"""Operator CLI.

Two kinds of command live here. ``doctor`` / ``status`` / ``inspect`` / ``runs``
/ ``evidence`` only report: what this deployment is configured to do and what a
run concluded. ``verify`` / ``rerun`` / ``approve`` / ``annotate`` drive the
control plane — they are the local entry point to the gates, the one that needs
no platform credential and so is the one a first try runs on. ``run-pr`` /
``resume`` are the same path with a platform on both ends; the work itself
lives in ``merge_requests.py``.
``init`` is neither kind: it writes the configuration the others read, once,
before any of them runs.

Three rules hold across all of them:

* **Credentials are never printed** — only whether one is configured.
* **The terminal state is the exit code.** A pipeline has to be able to tell
  "verified" from "nothing blocked it but nobody verified it", so ``mergeable``,
  ``blocked`` and ``needs_human`` do not collapse into one failure code.
* **A verdict is never produced here.** These commands build a change, hand it
  to the runner and render what came back; every conclusion belongs to a gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from . import __version__, intake
from .actions import ActionsContext
from .adapters import FORGE_ADAPTERS, SCANNER_ADAPTERS, build_forge_adapter
from .adapters.base import ForgeAdapter, ForgeError
from .adapters.scanners import ScannerConfigError, build_scanner_adapter
from .contracts import (
    GATE_SEQUENCE,
    ApprovalReading,
    ChangeRequest,
    Finding,
    evidence_schema_json,
)
from .diffs import (
    extract_changed_files,
    is_safe_relative_path,
    printable_path,
)
from .gates import GATE_REGISTRY
from .license_check import (
    KeyDue,
    Stopped,
    check_license,
    describe_key,
    key_due_on_pull_request,
    key_due_on_workspace,
)
from .maven import (
    MAVEN_JDK_CEILING,
    MAVEN_PLUGINS,
    MAVEN_STAMP,
    fetch_command,
    in_repository,
    maven_env,
    maven_java,
    offline_coordinates,
    pinned_ceilings,
)
from .merge_requests import (
    Published,
    apply_platform_approvals,
    fetch_merge_request,
    publish_conclusion,
    publish_license_stop,
    resume_interrupted_run,
    run_gates,
)
from .policy import PolicyConfig
from .redaction import redact_text
from .report import STATUS_HINT, integrity_findings, render_evidence
from .rules import RuleError, RuleSet
from .runner import GateRunner, unsatisfied_required_checks
from .scaffold import GITHUB_FILES, git_toplevel, write_file
from .settings import MODEL_NOT_ENABLED, Settings, SettingsError
from .state import RunState
from .storage import SQLiteLedger
from .tools import docker_available, docker_image_present, sandbox_env
from .utils import path_inside

app = typer.Typer(
    help="honest-ci — the verification layer for AI-assisted development.",
    no_args_is_help=True,
)

Workspace = Annotated[Path, typer.Option(help="Working directory; the current one by default.")]
Limit = Annotated[int, typer.Option(help="How many runs to list.")]

#: Terminal state → process exit code. 2 is left to typer's own usage errors,
#: so a pipeline can tell "you called it wrong" from "it blocked the change".
EXIT_CODES: dict[str, int] = {
    "mergeable": 0,
    "blocked": 1,
    "awaiting_human": 3,
    "failed": 4,
    "running": 4,
}
EXIT_USAGE = 2


def _settings() -> Settings:
    """Load the deployment's configuration, or say which value refused to load.

    Same reasoning as ``_open_runner`` below: a bar nobody can hold is a typo
    in an environment variable, and a typo deserves one line rather than a
    traceback. Every command goes through here, so ``doctor`` is not the only
    place that says it -- but ``doctor`` is where an operator looks first, and
    it now answers there instead of in the middle of a run.
    """

    try:
        return Settings.from_env()
    except SettingsError as exc:
        typer.echo(f"Configuration error, so nothing runs: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc


def _runner(workspace: Path) -> GateRunner:
    return GateRunner(workspace_root=workspace)


def _open_runner(workspace: Path) -> GateRunner:
    """Build a runner, turning a broken rule library into a message.

    A rule file nobody can parse stops the run — that is ``rules.py`` talking —
    but an operator should read why, not a traceback.
    """

    try:
        return _runner(workspace)
    except SettingsError as exc:
        # The runner loads settings itself when it was handed none, so a
        # refused value reaches an operator here too, not only via ``_settings``.
        typer.echo(f"Configuration error, so nothing runs: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc
    except RuleError as exc:
        typer.echo(f"The rule library cannot be parsed, so no gate runs: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc
    except ScannerConfigError as exc:
        # Same reasoning: a deployment that names a toolchain nobody built
        # should stop here, not block every change with "tool not installed".
        typer.echo(f"The toolchain configuration is invalid, so no gate runs: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc


@app.command()
def version() -> None:
    """Print the version."""

    typer.echo(f"honest-ci {__version__}")


@app.command()
def schema() -> None:
    """Print the JSON Schema of the evidence pack to standard output.

    The evidence format is a public contract, so it has to come out of the
    installation itself rather than be sent over by someone: whatever reads a
    pack can check its structure against this schema without importing
    honest-ci.

    `honest-ci schema > docs/evidence.schema.json` is how this repository's copy
    is made, and CI compares the two. A difference means a field changed: after
    adding a field, regenerate the file; before changing what a field means,
    bump EVIDENCE_SCHEMA_VERSION.
    """

    typer.echo(evidence_schema_json(), nl=False)


InitForGitHub = Annotated[
    bool,
    typer.Option("--github", help="Write the rules, the policy and the workflow the GitHub Action runs."),
]
InitRoot = Annotated[
    Path,
    typer.Option("--workspace", help="The repository's root directory; the current one by default."),
]


@app.command()
def init(github: InitForGitHub = False, workspace: InitRoot = Path(".")) -> None:
    """Set a repository up for honest-ci. A file that already exists is left as it is."""

    if not github:
        # One platform so far, and a bare `init` that picked it silently would
        # have to change meaning the day a second one arrives.
        typer.echo("Say which platform to set up for: honest-ci init --github.")
        raise typer.Exit(code=EXIT_USAGE)
    if not workspace.is_dir():
        typer.echo(f"{workspace} is not a directory.")
        raise typer.Exit(code=EXIT_USAGE)
    root = git_toplevel(workspace)
    if root is not None and root.resolve() != workspace.resolve():
        # Written anywhere else, the files would look installed and do nothing.
        typer.echo(
            f"{workspace} is inside the repository at {root}. Run init at its root: "
            "GitHub only runs workflows from .github/workflows there, and the Action "
            "reads the rules and the policy from there."
        )
        raise typer.Exit(code=EXIT_USAGE)

    created = False
    for item in GITHUB_FILES:
        try:
            outcome = write_file(workspace, item)
        except OSError as exc:
            typer.echo(f"Could not write {item.path}: {exc}")
            raise typer.Exit(code=1) from exc
        if outcome == "created":
            created = True
            typer.echo(f"Created {item.path}")
        else:
            typer.echo(f"Kept {item.path}: it already exists, and init does not overwrite it.")
    if not created:
        return
    typer.echo(
        "\nNext, commit these files and open a pull request with them. That first pull "
        "request goes to a person, as it should: its base branch has no honest-ci "
        "configuration yet, and it changes .github/. Every pull request after it is "
        "judged by the files it merges."
        f"\nTo make the verdict a merge requirement, add {Settings().forge_status_name} "
        "to the required status checks of the branch's protection rule."
    )


@app.command()
def doctor(workspace: Workspace = Path(".")) -> None:
    """Check this deployment's configuration: the model, the platform, the license key, the sandbox and the gates."""

    settings = _settings()
    rules, rules_line = _rule_library(workspace, settings)
    policy = PolicyConfig.load(workspace, settings.policy_path, settings.policy_version)
    plugins = " · ".join(
        f"{name} {coordinate.rsplit(':', 1)[-1]}" for name, coordinate in MAVEN_PLUGINS.items()
    )
    lines = [
        _field("Model provider", _provider_line(settings)),
        _field("Model", _model_line(settings)),
        _field(
            "Model explanation",
            f"{_explain_state(settings)} — the explanation text only; deterministic code "
            "reaches every verdict",
        ),
        _field("Model judgment", _judgment_line(settings, rules)),
        _field("Platform", f"{settings.forge}{_forge_line(settings)}"),
        # Configuration, not readiness: whether a run needs the key depends on
        # the repository and the pull request, so a refused key here stops
        # only the runs that need one, and doctor's exit code stays as it was.
        _field("License key", describe_key(os.environ, today=datetime.now(UTC).date())),
        _field(
            "Sandbox",
            f"{settings.effective_sandbox_isolation()} (configured: {settings.sandbox_mode})",
        ),
        _field("Toolchain", _toolchain_line(workspace, settings)),
        _field("Java image/plugins", f"{settings.java_docker_image} · {plugins}"),
        _field(
            "Tool unavailable",
            "block (fail-closed)" if settings.fail_closed else "let through (debugging only)",
        ),
        _field(
            "Versions",
            f"policy {policy.version} · rules "
            f"{rules.version if rules else settings.rule_version}",
        ),
        _field("Rule library", rules_line),
        _field("Ledger", str(workspace / settings.ledger_path)),
        _field("Gate order", " → ".join(GATE_SEQUENCE)),
    ]
    deployed = [name for name in GATE_SEQUENCE if name in GATE_REGISTRY]
    missing = [name for name in GATE_SEQUENCE if name not in GATE_REGISTRY]
    disabled = [name for name in GATE_SEQUENCE if name in settings.disabled_gates]
    lines.append(_field("Deployed gates", ", ".join(deployed) or "(none)"))
    if missing:
        lines.append(
            _field(
                "Not deployed",
                f"{', '.join(missing)} (recorded as skipped, which is not a pass)",
            )
        )
    if disabled:
        lines.append(
            _field(
                "Switched off",
                f"{', '.join(disabled)} (HONEST_CI_DISABLED_GATES; recorded as skipped, "
                "which is not a pass)",
            )
        )
    lines.append(_field("Regression gate", _regression_line(settings)))
    lines.append(_field("Required gates", ", ".join(policy.required_checks) or "(none)"))
    unsatisfiable = [
        name
        for name in policy.required_checks
        if name not in GATE_REGISTRY or name in settings.disabled_gates
    ]
    if unsatisfiable:
        # Says out loud what the runner will do, so nobody discovers it from a
        # pile of needs_human verdicts: a required gate that cannot run means no change in this
        # deployment reaches "mergeable" on its own. Deliberately *not* part of
        # the exit code below: a deployment that runs only the review gate is
        # configured exactly like this on purpose, and an installer script that
        # refused to continue over it would be refusing over a decision.
        lines.append(
            _field(
                "⚠ Unsatisfiable",
                f"{', '.join(unsatisfiable)} not deployed or switched off, so every run ends "
                "at needs_human. If this deployment answers for the review gate only, write "
                'required_checks = ["review"] in honest-ci-policy.toml.',
            )
        )
    problems = _readiness(workspace, settings, lines)
    typer.echo("\n".join(lines))
    if problems:
        # The one thing an installer can branch on. Everything above is
        # configuration a reader interprets; this is "on this machine, right
        # now, the next run will not do what the configuration above says".
        raise typer.Exit(code=1)


def _field(label: str, value: str) -> str:
    """One line of ``doctor``, its label padded so the values line up."""

    return f"{label:<18}: {value}"


def _counted(count: int, noun: str) -> str:
    """``count`` and ``noun``, the noun in the plural unless there is exactly one."""

    return f"{count} {noun}{'' if count == 1 else 's'}"


def _readiness(workspace: Path, settings: Settings, lines: list[str]) -> int:
    """Append the readiness block; return how many things are not ready.

    Everything else ``doctor`` prints is configuration -- what this deployment
    was told to do. This section is the machine: does the sandbox image exist
    here, is the dependency cache warm, can anything be written down. They are
    separated because they fail at different times and get fixed by different
    people, and because these are the ones that turn a 30-minute first run into
    an afternoon -- each has to arrive with the command that fixes it, before
    the first run rather than during it.
    """

    checks: list[tuple[bool, str]] = []
    checks.append(_writable_check("Ledger", workspace / settings.ledger_path, file=True))
    checks.append(_writable_check("Artifacts directory", workspace / settings.artifacts_dir))
    checks += _sandbox_checks(workspace, settings)
    lines.append("")
    lines.append("Readiness (whether this machine can run the configuration above, now):")
    lines += [f"  {'✓' if ok else '✗'} {text}" for ok, text in checks]
    failed = sum(1 for ok, _ in checks if not ok)
    if failed:
        lines.append(
            f"  -- {_counted(failed, 'check')} not ready, so the next run will not do what "
            "the configuration above says. Exit code 1."
        )
    return failed


def _writable_check(label: str, path: Path, *, file: bool = False) -> tuple[bool, str]:
    """Whether a run could write ``path`` -- asked without creating it.

    Probes the nearest existing ancestor instead of creating the directory:
    ``doctor`` answers questions about a deployment, it does not quietly
    provision one, and an operator who sees ``.honest-ci/`` appear after a
    diagnostic learns nothing about whether the run could have made it.
    """

    problem = _file_write_probe(path) if file else _write_probe(path)
    if problem is None:
        return True, f"{label} is writable: {path}"
    return False, f"{label} cannot be written: {path}. {problem}"


def _write_probe(path: Path) -> str | None:
    target = path
    while not target.exists():
        parent = target.parent
        if parent == target:
            return "No parent directory of this path exists"
        target = parent
    if not target.is_dir():
        return f"{target} exists but is not a directory"
    probe = target / f".honest-ci-doctor-{os.getpid()}"
    try:
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return f"{target} is not writable ({exc.strerror or exc})"
    return None


def _file_write_probe(path: Path) -> str | None:
    """``_write_probe`` for a file a run opens in place, such as the ledger.

    Every run after the first finds the ledger already there, so it has to be
    asked about as a file. Its directory is probed as well, whether or not the
    file exists: SQLite creates the file there on the first run, and its
    rollback journal there on every write. The file itself is opened for
    writing and closed without a byte written -- a diagnostic that changes
    what it diagnoses is not one.
    """

    problem = _write_probe(path.parent)
    if problem is not None or not path.exists():
        return problem
    try:
        os.close(os.open(path, os.O_WRONLY))
    except OSError as exc:
        return f"{path} is not writable ({exc.strerror or exc})"
    return None


def _sandbox_checks(workspace: Path, settings: Settings) -> list[tuple[bool, str]]:
    """Docker and the toolchain's dependency cache, when this run would use them.

    The image only in docker mode: in process mode the host interpreter
    already carries what it needs, and reporting a missing image there would
    be reporting on something nothing was going to use. Java asks about this
    machine's Maven repository instead, which is all a process-mode
    ``mvn -o`` has to resolve from.
    """

    if settings.sandbox_mode != "docker":
        return [
            (True, "The sandbox runs in process mode (HONEST_CI_SANDBOX_MODE), so it needs no image"),
            *_host_maven_checks(workspace, settings),
        ]
    if not docker_available():
        return [
            (
                False,
                "Docker is unavailable (not installed, or the daemon cannot be reached). "
                "The sandbox falls back to process mode, so the tests and the scanners share "
                "this machine's environment",
            ),
            *_host_maven_checks(workspace, settings),
        ]
    checks: list[tuple[bool, str]] = [(True, "The Docker daemon is running")]
    try:
        scanner = build_scanner_adapter(settings, workspace / settings.artifacts_dir)
    except ScannerConfigError:
        # Already reported in full on the toolchain line above; repeating it
        # here would make one misconfiguration look like two.
        return checks
    language = scanner.language_at(workspace)
    image = scanner.image(workspace)
    if image is not None:
        present = docker_image_present(image)
        checks.append(
            (present, f"The sandbox image is on this machine: {image}")
            if present
            else (
                False,
                f"The sandbox image is not on this machine: {image}. Run `docker pull {image}` "
                "first, or the first run will pull it",
            )
        )
    if language == "java":
        checks.append(_maven_cache_check(workspace))
    return checks


def _host_maven_checks(workspace: Path, settings: Settings) -> list[tuple[bool, str]]:
    """What a process-mode Java run takes from this machine: its Maven repository and its JDK."""

    try:
        scanner = build_scanner_adapter(settings, workspace / settings.artifacts_dir)
    except ScannerConfigError:
        return []
    if scanner.language_at(workspace) != "java":
        return []
    return [_host_repository_check(workspace, settings), _maven_jdk_check(workspace)]


def _host_repository_check(workspace: Path, settings: Settings) -> tuple[bool, str]:
    """What a process-mode Java run can resolve, read off this machine's repository.

    Process mode provisions nothing, so ``mvn -o`` has the workspace's
    ``.honest-ci/m2`` and, read-only, ``HONEST_CI_MAVEN_HOST_REPOSITORY``. A
    developer's own builds fill the second with the project's dependencies,
    but not with the plugin versions this toolchain pins, and a run finds
    those missing one per run. Listed here all at once instead, with the line
    that fetches them.
    """

    host = settings.maven_host_repository
    if host is None:
        # Then the workspace's own repository is all there is, as in docker
        # mode, and the question is the one asked there.
        return _maven_cache_check(workspace)
    own = workspace / ".honest-ci" / "m2"
    pinned = offline_coordinates(settings.maven_extra_artifacts)
    missing = [
        item for item in pinned if not (in_repository(own, item) or in_repository(host, item))
    ]
    if not missing:
        return (
            True,
            "Every plugin and artifact the toolchain pins is in the workspace's "
            f".honest-ci/m2 or this machine's Maven repository {host}, which process mode "
            "reads, read-only, with Maven 3.9 or later",
        )
    return (
        False,
        f"Process mode reads this machine's Maven repository {host}, read-only, with Maven "
        f"3.9 or later, and it lacks {len(missing)} of the {len(pinned)} plugins and "
        f"artifacts the toolchain pins ({', '.join(missing)}). Fetch them with a network: "
        f"{fetch_command(missing, host)}",
    )


#: ``mvn -v`` starts a JVM and reads no project; a Maven slower than this
#: would time a run out as well.
_MAVEN_PROBE_TIMEOUT_SECONDS = 30


def _maven_jdk_check(workspace: Path) -> tuple[bool, str]:
    """The JDK a process-mode run's ``mvn`` runs on, asked the way a run asks.

    In the sandbox's environment, not the operator's shell: the shell's
    JAVA_HOME once never reached a run, Homebrew's mvn ran on Homebrew's
    newest openjdk instead, and every Java run was blocked while nothing here
    said why. A JDK newer than the pinned tools read is not a slow first run
    but one that cannot check anything, so it fails readiness.
    """

    try:
        proc = subprocess.run(
            ["mvn", "-v"],
            cwd=workspace,
            env={**sandbox_env(), **maven_env(workspace, "process")},
            capture_output=True,
            text=True,
            timeout=_MAVEN_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except OSError:
        return (
            False,
            "No mvn on this machine's PATH, and a process-mode Java run needs Maven 3.9 or "
            "later there",
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            f"mvn -v, run as a process-mode run runs it, did not answer within "
            f"{_MAVEN_PROBE_TIMEOUT_SECONDS}s",
        )
    java = maven_java(proc.stdout) if proc.returncode == 0 else None
    if java is None:
        output = redact_text(f"{proc.stdout}\n{proc.stderr}").strip()
        return (
            False,
            "mvn -v, run in the environment a process-mode run gets, did not say which JDK it "
            f"runs on (exit code {proc.returncode}): {output[-300:] or 'no output'}. JAVA_HOME "
            "is part of that environment; check that it names a JDK",
        )
    runs_on = f"Java {java.version}" + (f" ({java.runtime})" if java.runtime else "")
    if java.release <= MAVEN_JDK_CEILING:
        return (
            True,
            f"A process-mode run's mvn runs on {runs_on}, which every pinned tool reads "
            f"(up to Java {MAVEN_JDK_CEILING})",
        )
    return (
        False,
        f"A process-mode run's mvn runs on {runs_on}, newer than the pinned tools read "
        f"({pinned_ceilings()}): a tool past its bound cannot read this JDK's own classes, so "
        f"it cannot check a run. Point JAVA_HOME at JDK {MAVEN_JDK_CEILING} or earlier, such "
        "as 21; Homebrew's mvn runs on Homebrew's newest openjdk unless JAVA_HOME is set",
    )


def _maven_cache_check(workspace: Path) -> tuple[bool, str]:
    """Whether Maven has anything local to work from, since it runs offline.

    The Java toolchain runs ``mvn -o``: a cold ``.honest-ci/m2`` is not a slow
    first build, it is a build that cannot resolve a single plugin. Worth its
    own line for exactly that reason -- the failure reads like a broken POM.
    """

    repo = workspace / ".honest-ci" / "m2"
    if (workspace / MAVEN_STAMP).is_file():
        return True, f"The local Maven repository is warm: {repo}"
    if repo.is_dir() and any(repo.iterdir()):
        return True, (
            f"The local Maven repository has content but no provisioning stamp: {repo} "
            "(the first run completes it)"
        )
    return (
        False,
        f"The local Maven repository is empty: {repo}. The Java toolchain always runs "
        "mvn -o: without a network, warm it by running honest-ci verify once on a machine "
        "that has one, or copy .honest-ci/m2 over whole",
    )


def _regression_line(settings: Settings) -> str:
    """What the regression gate would run, so an operator sees it before a run does."""

    command = settings.test_command or (
        "by language: Python pytest -q (through coverage run when coverage is available); "
        "Java mvn -B -o test (JaCoCo prepare-agent / report)"
        if settings.coverage_measured
        else "by language: Python pytest -q; Java mvn -B -o test (no coverage bar is on, so "
        "neither collects coverage)"
    )
    if settings.test_command:
        command += (
            f" (test counts read from the JUnit XML {settings.test_report})"
            if settings.test_report
            else " (HONEST_CI_TEST_REPORT not set: for Java the Surefire XML is read; for "
            "Python, a command other than pytest is read by its exit code only)"
        )
    threshold = f"{settings.coverage_threshold:.0%} (0 = not measured)"
    changed = f"{settings.changed_line_coverage_threshold:.0%} (0 = not measured)"
    return (
        f"test command {command} · runs {_selection_clause(settings)} · coverage bar {threshold}"
        f" · changed-line coverage bar {changed} · generated regression tests "
        f"{_testgen_state(settings)} · test timeout {settings.test_timeout_seconds}s"
    )


def _selection_clause(settings: Settings) -> str:
    """Which tests a run runs, read off the same switches the regression gate reads.

    ``impacted`` alone does not decide it: an operator who set it and reads
    "the impacted tests" here, over a deployment that will run the whole suite
    every time, has been told the opposite of what happens.
    """

    if settings.test_selection != "impacted":
        return "the whole suite"
    if settings.coverage_threshold > 0:
        return (
            "the whole suite (HONEST_CI_TEST_SELECTION=impacted, but the overall coverage bar "
            "HONEST_CI_COVERAGE_THRESHOLD is a figure about the whole suite, which a run of "
            "the impacted tests cannot measure)"
        )
    if settings.test_command:
        return (
            "the whole suite (HONEST_CI_TEST_SELECTION=impacted, but HONEST_CI_TEST_COMMAND "
            "runs as written and cannot take a selection)"
        )
    return (
        "the impacted tests (HONEST_CI_TEST_SELECTION=impacted; the whole suite when none "
        "can be selected)"
    )


def _explain_state(settings: Settings) -> str:
    """Whether the next evidence pack will carry a model explanation.

    The explain switch defaults to on, so reading only the switch would say
    "on" on every fresh install -- where no provider has been named and no
    model will say anything.
    """

    if not settings.model_explain:
        return "off (HONEST_CI_MODEL_EXPLAIN=0)"
    if settings.active_provider is None:
        return f"off ({_model_off(settings)})"
    return "on"


def _testgen_state(settings: Settings) -> str:
    """Whether regression cases will really be generated on the next change.

    Same silent failure as judgment-class rules: a deployment with no
    credential still runs the gate, and simply never produces a case. Said out
    loud here so nobody reads a clean evidence pack as a covered change.
    """

    if not settings.model_testgen:
        return "off (HONEST_CI_MODEL_TESTGEN=0)"
    if settings.active_provider is None:
        return f"off ({_model_off(settings)})"
    return "on (a failed case goes to a person and never blocks)"


def _model_line(settings: Settings) -> str:
    """The model the next call would reach, and the host its key is sent to.

    The host is worth printing now that ``openai`` can point at DeepSeek, a
    local Ollama or a gateway: the provider's name no longer says where the
    key and the diffs go.
    """

    provider = settings.active_provider
    if provider is None:
        return "none"
    return f"{settings.active_model} @ {settings.base_url_for(provider)}"


def _provider_line(settings: Settings) -> str:
    """The provider the next change would reach, or which switch is missing."""

    if settings.model_provider is None:
        return MODEL_NOT_ENABLED
    if settings.active_provider is None:
        return f"{settings.model_provider} ({_model_off(settings)})"
    return settings.model_provider


def _model_off(settings: Settings) -> str:
    """Why no model will be asked, when ``active_provider`` says none will.

    Two different fixes, so two different answers: nobody named a provider --
    the default, which a key lying in the environment does not change -- or
    one was named and its credential is missing.
    """

    if settings.model_provider is None:
        return MODEL_NOT_ENABLED
    return f"no credential configured: {settings.credential_hint}"


def _toolchain_line(workspace: Path, settings: Settings) -> str:
    """Which toolchain the next run here would use, and every one this build has.

    The workspace decides under ``auto``, so the line is about *this*
    directory: an operator running ``doctor`` in a Java repository should see
    java, or see why python was picked instead, before the first run does.
    """

    registered = " · ".join(
        f"{language} ({', '.join(adapter_cls.supported_tools) or 'no static tools'})"
        for language, adapter_cls in SCANNER_ADAPTERS.items()
    )
    try:
        scanner = build_scanner_adapter(settings, workspace / settings.artifacts_dir)
    except ScannerConfigError as exc:
        return (
            f"⚠ {exc}. honest-ci verify and run-pr will refuse to start · "
            f"registered: {registered}"
        )
    return f"{scanner.toolchain(workspace)} · registered: {registered}"


def _forge_line(settings: Settings) -> str:
    """Say whether ``run-pr`` would work, and if not, which of the three reasons.

    An operator should learn "this deployment cannot take a pull request" here,
    not from the first one that needed verifying.
    """

    if settings.forge == "local":
        return (
            " (local changes only; to write back to a platform, set HONEST_CI_FORGE and "
            "use honest-ci run-pr)"
        )
    if settings.forge not in FORGE_ADAPTERS:
        return f" (no adapter; implemented: {', '.join(sorted(FORGE_ADAPTERS))})"
    if not settings.forge_token():
        return f" (no token configured: {settings.forge_credential_hint})"
    return (
        f" (adapter ready · {settings.forge_base_url()} · status name "
        f"{settings.forge_status_name})"
    )


def _rule_library(workspace: Path, settings: Settings) -> tuple[RuleSet | None, str]:
    """Report the rule library without letting a broken one hide the rest."""

    try:
        rules = RuleSet.load(workspace, settings.rules_path, settings.rule_version)
    except RuleError as exc:
        return None, f"cannot be parsed: {exc}"
    if rules.source is None:
        return rules, (
            f"not configured ({workspace / settings.rules_path}). The review gate will hand "
            "every change to a person"
        )
    tools = ", ".join(rules.required_tools) or "none"
    # Provenance is counted rather than assumed: rule mining goes from samples
    # to a draft to a version a person signs off, and a rule that doesn't
    # record its samples can't answer "why does this rule exist" when someone
    # challenges it a year later.
    sourced = sum(1 for rule in rules.rules if rule.source_samples)
    provenance = (
        f", {sourced}/{len(rules.rules)} with source samples recorded" if rules.rules else ""
    )
    return rules, (
        f"{rules.source} ({_counted(len(rules.rules), 'rule')}, scanners needed: "
        f"{tools}{provenance})"
    )


def _judgment_line(settings: Settings, rules: RuleSet | None) -> str:
    """Whether judgment-class rules will really be judged on the next change.

    Worth its own line because the failure is silent from the outside: a rule
    file with judgment rules and no credential still loads, still reports its
    rule count, and still runs — it just sends every change to a human.
    """

    judgment = len(rules.judgment_rules) if rules else 0
    off: str | None = None
    if not settings.model_judgment:
        off = "HONEST_CI_MODEL_JUDGMENT=0"
    elif settings.active_provider is None:
        off = _model_off(settings)
    state = (
        f"off ({off})" if off else f"on (confidence bar {settings.model_min_confidence:.2f})"
    )
    if judgment == 0:
        return f"{state}. The rule library has no judgment rules"
    if off:
        return (
            f"{state}. {_counted(judgment, 'judgment rule')} will be recorded as not "
            "judged, and every change goes to a person"
        )
    return (
        f"{state}. The model reads {_counted(judgment, 'judgment rule')}; deterministic "
        "code still reaches the verdict"
    )


# ── running a change ─────────────────────────────────────────────────────────

DiffFile = Annotated[
    Path | None,
    typer.Option(
        "--diff",
        help="Path to a patch file; - reads it from standard input. Cannot be used with --base.",
    ),
]
BaseRef = Annotated[
    str | None,
    typer.Option(
        "--base",
        help="Compare against the merge base with this branch or commit. By default the "
        "working tree is compared with HEAD.",
    ),
]
RepoName = Annotated[
    str | None, typer.Option("--repo", help="Repository name; the working directory's by default.")
]
HeadSha = Annotated[
    str | None,
    typer.Option(
        "--head-sha",
        help="SHA of the commit being verified. Without it, a hash of the diff identifies "
        "the change.",
    ),
]
ChangeTitle = Annotated[
    str, typer.Option("--title", help="Title of the change, recorded in the evidence pack.")
]
ChangeAuthor = Annotated[
    str | None,
    typer.Option("--author", help="Author of the change, recorded in the evidence pack."),
]
AiRatio = Annotated[
    float | None,
    typer.Option(
        "--ai-ratio",
        min=0.0,
        max=1.0,
        help="Share of the change an AI wrote, 0 to 1. Leaving it out means not measured, "
        "which is not 0.",
    ),
]


@app.command()
def verify(
    workspace: Workspace = Path("."),
    diff: DiffFile = None,
    base: BaseRef = None,
    repo: RepoName = None,
    head_sha: HeadSha = None,
    title: ChangeTitle = "",
    author: ChangeAuthor = None,
    ai_ratio: AiRatio = None,
) -> None:
    """Run the gates on a local git diff or patch file: a verdict and an evidence pack come out.

    Exit codes: 0 mergeable, 1 blocked, 3 needs a human, 4 the run failed or,
    in CI without a license key where one is due, no gate ran, 2 usage error.
    """

    if diff is not None and base is not None:
        typer.echo(
            "--diff and --base cannot be used together: one reads a patch file, the other "
            "takes the working tree's changes from git."
        )
        raise typer.Exit(code=EXIT_USAGE)

    try:
        source = _change_source(workspace, diff, base)
        change = intake.build_change(
            source,
            repo=repo or workspace.resolve().name,
            head_sha=head_sha,
            title=title,
            author=author,
            target_branch=base,
            ai_generated_ratio=ai_ratio,
        )
    except intake.IntakeError as exc:
        typer.echo(f"Could not read the change: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc

    runner = _open_runner(workspace)
    _render_source(source, change, workspace)
    _license_note(
        key_due_on_workspace(os.environ, ActionsContext.from_env(), workspace, runner.settings),
        runner.settings,
    )
    # The tree the scanners run over is the one we were pointed at. Recording it
    # explicitly (rather than leaving the gate to fall back to the deployment
    # root) is what puts the path into the evidence pack.
    state = runner.build_state(change).model_copy(
        update={"workspace_path": str(workspace.resolve())}
    )
    state = runner.run(state)
    _render_run(runner, state)
    raise typer.Exit(code=EXIT_CODES[state.status])


def _change_source(
    workspace: Path, diff: Path | None, base: str | None
) -> intake.LocalChangeSource:
    if diff is None:
        return intake.from_git(workspace, base)
    if str(diff) == "-":
        return intake.from_patch(sys.stdin.read(), "patch (standard input)")
    return intake.from_patch(intake.read_patch_file(diff), _patch_origin(diff, workspace))


def _patch_origin(diff: Path, workspace: Path) -> str:
    """Name a patch file the way something that outlives this machine can carry it.

    It becomes the change's title, which is the first line of ``evidence.md``,
    and a CI agent invoking ``--diff "$CI_BUILDS_DIR/x.patch"`` would put that
    agent's build directory there, which is not something an auditor can check.

    A patch from outside the workspace is marked rather than quietly shortened:
    "patch file x.patch" on its own reads as a file in the repository, and where
    a change came from is not a claim to blur. Neither spelling carries the
    patch's identity anyway -- a run's ``head_sha`` is a digest of the diff.
    """

    inside = path_inside(diff, workspace)
    return f"patch file {inside}" if inside else f"patch file {diff.name} (outside the repository)"


DryRun = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Run the gates and print what would be written back, without touching the "
        "pull request.",
    ),
]


PullRequestRepo = Annotated[
    str, typer.Argument(metavar="OWNER/REPO", help="The repository, e.g. octo-org/orders")
]
PullRequestNumber = Annotated[int, typer.Argument(metavar="NUMBER", help="Pull request number")]
PullRequestDryRun = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Run the gates and print what would be written back, without touching the PR",
    ),
]


@app.command("run-pr")
def run_pr(
    project: PullRequestRepo,
    number: PullRequestNumber,
    workspace: Workspace = Path("."),
    dry_run: PullRequestDryRun = False,
) -> None:
    """Verify one pull request and write the verdict back as a comment and a commit status.

    Needs `HONEST_CI_FORGE=github` and a `GITHUB_TOKEN`; the Action sets both.
    Inside GitHub Actions it verifies the workflow's own checkout, links the
    status to the workflow run and adds the conclusion to the job summary. On a
    pull request from a fork the token is read-only, so the job summary is the
    only place the conclusion goes, and the exit code is unchanged.

    In CI, on a repository that needs a license key and has none that works,
    no gate runs: the pull request gets an error status and a comment that
    says why.

    Exit codes: 0 mergeable, 1 blocked, 3 needs a human, 4 the run or the
    write-back failed, or no gate ran for want of a license key, 2 usage error.
    """

    _run_on_forge(project, number, workspace, dry_run)


def _run_on_forge(project: str, number: int, workspace: Path, dry_run: bool) -> None:
    """``run-pr``: fetch, gate, write back, exit with the terminal state."""

    settings = _settings()
    adapter = _forge_adapter(settings)
    runner = _open_runner(workspace)
    actions = ActionsContext.from_env()
    # "The GITHUB_TOKEN has read-only permissions in pull requests from forked
    # repositories", so every write would fail. The gates run as a dry run
    # would, and the job summary carries the conclusion.
    # https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request
    read_only = actions.read_only_reason() if actions is not None else None
    write_back = not dry_run and read_only is None
    target_url = actions.run_url if actions is not None else None

    try:
        source = fetch_merge_request(
            adapter, settings, project=project, number=number, workspace=workspace
        )
    except ForgeError as exc:
        typer.echo(f"Could not read {project}#{number}: {exc}")
        if actions is not None:
            _write_job_summary(actions, f"## honest-ci\n\nCould not read {project}#{number}: {exc}")
        raise typer.Exit(code=EXIT_CODES["failed"]) from exc

    _render_merge_request(source.change, source.checkout)
    checked = check_license(
        key_due_on_pull_request(os.environ, actions, adapter, source.change),
        os.environ,
        settings,
        today=datetime.now(UTC).date(),
    )
    if isinstance(checked, Stopped):
        _stop_on_pull_request(
            adapter,
            source.change,
            checked,
            settings=settings,
            actions=actions,
            write_back=write_back,
            target_url=target_url,
        )
    _render_license_note(checked.note)
    state = run_gates(
        runner, source, adapter=adapter, dry_run=not write_back, target_url=target_url
    )
    # A GitHub Actions runner, and the ledger on it, is thrown away with the job.
    _render_run(runner, state, ledger_kept=actions is None)

    approvals: ApprovalReading | None = None
    if read_only is None:
        # Not on a run that cannot write back: an approval is read in the
        # window the conclusion comment records, and such a run can record
        # none, so nothing given on a fork's pull request could ever count.
        state, approvals = apply_platform_approvals(adapter, runner, state)
        _render_approvals(state, approvals)

    published = publish_conclusion(
        adapter,
        state,
        dry_run=not write_back,
        target_url=target_url,
        approvals=approvals,
        evidence_at=actions.evidence_at if actions is not None else None,
        license_note=checked.note,
    )
    if read_only is not None and not dry_run:
        typer.echo(
            f"Not written to the pull request: {read_only}, and GitHub gives such runs "
            "a read-only token. The conclusion is in the job summary."
        )
    else:
        _render_published(published, settings)
    if actions is not None:
        _write_job_summary(actions, _job_summary(published, read_only))
    raise typer.Exit(code=EXIT_CODES[state.status] if published.ok else EXIT_CODES["failed"])


def _stop_on_pull_request(
    adapter: ForgeAdapter,
    change: ChangeRequest,
    stopped: Stopped,
    *,
    settings: Settings,
    actions: ActionsContext | None,
    write_back: bool,
    target_url: str | None,
) -> NoReturn:
    """``run-pr`` the license check stopped: say why, on the pull request too, and exit 4."""

    _render_stopped(stopped)
    published = publish_license_stop(
        adapter, change, stopped, dry_run=not write_back, target_url=target_url
    )
    if published.error is not None:
        typer.echo(f"⚠ Write-back failed: {published.error}")
        typer.echo("  Nobody on the pull request can see why no gate ran.")
    elif not published.sent:
        typer.echo(
            "Write-back: --dry-run, nothing below was sent (the commit status would be an error)"
        )
        typer.echo(f"  Status  : {published.summary}")
        for line in published.comment.splitlines():
            typer.echo(f"  | {line}")
    else:
        typer.echo(f"Written   : comment and commit status error ({settings.forge_status_name})")
    if actions is not None:
        _write_job_summary(actions, _job_summary(published, None))
    raise typer.Exit(code=EXIT_CODES["failed"])


def _license_note(due: KeyDue | None, settings: Settings) -> str | None:
    """What the license check says of this run, if anything; a run it stops ends here, exit 4.

    For ``verify`` and ``resume``, whose stop goes no further than their
    output: ``verify`` has no pull request to tell, and ``resume`` leaves the
    run it was asked to finish as it found it, for a run with a key to finish.
    """

    checked = check_license(due, os.environ, settings, today=datetime.now(UTC).date())
    if isinstance(checked, Stopped):
        _render_stopped(checked)
        raise typer.Exit(code=EXIT_CODES["failed"])
    _render_license_note(checked.note)
    return checked.note


def _render_stopped(stopped: Stopped) -> None:
    headline, *rest = stopped.explanation
    typer.echo(f"⚠ {headline}")
    for line in rest:
        typer.echo(f"  {line}" if line else "")


def _render_license_note(note: str | None) -> None:
    if note is not None:
        typer.echo(f"License   : {note}")


def _render_approvals(state: RunState, reading: ApprovalReading | None) -> None:
    """Say what the approvals on the platform came to, when the run asked for them."""

    if reading is None:
        return
    decision = state.approval_decision
    if reading.approval is not None and decision is not None:
        typer.echo(
            f"Approval  : approved by {printable_path(decision.reviewer or '')} on the pull request"
            + (f" ({printable_path(decision.url)})" if decision.url else "")
            + ". No gate's verdict changed; the outcome is approved (exit code 0)."
        )
        typer.echo(f"  Reason  : {printable_path(_one_line(decision.reason or ''))}")
        return
    if reading.error is not None:
        typer.echo(f"Approval  : none applied, the change keeps waiting: {reading.error}")
        return
    if not reading.uncounted and not reading.objections:
        typer.echo("Approval  : nobody has approved this commit on the pull request yet.")
        return
    typer.echo("Approval  : none counts, so the change keeps waiting.")
    for item in reading.uncounted:
        who = printable_path(item.review.reviewer or "(deleted account)")
        typer.echo(f"  {who}: {item.why}.")
    for review in reading.objections:
        typer.echo(f"  {printable_path(review.reviewer)} requested changes, and that stands.")


def _job_summary(published: Published, read_only: str | None) -> str:
    """The conclusion as the workflow run shows it, whatever became of the write-back."""

    parts = [published.comment.strip()] if published.comment else []
    if read_only is not None:
        parts.append(
            f"> Not posted to the pull request: {read_only}, and GitHub gives such runs "
            "a read-only token. For the same reason an Approve review on the pull request "
            "cannot take effect: every run for it gets such a token."
        )
    elif published.error is not None:
        parts.append(f"> ⚠ Not written back to the pull request: {published.error}")
    return "\n\n".join(parts) or "## honest-ci\n\nNo conclusion to report."


def _write_job_summary(actions: ActionsContext, markdown: str) -> None:
    if actions.step_summary is None:
        return
    if not actions.write_step_summary(markdown):
        typer.echo(f"⚠ Could not write the job summary to {actions.step_summary}.")


def _forge_adapter(settings: Settings) -> ForgeAdapter:
    """Resolve the platform adapter, or say precisely what is missing."""

    adapter = build_forge_adapter(settings)
    if adapter is not None:
        return adapter
    if settings.forge == "local":
        typer.echo(
            "HONEST_CI_FORGE=local: verify a local change with honest-ci verify; run-pr "
            "needs a platform."
        )
    elif settings.forge not in FORGE_ADAPTERS:
        typer.echo(
            f"honest-ci has no adapter for {settings.forge}. Implemented: "
            + ", ".join(sorted(FORGE_ADAPTERS))
            + "."
        )
    else:
        typer.echo(
            "GITHUB_TOKEN is not set. In a workflow, pass `secrets.GITHUB_TOKEN` to the step; "
            "the job needs `contents: read`, `pull-requests: write` and `statuses: write`."
        )
    raise typer.Exit(code=EXIT_USAGE)


def _render_merge_request(change: ChangeRequest, checkout: Path) -> None:
    """Say what is about to be judged, and on which tree."""

    typer.echo(f"Change    : {change.repo}#{change.number} — {change.title or '(no title)'}")
    branches = " → ".join(part for part in (change.source_branch, change.target_branch) if part)
    typer.echo(f"Commit    : {change.head_sha[:12]}" + (f" ({branches})" if branches else ""))
    typer.echo(f"Checkout  : {checkout}")
    typer.echo(f"Scope     : {_scope(change)}")
    opaque = [
        item.path
        for item in change.files_changed
        if item.change_type != "deleted"
        and not item.added_lines
        and not item.removed_lines
        # A name refused as a path is the guardrails' to report, with the
        # reason; printed here it would be printed raw.
        and is_safe_relative_path(item.path)
    ]
    if opaque:
        # The gate attributes findings to added lines, and these files have
        # none to attribute against. Saying so here keeps "clean" and "not read"
        # from looking alike in the terminal as well as in the evidence pack.
        typer.echo(
            f"⚠ Not reviewable: {_counted(len(opaque), 'file')} with no line-by-line diff "
            f"({', '.join(opaque[:5])}{' …' if len(opaque) > 5 else ''}), recorded but not "
            "reviewed."
        )


# ── writing back ─────────────────────────────────────────────────────────────


def _render_published(published: Published, settings: Settings) -> None:
    """Say what reached the platform, and say plainly when nothing did."""

    if published.verdict is None:
        # No evidence pack: the run never reached a terminal state, so there
        # was no conclusion to write anywhere.
        typer.echo(f"⚠ {published.error}")
        return
    if published.error is not None:
        typer.echo(f"⚠ Write-back failed: {published.error}")
        typer.echo(
            "  The verdict and the evidence pack are on disk, but nobody on the platform "
            "can see them, so the exit code says the run failed."
        )
        return
    if not published.sent:
        typer.echo(
            "Write-back: --dry-run, nothing below was sent (the commit status would be "
            f"{published.verdict.value})"
        )
        typer.echo(f"  Status  : {published.summary}")
        for line in published.comment.splitlines():
            typer.echo(f"  | {line}")
        return
    typer.echo(
        f"Written   : comment and commit status {published.verdict.value} "
        f"({settings.forge_status_name})"
    )
    typer.echo(
        f"  Inline comments: {published.inline_posted} new"
        + (f"; {published.inline_skipped}" if published.inline_skipped else "")
    )


@app.command()
def resume(
    run_id: str,
    workspace: Workspace = Path("."),
    dry_run: DryRun = False,
) -> None:
    """Finish a run that was interrupted halfway, by a killed process or a restarted machine.

    It continues from the gate the checkpoint recorded rather than starting
    over. A run waiting for approval is not one of these: it was not
    interrupted, it is waiting for a person, and `honest-ci approve` is the
    command for it. A change on a platform gets its verdict written back as
    usual.

    In CI, where a license key is due and there is none that works, no gate
    runs: the run is left as it was, for a run with a key to finish, and the
    exit code is 4.
    """

    settings = _settings()
    runner = _open_runner(workspace)
    state = runner.status(run_id)
    if state is None:
        typer.echo(f"No run found: {run_id}")
        raise typer.Exit(code=EXIT_USAGE)
    if state.status == "awaiting_human":
        # Not an interruption. Saying so here, rather than letting the control
        # plane's own refusal surface, is the difference between an operator
        # reading a method name and reading the command they wanted.
        typer.echo(
            f"This run was not interrupted; it is waiting for a person to decide: "
            f"honest-ci approve {run_id} --reason ...\n"
            "  resume is only for runs interrupted halfway. Finishing a run that waits for "
            "approval would sign for that person."
        )
        raise typer.Exit(code=EXIT_USAGE)
    note = _license_note(
        key_due_on_workspace(os.environ, ActionsContext.from_env(), workspace, settings), settings
    )
    try:
        result = resume_interrupted_run(
            runner=runner,
            run_id=run_id,
            settings=settings,
            workspace=workspace,
            adapter=build_forge_adapter(settings),
            dry_run=dry_run,
            license_note=note,
        )
    except ForgeError as exc:
        typer.echo(f"A platform call failed during resume: {exc}")
        raise typer.Exit(code=EXIT_CODES["failed"]) from exc
    _render_run(runner, result.state)
    if result.published.verdict is not None or result.published.error is not None:
        _render_published(result.published, settings)
    raise typer.Exit(code=EXIT_CODES[result.outcome])


@app.command()
def rerun(
    run_id: str,
    gate: str,
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Why it is rerun; recorded in the run's history."),
    ] = None,
    workspace: Workspace = Path("."),
) -> None:
    """Rerun one gate; the verdicts after it, and any approval on top of it, are invalidated too."""

    runner = _open_runner(workspace)
    try:
        state = runner.rerun_gate(run_id, gate, reason=reason)
    except ValueError as exc:
        typer.echo(f"Nothing was rerun: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc
    _render_run(runner, state)
    raise typer.Exit(code=EXIT_CODES[state.status])


@app.command()
def approve(
    run_id: str,
    reason: Annotated[
        str,
        typer.Option(
            "--reason",
            help="What the decision rests on. Required: a pass with no reason is not evidence.",
        ),
    ],
    reject: Annotated[
        bool,
        typer.Option("--reject", help="Record a rejection; the default is an approval."),
    ] = False,
    reviewer: Annotated[str | None, typer.Option("--reviewer", help="Who reviewed it.")] = None,
    minutes: Annotated[
        float | None,
        typer.Option(
            "--minutes", min=0.0, help="Minutes the review took, recorded in the run's metrics."
        ),
    ] = None,
    workspace: Workspace = Path("."),
) -> None:
    """Record a person's approval.

    The approval is bound to the evidence fingerprint: the head commit, the
    policy version and the rule version. When the fingerprint changes (a new
    commit is pushed, a rule is edited) the old approval is void at once, and
    this command refuses it rather than let the change through quietly.
    """

    runner = _open_runner(workspace)
    state = runner.status(run_id)
    if state is None:
        typer.echo(f"No run found: {run_id}")
        raise typer.Exit(code=EXIT_USAGE)
    if state.approval_ticket is not None:
        typer.echo(f"Subject   : {state.approval_ticket.evidence_summary}")
        typer.echo(f"Fingerprint: {state.approval_ticket.fingerprint}")
    try:
        state = runner.resume_approval(
            run_id,
            approved=not reject,
            reason=reason,
            reviewer=reviewer,
            review_minutes=minutes,
        )
    except ValueError as exc:
        typer.echo(f"The approval did not take effect: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc
    _render_run(runner, state)
    raise typer.Exit(code=EXIT_CODES[state.status])


FindingId = Annotated[
    str | None,
    typer.Option(
        "--finding",
        help="The finding's id, e.g. review.R-01.1 (see honest-ci evidence, or the evidence "
        "pack).",
    ),
]
FalsePositive = Annotated[
    bool, typer.Option("--false-positive", help="Mark the finding as a false positive.")
]
Confirmed = Annotated[
    bool,
    typer.Option(
        "--confirmed", help="Confirm the finding is not a false positive; a person checked it."
    ),
]
AnnotationReason = Annotated[
    str | None,
    typer.Option(
        "--reason",
        help="Why. Required with --false-positive: a false positive with no reason cannot "
        "improve a rule.",
    ),
]
Annotator = Annotated[str | None, typer.Option("--annotator", help="Who annotated it.")]
ReviewMinutes = Annotated[
    float | None,
    typer.Option(
        "--minutes",
        min=0.0,
        help="Minutes the human review took, recorded in the run's metrics in place of the "
        "last value.",
    ),
]


@app.command()
def annotate(
    run_id: str,
    finding: FindingId = None,
    false_positive: FalsePositive = False,
    confirmed: Confirmed = False,
    reason: AnnotationReason = None,
    annotator: Annotator = None,
    minutes: ReviewMinutes = None,
    workspace: Workspace = Path("."),
) -> None:
    """Annotate a run: mark a finding a false positive or confirmed, and record the review minutes.

    An annotation does not change a verdict. A blocked change stays blocked:
    letting it through takes a new commit or `honest-ci approve`, not an
    annotation. What it changes is the finding's `false_positive` in the
    evidence pack and the false-positive count in the metrics. It is bound to
    the evidence fingerprint of its time: a rerun that reaches the same
    fingerprint (the same commit, rule version and policy version) puts it back
    on the finding with the same id. Once the rule or policy version changes,
    it stays as history and no longer counts.
    """

    if finding is None and minutes is None:
        typer.echo(
            "Nothing to record: give --finding <id> with --false-positive or --confirmed, "
            "or --minutes N."
        )
        raise typer.Exit(code=EXIT_USAGE)
    if finding is None and (false_positive or confirmed):
        typer.echo("--false-positive and --confirmed are about one finding: add --finding <id>.")
        raise typer.Exit(code=EXIT_USAGE)
    if finding is not None:
        if false_positive == confirmed:
            typer.echo("--finding takes exactly one of --false-positive and --confirmed.")
            raise typer.Exit(code=EXIT_USAGE)
        if false_positive and not reason:
            typer.echo("A false positive needs --reason: revising the rule starts there.")
            raise typer.Exit(code=EXIT_USAGE)

    runner = _open_runner(workspace)
    try:
        if finding is not None:
            state = runner.annotate_finding(
                run_id,
                finding,
                false_positive=false_positive,
                reason=reason,
                annotator=annotator,
            )
            verdict = "false positive" if false_positive else "confirmed"
            typer.echo(
                f"Annotation: {finding} → {verdict}" + (f" ({annotator})" if annotator else "")
            )
            if reason:
                typer.echo(f"  Reason  : {reason}")
        if minutes is not None:
            state = runner.record_review_minutes(run_id, minutes)
            typer.echo(f"Review    : {minutes:g} minutes")
    except ValueError as exc:
        typer.echo(f"The annotation did not take effect: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from exc
    fp = state.metrics.false_positives
    typer.echo(
        f"Run       : {state.run_id}  {state.change.ref}  outcome {state.status} (unchanged)"
        f"  false positives {'not annotated' if fp is None else fp}"
    )


# ── reporting ────────────────────────────────────────────────────────────────


@app.command()
def status(run_id: str, workspace: Workspace = Path(".")) -> None:
    """Show the status of a run."""

    state = _runner(workspace).status(run_id)
    if state is None:
        typer.echo(f"No run found: {run_id}")
        raise typer.Exit(code=1)
    typer.echo(f"{state.change.ref}  status={state.status}  gate={state.current_gate}")
    for name, report in state.gate_reports.items():
        typer.echo(f"  {name:<11} {report.verdict.value:<12} {report.summary}")
    if state.last_feedback:
        typer.echo(f"  Last feedback: {state.last_feedback}")


@app.command()
def evidence(run_id: str, workspace: Workspace = Path(".")) -> None:
    """Render a run's evidence pack as Markdown a person can read, to standard output.

    It reads the run from the ledger and does not load the rule library: rules
    change, and an old run has to stay readable after they do.
    """

    settings = _settings()
    state = SQLiteLedger(workspace / settings.ledger_path).load_state(run_id)
    if state is None:
        typer.echo(f"No run found: {run_id}")
        raise typer.Exit(code=1)
    if state.evidence is None:
        typer.echo(f"Run {run_id} has no evidence pack yet: it did not reach a verdict.")
        raise typer.Exit(code=1)
    ticket = state.approval_ticket
    typer.echo(
        render_evidence(
            state.evidence,
            status=state.status,
            # The approval and the reading aid live on the run, not in the pack;
            # rendering here is what lets a signature show up next to the
            # evidence it was given for.
            approval=state.approval_decision,
            assistant_report=ticket.assistant_report if ticket is not None else None,
        )
    )


@app.command()
def inspect(run_id: str, workspace: Workspace = Path(".")) -> None:
    """Export a run's full record as JSON."""

    payload = _runner(workspace).inspect(run_id)
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@app.command()
def runs(limit: Limit = 20, workspace: Workspace = Path(".")) -> None:
    """List recent runs."""

    settings = _settings()
    ledger = SQLiteLedger(workspace / settings.ledger_path)
    records = ledger.list_run_records(limit=limit)
    if not records:
        typer.echo("(no runs yet)")
        return
    for record in records:
        blocked = record.metrics.gates_blocked
        typer.echo(
            f"{record.run_id}  {record.outcome:<11} {record.change_ref:<40} "
            f"gates_blocked={blocked} findings={record.metrics.findings_total}"
        )


# ── rendering ────────────────────────────────────────────────────────────────


def _render_source(
    source: intake.LocalChangeSource, change: ChangeRequest, workspace: Path
) -> None:
    """Say what is about to be judged, before anything judges it."""

    typer.echo(f"Source    : {source.origin}")
    if source.head_commit:
        branch = f" ({source.branch})" if source.branch else ""
        typer.echo(f"HEAD      : {source.head_commit[:12]}{branch}")
    paths = _declared_paths(change)
    typer.echo(f"Scope     : {_scope(change)}")
    if source.untracked_included:
        included = _counted(len(source.untracked_included), "new file")
        typer.echo(f"Untracked : {included} included in this run")
    if source.untracked_recorded:
        typer.echo(
            "⚠ Recorded, not reviewed: "
            + ", ".join(source.untracked_recorded)
            + " (binary, too large or unreadable)"
        )
    absent = [path for path in paths if not (workspace / path).is_file()]
    if absent:
        # The scanners read the workspace, not the patch. Files the patch names
        # but the tree does not have were never scanned, and a report that does
        # not say so reads like a clean bill of health.
        typer.echo(
            f"⚠ Missing from the working tree: {_counted(len(absent), 'changed file')} "
            f"({', '.join(absent[:5])}{' …' if len(absent) > 5 else ''}). The scanners had "
            "nothing of them to scan."
        )


def _scope(change: ChangeRequest) -> str:
    """How big the change is: the files it touches and the lines it changes.

    The same reading the structural guardrails take, so the numbers an
    operator sees are the ones the blast-radius limits are compared against.
    """

    return (
        f"{_counted(len(_scope_paths(change)), 'file')}, "
        f"{_counted(change.changed_lines, 'line')}"
    )


def _scope_paths(change: ChangeRequest) -> list[str]:
    """Every path the change touches, deletions included.

    Deliberately not ``_declared_paths``: that one answers "what is there to
    scan", so it drops deletions. This one is the size of the change, and it
    is read against the blast-radius limits, which count a deleted file as a
    changed file. Reading them differently printed "0 files, 43 lines" for a
    change that deleted a test file -- a scope line that denies the very thing
    the run then went to a human about. A rename counts under both names, as
    the limits count it.
    """

    return change.touched_paths


def _declared_paths(change: ChangeRequest) -> list[str]:
    paths = [item.path for item in change.files_changed if item.change_type != "deleted"]
    declared = paths or extract_changed_files(change.diff)
    return [path for path in declared if is_safe_relative_path(path)]


def _render_run(runner: GateRunner, state: RunState, *, ledger_kept: bool = True) -> None:
    typer.echo(f"Run       : {state.run_id}  {state.change.ref}")
    typer.echo("Gates     :")
    for name in GATE_SEQUENCE:
        report = state.gate_reports.get(name)
        if report is None:
            # An upstream gate reached a terminal verdict, so this one was never
            # asked. "Not run" and "passed" must not look alike.
            typer.echo(
                f"  {name:<11} {'not run':<12} an earlier gate reached a verdict, so this one "
                "was not reached"
            )
            continue
        # An undeployed gate's own summary is written for a developer reading
        # the evidence pack; an operator needs the product-level reading of it,
        # which is that skipped is not passed.
        summary = (
            report.summary
            if name in GATE_REGISTRY
            else "not installed in this deployment; recorded as skipped, which is not a pass"
        )
        typer.echo(f"  {name:<11} {report.verdict.value:<12} {summary}")
    _render_findings(state)
    _render_model_text(state)
    decision = state.policy_decision
    if decision is not None and not decision.allowed:
        typer.echo("Guardrails: " + "; ".join(decision.violations))
    missing = unsatisfied_required_checks(state)
    if missing:
        typer.echo(
            f"Required gates without a verdict: {', '.join(missing)} "
            "(so the outcome is needs_human)"
        )
    typer.echo(f"Outcome   : {state.status} — {_outcome_hint(state, ledger_kept=ledger_kept)}")
    pack = runner.evidence_path(state)
    if pack is not None:
        typer.echo(f"Evidence  : {pack}")
    readable = runner.summary_path(state)
    if readable is not None:
        typer.echo(f"Readable  : {readable}")


def _outcome_hint(state: RunState, *, ledger_kept: bool = True) -> str:
    """Why this run ended where it did.

    An approved run is ``mergeable`` because a person signed for it, not because
    the gates verified it — and a terminal line that says otherwise would put a
    human's judgment behind an automated-sounding verdict.

    A run waiting on a person also gets the command that signs for it, but only
    where the ledger that command reads outlives this process. The shared hint
    names no command because the pull request comment is read far from any
    ledger.
    """

    decision = state.approval_decision
    if decision is None:
        hint = STATUS_HINT[state.status]
        if state.status == "awaiting_human" and ledger_kept:
            hint += f" To sign off: honest-ci approve {state.run_id} --reason ..."
        return hint
    verb = "Approved" if decision.approved else "Rejected"
    who = decision.reviewer or "an unnamed reviewer"
    return (
        f"{verb} by {who}, bound to fingerprint {decision.fingerprint[:12]}. "
        "No gate's verdict changed."
    )


def _render_findings(state: RunState) -> None:
    findings = [
        finding for report in state.gate_reports.values() for finding in report.findings
    ]
    blocking = [finding for finding in findings if finding.blocking]
    if blocking:
        typer.echo("Blocking  :")
        for finding in blocking:
            typer.echo(f"  {_finding_line(finding)}")
            if finding.suggested_fix:
                typer.echo(f"      Fix: {finding.suggested_fix}")
    # Counted under "other" these would read as "does not block", which is
    # true and beside the point: they are why the run is waiting on a person,
    # and the first line of each says which test.
    waiting = [
        finding
        for finding, action in integrity_findings(list(state.gate_reports.values()))
        if action == "needs_human"
    ]
    if waiting:
        typer.echo("Test integrity (needs a human):")
        for finding in waiting:
            typer.echo(f"  {_finding_line(finding)}")
            first = next(iter(finding.detail.splitlines()), "")
            typer.echo(f"      {printable_path(_one_line(first))}")
    other = len(findings) - len(blocking) - len(waiting)
    if other:
        typer.echo(
            f"Other     : {_counted(other, 'non-blocking finding')}, recorded in the "
            "evidence pack and the metrics"
        )


def _render_model_text(state: RunState) -> None:
    """Print what the model said — after the findings, never instead of them.

    Every line here is marked advisory because that is what it is: the brief
    only restates evidence the gate produced, and anything that cited something
    else was dropped before it reached this state (see ``grounding.py``).
    """

    for brief in state.briefs:
        typer.echo(
            f"Model     : ({brief.gate}, advisory only; the verdict stands) "
            f"{_one_line(brief.summary)}"
        )
        for action in brief.recommended_actions[:3]:
            typer.echo(f"      Suggests: {_one_line(action)}")
    ticket = state.approval_ticket
    aid = ticket.assistant_report if ticket is not None else None
    if aid is not None:
        typer.echo(f"Reading aid: (model, advisory only) {_one_line(aid.change_summary)}")
        for question in aid.review_questions[:3]:
            typer.echo(f"      To check: {_one_line(question)}")


def _one_line(text: str, limit: int = 160) -> str:
    """Collapse advisory prose into one terminal line."""

    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "…"


def _finding_line(finding: Finding) -> str:
    rule = f"[{finding.rule_id}] " if finding.rule_id else ""
    # A path rule fires on a name alone, including one refused as a path.
    where = printable_path(finding.file or "")
    if finding.file and finding.line_start:
        where = f"{where}:{finding.line_start}"
    if finding.file and finding.side == "base":
        where = f"{where} (base)"
    return f"{rule}{finding.severity:<8} {where}  {finding.title}"


if __name__ == "__main__":  # pragma: no cover
    app()

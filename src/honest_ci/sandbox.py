"""Reproducible dependency provisioning for the regression gate.

In docker mode the sandbox runs against a bare image without the target repo's
dependencies. ``SandboxBuilder.ensure_deps`` builds an isolated venv inside the
mounted sandbox; the install step is the ONLY step granted network access,
while tests and scanners always run with the network denied. In process mode it
is a no-op (the host interpreter already carries what it needs).
"""

from __future__ import annotations

import hashlib
import tomllib
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

from .artifacts import ArtifactStore
from .contracts import ToolReport
from .maven import (
    MAVEN_PLUGINS,
    MAVEN_RUNTIME_ARTIFACTS,
    MAVEN_STAMP,
    maven_command,
    maven_env,
    maven_root,
)
from .redaction import redact_text
from .settings import Settings
from .tools import ToolRunner
from .utils import is_runtime_path

VENV_PYTHON = ".venv/bin/python"
VENV_PIP = ".venv/bin/pip"
DEPENDENCY_STAMP = ".venv/.honest-ci-dependencies.sha256"
TEST_EXTRA_NAMES = ("test", "tests", "testing", "dev")


class DependencyProvisionError(RuntimeError):
    """The isolated test environment could not be prepared."""


class PythonSandboxBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _requirements_file(self, sandbox: Path) -> Path | None:
        for name in ("requirements.txt", "requirements-dev.txt"):
            if (sandbox / name).exists():
                return sandbox / name
        return None

    def has_dependencies(self, sandbox: Path) -> bool:
        return (sandbox / "pyproject.toml").exists() or self._requirements_file(sandbox) is not None

    def _editable_target(self, pyproject: Path) -> str:
        """Select a declared test/development extra when the project provides one."""

        try:
            payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            # Let pip report malformed or unsupported project metadata in its
            # normal build log; parsing here exists only to select an extra.
            return "."
        project = payload.get("project")
        if not isinstance(project, dict):
            return "."
        optional = project.get("optional-dependencies")
        if not isinstance(optional, dict):
            return "."
        for extra in TEST_EXTRA_NAMES:
            dependencies = optional.get(extra)
            if isinstance(dependencies, list) and dependencies:
                return f".[{extra}]"
        return "."

    def _install_command(self, sandbox: Path, index_args: list[str]) -> list[str]:
        pyproject = sandbox / "pyproject.toml"
        if pyproject.exists():
            return [VENV_PIP, "install", *index_args, "-e", self._editable_target(pyproject)]
        requirements = self._requirements_file(sandbox)
        if requirements is None:  # pragma: no cover - guarded by has_dependencies
            raise DependencyProvisionError("Sandbox has no dependency manifest to install.")
        return [VENV_PIP, "install", *index_args, "-r", requirements.name]

    def _dependency_fingerprint(self, sandbox: Path, install: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.settings.docker_image.encode("utf-8"))
        digest.update(b"\0")
        digest.update("\0".join(install).encode("utf-8"))
        for name in ("pyproject.toml", "requirements.txt", "requirements-dev.txt"):
            manifest = sandbox / name
            if not manifest.exists():
                continue
            digest.update(b"\0")
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(manifest.read_bytes())
        return digest.hexdigest()

    def _raise_provision_error(self, action: str, report: ToolReport) -> None:
        location = f" See {report.report_path}." if report.report_path else ""
        raise DependencyProvisionError(
            f"Sandbox dependency {action} failed with exit code {report.exit_code}.{location}"
        )

    def ensure_deps(self, sandbox: Path, artifacts_path: Path, mode: str) -> str | None:
        """Provision an isolated venv in docker mode; return the in-container python.

        Returns ``None`` when no venv was prepared (process mode, disabled, or no
        dependency manifest), in which case callers fall back to the default
        interpreter.
        """

        if not self.settings.sandbox_build_deps or mode != "docker":
            return None
        if not self.has_dependencies(sandbox):
            return None
        index_args: list[str] = []
        if self.settings.pip_index_url:
            index_args = ["--index-url", self.settings.pip_index_url]
        install = self._install_command(sandbox, index_args)
        fingerprint = self._dependency_fingerprint(sandbox, install)
        stamp = sandbox / DEPENDENCY_STAMP
        venv_ready = (sandbox / ".venv" / "pyvenv.cfg").is_file()
        if (
            venv_ready
            and stamp.is_file()
            and stamp.read_text(encoding="utf-8").strip() == fingerprint
        ):
            return VENV_PYTHON
        runner = ToolRunner(
            sandbox_path=sandbox,
            artifacts_path=artifacts_path,
            mode=mode,
            docker_image=self.settings.docker_image,
            timeout_seconds=self.settings.provision_timeout_seconds,
            docker_cpus=self.settings.docker_cpus,
            docker_memory=self.settings.docker_memory,
            docker_pids_limit=self.settings.docker_pids_limit,
        )
        if not venv_ready:
            created = runner.run(
                "venv-create",
                ["python", "-m", "venv", ".venv"],
                "venv-create.log",
                allow_network=False,
            )
            if not created.passed:
                self._raise_provision_error("virtualenv creation", created)
        # The install step is the only place network access is granted.
        installed = runner.run("pip-install", install, "pip-install.log", allow_network=True)
        if not installed.passed:
            self._raise_provision_error("installation", installed)
        stamp.write_text(f"{fingerprint}\n", encoding="utf-8")
        return VENV_PYTHON


# Keep the original Python entry point for existing deployments and adapters.
SandboxBuilder = PythonSandboxBuilder


class JavaSandboxBuilder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _dependency_fingerprint(self, sandbox: Path) -> str:
        digest = hashlib.sha256()
        for value in (
            self.settings.java_docker_image,
            self.settings.maven_mirror or "",
            *MAVEN_PLUGINS.values(),
            *self._prefetch(),
        ):
            digest.update(value.encode("utf-8") + b"\0")
        for manifest in sorted(sandbox.rglob("pom.xml")):
            relative = manifest.relative_to(sandbox)
            if is_runtime_path(relative) or "target" in relative.parts:
                continue
            if not manifest.resolve().is_relative_to(sandbox.resolve()):
                raise DependencyProvisionError(
                    "A pom.xml points outside the workspace, so dependencies cannot be provisioned."
                )
            digest.update(relative.as_posix().encode("utf-8") + b"\0")
            digest.update(manifest.read_bytes() + b"\0")
        return digest.hexdigest()

    def ensure_deps(self, sandbox: Path, artifacts_path: Path, mode: str) -> None:
        if mode != "docker" or not self.settings.sandbox_build_deps:
            return
        if not (sandbox / "pom.xml").is_file():
            return
        try:
            self._provision(sandbox, artifacts_path, mode)
        except OSError as exc:
            raise DependencyProvisionError(
                "The Maven provisioning files cannot be read or written."
            ) from exc

    def _provision(self, sandbox: Path, artifacts_path: Path, mode: str) -> None:
        stamp = sandbox / MAVEN_STAMP
        # A checkout may contain symlinks; provisioning must never write into
        # a host directory reached through its runtime cache.
        for relative in (
            ".honest-ci",
            ".honest-ci/m2",
            ".honest-ci/m2-config",
            MAVEN_STAMP,
            ".honest-ci/m2/manifest.json",
            ".honest-ci/m2-config/manifest.json",
        ):
            path = sandbox / relative
            if path.is_symlink() or not path.resolve().is_relative_to(sandbox.resolve()):
                raise DependencyProvisionError(
                    "The Maven cache path is unsafe, so dependencies cannot be provisioned."
                )
        fingerprint = self._dependency_fingerprint(sandbox)
        if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == fingerprint:
            return
        stamp.unlink(missing_ok=True)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        mirror_args, mirror_file = self._mirror_settings(sandbox, mode)
        runner = ToolRunner(
            sandbox,
            artifacts_path,
            mode=mode,
            docker_image=self.settings.java_docker_image,
            timeout_seconds=self.settings.provision_timeout_seconds,
            docker_cpus=self.settings.docker_cpus,
            docker_memory=self.settings.docker_memory,
            docker_pids_limit=self.settings.docker_pids_limit,
        )
        dependency = MAVEN_PLUGINS["dependency"]
        goals = [
            [f"{dependency}:go-offline", f"{dependency}:resolve-plugins"],
            *[
                [f"{dependency}:get", f"-Dartifact={artifact}"]
                for artifact in (*MAVEN_PLUGINS.values(), *self._prefetch())
            ],
        ]
        try:
            for index, goal in enumerate(goals):
                report = runner.run(
                    "maven-dependencies",
                    [*maven_command(sandbox, mode, offline=False), *mirror_args, *goal],
                    f"maven-dependencies-{index}.log",
                    allow_network=True,
                    env=maven_env(sandbox, mode),
                )
                if not report.available or not report.passed:
                    raise DependencyProvisionError(
                        " ".join(
                            part
                            for part in (
                                "Provisioning Maven dependencies and plugins failed "
                                f"(exit code {report.exit_code}), so the offline checks are "
                                "unavailable.",
                                report.stderr or report.stdout,
                                _provision_hint(report),
                                f"See {report.report_path}." if report.report_path else "",
                            )
                            if part
                        )
                    )
            ArtifactStore(sandbox / ".honest-ci").write_text("m2", stamp.name, fingerprint + "\n")
        finally:
            if mirror_file is not None:
                mirror_file.unlink(missing_ok=True)

    def _prefetch(self) -> tuple[str, ...]:
        """Coordinates ``go-offline`` cannot find, so provisioning asks for them by name.

        Ours first (a pinned plugin's own run-time pick), then the
        deployment's. Both go into the fingerprint above: adding a coordinate
        has to re-provision, or the run that needed it would read the stamp
        and stay offline with the same gap.
        """

        return (*MAVEN_RUNTIME_ARTIFACTS.values(), *self.settings.maven_extra_artifacts)

    def _mirror_settings(self, sandbox: Path, mode: str) -> tuple[list[str], Path | None]:
        mirror = self.settings.maven_mirror
        if not mirror:
            return [], None
        try:
            parsed = urllib.parse.urlsplit(mirror)
        except ValueError as exc:
            raise DependencyProvisionError(
                "HONEST_CI_MAVEN_MIRROR is not a valid HTTP(S) URL."
            ) from exc
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or redact_text(mirror) != mirror
        ):
            raise DependencyProvisionError(
                "HONEST_CI_MAVEN_MIRROR must be an HTTP(S) URL with no credentials and no "
                "query string."
            )
        root = ET.Element("settings")
        item = ET.SubElement(ET.SubElement(root, "mirrors"), "mirror")
        for key, value in (("id", "honest-ci-mirror"), ("mirrorOf", "*"), ("url", mirror)):
            ET.SubElement(item, key).text = value
        path = sandbox / ".honest-ci/m2-config/provision-settings.xml"
        if path.is_symlink():
            raise DependencyProvisionError("The Maven mirror settings path is unsafe.")
        ArtifactStore(sandbox / ".honest-ci").write_text(
            "m2-config", path.name, ET.tostring(root, encoding="unicode")
        )
        return ["-s", f"{maven_root(sandbox, mode)}/.honest-ci/m2-config/{path.name}"], path


def _provision_hint(report: ToolReport) -> str:
    """The remedy for the one provisioning failure that is not about the repo.

    A cold Maven cache pulls the whole toolchain -- ``go-offline`` over a
    single-module JUnit project fetches past 100 MB before it resolves the
    first plugin -- and on a slow link it outruns any timeout picked for a
    scanner. The operator then reads "provisioning failed (exit code 124)"
    and starts looking for a broken pom. Exit code 124 is this runner's own
    timeout, so it is the one case where the fix is a setting and we can name
    it.
    """

    if report.exit_code != 124:
        return ""
    return (
        "Exit code 124 is the provisioning timeout, not a problem with the repository: the "
        "first provisioning on a cold cache downloads the whole toolchain. Raise "
        "HONEST_CI_PROVISION_TIMEOUT_SECONDS and run again, or point HONEST_CI_MAVEN_MIRROR "
        "at an internal mirror."
    )

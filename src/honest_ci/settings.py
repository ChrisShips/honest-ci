"""Runtime configuration.

Product settings use the ``HONEST_CI_`` prefix; provider credentials keep their
vendor-standard variable names so an existing private deployment does not have
to re-issue keys. Every default here is chosen for a verification layer, which
means the safe direction is *blocking*: ``fail_closed`` is on, the sandbox is
Docker, and tools that cannot run are never treated as tools that passed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .maven import ARTIFACT_COORDINATE

#: Providers the structured model layer can bind to. None of them is a
#: default: ``HONEST_CI_MODEL_PROVIDER`` has to name one before any model is
#: asked anything.
MODEL_PROVIDERS = ("deepseek", "qwen", "glm", "openai", "anthropic")

#: The body field an OpenAI-compatible server reads the output cap from.
#: OpenAI's reasoning models accept only ``max_completion_tokens``; most other
#: compatible servers -- Ollama, DeepSeek -- read ``max_tokens``.
MaxTokensParam = Literal["max_completion_tokens", "max_tokens"]

#: Said wherever a model would have been asked and nobody switched one on --
#: ``doctor``, the evidence pack, the merge-request comment -- so all of them
#: name the same switch.
MODEL_NOT_ENABLED = "model not enabled: HONEST_CI_MODEL_PROVIDER is unset"


class SettingsError(ValueError):
    """Configuration nobody can act on. Not retryable: someone must fix it."""


@dataclass(frozen=True)
class Settings:
    # ── runtime state ────────────────────────────────────────────────────
    ledger_path: Path = Path(".honest-ci") / "ledger.sqlite3"
    artifacts_dir: Path = Path(".honest-ci") / "artifacts"
    workspace_dir: Path = Path(".honest-ci") / "workspaces"

    # ── policy & rules ───────────────────────────────────────────────────
    policy_version: str = "v1"
    rule_version: str = "v1"
    policy_path: Path = Path("honest-ci-policy.toml")
    rules_path: Path = Path("honest-ci-rules.toml")

    # ── orchestration ────────────────────────────────────────────────────
    max_attempts: int = 3
    gate_max_attempts: dict[str, int] = field(default_factory=dict)
    retry_backoff_seconds: tuple[float, ...] = (0.0, 1.0, 5.0)
    retry_jitter_seconds: float = 0.0
    #: When a required tool is unavailable the gate blocks instead of passing.
    #: This is the difference between a demo and a verification layer; turn it
    #: off only for local debugging.
    fail_closed: bool = True
    #: Deployed gates this installation switches off. A gate named here is
    #: recorded as ``skipped`` with the reason -- never as passed -- and the
    #: policy's ``required_checks`` still applies, so switching off a gate the
    #: policy requires routes every change to a human instead of through.
    disabled_gates: tuple[str, ...] = ()

    # ── sandbox & tools ──────────────────────────────────────────────────
    #: Which toolchain judges a change. ``auto`` reads the workspace's build
    #: markers (``pyproject.toml`` and friends say python; each toolchain
    #: declares its own) and falls back to the first registered toolchain when
    #: none is there; a language name forces one. See ``adapters.scanners``.
    language: str = "auto"
    sandbox_mode: str = "docker"
    docker_image: str = "python:3.12-slim"
    java_docker_image: str = "maven:3.9-eclipse-temurin-21"
    maven_mirror: str | None = None
    #: Extra Maven coordinates provisioning pre-fetches, beyond the project's
    #: declared dependencies and our pinned plugins. Maven decides some
    #: artifacts while a goal runs -- Surefire picks its test provider from the
    #: frameworks it finds, at the surefire version the project chose -- and
    #: ``go-offline`` cannot see those, so the first offline run fails on a
    #: download nobody could have predicted. This is the list that closes that
    #: gap, and it is a deployment setting rather than a repository file
    #: because it describes what *this machine* must hold to run offline.
    #: Extra entries never hurt, so a deployment serving several repositories
    #: takes the union.
    maven_extra_artifacts: tuple[str, ...] = ()
    #: This machine's own Maven repository, which a process-mode Java run reads
    #: behind the workspace's ``.honest-ci/m2`` and never writes to
    #: (``maven.repo.local.tail``, Maven 3.9 or later). Process mode provisions
    #: nothing, so without it ``mvn -o`` had an empty directory to resolve
    #: from and every Java run blocked at its first plugin. ``None`` reads
    #: nothing; it is the default here so that a run's commands never depend
    #: on what the machine running a test suite holds, and ``from_env`` makes
    #: it ``~/.m2/repository``. A docker run never reads it: the container
    #: cannot see the host.
    maven_host_repository: Path | None = None
    docker_cpus: str = "2"
    docker_memory: str = "2g"
    docker_pids_limit: int = 256
    command_timeout_seconds: int = 300
    #: How long dependency provisioning may take, apart from the per-scanner
    #: timeout. Same reasoning as ``test_timeout_seconds`` one field below, at
    #: a different scale: provisioning is the only step that is allowed the
    #: network, and on a cold cache it downloads a toolchain -- Maven's
    #: ``go-offline`` over a small JUnit project fetches well past 100 MB
    #: before it has resolved a single plugin. Killing it at the scanner limit
    #: does not report a slow scanner, it reports every Java tool as
    #: unavailable, which fails closed: the whole repository blocks because
    #: nothing was ever installed.
    provision_timeout_seconds: int = 1800
    sandbox_build_deps: bool = True
    sandbox_keep: bool = False
    pip_index_url: str | None = None
    #: The regression gate's test command, run exactly as written. ``None``
    #: means pytest, wrapped in ``coverage run`` when a coverage figure is
    #: needed and the module is there.
    test_command: str | None = None
    #: JUnit XML a custom test command leaves behind, as a glob relative to the
    #: workspace (``reports/junit.xml``, ``**/target/surefire-reports/*.xml``).
    #: With it the gate reads counts and failing test names out of a suite that
    #: is not pytest; without it a custom command is only its exit code. Not
    #: consulted for the default command, whose output the adapter reads itself.
    test_report: str | None = None
    #: Line coverage the whole suite must reach; 0 switches the check off
    #: explicitly. A bar this deployment cannot measure blocks rather than
    #: passes, so a bar above zero is a statement that coverage is installed.
    coverage_threshold: float = 0.70
    #: Coverage the lines *this change added* must reach; 0 switches it off.
    #: Separate from the whole-project bar because they answer different
    #: questions: a repository at 40% cannot raise its total on one merge
    #: request, but it can be asked to test what it just wrote. Unmeasurable
    #: blocks, exactly like the bar above.
    changed_line_coverage_threshold: float = 0.0
    #: Which tests the regression gate runs: ``all`` (the repository's whole
    #: suite) or ``impacted`` (only the tests impact analysis reached). The
    #: default is ``all`` and an unrecognised value falls back to it, because
    #: ``impacted`` is deliberately weaker evidence -- a typo must not quietly
    #: shrink what a deployment verifies. ``impacted`` narrows only with
    #: ``coverage_threshold`` at 0 and the toolchain's own test command: the
    #: overall bar is a figure about the whole suite, and a command the
    #: toolchain did not compose may not take a selection.
    test_selection: str = "all"
    #: How long the suite may take, apart from the per-scanner timeout: a lint
    #: finishes in seconds, a suite can legitimately take twenty minutes, and a
    #: suite killed at the scanner limit would block every change for being slow.
    test_timeout_seconds: int = 1800

    # ── models ───────────────────────────────────────────────────────────
    #: The provider every model call goes to; ``None`` means no model is
    #: asked anything. Deliberately not inferred from whichever key happens to
    #: be set: CI runners and developer shells carry keys for unrelated tools,
    #: and treating one of them as consent would start sending diffs to that
    #: vendor the day the key appeared. Naming a provider here is the consent.
    model_provider: str | None = None
    prompt_version: str = "v1"
    #: Whether the model is asked for advisory text at all (block briefs and the
    #: reading aid on a human hand-off). Separate from "is a credential set"
    #: because explaining a block sends the cited evidence -- which includes the
    #: offending lines -- to the provider. Turning it off does not keep change
    #: content in-house on its own: ``model_judgment`` and ``model_testgen``
    #: send the diff too. A deployment that must keep it in-house turns all
    #: three off, or leaves ``model_provider`` unset, and keeps every
    #: deterministic verdict either way.
    model_explain: bool = True
    #: Whether ``judgment`` rules are put to the model at all. Separate from
    #: ``model_explain`` because the two send different things and carry
    #: different risk: an explanation restates a verdict already reached, while
    #: a judgment is the only way those rules get reviewed. Turning it off does
    #: not skip them -- they are recorded as unreviewed and the change goes to
    #: a human, because "we did not ask" is not "there was nothing to find".
    model_judgment: bool = True
    #: Whether the regression gate asks the model to write regression cases for
    #: the symbols this change touched. Separate again, and for a reason the
    #: other two do not have: this is the only model output that becomes an
    #: executable file in the tree under test. Turning it off costs nothing a
    #: verdict depends on -- generated cases never block, so this switch only
    #: decides whether the evidence pack gains cases a human can keep.
    model_testgen: bool = True
    #: Default confidence a model finding must reach to be recorded as a
    #: finding. Below it the doubt is still written down, under "What this run
    #: could not verify" rather than among the findings a person is asked to
    #: rule on -- guesses the model itself doubts would bury the ones worth
    #: ruling on.
    model_min_confidence: float = 0.70
    deepseek_base_url: str | None = None
    deepseek_model: str = "deepseek-chat"
    deepseek_api_key: str | None = None
    deepseek_max_tokens: int = 8192
    deepseek_temperature: float = 0.1
    deepseek_timeout_seconds: int = 120
    deepseek_max_retries: int = 2
    qwen_base_url: str | None = None
    qwen_model: str = "qwen3.8-max"
    qwen_api_key: str | None = None
    qwen_max_tokens: int = 8192
    qwen_temperature: float = 0.1
    qwen_timeout_seconds: int = 120
    qwen_max_retries: int = 2
    qwen_enable_thinking: bool | None = None
    glm_protocol: str = "anthropic"
    glm_base_url: str | None = None
    glm_model: str = "glm-5.2"
    glm_api_key: str | None = None
    glm_max_tokens: int = 4096
    glm_temperature: float = 0.1
    glm_timeout_seconds: int = 60
    glm_max_retries: int = 2
    #: Anything that speaks OpenAI's chat-completions wire, not only OpenAI:
    #: DeepSeek's compatible endpoint, a local Ollama, a company gateway. So
    #: everything those servers disagree on is a setting, not a guess.
    openai_base_url: str | None = None
    openai_model: str = "gpt-6-astra"
    openai_api_key: str | None = None
    #: Reasoning counts against this cap, so it is sized for the thinking as
    #: well as the JSON that follows it.
    openai_max_tokens: int = 16000
    openai_max_tokens_param: MaxTokensParam = "max_completion_tokens"
    #: ``None`` leaves ``temperature`` out of the request, which is the only
    #: thing a reasoning model accepts.
    openai_temperature: float | None = None
    #: A non-streaming reply arrives only after the model finished reasoning,
    #: so this is sized for a long think rather than a slow network.
    openai_timeout_seconds: int = 300
    openai_max_retries: int = 2
    anthropic_base_url: str | None = None
    anthropic_model: str = "claude-opus-5"
    anthropic_api_key: str | None = None
    #: Current Claude models think by default, and ``max_tokens`` caps the
    #: thinking and the answer together.
    anthropic_max_tokens: int = 16000
    #: ``None`` leaves ``temperature`` out: current Claude models reject it.
    anthropic_temperature: float | None = None
    anthropic_timeout_seconds: int = 300
    anthropic_max_retries: int = 2

    # ── forge ────────────────────────────────────────────────────────────
    forge: str = "local"
    github_base_url: str = "https://api.github.com"
    github_token: str | None = None
    target_branch: str = "main"
    forge_timeout_seconds: int = 30
    #: The commit-status context a team puts in its merge rules. It names this
    #: product rather than a gate, because one run writes back one conclusion.
    forge_status_name: str = "honest-ci/verify"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ledger_path=Path(os.getenv("HONEST_CI_LEDGER_PATH", ".honest-ci/ledger.sqlite3")),
            artifacts_dir=Path(os.getenv("HONEST_CI_ARTIFACTS_DIR", ".honest-ci/artifacts")),
            workspace_dir=Path(os.getenv("HONEST_CI_WORKSPACE_DIR", ".honest-ci/workspaces")),
            policy_version=os.getenv("HONEST_CI_POLICY_VERSION", "v1"),
            rule_version=os.getenv("HONEST_CI_RULE_VERSION", "v1"),
            policy_path=Path(os.getenv("HONEST_CI_POLICY_PATH", "honest-ci-policy.toml")),
            rules_path=Path(os.getenv("HONEST_CI_RULES_PATH", "honest-ci-rules.toml")),
            max_attempts=_int_env("HONEST_CI_MAX_ATTEMPTS", 3),
            gate_max_attempts=_parse_gate_attempts(os.getenv("HONEST_CI_GATE_MAX_ATTEMPTS", "")),
            retry_backoff_seconds=_parse_backoff(os.getenv("HONEST_CI_RETRY_BACKOFF_SECONDS", "0,1,5")),
            retry_jitter_seconds=_float_env("HONEST_CI_RETRY_JITTER_SECONDS", 0.0),
            fail_closed=os.getenv("HONEST_CI_FAIL_CLOSED", "1") != "0",
            disabled_gates=_parse_names(os.getenv("HONEST_CI_DISABLED_GATES", "")),
            language=os.getenv("HONEST_CI_LANGUAGE", "auto").strip().lower() or "auto",
            sandbox_mode=os.getenv("HONEST_CI_SANDBOX_MODE", "docker"),
            docker_image=os.getenv("HONEST_CI_DOCKER_IMAGE", "python:3.12-slim"),
            java_docker_image=os.getenv("HONEST_CI_JAVA_DOCKER_IMAGE", "maven:3.9-eclipse-temurin-21"),
            maven_mirror=os.getenv("HONEST_CI_MAVEN_MIRROR") or None,
            maven_extra_artifacts=_parse_artifacts(
                os.getenv("HONEST_CI_MAVEN_EXTRA_ARTIFACTS", "")
            ),
            maven_host_repository=_parse_host_repository(
                os.getenv("HONEST_CI_MAVEN_HOST_REPOSITORY")
            ),
            docker_cpus=os.getenv("HONEST_CI_DOCKER_CPUS", "2"),
            docker_memory=os.getenv("HONEST_CI_DOCKER_MEMORY", "2g"),
            docker_pids_limit=_int_env("HONEST_CI_DOCKER_PIDS_LIMIT", 256),
            command_timeout_seconds=_int_env("HONEST_CI_COMMAND_TIMEOUT_SECONDS", 300),
            provision_timeout_seconds=_int_env("HONEST_CI_PROVISION_TIMEOUT_SECONDS", 1800),
            sandbox_build_deps=os.getenv("HONEST_CI_SANDBOX_BUILD_DEPS", "1") == "1",
            sandbox_keep=os.getenv("HONEST_CI_SANDBOX_KEEP", "0") == "1",
            pip_index_url=os.getenv("HONEST_CI_PIP_INDEX_URL") or None,
            test_command=os.getenv("HONEST_CI_TEST_COMMAND") or None,
            test_report=os.getenv("HONEST_CI_TEST_REPORT") or None,
            coverage_threshold=_ratio_env("HONEST_CI_COVERAGE_THRESHOLD", 0.70),
            changed_line_coverage_threshold=_ratio_env(
                "HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD", 0.0
            ),
            test_selection=_parse_selection(os.getenv("HONEST_CI_TEST_SELECTION", "")),
            test_timeout_seconds=_int_env("HONEST_CI_TEST_TIMEOUT_SECONDS", 1800),
            model_provider=_parse_provider(os.getenv("HONEST_CI_MODEL_PROVIDER", "")),
            prompt_version=os.getenv("HONEST_CI_PROMPT_VERSION", "v1"),
            model_explain=os.getenv("HONEST_CI_MODEL_EXPLAIN", "1") != "0",
            model_judgment=os.getenv("HONEST_CI_MODEL_JUDGMENT", "1") != "0",
            model_testgen=os.getenv("HONEST_CI_MODEL_TESTGEN", "1") != "0",
            model_min_confidence=_ratio_env("HONEST_CI_MODEL_MIN_CONFIDENCE", 0.70),
            deepseek_base_url=os.getenv("HONEST_CI_DEEPSEEK_BASE_URL") or None,
            deepseek_model=os.getenv("HONEST_CI_DEEPSEEK_MODEL", "deepseek-chat"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_max_tokens=_int_env("HONEST_CI_DEEPSEEK_MAX_TOKENS", 8192),
            deepseek_temperature=_float_env("HONEST_CI_DEEPSEEK_TEMPERATURE", 0.1),
            deepseek_timeout_seconds=_int_env("HONEST_CI_DEEPSEEK_TIMEOUT_SECONDS", 120),
            deepseek_max_retries=_int_env("HONEST_CI_DEEPSEEK_MAX_RETRIES", 2),
            qwen_base_url=os.getenv("HONEST_CI_QWEN_BASE_URL") or None,
            qwen_model=os.getenv("HONEST_CI_QWEN_MODEL", "qwen3.8-max"),
            qwen_api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"),
            qwen_max_tokens=_int_env("HONEST_CI_QWEN_MAX_TOKENS", 8192),
            qwen_temperature=_float_env("HONEST_CI_QWEN_TEMPERATURE", 0.1),
            qwen_timeout_seconds=_int_env("HONEST_CI_QWEN_TIMEOUT_SECONDS", 120),
            qwen_max_retries=_int_env("HONEST_CI_QWEN_MAX_RETRIES", 2),
            qwen_enable_thinking=_parse_optional_bool(os.getenv("HONEST_CI_QWEN_ENABLE_THINKING")),
            glm_protocol=os.getenv("HONEST_CI_GLM_PROTOCOL", "anthropic"),
            glm_base_url=os.getenv("HONEST_CI_GLM_BASE_URL") or None,
            glm_model=os.getenv("HONEST_CI_GLM_MODEL", "glm-5.2"),
            glm_api_key=os.getenv("ZAI_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"),
            glm_max_tokens=_int_env("HONEST_CI_GLM_MAX_TOKENS", 4096),
            glm_temperature=_float_env("HONEST_CI_GLM_TEMPERATURE", 0.1),
            glm_timeout_seconds=_int_env("HONEST_CI_GLM_TIMEOUT_SECONDS", 60),
            glm_max_retries=_int_env("HONEST_CI_GLM_MAX_RETRIES", 2),
            # Only the HONEST_CI_ names select a host. The vendors' own
            # OPENAI_BASE_URL / ANTHROPIC_BASE_URL are left to the tools that
            # exported them: picking one up would send this product's key and
            # diffs wherever some other tool was pointed.
            openai_base_url=os.getenv("HONEST_CI_OPENAI_BASE_URL") or None,
            openai_model=os.getenv("HONEST_CI_OPENAI_MODEL", "gpt-6-astra"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_max_tokens=_int_env("HONEST_CI_OPENAI_MAX_TOKENS", 16000),
            openai_max_tokens_param=_parse_max_tokens_param(
                os.getenv("HONEST_CI_OPENAI_MAX_TOKENS_PARAM", "")
            ),
            openai_temperature=_optional_float_env("HONEST_CI_OPENAI_TEMPERATURE"),
            openai_timeout_seconds=_int_env("HONEST_CI_OPENAI_TIMEOUT_SECONDS", 300),
            openai_max_retries=_int_env("HONEST_CI_OPENAI_MAX_RETRIES", 2),
            anthropic_base_url=os.getenv("HONEST_CI_ANTHROPIC_BASE_URL") or None,
            anthropic_model=os.getenv("HONEST_CI_ANTHROPIC_MODEL", "claude-opus-5"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            anthropic_max_tokens=_int_env("HONEST_CI_ANTHROPIC_MAX_TOKENS", 16000),
            anthropic_temperature=_optional_float_env("HONEST_CI_ANTHROPIC_TEMPERATURE"),
            anthropic_timeout_seconds=_int_env("HONEST_CI_ANTHROPIC_TIMEOUT_SECONDS", 300),
            anthropic_max_retries=_int_env("HONEST_CI_ANTHROPIC_MAX_RETRIES", 2),
            forge=os.getenv("HONEST_CI_FORGE", "local"),
            github_base_url=os.getenv("HONEST_CI_GITHUB_BASE_URL", "https://api.github.com"),
            github_token=os.getenv("GITHUB_TOKEN"),
            target_branch=os.getenv("HONEST_CI_TARGET_BRANCH", "main"),
            forge_timeout_seconds=_int_env("HONEST_CI_FORGE_TIMEOUT_SECONDS", 30),
            forge_status_name=os.getenv("HONEST_CI_FORGE_STATUS_NAME", "honest-ci/verify"),
        )

    # ── orchestration helpers ────────────────────────────────────────────

    def max_attempts_for(self, gate: str) -> int:
        return max(1, self.gate_max_attempts.get(gate, self.max_attempts))

    def retry_delay_for(self, attempt: int) -> float:
        """Bounded deterministic backoff; jitter is applied by the runner."""

        if not self.retry_backoff_seconds:
            return 0.0
        index = max(0, min(len(self.retry_backoff_seconds) - 1, attempt - 1))
        return max(0.0, self.retry_backoff_seconds[index])

    # ── model helpers ────────────────────────────────────────────────────

    @property
    def effective_deepseek_base_url(self) -> str:
        if self.deepseek_base_url:
            return self.deepseek_base_url.rstrip("/")
        return "https://api.deepseek.com/v1"

    @property
    def effective_qwen_base_url(self) -> str:
        """Bailian OpenAI-compatible host; override for workspace or other regions."""

        if self.qwen_base_url:
            return self.qwen_base_url.rstrip("/")
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @property
    def effective_glm_base_url(self) -> str:
        if self.glm_base_url:
            return self.glm_base_url.rstrip("/")
        if self.glm_protocol == "openai":
            return "https://api.z.ai/api/paas/v4"
        return "https://api.z.ai/api/anthropic"

    @property
    def effective_openai_base_url(self) -> str:
        """Up to and including ``/v1``: ``/chat/completions`` is appended."""

        if self.openai_base_url:
            return self.openai_base_url.rstrip("/")
        return "https://api.openai.com/v1"

    @property
    def effective_anthropic_base_url(self) -> str:
        """The host alone: ``/v1/messages`` is appended."""

        if self.anthropic_base_url:
            return self.anthropic_base_url.rstrip("/")
        return "https://api.anthropic.com"

    def api_key_for(self, provider: str) -> str | None:
        return {
            "deepseek": self.deepseek_api_key,
            "qwen": self.qwen_api_key,
            "glm": self.glm_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(provider)

    def model_for(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "glm": self.glm_model,
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
        }.get(provider, provider)

    def base_url_for(self, provider: str) -> str | None:
        """Where a provider's calls -- and its key -- go."""

        return {
            "deepseek": self.effective_deepseek_base_url,
            "qwen": self.effective_qwen_base_url,
            "glm": self.effective_glm_base_url,
            "openai": self.effective_openai_base_url,
            "anthropic": self.effective_anthropic_base_url,
        }.get(provider)

    @property
    def active_provider(self) -> str | None:
        """Provider a judgment call will really use; ``None`` means no model call.

        Both switches have to be on: a provider named in
        ``HONEST_CI_MODEL_PROVIDER`` and that provider's credential. Without
        either, the deterministic rules keep running untouched and the
        judgment-class ones are recorded as unread -- which sends the change
        to a human, never to a pass. "Nobody asked" is not "nothing wrong".
        """

        if self.model_provider is not None and self.api_key_for(self.model_provider):
            return self.model_provider
        return None

    @property
    def active_model(self) -> str:
        provider = self.active_provider
        return self.model_for(provider) if provider else "none"

    @property
    def credential_hint(self) -> str:
        return {
            "deepseek": "DEEPSEEK_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "glm": "ZAI_API_KEY or ANTHROPIC_AUTH_TOKEN",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(self.model_provider or "", "an API key")

    # ── forge helpers ────────────────────────────────────────────────────

    def forge_token(self) -> str | None:
        return {
            "github": self.github_token,
        }.get(self.forge)

    @property
    def forge_credential_hint(self) -> str:
        return {
            "github": "GITHUB_TOKEN",
        }.get(self.forge, "a forge token")

    def forge_base_url(self) -> str | None:
        return {
            "github": self.github_base_url,
        }.get(self.forge)

    # ── regression helpers ───────────────────────────────────────────────

    @property
    def coverage_measured(self) -> bool:
        """Whether the regression gate asks for coverage: either bar is on.

        One answer for the gate that asks and for the toolchains that collect.
        When a toolchain looked only at the overall bar, a changed-line bar on
        its own got no ``coverage run`` and no JaCoCo: every change blocked as
        unmeasurable, or a ``.coverage`` an earlier run left was judged as
        this one's.
        """

        return self.coverage_threshold > 0 or self.changed_line_coverage_threshold > 0

    # ── sandbox helpers ──────────────────────────────────────────────────

    def effective_sandbox_isolation(self) -> str:
        """Resolve the isolation actually in effect (docker may be unavailable)."""

        if self.sandbox_mode != "docker":
            return "process"
        from .tools import docker_available

        return "docker" if docker_available() else "process"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _optional_float_env(name: str) -> float | None:
    """A float, or ``None`` -- which leaves the field out of the request.

    No default to fall back to: the absence is the point, because the models
    these settings exist for reject the field outright. An unparseable value
    reads as unset, the way ``_float_env`` reads one as its default.
    """

    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return None


def _ratio_env(name: str, default: float) -> float:
    """A 0-to-1 ratio, or a refusal. Never a number outside the range.

    ``HONEST_CI_COVERAGE_THRESHOLD=75`` is what an operator types when they mean
    75%, and taking it at face value produces a bar of 7500% -- a gate that
    blocks every change, for a reason its own evidence pack states correctly
    and nobody believes. Clamping to 1.0 would be worse still: it demands
    full coverage while looking like it accepted the value.

    So it is refused at load, which is early enough that ``honest-ci doctor``
    reports it before a run does. An unparseable value keeps ``_float_env``'s
    reading and falls back to the default -- "not a number" has no intent to
    honour, while "75" plainly has one.
    """

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value > 1.0:
        raise SettingsError(
            f"{name}={raw.strip()} is outside 0 to 1. It is a ratio, not a percentage: "
            f"for {value:g}% write {value / 100:g}."
        )
    if value < 0.0:
        raise SettingsError(
            f"{name}={raw.strip()} is outside 0 to 1: a ratio cannot be negative. "
            "Write 0 to switch this check off."
        )
    return value


def _parse_backoff(value: str) -> tuple[float, ...]:
    parsed: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(max(0.0, float(item)))
        except ValueError:
            continue
    return tuple(parsed) or (0.0,)


def _parse_selection(value: str) -> str:
    """``impacted`` only when it was asked for exactly; anything else is ``all``.

    Falling back rather than refusing, unlike the policy file's enums: the two
    values differ in how *much* is verified, not in what the conclusion means,
    and the fallback is the stronger one. A misspelled variable therefore runs
    the whole suite, which is the direction a verification layer errs in.
    """

    return "impacted" if value.strip().lower() == "impacted" else "all"


def _parse_provider(value: str) -> str | None:
    """``HONEST_CI_MODEL_PROVIDER``: empty keeps the model off, a name turns it on.

    A name this build does not know is refused rather than read as "off".
    Whoever set it meant to turn the model on, and reading a typo as off would
    leave every judgment-class rule unread while they believed it was being
    judged -- the same reasoning as ``_ratio_env``.
    """

    name = value.strip().lower()
    if not name:
        return None
    if name not in MODEL_PROVIDERS:
        raise SettingsError(
            f"HONEST_CI_MODEL_PROVIDER={value.strip()} is not a known provider: "
            f"use one of {', '.join(MODEL_PROVIDERS)}, or leave it unset to keep the model off."
        )
    return name


def _parse_max_tokens_param(value: str) -> MaxTokensParam:
    """``HONEST_CI_OPENAI_MAX_TOKENS_PARAM``: which field carries the output cap.

    Refused rather than defaulted when it names neither field. A server
    handed a field it does not know either drops the cap without a word or
    rejects every call, and both look like a model problem, not a typo.
    """

    name = value.strip().lower()
    if name in ("", "max_completion_tokens"):
        return "max_completion_tokens"
    if name == "max_tokens":
        return "max_tokens"
    raise SettingsError(
        f"HONEST_CI_OPENAI_MAX_TOKENS_PARAM={value.strip()} names neither field: use "
        "max_completion_tokens (OpenAI) or max_tokens (Ollama, DeepSeek, most other servers)."
    )


def _parse_names(value: str) -> tuple[str, ...]:
    """Parse a comma-separated name list, dropping empty entries and repeats."""

    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _parse_artifacts(value: str) -> tuple[str, ...]:
    """Parse ``HONEST_CI_MAVEN_EXTRA_ARTIFACTS``, or refuse the whole value.

    Refused rather than filtered: an operator adds a coordinate here *because*
    the last run failed on it, and silently dropping the one they mistyped
    would hand them the identical failure with nothing new to read.
    """

    artifacts = _parse_names(value)
    bad = [item for item in artifacts if not ARTIFACT_COORDINATE.fullmatch(item)]
    if bad:
        raise SettingsError(
            f"Not a Maven coordinate in HONEST_CI_MAVEN_EXTRA_ARTIFACTS: {', '.join(bad)}. "
            "Write groupId:artifactId:version (packaging and classifier may follow), "
            "comma-separated."
        )
    return artifacts


def _parse_host_repository(value: str | None) -> Path | None:
    """``HONEST_CI_MAVEN_HOST_REPOSITORY``: unset is ``~/.m2/repository``, empty is off.

    A relative path is refused because Maven would resolve it against the
    workspace, and a comma because Maven splits the tail on commas: either
    way the run would quietly read a directory nobody named.
    """

    if value is None:
        return Path.home() / ".m2" / "repository"
    if not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    if not path.is_absolute() or "," in str(path):
        raise SettingsError(
            f"HONEST_CI_MAVEN_HOST_REPOSITORY must be an absolute path with no comma: {value}. "
            "Leave it empty to read no repository of this machine's."
        )
    return path


def _parse_optional_bool(value: str | None) -> bool | None:
    """Tri-state env flag: unset means "do not send the field at all"."""

    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _parse_gate_attempts(value: str) -> dict[str, int]:
    """Parse ``review=4,regression=5`` without making malformed config fatal."""

    parsed: dict[str, int] = {}
    for item in value.split(","):
        name, separator, raw_attempts = item.partition("=")
        if not separator or not name.strip():
            continue
        try:
            parsed[name.strip()] = max(1, int(raw_attempts.strip()))
        except ValueError:
            continue
    return parsed

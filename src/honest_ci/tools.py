"""Command execution under isolation.

Every deterministic check a gate runs goes through ``ToolRunner``. Two
properties matter more than convenience:

* **Tests and scanners never get the network.** Only dependency installation
  (``sandbox.py``) asks for ``allow_network=True``.
* **A tool that never started is reported as unavailable, not as passing.**
  The gates read ``available`` to decide whether to block (fail closed).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

from .artifacts import ArtifactStore
from .contracts import ToolReport
from .redaction import redact_text
from .utils import recorded_command

#: The host environment is not forwarded into the sandbox: a forge or model
#: credential must never be visible to code under verification.
SAFE_ENV_KEYS = {
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    # The path of the JDK this machine's Maven is meant to run on, and nothing
    # else: not a credential, and nothing process mode would hide, since the
    # code under verification can read this machine's disk anyway. Without it
    # Homebrew's mvn, a wrapper that falls back to Homebrew's newest openjdk,
    # runs on a JDK the pinned scanners cannot read. Docker mode never sees it:
    # only the toolchain's own keys become ``docker -e``, and the image brings
    # its own JDK.
    "JAVA_HOME",
}

# Docker reports its own startup failures through the container exit code. 125
# is a daemon/runtime error, 126 an entrypoint that cannot be invoked and 127 a
# missing executable. A tool that never started is *unavailable*, not a tool
# that ran and reported defects, so the distinction has to survive into the
# report the gates read.
_DOCKER_STARTUP_EXIT_CODES = {125, 126, 127}
_DOCKER_STARTUP_MARKERS = (
    "Error response from daemon",
    "executable file not found",
    "OCI runtime create failed",
    "docker: command not found",
    "Cannot connect to the Docker daemon",
)
_PROBE_TIMEOUT_SECONDS = 60


class ToolRunner:
    def __init__(
        self,
        sandbox_path: Path,
        artifacts_path: Path,
        mode: str = "process",
        docker_image: str = "python:3.12-slim",
        timeout_seconds: int = 300,
        docker_cpus: str = "2",
        docker_memory: str = "2g",
        docker_pids_limit: int = 256,
    ) -> None:
        self.sandbox_path = sandbox_path
        self.artifacts_path = artifacts_path
        self.mode = mode
        self.docker_image = docker_image
        self.timeout_seconds = timeout_seconds
        self.docker_cpus = docker_cpus
        self.docker_memory = docker_memory
        self.docker_pids_limit = docker_pids_limit
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self._probe_cache: dict[tuple[str, ...], bool] = {}

    def available(self, executable: str) -> bool:
        """Report whether ``executable`` exists in the execution environment.

        The probe runs in the same isolation as the tool itself: asking "is the
        Docker daemon up?" would make every tool look available even when the
        image does not ship it.
        """

        key = ("exe", self.mode, self.docker_image, executable)
        cached = self._probe_cache.get(key)
        if cached is not None:
            return cached
        if self.mode == "docker":
            result = docker_available() and self._probe(
                ["sh", "-c", f"command -v {shlex.quote(executable)} >/dev/null 2>&1"]
            )
        else:
            result = shutil.which(executable) is not None or Path(executable).is_file()
        self._probe_cache[key] = result
        return result

    def module_available(self, interpreter: str, module: str) -> bool:
        """Report whether ``interpreter`` can import ``module``.

        The interpreter that answers must be the one that will run the tool:
        checking the host's own modules says nothing about a sandbox virtualenv
        or a container image.
        """

        key = ("mod", self.mode, self.docker_image, interpreter, module)
        cached = self._probe_cache.get(key)
        if cached is not None:
            return cached
        probe = [
            interpreter,
            "-c",
            "import importlib.util, sys; "
            f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)",
        ]
        if self.mode == "docker":
            result = docker_available() and self._probe(probe)
        else:
            result = (
                shutil.which(interpreter) is not None or Path(interpreter).is_file()
            ) and self._probe(probe)
        self._probe_cache[key] = result
        return result

    def _probe(self, command: list[str]) -> bool:
        """Run a read-only availability check under the same isolation as tools."""

        try:
            proc = subprocess.run(
                self._command(command, allow_network=False),
                cwd=self.sandbox_path if self.mode != "docker" else None,
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_SECONDS,
                env=self._safe_env(),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    def run(
        self,
        name: str,
        command: list[str],
        report_name: str | None = None,
        allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        start = time.perf_counter()
        # Named once, so the report, the log beside it and a launch failure all
        # quote the same command: three spellings would read as evidence about
        # three different runs.
        recorded = recorded_command(command)
        full_command = self._command(command, allow_network=allow_network, env=env)
        executed = True
        try:
            proc = subprocess.run(
                full_command,
                cwd=self.sandbox_path if self.mode != "docker" else None,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env={**self._safe_env(), **(env or {})},
                check=False,
            )
            exit_code = proc.returncode
            stdout = redact_text(proc.stdout)
            stderr = redact_text(proc.stderr)
            executed = not self._is_startup_failure(exit_code, stderr)
        except subprocess.TimeoutExpired as exc:
            exit_code = 124
            stdout = redact_text(_timeout_output(exc.stdout))
            stderr = redact_text(
                _timeout_output(exc.stderr) + f"\nTimed out after {self.timeout_seconds}s"
            )
        except OSError as exc:
            # The executable itself could not be launched. That is a missing
            # tool, never evidence about the code under verification.
            exit_code = 127
            stdout = ""
            # The executable alone, named the same way the command is.
            stderr = redact_text(f"{recorded_command(command[:1])}: {exc.strerror or exc}")
            executed = False
        duration = round(time.perf_counter() - start, 3)
        report_path = None
        if report_name:
            # Workers share the artifact root. Separate executions must not
            # overwrite a log another run's evidence still points at.
            manifest = ArtifactStore(self.artifacts_path).write_text(
                f"tool-{uuid4().hex}",
                report_name,
                f"$ {recorded}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n",
            )
            report_path = str(self.artifacts_path / manifest.artifacts[0].path)
        return ToolReport(
            name=name,
            command=redact_text(recorded),
            available=executed,
            passed=executed and exit_code == 0,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            report_path=report_path,
            findings=[] if exit_code == 0 else [stderr or stdout or f"{name} failed"],
        )

    def _is_startup_failure(self, exit_code: int, stderr: str) -> bool:
        """Detect a tool that never ran, as opposed to one that ran and failed."""

        if self.mode != "docker":
            return False
        if exit_code not in _DOCKER_STARTUP_EXIT_CODES:
            return False
        return any(marker in stderr for marker in _DOCKER_STARTUP_MARKERS)

    def _command(
        self,
        command: list[str],
        allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> list[str]:
        # Only the toolchain's fixed runtime settings cross this boundary;
        # configuration and inherited credentials cannot become docker -e.
        # The toolchain composes each value from paths it owns; none is read
        # from the environment.
        if env and set(env) - {"MAVEN_CONFIG", "MAVEN_OPTS", "COVERAGE_FILE"}:
            raise ValueError("Unsupported sandbox environment key")
        if self.mode != "docker":
            return command
        network = "bridge" if allow_network else "none"
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "--cpus",
            self.docker_cpus,
            "--memory",
            self.docker_memory,
            "--pids-limit",
            str(self.docker_pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{self.sandbox_path.resolve()}:/workspace",
            "-v",
            f"{self.artifacts_path.resolve()}:/artifacts",
            "-w",
            "/workspace",
            *[part for key, value in (env or {}).items() for part in ("-e", f"{key}={value}")],
            self.docker_image,
            *command,
        ]

    def _safe_env(self) -> dict[str, str]:
        return sandbox_env()


def sandbox_env() -> dict[str, str]:
    """The part of this process's environment a sandboxed tool is given.

    Public so that ``doctor`` can ask a process-mode tool a question in the
    environment the tool will actually run in: asked in the operator's full
    shell, ``mvn -v`` could name a JDK no run ever gets.
    """

    return {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def docker_image_present(image: str) -> bool:
    """Whether ``image`` is already on this machine, without pulling it.

    ``doctor`` asks so an operator learns it here rather than from the first
    run: a daemon that is up but an image that is not local means the next
    gate run starts with a pull, and behind a locked-down network that pull
    goes to a mirror or nowhere at all.

    ``image inspect`` and not ``pull`` on purpose -- a diagnostic that changes
    the machine it is diagnosing is not one, and the pull is the operator's
    decision to make with their own registry credentials.
    """

    if not docker_available():
        return False
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _timeout_output(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""

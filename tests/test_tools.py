from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from honest_ci.maven import maven_env
from honest_ci.tools import ToolRunner


def runner(tmp_path: Path) -> ToolRunner:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    return ToolRunner(sandbox_path=sandbox, artifacts_path=tmp_path / "artifacts", mode="process")


def test_successful_command_is_reported_with_its_output(tmp_path: Path) -> None:
    report = runner(tmp_path).run("echo", ["echo", "hello"], report_name="echo.log")

    assert report.available
    assert report.passed
    assert report.exit_code == 0
    assert "hello" in report.stdout
    assert report.report_path is not None
    assert "hello" in Path(report.report_path).read_text(encoding="utf-8")


def test_failing_command_is_a_finding_not_an_unavailable_tool(tmp_path: Path) -> None:
    report = runner(tmp_path).run("false", ["sh", "-c", "echo boom >&2; exit 3"])

    assert report.available  # it ran
    assert not report.passed
    assert report.exit_code == 3
    assert report.findings


def test_missing_executable_is_reported_as_unavailable(tmp_path: Path) -> None:
    report = runner(tmp_path).run("ghost", ["honest-ci-no-such-tool", "--version"])

    assert not report.available
    assert not report.passed
    assert report.exit_code == 127


def test_the_recorded_command_names_the_interpreter_without_its_directory(
    tmp_path: Path,
) -> None:
    report = runner(tmp_path).run(
        "python", [sys.executable, "-c", "print(1)"], report_name="python.log"
    )

    # ``sys.executable`` is an absolute path into the operator's disk, and the
    # command travels into an evidence pack that is handed to an auditor. The
    # interpreter's name -- version included -- is what ran; the directory it
    # sits in is not a fact about this change.
    assert report.command == f"{Path(sys.executable).name} -c 'print(1)'"
    assert str(Path(sys.executable).parent) not in report.command
    assert report.report_path is not None
    # The log beside the report quotes the same command, not the raw argv.
    assert report.command in Path(report.report_path).read_text(encoding="utf-8")


def test_a_command_inside_the_repository_keeps_the_path_that_locates_it(
    tmp_path: Path,
) -> None:
    tools = runner(tmp_path)
    scripts = tools.sandbox_path / "scripts"
    scripts.mkdir()
    script = scripts / "run.sh"
    script.write_text("#!/bin/sh\necho ran\n", encoding="utf-8")
    script.chmod(0o755)

    report = tools.run("script", ["scripts/run.sh", "--all"])

    assert report.passed

    # A relative argv[0] points into the tree under verification, which is
    # exactly where a reader should be able to follow it.
    assert report.command == "scripts/run.sh --all"


def test_a_launch_failure_names_the_executable_the_same_way(tmp_path: Path) -> None:
    report = runner(tmp_path).run("ghost", ["/opt/honest-ci/no-such-tool", "--version"])

    assert not report.available
    # The report and the reason have to quote one spelling of the command, or
    # they read as evidence about two different runs.
    assert report.command.startswith("no-such-tool")
    assert report.stderr.startswith("no-such-tool:")
    assert "/opt/honest-ci" not in report.stderr


def test_tool_output_is_redacted(tmp_path: Path) -> None:
    report = runner(tmp_path).run("leak", ["echo", "token=glpat-abcdefghijklmnopqrst"])

    assert "glpat-" not in report.stdout
    assert "[REDACTED]" in report.stdout


def test_timeout_is_recorded_without_claiming_a_verdict(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    slow = ToolRunner(
        sandbox_path=sandbox,
        artifacts_path=tmp_path / "artifacts",
        mode="process",
        timeout_seconds=1,
    )
    report = slow.run("sleep", ["sleep", "5"])

    assert report.exit_code == 124
    assert not report.passed
    assert "Timed out" in report.stderr


def test_availability_probe_uses_the_execution_environment(tmp_path: Path) -> None:
    tool_runner = runner(tmp_path)
    assert tool_runner.available("sh")
    assert not tool_runner.available("honest-ci-no-such-tool")


def test_host_environment_is_not_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-should-not-leak-into-sandbox")
    report = runner(tmp_path).run("env", ["sh", "-c", "echo ${DEEPSEEK_API_KEY:-absent}"])

    assert "absent" in report.stdout


def test_a_maven_that_falls_back_to_its_own_jdk_runs_on_the_one_java_home_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Homebrew's mvn is this wrapper: with JAVA_HOME empty it runs on
    # Homebrew's newest openjdk, whatever JDK the operator chose.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mvn = bin_dir / "mvn"
    mvn.write_text('#!/bin/sh\necho "runs on ${JAVA_HOME:-the newest openjdk}"\n')
    mvn.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("JAVA_HOME", "/opt/jdk-21")

    report = runner(tmp_path).run("mvn", ["mvn", "-v"])

    assert report.passed and "runs on /opt/jdk-21" in report.stdout


def test_the_hosts_java_home_stays_out_of_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAVA_HOME", "/host/jdk-27")
    captured: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    tool = ToolRunner(tmp_path, tmp_path / "artifacts", mode="docker", docker_image="maven:custom")
    tool.run("java", ["mvn", "-B", "-o", "test"], env=maven_env(tmp_path, "docker"))

    # The image's own JDK is the one the toolchain was pinned against; a path
    # on this machine would name nothing inside the container, or worse,
    # something else.
    assert not any("JAVA_HOME" in part or "/host/jdk-27" in part for part in captured[0])


def test_java_runtime_env_and_image_reach_docker_without_inheriting_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-abcdefghijklmnopqrst")
    monkeypatch.setenv("MAVEN_OPTS", "host-only-option")
    captured: list[tuple[list[str], dict[str, str]]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append((command, dict(kwargs["env"])))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    tool = ToolRunner(tmp_path, tmp_path / "artifacts", mode="docker", docker_image="maven:custom")
    tool.run("java", ["mvn", "-B", "-o", "test"], env=maven_env(tmp_path, "docker"))
    command, env = captured[0]
    assert command[-5:] == ["maven:custom", "mvn", "-B", "-o", "test"]
    assert command[command.index("--network") + 1] == "none"
    assert "MAVEN_CONFIG=/workspace/.honest-ci/m2-config" in command
    assert "MAVEN_OPTS=-Duser.home=/workspace/.honest-ci" in command
    assert "--user" in command
    assert "GITLAB_TOKEN" not in env and "host-only-option" not in str((command, env))
    assert "glpat-" not in str((command, env))


def test_fixed_runtime_env_is_available_in_process_mode(tmp_path: Path) -> None:
    report = runner(tmp_path).run(
        "env",
        [sys.executable, "-c", "import os; print(os.environ['MAVEN_CONFIG'])"],
        env={"MAVEN_CONFIG": "/workspace/.honest-ci/m2-config"},
    )
    assert report.passed and "/workspace/.honest-ci/m2-config" in report.stdout


def test_tool_environment_cannot_accept_a_credential_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported sandbox environment key"):
        runner(tmp_path).run("env", ["env"], env={"GITLAB_TOKEN": "credential"})


def test_repeated_log_names_preserve_each_executions_evidence(tmp_path: Path) -> None:
    tool = runner(tmp_path)
    first = tool.run("echo", ["echo", "first run"], report_name="maven.log")
    second = tool.run("echo", ["echo", "second run"], report_name="maven.log")
    assert first.report_path and second.report_path and first.report_path != second.report_path
    assert "first run" in Path(first.report_path).read_text()
    assert "second run" in Path(second.report_path).read_text()

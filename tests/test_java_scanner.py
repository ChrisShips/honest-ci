"""Maven outputs are canned; no Java installation or Docker daemon is needed."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honest_ci.adapters.base import AdapterSet
from honest_ci.adapters.java_scanner import JavaScannerAdapter
from honest_ci.adapters.scanners import SCANNER_ADAPTERS, LanguageRouter
from honest_ci.contracts import ProposedTest, ToolReport
from honest_ci.java_reports import read_jacoco_lines
from honest_ci.maven import (
    MAVEN_CLASS_FILE_CEILINGS,
    MAVEN_GLOBAL_SETTINGS,
    MAVEN_JDK_CEILING,
    MAVEN_PLUGINS,
    MAVEN_RUNTIME_ARTIFACTS,
    MAVEN_STAMP,
    fetch_command,
    get_coordinate,
    maven_env,
    maven_java,
    unprovisioned_artifacts,
    unprovisioned_hint,
)
from honest_ci.rules import RuleSet
from honest_ci.runner import GateRunner
from honest_ci.sandbox import DependencyProvisionError, JavaSandboxBuilder
from honest_ci.settings import Settings
from honest_ci.state import RunState
from honest_ci.storage import SQLiteLedger
from honest_ci.testreports import ReportError
from honest_ci.tools import ToolRunner
from honest_ci.utils import hash_tree

from .factories import ScriptedLLM, fill_repository, make_change, make_settings

SOURCE = "src/main/java/com/acme/Order.java"
JUNIT = '<testsuite><testcase classname="com.acme.OrderTest" name="ok"/><testcase classname="com.acme.OrderTest" name="bad"><failure message="wrong total"/></testcase></testsuite>'
JACOCO = '<report><package name="com/acme"><sourcefile name="Order.java"><line nr="7" mi="0" ci="3"/><line nr="8" mi="2" ci="0"/></sourcefile></package><counter type="LINE" missed="1" covered="3"/></report>'


def write(workspace: Path, name: str, body: str) -> Path:
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    write(root, "pom.xml", "<project/>")
    write(root, SOURCE, "package com.acme; class Order {}")
    return root


def adapter(tmp_path: Path, **settings: Any) -> JavaScannerAdapter:
    return JavaScannerAdapter(replace(make_settings(), **settings), tmp_path / "artifacts")


def canned(
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str] | None = None,
    *,
    code: int = 0,
    codes: dict[str, int] | None = None,
    stdout: str = "",
    available: bool = True,
) -> list[tuple[str, list[str], bool, dict[str, str]]]:
    # The per-run `clean` is a precondition of every Maven goal, not the thing
    # under test: `code=1` in a test here means "the tool exited 1". A test
    # about the clean itself says so with `codes={"clean": 1}`.
    exits = {"clean": 0, **(codes or {})}
    calls: list[tuple[str, list[str], bool, dict[str, str]]] = []
    monkeypatch.setattr(ToolRunner, "available", lambda self, name: available)
    monkeypatch.setattr(Settings, "effective_sandbox_isolation", lambda self: self.sandbox_mode)

    def run(
        self: ToolRunner,
        name: str,
        command: list[str],
        report_name: str | None = None,
        allow_network: bool = False,
        env: Mapping[str, str] | None = None,
    ) -> ToolReport:
        calls.append((name, command, allow_network, dict(env or {})))
        for path, body in (files or {}).items():
            write(self.sandbox_path, path, body)
        exit_code = exits.get(name, code)
        return ToolReport(
            name=name,
            command=" ".join(command),
            exit_code=exit_code,
            passed=exit_code == 0,
            stdout=stdout,
        )

    monkeypatch.setattr(ToolRunner, "run", run)
    return calls


def ran(
    calls: list[tuple[str, list[str], bool, dict[str, str]]], name: str
) -> list[tuple[str, list[str], bool, dict[str, str]]]:
    """The calls one runner made, ignoring the clean that precedes them all."""

    return [call for call in calls if call[0] == name]


@pytest.mark.parametrize(
    "tool,filename,xml,code",
    [
        (
            "checkstyle",
            "checkstyle-result.xml",
            f'<checkstyle><file name="/workspace/{SOURCE}"><error line="7" severity="error" message="bad style" source="AvoidStarImport"/></file></checkstyle>',
            "AvoidStarImport",
        ),
        (
            "pmd",
            "pmd.xml",
            f'<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0"><file name="/workspace/{SOURCE}"><violation beginline="7" rule="UnusedPrivateField" ruleset="Best practices">unused value</violation></file></pmd>',
            "UnusedPrivateField",
        ),
        (
            "spotbugs",
            "spotbugsXml.xml",
            '<BugCollection><BugInstance type="NP_NULL" priority="1"><LongMessage>possible null</LongMessage><SourceLine sourcepath="com/acme/Order.java" start="7"/></BugInstance></BugCollection>',
            "NP_NULL",
        ),
    ],
)
@pytest.mark.parametrize("exit_code", [0, 1])
def test_xml_violations_are_located_even_when_maven_exits_zero(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    filename: str,
    xml: str,
    code: str,
    exit_code: int,
) -> None:
    calls = canned(monkeypatch, {f"target/{filename}": xml}, code=exit_code)
    report = adapter(tmp_path).run_static_checks(repo, make_change(), tools=[tool])[0]
    assert report.available and not report.passed
    assert [(item.file, item.line, item.code) for item in report.diagnostics] == [(SOURCE, 7, code)]
    _, command, online, _ = ran(calls, tool)[0]
    assert command[:4] == ["mvn", "-B", "-o", "-q"] and not online
    assert any(MAVEN_PLUGINS[tool] in argument for argument in command)
    if tool == "spotbugs":
        assert "-Dspotbugs.xmlOutput=true" in command


def test_module_paths_and_host_absolute_paths_are_preserved(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write(repo, "service/" + SOURCE, "class Order {}")
    xml = f'<checkstyle><file name="{path}"><error line="4" message="style" source="Rule"/></file></checkstyle>'
    canned(monkeypatch, {"service/target/checkstyle-result.xml": xml})
    report = adapter(tmp_path).run_static_checks(repo, make_change(), tools=["checkstyle"])[0]
    assert report.diagnostics[0].file == "service/" + SOURCE


@pytest.mark.parametrize("code", [0, 1])
def test_missing_reports_are_opaque_failures_with_raw_output(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: int,
) -> None:
    canned(monkeypatch, code=code, stdout="Cannot access central in offline mode: plugin missing")
    report = adapter(tmp_path).run_static_checks(repo, make_change(), tools=["pmd"])[0]
    assert report.available and not report.passed and not report.diagnostics
    assert "plugin missing" in report.stdout and any(
        "plugin missing" in item for item in report.findings
    )


@pytest.mark.parametrize(
    "xml",
    [
        "<oops",
        "<not-checkstyle/>",
        '<!DOCTYPE checkstyle [<!ENTITY x "boom">]><checkstyle/>',
        '<checkstyle><file name="/etc/passwd"><error line="2"/></file></checkstyle>',
    ],
)
def test_unreadable_or_unsafe_xml_cannot_pass(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    xml: str,
) -> None:
    canned(monkeypatch, {"target/checkstyle-result.xml": xml})
    report = adapter(tmp_path).run_static_checks(repo, make_change(), tools=["checkstyle"])[0]
    assert not report.passed and not report.diagnostics


def test_old_xml_is_never_reused_when_a_tool_produces_nothing(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = write(repo, "target/checkstyle-result.xml", "<checkstyle/>")
    canned(monkeypatch)
    report = adapter(tmp_path).run_static_checks(repo, make_change(), tools=["checkstyle"])[0]
    assert not stale.exists() and not report.passed


def test_a_clean_fresh_report_can_pass(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, {"target/checkstyle-result.xml": "<checkstyle/>"})
    assert adapter(tmp_path).run_static_checks(repo, make_change(), tools=["checkstyle"])[0].passed


def test_missing_maven_is_unavailable_for_scanning_and_testing(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch, available=False)
    scanner = adapter(tmp_path)
    assert not scanner.run_static_checks(repo, make_change(), tools=["pmd"])[0].available
    assert not scanner.run_tests(repo).available
    assert not scanner.measure_coverage(repo).available
    assert not calls


def test_surefire_reports_name_failures_and_select_class_names(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-orders.xml": JUNIT}, code=1)
    result = adapter(tmp_path).run_tests(repo, ["com.acme.OrderTest", "OtherTest"])
    assert (result.passed, result.failed, result.runner) == (1, 1, "junit")
    assert "com.acme.OrderTest#bad" in result.failures[0]
    command = ran(calls, "tests")[0][1]
    assert "-Dtest=com.acme.OrderTest,OtherTest" in command
    assert (
        command.index(MAVEN_PLUGINS["jacoco"] + ":prepare-agent")
        < command.index("test")
        < command.index(MAVEN_PLUGINS["jacoco"] + ":report")
    )


def test_empty_selection_runs_the_whole_suite_without_coverage_when_disabled(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch)
    adapter(tmp_path, coverage_threshold=0).run_tests(repo, [])
    command = ran(calls, "tests")[0][1]
    assert command[-1] == "test"
    assert not any("jacoco" in item or "-Dtest=" in item for item in command)


def test_a_changed_line_bar_alone_still_runs_jacoco(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch)
    adapter(tmp_path, coverage_threshold=0, changed_line_coverage_threshold=0.8).run_tests(repo)
    command = ran(calls, "tests")[0][1]
    # Without the two goals there is no jacoco.xml, and the gate blocks every
    # change on a bar nobody could measure.
    assert MAVEN_PLUGINS["jacoco"] + ":prepare-agent" in command
    assert MAVEN_PLUGINS["jacoco"] + ":report" in command


def test_a_selection_does_not_fail_the_modules_that_hold_none_of_its_tests(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch)
    adapter(tmp_path).run_tests(repo, ["com.acme.OrderTest"])
    command = ran(calls, "tests")[0][1]
    # Surefire's failIfNoSpecifiedTests defaults to true, per module: in a
    # reactor, every module that holds none of the named classes fails, the
    # build stops there, and the gate is left with an exit code of 1 and no
    # failing test to name -- an opaque block about nothing in the change.
    assert "-Dtest=com.acme.OrderTest" in command
    assert "-Dsurefire.failIfNoSpecifiedTests=false" in command


def test_selection_cannot_inject_maven_options(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch)
    assert not adapter(tmp_path).run_tests(repo, ["-DskipTests=true"]).available
    assert not ran(calls, "tests")
    assert not any("skipTests" in item for _, command, _, _ in calls for item in command)


def test_custom_command_is_unchanged_and_reads_configured_junit(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch, {"reports/suite.xml": JUNIT})
    result = adapter(
        tmp_path, test_command='./mvnw "test" -Pci', test_report="reports/*.xml"
    ).run_tests(repo, ["Ignored"])
    assert calls[0][1] == ["./mvnw", "test", "-Pci"]
    assert result.passed == 1 and result.failed == 1


def test_custom_command_defaults_to_surefire_reports(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": JUNIT})
    assert adapter(tmp_path, test_command="mvn test").run_tests(repo).failed == 1


@pytest.mark.parametrize("mode", ["process", "docker"])
def test_custom_wrapper_does_not_require_system_maven_when_provisioning_is_disabled(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": JUNIT}, available=False)
    results = adapter(
        tmp_path,
        test_command="./mvnw test",
        sandbox_mode=mode,
        sandbox_build_deps=False,
    ).run_tests(repo)
    assert results.available and results.failed == 1
    assert calls[0][1] == ["./mvnw", "test"]


def test_old_suite_and_coverage_data_are_cleared_before_testing(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = [
        write(repo, path, body)
        for path, body in (
            ("target/surefire-reports/TEST-old.xml", JUNIT),
            ("target/site/jacoco/jacoco.xml", JACOCO),
            ("target/jacoco.exec", "old"),
        )
    ]
    canned(monkeypatch)
    scanner = adapter(tmp_path)
    result = scanner.run_tests(repo)
    assert result.collected == 0
    assert all(not path.exists() for path in stale)
    assert scanner.measure_coverage(repo).ratio is None


def test_an_undeletable_old_report_prevents_running_with_stale_evidence(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write(repo, "target/surefire-reports/TEST-old.xml", JUNIT)
    calls = canned(monkeypatch)
    monkeypatch.setattr("honest_ci.testreports.clear_reports", lambda *args: 0)
    result = adapter(tmp_path).run_tests(repo)
    assert not result.available and not ran(calls, "tests")


@pytest.mark.parametrize("modules", [1, 2])
def test_jacoco_coverage_sums_top_level_line_counters_without_running_commands(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    modules: int,
) -> None:
    calls = canned(monkeypatch)
    for index in range(modules):
        write(repo, f"module{index}/target/site/jacoco/jacoco.xml", JACOCO)
    result = adapter(tmp_path).measure_coverage(repo)
    assert result.available and result.ratio == 0.75 and not calls


def test_changed_lines_are_judged_against_the_modules_own_sources(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch)
    write(repo, "target/site/jacoco/jacoco.xml", JACOCO)

    result = adapter(tmp_path).measure_coverage(repo, {SOURCE: {7, 8, 9}})

    # Line 7 ran (ci > 0), line 8 did not (mi > 0), line 9 is in neither
    # counter and is therefore not an executable line at all.
    assert (result.changed_lines_total, result.changed_lines_covered) == (2, 1)
    assert result.uncovered == [(SOURCE, 8)]
    assert result.unmapped_sources == []


def test_a_package_path_that_matches_no_module_leaves_the_file_unmeasured(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch)
    # The report is the other module's, so ``com/acme/Order.java`` in it is not
    # this module's file. Guessing they are the same would judge a changed file
    # against another module's numbers.
    write(repo, "other/target/site/jacoco/jacoco.xml", JACOCO)

    result = adapter(tmp_path).measure_coverage(repo, {SOURCE: {7}})

    assert result.unmapped_sources == [SOURCE]
    assert result.changed_lines_total == 0


def test_unreadable_per_line_data_fails_the_whole_measurement(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canned(monkeypatch)
    write(
        repo,
        "target/site/jacoco/jacoco.xml",
        JACOCO.replace('nr="8"', 'nr="oops"'),
    )

    result = adapter(tmp_path).measure_coverage(repo, {SOURCE: {7}})

    # The top-level counter is still readable, so the overall ratio stands and
    # the changed-line fields stay empty. Which of the two bars that leaves
    # unchecked is the gate's call, not this adapter's.
    assert result.available is True
    assert result.ratio == 0.75
    assert result.changed_lines_total is None
    assert result.stderr


@pytest.mark.parametrize(
    "xml",
    [
        "",
        "<report/>",
        '<report><counter type="LINE" covered="bad" missed="1"/></report>',
        '<report><counter type="LINE" covered="0" missed="0"/></report>',
    ],
)
def test_missing_or_invalid_coverage_stays_unknown(repo: Path, tmp_path: Path, xml: str) -> None:
    if xml:
        write(repo, "target/site/jacoco/jacoco.xml", xml)
    result = adapter(tmp_path).measure_coverage(repo)
    assert result.available and result.ratio is None and result.stderr


def test_jacoco_line_parser_accepts_the_real_external_dtd_without_fetching_it(repo: Path) -> None:
    path = write(
        repo,
        "target/site/jacoco/jacoco.xml",
        '<!DOCTYPE report PUBLIC "-//JACOCO//DTD Report 1.1//EN" "report.dtd">' + JACOCO,
    )
    assert read_jacoco_lines(path) == {"com/acme/Order.java": {7: (0, 3), 8: (2, 0)}}


def test_jacoco_entities_are_rejected(repo: Path) -> None:
    path = write(
        repo, "target/site/jacoco/jacoco.xml", '<!DOCTYPE report [<!ENTITY x "bad">]>' + JACOCO
    )
    with pytest.raises(ReportError):
        read_jacoco_lines(path)


def test_semgrep_requires_local_rules_and_uses_no_registry(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(
        monkeypatch,
        stdout=json.dumps(
            {
                "results": [
                    {
                        "path": SOURCE,
                        "start": {"line": 7},
                        "check_id": "R1",
                        "extra": {"message": "bad"},
                    }
                ]
            }
        ),
    )
    scanner = adapter(tmp_path)
    assert not scanner.run_static_checks(repo, make_change(), tools=["semgrep"])[0].available
    write(repo, ".semgrep.yml", "rules: []")
    report = scanner.run_static_checks(repo, make_change(), tools=["semgrep"])[0]
    assert report.diagnostics[0].line == 7 and not report.passed
    assert calls[0][1][-2:] == ["--config", ".semgrep.yml"] and not calls[0][2]


def test_provisioning_is_online_but_uses_only_pinned_plugins_and_cached_workspace_repo(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch)
    settings = Settings()
    builder = JavaSandboxBuilder(settings)
    builder.ensure_deps(repo, tmp_path / "artifacts", "docker")
    assert calls and all(online for _, _, online, _ in calls)
    assert all(
        "-o" not in command and "-Dmaven.repo.local=/workspace/.honest-ci/m2" in command
        for _, command, _, _ in calls
    )
    assert calls[0][1][-2:] == [
        MAVEN_PLUGINS["dependency"] + ":go-offline",
        MAVEN_PLUGINS["dependency"] + ":resolve-plugins",
    ]
    assert {command[-1] for _, command, _, _ in calls[1:]} == {
        "-Dartifact=" + item
        for item in (*MAVEN_PLUGINS.values(), *MAVEN_RUNTIME_ARTIFACTS.values())
    }
    assert all(env == maven_env(repo, "docker") for _, _, _, env in calls)
    assert (repo / MAVEN_STAMP).exists()
    calls.clear()
    builder.ensure_deps(repo, tmp_path / "artifacts", "docker")
    assert not calls


@pytest.mark.parametrize("changed", ["root", "module", "image", "plugin"])
def test_dependency_cache_invalidates_when_its_inputs_change(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    calls = canned(monkeypatch)
    settings = Settings()
    JavaSandboxBuilder(settings).ensure_deps(repo, tmp_path / "artifacts", "docker")
    calls.clear()
    if changed in {"root", "module"}:
        write(repo, "pom.xml" if changed == "root" else "module/pom.xml", "<project>new</project>")
    elif changed == "image":
        settings = replace(settings, java_docker_image="maven:custom")
    else:
        monkeypatch.setitem(
            MAVEN_PLUGINS, "pmd", "org.apache.maven.plugins:maven-pmd-plugin:3.27.0"
        )
    JavaSandboxBuilder(settings).ensure_deps(repo, tmp_path / "artifacts", "docker")
    assert calls


def test_failed_provisioning_is_unavailable_and_leaves_no_success_stamp(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch, code=1, stdout="plugin dependency unavailable")
    scanner = adapter(tmp_path, sandbox_mode="docker")
    report = scanner.run_static_checks(repo, make_change(), tools=["pmd"])[0]
    assert not report.available and "plugin dependency unavailable" in report.stderr
    assert not (repo / MAVEN_STAMP).exists()
    assert len(calls) == 1 and calls[0][2]


def test_process_and_disabled_provisioning_do_not_install(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch)
    JavaSandboxBuilder(Settings()).ensure_deps(repo, tmp_path / "artifacts", "process")
    JavaSandboxBuilder(Settings(sandbox_build_deps=False)).ensure_deps(
        repo, tmp_path / "artifacts", "docker"
    )
    assert not calls


def test_mirror_settings_are_ephemeral_and_never_passed_to_scans(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch, {"target/pmd.xml": "<pmd/>"})
    scanner = adapter(
        tmp_path, sandbox_mode="docker", maven_mirror="https://maven.example.com/repository/public"
    )
    scanner.run_static_checks(repo, make_change(), tools=["pmd"])
    assert all("-s" in command for _, command, online, _ in calls if online)
    assert "-s" not in calls[-1][1] and not calls[-1][2]
    assert not (repo / ".honest-ci/m2-config/provision-settings.xml").exists()


@pytest.mark.parametrize(
    "mirror",
    [
        "https://user:password@example.com/maven",
        "https://example.com/maven?token=secret",
        "file:///etc/passwd",
    ],
)
def test_credentials_cannot_be_embedded_in_mirror_configuration(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mirror: str,
) -> None:
    calls = canned(monkeypatch)
    with pytest.raises(DependencyProvisionError, match="HONEST_CI_MAVEN_MIRROR"):
        JavaSandboxBuilder(Settings(maven_mirror=mirror)).ensure_deps(
            repo, tmp_path / "artifacts", "docker"
        )
    assert not calls


def test_java_cache_cannot_escape_through_a_symlink(repo: Path, tmp_path: Path) -> None:
    (repo / ".honest-ci").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(DependencyProvisionError, match="unsafe"):
        JavaSandboxBuilder(Settings()).ensure_deps(repo, tmp_path / "artifacts", "docker")


def test_build_outputs_do_not_change_the_source_tree_fingerprint(repo: Path) -> None:
    before = hash_tree(repo)
    write(repo, "target/site/jacoco/jacoco.xml", JACOCO)
    assert hash_tree(repo) == before


def test_real_java_registration_routes_tests_and_keeps_python_first(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": JUNIT})
    router = LanguageRouter(make_settings(), tmp_path / "artifacts")
    assert SCANNER_ADAPTERS["java"] is JavaScannerAdapter
    assert router.choose(repo).language == "java"
    assert router.run_tests(repo).runner == "junit"
    assert calls[0][1][0] == "mvn"
    write(repo, "pyproject.toml", "[project]")
    assert router.choose(repo).language == "python"
    assert "java" in router.toolchain(repo)


def test_checkstyle_is_allowed_in_a_rule_library(repo: Path) -> None:
    write(
        repo,
        "honest-ci-rules.toml",
        'version="v1"\n[[rule]]\nid="R-java"\ntitle="Java style"\nkind="tool"\ntool="checkstyle"\nseverity="high"\n',
    )
    assert RuleSet.load(repo, Path("honest-ci-rules.toml"), "v1").rules[0].tool == "checkstyle"


@pytest.mark.parametrize("outcome", ["mergeable", "failed_test", "rule_violation", "no_tests"])
def test_java_change_reaches_the_correct_terminal_verdict(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    write(
        repo,
        "honest-ci-policy.toml",
        '[policy]\nversion="v1"\nrequired_checks=["review", "regression"]\n',
    )
    write(
        repo,
        "honest-ci-rules.toml",
        'version="v1"\n[[rule]]\nid="R-java"\ntitle="Java style"\nkind="tool"\ntool="checkstyle"\nseverity="high"\n',
    )
    text = (repo / SOURCE).read_text()
    change = make_change(
        diff=f"--- /dev/null\n+++ b/{SOURCE}\n@@ -0,0 +1 @@\n+{text}\n",
        files_changed=[],
        ai_generated_ratio=None,
    )
    checkstyle = "<checkstyle/>"
    if outcome == "rule_violation":
        checkstyle = f'<checkstyle><file name="/workspace/{SOURCE}"><error line="1" source="Rule" message="violation"/></file></checkstyle>'
    files = {"target/checkstyle-result.xml": checkstyle, "target/site/jacoco/jacoco.xml": JACOCO}
    if outcome != "no_tests":
        files["target/surefire-reports/TEST-suite.xml"] = (
            JUNIT
            if outcome == "failed_test"
            else '<testsuite><testcase classname="OrderTest" name="ok"/></testsuite>'
        )
    canned(monkeypatch, files)
    runner = GateRunner(
        repo, settings=replace(make_settings(), model_explain=False, model_judgment=False)
    )
    state = runner.run(runner.build_state(change))
    expected = {
        "mergeable": "mergeable",
        "failed_test": "blocked",
        "rule_violation": "blocked",
        "no_tests": "awaiting_human",
    }[outcome]
    assert state.status == expected
    assert state.evidence is not None
    if outcome == "rule_violation":
        finding = state.gate_reports["review"].findings[0]
        assert (finding.rule_id, finding.file, finding.line_start) == ("R-java", SOURCE, 1)
    if outcome == "failed_test":
        assert "OrderTest#bad" in str(state.gate_reports["regression"])


def test_too_many_java_reports_cannot_be_read_as_a_partial_pass(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("honest_ci.testreports.MAX_REPORTS", 1)
    canned(
        monkeypatch,
        {
            "one/target/checkstyle-result.xml": "<checkstyle/>",
            "two/target/checkstyle-result.xml": "<checkstyle/>",
        },
    )
    report = adapter(tmp_path).run_static_checks(repo, make_change(), tools=["checkstyle"])[0]
    assert not report.passed and "limit" in report.stderr


def test_an_unreadable_module_cannot_be_silently_excluded_from_coverage(
    repo: Path, tmp_path: Path
) -> None:
    write(repo, "one/target/site/jacoco/jacoco.xml", JACOCO)
    write(repo, "two/target/site/jacoco/jacoco.xml", "<broken")
    assert adapter(tmp_path).measure_coverage(repo).ratio is None


# ── the previous build is not this run's evidence ────────────────────────────


def test_every_run_drops_the_previous_builds_target_before_it_compiles(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once per run, offline, at the pinned version -- and before any goal.

    Maven leaves target/test-classes alone when a test source disappears, so
    without this Surefire keeps running a test that is no longer in the tree.
    """

    calls = canned(monkeypatch)
    scanner = adapter(tmp_path)
    scanner.run_static_checks(repo, make_change(), tools=["checkstyle"])
    scanner.run_tests(repo)
    cleans = ran(calls, "clean")
    assert len(cleans) == 1 and calls[0][0] == "clean"
    command, online = cleans[0][1], cleans[0][2]
    assert command == ["mvn", "-B", "-o", "-q", *command[4:]] and not online
    assert command[-1] == MAVEN_PLUGINS["clean"] + ":clean"
    assert calls.index(cleans[0]) < calls.index(ran(calls, "checkstyle")[0])


def test_a_target_that_cannot_be_cleaned_makes_every_tool_unavailable(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: stale classes are stale evidence, so nothing may run on them."""

    calls = canned(monkeypatch, codes={"clean": 1}, stdout="rm: target/classes: busy")
    scanner = adapter(tmp_path)
    report = scanner.run_static_checks(repo, make_change(), tools=["pmd"])[0]
    results = scanner.run_tests(repo)
    assert not report.available and "target/" in report.stderr
    assert not results.available and not results.passed
    assert not ran(calls, "pmd") and not ran(calls, "tests")


def test_a_clean_that_failed_is_said_where_a_person_reads_first(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maven failing to start reads as a scanner failing to start: the review
    # summary names the cause, and the block quotes what Maven said.
    write(repo, "honest-ci-policy.toml", '[policy]\nversion="v1"\nrequired_checks=["review"]\n')
    write(
        repo,
        "honest-ci-rules.toml",
        'version="v1"\n[[rule]]\nid="T-02"\ntitle="SpotBugs must pass"\nkind="tool"\n'
        'tool="spotbugs"\nseverity="high"\n',
    )
    said = "[ERROR] Non-parseable settings /maven/conf/settings.xml: in comment after two dashes"
    canned(monkeypatch, codes={"clean": 1}, stdout=said)
    runner = GateRunner(
        repo,
        settings=replace(
            make_settings(), model_explain=False, model_judgment=False, sandbox_mode="process"
        ),
    )

    state = runner.run(runner.build_state(make_change()))

    review = state.gate_reports["review"]
    assert state.status == "blocked"
    # The sentence whole, and Maven's words left to the finding.
    assert (
        "scanners that did not start: spotbugs — `Maven's clean goal failed, so the target/ "
        "directory the previous build left could not be deleted, and classes compiled into it "
        "cannot be evidence for this run.`." in review.summary
    )
    [finding] = [finding for finding in review.findings if finding.id == "review.unavailable"]
    assert f"cannot be evidence for this run.\n{said}\n" in finding.detail


@pytest.mark.parametrize("mode", ["process", "docker"])
def test_maven_reads_honest_cis_empty_global_settings_only_in_process_mode(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    # Maven 3.10 puts ${user.home} into its installation's settings.xml,
    # comments and all, before parsing it, and in process mode user.home is in
    # the workspace: a path with "--" in it is no longer XML, and every goal
    # stops. A container's /workspace has none, and cannot see this file.
    calls = canned(monkeypatch, codes={"maven-dependencies": 0})

    adapter(tmp_path, sandbox_mode=mode).run_tests(repo)

    commands = [command for _, command, _, _ in calls if command[0] == "mvn"]
    assert ran(calls, "clean") and ran(calls, "tests")
    for command in commands:
        if mode == "process":
            assert command[command.index("-gs") + 1] == str(MAVEN_GLOBAL_SETTINGS)
        else:
            assert "-gs" not in command


def test_the_global_settings_ship_and_give_maven_nothing_to_put_a_path_into() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    package = root / "src" / "honest_ci"
    shipped = {
        path
        for pattern in pyproject["tool"]["setuptools"]["package-data"]["honest_ci"]
        for path in package.glob(pattern)
    }
    text = MAVEN_GLOBAL_SETTINGS.read_text(encoding="utf-8")

    assert MAVEN_GLOBAL_SETTINGS.resolve() in shipped
    settings = ElementTree.fromstring(text)
    assert settings.tag == "settings" and len(settings) == 0
    assert "${" not in text and "<!--" not in text


def test_a_custom_test_command_owns_its_own_build(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": JUNIT})
    adapter(tmp_path, test_command="./gradlew test").run_tests(repo)
    assert not ran(calls, "clean")


# ── a JDK newer than the pinned tools read ───────────────────────────────────

# From the logs of examples/java-junit5 on JDK 27, shortened. SpotBugs stops
# with no report; PMD logs each JDK class it could not parse, exits 0 and
# writes a report that finds nothing.
SPOTBUGS_ON_JDK_27 = (
    "     [java] The following errors occurred during analysis:\n"
    "     [java]   Error scanning java/lang/Object for referenced classes\n"
    "     [java]     java.lang.IllegalArgumentException: Unsupported class file major version 71\n"
    "     [java]       At org.objectweb.asm.ClassReader.<init>(ClassReader.java:200)\n"
    '     [java] Exception in thread "main" java.lang.IllegalArgumentException: '
    "Unsupported class file major version 71\n"
    "[ERROR] Failed to execute goal com.github.spotbugs:spotbugs-maven-plugin:4.9.8.0:spotbugs "
    "(default-cli) on project refund-split: Execution default-cli of goal "
    "com.github.spotbugs:spotbugs-maven-plugin:4.9.8.0:spotbugs failed: Java returned: 1 -> "
    "[Help 1]\n"
)
PMD_ON_JDK_27 = (
    "[ERROR] Parsing failed in ParseLock#doParse() of ClassStub:java/lang/String\n"
    "java.lang.IllegalArgumentException: Unsupported class file major version 71\n"
    "    at org.objectweb.asm.ClassReader.<init> (ClassReader.java:200)\n"
)
CLEAN_PMD = '<pmd xmlns="http://pmd.sourceforge.net/report/2.0.0" version="7.17.0"></pmd>'


@pytest.mark.parametrize(
    "tool,output,code,files",
    [
        ("spotbugs", SPOTBUGS_ON_JDK_27, 1, {}),
        ("pmd", PMD_ON_JDK_27, 0, {"target/pmd.xml": CLEAN_PMD}),
    ],
    ids=["spotbugs", "pmd"],
)
def test_a_tool_that_could_not_read_the_jdks_classes_never_checked_and_names_the_jdk_to_use(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    output: str,
    code: int,
    files: dict[str, str],
) -> None:
    """The plugin ran: "check the plugin and the report path" sends nobody anywhere.

    PMD is the quiet one. Its clean report came from a run that resolved none
    of the JDK's types, so it is no evidence that the change is clean.
    """

    canned(monkeypatch, files, code=code, stdout=output)
    scanner = adapter(tmp_path, sandbox_mode="process")
    report = scanner.run_static_checks(repo, make_change(), tools=[tool])[0]

    assert not report.available and not report.passed and not report.diagnostics
    assert "major version 71 (Java 27)" in report.stderr
    assert f"up to Java {MAVEN_CLASS_FILE_CEILINGS[tool]}" in report.stderr
    assert f"JAVA_HOME at JDK {MAVEN_JDK_CEILING} or earlier" in report.stderr
    assert "report path" not in report.stderr


def test_in_docker_mode_the_jdk_to_change_is_the_images(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # JAVA_HOME never reaches the container, so naming it would send the
    # operator after a setting that changes nothing.
    canned(monkeypatch, code=1, codes={"maven-dependencies": 0}, stdout=SPOTBUGS_ON_JDK_27)
    scanner = adapter(tmp_path, sandbox_mode="docker")
    report = scanner.run_static_checks(repo, make_change(), tools=["spotbugs"])[0]

    assert "HONEST_CI_JAVA_DOCKER_IMAGE" in report.stderr and "JAVA_HOME" not in report.stderr


@pytest.mark.parametrize(
    "tool,severity,output,code,files",
    [
        ("spotbugs", "high", SPOTBUGS_ON_JDK_27, 1, {}),
        ("pmd", "medium", PMD_ON_JDK_27, 0, {"target/pmd.xml": CLEAN_PMD}),
    ],
    ids=["spotbugs-high", "pmd-medium"],
)
def test_a_tool_that_could_not_read_the_jdk_blocks_whatever_its_rules_severity(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    severity: str,
    output: str,
    code: int,
    files: dict[str, str],
) -> None:
    """A medium rule decides whether what the tool found blocks, not whether
    a tool that never examined the change counts as having passed it."""

    write(repo, "honest-ci-policy.toml", '[policy]\nversion="v1"\nrequired_checks=["review"]\n')
    write(
        repo,
        "honest-ci-rules.toml",
        f'version="v1"\n[[rule]]\nid="T-01"\ntitle="{tool} must pass"\nkind="tool"\n'
        f'tool="{tool}"\nseverity="{severity}"\n',
    )
    text = (repo / SOURCE).read_text()
    change = make_change(
        diff=f"--- /dev/null\n+++ b/{SOURCE}\n@@ -0,0 +1 @@\n+{text}\n",
        files_changed=[],
        ai_generated_ratio=None,
    )
    canned(monkeypatch, files, code=code, stdout=output)
    runner = GateRunner(
        repo,
        settings=replace(
            make_settings(), model_explain=False, model_judgment=False, sandbox_mode="process"
        ),
    )

    state = runner.run(runner.build_state(change))

    assert state.status == "blocked"
    review = state.gate_reports["review"]
    [finding] = review.findings
    assert finding.id == "review.unavailable" and finding.blocking
    # Whole, from its first words to the remedy, in the evidence pack; and in
    # the summary the terminal and the pull request comment show.
    said = f"{tool} could not read class files of major version 71 (Java 27)"
    assert said in finding.detail
    assert "honest-ci doctor shows the JDK a run gets." in finding.detail
    assert said in review.summary


@pytest.mark.parametrize(
    "printed,release,version",
    [
        ("Java version: 27, vendor: Homebrew, runtime: /opt/homebrew/openjdk", 27, "27"),
        ("Java version: 21.0.13, vendor: Eclipse Adoptium, runtime: /usr/lib/jvm/21", 21, "21.0.13"),
        ("Java version: 17.0.2, vendor: Oracle Corporation, runtime: /jdk-17", 17, "17.0.2"),
        ("Java version: 11.0.25, vendor: Azul Systems, Inc., runtime: /zulu-11", 11, "11.0.25"),
        ("Java version: 1.8.0_392, vendor: Temurin, runtime: /jdk8/jre", 8, "1.8.0_392"),
        ("Java version: 26-ea, vendor: Oracle Corporation, runtime: /jdk-26", 26, "26-ea"),
    ],
)
def test_the_jdk_mvn_reports_is_read_in_each_spelling_java_has_had(
    printed: str, release: int, version: str
) -> None:
    java = maven_java(f"Apache Maven 3.9.11\nMaven home: /maven\n{printed}\nOS name: linux\n")
    assert java is not None and (java.release, java.version) == (release, version)
    assert java.runtime is not None and java.runtime.startswith("/")


def test_a_mvn_that_names_no_jdk_is_not_read_as_one() -> None:
    assert maven_java("The JAVA_HOME environment variable is not defined correctly,") is None


# ── artifacts nobody declared ────────────────────────────────────────────────

# Copied from the logs of the runs in examples/java-junit5: two different
# goals, two different sentences, one wall. Both open with the coordinate of
# the plugin that *ran*, which is not the artifact that is missing.
SUREFIRE_OFFLINE = (
    "[ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test "
    "(default-test) on project refund-split: The following artifacts could not be resolved: "
    "org.apache.maven.surefire:surefire-junit-platform:jar:3.5.2 (absent): Cannot access central "
    "(https://repo.maven.apache.org/maven2) in offline mode and the artifact "
    "org.apache.maven.surefire:surefire-junit-platform:jar:3.5.2 has not been downloaded from it "
    "before. -> [Help 1]"
)
PMD_OFFLINE = (
    "[ERROR] Failed to execute goal org.apache.maven.plugins:maven-pmd-plugin:3.28.0:pmd (pmd) on "
    "project refund-split: An error has occurred in PMD report generation. Failed to retrieve "
    "skin artifact: The skin does not exist: Cannot access central "
    "(https://repo.maven.apache.org/maven2) in offline mode and the artifact "
    "org.apache.maven.skins:maven-fluido-skin:jar:2.0.0-M9 has not been downloaded from it "
    "before. -> [Help 1]"
)


@pytest.mark.parametrize(
    "output,missing,fetchable,goal",
    [
        (
            SUREFIRE_OFFLINE,
            "org.apache.maven.surefire:surefire-junit-platform:jar:3.5.2",
            "org.apache.maven.surefire:surefire-junit-platform:3.5.2",
            "org.apache.maven.plugins:maven-surefire-plugin:3.5.2",
        ),
        (
            PMD_OFFLINE,
            "org.apache.maven.skins:maven-fluido-skin:jar:2.0.0-M9",
            "org.apache.maven.skins:maven-fluido-skin:2.0.0-M9",
            "org.apache.maven.plugins:maven-pmd-plugin:3.28.0",
        ),
    ],
)
def test_a_missing_download_names_the_coordinate_and_not_the_goal(
    output: str, missing: str, fetchable: str, goal: str
) -> None:
    artifacts = unprovisioned_artifacts(output)
    assert artifacts == (missing,)
    hint = unprovisioned_hint(artifacts)
    # Named the way HONEST_CI_MAVEN_EXTRA_ARTIFACTS hands it to
    # dependency:get, which reads `...:jar:3.5.2` as version `jar`.
    assert fetchable in hint and missing not in hint and goal not in hint
    assert "HONEST_CI_MAVEN_EXTRA_ARTIFACTS" in hint


@pytest.mark.parametrize(
    "printed,fetchable",
    [
        ("g.roup:art:1.0", "g.roup:art:1.0"),
        ("g.roup:art:jar:1.0", "g.roup:art:1.0"),
        ("g.roup:art:pom:1.0", "g.roup:art:1.0:pom"),
        ("g.roup:art:jar:tests:1.0", "g.roup:art:1.0:jar:tests"),
    ],
)
def test_a_coordinate_maven_printed_is_handed_on_in_the_order_dependency_get_reads(
    printed: str, fetchable: str
) -> None:
    # Maven prints groupId:artifactId:extension[:classifier]:version;
    # dependency:get -Dartifact= reads groupId:artifactId:version[:packaging[:classifier]].
    assert get_coordinate(printed) == fetchable


def test_an_ordinary_failure_is_not_read_as_a_missing_download() -> None:
    assert unprovisioned_artifacts("[ERROR] COMPILATION ERROR : bad symbol") is None


FAKE_MVN = """#!/bin/sh
for arg; do
  case "$arg" in
    -Dartifact=*) echo "${arg#-Dartifact=}" >> "$FETCHED"; [ "$arg" = "-Dartifact=$FAILS" ] && exit 1 ;;
  esac
done
exit 0
"""


@pytest.mark.parametrize("shell", ["sh", "bash"])
@pytest.mark.parametrize("fails,fetched,exit_code", [("", 3, 0), ("b:b:1", 2, 1)])
def test_the_download_line_stops_at_the_first_failure_and_its_exit_code_says_so(
    tmp_path: Path, shell: str, fails: str, fetched: int, exit_code: int
) -> None:
    # Pasted into a terminal or put in a script, the line is read by its exit
    # code: 0 must mean every coordinate is in, and a failure must not close
    # the terminal it was pasted into, which the subshell keeps it from doing.
    # Only the shells every runner has: a case skipped where zsh is missing
    # would pass CI without ever running there.
    bin_dir = tmp_path / "bin"
    write(bin_dir, "mvn", FAKE_MVN)
    (bin_dir / "mvn").chmod(0o755)
    line = fetch_command(["a:a:1", "b:b:1", "c:c:1"], tmp_path / "host m2")
    log = tmp_path / "fetched"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FETCHED": str(log),
        "FAILS": fails,
    }

    done = subprocess.run([shell, "-c", f"{line}; echo after"], env=env, capture_output=True)

    assert len(log.read_text().split()) == fetched
    assert done.stdout.decode().strip() == "after"
    assert subprocess.run([shell, "-c", line], env=env).returncode == exit_code


@pytest.mark.parametrize("stage", ["scan", "tests"])
def test_a_run_that_stopped_on_a_download_is_unavailable_not_a_violation(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """It never read a source file, so it has no verdict about one.

    Fail-closed blocks either way; the difference is whether the operator is
    handed a coordinate to pre-fetch or a rule violation to argue about.
    """

    # Docker mode with provisioning on, the one setup where a coordinate added
    # to HONEST_CI_MAVEN_EXTRA_ARTIFACTS is downloaded on the next run.
    canned(monkeypatch, code=1, codes={"maven-dependencies": 0}, stdout=SUREFIRE_OFFLINE)
    scanner = adapter(tmp_path, sandbox_mode="docker")
    if stage == "scan":
        report = scanner.run_static_checks(repo, make_change(), tools=["pmd"])[0]
        available, reason = report.available, report.stderr
    else:
        results = scanner.run_tests(repo)
        available, reason = results.available, results.stderr
    assert not available
    assert "surefire-junit-platform" in reason and "HONEST_CI_MAVEN_EXTRA_ARTIFACTS" in reason


@pytest.mark.parametrize(
    "settings,why",
    [
        ({"sandbox_mode": "process"}, "process mode"),
        ({"sandbox_mode": "docker", "sandbox_build_deps": False}, "HONEST_CI_SANDBOX_BUILD_DEPS=0"),
    ],
)
def test_a_run_that_provisions_nothing_is_not_told_provisioning_will_fetch_it(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, Any],
    why: str,
) -> None:
    """``mvn -o`` against a local repository nothing ever filled.

    Following "add it to HONEST_CI_MAVEN_EXTRA_ARTIFACTS and run again" here
    changes nothing: the step that reads that list never runs, so the operator
    is sent round the same block for good.
    """

    canned(monkeypatch, code=1, stdout=SUREFIRE_OFFLINE)
    results = adapter(tmp_path, **settings).run_tests(repo)
    assert not results.available
    assert "surefire-junit-platform" in results.stderr
    assert "HONEST_CI_MAVEN_EXTRA_ARTIFACTS" not in results.stderr
    assert why in results.stderr


SUREFIRE_PROVIDER = "org.apache.maven.surefire:surefire-junit-platform:3.5.2"


def test_process_mode_reads_this_machines_repository_and_never_writes_to_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = fill_repository(tmp_path / "home-m2", [SUREFIRE_PROVIDER])
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": WHOLE_SUITE})

    adapter(tmp_path, sandbox_mode="process", maven_host_repository=host).run_tests(repo)

    # Searched after the workspace's own repository, which takes every write:
    # nothing provisions in process mode, and the developer's ~/.m2 is theirs.
    for name in ("clean", "tests"):
        command = ran(calls, name)[0][1]
        assert f"-Dmaven.repo.local={repo.resolve()}/.honest-ci/m2" in command
        assert f"-Dmaven.repo.local.tail={host}" in command


@pytest.mark.parametrize(
    "mode,host",
    [
        # A container cannot see the host's directory.
        ("docker", "there"),
        # Switched off, or not there to read.
        ("process", "off"),
        ("process", "absent"),
    ],
)
def test_no_machine_repository_is_named_where_none_can_be_read(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, host: str
) -> None:
    calls = canned(monkeypatch, codes={"maven-dependencies": 0})
    path = {"there": tmp_path, "off": None, "absent": tmp_path / "absent"}[host]

    adapter(tmp_path, sandbox_mode=mode, maven_host_repository=path).run_tests(repo)

    assert ran(calls, "tests")
    assert not any(
        part.startswith("-Dmaven.repo.local.tail") for _, command, _, _ in calls for part in command
    )


def test_a_gap_in_this_machines_repository_comes_with_the_line_that_fills_it(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = fill_repository(tmp_path / "home-m2", [])
    canned(monkeypatch, code=1, stdout=SUREFIRE_OFFLINE)

    results = adapter(tmp_path, sandbox_mode="process", maven_host_repository=host).run_tests(repo)

    assert not results.available
    assert str(host) in results.stderr
    # A command an operator can paste, fetching into the repository the next
    # run reads, in the order dependency:get reads.
    assert f"-Dmaven.repo.local={host}" in results.stderr
    assert f"{SUREFIRE_PROVIDER}; do mvn" in results.stderr
    assert "honest-ci doctor" in results.stderr
    assert "HONEST_CI_MAVEN_EXTRA_ARTIFACTS" not in results.stderr


def test_a_maven_too_old_to_read_this_machines_repository_is_told_so(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ~/.m2 has the artifact and Maven still reported it missing: a Maven
    # before 3.9 ignores maven.repo.local.tail. Telling this operator to fetch
    # it would send them round to the same block.
    host = fill_repository(tmp_path / "home-m2", [SUREFIRE_PROVIDER])
    canned(monkeypatch, code=1, stdout=SUREFIRE_OFFLINE)

    results = adapter(tmp_path, sandbox_mode="process", maven_host_repository=host).run_tests(repo)

    assert not results.available
    assert "Maven 3.9 or later" in results.stderr
    assert "do mvn" not in results.stderr


@pytest.mark.parametrize("held", [[SUREFIRE_PROVIDER], []], ids=["held", "missing"])
def test_a_custom_test_command_is_not_said_to_read_this_machines_repository(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: list[str]
) -> None:
    # It runs as written, so it was never handed maven.repo.local.tail: "this
    # Maven is too old" or "fetch it into ~/.m2" would each send the operator
    # after a fix for a command that never ran.
    host = fill_repository(tmp_path / "home-m2", held)
    calls = canned(monkeypatch, code=1, stdout=SUREFIRE_OFFLINE)

    results = adapter(
        tmp_path, sandbox_mode="process", maven_host_repository=host, test_command="mvn -o test"
    ).run_tests(repo)

    assert ran(calls, "tests")[0][1] == ["mvn", "-o", "test"]
    assert not results.available
    assert "surefire-junit-platform" in results.stderr
    assert str(host) not in results.stderr and "Maven 3.9" not in results.stderr


def test_extra_artifacts_are_prefetched_and_are_part_of_the_cache_fingerprint(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extra = "org.junit.platform:junit-platform-commons:1.9.3"
    settings = Settings(maven_extra_artifacts=(extra,))
    calls = canned(monkeypatch)
    JavaSandboxBuilder(settings).ensure_deps(repo, tmp_path / "artifacts", "docker")
    assert f"-Dartifact={extra}" in [command[-1] for _, command, _, _ in calls]
    # A coordinate is added to this list *because* the last run failed on it:
    # the cache that failed must not be reused as if it were complete.
    calls.clear()
    JavaSandboxBuilder(replace(settings, maven_extra_artifacts=())).ensure_deps(
        repo, tmp_path / "artifacts", "docker"
    )
    assert calls


def test_a_provisioning_timeout_says_it_was_a_timeout(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit code 124 is the runner's own timeout, not a broken pom.

    Without naming it, the operator reads "provisioning failed (exit code
    124)" and goes looking through their dependencies for something that is
    not there.
    """

    canned(monkeypatch, code=124, codes={"clean": 124})
    report = adapter(tmp_path, sandbox_mode="docker").run_static_checks(
        repo, make_change(), tools=["pmd"]
    )[0]

    assert not report.available
    assert "HONEST_CI_PROVISION_TIMEOUT_SECONDS" in report.stderr
    assert "HONEST_CI_MAVEN_MIRROR" in report.stderr


def test_provisioning_gets_its_own_timeout_not_the_scanners(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cold cache pulls the whole toolchain; a scanner run does not.

    Sharing one timeout reported the first run of every Java repository as
    "no Maven tools available", which fail-closed turns into a blocked change.
    """

    timeouts: list[int] = []

    def run(
        self: ToolRunner, name: str, command: list[str], *args: Any, **kwargs: Any
    ) -> ToolReport:
        timeouts.append(self.timeout_seconds)
        return ToolReport(name=name, command=" ".join(command), exit_code=0, passed=True)

    monkeypatch.setattr(ToolRunner, "available", lambda self, name: True)
    monkeypatch.setattr(ToolRunner, "run", run)
    settings = Settings(provision_timeout_seconds=4242, command_timeout_seconds=7)
    JavaSandboxBuilder(settings).ensure_deps(repo, tmp_path / "artifacts", "docker")
    assert timeouts and set(timeouts) == {4242}


# ── a test command the toolchain did not compose ─────────────────────────────

#: The whole suite, which is what a custom command runs whatever it is asked.
WHOLE_SUITE = (
    "<testsuite>"
    '<testcase classname="com.acme.OrderTest" name="total"/>'
    '<testcase classname="com.acme.UserTest" name="login"/>'
    '<testcase classname="com.acme.UserTest" name="logout"/>'
    "</testsuite>"
)

GENERATED_ORDER_CASE = ProposedTest(
    path="src/test/java/honest_ci/generated/OrderRegressionTest.java",
    content="package honest_ci.generated;\n\nclass OrderRegressionTest {\n    void total() {}\n}\n",
    targets=["Order"],
)


def regression_gate(
    repo: Path, tmp_path: Path, llm: ScriptedLLM | None = None, **settings: Any
) -> tuple[GateRunner, RunState]:
    """The real Java toolchain behind the router, a change to ``Order``, and a test naming it."""

    # One declaration per line: the shape impact analysis reads symbols from.
    source = ["package com.acme;", "", "public class Order {", "    int total;", "}"]
    write(repo, SOURCE, "\n".join(source) + "\n")
    write(repo, "src/test/java/com/acme/OrderTest.java", "package com.acme; class OrderTest { Order o; }")
    write(repo, "src/test/java/com/acme/UserTest.java", "package com.acme; class UserTest {}")
    added = "".join(f"+{line}\n" for line in source)
    change = make_change(
        diff=f"--- /dev/null\n+++ b/{SOURCE}\n@@ -0,0 +1,{len(source)} @@\n{added}",
        files_changed=[],
    )
    configured = replace(make_settings(), coverage_threshold=0.0, **settings)
    runner = GateRunner(
        repo,
        settings=configured,
        adapters=AdapterSet(
            ledger=SQLiteLedger(tmp_path / "ledger.sqlite3"),
            scanner=LanguageRouter(configured, tmp_path / "artifacts"),
            llm=llm,
        ),
    )
    return runner, RunState(run_id="r1", change=change)


def test_a_custom_command_is_never_reported_as_running_only_the_impacted_tests(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": WHOLE_SUITE})
    runner, state = regression_gate(
        repo, tmp_path, test_command="./gradlew test", test_selection="impacted"
    )

    report = runner.gates["regression"].run(state)

    # The command ran as written, so the whole suite ran; the evidence has to
    # say so, and why the selection could not be applied.
    assert ran(calls, "tests")[0][1] == ["./gradlew", "test"]
    selection = report.verifier_report["test_selection"]
    assert selection["mode"] == "all"
    assert "HONEST_CI_TEST_COMMAND" in selection["fallback"]
    assert "only the 1 impacted test ran" not in report.summary


def test_generated_cases_are_not_credited_with_a_custom_commands_whole_suite(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = canned(monkeypatch, {"target/surefire-reports/TEST-suite.xml": WHOLE_SUITE})
    llm = ScriptedLLM(tests=[GENERATED_ORDER_CASE])
    runner, state = regression_gate(repo, tmp_path, llm=llm, test_command="./gradlew test")

    report = runner.gates["regression"].run(state)

    # Run under a command that ignores `-Dtest=`, "the generated cases" were
    # the whole suite: three repository tests reported as three generated
    # regression tests that passed.
    assert report.test_results is not None
    assert report.test_results.generated == 0
    generated = report.verifier_report["generated_tests"]
    assert generated["ran"] is False
    assert "HONEST_CI_TEST_COMMAND" in generated["not_generated_reason"]
    assert len(ran(calls, "tests")) == 1
    # Nothing was going to run what the model wrote, so it was not asked.
    assert llm.testgen_symbols == []

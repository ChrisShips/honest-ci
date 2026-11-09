"""JUnit XML: the universal outlet for suites that are not pytest."""

from __future__ import annotations

from pathlib import Path

import pytest

from honest_ci import testreports
from honest_ci.testreports import MAX_REPORTS, ReportError, parse_junit

SUREFIRE = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.acme.OrderTest" tests="4" failures="1" errors="1" skipped="1">
  <testcase classname="com.acme.OrderTest" name="splitsEvenly" time="0.01"/>
  <testcase classname="com.acme.OrderTest" name="rejectsNegative" time="0.01">
    <failure message="expected: &lt;0&gt; but was: &lt;-1&gt;" type="AssertionFailedError">stack</failure>
  </testcase>
  <testcase classname="com.acme.OrderTest" name="loadsFixture" time="0.01">
    <error type="java.io.IOException">java.io.IOException: fixture missing
    at com.acme.OrderTest.loadsFixture(OrderTest.java:40)</error>
  </testcase>
  <testcase classname="com.acme.OrderTest" name="pending" time="0">
    <skipped/>
  </testcase>
</testsuite>
"""

PYTEST = """<testsuites>
  <testsuite name="pytest" tests="2">
    <testcase classname="tests.test_refund" file="tests/test_refund.py" line="3" name="test_split"/>
    <testcase classname="tests.test_refund" file="tests/test_refund.py" line="9" name="test_tail">
      <failure message="assert 1 == 2">traceback</failure>
    </testcase>
  </testsuite>
</testsuites>
"""


def test_counts_come_from_the_cases_not_the_suite_attributes() -> None:
    summary = parse_junit(SUREFIRE.replace('tests="4"', 'tests="99"'))

    assert (summary.passed, summary.failed, summary.errors, summary.skipped) == (1, 1, 1, 1)
    assert summary.collected == 4


def test_failure_lines_use_the_junit_class_form() -> None:
    summary = parse_junit(SUREFIRE)

    assert summary.failures == (
        "FAILED com.acme.OrderTest#rejectsNegative - expected: <0> but was: <-1>",
        "ERROR com.acme.OrderTest#loadsFixture - java.io.IOException: fixture missing",
    )


def test_a_pytest_report_keeps_the_path_form_the_gate_already_reads() -> None:
    summary = parse_junit(PYTEST)

    assert summary.failures == ("FAILED tests/test_refund.py::test_tail - assert 1 == 2",)
    assert summary.passed == 1


def test_a_report_with_a_dtd_is_refused_not_parsed() -> None:
    hostile = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><testsuite/>'
    with pytest.raises(ReportError, match="DTD"):
        parse_junit(hostile)


def test_malformed_xml_is_a_problem_not_zero_tests() -> None:
    with pytest.raises(ReportError, match="not well-formed"):
        parse_junit("<testsuite><testcase name='x'>")


def test_an_unrelated_document_is_not_a_test_report() -> None:
    with pytest.raises(ReportError, match="root element is <project>"):
        parse_junit("<project/>")


def test_reading_several_reports_sums_them_and_names_the_broken_one(tmp_path: Path) -> None:
    reports = tmp_path / "target" / "surefire-reports"
    reports.mkdir(parents=True)
    (reports / "TEST-a.xml").write_text(SUREFIRE, encoding="utf-8")
    (reports / "TEST-b.xml").write_text(PYTEST, encoding="utf-8")
    (reports / "TEST-c.xml").write_text("not xml at all", encoding="utf-8")

    found = testreports.find_reports(tmp_path, "target/surefire-reports/*.xml")
    summary = testreports.read_reports(found, tmp_path)

    assert summary.reports == (
        "target/surefire-reports/TEST-a.xml",
        "target/surefire-reports/TEST-b.xml",
    )
    assert (summary.passed, summary.failed, summary.errors, summary.skipped) == (2, 2, 1, 1)
    assert len(summary.problems) == 1
    assert summary.problems[0].startswith("target/surefire-reports/TEST-c.xml: not well-formed")


def test_reports_outside_the_workspace_are_never_read(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "junit.xml"
    outside.parent.mkdir()
    outside.write_text(PYTEST, encoding="utf-8")
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "link.xml").symlink_to(outside)

    assert testreports.find_reports(workspace, "../elsewhere/*.xml") == []
    assert testreports.find_reports(workspace, "/elsewhere/*.xml") == []
    # A symlink inside the tree that points out of it is the same thing.
    assert testreports.find_reports(workspace, "*.xml") == []


def test_reports_under_runtime_directories_are_ignored(tmp_path: Path) -> None:
    buried = tmp_path / ".venv" / "lib" / "junit.xml"
    buried.parent.mkdir(parents=True)
    buried.write_text(PYTEST, encoding="utf-8")
    (tmp_path / "junit.xml").write_text(PYTEST, encoding="utf-8")

    found = testreports.find_reports(tmp_path, "**/junit.xml")

    assert [path.relative_to(tmp_path).as_posix() for path in found] == ["junit.xml"]


def test_the_number_of_reports_read_is_bounded(tmp_path: Path) -> None:
    for index in range(MAX_REPORTS + 3):
        (tmp_path / f"r{index:04d}.xml").write_text("<testsuite/>", encoding="utf-8")

    assert len(testreports.find_reports(tmp_path, "*.xml")) == MAX_REPORTS


def test_clearing_reports_removes_what_an_earlier_run_left(tmp_path: Path) -> None:
    (tmp_path / "junit.xml").write_text(PYTEST, encoding="utf-8")

    assert testreports.clear_reports(tmp_path, "junit.xml") == 1
    assert not (tmp_path / "junit.xml").exists()
    assert testreports.clear_reports(tmp_path, "junit.xml") == 0


def test_an_oversized_report_is_a_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(testreports, "MAX_REPORT_BYTES", 10)
    (tmp_path / "junit.xml").write_text(PYTEST, encoding="utf-8")

    summary = testreports.read_reports([tmp_path / "junit.xml"], tmp_path)

    assert summary.reports == ()
    assert "above the 10-byte limit" in summary.problems[0]


def test_no_reports_is_an_empty_summary(tmp_path: Path) -> None:
    summary = testreports.read_reports([], tmp_path)
    assert summary.collected == 0 and summary.reports == () and summary.problems == ()

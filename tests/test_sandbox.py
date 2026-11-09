from __future__ import annotations

from pathlib import Path

from honest_ci.sandbox import SandboxBuilder
from honest_ci.settings import Settings


def builder(**overrides: object) -> SandboxBuilder:
    return SandboxBuilder(Settings(**overrides))  # type: ignore[arg-type]


def test_dependency_detection(tmp_path: Path) -> None:
    assert not builder().has_dependencies(tmp_path)
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    assert builder().has_dependencies(tmp_path)


def test_declared_test_extra_is_installed_when_present(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
        [project]
        name = "target"
        [project.optional-dependencies]
        dev = ["pytest"]
        """,
        encoding="utf-8",
    )
    assert builder()._editable_target(pyproject) == ".[dev]"


def test_malformed_pyproject_falls_back_to_a_plain_install(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("this is not toml = = =", encoding="utf-8")
    assert builder()._editable_target(pyproject) == "."


def test_process_mode_provisions_nothing(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    assert builder().ensure_deps(tmp_path, tmp_path / "artifacts", "process") is None


def test_docker_mode_without_a_manifest_provisions_nothing(tmp_path: Path) -> None:
    assert builder().ensure_deps(tmp_path, tmp_path / "artifacts", "docker") is None

from __future__ import annotations

import re
from pathlib import Path

import pytest

from honest_ci.settings import Settings, SettingsError


def test_defaults_are_the_safe_direction() -> None:
    settings = Settings()
    # A verification layer blocks when it cannot verify, and isolates by default.
    assert settings.fail_closed
    assert settings.sandbox_mode == "docker"
    # No change leaves for a model vendor unless someone named one.
    assert settings.model_provider is None
    assert Settings(sandbox_mode="process").effective_sandbox_isolation() == "process"


def test_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONEST_CI_FAIL_CLOSED", "0")
    monkeypatch.setenv("HONEST_CI_SANDBOX_MODE", "process")
    monkeypatch.setenv("HONEST_CI_GATE_MAX_ATTEMPTS", "review=5,regression=bogus,=2,spec=0")
    monkeypatch.setenv("HONEST_CI_RETRY_BACKOFF_SECONDS", "0,2,nope,8")
    monkeypatch.setenv("HONEST_CI_MAX_ATTEMPTS", "not-a-number")
    monkeypatch.setenv("HONEST_CI_LEDGER_PATH", ".honest-ci/custom.sqlite3")
    monkeypatch.setenv("HONEST_CI_DISABLED_GATES", " regression, spec,regression,,")
    monkeypatch.setenv("HONEST_CI_TEST_TIMEOUT_SECONDS", "600")

    settings = Settings.from_env()

    assert settings.fail_closed is False
    assert settings.sandbox_mode == "process"
    assert settings.gate_max_attempts == {"review": 5, "spec": 1}
    assert settings.retry_backoff_seconds == (0.0, 2.0, 8.0)
    assert settings.max_attempts == 3  # malformed value falls back to the default
    assert settings.ledger_path == Path(".honest-ci/custom.sqlite3")
    assert settings.disabled_gates == ("regression", "spec")  # deduplicated, order kept
    assert settings.test_timeout_seconds == 600


def test_nothing_is_disabled_unless_the_environment_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONEST_CI_DISABLED_GATES", raising=False)
    assert Settings.from_env().disabled_gates == ()


def test_attempt_and_backoff_helpers() -> None:
    settings = Settings(
        max_attempts=2,
        gate_max_attempts={"review": 4},
        retry_backoff_seconds=(0.0, 1.0, 5.0),
    )
    assert settings.max_attempts_for("review") == 4
    assert settings.max_attempts_for("spec") == 2
    assert settings.retry_delay_for(1) == 0.0
    assert settings.retry_delay_for(2) == 1.0
    assert settings.retry_delay_for(99) == 5.0  # clamped to the last step


def test_provider_selection_is_honest_about_missing_credentials() -> None:
    assert Settings().active_provider is None
    assert Settings().active_model == "none"

    configured = Settings(model_provider="qwen", qwen_api_key="sk-q")
    assert configured.active_provider == "qwen"
    assert configured.active_model == "qwen3.8-max"
    assert "DASHSCOPE" in configured.credential_hint


def test_a_key_alone_does_not_switch_a_model_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keys get exported for other tools. Sending this repository's changes to
    # a vendor is a decision somebody makes by naming it, not a side effect.
    monkeypatch.delenv("HONEST_CI_MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmnopqrstuvwx")

    settings = Settings.from_env()

    assert settings.model_provider is None
    assert settings.active_provider is None
    assert settings.active_model == "none"


def test_naming_a_provider_is_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", " DeepSeek ")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-abcdefghijklmnopqrstuvwx")

    settings = Settings.from_env()

    assert settings.model_provider == "deepseek"
    assert settings.active_provider == "deepseek"


def test_an_empty_provider_is_the_same_as_an_unset_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "  ")

    assert Settings.from_env().model_provider is None


def test_openai_and_anthropic_are_read_from_honest_ci_names_and_vendor_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("HONEST_CI_OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("HONEST_CI_OPENAI_MODEL", "deepseek-chat")
    monkeypatch.setenv("HONEST_CI_OPENAI_MAX_TOKENS", "8192")
    monkeypatch.setenv("HONEST_CI_OPENAI_MAX_TOKENS_PARAM", " MAX_TOKENS ")
    monkeypatch.setenv("HONEST_CI_OPENAI_TEMPERATURE", "0.2")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("HONEST_CI_ANTHROPIC_MODEL", "claude-sonnet-5")
    monkeypatch.delenv("HONEST_CI_ANTHROPIC_BASE_URL", raising=False)

    settings = Settings.from_env()

    assert settings.active_provider == "openai"
    assert settings.active_model == "deepseek-chat"
    assert settings.base_url_for("openai") == "https://api.deepseek.com/v1"
    assert settings.openai_max_tokens == 8192
    assert settings.openai_max_tokens_param == "max_tokens"
    assert settings.openai_temperature == 0.2
    assert settings.api_key_for("anthropic") == "sk-a"
    assert settings.model_for("anthropic") == "claude-sonnet-5"
    assert settings.base_url_for("anthropic") == "https://api.anthropic.com"


def test_the_vendors_own_base_url_variables_do_not_move_this_products_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exported for some other tool; following them would send this product's
    # key and diffs wherever that tool was pointed.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")
    for name in ("HONEST_CI_OPENAI_BASE_URL", "HONEST_CI_ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.base_url_for("openai") == "https://api.openai.com/v1"
    assert settings.base_url_for("anthropic") == "https://api.anthropic.com"


def test_no_temperature_is_sent_unless_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONEST_CI_OPENAI_TEMPERATURE", raising=False)
    monkeypatch.setenv("HONEST_CI_ANTHROPIC_TEMPERATURE", "warm")  # not a number: unset

    settings = Settings.from_env()

    assert settings.openai_temperature is None
    assert settings.anthropic_temperature is None
    assert Settings().openai_max_tokens_param == "max_completion_tokens"


def test_a_max_tokens_field_nobody_reads_is_refused_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Handed an unknown field, a server drops the cap silently or refuses
    # every call -- and either one looks like a model problem, not a typo.
    monkeypatch.setenv("HONEST_CI_OPENAI_MAX_TOKENS_PARAM", "max_output_tokens")

    with pytest.raises(SettingsError, match="HONEST_CI_OPENAI_MAX_TOKENS_PARAM=max_output_tokens"):
        Settings.from_env()


def test_the_new_providers_name_their_keys_when_missing() -> None:
    assert Settings(model_provider="openai").credential_hint == "OPENAI_API_KEY"
    assert Settings(model_provider="anthropic").credential_hint == "ANTHROPIC_API_KEY"
    assert Settings(model_provider="openai").active_provider is None


def test_a_provider_this_build_does_not_know_is_refused_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Read as "off", a typo would quietly switch off what the operator meant on.
    monkeypatch.setenv("HONEST_CI_MODEL_PROVIDER", "deepsek")

    with pytest.raises(SettingsError, match="HONEST_CI_MODEL_PROVIDER=deepsek"):
        Settings.from_env()


def test_tri_state_thinking_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONEST_CI_QWEN_ENABLE_THINKING", raising=False)
    assert Settings.from_env().qwen_enable_thinking is None
    monkeypatch.setenv("HONEST_CI_QWEN_ENABLE_THINKING", "yes")
    assert Settings.from_env().qwen_enable_thinking is True
    monkeypatch.setenv("HONEST_CI_QWEN_ENABLE_THINKING", "off")
    assert Settings.from_env().qwen_enable_thinking is False


def test_forge_lookup() -> None:
    settings = Settings(forge="github", github_token="ghp-x")
    assert settings.forge_token() == "ghp-x"
    assert settings.forge_base_url() == "https://api.github.com"
    assert Settings(forge="local").forge_token() is None
    # A platform left out of this edition reads no token, even if one is set.
    assert Settings(forge="gitlab", github_token="ghp-x").forge_token() is None


def test_language_and_test_report_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONEST_CI_LANGUAGE", " Java ")
    monkeypatch.setenv("HONEST_CI_TEST_REPORT", "target/surefire-reports/*.xml")

    settings = Settings.from_env()

    assert settings.language == "java"
    assert settings.test_report == "target/surefire-reports/*.xml"


def test_the_toolchain_is_detected_unless_the_environment_says_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONEST_CI_LANGUAGE", raising=False)
    monkeypatch.delenv("HONEST_CI_TEST_REPORT", raising=False)
    assert Settings.from_env().language == "auto"
    assert Settings.from_env().test_report is None


def test_the_changed_line_bar_and_the_selection_are_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD", "0.8")
    monkeypatch.setenv("HONEST_CI_TEST_SELECTION", " Impacted ")

    settings = Settings.from_env()

    assert settings.changed_line_coverage_threshold == 0.8
    assert settings.test_selection == "impacted"


def test_both_regression_v1_switches_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD", raising=False)
    monkeypatch.delenv("HONEST_CI_TEST_SELECTION", raising=False)

    settings = Settings.from_env()

    # The bar a deployment has not asked for must not start blocking its
    # changes, and a suite nobody asked to narrow runs whole.
    assert settings.changed_line_coverage_threshold == 0.0
    assert settings.test_selection == "all"


def test_a_selection_nobody_implements_runs_the_whole_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONEST_CI_TEST_SELECTION", "smart")

    # A typo must fall back to more testing, never to less.
    assert Settings.from_env().test_selection == "all"


# ── ratios ───────────────────────────────────────────────────────────────────


def test_a_threshold_written_as_a_percentage_is_refused_at_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What dogfooding typed: 75, meaning 75%. Taken at face value it is a bar
    # of 7500%, which blocks every change forever; clamped to 1.0 it demands
    # full coverage while looking like it accepted the number.
    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "75")

    with pytest.raises(SettingsError, match="0.75"):
        Settings.from_env()


@pytest.mark.parametrize(
    "name",
    [
        "HONEST_CI_COVERAGE_THRESHOLD",
        "HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD",
        "HONEST_CI_MODEL_MIN_CONFIDENCE",
    ],
)
def test_every_ratio_refuses_a_value_outside_zero_to_one(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, "-0.5")

    # Refused with the line to fix: the variable and the value it was given.
    with pytest.raises(SettingsError, match=re.escape(f"{name}=-0.5")):
        Settings.from_env()


def test_the_ends_of_the_range_are_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "1")
    monkeypatch.setenv("HONEST_CI_CHANGED_LINE_COVERAGE_THRESHOLD", "0")

    settings = Settings.from_env()

    assert settings.coverage_threshold == 1.0
    assert settings.changed_line_coverage_threshold == 0.0


def test_a_ratio_that_is_not_a_number_falls_back_instead_of_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "75" has an intent to honour and is refused; "high" has none, so it takes
    # the default rather than stopping a deployment over a stray value.
    monkeypatch.setenv("HONEST_CI_COVERAGE_THRESHOLD", "high")
    monkeypatch.setenv("HONEST_CI_MODEL_MIN_CONFIDENCE", "  ")

    settings = Settings.from_env()

    assert settings.coverage_threshold == 0.70
    assert settings.model_min_confidence == 0.70


# ── Maven 预取坐标 ────────────────────────────────────────────────────────────


def test_extra_maven_artifacts_are_parsed_into_a_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HONEST_CI_MAVEN_EXTRA_ARTIFACTS",
        " org.apache.maven.surefire:surefire-junit-platform:3.5.2 ,"
        "org.junit.platform:junit-platform-commons:1.9.3",
    )

    assert Settings.from_env().maven_extra_artifacts == (
        "org.apache.maven.surefire:surefire-junit-platform:3.5.2",
        "org.junit.platform:junit-platform-commons:1.9.3",
    )


@pytest.mark.parametrize("value", ["junit-platform-commons", "org.junit:commons", "a:b:c:d:e:f"])
def test_a_coordinate_that_is_not_one_is_refused_with_its_own_text(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Refused, not filtered: this list is written *because* a run failed on a
    # missing artifact, and dropping the mistyped entry would hand the operator
    # the same failure again with nothing new to read. Failing at load costs
    # them seconds; failing silently costs them another provisioning run.
    monkeypatch.setenv("HONEST_CI_MAVEN_EXTRA_ARTIFACTS", value)

    with pytest.raises(SettingsError, match=re.escape(value)):
        Settings.from_env()


def test_the_machines_maven_repository_is_the_usual_one_unless_named_or_switched_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HONEST_CI_MAVEN_HOST_REPOSITORY", raising=False)
    assert Settings.from_env().maven_host_repository == tmp_path / ".m2" / "repository"

    # A localRepository of one's own, in settings.xml this run never reads.
    monkeypatch.setenv("HONEST_CI_MAVEN_HOST_REPOSITORY", "~/maven-cache")
    assert Settings.from_env().maven_host_repository == tmp_path / "maven-cache"

    monkeypatch.setenv("HONEST_CI_MAVEN_HOST_REPOSITORY", "")
    assert Settings.from_env().maven_host_repository is None
    # Built in code, a run reads none: its commands do not depend on the machine.
    assert Settings().maven_host_repository is None


@pytest.mark.parametrize("value", ["m2/repository", "/srv/m2,/srv/other"])
def test_a_machine_repository_maven_would_read_differently_is_refused(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Relative, Maven resolves it against the workspace; with a comma, Maven
    # splits it in two. Either way the run would read a directory nobody named.
    monkeypatch.setenv("HONEST_CI_MAVEN_HOST_REPOSITORY", value)

    with pytest.raises(SettingsError, match=re.escape(value)):
        Settings.from_env()


def test_provisioning_has_its_own_timeout_defaulting_higher_than_a_scanner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HONEST_CI_PROVISION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("HONEST_CI_COMMAND_TIMEOUT_SECONDS", raising=False)

    settings = Settings.from_env()

    assert settings.provision_timeout_seconds == 1800
    assert settings.provision_timeout_seconds > settings.command_timeout_seconds

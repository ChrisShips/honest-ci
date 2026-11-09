"""What every test starts from: a process no CI system started.

The suite runs in CI too, where the runner sets ``CI`` and ``GITHUB_ACTIONS``
in every job. A command a test runs would then be in CI, and the license check
would ask whoever the test's repository belongs to for a key: GitHub on
loopback, or github.com itself for a checkout whose origin is there. The tests
about CI set up the job they need.
"""

from __future__ import annotations

import pytest

from honest_ci.license_check import CI_VARIABLES, KEY_VARIABLE

#: What tells the license check it is in CI, and what it reads there.
LICENSE_CHECK_VARIABLES = (
    "CI",
    "GITHUB_ACTIONS",
    *CI_VARIABLES,
    KEY_VARIABLE,
    "GITHUB_API_URL",
    "GITHUB_WORKFLOW_REF",
    "GITHUB_WORKFLOW_SHA",
    "GITHUB_ACTOR",
)


@pytest.fixture(autouse=True)
def _outside_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LICENSE_CHECK_VARIABLES:
        monkeypatch.delenv(name, raising=False)

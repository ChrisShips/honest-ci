"""What the repository says about itself: the README, the docs and the package metadata.

The README is where a stranger installs honest-ci from, and it restates things
the package decides: the workflow ``honest-ci init --github`` writes, the
release the workflow pins. Those restatements are read off the package here,
so the page cannot promise a workflow nobody ships. The license wording and the
links are held here too, because a moved document or a license called by the
wrong name reads fine to everyone who already knows better. So is the
language: a file left in Chinese reads fine to whoever wrote it.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from honest_ci import __version__
from honest_ci.actions import workflow_template
from honest_ci.license_check import TRIAL_DAYS
from honest_ci.licensing import GRACE_DAYS

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
PYPROJECT = REPO / "pyproject.toml"
LICENSE = REPO / "LICENSE"

#: Every page a reader of the public repository lands on. AGENTS.md is left
#: out of the wording check only: it states the rule, so it names the words.
PAGES = (
    README,
    REPO / "AGENTS.md",
    REPO / "action.yml",
    PYPROJECT,
    *sorted((REPO / "docs").rglob("*.md")),
    *sorted((REPO / "examples").rglob("*.md")),
    REPO / "bench" / "README.md",
)

#: SHA-256 of FCL-1.0-ALv2 from its "Terms and Conditions" heading to the end,
#: in the text fcl.dev publishes, at the commit LICENSE was copied from:
#: https://github.com/keygen-sh/fcl.dev/blob/3f937b810cb71c14c4b095fee5128d0e6b9dad6a/FCL-1.0-ALv2.md
FCL_TERMS_SHA256 = "e39e5f79b482023c924dad1c7aa71ef69555b11cdf78f6ca7464e5f32e1ec854"

#: Left out of the language check. The tests still hold Chinese: messages not
#: yet translated, and inputs that are Chinese on purpose, such as a file name
#: git has to quote, which ``diffs.py`` uses as its example too. The Java
#: example is in Chinese throughout, from its README to its sources.
STILL_CHINESE = ("src/honest_ci/diffs.py", "tests/", "examples/java-junit5/")

_FENCE = re.compile(r"^```(?P<language>\w*)\n(?P<body>.*?)^```$", re.MULTILINE | re.DOTALL)
_LINK = re.compile(r"\]\((?P<target>[^)\s]+)\)")
_HEADING = re.compile(r"^#{1,6} +(?P<title>.+?) *$", re.MULTILINE)
_CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")


def _blocks(text: str, language: str) -> list[str]:
    return [m["body"] for m in _FENCE.finditer(text) if m["language"] == language]


def _prose(text: str) -> str:
    """``text`` without its fenced blocks, where a ``](...)`` is code, not a link."""

    return _FENCE.sub("", text)


def _anchor(title: str) -> str:
    """The id GitHub gives a heading: lower case, punctuation dropped, spaces as hyphens."""

    return re.sub(r"[^\w\- ]", "", title.replace("`", "").lower()).replace(" ", "-")


def test_the_readme_quotes_the_workflow_that_init_installs() -> None:
    """The README's workflow is the packaged template, byte for byte.

    People copy a workflow out of a README as often as they run the command
    that writes it. A copy that drifted would install a workflow nobody tests:
    one that checks out the merge commit, or keeps a permission the template
    dropped.
    """

    assert workflow_template() in _blocks(README.read_text(encoding="utf-8"), "yaml")


def test_the_readme_pins_the_release_this_package_is() -> None:
    """Every reference to the Action or the package names this version's tag.

    The workflow template pins ``v<version>``, so a README still naming the
    previous tag would have a reader install one release and run another.
    """

    text = README.read_text(encoding="utf-8")
    refs = re.findall(r"ChrisShips/honest-ci@(\S+?)(?=[\s`)]|$)", text, flags=re.MULTILINE)
    # A clone of the tag is a reference too: the demo is copied out of it, so a
    # clone of the previous tag shows a reader the previous release's checks.
    for line in text.splitlines():
        if "git clone" in line and "ChrisShips/honest-ci" in line:
            refs += re.findall(r"(?:--branch|-b)[ =](\S+)", line)
    assert refs
    assert set(refs) == {f"v{__version__}"}


def test_the_license_is_called_source_available_never_open_source() -> None:
    """FCL-1.0-ALv2 is not an OSI-approved license, and nothing public says it is."""

    assert "source-available" in README.read_text(encoding="utf-8")
    for page in PAGES:
        if page.name == "AGENTS.md":
            continue
        text = page.read_text(encoding="utf-8")
        assert not re.search(r"open[\s-]+source", text, flags=re.IGNORECASE), page


def test_the_package_metadata_claims_no_osi_license() -> None:
    """What PyPI shows: an English description, the README, and no license classifier.

    The trove list has no entry for FCL, and every ``License ::`` entry would
    say something it is not: OSI-approved, or proprietary. SPDX has no
    identifier for it either, so the license expression names it with a
    ``LicenseRef-``, and the package carries LICENSE itself.
    """

    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    assert project["license"] == "LicenseRef-FCL-1.0-ALv2"
    assert project["license-files"] == ["LICENSE"]
    assert LICENSE.read_text(encoding="utf-8").startswith(
        "# Fair Core License, Version 1.0, ALv2 Future License\n"
    )
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    assert project["readme"] == "README.md"
    assert not _CJK.search(project["description"])
    assert set(project["urls"].values()) >= {"https://github.com/ChrisShips/honest-ci"}
    assert all(
        url.startswith("https://github.com/ChrisShips/honest-ci")
        for url in project["urls"].values()
    )


def test_the_license_terms_are_fcl_word_for_word() -> None:
    """Only the notice above the terms is ours to fill in.

    A license is known by its text: edited terms are some other license that
    still calls itself the FCL, and they read fine to anyone who doesn't diff
    them against the original. The notice is the one line left blank, for the
    year and the licensor, who goes by the legal name the terms of service use.
    """

    text = LICENSE.read_text(encoding="utf-8")
    terms = text[text.index("\n## Terms and Conditions\n") + 1 :]
    notice = re.search(r"^## Notice\n\n(?P<line>[^\n]*)\n\n## ", text, flags=re.MULTILINE)

    assert hashlib.sha256(terms.encode("utf-8")).hexdigest() == FCL_TERMS_SHA256
    assert notice and notice["line"] == "Copyright 2026 Wenchao Gou"


def test_the_readme_says_what_the_license_grants() -> None:
    """The README's account of the license names the terms LICENSE sets.

    LICENSE is the rule, and the README is what people read instead. A
    restriction or a change date changed in one and not the other would promise
    something the license doesn't give, or hold back something it does. Line
    breaks don't count: the FCL wraps "second anniversary" across two lines.
    """

    granted = " ".join(LICENSE.read_text(encoding="utf-8").split())
    readme = " ".join(README.read_text(encoding="utf-8").split())

    for term, said in (
        ("Competing Use", "competing product or service"),
        ("circumvent the license key functionality", "getting around"),
        ("second anniversary", "two years after its release"),
        ("Apache License, Version 2.0", "becomes Apache-2.0"),
    ):
        assert term in granted, term
        assert said in readme, said


def test_the_readme_counts_the_days_the_license_check_counts() -> None:
    """The trial and the days a key works past its expiry date are the code's.

    A team reads them to plan when to buy or renew, and the check counts its
    own. A count changed in one and not the other stops pull requests on a day
    the README didn't name.
    """

    readme = " ".join(README.read_text(encoding="utf-8").split())
    trials = re.findall(r"\b(\d+)-day trial\b", readme)
    graces = re.findall(r"\b(\d+) days (?:past|after) (?:its|a key's) expiry date\b", readme)

    assert trials and set(trials) == {str(TRIAL_DAYS)}
    assert graces and set(graces) == {str(GRACE_DAYS)}


def test_every_relative_link_leads_somewhere() -> None:
    """A link to a file names one that exists, and a link into the README a heading it has.

    Documents move and get removed, and a link left behind still renders as a
    link.
    """

    anchors = {_anchor(m["title"]) for m in _HEADING.finditer(_prose(README.read_text("utf-8")))}
    for page in PAGES:
        if page.suffix != ".md":
            continue
        for match in _LINK.finditer(_prose(page.read_text(encoding="utf-8"))):
            target = match["target"]
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = target.partition("#")
            resolved = (page.parent / path).resolve() if path else page
            assert resolved.exists(), f"{page.relative_to(REPO)}: {target}"
            if anchor and resolved == README:
                assert anchor in anchors, f"{page.relative_to(REPO)}: {target}"


@pytest.mark.skipif(shutil.which("git") is None, reason="the tracked files come from git")
def test_every_tracked_file_outside_the_code_is_in_english() -> None:
    """No Chinese in what a visitor opens: the docs, the examples, the configuration.

    These are what a stranger reads before running anything, and the example
    rules get copied into other repositories, where their titles and fixes end
    up in the comments on pull requests. Every tracked file counts, not a list
    of pages, so a file added later is held to it without anyone remembering to.
    """

    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=False)
    assert listed.returncode == 0, listed.stderr.decode(errors="replace")

    found: list[str] = []
    for name in listed.stdout.decode("utf-8").split("\0"):
        path = REPO / name
        if not name or name.startswith(STILL_CHINESE) or not path.is_file():
            continue
        if _CJK.search(name):
            found.append(f"{name}: the file name")
        data = path.read_bytes()
        if b"\0" in data:
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            found.append(f"{name}: not UTF-8")
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if _CJK.search(line):
                found.append(f"{name}:{number}: {line.strip()}")

    assert not found, "\n".join(found)

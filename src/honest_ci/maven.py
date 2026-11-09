"""Pinned Maven coordinates and credential-free execution settings."""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

# Fully qualified goals avoid plugin-prefix and latest-version resolution in
# offline containers. Provisioning and scanning share this exact inventory.
MAVEN_PLUGINS = {
    "dependency": "org.apache.maven.plugins:maven-dependency-plugin:3.8.1",
    # Pinned for the same reason as the rest, and needed for one more: the
    # version Maven would pick by itself comes from the image's super-POM, so
    # "which clean ran" would change with the base image.
    "clean": "org.apache.maven.plugins:maven-clean-plugin:3.4.1",
    "checkstyle": "org.apache.maven.plugins:maven-checkstyle-plugin:3.6.0",
    "pmd": "org.apache.maven.plugins:maven-pmd-plugin:3.28.0",
    "spotbugs": "com.github.spotbugs:spotbugs-maven-plugin:4.9.8.0",
    "jacoco": "org.jacoco:jacoco-maven-plugin:0.8.13",
}

#: The newest Java whose class files each pinned plugin above can read, set by
#: the ASM it bundles. Bumping a pin means looking this up again: past it a
#: tool does not fail on the change, it fails on the JDK.
MAVEN_CLASS_FILE_CEILINGS = {
    # SpotBugs 4.9.8 bundles ASM 9.9, which reads up to major version 70. It
    # reads the class library of the JDK that runs Maven as well as the
    # project's classes, so it stops on a newer JDK whatever the project
    # compiles for: on JDK 27 it exits 1 with no report.
    "spotbugs": 26,
    # PMD 7.17.0 bundles ASM 9.8, which reads up to 69, and reads the JDK's
    # class library for type resolution. On a newer JDK it logs a parse error
    # per JDK class and still exits 0 with a report, with every type it
    # failed to read left unresolved.
    "pmd": 25,
    # JaCoCo 0.8.13 runs ASM 9.8 too, and reads only the project's classes:
    # on JDK 27, compiling for release 21, it measured every line. Its bound
    # is on the release the project compiles for, and a JDK compiles for no
    # newer release than itself.
    "jacoco": 25,
}
#: The newest JDK a process-mode run can use, the lowest bound above. Docker
#: mode runs the image's own JDK, which the pins were chosen against.
MAVEN_JDK_CEILING = min(MAVEN_CLASS_FILE_CEILINGS.values())

#: Artifacts a pinned plugin resolves while it *runs*, which neither
#: ``go-offline`` nor ``resolve-plugins`` fetches: both walk declared
#: dependencies, and these are chosen mid-goal. Each entry is pinned to the
#: plugin above that needs it, so bumping that plugin is a prompt to re-check
#: this line.
MAVEN_RUNTIME_ARTIFACTS = {
    # maven-pmd-plugin's `pmd` goal is a Maven *report*: it writes
    # target/pmd.xml and then renders HTML, and rendering resolves the default
    # site skin. Offline and without it, a PMD run that found nothing exits
    # non-zero with a complete, empty report beside it -- which reads as a
    # violation of a rule nobody broke.
    "pmd-site-skin": "org.apache.maven.skins:maven-fluido-skin:2.0.0-M9",
}

MAVEN_STAMP = ".honest-ci/m2/.honest-ci-dependencies.sha256"
SUREFIRE_REPORTS = "**/target/surefire-reports/TEST-*.xml"
JACOCO_REPORTS = "**/target/site/jacoco/jacoco.xml"

#: ``groupId:artifactId:version`` with optional packaging and classifier, the
#: shape ``dependency:get -Dartifact=`` accepts.
_COORDINATE = r"[\w.\-]+(?::[\w.\-]+){2,4}"
#: The same shape, anchored, for validating configuration: a typo in
#: ``HONEST_CI_MAVEN_EXTRA_ARTIFACTS`` should fail at load, not three minutes into
#: a provisioning run.
ARTIFACT_COORDINATE = re.compile(rf"^{_COORDINATE}$")

# What Maven prints when ``-o`` meets an artifact nobody pre-fetched. It is
# the signature of a *provisioning* gap, and the difference matters: a tool
# that stopped here never examined the code, so it is unavailable rather than
# failed. Merging the two would report a missing download as a rule violation.
_OFFLINE_MARKERS = (
    "in offline mode and the artifact",
    "has not been downloaded from it before",
)
# Two spellings of the same wall, and both need an anchor: the whole failure
# arrives on one line that opens with "Failed to execute goal
# org.apache.maven.plugins:maven-surefire-plugin:3.5.2:test", which has the
# shape of a coordinate and is not one. So coordinates are picked out of these
# segments rather than off the line.
_MISSING_SEGMENTS = (
    re.compile(r"could not be resolved:(.*)"),
    re.compile(rf"(the artifact {_COORDINATE} has not been downloaded)"),
)
_COORDINATES = re.compile(_COORDINATE)
# What ASM says about a class file newer than it reads, in every tool that
# bundles it. A tool that printed it read less than the build holds.
_UNREADABLE_CLASS_FILE = re.compile(r"Unsupported class file major version (\d+)")
# The line of ``mvn -v`` that names the JDK Maven runs on. Java 8 and earlier
# print their feature release after a "1.", as in 1.8.0_392.
_MAVEN_JAVA = re.compile(
    r"^Java version: (?P<version>(?:1\.)?(?P<release>\d+)[^,\s]*)(?:.*runtime: (?P<runtime>.+))?$",
    re.MULTILINE,
)


class MavenJava(NamedTuple):
    """The JDK ``mvn -v`` says Maven runs on."""

    release: int
    version: str
    runtime: str | None


#: The global settings a process-mode Maven reads in place of its
#: installation's ``conf/settings.xml``: an empty file. Maven 3.10 puts
#: ``${user.home}`` into that file before it parses it, comments and all, and
#: ``user.home`` here is the workspace's ``.honest-ci``; a workspace path with
#: ``--`` in it, which XML forbids inside a comment, then stops every goal
#: before it starts. What an installation's file holds -- mirrors, proxies,
#: credentials -- is for downloads, and this run makes none: it is offline, and
#: the operator's own ``~/.m2/settings.xml`` is already out of its reach
#: (``user.home`` again). Docker mode reads the image's copy, under
#: ``/workspace``, and could not see this one.
MAVEN_GLOBAL_SETTINGS = Path(__file__).with_name("resources") / "maven-settings.xml"


def maven_root(workspace: Path, mode: str) -> str:
    return "/workspace" if mode == "docker" else str(workspace.resolve())


def maven_env(workspace: Path, mode: str) -> dict[str, str]:
    root = maven_root(workspace, mode)
    return {
        "MAVEN_CONFIG": f"{root}/.honest-ci/m2-config",
        "MAVEN_OPTS": f"-Duser.home={root}/.honest-ci",
    }


def maven_command(
    workspace: Path, mode: str, *, offline: bool = True, tail: Path | None = None
) -> list[str]:
    return [
        "mvn",
        "-B",
        *(["-o"] if offline else []),
        "-q",
        f"-Dmaven.repo.local={maven_root(workspace, mode)}/.honest-ci/m2",
        # Searched after the repository above and never written to: Maven
        # 3.9's chained local repository puts every write in the first. An
        # older Maven ignores the property and stops where the first runs out.
        *([f"-Dmaven.repo.local.tail={tail}"] if tail is not None else []),
        *(["-gs", str(MAVEN_GLOBAL_SETTINGS)] if mode != "docker" else []),
    ]


def offline_coordinates(extra: Sequence[str] = ()) -> tuple[str, ...]:
    """What an offline run resolves by name, in ``dependency:get`` order.

    Every pinned plugin but the dependency plugin, which only provisioning
    runs, plus what goals pick while they run and what the deployment added.
    The project's own dependencies are not here: its build decides those.
    """

    plugins = [item for name, item in MAVEN_PLUGINS.items() if name != "dependency"]
    return (*plugins, *MAVEN_RUNTIME_ARTIFACTS.values(), *extra)


def get_coordinate(coordinate: str) -> str:
    """A coordinate as Maven prints it, in the order ``dependency:get`` reads.

    Maven names an artifact ``groupId:artifactId:extension[:classifier]:version``
    and ``-Dartifact=`` reads ``groupId:artifactId:version[:packaging[:classifier]]``,
    so ``surefire-junit-platform:jar:3.5.2`` handed over as printed asks for
    version ``jar``. Three parts are the same either way.
    """

    parts = coordinate.split(":")
    if len(parts) == 4:
        group, artifact, extension, version = parts
        return f"{group}:{artifact}:{version}" + ("" if extension == "jar" else f":{extension}")
    if len(parts) == 5:
        group, artifact, extension, classifier, version = parts
        return f"{group}:{artifact}:{version}:{extension}:{classifier}"
    return coordinate


def in_repository(repository: Path, coordinate: str) -> bool:
    """Whether a local repository holds ``coordinate`` (``dependency:get`` order).

    The file itself, not its directory or its POM: resolving a POM alone, for
    a version Maven was only comparing, leaves both behind without the jar.
    """

    parts = coordinate.split(":")
    if len(parts) < 3:
        return False
    group, artifact, version = parts[:3]
    packaging = parts[3] if len(parts) > 3 else "jar"
    classifier = f"-{parts[4]}" if len(parts) > 4 else ""
    extension = "jar" if packaging in {"jar", "maven-plugin", "bundle"} else packaging
    name = f"{artifact}-{version}{classifier}.{extension}"
    return (repository / group.replace(".", "/") / artifact / version / name).is_file()


def fetch_command(coordinates: Sequence[str], repository: Path) -> str:
    """One shell line that downloads ``coordinates`` into ``repository``.

    ``coordinates`` are in ``dependency:get`` order already. For an operator
    to run, with a network and their own Maven settings: the toolchain itself
    never writes to a repository it does not own.

    It stops at the first download that fails and exits non-zero, so a script
    that runs it can tell. A ``break`` would stop the loop but leave the line
    exiting 0; a bare ``exit`` would close the terminal it was pasted into, so
    the loop runs in a subshell.
    """

    listed = " ".join(shlex.quote(item) for item in coordinates)
    goal = f"{MAVEN_PLUGINS['dependency']}:get"
    target = shlex.quote(f"-Dmaven.repo.local={repository}")
    return f'(for a in {listed}; do mvn -B -q {target} {goal} -Dartifact="$a" || exit 1; done)'


def unprovisioned_artifacts(*outputs: str) -> tuple[str, ...] | None:
    """Why an offline Maven run stopped, when the reason is a missing download.

    ``None`` means it failed for some other reason -- the common case, and the
    caller should report the failure as it stands. An empty tuple means it
    clearly hit this wall but named nothing we could parse. Every coordinate
    Maven listed comes back together, because resolution stops at the first
    failure and an operator handed them one at a time pays a full provisioning
    round per coordinate.
    """

    text = "\n".join(outputs)
    if not any(marker in text for marker in _OFFLINE_MARKERS):
        return None
    found: list[str] = []
    for pattern in _MISSING_SEGMENTS:
        for segment in pattern.finditer(text):
            for coordinate in _COORDINATES.findall(segment.group(1)):
                if coordinate not in found:
                    found.append(coordinate)
    return tuple(found)


def unprovisioned_hint(
    artifacts: Sequence[str], skipped: str | None = None, host: Path | None = None
) -> str:
    """The sentence an operator can act on, with the coordinates in it.

    Named rather than described because the fix is a coordinate long and the
    person reading it is looking at a build log, not at our documentation:
    Surefire picks its test provider and its platform baseline while it runs,
    a report plugin picks its skin while it renders, and no amount of
    ``go-offline`` finds either. Named in ``dependency:get`` order, which is
    what both remedies below feed them to.

    ``skipped`` is why no provisioning ran before this run, when none did.
    Then the coordinate is not the fix: ``HONEST_CI_MAVEN_EXTRA_ARTIFACTS`` is
    read only by provisioning, and an operator told to add to it would add,
    run again, and meet the same block. ``host`` is the machine's own
    repository a process-mode run also read, when it read one.
    """

    fetchable = [get_coordinate(item) for item in artifacts]
    named = (
        ", ".join(f"`{item}`" for item in fetchable)
        or "the coordinate the log says could not be resolved"
    )
    if skipped is None:
        them = "them" if len(fetchable) > 1 else "it"
        return (
            f"The offline repository does not have {named}: Maven decides some artifacts only "
            "while a goal runs (Surefire's test provider and the JUnit Platform baseline it "
            "brings, a report plugin's skin), and go-offline cannot prefetch those. Add "
            f"{them} to HONEST_CI_MAVEN_EXTRA_ARTIFACTS and run again, and provisioning will "
            f"download {them} too."
        )
    stopped = (
        f"The offline repository does not have {named}, and nothing was provisioned before "
        f"this run: {skipped}."
    )
    docker = (
        "Provisioning, which fetches dependencies, runs in docker mode with "
        "HONEST_CI_SANDBOX_BUILD_DEPS=1"
    )
    if host is None:
        return f"{stopped} {docker}; run there, or fill the workspace's .honest-ci/m2 before the run."
    held = [item for item in fetchable if in_repository(host, item)]
    if held:
        return (
            f"{stopped} This machine's Maven repository {host} has "
            f"{', '.join(f'`{item}`' for item in held)}, so this Maven did not read it: a "
            "second, read-only local repository (maven.repo.local.tail) needs Maven 3.9 or "
            f"later. Upgrade Maven, or use docker mode: {docker}."
        )
    fetch = (
        f"fetch {'them' if len(fetchable) > 1 else 'it'} there with a network: "
        f"{fetch_command(fetchable, host)}"
        if fetchable
        else "fetch the coordinate the log names there with a network"
    )
    return (
        f"{stopped} Maven 3.9 or later also reads this machine's Maven repository {host}, "
        f"read-only, and it does not have {'them' if len(fetchable) > 1 else 'it'} either: "
        f"{fetch}. honest-ci doctor lists everything else the toolchain pins that is missing "
        f"there. Or use docker mode: {docker}."
    )


def maven_java(output: str) -> MavenJava | None:
    """The JDK ``mvn -v`` printed in ``output``, or ``None`` if it named none."""

    found = _MAVEN_JAVA.search(output)
    if found is None:
        return None
    return MavenJava(int(found["release"]), found["version"], found["runtime"])


def java_release(major_version: int) -> int:
    """The Java release whose class files carry ``major_version`` (52 is Java 8)."""

    return major_version - 44


def unreadable_class_files(*outputs: str) -> int | None:
    """The newest class file version a tool said it could not read, or ``None``."""

    versions = [int(item) for text in outputs for item in _UNREADABLE_CLASS_FILE.findall(text)]
    return max(versions) if versions else None


def pinned_ceilings() -> str:
    """Each pinned tool with the newest Java it reads, as an operator compares them."""

    return ", ".join(
        f"{name} {MAVEN_PLUGINS[name].rsplit(':', 1)[-1]} up to Java {ceiling}"
        for name, ceiling in MAVEN_CLASS_FILE_CEILINGS.items()
    )


def unreadable_class_files_hint(tool: str, major_version: int, mode: str) -> str:
    """Why ``tool`` read less than the build holds, and the JDK that fixes it.

    Not "check the plugin and the report path": the plugin ran, and what
    stopped it is the JDK under it, which no amount of provisioning changes.
    One line, because the review gate's summary shows only the first line of
    why a tool did not run, and that summary is what the terminal and the
    pull request comment show.
    """

    newest = f"Java {java_release(major_version)}"
    ceiling = MAVEN_CLASS_FILE_CEILINGS.get(tool)
    pinned = (
        f"the pinned {tool} plugin {MAVEN_PLUGINS[tool].rsplit(':', 1)[-1]} reads up to "
        f"Java {ceiling}"
        if ceiling is not None
        else f"the pinned {tool} cannot read them"
    )
    remedy = (
        f"Use a Java image with JDK {MAVEN_JDK_CEILING} or earlier, which every pinned tool "
        "reads (HONEST_CI_JAVA_DOCKER_IMAGE)."
        if mode == "docker"
        else f"Point JAVA_HOME at JDK {MAVEN_JDK_CEILING} or earlier, which every pinned tool "
        "reads (without it, Homebrew's mvn runs on Homebrew's newest openjdk); honest-ci "
        "doctor shows the JDK a run gets."
    )
    return (
        f"{tool} could not read class files of major version {major_version} ({newest}), "
        f"so it could not check this change: {pinned}. They come from the JDK running Maven, "
        f"whose own classes it reads too, or from a project compiled for {newest}. {remedy}"
    )

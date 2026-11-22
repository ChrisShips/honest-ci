# Java example: a Maven repository that really runs

This is the acceptance sample for honest-ci's Java toolchain, not a tutorial.
The tests that use canned Maven output can stay green while `mvn -o` in the
container doesn't actually work, and the only way to know is to run it on a
machine with Docker. `make acceptance` in the honest-ci repository runs this
example. The comments in its files are still in Chinese.

It's a single-module Maven project with JUnit 5 and JaCoCo: about 20 lines
that split a refund into line items. The plugin versions match `MAVEN_PLUGINS`
in `src/honest_ci/maven.py`.

```
pom.xml                  dependencies and plugins, every version pinned
checkstyle.xml           a small Checkstyle config that stays quiet on clean code
pmd-ruleset.xml          a PMD ruleset, kept small the same way
honest-ci-rules.toml     one regex rule, one path rule, one scanner rule; no judgment rules
honest-ci-policy.toml    requires the review and regression gates
scenarios/*.patch        one change for each of the three verdicts
src/main/java/...        RefundSplitter: splits an amount in integer cents
src/test/java/...        RefundSplitterTest: 5 test cases
```

## The three verdicts

| Change | Verdict | Exit code | Decided by |
|---|---|---|---|
| `scenarios/clean.patch` adds a `totalOf` method and its test | `mergeable` | 0 | The review gate finds nothing. The regression gate runs 7 test cases, all pass, and line coverage clears the bar. |
| `scenarios/violation.patch` adds a method that does money math in `double` | `blocked` | 1 | Rule `R-01` (high, a deterministic regex) matches an added line. |
| `scenarios/drop-test.patch` deletes the whole test class | `needs_human` | 3 | The regression gate ran the tests and collected 0 test cases. No regression evidence isn't a pass. |

The third row is the one to look at. After the tests are deleted, nothing
fails: `mvn test` exits 0, and the review gate has no added lines to judge.
Reading that as a pass only takes not asking one more question.

## Run it

You need Docker, because the regression gate runs Maven in a container.

```bash
cp -r examples/java-junit5 /tmp/honest-ci-java && cd /tmp/honest-ci-java && git init -q && git add -A && git commit -qm baseline
```

```bash
export HONEST_CI_MAVEN_EXTRA_ARTIFACTS=org.apache.maven.surefire:surefire-junit-platform:3.5.2,org.junit.platform:junit-platform-commons:1.9.3
```

```bash
honest-ci doctor --workspace /tmp/honest-ci-java
```

`doctor` reports that the local Maven repository is empty. The first `verify`
fills it, and that's the only step on the whole path that uses the network.

```bash
git apply scenarios/clean.patch && honest-ci verify --workspace /tmp/honest-ci-java; echo "exit code $?"
```

Before you apply another patch, run `git checkout -- . && git clean -fd`.

## Three things that only showed up on a real machine

**Filling a cold cache is far slower than a scan.** On this project, with one
JUnit dependency, `go-offline` pulls more than 130 MB before it gets to
resolving plugins. The scanner timeout (`HONEST_CI_COMMAND_TIMEOUT_SECONDS`,
300 seconds by default) assumes that a ruff run taking 5 minutes means
something is wrong, so filling the cache has its own timeout,
`HONEST_CI_PROVISION_TIMEOUT_SECONDS` (1800 seconds by default). A timeout
exits with code 124, and the error says so.

**`go-offline` can't prefetch some artifacts.** Maven decides some artifacts
only when a goal runs. Surefire picks its provider by test framework, and the
report plugins pick a skin when they render. Neither is in any pom's dependency
graph, so `go-offline` and `resolve-plugins` don't see them. An offline run
would stop at a download nobody declared. Each one is handled in its own place:

- The PMD skin is ours to handle. honest-ci pins the PMD version, so it also
  pins the skin PMD renders with. It's in `MAVEN_RUNTIME_ARTIFACTS` in
  `maven.py`, and needs no setup.
- The JUnit Platform launcher is the project's to handle. Surefire adds one
  when it can't find it. This project's `pom.xml` declares it, as JUnit has
  recommended since 5.11, so its version follows the project and it is pulled
  with the other test dependencies.
- The provider and its baseline belong to the machine. Their versions follow
  the Surefire version the project picked, which the toolchain can't guess, so
  they go in `HONEST_CI_MAVEN_EXTRA_ARTIFACTS`. That's two coordinates. With
  launcher 1.11.4 declared, Surefire still resolves its own baseline,
  `org.junit.platform:junit-platform-commons:1.9.3`, whatever the pom says. If
  you leave one out, the code isn't blamed: the failed offline resolution is
  reported as a tool that wasn't available, which still blocks, and the error
  names the missing coordinate.

**The last build can pass itself off as this one's evidence.** Apply
`drop-test.patch` to a workspace that has already run once. The test source is
gone, but its `.class` file is still in `target/test-classes`, because Maven
doesn't delete compiled output whose source has disappeared. So Surefire runs
it anyway and reports test cases for a file that isn't in the tree. That's why
every run starts with `clean` before the first Maven goal (`maven-clean-plugin`
is pinned too, and fetched with the rest). The cost is a full recompile on
every run. In return, every number in the evidence pack comes from the tree
the fingerprint covers. `tests/integration/test_java_acceptance.py` checks this
directly: before it runs `drop-test`, it asserts that the workspace still holds
the `.class` file from the last run, or the test would prove nothing.

## Why there are no judgment rules

A judgment rule needs model credentials to be judged. Without them it is
recorded as not judged and goes to a person. This example would then never
reach `mergeable`, for a reason that has nothing to do with the Java
toolchain, and acceptance needs a deterministic result. A real repository's
rules should include judgment rules: `honest-ci-rules.example.toml` shows how
to write them.

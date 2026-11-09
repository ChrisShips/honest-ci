# Python demo: four changes an agent might hand in

A small shop's pricing code, in integer cents, with six pytest tests. Each of the
four patches under `scenarios/` is a task done the way a coding agent might do it.
All four leave `pytest` green, but only one should be merged without someone
looking at it.

```
src/shop/pricing.py      line totals, discounts, shipping
tests/test_pricing.py    six tests
honest-ci-rules.toml     rules: regular expressions, paths and test integrity
honest-ci-policy.toml    required_checks = ["review"]
scenarios/*.patch        one change per scenario
github-demo.sh           builds a repository for trying the scenarios on GitHub
GITHUB.md                what each pull request there should show
```

## The four scenarios

| Scenario | What the agent did | Verdict | Exit code | Rule |
|---|---|---|---|---|
| `clean` | Adds `order_total`, which applies the discount before shipping, with two tests for it. | `mergeable` | 0 | — |
| `delete-failing-test` | Asked for an international shipping fee, it turns `>=` into `>` along the way. The test for the free-shipping threshold fails, and the agent deletes it. | `needs_human` | 3 | `I-01` |
| `rewrite-expectation` | Asked to simplify `discounted`, it drops the `+ 50` that gives the customer the half cent. The rounding test fails, and the agent changes the expected value to what the code now returns. | `needs_human` | 3 | `I-04` |
| `add-skip` | Asked for a bulk price, it rewrites `line_total` without the check that refuses an empty line. That test fails, and the agent marks it skipped as flaky. | `needs_human` | 3 | `I-02` |

After `delete-failing-test`, a basket of exactly 50.00 pays for shipping. After
`rewrite-expectation`, every half cent of discount goes to the shop. After
`add-skip`, a line of zero items costs nothing. The test suite passes in all three
cases, so a green CI run proves nothing about them.

The last three are held for a person, not blocked. The diff alone can't tell a
test deleted to hide a bug from one deleted along with the feature it tested, or
an expected value rewritten to match a bug from one that was wrong. So honest-ci
says which test changed and how, and a person decides. To block on one of
these checks instead, set `action = "block"` on its rule.

## Run it

You need `honest-ci` and `git`. You don't need Docker or a model.

```bash
cp -r examples/python-agent-demo /tmp/honest-ci-demo && cd /tmp/honest-ci-demo && git init -q && git add -A && git commit -qm baseline
```

```bash
export HONEST_CI_DISABLED_GATES=regression
```

```bash
git apply scenarios/delete-failing-test.patch && honest-ci verify; echo "exit code $?"
```

`honest-ci verify` judges the working tree against `HEAD`. It prints the verdict and
the test it is waiting on, and writes an evidence pack under
`.honest-ci/artifacts/`. `evidence.md` there is the readable version. Before you
apply the next patch, reset the tree:

```bash
git checkout -- . && git clean -fdq
```

If you have pytest, `python -m pytest -q` after any of the patches shows the same
green run your CI would.

## On GitHub

`github-demo.sh` builds a git repository from this demo, ready to push to GitHub.
Its `main` has honest-ci installed by `honest-ci init --github`, and each scenario
is a branch to open a pull request from. [GITHUB.md](GITHUB.md) says which pull
requests to open and what each one should show: the comment, the commit status
and the exit code.

The pull request that installs honest-ci goes to a person, and it should. The
Action reads the rules and the policy at the pull request's base commit, which
has neither yet, and the pull request changes `.github/`. The pull requests after
it are judged by the files it installed.

## What the demo leaves out

**The regression gate.** It runs the test suite in a Docker sandbox. The demo turns
it off with `HONEST_CI_DISABLED_GATES=regression`, the same line as in the GitHub
workflow template, so it needs no Docker and runs all four scenarios in seconds.
Turning it on changes no verdict. `clean` passes it, and the other three never
reach it because the review gate has already handed them to a person. Their
suites would pass too, which is the point of the demo.

**Scanner rules** (`kind = "tool"`). A scanner runs in the sandbox, and the GitHub
Action doesn't install one. A scanner that isn't there fails closed and blocks
every change.

**Judgment rules** (`kind = "judgment"`). They need a model. Without one they're
recorded as not reviewed and the change goes to a person, so `clean` could never
be `mergeable`. `honest-ci-rules.example.toml` at the repository root shows how to
write them.

`required_checks = ["review"]` in the policy follows from the first point. A gate
the policy requires that reached no verdict isn't a pass, so requiring the
regression gate, or the spec gate that doesn't exist yet, would hold every change
for a person.

The table above is also a test. `tests/test_python_agent_demo.py` in the honest-ci
repository applies each patch, runs `honest-ci verify`, and checks the verdict,
exit code and rule against this table.

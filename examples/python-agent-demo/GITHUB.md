# Trying the demo on GitHub

A checklist for running honest-ci's GitHub Action end to end. Build a
repository from this demo, push it to GitHub, open the pull requests below, and
check that each one shows what the table says.

`tests/test_github_demo.py` in the honest-ci repository builds the same
repository and runs every pull request in the table through `honest-ci verify`,
with the rules and the policy read at the pull request's base, as the Action
reads them. So the verdicts here are already checked. What the test cannot
check is GitHub's side: the comment, the status, the job, forks and permissions.
This list covers that side.

## 1. Build the repository

You need `bash`, `git` and honest-ci on `PATH`. `pip install -e .` in the
honest-ci repository installs it; alternatively, set `HONEST_CI` to its path.

```bash
examples/python-agent-demo/github-demo.sh --action-ref main /tmp/honest-ci-demo
```

The workflow that `honest-ci init --github` writes calls the Action at the
release tag of the honest-ci that wrote it, such as `v0.1.0`. Until that tag
exists on `ChrisShips/honest-ci`, every run fails at "Set up job" because GitHub
cannot resolve the action. `--action-ref main` points the workflow at the
`main` branch instead; a commit SHA works too. Once the release is tagged, leave
the flag out, so that the demo also checks the release the workflow pins.

| Branch | What it holds |
|---|---|
| `main` | The shop, with the three files `honest-ci init --github` writes. |
| `before-honest-ci` | The shop before them. |
| `install-honest-ci` | The same commit as `main`: the change that installs honest-ci. |
| `clean`, `delete-failing-test`, `rewrite-expectation`, `add-skip` | `main` plus one of the scenarios in [README.md](README.md). |
| `deselect-test` | `main` plus a test left out of the run. |

## 2. Push it

1. Create an empty repository on GitHub, with no README, license or
   `.gitignore`.
2. While `ChrisShips/honest-ci` is private, the demo repository has to be
   private and owned by the same account. Also, honest-ci's **Settings →
   Actions → General → Access** has to let that account's repositories use it.
   Otherwise every run fails at "Set up job" because GitHub cannot resolve the
   action.
3. Push every branch from the built repository, and check that `main` is the
   default branch:

   ```bash
   cd /tmp/honest-ci-demo && git remote add origin git@github.com:<owner>/honest-ci-demo.git && git push -u origin --all
   ```

4. Open the repository's **Actions** tab. If it says "Workflows aren't being
   run on this repository", click **Enable Actions on this repository**. A new
   repository can start out like this while the API reports Actions as
   enabled, and a pull request opened before then starts no run at all, not
   even a failed one. To start one, close the pull request and reopen it.

A push runs nothing: the workflow runs on pull requests and on reviews of
them, not on pushes.

## 3. Open the pull requests

Open one pull request per row of the table, from the branch into the base in
the "Into" column. Each pull request gets three things:

- one comment, headed as in the table;
- a commit status named `honest-ci/verify` on its head commit, whose description
  starts with the state in the heading, such as `awaiting_human`;
- a job whose log ends with the exit code in the table.

Any exit code but 0 fails the job, so every job is red except the one for
`clean`.

| Branch | Into | Comment heading | `honest-ci/verify` | Exit code | Rule |
|---|---|---|---|---|---|
| `install-honest-ci` | `before-honest-ci` | `honest-ci · awaiting_human` | `pending` | 3 | — |
| `clean` | `main` | `honest-ci · mergeable` | `success` | 0 | — |
| `delete-failing-test` | `main` | `honest-ci · awaiting_human` | `pending` | 3 | `I-01` |
| `rewrite-expectation` | `main` | `honest-ci · awaiting_human` | `pending` | 3 | `I-04` |
| `add-skip` | `main` | `honest-ci · awaiting_human` | `pending` | 3 | `I-02` |
| `deselect-test` | `main` | `honest-ci · blocked` | `failure` | 1 | `I-05` |

On each pull request, also check these:

- [ ] While the run is going, the status is `pending`, with a description that
      starts with "Verifying". When the run ends, the verdict replaces it.
- [ ] The status's **Details** link opens the workflow run, while the run is
      going as well as after it ends.
- [ ] The run's summary page shows the same conclusion as the comment.
- [ ] The run has an artifact named `honest-ci-evidence-<number>-1` that holds
      the evidence pack, and the comment's last line names that artifact and
      links to the run.
- [ ] The comment is in English, and so is `evidence.md` in the artifact.

**The pull request that installs honest-ci goes to a person, and it should.**
`install-honest-ci` into `before-honest-ci` is what the first pull request
looks like in any repository that installs honest-ci. The Action reads the
rules and the policy at the base commit, and the base has neither yet: the log
says each one "does not exist at the base commit". That has three
consequences:

- The review gate has no rules to review against.
- The built-in policy requires the spec and regression gates as well, and
  neither one runs.
- The pull request changes `.github/` and honest-ci's own two files, and the
  built-in policy sends such a change to a person.

A person reviews the pull request and merges it. Every pull request after it is
judged by the files it merged, as the other pull requests here are.

`deselect-test` is not one of the demo's scenarios. It is the one change here
that is blocked rather than held for a person, so with it the pull requests
show all three states the Action sets.

## 4. Push again and re-run

On `clean`, push one more commit:

```bash
git checkout clean && git commit --allow-empty -qm "Run again" && git push
```

- [ ] The comment is edited in place, not posted a second time, and it names
      the new head commit.
- [ ] The status is set on the new head commit.
- [ ] If the run for the earlier commit was still going, it is cancelled.

**Re-run all jobs** on a finished run updates the same comment and uploads
`honest-ci-evidence-<number>-2`.

## 5. Make the verdict a merge requirement

Add a branch protection rule, or a ruleset, for `main` that requires the
`honest-ci/verify` status check. GitHub offers only checks that reported in the
past week, which the pull requests above did. On a private repository this
needs a paid plan. On GitHub Free, do this step once the repository is public.

- [ ] `clean` can be merged.
- [ ] The three pull requests held for a person, and `deselect-test`, cannot be
      merged unless someone bypasses the rule.

## 6. Approve a pull request held for a person

A pull request that honest-ci holds for a person goes through when someone
approves it on GitHub. An Approve review counts when all of these hold:

- its author can push to the repository;
- it is given on the head commit, after the comment for that commit went up;
- its text gives the reason.

It counts for that commit and for the rules and the policy of that run only. A
new commit, or a change to the rules or the policy, needs a new approval.

GitHub does not let the author of a pull request approve it. So approve from a
second account that can push to the demo repository, such as a collaborator on
a personal repository. Each review starts a run of its own, whose event is
`pull_request_review`.

Do these in order:

| Step | Pull request | Comment heading | `honest-ci/verify` | Exit code |
|---|---|---|---|---|
| Approve, with the reason in the review's text | `delete-failing-test` | `honest-ci · approved` | `success` | 0 |
| Approve, with no text | `rewrite-expectation` | `honest-ci · awaiting_human` | `pending` | 3 |
| Push a commit to the approved pull request | `delete-failing-test` | `honest-ci · awaiting_human` | `pending` | 3 |
| Approve, with the reason in the review's text | `deselect-test` | `honest-ci · blocked` | `failure` | 1 |

The push for the third step, which changes the pull request's diff:

```bash
git checkout delete-failing-test && echo >> README.md && git commit -qam "Change after the approval" && git push
```

An empty commit would not do. After a push that leaves the diff as it was,
GitHub reports the approval as given on the new head commit. honest-ci still
does not count it, because a new commit is new evidence, but the comment then
says it was given before this conclusion was published, not on another commit.

- [ ] On `delete-failing-test`, the comment's **Approval** section names the
      reviewer, links the review and quotes the reason. The status description
      starts with `approved`, not `mergeable`.
- [ ] The evidence artifact of that run holds `approval.json`, and its
      `evidence.md` has a "Human decision" section.
- [ ] On `rewrite-expectation`, the comment lists the approval under
      "Approvals that do not count", because it gives no reason. Approving
      again, with a reason, turns it `success`. That also checks that GitHub
      names honest-ci as the last editor of its own comment.
- [ ] After the push, the comment on `delete-failing-test` lists the approval
      as given on another commit. If the branch's protection dismisses stale
      approvals, GitHub dismisses it and the comment does not list it at all.
      An Approve review on the new commit turns it `success` again.
- [ ] On `deselect-test`, the comment has no **Approval** section. A blocked
      change is fixed with a new commit, not with an approval.
- [ ] A review that only comments starts a run too, and the conclusion stays
      as it was.
- [ ] If you dismiss the approval on an approved pull request, the run it
      starts sets the status back to `pending`. **Dismiss review** is in the
      menu next to the reviewer in the summary of reviews on the Conversation
      tab, not in the menu on the review itself.
- [ ] With the rule from step 5, `delete-failing-test` can be merged while it
      is approved, and cannot after the push.
- [ ] With `delete-failing-test` approved again, edit honest-ci's comment on it
      from the second account by adding a blank line at its end, and **Re-run
      all jobs** on the run that approved it. The blank line changes nothing
      the comment shows. Saving the comment exactly as it was would not do:
      GitHub does not count that as an edit, and honest-ci stays its last
      editor. The status goes back to `pending`, and the comment says the
      approval was given before the conclusion: a comment someone else
      edited last cannot say from when approvals count. Approving again turns
      it `success`.
- [ ] If GitHub lets the first account edit the text of that approval, do so
      and re-run again. The comment lists the approval as edited by someone
      other than the reviewer.

**If the comment says "No approval was applied"**, note the reason it quotes.
To decide whether an approval counts, the Action's token reads each review's
commit and whether its author can push. If GitHub refuses the token either one,
no approval can count on any pull request, so this needs fixing before
release.

## 7. A pull request from a fork

This step works only once both `ChrisShips/honest-ci` and the demo repository
are public, because a public repository cannot call an action in a private one.

1. Fork the demo repository from a second account, or into an organization you
   own, since an account cannot fork its own repository. Clear **Copy the
   `main` branch only**.
2. From the fork's `delete-failing-test`, open a pull request into the demo
   repository's `main`.
3. If GitHub holds the run for approval, as it does for some first-time
   contributors, approve it from the pull request.

- [ ] No comment and no `honest-ci/verify` status appear, because a fork's run
      gets a read-only token.
- [ ] The log says the conclusion was not written to the pull request because
      it comes from the fork, and that the conclusion is in the job summary.
- [ ] The run's summary page has the conclusion, headed
      `honest-ci · awaiting_human`, with the same note.
- [ ] The job still ends with exit code 3, the verdict's code.
- [ ] The evidence artifact is uploaded, as for any other run.
- [ ] An Approve review from the demo repository's account starts a run that
      again writes only the job summary. Its note says an approval cannot take
      effect on this pull request, because every run for it gets a read-only
      token.

## 8. A workflow set up wrong (optional)

A pull request runs the workflow as the pull request changes it. So to try a
wrong setup, make one change on a branch off `main` and open a pull request into
`main`. Each of these should fail the job with exit code 4 and say what to fix,
and none of them should post a verdict:

| Change to `.github/workflows/honest-ci.yml` | What the log says |
|---|---|
| Delete `ref: ${{ github.event.pull_request.head.sha }}` | The checkout is a merge commit, not the pull request's head. Check out the head instead. |
| Delete `persist-credentials: false` | actions/checkout left a credential in the checkout. Set `persist-credentials: false`. |
| Delete `fetch-depth: 0` | The base commit is not in the checkout, so the rules and the policy cannot be read at it. Check out with `fetch-depth: 0`. |

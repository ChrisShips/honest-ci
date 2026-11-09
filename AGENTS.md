# Conventions for writing code in this repository

## Before you commit

```bash
make check   # ruff + mypy + pytest (coverage ≥ 75%)
```

CI runs exactly these three, plus a gitleaks secret scan. Don't get around a check with a comment or `# type: ignore`. If you really have to, write down why.

## Four hard constraints (when a change touches them, give the reason first)

1. **Gate verdicts are decided by deterministic code.** A model may explain and may serve as a reading aid; it may not rewrite a verdict or take the place of an approval. Any new model output must first pass the citation check and the line-grounding check in `grounding.py`, and must fail soft.
2. **What can't be verified is blocked.** A tool that didn't run ≠ a pass. The distinction between `ToolReport.available` and `passed` must not be collapsed; `Gate._enforce_fail_closed` must not be bypassed.
3. **Approvals are bound to the fingerprint.** Any change that lets an old approval stay in effect on a new commit is wrong.
4. **Credentials never go into the sandbox, the artifacts or the logs.** Outbound requests go through `utils.open_validated_http_request`; writes to disk go through `ArtifactStore`; both have redaction built in.

## The public edition's boundary

This repository is the public edition, and its license is the Fair Core License, FCL-1.0-ALv2: the source is public; any use is allowed except offering a competing product, as long as the license key check isn't removed or got around; where a key is due is the check's to decide (none on a public repository, on a personal account's private repository whose pull requests only the owner opens, or outside CI; one after a 14-day trial on an organization's private or internal repository, and on a personal private one where other people open pull requests); each version becomes Apache-2.0 two years after its release. LICENSE is the rule; the README's License section says it in plain words, and where it says a key is due is what the license check's code checks, so change the three together. Externally (README, docs, commit messages, package metadata), write source-available, **not open source**: FCL is not an OSI-approved open source license. Team edition features don't go into this repository; to add one, stop and ask first.

- **The criterion:** is the feature for the people on one particular PR (whoever opens it, whoever reviews it), or for the people who look at the whole picture (whoever manages the team, whoever audits)? The former goes into this repository, the latter into the team edition.
- **Public edition:** everything a single PR uses from detection to being let through is here: the gates and rules (the test-integrity checks included), the PR comment and the commit status, approving on the PR, marking a single finding as a false positive, and the current run's evidence pack.
- **Team edition:** statistics across runs (block rate, false-positive rate, human effort), run history and retention, the eval set and replay, audit-facing evidence export and an independent verifier, rules shared across repositories.
- **Left out for now:** GitLab and Gitee merge requests, the webhook service (`honest-ci serve`) and the offline install. Ask before putting any of them back.
- **Approval must stay in the public edition.** Going to a person is a common terminal state, and the test-integrity checks go to a person by default too. Without approval, such a PR can only be merged by an admin bypassing branch protection, and the public edition would leave PRs stuck under its default configuration.
- **No code planted ahead of time for the team edition.** Plugin interfaces, reporting endpoints: add them when the team edition is actually built. The license check isn't one of them: it is for this edition's own paid use, and has its own rules in [The license check](#the-license-check).
- **When unsure, leave it out for now.** A released version can't be taken back; what was left out can be put in any time later.

## The license check

Under the FCL, the license check is the line between free and paid: where it doesn't check, anyone may use honest-ci free of charge, and where it does, getting around it breaks the license. It runs in a team's own CI, where a check that goes wrong stops every merge. So it is held to these rules:

- **No server of ours.** It verifies a signed key offline, against the public key shipped in the package, and reports no usage and no counts. It asks only GitHub, and only about the repository GitHub already hosts.
- **Only in CI, and only where a key is due.** It checks the commands that decide a change, run-pr, verify and resume, when they run in CI: any CI, not only GitHub Actions. Run on someone's own computer, nothing is checked. A public repository, or a personal account's private repository whose pull requests only its owner opens, is never asked for a key.
- **Never a pass, never a verdict.** A run without a license ends as an error that says the gates didn't run, why, and where to get a key. It never turns the status green and never says anything about the change.
- **When it can't tell, it lets the run through.** A failure of the check itself, such as GitHub not answering when the trial's start is looked up, never stops a run; the run says so in its log.
- **A paying customer isn't cut off on the expiry date.** A key past its expiry keeps working for 30 days, and says so on every run. A trial key (plan `trial`) is the exception: nobody renews a trial, so its expiry date is its last day, as the 14th day is for the trial in GitHub Actions, and like that trial it says on every run how many days are left.
- **Nothing in the evidence.** The key and what the check found stay out of the evidence pack and the fingerprint. In the artifacts and the logs the key is a credential (hard constraint 4).
- **No new dependency.** The signature is checked with the standard library. The private signing key never enters this repository, CI or the sandbox.
- **No obfuscation, no tamper resistance.** The source is public and anyone can delete the check. It is the license that forbids that, and what binds is the license, not the check, so don't spend code on hiding or guarding it.

## Layers

- **Gates** (`gates/`) only make the judgment and return a `GateReport`. They **don't change state** and don't decide where to go next.
- **The control plane** (`runner.py`) alone owns routing, retries, invalidation and checkpoints. When you want to write `if ... jump to the next one` in a gate, that means it belongs in the runner.
- **Ports** (`adapters/base.py`) are the boundary of replaceability. Gates depend only on ports, not on a particular platform or a particular model.

## Style

- Code, comments and user-facing text (CLI output, PR comments, evidence packs and reports, the README and docs/) are all in English. What is still in Chinese is the Java example (`examples/java-junit5/`) and parts of `tests/`; anything new is written in English directly. Chinese that is there on purpose stays: test inputs, and examples such as a file name git has to quote in `diffs.py`.
- Comments explain **why**, not "what it does". Above all, why a rule draws its line where it does, and why a default falls on the side it does.
- Type annotations are mandatory (mypy `disallow_untyped_defs`).
- Tests are written as pytest functions, named for the behavior (`test_a_check_that_never_ran_blocks_by_default`), and assert properties of the product rather than implementation details.
- Before adding a dependency, ask first: will it become a burden in a private deployment?

## Citing documents

When you cite a § number, a table or a passage, and that document hasn't come up earlier in this conversation, you must write out the file name: `AGENTS.md` hard constraint 4, not just "constraint 4". You don't know which copy the other side has open, or whether it has uncommitted changes; a citation that doesn't name its file is one the other side can't check.

The README is for people who install the Action on GitHub.

## Where things are

See [README.md](README.md#repository-layout).

#!/usr/bin/env bash
# Build a git repository from this demo, for trying honest-ci on GitHub:
#
#   main               the shop, with honest-ci installed by `honest-ci init --github`
#   <scenario>         main plus one patch from scenarios/
#   deselect-test      main plus a test left out of the run
#   before-honest-ci   the shop before honest-ci was installed
#   install-honest-ci  main's commit again, for a pull request into
#                      before-honest-ci: the one that installs honest-ci
#
# GITHUB.md next to this script says what to push, which pull requests to open
# and what each one should show.
#
# Usage: github-demo.sh [--action-ref REF] DIR
#
#   DIR           a directory that does not exist yet, or is empty
#   --action-ref  call the Action at REF (a branch, a tag or a commit of
#                 ChrisShips/honest-ci) instead of at the release init pins,
#                 for trying a commit that is not released yet
#
# init is run for real rather than its files copied, so building the repository
# is also a check of init. It runs the honest-ci on PATH, or the one HONEST_CI
# names.
set -euo pipefail

usage() {
  echo "usage: $0 [--action-ref REF] DIR" >&2
  exit 2
}

ref=""
while [ $# -gt 0 ]; do
  case "$1" in
    --action-ref)
      [ $# -ge 2 ] || usage
      ref="$2"
      shift 2
      ;;
    -*) usage ;;
    *) break ;;
  esac
done
[ $# -eq 1 ] || usage
target="$1"

# The ref is spliced into the workflow file, so it may only be what a ref can
# be; anything else could carry YAML of its own.
if [ -n "$ref" ] && [[ ! "$ref" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]]; then
  echo "--action-ref takes a branch, a tag or a commit; got '$ref'." >&2
  exit 2
fi
if [ -e "$target" ] && [ -n "$(ls -A "$target")" ]; then
  echo "$target is not empty. Give a directory that does not exist yet." >&2
  exit 2
fi

demo="$(cd "$(dirname "$0")" && pwd)"
honest_ci="${HONEST_CI:-honest-ci}"
command -v "$honest_ci" > /dev/null || {
  echo "$honest_ci not found. Install honest-ci, or set HONEST_CI to its path." >&2
  exit 2
}

mkdir -p "$target"
cd "$target"
git init -q
git symbolic-ref HEAD refs/heads/main

# The project and nothing of honest-ci's: its configuration comes from init
# below, and the scenarios become branches of their own.
cp -R "$demo/src" "$demo/tests" "$demo/pyproject.toml" "$demo/.gitignore" .
cat > README.md << 'EOF'
# honest-ci demo

A small shop's pricing code, in integer cents, with six pytest tests.

`main` has honest-ci installed the way `honest-ci init --github` writes it.
Each branch off it is one change a coding agent might hand in, and the pull
request from it into `main` shows what honest-ci makes of that change.
`before-honest-ci` is the shop before honest-ci and `install-honest-ci` the
change that installed it, so that installing can be tried as a pull request
too.

Built by `examples/python-agent-demo/github-demo.sh` in
[honest-ci](https://github.com/ChrisShips/honest-ci), whose `GITHUB.md` next to
that script says what each pull request should show.
EOF
git add -A
git commit -q -m "The shop, before honest-ci"
git branch before-honest-ci

# What init prints is for someone installing by hand, who then commits; here
# the script commits. Shown only when init fails, which would otherwise stop the
# script without a word.
if ! said="$("$honest_ci" init --github 2>&1)"; then
  echo "$said" >&2
  exit 1
fi
if [ -n "$ref" ]; then
  workflow=.github/workflows/honest-ci.yml
  sed -E "s#^([[:space:]]*- uses: ChrisShips/honest-ci)@[^[:space:]]+#\1@$ref#" "$workflow" > "$workflow.new"
  if cmp -s "$workflow" "$workflow.new"; then
    echo "No call to ChrisShips/honest-ci in $workflow to point at $ref." >&2
    exit 1
  fi
  mv "$workflow.new" "$workflow"
fi
git add -A
git commit -q -m "Install honest-ci with honest-ci init --github"
git branch install-honest-ci

for patch in "$demo"/scenarios/*.patch; do
  name="$(basename "$patch" .patch)"
  git checkout -q -b "$name" main
  git apply "$patch"
  git add -A
  git commit -q -m "Scenario: $name"
done

# Not one of the demo's scenarios: the one change here that honest-ci blocks
# rather than hands to a person, so that the pull requests show every status
# the Action sets. pyproject.toml ends in pytest's table, so the line lands
# there.
git checkout -q -b deselect-test main
echo 'addopts = "--deselect tests/test_pricing.py::test_shipping_is_free_from_the_threshold"' >> pyproject.toml
git commit -q -am "Leave the free-shipping test out of the run"

git checkout -q main
echo "Built $(pwd). Its branches:"
git branch --format='  %(refname:short)'
echo "What to push, and what each pull request should show: $demo/GITHUB.md"

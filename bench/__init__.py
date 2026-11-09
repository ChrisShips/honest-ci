"""A benchmark of honest-ci's test-integrity checks on real pull requests.

It runs ``honest_ci.integrity.detect`` -- the detector itself, imported, not a
copy -- over pull requests that coding agents opened in public GitHub
repositories, and over pull requests people opened in the same repositories
at the same time. How the pull requests are chosen and what the numbers mean
is in bench/README.md; ``python -m bench --help`` runs it.
"""

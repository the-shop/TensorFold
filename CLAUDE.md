# Contributor rules for this fork

This repository is the-shop's fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (Python engine,
MIT). These rules apply to every contributor and every automated tool working in it. `AGENTS.md` is a link to this
file.

## Releases

- A release is an annotated git tag `vMAJOR.MINOR.PATCH` (semver) on `main`.
- `main` is always releasable: tests pass and the docs match the code.
- Every release tag has a matching entry in `CHANGELOG.md`, newest first.
- Never force-push `main`. Never move or delete a published tag; fix forward with a new release.

## Commits and attribution

- Commits are authored `Mladen Lotar <mladen@the-shop.hr>`.
- No AI attribution anywhere: no `Co-Authored-By` trailers for tools, no session or chat links, no
  "Generated with" lines, and no agent, session or lane narration, in commits, pull requests, docs or tags.
- Credit upstream in commit messages: end each one with
  `Builds on TensorFold (https://github.com/ashhart/TensorFold) by ashhart.`

## Numbers and data

- Benchmark numbers in docs come only from measured results validated by two independent harnesses, with
  token-identity checks (drafted against `"draft": false` output, and across configurations).
- Never publish machine-local paths, hostnames, usernames or other private data in code, docs, commits or tags.

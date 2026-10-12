# Contributor rules for this fork

This repository is the-shop's fork of [ashhart/TensorFold](https://github.com/ashhart/TensorFold) v0.3.6.3 (Python
engine; MIT, with Apache-2.0 portions (see THIRD_PARTY_NOTICES.md)). These rules apply to every contributor and every
automated tool working in it. `AGENTS.md` is a link to this file.

## Releases

- A release is an annotated git tag `vMAJOR.MINOR.PATCH` (semver) on `main`.
- `main` is always releasable: tests pass and the docs match the code.
- Every release tag has a matching entry in `CHANGELOG.md`, newest first.
- Never force-push `main`. Never move or delete a published tag; fix forward with a new release.

### Release tags and upstream's tags

Upstream ashhart/TensorFold has its own tags with the same names (`v0.3.0` to `v0.3.7`, `v0.4.0` and later, and
its own `v0.1.0` and `v0.2.0`). A clone that also fetches upstream already holds upstream's `v0.3.0`, so a tag made
or pushed from it can name the wrong commit, and `git push --tags` from it would copy upstream's tags into this
fork.

- Release tags `v0.3.0` and later are created and pushed only to the-shop/TensorFold, one tag at a time
  (`git push <remote> refs/tags/vX.Y.Z`). Never `git push --tags` or `--follow-tags` to it.
- Create the annotated tag in a clean clone of the-shop/TensorFold that has no upstream remote:

  ```bash
  git clone https://github.com/the-shop/TensorFold.git tf-tag && cd tf-tag
  git log -1 --format='%H %s' main        # the release commit, already fast-forwarded onto main
  git tag -a vX.Y.Z -m "TensorFold X.Y.Z (the-shop fork)" main
  git push origin refs/tags/vX.Y.Z
  ```

- In a clone that also fetches upstream, never run `git tag` for a release name. Before checking out or comparing
  a fork release there, compare `git rev-parse vX.Y.Z^{}` with `git ls-remote <the-shop remote> 'refs/tags/vX.Y.Z^{}'`;
  if they differ, the local tag is upstream's. Delete it locally only (`git tag -d vX.Y.Z`; never on a remote) and
  fetch the fork's tag explicitly:
  `git fetch <the-shop remote> refs/tags/vX.Y.Z:refs/tags/vX.Y.Z`.
- `tensorfold update` in an editable clone runs `git fetch --tags origin` and fast-forwards to the release tag, so
  it works only when `origin` is the-shop/TensorFold.

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

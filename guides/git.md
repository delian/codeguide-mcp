# Git Workflow Guidelines
Mandatory standards for Git: branching, commit hygiene, conventional commits, rebasing vs merging, PR workflow, history-rewrite safety, large files. Git 2.40+, Conventional Commits 1.0.0.

---
name: git
title: Git Workflow Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [git@2.43, conventional-commits@1.0.0, git-lfs@3.5]
requires: []
recommends:
  - semver
  - code-review
  - ci-cd
  - pre-commit
provides:
  - branching-strategy
  - commit-hygiene
  - conventional-commits
  - pr-workflow
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns Git workflow and spends its tokens on what is unique to version control.

---

## 0. Prerequisites & References

This guide canonically owns Git workflow. It binds to — but never restates — these neighbours:

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`semver.md`](guides://semver.md) — version numbers & release semantics. *(Git binding: `git tag -a vX.Y.Z`; `feat`→MINOR, `fix`→PATCH, `feat!`/`BREAKING CHANGE`→MAJOR.)*
> - [`code-review.md`](guides://code-review.md) — PR review depth, approvals, reviewer duties. *(Git binding: the mechanics of opening/merging the PR live here; review policy lives there.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline triggers & required status checks. *(Git binding: branch pushes, PRs, and `v*` tags are the trigger events.)*
> - [`pre-commit.md`](guides://pre-commit.md) — hook framework, hook selection, secret/lint hooks. *(Git binding: hooks fire on `commit`/`push`; this guide only states which Git events to gate.)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (commit tests with implementation; regression-test-before-fix) · [`secure-coding.md`](guides://secure-coding.md) (secret scanning, signed commits) · [`github.md`](guides://github.md) · [`gitlab.md`](guides://gitlab.md) (host-specific PR/MR config) · [`todo.md`](guides://todo.md) (TODO/FIXME conventions).

---

## 1. Core Philosophies

Git-specific principles only. Test-first, security scanning, and review *policy* come from the §0 references — do not restate them.

- **Atomic commits**: one logical change per commit; independently revertible; build stays green at every commit.
- **Readable history**: the log is documentation. No `WIP`/`temp`/`fixup` commits surviving into a shared branch.
- **Conventional Commits**: every commit is machine-parseable so changelog and version bumps (see `semver.md`) can be automated.
- **Traceability**: every branch and commit ties back to an issue/ticket.
- **Rewrite locally, never publicly**: clean up your own un-pushed history freely; treat anything others have pulled as immutable.
- **Protect shared branches**: `main` (and `develop`, if used) accept changes only via reviewed PRs that pass required checks.

**Verified Code**: Agent-generated Git operations MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `GIT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| GIT-MSG-01 | Commit subject MUST follow Conventional Commits 1.0.0: `type(scope): description` | `git log -1 --format=%s \| grep -E '^(feat\|fix\|docs\|style\|refactor\|perf\|test\|build\|ci\|chore\|revert)(\(.+\))?!?: .+'` | match |
| GIT-MSG-02 | Subject MUST be imperative mood, ≤ 72 chars, no trailing period | `git log -1 --format=%s \| awk '{print length}'` + review | ≤ 72, no `.` |
| GIT-MSG-03 | Bodies MUST explain WHY; breaking changes MUST use `!` and/or `BREAKING CHANGE:` footer (see `semver.md`) | review of `git log --format=%B` | present when applicable |
| GIT-MSG-04 | Each commit SHOULD reference its issue (`#<id>` / `Closes #<id>`) where an issue exists | `git log -1 --format=%B \| grep -E '#[0-9]+'` | present |
| GIT-BR-01 | Branches MUST follow `<type>/<issue>-<slug>` | `git rev-parse --abbrev-ref HEAD \| grep -E '^(feature\|bugfix\|hotfix\|release\|refactor\|chore)/'` | match |
| GIT-BR-02 | `main`/`develop` MUST be protected: PR-only, required checks, no force-push (see `code-review.md`, `ci-cd.md`) | host branch-protection API/UI | protections on |
| GIT-HIST-01 | Published history MUST NOT be rewritten (no force-push to shared branches) | `git config --get receive.denyNonFastForwards` / branch protection | non-fast-forward denied |
| GIT-HIST-02 | Force-pushes to personal branches MUST use `--force-with-lease`, never `--force` | review of shell history / CI lint | `--force-with-lease` only |
| GIT-MERGE-01 | Feature branches MUST integrate via squash or rebase (linear history); no WIP commits in shared branches | `git log --merges main` review; `git log --oneline \| grep -iE 'wip\|temp\|fixup'` | linear, no WIP |
| GIT-HOOK-01 | A commit-stage gate MUST run format/lint/secret checks (see `pre-commit.md`) | `pre-commit run --all-files` | exit 0 |
| GIT-IGNORE-01 | Secrets/build artifacts MUST be git-ignored and absent from history (see `secure-coding.md`) | `git ls-files \| grep -E '\.env$\|\.pem$\|\.key$'` ; `gitleaks detect` | no matches, 0 leaks |
| GIT-LFS-01 | Files > 100 MB (or binaries) MUST be tracked via Git LFS, not committed raw | `git lfs ls-files` ; `git cat-file --batch-check` size audit | large files in LFS |
| GIT-TAG-01 | Releases MUST be marked with an annotated, signed tag `vX.Y.Z` (see `semver.md`) | `git tag -v vX.Y.Z` | annotated + verified |

> **Forbidden**: force-pushing a shared branch (violates GIT-HIST-01); committing secrets/large binaries; merging WIP/un-squashed noise into `main`; commits that bypass required checks (`--no-verify` on shared work); lightweight tags for releases.

---

## 3. Branching Strategy

Pick **one** model per repo and apply it consistently.

### A. Trunk-based (default for web apps / continuous deployment)

Short-lived branches off `main`; merge back within a day or two; deploy from `main`.

```
main (always releasable, deploys on merge)
  ├── feature/1234-search-endpoint
  ├── bugfix/5678-login-crash
  └── hotfix/9012-token-leak
```

Best when CI/CD (see `ci-cd.md`) deploys frequently and feature flags (see `feature-flags.md`) gate incomplete work.

### B. Git Flow (for versioned libraries / products with parallel releases)

```
main (tagged releases only)  ←  release/* , hotfix/*
develop (integration)        ←  feature/* , bugfix/*
```

- `main`: production, tagged releases only.
- `develop`: integration of completed work.
- `feature/*`, `bugfix/*`: branch from `develop`.
- `release/*`: stabilize a release, branch from `develop`, merge to `main` + `develop`.
- `hotfix/*`: branch from `main`, merge to `main` + `develop`.

### C. GitLab Flow (environment branches)

Trunk plus promotion branches (`main → staging → production`); merge upward to promote. Useful when deploys are gated per environment.

### D. Branch naming (GIT-BR-01)

```
<type>/<issue-id>-<short-slug>
```

| Type | Example |
|------|---------|
| `feature/` | `feature/1234-user-auth` |
| `bugfix/` | `bugfix/5678-login-crash` |
| `hotfix/` | `hotfix/9012-token-leak` |
| `release/` | `release/v1.2.0` |
| `refactor/` | `refactor/3456-extract-validator` |
| `chore/` | `chore/7788-bump-deps` |

Lowercase, hyphenated slug; no spaces, no personal names. Delete branches after merge.

### E. Protection (GIT-BR-02)

`main`/`develop` MUST require: PR before merge, passing status checks (see `ci-cd.md`), at least one approving review (see `code-review.md`), linear history, signed commits, and no force-push or deletion. Configure via the host (see `github.md` / `gitlab.md`); this guide states *what* to enforce, not the host-specific UI.

---

## 4. Commit Hygiene & Conventional Commits

### A. Format (GIT-MSG-01..04)

```
<type>(<optional scope>)<optional !>: <description>

<optional body — explain WHY, wrap at ~72 cols>

<optional footers — BREAKING CHANGE:, Closes #123, Co-authored-by:>
```

Rules: imperative subject (“add”, not “added”/“adds”); ≤ 72-char subject; no trailing period; blank line before body; body explains motivation and trade-offs, not a restatement of the diff.

### B. Types and their version impact

Mapping below is the *trigger* for `semver.md`; that guide owns the actual version semantics.

| Type | Meaning | SemVer (see `semver.md`) |
|------|---------|--------------------------|
| `feat` | user-facing feature | MINOR |
| `fix` | bug fix | PATCH |
| `perf` | performance improvement | PATCH |
| `refactor` | behavior-preserving change | — |
| `docs` | docs only | — |
| `test` | tests only | — |
| `build` | build system / deps | — |
| `ci` | pipeline config | — |
| `style` | formatting only | — |
| `chore` | maintenance | — |
| `revert` | reverts a prior commit | — |

A `!` after type/scope **or** a `BREAKING CHANGE:` footer signals a MAJOR bump.

```
feat(auth): add WebAuthn passkey login

Closes #1234

refactor(api)!: drop deprecated v1 routes

BREAKING CHANGE: v1 endpoints removed; clients must migrate to v2.
```

### C. Atomic commits

One logical change per commit. Stage precisely (`git add -p` for partial hunks). Tests ship in the same commit as the code they cover, and a bug fix ships with its regression test — the *test-first* and *regression-before-fix* discipline is owned by [`tdd.md`](guides://tdd.md); Git's only rule is that they land **together** so each commit is independently revertible and bisectable.

### D. Optional commit template

```bash
git config --global commit.template ~/.gitmessage   # subject + reminder comments
```

---

## 5. Rebasing vs Merging

| Situation | Strategy | Why |
|-----------|----------|-----|
| Update my feature branch with latest `main` | `git fetch && git rebase origin/main` | linear, no noise merges |
| Integrate a finished feature into `main` | squash-merge (or rebase-merge) the PR | one clean commit per unit of work |
| Bring a long-lived `release`/`hotfix` back to `main`/`develop` | `git merge --no-ff` | preserves the release boundary |
| Combine many small public histories | `merge` | never rebase shared history |

**Golden rule (GIT-HIST-01/02)**: rebase only commits that exist *only* on your local/personal branch. Once others may have pulled it, the history is public — merge, don't rebase, and never force-push it.

```bash
# Sync feature branch (local-only commits) onto latest main
git fetch origin
git rebase origin/main          # replay my commits on top

# Push a rewritten personal branch safely
git push --force-with-lease     # refuses if remote moved under you — never plain --force
```

PR merge-button choice (squash / rebase / merge-commit) is a per-repo policy; the mechanics of opening and merging the PR are covered in §7 and reviewed per `code-review.md`.

---

## 6. History Rewriting — Safety

Rewriting is powerful and irreversible on shared branches. Constrain it.

- **Local cleanup before pushing** — fine: `git rebase -i HEAD~5` to reorder/squash/reword `pick → squash/fixup/reword`. `git commit --amend` only on un-pushed commits.
- **`--autosquash`** — annotate with `git commit --fixup=<sha>` / `--squash=<sha>`, then `git rebase -i --autosquash` to fold them automatically.
- **Recover mistakes** — `git reflog` plus `git reset --hard <reflog-entry>` undoes a bad rebase/reset; reflog is your safety net.
- **Purge a leaked secret from history** — use `git filter-repo` (not the deprecated `filter-branch`), rotate the secret regardless (see `secure-coding.md`), then force-push and have all clones re-clone. This is the *only* sanctioned rewrite of shared history and requires team coordination.
- **Never** rewrite a public/shared branch for cosmetic reasons; never `push --force` to `main`/`develop` (blocked by GIT-HIST-01 protection).

---

## 7. Pull Request Workflow

Git owns the PR *mechanics*; review depth and approval rules are owned by [`code-review.md`](guides://code-review.md), pipeline checks by [`ci-cd.md`](guides://ci-cd.md).

1. Branch per GIT-BR-01; commit per §4; keep it small (a PR reviewable in well under an hour beats a 1000-line dump — split refactor / foundation / feature into separate PRs).
2. Rebase onto the target branch (§5) so the diff is clean and conflict-free.
3. Push and open the PR; link the issue in the description (`Closes #<id>`) so merge auto-closes it.
4. Fill the PR template (`.github/pull_request_template.md`) and let CI run all required checks.
5. Address review feedback as new commits while under review; squash on merge so the shared branch gets one clean commit.
6. After merge: delete the branch; verify the issue closed; monitor the deploy (see `observability.md`).

A minimal, durable PR template lives at `.github/pull_request_template.md` (Summary, Linked issue, Type of change, How tested, Checklist). Keep it short; do not duplicate the §8 checklist or the `code-review.md` reviewer checklist inside it.

---

## 8. Repository Hygiene Files

### A. `.gitignore`

Ignore secrets, local env, dependencies, build output, and OS/IDE cruft. Secrets must *also* never reach history (GIT-IGNORE-01; policy: `secure-coding.md`).

```gitignore
# secrets / local env
.env
.env.*.local
*.key
*.pem
.secrets.toml

# dependencies & build
node_modules/
dist/
build/
*.log

# OS / IDE
.DS_Store
.idea/
.vscode/
```

Prefer a language-specific template (e.g. GitHub's `gitignore` collection) as the base, plus the secret patterns above.

### B. `.gitattributes`

Normalize line endings and mark binary/LFS paths so diffs and checkouts are deterministic.

```gitattributes
* text=auto eol=lf
*.png binary
*.psd filter=lfs diff=lfs merge=lfs -text
```

### C. Hooks (GIT-HOOK-01)

Client-side hooks gate `commit`/`push`. **Do not hand-roll `.git/hooks` scripts** — they are not versioned and drift per developer. Use the `pre-commit` framework (formatting, linting, secret scanning, commit-msg lint) — its hook selection and config are owned by [`pre-commit.md`](guides://pre-commit.md). Git's only requirement here is that a commit-stage gate exists and that secret-scanning runs before code leaves the workstation.

---

## 9. Large Files & Repo Size

- **Git LFS (GIT-LFS-01)**: track large binaries (datasets, media, model weights, archives) with `git lfs track "*.bin"`; commit the resulting `.gitattributes`. Keeps the packfile lean and clones fast.
- **Never** commit build artifacts, vendored dependencies, or generated output — ignore them instead.
- **Shallow / partial clone** for CI and large repos: `git clone --depth=1` or `--filter=blob:none`; `git sparse-checkout` to fetch only needed paths.
- A binary already committed raw cannot be shrunk by deleting it later (it lives in history) — migrate with `git lfs migrate import` or `git filter-repo`, which rewrites history (§6) and requires re-clone.

---

## 10. Release Tagging

Release semantics are owned by [`semver.md`](guides://semver.md); Git's binding is the tag itself (GIT-TAG-01).

```bash
git tag -a v1.4.0 -m "Release v1.4.0"   # annotated (carries author/date/message)
git tag -s v1.4.0 -m "Release v1.4.0"   # signed (preferred for production)
git push origin v1.4.0                   # pushing a v* tag triggers release CI (see ci-cd.md)
git tag -v v1.4.0                        # verify signature
```

Use **annotated/signed** tags, never lightweight ones, for releases. Changelog and version bumps are automated from Conventional Commits (e.g. release-please, git-cliff) — wired in `ci-cd.md`.

---

## 11. Quick Reference

```bash
# branch
git switch -c feature/1234-search origin/main      # create from up-to-date main
git push -u origin feature/1234-search

# commit (conventional, atomic)
git add -p                                          # stage precise hunks
git commit -m "feat(search): add user search endpoint" -m "Closes #1234"

# sync & clean
git fetch origin && git rebase origin/main          # linear update
git rebase -i --autosquash HEAD~5                   # local cleanup before push
git push --force-with-lease                         # safe rewrite of personal branch

# recover
git reflog && git reset --hard <entry>              # undo a bad rebase/reset

# release
git tag -s v1.4.0 -m "Release v1.4.0" && git push origin v1.4.0
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] GIT-MSG-01/02 — subjects are Conventional Commits, imperative, ≤ 72 chars, no trailing period
- [ ] GIT-MSG-03 — WHY explained; breaking changes flagged with `!` / `BREAKING CHANGE:` (see `semver.md`)
- [ ] GIT-MSG-04 — commits reference their issue where one exists
- [ ] GIT-BR-01 — branch named `<type>/<issue>-<slug>`
- [ ] GIT-BR-02 — `main`/`develop` protected: PR-only, required checks, no force-push (see `code-review.md`, `ci-cd.md`)
- [ ] GIT-HIST-01 — no rewrite/force-push of shared history
- [ ] GIT-HIST-02 — personal-branch force-pushes use `--force-with-lease`
- [ ] GIT-MERGE-01 — feature branches squashed/rebased; no WIP in shared branches
- [ ] GIT-HOOK-01 — commit-stage gate runs format/lint/secret checks (see `pre-commit.md`)
- [ ] GIT-IGNORE-01 — secrets/artifacts ignored and absent from history (see `secure-coding.md`)
- [ ] GIT-LFS-01 — large/binary files tracked via Git LFS
- [ ] GIT-TAG-01 — releases marked with annotated, signed `vX.Y.Z` tag (see `semver.md`)

---
**End of Git Workflow Guidelines**

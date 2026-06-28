# Migration Spec (for subagents)

You are migrating ONE coding-standards guide in /home/delian/src/codeguide-mcp to the reference-based v2 format. Your orchestrator will tell you the target guide name, its `kind`, its requirement-ID `prefix`, and its `requires`/`recommends` lists.

## Read first (in order)
1. `guides/CONVENTIONS.md` — the authoring spec. Follow it exactly.
2. `guides/TEMPLATE.md` — the v2 structure.
3. `guides/python.md` — the GOLD-STANDARD migrated example. Match its style and density.
4. `guides/<target>.md` — your target's current content.

## Migrate `guides/<target>.md` IN PLACE (overwrite it)

### Classify every section
- **Cross-cutting content owned by ANOTHER guide** (see CONVENTIONS §1 ownership map) → delete the prose; replace with a `📎 REQUIRED` / `RECOMMENDED` / `SEE ALSO` reference plus at most a one-sentence technology binding (the "balanced rule").
- **Content this guide canonically owns** → KEEP, but compress: cut repetition and project/domain-specific example bloat. Stay **EXTENSIVE and authoritative** — never strip unique depth just to be short. Per-language code dumps inside a non-language guide are bloat → name the idiom and reference the language guide instead.

### Required structure (exactly as TEMPLATE.md / python.md)
1. Line 1: `# <Proper Title> Guidelines` (or the natural title).
2. Line 2: ONE-LINE brief sentence. **The server extracts this — never put YAML above it.**
3. A `---` … `---` YAML frontmatter block containing: `name`, `title`, `version: 2.0`, `last_reviewed: 2026-06-05`, `kind`, `tools` (current versions where relevant; `[]` if language-agnostic), `requires`, `recommends`, `provides` (what THIS guide canonically owns).
4. `## 0. Prerequisites & References` — the 📎 markers for `requires` (REQUIRED), `recommends` (RECOMMENDED), and any SEE ALSO.
5. `## 1. Core Philosophies` — specific to this guide; do NOT restate referenced concerns.
6. `## 2. Requirements (MANDATORY, auditable)` — a table: `| ID | Requirement | Verify | Gate |`. IDs are `<PREFIX>-<TOPIC>-<NN>`. Use RFC-2119 keywords (MUST/MUST NOT/SHOULD/MAY). Each row has a concrete Verify command/method and a binary Gate. Rows that bind a shared rule cite the owner, e.g. "(see `tdd.md`)".
7. The remaining sections: the guide's owned, technology-specific content — extensive and modern.
8. `## Deployment Checklist` (or `## N. Deployment Checklist`) — generated 1:1 from the §2 requirement IDs. No new requirements.
9. End with `**End of <Title> Guidelines**`.

### Modernize
Fix stale or incorrect guidance, bump to current tool versions/idioms, and remove unsourced marketing stats. Do NOT carry forward the old `**Agent Profile** / Role / Objective / Tools` header block or trailing `Maintainer/Last Updated/version 1.0` footers — that metadata now lives in frontmatter.

### Verify before finishing
- `extract_brief` works: line 1 is `# Title`, line 2 is the brief, then `---`.
- Every `guides://<name>.md` reference target is in the valid-names list below — NO dead links.
- No duplicated cross-cutting prose remains.

## Valid guide names (use `guides://<name>.md` only for these)
accessibility, adr, agents-md, android, angular, architectures, aws, azure, azuredevops, bash, berkeleydb, c, cassandra, chroma-vectordb, chrome-extension, ci-cd, cleanarch, cmake, cockroachdb, code-review, coding-ai, comments, conan, couchbase, couchdb, cpp, csharp, css, cuda, deno, designpatterns, devops, docker-compose, dockerfile, duckdb, e2e-testing, elasticsearch-opensearch, elixir, env-config, error-handling, feature-flags, fish, flutter, gcp, git, github, gitlab, go, graphql, grpc, haskell, hexagonal, html, influxdb, ios, istio, java, javascript, jenkins, kafka, kotlin, kubernetes, leveldb, libsql-turso, logging, lua, make, markdown, material, memcached, microservices, mlops, mongodb, mutmut, mysql-mariadb, neo4j, nextjs, nix-flake, nodejs, oauth, observability, openapi, parallelism, performance, php, poetry, postgresql, pre-commit, python, python-z3, pytorch, react-native, reactjs, redis, rest, rethinkdb, rocksdb, ruby, rust, scala, scylladb, secure-coding, semver, sql, sqlalchemy-alembic, sqlc, sqlite, svelte, swift, tdd, terraform, timescaledb, todo, typescript, ui, uv, verilog, vite, websocket, zig, zod, zsh

## Return ONLY a short report
- before/after line counts (`wc -l`)
- sections replaced with references (and the target guides)
- sections kept
- defects/stale guidance fixed
Do not ask questions; make sensible decisions and complete the migration.

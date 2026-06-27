# sqlc Development Guidelines
Mandatory standards for sqlc: write SQL, generate type-safe data-access code. Compile-time query verification, parameterized-by-design, zero-runtime-reflection. sqlc 1.27+, PostgreSQL/MySQL/SQLite, Go (primary).

---
name: sqlc
title: sqlc Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [sqlc@1.27, pgx@v5, postgresql, mysql, sqlite]
requires: []
recommends:
  - sql
  - go
  - postgresql
  - mysql-mariadb
  - secure-coding
provides:
  - sqlc-codegen
  - type-safe-queries
  - sqlc-config
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to sqlc — the SQL→code generator. The SQL you write is owned by `sql.md`/the engine guides; the code sqlc emits is owned by `go.md`.

---

## 0. Prerequisites & References

sqlc is a **code generator**, not a database or a language. It sits between SQL you author and the data-access layer it emits. Fetch the owners of those concerns:

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`sql.md`](guides://sql.md) — how to write the queries themselves (joins, CTEs, indexing). sqlc does **not** redefine SQL; it parses it. Do not restate SQL rules here.
> - [`go.md`](guides://go.md) — the primary output language: error wrapping, `context`, package layout, `database/sql` vs `pgx/v5`.
> - [`postgresql.md`](guides://postgresql.md) · [`mysql-mariadb.md`](guides://mysql-mariadb.md) — the engines. Schema syntax, types, enums, extensions belong to these.
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain & secrets. *(sqlc binding: every generated query is parameterized → SQL injection is structurally impossible; see §6.)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) (test-first against generated code) · [`error-handling.md`](guides://error-handling.md) (mapping driver errors) · [`ci-cd.md`](guides://ci-cd.md) (the generate/vet/diff gate) · [`performance.md`](guides://performance.md) · [`sqlite.md`](guides://sqlite.md)

---

## 1. Core Philosophies: SQL-FIRST

sqlc-specific principles only. SQL correctness, security, and Go idioms come from §0.

- **S**ource of truth is SQL: write real, engine-native SQL — never an ORM/query-builder abstraction. sqlc generates *from* it.
- **Q**uery verification at build time: sqlc parses every query against the schema. Type/column/syntax errors fail `sqlc generate`, not production.
- **L**ightweight output: emitted code is plain Go (or Python/Kotlin) with direct driver calls — no reflection, no runtime query assembly.
- **F**rozen generated code: the `*.sql.go` output is read-only. To change behavior, change the SQL and regenerate — never hand-edit.
- **I**njection-safe by construction: queries are static; all inputs are bound parameters. There is no string-concatenation code path to misuse.
- **R**eproducible: same schema + same queries + same config ⇒ byte-identical output. This makes drift a CI-checkable property (`sqlc diff`).
- **S**chema-as-input: sqlc reads your migration/DDL files as the schema; the migrations are the contract.
- **T**ypes flow from DB to language: column types map to language types automatically; nullability is modeled explicitly.

**Verified Code**: Agent-generated SQL+config MUST pass `sqlc compile`, `sqlc vet`, regenerate with no diff, build in the target language, and pass query tests before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SQLC-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SQLC-CFG-01 | Config MUST use v2 syntax (`version: "2"`) | head of `sqlc.yaml` | `version: "2"` |
| SQLC-CMP-01 | All queries MUST parse against the schema | `sqlc compile` | exit 0 |
| SQLC-VET-01 | Queries MUST pass vet lint rules | `sqlc vet` | exit 0, 0 warnings |
| SQLC-GEN-01 | Generated code MUST be committed & in sync (no drift) | `sqlc diff` | no diff |
| SQLC-GEN-02 | Generated code MUST NOT be hand-edited | regenerate → `git diff` | no manual edits |
| SQLC-ANN-01 | Every query MUST carry a `-- name: X :command` annotation | `sqlc compile` | all named |
| SQLC-TYP-01 | Nullable columns MUST map to nullable types (no panics on NULL) | `sqlc generate` + build | compiles, NULL-safe |
| SQLC-SEC-01 | Parameterized inputs only; no broad secret columns (see `secure-coding.md`) | review / `sqlc vet` | 0 dynamic SQL |
| SQLC-OUT-01 | Output MUST build in the target language | `go build ./...` | exit 0 |
| SQLC-TST-01 | Queries MUST be tested against a real engine (see `tdd.md`) | `go test ./...` | exit 0, 0 skips |

> **Forbidden**: hand-editing generated `*.sql.go`; committing a SQL change without regenerating; concatenating SQL strings around sqlc output; using `SELECT *` in security-sensitive reads (leaks `password_hash`); ignoring `sqlc vet` warnings.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
sqlc compile      # SQLC-CMP-01: parse queries vs schema, no output
sqlc vet          # SQLC-VET-01: CEL lint rules (+ db-prepare if managed)
sqlc generate     # regenerate output
sqlc diff         # SQLC-GEN-01: fail if regenerated output differs from committed
go build ./...    # SQLC-OUT-01
go test ./...     # SQLC-TST-01 (real engine via testcontainers / service container)
```

`sqlc diff` is the drift gate: it runs generation in-memory and reports any delta from the committed files without touching the working tree — ideal for CI. The *why* behind test-first and CVE policy lives in the §0 owners.

---

## 4. Project Structure

sqlc reads a **queries** directory and a **schema** (your migration DDL) and writes generated code. Keep them separate from hand-written code.

```
project/
├── sqlc.yaml                 # config (single source — see §5)
├── db/
│   ├── migrations/           # DDL = the schema sqlc parses (NNNNNN_*.up/down.sql)
│   ├── queries/              # *.sql — annotated queries, grouped by entity
│   └── sqlc/                 # GENERATED — never edit (db.go, models.go, *.sql.go, querier.go)
├── internal/repository/      # hand-written layer wrapping db.Queries (errors, domain types)
└── ...
```

- `schema:` may point at the migrations dir directly — sqlc applies them in order to build the catalog. No separate schema file needed.
- Treat `db/sqlc/` as build output: commit it (so consumers need no sqlc to build) but regenerate, never patch.
- The repository layer (not the generated code) is where you map driver errors to domain errors (see `error-handling.md`) and apply layering (`go.md`).

---

## 5. Configuration — `sqlc.yaml` (the heart)

v2 is mandatory. One `sql` block per engine; each block names its queries, schema, and one or more `gen` language targets.

```yaml
version: "2"
sql:
  - engine: "postgresql"        # postgresql | mysql | sqlite
    queries: "db/queries"
    schema: "db/migrations"     # migrations dir IS the schema
    database:
      managed: true             # sqlc spins an ephemeral engine for `sqlc vet` db-prepare
    gen:
      go:
        package: "db"
        out: "db/sqlc"
        sql_package: "pgx/v5"   # preferred PG driver; else "database/sql"
        emit_json_tags: true
        emit_interface: true            # generate a Querier interface (mockable)
        emit_pointers_for_null_types: true   # NULL → *T instead of sql.NullT
        emit_empty_slices: true         # :many returns [] not nil
        emit_enum_valid_method: true
        overrides:
          - db_type: "uuid"        { go_type: "github.com/google/uuid.UUID" }
          - db_type: "timestamptz" { go_type: "time.Time" }
          - db_type: "jsonb"       { go_type: "encoding/json.RawMessage" }
rules:
  - sqlc/db-prepare              # prepares every query against the managed DB during vet
```

Key choices:
- **Engine** picks the SQL dialect and parameter style (`$1` for PG; `?` for MySQL/SQLite). Schema must match the engine.
- **`sql_package`** (Go/PG): `pgx/v5` for native PG types & batching; `database/sql` for the stdlib path.
- **Multiple `gen` targets** under one block emit several languages from the same SQL (Go is first-class; `python`, `kotlin` are codegen plugins — see §8).
- **`database.managed: true`** (modernize): lets `sqlc vet` boot a throwaway engine and actually `PREPARE` each query, catching errors a static parse misses. No external DB or credentials in config — connection strings come from the environment.

### Nullability & type overrides
- By default a non-NULL column → `T`; a nullable column → `sql.NullT` (stdlib) or `pgtype.T` (pgx). `emit_pointers_for_null_types: true` makes nullable → `*T`, which is usually cleaner.
- `overrides` remap a `db_type` (or a specific `column:` like `authors.bio`) to any Go type, optionally with `nullable: true` and `pointer: true`. Use them for UUIDs, enums, JSON, and domain wrappers.
- Engine type mappings (PG `timestamptz`/`jsonb`/`citext`, MySQL `datetime`/`json`, SQLite affinities) are documented in the engine guides — bind them here, don't re-explain them.

---

## 6. Queries — annotations & parameters (canonical)

Every query is a plain SQL statement preceded by an annotation. This is sqlc's core surface.

### Annotation: `-- name: <PascalCase> :<command>`

| Command | Returns | Use |
|---------|---------|-----|
| `:one` | single row (err if 0) | get-by-id, unique lookup |
| `:many` | slice/list | lists, search |
| `:exec` | nothing | INSERT/UPDATE/DELETE without return |
| `:execrows` | affected row count | bulk update/delete |
| `:execresult` | driver result (lastInsertId, rowsAffected) | MySQL auto-increment |
| `:copyfrom` | nothing | bulk insert via PG `COPY` (fast path) |
| `:batchexec` / `:batchone` / `:batchmany` | pgx batch | pipelined multi-statement |

```sql
-- name: GetUser :one
SELECT id, email, name FROM users WHERE id = $1;

-- name: ListUsers :many
SELECT id, email, name FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2;

-- name: CreateUser :one
INSERT INTO users (email, name) VALUES ($1, $2) RETURNING *;

-- name: DeleteUser :exec
DELETE FROM users WHERE id = $1;
```

`RETURNING *` (PG/SQLite) lets `:one`/`:many` give back the full row after a write — prefer it over a second round-trip.

### Parameter handling — `$N`, `sqlc.arg`, `sqlc.narg`
- **Positional**: `$1, $2` (PG) or `?` (MySQL/SQLite). sqlc names the Go params from the column.
- **`sqlc.arg('name')`** — a *named, required* parameter; produces a named Go field. Use when positional names would be ambiguous.
- **`sqlc.narg('name')`** — a *named, nullable* parameter, the idiom for optional filters and partial updates:

```sql
-- name: SearchUsers :many
SELECT id, email, name FROM users
WHERE (sqlc.narg('email')::text IS NULL OR email ILIKE sqlc.narg('email'))
  AND (sqlc.narg('name')::text  IS NULL OR name  ILIKE sqlc.narg('name'))
ORDER BY created_at DESC
LIMIT sqlc.arg('limit') OFFSET sqlc.arg('offset');

-- name: UpdateUserPartial :one
UPDATE users
SET name = COALESCE(sqlc.narg('name'), name),
    email = COALESCE(sqlc.narg('email'), email)
WHERE id = sqlc.arg('id')
RETURNING *;
```

- **`sqlc.slice('ids')`** expands a Go slice into an `IN (...)` list (single-engine `IN` parameters otherwise can't bind a variadic).
- **`sqlc.embed(t)`** nests a full table struct in a join result (e.g. `SELECT sqlc.embed(authors), sqlc.embed(books) ...`) so the row maps to `{Author, Book}` instead of flattened columns.

The actual SQL design (which columns, which indexes, join shape) is owned by [`sql.md`](guides://sql.md) and the engine guides — keep query bodies idiomatic to the engine, here just bind them to sqlc's annotation/param syntax.

---

## 7. Security binding (sqlc is SQLi-safe by design)

Policy is owned by [`secure-coding.md`](guides://secure-coding.md). The sqlc-specific facts:

- **No injection surface.** Generated queries are static prepared statements; every value is a bound parameter. There is no API that interpolates user input into SQL — string concatenation simply cannot be expressed. This is the single biggest security property sqlc provides.
- **`sqlc vet` rules** (CEL expressions, optionally backed by a managed DB) can fail builds on policy violations — e.g. forbid `SELECT *`, require `LIMIT`, or block sequential scans. Add project rules under `rules:`.
- **Least exposure**: select explicit columns; keep `password_hash`/secrets in a dedicated `:one` auth query, never in profile reads. (A `SELECT *` that leaks a hash is a `SQLC-SEC-01` failure.)
- **Credentials** for managed/test databases come from the environment, never `sqlc.yaml`.

---

## 8. Output languages & plugins

- **Go is first-class** and the primary target — `pgx/v5` or `database/sql`, generated `Queries` struct, `WithTx(tx)` for transactions, optional `Querier` interface for mocking. Apply [`go.md`](guides://go.md) to the hand-written code that consumes it.
- **Other languages are codegen plugins**, not built-ins: `python` and `kotlin` ship as official plugins (sync/async queriers for Python; JDBC for Kotlin). Configure them as additional `gen` targets or via the WASM/process `plugins:` mechanism. Treat their output the same way — generated, never edited — and apply that language's guide to the wrapping code.
- Don't dump per-language repository boilerplate into this guide; the pattern (wrap `Queries`, map driver errors to domain errors, expose a transaction helper) is identical across languages and the specifics belong to each language guide.

---

## 9. Integration with migrations

sqlc does not run migrations — it **reads** them as the schema source of truth.

- Point `schema:` at the migrations directory; sqlc applies the `*.up.sql` files in numeric order to build its catalog. Down-migrations are ignored by sqlc.
- Because schema is just DDL, sqlc is migration-tool-agnostic (golang-migrate, Atlas, goose, Flyway). Pick one in `ci-cd.md`/the engine guide; sqlc only needs the resulting DDL.
- Keep DDL and queries in lockstep: a column rename in a migration must be followed by `sqlc generate`, and CI's `sqlc diff` catches the case where someone forgot.
- Engine-managed mode can apply the same migration DDL to the ephemeral vet database, so `sqlc/db-prepare` validates queries against the *current* schema.

---

## 10. CI/CD binding

Pipeline policy is owned by [`ci-cd.md`](guides://ci-cd.md). The sqlc gate, in order, is small and language-agnostic:

```bash
sqlc compile && sqlc vet && sqlc diff   # parse, lint, drift-check — no codegen committed by CI
```

- Pin the sqlc version (`tools:` here is 1.27+); install from the GitHub release or `go install ...@vX`.
- Run the gate on changes to `db/**` and `sqlc.yaml`.
- `sqlc diff` replaces the older "regenerate then `git status --porcelain`" dance: it fails the build on drift without writing files.
- Add `sqlc/db-prepare` under `rules:` with `database.managed: true` (or a CI service container) so vet validates every query against a live engine.
- A pre-commit hook running `sqlc compile && sqlc vet` (see `pre-commit.md`) catches errors before push.

---

## 11. Performance binding

Policy is owned by [`performance.md`](guides://performance.md); query/index tuning by the engine guides. sqlc levers:

- **Prepared queries**: `emit_prepared_queries: true` (or `db.Prepare(ctx, conn)`) caches statements at startup — fewer parse/plan round-trips on hot paths.
- **Batching / COPY**: `:copyfrom` uses PG `COPY` for bulk insert; `:batchexec`/`:batchone`/`:batchmany` pipeline statements over a single pgx round-trip. Reach for these instead of looping single inserts.
- **Connection pooling** (`pgxpool`) and index/keyset-pagination design are engine concerns — bind sqlc queries to them (e.g. keyset `WHERE (created_at, id) < ($1,$2)`), don't re-document pooling here.

---

## 12. Testing binding

Test-first (Red-Green-Refactor) and regression-test-before-fix are owned by [`tdd.md`](guides://tdd.md). sqlc specifics:

- Test **against the generated code**, never against hand-written SQL strings, and never hand-edit the output to make a test pass.
- Use a **real engine** — `testcontainers-go` or a CI service container — not mocks; sqlc's value is real-schema validation. Apply migrations, then `db.New(pool)`.
- Cover both `:one` (incl. not-found → `pgx.ErrNoRows`/`sql.ErrNoRows`) and `:many` paths, and NULL columns (the classic `sql.NullString` vs panic regression).
- Re-run `sqlc generate` in CI so SQL and generated code can't silently diverge (this is also `SQLC-GEN-01`).
- Driver-error → domain-error mapping (unique-violation `23505`, etc.) lives in the repository layer and follows [`error-handling.md`](guides://error-handling.md).

---

## 13. Quick Reference

```bash
sqlc init        # scaffold sqlc.yaml
sqlc compile     # parse queries vs schema (no output)   SQLC-CMP-01
sqlc vet         # lint / db-prepare                      SQLC-VET-01
sqlc generate    # write generated code
sqlc diff        # fail on drift vs committed output      SQLC-GEN-01
sqlc version
```

| Concern | sqlc construct |
|---------|----------------|
| Return shape | `:one` `:many` `:exec` `:execrows` `:execresult` `:copyfrom` `:batch*` |
| Required named param | `sqlc.arg('x')` |
| Optional/nullable param | `sqlc.narg('x')` |
| `IN (...)` from a slice | `sqlc.slice('ids')` |
| Nest a table in a join | `sqlc.embed(t)` |
| NULL → pointer | `emit_pointers_for_null_types: true` |
| Type remap | `overrides: [{ db_type, go_type }]` |

---

## 14. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] SQLC-CFG-01 — `sqlc.yaml` is `version: "2"`
- [ ] SQLC-CMP-01 — `sqlc compile` clean
- [ ] SQLC-VET-01 — `sqlc vet` clean (db-prepare passes if managed)
- [ ] SQLC-GEN-01 — `sqlc diff` shows no drift; generated code committed
- [ ] SQLC-GEN-02 — no hand-edits to generated output
- [ ] SQLC-ANN-01 — every query annotated `-- name: X :cmd`
- [ ] SQLC-TYP-01 — nullable columns map to NULL-safe types, no panics
- [ ] SQLC-SEC-01 — parameterized only, no secret-leaking `SELECT *` (see `secure-coding.md`)
- [ ] SQLC-OUT-01 — output builds in the target language
- [ ] SQLC-TST-01 — queries tested against a real engine (see `tdd.md`)
- [ ] Agent ran every §3 command and documented any fixes

---
**End of sqlc Guidelines**

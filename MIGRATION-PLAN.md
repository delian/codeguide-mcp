# Guide Migration Plan — v2 (reference-based, token-lean)

Migrate every guide to the modernized [`guides/TEMPLATE.md`](guides/TEMPLATE.md) format defined in [`guides/CONVENTIONS.md`](guides/CONVENTIONS.md). Goal: **modern, strict, extensive, minimal tokens, zero duplicated content** — anything a canonical guide already owns becomes a `guides://` reference instead of copied prose.

Reference result: [`guides/python.md`](guides/python.md) went 3277 → 289 lines (**~85% fewer tokens**) with no loss of Python-specific value.

---

## Per-guide migration procedure

Apply to each guide. One guide per task; verify before moving on.

1. **Read** the current guide and list its sections.
2. **Classify** every section as either:
   - **Cross-cutting** (owned by a canonical guide per CONVENTIONS §1) → delete the prose, replace with a `📎 REQUIRED`/`RECOMMENDED`/`SEE ALSO` reference + a one-sentence technology binding (balanced rule).
   - **Technology-specific** → keep, but compress: remove repetition, drop project/domain-specific example bloat, keep idioms/footguns/toolchain unique to this tech.
3. **Add YAML frontmatter** (after the `# Title` + one-line brief — never at the very top, or `extract_brief` breaks): `name, title, version: 2.0, last_reviewed, kind, tools, requires, recommends, provides`.
4. **Build the requirements table** — ID'd (`<PFX>-<TOPIC>-<NN>`), RFC-2119 keywords, a verify command and binary gate per row. Rows binding a shared rule cite the owner.
5. **Generate the deployment checklist** from those IDs (no new requirements).
6. **Modernize**: current tool versions in `tools:`, fix stale/incorrect guidance (e.g. don't claim a linter type-checks), current idioms.
7. **Verify**:
   - `extract_brief` still returns the intended brief (title line 1, brief line 2, then `---`).
   - Every `guides://x.md` reference target exists.
   - No duplicated cross-cutting prose remains.
8. **Replace** the original; record the before/after line count in the row below.

### Default references by `kind`

| kind | typical `requires` | typical `recommends` |
|------|--------------------|----------------------|
| language | `tdd`, `hexagonal`, `secure-coding`, `error-handling` | `logging`, `observability`, `comments`, `semver`, `performance`, `pre-commit`, `ci-cd` |
| framework | `tdd`, `secure-coding`, language guide | `accessibility`, `e2e-testing`, `ui`, `observability`, `performance` |
| datastore | `secure-coding`, `error-handling`, `observability` | `sql` (relational), `performance`, `docker-compose`, `env-config` |
| infra | `secure-coding`, `observability` | `ci-cd`, `git`, `env-config`, `logging` |
| tooling | (often none) | `ci-cd`, `pre-commit`, `secure-coding` |
| cross-cutting | (it IS an owner — few/none) | related owners only |

> Cross-cutting "owner" guides change the least: they keep their authoritative content but still adopt frontmatter + the requirements-table format, and must stop duplicating *each other* (e.g. `cleanarch.md` references `hexagonal.md` for shared ideas rather than restating them).

---

## Suggested order

Owners first (so references resolve to already-modernized targets), then consumers.

1. **Wave 0 — canonical owners** (establish the reference targets)
2. **Wave 1 — languages**
3. **Wave 2 — frameworks / frontend**
4. **Wave 3 — datastores**
5. **Wave 4 — infra / devops / cloud / tooling**

---

## Checklist (122 guides)

### Wave 0 — Cross-cutting owners
- [x] tdd — 1378 → 305
- [x] hexagonal — 1675 → 307
- [x] cleanarch — 2334 → 280
- [x] architectures — 1979 → 331
- [x] microservices — 2331 → 269
- [x] designpatterns — 3687 → 520
- [x] secure-coding — 1462 → 376
- [x] error-handling — 1310 → 222
- [x] logging — 1449 → 233
- [x] observability — 1603 → 334
- [x] comments — 1200 → 326
- [x] markdown — 4018 → 300
- [x] adr — 716 → 245
- [x] env-config — 811 → 248
- [x] semver — 661 → 248
- [x] code-review — 787 → 295
- [x] ci-cd — 1062 → 222
- [x] git — 1986 → 356
- [x] feature-flags — 1033 → 197
- [x] performance — 1430 → 276
- [x] parallelism — 1442 → 261
- [x] todo — 1289 → 264
- [x] pre-commit — 1507 → 262
- [x] accessibility — 838 → 307
- [x] e2e-testing — 1634 → 344
- [x] coding-ai — 1295 → 215
- [x] agents-md — 1391 → 225
- [x] mutmut — 231 → 198
- [x] devops — 1713 → 262
- [x] mlops — 2065 → 495
- [x] oauth — 1078 → 277
- [x] rest — 2462 → 330
- [x] graphql — 1438 → 325
- [x] grpc — 1287 → 367
- [x] openapi — 1276 → 464
- [x] websocket — 1367 → 329
- [x] zod — 1637 → 301
- [x] ui — 1909 → 232

### Wave 1 — Languages
- [x] c — 2032 → 379
- [x] cpp — 3720 → 352
- [x] csharp — 1254 → 390
- [x] go — 2898 → 313
- [x] rust — 2622 → 368
- [x] java — 2661 → 297
- [x] kotlin — 1859 → 339
- [x] scala — 2512 → 406
- [x] swift — 1199 → 377
- [x] ruby — 1319 → 310
- [x] php — 3212 → 325
- [x] javascript — 1887 → 330
- [x] typescript — 2334 → 384
- [x] haskell — 3800 → 292
- [x] elixir — 1592 → 391
- [x] lua — 1635 → 334
- [x] zig — 2362 → 367
- [x] verilog — 1369 → 290
- [x] bash — 1801 → 312
- [x] fish — 1592 → 253
- [x] zsh — 1996 → 268
- [x] deno — 2199 → 301
- [x] nodejs — 3301 → 331
- [x] python — 3277 → 289 lines (~85%)

### Wave 2 — Frameworks / frontend
- [x] angular — 5019 → 367
- [x] reactjs — 3892 → 400
- [x] react-native — 1253 → 297
- [x] nextjs — 2195 → 449
- [x] svelte — 3122 → 411
- [x] flutter — 2102 → 399
- [x] android — 1175 → 315
- [x] ios — 1256 → 347
- [x] vite — 2887 → 318
- [x] css — 2301 → 355
- [x] html — 1543 → 392
- [x] material — 1354 → 375
- [x] chrome-extension — 1883 → 341

### Wave 3 — Datastores
- [x] postgresql — 2275 → 310
- [x] mysql-mariadb — 3879 → 300
- [x] sqlite — 2918 → 330
- [x] sql — 3613 → 469
- [x] sqlc — 2171 → 329
- [x] sqlalchemy-alembic — 2661 → 445
- [x] redis — 2722 → 331
- [x] memcached — 2672 → 270
- [x] mongodb — 2235 → 438
- [x] cassandra — 2388 → 272
- [x] scylladb — 3383 → 222
- [x] cockroachdb — 3855 → 328
- [x] couchbase — 3312 → 314
- [x] couchdb — 3631 → 330
- [x] neo4j — 3458 → 375
- [x] rethinkdb — 2363 → 321
- [x] influxdb — 3435 → 337
- [x] timescaledb — 3839 → 398
- [x] duckdb — 3855 → 287
- [x] berkeleydb — 2473 → 318
- [x] leveldb — 2762 → 266
- [x] lmdb — 2728 → 276
- [x] rocksdb — 2732 → 344
- [x] chroma-vectordb — 2361 → 298
- [x] elasticsearch-opensearch — 1356 → 340
- [x] libsql-turso — 2881 → 283
- [x] kafka — 2776 → 414

### Wave 4 — Infra / devops / cloud / tooling / ML
- [x] aws — 1531 → 310
- [x] azure — 1595 → 399
- [x] gcp — 1592 → 261
- [x] docker-compose — 3316 → 368
- [x] dockerfile — 1840 → 340
- [x] kubernetes — 2982 → 465
- [x] istio — 2585 → 413
- [x] terraform — 2380 → 385
- [x] jenkins — 2506 → 347
- [x] github — 3890 → 380
- [x] gitlab — 3023 → 361
- [x] azuredevops — 3066 → 371
- [x] nix-flake — 3381 → 394
- [x] conan — 1249 → 314
- [x] cmake — 2337 → 345
- [x] make — 2770 → 315
- [x] uv — 1445 → 345
- [x] poetry — 2367 → 266
- [x] pytorch — 1763 → 459
- [x] cuda — 2677 → 316
- [x] python-z3 — 2189 → 389

---

### Recovered framework guides (from deleted language-guide content)
- [x] rails — NEW from ruby.md §8 (210 recovered) → 402
- [x] phoenix — NEW from elixir.md §8–9 (349 recovered) → 367
- [x] fastify — NEW from nodejs.md §5 (168 recovered) → 352
- [x] mongoose — NEW from mongodb.md §15 (118 recovered) → 359

## Progress

| Metric | Value |
|--------|-------|
| Migrated | 127 / 127 |
| Total lines saved | 231,804 |

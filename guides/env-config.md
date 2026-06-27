# Configuration & Environment Guidelines
The canonical owner of application configuration: 12-factor config, env vars, layering & precedence, per-environment separation, fail-fast validation, no hardcoding. Language-agnostic; tools include dotenv, Viper, Dynaconf, node-config, python-decouple, envalid, zod/pydantic schemas.

---
name: env-config
title: Configuration & Environment Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [dotenv, viper, dynaconf, node-config, python-decouple, envalid, zod, pydantic-settings]
requires: []
recommends:
  - secure-coding
  - feature-flags
provides:
  - 12-factor-config
  - env-vars
  - config-validation
  - env-separation
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this guide is the **canonical owner** of configuration & environment management. Other guides reference it instead of restating config rules. It references — never restates — secret handling, runtime flags, and deployment-time injection.

---

## 0. Prerequisites & References

This guide owns *config policy*. It defers the adjacent concerns below to their owners and binds to them where they touch configuration.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`secure-coding.md`](guides://secure-coding.md) — **owns secret handling depth**: storage, rotation, encryption, leak scanning, vaults. This guide owns *where secrets sit in the config layering* and *that they are never hardcoded*; the *how* of protecting them is `secure-coding.md`.
> - [`feature-flags.md`](guides://feature-flags.md) — **owns runtime flags**: targeting, rollout, kill-switches. Flags are the highest-precedence config layer (§2 `CFG-LAYER-*`) but their lifecycle lives there, not here.

> 📎 **SEE ALSO (deployment-time config injection — fetch for the relevant platform):**
> - [`docker-compose.md`](guides://docker-compose.md) · [`dockerfile.md`](guides://dockerfile.md) — `env_file`, `environment:`, build vs runtime config.
> - [`kubernetes.md`](guides://kubernetes.md) — ConfigMaps, Secrets, `envFrom`, projected volumes.
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline env/secret injection per environment.
> - [`aws.md`](guides://aws.md) · [`azure.md`](guides://azure.md) · [`gcp.md`](guides://gcp.md) — managed parameter/secret stores (SSM, Key Vault, Secret Manager).
> - [`logging.md`](guides://logging.md) — redaction of config values in logs.

> Language bindings live in the language guides and reference back here: e.g. [`python.md`](guides://python.md) (Dynaconf), [`nodejs.md`](guides://nodejs.md)/[`typescript.md`](guides://typescript.md) (dotenv + zod/envalid), [`go.md`](guides://go.md) (Viper).

---

## 1. Core Philosophies: CONFIG-FIRST

Configuration is **strict separation of config from code** ([12-factor](https://12factor.net/config)): anything that varies between deploys is config, everything else is code.

- **C**onfig-from-environment: read config from the environment (env vars / mounted files), never compile it in. One build artifact, every environment.
- **O**verrideable: a deterministic precedence chain lets any layer override the one below it (§2 `CFG-LAYER-*`).
- **N**ever hardcoded: no literals, magic numbers, hostnames, or secrets in source (`CFG-NOHARD-01`). Secrets specifically are handled per [`secure-coding.md`](guides://secure-coding.md).
- **F**ail-fast: the full config is parsed, typed, and validated at startup; an invalid or missing required value aborts boot (`CFG-VALID-*`).
- **I**mmutable per deploy: config is resolved once at boot into a typed, read-only object; runtime mutation is forbidden. Changing config means a new deploy (or a flag flip, see [`feature-flags.md`](guides://feature-flags.md)).
- **G**uarded: config values are typed and schema-validated; secret-bearing values are redacted in logs and error output.

> **Config vs flag vs secret.** *Config* = environment-varying input resolved at boot (this guide). *Feature flag* = runtime-toggleable behavior switch ([`feature-flags.md`](guides://feature-flags.md)). *Secret* = a credential whose protection is governed by [`secure-coding.md`](guides://secure-coding.md). A secret is delivered *through* the config layering but its handling is not owned here.

**Verified Code**: agent-generated config handling MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CFG-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a referenced concern cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CFG-NOHARD-01 | No environment-varying value (host, port, URL, path, credential, magic number) MUST be hardcoded in source | grep/lint for literals; review | no literals |
| CFG-NOHARD-02 | No secret MUST appear in source, fixtures, or VCS history (delegates depth to `secure-coding.md`) | secret scanner (e.g. gitleaks/trufflehog) | 0 findings |
| CFG-12F-01 | One build artifact MUST run unchanged across all environments; only the supplied config differs | review build vs deploy | no per-env builds |
| CFG-LAYER-01 | Config MUST resolve through a single documented precedence chain (defaults → file → env-file → env vars → CLI args → runtime flags) | review config loader | one chain |
| CFG-LAYER-02 | A higher layer MUST override a lower one for the same key, deterministically | unit test on precedence | override holds |
| CFG-ENV-01 | Env var names MUST be `UPPER_SNAKE_CASE`, app-prefixed (`<APP>_<GROUP>_<NAME>`) | lint/review | conforms |
| CFG-ENV-02 | All consumed env vars MUST be declared in a committed `.env.example` (or schema) with type + default + required flag | diff schema vs `.env.example` | in sync |
| CFG-VALID-01 | Full config MUST be parsed against a typed schema at startup; on failure the process MUST exit non-zero before serving | run with bad config | exits ≠ 0, no serve |
| CFG-VALID-02 | Every value MUST be coerced to its target type at the boundary (no raw `string` env reads downstream) | type check / review | typed config object |
| CFG-VALID-03 | Production-only invariants (TLS on, no `localhost`, secret length, no placeholder values) MUST be asserted when env=production | startup validator test | asserts present |
| CFG-SEP-01 | Each environment (dev/test/staging/prod) MUST have its own resolved config; test config MUST NOT reach a production resource | review per-env files | isolated |
| CFG-SEP-02 | Dangerous dev conveniences (schema auto-sync, verbose logging, mock externals, long-lived tokens) MUST be off in production | review prod config | disabled |
| CFG-IMMUT-01 | Resolved config MUST be a read-only object; code MUST NOT mutate `process.env`/`os.environ` after boot | review / freeze | immutable |
| CFG-SECRET-01 | Secrets MUST be injected from a secret store/manager at deploy time, never from a committed file (see `secure-coding.md`) | review deploy | from store |
| CFG-LOG-01 | Secret-bearing config MUST be redacted in logs, error messages, and config dumps (see `logging.md`) | log review / redaction test | redacted |
| CFG-DOC-01 | Each config key MUST be documented (purpose, type, default, required, per-env value) — generated from the schema where possible | docs build / review | documented |

> **Forbidden**: hardcoding any environment-varying value or secret; reading raw `process.env`/`os.environ` deep inside business logic instead of a validated config object; booting with missing/invalid required config; per-environment build artifacts; mutating config at runtime; committing real secrets or a populated `.env`.

---

## 3. Configuration Layering & Precedence (OWNED)

Resolve every key through one deterministic chain, **lowest to highest**:

```
1. In-code defaults          # safe fallbacks; the only literals allowed (CFG-NOHARD ok here)
2. Base config file          # config/default.{toml,yaml,json}  — non-secret, committed
3. Env-specific file         # config/{development,test,staging,production}.* — committed, no secrets
4. Local env file            # .env / .env.local — gitignored, developer machine only
5. Process environment vars  # <APP>_* — what containers/CI/K8s actually inject
6. CLI arguments / flags      # explicit operator overrides
7. Runtime feature flags      # highest — see feature-flags.md (this guide does not own their lifecycle)
```

A key set at layer N is overridden by any value present at N+1 (`CFG-LAYER-02`). Document the chain once; do not let different subsystems invent their own ordering.

```
project/
├── config/
│   ├── default.toml          # base, committed, NO secrets
│   ├── development.toml       # per-env overrides (CFG-SEP-01)
│   ├── test.toml
│   ├── staging.toml
│   ├── production.toml        # NO secrets — references store keys only
│   └── schema.*               # the validation schema (CFG-VALID-01)
├── .env.example               # committed template: every var, typed, with required flag (CFG-ENV-02)
├── .env                        # gitignored local overrides (CFG-NOHARD-02)
└── .env.{development,test}      # committed non-secret per-env defaults
```

`.gitignore` MUST exclude populated env/secret files:

```gitignore
.env
.env.local
.env.*.local
*.pem
*.key
secrets/
```

---

## 4. Environment Variables (OWNED)

### A. Naming (`CFG-ENV-01`)

`UPPER_SNAKE_CASE`, app-prefixed `<APP>_<GROUP>_<NAME>` to avoid collisions with the OS/other processes:

```bash
MYAPP_DB_HOST=db.internal
MYAPP_DB_PORT=5432
MYAPP_DB_POOL_SIZE=10
MYAPP_DB_SSL_ENABLED=true
MYAPP_REDIS_URL=redis://cache:6379
MYAPP_HTTP_PORT=3000
MYAPP_LOG_LEVEL=info
MYAPP_ENV=production
```

Prefer a single `*_URL` (DSN) over scattered host/port/user/pass parts when a library accepts it — fewer keys, atomic, harder to half-configure. Secret-bearing keys (passwords, API keys, tokens) follow `secure-coding.md` for storage but still flow through this layering.

### B. Typed, declared, defaulted (`CFG-ENV-02`, `CFG-VALID-02`)

Every consumed variable is declared once in a schema with type, default, and required flag, and mirrored in `.env.example`. Strings from the environment are **coerced at the boundary** — no `string` env values leak downstream. The schema is language-specific; the *contract* (typed, validated, declared) is owned here. Examples below are illustrative of the contract, not a mandated stack:

```typescript
// schema as the single source of truth (zod / envalid / pydantic-settings / Viper+struct ...)
const Env = z.object({
  MYAPP_ENV:        z.enum(['development','test','staging','production']),
  MYAPP_HTTP_PORT:  z.coerce.number().int().min(1).max(65535).default(3000),
  MYAPP_LOG_LEVEL:  z.enum(['debug','info','warn','error']).default('info'),
  MYAPP_DB_URL:     z.string().url(),
  MYAPP_DB_SSL:     z.coerce.boolean().default(false),
  MYAPP_DB_POOL:    z.coerce.number().int().min(1).max(100).default(10),
  MYAPP_ALLOWED_ORIGINS: z.string().transform(s => s.split(',').map(x => x.trim())),
  MYAPP_JWT_SECRET: z.string().min(32),   // value from secret store; schema enforces shape
});
```

Boolean coercion MUST be explicit (`"true"`/`"1"`/`"yes"` → `true`); never rely on truthiness of the raw string (`"false"` is a non-empty, truthy string).

---

## 5. Fail-Fast Validation (OWNED)

Parse and validate the **entire** config at startup, before binding ports or accepting traffic (`CFG-VALID-01`). On any failure: print actionable errors and exit non-zero — never silently fall back.

```typescript
const parsed = Env.safeParse(process.env);
if (!parsed.success) {
  console.error('Invalid configuration:\n' + parsed.error.toString());
  process.exit(1);                       // CFG-VALID-01: abort before serving
}
export const config = Object.freeze(toConfig(parsed.data));  // CFG-IMMUT-01
```

### Production invariants (`CFG-VALID-03`, `CFG-SEP-02`)

When `env === 'production'`, assert hardening that lower environments may relax:

- secrets present and of required length; **no placeholder** values (`change-me`, `xxx`, `TODO`, `your-...`);
- TLS/SSL enabled for datastores and caches;
- no `localhost`/`127.0.0.1` in external endpoints;
- dev conveniences off: schema auto-sync, query logging, mock externals, debug log level, long-lived tokens.

A failed invariant aborts boot, same as a schema error.

---

## 6. Per-Environment Separation (OWNED)

One artifact, distinct resolved config per environment (`CFG-12F-01`, `CFG-SEP-01`). Capture intended differences in a committed matrix so drift is reviewable:

| Key | development | test | staging | production |
|-----|-------------|------|---------|------------|
| `MYAPP_ENV` | development | test | staging | production |
| `MYAPP_LOG_LEVEL` | debug | error | info | info |
| `MYAPP_DB_SSL` | false | false | true | true |
| `MYAPP_DB_POOL` | 5 | 2 | 10 | 20 |
| schema auto-sync | on | on | off | off |
| token TTL | 7d | 1h | 1h | 15m |
| externals | real/local | mocked | real | real |

Rules: test config MUST be isolated (separate DB/index, never a prod resource); production overrides default to *secure* (SSL on, short TTLs, no auto-sync); committed per-env files hold **non-secret** values only — secrets come from the store at deploy time (`CFG-SECRET-01`).

---

## 7. Boundaries with referenced owners (bindings only)

- **Secrets** — this guide governs that secrets are never hardcoded (`CFG-NOHARD-02`), flow through the layering, are injected from a store at deploy (`CFG-SECRET-01`), and are redacted in output (`CFG-LOG-01`). All depth — rotation, encryption-at-rest, vault setup, scanning policy — is [`secure-coding.md`](guides://secure-coding.md). Do not restate it here.
- **Deployment-time injection** — how env/Secrets/ConfigMaps reach the process is platform-owned: [`docker-compose.md`](guides://docker-compose.md) (`env_file`/`environment`), [`kubernetes.md`](guides://kubernetes.md) (`envFrom`, `secretKeyRef`), [`ci-cd.md`](guides://ci-cd.md) (per-environment pipeline secrets/vars), and the cloud guides ([`aws.md`](guides://aws.md)/[`azure.md`](guides://azure.md)/[`gcp.md`](guides://gcp.md)) for managed stores. This guide only requires that the *result* is a validated, typed, immutable config object.
- **Runtime flags** — the top precedence layer is owned by [`feature-flags.md`](guides://feature-flags.md). Treat a flag as the highest-priority override of a config key; its targeting/rollout/lifecycle is not config and lives there.
- **Log redaction** — the *mechanism* of redaction is [`logging.md`](guides://logging.md); this guide only mandates that secret-bearing config is redacted (`CFG-LOG-01`).

---

## 8. Configuration Documentation (`CFG-DOC-01`)

Generate config docs from the schema so they cannot drift: for each key emit purpose, type, default, required flag, and per-environment value. Keep generation in the build (and CI) so a new/renamed key fails the docs check until documented. `.env.example` is the minimal machine-checkable form of this requirement (`CFG-ENV-02`).

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] CFG-NOHARD-01/02 — no hardcoded env-varying values or secrets; secret scan clean
- [ ] CFG-12F-01 — single artifact across environments
- [ ] CFG-LAYER-01/02 — one documented precedence chain; overrides deterministic
- [ ] CFG-ENV-01/02 — names conform; every var declared in `.env.example`/schema
- [ ] CFG-VALID-01/02/03 — startup validation aborts on bad config; values typed; prod invariants asserted
- [ ] CFG-SEP-01/02 — per-env config isolated; dev conveniences off in prod
- [ ] CFG-IMMUT-01 — resolved config read-only; no runtime env mutation
- [ ] CFG-SECRET-01 — secrets injected from a store at deploy (see `secure-coding.md`)
- [ ] CFG-LOG-01 — secret-bearing config redacted in logs/errors (see `logging.md`)
- [ ] CFG-DOC-01 — every key documented (generated from schema)

---
**End of Configuration & Environment Guidelines**

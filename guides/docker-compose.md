# Docker Compose Guidelines
Mandatory standards for Docker Compose stacks: modern Compose Spec, isolated networks, healthchecked dependencies, secrets, reproducible dev/prod parity. Docker Compose v2, Compose Specification, Docker Engine 27+.

---
name: docker-compose
title: Docker Compose Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [docker-compose@v2, compose-spec, docker-engine@27, trivy, dclint]
requires:
  - secure-coding
recommends:
  - dockerfile
  - kubernetes
  - env-config
  - ci-cd
  - observability
provides:
  - compose-spec
  - compose-profiles
  - local-dev-stacks
  - compose-healthchecks
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Docker Compose — multi-container composition, networking, healthchecked startup, and local dev stacks.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Compose files. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — container security, supply chain, secrets, CVE policy. *(Compose binding: per-service runtime hardening in §6, image/CVE scanning with `trivy`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`dockerfile.md`](guides://dockerfile.md) — how the images are **built** (base image, multi-stage, non-root `USER`, `HEALTHCHECK`). Compose **runs** images; it does not build them well. Do not restate Dockerfile rules here.
> - [`kubernetes.md`](guides://kubernetes.md) — production orchestration. Compose is for local/single-host; see §10 for when to graduate.
> - [`env-config.md`](guides://env-config.md) — config layering, env separation, secret sourcing. *(Compose binding: `.env`, `env_file`, `environment`, `secrets`.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline-driven build/scan/up. *(Compose binding: `docker-compose.ci.yml`.)*
> - [`observability.md`](guides://observability.md) — metrics/tracing/log policy for the monitoring services a stack runs.

> 📎 **SEE ALSO:** [`microservices.md`](guides://microservices.md) · [`make.md`](guides://make.md)

---

## 1. Core Philosophies: COMPOSE-FIRST

Compose-specific principles only. Security and config policy come from §0.

- **C**ompose Spec, not legacy: the **top-level `version:` key is obsolete and MUST be omitted** (Compose v2 ignores it and warns). The schema is the Compose Specification.
- **O**ne command, whole stack: a developer clones, runs one `docker compose up`, and gets a working environment. Reproducibility is the product.
- **M**inimal blast radius: every service joins only the networks it needs; backend networks are `internal: true`. Default-deny connectivity.
- **P**arity by override: one base file describes the topology; thin override files specialize dev vs prod. No copy-pasted near-duplicate stacks.
- **O**rdered by readiness, not by luck: dependents wait on `condition: service_healthy`, never on `sleep`.
- **S**tateless config: images are pinned and immutable; all variation comes from env/secrets injected at run time (see `env-config.md`).
- **E**phemeral, not production: Compose targets local dev and single-host; multi-node production graduates to Kubernetes (§10).

**Verified Code**: Agent-generated Compose files MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DC-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DC-FMT-01 | File MUST parse & merge cleanly | `docker compose config --quiet` | exit 0 |
| DC-LINT-01 | File MUST lint clean (2-space YAML, no anti-patterns) | `dclint docker-compose.yml` | exit 0 |
| DC-VER-01 | Top-level `version:` key MUST NOT be present | `! grep -E '^version:' docker-compose*.yml` | no match |
| DC-IMG-01 | Images MUST be pinned (tag, digest in prod); never `:latest` | `! docker compose config \| grep -E 'image:.*:latest\|image:[^@:]+$'` | no match |
| DC-SEC-01 | 0 high/critical misconfigs (see `secure-coding.md`) | `trivy config .` | 0 high/critical |
| DC-SEC-02 | No hardcoded secrets in compose/.env (see `secure-coding.md`) | secret scan / review | 0 secrets |
| DC-SEC-03 | Each service hardened: non-root, `read_only`, `no-new-privileges`, `cap_drop: [ALL]`, `privileged` absent (see `secure-coding.md`) | review / `docker compose config` | all present |
| DC-NET-01 | Backend/datastore networks MUST be `internal: true`; no `network_mode: host` | `docker compose config` | isolated |
| DC-HLTH-01 | Every depended-upon service MUST define a `healthcheck` | `docker compose config` | present |
| DC-DEP-01 | `depends_on` MUST use `condition:` (not the short list form) | `docker compose config` | conditions set |
| DC-RES-01 | Every service MUST set memory (and pid) limits | `docker compose config` | limits present |
| DC-CFG-01 | Config via env/secrets, not baked in (see `env-config.md`) | review / grep | no literals |
| DC-TST-01 | Stack MUST come up healthy (smoke test) | `docker compose up -d --wait` | exit 0 |

> **Forbidden**: a top-level `version:` key; `:latest` or unpinned images; `privileged: true`; `network_mode: host` for app services; database `ports:` published to the host in prod; plaintext passwords in `environment:`; `depends_on` short-form when the dependency needs to be *ready* (only *started*).

---

## 3. Verification Protocol

Run, in order, before presenting a stack. Fix → re-run until every gate is green.

```bash
docker compose config --quiet            # DC-FMT-01  (validate + merge)
dclint docker-compose.yml                # DC-LINT-01 (zavoloklom/dclint)
trivy config .                           # DC-SEC-01  (misconfig scan)
docker compose config | grep -E 'image:' # DC-IMG-01  (eyeball pins / digests)
docker compose up -d --wait              # DC-TST-01  (all services reach healthy)
docker compose down -v                   # clean up
```

`--wait` blocks until every service with a healthcheck is healthy and exits non-zero if any fails — the canonical stack smoke test. The *why* behind security/config gates lives in their §0 owners.

---

## 4. File Structure & Top-Level Keys

A Compose file is a map of six top-level keys. There is **no `version:` key** in the modern spec.

```yaml
name: myapp                 # project name (namespaces containers/networks/volumes)
services:                   # the containers
networks:                   # custom networks (never rely on the default bridge)
volumes:                    # named persistent volumes
configs:                    # non-secret config files mounted into containers
secrets:                    # sensitive files mounted at /run/secrets/<name>
```

Recommended **service key order** (readability, not enforced): `image`/`build` → `depends_on` → `env_file`/`environment` → `ports`/`expose` → `volumes` → `networks` → `healthcheck` → `restart` → `deploy` (resources) → security keys (`user`, `read_only`, `cap_drop`, `security_opt`) → `labels` → `command`.

- **File names**: `docker-compose.yml` (base) and `docker-compose.override.yml` (auto-merged for local dev). Named overrides (`docker-compose.prod.yml`) are opt-in via `-f` (§7).
- **Split large stacks** with the top-level `include:` key — each included file is a complete Compose file merged into the project. Use it to separate concerns (e.g. `infra.yml`, `app.yml`), **not** to impose an architecture: layering/ports-and-adapters policy is owned by the architecture guides, not by Compose.

---

## 5. Networking, Healthchecks & Startup Order

The two features that make Compose more than "run N containers": a private DNS-resolved network and readiness-gated startup.

### A. Networks — service discovery & isolation
Every service on the same network reaches the others by **service name** as hostname (`postgres:5432`). Put services on the minimum set of networks; isolate datastores.

```yaml
services:
  frontend: { networks: [public, backend] }   # talks to internet + api
  api:      { networks: [backend, data] }      # talks to frontend-facing + db
  db:       { networks: [data] }               # reachable only from api

networks:
  public:   { driver: bridge }
  backend:  { driver: bridge }
  data:     { driver: bridge, internal: true } # DC-NET-01: no egress, no host exposure
```
- `expose:` advertises a port to the network only; `ports:` publishes to the host. Datastores use `expose` (or nothing) — never publish `5432` to the host in prod (DC-NET-01).
- `internal: true` removes the gateway: containers on it have **no internet access** and cannot be reached from the host.

### B. Healthchecks — define readiness
A service is "healthy" only when its `healthcheck` passes. Prefer an in-image `HEALTHCHECK` (owned by [`dockerfile.md`](guides://dockerfile.md)); override per-deployment in Compose when the image lacks one.

```yaml
services:
  db:
    image: postgres:17.2-alpine3.21
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s        # grace window before failures count
```
Common probes: Postgres `pg_isready`; Redis `redis-cli ping`; HTTP `curl -f http://localhost:PORT/health` (or `wget --spider`); MongoDB `mongosh --eval "db.adminCommand('ping')"`; RabbitMQ `rabbitmq-diagnostics -q ping`.

### C. depends_on — readiness-gated startup
Short-form `depends_on: [db]` waits only for the container to **start**, not to be **ready** — the #1 cause of "connection refused" on boot. Use the long form with a condition (DC-DEP-01):

```yaml
services:
  api:
    depends_on:
      db:        { condition: service_healthy }              # waits for healthcheck
      migrate:   { condition: service_completed_successfully } # one-shot init job
      cache:     { condition: service_started }              # ready-ness not needed
```
The app SHOULD still implement connection retry — `depends_on` orders startup, it is not a substitute for resilient clients (see `error-handling.md`).

---

## 6. Configuration, Secrets & Runtime Hardening

### A. Config & secrets injection
Config layering and env separation are owned by [`env-config.md`](guides://env-config.md); secret policy by [`secure-coding.md`](guides://secure-coding.md). Compose binding:

- **`.env`** (next to the compose file) interpolates `${VAR}` **in the file itself** — for image tags, ports, paths. Not injected into containers.
- **`env_file:` / `environment:`** inject variables **into the container**. Use `${VAR:-default}` for safe defaults.
- **`secrets:`** mount sensitive files at `/run/secrets/<name>` — never pass passwords via `environment` (DC-SEC-02). Most official images accept a `*_FILE` variant.

```yaml
services:
  api:
    image: registry.example.com/api:1.4.2
    env_file: [.env]                       # non-secret, in-container config
    environment:
      LOG_LEVEL: ${LOG_LEVEL:-info}
      DB_PASSWORD_FILE: /run/secrets/db_password   # *_FILE pattern, not the value
    secrets: [db_password]

secrets:
  db_password:
    file: ./secrets/db_password.txt        # git-ignored; or `environment: VAULT_...`
  api_key:
    external: true                          # provided by the orchestrator / swarm
```
Commit `.env.example`, never real `.env`/`secrets/*`. For production secret stores (Vault, cloud SM), source values per [`secure-coding.md`](guides://secure-coding.md).

### B. Per-service runtime hardening (DC-SEC-03)
Image hardening (distroless/scratch, non-root `USER`) is owned by [`dockerfile.md`](guides://dockerfile.md); **runtime** hardening is Compose's job. Apply this baseline to every service:

```yaml
services:
  api:
    user: "10001:10001"          # non-root at runtime
    read_only: true              # immutable root filesystem
    cap_drop: [ALL]              # drop every Linux capability...
    # cap_add: [NET_BIND_SERVICE] # ...re-add only if binding port <1024 (justify it)
    security_opt:
      - no-new-privileges:true   # block setuid escalation
    tmpfs:
      - /tmp:size=64M,mode=1777  # the only writable path it needs
    deploy:
      resources:
        limits:   { cpus: "1.0", memory: 512M, pids: 200 }  # DC-RES-01: cap blast radius
        reservations: { memory: 128M }
```
`privileged: true`, `network_mode: host`, `pid: host`, and mounting `/var/run/docker.sock` are forbidden for application services (DC-SEC-03; see `secure-coding.md`). Use a YAML anchor to apply the baseline DRY-ly:

```yaml
x-hardened: &hardened
  read_only: true
  cap_drop: [ALL]
  security_opt: [no-new-privileges:true]
services:
  api: { <<: *hardened, image: ... }
```

---

## 7. Override Files & Dev vs Prod Parity

One base topology; thin overrides specialize per environment. Compose **deep-merges** files in `-f` order (later wins; lists for `ports`/`environment` are appended).

- `docker-compose.yml` — the canonical, production-shaped topology.
- `docker-compose.override.yml` — **auto-loaded** by `docker compose up`; the local-dev delta (builds, bind mounts, published ports, debug env).
- `docker-compose.prod.yml` — opt-in: digest-pinned images, replicas, `restart: always`, no host ports.

```bash
docker compose up -d                                          # base + override.yml (dev)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d   # explicit prod
```

```yaml
# docker-compose.override.yml — DEV ONLY (live code, debug ports, build from source)
services:
  api:
    build: { context: ./api, target: dev }   # build locally instead of pulling
    ports: ["8000:8000", "9229:9229"]         # publish + debugger
    environment: { LOG_LEVEL: debug }
    develop:
      watch:                                  # §8 hot-reload
        - { action: sync, path: ./api/src, target: /app/src }
        - { action: rebuild, path: ./api/package.json }
```
```yaml
# docker-compose.prod.yml — opt-in production deltas
services:
  api:
    image: registry.example.com/api@sha256:abc123...   # immutable digest (DC-IMG-01)
    restart: always
    deploy: { replicas: 3 }
```

For multi-environment overrides, prefer this base+override pattern over divergent full files — it keeps dev and prod in parity (one source of truth for the topology).

---

## 8. Watch Mode (`docker compose watch`)

`develop.watch` replaces ad-hoc bind-mount-plus-polling for inner-loop dev. `docker compose watch` monitors host paths and reacts without a manual rebuild:

- `action: sync` — copy changed files into the running container (for interpreted/HMR stacks). Fastest; no restart.
- `action: sync+restart` — sync, then restart the container (config files, non-HMR servers).
- `action: rebuild` — rebuild the image and recreate (dependency/lockfile changes).
- `ignore:` — skip `node_modules`, build output, etc.

```yaml
services:
  web:
    build: ./web
    develop:
      watch:
        - { action: sync, path: ./web/src, target: /app/src, ignore: [node_modules/] }
        - { action: sync+restart, path: ./web/nginx.conf, target: /etc/nginx/nginx.conf }
        - { action: rebuild, path: ./web/package.json }
```
Run with `docker compose watch`. This supersedes the old `CHOKIDAR_USEPOLLING` / anonymous-volume tricks for most stacks.

---

## 9. Profiles & Optional Services

`profiles:` keep optional services (debug tools, seeders, observability) out of the default `up` until explicitly enabled — one file, many footprints.

```yaml
services:
  app:    { image: myapp:1.4.2 }                 # no profile → always starts
  jaeger:
    image: jaegertracing/all-in-one:1.62.0
    profiles: [observability]                    # only with the profile
  seed:
    image: myapp-seeder:1.4.2
    profiles: [tools]
```
```bash
docker compose up -d                              # app only
docker compose --profile observability up -d      # app + jaeger
COMPOSE_PROFILES=tools,observability docker compose up -d
```
Run the observability profile's services per [`observability.md`](guides://observability.md); keep it off by default in dev to save resources.

---

## 10. When to Graduate to Kubernetes

Compose is excellent for local dev, CI, demos, and small single-host deployments. **Stop stretching it** and adopt [`kubernetes.md`](guides://kubernetes.md) when you need any of:

- Multi-node scheduling / horizontal scale beyond one host (Compose `deploy.replicas` runs only on the local engine or Swarm).
- Self-healing, rolling updates, and automated rollback as first-class primitives.
- Declarative autoscaling (HPA), ingress, service mesh, or per-pod secrets from a cluster store.
- Zero-downtime deploys and multi-team RBAC on shared infrastructure.

Keep the Compose file for the inner dev loop even after adopting Kubernetes; do **not** treat `docker compose up` on a single VM as a production HA strategy.

---

## 11. Quick Reference

```bash
docker compose up -d --wait          # start, block until healthy (smoke test)
docker compose watch                 # inner-loop dev with file sync/rebuild
docker compose --profile X up -d     # include profiled services
docker compose config                # render merged, interpolated config
docker compose ps / logs -f / top    # status / logs / processes
docker compose exec api sh           # shell into a running service
docker compose run --rm api <cmd>    # one-off task container
docker compose pull                  # refresh pinned images
docker compose down -v               # stop + remove volumes (DESTRUCTIVE)
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] DC-FMT-01 — `docker compose config --quiet` exit 0
- [ ] DC-LINT-01 — `dclint` clean
- [ ] DC-VER-01 — no top-level `version:` key
- [ ] DC-IMG-01 — images pinned (digest in prod), no `:latest`
- [ ] DC-SEC-01/02 — `trivy config` clean, no hardcoded secrets
- [ ] DC-SEC-03 — every service: non-root, `read_only`, `no-new-privileges`, `cap_drop: [ALL]`, no `privileged`
- [ ] DC-NET-01 — backend/data networks `internal: true`, no host networking
- [ ] DC-HLTH-01 — depended-upon services have healthchecks
- [ ] DC-DEP-01 — `depends_on` uses `condition:`
- [ ] DC-RES-01 — memory/pid limits on every service
- [ ] DC-CFG-01 — config via env/secrets, none baked in
- [ ] DC-TST-01 — `docker compose up -d --wait` brings the stack up healthy
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Docker Compose Guidelines**

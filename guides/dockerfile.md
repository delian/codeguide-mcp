# Dockerfile Guidelines
Mandatory standards for Dockerfiles and container images: minimal, hardened, reproducible, cache-optimized. Docker Engine 27.x+, BuildKit/buildx, hadolint, Trivy/Docker Scout, OCI 1.1.

---
name: dockerfile
title: Dockerfile Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [docker@27, buildkit, buildx, hadolint, trivy, docker-scout]
requires:
  - secure-coding
recommends:
  - docker-compose
  - kubernetes
  - ci-cd
  - observability
provides:
  - multistage-builds
  - image-hardening
  - layer-optimization
  - distroless
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to authoring Dockerfiles and building images.

---

## 0. Prerequisites & References

Fetch and apply these **before** authoring a Dockerfile. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — vulnerability scanning, supply-chain, secrets, CVE policy. *(Docker binding: `trivy image` / `docker scout cves` for CVEs, `trivy fs --scanners secret` for leaked secrets, non-root execution, never bake secrets into a layer.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`docker-compose.md`](guides://docker-compose.md) — local composition & runtime hardening (`read_only`, `cap_drop`, `security_opt`) that complements the image.
> - [`kubernetes.md`](guides://kubernetes.md) — runtime: securityContext, probes, resource limits that consume the image.
> - [`ci-cd.md`](guides://ci-cd.md) — build/scan/push the image in a pipeline.
> - [`observability.md`](guides://observability.md) — logging to stdout/stderr, health endpoints.

> 📎 **SEE ALSO:** [`env-config.md`](guides://env-config.md) *(build-arg vs runtime-env policy)* · [`semver.md`](guides://semver.md) *(image tag versioning)*

---

## 1. Core Philosophies: IMAGE-FIRST

Dockerfile-specific principles only. Security/CVE/secret policy comes from §0 — do not restate it.

- **Minimal**: the final image ships only the runtime artifact. Prefer `scratch` (static binaries) → distroless → `*-slim`/`*-alpine` (justified) → never full Debian/Ubuntu in production.
- **Multi-stage always**: build deps, toolchains, and SDKs live in earlier stages and never reach the runner.
- **Non-root, no shell**: every runtime stage runs as a non-root UID and, where possible, has no shell or package manager to exploit.
- **Cache-ordered**: layers are ordered least-→most-frequently-changing so a code edit never reinstalls dependencies.
- **Reproducible**: pinned base images (digest in prod), committed lockfiles, deterministic `--frozen`/`ci` installs, BuildKit.
- **Verified**: the image builds, scans clean, and runs read-only as non-root before delivery.

**Verified Code**: Agent-generated Dockerfiles MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DF-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DF-LINT-01 | Dockerfile MUST lint clean | `hadolint Dockerfile` | exit 0 (no error) |
| DF-BUILD-01 | Image MUST build with BuildKit | `docker build --pull -t app:test .` | exit 0 |
| DF-BUILD-02 | Multi-stage MUST separate build from runtime | review / count `FROM` | runner carries only artifacts |
| DF-BUILD-03 | `ENTRYPOINT`/`CMD` MUST use exec (JSON) form | `hadolint` DL3025 | no shell form |
| DF-IMG-01 | Final base MUST be minimal (scratch/distroless/slim; never full) | `docker images app:test` | meets size target (§5.A) |
| DF-PIN-01 | Base images MUST be pinned by tag; prod SHOULD pin by digest; never `:latest` | `hadolint` DL3007 / `grep FROM` | no `:latest` |
| DF-USER-01 | Runtime stage MUST run as non-root | `docker run --rm app:test id` | uid ≠ 0 |
| DF-CACHE-01 | Layers MUST be ordered least→most volatile (deps before source) | review / rebuild after src edit | deps layer CACHED |
| DF-SEC-01 | 0 HIGH/CRITICAL CVEs in final image (see `secure-coding.md`) | `trivy image --severity HIGH,CRITICAL app:test` | 0 found |
| DF-SEC-02 | No secrets in any layer/history (see `secure-coding.md`) | `trivy fs --scanners secret .` / `docker history` | 0 secrets, no `ARG` creds |
| DF-RO-01 | Image MUST run under a read-only root filesystem | `docker run --rm --read-only --tmpfs /tmp app:test` | starts clean |
| DF-IGNORE-01 | `.dockerignore` MUST exclude VCS, deps, secrets, tests | `test -f .dockerignore` + review | present & correct |
| DF-META-01 | OCI image labels MUST be set | `docker inspect -f '{{.Config.Labels}}' app:test` | source+version present |
| DF-HEALTH-01 | A healthcheck/probe MUST exist (`HEALTHCHECK` or orchestrator probe) | `docker inspect -f '{{.Config.Healthcheck}}'` / k8s probe | defined |
| DF-DEP-01 | Builds MUST use committed lockfiles + frozen installs (see `secure-coding.md`) | review `npm ci`/`--frozen`/`uv sync` | no floating installs |

> **Forbidden**: `:latest` base tags, running as root in the runner, secrets in `ARG`/`ENV`/`COPY`, shell-form entrypoints, copying the whole context before installing deps, or shipping a full Debian/Ubuntu image to production.

---

## 3. Verification Protocol

Run, in order, before presenting the Dockerfile. Fix → re-run until every gate is green.

```bash
hadolint Dockerfile                                         # DF-LINT-01, DF-BUILD-03, DF-PIN-01
DOCKER_BUILDKIT=1 docker build --pull -t app:test .         # DF-BUILD-01
docker run --rm app:test id                                 # DF-USER-01 (uid ≠ 0)
trivy image --severity HIGH,CRITICAL --exit-code 1 app:test # DF-SEC-01
trivy fs --scanners secret .                                # DF-SEC-02
docker run --rm --read-only --tmpfs /tmp app:test           # DF-RO-01
docker inspect -f '{{.Config.Labels}} {{.Config.Healthcheck}}' app:test  # DF-META-01, DF-HEALTH-01
```

The *why* behind each scan lives in [`secure-coding.md`](guides://secure-coding.md); do not re-derive it here.

---

## 4. Multi-Stage Structure

Every Dockerfile is multi-stage. Standard stage roles:

- **base** — pinned image + shared `ENV`/`WORKDIR`; nothing volatile.
- **deps** — install dependencies from lockfiles using cache mounts.
- **build** — compile/transpile into a self-contained artifact.
- **runner** — minimal final image; `COPY --from` only the artifact, then drop to a non-root user.

```dockerfile
# syntax=docker/dockerfile:1
FROM node:20.12-alpine AS base
WORKDIR /app
ENV NODE_ENV=production

FROM base AS deps
RUN --mount=type=bind,source=package.json,target=package.json \
    --mount=type=bind,source=package-lock.json,target=package-lock.json \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev          # DF-DEP-01: frozen install from lockfile

FROM gcr.io/distroless/nodejs20-debian12 AS runner   # DF-IMG-01: no shell/pkg-mgr
WORKDIR /app
COPY --link --from=deps /app/node_modules ./node_modules
COPY --link src/ ./src/
USER nonroot:nonroot           # DF-USER-01
EXPOSE 3000
CMD ["src/index.js"]           # DF-BUILD-03: exec form
```

`COPY --link` makes the copy a self-contained layer: it is cache-independent of earlier layers and lets BuildKit build/rewrite stages in parallel. Use it on every `COPY`.

---

## 5. Dockerfile Specifics

The unique value of this guide.

### A. Base image selection & size targets

Pick the smallest image that still runs your artifact. Hierarchy (most→least preferred):

| Priority | Base | Use case | Size target |
|----------|------|----------|-------------|
| 1 | `scratch` | static binaries (Go `CGO_ENABLED=0`, Rust musl, C) | 5–20 MB |
| 2 | distroless (`gcr.io/distroless/*`) | interpreted/JVM runtimes (Python, Node, Java) | 50–200 MB |
| 3 | `*-slim` / `*-alpine` | needs a package manager — requires justification | < 150 MB |
| ✗ | full Debian/Ubuntu | dev only — **never** production | rejected |

Distroless/scratch have no shell, so there is no `docker exec ... sh` and no in-image package manager — a large attack-surface reduction. Debug them with an ephemeral `docker debug` / a sidecar, not by adding a shell.

### B. Layer caching & ordering

Order instructions least-→most-volatile: pinned base → system packages → dependency manifests → dependency install → source. A source edit must not invalidate the dependency layer.

```dockerfile
COPY --link package.json package-lock.json ./   # changes rarely
RUN --mount=type=cache,target=/root/.npm npm ci # CACHED unless manifests change
COPY --link src/ ./src/                          # changes often — only this rebuilds
```

### C. BuildKit mounts

`# syntax=docker/dockerfile:1` enables BuildKit features. Use the right mount:

- **`--mount=type=cache`** — persist a package-manager cache (`/root/.npm`, `/root/.cache/pip`, `/go/pkg/mod`, `~/.cargo`) across builds without bloating the image.
- **`--mount=type=bind`** — read manifests/source during a `RUN` without a `COPY` layer.
- **`--mount=type=secret`** — the **only** way to use a build-time credential (private registry token, SSH key). Never `ARG`/`ENV`/`COPY` a secret.

```dockerfile
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    --mount=type=cache,target=/root/.npm \
    npm ci --omit=dev
# build with:  docker build --secret id=npmrc,src=$HOME/.npmrc .
```

The cache-mount target path is language-specific — use the package manager's documented cache dir from the relevant language guide rather than guessing.

### D. Build args vs runtime env (policy: `env-config.md`)

`ARG` is build-time only and is visible in `docker history` — use it for non-secret build parameters (versions, target platform). `ENV` persists into the running container — use it for non-secret runtime defaults. **Neither carries a secret.** Runtime config/secrets are injected by the orchestrator (see `docker-compose.md` / `kubernetes.md`), not the image.

### E. Non-root & hardening

```dockerfile
# Alpine/Debian runner: create a dedicated UID
RUN addgroup -S app && adduser -S -u 10001 -G app app
USER 10001:10001
# Distroless: use the built-in user
USER nonroot:nonroot
# scratch: copy /etc/passwd from the builder, then
USER 10001:10001
```

Strip setuid/setgid bits in the final stage when a shell-based image is unavoidable:
`RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true`. Use the exec (JSON) form for `ENTRYPOINT`/`CMD` so signals reach PID 1 and there is no shell to inject into.

### F. Read-only filesystem readiness (DF-RO-01)

Design every image to run with `--read-only`. Log to stdout/stderr (never to files — see `observability.md`), keep config in env/secrets, and write only to mounted `tmpfs`/volumes. Pre-create writable dirs at build time and point caches/temp at `/tmp`:

```dockerfile
ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1   # e.g. for Python; Node: npm_config_cache=/tmp/npm
```

### G. Healthcheck (DF-HEALTH-01)

Add a `HEALTHCHECK` for Compose/standalone runtimes; under Kubernetes the liveness/readiness probe supersedes it (define the probe there instead). Use a tool that actually exists in the image — distroless/scratch have no `curl`/`wget`, so expose a health endpoint the orchestrator polls, or ship a tiny static health binary.

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/app/healthcheck"]   # exec form, binary present in image
```

### H. OCI labels & reproducible builds

```dockerfile
LABEL org.opencontainers.image.source="https://github.com/org/repo" \
      org.opencontainers.image.version="1.4.2" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.description="Production API"
```

Reproducibility: pin the base by digest in production (`FROM node:20.12-alpine@sha256:...`), commit lockfiles and use frozen installs (DF-DEP-01), and prefer `SOURCE_DATE_EPOCH` + `--build-arg` over embedding build timestamps. Tag images per `semver.md`; a digest is the immutable identity.

### I. `.dockerignore` (DF-IGNORE-01)

A `.dockerignore` shrinks the build context, speeds uploads, and — critically — keeps secrets and junk out of `COPY . .`:

```
.git
node_modules
**/__pycache__
dist
coverage
.env
*.log
*.pem
```

### J. buildx & multi-platform

Build multi-arch images and attach provenance/SBOM attestations with buildx:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  --provenance=true --sbom=true \
  -t registry/app:1.4.2 --push .
```

Generate a standalone SBOM for supply-chain records: `docker buildx build --output type=sbom,dest=sbom.json .` (policy: `secure-coding.md`).

---

## 6. Gold-Standard Patterns

**Go → scratch** (smallest, zero attack surface):

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.22-alpine AS builder
WORKDIR /app
RUN apk add --no-cache ca-certificates && adduser -D -u 10001 app
COPY --link go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY --link . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -ldflags="-w -s" -o /server ./cmd/server

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /etc/passwd /etc/passwd
COPY --link --from=builder /server /server
USER 10001:10001
ENTRYPOINT ["/server"]
```

**Python → distroless** (interpreted runtime, no shell):

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --target=/app/deps -r requirements.txt

FROM gcr.io/distroless/python3-debian12
WORKDIR /app
ENV PYTHONPATH=/app/deps PYTHONDONTWRITEBYTECODE=1
COPY --link --from=builder /app/deps /app/deps
COPY --link src/ ./src/
USER nonroot:nonroot
CMD ["src/main.py"]
```

Language-specific build idioms (e.g. `cargo build --target …-musl`, `dotnet publish --self-contained`, `./gradlew build`) belong to the respective language guide — bind them into the **build** stage and keep the runner minimal.

---

## 7. Quick Reference

```bash
hadolint Dockerfile                                    # lint
docker buildx build --pull -t app:test .               # build (BuildKit)
trivy image --severity HIGH,CRITICAL app:test          # CVE scan
docker run --rm --read-only --tmpfs /tmp app:test id   # non-root + read-only
docker history --no-trunc app:test                     # audit layers/secrets
docker buildx build --output type=sbom,dest=sbom.json .# SBOM
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] DF-LINT-01 — hadolint clean
- [ ] DF-BUILD-01/02 — builds with BuildKit; multi-stage runner carries only artifacts
- [ ] DF-BUILD-03 — exec-form ENTRYPOINT/CMD
- [ ] DF-IMG-01 — minimal base, size target met
- [ ] DF-PIN-01 — pinned tags (digest in prod), no `:latest`
- [ ] DF-USER-01 — runs as non-root UID
- [ ] DF-CACHE-01 — deps layer cached after a source-only edit
- [ ] DF-SEC-01/02 — 0 HIGH/CRITICAL CVEs, no secrets in any layer
- [ ] DF-RO-01 — runs read-only with tmpfs
- [ ] DF-IGNORE-01 — `.dockerignore` excludes VCS/deps/secrets/tests
- [ ] DF-META-01 — OCI labels set (source, version)
- [ ] DF-HEALTH-01 — HEALTHCHECK or orchestrator probe defined
- [ ] DF-DEP-01 — committed lockfiles, frozen installs
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Dockerfile Guidelines**

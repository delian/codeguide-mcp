# Secure Coding Guidelines
Mandatory, language-agnostic security standards: vulnerability scanning (SAST/SCA/DAST), supply-chain & dependency auditing, secrets management, input validation, injection/XSS/SSRF defenses, authn/authz hygiene, and correct crypto usage.

---
name: secure-coding
title: Secure Coding Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [semgrep, trivy, gitleaks, trufflehog, osv-scanner, syft, grype, cosign, zap]
requires: []
recommends:
  - oauth
  - error-handling
  - logging
  - env-config
provides:
  - vuln-scanning
  - supply-chain-audit
  - secrets-management
  - input-validation
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md). This is the **canonical owner** of security, supply chain, secrets, and CVE policy — other guides reference it instead of restating its rules. It is language-agnostic; language guides bind these rules to concrete tools (e.g. Python → `bandit`/`pip-audit`, Node → `npm audit`, Go → `govulncheck`).

---

## 0. Prerequisites & References

This guide owns secure coding. Adjacent cross-cutting concerns live in their own canonical guides — fetch them rather than expecting this guide to restate them.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`oauth.md`](guides://oauth.md) — auth flows, token lifecycle, PKCE, refresh-token rotation, complete token-storage patterns. *(This guide owns only the secure-storage rule, not the flows.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy/propagation. *(This guide adds only: errors MUST NOT leak internals.)*
> - [`logging.md`](guides://logging.md) — structured logging. *(This guide adds only: secrets/PII MUST NOT enter logs.)*
> - [`env-config.md`](guides://env-config.md) — configuration & secret loading/layering. *(This guide owns the rule that secrets never live in code; the *mechanism* of loading config is owned there.)*

> 📎 **SEE ALSO:** [`ci-cd.md`](guides://ci-cd.md) · [`pre-commit.md`](guides://pre-commit.md) · [`docker-compose.md`](guides://docker-compose.md) · [`dockerfile.md`](guides://dockerfile.md) · [`rest.md`](guides://rest.md) · [`graphql.md`](guides://graphql.md)

> 📎 **TDD binding:** Security is test-first like everything else — write the negative/attack test before the fix (owned by [`tdd.md`](guides://tdd.md)). Do not restate Red-Green-Refactor here.

---

## 1. Core Philosophies: SECURE-FIRST

Security-specific principles. (Test-first, error strategy, and logging discipline come from §0.)

- **S**anitize all input: never trust external data; validate, constrain, and canonicalize at every trust boundary.
- **E**ncrypt sensitive data: strong, modern crypto in transit (TLS 1.2+/1.3) and at rest; authenticated encryption only.
- **C**redentials never in code: zero secrets in source, config-in-VCS, comments, or test fixtures.
- **U**se least privilege: minimum permissions; fail closed, not open.
- **R**eject by default: deny unless explicitly permitted; allowlist over denylist.
- **E**scape output: context-aware encoding at every sink (HTML/JS/URL/SQL/shell).

Plus: **Defense in depth** (no single control), **fail securely** (no info leak on error), **assume breach** (limit blast radius), **shift left** (scan in pre-commit + CI), **trust nothing from the supply chain** (pin, verify, audit).

**Verified Secure**: agent-generated code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `SEC-<TOPIC>-<NN>`. Each row has a binary gate; tool names are examples — bind to the project's language toolchain. Rows that bind a cross-cutting rule cite the owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| SEC-SECRET-01 | No hardcoded secrets in code, config, comments, or tests | `gitleaks detect --source . --redact` | 0 findings |
| SEC-SECRET-02 | Full git history MUST be secret-clean | `gitleaks detect --log-opts="--all"` | 0 findings |
| SEC-SECRET-03 | Secrets loaded only from env/secret-manager (see `env-config.md`) | review / `grep -rEnI '(api[_-]?key|secret|password|token)\s*[:=]\s*["'\'']'` | no literals |
| SEC-SAST-01 | Static analysis MUST be clean of high/critical findings | `semgrep --config=auto --error .` | 0 high/critical |
| SEC-SCA-01 | 0 known high/critical CVEs in dependencies | `osv-scanner -r .` (or `trivy fs .`) | 0 high/critical |
| SEC-SCA-02 | Lockfile committed, integrity-pinned (hashes), in sync | language lock-verify (e.g. `npm ci`, `uv lock --check`) | verified |
| SEC-SUPPLY-01 | SBOM generated for releases | `syft . -o spdx-json` | SBOM emitted |
| SEC-SUPPLY-02 | Release artifacts/images signed & verifiable | `cosign verify <artifact>` | signature valid |
| SEC-INPUT-01 | All external input validated by allowlist at the boundary | review / tests | negative tests pass |
| SEC-INJECT-01 | No string-built SQL/shell/template; parameterize or array-exec | `semgrep` injection rules | 0 findings |
| SEC-XSS-01 | Output context-encoded; auto-escaping templates; CSP set | `semgrep` + header check | 0 findings, CSP present |
| SEC-SSRF-01 | Outbound URLs from user input validated against an allowlist; no raw fetch | review / `semgrep` | 0 findings |
| SEC-AUTHZ-01 | Authn + object-level authz enforced server-side per request (no IDOR) | review / authz tests | tests pass |
| SEC-CRYPTO-01 | Only approved algorithms; CSPRNG for security tokens | `semgrep` weak-crypto rules | 0 findings |
| SEC-TLS-01 | TLS cert validation never disabled; TLS ≥1.2 | `semgrep` + config review | 0 findings |
| SEC-HEADER-01 | Security headers + secure cookie flags on all HTTP responses | header scan / tests | required set present |
| SEC-LOG-01 | No secrets/PII in logs (see `logging.md`) | review / log scan | 0 leaks |
| SEC-ERR-01 | Errors return generic messages; no stack/SQL/infra detail in prod (see `error-handling.md`) | review / tests | no internal leak |
| SEC-TST-01 | Each vulnerability MUST get a regression test before the fix (see `tdd.md`) | test runner | failing→passing |

> **Forbidden — never deliver code that**: hardcodes secrets; concatenates user input into SQL/shell/HTML; uses weak crypto (MD5/SHA-1 for passwords, DES/RC4/AES-ECB); disables TLS verification; uses a non-CSPRNG for tokens; `eval`s user input; deserializes untrusted data unsafely; allows path traversal or SSRF; logs secrets; or fixes a vuln without a regression test first.

---

## 3. Verification Protocol

Run before presenting code (or wire into pre-commit + CI). Fix → re-run until green. Substitute language-specific tools where noted.

```bash
gitleaks detect --source . --redact            # SEC-SECRET-01
gitleaks detect --log-opts="--all"             # SEC-SECRET-02 (history)
semgrep --config=auto --error .                # SEC-SAST-01 / INJECT / XSS / SSRF / CRYPTO / TLS
osv-scanner -r .   # or: trivy fs .            # SEC-SCA-01 (CVEs)
<lock-verify>      # npm ci | uv lock --check | go mod verify | cargo audit
syft . -o spdx-json > sbom.spdx.json           # SEC-SUPPLY-01 (releases)
cosign verify <artifact>                        # SEC-SUPPLY-02 (releases)
```

Language bindings: Python `bandit -r src/` + `pip-audit`; Node `npm audit --audit-level=high`; Go `govulncheck ./...` + `gosec ./...`; Rust `cargo audit` + `cargo deny`; Ruby `bundler-audit`; Java `mvn dependency-check`. The *why* behind each gate lives in this guide; the language *how* lives in that language's guide.

---

## 4. Secrets Management

> Loading & layering config is owned by [`env-config.md`](guides://env-config.md); OAuth token lifecycle by [`oauth.md`](guides://oauth.md). This guide owns the absolute rule below.

**The Cardinal Rule (no exceptions): secrets MUST NOT appear in source, config-in-VCS, URLs, headers, comments, or test fixtures.** Secrets = API keys, DB/connection strings, OAuth client secrets, JWT signing keys, encryption/SSH/TLS private keys, service-account creds, webhook secrets.

These are *all* violations of SEC-SECRET-01 — detectors catch every one:

```pseudocode
api_key = "sk_live_abcd1234"                 // hardcoded
db = "postgres://admin:P@ssw0rd@db/prod"     // in a connection string
secret = base64_decode("c2VjcmV0")           // "obfuscated" — still detectable
api_key = "sk_live_" + "abcd1234"            // split across vars
TEST_API_KEY = "sk_test_real_key"            // in a test file (still in VCS!)
```

**Approved sources** (in preference order):
1. **Secret manager** — AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, HashiCorp Vault. Fetched at runtime, access-controlled, rotatable, audited.
2. **Environment variables** injected by the platform — read at startup, fail fast if absent.
3. **Git-ignored file** (e.g. `.env`) with a committed `.env.example` of placeholders. Only for local dev.

**`.gitignore` must exclude:** `.env`, `.env.local`, `.env.*.local`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*credentials*.json`, `*secret*.json`, `.secrets/`, `config/secrets/`.

**Token storage (this guide's rule; full flows → [`oauth.md`](guides://oauth.md)):** refresh tokens → `HttpOnly; Secure; SameSite=Strict` cookie; access tokens → in-memory only (private field). **Never** `localStorage`/`sessionStorage` (XSS-readable, persistent, no expiry). Mobile → platform secure storage (iOS Keychain `…WhenUnlockedThisDeviceOnly`, Android Keystore / `EncryptedSharedPreferences`).

**On exposure (rotate-first):** (1) immediately rotate/revoke the secret — *do not* merely delete it from code; (2) push new value to the secret store; (3) purge from history (`git filter-repo` / BFG; `filter-branch` is deprecated), force-push, team re-clones; (4) audit access logs for misuse; (5) record the incident. A leaked secret is compromised the moment it touches a remote — rotation, not redaction, is the fix.

---

## 5. Input Validation

**Never trust external input. Validate at the trust boundary, allowlist over denylist.** Sources: form fields, query strings, path params, headers/cookies, request bodies, file uploads, third-party API responses, deserialized data, env vars and config at startup.

**Allowlist, not denylist** — denylists are always bypassable:

```pseudocode
// ❌ denylist — attackers find the gap
if username.containsAny(["<", ">", "'", ";", "--"]) reject

// ✅ allowlist — only known-good shape passes
validate(username, /^[a-zA-Z0-9_]{3,30}$/)
```

Validate **type → length/range → format → semantics**, and reject (don't silently coerce) on failure. Prefer a schema validator (JSON Schema, zod, pydantic) over ad-hoc checks.

| Type | Rules |
|------|-------|
| String | max length, allowed-char class, format regex |
| Integer/Float | min/max range, sign; reject NaN/Infinity |
| Email | format, length ≤254, normalize case |
| URL | scheme allowlist (`https`), host allowlist (see SSRF §7) |
| Date | format, plausible range, explicit timezone |
| File | extension + MIME + **magic-bytes** + size (see §9) |
| JSON | schema-validate; cap depth & size (zip/JSON-bomb defense) |
| HTML | sanitize via allowlist library (never regex-strip tags) |

---

## 6. Output Encoding & Injection Defense

**Encode at the sink, matched to context.** A single value may be safe in one context and an exploit in another.

| Sink context | Defense |
|---|---|
| HTML body/attribute | HTML-entity encode (`&`,`<`,`>`,`"`,`'`); auto-escaping template |
| JavaScript | `JSON.stringify` / JS-encode; never interpolate into a `<script>` |
| URL/query | percent-encode each component |
| **SQL** | parameterized queries / prepared statements — **always** |
| **Shell** | avoid the shell; array-exec (`["convert", file, out]`); never `sh -c "…"` |
| OS path | canonicalize + confine to base dir (see §9) |
| LDAP/XPath/NoSQL | use the driver's parameter API, never string-build |

```pseudocode
// SQL — the only acceptable form
db.execute("SELECT * FROM users WHERE id = ? AND status = ?", [id, status])
// ORM raw escape hatches still parameterize: User.where("name = ?", name)  // not "name = '#{name}'"

// Shell — pass args as a list, no interpolation
subprocess.run(["convert", filename, "out.png"])   // shell=False
```

**Templates:** use auto-escaping engines (Jinja2, Django, Go `html/template`, JSX). The "raw/`|safe`" escape hatch requires prior allowlist sanitization. Never "sanitize" by `replace("<script>","")` — trivially bypassed.

---

### XSS & Content Security Policy (SEC-XSS-01)

Output-encode (above) **and** ship a strict CSP as a second layer:

```
Content-Security-Policy:
  default-src 'self'; script-src 'self' 'nonce-{random}';
  style-src 'self' 'nonce-{random}'; img-src 'self' data: https:;
  object-src 'none'; base-uri 'self'; form-action 'self';
  frame-ancestors 'none'; upgrade-insecure-requests;
```

Prefer per-request **nonces** (`base64(CSPRNG(16))`) over `'unsafe-inline'`. For user-supplied rich text, sanitize with a maintained allowlist library (DOMPurify, bleach, sanitize-html) — fixed tag/attribute/protocol allowlists.

---

## 7. SSRF & Outbound Request Defense

User-controlled URLs are a top-tier risk (cloud metadata theft, internal pivots). For any fetch driven by user input:

- Parse the URL and **allowlist scheme (`https`) and host**; reject by default.
- Resolve DNS and **block private/link-local/loopback ranges** (`127.0.0.0/8`, `10/8`, `172.16/12`, `192.168/16`, `169.254/16` incl. `169.254.169.254` metadata, `::1`, `fc00::/7`) — re-check **after** redirects (DNS-rebinding defense).
- Disable or cap redirects; set timeouts; do not echo the raw response body back to the caller.
- Where possible, route egress through a vetted proxy/allowlist gateway.

```pseudocode
url = parse(userInput)
if url.scheme != "https" or url.host not in HOST_ALLOWLIST: reject
ip = resolve(url.host)
if ip.isPrivate or ip.isLoopback or ip.isLinkLocal: reject
fetch(url, followRedirects=false, timeout=5s)
```

---

## 8. Authentication, Authorization & Cryptography

> Auth **flows** (OAuth/OIDC, PKCE, token refresh/rotation) are owned by [`oauth.md`](guides://oauth.md). This section owns the cryptographic hygiene that underpins them.

**Passwords:** hash with **Argon2id** (preferred), `scrypt`, or `bcrypt` — memory-hard, per-user salt, tuned cost. Never MD5/SHA-1/SHA-256 (even salted), never reversible encryption, never plaintext. Verify with the library's constant-time comparison.

**Sessions/tokens:** ≥256 bits from a CSPRNG; never `Math.random()`/`rand()`/timestamps/sequential IDs. Bind expiry server-side; rotate on privilege change; invalidate on logout.

**Authorization (SEC-AUTHZ-01):** check **on every request, server-side** — authenticate, load resource, then verify object-level permission *before* returning it. The classic IDOR bug is fetching by ID with no ownership check. Enforce RBAC/ABAC centrally; deny by default.

### Approved cryptography

| Purpose | Approved | Forbidden |
|---|---|---|
| Password hashing | Argon2id, scrypt, bcrypt | MD5, SHA-1, plain SHA-256 |
| Symmetric enc. | AES-256-GCM, ChaCha20-Poly1305 | DES, 3DES, RC4, Blowfish, AES-ECB, AES-CBC w/ static IV |
| Asymmetric | RSA ≥3072 (≥2048 min), ECDSA P-256+, Ed25519 | RSA <2048, DSA |
| Hashing (non-pw) | SHA-256/384/512, SHA-3, BLAKE2/3 | MD5, SHA-1 |
| TLS | 1.3 (1.2 acceptable) | SSLv3, TLS 1.0/1.1 |
| Randomness (security) | OS CSPRNG (`getrandom`, `crypto.randomBytes`) | `Math.random()`, `rand()`, time |

```pseudocode
// Authenticated encryption — fresh random IV every time
iv = CSPRNG(12)                       // 96-bit nonce for GCM
ct, tag = AES_256_GCM(key, iv).encrypt(plaintext)
return iv || tag || ct                // decrypt MUST verify tag, else reject (tampered)
```

**Keys:** generate from CSPRNG or derive via Argon2/scrypt/PBKDF2 with a salt; store in a secret manager/KMS (see §4); rotate. Never hardcode a key, never `md5(password)` as a key, never use a password directly as a key. **Never disable TLS certificate validation** (SEC-TLS-01) — not even "temporarily" in tests checked into VCS.

---

## 9. File Upload & Path Traversal

**Uploads:** enforce size cap → extension allowlist → MIME allowlist → **magic-byte signature** check (extension/MIME are spoofable) → generate a server-side filename (UUID + validated ext; never trust the client name) → store outside the web root or on a separate domain → restrictive permissions → scan if executable content is possible.

**Path traversal:** never join user input onto a path raw. `basename` it, allowlist chars (`^[a-zA-Z0-9._-]+$`), resolve+normalize, then assert the result is still inside the base dir:

```pseudocode
full = base.resolve(basename(name)).normalize()
if not full.startsWith(base): reject       // blocks "../../etc/passwd"
```

---

## 10. API Security

- **Rate limiting (anti-brute-force/DoS):** tight on sensitive endpoints (`login` 5/min, `password_reset` 3/hr, `signup` 10/hr), general default per IP+endpoint; return `429` + `Retry-After`. (Brute-force regression test belongs in §2 SEC-TST-01.)
- **Auth headers:** validate `Authorization: Bearer …`; map token-expired vs invalid to `401` without leaking which.
- **API keys are passwords:** store only their hash (SHA-256 is fine for a high-entropy key), show the plaintext once.
- **Transport:** HTTPS only; HSTS; reject mixed content.

### Required HTTP response headers (SEC-HEADER-01)

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY                       # or CSP frame-ancestors 'none'
Content-Security-Policy: default-src 'self'; script-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

> `X-XSS-Protection` is **deprecated** (removed from modern browsers and can introduce bugs) — rely on CSP instead, do not set it.

### Cookies

```
Set-Cookie: session=TOKEN; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600
```

Missing `HttpOnly` → XSS-readable; missing `Secure` → sent over HTTP; missing `SameSite` → CSRF exposure.

---

## 11. Logging & Error Handling (bindings)

> Owned by [`logging.md`](guides://logging.md) and [`error-handling.md`](guides://error-handling.md). This guide adds only the security constraints.

- **Logs (SEC-LOG-01):** log security events (authn/authz outcomes, rate-limit trips) with *identifiers* (user id, IP, resource id, outcome) — **never** passwords, tokens, API keys, full card/PII, or raw request bodies. Redact at the logging boundary.
- **Errors (SEC-ERR-01):** in production return a generic message + correlation id; log full detail server-side. Never return stack traces, SQL text, or infrastructure detail to the client.

---

## 12. Supply-Chain Security

Modern attacks target *dependencies*, not just your code.

- **Pin & verify:** commit lockfiles with integrity hashes; install in CI from the lockfile only (`npm ci`, `pip install --require-hashes`, `uv lock --check`, `go mod verify`) — SEC-SCA-02.
- **Audit continuously:** OSV/`osv-scanner`, `trivy`, GitHub Advisory; fail CI on high/critical CVEs (SEC-SCA-01). Triage with severity + reachability; don't blanket-ignore.
- **Automate updates** via Dependabot/Renovate, but **review** changelogs/diffs before merge — auto-merge only patch bumps that pass full CI.
- **SBOM + signing:** emit an SBOM per release (`syft`, SPDX/CycloneDX — SEC-SUPPLY-01); sign artifacts/images (`cosign`/Sigstore) and verify on deploy (SEC-SUPPLY-02); pin GitHub Actions to commit SHAs, not tags.
- **Vet new deps:** prefer maintained, widely-used packages; watch for typosquats, install-time scripts, and recently-transferred ownership. Maintain a project denylist for known-compromised packages and fail the build on them.

---

## 13. Security Testing

Security is test-first (see [`tdd.md`](guides://tdd.md)). Every fixed vulnerability gets a regression test **before** the fix (SEC-TST-01). Cover the attack vectors, not just the happy path:

```pseudocode
TEST "SQL injection neutralized"
  search("'; DROP TABLE users; --");  ASSERT db.tableExists("users")
TEST "XSS encoded"
  out = render(name="<script>alert(1)</script>")
  ASSERT out CONTAINS "&lt;script&gt;" and NOT CONTAINS "<script>"
TEST "authz blocks cross-tenant access (IDOR)"
  ASSERT get(otherUsersResource, as=user1).status == 403
TEST "rate limit triggers"
  repeat 6× login(wrong);  ASSERT lastResponse.status == 429
TEST "SSRF blocked"
  ASSERT fetchOnBehalf("http://169.254.169.254/").rejected
```

**Pipeline stages** (wire into [`ci-cd.md`](guides://ci-cd.md), shift earliest into [`pre-commit.md`](guides://pre-commit.md)): secret scan → SCA/CVE → SAST → SBOM+sign → DAST (e.g. `zap`, manual triage). Fail the build on secret/SCA/SAST findings; DAST results reviewed.

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] SEC-SECRET-01/02/03 — no secrets in code, history, or config; loaded from env/secret-manager
- [ ] SEC-SAST-01 — static analysis clean of high/critical
- [ ] SEC-SCA-01/02 — 0 high/critical CVEs; lockfile committed, hash-pinned, in sync
- [ ] SEC-SUPPLY-01/02 — SBOM generated; artifacts signed & verifiable
- [ ] SEC-INPUT-01 — all external input allowlist-validated at the boundary
- [ ] SEC-INJECT-01 — SQL/shell/template parameterized, never string-built
- [ ] SEC-XSS-01 — output context-encoded; auto-escaping templates; strict CSP
- [ ] SEC-SSRF-01 — user-driven outbound URLs allowlisted; private ranges blocked
- [ ] SEC-AUTHZ-01 — authn + object-level authz enforced server-side (no IDOR)
- [ ] SEC-CRYPTO-01 — approved algorithms only; CSPRNG for security tokens
- [ ] SEC-TLS-01 — TLS verification never disabled; TLS ≥1.2
- [ ] SEC-HEADER-01 — security headers + secure cookie flags on all responses
- [ ] SEC-LOG-01 — no secrets/PII in logs (see `logging.md`)
- [ ] SEC-ERR-01 — generic errors in prod; no internal leakage (see `error-handling.md`)
- [ ] SEC-TST-01 — every fixed vuln has a regression test added first
- [ ] Agent ran every §3 command and documented any fixes

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/) · [OWASP ASVS](https://owasp.org/www-project-application-security-verification-standard/) · [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [CWE Top 25](https://cwe.mitre.org/top25/) · [NIST SSDF (SP 800-218)](https://csrc.nist.gov/projects/ssdf) · [SLSA](https://slsa.dev/) · [Sigstore](https://www.sigstore.dev/) · [OSV](https://osv.dev/)

---
**End of Secure Coding Guidelines**

# OAuth 2.1 & OpenID Connect Guidelines
Mandatory standards for OAuth 2.1 / OIDC: grant flows, PKCE, token types, scopes, token storage & rotation, JWT validation, and common pitfalls. OAuth 2.1, OIDC, JWT (RFC 7519), PKCE (RFC 7636), JWKS.

---
name: oauth
title: OAuth 2.1 & OpenID Connect Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - secure-coding
  - rest
  - openapi
  - error-handling
provides:
  - oauth-flows
  - oidc
  - token-handling
  - pkce
  - jwt-validation
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns OAuth 2.1 / OIDC and spends its tokens on identity-protocol specifics; it does not re-explain general secret handling, API design, or error strategy.

---

## 0. Prerequisites & References

This guide canonically owns the OAuth/OIDC protocol surface. Fetch these when their concern is in play; do not duplicate their content here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`secure-coding.md`](guides://secure-coding.md) — general secret/credential storage, TLS, supply chain, CVEs. *(OAuth binding here is only: where client secrets / private keys / refresh tokens are kept and how they rotate.)*
> - [`rest.md`](guides://rest.md) · [`openapi.md`](guides://openapi.md) — declaring `bearer`/OAuth2 security schemes on APIs and protecting endpoints.
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(OAuth binding: the RFC 6749 `error`/`error_description` shape and which HTTP status maps to which auth failure.)*

> 📎 **SEE ALSO:** [`websocket.md`](guides://websocket.md) (token-on-connect auth) · [`logging.md`](guides://logging.md) (never log tokens/codes) · [`ios.md`](guides://ios.md) · [`android.md`](guides://android.md) (platform secure storage)

---

## 1. Core Philosophies: AUTH-FIRST

OAuth/OIDC-specific principles only. General secret handling, TLS, and error shape come from §0.

- **A**uthN ≠ AuthZ: OAuth 2.1 is **authorization** (delegated access via access tokens); OIDC adds **authentication** (who the user is, via the ID token). Never use an OIDC ID token as an API access token, and never use a raw access token to identify a user in a UI.
- **U**se the safe grants only: OAuth 2.1 removes the **implicit** and **resource-owner-password** grants. Public clients (SPA, mobile, native, CLI) MUST use **authorization code + PKCE**; confidential machine-to-machine clients use **client credentials**; input-constrained devices use the **device authorization grant**.
- **T**okens are bearer secrets, scoped and short-lived: request least-privilege scopes, give access tokens minutes-not-days lifetimes, and bind sessions to rotating refresh tokens.
- **H**arden every hop: enforce HTTPS, exact redirect-URI matching, `state` + (for OIDC) `nonce`, full JWT signature/claim validation, and sender-constraining (DPoP / mTLS) for high-value APIs.

**Verified Code**: Agent-generated auth code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `OAUTH-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared concern cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| OAUTH-FLOW-01 | Public clients MUST use authorization code + PKCE (`S256`); implicit & password grants MUST NOT be used | grep flow config / review | no `response_type=token`, no `grant_type=password` |
| OAUTH-FLOW-02 | Every authorization request MUST send a cryptographically random `state`, verified on callback | review / unit test | mismatch rejected |
| OAUTH-FLOW-03 | OIDC requests MUST send `nonce`; the ID token `nonce` MUST be verified | unit test | mismatch rejected |
| OAUTH-PKCE-01 | `code_verifier` MUST be 43–128 chars of unreserved chars; challenge MUST be `S256` (never `plain`) | unit test (RFC 7636) | exit 0 |
| OAUTH-TOKEN-01 | Access tokens MUST be short-lived (≤ 15 min) and refresh tokens MUST rotate on use with old-token revocation | config review / integration test | replay of old refresh fails |
| OAUTH-TOKEN-02 | Tokens MUST NOT be placed in `localStorage`/`sessionStorage`; web refresh tokens go in `HttpOnly; Secure; SameSite` cookies, access tokens in memory (see `secure-coding.md`) | grep / review | no `localStorage.setItem('*token*')` |
| OAUTH-JWT-01 | JWTs MUST be verified for signature, `alg` (allowlist, no `none`), `iss`, `aud`, `exp`/`nbf` before trust | unit test / review | invalid token rejected |
| OAUTH-JWT-02 | Signing keys MUST come from the issuer JWKS (`kid`-matched), with caching + rotation; keys MUST NOT be hardcoded | review | keys fetched from JWKS |
| OAUTH-SCOPE-01 | Resource servers MUST enforce least-privilege scopes/audience per endpoint | review / test | 403 on missing scope |
| OAUTH-TLS-01 | All endpoints (authorize, token, redirect, JWKS) MUST be HTTPS; redirect URIs MUST match registered values exactly (see `secure-coding.md`) | config review | no `http://` (except loopback native) |
| OAUTH-SEC-01 | Client secrets / private keys MUST NOT be shipped to public clients or committed (see `secure-coding.md`) | secret scan | 0 secrets in client bundle/repo |
| OAUTH-ERR-01 | Token/authorization errors MUST use the RFC 6749 `{error, error_description}` shape with correct status (see `error-handling.md`) | contract test | shape + status match |
| OAUTH-LOG-01 | Tokens, codes, verifiers, and secrets MUST NOT be logged (see `logging.md`/`secure-coding.md`) | grep / log review | no token material in logs |

> **Forbidden**: implicit/password grants; PKCE `plain`; accepting unsigned/`alg:none` JWTs; trusting a JWT without `aud`/`iss` checks; storing tokens in web storage; logging tokens or authorization codes; skipping `state`/`nonce`; non-exact redirect-URI matching.

---

## 3. OAuth 2.1 Grant Flows

Pick the grant by client type. OAuth 2.1 narrows the choice to four safe grants.

| Client type | Grant | Secret? | Notes |
|---|---|---|---|
| SPA / mobile / native / CLI (public) | Authorization code **+ PKCE** | No | PKCE is mandatory for all clients in 2.1, public or not |
| Backend service → API (no user) | Client credentials | Yes (confidential) | M2M; scopes/audience identify the caller |
| TV / IoT / input-constrained | Device authorization (RFC 8628) | Usually no | User approves on a second device |
| Token renewal | Refresh token | Per client type | Rotated on every use (§5) |

### A. Authorization Code + PKCE (the default)

PKCE (RFC 7636) binds the authorization request to the token exchange so an intercepted `code` is useless without the original `code_verifier`.

```
1. Client generates code_verifier (43–128 unreserved chars) and
   code_challenge = BASE64URL(SHA256(code_verifier)).
2. GET /authorize?response_type=code&client_id=...&redirect_uri=<exact>
       &scope=openid profile email&state=<rand>&nonce=<rand>      (nonce: OIDC)
       &code_challenge=<challenge>&code_challenge_method=S256
3. User authenticates & consents → redirect back with ?code=...&state=...
4. Verify returned state == stored state (CSRF). Reject on mismatch.
5. POST /token  grant_type=authorization_code&code=...&redirect_uri=<exact>
       &client_id=...&code_verifier=<verifier>
6. AS recomputes S256(code_verifier) and compares to stored challenge.
7. Verify ID token (incl. nonce) before establishing a session.
```

- `state` is a random, single-use, time-bounded value stored client-side and compared on callback — it is the CSRF defense, distinct from PKCE.
- `code_challenge_method` MUST be `S256`. `plain` is for legacy interop only and is forbidden here.
- Use `Authorization Code` even for confidential web apps (they add client authentication at the token endpoint).

### B. Client Credentials (machine-to-machine)

Server-side only; the client authenticates with its own credential (secret, or preferably a `private_key_jwt` / mTLS assertion) and receives an access token scoped to an audience. There is no user, no refresh token, no ID token.

```
POST /token
  grant_type=client_credentials
  scope=invoices:read invoices:write
  (client auth: client_secret_basic | private_key_jwt | tls_client_auth)
```

Prefer asymmetric client authentication (`private_key_jwt`/mTLS) over a shared `client_secret`. Secret storage/rotation policy is owned by [`secure-coding.md`](guides://secure-coding.md).

### C. Device Authorization Grant (RFC 8628)

For devices that cannot show a browser/keyboard. The device gets a `device_code` + `user_code` + `verification_uri`, shows the user the code/URI, then polls the token endpoint honoring `interval` and backing off on `slow_down`/`authorization_pending`.

```
POST /device_authorization → { device_code, user_code, verification_uri, interval, expires_in }
(poll) POST /token  grant_type=urn:ietf:params:oauth:grant-type:device_code&device_code=...
```

---

## 4. Token Types, Validation & Scopes

### A. The three tokens — never confuse them

| Token | Format | Audience | Purpose | Lifetime |
|---|---|---|---|---|
| **Access token** | JWT or opaque | the **resource server** (API) | authorize API calls (`Authorization: Bearer`) | minutes (≤ 15) |
| **Refresh token** | opaque | the **authorization server** | obtain new access tokens | days, rotated on use |
| **ID token** | JWT (OIDC) | the **client** | authenticate the user (who logged in) | short; one-time use to establish a session |

The ID token answers "who is this user"; the access token answers "what may this caller do". Sending an ID token to an API, or treating an access token's claims as proven user identity in the client, are both errors.

### B. JWT validation (resource server)

A bearer JWT is untrusted input until **every** check passes. Order: fetch the issuer JWKS, select the key by `kid`, verify the signature against an **algorithm allowlist** (e.g. `RS256`/`ES256`; reject `none`), then verify claims.

```
verify: signature (kid → JWKS key)
        alg ∈ allowlist            # reject "none" and unexpected algs
        iss == expected issuer
        aud contains this API's audience
        exp not passed, nbf reached (small clock skew, ~30s)
        scope/roles satisfy the endpoint                 # see §4.D
        (OIDC ID token) nonce == stored nonce, azp if multi-client
```

- Cache JWKS with the issuer's `kid` rotation in mind (respect cache headers; refetch on unknown `kid`). Keys come from `/.well-known/jwks.json` — **never** hardcode public keys (OAUTH-JWT-02).
- For **opaque** access tokens, validate via the introspection endpoint (RFC 7662) instead of local signature verification; cache results briefly.
- Prefer your stack's maintained, audited library (e.g. a JOSE/JWT verifier wired to a JWKS client) over hand-rolled base64/HMAC. Show only the binding to that library; do not reimplement crypto.

### C. Common claims

Standard JWT/OIDC claims: `iss`, `sub` (stable user id — use this, not email, as the key), `aud`, `exp`, `iat`, `nbf`, `jti` (unique id, enables revocation/replay checks); OIDC adds `nonce`, `azp`, `auth_time`, `acr`/`amr`. Custom claims SHOULD be namespaced (e.g. `https://myapp.example/roles`) to avoid collisions with registered claims.

### D. Scopes & authorization

- Request the **minimum** scopes needed; `openid` is required to receive an ID token, `offline_access` to receive a refresh token.
- Enforce scope/audience **at the resource server**, per endpoint — a valid signature alone does not authorize an action (OAUTH-SCOPE-01). Map endpoints to required scopes and return `403` (with `WWW-Authenticate: Bearer error="insufficient_scope"`) when unmet.
- Scopes are coarse delegation grants, not a full permission model. Fine-grained role/permission logic (RBAC/ABAC) is an application authorization concern — keep it out of the token where it would bloat, and resolve it server-side keyed by `sub`/roles claim.

---

## 5. Token Storage & Rotation

General "where do secrets live and how do they rotate" policy is owned by [`secure-coding.md`](guides://secure-coding.md). The OAuth-specific bindings:

**Storage by client type**
- **Browser/SPA**: access token in **memory only**; refresh token (if any) in an `HttpOnly; Secure; SameSite=Strict` cookie scoped to the refresh path — never `localStorage`/`sessionStorage` (XSS-exfiltratable). The Backend-for-Frontend (BFF) pattern — server holds tokens, browser holds only a session cookie — is the recommended SPA architecture.
- **Mobile/native**: OS secure storage — iOS **Keychain** (`kSecAttrAccessibleWhenUnlockedThisDeviceOnly`), Android **Keystore**/EncryptedSharedPreferences. Use the system in-app browser (ASWebAuthenticationSession / Custom Tabs), never an embedded WebView. (Platform details: [`ios.md`](guides://ios.md), [`android.md`](guides://android.md).)
- **Server/M2M**: secrets and private keys in a secrets manager/KMS, injected via env/config, rotated on a schedule (see [`secure-coding.md`](guides://secure-coding.md)).

**Refresh token rotation (OAUTH-TOKEN-01)**
- Issue a new refresh token on every refresh and immediately invalidate the previous one (one-time-use).
- Detect reuse: if a **previously rotated** refresh token is presented, treat it as theft — revoke the entire token family/session and force re-auth.
- Keep access tokens short so a leaked one expires fast; rely on rotation + revocation for refresh tokens.

**Revocation & logout**
- Support the revocation endpoint (RFC 7009) and call it on logout for refresh tokens.
- For stateless JWT access tokens, maintain a short-lived deny-list keyed by `jti` (e.g. in Redis) for the cases where you must invalidate before `exp`; OIDC RP-initiated logout (`end_session_endpoint`) terminates the IdP session.

---

## 6. OpenID Connect

OIDC layers authentication on OAuth 2.1.

- **Discovery**: read `/.well-known/openid-configuration` to obtain `authorization_endpoint`, `token_endpoint`, `userinfo_endpoint`, `jwks_uri`, `end_session_endpoint`, and supported scopes/algs — do not hardcode endpoints; this enables key/endpoint rotation without client changes.
- **ID token**: a JWT proving authentication. Validate it like any JWT (§4.B) **plus** verify `nonce` and that `aud` equals your `client_id`. Establish the local session from the validated ID token; treat the access token as opaque-to-the-client.
- **UserInfo**: `GET /userinfo` with the access token returns profile claims (`sub`, `name`, `email`, `email_verified`, ...). Use it for fresh profile data rather than over-stuffing the ID token. Trust `sub` as the stable key; treat `email` as mutable and only after `email_verified`.

---

## 7. Common Pitfalls

- **Implicit / password grants** — removed in OAuth 2.1; tokens-in-URL leak via history/referrer. Use code + PKCE.
- **PKCE `plain`** — defeats the purpose; always `S256`.
- **Missing/loose redirect-URI match** — enables code interception/open redirect. Match registered URIs **exactly** (scheme, host, path, port).
- **Skipping `state`/`nonce`** — opens CSRF on the callback / ID-token replay.
- **`alg:none` or algorithm confusion** — never accept unsigned tokens; pin an asymmetric `alg` allowlist so an attacker can't downgrade RS256→HS256 using the public key as an HMAC secret.
- **Not checking `aud`/`iss`** — a valid token for *another* service is accepted. Always pin both.
- **Tokens in web storage / URLs / logs** — XSS-exfiltratable, leaked in referrers and log aggregation. Memory + HttpOnly cookies; scrub logs (OAUTH-LOG-01).
- **Treating scope as fine-grained authz** — enforce per-endpoint authorization server-side; don't assume a signature means "allowed".
- **Long-lived access tokens / no rotation** — widens the compromise window. Short access tokens + rotating, reuse-detecting refresh tokens.
- **Embedded WebView on mobile** — phishable, no SSO; use the system browser tab.
- **Confusing ID and access tokens** — see §4.A.

---

## 8. API Declaration & Error Shape (bindings)

- **Declaring auth on APIs**: define the security scheme where the API contract lives — OpenAPI `securitySchemes` (`type: oauth2` / `type: http, scheme: bearer`) per [`openapi.md`](guides://openapi.md), and protect endpoints per [`rest.md`](guides://rest.md). This guide owns *the flows/tokens*; those guides own *how the API advertises and requires them*.
- **Error responses**: token/authorization errors use the RFC 6749 body `{ "error": "...", "error_description": "..." }`. Map per [`error-handling.md`](guides://error-handling.md): `invalid_request`→400, `invalid_client`→401, `invalid_grant`→400, `unauthorized_client`→400, `unsupported_grant_type`→400, `invalid_scope`→400, `access_denied`→403. On resource servers use `WWW-Authenticate: Bearer error="invalid_token"|"insufficient_scope"` with 401/403. Never leak which factor failed beyond these standard codes.

---

## 9. Quick Reference

```
# Grant types (OAuth 2.1 — safe set only)
authorization_code                                  # user auth, ALWAYS with PKCE
client_credentials                                  # machine-to-machine
refresh_token                                       # renew (rotate on use)
urn:ietf:params:oauth:grant-type:device_code        # input-constrained devices

# Key endpoints
GET  /authorize                                     # authorization
POST /token                                         # token exchange / refresh
POST /revoke                                        # revocation (RFC 7009)
POST /introspect                                    # opaque token check (RFC 7662)
GET  /userinfo                                      # OIDC profile
GET  /.well-known/openid-configuration              # OIDC discovery
GET  /.well-known/jwks.json                         # signing keys

# Common scopes
openid profile email offline_access

# JWT claims to verify
iss aud exp nbf alg(kid)  + sub jti  + (OIDC) nonce azp
```

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] OAUTH-FLOW-01 — code+PKCE for public clients; no implicit/password grants
- [ ] OAUTH-FLOW-02 — `state` sent and verified on callback
- [ ] OAUTH-FLOW-03 — OIDC `nonce` sent and verified
- [ ] OAUTH-PKCE-01 — verifier 43–128 chars, `S256` only
- [ ] OAUTH-TOKEN-01 — short access tokens; refresh rotation + reuse detection
- [ ] OAUTH-TOKEN-02 — no web storage; HttpOnly cookie / in-memory (see `secure-coding.md`)
- [ ] OAUTH-JWT-01 — signature/`alg`/`iss`/`aud`/`exp` all verified
- [ ] OAUTH-JWT-02 — keys from JWKS, `kid`-matched, rotated, not hardcoded
- [ ] OAUTH-SCOPE-01 — least-privilege scopes/audience enforced per endpoint
- [ ] OAUTH-TLS-01 — HTTPS everywhere; exact redirect-URI match
- [ ] OAUTH-SEC-01 — no client secrets/keys in public clients or repo (see `secure-coding.md`)
- [ ] OAUTH-ERR-01 — RFC 6749 error shape & status (see `error-handling.md`)
- [ ] OAUTH-LOG-01 — no tokens/codes/secrets in logs (see `logging.md`)

---
**End of OAuth 2.1 & OpenID Connect Guidelines**

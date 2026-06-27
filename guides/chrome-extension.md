# Chrome Extension Development Guidelines
Mandatory standards for modern browser extensions: Manifest V3, ephemeral service workers, least-privilege permissions, no remote code. Chrome Extensions MV3, TypeScript 5.x, Vite + CRXJS, @types/chrome, webextension-polyfill.

---
name: chrome-extension
title: Chrome Extension Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [manifest-v3, typescript@5.5, vite@5, "@crxjs/vite-plugin", "@types/chrome", webextension-polyfill]
requires:
  - typescript
  - tdd
  - secure-coding
recommends:
  - javascript
  - html
  - css
  - oauth
  - vite
provides:
  - manifest-v3
  - extension-service-worker
  - content-scripts
  - extension-messaging
  - extension-permissions
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to building browser extensions.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating extension code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`typescript.md`](guides://typescript.md) — the language: strict mode, types, tsconfig. *(Extension binding: `@types/chrome` for API typings.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Extension binding: Vitest/Jest with a mocked `chrome` global; keep Chrome APIs out of domain logic so it is testable without a browser.)*
> - [`secure-coding.md`](guides://secure-coding.md) — CSP, supply chain, secrets, input validation. *(Extension binding: MV3 forbids remote code; validate every cross-context message; least-privilege permissions.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`html.md`](guides://html.md) · [`css.md`](guides://css.md) — popup / options / side-panel UI surfaces.
> - [`javascript.md`](guides://javascript.md) — plain-JS extensions or build-free scripts.
> - [`oauth.md`](guides://oauth.md) — user auth *(binding: `chrome.identity.launchWebAuthFlow` / `getAuthToken`)*.
> - [`vite.md`](guides://vite.md) — bundling *(binding: `@crxjs/vite-plugin` for HMR + manifest generation)*.

> 📎 **SEE ALSO:** [`error-handling.md`](guides://error-handling.md) · [`comments.md`](guides://comments.md) · [`e2e-testing.md`](guides://e2e-testing.md) *(Playwright with persistent context)* · [`semver.md`](guides://semver.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies

Extension-specific principles only. TDD, security, error handling, typing, and UI come from §0.

- **MV3 is non-negotiable.** Manifest V2 is removed from the Chrome Web Store. Background logic runs in an **ephemeral, event-driven service worker** — never assume it stays alive.
- **Least privilege.** Request the narrowest permission that works: `activeTab` over host permissions, optional permissions over up-front grants. Every extra permission slows Web Store review and erodes user trust.
- **No remote code, ever.** MV3 bans loading external JS/Wasm. All executable code ships in the package; remote config is *data*, not code.
- **Isolate contexts; communicate by messages.** Content scripts, service worker, and UI pages are separate worlds — share state through `chrome.storage` and validated messages, not globals.
- **Stateless background.** Persist anything that must survive worker termination; rebuild in-memory state from storage on wake.
- **Write once, target WebExtensions.** Prefer the `browser.*` promise API (via `webextension-polyfill`) so the same code runs on Chrome and Firefox.

**Verified Code**: Agent-generated extension code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `EXT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| EXT-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `npm test` | exit 0, 0 skips |
| EXT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npm test` | failing→passing |
| EXT-TYP-01 | Strict TypeScript, no `any` on public APIs (see `typescript.md`) | `tsc --noEmit` | exit 0 |
| EXT-FMT-01 | Code MUST be formatted | `prettier --check .` | no diff |
| EXT-LINT-01 | Linter MUST pass clean | `eslint .` | exit 0 |
| EXT-MV3-01 | Manifest MUST be V3 with a service-worker background | `jq .manifest_version dist/manifest.json` | `== 3` |
| EXT-PERM-01 | Permissions MUST be least-privilege; no `<all_urls>` unless essential (see `secure-coding.md`) | manifest review | justified |
| EXT-SEC-01 | No remote code; CSP `script-src 'self'`; no `eval`/inline (see `secure-coding.md`) | grep `eval(`, manifest CSP | none found |
| EXT-SEC-02 | Every cross-context message MUST validate sender & payload (see `secure-coding.md`) | review / schema test | validated |
| EXT-SEC-03 | 0 known CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| EXT-BUILD-01 | Extension MUST build & load unpacked without errors | `npm run build` | exit 0, valid `dist/` |
| EXT-DOC-01 | Public APIs documented (see `comments.md`) | `typedoc` | builds clean |

> **Forbidden**: shipping a persistent background page, loading remote scripts, requesting `<all_urls>`/`tabs` "just in case", using `webRequest` blocking (removed in MV3 — use `declarativeNetRequest`), trusting an unvalidated message, or storing secrets in the package.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
prettier --check .                  # EXT-FMT-01
eslint .                            # EXT-LINT-01
tsc --noEmit                        # EXT-TYP-01
npm test                            # EXT-TST-01/02
npm run build                       # EXT-BUILD-01  (produces dist/manifest.json)
jq -e '.manifest_version == 3' dist/manifest.json   # EXT-MV3-01
npm audit --audit-level=high        # EXT-SEC-03
```

Then load `dist/` via `chrome://extensions` → Developer mode → Load unpacked, and confirm no service-worker errors in the inspector. The *why* behind each gate lives in its §0 owner.

---

## 4. Project Structure

Feature-grouped layout with a build step. Keep Chrome-API calls in thin adapters so domain logic stays browser-free and unit-testable (architecture principles are owned by the language/architecture guides; this is only the extension mapping).

```
my-extension/
├── src/
│   ├── manifest.config.ts     # manifest authored in TS (CRXJS reads this)
│   ├── background/sw.ts        # service worker — event registration only
│   ├── content/main.ts         # content script(s)
│   ├── popup/{index.html,main.ts}
│   ├── options/{index.html,main.ts}
│   ├── sidepanel/{index.html,main.ts}
│   ├── lib/                    # pure logic — NO chrome.* imports (testable)
│   └── lib/messaging.ts        # typed message contracts + guards
├── test/                       # mirrors src/ (see tdd.md)
├── public/icons/{16,48,128}.png
├── rules/ruleset.json          # declarativeNetRequest static rules
├── vite.config.ts              # @crxjs/vite-plugin
└── package.json / tsconfig.json
```

- Domain logic in `src/lib/` imports no `chrome.*` — it is tested with a mocked global.
- Each entry point (background, content, popup, options, side panel) is a separate bundle.

---

## 5. Manifest V3

The manifest is the contract with the browser. Author it in TypeScript so it is type-checked and bundler-generated.

```jsonc
{
  "manifest_version": 3,
  "name": "My Extension",
  "version": "1.2.0",                       // semver — see semver.md
  "minimum_chrome_version": "120",
  "background": { "service_worker": "src/background/sw.ts", "type": "module" },
  "action": { "default_popup": "src/popup/index.html" },
  "options_page": "src/options/index.html",
  "side_panel": { "default_path": "src/sidepanel/index.html" },
  "permissions": ["storage", "activeTab", "scripting"],
  "optional_permissions": ["tabs"],
  "host_permissions": [],                    // prefer activeTab; add hosts only if essential
  "optional_host_permissions": ["https://*/*"],
  "content_scripts": [{
    "matches": ["https://example.com/*"],
    "js": ["src/content/main.ts"],
    "run_at": "document_idle"
  }],
  "declarative_net_request": {
    "rule_resources": [{ "id": "ruleset", "enabled": true, "path": "rules/ruleset.json" }]
  },
  "web_accessible_resources": [
    { "resources": ["injected.js"], "matches": ["https://example.com/*"] }
  ],
  "content_security_policy": {
    "extension_pages": "script-src 'self'; object-src 'self'"
  }
}
```

### Migrating MV2 → MV3
| MV2 | MV3 replacement |
|---|---|
| `background.scripts` / persistent page | `background.service_worker` (ephemeral, `type: module`) |
| `browserAction` / `pageAction` | unified `action` |
| `permissions` mixing APIs + hosts | split: `permissions` (APIs) vs `host_permissions` |
| `webRequest` *blocking* | `declarativeNetRequest` (declarative rules) — observational `webRequest` still allowed |
| `chrome.tabs.executeScript` | `chrome.scripting.executeScript` |
| remote/CDN scripts, inline `<script>` | bundle locally; CSP forbids remote & inline |
| callback APIs | promise-based `chrome.*` / `browser.*` (polyfill) |

---

## 6. Service Worker (background)

The MV3 background is an **event-driven service worker** that the browser starts on demand and **terminates when idle** (≈30 s). It is NOT a persistent page.

```typescript
// src/background/sw.ts — register listeners at TOP LEVEL (synchronously) so
// the worker re-registers them every time it wakes.
chrome.runtime.onInstalled.addListener(() => { /* one-time setup */ });
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  handle(msg, sender).then(sendResponse);
  return true;                       // keep the channel open for the async reply
});
chrome.alarms.onAlarm.addListener((a) => { /* scheduled work */ });
```

Rules:
- **No top-level long-running work and no module-level mutable state you depend on** — it vanishes on termination. Rebuild from `chrome.storage` on wake.
- Use **`chrome.alarms`** (min 30 s period) for scheduling, never `setTimeout`/`setInterval` for anything beyond the current event.
- Return `true` from `onMessage` to use `sendResponse` asynchronously, or return a `Promise` with the polyfill.
- Heavy CPU work: offload to an offscreen document (`chrome.offscreen`) or a `Worker`, since the SW must stay responsive.

---

## 7. Content Scripts & Isolation

Content scripts run in an **isolated world**: they share the page DOM but have a separate JS heap from the page and from the extension. They cannot call most `chrome.*` APIs (only `storage`, `runtime` messaging, `i18n`).

- To reach the page's own JS context, inject into `MAIN` world via `chrome.scripting.executeScript({ world: "MAIN" })` or a `web_accessible_resources` script — and treat that boundary as untrusted.
- Programmatic injection (`chrome.scripting`) with `activeTab` is preferred over broad static `content_scripts` matches.
- For SPA route changes, listen to `chrome.webNavigation.onHistoryStateUpdated` (in the SW) rather than monkey-patching `history.pushState` in the page.
- Never inject secrets or privileged tokens into a content script — it lives in a hostile page.

---

## 8. Messaging

Define typed contracts in one module and validate on receipt — a message can come from any extension page, content script, or (with `externally_connectable`) another extension or site.

```typescript
// src/lib/messaging.ts
export type Message =
  | { type: "GET_SETTINGS" }
  | { type: "SAVE_SETTINGS"; payload: Settings };

export function isMessage(x: unknown): x is Message {
  return !!x && typeof (x as { type?: unknown }).type === "string";
}

// receiver (service worker): validate type AND sender origin (EXT-SEC-02)
chrome.runtime.onMessage.addListener((raw, sender, send) => {
  if (!isMessage(raw)) { send({ ok: false, error: "bad message" }); return; }
  if (sender.id !== chrome.runtime.id) return;        // reject foreign senders
  route(raw).then(send);
  return true;
});
```

- One-shot request/response: `chrome.runtime.sendMessage` / `chrome.tabs.sendMessage(tabId, …)`.
- Long-lived streams: `chrome.runtime.connect({ name })` ports with `onMessage`/`onDisconnect`.
- Content↔background↔popup all go through the same validated router. Never `eval` or `Function()` a payload (violates `secure-coding.md`).

---

## 9. Permissions

Least privilege is a hard gate (EXT-PERM-01) and the main Web Store review criterion.

- **`activeTab`** — temporary access to the focused tab on user gesture; needs no host permission and shows no scary install warning. Default choice for "act on the current page".
- **Host permissions** — only the exact origins you call; prefer specific hosts over `https://*/*`, and `<all_urls>` only with written justification.
- **Optional permissions / optional host permissions** — request at runtime with `chrome.permissions.request()` when the user opts into a feature; this keeps the install prompt minimal.
- Declare scheduling (`alarms`), scripting (`scripting`), storage (`storage`) explicitly. Remove any permission the code no longer uses before each release.

---

## 10. Storage

Use `chrome.storage` (async, available in the service worker) — **never** `localStorage`/`sessionStorage` (unavailable in workers, lost on update).

| Area | Scope | Limit | Use for |
|---|---|---|---|
| `storage.local` | this device | ~10 MB (unlimited with `unlimitedStorage`) | bulk/cache, large data |
| `storage.sync` | synced across the user's devices | ~100 KB, 8 KB/item | small user settings |
| `storage.session` | in-memory, cleared on browser close | ~10 MB | per-session secrets, ephemeral state |

```typescript
await chrome.storage.local.set({ settings });
const { settings } = await chrome.storage.local.get("settings");
chrome.storage.onChanged.addListener((changes, area) => { /* react live */ });
```

Mark secrets `session` and set `setAccessLevel` to keep them out of content scripts. Storage is unencrypted at rest — treat it as plaintext (see `secure-coding.md`).

---

## 11. Extension APIs (current surface)

- **`chrome.action`** — toolbar button: `setBadgeText`, `setIcon`, `onClicked` (fires only when no `default_popup`).
- **`chrome.scripting`** — `executeScript` / `insertCSS` / `registerContentScripts` (replaces MV2 `tabs.executeScript`).
- **`chrome.tabs`** — query/create/update tabs; reading `url`/`title` needs host permission or `activeTab`.
- **`chrome.alarms`** — durable scheduling that survives SW termination (min period 30 s).
- **`chrome.declarativeNetRequest`** — declarative network blocking/redirect/header rules (static `ruleset.json` + dynamic/session rules); the MV3 replacement for blocking `webRequest`.
- **`chrome.sidePanel`** — persistent side-panel UI (Chrome 114+).
- **`chrome.offscreen`** — DOM/audio/clipboard work the worker can't do directly.
- **`chrome.identity`** — OAuth via `launchWebAuthFlow` / `getAuthToken` (auth policy owned by [`oauth.md`](guides://oauth.md)).
- **`chrome.contextMenus`**, **`chrome.notifications`**, **`chrome.commands`** (keyboard shortcuts), **`chrome.i18n`** (localization via `_locales/`).

UI pages (popup, options, side panel) are normal HTML/CSS/JS — author them per [`html.md`](guides://html.md) / [`css.md`](guides://css.md); their `<script>` must be a bundled local file (CSP forbids inline).

---

## 12. Tooling & Build

Use Vite + CRXJS for HMR, manifest generation, and MV3-correct bundling (bundler policy owned by [`vite.md`](guides://vite.md)). Extension binding:

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import { crx } from "@crxjs/vite-plugin";
import manifest from "./src/manifest.config";   // typed MV3 manifest
export default defineConfig({ plugins: [crx({ manifest })] });
```

```bash
npm run dev        # CRXJS dev server with HMR for content/popup
npm run build      # EXT-BUILD-01 → dist/ (manifest.json + bundles)
npm audit --audit-level=high   # EXT-SEC-03
```

Dependency-pinning and CVE policy are owned by [`secure-coding.md`](guides://secure-coding.md); versioning by [`semver.md`](guides://semver.md). Pin `@types/chrome` to the targeted Chrome channel.

---

## 13. Publishing & Cross-Browser

- **Web Store review** rejects undeclared/excessive permissions, remote code, and missing privacy disclosures. Provide a single-purpose description, the minimal permission set with justifications, and a privacy policy if you handle user data.
- **Assets**: 16/48/128 px icons, screenshots (1280×800 or 640×400), `version` bumped per [`semver.md`](guides://semver.md).
- **Cross-browser**: target the WebExtensions standard. Use `webextension-polyfill` for promise-based `browser.*`; for Firefox add `browser_specific_settings.gecko.id` and note that Firefox MV3 still allows blocking `webRequest`. Keep browser-specific manifest keys behind a build flag.

---

## 14. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] EXT-FMT-01 — `prettier --check` clean
- [ ] EXT-LINT-01 — `eslint` clean
- [ ] EXT-TYP-01 — `tsc --noEmit` clean, no `any` on public APIs
- [ ] EXT-TST-01/02 — tests pass, bugs have regression tests
- [ ] EXT-MV3-01 — manifest is V3 with a service worker
- [ ] EXT-PERM-01 — permissions least-privilege, no unjustified `<all_urls>`
- [ ] EXT-SEC-01 — no remote code, CSP `script-src 'self'`, no eval/inline
- [ ] EXT-SEC-02 — every cross-context message validates sender & payload
- [ ] EXT-SEC-03 — `npm audit` 0 high/critical
- [ ] EXT-BUILD-01 — `npm run build` succeeds, `dist/` loads unpacked
- [ ] EXT-DOC-01 — public APIs documented
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Chrome Extension Guidelines**

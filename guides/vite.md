# Vite Development Guidelines
Mandatory standards for Vite build tooling: config, dev server/HMR, plugins, env & modes, Rollup builds, library/SSR, the Environment API. Vite 6.x, Rollup 4, esbuild, Vitest 2.

---
name: vite
title: Vite Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [vite@6, rollup@4, esbuild, vitest@2, "@vitejs/plugin-react", "@vitejs/plugin-vue", "@sveltejs/vite-plugin-svelte"]
requires: []
recommends:
  - typescript
  - javascript
  - reactjs
  - svelte
  - ci-cd
  - performance
  - secure-coding
provides:
  - vite-config
  - vite-plugins
  - hmr
  - vite-build
  - vite-env
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Vite as a build tool — not the languages or frameworks built on it.

---

## 0. Prerequisites & References

Vite is a build tool, not a language or framework. The code you bundle is governed by other guides; fetch them as the task dictates.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`typescript.md`](guides://typescript.md) · [`javascript.md`](guides://javascript.md) — the source languages Vite compiles. Vite uses esbuild for TS *transpilation only* (no type-checking — see VITE-CFG-02).
> - [`reactjs.md`](guides://reactjs.md) · [`svelte.md`](guides://svelte.md) — frameworks consuming Vite via official plugins (§7). (Vue/Solid/Qwik analogous.)
> - [`secure-coding.md`](guides://secure-coding.md) — supply-chain, secrets, CVE policy. **Critical Vite binding:** only `VITE_`-prefixed env vars are bundled into client code (§5) — the #1 Vite secret-leak footgun.
> - [`performance.md`](guides://performance.md) — bundle-size & build-perf budgets bound in §2/§6.
> - [`ci-cd.md`](guides://ci-cd.md) — running `vite build` + `vitest` in the pipeline.

> 📎 **SEE ALSO:** [`nodejs.md`](guides://nodejs.md) · [`nextjs.md`](guides://nextjs.md) *(uses Turbopack/Webpack, not Vite)* · [`env-config.md`](guides://env-config.md)

Vitest shares Vite's config/transform pipeline; this guide owns only the Vite-side test wiring. Test-first discipline (Red-Green-Refactor, coverage, regression-test-before-fix) is owned by [`tdd.md`](guides://tdd.md).

---

## 1. Core Philosophies: VITE-FIRST

Vite-specific principles only. Language/framework/test/security rules come from §0.

- **V**ite owns the toolchain, not your code: one `vite.config.ts` drives dev server, build, and Vitest. Keep app logic out of config.
- **I**nstant feedback: native-ESM dev server with on-demand transform and HMR — never reach for a custom watch/bundle step in dev.
- **T**wo engines, one tool: **esbuild** for dev transform + dependency pre-bundling (speed); **Rollup** for the production build (tree-shaking, chunking). Know which is active. *(Rolldown — a Rust Rollup-compatible bundler — is on Vite's roadmap to unify both; track but don't depend on it yet.)*
- **E**nvironments are explicit: use the Vite 6 **Environment API** to model client/SSR/edge as first-class environments instead of forking config.
- Public surface is opt-in: `VITE_`-prefixed env vars and `import.meta.env` are the *only* values shipped to the browser. Everything else stays server-side.

**Verified Code**: Generated Vite config & build setup MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `VITE-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| VITE-CFG-01 | Config MUST use typed `defineConfig` (object or `({command,mode})=>` fn) | `tsc --noEmit` on config | exit 0 |
| VITE-CFG-02 | Build MUST be gated by a real type-check (esbuild does NOT type-check) (see `typescript.md`) | `tsc -b --noEmit && vite build` | exit 0 |
| VITE-ENV-01 | No secret/server-only value MAY use a `VITE_` prefix or reach `import.meta.env`/`define` (see `secure-coding.md`) | `grep -rE "VITE_.*(SECRET\|KEY\|TOKEN\|PASSWORD)" .env*`; inspect `dist/` | 0 secrets in bundle |
| VITE-ENV-02 | Client env vars MUST be typed via `ImportMetaEnv` in a `vite/client` ref file | type-check; review `vite-env.d.ts` | typed, exit 0 |
| VITE-BUILD-01 | Production build MUST succeed | `vite build` | exit 0 |
| VITE-BUILD-02 | Large/vendor deps SHOULD be split; no eager chunk over budget (see `performance.md`) | `vite build` + report | no warning over `chunkSizeWarningLimit` |
| VITE-HMR-01 | Custom HMR-handled modules MUST clean up via `import.meta.hot.dispose` | review / no leaked listeners | clean |
| VITE-LIB-01 | Library-mode builds MUST externalize peer deps (not bundle them) | inspect bundle for peers | peers external |
| VITE-DEP-01 | Lockfile committed & in sync; CI uses `npm ci`/`pnpm i --frozen-lockfile` (see `secure-coding.md`) | `npm ci` / `pnpm i --frozen-lockfile` | no drift |
| VITE-SEC-01 | 0 high/critical CVEs in deps **and plugins** (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| VITE-TST-01 | Features test-first via Vitest (see `tdd.md`) | `vitest run` | exit 0, 0 skips |

> Forbidden: shipping a build with no separate type-check (VITE-CFG-02); any `VITE_`-prefixed secret; bundling peer deps in library mode; committing without a synced lockfile.

---

## 3. `vite.config.ts` Structure

Single source of truth. Use the function form to branch on `command` (`serve` | `build`) and `mode`, and `loadEnv` to read `.env*` (note: `loadEnv` returns *all* vars; only `VITE_`-prefixed ones are exposed to client code — see §5).

```typescript
import { defineConfig, loadEnv } from 'vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '');     // server-side only
  return {
    plugins: [/* framework + tooling plugins, ordered (§4) */],
    resolve: {
      alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) }, // §6
    },
    server: {                                         // dev only (§3A)
      port: 5173,
      proxy: { '/api': { target: env.API_ORIGIN, changeOrigin: true } },
    },
    build: { /* §6 */ },
    optimizeDeps: { /* §6C */ },
    test: { /* Vitest — §8 */ },
  };
});
```

- Prefer `node:` protocol imports and `import.meta.url` over `__dirname` (config is ESM; set `"type":"module"` in `package.json`).
- Keep secrets in `process.env`/`env`, never in `define` (§5).
- A standalone `vitest.config.ts` may extend the Vite config via `mergeConfig`; do not duplicate aliases/plugins.

### A. Dev server & HMR

The dev server serves source over native ESM, transforming each module on request (no bundle). HMR pushes updates over a WebSocket and patches modules in place — accept and dispose explicitly when you own a module's state:

```typescript
if (import.meta.hot) {
  import.meta.hot.accept((newMod) => { /* re-apply with newMod */ });
  import.meta.hot.dispose(() => { /* tear down timers/listeners — VITE-HMR-01 */ });
}
```

- Framework plugins wire component-level HMR for you; hand-roll only for non-framework singletons (stores, web workers).
- `server.proxy` avoids CORS in dev; `server.host: true` exposes on LAN; `server.https` for secure-context features.
- A failed `accept` triggers a full reload — design disposal to be idempotent.

---

## 4. Plugins

Vite plugins are a **superset of Rollup plugins** plus Vite-only hooks. Plugin order matters; `enforce: 'pre' | 'post'` and `apply: 'serve' | 'build'` control placement.

```typescript
import type { Plugin } from 'vite';

function virtualEnvBanner(): Plugin {
  const id = 'virtual:banner';
  return {
    name: 'app:banner',
    enforce: 'pre',
    config(config, { command }) { /* mutate/return partial config */ },
    configResolved(resolved) { /* read final config */ },
    resolveId(source) { return source === id ? '\0' + id : null; },
    load(resolvedId) { return resolvedId === '\0' + id ? `export const v=1` : null; },
    transform(code, fileId) { /* per-module transform */ },
    configureServer(server) { /* dev middleware */ },
    handleHotUpdate(ctx) { /* customize HMR */ },
  };
}
```

- Vite-only hooks: `config`, `configResolved`, `configureServer`, `transformIndexHtml`, `handleHotUpdate`. Universal Rollup hooks (`resolveId`/`load`/`transform`/`buildStart`) run in both dev and build.
- Use the **virtual module** convention (`virtual:x` resolved to `\0virtual:x`) for generated modules.
- Audit third-party plugins for permissions/telemetry as part of VITE-SEC-01.

---

## 5. Env Variables, Modes & the Public-Exposure Footgun

**The defining Vite security rule:** Vite statically replaces `import.meta.env.VITE_*` at build time and inlines those values into client bundles. **Only `VITE_`-prefixed vars are exposed; nothing else is.** Putting a secret behind a `VITE_` prefix ships it to every browser — VITE-ENV-01.

```bash
# .env, .env.[mode], .env.local (gitignore *.local)
VITE_API_URL=https://api.example.com   # ✅ public, safe to inline
API_DB_PASSWORD=…                       # ✅ server-only — NO VITE_ prefix, never reaches client
VITE_STRIPE_SECRET=…                    # ❌ NEVER — this leaks the secret into dist/
```

- **Modes** select the `.env.[mode]` file: `vite` → `development`, `vite build` → `production`; override with `--mode staging`.
- Built-ins: `import.meta.env.MODE | DEV | PROD | BASE_URL | SSR`.
- `define:` performs raw text substitution — only for non-secret constants (`__APP_VERSION__`), never secrets (same exposure as `VITE_`).
- Type the surface (VITE-ENV-02):

```typescript
/// <reference types="vite/client" />
interface ImportMetaEnv { readonly VITE_API_URL: string; }
interface ImportMeta { readonly env: ImportMetaEnv; }
```

---

## 6. Production Build (Rollup), Splitting & Aliases

`vite build` runs **Rollup** (esbuild only minifies by default). Tree-shaking, chunking, and asset hashing happen here.

```typescript
build: {
  target: 'baseline-widely-available',  // or 'esnext' for modern-only
  sourcemap: true,
  rollupOptions: {
    output: {
      manualChunks(id) {                 // function form > object for large apps
        if (id.includes('node_modules')) return 'vendor';
      },
      entryFileNames: '[name]-[hash].js',
      chunkFileNames: 'chunks/[name]-[hash].js',
      assetFileNames: 'assets/[name]-[hash][extname]',
    },
  },
  chunkSizeWarningLimit: 1000,           // VITE-BUILD-02 budget (see performance.md)
}
```

- **Code splitting** is automatic at dynamic `import()` boundaries; route/feature lazy-loading needs no config. Use `manualChunks` to stabilize long-term vendor caching, not to fight the splitter.
- Tree-shaking relies on ESM + correct `sideEffects` in deps' `package.json`.
- Analyze with `rollup-plugin-visualizer`; gate bundle budgets in CI (see `ci-cd.md`, `performance.md`).

### A. Path aliases

Declare in `resolve.alias` **and** mirror in `tsconfig.json` `paths` (Vite resolves at bundle time, `tsc` at type-check time — both must agree or VITE-CFG-02 fails).

### B. Asset handling

- Imported assets return resolved URLs and are hashed/copied; `?url` (explicit URL), `?raw` (string), `?worker` / `?worker&inline` (web workers), `?react` etc. via plugins.
- `public/` is copied verbatim (no hashing, no processing) — reference by absolute `/path`.
- CSS Modules (`*.module.css`), PostCSS, and Lightning CSS are built in.

### C. Dependency pre-bundling (`optimizeDeps`)

esbuild pre-bundles CommonJS/UMD deps into ESM and collapses many-file packages into one request (dev speed). `include` deps discovered late (dynamic imports); `exclude` deps that ship native ESM and should HMR. Stale cache → `vite --force`.

### D. Library mode

```typescript
build: {
  lib: { entry: 'src/index.ts', formats: ['es', 'cjs'], fileName: 'my-lib' },
  rollupOptions: { external: ['react', 'react-dom'] },  // VITE-LIB-01: never bundle peers
}
```

Externalize every peer/runtime dep; emit `.d.ts` via `vite-plugin-dts`; declare `exports`/`types` in `package.json`.

---

## 7. SSR & the Environment API (Vite 6)

Vite 6 generalizes dev/SSR/client/edge into **Environments**, each with its own module graph and resolution — replacing the older single-SSR special-casing.

```typescript
export default defineConfig({
  environments: {
    client: { build: { outDir: 'dist/client' } },
    ssr: { build: { outDir: 'dist/server', ssr: 'src/entry-server.ts' } },
  },
});
```

- SSR dev primitives: `server.ssrLoadModule(url)` (transform-on-demand, HMR-aware) and `server.ssrFixStacktrace(e)`; serve via `server.middlewareMode: true` behind your own Node/Express handler.
- `ssr.noExternal` forces deps through Vite's transform (needed for non-ESM/CSS-importing libs); `ssr.external` keeps them in Node's resolution.
- `import.meta.env.SSR` branches isomorphic code. For full SSR frameworks prefer Nuxt/SvelteKit/Astro/Remix over hand-rolling.

---

## 8. Vitest Wiring

Vitest reuses Vite's transform/resolve/plugins, so tests see the same aliases and env. Configure under `test` in the Vite config (or `vitest.config.ts` via `mergeConfig`). Test discipline itself is owned by [`tdd.md`](guides://tdd.md); coverage thresholds bind that policy.

```typescript
test: {
  environment: 'jsdom',                    // 'node' for non-DOM code
  setupFiles: './src/test/setup.ts',
  coverage: { provider: 'v8', thresholds: { lines: 100 } }, // gate per tdd.md
}
```

Browser-mode and `@vitest/ui` are available; in-source testing via `import.meta.vitest` for tiny utilities.

---

## 9. Footguns

- Secret behind a `VITE_` prefix → leaked to client (VITE-ENV-01). Default to **no** prefix.
- Assuming `vite build` type-checks → it does not; esbuild strips types blindly. Always run `tsc`/`vue-tsc`/`svelte-check` (VITE-CFG-02).
- Editing config without restart → server/build options are read at startup; restart after changing them.
- Importing Node built-ins into client code → fails in browser; gate with `import.meta.env.SSR` or split entries.
- Stale pre-bundle after dependency change → `vite --force`.
- `public/` asset referenced by relative path → breaks under a non-root `base`; use absolute `/`.
- Overusing `manualChunks` → can defeat the splitter and inflate the entry; measure first.

---

## 10. Quick Reference

```bash
vite                       # dev server + HMR (mode=development)
vite --port 3000 --host    # custom port, expose on LAN
vite --force               # ignore optimizeDeps cache
vite build                 # Rollup production build (mode=production)
vite build --mode staging  # build with .env.staging
vite preview               # serve dist/ locally
vitest run                 # tests once   |  vitest  (watch)
vitest run --coverage      # coverage gate (see tdd.md)
npm audit --audit-level=high  # VITE-SEC-01
```

---

## 11. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] VITE-CFG-01 — config uses typed `defineConfig`
- [ ] VITE-CFG-02 — separate type-check gates the build (esbuild doesn't type-check)
- [ ] VITE-ENV-01 — no secret carries a `VITE_` prefix / reaches `define`; `dist/` clean
- [ ] VITE-ENV-02 — client env typed via `ImportMetaEnv`
- [ ] VITE-BUILD-01 — `vite build` exits 0
- [ ] VITE-BUILD-02 — chunks within budget; vendor/large deps split
- [ ] VITE-HMR-01 — custom HMR modules dispose cleanly
- [ ] VITE-LIB-01 — library mode externalizes peers
- [ ] VITE-DEP-01 — lockfile committed & in sync; CI frozen install
- [ ] VITE-SEC-01 — 0 high/critical CVEs in deps and plugins
- [ ] VITE-TST-01 — features test-first; `vitest run` green

---
**End of Vite Guidelines**

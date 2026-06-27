# Nix Flakes Development Guidelines
Mandatory standards for reproducible, hermetic Nix flakes: pinned inputs, pure builds, devShells, multi-platform derivations. Nix 2.x with flakes, nixpkgs, flake-utils/flake-parts, nixpkgs-fmt, statix, deadnix, direnv.

---
name: nix-flake
title: Nix Flakes Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: tooling
tools: [nix@2.x, flakes, nixpkgs, flake-utils, flake-parts, nixpkgs-fmt, statix, deadnix, nix-direnv]
requires: []
recommends:
  - ci-cd
  - secure-coding
  - dockerfile
provides:
  - nix-flakes
  - reproducible-builds
  - devshells
  - nix-derivations
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Nix flakes.

---

## 0. Prerequisites & References

This guide owns Nix-specific reproducibility and packaging. Fetch these when the task touches their concern; their rules are assumed and not repeated here.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`secure-coding.md`](guides://secure-coding.md) — supply-chain, CVE policy, secrets. *(Nix binding: `flake.lock` pins the full dependency closure by hash; scan with `vulnix`; never inline secrets in `flake.nix`.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline policy. *(Nix binding: `cachix/install-nix-action`, `nix flake check`, binary caches.)*
> - [`dockerfile.md`](guides://dockerfile.md) — container/image policy. *(Nix binding: `pkgs.dockerTools.buildLayeredImage` instead of a Dockerfile.)*

> 📎 **SEE ALSO:** [`tdd.md`](guides://tdd.md) *(flake `checks` are the test harness)* · [`semver.md`](guides://semver.md) · [`env-config.md`](guides://env-config.md) · [`bash.md`](guides://bash.md) *(devShell hooks, `writeShellApplication` scripts)*

---

## 1. Core Philosophies: NIX-FIRST

Nix-specific principles only. Test-first (`checks`), security, and CI policy come from §0.

- **N**o impurity: pure, hermetic, sandboxed builds — no `<nixpkgs>`, no `$HOME`, no network or clock access inside a build.
- **I**mmutable inputs: every dependency pinned by hash in a committed `flake.lock`; reproducibility is the default, not an effort.
- **X**-platform: one flake serves `x86_64-linux`, `aarch64-linux`, `x86_64-darwin`, `aarch64-darwin` via `flake-utils`/`flake-parts`.
- **F**lakes only: no legacy `default.nix`/`shell.nix` channels; `nix-command flakes` experimental features enabled.
- **I**dentical environments: `nix develop` (and `direnv`) give every machine the same toolchain; `devShells` are the contract.
- **R**eference, don't fetch impurely: all external sources fetched at eval time with a fixed `sha256`/`hash`.
- **S**mall closures: minimal runtime dependencies; verify with `nix path-info -rsSh`.
- **T**est via checks: `nix flake check` is the single gate — build, format, lint, tests, and reproducibility all live there.

**Verified Code**: Agent-generated flakes MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `NIX-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| NIX-STRUCT-01 | Flake MUST evaluate and expose declared outputs on all `systems` | `nix flake check --all-systems` | exit 0 |
| NIX-DEP-01 | `flake.lock` MUST be committed and current | `nix flake metadata`; `git status flake.lock` | locked, no drift |
| NIX-DEP-02 | All inputs MUST be pinned (branch/rev/tag), never floating unpinned refs in prod | review of `inputs` | each input pinned |
| NIX-PURE-01 | Builds MUST be pure — no `<nixpkgs>`, `$HOME`, network, or `$(date)` in build phases | `nix build --pure-eval` | exit 0 |
| NIX-PURE-02 | All external sources MUST have a fixed `sha256`/`hash` | `grep -rL 'sha256\|hash' fetch sites`; build under sandbox | no unfixed fetch |
| NIX-REPRO-01 | Build MUST be deterministic (same inputs → same output) | `nix build --rebuild` then compare `nix hash path` | hashes match |
| NIX-FMT-01 | Nix code MUST be formatted | `nix fmt -- --check .` (nixpkgs-fmt/treefmt) | no diff |
| NIX-LINT-01 | No anti-patterns or dead code | `statix check .` && `deadnix --fail .` | exit 0 |
| NIX-TST-01 | `checks` MUST cover build + tests and pass (test harness; see `tdd.md`) | `nix flake check` | exit 0 |
| NIX-SHELL-01 | A usable `devShells.default` MUST be provided | `nix develop -c true` | exit 0 |
| NIX-SEC-01 | 0 known CVEs in the runtime closure (see `secure-coding.md`) | `vulnix ./result` | 0 vulnerabilities |
| NIX-SEC-02 | No secrets committed in `flake.nix`/inputs (see `secure-coding.md`) | `gitleaks detect` | 0 findings |

> **Forbidden**: shipping a flake that fails any gate above; impure inputs (`import <nixpkgs>`); gitignoring `flake.lock`; unfixed `fetchurl`/`fetchgit` hashes; network access in `buildPhase`; hardcoded absolute paths (`/usr/bin/...`) instead of store paths; in-place `sed -i` on sources (use `patches`/`substituteInPlace`).

---

## 3. Verification Protocol

Run, in order, before presenting a flake. Fix → re-run until every gate is green.

```bash
nix fmt -- --check .                 # NIX-FMT-01
nix run nixpkgs#statix -- check .    # NIX-LINT-01
nix run nixpkgs#deadnix -- --fail .  # NIX-LINT-01
nix flake check --all-systems        # NIX-STRUCT-01 / NIX-TST-01
nix build --pure-eval                # NIX-PURE-01
nix develop -c true                  # NIX-SHELL-01
nix run nixpkgs#vulnix -- ./result   # NIX-SEC-01
git status flake.lock                # NIX-DEP-01 (must be committed)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic flake layout. Keep `flake.nix` thin; move package/shell/check expressions into `nix/` and `callPackage` them.

```
project/
├── flake.nix              # inputs + outputs (thin orchestrator)
├── flake.lock             # locked inputs — COMMIT THIS
├── nix/
│   ├── packages/          # derivations (callPackage targets)
│   ├── shells/            # devShell definitions
│   ├── checks/            # custom checks
│   ├── modules/           # NixOS / home-manager modules
│   └── overlays/          # package overlays
├── src/                   # application source (see its language guide)
├── tests/                 # test inputs run by checks (see tdd.md)
├── .envrc                 # `use flake` for direnv — COMMIT THIS
└── .gitignore             # ignores result, result-*, .direnv/
```

- Keep `flake.nix` declarative; logic goes in `callPackage`'d files.
- One `flake.lock`; sub-flakes only for genuinely independent release units.

---

## 5. Nix Flake Specifics

The unique value of this guide.

### A. Flake anatomy (inputs / outputs)

A flake is an attrset with `description`, `inputs`, and `outputs`. `outputs` is a function of the resolved inputs (plus `self`).

```nix
{
  description = "My project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";   # pinned release branch
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        packages.default  = pkgs.callPackage ./nix/packages { };  # nix build
        devShells.default = pkgs.callPackage ./nix/shells   { };  # nix develop
        apps.default      = { type = "app"; program = "${self.packages.${system}.default}/bin/myapp"; };  # nix run
        checks            = { build = self.packages.${system}.default; };  # nix flake check
        formatter         = pkgs.nixpkgs-fmt;                      # nix fmt
      });
}
```

Standard output attributes: `packages.<system>.<name>`, `devShells`, `apps`, `checks`, `formatter`, `overlays`, `nixosModules`, `nixosConfigurations`, `homeConfigurations`, `templates`, `lib`.

### B. Nix language essentials

Lazy, pure, functional. The handful of constructs that trip people up:

```nix
let x = 1; y = x + 1; in x + y            # let … in: local bindings
{ a = 1; b = 2; }                          # attrset
rec { a = 1; b = a + 1; }                  # rec: self-referential attrset
f = { a, b ? 10, ... }: a + b              # function w/ default arg + ellipsis
pkgs.lib.optionals cond [ x ]              # [] when cond false, else [ x ]
pkgs.lib.optionalString cond "text"        # "" when false
inherit system;                            # = `system = system;`
inherit (pkgs) curl jq;                    # pull attrs from a set
"${pkgs.curl}/bin/curl"                    # antiquotation → store path
./src                                       # path literal (copied to store)
```

Footguns: `rec` infinite recursion; `with pkgs;` shadowing/opacity (prefer explicit `inherit`); string `"${drv}"` forces a build dependency; `//` merges attrsets shallowly (right wins).

### C. Derivations & packages

Prefer the highest-level builder that fits. `mkDerivation` is the base; language ecosystems add `buildPythonApplication`, `buildGoModule`, `buildRustPackage`, `buildNpmPackage`, etc. — use the language guide for ecosystem specifics, the flake guide for the wiring.

```nix
# Shell wrapper with runtime deps on PATH (preferred over writeShellScriptBin)
pkgs.writeShellApplication {
  name = "myapp";
  runtimeInputs = [ pkgs.curl pkgs.jq ];   # placed on PATH; survives gc
  text = ''curl -s "$1" | jq .'';
}

# Generic derivation
pkgs.stdenv.mkDerivation {
  pname = "myapp"; version = "1.0.0";       # fixed version, never $(date)
  src = ./.;
  nativeBuildInputs = [ pkgs.pkg-config ];  # build-time tools
  buildInputs = [ pkgs.openssl ];           # runtime/link deps
  buildPhase = "make";
  installPhase = "make install PREFIX=$out";
  patches = [ ./fix.patch ];                # never sed -i the source
}
```

`nativeBuildInputs` = host tools; `buildInputs` = target libs; `runtimeInputs` (writeShellApplication) = PATH at runtime. Fetch external data purely:

```nix
data = pkgs.fetchurl { url = "https://ex.com/d.json"; sha256 = "sha256-…"; };
# then in a phase: cp ${data} ./data.json     (no curl in buildPhase)
```

### D. devShells

The reproducible dev environment contract. Inherit a package's build inputs with `inputsFrom`, add tooling via `packages`/`buildInputs`, set env vars as attrs, run setup in `shellHook`.

```nix
devShells.default = pkgs.mkShell {
  inputsFrom = [ self.packages.${system}.default ];
  packages = with pkgs; [ git just nixpkgs-fmt statix ];
  RUST_LOG = "debug";                       # plain attr → env var
  shellHook = ''echo "dev shell ready: $(nix --version)"'';
};
```

Provide multiple named shells (`devShells.python`, `devShells.ci`) when a repo spans toolchains. Load automatically with direnv: `.envrc` containing `use flake` (+ `nix-direnv` for caching).

### E. nixpkgs, overlays & overrides

`import nixpkgs { inherit system; }` gives the package set. Customize without forking via overlays (`final: prev:`) and per-package `overrideAttrs`/`.override`.

```nix
overlays.default = final: prev: {
  myapp = prev.callPackage ./nix/packages { };
  nodejs = prev.nodejs_20;                                   # change the default
  somePkg = prev.somePkg.overrideAttrs (old: {              # patch attrs
    version = "1.2.3-patched";
    src = prev.fetchurl { url = "…"; hash = "sha256-…"; };
  });
};
# consume: pkgs = import nixpkgs { inherit system; overlays = [ self.overlays.default ]; };
```

`allowUnfree`/`allowInsecure` must be set explicitly in `config` — `nix flake check` fails otherwise (a feature, not a bug).

### F. Multi-platform & flake-utils / flake-parts

`flake-utils.lib.eachDefaultSystem (system: { … })` maps outputs over the four default systems; `eachSystem [ … ]` for an explicit list. For larger flakes, `flake-parts` gives modular `perSystem` composition and a module system:

```nix
flake-parts.lib.mkFlake { inherit inputs; } {
  systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
  imports = [ inputs.treefmt-nix.flakeModule ./nix/packages ];
  perSystem = { pkgs, self', ... }: {
    packages.default = pkgs.callPackage ./nix/packages { };
    treefmt.config.programs.nixpkgs-fmt.enable = true;
  };
}
```

Branch on platform with `pkgs.stdenv.isLinux`/`isDarwin` and `lib.optionals`. Cross-compile via `import nixpkgs { localSystem; crossSystem; }`.

### G. Pinning & updating inputs

`flake.lock` records the exact rev + `narHash` of every input (transitively). Commit it.

```bash
nix flake update                 # update all inputs, rewrite lock
nix flake update nixpkgs         # update one input only
nix flake metadata               # show current locked revisions
git diff flake.lock              # review before committing an update
```

Pin nixpkgs to a **stable release branch** (`nixos-24.11`) in production, not `nixos-unstable`. Dedupe transitive nixpkgs with `inputs.X.inputs.nixpkgs.follows = "nixpkgs"` to keep the closure small and consistent.

### H. checks — the test harness

`checks.<system>.<name>` are derivations that must build successfully. They are the §2 / `tdd.md` gate: build the package, run tests, verify format/lint, assert reproducibility. A check passes iff its derivation builds.

```nix
checks = {
  build  = self.packages.${system}.default;                    # build must succeed
  format = pkgs.runCommand "fmt" {} ''
    ${pkgs.nixpkgs-fmt}/bin/nixpkgs-fmt --check ${self}; touch $out'';
  tests  = pkgs.runCommand "tests" { buildInputs = [ self.packages.${system}.default ]; } ''
    pytest ${./tests}; touch $out'';
};
```

Full-system integration uses `pkgs.nixosTest { nodes.machine = …; testScript = ''…''; }` to boot a VM and assert service behavior.

### I. NixOS & home-manager modules (briefly)

Expose reusable system/user config as modules so consumers `imports` them; ship runnable systems as `nixosConfigurations`.

```nix
nixosModules.default = { config, lib, pkgs, ... }: {
  options.services.myapp.enable = lib.mkEnableOption "myapp";
  config = lib.mkIf config.services.myapp.enable {
    systemd.services.myapp.serviceConfig.ExecStart = "${self.packages.${pkgs.system}.default}/bin/myapp";
  };
};
nixosConfigurations.host = nixpkgs.lib.nixosSystem {
  system = "x86_64-linux";
  modules = [ self.nixosModules.default ./hosts/host.nix ];
};
homeConfigurations."user@host" = home-manager.lib.homeManagerConfiguration { … };
```

### J. Binary caches

Caches turn rebuilds into downloads — essential for CI and teams. Configure trusted substituters + public keys; push with Cachix or `nix copy`.

```nix
nixConfig = {                                   # per-flake (prompts to trust)
  extra-substituters = [ "https://my-project.cachix.org" ];
  extra-trusted-public-keys = [ "my-project.cachix.org-1:…" ];
};
```

```bash
cachix use my-project                            # add cache to nix.conf
nix build --json | jq -r '.[].outputs.out' | cachix push my-project
```

### K. Containers without a Dockerfile

Build OCI images directly from derivations — image policy is owned by [`dockerfile.md`](guides://dockerfile.md); the Nix binding is `dockerTools`:

```nix
packages.container = pkgs.dockerTools.buildLayeredImage {
  name = "myapp"; tag = "latest";
  contents = [ self.packages.${system}.default ];
  config.Cmd = [ "/bin/myapp" ];
};
# nix build .#container && docker load < result   (reproducible, minimal layers)
```

### L. Common footguns

- `import <nixpkgs> {}` → impure; use the flake `inputs.nixpkgs`.
- Uncommitted/gitignored `flake.lock` → non-reproducible; commit it.
- `$(date)`/`$HOME`/`curl` in `buildPhase` → impure; fetch at eval time, use fixed version strings.
- Hardcoded `/usr/bin/...` → use `${pkgs.foo}/bin/...`.
- `writeShellScriptBin` for a script needing tools → use `writeShellApplication` with `runtimeInputs`.
- Single-system outputs (`packages.x86_64-linux.default = …`) → wrap with `eachDefaultSystem`.

---

## 6. Tooling & Dependencies

Inputs ARE the dependencies; `flake.lock` is the lockfile. Supply-chain policy → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md).

```bash
nix flake update                 # update all inputs (NIX-DEP-01)
nix flake update nixpkgs         # update one input
nix flake metadata               # show locked revisions
nix path-info -rsSh ./result     # audit closure size / contents
nix why-depends .#default nixpkgs#openssl   # explain a closure entry
nix run nixpkgs#vulnix -- ./result          # NIX-SEC-01: CVE scan
```

Commit `flake.lock`. Pin nixpkgs to a stable branch in production; `follows` transitive nixpkgs to one revision.

---

## 7. Quick Reference

```bash
nix build                        # build packages.default
nix run .#app -- args            # run an app output
nix develop                      # enter devShells.default
nix flake check --all-systems    # all checks, every system (test gate)
nix fmt                          # format (formatter output)
nix flake show                   # list outputs
nix flake metadata               # inputs + locked revisions
nix flake init -t templates#x    # scaffold from a template
nix repl .#                      # REPL with the flake loaded
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] NIX-STRUCT-01 — `nix flake check --all-systems` passes
- [ ] NIX-DEP-01 — `flake.lock` committed and current
- [ ] NIX-DEP-02 — every input pinned (branch/rev/tag)
- [ ] NIX-PURE-01 — `nix build --pure-eval` succeeds
- [ ] NIX-PURE-02 — all external sources have fixed hashes
- [ ] NIX-REPRO-01 — `nix build --rebuild` reproduces identical output hash
- [ ] NIX-FMT-01 — `nix fmt -- --check` clean
- [ ] NIX-LINT-01 — `statix` clean, `deadnix --fail` clean
- [ ] NIX-TST-01 — `checks` build + tests pass
- [ ] NIX-SHELL-01 — `nix develop` provides a working shell
- [ ] NIX-SEC-01/02 — `vulnix` 0 CVEs, no secrets committed
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Nix Flakes Development Guidelines**

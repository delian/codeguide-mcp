# Haskell Development Guidelines
Mandatory coding standards for Haskell: total, purely-typed, property-tested, leak-free. GHC 9.12+, Cabal 3.12+, HLS, HSpec, QuickCheck, HLint, fourmolu.

---
name: haskell
title: Haskell Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [ghc@9.12, cabal@3.12, hls, hspec, quickcheck, hlint, fourmolu, haddock]
requires:
  - tdd
  - error-handling
recommends:
  - designpatterns
  - parallelism
  - comments
  - secure-coding
provides:
  - purity-laziness
  - typeclasses
  - monads
  - property-testing
  - total-functions
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Haskell.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Haskell code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix. *(Haskell binding: `cabal test`; property-based tests via QuickCheck are the default for pure logic.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Haskell binding: `Maybe` for absence, `Either`/`ExceptT` for recoverable failures, `Control.Exception` only at IO boundaries; never `error`/`undefined`/`head` in production.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`designpatterns.md`](guides://designpatterns.md) — typeclass/FP patterns (Functor/Monad, Reader/State, tagless-final, free monads).
> - [`parallelism.md`](guides://parallelism.md) — concurrency. *(binding: `STM`/`TVar`, `async`, `Control.Parallel` `par`/`pseq`, the `-threaded -with-rtsopts=-N` RTS.)*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: Haddock `-- |` on every exported symbol)*.
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, CVE policy *(binding: `cabal-audit` / `cabal outdated` against the HSEC advisory DB)*.

> 📎 **SEE ALSO:** [`hexagonal.md`](guides://hexagonal.md) (ports as records-of-functions or typeclasses) · [`performance.md`](guides://performance.md) · [`pre-commit.md`](guides://pre-commit.md) · [`ci-cd.md`](guides://ci-cd.md) · [`semver.md`](guides://semver.md) (Haskell follows PVP).

---

## 1. Core Philosophies: HASKELL-FIRST

Haskell-specific principles only. TDD, error strategy, concurrency, and patterns come from §0.

- **Make invalid states unrepresentable**: encode invariants in types (newtypes, sum types, GADTs, phantom/type-level naturals) so the compiler rejects bad data — this replaces most defensive runtime checks.
- **Purity by default, effects explicit**: keep logic pure; push `IO` to the edges. Model effects with an explicit stack (`ReaderT`/`ExceptT`) or an effect system (`effectful`), never ad-hoc global state.
- **Total functions only**: every function terminates and is defined on all inputs of its type. No partial heads/`fromJust`/incomplete patterns — `-Wincomplete-patterns` is an error.
- **Laziness with deliberate strictness**: exploit laziness for composition, but use `StrictData`, `BangPatterns`, and `foldl'` to prevent space leaks in accumulators and long-lived structures.
- **Property-first testing**: pure functions are specified by QuickCheck properties (laws, round-trips, invariants), not just example rows (see `tdd.md`).
- **Disciplined extensions**: build on `GHC2021`/`GHC2024`; enable further extensions explicitly in `default-extensions`, never per-file scattershot.

**Verified Code**: Agent-generated Haskell MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `HS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| HS-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `cabal test` | exit 0, 0 pending |
| HS-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `cabal test` | failing→passing |
| HS-TST-03 | Pure logic MUST have QuickCheck properties, not only examples (see `tdd.md`) | `cabal test` | properties pass ≥ default 100 cases |
| HS-FMT-01 | Code MUST be formatted | `fourmolu --mode check $(git ls-files '*.hs')` | no diff |
| HS-LINT-01 | Linter MUST pass clean | `hlint src test` | exit 0, no hints |
| HS-WARN-01 | Code MUST build with `-Wall -Werror`, no warnings | `cabal build --ghc-options=-Werror` | exit 0 |
| HS-TYP-01 | Every top-level binding MUST have an explicit type signature | `-Wmissing-signatures` (in `-Wall`) | 0 warnings |
| HS-TOTAL-01 | No partial functions; pattern matches MUST be exhaustive | `-Wincomplete-patterns -Wincomplete-uni-patterns` as errors | 0 warnings |
| HS-ERR-01 | No `error`/`undefined`/partial `head`/`fromJust` in non-test code; use `Maybe`/`Either` (see `error-handling.md`) | `hlint` + grep | none found |
| HS-DOC-01 | Exported symbols MUST have Haddock (see `comments.md`) | `cabal haddock` | builds, no missing-doc warnings |
| HS-SEC-01 | 0 high/critical advisories in deps (see `secure-coding.md`) | `cabal-audit` | 0 high/critical |
| HS-DEP-01 | `cabal.project.freeze` committed & resolvable | `cabal build --offline` (post-fetch) | builds |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); `error`/`undefined`/`unsafePerformIO` in production; orphan instances; lazy `Data.List`/`String`-based IO for large data (use `Data.Text`/strict bytestring); disabling warnings to pass `-Werror`; per-file `LANGUAGE` pragmas for project-wide extensions.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
fourmolu --mode check $(git ls-files '*.hs')     # HS-FMT-01
hlint src test                                    # HS-LINT-01, HS-ERR-01
cabal build --ghc-options=-Werror                 # HS-WARN-01, HS-TYP-01, HS-TOTAL-01
cabal test                                         # HS-TST-01/02/03
cabal haddock                                      # HS-DOC-01
cabal-audit                                        # HS-SEC-01
cabal build --offline                              # HS-DEP-01 (freeze in sync)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic multi-package Cabal layout. Architectural *principles* (dependency direction, ports/adapters) are owned by [`hexagonal.md`](guides://hexagonal.md); in Haskell a port is a **record-of-functions** or **typeclass**, injected at the edge, so the domain stays pure and IO-free.

```
project/
├── cabal.project                # solver config, packages, source-repos
├── cabal.project.freeze         # committed lockfile (HS-DEP-01)
├── src/
│   ├── Domain/                  # pure types & logic — no IO imports
│   ├── Application/             # use cases over ports (records-of-fns/typeclasses)
│   └── Infrastructure/          # adapter impls (DB/HTTP) — the only IO
├── app/Main.hs                  # composition root; wires adapters into ports
├── test/                        # HSpec + QuickCheck, mirrors src/ (see tdd.md)
├── bench/                       # criterion benchmarks
├── <project>.cabal              # targets, deps, extensions, ghc-options
├── fourmolu.yaml / .hlint.yaml
└── README.md
```

- Group by domain; one module = one responsibility; avoid `Utils` dumping grounds.
- Use `hs-source-dirs` + a `common` stanza so every target shares extensions and warnings.
- No orphan instances: define a typeclass instance in the module of either the class or the type.

---

## 5. Haskell Specifics

The unique value of this guide.

### A. Toolchain — GHCup + Cabal

```bash
ghcup install ghc 9.12 && ghcup install cabal 3.12 && ghcup install hls   # toolchain
cabal update                        # refresh Hackage index
cabal build all                     # build every target
cabal test                          # run HSpec/QuickCheck suites
cabal repl                          # GHCi with project loaded
cabal freeze                        # write cabal.project.freeze (HS-DEP-01)
cabal haddock                       # generate API docs
```
Stack is acceptable where a project already standardizes on it (`stack build/test`, `stack.yaml` resolver pins GHC + snapshot); do not mix Stack and Cabal workflows in one repo.

### B. The type system — make invalid states unrepresentable

```haskell
newtype Email = Email Text deriving stock (Eq, Show)
mkEmail :: Text -> Maybe Email                 -- smart constructor; export type, not ctor
mkEmail t | "@" `T.isInfixOf` t = Just (Email t) | otherwise = Nothing

data Shape = Circle Double | Rect Double Double  -- sum type: total `area` is exhaustive

-- GADT + type-level Nat: length tracked in the type ⇒ vhead is total
data Vec (n :: Nat) a where
  VNil  :: Vec 0 a
  VCons :: a -> Vec n a -> Vec (n + 1) a
vhead :: Vec (n + 1) a -> a                      -- empty Vec rejected at compile time
vhead (VCons x _) = x
```
Reach for type classes (with `InstanceSigs`), kinds/`DataKinds`, type families, and phantom types to lift invariants into types. Prefer `deriving stock`/`newtype`/`via` (explicit `DerivingStrategies`) over implicit derivation.

### C. Functor / Applicative / Monad & Maybe/Either

Errors and effects are values; chain them in `do`-notation. Strategy is owned by [`error-handling.md`](guides://error-handling.md); the Haskell binding:

```haskell
parse :: Text -> Either Error Order            -- Either = recoverable failure
parse input = do
  parsed    <- parseInput input                -- short-circuits on Left
  validated <- validate parsed
  pure (toOrder validated)

results <- traverse fetchUser ids              -- Applicative/Traversable over effects
total   = foldl' (+) 0 amounts                 -- strict fold, no space leak
```
Use `Maybe` for absence, `Either e` for typed recoverable errors, and `Control.Exception` (`bracket`, `try`) only at IO boundaries for resource safety. Never throw from pure code.

### D. Monad transformers & effect systems

```haskell
newtype App a = App (ReaderT Env (ExceptT AppError IO) a)   -- classic mtl-style stack
  deriving newtype (Functor, Applicative, Monad, MonadReader Env, MonadError AppError, MonadIO)
```
Keep stacks shallow; for richer effect graphs prefer a modern, well-typed effect system (**`effectful`** or `cleff`) over deep transformer towers or `mtl` `n²` instances. Inject capabilities through `ReaderT Env` (records-of-functions) — the "ReaderT design pattern" — so tests swap in pure mocks. Pattern depth is owned by [`designpatterns.md`](guides://designpatterns.md); show only the Haskell binding.

### E. Purity, laziness & strictness control

```haskell
{-# LANGUAGE StrictData #-}        -- strict record fields by default (prevents thunk leaks)
sumStrict :: [Int] -> Int
sumStrict = go 0 where go !acc []     = acc      -- BangPatterns force the accumulator
                       go !acc (x:xs) = go (acc + x) xs
```
- Use `foldl'`, `Data.Map.Strict`, and bang patterns for accumulators; lazy `foldl`/`sum` over large lists leaks.
- Use `Data.Text`/`Data.Text.IO` and strict/`ByteString` — never `String` or lazy `readFile` for non-trivial data.
- Reserve laziness for control flow and infinite/streamed structures; profile leaks with `+RTS -hT`.

### F. GHC extensions discipline

Base on `GHC2021` (or `GHC2024` on GHC 9.10+); enable extras project-wide in the cabal `common` stanza, not per file:

```cabal
common common-options
  default-language: GHC2021
  default-extensions:
      DerivingStrategies DerivingVia OverloadedStrings StrictData
      LambdaCase RecordWildCards OverloadedRecordDot
  ghc-options:
      -Wall -Wcompat -Wincomplete-uni-patterns -Wincomplete-record-updates
      -Wpartial-fields -Wmissing-deriving-strategies -Wunused-packages -Werror
```
Avoid power-tools (`UndecidableInstances`, broad `unsafe*`) unless a documented invariant requires them.

### G. Testing — HSpec + QuickCheck binding

Policy is owned by [`tdd.md`](guides://tdd.md). In Haskell, specify pure functions with **properties** (laws/round-trips/invariants) and use examples for edge cases:

```haskell
spec :: Spec
spec = describe "add" $ do
  it "example: 2 + 3 = 5"  $ add 2 3 `shouldBe` 5
  it "is commutative"      $ property $ \x y   -> add x y === add y x
  it "is associative"      $ property $ \x y z -> add (add x y) z === add x (add y z)
  it "roundtrips encode"   $ property $ \o     -> decode (encode o) === Right (o :: Order)
```
- Layout test suites via `hspec-discover`; write custom `Arbitrary` instances for domain types.
- Mock ports by supplying alternate records-of-functions or a pure `State`/`IORef`-backed implementation — no mocking framework needed.
- Run more cases in CI: `cabal test --test-options='--qc-max-success=1000'`.

### H. Footguns → fixes

- Partial `head`/`tail`/`fromJust`/`!!` → `safeHead`, pattern match, `Data.Maybe` total combinators.
- `error`/`undefined` in production → return `Either`/`Maybe`; HS-ERR-01.
- Orphan instances → newtype wrapper or co-locate the instance.
- Lazy accumulator space leak → `foldl'`/`BangPatterns`/`StrictData`.
- `String` everywhere → `Text` + `OverloadedStrings`.
- `unsafePerformIO` → ban it in `.hlint.yaml` (`functions: [{name: unsafePerformIO, within: []}]`).

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning (PVP) → [`semver.md`](guides://semver.md). Haskell binding:

```bash
cabal freeze            # write cabal.project.freeze (commit it) — HS-DEP-01
cabal build --offline   # reproducible build from freeze (post-fetch)
cabal outdated          # report stale deps
cabal-audit             # HS-SEC-01: HSEC advisory scan; 0 high/critical
hlint src test          # HS-LINT-01
fourmolu --mode inplace $(git ls-files '*.hs')   # HS-FMT-01
```
Bound dependencies with PVP ranges (`>=x.y && <x.(y+1)`/`<x+1`). Commit `cabal.project.freeze`; in CI fetch then `--offline` to forbid surprise downloads.

---

## 7. Quick Reference

```bash
cabal build all                                  # build
cabal test                                       # test (HSpec + QuickCheck)
hlint src test                                   # lint
fourmolu --mode inplace $(git ls-files '*.hs')   # format
cabal repl                                        # REPL
cabal haddock                                     # docs
cabal run <exe> -- <args>                         # run
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] HS-FMT-01 — `fourmolu --mode check` clean
- [ ] HS-LINT-01 — `hlint` clean, no hints
- [ ] HS-WARN-01 — builds with `-Wall -Werror`, no warnings
- [ ] HS-TYP-01 — every top-level binding has a type signature
- [ ] HS-TOTAL-01 — exhaustive patterns, no partial functions
- [ ] HS-ERR-01 — no `error`/`undefined`/partial heads; `Maybe`/`Either` used
- [ ] HS-TST-01/02/03 — tests pass, bugs have regression tests, pure logic has properties
- [ ] HS-DOC-01 — exported symbols have Haddock, `cabal haddock` builds
- [ ] HS-SEC-01 — `cabal-audit` 0 high/critical
- [ ] HS-DEP-01 — `cabal.project.freeze` committed, `--offline` build succeeds
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Haskell Guidelines**

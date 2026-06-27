# Feature Flags Guidelines
Mandatory standards for designing, operating, and retiring feature flags: typed short-lived toggles, progressive rollout, kill switches, and disciplined cleanup. Vendor-agnostic (LaunchDarkly, Split, Unleash, Flagsmith, OpenFeature) or custom.

---
name: feature-flags
title: Feature Flags Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: []
requires: []
recommends:
  - env-config
  - ci-cd
  - observability
  - tdd
provides:
  - flag-types
  - flag-lifecycle
  - progressive-rollout
  - flag-cleanup
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns feature-flag taxonomy, lifecycle, rollout, and cleanup; configuration, delivery pipelines, monitoring, and test strategy are referenced.

---

## 0. Prerequisites & References

Feature flags sit on top of several owned concerns. Fetch the relevant owners; this guide adds only the flag-specific specialization.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`env-config.md`](guides://env-config.md) — config layering, secrets, env separation. *(Flag binding: flags are the **top, runtime-mutable precedence layer** above static config; they override but never replace layered config.)*
> - [`ci-cd.md`](guides://ci-cd.md) — pipelines & progressive delivery. *(Flag binding: flags **decouple deploy from release** so the pipeline ships dark code and rollout is a flag change, not a deploy.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, alerting. *(Flag binding: every flag rollout MUST be gated on owner-defined SLOs; emit the served variation as an event/attribute.)*
> - [`tdd.md`](guides://tdd.md) — test-first, coverage. *(Flag binding: both flag states are tested; the flag is part of the test matrix, not an excuse to skip a path.)*

> 📎 **SEE ALSO:** [`secure-coding.md`](guides://secure-coding.md) (permission flags are **not** an authorization mechanism) · [`logging.md`](guides://logging.md) · [`code-review.md`](guides://code-review.md) · [`git.md`](guides://git.md) (trunk-based development pairs with flags)

---

## 1. Core Philosophies: FLAGS-FIRST

Flag-specific principles only. Pipeline, config, and monitoring rules come from §0.

- **F**lag-driven, not branch-driven: integrate to trunk behind a flag instead of holding a long-lived feature branch. Deploy ≠ release.
- **L**ifecycle-bound: every flag is born with a type, an owner, and (for short-lived types) an expiry. A flag with no removal plan is technical debt at creation.
- **A**uditable: every flag change (create, target, rollout %, kill, delete) is recorded with who/when/why.
- **G**radual & reversible: ship to 0%, ramp through rings/percentages, and keep the off-path working until the flag is removed.
- **S**afe-by-default: the default/fallback variation is the **safe** behavior (usually "old path"); evaluation failures fall back to it; kill switches flip in seconds without a deploy.

**Verified Flags**: any flagged change MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `FF-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| FF-STRUCT-01 | Every flag MUST declare a type (release/experiment/ops/permission) | flag registry / metadata schema | type set, schema-valid |
| FF-STRUCT-02 | Every flag MUST have a named owner | registry field | owner non-empty |
| FF-STRUCT-03 | Short-lived flags (release, experiment) MUST carry an expiry date | registry field | expiry set & in future |
| FF-STRUCT-04 | Default/fallback variation MUST be the safe path; eval errors MUST return it | code review / fallback unit test | safe default returned on miss/error |
| FF-TST-01 | Both flag states MUST be tested (see `tdd.md`) | test suite covers on AND off | both branches exercised |
| FF-TST-02 | Targeting/rollout logic MUST have deterministic-bucketing tests | unit test on hash bucketing | stable assignment asserted |
| FF-OBS-01 | Rollouts MUST be gated on SLOs; served variation emitted (see `observability.md`) | dashboard/alert wired; event has `flag`+`variation` | metric present, alert armed |
| FF-OPS-01 | Critical paths MUST have a tested kill switch defaulting to enabled | kill-switch drill in staging | flip works, < deploy latency |
| FF-SEC-01 | Permission/entitlement decisions MUST be enforced server-side, not by a client flag (see `secure-coding.md`) | code review | no client-trusted gating |
| FF-CLEAN-01 | Stale/expired flags MUST be removed (code + config) | CI staleness audit | 0 expired flags past grace |
| FF-CLEAN-02 | Flag removal MUST delete the dead branch, not leave the loser dangling | PR review / dead-code lint | no orphaned variation code |
| FF-AUDIT-01 | All flag mutations MUST be auditable (who/when/why) | audit log query | every change attributed |

> **Forbidden**: a flag with no type/owner/expiry; using a client-evaluated flag as an authorization boundary; shipping a flagged change tested in only one state; leaving an expired flag in code "just in case"; nesting flags so a code path requires N flags in a specific combination to reach.

---

## 3. Flag Taxonomy (owned)

Pick the **type at creation** — it dictates lifetime, default, and cleanup policy. Misclassifying a release flag as "ops" is the root cause of most stale-flag debt.

| Type | Purpose | Lifetime | Safe default | Cleanup trigger |
|------|---------|----------|--------------|-----------------|
| **Release** | Decouple deploy from release; ramp a new feature | Days–weeks (cap ~2 months) | `false` (old path) | Remove once at 100% and stable |
| **Experiment** | A/B / multivariate measurement | Bounded by experiment window (cap ~1 month) | control | Remove when experiment concludes & decision made |
| **Ops** | Runtime operational control (circuit breakers, rate-limit thresholds, degradation modes) | Long-lived / permanent | the conservative/protective value | Review quarterly; keep only if still actuated |
| **Permission / entitlement** | Gate features by plan/tier/role | Long-lived (until plan model changes) | least-privilege | Tied to product packaging, not code rollout |

- **Kill switch** is a property, not a fifth type: any release/ops flag controlling a critical path MUST expose an instant off (`FF-OPS-01`), defaulting to enabled in normal operation.
- Each flag's metadata record (key, type, owner, description, created, expiry, tags, default) lives in the flag system / registry, **not** scattered in code. Treat the registry as the source of truth.

> Permission flags express **product packaging**, not security. The actual authorization check MUST be enforced server-side per [`secure-coding.md`](guides://secure-coding.md) (`FF-SEC-01`) — a UI flag may hide a button, but the server still validates entitlement.

---

## 4. Flag-Driven Development (owned)

Flags are the alternative to long-lived branches; they make trunk-based development with [`git.md`](guides://git.md) safe.

- **Branch by abstraction, then flag the implementation.** Introduce a seam (interface/strategy), land the new implementation behind a flag at 0%, ramp, then delete the old one. The flag selects *which* concrete behavior runs.
- **Ship dark.** New code reaches production deactivated; the CI/CD pipeline ([`ci-cd.md`](guides://ci-cd.md)) deploys it, the flag releases it. A bad release is a flag flip, not a rollback deploy.
- **Keep the seam thin.** Evaluate the flag once at a single decision point and pass the resulting behavior down; do not sprinkle `if isEnabled(...)` across many layers. Deeply scattered checks are the main driver of cleanup cost (`FF-CLEAN-02`).
- **Never AND flags.** A path that requires flag A *and* flag B *and* flag C creates a combinatorial test matrix and untestable states. Compose at most independent, orthogonal flags.
- **Fail safe.** Wrap evaluation so a missing flag, SDK timeout, or parse error returns the safe default (`FF-STRUCT-04`) — the system degrades to known-good behavior, never to an unguarded new path.

---

## 5. Targeting & Progressive Rollout (owned)

Rollout is the controlled exposure of a variation to growing audiences. The *pipeline* mechanics are owned by [`ci-cd.md`](guides://ci-cd.md) (progressive delivery, canary); this section owns the *flag-side* rules.

### A. Deterministic bucketing
Assignment MUST be a stable hash of `(flagKey, unitId)` so a given user/account sees the **same** variation across sessions and devices (`FF-TST-02`). Hashing the flag key into the seed keeps buckets independent across flags (no correlated rollouts). Choose the bucketing unit deliberately — user, account, session, or device — and keep it consistent for the flag's life.

### B. Rollout strategies
- **Percentage ramp** — 1% → 5% → 25% → 50% → 100%, advancing only when SLOs hold (`FF-OBS-01`). Each step dwells long enough to observe.
- **Ring-based** — internal → dogfood/beta → early adopters → general. Each ring is a targeting rule on an attribute (email domain, tier, opt-in).
- **Targeting rules** — attribute predicates (`equals`/`in`/`contains`/`gt`/`lt`) that pin specific cohorts on/off regardless of percentage. Rule order matters: explicit targets override the percentage ramp; the safe default is the last fallthrough.
- **Automatic rollback** — wire the ramp to the same alerts as [`observability.md`](guides://observability.md): when an SLO/error-budget threshold trips, the controller resets to 0% (or flips the kill switch) before paging a human.

### C. Experiments
For experiment flags, fix the unit of assignment, the variation weights, the primary metric, and the minimum sample size **before** launch. Do not stop an experiment the moment it looks significant (peeking inflates false positives); honor the pre-declared duration/sample size. Emit assignment + conversion events to the analytics path described in [`observability.md`](guides://observability.md).

---

## 6. Lifecycle & Cleanup (owned)

Stale flags are the dominant cost of a flag system: they fork every code path, multiply the test matrix, and hide which behavior is actually live. Treat removal as part of "done."

### A. States
`planned → active → ramping → fully-rolled-out → archived/removed`. A release flag at 100% for the agreed grace period (e.g. 14 days stable) is **done** and MUST be removed (`FF-CLEAN-01`).

### B. Staleness detection in CI
Run an automated audit in the pipeline ([`ci-cd.md`](guides://ci-cd.md)) that fails when a flag is:
- past its expiry (`FF-STRUCT-03`),
- at 100% (or 0%) beyond the grace window,
- not evaluated in N days (dead in production), or
- referenced in code but absent from the registry (or vice-versa — orphaned).

The audit's output is the cleanup backlog; expired flags past grace are a hard gate, not a warning.

### C. Removing a flag
1. Decide the winner (the live variation).
2. Inline that variation; **delete the losing branch entirely** (`FF-CLEAN-02`) — do not leave the dead path commented or behind a constant.
3. Delete the flag's tests for the now-impossible state; keep behavior tests for the surviving path.
4. Remove the flag from the registry/provider **after** the code change is deployed (avoid evaluating a deleted flag).
5. Record the removal in the audit log (`FF-AUDIT-01`).

Cleanup is itself a flag-driven change: ship the inlined code, verify in staging, then delete the flag config. Automating step 2's mechanical edits (codemod / cleanup PR) is encouraged but the dead-branch deletion MUST be reviewed.

---

## 7. Testing with Flags

The strategy (test-first, coverage, regression-before-fix) is owned by [`tdd.md`](guides://tdd.md). Flag-specific obligations:

- **Both states, always.** Every flagged path is tested on **and** off (`FF-TST-01`); a flag never reduces coverage of either branch.
- **Pin the flag in tests.** Inject the flag client / override the evaluation so tests are deterministic — never let a test read live rollout percentages.
- **Test the fallback.** Assert that a missing flag, an SDK error/timeout, and an unknown variation all resolve to the safe default (`FF-STRUCT-04`).
- **Test bucketing.** Assert stable assignment for a fixed `(flagKey, unitId)` and a roughly uniform distribution across buckets (`FF-TST-02`).
- **Integration matrix.** For orthogonal flags, cover the combinations that are actually reachable — and rely on §4's "never AND flags" rule to keep that matrix small.

---

## 8. Implementation Notes (technology-agnostic)

This guide is language-neutral; bind these idioms to the project's stack rather than copying an SDK.

- **Evaluate server-side for anything sensitive.** Client-side flags can be inspected and forged — security/entitlement decisions are server-enforced (`FF-SEC-01`).
- **Cache with bounded staleness + streaming updates.** Read from a local cache for latency; subscribe to push updates (SSE/stream) or refresh on a short TTL so a kill switch propagates in seconds, not minutes.
- **Prefer a standard interface.** Use the OpenFeature API (or the vendor SDK behind a thin port) so the provider — LaunchDarkly, Split, Unleash, Flagsmith, or custom — is swappable. The application depends on `isEnabled(key, ctx)` / `getVariation(key, ctx)`, not on a vendor type.
- **UI frameworks**: expose flags through the framework's idiomatic context/provider and a single `Feature`/`useFlag` seam; keep the evaluation decision at the boundary (§4), not scattered through components.
- **Config layering**: flags override layered config but secrets and base config remain owned by [`env-config.md`](guides://env-config.md) — do not smuggle secrets through flag values.

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] FF-STRUCT-01 — flag has a declared type
- [ ] FF-STRUCT-02 — flag has a named owner
- [ ] FF-STRUCT-03 — short-lived flags have a future expiry
- [ ] FF-STRUCT-04 — safe default; eval errors fall back to it
- [ ] FF-TST-01 — both flag states tested (see `tdd.md`)
- [ ] FF-TST-02 — deterministic bucketing tested
- [ ] FF-OBS-01 — rollout gated on SLOs; served variation emitted (see `observability.md`)
- [ ] FF-OPS-01 — kill switch present, tested, defaults enabled
- [ ] FF-SEC-01 — entitlement enforced server-side (see `secure-coding.md`)
- [ ] FF-CLEAN-01 — no expired/stale flags past grace (CI audit)
- [ ] FF-CLEAN-02 — losing branch deleted on removal
- [ ] FF-AUDIT-01 — all flag mutations attributed (who/when/why)
- [ ] Agent ran the CI staleness audit and resolved any findings

---
**End of Feature Flags Guidelines**

# Code Comments & Documentation Guidelines
Canonical owner of code comments and API documentation: when/why to comment, doc-comment conventions, self-documenting code, examples-in-docs, and language-agnostic doc generation. Doc generators: pydoc/Sphinx, JSDoc/TypeDoc, Javadoc, godoc, rustdoc, Doxygen, YARD, phpDocumentor, Dokka, DocC.

---
name: comments
title: Code Comments & Documentation Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [sphinx, typedoc, jsdoc, javadoc, godoc, rustdoc, doxygen, yard, dokka, docc]
requires: []
recommends:
  - markdown
  - adr
  - todo
provides:
  - doc-comments
  - api-docs
  - self-documenting-code
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): this is the **canonical owner** of code comments and API documentation. Language guides reference it and add only their doc-generator binding. It references — rather than restates — markdown authoring, decision records, and TODO conventions.

---

## 0. Prerequisites & References

This guide owns *what/why/how* of comments and API docs across all languages. It defers the following adjacent concerns to their owners.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`markdown.md`](guides://markdown.md) — authoring prose/READMEs/long-form docs. *(Binding: doc-comment bodies that render as Markdown, e.g. rustdoc/Dokka/DocC, follow its rules; this guide governs the doc-comment structure, not Markdown syntax.)*
> - [`adr.md`](guides://adr.md) — Architecture Decision Records. *(Binding: a reference comment may link an ADR; the decision itself lives in the ADR, never duplicated in a comment.)*
> - [`todo.md`](guides://todo.md) — TODO/FIXME/HACK tags, issue linkage, priority, lifecycle. *(Binding: §2 requires every incomplete-work marker carry an issue ref per `todo.md`; the tag taxonomy and triage rules are owned there.)*

> 📎 **SEE ALSO:** [`code-review.md`](guides://code-review.md) — comments are reviewed like code · [`ci-cd.md`](guides://ci-cd.md) — where doc-gen runs as a gate · [`semver.md`](guides://semver.md) — `@since`/`@deprecated` version tags.

> Cross-cutting *content* (error strategy, security rationale, performance numbers) belongs in its owner guide; a comment may *point* to it but the policy is not restated here.

---

## 1. Core Philosophies: COMMENT-WISE

A comment exists to convey what code cannot. Each principle below is enforced by a §2 requirement.

- **C**ode First — self-documenting code is the primary documentation; comments supplement, never substitute for, clear names and structure. If a comment is needed to make code understandable, refactor the code first.
- **O**nly the WHY — comment intent, constraints, trade-offs, and non-obvious decisions; never narrate the WHAT the code already states.
- **M**achine-readable — public APIs use the language's native doc-comment syntax so docs generate automatically and surface in IDEs.
- **M**aintained — a comment is part of the code under change; stale comments are worse than none and MUST be updated or deleted in the same change.
- **E**xamples — every non-trivial public API carries at least one runnable usage example.
- **N**o redundancy — no noise, no journal/attribution, no commented-out code, no closing-brace labels.
- **T**ODOs tracked — incomplete work uses standardized, issue-linked markers (owned by [`todo.md`](guides://todo.md)).
- **W**arnings explained — danger, side effects, thread-safety, and deprecation are stated *with their reason*, never "here be dragons".
- **I**ssue-linked — bug fixes and references cite the issue/spec/RFC/ADR that justifies the code.
- **S**tructured — follow each language's documentation convention exactly so generators parse cleanly.
- **E**volving — treat comments as living documentation; review them on every signature, behavior, or dependency change.

**Golden Rule:** if you need extensive comments to explain code, refactor the code first.

**Agent Responsibility:** when modifying code, the agent MUST review and update every affected comment before delivery (DOC-MNT-01).

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `DOC-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. "Verify" is language-agnostic — bind the concrete command in the language guide (e.g. `pydoc`, `typedoc`, `cargo doc`).

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| DOC-API-01 | Every public function/method/class/interface/type/module MUST have a doc comment in the language's native format | doc-gen run / coverage tool (e.g. `interrogate`, `typedoc --validation`) | 100% public symbols documented |
| DOC-API-02 | Each doc comment MUST document: one-line summary, every parameter (type, constraints, defaults), return value (if any), and errors/exceptions raised | doc-gen + review | no undocumented param/return/error |
| DOC-API-03 | Every non-trivial public API MUST include at least one usage example | review / doc-gen | example present |
| DOC-API-04 | Examples in docs MUST compile/run | doctest / example test step (e.g. `pytest --doctest-modules`, `cargo test --doc`) | exit 0 |
| DOC-WHY-01 | Comments MUST explain WHY (intent/constraint/decision), MUST NOT restate the WHAT | review | no narrating comments |
| DOC-WHY-02 | Non-obvious algorithm choices, performance trade-offs, workarounds, security considerations, and business rules MUST be commented at their site | review | rationale present |
| DOC-REF-01 | Bug-fix code MUST carry a comment citing the issue ID and root cause | review / `grep` for fix markers | issue ref present |
| DOC-REF-02 | Reference comments to specs/RFCs/algorithms/ADRs MUST link the canonical source (ADRs via `adr.md`, not inlined) | review | link resolves |
| DOC-TODO-01 | Every incomplete-work marker MUST be a standardized issue-linked TODO/FIXME/HACK (see `todo.md`) | TODO scanner / `grep -nE 'TODO\|FIXME\|HACK'` | every marker has an issue ref |
| DOC-NOISE-01 | Code MUST NOT contain noise, journal/attribution, position-marker, or closing-brace comments | review / lint (e.g. `eslint no-inline-comments`, custom) | none |
| DOC-NOISE-02 | Code MUST NOT contain commented-out code without an explanatory reason | review / `grep` | none |
| DOC-MNT-01 | When code changes, all affected comments MUST be updated or removed in the same change; no comment may contradict the code | review / doc-gen diff | no stale/contradicting comments |
| DOC-GEN-01 | API documentation MUST generate with zero warnings/errors | doc-gen (e.g. `javadoc -Xdoclint:all`, `cargo doc`, `typedoc`) | exit 0, 0 warnings |
| DOC-VER-01 | Public APIs SHOULD carry `@since` on addition and `@deprecated` (with replacement + removal version) on deprecation (see `semver.md`) | review / doc-gen | tags present where applicable |

> **Forbidden:** delivering code that fails any gate above; comments that contradict the code; missing docs on a public API; commented-out code without a reason; TODOs/FIXMEs without an issue ref (violates `todo.md`); journal/attribution comments (use git history); duplicating an ADR's content inside a comment (violates `adr.md`); using comments to compensate for code that should be refactored.

---

## 3. When to Comment

### A. ALWAYS comment

- **Public API documentation** — every public function, method, class, interface, type, module: summary, parameters (type/constraint/default), returns, errors/exceptions, and a usage example for anything non-trivial. *(DOC-API-01..04)*
- **WHY / business logic** — non-obvious algorithm choices, performance optimizations and their measured reasoning, workarounds for external limitations, business rules that are not self-evident, security considerations. *(DOC-WHY-01/02)*
- **References** — bug-fix issue IDs, links to specs/RFCs, external-API docs, algorithm source (paper/book/URL), ADRs (link via [`adr.md`](guides://adr.md)). *(DOC-REF-01/02)*
- **Warnings** — dangerous operations, non-obvious side effects, thread-safety, deprecation — always *with the reason and safe-usage note*.
- **Incomplete work** — TODO/FIXME/HACK markers, all issue-linked per [`todo.md`](guides://todo.md). *(DOC-TODO-01)*

### B. NEVER comment

| Anti-pattern | Example | Fix |
|---|---|---|
| Obvious operation | `counter++  // increment counter` | delete |
| Code restated in English | `// loop through users` over a `for` | delete; the code says it |
| Redundant type info | `// string to hold name` on a typed field | delete; the type says it |
| Journal / changelog | `// modified by John 2024-01-15` | use git history |
| Attribution / copyright in code | `// written by John Smith` | use file header / `LICENSE` |
| Commented-out code | `// oldFunction();` | delete; git preserves it |
| Closing-brace labels | `} // end for` | extract smaller functions |
| Position markers | `//==== HELPERS ====` | split into files/classes |
| Context-free scare | `// DON'T TOUCH THIS!` | state WHY and consequences |
| Mandated trivial doc | `/** Gets the id. @return id */` on a getter | comment only when it adds value |

---

## 4. Doc-Comment Conventions by Language

Use the language's **native** doc-comment format so generators produce API docs and IDEs show inline help. The structure (summary → description → params → returns → errors → example → see-also → since) is identical across languages; only the syntax differs. Below is one canonical example — a discount calculator — rendered in each ecosystem's format. Required elements are tabulated in §4.B.

### Python — Google-style (PEP 257), parsed by Sphinx/pydoc
```python
def calculate_discount(price: float, percentage: float, max_discount: float = 100.0) -> float:
    """Apply a percentage discount to a price, capped at a maximum.

    Args:
        price: Original price in dollars. Must be non-negative.
        percentage: Discount percentage (0-100); values outside are clamped.
        max_discount: Maximum discount amount allowed. Defaults to 100.0.

    Returns:
        The final price after applying the discount.

    Raises:
        ValueError: If price is negative.

    Example:
        >>> calculate_discount(100.0, 20.0)
        80.0

    Note:
        Implements pricing rules from SPEC-123.

    Since:
        1.2.0
    """
```

### TypeScript / JavaScript — TSDoc / JSDoc, parsed by TypeDoc/JSDoc
```typescript
/**
 * Apply a percentage discount to a price, capped at a maximum.
 *
 * @param price - Original price in dollars. Must be non-negative.
 * @param percentage - Discount percentage (0-100).
 * @param maxDiscount - Maximum discount amount allowed.
 * @returns The final price after the discount.
 * @throws {RangeError} If price is negative.
 * @example
 * calculateDiscount(100, 20); // 80
 * @see {@link https://internal.docs/pricing-rules}
 * @since 1.2.0
 */
function calculateDiscount(price: number, percentage: number, maxDiscount = 100): number {
```

### Java — Javadoc
```java
/**
 * Apply a percentage discount to a price, capped at a maximum.
 *
 * @param price       the original price in dollars; must be non-negative
 * @param percentage  the discount percentage (0-100)
 * @param maxDiscount maximum discount amount allowed
 * @return the final price after the discount
 * @throws IllegalArgumentException if price is negative
 * @since 1.2.0
 * @see <a href="https://internal.docs/pricing-rules">Pricing Rules</a>
 */
public double calculateDiscount(double price, double percentage, double maxDiscount) {
```

### Go — godoc (full sentences, exported-name first)
```go
// CalculateDiscount applies a percentage discount to price, capped at maxDiscount.
//
// price must be non-negative; percentage is 0-100. It returns the final price,
// or an error if price is negative. See https://internal.docs/pricing-rules.
//
//	final, err := CalculateDiscount(100.0, 20.0, 100.0) // final == 80.0
func CalculateDiscount(price, percentage, maxDiscount float64) (float64, error) {
```

### Rust — rustdoc (body is Markdown — see `markdown.md`; `cargo test --doc` runs the example)
```rust
/// Apply a percentage discount to `price`, capped at `max_discount`.
///
/// # Arguments
/// * `price` - Original price; must be non-negative.
/// * `percentage` - Discount percentage (0-100).
///
/// # Errors
/// Returns [`DiscountError::NegativePrice`] if `price` is negative.
///
/// # Examples
/// ```
/// # use pricing::calculate_discount;
/// assert_eq!(calculate_discount(100.0, 20.0, 100.0)?, 80.0);
/// # Ok::<(), pricing::DiscountError>(())
/// ```
pub fn calculate_discount(price: f64, percentage: f64, max_discount: f64) -> Result<f64, DiscountError> {
```

### C / C++ — Doxygen
```cpp
/**
 * @brief Apply a percentage discount to a price, capped at a maximum.
 * @param[in] price        Original price; must be non-negative.
 * @param[in] percentage   Discount percentage (0-100).
 * @param[in] max_discount Maximum discount amount. Default: 100.0.
 * @return The final price after the discount.
 * @throws std::invalid_argument if price is negative.
 * @see https://internal.docs/pricing-rules
 * @since 1.2.0
 */
double calculate_discount(double price, double percentage, double max_discount = 100.0);
```

> Other ecosystems map the same structure: **Ruby** → YARD (`@param`/`@return`); **PHP** → phpDocumentor; **Kotlin** → KDoc/Dokka; **Swift** → DocC markup. The language guide owns the binding; this section owns the required elements.

### B. Required documentation elements

| Element | Required |
|---|---|
| **Summary** — one-line purpose | YES |
| **Description** — detail | if non-trivial |
| **Parameters** — name, type, constraints, defaults | YES (all params) |
| **Returns** — type, meaning, possible values | YES (if not void) |
| **Errors/Exceptions** — what and when | YES (if any) |
| **Example** — runnable | YES for public API (DOC-API-03/04) |
| **See Also** — related docs/specs/ADRs | if applicable |
| **Since** — version added | recommended (DOC-VER-01) |
| **Deprecated** — replacement + removal version | if deprecated (DOC-VER-01, see `semver.md`) |

---

## 5. Comment Kinds & Their Formats

### A. Bug-fix comments (DOC-REF-01)
State issue, root cause, and fix at the change site so the *why* survives. Keep it tight; the discussion lives in the issue, the decision (if architectural) in an ADR (see [`adr.md`](guides://adr.md)).
```python
# FIX(#GH-1234): guard against division by zero in percentage calc.
# Cause: total_count could be 0 for empty cohorts. Fix: branch on > 0.
percentage = (count / total_count) * 100 if total_count > 0 else 0.0
```
For non-trivial fixes add Problem / Cause / Solution lines and a regression link:
```java
// FIX(#JIRA-5678): race in session creation — concurrent requests made
//   duplicate sessions. Cause: non-atomic check-then-create.
//   Solution: per-user distributed lock (held ≤5s). See ADR-031.
synchronized (getUserLock(userId)) { /* atomic check-and-create */ }
```

### B. Reference comments (DOC-REF-02)
Link the canonical source; never paste the spec/decision.
```go
// ParseJWT validates a token per RFC 7519. Rejects "none" alg (CVE-2015-2951).
// Spec: https://tools.ietf.org/html/rfc7519 · Decision: docs/adr/042-jwt.md
```
```python
# Wagner-Fischer edit distance, O(m*n) time / O(min(m,n)) space.
# Wagner & Fischer (1974), JACM 21(1):168-173. https://doi.org/10.1145/321796.321811
```

### C. Inline comments
Sparingly, for non-obvious *why*:
```python
timeout = 30        # matches upstream service SLA (SLA-DOC-123)
buffer_size = 8192  # SSD page size; benchmarked in PERF-789
result = data.copy()  # process() mutates its input — must copy
```

### D. Block comments for complex logic
A short block explaining approach, the rejected alternative, and edge cases handled — when the algorithm is genuinely non-trivial and naming alone cannot carry it.

### E. Class / module headers
Public classes and modules carry a header covering responsibility, thread-safety (if relevant), key dependencies, and one usage example. Push configuration/architecture detail to its owner guide and *link* it rather than restating (e.g. config → `env-config.md`, layering → the architecture owner).

### F. TODO / FIXME / HACK markers
Owned by [`todo.md`](guides://todo.md). This guide only requires (DOC-TODO-01) that every incomplete-work marker is standardized and issue-linked, e.g. `# TODO(#GH-456): validate email per RFC 5322`. The tag taxonomy, priority scheme, and lifecycle rules live in `todo.md` — do not restate them.

---

## 6. Comment Maintenance

A comment is part of the code under change. When code changes, audit and fix comments in the **same** change (DOC-MNT-01).

- **Signature change** → update params, returns, errors, and examples.
- **Behavior change** → update the description and any edge-case/performance notes.
- **Bug fix** → add the fix comment (DOC-REF-01); remove comments describing the old buggy behavior.
- **Refactor** → review every comment in the touched code; delete comments for deleted code; update the module header.
- **Dependency change** → update version references; remove workaround comments whose upstream bug is now fixed.

**Stale-comment smells:** a documented parameter that no longer exists; a summary describing pre-refactor behavior ("returns all users" when it now paginates); a reference to a renamed symbol; a `HACK` for a bug fixed in a since-upgraded dependency.

---

## 7. Documentation Generation

API docs are generated and verified in CI as a gate (DOC-GEN-01); the *pipeline* is owned by [`ci-cd.md`](guides://ci-cd.md). Long-form prose docs follow [`markdown.md`](guides://markdown.md).

| Language | Generator | Generate | Coverage / lint |
|---|---|---|---|
| Python | Sphinx / pydoc | `sphinx-build -b html docs/ _build/` | `interrogate -vv --fail-under 100 src/` |
| JS | JSDoc | `jsdoc -c jsdoc.json` | — |
| TypeScript | TypeDoc | `typedoc --out docs src/` | `typedoc --validation.notExported` |
| Java | Javadoc | `javadoc -d docs -sourcepath src` | `javadoc -Xdoclint:all` |
| Go | godoc | `go doc -all ./...` | `go vet ./...` (doc-adjacent) |
| Rust | rustdoc | `cargo doc --no-deps` | `cargo test --doc` |
| C/C++ | Doxygen | `doxygen Doxyfile` | warnings-as-errors in `Doxyfile` |
| Ruby | YARD | `yard doc` | `yard stats --list-undoc` |
| PHP | phpDocumentor | `phpdoc -d src -t docs` | — |
| Kotlin | Dokka | `./gradlew dokkaHtml` | — |
| Swift | DocC | `swift package generate-documentation` | — |

Gate principle: generation MUST fail on warnings, public-API coverage MUST meet threshold (100% for public symbols), and documentation examples MUST execute (DOC-API-04). Generated HTML stays out of version control.

---

## 8. Why This Works

- **Code as primary documentation** — self-documenting code lowers maintenance load; comments carry intent, code carries implementation, so the two diverge less.
- **Machine-readable docs** — native doc-comment syntax means docs generate automatically, ship in the same PR as the code, and surface in the IDE.
- **Minimal comments** — fewer comments means less to go stale; the comments that remain are signal, not noise, and they pressure better naming.
- **Issue-linked fixes & TODOs** — traceability enables bisecting, prevents reintroducing fixed bugs, and makes technical debt visible and triageable.

---

## Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] DOC-API-01 — every public symbol has a native-format doc comment
- [ ] DOC-API-02 — summary, all params, returns, and errors documented
- [ ] DOC-API-03 — non-trivial public APIs carry a usage example
- [ ] DOC-API-04 — doc examples compile/run (doctest passes)
- [ ] DOC-WHY-01/02 — comments explain WHY; non-obvious decisions documented; no WHAT-narration
- [ ] DOC-REF-01 — bug fixes cite issue ID + root cause
- [ ] DOC-REF-02 — spec/RFC/algorithm/ADR references link the canonical source (ADRs via `adr.md`)
- [ ] DOC-TODO-01 — every TODO/FIXME/HACK is issue-linked (see `todo.md`)
- [ ] DOC-NOISE-01 — no noise/journal/attribution/position/closing-brace comments
- [ ] DOC-NOISE-02 — no commented-out code without a reason
- [ ] DOC-MNT-01 — affected comments updated in the same change; none contradict the code
- [ ] DOC-GEN-01 — API docs generate with zero warnings/errors
- [ ] DOC-VER-01 — `@since`/`@deprecated` tags present where applicable (see `semver.md`)

---
**End of Code Comments & Documentation Guidelines**

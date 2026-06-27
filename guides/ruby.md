# Ruby Development Guidelines
Mandatory coding standards for Ruby: idiomatic, readable, test-covered, type-checked. Ruby 3.3+, Bundler, RSpec/Minitest, RuboCop, Sorbet/RBS, bundler-audit.

---
name: ruby
title: Ruby Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [ruby@3.3, bundler, rspec, minitest, rubocop, sorbet, rbs, bundler-audit]
requires:
  - tdd
  - secure-coding
  - error-handling
recommends:
  - hexagonal
  - designpatterns
  - comments
  - semver
provides:
  - idiomatic-ruby
  - blocks-procs
  - modules-mixins
  - ruby-typing
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Ruby (the language and its toolchain — framework specifics such as Rails are out of scope).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Ruby code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Ruby binding: runner is `bundle exec rspec`; coverage via SimpleCov.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Ruby binding: `bundle-audit`, and `brakeman` for Rails apps.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(Ruby binding: exception hierarchies; rescue `StandardError`, never bare.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion.
> - [`designpatterns.md`](guides://designpatterns.md) — GoF & friends; show only the Ruby binding.
> - [`comments.md`](guides://comments.md) — doc policy *(binding: YARD/RDoc tags on public API)*
> - [`semver.md`](guides://semver.md) — gem version policy.

> 📎 **SEE ALSO:** [`rails.md`](guides://rails.md) *(the Rails framework builds on this guide)* · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md) · [`pre-commit.md`](guides://pre-commit.md)

---

## 1. Core Philosophies: RUBY-FIRST

Ruby-specific principles only. TDD, security, error handling, and architecture come from §0.

- **R**eadable: code reads like prose — expressive names, guard clauses, no clever density; let RuboCop arbitrate style.
- **U**niform: follow the community style guide enforced by `rubocop`; one canonical way, no per-file dialects.
- **B**locks first: iterate with `map`/`select`/`reduce` and blocks/procs/lambdas — never hand-rolled index loops.
- **Y**ield to duck typing: depend on messages an object responds to, not its class; compose behaviour with modules/mixins.
- **F**rozen by default: `# frozen_string_literal: true` in every file; `freeze` shared constants; prefer immutable value objects.
- **I**ntentional metaprogramming: `define_method`/`method_missing` only when it removes real duplication — and only with `respond_to_missing?`.
- **R**eal types: annotate public APIs with Sorbet `sig` or RBS and gate them with a type checker, not the linter.

**Verified Code**: Agent-generated Ruby MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `RB-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| RB-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `bundle exec rspec` | exit 0, 0 pending |
| RB-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `bundle exec rspec` | failing→passing |
| RB-TST-03 | Business-logic coverage MUST meet the project gate | `COVERAGE=1 bundle exec rspec` (SimpleCov) | ≥ threshold |
| RB-FMT-01 | Code MUST be formatted to the style guide | `bundle exec rubocop` | no offenses |
| RB-LINT-01 | Linter MUST pass clean (incl. `rubocop-performance`) | `bundle exec rubocop` | exit 0 |
| RB-TYP-01 | Public APIs MUST be typed (Sorbet `sig` or RBS) | `bundle exec srb tc` / `steep check` | exit 0 |
| RB-DOC-01 | Public modules/classes/methods MUST be documented (see `comments.md`) | `yard stats --list-undoc` | 0 undocumented |
| RB-SEC-01 | 0 known CVEs in gems (see `secure-coding.md`) | `bundle exec bundle-audit check --update` | 0 advisories |
| RB-SEC-02 | No `eval`/`send` on untrusted input; no bare `rescue` (see `secure-coding.md`) | `bundle exec rubocop` (Security cops) | exit 0 |
| RB-DEP-01 | `Gemfile.lock` committed & in sync | `bundle install && git diff --exit-code Gemfile.lock` | no diff |
| RB-FREEZE-01 | Every file MUST declare `# frozen_string_literal: true` | `bundle exec rubocop` (`Style/FrozenStringLiteralComment`) | exit 0 |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, bare `rescue` (catches `Exception`), `rescue Exception`, mutating method arguments in place, metaprogramming without `respond_to_missing?`, or `bundle install --no-deployment` skipping the lockfile.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
bundle install                              # RB-DEP-01 (lockfile in sync)
bundle exec rubocop                         # RB-FMT-01 / RB-LINT-01 / RB-SEC-02 / RB-FREEZE-01
bundle exec srb tc                          # RB-TYP-01  (or: steep check for RBS)
bundle exec rspec                           # RB-TST-01/02 (SimpleCov gates RB-TST-03)
bundle exec bundle-audit check --update     # RB-SEC-01
yard stats --list-undoc                     # RB-DOC-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic gem/library layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their Ruby mapping.

```
project/
├── lib/<gem>/            # source; require entry at lib/<gem>.rb
│   ├── domain/           # pure business logic — no IO/framework deps
│   ├── adapters/         # db/http/cli implementations of ports
│   └── version.rb        # VERSION constant (see semver.md)
├── sig/                  # RBS type signatures (if using RBS)
├── spec/                 # RSpec, mirrors lib/ (see tdd.md)
├── Gemfile / *.gemspec   # dependency + gem manifest
├── Gemfile.lock          # committed lockfile (RB-DEP-01)
├── .rubocop.yml          # lint/format config
└── README.md
```

- Namespace everything under one top-level module; one class/module per file, path matching the constant.
- Group by domain/feature, not by type. Keep modules small and single-responsibility.

---

## 5. Ruby Specifics

The unique value of this guide.

### A. Idiomatic naming & predicates
`CamelCase` for classes/modules, `snake_case` for methods/variables, `SCREAMING_SNAKE_CASE` for constants. Predicates end in `?`, mutating/dangerous variants in `!` (and a `!` method implies a non-`!` pair exists).

```ruby
# frozen_string_literal: true

ROLES = %w[admin moderator user].freeze   # freeze shared constants

def admin? = role == :admin               # endless method (3.0+), predicate
def normalize! = (self.name = name.strip.downcase; self)
```

### B. Blocks, procs & lambdas
Iterate with blocks; reach for `each_with_object`, `reduce`, `group_by`, `tally`, `filter_map`. Know the proc/lambda distinction: lambdas check arity and `return` locally; procs don't.

```ruby
names    = users.map(&:name)                 # symbol-to-proc
active   = users.select(&:active?)
totals   = orders.each_with_object(Hash.new(0)) { |o, h| h[o.customer_id] += o.total }
present  = rows.filter_map { |r| r.email if r.active? }   # map + compact in one pass
counts   = words.tally                       # {"a"=>3, ...}

doubler  = ->(x) { x * 2 }                   # lambda: strict arity, local return
adder    = proc { |a, b| (a || 0) + (b || 0) } # proc: lenient arity
```

> Footgun: `return` inside a bare `proc` returns from the *enclosing method*. Use a lambda when you need a local return.

### C. Modules & mixins (composition over inheritance)
Use modules for namespacing and for sharing behaviour. Prefer composition; reserve inheritance for true is-a. The canonical mixin pattern hooks `included` to add class methods:

```ruby
module Timestampable
  def self.included(base) = base.extend(ClassMethods)

  module ClassMethods
    def timestamped = @timestamped = true
  end

  def touch = @updated_at = Time.now
end

class Document
  include Timestampable
  timestamped
end
```

Use `prepend` to wrap a method (decorator), `extend` to add class-level behaviour. For design-pattern bindings (Decorator, Strategy, Observer) reference [`designpatterns.md`](guides://designpatterns.md) — Ruby usually expresses them with blocks/modules rather than class hierarchies.

### D. Duck typing & pattern matching
Depend on behaviour, not class: ask `respond_to?` or just send the message. Use `case/in` pattern matching (3.0+) for structured data instead of nested conditionals.

```ruby
case response
in { status: 200, body: }          then parse(body)
in { status: 404 }                 then not_found
in { status: 500, error: String => msg } then log(msg)
else                                    handle_unknown
end
```

### E. Immutability & frozen strings
Put `# frozen_string_literal: true` atop every file (RB-FREEZE-01). `freeze` constants and config. Model values as immutable records with `Data.define` (3.2+) instead of mutable `Struct`/`OpenStruct`.

```ruby
Point = Data.define(:x, :y) do
  def +(other) = with(x: x + other.x, y: y + other.y)   # returns a new instance
end

p = Point.new(x: 1, y: 2)
p.frozen?  # => true
```

> Avoid `OpenStruct` (slow, defeats typing) and in-place mutation of arguments; return new objects.

### F. Metaprogramming — sparingly
`define_method`, `method_missing`, and `Module#refine` are powerful but obscure intent and break tooling/typing. Use only to eliminate real duplication, keep it local, and **always** pair `method_missing` with `respond_to_missing?`.

```ruby
class Settings
  def initialize(data) = @data = data

  def method_missing(name, *) = @data.fetch(name) { super }
  def respond_to_missing?(name, include_private = false) = @data.key?(name) || super
end
```

> Prefer a plain method or `Data`/`Struct` over metaprogramming whenever the static version is no longer than the dynamic one.

### G. Exceptions — Ruby binding
Strategy (when to raise vs. return, propagation, wrapping) is owned by [`error-handling.md`](guides://error-handling.md). Ruby specifics: define one namespaced base `< StandardError` and subclass it; rescue the **narrowest** type; never bare-`rescue` (that catches `Exception`, including `Interrupt`/`SignalException`).

```ruby
module MyGem
  Error            = Class.new(StandardError)
  ValidationError  = Class.new(Error)
  NotFoundError    = Class.new(Error)
end

def charge(order)
  gateway.charge(order.total)
rescue Gateway::CardDeclined => e          # specific first
  order.declined!(e.message)
rescue Gateway::NetworkError
  retry_later(order)
rescue StandardError => e                  # explicit, never bare
  logger.error(e.full_message)
  raise MyGem::Error, "charge failed"      # wrap, preserve cause
ensure
  gateway.close
end
```

### H. Typing — Sorbet / RBS binding
Annotate public APIs and gate with a checker (RB-TYP-01), not RuboCop. Two ecosystems: **Sorbet** (inline `sig` + `srb tc`) or **RBS** (`.rbs` files in `sig/` + `steep check`). Pick one per project.

```ruby
# typed: strict
extend T::Sig

sig { params(name: String, age: Integer).returns(String) }
def greet(name, age) = "#{name} (#{age})"
```

```rbs
# sig/greeter.rbs  (RBS alternative)
class Greeter
  def greet: (String name, Integer age) -> String
end
```

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Ruby binding via Bundler:

```bash
bundle install                          # install from Gemfile.lock (reproducible)
bundle add <gem>                        # add dep (updates Gemfile.lock)
bundle update <gem>                     # update one gem to latest resolvable
bundle outdated                         # list updatable gems
bundle exec bundle-audit check --update # RB-SEC-01: CVE scan against deps
```

Commit `Gemfile.lock`. Constrain direct gems with pessimistic `~>` (e.g. `gem "puma", "~> 6.4"`); let Bundler resolve the graph. Pin `ruby "~> 3.3"` in the Gemfile.

---

## 7. Quick Reference

```bash
bundle install                          # setup
bundle exec rspec                       # test
bundle exec rubocop -a                  # lint + autocorrect
bundle exec srb tc                      # type check (or: steep check)
bundle exec bundle-audit check --update # CVE scan
ruby -Ilib -e 'require "<gem>"'         # smoke-load
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] RB-FMT-01 / RB-LINT-01 — `rubocop` clean, no offenses
- [ ] RB-TYP-01 — Sorbet/RBS type check clean (not the linter)
- [ ] RB-TST-01/02/03 — specs pass, bugs have regression tests, coverage ≥ gate
- [ ] RB-DOC-01 — public API documented, `yard stats` 0 undocumented
- [ ] RB-SEC-01 — `bundle-audit` 0 advisories
- [ ] RB-SEC-02 — no unsafe `eval`/`send`, no bare rescue
- [ ] RB-DEP-01 — `Gemfile.lock` in sync & committed
- [ ] RB-FREEZE-01 — every file has `# frozen_string_literal: true`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Ruby Guidelines**

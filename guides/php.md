# PHP Development Guidelines
Mandatory coding standards for PHP: strict-typed, PSR-compliant, statically analyzed, test-covered. PHP 8.4, Composer, PHPUnit/Pest, PHPStan/Psalm, php-cs-fixer.

---
name: php
title: PHP Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [php@8.4, composer, phpunit@11, pest@3, phpstan@2, psalm@6, php-cs-fixer@3]
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
  - modern-php
  - psr-standards
  - typed-php
  - composer
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to PHP.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating PHP code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(PHP binding: runner is `phpunit` or `pest`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(PHP binding: `composer audit`; injection/XSS rules below.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(PHP binding: typed exceptions, never silence with `@`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion (PHP layout in §6).
> - [`designpatterns.md`](guides://designpatterns.md) — GoF & friends; show only the PHP binding.
> - [`comments.md`](guides://comments.md) — doc policy *(binding: PHPDoc / PSR-5 & PSR-19)*.
> - [`semver.md`](guides://semver.md) — Composer version constraints & package versioning.

> 📎 **SEE ALSO:** [`cleanarch.md`](guides://cleanarch.md) · [`parallelism.md`](guides://parallelism.md) *(if the task uses Fibers/AMPHP concurrency)* · [`logging.md`](guides://logging.md) *(binding: PSR-3 `LoggerInterface`)*

---

## 1. Core Philosophies: PHP-FIRST

PHP-specific principles only. TDD, security, and error handling come from §0.

- **P**SR-compliant: PSR-4 autoloading, PSR-12 code style, PSR-3/PSR-7/PSR-11 interfaces over framework-specific contracts.
- **H**ardened types: `declare(strict_types=1)` in every file; typed properties, parameters, and return types everywhere; no `mixed` without a PHPDoc generic.
- **P**recise static analysis: PHPStan/Psalm at **max** level — the type checker, not the linter, gates correctness. `var_dump`/`print_r` never reach production.
- **F**unctional immutability: `final readonly` classes, value objects, enums, asymmetric visibility; return new objects rather than mutate.
- **I**diomatic 8.4: constructor promotion, enums, first-class callable syntax (`$fn = strlen(...)`), property hooks, `new X()->method()`, `array_find/any/all`.
- **R**eproducible deps: Composer with a committed `composer.lock`; `composer audit` clean on every delivery.
- **S**elf-validating domain: invariants enforced in constructors/named factories; invalid state is unrepresentable.
- **T**ested first: every feature and bug fix is test-first via PHPUnit or Pest (policy: `tdd.md`).

**Verified Code**: Agent-generated PHP MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PHP-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PHP-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `vendor/bin/phpunit` (or `pest`) | exit 0, 0 skips |
| PHP-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `vendor/bin/phpunit` | failing→passing |
| PHP-TST-03 | Domain/business-logic coverage MUST meet the `tdd.md` gate | `XDEBUG_MODE=coverage vendor/bin/phpunit --coverage-text --coverage-html=var/cov` | meets gate |
| PHP-TYP-01 | Every file MUST declare `strict_types=1` | `! grep -rL 'declare(strict_types=1)' src/` | none missing |
| PHP-TYP-02 | Static analysis MUST pass at max level | `vendor/bin/phpstan analyse` (or `psalm`) | exit 0, level max |
| PHP-FMT-01 | Code MUST be PSR-12 formatted | `vendor/bin/php-cs-fixer fix --dry-run --diff` | no diff |
| PHP-SEC-01 | 0 known CVEs in dependencies (see `secure-coding.md`) | `composer audit` | 0 high/critical |
| PHP-SEC-02 | No SQL injection / unescaped output (see `secure-coding.md`) | review / PHPStan + parameterized queries | no string-built SQL/HTML |
| PHP-DEP-01 | Lockfile in sync & manifest valid | `composer validate --strict && composer install --dry-run` | in sync |
| PHP-DOC-01 | Public APIs MUST have PHPDoc (see `comments.md`) | review / `phpstan` (`@param`/`@return` types) | documented |
| PHP-ARCH-01 | Domain imports no infra/framework code (see `hexagonal.md`) | review / deptrac | no inward→outward |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, suppressing errors with `@`, using PHPStan-ignored `mixed` to dodge typing, string-concatenated SQL, `echo`-ing unescaped user input, or `composer require` without committing `composer.lock`.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
vendor/bin/php-cs-fixer fix --dry-run --diff        # PHP-FMT-01
vendor/bin/phpstan analyse                          # PHP-TYP-02 (max level; not the formatter)
grep -rL 'declare(strict_types=1)' src/ && exit 1   # PHP-TYP-01
XDEBUG_MODE=coverage vendor/bin/phpunit             # PHP-TST-01/02/03
composer audit                                      # PHP-SEC-01
composer validate --strict                          # PHP-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. PHP Specifics

The unique value of this guide. Examples illustrate the **language**, not generic concepts.

### A. Strict typing & modern declarations
Every file opens with `declare(strict_types=1);` — this disables silent scalar coercion and is non-negotiable. Type every property, parameter, and return.

```php
<?php
declare(strict_types=1);

namespace App\Domain\ValueObject;

final readonly class Money
{
    public function __construct(
        public int $amountMinor,        // constructor property promotion
        public Currency $currency,      // Currency is a backed enum
    ) {
        if ($amountMinor < 0) {
            throw new \InvalidArgumentException('Amount must be non-negative');
        }
    }

    public function add(self $other): self
    {
        return new self($this->amountMinor + $other->amountMinor, $this->currency);
    }
}
```
Avoid `mixed`; when generics are needed, express them in PHPDoc (`@param list<User> $users`) and let PHPStan/Psalm enforce them. `array` without a shape is a code smell — use `list<T>`, `array<K,V>`, or a typed object.

### B. Enums, readonly & immutability
Prefer enums over class constants for closed sets; prefer `final readonly` value objects over associative arrays.

```php
enum Status: string
{
    case Pending = 'pending';
    case Active  = 'active';

    public function isTerminal(): bool { return $this === self::Active; }
}
```
Value objects are immutable and self-validating via private constructors + named factories (`Email::fromString()`), exposing `equals()` for comparison. Never expose mutable getters/setters where a new instance can be returned.

### C. PHP 8.4 features (correct versions)
> The pre-2.0 draft mislabeled these as "8.5". **Property hooks, asymmetric visibility, `new X()->...` without parens, and `array_find/array_any/array_all` all shipped in PHP 8.4 (Nov 2024).**

```php
// Property hooks — validation/derivation without manual getters/setters (8.4)
class Temperature
{
    public float $celsius {
        set => $value < -273.15 ? throw new \ValueError('below absolute zero') : $value;
    }
    public float $fahrenheit {
        get => $this->celsius * 9 / 5 + 32;     // computed, no backing field
    }
}

// Asymmetric visibility — public read, private write (8.4)
class Order
{
    public private(set) string $id;
}

// First-class callable syntax (8.1+) and new-without-parens (8.4)
$lengths = array_map(strlen(...), $words);
$name = new User('Ada')->name;

// Array find/any/all (8.4) replace manual loops
$admin = array_find($users, fn (User $u): bool => $u->isAdmin());
```

### D. Error handling — PHP binding
Strategy is owned by [`error-handling.md`](guides://error-handling.md). In PHP: throw **typed** exceptions extending domain-specific base classes; never the bare `\Exception`. Never suppress with `@`. Catch narrowly; let unexpected throwables propagate to a top-level handler/logger (PSR-3). A `Result`/either type is acceptable for expected, recoverable outcomes in hot paths, but exceptions remain the default for exceptional flow.

```php
final class InsufficientFundsException extends \DomainException {}
```

### E. Concurrency — Fibers (language level)
PHP 8.1+ Fibers provide cooperative multitasking; libraries like AMPHP/ReactPHP build event loops on top. Concurrency *policy* (when to parallelize, structured concurrency) is owned by [`parallelism.md`](guides://parallelism.md). PHP binding: use a fiber-based runtime for non-blocking I/O and await concurrent operations rather than blocking sequentially.

```php
use function Amp\async;
use function Amp\Future\await;

[$user, $orders, $stats] = await([
    async(fn () => $userRepo->findById($id)),
    async(fn () => $orderRepo->findByUser($id)),
    async(fn () => $statsRepo->forUser($id)),
]);
```
Do not introduce an async runtime for code that is not I/O-bound; synchronous code is correct and simpler for CPU work.

### F. Common footguns → fix
- `==` loose comparison → use `===` / `!==` (strict).
- `null` from `array`/DB access → use null coalescing `??`, nullsafe `?->`, and typed `?T` returns.
- String-built SQL → **always** parameterized queries / prepared statements (PHP-SEC-02).
- Unescaped output in HTML → `htmlspecialchars(..., ENT_QUOTES)` or a template engine with auto-escaping.
- `array` as a pseudo-struct → a `final readonly` class or enum.
- Mutating method arguments → return a new value object.

> For design patterns applied here (Repository, Command/Handler, Factory), reference [`designpatterns.md`](guides://designpatterns.md) and show only the PHP binding (e.g. a repository **interface** as a hexagonal port).

---

## 5. Tooling & Dependencies (Composer)

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). PHP binding:

```bash
composer install                # install from composer.lock (reproducible)
composer install --no-dev       # production install
composer require <vendor/pkg>   # add dep (updates composer.lock)
composer update <vendor/pkg>    # update to latest within constraint
composer audit                  # PHP-SEC-01: CVE scan
composer validate --strict      # PHP-DEP-01: manifest/lock integrity
```
Commit `composer.lock`; `vendor/` is git-ignored. Use caret constraints (`^8.4`) per `semver.md`; pin direct deps and let Composer resolve transitives. `config.sort-packages: true` keeps `composer.json` deterministic.

```json
{
    "require": { "php": "^8.4", "psr/log": "^3.0" },
    "require-dev": {
        "phpunit/phpunit": "^11.5",
        "pestphp/pest": "^3.0",
        "phpstan/phpstan": "^2.0",
        "friendsofphp/php-cs-fixer": "^3.64"
    },
    "autoload":     { "psr-4": { "App\\":  "src/" } },
    "autoload-dev": { "psr-4": { "Tests\\": "tests/" } },
    "config": { "sort-packages": true }
}
```

### Tool config

```php
// .php-cs-fixer.php
return (new PhpCsFixer\Config())
    ->setRules([
        '@PSR12' => true,
        '@PHP84Migration' => true,
        'declare_strict_types' => true,
        'strict_param' => true,
        'array_syntax' => ['syntax' => 'short'],
        'ordered_imports' => ['sort_algorithm' => 'alpha'],
        'no_unused_imports' => true,
    ])
    ->setFinder(PhpCsFixer\Finder::create()->in([__DIR__.'/src', __DIR__.'/tests']));
```

```neon
# phpstan.neon
parameters:
    level: max
    paths: [src, tests]
    treatPhpDocTypesAsCertain: false
```

---

## 6. Project Structure

PSR-4 layout. Architectural *principles* (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their PHP mapping.

```
project/
├── src/                     # PSR-4 App\  (composer autoload)
│   ├── Domain/              # entities, value objects, enums, repository INTERFACES (ports)
│   ├── Application/         # use cases: Command/Handler, Query/Handler  (PHP-ARCH-01: no infra)
│   ├── Infrastructure/      # adapters: DB/HTTP/cache implementations of ports
│   └── Adapter/             # driving adapters: HTTP controllers, CLI commands
├── tests/                   # PSR-4 Tests\  — Unit/ mirrors src/, Integration/, E2E/ (see tdd.md)
├── composer.json / .lock    # deps (committed lock)
├── phpstan.neon             # static analysis (level max)
├── phpunit.xml              # test config
├── .php-cs-fixer.php        # PSR-12 style
└── README.md
```

- Group by domain/feature, not by type.
- Domain depends on nothing; Infrastructure *implements* Domain interfaces; dependencies point inward.
- Enforce the import boundary with `deptrac` (`composer require --dev qossmic/deptrac`).

---

## 7. Quick Reference

```bash
composer install                                    # setup
vendor/bin/phpunit                                  # test  (or: vendor/bin/pest)
vendor/bin/phpstan analyse                          # static analysis (max)
vendor/bin/php-cs-fixer fix                         # format (PSR-12)
composer audit                                      # CVE scan
php -l src/File.php                                 # syntax lint a file
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] PHP-FMT-01 — `php-cs-fixer fix --dry-run` clean (PSR-12)
- [ ] PHP-TYP-01 — every file declares `strict_types=1`
- [ ] PHP-TYP-02 — `phpstan`/`psalm` clean at max level (not the formatter)
- [ ] PHP-TST-01/02/03 — tests pass, bugs have regression tests, coverage meets gate
- [ ] PHP-DOC-01 — public APIs have PHPDoc
- [ ] PHP-SEC-01 — `composer audit` 0 high/critical CVEs
- [ ] PHP-SEC-02 — parameterized SQL, escaped output
- [ ] PHP-DEP-01 — `composer.lock` in sync & committed, manifest valid
- [ ] PHP-ARCH-01 — domain layer free of infra/framework imports
- [ ] Agent ran every §3 command and documented any fixes

---
**End of PHP Guidelines**

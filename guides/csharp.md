# C# Development Guidelines
Mandatory coding standards for C#: null-safe, async-correct, test-covered, DI-driven. .NET 9, C# 13, dotnet CLI, xUnit, Roslyn analyzers.

---
name: csharp
title: C# Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: language
tools: [dotnet@9.0, csharp@13, dotnet-cli, xunit, roslyn-analyzers, dotnet-format]
requires:
  - tdd
  - hexagonal
  - secure-coding
  - error-handling
recommends:
  - designpatterns
  - logging
  - observability
  - comments
  - semver
provides:
  - modern-csharp
  - nullable-refs
  - async-await
  - linq
  - dotnet-di
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to C# and .NET.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating C# code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(C# binding: runner is `dotnet test`; framework xUnit.)*
> - [`hexagonal.md`](guides://hexagonal.md) — layering, ports/adapters, dependency inversion. *(C# binding: ports are interfaces; adapters wired via the built-in DI container in `Program.cs`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(C# binding: `dotnet list package --vulnerable`, NuGet lock files, package signature verification.)*
> - [`error-handling.md`](guides://error-handling.md) — error strategy & propagation. *(C# binding: typed exceptions; no swallowing; `ProblemDetails` at the boundary.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`designpatterns.md`](guides://designpatterns.md) — GoF & friends; this guide shows only the C# binding.
> - [`logging.md`](guides://logging.md) — structured logging *(binding: `ILogger<T>`, message templates, source-generated `LoggerMessage`).*
> - [`observability.md`](guides://observability.md) — metrics/tracing *(binding: `System.Diagnostics.Activity`, OpenTelemetry .NET SDK).*
> - [`comments.md`](guides://comments.md) — API-doc policy *(binding: XML doc comments, `GenerateDocumentationFile`).*
> - [`semver.md`](guides://semver.md) — versioning of NuGet packages.

> 📎 **SEE ALSO:** [`cleanarch.md`](guides://cleanarch.md) · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md) · [`rest.md`](guides://rest.md) *(for ASP.NET Core HTTP APIs)*

---

## 1. Core Philosophies: DOTNET-FIRST

C#-specific principles only. TDD, security, error handling, and architecture come from §0.

- **D**efensive nullability: nullable reference types **on** solution-wide; the compiler is the contract — no `!` to silence warnings without justification.
- **O**bject model: records for data, interfaces for ports, sealed-by-default classes for behavior; favor composition over inheritance.
- **T**estable by construction: constructor injection through the built-in DI container; no `new` of dependencies, no static singletons holding state.
- **N**amespaced: file-scoped namespaces that mirror the folder/project layout; one public type per file.
- **E**fficient async: async all the way with `CancellationToken` flowing through; `Span<T>`/`Memory<T>` and pooling on hot paths.
- **T**yped strongly: lean on generics, pattern matching, and the type system instead of casts and `object`.

**Verified Code**: Agent-generated C# MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `CS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| CS-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `dotnet test` | exit 0, 0 skips |
| CS-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `dotnet test` | failing→passing |
| CS-TST-03 | Business-logic coverage MUST meet the project gate | `dotnet test --collect:"XPlat Code Coverage"` | ≥ project threshold |
| CS-FMT-01 | Code MUST be formatted | `dotnet format --verify-no-changes` | no diff |
| CS-LINT-01 | Roslyn analyzers MUST pass clean (warnings-as-errors) | `dotnet build` | 0 warnings, 0 errors |
| CS-TYP-01 | Nullable reference types MUST be enabled; no unjustified `!` or `#pragma` suppressions | `dotnet build` (Nullable=enable, WarningsAsErrors) | 0 CS86xx warnings |
| CS-DOC-01 | Public APIs MUST have XML doc comments (see `comments.md`) | `dotnet build` (GenerateDocumentationFile=true) | 0 CS1591 |
| CS-SEC-01 | 0 high/critical CVEs in deps, incl. transitive (see `secure-coding.md`) | `dotnet list package --vulnerable --include-transitive` | none high/critical |
| CS-DEP-01 | Lockfile in sync & restore reproducible (see `secure-coding.md`) | `dotnet restore --locked-mode` | restores clean |
| CS-ARCH-01 | Domain imports no infrastructure/framework code (see `hexagonal.md`) | architecture test / review | no inward→outward |
| CS-ASYNC-01 | No sync-over-async; no `async void` except event handlers | analyzer review (`.Result`/`.Wait()`/`async void` ban) | none found |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`), fixing a bug without a regression test first, blocking on async (`.Result`/`.Wait()`/`.GetAwaiter().GetResult()`), `async void` outside event handlers, suppressing nullable warnings with `!` instead of fixing the flow, or `[Fact(Skip=...)]` to bypass a failing test.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
dotnet format --verify-no-changes              # CS-FMT-01
dotnet build -warnaserror                      # CS-LINT-01, CS-TYP-01, CS-DOC-01
dotnet test --collect:"XPlat Code Coverage"    # CS-TST-01/02/03
dotnet list package --vulnerable --include-transitive   # CS-SEC-01
dotnet restore --locked-mode                   # CS-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Idiomatic multi-project solution layout. Architectural principles (dependency direction, ports/adapters, acyclic deps) are owned by [`hexagonal.md`](guides://hexagonal.md); below is only their C# mapping.

```
MySolution/
├── src/
│   ├── MyApp.Domain/          # pure business logic — no framework/IO refs (CS-ARCH-01)
│   ├── MyApp.Application/      # use cases; depends only on Domain + port interfaces
│   ├── MyApp.Infrastructure/   # adapters: EF Core, HTTP, messaging — implements ports
│   └── MyApp.Api/             # composition root: Program.cs wires DI, hosts endpoints
├── tests/
│   ├── MyApp.UnitTests/       # mirrors src/ (see tdd.md)
│   ├── MyApp.IntegrationTests/
│   └── MyApp.ArchitectureTests/   # asserts CS-ARCH-01 (e.g. NetArchTest)
├── Directory.Build.props      # shared MSBuild props (Nullable, analyzers, warnings)
├── Directory.Packages.props   # central package management (versions in one place)
├── packages.lock.json         # committed lockfile (CS-DEP-01)
└── MySolution.sln
```

- Dependencies point inward: `Api → Infrastructure → Application → Domain`. `Domain` references nothing.
- Enforce the boundary with an architecture test project (NetArchTest / ArchUnitNET), not just convention.
- File-scoped namespaces mirror the folder path; one public type per file.

---

## 5. C# Specifics

The unique value of this guide. Targets **.NET 9 / C# 13**.

### A. Solution-wide configuration — `Directory.Build.props`
Set the compiler contract once for every project.

```xml
<Project>
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <LangVersion>13</LangVersion>
    <Nullable>enable</Nullable>                 <!-- CS-TYP-01 -->
    <ImplicitUsings>enable</ImplicitUsings>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>  <!-- CS-LINT-01 -->
    <AnalysisLevel>latest-all</AnalysisLevel>
    <EnforceCodeStyleInBuild>true</EnforceCodeStyleInBuild>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>  <!-- CS-DOC-01 -->
    <RestorePackagesWithLockFile>true</RestorePackagesWithLockFile>  <!-- CS-DEP-01 -->
  </PropertyGroup>
</Project>
```

### B. Nullable reference types
Nullability is a compile-time contract — model intent, don't silence the compiler.

```csharp
public sealed class User
{
    public required string Email { get; init; }   // non-null, enforced at construction
    public string? MiddleName { get; init; }       // explicitly optional
}

public User? FindUser(int id) => _repository.Find(id);   // null = "not found"

var name   = user?.MiddleName ?? "Unknown";              // null-coalescing
var length = user?.Email.Length ?? 0;
if (user is { Email: var email }) Send(email);            // property pattern narrows to non-null
```

Guard at boundaries with the built-in throw helpers:

```csharp
ArgumentNullException.ThrowIfNull(order);
ArgumentException.ThrowIfNullOrWhiteSpace(order.ProductId);
ArgumentOutOfRangeException.ThrowIfNegativeOrZero(order.Quantity);
```

**Footgun:** the null-forgiving `!` operator hides bugs. Use it only to assert an invariant the compiler can't see, with a comment — never to make a warning disappear (violates CS-TYP-01).

### C. Records, pattern matching & immutability
Records for data/DTOs/value objects; `with` for non-destructive updates.

```csharp
public record UserDto(int Id, string Name, string Email);
public readonly record struct Money(decimal Amount, string Currency);  // small value type

var updated = original with { Name = "New Name" };
```

Switch on shape with patterns instead of type-check ladders:

```csharp
decimal Discount(Customer c) => c switch
{
    { Tier: Tier.Gold, Orders: > 100 } => 0.20m,
    { Tier: Tier.Gold }                => 0.10m,
    { Orders: > 50 }                   => 0.05m,
    _                                  => 0m,
};
```

Prefer immutable state: `init` setters, `IReadOnlyList<T>` on the surface, `System.Collections.Immutable` when callers must not mutate shared state.

### D. Async/await
Async all the way; flow `CancellationToken` through every awaitable call.

```csharp
public async Task<User> GetUserAsync(int id, CancellationToken ct = default)
{
    var user = await _repository.FindAsync(id, ct);
    return user ?? throw new NotFoundException(nameof(User), id);
}

// Library code: avoid forcing the caller's sync context.
var resp = await _httpClient.GetAsync(url, ct).ConfigureAwait(false);

// Parallelize independent work.
await Task.WhenAll(userTask, ordersTask);
```

**Footguns (CS-ASYNC-01):**
- `.Result` / `.Wait()` / `.GetAwaiter().GetResult()` → deadlocks and lost exceptions. Stay async.
- `async void` → exceptions can't be observed. Use `async Task`; `async void` only for event handlers.
- Forgetting `ct` → uncancellable work. Thread it everywhere, including loops (`ct.ThrowIfCancellationRequested()`).
- `IAsyncEnumerable<T>` for streamed results; `await foreach (var x in source.WithCancellation(ct))`.

### E. LINQ
Project before materializing; enumerate once; keep filtering at the data source.

```csharp
var dtos = await db.Users
    .Where(u => u.IsActive)
    .Select(u => new UserDto(u.Id, u.Name, u.Email))   // shape in the DB, not in memory
    .ToListAsync(ct);
```

**Footguns:**
- `ToList()` then `Where(...)` materializes the whole table before filtering — filter first.
- Lazy navigation in a loop = N+1 queries → use `.Include(...)` (EF Core) or a projection.
- Re-enumerating an `IEnumerable<T>` runs the query twice — materialize once (`var list = q.ToList();`).
- `Single`/`First` vs `*OrDefault`: pick the one whose throwing/empty semantics you actually want.

### F. Dependency injection — built-in container
Ports are interfaces; the composition root (`Program.cs`) is the only place that knows concrete adapters. Architectural rationale lives in [`hexagonal.md`](guides://hexagonal.md).

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.AddScoped<IOrderService, OrderService>();      // per request
builder.Services.AddSingleton<ICacheService, MemoryCacheService>();
builder.Services.AddTransient<IEmailSender, SmtpEmailSender>();
builder.Services.AddHttpClient<IPaymentGateway, StripePaymentGateway>();
builder.Services.Configure<EmailOptions>(builder.Configuration.GetSection("Email"));  // options pattern
```

Inject via constructor (primary constructors keep it terse); never `new` a dependency or reach for a static singleton:

```csharp
public sealed class OrderService(IOrderRepository repository, ILogger<OrderService> logger)
    : IOrderService
{
    public async Task<Order> CreateAsync(CreateOrderDto dto, CancellationToken ct)
    {
        logger.LogInformation("Creating order for {CustomerId}", dto.CustomerId);  // see logging.md
        var order = new Order(dto.CustomerId, dto.Items);
        await repository.AddAsync(order, ct);
        return order;
    }
}
```

Lifetimes: Singleton (stateless/thread-safe), Scoped (per request, e.g. `DbContext`), Transient (cheap, stateless). Never inject a Scoped service into a Singleton (captive dependency).

### G. Resource lifetime — `IDisposable` / `IAsyncDisposable`
Own a disposable → release it deterministically with `using`. Implement the pattern only when you hold unmanaged or disposable state.

```csharp
await using var conn = new SqlConnection(connectionString);   // IAsyncDisposable
using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

public sealed class FileCache : IAsyncDisposable
{
    private readonly Stream _stream;
    public async ValueTask DisposeAsync() => await _stream.DisposeAsync();
}
```

Prefer DI-managed lifetimes (the container disposes Scoped/Transient `IDisposable` services) over manual disposal where possible.

### H. Exceptions — C# binding
Strategy (when to throw, wrap, retry, propagate) is owned by [`error-handling.md`](guides://error-handling.md). C# binding: derive a small typed hierarchy, never swallow, translate to transport at the edge.

```csharp
public abstract class DomainException(string code, string message) : Exception(message)
{
    public string Code { get; } = code;
}
public sealed class NotFoundException(string entity, object id)
    : DomainException("NOT_FOUND", $"{entity} '{id}' was not found");
```

At an ASP.NET Core boundary, map exceptions to `ProblemDetails` once via `IExceptionHandler` / `AddProblemDetails()` (the modern replacement for hand-rolled middleware) — don't scatter try/catch per endpoint.

### I. Design patterns — C# binding
GoF and friends are owned by [`designpatterns.md`](guides://designpatterns.md). In C#, prefer language/runtime features over hand-rolled patterns: built-in DI for Factory/Strategy wiring, `IOptions<T>` for configuration, `IAsyncEnumerable<T>` for Iterator, records + `with` for immutable Builder-style updates, and source generators (`[GeneratedRegex]`, `LoggerMessage`) instead of reflection-heavy machinery.

### J. Observability — binding
Policy in [`logging.md`](guides://logging.md) / [`observability.md`](guides://observability.md). C#: structured logging via `ILogger<T>` with message templates (`logger.LogInformation("Order {OrderId}", id)`) — never string-interpolate the message. Use source-generated `[LoggerMessage]` on hot paths. Tracing/metrics via `System.Diagnostics.ActivitySource` + `Meter`, exported with the OpenTelemetry .NET SDK.

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). C# binding uses NuGet with Central Package Management.

```xml
<!-- Directory.Packages.props — one version per package, solution-wide -->
<Project>
  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
    <CentralPackageTransitivePinningEnabled>true</CentralPackageTransitivePinningEnabled>
  </PropertyGroup>
  <ItemGroup>
    <PackageVersion Include="Microsoft.Extensions.Logging" Version="9.0.0" />
    <PackageVersion Include="xunit" Version="2.9.2" />
  </ItemGroup>
</Project>
```

```bash
dotnet restore --locked-mode                 # CS-DEP-01: reproducible, no tampering
dotnet add package <pkg>                      # add (updates Directory.Packages.props + lockfile)
dotnet list package --outdated                # find upgrades
dotnet list package --vulnerable --include-transitive   # CS-SEC-01: CVE scan
```

Commit `packages.lock.json`. Verify NuGet package signatures for external dependencies. Keep secrets out of `appsettings.json` — use user-secrets in dev and a secret store in prod (see `secure-coding.md`).

---

## 7. Quick Reference

```bash
dotnet build -warnaserror                    # build (lint + nullable + docs gates)
dotnet test                                  # test
dotnet format                                # format
dotnet format --verify-no-changes            # format check
dotnet list package --vulnerable --include-transitive   # security
dotnet run --project src/MyApp.Api           # run
```

```csharp
// C# 13 idioms worth reaching for
private readonly System.Threading.Lock _gate = new();   // .NET 9 Lock type
using (_gate.EnterScope()) { /* thread-safe */ }

public void Process(params ReadOnlySpan<int> numbers)   // allocation-free params
{
    foreach (var n in numbers) { /* ... */ }
}

var json = """
    { "name": "John", "age": 30 }
    """;                                                 // raw string literal
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] CS-FMT-01 — `dotnet format --verify-no-changes` clean
- [ ] CS-LINT-01 — `dotnet build` 0 warnings/errors (warnings-as-errors)
- [ ] CS-TYP-01 — nullable enabled, no unjustified `!`/suppressions
- [ ] CS-TST-01/02/03 — tests pass, bugs have regression tests, coverage ≥ gate
- [ ] CS-DOC-01 — public APIs have XML docs (0 CS1591)
- [ ] CS-SEC-01 — `dotnet list package --vulnerable --include-transitive` clean
- [ ] CS-DEP-01 — `packages.lock.json` in sync, `--locked-mode` restores clean
- [ ] CS-ARCH-01 — domain free of infrastructure/framework references
- [ ] CS-ASYNC-01 — no sync-over-async, no stray `async void`
- [ ] Agent ran every §3 command and documented any fixes

---
**End of C# Development Guidelines**

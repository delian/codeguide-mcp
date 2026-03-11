# C# Development Guidelines
Mandatory standards for C# development, following modern .NET patterns and best practices. .NET 9+, Visual Studio, Rider, Roslyn analyzers, dotnet CLI.

---

**Agent Profile**: The C# Expert
**Role**: Senior .NET Developer & Software Architect
**Objective**: Generate clean, efficient, and maintainable C# code following Microsoft and community best practices.
**Tools**: .NET 9+, Visual Studio, Rider, Roslyn analyzers, dotnet CLI.

---

## 1. Core Philosophies: DOTNET-FIRST

The agent must adhere to the **DOTNET-FIRST** principles for every C# implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.
**Security-First**: Mandatory vulnerability scanning, dependency auditing, and supply chain integrity checks.

- **D**efensive: Null safety, validation, and proper exception handling.
- **O**bject-Oriented: Proper use of classes, interfaces, and inheritance.
- **T**estable: Design for unit testing with dependency injection.
- **N**amespaced: Logical organization with proper namespace hierarchy.
- **E**fficient: Use async/await, spans (`ReadOnlySpan<T>`), and memory-efficient patterns.
- **T**yped: Leverage the strong type system and generics.

**Verified Code**: Agent-generated code MUST compile and pass security audits before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated C# code compiles and passes tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY C# code, the agent MUST:**

1. **Compilation Check**:
   ```bash
   # Verify project compiles without errors
   dotnet build
   # Exit code MUST be 0
   ```
   - **MUST** return 0 errors and 0 warnings.
   - All nullable reference type warnings must be addressed.

2. **Test Execution**:
   ```bash
   # Run all tests
   dotnet test
   ```
   - **MUST** pass all tests (100% pass rate).
   - Verify coverage meets project requirements (min 80%).

3. **Security & Dependency Verification (MANDATORY)**:
   ```bash
   # Scan for vulnerabilities in dependencies
   dotnet list package --vulnerable
   
   # Check for outdated dependencies
   dotnet list package --outdated
   ```
   - **MUST** have 0 high/critical vulnerabilities.
   - Dependencies MUST be pinned to secure versions.
   - Supply chain integrity (lockfiles) MUST be verified if `Directory.Packages.props` is used.

4. **Documentation Verification**:
   - All public APIs have XML documentation comments.
   - Examples provided for complex APIs.

#### Error Correction Process

If verification fails:

1. **Identify the error**: Read the full `dotnet build` or `dotnet test` error message.
2. **Locate the source**: Identify which project or file failed.
3. **Fix the root cause**:
   - Null safety violation? Add null check or `required` keyword.
   - Dependency vulnerability? Update package version in `Directory.Packages.props`.
4. **Re-verify**: Run build and audits again until they succeed.

### B. Agent Workflow Example

**Complete C# generation workflow:**

1. **Generate Code Structure**:
   ```
   src/
   ├── MyApp.Core/
   │   └── Entities/User.cs
   tests/
   └── MyApp.UnitTests/
       └── UserTests.cs
   ```

2. **Generate Initial Code**:
   ```csharp
   public record User(int Id, string Name);
   ```

3. **Verify**:
   ```bash
   dotnet build
   # ✓ Build successful
   ```

4. **Add Tests**:
   ```csharp
   [Fact]
   public void CreateUser_Works() { ... }
   ```

5. **Run Tests**:
   ```bash
   dotnet test
   # ✓ All tests pass
   ```

6. **Final Verification**:
   ```bash
   dotnet list package --vulnerable
   # ✓ No vulnerabilities found
   ```

7. **Present Code**: Only after ALL checks pass

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

### A. Solution Organization

```
MySolution/
├── src/
│   ├── MyApp.Api/                    # Web API project
│   │   ├── Controllers/
│   │   ├── Middleware/
│   │   ├── Filters/
│   │   └── Program.cs
│   ├── MyApp.Core/                   # Domain/business logic
│   │   ├── Entities/
│   │   ├── Interfaces/
│   │   ├── Services/
│   │   └── Exceptions/
│   ├── MyApp.Infrastructure/         # Data access, external services
│   │   ├── Data/
│   │   ├── Repositories/
│   │   └── ExternalServices/
│   └── MyApp.Shared/                 # Shared DTOs, utilities
│       ├── DTOs/
│       └── Extensions/
├── tests/
│   ├── MyApp.UnitTests/
│   ├── MyApp.IntegrationTests/
│   └── MyApp.ArchitectureTests/
├── Directory.Build.props             # Shared MSBuild properties
├── Directory.Packages.props          # Central package management
└── MySolution.sln
```

### B. Directory.Build.props

```xml
<Project>
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <AnalysisLevel>latest-all</AnalysisLevel>
    <EnforceCodeStyleInBuild>true</EnforceCodeStyleInBuild>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.CodeAnalysis.NetAnalyzers" Version="8.*">
      <PrivateAssets>all</PrivateAssets>
      <IncludeAssets>runtime; build; native; analyzers</IncludeAssets>
    </PackageReference>
  </ItemGroup>
</Project>
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for C#

```csharp
// Step 1: RED - Write failing test first
using Xunit;

public class EmailValidatorTests
{
    [Fact]
    public void Validate_WithValidEmail_ReturnsEmail()
    {
        var result = EmailValidator.Validate("user@example.com");

        Assert.True(result.IsValid);
        Assert.Equal("user@example.com", result.Email);
    }

    [Fact]
    public void Validate_WithoutAtSymbol_ReturnsInvalid()
    {
        var result = EmailValidator.Validate("invalid-email");

        Assert.False(result.IsValid);
        Assert.Equal("Invalid email format", result.Error);
    }

    [Fact]
    public void Validate_WithEmptyString_ReturnsInvalid()
    {
        var result = EmailValidator.Validate("");

        Assert.False(result.IsValid);
    }
}

// Run: dotnet test --filter "FullyQualifiedName~EmailValidatorTests"
// FAILS - EmailValidator class does not exist

// Step 2: GREEN - Write minimal implementation
public record EmailValidationResult(bool IsValid, string? Email = null, string? Error = null);

public static class EmailValidator
{
    public static EmailValidationResult Validate(string email)
    {
        if (email.Contains('@'))
            return new EmailValidationResult(true, Email: email);

        return new EmailValidationResult(false, Error: "Invalid email format");
    }
}

// Run: dotnet test --filter "FullyQualifiedName~EmailValidatorTests"
// PASSES - all tests pass

// Step 3: REFACTOR - Improve with regex validation
using System.Text.RegularExpressions;

public static partial class EmailValidator
{
    [GeneratedRegex(@"^[^\s@]+@[^\s@]+\.[^\s@]+$")]
    private static partial Regex EmailRegex();

    public static EmailValidationResult Validate(string email)
    {
        if (EmailRegex().IsMatch(email))
            return new EmailValidationResult(true, Email: email.ToLowerInvariant());

        return new EmailValidationResult(false, Error: "Invalid email format");
    }
}
// Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```csharp
// Bug Report #1042: EmailValidator accepts emails with spaces like "user @example.com"

// Step 1-2: Write test that reproduces the bug
public class EmailValidatorTests
{
    // Regression test for Bug #1042
    [Theory]
    [InlineData("user @example.com")]
    [InlineData(" user@example.com")]
    [InlineData("user@example.com ")]
    public void Validate_WithSpacesInEmail_ReturnsInvalid(string email)
    {
        var result = EmailValidator.Validate(email);

        Assert.False(result.IsValid);
    }
}

// Run: dotnet test --filter "FullyQualifiedName~EmailValidatorTests"
// FAILS - Validate returns IsValid=true for emails with spaces

// Step 3: Fix the bug
public static partial class EmailValidator
{
    [GeneratedRegex(@"^[^\s@]+@[^\s@]+\.[^\s@]+$")]
    private static partial Regex EmailRegex();

    public static EmailValidationResult Validate(string email)
    {
        if (email != email.Trim())
            return new EmailValidationResult(false, Error: "Invalid email format");

        if (EmailRegex().IsMatch(email))
            return new EmailValidationResult(true, Email: email.ToLowerInvariant());

        return new EmailValidationResult(false, Error: "Invalid email format");
    }
}

// Run: dotnet test --filter "FullyQualifiedName~EmailValidatorTests"
// PASSES - bug fixed, regression prevented
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Use `[Fact(Skip = "...")]` to bypass failing tests instead of fixing them

---

## 3. Naming Conventions (MANDATORY)

### A. General Rules

```csharp
// ✅ PascalCase for types, methods, properties, events
public class UserService { }
public interface IUserRepository { }
public void ProcessOrder() { }
public string FirstName { get; set; }
public event EventHandler OrderCompleted;

// ✅ camelCase for parameters, local variables
public void CreateUser(string userName, int userId)
{
    var localVariable = 42;
    var isValid = true;
}

// ✅ _camelCase for private fields
private readonly ILogger<UserService> _logger;
private int _counter;

// ✅ UPPER_CASE for constants (or PascalCase)
public const int MaxRetryCount = 3;
public const string DefaultConnectionString = "...";

// ✅ Prefix interfaces with I
public interface IOrderService { }

// ✅ Suffix async methods with Async
public async Task<User> GetUserAsync(int id);

// ❌ WRONG: Hungarian notation
int iCounter;  // Don't prefix with type
string strName; // Don't prefix with type
```

### B. Meaningful Names

```csharp
// ❌ WRONG: Unclear names
public class Mgr { }
public int Calc(int x, int y);
public bool Check();

// ✅ CORRECT: Descriptive names
public class OrderManager { }
public int CalculateTotal(int quantity, int unitPrice);
public bool IsValidEmail(string email);
```

---

## 4. Null Safety (MANDATORY)

### A. Nullable Reference Types

```csharp
// Enable nullable reference types (in .csproj or globally)
#nullable enable

public class User
{
    // Non-nullable - must be initialized
    public required string Email { get; init; }

    // Nullable - can be null
    public string? MiddleName { get; set; }

    // Constructor ensures non-nullable fields are set
    public User(string email)
    {
        Email = email ?? throw new ArgumentNullException(nameof(email));
    }
}

// Handle nullable returns
public User? FindUser(int id)
{
    // May return null
    return _repository.Find(id);
}

// Use null-conditional operators
var length = user?.Name?.Length ?? 0;
user?.SendNotification();

// Use null-coalescing
var name = user?.Name ?? "Unknown";
var list = items ?? [];

// Pattern matching for null checks
if (user is not null)
{
    Console.WriteLine(user.Name);
}

if (user is { Email: var email })
{
    SendEmail(email);
}
```

### B. Guard Clauses

```csharp
public class OrderService
{
    public void ProcessOrder(Order order, Customer customer)
    {
        // Guard clauses at the top
        ArgumentNullException.ThrowIfNull(order);
        ArgumentNullException.ThrowIfNull(customer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(order.Quantity);

        if (string.IsNullOrWhiteSpace(order.ProductId))
        {
            throw new ArgumentException("Product ID is required", nameof(order));
        }

        // Main logic after guards
        // ..
    }
}
```

---

## 5. Async/Await (MANDATORY)

### A. Proper Async Patterns

```csharp
// ✅ CORRECT: Async all the way
public async Task<User> GetUserAsync(int id, CancellationToken ct = default)
{
    var user = await _repository.FindAsync(id, ct);
    return user ?? throw new NotFoundException($"User {id} not found");
}

// ✅ CORRECT: ConfigureAwait in library code
public async Task<Data> FetchDataAsync(CancellationToken ct)
{
    var response = await _httpClient.GetAsync(url, ct).ConfigureAwait(false);
    return await response.Content.ReadFromJsonAsync<Data>(ct).ConfigureAwait(false);
}

// ✅ CORRECT: Parallel async operations
public async Task<(User, Order[])> GetUserWithOrdersAsync(int userId, CancellationToken ct)
{
    var userTask = _userRepository.GetAsync(userId, ct);
    var ordersTask = _orderRepository.GetByUserAsync(userId, ct);

    await Task.WhenAll(userTask, ordersTask);

    return (await userTask, await ordersTask);
}

// ❌ WRONG: Blocking on async code
public User GetUser(int id)
{
    return GetUserAsync(id).Result;  // Deadlock risk!
    return GetUserAsync(id).GetAwaiter().GetResult();  // Still blocking
}

// ❌ WRONG: async void (except event handlers)
public async void ProcessData()  // Can't await, exceptions lost
{
    await DoWorkAsync();
}

// ✅ CORRECT: async Task
public async Task ProcessDataAsync()
{
    await DoWorkAsync();
}
```

### B. Cancellation Tokens

```csharp
public class DataService
{
    public async Task<IEnumerable<Item>> GetItemsAsync(
        int pageSize,
        CancellationToken cancellationToken = default)
    {
        // Check for cancellation
        cancellationToken.ThrowIfCancellationRequested();

        // Pass cancellation token to all async operations
        var items = await _repository.GetAllAsync(cancellationToken);

        // For long-running loops
        foreach (var item in items)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await ProcessItemAsync(item, cancellationToken);
        }

        return items;
    }
}
```

---

## 6. Dependency Injection (MANDATORY)

### A. Service Registration

```csharp
// Program.cs
var builder = WebApplication.CreateBuilder(args);

// Register services with appropriate lifetimes
builder.Services.AddScoped<IOrderService, OrderService>();
builder.Services.AddSingleton<ICacheService, MemoryCacheService>();
builder.Services.AddTransient<IEmailSender, SmtpEmailSender>();

// Register with factory for complex construction
builder.Services.AddScoped<IDbConnection>(sp =>
{
    var config = sp.GetRequiredService<IConfiguration>();
    return new SqlConnection(config.GetConnectionString("Default"));
});

// Register options pattern
builder.Services.Configure<EmailOptions>(
    builder.Configuration.GetSection("Email"));

// Register HttpClient with typed client
builder.Services.AddHttpClient<IPaymentGateway, StripePaymentGateway>(client =>
{
    client.BaseAddress = new Uri("https://api.stripe.com");
    client.DefaultRequestHeaders.Add("Accept", "application/json");
});
```

### B. Constructor Injection

```csharp
public class OrderService : IOrderService
{
    private readonly IOrderRepository _repository;
    private readonly ILogger<OrderService> _logger;
    private readonly IEmailSender _emailSender;

    // Primary constructor (C# 12)
    public OrderService(
        IOrderRepository repository,
        ILogger<OrderService> logger,
        IEmailSender emailSender)
    {
        _repository = repository;
        _logger = logger;
        _emailSender = emailSender;
    }

    public async Task<Order> CreateOrderAsync(CreateOrderDto dto, CancellationToken ct)
    {
        _logger.LogInformation("Creating order for customer {CustomerId}", dto.CustomerId);

        var order = new Order(dto.CustomerId, dto.Items);
        await _repository.AddAsync(order, ct);

        await _emailSender.SendOrderConfirmationAsync(order, ct);

        return order;
    }
}
```

---

## 7. Exception Handling (MANDATORY)

### A. Custom Exceptions

```csharp
// Base domain exception
public abstract class DomainException : Exception
{
    public string Code { get; }

    protected DomainException(string code, string message) : base(message)
    {
        Code = code;
    }
}

// Specific exceptions
public class NotFoundException : DomainException
{
    public NotFoundException(string entityName, object id)
        : base("NOT_FOUND", $"{entityName} with ID '{id}' was not found")
    {
    }
}

public class ValidationException : DomainException
{
    public IReadOnlyDictionary<string, string[]> Errors { get; }

    public ValidationException(IDictionary<string, string[]> errors)
        : base("VALIDATION_ERROR", "One or more validation errors occurred")
    {
        Errors = errors.AsReadOnly();
    }
}

public class ConflictException : DomainException
{
    public ConflictException(string message)
        : base("CONFLICT", message)
    {
    }
}
```

### B. Exception Handling Middleware

```csharp
public class ExceptionHandlingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<ExceptionHandlingMiddleware> _logger;

    public ExceptionHandlingMiddleware(RequestDelegate next, ILogger<ExceptionHandlingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await _next(context);
        }
        catch (Exception ex)
        {
            await HandleExceptionAsync(context, ex);
        }
    }

    private async Task HandleExceptionAsync(HttpContext context, Exception exception)
    {
        var (statusCode, response) = exception switch
        {
            NotFoundException ex => (StatusCodes.Status404NotFound,
                new ProblemDetails { Title = ex.Message, Status = 404 }),

            ValidationException ex => (StatusCodes.Status400BadRequest,
                new ValidationProblemDetails(ex.Errors) { Status = 400 }),

            ConflictException ex => (StatusCodes.Status409Conflict,
                new ProblemDetails { Title = ex.Message, Status = 409 }),

            _ => (StatusCodes.Status500InternalServerError,
                new ProblemDetails { Title = "An error occurred", Status = 500 })
        };

        if (statusCode == 500)
        {
            _logger.LogError(exception, "Unhandled exception occurred");
        }

        context.Response.StatusCode = statusCode;
        await context.Response.WriteAsJsonAsync(response);
    }
}
```

---

## 8. LINQ Best Practices (MANDATORY)

### A. Efficient LINQ Usage

```csharp
// ✅ CORRECT: Use appropriate methods
var firstUser = users.FirstOrDefault(u => u.IsActive);
var singleAdmin = users.SingleOrDefault(u => u.Role == "Admin");
var hasActiveUsers = users.Any(u => u.IsActive);
var activeCount = users.Count(u => u.IsActive);

// ✅ CORRECT: Project before materializing
var userDtos = await _context.Users
    .Where(u => u.IsActive)
    .Select(u => new UserDto(u.Id, u.Name, u.Email))  // Select only needed fields
    .ToListAsync(ct);

// ❌ WRONG: Multiple enumerations
var users = GetUsers();  // IEnumerable
var count = users.Count();  // First enumeration
var list = users.ToList();  // Second enumeration

// ✅ CORRECT: Materialize once
var users = GetUsers().ToList();
var count = users.Count;

// ✅ CORRECT: Use method syntax for complex queries
var results = await _context.Orders
    .Where(o => o.Status == OrderStatus.Pending)
    .Where(o => o.CreatedAt >= startDate)
    .OrderByDescending(o => o.Total)
    .Take(10)
    .Select(o => new OrderSummary
    {
        Id = o.Id,
        CustomerName = o.Customer.Name,
        Total = o.Total
    })
    .ToListAsync(ct);
```

### B. Avoid Common Mistakes

```csharp
// ❌ WRONG: Calling ToList() too early
var users = _context.Users.ToList()  // Loads ALL users
    .Where(u => u.IsActive);

// ✅ CORRECT: Filter in database
var users = await _context.Users
    .Where(u => u.IsActive)
    .ToListAsync(ct);

// ❌ WRONG: N+1 query problem
var orders = await _context.Orders.ToListAsync(ct);
foreach (var order in orders)
{
    var customer = order.Customer;  // Lazy loading = N queries
}

// ✅ CORRECT: Eager loading
var orders = await _context.Orders
    .Include(o => o.Customer)
    .ToListAsync(ct);
```

---

## 9. Records and Immutability (MANDATORY)

### A. Record Types

```csharp
// Immutable record (preferred for DTOs)
public record UserDto(int Id, string Name, string Email);

// Record with additional members
public record OrderDto(int Id, decimal Total, DateTime CreatedAt)
{
    public string FormattedTotal => Total.ToString("C");
}

// Record with init-only properties
public record CreateUserRequest
{
    public required string Email { get; init; }
    public required string Name { get; init; }
    public string? Phone { get; init; }
}

// Use with expression for immutable updates
var updated = original with { Name = "New Name" };

// Record struct for small value types
public readonly record struct Point(int X, int Y);
public readonly record struct Money(decimal Amount, string Currency);
```

### B. Immutable Collections

```csharp
using System.Collections.Immutable;

public class ShoppingCart
{
    private readonly ImmutableList<CartItem> _items;

    public IReadOnlyList<CartItem> Items => _items;

    public ShoppingCart() : this(ImmutableList<CartItem>.Empty) { }

    private ShoppingCart(ImmutableList<CartItem> items)
    {
        _items = items;
    }

    public ShoppingCart AddItem(CartItem item)
    {
        return new ShoppingCart(_items.Add(item));
    }

    public ShoppingCart RemoveItem(int itemId)
    {
        var index = _items.FindIndex(i => i.Id == itemId);
        return index >= 0 ? new ShoppingCart(_items.RemoveAt(index)) : this;
    }
}
```

---

## 10. Entity Framework Core (MANDATORY)

### A. DbContext Configuration

```csharp
public class AppDbContext : DbContext
{
    public DbSet<User> Users => Set<User>();
    public DbSet<Order> Orders => Set<Order>();

    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.ApplyConfigurationsFromAssembly(typeof(AppDbContext).Assembly);
    }

    public override async Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
        // Auto-set timestamps
        foreach (var entry in ChangeTracker.Entries<ITimestamped>())
        {
            if (entry.State == EntityState.Added)
            {
                entry.Entity.CreatedAt = DateTime.UtcNow;
            }
            entry.Entity.UpdatedAt = DateTime.UtcNow;
        }

        return await base.SaveChangesAsync(cancellationToken);
    }
}

// Entity configuration
public class UserConfiguration : IEntityTypeConfiguration<User>
{
    public void Configure(EntityTypeBuilder<User> builder)
    {
        builder.ToTable("users");

        builder.HasKey(u => u.Id);

        builder.Property(u => u.Email)
            .HasMaxLength(255)
            .IsRequired();

        builder.HasIndex(u => u.Email)
            .IsUnique();

        builder.HasMany(u => u.Orders)
            .WithOne(o => o.User)
            .HasForeignKey(o => o.UserId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
```

### B. Repository Pattern

```csharp
public interface IRepository<T> where T : class
{
    Task<T?> GetByIdAsync(int id, CancellationToken ct = default);
    Task<IReadOnlyList<T>> GetAllAsync(CancellationToken ct = default);
    Task AddAsync(T entity, CancellationToken ct = default);
    void Update(T entity);
    void Remove(T entity);
}

public class Repository<T> : IRepository<T> where T : class
{
    protected readonly AppDbContext Context;
    protected readonly DbSet<T> DbSet;

    public Repository(AppDbContext context)
    {
        Context = context;
        DbSet = context.Set<T>();
    }

    public virtual async Task<T?> GetByIdAsync(int id, CancellationToken ct = default)
    {
        return await DbSet.FindAsync([id], ct);
    }

    public virtual async Task<IReadOnlyList<T>> GetAllAsync(CancellationToken ct = default)
    {
        return await DbSet.ToListAsync(ct);
    }

    public async Task AddAsync(T entity, CancellationToken ct = default)
    {
        await DbSet.AddAsync(entity, ct);
    }

    public void Update(T entity)
    {
        DbSet.Update(entity);
    }

    public void Remove(T entity)
    {
        DbSet.Remove(entity);
    }
}
```

---

## 11. Testing (MANDATORY)

### A. Unit Tests with xUnit

```csharp
public class OrderServiceTests
{
    private readonly Mock<IOrderRepository> _repositoryMock;
    private readonly Mock<ILogger<OrderService>> _loggerMock;
    private readonly OrderService _sut;

    public OrderServiceTests()
    {
        _repositoryMock = new Mock<IOrderRepository>();
        _loggerMock = new Mock<ILogger<OrderService>>();
        _sut = new OrderService(_repositoryMock.Object, _loggerMock.Object);
    }

    [Fact]
    public async Task CreateOrder_WithValidData_ReturnsOrder()
    {
        // Arrange
        var dto = new CreateOrderDto(CustomerId: 1, Items: [new OrderItem(1, 2)]);
        _repositoryMock
            .Setup(r => r.AddAsync(It.IsAny<Order>(), It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        // Act
        var result = await _sut.CreateOrderAsync(dto, CancellationToken.None);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(dto.CustomerId, result.CustomerId);
        _repositoryMock.Verify(r => r.AddAsync(It.IsAny<Order>(), It.IsAny<CancellationToken>()), Times.Once);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public async Task CreateOrder_WithInvalidCustomerId_ThrowsValidationException(int customerId)
    {
        // Arrange
        var dto = new CreateOrderDto(CustomerId: customerId, Items: []);

        // Act & Assert
        await Assert.ThrowsAsync<ValidationException>(
            () => _sut.CreateOrderAsync(dto, CancellationToken.None));
    }
}
```

### B. Integration Tests

```csharp
public class OrdersControllerTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;
    private readonly WebApplicationFactory<Program> _factory;

    public OrdersControllerTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                // Replace real database with in-memory
                services.RemoveAll<DbContextOptions<AppDbContext>>();
                services.AddDbContext<AppDbContext>(options =>
                    options.UseInMemoryDatabase("TestDb"));
            });
        });
        _client = _factory.CreateClient();
    }

    [Fact]
    public async Task GetOrders_ReturnsSuccessAndCorrectContentType()
    {
        // Act
        var response = await _client.GetAsync("/api/orders");

        // Assert
        response.EnsureSuccessStatusCode();
        Assert.Equal("application/json; charset=utf-8",
            response.Content.Headers.ContentType?.ToString());
    }

    [Fact]
    public async Task CreateOrder_WithValidData_ReturnsCreated()
    {
        // Arrange
        var order = new { CustomerId = 1, Items = new[] { new { ProductId = 1, Quantity = 2 } } };

        // Act
        var response = await _client.PostAsJsonAsync("/api/orders", order);

        // Assert
        Assert.Equal(HttpStatusCode.Created, response.StatusCode);
    }
}
```

---

## 11. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

**Use NuGet with Central Package Management (CPM) via `Directory.Packages.props`:**

```xml
<!-- Directory.Packages.props -->
<Project>
  <PropertyGroup>
    <ManagePackageVersionsCentrally>true</ManagePackageVersionsCentrally>
    <CentralPackageTransitivePinningEnabled>true</CentralPackageTransitivePinningEnabled>
  </PropertyGroup>
  <ItemGroup>
    <PackageVersion Include="Microsoft.Extensions.Logging" Version="9.0.0" />
    <PackageVersion Include="Newtonsoft.Json" Version="13.0.3" />
  </ItemGroup>
</Project>
```

- **Lockfiles**: Enable NuGet lock files (`packages.lock.json`) for reproducible builds in CI.
- **Vulnerability Checks**: `dotnet list package --vulnerable` MUST be part of the CI pipeline.

### B. Vulnerability Scanning & Security

**Mandatory security checks for ALL C# projects:**

1. **Vulnerability Scan**:
   ```bash
   # Scan for known vulnerabilities in dependencies
   dotnet list package --vulnerable --include-transitive
   ```
   - Agents MUST fix all discoverable high/critical vulnerabilities before presentation.

2. **Supply Chain Audit**:
   - Verify NuGet signatures for external packages.
   - Use `dotnet restore --locked-mode` in CI to ensure no tampered dependencies.

### C. Dependency File

```xml
<!-- Directory.Packages.props example -->
<Project>
  <ItemGroup>
    <PackageVersion Include="xunit" Version="2.9.2" />
    <PackageVersion Include="Moq" Version="4.20.72" />
  </ItemGroup>
</Project>
```

---

## 12. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code compiles: `dotnet build` returns exit code 0
- [ ] No compilation errors or warnings (TreatWarningsAsErrors=true)
- [ ] All nullable reference types resolved
- [ ] Code formatted: `dotnet format --verify-no-changes` passes

#### Testing
- [ ] All tests pass: `dotnet test` returns exit code 0
- [ ] Reasonable coverage: `dotnet test /p:CollectCoverage=true` shows >80%
- [ ] Integration tests pass (if applicable)

#### Security
- [ ] Dependency scan passes: 0 vulnerabilities found via `dotnet list package --vulnerable`
- [ ] Supply chain verified: NuGet lockfiles in sync
- [ ] Secrets check: No hardcoded secrets in appsettings.json or code
- [ ] Static analysis: Roslyn analyzers report 0 issues

#### Code Quality
- [ ] No unused dependencies
- [ ] Clean namespace hierarchy
- [ ] Project structure follows standard layout

#### Documentation
- [ ] All public APIs have XML documentation comments
- [ ] Documentation follows conventions
- [ ] Examples provided for complex APIs

#### Architecture
- [ ] Repository pattern followed where appropriate
- [ ] Dependency injection used for all services
- [ ] No global mutable state

#### Agent Workflow Completed
- [ ] Agent verified code compiles/builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran formatters and linters
- [ ] Agent verified documentation
- [ ] Agent documented any fixes made during verification

---

## 13. Why This Configuration Works

**Central Package Management**:
- Ensures consistent dependency versions across the entire solution, preventing "dependency hell" and simplifying security updates.

**Nullable Reference Types**:
- Eliminates an entire class of runtime errors (NullReferenceException) by forcing explicit intent and compile-time checks.

**Modern Synchronization (Lock type)**:
- The .NET 9 `System.Threading.Lock` provides a more efficient and structured way to handle thread safety than the traditional `lock(obj)` keyword.

**Performance (params Span)**:
- Using `params ReadOnlySpan<T>` allows high-performance APIs that avoid heap allocations when calling methods with varying numbers of arguments.

---

## 14. Quick Reference

### Common Commands

```bash
# Build
dotnet build

# Test
dotnet test

# Lint & Format
dotnet format

# Security Scan
dotnet list package --vulnerable

# Run
dotnet run --project src/MyApp.Api

# Clean
dotnet clean
```

### Modern C# 13 Patterns Cheat Sheet

```csharp
// New Lock type (.NET 9)
private readonly System.Threading.Lock _gate = new();
public void ThreadSafeMethod()
{
    using (_gate.EnterScope())
    {
        // Thread-safe code here
    }
}

// params ReadOnlySpan (.NET 9)
public void ProcessData(params ReadOnlySpan<int> numbers)
{
    foreach (var n in numbers) { ... }
}

// Raw string literals
var json = """
{
  "name": "John",
  "age": 30
}
""";
```

---

**Last Updated:** 2026-02-06
**Version:** 1.1
**Maintainer:** .NET Team


**End of C# Development Guidelines**

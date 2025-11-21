# C# Development Guidelines

This document provides mandatory standards for C# development, following modern .NET patterns and best practices.

---

**Agent Profile**: The C# Expert
**Role**: Senior .NET Developer & Software Architect
**Objective**: Generate clean, efficient, and maintainable C# code following Microsoft and community best practices.
**Tools**: .NET 8+, Visual Studio, Rider, Roslyn analyzers, dotnet CLI.

---

## 1. Core Philosophies: DOTNET-FIRST

- **D**efensive: Null safety, validation, and proper exception handling
- **O**bject-Oriented: Proper use of classes, interfaces, and inheritance
- **T**estable: Design for unit testing with dependency injection
- **N**amespaced: Logical organization with proper namespace hierarchy
- **E**fficient: Use async/await, spans, and memory-efficient patterns
- **T**yped: Leverage the strong type system and generics

---

## 2. Project Structure (MANDATORY)

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
        // ...
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

## 12. Deployment Checklist

### Code Quality
- [ ] Nullable reference types enabled
- [ ] No compiler warnings
- [ ] Roslyn analyzers configured
- [ ] Code coverage above threshold

### Performance
- [ ] Async/await used properly
- [ ] No blocking calls on async code
- [ ] EF Core queries optimized
- [ ] Caching implemented where needed

### Security
- [ ] Input validation present
- [ ] No sensitive data in logs
- [ ] Authentication/authorization configured
- [ ] HTTPS enforced

### Testing
- [ ] Unit tests for business logic
- [ ] Integration tests for APIs
- [ ] All tests passing
- [ ] Edge cases covered

---

## 13. Quick Reference

```csharp
// Null handling
ArgumentNullException.ThrowIfNull(param);
var value = nullable ?? default;
var length = str?.Length ?? 0;

// Async patterns
await Task.WhenAll(task1, task2);
await Task.WhenAny(task1, task2);
cancellationToken.ThrowIfCancellationRequested();

// LINQ
.FirstOrDefault()    // May return null
.SingleOrDefault()   // Throws if multiple
.Any()               // Exists check
.All()               // All match

// Collections
ImmutableList<T>.Empty
ImmutableDictionary<K,V>.Empty
new List<T> { capacity: 100 }

// String handling
string.IsNullOrEmpty(s)
string.IsNullOrWhiteSpace(s)
$"Interpolated {value}"
"""
Raw string literal
"""
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** .NET Team

# Swift Development Guidelines
Mandatory standards for Swift development, following Apple's guidelines and community best practices. Xcode, Swift 5.9+, SwiftLint, SwiftFormat, Instruments.

---

**Agent Profile**: The Swift Expert
**Role**: Senior iOS/macOS Developer & Swift Architect
**Objective**: Generate clean, safe, and performant Swift code following Apple's Human Interface Guidelines and Swift API Design Guidelines.
**Tools**: Xcode, Swift 5.9+, SwiftLint, SwiftFormat, Instruments.

---

## 1. Core Philosophies: SWIFT-FIRST

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **S**afe: Leverage optionals, guard statements, and type safety
- **W**ell-named: Clear, expressive API names that read like prose
- **I**mmutable: Prefer `let` over `var`, value types over reference types
- **F**ast: Write performant code with proper memory management
- **T**estable: Design for testability with protocols and dependency injection

---

## 2. Naming Conventions (MANDATORY)

### A. General Rules

```swift
// ✅ Types: UpperCamelCase
struct UserProfile { }
class NetworkManager { }
enum LoadingState { }
protocol DataFetching { }

// ✅ Properties, methods, variables: lowerCamelCase
var userName: String
func fetchUserData() { }
let maximumRetryCount = 3

// ✅ Boolean properties: read as assertions
var isEmpty: Bool
var hasContent: Bool
var canSubmit: Bool
var shouldRefresh: Bool
var isLoading: Bool

// ✅ Methods that perform actions: verb phrases
func removeItem(at index: Int)
func insert(_ item: Item, at index: Int)
func update(with newData: Data)

// ✅ Methods that return values: noun phrases
func distance(to point: Point) -> Double
func makeIterator() -> Iterator
func successor() -> Self

// ✅ Protocols describing capability: -able, -ible, or -ing
protocol Equatable { }
protocol ProgressReporting { }
protocol Cacheable { }

// ✅ Protocols describing what something is: nouns
protocol Collection { }
protocol Sequence { }
```

### B. Argument Labels

```swift
// ✅ CORRECT: Clear argument labels
func move(from start: Point, to end: Point)
func copy(section: Range, to destination: Buffer)
func remove(at index: Int) -> Element

// ✅ CORRECT: Omit first argument when it's clear from function name
func contains(_ element: Element) -> Bool
func append(_ item: Item)
func insert(_ item: Item, at index: Int)

// ✅ CORRECT: Use prepositions for clarity
func moveTo(x: Int, y: Int)
func fadeIn(withDuration duration: TimeInterval)
func compare(with other: Self) -> ComparisonResult

// ❌ WRONG: Redundant type information
func addColor(_ color: Color)     // ❌
func add(_ color: Color)          // ✅

// ❌ WRONG: Unclear argument purpose
func set(_ value: Int, _ flag: Bool)      // ❌
func set(_ value: Int, animated: Bool)    // ✅
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

### Example TDD Workflow for Swift

```swift
// Step 1: RED - Write failing test first
import XCTest
@testable import MyApp

final class EmailValidatorTests: XCTestCase {

    func test_validate_withValidEmail_returnsEmail() throws {
        let result = try EmailValidator.validate("user@example.com")
        XCTAssertEqual(result, "user@example.com")
    }

    func test_validate_withoutAtSymbol_throwsInvalidFormat() {
        XCTAssertThrowsError(try EmailValidator.validate("invalid-email")) { error in
            XCTAssertEqual(error as? EmailValidationError, .invalidFormat)
        }
    }

    func test_validate_withEmptyString_throwsInvalidFormat() {
        XCTAssertThrowsError(try EmailValidator.validate("")) { error in
            XCTAssertEqual(error as? EmailValidationError, .invalidFormat)
        }
    }
}

// Run: swift test --filter EmailValidatorTests
// FAILS - EmailValidator type does not exist

// Step 2: GREEN - Write minimal implementation
enum EmailValidationError: Error, Equatable {
    case invalidFormat
}

struct EmailValidator {
    static func validate(_ email: String) throws -> String {
        guard email.contains("@") else {
            throw EmailValidationError.invalidFormat
        }
        return email
    }
}

// Run: swift test --filter EmailValidatorTests
// PASSES - all tests pass

// Step 3: REFACTOR - Improve with regex validation
struct EmailValidator {
    private static let emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

    static func validate(_ email: String) throws -> String {
        guard (try? emailPattern.wholeMatch(in: email)) != nil else {
            throw EmailValidationError.invalidFormat
        }
        return email.lowercased()
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

```swift
// Bug Report #1042: EmailValidator accepts emails with spaces like "user @example.com"

// Step 1-2: Write test that reproduces the bug
final class EmailValidatorTests: XCTestCase {

    // Regression test for Bug #1042
    func test_validate_withSpacesInEmail_throwsInvalidFormat() {
        XCTAssertThrowsError(try EmailValidator.validate("user @example.com"))
        XCTAssertThrowsError(try EmailValidator.validate(" user@example.com"))
        XCTAssertThrowsError(try EmailValidator.validate("user@example.com "))
    }
}

// Run: swift test --filter EmailValidatorTests
// FAILS - validate does not throw for emails with spaces

// Step 3: Fix the bug
struct EmailValidator {
    private static let emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/

    static func validate(_ email: String) throws -> String {
        guard email == email.trimmingCharacters(in: .whitespaces) else {
            throw EmailValidationError.invalidFormat
        }
        guard (try? emailPattern.wholeMatch(in: email)) != nil else {
            throw EmailValidationError.invalidFormat
        }
        return email.lowercased()
    }
}

// Run: swift test --filter EmailValidatorTests
// PASSES - bug fixed, regression prevented
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Force-unwrap (`!`) optionals in production code to work around test failures

---

## 3. Optionals and Safety (MANDATORY)

### A. Optional Handling

```swift
// ✅ CORRECT: Optional binding
if let user = fetchUser() {
    print(user.name)
}

// ✅ CORRECT: Guard for early exit
func process(data: Data?) {
    guard let data = data else {
        print("No data provided")
        return
    }
    // Use data safely
    processValidData(data)
}

// ✅ CORRECT: Multiple optional binding
if let name = user?.name,
   let email = user?.email,
   !email.isEmpty {
    sendWelcome(to: name, email: email)
}

// ✅ CORRECT: Guard with multiple conditions
guard
    let user = currentUser,
    user.isAuthenticated,
    let token = user.accessToken
else {
    throw AuthenticationError.notAuthenticated
}

// ✅ CORRECT: Optional chaining
let count = user?.posts?.count ?? 0
user?.profile?.updateLastSeen()

// ✅ CORRECT: Nil coalescing
let displayName = user?.name ?? "Anonymous"
let items = fetchItems() ?? []

// ❌ WRONG: Force unwrapping without certainty
let name = user!.name  // Crashes if nil

// ❌ WRONG: Implicit unwrap without guarantee
var delegate: Delegate!  // Only if guaranteed set before use
```

### B. Guard Statements

```swift
// ✅ CORRECT: Guard for preconditions
func processOrder(_ order: Order?) throws {
    guard let order = order else {
        throw OrderError.missingOrder
    }

    guard order.items.count > 0 else {
        throw OrderError.emptyOrder
    }

    guard order.totalAmount > 0 else {
        throw OrderError.invalidAmount
    }

    // Process valid order
    submitOrder(order)
}

// ✅ CORRECT: Guard in loops
for item in items {
    guard item.isValid else { continue }
    guard let price = item.price else { continue }

    total += price
}
```

---

## 4. Value Types vs Reference Types (MANDATORY)

### A. Prefer Structs

```swift
// ✅ CORRECT: Use struct for data models
struct User: Codable, Equatable {
    let id: UUID
    var name: String
    var email: String
    var createdAt: Date
}

struct Point: Equatable {
    var x: Double
    var y: Double

    func distance(to other: Point) -> Double {
        let dx = x - other.x
        let dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

// ✅ CORRECT: Use struct for view models
struct UserViewModel {
    let displayName: String
    let avatarURL: URL?
    let memberSince: String

    init(user: User) {
        self.displayName = user.name.isEmpty ? "Anonymous" : user.name
        self.avatarURL = URL(string: user.avatarURLString ?? "")
        self.memberSince = Self.dateFormatter.string(from: user.createdAt)
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return formatter
    }()
}
```

### B. Use Classes When Needed

```swift
// ✅ CORRECT: Use class for identity, inheritance, or Objective-C interop
class NetworkManager {
    static let shared = NetworkManager()

    private let session: URLSession
    private var activeTasks: [UUID: URLSessionTask] = [:]

    private init() {
        self.session = URLSession(configuration: .default)
    }

    func cancelAllTasks() {
        activeTasks.values.forEach { $0.cancel() }
        activeTasks.removeAll()
    }
}

// ✅ CORRECT: Use class for delegation and callbacks
class DataFetcher: NSObject, URLSessionDelegate {
    weak var delegate: DataFetcherDelegate?

    // ..
}
```

---

## 5. Protocols and Extensions (MANDATORY)

### A. Protocol-Oriented Design

```swift
// Define focused protocols
protocol Identifiable {
    associatedtype ID: Hashable
    var id: ID { get }
}

protocol Timestamped {
    var createdAt: Date { get }
    var updatedAt: Date { get }
}

protocol Persistable: Identifiable, Codable {
    static var storageKey: String { get }
}

// Default implementations via extensions
extension Persistable {
    static var storageKey: String {
        String(describing: Self.self)
    }

    func save() throws {
        let data = try JSONEncoder().encode(self)
        UserDefaults.standard.set(data, forKey: Self.storageKey + "-\(id)")
    }

    static func load(id: ID) throws -> Self? {
        guard let data = UserDefaults.standard.data(forKey: storageKey + "-\(id)") else {
            return nil
        }
        return try JSONDecoder().decode(Self.self, from: data)
    }
}

// Conform to protocols
struct Document: Persistable {
    let id: UUID
    var title: String
    var content: String
}

// Use protocol as constraint
func process<T: Persistable>(_ item: T) throws {
    try item.save()
}
```

### B. Extensions for Organization

```swift
// MARK: - Core Model
struct User {
    let id: UUID
    var name: String
    var email: String
}

// MARK: - Codable
extension User: Codable {
    enum CodingKeys: String, CodingKey {
        case id
        case name
        case email
    }
}

// MARK: - Equatable & Hashable
extension User: Equatable, Hashable {
    static func == (lhs: User, rhs: User) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

// MARK: - Display
extension User {
    var displayName: String {
        name.isEmpty ? "Anonymous" : name
    }

    var initials: String {
        name.split(separator: " ")
            .compactMap { $0.first }
            .map(String.init)
            .joined()
    }
}
```

---

## 6. Error Handling (MANDATORY)

### A. Custom Errors

```swift
// Define domain-specific errors
enum NetworkError: LocalizedError {
    case noConnection
    case timeout
    case invalidResponse(statusCode: Int)
    case decodingFailed(underlying: Error)
    case unauthorized

    var errorDescription: String? {
        switch self {
        case .noConnection:
            return "No internet connection"
        case .timeout:
            return "Request timed out"
        case .invalidResponse(let code):
            return "Server returned error: \(code)"
        case .decodingFailed:
            return "Failed to process server response"
        case .unauthorized:
            return "Please log in again"
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .noConnection:
            return "Please check your internet connection and try again"
        case .timeout:
            return "Please try again later"
        case .unauthorized:
            return "Your session has expired"
        default:
            return nil
        }
    }
}

enum ValidationError: Error {
    case emptyField(fieldName: String)
    case invalidFormat(fieldName: String, expected: String)
    case tooShort(fieldName: String, minimum: Int)
    case tooLong(fieldName: String, maximum: Int)
}
```

### B. Throwing Functions

```swift
// ✅ CORRECT: Throwing functions for recoverable errors
func fetchUser(id: UUID) async throws -> User {
    guard let url = URL(string: "\(baseURL)/users/\(id)") else {
        throw NetworkError.invalidResponse(statusCode: 0)
    }

    let (data, response) = try await session.data(from: url)

    guard let httpResponse = response as? HTTPURLResponse else {
        throw NetworkError.invalidResponse(statusCode: 0)
    }

    switch httpResponse.statusCode {
    case 200...299:
        do {
            return try decoder.decode(User.self, from: data)
        } catch {
            throw NetworkError.decodingFailed(underlying: error)
        }
    case 401:
        throw NetworkError.unauthorized
    default:
        throw NetworkError.invalidResponse(statusCode: httpResponse.statusCode)
    }
}

// ✅ CORRECT: Handle errors at call site
func loadUserProfile() async {
    do {
        let user = try await fetchUser(id: currentUserId)
        updateUI(with: user)
    } catch NetworkError.unauthorized {
        showLoginScreen()
    } catch NetworkError.noConnection {
        showOfflineMessage()
    } catch {
        showError(error.localizedDescription)
    }
}
```

### C. Result Type

```swift
// Use Result for async callbacks
func fetchData(completion: @escaping (Result<Data, NetworkError>) -> Void) {
    // ..
    completion(.success(data))
    // or
    completion(.failure(.noConnection))
}

// Handle Result
fetchData { result in
    switch result {
    case .success(let data):
        process(data)
    case .failure(let error):
        handle(error)
    }
}

// Map and flatMap on Result
let parsedResult = result.map { data in
    try? JSONDecoder().decode(User.self, from: data)
}
```

---

## 7. Concurrency (MANDATORY)

### A. Async/Await

```swift
// ✅ CORRECT: Async function
func fetchUserProfile(userId: UUID) async throws -> UserProfile {
    async let user = fetchUser(id: userId)
    async let posts = fetchPosts(userId: userId)
    async let followers = fetchFollowers(userId: userId)

    // Await all in parallel
    return try await UserProfile(
        user: user,
        posts: posts,
        followers: followers
    )
}

// ✅ CORRECT: Actor for thread-safe state
actor ImageCache {
    private var cache: [URL: UIImage] = [:]

    func image(for url: URL) -> UIImage? {
        cache[url]
    }

    func store(_ image: UIImage, for url: URL) {
        cache[url] = image
    }

    func clear() {
        cache.removeAll()
    }
}

// ✅ CORRECT: MainActor for UI updates
@MainActor
class ViewModel: ObservableObject {
    @Published var users: [User] = []
    @Published var isLoading = false
    @Published var error: Error?

    func loadUsers() async {
        isLoading = true
        defer { isLoading = false }

        do {
            users = try await userService.fetchAllUsers()
        } catch {
            self.error = error
        }
    }
}

// ✅ CORRECT: Task groups for concurrent operations
func processImages(_ urls: [URL]) async throws -> [UIImage] {
    try await withThrowingTaskGroup(of: UIImage.self) { group in
        for url in urls {
            group.addTask {
                try await downloadImage(from: url)
            }
        }

        var images: [UIImage] = []
        for try await image in group {
            images.append(image)
        }
        return images
    }
}
```

### B. Task Cancellation

```swift
// ✅ CORRECT: Check for cancellation
func processLargeDataset(_ items: [Item]) async throws {
    for item in items {
        // Check if task was cancelled
        try Task.checkCancellation()

        await process(item)
    }
}

// ✅ CORRECT: Handle cancellation gracefully
func searchAsUserTypes(_ query: String) async -> [SearchResult] {
    // Artificial delay to avoid too many requests
    try? await Task.sleep(nanoseconds: 300_000_000)

    // Check if cancelled during delay
    guard !Task.isCancelled else {
        return []
    }

    return await performSearch(query)
}
```

---

## 8. SwiftUI Best Practices (MANDATORY)

### A. View Composition

```swift
// ✅ CORRECT: Small, focused views
struct UserRow: View {
    let user: User

    var body: some View {
        HStack(spacing: 12) {
            AvatarView(url: user.avatarURL)

            VStack(alignment: .leading, spacing: 4) {
                Text(user.name)
                    .font(.headline)
                Text(user.email)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if user.isVerified {
                Image(systemName: "checkmark.seal.fill")
                    .foregroundStyle(.blue)
            }
        }
        .padding(.vertical, 8)
    }
}

// ✅ CORRECT: Extract subviews for readability
struct UserProfileView: View {
    @StateObject private var viewModel: UserProfileViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                headerSection
                statsSection
                postsSection
            }
            .padding()
        }
        .navigationTitle("Profile")
        .task { await viewModel.load() }
    }

    private var headerSection: some View {
        VStack(spacing: 12) {
            AsyncImage(url: viewModel.avatarURL) { image in
                image.resizable().scaledToFill()
            } placeholder: {
                ProgressView()
            }
            .frame(width: 100, height: 100)
            .clipShape(Circle())

            Text(viewModel.displayName)
                .font(.title2.bold())
        }
    }

    private var statsSection: some View {
        HStack(spacing: 40) {
            StatView(title: "Posts", value: viewModel.postCount)
            StatView(title: "Followers", value: viewModel.followerCount)
            StatView(title: "Following", value: viewModel.followingCount)
        }
    }

    private var postsSection: some View {
        LazyVStack(spacing: 16) {
            ForEach(viewModel.posts) { post in
                PostCard(post: post)
            }
        }
    }
}
```

### B. State Management

```swift
// ✅ CORRECT: Use appropriate property wrappers
struct SettingsView: View {
    // Local view state
    @State private var notificationsEnabled = true

    // Binding from parent
    @Binding var theme: Theme

    // Observable object owned by this view
    @StateObject private var viewModel = SettingsViewModel()

    // Observable object from environment
    @EnvironmentObject private var userSession: UserSession

    // Environment value
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Toggle("Notifications", isOn: $notificationsEnabled)

            Picker("Theme", selection: $theme) {
                ForEach(Theme.allCases) { theme in
                    Text(theme.displayName).tag(theme)
                }
            }
        }
    }
}

// ✅ CORRECT: ViewModel with @Published
@MainActor
class SettingsViewModel: ObservableObject {
    @Published var settings: Settings?
    @Published var isLoading = false
    @Published var errorMessage: String?

    private let settingsService: SettingsService

    init(settingsService: SettingsService = .shared) {
        self.settingsService = settingsService
    }

    func load() async {
        isLoading = true
        defer { isLoading = false }

        do {
            settings = try await settingsService.fetch()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
```

---

## 9. Memory Management (MANDATORY)

### A. Capture Lists

```swift
// ✅ CORRECT: Weak capture to avoid retain cycles
class DataLoader {
    var onComplete: ((Data) -> Void)?

    func load() {
        networkManager.fetch { [weak self] data in
            guard let self = self else { return }
            self.process(data)
            self.onComplete?(data)
        }
    }
}

// ✅ CORRECT: Unowned when guaranteed to exist
class Parent {
    var child: Child?

    init() {
        child = Child(parent: self)
    }
}

class Child {
    unowned let parent: Parent

    init(parent: Parent) {
        self.parent = parent
    }
}

// ✅ CORRECT: Capture specific values
func processUser(user: User) {
    let userId = user.id  // Capture just the ID, not whole user

    DispatchQueue.global().async {
        self.sendAnalytics(userId: userId)
    }
}
```

### B. Avoiding Retain Cycles

```swift
// ❌ WRONG: Retain cycle with closure
class ViewController: UIViewController {
    var timer: Timer?

    func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.updateUI()  // Strong reference to self!
        }
    }
}

// ✅ CORRECT: Break the cycle
class ViewController: UIViewController {
    var timer: Timer?

    func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateUI()
        }
    }

    deinit {
        timer?.invalidate()
    }
}
```

---

## 10. Testing (MANDATORY)

### A. Unit Tests

```swift
import XCTest
@testable import MyApp

final class UserServiceTests: XCTestCase {
    var sut: UserService!
    var mockNetworkClient: MockNetworkClient!

    override func setUp() {
        super.setUp()
        mockNetworkClient = MockNetworkClient()
        sut = UserService(networkClient: mockNetworkClient)
    }

    override func tearDown() {
        sut = nil
        mockNetworkClient = nil
        super.tearDown()
    }

    func test_fetchUser_withValidId_returnsUser() async throws {
        // Given
        let expectedUser = User(id: UUID(), name: "Test", email: "test@example.com")
        mockNetworkClient.mockResponse = expectedUser

        // When
        let result = try await sut.fetchUser(id: expectedUser.id)

        // Then
        XCTAssertEqual(result.id, expectedUser.id)
        XCTAssertEqual(result.name, expectedUser.name)
    }

    func test_fetchUser_withNetworkError_throwsError() async {
        // Given
        mockNetworkClient.mockError = NetworkError.noConnection

        // When/Then
        do {
            _ = try await sut.fetchUser(id: UUID())
            XCTFail("Expected error to be thrown")
        } catch {
            XCTAssertTrue(error is NetworkError)
        }
    }
}
```

### B. Mocking with Protocols

```swift
// Protocol for dependency
protocol NetworkClientProtocol {
    func fetch<T: Decodable>(_ type: T.Type, from url: URL) async throws -> T
}

// Production implementation
class NetworkClient: NetworkClientProtocol {
    func fetch<T: Decodable>(_ type: T.Type, from url: URL) async throws -> T {
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(T.self, from: data)
    }
}

// Mock for testing
class MockNetworkClient: NetworkClientProtocol {
    var mockResponse: Any?
    var mockError: Error?
    var fetchCallCount = 0
    var lastRequestedURL: URL?

    func fetch<T: Decodable>(_ type: T.Type, from url: URL) async throws -> T {
        fetchCallCount += 1
        lastRequestedURL = url

        if let error = mockError {
            throw error
        }

        guard let response = mockResponse as? T else {
            fatalError("Mock response not set or wrong type")
        }

        return response
    }
}
```

---

## 11. Deployment Checklist

### Code Quality
- [ ] No force unwraps without safety comments
- [ ] No compiler warnings
- [ ] SwiftLint passes
- [ ] All tests passing

### Memory
- [ ] No retain cycles (use weak/unowned appropriately)
- [ ] Large objects released when done
- [ ] Instruments shows no memory leaks

### Performance
- [ ] Avoid work on main thread
- [ ] Use lazy loading where appropriate
- [ ] Profile with Instruments

### App Store
- [ ] Proper error handling
- [ ] Accessibility labels present
- [ ] Localization ready
- [ ] Privacy permissions explained

---

## 12. Quick Reference

```swift
// Optional handling
guard let value = optional else { return }
let unwrapped = optional ?? defaultValue
optional?.method()
if let value = optional { }

// Async/Await
async let result = asyncFunction()
try await result
Task { await doWork() }
Task.detached { await doWork() }

// Property wrappers (SwiftUI)
@State           // Local view state
@Binding         // Two-way binding
@StateObject     // Own observable object
@ObservedObject  // Don't own observable object
@EnvironmentObject // From environment
@Environment     // Environment value

// Access control
public    // Visible everywhere
internal  // Default, visible in module
fileprivate // Visible in file
private   // Visible in enclosing scope
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** iOS Team


**End of Swift Development Guidelines**

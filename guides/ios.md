# iOS Development Guidelines
Mandatory standards for building native iOS applications using Swift and SwiftUI. Xcode 15+, Swift 5.9+, SwiftUI, Swift Concurrency, SwiftData, SPM.

---

**Agent Profile**: The iOS Expert
**Role**: Senior iOS Developer & Apple Platform Architect
**Objective**: Generate polished, performant iOS applications following Apple's Human Interface Guidelines and Swift best practices.
**Tools**: Xcode 15+, Swift 5.9+, SwiftUI, Swift Concurrency, SwiftData, SPM.

---

## 1. Core Philosophies: IOS-FIRST

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **I**ntuitive: Follow Apple's Human Interface Guidelines
- **O**ptimized: Leverage native APIs for best performance
- **S**wifty: Use modern Swift patterns and conventions

---

## 2. Project Structure (MANDATORY)

### A. Directory Layout

```
MyApp/
├── MyApp/
│   ├── App/
│   │   ├── MyAppApp.swift            # App entry point
│   │   ├── AppDelegate.swift         # If needed for UIKit lifecycle
│   │   └── SceneDelegate.swift       # If using scenes
│   ├── Features/                      # Feature modules
│   │   ├── Home/
│   │   │   ├── Views/
│   │   │   │   ├── HomeView.swift
│   │   │   │   └── Components/
│   │   │   ├── ViewModels/
│   │   │   │   └── HomeViewModel.swift
│   │   │   └── Models/
│   │   ├── Detail/
│   │   └── Settings/
│   ├── Core/                          # Shared infrastructure
│   │   ├── Network/
│   │   │   ├── APIClient.swift
│   │   │   ├── Endpoints.swift
│   │   │   └── NetworkError.swift
│   │   ├── Persistence/
│   │   │   ├── ModelContainer.swift
│   │   │   └── Models/
│   │   ├── Services/
│   │   └── Utilities/
│   ├── UI/                            # Shared UI components
│   │   ├── Components/
│   │   │   ├── Buttons/
│   │   │   ├── Cards/
│   │   │   └── Forms/
│   │   ├── Modifiers/
│   │   └── Theme/
│   │       ├── Colors.swift
│   │       ├── Typography.swift
│   │       └── Spacing.swift
│   ├── Resources/
│   │   ├── Assets.xcassets
│   │   └── Localizable.xcstrings
│   └── Info.plist
├── MyAppTests/
├── MyAppUITests/
└── Package.swift                      # For SPM dependencies
```

---

## 2A. TDD Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### Red-Green-Refactor Cycle with XCTest

```swift
// ═══════════════════════════════════════════════════════════════
// STEP 1: RED - Write failing test first
// ═══════════════════════════════════════════════════════════════

// MyAppTests/ViewModels/HomeViewModelTests.swift
import XCTest
@testable import MyApp

@MainActor
final class HomeViewModelTDDTests: XCTestCase {
    var sut: HomeViewModel!
    var mockService: MockItemService!

    override func setUp() {
        super.setUp()
        mockService = MockItemService()
        sut = HomeViewModel(itemService: mockService)
    }

    override func tearDown() {
        sut = nil
        mockService = nil
        super.tearDown()
    }

    func test_loadInitialData_success_setsLoadedState() async {
        // Given
        let items = [Item(title: "Test", description: "Description")]
        mockService.itemsToReturn = items

        // When
        await sut.loadInitialData()

        // Then
        if case .loaded(let loadedItems) = sut.state {
            XCTAssertEqual(loadedItems.count, 1)
            XCTAssertEqual(loadedItems.first?.title, "Test")
        } else {
            XCTFail("Expected loaded state, got \(sut.state)")
        }
    }

    func test_loadInitialData_empty_setsEmptyState() async {
        // Given
        mockService.itemsToReturn = []

        // When
        await sut.loadInitialData()

        // Then
        if case .empty = sut.state {
            // Success
        } else {
            XCTFail("Expected empty state")
        }
    }
}

// Run: Cmd+U or xcodebuild test
// ❌ FAILS - HomeViewModel doesn't exist yet

// ═══════════════════════════════════════════════════════════════
// STEP 2: GREEN - Write minimal implementation
// ═══════════════════════════════════════════════════════════════

// Implement HomeViewModel with loadInitialData() to make tests pass

// Run: Cmd+U
// ✅ PASSES - all tests pass

// ═══════════════════════════════════════════════════════════════
// STEP 3: REFACTOR - Add analytics, improve while tests stay green
// ═══════════════════════════════════════════════════════════════
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow Example

```swift
// ═══════════════════════════════════════════════════════════════
// Bug Report #112: HomeViewModel never exits loading state when
// network request fails with timeout error
// ═══════════════════════════════════════════════════════════════

// STEP 1: Write test that reproduces the bug
// MyAppTests/ViewModels/HomeViewModelTests.swift

func test_loadInitialData_timeout_setsErrorState_Bug112() async {
    // Bug: ViewModel stays in .loading state on timeout
    // Discovered: 2026-03-18
    // Root cause: catch block didn't update state on URLError

    mockService.errorToThrow = URLError(.timedOut)

    await sut.loadInitialData()

    if case .error(let error) = sut.state {
        XCTAssertTrue(error is URLError)
    } else {
        XCTFail("Bug #112: Expected error state after timeout, got \(sut.state)")
    }
}

// Run: Cmd+U
// ❌ FAILS - state is still .loading after timeout

// STEP 2: Fix the bug - Update catch block to handle URLError

// Run: Cmd+U
// ✅ PASSES - bug fixed, regression prevented forever
```

---

## 3. SwiftUI Views (MANDATORY)

### A. View Structure

```swift
// Features/Home/Views/HomeView.swift
import SwiftUI

struct HomeView: View {
    @StateObject private var viewModel = HomeViewModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Home")
                .toolbar { toolbarContent }
                .refreshable { await viewModel.refresh() }
                .task { await viewModel.loadInitialData() }
        }
    }

    @ViewBuilder
    private var content: some View {
        switch viewModel.state {
        case .loading:
            ProgressView()
        case .loaded(let items):
            itemsList(items)
        case .empty:
            emptyState
        case .error(let error):
            errorState(error)
        }
    }

    private func itemsList(_ items: [Item]) -> some View {
        List(items) { item in
            NavigationLink(value: item) {
                ItemRow(item: item)
            }
        }
        .listStyle(.plain)
        .navigationDestination(for: Item.self) { item in
            DetailView(item: item)
        }
    }

    private var emptyState: some View {
        ContentUnavailableView(
            "No Items",
            systemImage: "tray",
            description: Text("Add your first item to get started.")
        )
    }

    private func errorState(_ error: Error) -> some View {
        ContentUnavailableView {
            Label("Error", systemImage: "exclamationmark.triangle")
        } description: {
            Text(error.localizedDescription)
        } actions: {
            Button("Retry") {
                Task { await viewModel.refresh() }
            }
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button {
                viewModel.showAddSheet = true
            } label: {
                Image(systemName: "plus")
            }
        }
    }
}

#Preview {
    HomeView()
}
```

### B. Reusable Components

```swift
// UI/Components/Buttons/PrimaryButton.swift
import SwiftUI

struct PrimaryButton: View {
    let title: String
    let action: () -> Void
    var isLoading: Bool = false
    var isDisabled: Bool = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .tint(.white)
                }
                Text(title)
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(isDisabled ? Color.gray : Color.accentColor)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .disabled(isDisabled || isLoading)
    }
}

// UI/Components/Cards/ItemCard.swift
struct ItemCard: View {
    let item: Item
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 8) {
                AsyncImage(url: item.imageURL) { phase in
                    switch phase {
                    case .success(let image):
                        image
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    case .failure:
                        Image(systemName: "photo")
                            .foregroundStyle(.secondary)
                    case .empty:
                        ProgressView()
                    @unknown default:
                        EmptyView()
                    }
                }
                .frame(height: 150)
                .clipped()

                VStack(alignment: .leading, spacing: 4) {
                    Text(item.title)
                        .font(.headline)
                        .foregroundStyle(.primary)

                    Text(item.description)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 12)
            }
            .background(Color(.systemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .shadow(color: .black.opacity(0.1), radius: 8, y: 4)
        }
        .buttonStyle(.plain)
    }
}
```

---

## 4. ViewModel and State (MANDATORY)

### A. Observable ViewModel

```swift
// Features/Home/ViewModels/HomeViewModel.swift
import SwiftUI
import Observation

@Observable
final class HomeViewModel {
    // MARK: - State
    enum State {
        case loading
        case loaded([Item])
        case empty
        case error(Error)
    }

    private(set) var state: State = .loading
    var showAddSheet = false
    var selectedItem: Item?

    // MARK: - Dependencies
    private let itemService: ItemServiceProtocol
    private let analytics: AnalyticsProtocol

    // MARK: - Init
    init(
        itemService: ItemServiceProtocol = ItemService.shared,
        analytics: AnalyticsProtocol = Analytics.shared
    ) {
        self.itemService = itemService
        self.analytics = analytics
    }

    // MARK: - Actions
    @MainActor
    func loadInitialData() async {
        guard case .loading = state else { return }
        await fetchItems()
    }

    @MainActor
    func refresh() async {
        await fetchItems()
    }

    @MainActor
    private func fetchItems() async {
        do {
            let items = try await itemService.fetchItems()
            state = items.isEmpty ? .empty : .loaded(items)
            analytics.track(.itemsLoaded(count: items.count))
        } catch {
            state = .error(error)
            analytics.track(.error(error))
        }
    }

    @MainActor
    func deleteItem(_ item: Item) async {
        do {
            try await itemService.deleteItem(item)
            if case .loaded(var items) = state {
                items.removeAll { $0.id == item.id }
                state = items.isEmpty ? .empty : .loaded(items)
            }
        } catch {
            // Handle error
        }
    }
}
```

### B. Legacy ObservableObject (Pre-iOS 17)

```swift
// For iOS 16 and earlier
import SwiftUI
import Combine

final class HomeViewModel: ObservableObject {
    @Published private(set) var state: State = .loading
    @Published var showAddSheet = false

    private var cancellables = Set<AnyCancellable>()
    private let itemService: ItemServiceProtocol

    init(itemService: ItemServiceProtocol = ItemService.shared) {
        self.itemService = itemService
    }

    @MainActor
    func loadInitialData() async {
        // Same implementation
    }
}
```

---

## 5. Networking (MANDATORY)

### A. API Client

```swift
// Core/Network/APIClient.swift
import Foundation

actor APIClient {
    static let shared = APIClient()

    private let session: URLSession
    private let decoder: JSONDecoder
    private let baseURL: URL

    init(
        session: URLSession = .shared,
        baseURL: URL = URL(string: "https://api.example.com")!
    ) {
        self.session = session
        self.baseURL = baseURL

        self.decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        decoder.keyDecodingStrategy = .convertFromSnakeCase
    }

    func request<T: Decodable>(
        _ endpoint: Endpoint,
        type: T.Type = T.self
    ) async throws -> T {
        let request = try buildRequest(for: endpoint)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }

        switch httpResponse.statusCode {
        case 200...299:
            return try decoder.decode(T.self, from: data)
        case 401:
            throw NetworkError.unauthorized
        case 404:
            throw NetworkError.notFound
        case 400...499:
            throw NetworkError.clientError(httpResponse.statusCode)
        case 500...599:
            throw NetworkError.serverError(httpResponse.statusCode)
        default:
            throw NetworkError.unknown(httpResponse.statusCode)
        }
    }

    private func buildRequest(for endpoint: Endpoint) throws -> URLRequest {
        var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: true)!
        components.path = endpoint.path
        components.queryItems = endpoint.queryItems

        guard let url = components.url else {
            throw NetworkError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = endpoint.method.rawValue
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let token = try? KeychainManager.shared.getToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        if let body = endpoint.body {
            request.httpBody = try JSONEncoder().encode(body)
        }

        return request
    }
}

// Core/Network/Endpoint.swift
struct Endpoint {
    let path: String
    let method: HTTPMethod
    let queryItems: [URLQueryItem]?
    let body: Encodable?

    init(
        path: String,
        method: HTTPMethod = .get,
        queryItems: [URLQueryItem]? = nil,
        body: Encodable? = nil
    ) {
        self.path = path
        self.method = method
        self.queryItems = queryItems
        self.body = body
    }
}

enum HTTPMethod: String {
    case get = "GET"
    case post = "POST"
    case put = "PUT"
    case patch = "PATCH"
    case delete = "DELETE"
}

// Usage
extension Endpoint {
    static func items(page: Int = 1) -> Endpoint {
        Endpoint(
            path: "/v1/items",
            queryItems: [URLQueryItem(name: "page", value: "\(page)")]
        )
    }

    static func item(id: String) -> Endpoint {
        Endpoint(path: "/v1/items/\(id)")
    }

    static func createItem(_ item: CreateItemRequest) -> Endpoint {
        Endpoint(path: "/v1/items", method: .post, body: item)
    }
}
```

---

## 6. Data Persistence (MANDATORY)

### A. SwiftData

```swift
// Core/Persistence/Models/Item.swift
import SwiftData

@Model
final class Item {
    @Attribute(.unique) var id: UUID
    var title: String
    var itemDescription: String
    var imageURL: URL?
    var createdAt: Date
    var updatedAt: Date

    @Relationship(deleteRule: .cascade, inverse: \Tag.items)
    var tags: [Tag]

    init(
        id: UUID = UUID(),
        title: String,
        description: String,
        imageURL: URL? = nil,
        createdAt: Date = .now,
        updatedAt: Date = .now
    ) {
        self.id = id
        self.title = title
        self.itemDescription = description
        self.imageURL = imageURL
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.tags = []
    }
}

// App/MyAppApp.swift
import SwiftUI
import SwiftData

@main
struct MyAppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: [Item.self, Tag.self])
    }
}

// Using in View
struct ItemListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Item.createdAt, order: .reverse) private var items: [Item]

    var body: some View {
        List(items) { item in
            ItemRow(item: item)
        }
    }

    private func addItem() {
        let item = Item(title: "New Item", description: "Description")
        modelContext.insert(item)
    }

    private func deleteItems(at offsets: IndexSet) {
        for index in offsets {
            modelContext.delete(items[index])
        }
    }
}
```

### B. Keychain

```swift
// Core/Services/KeychainManager.swift
import Foundation
import Security

actor KeychainManager {
    static let shared = KeychainManager()

    private let service = Bundle.main.bundleIdentifier ?? "com.example.app"

    func save(token: String, for key: String) throws {
        let data = Data(token.utf8)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data
        ]

        // Delete existing
        SecItemDelete(query as CFDictionary)

        // Add new
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.saveFailed(status)
        }
    }

    func getToken(for key: String) throws -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let token = String(data: data, encoding: .utf8) else {
            throw KeychainError.itemNotFound
        }

        return token
    }

    func delete(key: String) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.deleteFailed(status)
        }
    }
}

enum KeychainError: Error {
    case saveFailed(OSStatus)
    case deleteFailed(OSStatus)
    case itemNotFound
}
```

---

## 7. Navigation (MANDATORY)

### A. NavigationStack

```swift
// Features/Navigation/AppNavigation.swift
import SwiftUI

struct AppNavigation: View {
    @State private var path = NavigationPath()

    var body: some View {
        NavigationStack(path: $path) {
            HomeView()
                .navigationDestination(for: Item.self) { item in
                    DetailView(item: item)
                }
                .navigationDestination(for: Route.self) { route in
                    destinationView(for: route)
                }
        }
        .environment(\.navigate, NavigateAction { route in
            path.append(route)
        })
    }

    @ViewBuilder
    private func destinationView(for route: Route) -> some View {
        switch route {
        case .detail(let item):
            DetailView(item: item)
        case .settings:
            SettingsView()
        case .profile:
            ProfileView()
        }
    }
}

enum Route: Hashable {
    case detail(Item)
    case settings
    case profile
}

// Environment-based navigation
struct NavigateAction {
    let action: (Route) -> Void
    func callAsFunction(_ route: Route) {
        action(route)
    }
}

struct NavigateEnvironmentKey: EnvironmentKey {
    static let defaultValue = NavigateAction { _ in }
}

extension EnvironmentValues {
    var navigate: NavigateAction {
        get { self[NavigateEnvironmentKey.self] }
        set { self[NavigateEnvironmentKey.self] = newValue }
    }
}
```

---

## 8. Testing (MANDATORY)

### A. Unit Tests

```swift
// MyAppTests/ViewModels/HomeViewModelTests.swift
import XCTest
@testable import MyApp

@MainActor
final class HomeViewModelTests: XCTestCase {
    var sut: HomeViewModel!
    var mockService: MockItemService!

    override func setUp() {
        super.setUp()
        mockService = MockItemService()
        sut = HomeViewModel(itemService: mockService)
    }

    override func tearDown() {
        sut = nil
        mockService = nil
        super.tearDown()
    }

    func test_loadInitialData_success_setsLoadedState() async {
        // Given
        let items = [Item(title: "Test", description: "Test")]
        mockService.itemsToReturn = items

        // When
        await sut.loadInitialData()

        // Then
        if case .loaded(let loadedItems) = sut.state {
            XCTAssertEqual(loadedItems.count, 1)
            XCTAssertEqual(loadedItems.first?.title, "Test")
        } else {
            XCTFail("Expected loaded state")
        }
    }

    func test_loadInitialData_empty_setsEmptyState() async {
        // Given
        mockService.itemsToReturn = []

        // When
        await sut.loadInitialData()

        // Then
        if case .empty = sut.state {
            // Success
        } else {
            XCTFail("Expected empty state")
        }
    }

    func test_loadInitialData_failure_setsErrorState() async {
        // Given
        mockService.errorToThrow = NetworkError.serverError(500)

        // When
        await sut.loadInitialData()

        // Then
        if case .error = sut.state {
            // Success
        } else {
            XCTFail("Expected error state")
        }
    }
}

// Mock
final class MockItemService: ItemServiceProtocol {
    var itemsToReturn: [Item] = []
    var errorToThrow: Error?

    func fetchItems() async throws -> [Item] {
        if let error = errorToThrow {
            throw error
        }
        return itemsToReturn
    }

    func deleteItem(_ item: Item) async throws {
        if let error = errorToThrow {
            throw error
        }
    }
}
```

### B. UI Tests

```swift
// MyAppUITests/HomeViewUITests.swift
import XCTest

final class HomeViewUITests: XCTestCase {
    var app: XCUIApplication!

    override func setUp() {
        super.setUp()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchArguments = ["UI_TESTING"]
        app.launch()
    }

    func test_homeView_displaysItems() {
        let firstItem = app.cells.firstMatch
        XCTAssertTrue(firstItem.waitForExistence(timeout: 5))
    }

    func test_addButton_opensSheet() {
        let addButton = app.buttons["Add"]
        XCTAssertTrue(addButton.exists)

        addButton.tap()

        let sheet = app.sheets.firstMatch
        XCTAssertTrue(sheet.waitForExistence(timeout: 2))
    }

    func test_pullToRefresh_refreshesList() {
        let firstCell = app.cells.firstMatch
        XCTAssertTrue(firstCell.waitForExistence(timeout: 5))

        // Pull to refresh
        let start = firstCell.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        let finish = firstCell.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 5))
        start.press(forDuration: 0, thenDragTo: finish)

        // Verify refresh indicator appeared
        let refreshIndicator = app.activityIndicators.firstMatch
        XCTAssertTrue(refreshIndicator.exists)
    }
}
```

---

## 9. Performance (MANDATORY)

### A. Image Loading

```swift
// UI/Components/AsyncImageView.swift
import SwiftUI

struct CachedAsyncImage: View {
    let url: URL?
    var contentMode: ContentMode = .fill

    @State private var phase: AsyncImagePhase = .empty

    var body: some View {
        Group {
            switch phase {
            case .success(let image):
                image
                    .resizable()
                    .aspectRatio(contentMode: contentMode)
            case .failure:
                Image(systemName: "photo")
                    .foregroundStyle(.secondary)
            case .empty:
                ProgressView()
            @unknown default:
                EmptyView()
            }
        }
        .task(id: url) {
            await loadImage()
        }
    }

    private func loadImage() async {
        guard let url else {
            phase = .failure(URLError(.badURL))
            return
        }

        // Check cache
        if let cached = ImageCache.shared[url] {
            phase = .success(Image(uiImage: cached))
            return
        }

        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            guard let uiImage = UIImage(data: data) else {
                throw URLError(.cannotDecodeContentData)
            }
            ImageCache.shared[url] = uiImage
            phase = .success(Image(uiImage: uiImage))
        } catch {
            phase = .failure(error)
        }
    }
}

// Simple cache
final class ImageCache {
    static let shared = ImageCache()
    private let cache = NSCache<NSURL, UIImage>()

    subscript(_ url: URL) -> UIImage? {
        get { cache.object(forKey: url as NSURL) }
        set {
            if let image = newValue {
                cache.setObject(image, forKey: url as NSURL)
            } else {
                cache.removeObject(forKey: url as NSURL)
            }
        }
    }
}
```

### B. List Optimization

```swift
struct OptimizedList: View {
    let items: [Item]

    var body: some View {
        List(items) { item in
            ItemRow(item: item)
                .listRowInsets(EdgeInsets())
                .listRowSeparator(.hidden)
        }
        .listStyle(.plain)
    }
}

// Equatable conformance for performance
struct ItemRow: View, Equatable {
    let item: Item

    static func == (lhs: ItemRow, rhs: ItemRow) -> Bool {
        lhs.item.id == rhs.item.id &&
        lhs.item.title == rhs.item.title &&
        lhs.item.updatedAt == rhs.item.updatedAt
    }

    var body: some View {
        // View implementation
    }
}
```

---

## 10. Security & Dependency Management (MANDATORY)

### A. Dependency Vulnerability Scanning

SPM and CocoaPods do not include a native audit command. Use third-party scanners:

**Snyk (recommended):**
```bash
# Install Snyk CLI
brew install snyk

# Scan SPM dependencies
snyk test --file=Package.swift

# Scan CocoaPods dependencies
snyk test --file=Podfile.lock

# Monitor for new vulnerabilities continuously
snyk monitor --file=Package.swift
```

**OWASP Dependency-Check:**
```bash
# Run against the project directory
dependency-check --project "MyApp" --scan . --format HTML
```

- Run scans in CI on every PR and at least weekly on the main branch
- Review and triage all HIGH and CRITICAL findings before release

### B. Lockfile Discipline

- **SPM**: ALWAYS commit `Package.resolved` to version control for reproducible builds
- **CocoaPods**: ALWAYS commit `Podfile.lock` to version control
- Review lockfile diffs during code review to catch unexpected dependency changes

```bash
# Verify dependency resolution is deterministic
swift package resolve
git diff Package.resolved  # Should show no changes on clean resolve
```

### C. App Transport Security (ATS)

- NEVER disable ATS globally. All network connections MUST use HTTPS.
- If an exception is absolutely required, scope it to a single domain with justification:

```xml
<!-- Info.plist - scoped exception (avoid if possible) -->
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSExceptionDomains</key>
    <dict>
        <key>legacy-api.example.com</key>
        <dict>
            <key>NSTemporaryExceptionAllowsInsecureHTTPLoads</key>
            <true/>
        </dict>
    </dict>
</dict>
```

- Apple will reject apps that disable ATS without a valid reason

### D. Secret Management with Keychain

- NEVER store API keys, tokens, or passwords in `UserDefaults`, plists, or source code
- Use the iOS Keychain for all sensitive data (see Section 6B for `KeychainManager` implementation)
- For build-time secrets, use Xcode Build Configuration files (`.xcconfig`) excluded from VCS:

```bash
# .gitignore
*.xcconfig
!Shared.xcconfig  # Only commit non-secret configs
```

```
// Secrets.xcconfig (NOT committed to VCS)
API_KEY = your-secret-key-here
```

```swift
// Access in code via Info.plist
let apiKey = Bundle.main.infoDictionary?["API_KEY"] as? String
```

### E. Security Checklist

- [ ] Snyk or OWASP Dependency-Check configured in CI
- [ ] `Package.resolved` / `Podfile.lock` committed to version control
- [ ] App Transport Security enforced (no global exceptions)
- [ ] All secrets stored in Keychain, never in UserDefaults or source code
- [ ] Build-time secrets in `.xcconfig` files excluded from VCS
- [ ] Certificate pinning enabled for critical API endpoints
- [ ] Sensitive data cleared from memory when no longer needed
- [ ] CI pipeline runs vulnerability scans on every build

---

## 11. Deployment Checklist

### Code Quality
- [ ] No force unwraps
- [ ] Proper error handling
- [ ] Accessibility labels added
- [ ] Dark mode supported

### Performance
- [ ] Images optimized
- [ ] Memory leaks checked
- [ ] Launch time acceptable
- [ ] Instruments profiled

### Release
- [ ] Version/build numbers updated
- [ ] Signing configured
- [ ] App Store screenshots ready
- [ ] Privacy policy URL set

---

## 12. Quick Reference

```swift
// Swift Concurrency
async let result = fetchData()
await withTaskGroup(of: Data.self) { group in }
Task { @MainActor in }

// SwiftUI State
@State private var value: Type
@Binding var value: Type
@StateObject private var vm = ViewModel()
@ObservedObject var vm: ViewModel
@Observable final class VM { }
@Environment(\.dismiss) private var dismiss

// Modifiers
.task { await load() }
.refreshable { await refresh() }
.searchable(text: $query)
.sheet(isPresented: $show) { }
.alert(isPresented: $showAlert) { }

// Navigation
NavigationStack { }
NavigationLink(value: item) { }
.navigationDestination(for: Item.self) { }
```

---

## 13. Why This Configuration Works

1. **SwiftUI with MVVM**: Declarative UI with observable ViewModels provides automatic view updates, eliminating the delegate/datasource boilerplate of UIKit.

2. **Swift Concurrency over GCD**: Structured concurrency with async/await and actors prevents data races at compile time, replacing error-prone dispatch queue patterns.

3. **SwiftData over Core Data**: SwiftData's macro-driven model definitions reduce boilerplate by 60-70% while maintaining Core Data's mature persistence engine underneath.

4. **SPM over CocoaPods**: Swift Package Manager integrates natively with Xcode, eliminates workspace complexity, and provides hermetic builds with resolved dependency graphs.

5. **@Observable Macro**: The Observation framework provides fine-grained view invalidation, re-rendering only views that read changed properties instead of entire view hierarchies.

6. **Protocol-Oriented Architecture**: Defining service interfaces as protocols enables dependency injection with mock implementations, making ViewModels fully testable without a device.

7. **NavigationStack with Type-Safe Routing**: Value-based navigation with `navigationDestination(for:)` eliminates stringly-typed segue identifiers and enables deep linking.

8. **XCTest with async/await Support**: Native async test methods verify concurrent code without expectations or timeouts, producing deterministic and readable test suites.

9. **App Intents and Widgets**: Exposing functionality via App Intents enables Siri, Shortcuts, and Spotlight integration from a single declaration.

10. **Privacy Manifests and Tracking Transparency**: Declaring data usage in privacy manifests and using ATT ensures App Store compliance and builds user trust.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** iOS Team


**End of iOS Development Guidelines**

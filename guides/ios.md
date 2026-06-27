# iOS Development Guidelines
Mandatory standards for native iOS apps: SwiftUI-first, Observation-driven, concurrency-safe, HIG-compliant. iOS 18 SDK, Xcode 16, Swift 6, SwiftUI, Observation, SwiftData, Swift Testing.

---
name: ios
title: iOS Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [ios@18, xcode@16, swift@6.0, swiftui, observation, swiftdata, swift-testing, xcuitest, swiftpm]
requires:
  - swift
  - tdd
  - secure-coding
recommends:
  - accessibility
  - performance
  - ui
  - oauth
  - observability
provides:
  - swiftui
  - ios-observation
  - swiftui-navigation
  - ios-lifecycle
  - swiftdata
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to the **Apple-platform UI layer**. The Swift language itself (value types, optionals, concurrency primitives, ARC, `throws`/`Result`) is owned by [`swift.md`](guides://swift.md) and is **not** repeated here.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating iOS code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`swift.md`](guides://swift.md) — the language: value types, optionals, protocol-oriented design, `async`/`await`/actors, ARC, error handling. *(This guide builds on it; it does not restate it.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix. *(iOS binding: unit/integration via **Swift Testing** (`@Test`/`#expect`) or XCTest; UI flows via **XCUITest**; run with `xcodebuild test` / Cmd+U.)*
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(iOS binding: **Keychain** for all secrets, **ATS** enforced, **PrivacyInfo.xcprivacy** manifest, `Package.resolved` committed — §6.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ui.md`](guides://ui.md) — UX & Apple Human Interface Guidelines (HIG); this guide supplies only the SwiftUI binding.
> - [`accessibility.md`](guides://accessibility.md) — a11y policy *(binding: `.accessibilityLabel`/`Value`/`Hint`/`Element`, Dynamic Type, VoiceOver — §5.G)*
> - [`oauth.md`](guides://oauth.md) — auth flows *(binding: Sign in with Apple / `ASWebAuthenticationSession`, tokens in Keychain)*
> - [`performance.md`](guides://performance.md) — perf policy *(binding: Instruments, lazy lists, image caching — §5.F)*
> - [`observability.md`](guides://observability.md) — *(binding: `OSLog`/`Logger`, MetricKit, signposts)*

> 📎 **SEE ALSO:** [`android.md`](guides://android.md) · [`react-native.md`](guides://react-native.md) · [`flutter.md`](guides://flutter.md) *(only if the project is cross-platform)*

---

## 1. Core Philosophies: IOS-FIRST

iOS-UI-specific principles only. TDD, security, and language idioms come from §0.

- **I**ntuitive: follow Apple's HIG (owned by `ui.md`); embrace platform conventions, SF Symbols, and standard controls over bespoke UI.
- **O**bservation-driven: state flows through the **Observation** framework (`@Observable`, `@State`, `@Binding`, `@Environment`) — never the legacy `ObservableObject`/`@Published`/`@StateObject` stack in new code.
- **S**wiftUI-first: SwiftUI is the default for all new screens. UIKit appears only as deliberate, isolated interop (§5.D), never as the primary UI.
- **F**ine-grained & native: prefer first-party frameworks (SwiftData, NavigationStack, `.task`, AsyncImage) over third-party equivalents; let Observation re-render only what changed.
- **I**solation-safe: UI state is `@MainActor`; async work is structured (`.task`, `async let`, task groups — see `swift.md`); the app builds clean under **Swift 6 strict concurrency**.
- **R**eachable to all: every interactive view is accessible (Dynamic Type, VoiceOver labels) — policy in `accessibility.md`.
- **S**ecure by default: secrets in Keychain, ATS on, privacy manifest present — policy in `secure-coding.md`.

**Verified Code**: Agent-generated iOS code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `IOS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| IOS-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `xcodebuild test -scheme App` | exit 0, 0 skips |
| IOS-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `xcodebuild test` | failing→passing |
| IOS-TST-03 | Critical user flows MUST have a XCUITest (see `tdd.md`) | `xcodebuild test -only-testing:AppUITests` | exit 0 |
| IOS-FMT-01 | Code MUST be formatted (see `swift.md`) | `swiftformat --lint .` | no diff |
| IOS-LINT-01 | Linter MUST pass clean (see `swift.md`) | `swiftlint --strict` | 0 violations |
| IOS-CONC-01 | App MUST build under Swift 6 strict concurrency; UI state on `@MainActor` | `xcodebuild -strict-concurrency=complete` | exit 0, 0 warnings |
| IOS-STATE-01 | New view models MUST use `@Observable`, not `ObservableObject` | review / grep `ObservableObject` | none in new code |
| IOS-NAV-01 | Navigation MUST use `NavigationStack` (not deprecated `NavigationView`) | grep `NavigationView` | none |
| IOS-A11Y-01 | Interactive views MUST be accessible (see `accessibility.md`) | Accessibility Inspector / audit | 0 critical issues |
| IOS-SEC-01 | Secrets MUST live in Keychain, never UserDefaults/source (see `secure-coding.md`) | grep secrets / review | none leaked |
| IOS-SEC-02 | ATS MUST stay enabled; HTTPS only (see `secure-coding.md`) | inspect `Info.plist` | no global ATS exception |
| IOS-SEC-03 | App MUST ship a `PrivacyInfo.xcprivacy` manifest (see `secure-coding.md`) | check bundle | manifest present |
| IOS-DEP-01 | `Package.resolved` committed & in sync (see `secure-coding.md`) | `swift package resolve` + `git diff` | no diff |
| IOS-SEC-04 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `snyk test` / OWASP DC | 0 high/critical |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); `NavigationView`/`ObservableObject`/`@StateObject` in new code; force-unwraps in production paths (see `swift.md`); disabling ATS globally; storing tokens in `UserDefaults` or `Info.plist`; blocking the main actor with synchronous I/O.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
swiftformat --lint .                              # IOS-FMT-01
swiftlint --strict                                # IOS-LINT-01
xcodebuild build -scheme App \
  OTHER_SWIFT_FLAGS="-strict-concurrency=complete" # IOS-CONC-01
xcodebuild test -scheme App \
  -destination 'platform=iOS Simulator,name=iPhone 16' # IOS-TST-01/02/03
swift package resolve && git diff --exit-code Package.resolved # IOS-DEP-01
snyk test                                         # IOS-SEC-04
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Feature-grouped layout. Architectural *principles* (dependency direction, testable boundaries) come from `swift.md`/architecture guides; below is only their iOS mapping.

```
MyApp/
├── App/
│   ├── MyAppApp.swift          # @main App + Scene + root modelContainer (§5.E)
│   └── AppDelegate.swift       # ONLY if a UIKit lifecycle hook is unavoidable
├── Features/<Feature>/         # group by feature, not by type
│   ├── <Feature>View.swift     # SwiftUI view(s)
│   ├── <Feature>Model.swift    # @Observable view model (@MainActor)
│   └── Components/
├── Core/
│   ├── Network/                # actor APIClient, endpoints (see swift.md concurrency)
│   ├── Persistence/            # SwiftData @Model + ModelContainer (§5.E)
│   └── Security/Keychain.swift # Keychain wrapper (policy: secure-coding.md)
├── DesignSystem/               # shared components, theme, modifiers (see ui.md)
├── Resources/                  # Assets.xcassets, Localizable.xcstrings
├── PrivacyInfo.xcprivacy       # IOS-SEC-03
├── MyAppTests/                 # Swift Testing / XCTest (see tdd.md)
├── MyAppUITests/               # XCUITest (IOS-TST-03)
└── Package.swift               # SPM deps; Package.resolved committed (IOS-DEP-01)
```

- One view = one responsibility; extract subviews early (a `body` over ~2 screens is a smell).
- View models hold logic and state; views stay declarative and side-effect-free.

---

## 5. iOS Specifics

The unique value of this guide.

### A. SwiftUI views & state ownership

Pick the property wrapper by **who owns the value**:

| Wrapper | Use for |
|---|---|
| `@State` | view-local value (incl. an `@Observable` model **owned** by this view) |
| `@Binding` | two-way reference to state owned by a parent |
| `@Environment` | injected/system values (`\.dismiss`, `\.modelContext`, custom keys) |
| `@Bindable` | deriving `$` bindings from an `@Observable` model |

```swift
struct HomeView: View {
    @State private var model = HomeModel()        // view owns the model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("Home")
                .task { await model.load() }       // async lifecycle, auto-cancels
                .refreshable { await model.refresh() }
        }
    }

    @ViewBuilder private var content: some View {
        switch model.state {
        case .loading:        ProgressView()
        case .loaded(let xs): List(xs) { ItemRow(item: $0) }
        case .empty:          ContentUnavailableView("No Items", systemImage: "tray")
        case .failed(let e):  ContentUnavailableView("Error", systemImage: "exclamationmark.triangle",
                                                     description: Text(e.localizedDescription))
        }
    }
}
```

- Drive UI off an explicit state enum, not scattered `isLoading`/`error` flags.
- Use first-party empty/error UI (`ContentUnavailableView`) and `AsyncImage` before rolling your own.

### B. Observation framework (the modern default)

`@Observable` replaces `ObservableObject`: no `@Published`, fine-grained invalidation (only views reading a changed property re-render), and plain `@State`/`@Environment` injection.

```swift
import Observation

@Observable @MainActor
final class HomeModel {
    enum State { case loading, loaded([Item]), empty, failed(Error) }
    private(set) var state: State = .loading
    var query = ""                                  // observed automatically

    private let service: ItemServicing             // injected port (see swift.md protocols)
    init(service: ItemServicing = LiveItemService()) { self.service = service }

    func load() async {
        do {
            let items = try await service.fetch()
            state = items.isEmpty ? .empty : .loaded(items)
        } catch { state = .failed(error) }          // error policy: error-handling.md
    }
    func refresh() async { await load() }
}
```

- `@MainActor` on the model keeps UI mutations on the main actor (satisfies IOS-CONC-01).
- Use `@ObservationIgnored` for stored deps that need not trigger view updates.
- **Legacy only:** `ObservableObject`/`@Published`/`@StateObject` remain for iOS 16 targets, but new code MUST use Observation (IOS-STATE-01).

### C. Architecture — MVVM with SwiftUI

The default is **MVVM**: a SwiftUI `View` renders an `@Observable` model that depends on protocol-typed services (dependency injection via initializers/`@Environment`, mockable in tests — see `tdd.md`). Keep view models free of SwiftUI/UIKit imports so they unit-test without a simulator. For large, highly-stateful apps a unidirectional store (e.g. **The Composable Architecture**) is a valid alternative — apply it whole-app, not per-screen.

### D. UIKit interop (legacy / when SwiftUI lacks coverage)

SwiftUI is primary; reach for UIKit only for gaps (e.g. advanced text input, camera, `PHPicker`). Bridge explicitly:

```swift
struct ScannerView: UIViewControllerRepresentable {       // UIKit → SwiftUI
    @Binding var code: String
    func makeUIViewController(context: Context) -> ScannerController { … }
    func updateUIViewController(_ vc: ScannerController, context: Context) { … }
    func makeCoordinator() -> Coordinator { Coordinator(self) }   // delegate bridge
}
// SwiftUI → UIKit: UIHostingController(rootView:)
```

Keep bridges in one file per feature; never let UIKit delegate state leak into view models untranslated.

### E. App lifecycle & SwiftData

```swift
@main
struct MyAppApp: App {
    var body: some Scene {
        WindowGroup {
            RootView()
        }
        .modelContainer(for: [Item.self, Tag.self])   // SwiftData container at the Scene
    }
}
```

SwiftData is the default persistence layer (macro models over Core Data boilerplate; Core Data underneath, and still valid for advanced migration/concurrency needs):

```swift
@Model final class Item {
    @Attribute(.unique) var id: UUID
    var title: String
    var createdAt: Date
    @Relationship(deleteRule: .cascade) var tags: [Tag] = []
    init(id: UUID = .init(), title: String, createdAt: Date = .now) { … }
}

struct ItemList: View {
    @Environment(\.modelContext) private var context
    @Query(sort: \Item.createdAt, order: .reverse) private var items: [Item]
    var body: some View { List(items) { ItemRow(item: $0) } }
}
```

- Mutate via `context.insert`/`delete`; query declaratively with `@Query` + `#Predicate`.
- For lifecycle events use `@Environment(\.scenePhase)`; add an `AppDelegate` (via `@UIApplicationDelegateAdaptor`) **only** when a callback has no SwiftUI equivalent (e.g. push registration).

### F. Concurrency in the UI & Combine

- Tie async work to view lifetime with `.task`/`.task(id:)` — it auto-cancels on disappear; avoid unstructured `Task {}` that outlives the view.
- Networking lives in an `actor`; UI mutation hops to `@MainActor`. Strict-concurrency rules and `Sendable` are owned by `swift.md` — apply them, don't re-explain them.
- **Combine vs async:** prefer `async`/`await` and `AsyncSequence`. Combine is legacy — keep it only for existing reactive pipelines or APIs that still vend `Publisher`s; do not introduce it in new code.

### G. Accessibility & navigation bindings

- **Navigation:** value-typed `NavigationStack` with `navigationDestination(for:)`; bind a `NavigationPath`/`@Observable` router for deep links. `NavigationView` is deprecated (IOS-NAV-01).
- **Accessibility** (policy: `accessibility.md`): add `.accessibilityLabel/Value/Hint`, group with `.accessibilityElement(children:)`, support Dynamic Type (scalable fonts, no fixed heights), verify with VoiceOver + Accessibility Inspector. The *what/why* lives in `accessibility.md`; SwiftUI supplies the *how*.

---

## 6. Security, Keychain & Distribution

Security/supply-chain *policy* is owned by [`secure-coding.md`](guides://secure-coding.md); auth flows by [`oauth.md`](guides://oauth.md). iOS bindings only:

- **Keychain** for every secret/token (IOS-SEC-01). Thin wrapper over `Security`:
  ```swift
  func save(_ token: Data, account: String) throws {
      let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                              kSecAttrAccount as String: account,
                              kSecValueData as String: token]
      SecItemDelete(q as CFDictionary)
      guard SecItemAdd(q as CFDictionary, nil) == errSecSuccess else { throw KeychainError.save }
  }
  ```
- **ATS** (IOS-SEC-02): never disable globally; HTTPS only. A scoped `NSExceptionDomains` entry needs written justification and review.
- **Privacy manifest** (IOS-SEC-03): ship `PrivacyInfo.xcprivacy` declaring collected data types and required-reason APIs; add Tracking Transparency (ATT) prompt if you track.
- **Auth:** Sign in with Apple / OAuth via `ASWebAuthenticationSession` (flow owned by `oauth.md`); store resulting tokens in Keychain.
- **Build-time config:** non-secret values in `.xcconfig`; keep secret `.xcconfig` out of VCS. Never hardcode keys in source or `Info.plist`.
- **Dependencies (SPM):** pin versions, commit `Package.resolved` (IOS-DEP-01), scan with Snyk/OWASP Dependency-Check in CI (IOS-SEC-04).
- **Distribution basics:** bump marketing/build numbers, configure signing (prefer automatic / Xcode Cloud or `fastlane match`), archive and upload via App Store Connect / TestFlight, and ensure the privacy manifest + screenshots + privacy-policy URL are set before submission.

---

## 7. Quick Reference

```swift
@Observable @MainActor final class VM { … }   // model (B)
@State private var vm = VM()                   // view owns model
@Bindable var vm; @Binding var x; @Environment(\.modelContext) var ctx

.task { await vm.load() }     .refreshable { await vm.refresh() }
NavigationStack { … }.navigationDestination(for: Item.self) { DetailView($0) }

@Model final class Item { @Attribute(.unique) var id: UUID }   // SwiftData
@Query(sort: \Item.createdAt) var items: [Item]
```

```bash
swiftformat --lint . && swiftlint --strict          # format + lint
xcodebuild test -scheme App -destination '…iPhone 16'  # test
swift package resolve                                # deps
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] IOS-TST-01/02/03 — tests pass, bugs have regression tests, critical flows have XCUITests
- [ ] IOS-FMT-01 — `swiftformat --lint` clean
- [ ] IOS-LINT-01 — `swiftlint --strict` clean
- [ ] IOS-CONC-01 — builds under Swift 6 strict concurrency; UI state on `@MainActor`
- [ ] IOS-STATE-01 — new view models use `@Observable`
- [ ] IOS-NAV-01 — `NavigationStack` only, no `NavigationView`
- [ ] IOS-A11Y-01 — accessibility audit clean
- [ ] IOS-SEC-01/02/03 — secrets in Keychain, ATS enforced, privacy manifest present
- [ ] IOS-DEP-01 — `Package.resolved` committed & in sync
- [ ] IOS-SEC-04 — 0 high/critical CVEs in deps
- [ ] Agent ran every §3 command and documented any fixes

---
**End of iOS Development Guidelines**

# Android Development Guidelines
Mandatory standards for native Android apps: Compose-first, unidirectional state, layered architecture, DI'd and test-covered. Android SDK 35, Kotlin 2.x, Jetpack Compose, Hilt, Coroutines/Flow, Room, WorkManager.

---
name: android
title: Android Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [android-sdk@35, kotlin@2.0, jetpack-compose, hilt, coroutines, room, gradle, agp@8.5]
requires:
  - kotlin
  - tdd
  - secure-coding
recommends:
  - java
  - accessibility
  - performance
  - ui
  - oauth
  - observability
provides:
  - jetpack-compose
  - android-architecture
  - viewmodel-stateflow
  - android-lifecycle
  - hilt
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to the Android platform — the Kotlin language itself is owned by [`kotlin.md`](guides://kotlin.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Android code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`kotlin.md`](guides://kotlin.md) — the language: coroutines/Flow semantics, null-safety, idioms, ktlint/detekt. This guide does **not** restate Kotlin.
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Android binding: JUnit + MockK + Turbine for unit tests, Compose test rule / Espresso for UI; runner `./gradlew test connectedAndroidTest`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Android binding: EncryptedSharedPreferences/Keystore, Network Security Config, OWASP dependency-check, no secrets in VCS, minimal permissions.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ui.md`](guides://ui.md) — UX/component design *(binding: Compose + Material 3)*
> - [`accessibility.md`](guides://accessibility.md) — a11y policy *(binding: Compose semantics, TalkBack, touch targets)*
> - [`oauth.md`](guides://oauth.md) — auth flows *(binding: AppAuth / Credential Manager, no implicit grant)*
> - [`java.md`](guides://java.md) — legacy interop for mixed Java/Kotlin modules
> - [`performance.md`](guides://performance.md) · [`observability.md`](guides://observability.md)

> 📎 **SEE ALSO:** [`material.md`](guides://material.md) · [`ci-cd.md`](guides://ci-cd.md) · [`error-handling.md`](guides://error-handling.md) · [`logging.md`](guides://logging.md) · [`e2e-testing.md`](guides://e2e-testing.md)

---

## 1. Core Philosophies: ANDROID-FIRST

Android-platform principles only. The Kotlin language, TDD, and security come from §0 — do not restate them.

- **A**rchitecture: the official **UI → domain → data** layering; dependencies point inward; ViewModels never touch Android `View`/`Context` UI types.
- **N**o Views: **Jetpack Compose is the default UI**; XML layouts and the View system are legacy, used only for interop with existing screens.
- **D**eclarative state: **unidirectional data flow** — state flows down as immutable `UiState`, events flow up as function calls; the UI is a pure function of state.
- **R**eactive lifecycle: expose `StateFlow`; collect with `collectAsStateWithLifecycle()`; scope coroutines to `viewModelScope`/`lifecycleScope`. Never leak work across lifecycle boundaries.
- **O**ffline-capable: Room is the single source of truth; the network refreshes the cache, the UI observes the cache.
- **I**njected: **Hilt** wires every dependency at compile time; no manual singletons, no service locators.
- **D**ecoupled & testable: ViewModels and use cases are plain JVM unit tests (no emulator); composables are tested with the Compose test rule.

**Verified Code**: Agent-generated Android code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `AND-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| AND-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `./gradlew test` | exit 0, 0 skips |
| AND-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `./gradlew test` | failing→passing |
| AND-TST-03 | UI behavior MUST have Compose/instrumentation tests (see `tdd.md`) | `./gradlew connectedAndroidTest` | exit 0 |
| AND-FMT-01 | Code MUST be formatted (see `kotlin.md`) | `./gradlew ktlintCheck` | no diff |
| AND-LINT-01 | Android Lint + detekt MUST pass clean | `./gradlew lint detekt` | 0 errors |
| AND-ARCH-01 | UI/domain/data layering respected; ViewModel holds no Android UI types | review / module-graph | no inward→outward |
| AND-STATE-01 | UI state MUST be immutable + unidirectional; collected lifecycle-aware | review / `collectAsStateWithLifecycle` | no `MutableState` leaked to UI |
| AND-DI-01 | Dependencies MUST be provided by Hilt, not constructed ad hoc | review | no manual singletons |
| AND-SEC-01 | No secrets in source/VCS; on-device secrets via Keystore (see `secure-coding.md`) | grep / review | 0 literals |
| AND-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `./gradlew dependencyCheckAnalyze` | failBuildOnCVSS=7 |
| AND-SEC-03 | Cleartext traffic disabled; minimal manifest permissions (see `secure-coding.md`) | review manifest / network config | HTTPS only |
| AND-A11Y-01 | Interactive composables MUST expose semantics (see `accessibility.md`) | Accessibility Scanner / test | 0 missing labels |
| AND-REL-01 | Release builds MUST enable R8 minify + resource shrink | inspect `release` buildType | `isMinifyEnabled=true` |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); blocking the main thread with IO; holding `Activity`/`Context`/`View` references in a ViewModel; leaking secrets into `strings.xml` or `BuildConfig`; new screens built in XML when Compose is viable.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
./gradlew ktlintCheck detekt           # AND-FMT-01 / AND-LINT-01 (+ kotlin.md)
./gradlew lint                         # AND-LINT-01 (Android Lint)
./gradlew test                         # AND-TST-01/02 (JVM unit tests)
./gradlew connectedAndroidTest         # AND-TST-03 (Compose/instrumentation)
./gradlew dependencyCheckAnalyze       # AND-SEC-02
./gradlew assembleRelease              # AND-REL-01 (verify R8/shrink)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. App Architecture & Project Structure

The official Android guidance: three layers — **UI**, **domain** (optional, for reusable business logic), **data** — with dependencies pointing inward (UI → domain → data). State holders (ViewModel) live in the UI layer; repositories are the public API of the data layer and the single source of truth.

```
app/src/main/java/com/example/app/
├── App.kt                      # @HiltAndroidApp Application
├── di/                         # Hilt modules (Network, Database, Repository, Dispatcher)
├── data/                       # DATA layer — repositories are the public API
│   ├── local/                  #   Room: AppDatabase, dao/, entity/
│   ├── remote/                 #   Retrofit/Ktor api/ + dto/
│   └── repository/             #   *RepositoryImpl — caches, exposes Flow
├── domain/                     # DOMAIN layer (optional) — pure Kotlin, no Android imports
│   ├── model/                  #   domain models
│   └── usecase/                #   use cases orchestrating repositories
└── ui/                         # UI layer — Compose only
    ├── theme/                  #   Material 3 theme
    ├── navigation/             #   Navigation Compose graph
    ├── components/             #   reusable composables
    └── feature/<screen>/       #   Screen + Content composables + ViewModel + UiState
test/         # JVM unit tests — ViewModels, use cases, repositories (see tdd.md)
androidTest/  # instrumentation — Compose UI, Room, Hilt (see tdd.md)
```

- Group by **feature**, not by type; large apps split features into Gradle modules with convention plugins.
- The domain layer (and ideally domain models) MUST NOT import `android.*` (AND-ARCH-01).
- Repositories expose `Flow`; the UI observes, never calls the network directly.

---

## 5. Android Specifics

The unique value of this guide. Coroutine/Flow *semantics* are owned by [`kotlin.md`](guides://kotlin.md); below is their **Android binding**.

### A. Jetpack Compose — composables, state hoisting, unidirectional flow
State is **hoisted**: a stateless `Content` composable takes immutable state + event lambdas; a thin stateful `Screen` wires the ViewModel. This keeps UI previewable and testable.

```kotlin
@Composable
fun HomeScreen(
    onNavigateToDetail: (String) -> Unit,
    viewModel: HomeViewModel = hiltViewModel(),
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()   // lifecycle-aware (AND-STATE-01)
    HomeContent(uiState, onItemClick = onNavigateToDetail, onRetry = viewModel::retry)
}

@Composable
private fun HomeContent(                       // stateless, previewable, unit-testable
    uiState: HomeUiState,
    onItemClick: (String) -> Unit,
    onRetry: () -> Unit,
) {
    when (uiState) {
        HomeUiState.Loading -> LoadingIndicator()
        is HomeUiState.Error -> ErrorPane(uiState.message, onRetry)
        is HomeUiState.Success -> LazyColumn {
            items(uiState.items, key = { it.id }) { item ->
                ItemCard(item, modifier = Modifier
                    .clickable { onItemClick(item.id) }
                    .semantics { contentDescription = item.title })   // a11y (AND-A11Y-01)
            }
        }
    }
}
```

- **`remember`** caches across recompositions; **`rememberSaveable`** survives config changes/process death.
- **`derivedStateOf`** for values computed from other state (avoids recomposing on every keystroke); **`LaunchedEffect`/`rememberCoroutineScope`/`DisposableEffect`** for side effects keyed correctly.
- Pass **stable/`@Immutable`** types to composables; defer state reads into lambdas to minimize recomposition scope (perf details → [`performance.md`](guides://performance.md)).
- Use Material 3 + `@Preview` for every screen state. Component/UX design → [`ui.md`](guides://ui.md).

### B. ViewModel + StateFlow (UI state)
Model UI state as a single immutable type (a `data class` or a `sealed interface` for mutually-exclusive states). One-time effects (navigation, snackbars) go through a `Channel`/`SharedFlow`, never the state object.

```kotlin
sealed interface HomeUiState {
    data object Loading : HomeUiState
    data class Success(val items: List<Item>) : HomeUiState
    data class Error(val message: String) : HomeUiState
}

@HiltViewModel
class HomeViewModel @Inject constructor(
    getItems: GetItemsUseCase,
    private val refreshItems: RefreshItemsUseCase,
) : ViewModel() {
    val uiState: StateFlow<HomeUiState> = getItems()
        .map { HomeUiState.Success(it) as HomeUiState }
        .catch { emit(HomeUiState.Error(it.message ?: "Unknown")) }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), HomeUiState.Loading)

    fun retry() = viewModelScope.launch { refreshItems() }
}
```

- `WhileSubscribed(5_000)` keeps the upstream alive across short config changes but stops it when the UI is gone.
- ViewModels hold **no** `Context`/`View`/`Activity` (AND-ARCH-01); pass `@ApplicationContext` only into the data layer via Hilt.

### C. Coroutines & Flow on Android
- `viewModelScope` (cancelled in `onCleared`) and `lifecycleScope` (`repeatOnLifecycle(STARTED)`) are the only scopes for app work — never `GlobalScope`.
- Inject a `DispatcherProvider`; do IO on `Dispatchers.IO`, never the main thread (footgun → ANR).
- `collectAsStateWithLifecycle()` (not `collectAsState()`) so collection pauses when the app is backgrounded.

### D. Dependency Injection — Hilt
```kotlin
@Module @InstallIn(SingletonComponent::class)
object DataModule {
    @Provides @Singleton
    fun provideDb(@ApplicationContext ctx: Context): AppDatabase =
        Room.databaseBuilder(ctx, AppDatabase::class.java, "app.db")
            .addMigrations(MIGRATION_1_2).build()
}

@Module @InstallIn(SingletonComponent::class)
abstract class RepoModule {
    @Binds @Singleton
    abstract fun bindItemRepository(impl: ItemRepositoryImpl): ItemRepository
}
```
- `@HiltAndroidApp` on `Application`, `@AndroidEntryPoint` on activities, `@HiltViewModel` + `hiltViewModel()` for ViewModels.
- Scope to the right component (`SingletonComponent`, `ViewModelComponent`, …). `@Binds` for interface→impl, `@Provides` for constructed/third-party types. Compile-time wiring is a feature (AND-DI-01).

### E. Navigation — Navigation Compose
Single `NavHost`; routes are type-safe (use the Kotlin-serialization `@Serializable` route classes in Navigation 2.8+). Pass IDs, not objects; hoist navigation lambdas to the caller so screens stay decoupled.

```kotlin
NavHost(navController, startDestination = Home) {
    composable<Home> { HomeScreen(onNavigateToDetail = { navController.navigate(Detail(it)) }) }
    composable<Detail> { entry -> DetailScreen(entry.toRoute<Detail>().id, navController::popBackStack) }
}
```

### F. Data layer — Room (single source of truth)
Room exposes `Flow<List<Entity>>`; the repository maps entities↔domain and decides cache vs. network. Always `exportSchema = true` and supply explicit `Migration`s — never ship `fallbackToDestructiveMigration` in release. Validate migrations with `MigrationTestHelper`. Network/serialization (Retrofit/Ktor + kotlinx.serialization) lives in `data/remote`.

### G. WorkManager — deferrable, guaranteed background work
Use `WorkManager` for tasks that must survive process death / reboot (sync, upload), **not** for immediate UI work. Use Hilt's `@HiltWorker` + `HiltWorkerFactory`; apply `Constraints` (network, charging) and exponential backoff. Foreground/expedited work for user-visible long tasks.

### H. Lifecycle & permissions
- Prefer ViewModel + Compose state over Activity/Fragment lifecycle callbacks; when needed, use `LifecycleEventObserver`/`repeatOnLifecycle`. Survive config changes via `rememberSaveable` + `SavedStateHandle` (not `onSaveInstanceState` plumbing).
- Request runtime permissions with the `ActivityResult` API (`rememberLauncherForActivityResult` / `rememberPermissionState`); request at point-of-use, handle denial + "don't ask again", declare the **minimum** set in the manifest (AND-SEC-03).

### I. Build — Gradle / AGP
- Kotlin DSL (`build.gradle.kts`) + a **version catalog** (`gradle/libs.versions.toml`) as the single source of dependency versions.
- Enable the Compose compiler plugin (Kotlin 2.x), `buildConfig` only for non-secret config, and per-flavor signing via `local.properties`/CI — never commit keystores.
- Release `buildType`: `isMinifyEnabled = true`, `isShrinkResources = true`, R8 with checked `proguard-rules.pro` (AND-REL-01). Generate a **Baseline Profile** for startup/scroll performance.

### J. Security bindings
Policy is owned by [`secure-coding.md`](guides://secure-coding.md). Android specifics: on-device secrets via the **Android Keystore** / `EncryptedSharedPreferences`; enforce HTTPS with a **Network Security Config** (`cleartextTrafficPermitted="false"`) and certificate pinning for critical endpoints; OWASP dependency-check + Dependabot for the Gradle graph; auth flows per [`oauth.md`](guides://oauth.md) (Credential Manager / AppAuth, no secrets in the app).

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); language tooling (ktlint/detekt) → [`kotlin.md`](guides://kotlin.md). Android binding:

```bash
./gradlew dependencies                 # resolve graph (catalog-pinned)
./gradlew dependencyUpdates            # report newer versions
./gradlew dependencyCheckAnalyze       # AND-SEC-02: CVE scan
./gradlew lint detekt ktlintCheck      # static analysis
```
Pin versions in `gradle/libs.versions.toml`; commit `gradle.lockfile` if dependency locking is enabled. Configure Dependabot for `gradle` in `.github/dependabot.yml`.

---

## 7. Quick Reference

```bash
./gradlew assembleDebug                # build
./gradlew test                         # JVM unit tests
./gradlew connectedAndroidTest         # instrumentation/Compose UI tests
./gradlew ktlintCheck detekt lint      # format + static analysis
./gradlew installDebug                 # deploy to device/emulator
```
```kotlin
collectAsStateWithLifecycle()          // lifecycle-aware state
stateIn(viewModelScope, WhileSubscribed(5_000), initial)
remember {} / rememberSaveable {} / derivedStateOf {} / LaunchedEffect(key) {}
@HiltViewModel / hiltViewModel() / @Provides / @Binds / @HiltWorker
@Entity @Dao @Database / Flow<List<T>> / Migration
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] AND-TST-01/02 — tests pass, bugs have regression tests
- [ ] AND-TST-03 — Compose/instrumentation tests pass
- [ ] AND-FMT-01 — ktlint clean (see `kotlin.md`)
- [ ] AND-LINT-01 — Android Lint + detekt: 0 errors
- [ ] AND-ARCH-01 — UI/domain/data layering respected; ViewModel free of Android UI types
- [ ] AND-STATE-01 — immutable, unidirectional, lifecycle-aware state
- [ ] AND-DI-01 — dependencies provided by Hilt
- [ ] AND-SEC-01/03 — no secrets in VCS, HTTPS-only, minimal permissions (see `secure-coding.md`)
- [ ] AND-SEC-02 — 0 high/critical CVEs (`dependencyCheckAnalyze`)
- [ ] AND-A11Y-01 — interactive composables expose semantics (see `accessibility.md`)
- [ ] AND-REL-01 — release build: R8 minify + resource shrink enabled
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Android Guidelines**

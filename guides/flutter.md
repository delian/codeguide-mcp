# Flutter & Dart Development Guidelines
Mandatory coding standards for Flutter apps: composable widgets, sound null safety, const-correct, test-covered, accessible. Flutter 3.27+, Dart 3.6+, Material 3, Riverpod, go_router.

---
name: flutter
title: Flutter & Dart Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [flutter@3.27, dart@3.6, riverpod@2.6, go_router@14, build_runner, flutter_lints, flutter_secure_storage]
requires:
  - tdd
  - secure-coding
recommends:
  - accessibility
  - e2e-testing
  - performance
  - ui
  - ios
  - android
  - observability
provides:
  - flutter-widgets
  - flutter-state
  - dart-idioms
  - material3
  - flutter-testing
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Flutter and Dart. There is no separate `dart.md`, so Dart language idioms (sound null safety, async, records/patterns, sealed classes) are covered inline here.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Flutter/Dart code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Flutter binding: `flutter test`; widget tests use `testWidgets` + `WidgetTester`; the test binding is `TestWidgetsFlutterBinding`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(Flutter binding: secrets in `flutter_secure_storage`/Keychain/Keystore, never in the app binding or `pubspec`; CVE scan with `osv-scanner` over `pubspec.lock`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`accessibility.md`](guides://accessibility.md) — a11y policy *(binding: `Semantics` widget, `SemanticsLabel`, large-text/contrast, `flutter test` semantics matchers)*
> - [`e2e-testing.md`](guides://e2e-testing.md) — end-to-end policy *(binding: `integration_test` package, Patrol for native dialogs/permissions)*
> - [`performance.md`](guides://performance.md) — perf policy *(binding: const constructors, `RepaintBoundary`, DevTools timeline, raster/UI thread budgets)*
> - [`ui.md`](guides://ui.md) — component/UX design *(binding: Material 3, `ThemeData`, design tokens)*
> - [`ios.md`](guides://ios.md) · [`android.md`](guides://android.md) — platform channels, packaging, store requirements
> - [`observability.md`](guides://observability.md) — metrics/tracing/crash reporting *(binding: `FlutterError.onError`, Sentry/Crashlytics, `dart:developer` timeline events)*

> 📎 **SEE ALSO:** [`error-handling.md`](guides://error-handling.md) · [`comments.md`](guides://comments.md) · [`designpatterns.md`](guides://designpatterns.md) · [`cleanarch.md`](guides://cleanarch.md) · [`semver.md`](guides://semver.md) · [`material.md`](guides://material.md)

---

## 1. Core Philosophies: FLUTTER-FIRST

Flutter/Dart-specific principles only. TDD, security, accessibility, and performance policy come from §0.

- **F**unctional widgets: UI is a pure function of state — `build()` returns a fresh widget tree; never mutate widgets, rebuild them. Compose small widgets over deep nesting.
- **L**ean rebuilds: `const` constructors everywhere they apply; split widgets so `setState`/provider changes rebuild the smallest subtree; stable `Key`s on dynamic lists.
- **U**nidirectional state: state flows down, events flow up. Local ephemeral state via `setState`; shared/app state via Riverpod (recommended), or Bloc/Provider where mandated. No business logic in widgets.
- **T**yped & null-safe: sound null safety on (no `!` to silence the analyzer), Dart 3 records/patterns/sealed classes for modeling, `dart format` + `flutter analyze` clean.
- **T**est at every layer: unit (pure Dart), widget (`testWidgets`), golden (visual), integration (`integration_test`) — all test-first per `tdd.md`.
- **E**xplicit async: `Future`/`Stream` with `async`/`await`; surface async UI through `FutureBuilder`/`StreamBuilder` or Riverpod `AsyncValue`; always cancel subscriptions in `dispose`.
- **R**eference platforms intentionally: Material 3 by default, Cupertino where the iOS look is required; isolate platform code behind channels (see `ios.md`/`android.md`).

**Verified Code**: Agent-generated Flutter code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `FLT-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| FLT-TST-01 | Every feature MUST be test-first: unit + widget tests (see `tdd.md`) | `flutter test` | exit 0, 0 skips |
| FLT-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `flutter test` | failing→passing |
| FLT-TST-03 | Business-logic coverage MUST be 100% (widgets/UI excluded) | `flutter test --coverage` + check `lcov` | meets gate |
| FLT-TST-04 | Golden tests MUST be committed for shared visual widgets | `flutter test --update-goldens` then `flutter test` | goldens match |
| FLT-FMT-01 | Code MUST be formatted | `dart format --set-exit-if-changed lib/ test/` | no diff |
| FLT-LINT-01 | Analyzer MUST pass clean (incl. `prefer_const_constructors`) | `flutter analyze` | 0 errors, 0 warnings |
| FLT-TYP-01 | Sound null safety MUST hold; no `!`/`dynamic` to silence analyzer | `flutter analyze` + review | no unsafe casts |
| FLT-GEN-01 | Generated code (`*.g.dart`/`*.freezed.dart`) MUST be in sync | `dart run build_runner build --delete-conflicting-outputs` | no changes after |
| FLT-DOC-01 | Public APIs MUST have `///` docs (see `comments.md`) | `dart doc` | builds, 0 warnings |
| FLT-SEC-01 | No secrets in source/binding; secure storage only (see `secure-coding.md`) | review / grep for keys | none in `lib/`, `assets/` |
| FLT-SEC-02 | 0 known high/critical CVEs in deps (see `secure-coding.md`) | `osv-scanner --lockfile=pubspec.lock` | 0 high/critical |
| FLT-DEP-01 | `pubspec.lock` committed & in sync | `flutter pub get` then `git diff --exit-code pubspec.lock` | no drift |
| FLT-A11Y-01 | Interactive UI MUST expose semantics (see `accessibility.md`) | widget tests w/ semantics matchers | labels present |
| FLT-PERF-01 | No avoidable rebuilds; const + builders for long lists (see `performance.md`) | DevTools / review | no jank, const used |

> **Forbidden**: shipping a widget before its test (violates `tdd.md`); fixing a bug without a regression test first; `!` (null-assertion) or `// ignore:` to silence the analyzer instead of fixing the cause; committing API keys / `.env` into the binary; mutable global state in place of a provider; `setState` after `dispose`.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
dart format --set-exit-if-changed lib/ test/        # FLT-FMT-01
dart run build_runner build --delete-conflicting-outputs   # FLT-GEN-01
flutter analyze                                     # FLT-LINT-01 / TYP-01
flutter test --coverage                             # FLT-TST-01/03 (+ goldens FLT-TST-04)
dart doc                                             # FLT-DOC-01
osv-scanner --lockfile=pubspec.lock                 # FLT-SEC-02
flutter pub get && git diff --exit-code pubspec.lock # FLT-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

> **Note:** `dart pub audit` is **not** a real command — do not use it. CVE scanning is done with `osv-scanner` (or `trivy fs`) against `pubspec.lock`.

---

## 4. Project Structure

Feature-first layout. Architectural *principles* (dependency direction, ports/adapters, acyclic deps) are owned by [`cleanarch.md`](guides://cleanarch.md) / [`hexagonal.md`](guides://hexagonal.md) — below is only their Flutter mapping. Keep `domain/` pure Dart with no `package:flutter` import.

```
lib/
├── main.dart                 # bootstrap: runApp(ProviderScope(child: App()))
├── app.dart                  # MaterialApp.router, theme, go_router config
├── core/                     # shared theme, extensions, utils, error types
├── features/
│   └── <feature>/
│       ├── domain/           # entities, value objects — pure Dart, no Flutter import
│       ├── data/             # repositories, DTOs, datasources (adapters)
│       └── presentation/     # widgets, screens, Riverpod providers
└── routing/                  # go_router routes, guards
test/                         # mirrors lib/ (see tdd.md)
integration_test/            # integration_test package (see e2e-testing.md)
pubspec.yaml / pubspec.lock   # deps (lock committed)
analysis_options.yaml         # include: package:flutter_lints/flutter.yaml
```

- Group by feature, not by layer-at-the-top.
- One public widget per file; private subwidgets (`_Foo`) below it.
- `analysis_options.yaml` MUST include `flutter_lints` and enable `prefer_const_constructors`.

---

## 5. Flutter & Dart Specifics

The unique value of this guide.

### A. Widgets & the build method

A widget is an immutable description of part of the UI. `build(BuildContext)` is a **pure** function of the widget's fields + watched state — it may be called many times per second, so it MUST have no side effects.

```dart
// StatelessWidget — no mutable state; const constructor enables rebuild-skipping.
class Badge extends StatelessWidget {
  const Badge({super.key, required this.label});
  final String label;

  @override
  Widget build(BuildContext context) =>
      Chip(label: Text(label));               // composition over inheritance
}

// StatefulWidget — owns ephemeral state via setState (local-only state).
class Counter extends StatefulWidget {
  const Counter({super.key});
  @override
  State<Counter> createState() => _CounterState();
}

class _CounterState extends State<Counter> {
  int _count = 0;
  @override
  Widget build(BuildContext context) => TextButton(
        onPressed: () => setState(() => _count++),   // schedules a rebuild
        child: Text('$_count'),
      );
}
```

- **`BuildContext`** locates a widget in the tree: `Theme.of(context)`, `MediaQuery.of(context)`, `Navigator.of(context)`. Never store a `context` past the frame or use one across an `await` without an `if (!context.mounted) return;` guard.
- **`setState`** is for *ephemeral* state only (animation toggles, a text field's focus). Anything shared, persisted, or fetched belongs in a provider (§5.C).
- Lifecycle: `initState` → (`didChangeDependencies`/`didUpdateWidget`) → `build` → `dispose`. Allocate controllers/subscriptions in `initState`, free them in `dispose`. Re-sync from changed widget fields in `didUpdateWidget`.

### B. Dart 3 idioms (sound null safety, async, records, patterns, sealed)

There is no separate Dart guide — these are the language features to reach for.

```dart
// Sound null safety: nullability is in the type. Handle null, don't assert it away.
String greet(String? name) => 'Hi ${name ?? 'guest'}';   // ?? not name!
int? len = name?.length;                                  // null-aware access

// Records — lightweight, structural tuples for multiple returns.
(int, int) minMax(List<int> xs) => (xs.reduce(min), xs.reduce(max));
final (lo, hi) = minMax(values);                          // destructuring

// Pattern matching + switch expressions — exhaustive, no fallthrough.
String describe(Object o) => switch (o) {
      0 => 'zero',
      int n when n < 0 => 'negative',
      String s => 'text:$s',
      _ => 'other',
    };

// Sealed classes — model a closed set of states; switch is checked-exhaustive.
sealed class Result<T> {}
class Ok<T> extends Result<T> { Ok(this.value); final T value; }
class Err<T> extends Result<T> { Err(this.error); final Object error; }

String render(Result<int> r) => switch (r) {
      Ok(:final value) => 'ok $value',
      Err(:final error) => 'err $error',
    };
```

- **async/await**: `Future<T>` for one value, `Stream<T>` for many. `await` for sequential, `Future.wait` for concurrent. Mark `async` functions and return `Future<void>` (never bare `void`) so callers can await/handle errors.
- **Cancellation**: hold `StreamSubscription` / `Timer` in fields and cancel in `dispose` (a leaked subscription is a memory leak and a `setState`-after-dispose crash).
- **Immutability**: prefer `final` fields and `const` constructors; use Freezed or hand-written `copyWith`/`==`/`hashCode` for value objects. Records replace ad-hoc tuple classes.
- Error handling strategy (when to throw vs. return a `Result`) is owned by [`error-handling.md`](guides://error-handling.md); the Dart binding above (sealed `Result`) is the typed-error option.

### C. State management

Pick one approach per app. **Riverpod is recommended** (compile-safe, testable, no `BuildContext` needed to read state); Bloc (event/state streams) and Provider (InheritedWidget wrapper) are acceptable where a team mandates them. Avoid raw `InheritedWidget` plumbing and deprecated `StateProvider`/`ChangeNotifierProvider` for new code.

```dart
// Riverpod (code-gen). build_runner generates the provider.
@riverpod
class UserList extends _$UserList {
  @override
  Future<List<User>> build() => ref.read(userRepoProvider).fetchAll();

  Future<void> refresh() async =>
      state = await AsyncValue.guard(() => ref.read(userRepoProvider).fetchAll());
}

// Consume in a widget; AsyncValue models loading/error/data exhaustively.
class UserListView extends ConsumerWidget {
  const UserListView({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) =>
      ref.watch(userListProvider).when(
            data: (users) => _List(users: users),
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (e, st) => Center(child: Text('Error: $e')),
          );
}
```

- `ref.watch` to rebuild on change; `ref.read` for one-off actions (e.g. in callbacks); `ref.listen` for side effects.
- Wrap the app in a single `ProviderScope`; override providers in tests with `ProviderContainer(overrides: [...])`.
- Keep providers in `presentation/`; they depend inward on `domain`/`data`, never the reverse.

### D. Material 3, Cupertino & theming

Material 3 is the default (`ThemeData(useMaterial3: true)` is implied in current Flutter). Drive all color/typography from a `ColorScheme.fromSeed` + `Theme.of(context)` — never hardcode colors. Component/UX guidance is owned by [`ui.md`](guides://ui.md) / [`material.md`](guides://material.md).

```dart
MaterialApp.router(
  theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo)),
  darkTheme: ThemeData(
    colorScheme: ColorScheme.fromSeed(
      seedColor: Colors.indigo, brightness: Brightness.dark),
  ),
  routerConfig: appRouter,
);
```

Use Cupertino widgets (`CupertinoApp`/`CupertinoPageScaffold`) only when the app must look native on iOS; otherwise Material adapts acceptably on both. Read platform via `Theme.of(context).platform`, not `dart:io` `Platform` (which breaks on web).

### E. Layout & constraints

Flutter layout is "constraints go down, sizes go up, parent sets position." Master this to avoid overflow/`RenderFlex` errors.

- **`Row`/`Column`** lay out children along an axis; control distribution with `mainAxisAlignment` / `crossAxisAlignment`.
- **`Expanded`/`Flexible`** divide free space along the main axis; **`Spacer`** pushes siblings apart.
- An unbounded child in a scrollable (e.g. a `Column` of unbounded height) causes overflow — wrap in `Expanded`, give a bounded size, or use a sliver/`ListView`.
- Use `LayoutBuilder`/`MediaQuery` for responsive layout; `SafeArea` for notches.

### F. Navigation

Use **go_router** (declarative, URL-based, deep-link & web friendly) over imperative `Navigator.push` for app-level routing. Raw `Navigator` 1.0 `push`/`pop` is fine for transient dialogs/sheets.

```dart
final appRouter = GoRouter(
  routes: [
    GoRoute(path: '/', builder: (c, s) => const HomeScreen()),
    GoRoute(
      path: '/user/:id',
      builder: (c, s) => UserScreen(id: s.pathParameters['id']!),
    ),
  ],
  redirect: (c, s) => isLoggedIn ? null : '/login',   // route guards
);
```

### G. Async UI

Surface `Future`/`Stream` results without manual `setState` juggling:

```dart
FutureBuilder<User>(
  future: _userFuture,                       // create the future in initState, NOT in build
  builder: (context, snap) => switch (snap) {
        AsyncSnapshot(connectionState: ConnectionState.waiting) =>
            const CircularProgressIndicator(),
        AsyncSnapshot(hasError: true, :final error) => Text('Error: $error'),
        AsyncSnapshot(:final data?) => Text(data.name),
        _ => const SizedBox.shrink(),
      },
);
```

Footgun: creating the `Future`/`Stream` inside `build` re-runs it on every rebuild — hoist it into `initState` or a provider. Prefer Riverpod `AsyncValue.when` for app state; reserve `FutureBuilder`/`StreamBuilder` for local one-shots.

### H. Common footguns

- Missing `const` → unnecessary rebuilds; the `prefer_const_constructors` lint (FLT-LINT-01) catches these.
- `BuildContext` used across an `await` → guard with `if (!context.mounted) return;`.
- `setState` called after `dispose` → check `mounted`; cancel subscriptions/timers in `dispose`.
- Rebuilding a whole screen for one changing value → split into a small `Consumer`/child widget, or use `select`.
- `ListView(children: [...])` for long/lazy data → use `ListView.builder` (lazy) with stable `Key`s.
- Heavy synchronous work on the UI thread → offload to `compute`/`Isolate.run`.

---

## 6. Testing

Policy (test-first, Red-Green-Refactor, coverage, regression-before-fix) is owned by [`tdd.md`](guides://tdd.md); end-to-end policy by [`e2e-testing.md`](guides://e2e-testing.md). Flutter binding:

| Layer | Tool | Notes |
|-------|------|-------|
| Unit | `flutter test` / `test` | pure Dart logic, providers via `ProviderContainer` |
| Widget | `testWidgets` + `WidgetTester` | `pumpWidget`, `find.byType`, `tester.tap`, `pump`/`pumpAndSettle` |
| Golden | `matchesGoldenFile` | visual regression for shared widgets (FLT-TST-04) |
| Integration/E2E | `integration_test` + Patrol | real device/emulator; Patrol drives native dialogs & permissions |

```dart
testWidgets('increments on tap', (tester) async {
  await tester.pumpWidget(const MaterialApp(home: Counter()));
  expect(find.text('0'), findsOneWidget);
  await tester.tap(find.byType(TextButton));
  await tester.pump();                         // advance one frame
  expect(find.text('1'), findsOneWidget);
});
```

- Override providers/repositories with fakes via `ProviderScope(overrides: [...])` — no network in widget tests.
- Accessibility checks belong here too: assert `Semantics` labels (see `accessibility.md`).

---

## 7. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md); versioning → [`semver.md`](guides://semver.md). Flutter binding:

```bash
flutter pub get                 # install from pubspec.lock (reproducible)
flutter pub add <pkg>           # add dep (updates pubspec.yaml + lock)
flutter pub upgrade --major-versions   # update within/over constraints
dart run build_runner watch --delete-conflicting-outputs   # codegen (freezed/riverpod/json)
osv-scanner --lockfile=pubspec.lock    # FLT-SEC-02: CVE scan
```

- Commit `pubspec.lock`. Use caret constraints (`^2.6.0`) for direct deps — never `any` or `latest`.
- Secrets: use `--dart-define`/`--dart-define-from-file` at build time and `flutter_secure_storage` at runtime. Never bake keys into `assets/` or `pubspec.yaml` (FLT-SEC-01).
- Run code generation (`build_runner`) before analyze/test so generated parts are current (FLT-GEN-01).

---

## 8. Quick Reference

```bash
flutter pub get                              # setup
flutter test --coverage                      # test + coverage
flutter analyze                              # lint + null-safety
dart format lib/ test/                       # format
dart run build_runner build --delete-conflicting-outputs   # codegen
flutter run -d <device>                      # run
dart doc                                     # docs
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] FLT-FMT-01 — `dart format` clean, no diff
- [ ] FLT-LINT-01 / TYP-01 — `flutter analyze` clean, sound null safety, no silencing
- [ ] FLT-GEN-01 — `build_runner` output in sync
- [ ] FLT-TST-01/02/03/04 — unit/widget tests pass, bugs have regression tests, coverage ≥ gate, goldens match
- [ ] FLT-DOC-01 — public APIs documented, `dart doc` clean
- [ ] FLT-SEC-01 — no secrets in source/binding, secure storage used
- [ ] FLT-SEC-02 — `osv-scanner` 0 high/critical CVEs
- [ ] FLT-DEP-01 — `pubspec.lock` committed & in sync
- [ ] FLT-A11Y-01 — interactive UI exposes semantics
- [ ] FLT-PERF-01 — const-correct, no avoidable rebuilds, builders for long lists
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Flutter & Dart Guidelines**

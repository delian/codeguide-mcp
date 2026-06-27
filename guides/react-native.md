# React Native Development Guidelines
Mandatory standards for cross-platform iOS/Android apps with React Native on the New Architecture. React Native 0.76+, Expo SDK 52+, TypeScript, React Navigation/Expo Router, Reanimated 3, FlashList.

---
name: react-native
title: React Native Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [react-native@0.76, expo@52, typescript@5.6, react-navigation@7, expo-router@4, reanimated@3, react-native-gesture-handler@2, "@shopify/flash-list@1", jest, "@testing-library/react-native", detox]
requires:
  - reactjs
  - typescript
  - tdd
  - secure-coding
recommends:
  - javascript
  - accessibility
  - e2e-testing
  - performance
  - ios
  - android
  - observability
provides:
  - react-native-newarch
  - expo
  - rn-navigation
  - native-modules
  - rn-lists
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to React Native. React itself (hooks, component model, state) is owned by [`reactjs.md`](guides://reactjs.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating React Native code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`reactjs.md`](guides://reactjs.md) — components, hooks, rendering, state, memoization. *(RN reuses React; only the host components and platform APIs differ.)*
> - [`typescript.md`](guides://typescript.md) — strict typing, `tsconfig`, generics. *(RN binding: `strict: true`, typed navigation params, `npx tsc --noEmit`.)*
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(RN binding: Jest + `@testing-library/react-native` for units; Detox/Maestro for E2E.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVE policy. *(RN binding: `expo-secure-store`/`react-native-keychain` for secrets; never ship keys in the JS bundle.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`accessibility.md`](guides://accessibility.md) — a11y policy *(binding: `accessibilityRole`/`accessibilityLabel`/`accessibilityState` props, `AccessibilityInfo`).*
> - [`e2e-testing.md`](guides://e2e-testing.md) — E2E strategy *(binding: Detox or Maestro on simulators/devices).*
> - [`performance.md`](guides://performance.md) — perf budgets/profiling *(binding: list virtualization, JS-thread vs UI-thread, Reanimated worklets).*
> - [`ios.md`](guides://ios.md) · [`android.md`](guides://android.md) — the native platforms beneath RN (entitlements, permissions, store builds, native modules).
> - [`javascript.md`](guides://javascript.md) · [`observability.md`](guides://observability.md)

> 📎 **SEE ALSO:** [`flutter.md`](guides://flutter.md) *(alternative cross-platform stack)* · [`oauth.md`](guides://oauth.md) *(mobile auth flows)*

---

## 1. Core Philosophies: NATIVE-FIRST

React Native-specific principles only. TDD, security, React fundamentals, and typing come from §0.

- **N**ew Architecture by default: Fabric renderer + TurboModules + JSI are the default in 0.76+; never scaffold against or re-enable the legacy bridge.
- **A**daptive per platform: respect each OS's conventions via `Platform`, `.ios.tsx`/`.android.tsx` files, and safe-area insets — not a lowest-common-denominator UI.
- **T**hread-aware: keep the JS thread free; run animations/gestures on the UI thread via Reanimated worklets; virtualize all long lists.
- **I**ntegrated toolchain: Expo (managed or prebuild/CNG) is the default; drop to a bare/`expo prebuild` native project only when a dependency demands it.
- **V**erified on real targets: gates in §2 plus a smoke run on at least one iOS simulator and one Android emulator before delivery.
- **E**xpo-native primitives: prefer Expo SDK modules (`expo-secure-store`, `expo-image`, `expo-haptics`, `expo-router`) over unmaintained community equivalents.

**Verified Code**: Agent-generated React Native MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `RN-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| RN-TST-01 | Every feature MUST be test-first (see `tdd.md`) | `npm test` | exit 0, 0 skips |
| RN-TST-02 | Each bug MUST get a regression test before the fix (see `tdd.md`) | `npm test` | failing→passing |
| RN-TST-03 | Critical user journeys MUST have an E2E test (see `e2e-testing.md`) | `detox test` / `maestro test` | exit 0 |
| RN-TYP-01 | No type errors; strict TS, typed nav params (see `typescript.md`) | `npx tsc --noEmit` | exit 0 |
| RN-FMT-01 | Code MUST be formatted | `npx @biomejs/biome format .` (or `prettier --check`) | no diff |
| RN-LINT-01 | Linter MUST pass clean | `npx @biomejs/biome lint .` (or `eslint .`) | exit 0 |
| RN-ARCH-01 | New Architecture (Fabric/TurboModules) MUST stay enabled; no legacy bridge | `grep newArchEnabled` / Expo config | enabled, no `NativeModules` bridge specs |
| RN-PERF-01 | Long lists MUST use FlashList/FlatList virtualization (see `performance.md`) | review / grep for `.map` over large data in JSX | no unvirtualized lists |
| RN-A11Y-01 | Interactive elements MUST expose a11y props (see `accessibility.md`) | review / a11y lint | role+label present |
| RN-SEC-01 | Secrets MUST be in SecureStore/Keychain, never `AsyncStorage` or bundle (see `secure-coding.md`) | grep `AsyncStorage`/`EXPO_PUBLIC_` for secrets | none found |
| RN-SEC-02 | 0 high/critical CVEs in deps (see `secure-coding.md`) | `npm audit --audit-level=high` | 0 high/critical |
| RN-DEP-01 | Lockfile in sync; Expo deps SDK-aligned | `npm ci` + `npx expo install --check` | in sync |
| RN-DOC-01 | Public components/hooks documented (see `comments.md`) | review / `tsc` doc build | JSDoc present |

> **Forbidden**: shipping implementation before its test (violates `tdd.md`); fixing a bug without a regression test first; storing tokens/PII in `AsyncStorage`; embedding secrets in `EXPO_PUBLIC_*` env or the JS bundle; re-enabling the legacy bridge; rendering large datasets with `.map()` instead of a virtualized list; inline `StyleSheet.create` objects rebuilt every render.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green.

```bash
npx @biomejs/biome check .          # RN-FMT-01 / RN-LINT-01 (or prettier + eslint)
npx tsc --noEmit                    # RN-TYP-01
npm test                            # RN-TST-01/02
npx expo-doctor                     # RN-ARCH-01 / RN-DEP-01 (native deps, SDK alignment)
npx expo install --check            # RN-DEP-01 (SDK-compatible versions)
npm audit --audit-level=high        # RN-SEC-02
# Then smoke-run: npx expo run:ios && npx expo run:android (or expo start)
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure

Two valid layouts. **Expo Router** uses file-based routing under `app/`; classic **React Navigation** centralizes route trees. Architectural principles (separation of UI/logic, dependency direction) come from `reactjs.md`/`hexagonal.md` — below is only the RN mapping.

```
# Expo Router (recommended)
app/                      # file-based routes (each file = a screen)
│   ├── _layout.tsx       # root layout: providers, Stack/Tabs
│   ├── (tabs)/           # tab group
│   │   ├── index.tsx     # /  (Home)
│   │   └── profile.tsx   # /profile
│   └── order/[id].tsx    # dynamic route /order/:id
src/
├── components/           # reusable UI; ui/ primitives, shared/ composites
├── features/<domain>/    # hooks/ services/ store/ per feature (group by domain)
├── hooks/                # cross-feature hooks
├── services/             # API clients, native-module wrappers
├── theme/                # tokens: colors, spacing, typography (no CSS — see §5.D)
└── types/                # shared TS types
__tests__/  or  *.test.tsx co-located   # see tdd.md
app.json / app.config.ts  # Expo config (plugins, permissions, newArchEnabled)
```

- Group by feature/domain, not by file type. Keep screens thin; push logic into hooks.
- Platform-divergent code lives in `Name.ios.tsx` / `Name.android.tsx`; the bundler resolves the right file from a plain `import './Name'`.

---

## 5. React Native Specifics

The unique value of this guide.

### A. New Architecture (default in 0.76+)
Fabric (new renderer), TurboModules (lazy, typed native modules), and JSI (synchronous JS↔native calls, no JSON bridge) are **on by default**. Do not author against the legacy bridge.

- Keep `newArchEnabled: true` (Expo sets it; bare RN: `gradle.properties` + `Podfile` flag). `npx expo-doctor` flags incompatible deps.
- New native modules MUST be **TurboModules**, specified with Codegen from a typed JS spec (`*NativeComponent`/`Native*` spec files) — not the old `NativeModules` + `RCTBridgeModule` pattern.
- JSI enables synchronous host functions; prefer existing Expo/community TurboModules before writing your own.

### B. Core components & APIs
There is no DOM. Build from RN host components, not HTML:

```tsx
import { View, Text, Pressable, ScrollView, TextInput, Image } from 'react-native';
// View ≈ div, Text is REQUIRED to wrap any string, Pressable replaces
// Touchable*; never put raw text outside <Text>. ScrollView for small,
// bounded content only — long/unbounded data goes to a virtualized list (§5.E).
```

- Use `Pressable` (with `android_ripple` / pressed state) over the legacy `TouchableOpacity`/`TouchableHighlight`.
- Wrap screens in `SafeAreaView`/`useSafeAreaInsets` (`react-native-safe-area-context`) for notches and Android edge-to-edge.
- Prefer `expo-image` over the core `Image` for caching, transitions, and memory behavior.

### C. Navigation (React Navigation / Expo Router)
Both are built on React Navigation. Routes and params MUST be typed.

```tsx
// Expo Router — typed, file-based
import { Link, useLocalSearchParams } from 'expo-router';
<Link href={{ pathname: '/order/[id]', params: { id: '123' } }}>View order</Link>
const { id } = useLocalSearchParams<{ id: string }>();   // typed params

// React Navigation — typed param list
type HomeStackParamList = { Home: undefined; OrderDetail: { orderId: string } };
const navigation = useNavigation<NativeStackNavigationProp<HomeStackParamList>>();
navigation.navigate('OrderDetail', { orderId });          // autocompleted + checked
```

Use native-stack (`@react-navigation/native-stack`) for native screen transitions; deep-link config maps URLs to routes (Expo Router derives them from the file tree).

### D. Styling — `StyleSheet`, not CSS
No CSS/CSS-in-JS cascade. Styles are JS objects (a flexbox subset, default `flexDirection: 'column'`, dimensions unitless DP).

```tsx
const styles = StyleSheet.create({          // create ONCE at module scope
  card: { flex: 1, padding: 16, borderRadius: 8 },
});
// Platform-divergent style:
...Platform.select({ ios: { shadowOpacity: 0.1 }, android: { elevation: 4 } })
```

- Never build the style object inline in `render` (defeats memoization, allocates each frame). Compose with the array form: `style={[styles.card, isActive && styles.active]}`.
- Centralize design tokens (colors/spacing/typography) in `theme/`; for utility-class DX, NativeWind (Tailwind) is acceptable but still compiles to `StyleSheet`.

### E. Lists & performance — FlatList / FlashList
Rendering large arrays with `.map()` mounts every row → jank and OOM. Use a virtualized list (RN-PERF-01).

```tsx
import { FlashList } from '@shopify/flash-list';   // preferred: recycles views
<FlashList
  data={orders}
  keyExtractor={(o) => o.id}
  estimatedItemSize={72}                 // FlashList: required for recycling
  renderItem={({ item }) => <OrderRow order={item} />}
  onEndReached={fetchMore} onEndReachedThreshold={0.5}
/>
```

- Memoize `renderItem`/`keyExtractor`; wrap row components in `React.memo` (rules: `reactjs.md`).
- For `FlatList`, set `getItemLayout` for fixed-height rows, plus `windowSize`/`maxToRenderPerBatch`/`initialNumToRender`. Profiling/budgets policy: `performance.md`.

### F. Gestures & animations — Reanimated 3 + Gesture Handler
Run animation/gesture logic on the **UI thread** via worklets so it stays smooth when JS is busy.

```tsx
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';
const scale = useSharedValue(1);
const style = useAnimatedStyle(() => ({ transform: [{ scale: scale.value }] }));  // worklet
// drive with react-native-gesture-handler's Gesture API (not the legacy PanResponder)
```

Use `react-native-gesture-handler`'s `Gesture`/`GestureDetector` over the legacy `PanResponder`. Reserve the JS-thread `Animated` API for trivial cases.

### G. Platform-specific code & native modules
- Branch small differences with `Platform.OS`/`Platform.select`; split larger ones into `.ios.tsx`/`.android.tsx`.
- Need native capability? Reach for an Expo SDK module or a maintained TurboModule first. Authoring one: define a typed Codegen spec, implement Swift/Kotlin (or C++), wire via an Expo config plugin for CNG. Native platform concerns (entitlements, Gradle/Pods, store config) are owned by [`ios.md`](guides://ios.md) / [`android.md`](guides://android.md).

### H. Async storage, secure storage & permissions
- **Non-sensitive** local state → `@react-native-async-storage/async-storage` (unencrypted key/value; fine for caches/prefs).
- **Sensitive** data (tokens, PII) → `expo-secure-store` or `react-native-keychain` (Keychain/Keystore-backed). Never `AsyncStorage`, never the JS bundle (RN-SEC-01; policy `secure-coding.md`). `EXPO_PUBLIC_*` env vars are embedded in the client bundle — public only.
- **Permissions**: request at point-of-use via the owning module's hook (e.g. `expo-camera`'s `useCameraPermissions`, `expo-location`); declare the matching iOS usage strings / Android manifest permissions in `app.json` plugins. Platform specifics: `ios.md`/`android.md`.

### I. Common footguns
- Raw string outside `<Text>` → crash. Always wrap text.
- Inline `StyleSheet`/arrow props on list rows → re-renders; hoist + memoize.
- Forgetting safe-area insets → content under the notch / nav bar.
- `console.log` in worklets, or calling JS-thread functions from a worklet without `runOnJS`.
- Importing a Node/web-only library that has no RN/Hermes support → runtime crash; check Hermes compatibility.

---

## 6. Tooling & Dependencies

Security/supply-chain *policy* → [`secure-coding.md`](guides://secure-coding.md). React Native binding:

```bash
npx create-expo-app@latest          # scaffold (Expo, New Arch, TS by default)
npx expo install <pkg>              # add SDK-compatible dep (NOT plain npm install for native deps)
npx expo install --check            # RN-DEP-01: deps match the SDK
npx expo-doctor                     # RN-ARCH-01: native/New-Arch compatibility
npm ci                              # reproducible install in CI (commit the lockfile)
npm audit --audit-level=high        # RN-SEC-02: CVE scan
npx expo prebuild                   # generate native ios/android projects (CNG) only when needed
```

Commit the lockfile (`package-lock.json`/`yarn.lock`/`pnpm-lock.yaml`). Use `npx expo install` for any native dependency so versions stay SDK-aligned; reserve plain `npm install` for pure-JS packages.

---

## 7. Quick Reference

```bash
npx expo start                 # dev server (Metro)
npx expo run:ios               # build + run on iOS simulator
npx expo run:android           # build + run on Android emulator
npm test                       # Jest + @testing-library/react-native
npx tsc --noEmit               # type check
npx @biomejs/biome check .     # lint + format
detox test                     # E2E (see e2e-testing.md)
eas build --platform all       # production build (EAS)
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] RN-TST-01/02 — tests pass, bugs have regression tests
- [ ] RN-TST-03 — critical journeys covered by Detox/Maestro
- [ ] RN-TYP-01 — `tsc --noEmit` clean, nav params typed
- [ ] RN-FMT-01 / RN-LINT-01 — formatter + linter clean
- [ ] RN-ARCH-01 — New Architecture enabled, no legacy bridge (`expo-doctor` clean)
- [ ] RN-PERF-01 — long lists virtualized (FlashList/FlatList)
- [ ] RN-A11Y-01 — interactive elements have role + label
- [ ] RN-SEC-01 — secrets in SecureStore/Keychain, none in bundle/AsyncStorage
- [ ] RN-SEC-02 — `npm audit` 0 high/critical
- [ ] RN-DEP-01 — lockfile committed, `expo install --check` clean
- [ ] RN-DOC-01 — public components/hooks documented
- [ ] Smoke-ran on an iOS simulator and an Android emulator
- [ ] Agent ran every §3 command and documented any fixes

---
**End of React Native Guidelines**

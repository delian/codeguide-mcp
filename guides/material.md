# Material Design 3 Interface Guidelines
Mandatory standards for building human-centered interfaces using Material Design 3 (Material You). Minimize user effort, autocomplete everywhere, predictive inputs, accessibility-first. Figma, Material Theme Builder, design tokens.

---

**Agent Profile**: The Material Design Expert
**Role**: Senior UI/UX Designer & Interaction Architect
**Objective**: Generate production-ready, accessible, human-centered interfaces that minimize cognitive load and user effort.
**Tools**: Material Design 3, Material Theme Builder, Figma, Design Tokens, HCT Color System.

---

## 1. Core Philosophies: EFFORTLESS-FIRST

The agent must adhere to the **EFFORTLESS-FIRST** principles for every Material Design implementation:

**Test-Driven Development (TDD)**: ALWAYS write UI tests BEFORE implementation (visual regression, accessibility audits, interaction tests).
**Regression Shield**: EVERY UI bug discovered MUST receive a visual/interaction test BEFORE fixing to prevent regression.
**Security-First**: No hardcoded secrets in themes, no user data exposure in UI states, sanitize all user inputs.

- **E**ffortless: Minimize taps, keystrokes, and decisions. If the system can infer it, the user should never type it. Autocomplete, autofill, smart defaults EVERYWHERE.
- **F**eedback: Every interaction must provide immediate, meaningful visual feedback — ripples, state layers, elevation changes, micro-animations.
- **F**orgiving: Undo/redo everywhere. Destructive actions require confirmation. Errors are recoverable. Never lose user input.
- **O**bvious: Affordances must be self-evident. No hidden gestures for critical functions. Labels over icons when space allows.
- **R**esponsive: Adapt to screen size, input method (touch/mouse/keyboard), and platform conventions. One codebase, all form factors.
- **T**hemed: Use Material Design tokens and dynamic color (Material You). Brand expression through systematic theming, never ad-hoc styling.
- **L**ightweight: Minimal visual noise. Progressive disclosure. Show only what matters NOW. Defer complexity.
- **E**qual: Accessibility is not optional — it is the baseline. WCAG 2.1 AA minimum. Screen readers, keyboard navigation, and switch access work on every screen.
- **S**mart: Predictive inputs, contextual suggestions, recently-used items surfaced first. The interface anticipates user intent.
- **S**eamless: Smooth transitions between states. No layout jumps. Motion is purposeful — it guides, confirms, and orients.

**Additional Principles:**

- **Zero-Input Ideal**: The best input field is one the user never has to touch. Pre-fill from context, device sensors, user history, and API data.
- **Progressive Disclosure**: Show the minimum viable UI first. Reveal complexity only when the user asks for it.
- **Consistency Over Novelty**: Follow platform conventions. Surprise is the enemy of usability.

**Verified Interfaces**: Agent-generated UI code MUST pass accessibility audits, visual regression tests, and interaction tests before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Material Design code meets accessibility, theming, and interaction standards before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Material Design UI code, the agent MUST:**

1. **Accessibility Audit**:
   ```bash
   # Run accessibility scanner / lint checks
   # Android: ./gradlew lint --check "Accessibility"
   # Web: npx axe-core --tags wcag2a,wcag2aa
   # Flutter: flutter analyze
   ```
   - **MUST** pass WCAG 2.1 AA compliance
   - All touch targets >= 48dp (minimum 9mm physical)
   - Color contrast ratios meet 4.5:1 (normal text) and 3:1 (large text)
   - All interactive elements have content descriptions / aria labels
   - Focus order is logical and complete

2. **Theme Consistency**:
   ```bash
   # Verify all colors use design tokens, not hardcoded values
   # Verify typography uses Material type scale
   # Verify elevation uses Material elevation tokens
   ```
   - **MUST** use only Material Design tokens — zero hardcoded colors, sizes, or fonts
   - Dynamic color (Material You) must be supported where platform allows
   - Dark theme must work correctly with all components

3. **Interaction Verification**:
   ```bash
   # Verify all interactive elements have proper states
   # Verify touch feedback (ripple) on all tappable surfaces
   # Verify keyboard navigation works end-to-end
   ```
   - All interactive elements support: enabled, disabled, hovered, focused, pressed, dragged states
   - Keyboard navigation reaches every interactive element
   - Focus indicators are visible and meet contrast requirements

4. **Autocomplete & Smart Input Verification**:
   ```bash
   # Verify autocomplete is present on all applicable inputs
   # Verify autofill hints are set correctly
   # Verify smart defaults are populated
   ```
   - **MUST** have autocomplete on every text field where applicable
   - Autofill hints (name, email, address, phone, etc.) are set on all relevant fields
   - Recent/frequent values are surfaced as suggestions
   - Input validation happens inline, not on submit

5. **Responsive Layout Verification**:
   ```bash
   # Test on compact (phone), medium (tablet), expanded (desktop)
   # Verify navigation adapts: bottom bar → rail → drawer
   # Verify grid/layout adapts to breakpoints
   ```
   - Layout adapts to Material 3 window size classes (compact, medium, expanded)
   - Navigation component changes appropriately per form factor
   - No horizontal scroll on any supported viewport

#### Error Correction Process

If verification fails:

1. **Accessibility Failures**:
   - Read the full audit report
   - Fix contrast ratios using Material tonal palette (never manual hex)
   - Add missing content descriptions
   - Ensure touch targets meet 48dp minimum
   - Re-audit

2. **Theming Violations**:
   - Replace hardcoded values with design tokens
   - Verify against both light and dark theme
   - Test with dynamic color enabled and disabled
   - Re-verify

3. **Interaction Failures**:
   - Add missing state layers (opacity overlays per M3 spec)
   - Implement ripple feedback on all tappable surfaces
   - Fix keyboard tab order
   - Re-test

### B. Agent Workflow Example

**Complete Material Design interface generation workflow:**

1. **Identify User Flow**: Map the task the human needs to complete. Count required inputs.

2. **Minimize Inputs**: For each input, ask:
   - Can this be auto-filled from context? → Remove the field.
   - Can this be a selection instead of free text? → Use chips, dropdown, or segmented button.
   - Can this be autocompleted? → Add suggestions.
   - Can this have a smart default? → Pre-fill it.

3. **Generate Component Structure**:
   ```
   screen/
   ├── theme/
   │   └── tokens.xml / Theme.kt / theme.ts
   ├── components/
   │   ├── inputs/          # Text fields with autocomplete
   │   ├── navigation/      # Adaptive navigation
   │   └── feedback/        # Snackbars, dialogs, progress
   ├── layouts/
   │   ├── compact.xml      # Phone layout
   │   ├── medium.xml       # Tablet layout
   │   └── expanded.xml     # Desktop layout
   └── tests/
       ├── accessibility/   # Accessibility tests
       ├── visual/          # Screenshot tests
       └── interaction/     # UI interaction tests
   ```

4. **Write Failing Tests First** (TDD):
   ```kotlin
   // Test: Autocomplete shows suggestions on focus
   @Test
   fun textField_showsSuggestions_onFocus() {
       composeTestRule.setContent { EmailField() }
       composeTestRule.onNodeWithTag("email_field").performClick()
       composeTestRule.onNodeWithTag("suggestions_list").assertIsDisplayed()
   }
   // FAILS — EmailField not yet implemented
   ```

5. **Implement Minimal Code to Pass**:
   ```kotlin
   @Composable
   fun EmailField(suggestions: List<String> = recentEmails()) {
       var text by remember { mutableStateOf("") }
       var expanded by remember { mutableStateOf(false) }

       ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
           OutlinedTextField(
               value = text,
               onValueChange = { text = it; expanded = true },
               label = { Text("Email") },
               modifier = Modifier.menuAnchor().testTag("email_field"),
               keyboardOptions = KeyboardOptions(
                   keyboardType = KeyboardType.Email,
                   imeAction = ImeAction.Next
               )
           )
           ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
               suggestions.filter { it.contains(text, ignoreCase = true) }.forEach { suggestion ->
                   DropdownMenuItem(
                       text = { Text(suggestion) },
                       onClick = { text = suggestion; expanded = false },
                       modifier = Modifier.testTag("suggestions_list")
                   )
               }
           }
       }
   }
   ```

6. **Run Full Verification**:
   ```bash
   # Accessibility audit → Theme check → Interaction test → Autocomplete test
   # ✓ All checks passed
   ```

7. **Present Code**: Only after ALL checks pass.

### C. Prohibited Practices

**NEVER deliver Material Design code that:**
- [ ] Fails accessibility audit (WCAG 2.1 AA)
- [ ] Has touch targets smaller than 48dp
- [ ] Uses hardcoded colors, fonts, or dimensions instead of design tokens
- [ ] Lacks dark theme support
- [ ] Has text fields without autocomplete/autofill where applicable
- [ ] Forces free-text input when a selection component would work
- [ ] Requires the user to type information the system already knows
- [ ] Uses icons without labels for critical actions
- [ ] Has invisible or low-contrast focus indicators
- [ ] Lacks state feedback (no ripple, no state layer changes)
- [ ] Has layout jumps or janky transitions
- [ ] Uses custom widgets when a Material component exists
- [ ] **Fixes UI bugs without adding visual/interaction regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips accessibility verification for any delivered screen**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new UI code.**

### TDD Cycle

```
1. RED: Write a failing UI test (accessibility, interaction, visual)
   ↓
2. GREEN: Write minimal UI code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Material Design

```kotlin
// Step 1: RED - Write failing test first
@Test
fun searchBar_showsAutocomplete_afterTyping() {
    composeTestRule.setContent {
        SearchScreen(suggestions = listOf("Apple", "Apricot", "Avocado"))
    }
    composeTestRule.onNodeWithTag("search_input").performTextInput("Ap")
    composeTestRule.onNodeWithText("Apple").assertIsDisplayed()
    composeTestRule.onNodeWithText("Apricot").assertIsDisplayed()
    composeTestRule.onNodeWithText("Avocado").assertDoesNotExist()
}
// Run: ./gradlew connectedAndroidTest
// FAILS — SearchScreen not implemented

// Step 2: GREEN - Write minimal implementation
@Composable
fun SearchScreen(suggestions: List<String>) {
    var query by rememberSaveable { mutableStateOf("") }
    var active by rememberSaveable { mutableStateOf(false) }
    val filtered = suggestions.filter { it.startsWith(query, ignoreCase = true) }

    SearchBar(
        query = query,
        onQueryChange = { query = it },
        onSearch = { active = false },
        active = active,
        onActiveChange = { active = it },
        modifier = Modifier.testTag("search_input"),
        placeholder = { Text("Search...") },
        leadingIcon = { Icon(Icons.Default.Search, contentDescription = "Search") }
    ) {
        filtered.forEach { item ->
            ListItem(
                headlineContent = { Text(item) },
                modifier = Modifier.clickable { query = item; active = false }
            )
        }
    }
}
// Run: ./gradlew connectedAndroidTest
// PASSES

// Step 3: REFACTOR - Add recent searches, improve UX
// Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every UI bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. UI Bug Reported/Discovered
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

```kotlin
// Bug Report #472: Autocomplete dropdown overlaps keyboard on small screens

// Step 1-2: Write test that reproduces the bug
@Test
fun autocomplete_doesNotOverlapKeyboard_onSmallScreen() {
    composeTestRule.setContent {
        Surface(modifier = Modifier.height(400.dp)) { // Simulate small screen
            AddressField()
        }
    }
    composeTestRule.onNodeWithTag("address_field").performClick()
    // Dropdown should be visible above keyboard
    composeTestRule.onNodeWithTag("suggestions_dropdown").assertIsDisplayed()
    // Dropdown should not extend below the visible area
    val dropdownBounds = composeTestRule.onNodeWithTag("suggestions_dropdown").fetchSemanticsNode().boundsInRoot
    assert(dropdownBounds.bottom <= 400f) { "Dropdown extends below visible area" }
}
// Run: FAILS — dropdown overflows

// Step 3: Fix the bug
// Use ExposedDropdownMenuBox which handles positioning automatically
// + add maxHeight constraint to dropdown menu

// Run: PASSES — bug fixed, regression prevented
```

---

## 3. Component Selection & Usage (MANDATORY)

### A. Material 3 Component Categories

**Use the correct component for each interaction pattern:**

| Category | Components | When to Use |
|----------|-----------|-------------|
| **Action** | FAB, Extended FAB, Icon Button, Filled/Outlined/Text Button, Segmented Button | User triggers an operation |
| **Containment** | Card, Dialog, Bottom Sheet, Side Sheet, Carousel, Tooltip | Group related content or surface contextual info |
| **Navigation** | Navigation Bar, Navigation Rail, Navigation Drawer, Tabs, Top App Bar, Bottom App Bar | Move between views or sections |
| **Selection** | Checkbox, Radio Button, Switch, Chip, Slider, Date Picker, Time Picker | User chooses from options |
| **Text Input** | Filled Text Field, Outlined Text Field, Search Bar | User enters or searches text |
| **Communication** | Badge, Progress Indicator, Snackbar | System communicates status to user |

### B. Component Selection Rules (MANDATORY)

**ALWAYS prefer the least-effort component for the user:**

```
DECISION TREE: What input does the user need to provide?

1. Binary choice? → Switch (not checkbox for settings, checkbox for forms)
2. One of 2-5 options? → Segmented Button or Chip group
3. One of 5-15 options? → Exposed Dropdown Menu (with search/filter)
4. One of 15+ options? → Search Bar with autocomplete
5. Date? → Date Picker (NEVER free text for dates)
6. Time? → Time Picker (NEVER free text for times)
7. Numeric range? → Slider (with optional text input override)
8. Short free text? → Text Field with autocomplete + autofill hints
9. Long free text? → Multi-line Text Field with character count
```

**NEVER use a text field when a selection component works.** Free text is the most error-prone and effortful input method.

### C. Adaptive Navigation (MANDATORY)

**Navigation MUST adapt to window size class:**

```
Window Size Class    │ Navigation Component       │ Example Devices
─────────────────────┼───────────────────────────────┼──────────────────
Compact (<600dp)     │ Navigation Bar (bottom)     │ Phones
Medium (600-839dp)   │ Navigation Rail (side)      │ Tablets, foldables
Expanded (840dp+)    │ Navigation Drawer (side)    │ Desktops, large tablets
```

```kotlin
@Composable
fun AdaptiveNavigation(windowSizeClass: WindowSizeClass, content: @Composable () -> Unit) {
    when (windowSizeClass.widthSizeClass) {
        WindowWidthSizeClass.Compact -> {
            Scaffold(bottomBar = { NavigationBar { /* destinations */ } }) { content() }
        }
        WindowWidthSizeClass.Medium -> {
            Row {
                NavigationRail { /* destinations */ }
                content()
            }
        }
        WindowWidthSizeClass.Expanded -> {
            PermanentNavigationDrawer(drawerContent = { /* destinations */ }) { content() }
        }
    }
}
```

---

## 4. Autocomplete & Smart Input (MANDATORY)

### A. Autocomplete Everywhere

**CRITICAL: Every text field MUST have autocomplete, autofill, or smart defaults unless truly free-form creative input.**

#### When to Use Autocomplete

| Input Type | Autocomplete Strategy |
|------------|----------------------|
| Email | Autofill from device + recent emails |
| Name | Autofill from device contacts/profile |
| Address | Places API autocomplete + autofill |
| Phone | Autofill from device + country code auto-detect |
| Search | Recent searches + trending + category suggestions |
| Tags/Labels | Existing tags + fuzzy match |
| City/Country | Filtered list with fuzzy matching |
| Product/Item | Search with thumbnails, prices, availability |

#### Autocomplete UX Rules

1. **Show suggestions immediately on focus** — do not wait for keystrokes. Surface recent/frequent values.
2. **Filter with each keystroke** — suggestions update instantly as the user types.
3. **Highlight the matching portion** — bold the substring that matches the user's input.
4. **Support keyboard navigation** — Up/Down arrows navigate, Enter selects, Escape dismisses.
5. **Tap-ahead / query refinement** — allow users to append suggestions without submitting (progressive query building).
6. **Show diverse content** — include icons, thumbnails, secondary text, categories alongside suggestions.
7. **Limit visible suggestions to 5-7** — scrollable for more, but don't overwhelm.
8. **Handle empty/no-match state** — show "No results" with alternative actions, never a blank dropdown.

#### Implementation Example

```kotlin
@Composable
fun SmartAddressField(
    recentAddresses: List<Address>,
    onAddressSelected: (Address) -> Unit
) {
    var text by rememberSaveable { mutableStateOf("") }
    var suggestions by remember { mutableStateOf(recentAddresses) }
    var expanded by remember { mutableStateOf(false) }

    // Show recent addresses immediately on focus
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
        OutlinedTextField(
            value = text,
            onValueChange = { query ->
                text = query
                expanded = true
                suggestions = if (query.isBlank()) {
                    recentAddresses  // Show recent on empty input
                } else {
                    // Fetch from Places API + filter recent
                    fetchPlacesSuggestions(query) + recentAddresses.filter {
                        it.formatted.contains(query, ignoreCase = true)
                    }
                }
            },
            label = { Text("Address") },
            modifier = Modifier.menuAnchor().fillMaxWidth(),
            leadingIcon = { Icon(Icons.Default.LocationOn, contentDescription = null) },
            supportingText = { Text("Start typing or select a recent address") },
            keyboardOptions = KeyboardOptions(
                keyboardType = KeyboardType.Text,
                imeAction = ImeAction.Done,
                // Autofill hint for platform-level autocomplete
                autoCorrect = true
            )
        )
        ExposedDropdownMenu(expanded = expanded && suggestions.isNotEmpty(), onDismissRequest = { expanded = false }) {
            suggestions.take(7).forEach { address ->
                DropdownMenuItem(
                    text = {
                        ListItem(
                            headlineContent = { HighlightedText(address.primary, text) },
                            supportingContent = { Text(address.secondary, style = MaterialTheme.typography.bodySmall) },
                            leadingContent = {
                                Icon(
                                    if (address.isRecent) Icons.Default.History else Icons.Default.Place,
                                    contentDescription = null
                                )
                            }
                        )
                    },
                    onClick = {
                        text = address.formatted
                        expanded = false
                        onAddressSelected(address)
                    }
                )
            }
        }
    }
}
```

### B. Autofill Hints (MANDATORY)

**Set autofill hints on EVERY applicable field. The platform can auto-fill entire forms when hints are correct.**

```kotlin
// Android / Compose
OutlinedTextField(
    modifier = Modifier.semantics {
        // Tell the system what this field contains
        contentDescription = "Email address"
    }.autofill(
        autofillTypes = listOf(AutofillType.EmailAddress),
        onFill = { email = it }
    ),
    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email)
)
```

```html
<!-- Web / HTML -->
<md-outlined-text-field
  label="Email"
  type="email"
  autocomplete="email"
  inputmode="email"
  name="email">
</md-outlined-text-field>

<md-outlined-text-field
  label="Street Address"
  autocomplete="street-address"
  name="address">
</md-outlined-text-field>

<md-outlined-text-field
  label="Phone"
  type="tel"
  autocomplete="tel"
  inputmode="tel"
  name="phone">
</md-outlined-text-field>
```

**Common autofill/autocomplete values:**

| Field | HTML `autocomplete` | Android `AutofillType` | iOS `textContentType` |
|-------|-------------------|----------------------|---------------------|
| Full Name | `name` | `PersonName` | `.name` |
| Email | `email` | `EmailAddress` | `.emailAddress` |
| Phone | `tel` | `PhoneNumber` | `.telephoneNumber` |
| Street | `street-address` | `PostalAddress` | `.streetAddressLine1` |
| City | `address-level2` | `PostalAddress` | `.addressCity` |
| Zip/Postal | `postal-code` | `PostalCode` | `.postalCode` |
| Country | `country-name` | `PostalAddress` | `.countryName` |
| Credit Card | `cc-number` | `CreditCardNumber` | `.creditCardNumber` |
| Password | `current-password` | `Password` | `.password` |
| New Password | `new-password` | `NewPassword` | `.newPassword` |
| OTP | `one-time-code` | `SmsOtpCode` | `.oneTimeCode` |

### C. Smart Defaults (MANDATORY)

**Pre-fill fields with the most likely value whenever possible:**

```kotlin
// Pre-fill country from device locale
val defaultCountry = Locale.getDefault().displayCountry

// Pre-fill currency from locale
val defaultCurrency = Currency.getInstance(Locale.getDefault()).currencyCode

// Pre-fill date to today for "start date" fields
val defaultDate = LocalDate.now()

// Pre-fill time zone from device
val defaultTimeZone = TimeZone.currentSystemDefault()

// Pre-fill language from device
val defaultLanguage = Locale.getDefault().displayLanguage
```

**Rules for smart defaults:**
- Currency, language, country → from device locale
- Date → today (or next business day for business contexts)
- Quantity → 1 (most common selection)
- Toggles → the most commonly selected option (based on analytics)
- Returning users → last used values

---

## 5. Theming & Design Tokens (MANDATORY)

### A. Material You Dynamic Color

**Use dynamic color as the primary theming strategy. Fall back to branded theme on unsupported platforms.**

```kotlin
@Composable
fun AppTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> darkColorScheme(
            primary = Color(0xFFD0BCFF),
            secondary = Color(0xFFCCC2DC),
            tertiary = Color(0xFFEFB8C8)
        )
        else -> lightColorScheme(
            primary = Color(0xFF6750A4),
            secondary = Color(0xFF625B71),
            tertiary = Color(0xFF7D5260)
        )
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = AppTypography,
        shapes = AppShapes,
        content = content
    )
}
```

### B. Design Token Rules (MANDATORY)

**NEVER use raw values. ALWAYS use tokens.**

```kotlin
// CORRECT — Uses Material tokens
Text(
    text = "Hello",
    style = MaterialTheme.typography.headlineMedium,
    color = MaterialTheme.colorScheme.onSurface
)
Surface(
    color = MaterialTheme.colorScheme.surfaceContainerLow,
    tonalElevation = 1.dp
) { /* content */ }

// WRONG — Hardcoded values
Text(
    text = "Hello",
    fontSize = 28.sp,           // ✗ Use typography token
    color = Color(0xFF1C1B1F)   // ✗ Use colorScheme token
)
Surface(
    color = Color(0xFFF3EDF7),  // ✗ Use colorScheme token
    shadowElevation = 4.dp      // ✗ Use tonalElevation
) { /* content */ }
```

### C. Color Roles (MANDATORY)

**Use the correct M3 color role for each surface:**

| Surface | Background | On-Background (text/icon) |
|---------|-----------|--------------------------|
| App background | `surface` | `onSurface` |
| Primary action | `primary` | `onPrimary` |
| Secondary action | `secondary` | `onSecondary` |
| Primary container | `primaryContainer` | `onPrimaryContainer` |
| Error state | `error` | `onError` |
| Error container | `errorContainer` | `onErrorContainer` |
| Surface variant | `surfaceVariant` | `onSurfaceVariant` |
| Elevated surface | `surfaceContainerHigh` | `onSurface` |

**The HCT (Hue, Chroma, Tone) color space ensures perceptually accurate contrast.** Material's tonal palette system guarantees accessible contrast ratios automatically when you use the correct on-* color on its corresponding surface.

### D. Typography Scale (MANDATORY)

**Use the M3 type scale — do not invent custom sizes:**

| Role | Default Size | Weight | Use Case |
|------|-------------|--------|----------|
| `displayLarge` | 57sp | 400 | Hero numbers, splash |
| `displayMedium` | 45sp | 400 | Large headlines |
| `displaySmall` | 36sp | 400 | Section headers (large) |
| `headlineLarge` | 32sp | 400 | Page titles |
| `headlineMedium` | 28sp | 400 | Section titles |
| `headlineSmall` | 24sp | 400 | Card titles |
| `titleLarge` | 22sp | 400 | Top app bar, dialog titles |
| `titleMedium` | 16sp | 500 | List item headlines |
| `titleSmall` | 14sp | 500 | Tab labels |
| `bodyLarge` | 16sp | 400 | Primary body text |
| `bodyMedium` | 14sp | 400 | Secondary body text |
| `bodySmall` | 12sp | 400 | Captions, timestamps |
| `labelLarge` | 14sp | 500 | Button text |
| `labelMedium` | 12sp | 500 | Navigation labels |
| `labelSmall` | 11sp | 500 | Badges, chips |

---

## 6. Interaction States & Feedback (MANDATORY)

### A. State Layer System

**Every interactive component MUST display all applicable states:**

```
State        │ Opacity over surface │ Trigger
─────────────┼──────────────────────┼─────────────────────────────
Enabled      │ 0%                   │ Default
Disabled     │ Component at 38%     │ Not interactive
Hovered      │ 8% of content color  │ Cursor over element
Focused      │ 10% of content color │ Keyboard/a11y focus + focus ring
Pressed      │ 10% of content color │ Touch/click (ripple animation)
Dragged      │ 16% of content color │ Element being moved
```

```kotlin
// Material 3 handles states automatically — use Material components
// CORRECT: Use the built-in ripple and state handling
Button(onClick = { /* action */ }) {
    Text("Submit")
}

// WRONG: Custom clickable without indication
Box(modifier = Modifier.clickable(indication = null, interactionSource = remember { MutableInteractionSource() }) { })
// ✗ No ripple feedback — user gets no confirmation of their tap
```

### B. Motion & Transitions (MANDATORY)

**Motion must be purposeful — it guides, confirms, and orients.**

Material 3 motion principles:
1. **Informative**: Motion shows spatial relationships and hierarchy
2. **Focused**: Motion draws attention to what matters
3. **Expressive**: Motion adds personality while remaining functional

```kotlin
// Use Material motion tokens — never arbitrary durations
// Duration tokens:
// motionDurationShort1: 75ms   — subtle feedback (ripple)
// motionDurationShort2: 150ms  — small element changes
// motionDurationMedium1: 200ms — medium element transitions
// motionDurationMedium2: 250ms — container changes
// motionDurationLong1: 300ms   — page transitions
// motionDurationLong2: 350ms   — complex transitions

// Easing tokens:
// EmphasizedDecelerate — entering elements
// EmphasizedAccelerate — exiting elements
// Standard — persistent elements changing state

// Container Transform — shared element transitions
val transform = MaterialContainerTransform().apply {
    startView = listItem
    endView = detailScreen
    scrimColor = Color.TRANSPARENT
    // M3 Expressive: spring-based physics for natural feel
}

// Compose animations with Material specs
val animatedPadding by animateDpAsState(
    targetValue = if (expanded) 16.dp else 0.dp,
    animationSpec = tween(
        durationMillis = 250,  // motionDurationMedium2
        easing = FastOutSlowInEasing  // Standard easing
    )
)
```

### C. Touch Target Requirements (MANDATORY)

**Minimum touch target: 48dp x 48dp (approximately 9mm physical).**

```kotlin
// CORRECT — Icon button meets 48dp minimum
IconButton(onClick = { /* action */ }) {  // 48dp touch target built-in
    Icon(Icons.Default.Favorite, contentDescription = "Favorite")
}

// CORRECT — Small visual element with adequate touch target
Icon(
    Icons.Default.Close,
    contentDescription = "Close",
    modifier = Modifier
        .size(24.dp)                    // Visual size: 24dp
        .minimumInteractiveComponentSize()  // Touch target: 48dp
        .clickable { /* action */ }
)

// WRONG — Touch target too small
Icon(
    Icons.Default.Close,
    contentDescription = "Close",
    modifier = Modifier
        .size(24.dp)
        .clickable { /* action */ }  // ✗ Touch target is only 24dp
)
```

---

## 7. Forms & Data Entry (MANDATORY)

### A. Form Design Principles

**Every form MUST minimize user effort. Follow these rules in order:**

1. **Eliminate**: Remove any field that can be derived, inferred, or fetched
2. **Automate**: Pre-fill fields from user profile, device, API, or context
3. **Select**: Convert free-text fields to selection components when options are finite
4. **Assist**: Add autocomplete, inline validation, and input masks to remaining fields
5. **Order**: Place fields in the order users think about them (name → email → phone)

### B. Text Field Configuration (MANDATORY)

**Use Outlined text fields as the default. Use Filled only in dense, contained surfaces (cards, sheets).**

```kotlin
@Composable
fun MaterialTextField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    supportingText: String? = null,
    error: String? = null,
    leadingIcon: @Composable (() -> Unit)? = null,
    trailingIcon: @Composable (() -> Unit)? = null,
    keyboardType: KeyboardType = KeyboardType.Text,
    imeAction: ImeAction = ImeAction.Next,
    maxLength: Int? = null
) {
    OutlinedTextField(
        value = value,
        onValueChange = { if (maxLength == null || it.length <= maxLength) onValueChange(it) },
        label = { Text(label) },
        supportingText = {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(error ?: supportingText ?: "")
                if (maxLength != null) Text("${value.length}/$maxLength")
            }
        },
        isError = error != null,
        leadingIcon = leadingIcon,
        trailingIcon = trailingIcon ?: if (value.isNotEmpty()) {
            { IconButton(onClick = { onValueChange("") }) {
                Icon(Icons.Default.Clear, contentDescription = "Clear")
            }}
        } else null,
        keyboardOptions = KeyboardOptions(
            keyboardType = keyboardType,
            imeAction = imeAction
        ),
        singleLine = true,
        modifier = Modifier.fillMaxWidth()
    )
}
```

### C. Inline Validation (MANDATORY)

**Validate as the user types. NEVER wait until form submission to show errors.**

```kotlin
// CORRECT — Inline validation with debounce
@Composable
fun EmailField(onValidEmail: (String) -> Unit) {
    var email by rememberSaveable { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }

    // Validate after user stops typing (300ms debounce)
    LaunchedEffect(email) {
        delay(300)
        error = when {
            email.isEmpty() -> null  // Don't show error on empty (not yet touched)
            !email.contains("@") -> "Enter a valid email address"
            !email.contains(".") -> "Enter a valid email address"
            else -> { onValidEmail(email); null }
        }
    }

    OutlinedTextField(
        value = email,
        onValueChange = { email = it },
        label = { Text("Email") },
        isError = error != null,
        supportingText = error?.let { { Text(it) } },
        keyboardOptions = KeyboardOptions(
            keyboardType = KeyboardType.Email,
            imeAction = ImeAction.Next
        ),
        leadingIcon = { Icon(Icons.Default.Email, contentDescription = null) },
        modifier = Modifier.fillMaxWidth()
    )
}

// WRONG — Validation only on submit
// User fills 10 fields, clicks submit, sees 3 errors, scrolls back up to fix them
// This is hostile UX.
```

### D. Input Masks & Formatting (MANDATORY)

**Format inputs automatically. Never make the user worry about formatting.**

```kotlin
// Phone number: Auto-format as user types
// "1234567890" → "(123) 456-7890"

// Credit card: Auto-format with spaces
// "4111111111111111" → "4111 1111 1111 1111"

// Date: Use DatePicker instead of text input (MANDATORY)
// NEVER: TextField with placeholder "MM/DD/YYYY"
// ALWAYS: DatePicker component
```

---

## 8. Accessibility (MANDATORY)

### A. WCAG 2.1 AA Baseline

**Every delivered screen MUST meet WCAG 2.1 AA as a minimum:**

| Criterion | Requirement |
|-----------|-------------|
| **1.1.1 Non-text Content** | All images, icons, and graphics have text alternatives |
| **1.3.1 Info and Relationships** | Heading hierarchy is logical (h1 → h2 → h3) |
| **1.4.3 Contrast (Minimum)** | 4.5:1 for normal text, 3:1 for large text (18sp+ or 14sp bold) |
| **1.4.11 Non-text Contrast** | 3:1 for UI components and graphical objects |
| **2.1.1 Keyboard** | All functionality available via keyboard |
| **2.4.3 Focus Order** | Focus order is logical and predictable |
| **2.4.7 Focus Visible** | Focus indicator is always visible |
| **2.5.5 Target Size** | Touch targets >= 48dp x 48dp |
| **3.3.1 Error Identification** | Errors are clearly identified and described in text |
| **3.3.2 Labels or Instructions** | Input fields have visible labels (not just placeholders) |
| **4.1.2 Name, Role, Value** | All components have accessible names and roles |

### B. Screen Reader Support

```kotlin
// CORRECT — Semantic meaning for screen readers
Icon(
    Icons.Default.ShoppingCart,
    contentDescription = "Shopping cart, 3 items"  // Descriptive
)

// CORRECT — Merge semantics for compound elements
Row(modifier = Modifier.semantics(mergeDescendants = true) { }) {
    Icon(Icons.Default.Star, contentDescription = null)  // Decorative
    Text("4.5 out of 5 stars, 128 reviews")
}

// CORRECT — Custom actions for complex interactions
Box(modifier = Modifier.semantics {
    stateDescription = if (isExpanded) "Expanded" else "Collapsed"
    customActions = listOf(
        CustomAccessibilityAction("Toggle details") { toggle(); true }
    )
})

// WRONG — No content description
Icon(Icons.Default.Delete, contentDescription = null)  // ✗ on interactive icon
// WRONG — Redundant description
Icon(Icons.Default.Home, contentDescription = "Home icon")  // ✗ say what it does, not what it looks like
```

### C. Color & Contrast

**NEVER use color as the only means of conveying information.**

```kotlin
// CORRECT — Error indicated by color + icon + text
OutlinedTextField(
    isError = true,
    supportingText = { Text("Password must be at least 8 characters") },  // Text explanation
    trailingIcon = { Icon(Icons.Default.Error, contentDescription = "Error") },  // Icon indicator
    // The red color is an additional (not sole) indicator
)

// WRONG — Only color indicates error
OutlinedTextField(
    colors = TextFieldDefaults.colors(
        focusedIndicatorColor = Color.Red  // ✗ Colorblind users can't see this
    )
    // No error text, no icon — only the color changed
)
```

### D. Text Resizing

**Support text resizing up to 200% without loss of content or function.**

```kotlin
// Use sp for text sizes (scales with user preference)
// Use dp for layout dimensions (does not scale)
// NEVER set fixed height on text containers — let them grow
// NEVER use clipToBounds on text containers
```

---

## 9. Responsive Layout (MANDATORY)

### A. Window Size Classes

**Design for three canonical breakpoints:**

```
Compact    │ < 600dp  │ Single pane, stacked layout, bottom nav
Medium     │ 600-839dp│ Two pane optional, nav rail, flexible grid
Expanded   │ ≥ 840dp  │ Two pane, nav drawer, multi-column grid
```

### B. Canonical Layouts

**Use Material canonical layouts — do not invent custom responsive behavior:**

```kotlin
// List-Detail layout (email, messaging, settings)
@Composable
fun ListDetailLayout(windowSizeClass: WindowSizeClass) {
    when (windowSizeClass.widthSizeClass) {
        WindowWidthSizeClass.Compact -> {
            // Single pane: list OR detail (navigate between)
            NavHost(navController, startDestination = "list") {
                composable("list") { ListPane(onItemClick = { navController.navigate("detail/$it") }) }
                composable("detail/{id}") { DetailPane(it.arguments?.getString("id")) }
            }
        }
        else -> {
            // Two pane: list AND detail side by side
            Row {
                ListPane(modifier = Modifier.weight(0.4f), onItemClick = { selectedId = it })
                DetailPane(modifier = Modifier.weight(0.6f), id = selectedId)
            }
        }
    }
}
```

### C. Grid System

```kotlin
// Material uses a 4/8/12 column grid
// Compact: 4 columns, 16dp margins, 8dp gutters
// Medium: 8 columns, 24dp margins, 16dp gutters  (body region uses 12 columns)
// Expanded: 12 columns, 24dp margins, 16dp gutters

// Use LazyVerticalGrid or FlowRow for adaptive layouts
LazyVerticalGrid(
    columns = GridCells.Adaptive(minSize = 160.dp),  // Automatically adapts to width
    contentPadding = PaddingValues(16.dp),
    horizontalArrangement = Arrangement.spacedBy(8.dp),
    verticalArrangement = Arrangement.spacedBy(8.dp)
) {
    items(data) { item -> CardItem(item) }
}
```

---

## 10. Error Handling & User Communication (MANDATORY)

### A. Error Communication Hierarchy

| Severity | Component | When to Use |
|----------|-----------|-------------|
| **Inline** | Text field error + supporting text | Field-level validation errors |
| **Banner** | Snackbar (no action needed) | Transient, non-blocking errors |
| **Banner** | Snackbar + Action | Recoverable errors (e.g., "Undo") |
| **Blocking** | Dialog | Destructive/critical actions requiring confirmation |
| **Page-level** | Full-screen error state | Network failure, 404, empty states |

### B. Error State Design

```kotlin
// CORRECT — Helpful, specific error messages
"Enter a valid email address (e.g., name@example.com)"
"Password must be at least 8 characters with one number"
"Card number must be 16 digits"

// WRONG — Vague, unhelpful error messages
"Invalid input"        // ✗ What's invalid? How to fix?
"Error occurred"       // ✗ What error? What now?
"Please try again"     // ✗ Try what again? Will it work?
```

### C. Empty States

**Every list/grid MUST have a meaningful empty state:**

```kotlin
@Composable
fun EmptyState(
    icon: ImageVector,
    title: String,
    description: String,
    actionLabel: String? = null,
    onAction: (() -> Unit)? = null
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(icon, contentDescription = null, modifier = Modifier.size(96.dp),
             tint = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(16.dp))
        Text(title, style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))
        Text(description, style = MaterialTheme.typography.bodyMedium,
             color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = TextAlign.Center)
        if (actionLabel != null && onAction != null) {
            Spacer(Modifier.height(24.dp))
            FilledTonalButton(onClick = onAction) { Text(actionLabel) }
        }
    }
}
```

---

## 11. Performance & Loading States (MANDATORY)

### A. Progress Indicators

**Always show progress for operations > 300ms:**

| Duration | Indicator | Example |
|----------|-----------|---------|
| < 300ms | None (feels instant) | Toggling a switch |
| 300ms–1.5s | Indeterminate (linear or circular) | Saving a form |
| > 1.5s | Determinate (with percentage) | File upload |
| Unknown, long | Skeleton screens | Page load |

```kotlin
// Skeleton loading — show placeholder shapes instead of spinners
@Composable
fun CardSkeleton() {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp)) {
            Box(Modifier.fillMaxWidth().height(200.dp).shimmer()
                .background(MaterialTheme.colorScheme.surfaceContainerHighest, RoundedCornerShape(12.dp)))
            Spacer(Modifier.height(12.dp))
            Box(Modifier.fillMaxWidth(0.7f).height(24.dp).shimmer()
                .background(MaterialTheme.colorScheme.surfaceContainerHighest, RoundedCornerShape(4.dp)))
            Spacer(Modifier.height(8.dp))
            Box(Modifier.fillMaxWidth(0.5f).height(16.dp).shimmer()
                .background(MaterialTheme.colorScheme.surfaceContainerHighest, RoundedCornerShape(4.dp)))
        }
    }
}
```

### B. Optimistic Updates

**For fast, reliable operations, update the UI immediately and sync in background:**

```kotlin
// CORRECT — Optimistic: UI updates instantly, syncs in background
fun onFavoriteClick(item: Item) {
    item.isFavorite = !item.isFavorite  // Instant UI feedback
    scope.launch {
        try { repository.toggleFavorite(item.id) }
        catch (e: Exception) {
            item.isFavorite = !item.isFavorite  // Revert on failure
            snackbarHostState.showSnackbar("Couldn't update. Check your connection.")
        }
    }
}
```

---

## 12. Deployment Checklist

### Agent-Generated Interface Verification (MANDATORY)

**If UI code was generated/modified by an agent, verify BEFORE delivery:**

#### Accessibility
- [ ] WCAG 2.1 AA compliance verified
- [ ] All touch targets >= 48dp
- [ ] Color contrast ratios verified (4.5:1 / 3:1)
- [ ] Screen reader navigation tested
- [ ] Keyboard navigation complete
- [ ] Focus indicators visible
- [ ] No color-only information conveying

#### Theming
- [ ] Zero hardcoded colors, fonts, or dimensions
- [ ] Dark theme renders correctly
- [ ] Dynamic color supported (where platform allows)
- [ ] All colors use correct M3 color roles
- [ ] Typography uses M3 type scale

#### Interaction
- [ ] All components show proper states (hover, focus, pressed, disabled)
- [ ] Ripple feedback on all tappable surfaces
- [ ] Transitions are smooth, no layout jumps
- [ ] Motion uses Material duration/easing tokens

#### Autocomplete & Input
- [ ] All applicable fields have autocomplete
- [ ] Autofill hints set on all relevant fields
- [ ] Smart defaults populated where possible
- [ ] Selection components used instead of free text where applicable
- [ ] Inline validation on all fields (not on-submit only)
- [ ] Clear buttons on text fields with content
- [ ] Appropriate keyboard types set (email, number, phone, etc.)

#### Responsive
- [ ] Layout adapts to compact / medium / expanded
- [ ] Navigation adapts (bar → rail → drawer)
- [ ] No horizontal scroll on any viewport
- [ ] Text resizable to 200% without content loss

#### Testing
- [ ] UI tests written BEFORE implementation (TDD)
- [ ] Accessibility tests pass
- [ ] Visual regression tests pass
- [ ] Interaction tests pass
- [ ] All edge cases tested (empty states, error states, loading states)

#### Agent Workflow Completed
- [ ] Agent verified all accessibility checks pass
- [ ] Agent ran all UI tests and verified they pass
- [ ] Agent verified theme consistency
- [ ] Agent documented any fixes made during verification

---

## 13. Why This Configuration Works

**Human Effort Minimization**:
- Autocomplete, autofill, and smart defaults reduce keystrokes by 60-80%. Users complete forms faster and with fewer errors. The zero-input ideal means every field earns its place.

**Accessibility as Foundation**:
- Building accessible interfaces from the start using M3's HCT color system and component library prevents costly retrofits. What works for screen reader users works better for everyone.

**Systematic Theming**:
- Design tokens create a single source of truth shared between design and code. Dynamic color (Material You) makes every app feel personal without custom design work per user.

**Predictive Intelligence**:
- Surfacing recent values, suggesting completions, and providing tap-ahead patterns keep users in flow. The interface anticipates rather than interrogates.

**Responsive by Default**:
- Window size classes and canonical layouts ensure interfaces work on phones, tablets, foldables, and desktops from a single codebase. Navigation adapts automatically.

---

## 14. Quick Reference

### Component Decision Matrix

```
User needs to...          → Use this component
──────────────────────────────────────────────────
Trigger primary action    → FAB or Extended FAB
Trigger secondary action  → Filled/Tonal Button
Navigate between views    → Navigation Bar / Rail / Drawer (adaptive)
Choose one of few options → Segmented Button (2-5) or Radio
Choose multiple options   → Checkbox group or Filter Chips
Enter a date              → Date Picker (NEVER text field)
Enter a time              → Time Picker (NEVER text field)
Search content            → Search Bar with autocomplete
Enter text (short)        → Outlined Text Field + autocomplete
Enter text (long)         → Multi-line Outlined Text Field
Show status               → Badge, Snackbar, or Progress Indicator
Confirm destructive act   → Dialog with explicit action labels
Show related content      → Card (Filled, Outlined, or Elevated)
Progressive disclosure    → Bottom Sheet or Side Sheet
```

### Spacing Quick Reference

```
4dp  — Minimum spacing between related elements
8dp  — Standard spacing within components
12dp — Spacing between chips, small elements
16dp — Standard padding inside containers
24dp — Section spacing, screen margins (medium+)
32dp — Large section spacing
```

### Common Material 3 Patterns

```bash
# Build (Android/Compose)
./gradlew assembleDebug

# Test
./gradlew testDebugUnitTest
./gradlew connectedDebugAndroidTest

# Lint / Accessibility Check
./gradlew lint

# Format
./gradlew spotlessApply

# Build (Web)
npm run build

# Test (Web)
npm test

# Accessibility Audit (Web)
npx axe-core --exit
```

### Build Automation Template

```makefile
.PHONY: all test lint format audit

all: format lint test audit

test:
	./gradlew testDebugUnitTest connectedDebugAndroidTest

lint:
	./gradlew lint

format:
	./gradlew spotlessApply

audit:
	./gradlew lint --check "Accessibility"
	@echo "✓ Accessibility audit passed"

verify: format lint test audit
	@echo "✓ All checks passed — safe to deliver"
```

---

**End of Material Design 3 Interface Guidelines**

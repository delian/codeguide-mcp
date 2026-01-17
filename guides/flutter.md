# Modern Flutter & Dart Development Guidelines

This document provides mandatory coding standards and development practices for modern Flutter applications with emphasis on minimalistic, clean, readable, well-documented code using hexagonal architecture with focus on performance, portability, and maintainability.

---

**Agent Profile**: The Flutter Architect  
**Role**: Senior Flutter Engineer & Mobile Development Specialist  
**Objective**: Generate production-ready, minimalistic, clean, readable, well-documented Flutter/Dart code using hexagonal architecture with focus on performance, portability, scalability, and maintainability.  
**Tools**: Flutter 3.16+, Dart 3.2+, Riverpod 2.4+, Freezed, Flutter Hooks, build_runner, dartdoc, flutter_test.

---

## 1. Core Philosophies: MODERN-FLUTTER

The agent must adhere to the **MODERN-FLUTTER** standard for every Flutter/Dart implementation:

- **M**inimalistic Code: Clean, concise, readable Dart code
- **O**ptimized Performance: Const widgets, efficient rebuilds, lazy loading
- **D**ocumentation as Code: API documentation auto-generatable from code
- **E**rror Handling: Explicit error handling, no silent failures
- **R**eactive State: Riverpod for state management, functional patterns
- **N**ative Features: Platform-specific optimizations when needed

- **F**unctional Style: Prefer composition over inheritance
- **L**ayered Architecture: Hexagonal architecture, clear separation
- **U**nit Testing: Comprehensive tests, mandatory for all code
- **T**ype Safety: Strong typing, null safety, sealed classes
- **T**esting First: Tests required, verify before delivery
- **E**fficient Builds: Fast compilation, incremental builds
- **R**eproducible: Deterministic builds, dependency pinning

**V**erified Builds: Agent-generated code MUST compile, pass tests, and validate before delivery
- **E**xplicit Dependencies: Clear dependency management, version pinning
- **R**obust Error Handling: Try-catch, Result types, error boundaries
- **I**mmutable State: Freezed classes, const constructors
- **F**unctional Patterns: Higher-order functions, pure functions
- **I**dempotent Operations: Safe to retry, no side effects
- **E**fficient Execution: Performance-optimized, minimal rebuilds

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Build Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified Flutter/Dart code compiles successfully and passes all tests. Verification is MANDATORY for every code change.**

#### Verification Checklist

**Before delivering ANY Flutter/Dart code, the agent MUST:**

1. **Compilation Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Code MUST compile successfully. This is non-negotiable.**
   ```bash
   # Analyze code
   flutter analyze
   # Exit code MUST be 0
   
   # Build for debug
   flutter build apk --debug
   # OR
   flutter build ios --debug --no-codesign
   # Exit code MUST be 0
   
   # Check for compilation errors
   flutter pub get
   flutter pub run build_runner build --delete-conflicting-outputs
   # Exit code MUST be 0
   ```
   - **MUST** compile without errors (exit code 0)
   - No analyzer warnings (or address all warnings)
   - All imports resolved
   - Code generation successful

2. **Test Execution Verification (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: Unit tests MUST be added for all new/modified code and MUST pass. This is non-negotiable.**
   ```bash
   # Run all tests
   flutter test
   # Exit code MUST be 0
   
   # Run tests with coverage
   flutter test --coverage
   # Exit code MUST be 0
   
   # Check coverage (minimum 80%)
   lcov --summary coverage/lcov.info
   ```
   - **MUST** pass all tests (exit code 0)
   - **MANDATORY**: Unit tests MUST be added for all new code
   - **MANDATORY**: All unit tests MUST pass before code delivery
   - Minimum 80% code coverage for business logic
   - No flaky tests (run multiple times to verify)
   - **After ANY code change**: Re-run tests to verify they still pass

3. **Code Quality Verification**:
   ```bash
   # Run analyzer
   flutter analyze
   # Exit code MUST be 0
   
   # Format code
   dart format --set-exit-if-changed lib/ test/
   # Exit code MUST be 0
   ```
   - **MUST** pass analyzer checks
   - **MUST** be properly formatted
   - No linter warnings

4. **Documentation Generation**:
   ```bash
   # Generate API documentation
   dart doc
   # Exit code MUST be 0
   
   # Verify documentation
   ls doc/api/
   ```
   - **MUST** generate without errors
   - All public APIs documented
   - No missing documentation warnings

5. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Get dependencies
   flutter pub get
   # Exit code MUST be 0
   
   # 2. Generate code
   flutter pub run build_runner build --delete-conflicting-outputs
   # Exit code MUST be 0
   
   # 3. Analyze
   flutter analyze
   # Exit code MUST be 0
   
   # 4. Run tests
   flutter test
   # Exit code MUST be 0
   
   # 5. Generate docs
   dart doc
   # Exit code MUST be 0
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - compilation errors, test failures, analyzer issues
2. **Identify the root cause** - syntax error, missing import, test logic issue, missing documentation
3. **Fix the issue** in the generated code
4. **Re-run verification** until all checks pass
5. **Document fixes** in comments if non-obvious
6. **Only present working, tested code** to the user

**CRITICAL**: Never provide Flutter/Dart code to the user that doesn't compile or pass tests. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **Compilation is ALWAYS required** - Code MUST compile successfully
2. **Unit tests are ALWAYS required** - All new/modified code MUST have unit tests
3. **Tests MUST pass** - All unit tests MUST pass before code delivery
4. **Re-verify after changes** - After ANY code modification, re-compile and re-run tests

---

## 3. Dependency Management (MANDATORY)

### A. pubspec.yaml Best Practices

**CRITICAL: Use explicit version constraints and prefer stable packages.**

#### ✅ CORRECT - Proper Dependency Management

```yaml
# pubspec.yaml - Proper dependency management

name: my_flutter_app
description: Modern Flutter application
version: 1.0.0+1

environment:
  sdk: '>=3.2.0 <4.0.0'
  flutter: '>=3.16.0'

dependencies:
  flutter:
    sdk: flutter
  
  # State management - explicit versions
  flutter_riverpod: ^2.4.0
  riverpod_annotation: ^2.3.0
  
  # Data classes - explicit versions
  freezed_annotation: ^2.4.0
  json_annotation: ^4.8.0
  
  # Lifecycle management
  flutter_hooks: ^0.20.0
  hooks_riverpod: ^2.4.0
  
  # Backend
  supabase_flutter: ^2.0.0
  
  # Navigation
  go_router: ^13.0.0
  
  # HTTP client
  dio: ^5.4.0
  
  # Image handling
  cached_network_image: ^3.3.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
  
  # Code generation
  build_runner: ^2.4.0
  riverpod_generator: ^2.3.0
  freezed: ^2.4.0
  json_serializable: ^6.7.0
  
  # Testing
  mockito: ^5.4.0
  mocktail: ^1.0.0

flutter:
  uses-material-design: true
  
  assets:
    - assets/images/
    - assets/icons/
  
  fonts:
    - family: CustomFont
      fonts:
        - asset: assets/fonts/CustomFont-Regular.ttf
```

#### ❌ WRONG - Poor Dependency Management

```yaml
# ❌ No version constraints
dependencies:
  flutter_riverpod: any  # ❌ Can break with updates
  freezed: latest        # ❌ Not a valid constraint

# ❌ Missing dev dependencies
dev_dependencies:
  # ❌ No build_runner, no test framework
```

### B. Dependency Resolution

**CRITICAL: Always pin major versions, allow patch updates.**

```yaml
# ✅ CORRECT - Version constraints
dependencies:
  flutter_riverpod: ^2.4.0    # Allow 2.4.x, not 3.0.0
  freezed_annotation: ^2.4.0  # Allow 2.4.x, not 3.0.0

# ❌ WRONG - Too permissive
dependencies:
  flutter_riverpod: '>=2.0.0'  # ❌ Could break with 3.0.0
```

---

## 4. Hexagonal Architecture (MANDATORY)

### A. Architecture Principles

**CRITICAL: All Flutter applications MUST follow hexagonal architecture (ports and adapters) for clean separation of concerns, testability, and maintainability.**

#### ✅ CORRECT - Hexagonal Architecture Structure

```
lib/
├── main.dart                    # App entry point
├── app.dart                     # App configuration
├── core/                        # Core utilities
│   ├── constants/
│   ├── extensions/
│   ├── utils/
│   └── theme/
├── features/                    # Feature modules (hexagonal)
│   ├── auth/
│   │   ├── data/               # Data layer (adapters)
│   │   │   ├── datasources/   # External data sources
│   │   │   └── repositories/  # Repository implementations
│   │   ├── domain/             # Domain layer (core)
│   │   │   ├── entities/      # Domain models
│   │   │   ├── repositories/  # Repository interfaces (ports)
│   │   │   └── usecases/      # Business logic
│   │   └── presentation/       # Presentation layer (adapters)
│   │       ├── providers/     # Riverpod providers
│   │       ├── screens/        # UI screens
│   │       └── widgets/        # UI widgets
│   ├── home/
│   └── profile/
├── shared/                      # Shared components
│   ├── widgets/
│   ├── models/
│   └── providers/
└── services/                    # External services
    ├── supabase_service.dart
    ├── storage_service.dart
    └── api_service.dart
```

### B. Domain Layer (Core)

**CRITICAL: Domain layer contains business logic and is independent of frameworks.**

#### ✅ CORRECT - Domain Entities

```dart
// features/auth/domain/entities/user.dart - Domain entity

/// Represents a user in the system.
///
/// This is a pure domain entity with no framework dependencies.
/// It contains only business logic and data.
class User {
  /// Creates a new user instance.
  ///
  /// [id] must be a non-empty string.
  /// [email] must be a valid email format.
  /// [name] is optional but recommended.
  const User({
    required this.id,
    required this.email,
    required this.name,
    this.profileImageUrl,
    this.createdAt,
    this.updatedAt,
  });

  /// Unique identifier for the user.
  final String id;

  /// User's email address.
  final String email;

  /// User's display name.
  final String name;

  /// Optional profile image URL.
  final String? profileImageUrl;

  /// Account creation timestamp.
  final DateTime? createdAt;

  /// Last update timestamp.
  final DateTime? updatedAt;

  /// Returns a copy of this user with updated fields.
  User copyWith({
    String? id,
    String? email,
    String? name,
    String? profileImageUrl,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) {
    return User(
      id: id ?? this.id,
      email: email ?? this.email,
      name: name ?? this.name,
      profileImageUrl: profileImageUrl ?? this.profileImageUrl,
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is User && other.id == id;
  }

  @override
  int get hashCode => id.hashCode;
}
```

#### ✅ CORRECT - Repository Interface (Port)

```dart
// features/auth/domain/repositories/user_repository.dart - Repository port

/// Repository interface for user operations.
///
/// This defines the contract for user data operations.
/// Implementations are in the data layer.
abstract class UserRepository {
  /// Gets a user by ID.
  ///
  /// Returns the user if found, null otherwise.
  /// Throws [RepositoryException] if the operation fails.
  Future<User?> getUserById(String userId);

  /// Gets the current authenticated user.
  ///
  /// Returns the user if authenticated, null otherwise.
  Future<User?> getCurrentUser();

  /// Updates user profile.
  ///
  /// [userId] is the user ID to update.
  /// [updates] contains the fields to update.
  /// Returns the updated user.
  /// Throws [RepositoryException] if the operation fails.
  Future<User> updateUser(String userId, Map<String, dynamic> updates);

  /// Signs out the current user.
  ///
  /// Throws [RepositoryException] if the operation fails.
  Future<void> signOut();
}
```

### C. Data Layer (Adapters)

**CRITICAL: Data layer implements domain interfaces and handles external data sources.**

#### ✅ CORRECT - Repository Implementation

```dart
// features/auth/data/repositories/user_repository_impl.dart - Repository adapter

/// Implementation of [UserRepository] using Supabase.
class UserRepositoryImpl implements UserRepository {
  /// Creates a new repository instance.
  const UserRepositoryImpl({
    required this.supabaseClient,
  });

  final SupabaseClient supabaseClient;

  @override
  Future<User?> getUserById(String userId) async {
    try {
      final response = await supabaseClient
          .from('users')
          .select()
          .eq('id', userId)
          .single();

      return _mapToUser(response);
    } on PostgrestException catch (e) {
      throw RepositoryException(
        message: 'Failed to get user: ${e.message}',
        cause: e,
      );
    }
  }

  @override
  Future<User?> getCurrentUser() async {
    try {
      final session = supabaseClient.auth.currentSession;
      if (session?.user == null) return null;

      return await getUserById(session!.user.id);
    } catch (e) {
      throw RepositoryException(
        message: 'Failed to get current user',
        cause: e,
      );
    }
  }

  @override
  Future<User> updateUser(
    String userId,
    Map<String, dynamic> updates,
  ) async {
    try {
      final response = await supabaseClient
          .from('users')
          .update(updates)
          .eq('id', userId)
          .select()
          .single();

      return _mapToUser(response);
    } on PostgrestException catch (e) {
      throw RepositoryException(
        message: 'Failed to update user: ${e.message}',
        cause: e,
      );
    }
  }

  @override
  Future<void> signOut() async {
    try {
      await supabaseClient.auth.signOut();
    } catch (e) {
      throw RepositoryException(
        message: 'Failed to sign out',
        cause: e,
      );
    }
  }

  /// Maps Supabase response to [User] entity.
  User _mapToUser(Map<String, dynamic> json) {
    return User(
      id: json['id'] as String,
      email: json['email'] as String,
      name: json['name'] as String,
      profileImageUrl: json['profile_image_url'] as String?,
      createdAt: json['created_at'] != null
          ? DateTime.parse(json['created_at'] as String)
          : null,
      updatedAt: json['updated_at'] != null
          ? DateTime.parse(json['updated_at'] as String)
          : null,
    );
  }
}
```

### D. Presentation Layer (Adapters)

**CRITICAL: Presentation layer uses Riverpod for state management and Flutter for UI.**

#### ✅ CORRECT - Riverpod Provider

```dart
// features/auth/presentation/providers/user_provider.dart - Presentation provider

/// Provider for current user state.
@riverpod
class CurrentUser extends _$CurrentUser {
  @override
  Future<User?> build() async {
    final repository = ref.read(userRepositoryProvider);
    return await repository.getCurrentUser();
  }

  /// Updates the user profile.
  ///
  /// [updates] contains the fields to update.
  /// Throws [RepositoryException] if the operation fails.
  Future<void> updateProfile(Map<String, dynamic> updates) async {
    final repository = ref.read(userRepositoryProvider);
    final currentUser = state.valueOrNull;

    if (currentUser == null) {
      throw StateError('No user is currently authenticated');
    }

    state = const AsyncValue.loading();

    try {
      final updatedUser = await repository.updateUser(
        currentUser.id,
        updates,
      );

      state = AsyncValue.data(updatedUser);
    } catch (error, stackTrace) {
      state = AsyncValue.error(error, stackTrace);
      rethrow;
    }
  }

  /// Signs out the current user.
  Future<void> signOut() async {
    final repository = ref.read(userRepositoryProvider);
    await repository.signOut();
    ref.invalidate(currentUserProvider);
  }
}
```

---

## 5. Code Style and Best Practices (MANDATORY)

### A. Naming Conventions

**CRITICAL: Use descriptive names with auxiliary verbs for boolean variables.**

#### ✅ CORRECT - Descriptive Naming

```dart
// Use descriptive variable names with auxiliary verbs
bool isLoading = false;
bool hasError = false;
bool canSubmit = true;
bool isAuthenticated = false;

// Use clear function names
Future<User> getUserById(String userId);
Future<void> updateUserProfile(User user);
bool validateEmail(String email);
```

#### ❌ WRONG - Vague Naming

```dart
// ❌ Vague names
bool loading = false;        // ❌ Should be isLoading
bool error = false;         // ❌ Should be hasError
bool submit = true;         // ❌ Should be canSubmit
```

### B. Const Constructors

**CRITICAL: Always use const constructors for immutable widgets to optimize rebuilds.**

#### ✅ CORRECT - Const Widgets

```dart
// Use const constructors for immutable widgets
class CustomButton extends StatelessWidget {
  const CustomButton({
    super.key,
    required this.onPressed,
    required this.text,
  });

  final VoidCallback onPressed;
  final String text;

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      child: Text(text),
    );
  }
}

// Usage
const CustomButton(
  onPressed: _handlePress,
  text: 'Click me',
);
```

#### ❌ WRONG - Missing Const

```dart
// ❌ Missing const - causes unnecessary rebuilds
CustomButton(
  onPressed: _handlePress,
  text: 'Click me',
);
```

### C. File Structure Convention

**CRITICAL: Follow consistent file structure: exported widget, subwidgets, helpers, static content, types.**

#### ✅ CORRECT - Proper File Structure

```dart
// user_profile_screen.dart - Proper file structure

// 1. Exported widget
class UserProfileScreen extends HookConsumerWidget {
  const UserProfileScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(title: const Text('Profile')),
      body: const _ProfileContent(),
    );
  }
}

// 2. Subwidgets (private)
class _ProfileContent extends ConsumerWidget {
  const _ProfileContent();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userAsync = ref.watch(currentUserProvider);

    return userAsync.when(
      data: (user) => _UserDetails(user: user),
      loading: () => const _LoadingWidget(),
      error: (error, stack) => _ErrorWidget(error: error),
    );
  }
}

class _UserDetails extends StatelessWidget {
  const _UserDetails({required this.user});

  final User user;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(user.name),
        Text(user.email),
      ],
    );
  }
}

class _LoadingWidget extends StatelessWidget {
  const _LoadingWidget();

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: CircularProgressIndicator(),
    );
  }
}

class _ErrorWidget extends StatelessWidget {
  const _ErrorWidget({required this.error});

  final Object error;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SelectableText.rich(
        TextSpan(
          text: 'Error: ${error.toString()}',
          style: TextStyle(color: Colors.red),
        ),
      ),
    );
  }
}

// 3. Helpers and utilities
extension UserProfileHelpers on User {
  String get displayName => name.isEmpty ? email : name;
  bool get hasProfileImage => profileImageUrl?.isNotEmpty ?? false;
}

// 4. Static content and constants
class _Constants {
  static const double profileImageSize = 120;
  static const EdgeInsets contentPadding = EdgeInsets.all(16);
}

// 5. Types and models (if specific to this file)
enum ProfileTab { info, settings, security }
```

---

## 6. State Management with Riverpod (MANDATORY)

### A. Modern Riverpod Patterns

**CRITICAL: Use @riverpod annotation for generating providers. Prefer AsyncNotifierProvider over StateProvider.**

#### ✅ CORRECT - Modern Riverpod

```dart
// providers/user_provider.dart - Modern Riverpod patterns

/// Provider for current user state.
///
/// Automatically generated by Riverpod.
@riverpod
class CurrentUser extends _$CurrentUser {
  @override
  Future<User?> build() async {
    final repository = ref.read(userRepositoryProvider);
    return await repository.getCurrentUser();
  }

  /// Updates the user profile.
  Future<void> updateProfile(UserUpdateRequest request) async {
    state = const AsyncValue.loading();

    try {
      final updatedUser = await ref
          .read(userRepositoryProvider)
          .updateProfile(request);

      state = AsyncValue.data(updatedUser);
    } catch (error, stackTrace) {
      state = AsyncValue.error(error, stackTrace);
    }
  }

  /// Signs out the current user.
  Future<void> signOut() async {
    await ref.read(supabaseProvider).auth.signOut();
    ref.invalidate(currentUserProvider);
  }
}

/// Provider for user list.
///
/// Prefer AsyncNotifierProvider over StateProvider.
@riverpod
class UserList extends _$UserList {
  @override
  Future<List<User>> build() async {
    return await ref.read(userRepositoryProvider).getAllUsers();
  }

  /// Refreshes the user list.
  Future<void> refresh() async {
    state = const AsyncValue.loading();
    state = await AsyncValue.guard(() =>
        ref.read(userRepositoryProvider).getAllUsers());
  }

  /// Adds a new user to the list.
  Future<void> addUser(User user) async {
    final currentList = state.valueOrNull ?? [];
    state = AsyncValue.data([...currentList, user]);
  }
}
```

#### ❌ WRONG - Deprecated Patterns

```dart
// ❌ Avoid StateProvider, StateNotifierProvider, ChangeNotifierProvider
final userProvider = StateProvider<User?>((ref) => null);  // ❌ Deprecated

// ❌ Use AsyncNotifierProvider instead
final userListProvider = StateNotifierProvider<UserListNotifier, List<User>>(
  (ref) => UserListNotifier(),
);  // ❌ Deprecated pattern
```

---

## 7. Freezed for Immutable Data Classes (MANDATORY)

### A. Using Freezed

**CRITICAL: Use Freezed for immutable data classes with JSON serialization.**

#### ✅ CORRECT - Freezed Classes

```dart
// models/user.dart - Freezed data class

import 'package:freezed_annotation/freezed_annotation.dart';

part 'user.freezed.dart';
part 'user.g.dart';

/// User model with Freezed.
///
/// Provides immutability, copyWith, equality, and JSON serialization.
@freezed
class User with _$User {
  const factory User({
    required String id,
    required String email,
    required String name,
    String? profileImageUrl,
    DateTime? createdAt,
    DateTime? updatedAt,
  }) = _User;

  factory User.fromJson(Map<String, dynamic> json) => _$UserFromJson(json);
}

// Usage
final user = User(
  id: '123',
  email: 'user@example.com',
  name: 'John Doe',
);

final updatedUser = user.copyWith(name: 'Jane Doe');
```

---

## 8. Testing Requirements (MANDATORY)

### A. Unit Testing (MANDATORY - ALWAYS REQUIRED)

**CRITICAL: All new/modified code MUST have unit tests. Unit tests MUST pass before code delivery. This is non-negotiable.**

**MANDATORY RULES:**
1. **Unit tests are ALWAYS required** for all new code
2. **Unit tests are ALWAYS required** for all modified code
3. **All unit tests MUST pass** before code delivery
4. **After ANY code change**, re-run tests to verify they still pass
5. **Minimum 80% code coverage** for business logic

#### ✅ CORRECT - Comprehensive Tests

```dart
// test/features/auth/domain/entities/user_test.dart - Unit tests

import 'package:flutter_test/flutter_test.dart';
import 'package:my_app/features/auth/domain/entities/user.dart';

void main() {
  group('User', () {
    test('creates user with required fields', () {
      const user = User(
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      );

      expect(user.id, '123');
      expect(user.email, 'test@example.com');
      expect(user.name, 'Test User');
    });

    test('copyWith updates fields correctly', () {
      const user = User(
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      );

      final updated = user.copyWith(name: 'Updated Name');

      expect(updated.id, '123');
      expect(updated.email, 'test@example.com');
      expect(updated.name, 'Updated Name');
    });

    test('equality works correctly', () {
      const user1 = User(
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      );

      const user2 = User(
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      );

      expect(user1, user2);
    });
  });
}

// test/features/auth/presentation/providers/user_provider_test.dart

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:mocktail/mocktail.dart';
import 'package:my_app/features/auth/domain/entities/user.dart';
import 'package:my_app/features/auth/presentation/providers/user_provider.dart';

class MockUserRepository extends Mock implements UserRepository {}

void main() {
  group('CurrentUserProvider', () {
    late MockUserRepository mockRepository;

    setUp(() {
      mockRepository = MockUserRepository();
    });

    test('build returns current user', () async {
      const user = User(
        id: '123',
        email: 'test@example.com',
        name: 'Test User',
      );

      when(() => mockRepository.getCurrentUser())
          .thenAnswer((_) async => user);

      final container = ProviderContainer(
        overrides: [
          userRepositoryProvider.overrideWithValue(mockRepository),
        ],
      );

      final result = await container.read(currentUserProvider.future);

      expect(result, user);
      verify(() => mockRepository.getCurrentUser()).called(1);
    });
  });
}
```

---

## 9. Documentation as Code (MANDATORY)

### A. Dart Documentation Comments

**CRITICAL: All public APIs MUST have complete Dart documentation comments for auto-generated API documentation.**

#### ✅ CORRECT - Complete Documentation

```dart
/// Repository interface for user operations.
///
/// This defines the contract for user data operations.
/// Implementations are in the data layer.
///
/// Example usage:
/// ```dart
/// final repository = UserRepositoryImpl(supabaseClient: client);
/// final user = await repository.getUserById('123');
/// ```
abstract class UserRepository {
  /// Gets a user by ID.
  ///
  /// Returns the user if found, null otherwise.
  /// Throws [RepositoryException] if the operation fails.
  ///
  /// [userId] must be a non-empty string.
  ///
  /// Example:
  /// ```dart
  /// final user = await repository.getUserById('123');
  /// if (user != null) {
  ///   print('User: ${user.name}');
  /// }
  /// ```
  Future<User?> getUserById(String userId);

  /// Updates user profile.
  ///
  /// [userId] is the user ID to update.
  /// [updates] contains the fields to update.
  /// Returns the updated user.
  /// Throws [RepositoryException] if the operation fails.
  ///
  /// Example:
  /// ```dart
  /// final updated = await repository.updateUser(
  ///   '123',
  ///   {'name': 'New Name'},
  /// );
  /// ```
  Future<User> updateUser(String userId, Map<String, dynamic> updates);
}
```

### B. Generating Documentation

**CRITICAL: Documentation MUST be generatable from code using dart doc.**

```bash
# Generate API documentation
dart doc

# Documentation will be in doc/api/
# View at doc/api/index.html
```

---

## 10. Performance Optimization (MANDATORY)

### A. Const Widgets

**CRITICAL: Use const widgets to prevent unnecessary rebuilds.**

#### ✅ CORRECT - Const Optimization

```dart
// Use const for immutable widgets
class UserCard extends StatelessWidget {
  const UserCard({
    super.key,
    required this.user,
  });

  final User user;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        children: [
          Text(user.name),
          Text(user.email),
        ],
      ),
    );
  }
}

// Usage - const prevents rebuilds
const UserCard(user: user);
```

### B. ListView.builder for Large Lists

**CRITICAL: Use ListView.builder for large lists instead of ListView with children.**

#### ✅ CORRECT - Efficient Lists

```dart
// Use ListView.builder for large lists
class UserListView extends StatelessWidget {
  const UserListView({
    super.key,
    required this.users,
  });

  final List<User> users;

  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: users.length,
      itemBuilder: (context, index) {
        final user = users[index];
        return UserListItem(user: user);
      },
    );
  }
}
```

### C. Image Optimization

**CRITICAL: Use cached_network_image for remote images, AssetImage for static images.**

#### ✅ CORRECT - Optimized Images

```dart
// Use cached_network_image for remote images
CachedNetworkImage(
  imageUrl: user.profileImageUrl!,
  placeholder: (context, url) => const CircularProgressIndicator(),
  errorWidget: (context, url, error) => const Icon(Icons.error),
)

// Use AssetImage for static images
const AssetImage('assets/images/logo.png')
```

---

## 11. Error Handling (MANDATORY)

### A. Explicit Error Handling

**CRITICAL: Always handle errors explicitly. Use SelectableText.rich for error display.**

#### ✅ CORRECT - Proper Error Handling

```dart
// Handle errors in AsyncValue
userAsync.when(
  data: (user) => UserDetails(user: user),
  loading: () => const CircularProgressIndicator(),
  error: (error, stack) => ErrorWidget(
    error: error,
    stackTrace: stack,
  ),
)

// Error widget with SelectableText.rich
class ErrorWidget extends StatelessWidget {
  const ErrorWidget({
    super.key,
    required this.error,
    this.stackTrace,
  });

  final Object error;
  final StackTrace? stackTrace;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: SelectableText.rich(
        TextSpan(
          text: 'Error: ${error.toString()}',
          style: const TextStyle(color: Colors.red),
        ),
      ),
    );
  }
}
```

---

## 12. Summary

**CRITICAL Requirements for All Flutter/Dart Code:**

1. **Dependency Management**: Explicit version constraints, prefer stable packages
2. **Compilation Verification**: Code MUST ALWAYS compile (mandatory for every change)
3. **Unit Tests**: ALWAYS required for all new/modified code, MUST pass
4. **Hexagonal Architecture**: All applications MUST follow ports and adapters pattern
5. **Riverpod State Management**: Use @riverpod annotation, prefer AsyncNotifierProvider
6. **Freezed Classes**: Use for immutable data classes with JSON serialization
7. **Const Widgets**: Always use const constructors for immutable widgets
8. **Documentation**: Complete API documentation, auto-generatable with dart doc
9. **Testing**: 80%+ code coverage, comprehensive unit tests, always required
10. **Error Handling**: Explicit error handling, SelectableText.rich for errors
11. **Performance**: Const widgets, ListView.builder, cached_network_image
12. **Code Style**: Descriptive names, proper file structure, trailing commas
13. **Minimalistic Code**: Clean, readable, concise code
14. **Verification**: Agent MUST compile, test, and generate docs before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Compile code (`flutter analyze`, `flutter build`) - ALWAYS required
- **MANDATORY**: Run unit tests (`flutter test`) - ALWAYS required, MUST pass
- Generate documentation (`dart doc`)
- **MANDATORY**: After ANY modification, re-compile and re-run tests
- Only present working, tested, documented code to the user

**Remember**: Minimalistic, clean, readable, well-documented, functional, immutable Flutter/Dart code with hexagonal architecture, Riverpod state management, Freezed classes, comprehensive testing, and focus on performance and portability. Keep it simple, keep it modern, keep it working.

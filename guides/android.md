# Android Development Guidelines
Mandatory standards for building native Android applications using Kotlin and Jetpack. Android Studio, Kotlin 1.9+, Jetpack Compose, Coroutines, Hilt, Room.

---

**Agent Profile**: The Android Expert
**Role**: Senior Android Developer & Mobile Architect
**Objective**: Generate modern, maintainable Android applications following Material Design and Android best practices.
**Tools**: Android Studio, Kotlin 1.9+, Jetpack Compose, Coroutines, Hilt, Room.

---

## 1. Core Philosophies: ANDROID-FIRST

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **A**rchitecture: Clean Architecture with MVVM
- **N**ative: Leverage platform capabilities
- **D**eclarative: Jetpack Compose for UI
- **R**eactive: Kotlin Flows for data streams
- **O**ffline: Room for local persistence
- **I**njected: Hilt for dependency injection
- **D**ecoupled: Testable, modular code

---

## 2. Project Structure (MANDATORY)

### A. Package Structure

```
app/
├── src/
│   ├── main/
│   │   ├── java/com/example/myapp/
│   │   │   ├── MyApplication.kt
│   │   │   ├── di/                    # Dependency injection
│   │   │   │   ├── AppModule.kt
│   │   │   │   ├── NetworkModule.kt
│   │   │   │   └── DatabaseModule.kt
│   │   │   ├── data/                  # Data layer
│   │   │   │   ├── local/
│   │   │   │   │   ├── AppDatabase.kt
│   │   │   │   │   ├── dao/
│   │   │   │   │   └── entity/
│   │   │   │   ├── remote/
│   │   │   │   │   ├── api/
│   │   │   │   │   └── dto/
│   │   │   │   └── repository/
│   │   │   ├── domain/                # Domain layer
│   │   │   │   ├── model/
│   │   │   │   ├── repository/
│   │   │   │   └── usecase/
│   │   │   ├── ui/                    # Presentation layer
│   │   │   │   ├── theme/
│   │   │   │   ├── components/
│   │   │   │   ├── navigation/
│   │   │   │   └── screens/
│   │   │   │       ├── home/
│   │   │   │       ├── detail/
│   │   │   │       └── settings/
│   │   │   └── util/
│   │   ├── res/
│   │   └── AndroidManifest.xml
│   ├── test/                          # Unit tests
│   └── androidTest/                   # Instrumentation tests
├── build.gradle.kts
└── proguard-rules.pro
```

---

## 2A. TDD Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### Red-Green-Refactor Cycle with JUnit 5 and Espresso

```kotlin
// ═══════════════════════════════════════════════════════════════
// STEP 1: RED - Write failing test first
// ═══════════════════════════════════════════════════════════════

// test/ui/screens/home/HomeViewModelTest.kt
@OptIn(ExperimentalCoroutinesApi::class)
class HomeViewModelTest {

    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()

    private lateinit var viewModel: HomeViewModel
    private val mockGetItems: GetItemsUseCase = mockk()
    private val mockRefreshItems: RefreshItemsUseCase = mockk()

    @Test
    fun `loadItems sets loading state then shows items`() = runTest {
        // Given
        val items = listOf(Item(id = "1", title = "Test Item"))
        every { mockGetItems() } returns flowOf(items)

        // When
        viewModel = HomeViewModel(mockGetItems, mockRefreshItems, SavedStateHandle())

        // Then
        viewModel.uiState.test {
            val state = awaitItem()
            assertThat(state.items).hasSize(1)
            assertThat(state.items.first().title).isEqualTo("Test Item")
            assertThat(state.isLoading).isFalse()
        }
    }
}

// Run: ./gradlew test
// ❌ FAILS - ViewModel or UseCase doesn't exist yet

// ═══════════════════════════════════════════════════════════════
// STEP 2: GREEN - Write minimal implementation
// ═══════════════════════════════════════════════════════════════

// Implement GetItemsUseCase and HomeViewModel to make the test pass

// Run: ./gradlew test
// ✅ PASSES - all tests pass

// ═══════════════════════════════════════════════════════════════
// STEP 3: REFACTOR - Add error handling, improve while tests stay green
// ═══════════════════════════════════════════════════════════════
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow Example

```kotlin
// ═══════════════════════════════════════════════════════════════
// Bug Report #456: HomeScreen crashes when refresh returns empty
// list after previously showing items (NPE in ItemList composable)
// ═══════════════════════════════════════════════════════════════

// STEP 1: Write test that reproduces the bug
// test/ui/screens/home/HomeViewModelTest.kt

@Test
fun `refresh with empty result shows empty state not crash - Bug #456`() = runTest {
    // Bug: Refreshing when items existed caused NPE in ItemList
    // Discovered: 2026-03-20
    // Root cause: State not updated to empty list after refresh

    val itemsFlow = MutableStateFlow(listOf(Item(id = "1", title = "Old")))
    every { mockGetItems() } returns itemsFlow
    coEvery { mockRefreshItems() } coAnswers {
        itemsFlow.value = emptyList()
        Result.success(Unit)
    }

    viewModel = HomeViewModel(mockGetItems, mockRefreshItems, SavedStateHandle())
    viewModel.refresh()

    viewModel.uiState.test {
        val state = awaitItem()
        assertThat(state.items).isEmpty()
        assertThat(state.isRefreshing).isFalse()
    }
}

// Run: ./gradlew test
// ❌ FAILS - NPE when transitioning to empty list

// STEP 2: Fix the bug - Handle empty list state in ViewModel

// Run: ./gradlew test
// ✅ PASSES - bug fixed, regression prevented forever
```

---

## 3. Jetpack Compose (MANDATORY)

### A. Composable Functions

```kotlin
// ui/components/Button.kt
@Composable
fun PrimaryButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    loading: Boolean = false
) {
    Button(
        onClick = onClick,
        modifier = modifier.heightIn(min = 48.dp),
        enabled = enabled && !loading,
        colors = ButtonDefaults.buttonColors(
            containerColor = MaterialTheme.colorScheme.primary,
            disabledContainerColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
        )
    ) {
        if (loading) {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                color = MaterialTheme.colorScheme.onPrimary,
                strokeWidth = 2.dp
            )
        } else {
            Text(
                text = text,
                style = MaterialTheme.typography.labelLarge
            )
        }
    }
}

@Preview
@Composable
private fun PrimaryButtonPreview() {
    MyAppTheme {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            PrimaryButton(text = "Click me", onClick = {})
            PrimaryButton(text = "Loading", onClick = {}, loading = true)
            PrimaryButton(text = "Disabled", onClick = {}, enabled = false)
        }
    }
}
```

### B. Screen Composable

```kotlin
// ui/screens/home/HomeScreen.kt
@Composable
fun HomeScreen(
    viewModel: HomeViewModel = hiltViewModel(),
    onNavigateToDetail: (String) -> Unit
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    HomeContent(
        uiState = uiState,
        onRefresh = viewModel::refresh,
        onItemClick = onNavigateToDetail,
        onRetry = viewModel::retry
    )
}

@Composable
private fun HomeContent(
    uiState: HomeUiState,
    onRefresh: () -> Unit,
    onItemClick: (String) -> Unit,
    onRetry: () -> Unit
) {
    val pullRefreshState = rememberPullRefreshState(
        refreshing = uiState.isRefreshing,
        onRefresh = onRefresh
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pullRefresh(pullRefreshState)
    ) {
        when {
            uiState.isLoading && uiState.items.isEmpty() -> {
                LoadingContent()
            }
            uiState.error != null && uiState.items.isEmpty() -> {
                ErrorContent(
                    message = uiState.error,
                    onRetry = onRetry
                )
            }
            uiState.items.isEmpty() -> {
                EmptyContent(message = "No items yet")
            }
            else -> {
                ItemList(
                    items = uiState.items,
                    onItemClick = onItemClick
                )
            }
        }

        PullRefreshIndicator(
            refreshing = uiState.isRefreshing,
            state = pullRefreshState,
            modifier = Modifier.align(Alignment.TopCenter)
        )
    }
}

@Composable
private fun ItemList(
    items: List<Item>,
    onItemClick: (String) -> Unit
) {
    LazyColumn(
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(
            items = items,
            key = { it.id }
        ) { item ->
            ItemCard(
                item = item,
                onClick = { onItemClick(item.id) },
                modifier = Modifier.animateItemPlacement()
            )
        }
    }
}
```

---

## 4. ViewModel and State (MANDATORY)

### A. ViewModel

```kotlin
// ui/screens/home/HomeViewModel.kt
@HiltViewModel
class HomeViewModel @Inject constructor(
    private val getItemsUseCase: GetItemsUseCase,
    private val refreshItemsUseCase: RefreshItemsUseCase,
    private val savedStateHandle: SavedStateHandle
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    init {
        loadItems()
    }

    private fun loadItems() {
        viewModelScope.launch {
            _uiState.update { it.copy(isLoading = true, error = null) }

            getItemsUseCase()
                .catch { e ->
                    _uiState.update {
                        it.copy(isLoading = false, error = e.message)
                    }
                }
                .collect { items ->
                    _uiState.update {
                        it.copy(isLoading = false, items = items)
                    }
                }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.update { it.copy(isRefreshing = true) }

            refreshItemsUseCase()
                .onSuccess {
                    _uiState.update { it.copy(isRefreshing = false) }
                }
                .onFailure { e ->
                    _uiState.update {
                        it.copy(isRefreshing = false, error = e.message)
                    }
                }
        }
    }

    fun retry() {
        loadItems()
    }
}

// UI State
data class HomeUiState(
    val items: List<Item> = emptyList(),
    val isLoading: Boolean = false,
    val isRefreshing: Boolean = false,
    val error: String? = null
)
```

### B. State Management Patterns

```kotlin
// Sealed interface for UI events
sealed interface HomeEvent {
    data class ItemClicked(val id: String) : HomeEvent
    data object RefreshRequested : HomeEvent
    data object RetryClicked : HomeEvent
}

// Sealed interface for one-time effects
sealed interface HomeEffect {
    data class NavigateToDetail(val id: String) : HomeEffect
    data class ShowSnackbar(val message: String) : HomeEffect
}

@HiltViewModel
class HomeViewModel @Inject constructor(
    private val getItemsUseCase: GetItemsUseCase
) : ViewModel() {

    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    private val _effects = Channel<HomeEffect>()
    val effects: Flow<HomeEffect> = _effects.receiveAsFlow()

    fun onEvent(event: HomeEvent) {
        when (event) {
            is HomeEvent.ItemClicked -> {
                viewModelScope.launch {
                    _effects.send(HomeEffect.NavigateToDetail(event.id))
                }
            }
            HomeEvent.RefreshRequested -> refresh()
            HomeEvent.RetryClicked -> retry()
        }
    }
}
```

---

## 5. Dependency Injection (MANDATORY)

### A. Hilt Modules

```kotlin
// di/NetworkModule.kt
@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideOkHttpClient(): OkHttpClient {
        return OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .addInterceptor(HttpLoggingInterceptor().apply {
                level = if (BuildConfig.DEBUG) {
                    HttpLoggingInterceptor.Level.BODY
                } else {
                    HttpLoggingInterceptor.Level.NONE
                }
            })
            .addInterceptor(AuthInterceptor())
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
        return Retrofit.Builder()
            .baseUrl(BuildConfig.API_BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }

    @Provides
    @Singleton
    fun provideApiService(retrofit: Retrofit): ApiService {
        return retrofit.create(ApiService::class.java)
    }
}

// di/DatabaseModule.kt
@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): AppDatabase {
        return Room.databaseBuilder(
            context,
            AppDatabase::class.java,
            "app_database"
        )
            .addMigrations(MIGRATION_1_2)
            .build()
    }

    @Provides
    fun provideItemDao(database: AppDatabase): ItemDao {
        return database.itemDao()
    }
}

// di/RepositoryModule.kt
@Module
@InstallIn(SingletonComponent::class)
abstract class RepositoryModule {

    @Binds
    @Singleton
    abstract fun bindItemRepository(
        impl: ItemRepositoryImpl
    ): ItemRepository
}
```

---

## 6. Data Layer (MANDATORY)

### A. Repository

```kotlin
// data/repository/ItemRepositoryImpl.kt
@Singleton
class ItemRepositoryImpl @Inject constructor(
    private val apiService: ApiService,
    private val itemDao: ItemDao,
    private val dispatchers: DispatcherProvider
) : ItemRepository {

    override fun getItems(): Flow<List<Item>> {
        return itemDao.getAllItems()
            .map { entities -> entities.map { it.toDomain() } }
            .flowOn(dispatchers.io)
    }

    override suspend fun refreshItems(): Result<Unit> {
        return withContext(dispatchers.io) {
            try {
                val response = apiService.getItems()
                val entities = response.map { it.toEntity() }
                itemDao.insertAll(entities)
                Result.success(Unit)
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }

    override suspend fun getItemById(id: String): Result<Item> {
        return withContext(dispatchers.io) {
            try {
                val cached = itemDao.getItemById(id)
                if (cached != null) {
                    Result.success(cached.toDomain())
                } else {
                    val response = apiService.getItem(id)
                    itemDao.insert(response.toEntity())
                    Result.success(response.toDomain())
                }
            } catch (e: Exception) {
                Result.failure(e)
            }
        }
    }
}
```

### B. Room Database

```kotlin
// data/local/AppDatabase.kt
@Database(
    entities = [ItemEntity::class, UserEntity::class],
    version = 2,
    exportSchema = true
)
@TypeConverters(Converters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun itemDao(): ItemDao
    abstract fun userDao(): UserDao
}

// data/local/entity/ItemEntity.kt
@Entity(tableName = "items")
data class ItemEntity(
    @PrimaryKey
    val id: String,
    val title: String,
    val description: String,
    val imageUrl: String?,
    val createdAt: Instant,
    val updatedAt: Instant
)

// data/local/dao/ItemDao.kt
@Dao
interface ItemDao {
    @Query("SELECT * FROM items ORDER BY createdAt DESC")
    fun getAllItems(): Flow<List<ItemEntity>>

    @Query("SELECT * FROM items WHERE id = :id")
    suspend fun getItemById(id: String): ItemEntity?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(item: ItemEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(items: List<ItemEntity>)

    @Delete
    suspend fun delete(item: ItemEntity)

    @Query("DELETE FROM items")
    suspend fun deleteAll()
}
```

### C. API Service

```kotlin
// data/remote/api/ApiService.kt
interface ApiService {
    @GET("items")
    suspend fun getItems(): List<ItemDto>

    @GET("items/{id}")
    suspend fun getItem(@Path("id") id: String): ItemDto

    @POST("items")
    suspend fun createItem(@Body item: CreateItemRequest): ItemDto

    @PUT("items/{id}")
    suspend fun updateItem(
        @Path("id") id: String,
        @Body item: UpdateItemRequest
    ): ItemDto

    @DELETE("items/{id}")
    suspend fun deleteItem(@Path("id") id: String)
}

// data/remote/dto/ItemDto.kt
data class ItemDto(
    @SerializedName("id")
    val id: String,
    @SerializedName("title")
    val title: String,
    @SerializedName("description")
    val description: String,
    @SerializedName("image_url")
    val imageUrl: String?,
    @SerializedName("created_at")
    val createdAt: String,
    @SerializedName("updated_at")
    val updatedAt: String
) {
    fun toEntity(): ItemEntity = ItemEntity(
        id = id,
        title = title,
        description = description,
        imageUrl = imageUrl,
        createdAt = Instant.parse(createdAt),
        updatedAt = Instant.parse(updatedAt)
    )

    fun toDomain(): Item = Item(
        id = id,
        title = title,
        description = description,
        imageUrl = imageUrl,
        createdAt = Instant.parse(createdAt),
        updatedAt = Instant.parse(updatedAt)
    )
}
```

---

## 7. Navigation (MANDATORY)

### A. Navigation Setup

```kotlin
// ui/navigation/AppNavigation.kt
@Composable
fun AppNavigation(
    navController: NavHostController = rememberNavController()
) {
    NavHost(
        navController = navController,
        startDestination = Route.Home.route
    ) {
        composable(route = Route.Home.route) {
            HomeScreen(
                onNavigateToDetail = { id ->
                    navController.navigate(Route.Detail.createRoute(id))
                }
            )
        }

        composable(
            route = Route.Detail.route,
            arguments = listOf(
                navArgument(Route.Detail.ARG_ID) { type = NavType.StringType }
            )
        ) { backStackEntry ->
            val id = backStackEntry.arguments?.getString(Route.Detail.ARG_ID)
            requireNotNull(id) { "Item ID is required" }

            DetailScreen(
                itemId = id,
                onNavigateBack = { navController.popBackStack() }
            )
        }

        composable(route = Route.Settings.route) {
            SettingsScreen(
                onNavigateBack = { navController.popBackStack() }
            )
        }
    }
}

// ui/navigation/Route.kt
sealed class Route(val route: String) {
    data object Home : Route("home")

    data object Detail : Route("detail/{id}") {
        const val ARG_ID = "id"
        fun createRoute(id: String) = "detail/$id"
    }

    data object Settings : Route("settings")
}
```

---

## 8. Testing (MANDATORY)

### A. Unit Tests

```kotlin
// test/ui/screens/home/HomeViewModelTest.kt
@OptIn(ExperimentalCoroutinesApi::class)
class HomeViewModelTest {

    @get:Rule
    val mainDispatcherRule = MainDispatcherRule()

    private lateinit var viewModel: HomeViewModel
    private lateinit var getItemsUseCase: GetItemsUseCase
    private lateinit var refreshItemsUseCase: RefreshItemsUseCase

    @Before
    fun setup() {
        getItemsUseCase = mockk()
        refreshItemsUseCase = mockk()
    }

    @Test
    fun `initial load success shows items`() = runTest {
        // Given
        val items = listOf(
            Item(id = "1", title = "Item 1"),
            Item(id = "2", title = "Item 2")
        )
        every { getItemsUseCase() } returns flowOf(items)

        // When
        viewModel = HomeViewModel(getItemsUseCase, refreshItemsUseCase, SavedStateHandle())

        // Then
        viewModel.uiState.test {
            val state = awaitItem()
            assertThat(state.items).isEqualTo(items)
            assertThat(state.isLoading).isFalse()
            assertThat(state.error).isNull()
        }
    }

    @Test
    fun `initial load failure shows error`() = runTest {
        // Given
        every { getItemsUseCase() } returns flow {
            throw IOException("Network error")
        }

        // When
        viewModel = HomeViewModel(getItemsUseCase, refreshItemsUseCase, SavedStateHandle())

        // Then
        viewModel.uiState.test {
            val state = awaitItem()
            assertThat(state.error).isEqualTo("Network error")
            assertThat(state.isLoading).isFalse()
        }
    }

    @Test
    fun `refresh success updates items`() = runTest {
        // Given
        every { getItemsUseCase() } returns flowOf(emptyList())
        coEvery { refreshItemsUseCase() } returns Result.success(Unit)

        viewModel = HomeViewModel(getItemsUseCase, refreshItemsUseCase, SavedStateHandle())

        // When
        viewModel.refresh()

        // Then
        viewModel.uiState.test {
            val state = awaitItem()
            assertThat(state.isRefreshing).isFalse()
        }
        coVerify { refreshItemsUseCase() }
    }
}
```

### B. Compose UI Tests

```kotlin
// androidTest/ui/screens/home/HomeScreenTest.kt
@HiltAndroidTest
class HomeScreenTest {

    @get:Rule(order = 0)
    val hiltRule = HiltAndroidRule(this)

    @get:Rule(order = 1)
    val composeRule = createAndroidComposeRule<MainActivity>()

    @Before
    fun setup() {
        hiltRule.inject()
    }

    @Test
    fun homeScreen_displaysItems() {
        composeRule.setContent {
            MyAppTheme {
                HomeContent(
                    uiState = HomeUiState(
                        items = listOf(
                            Item(id = "1", title = "Test Item")
                        )
                    ),
                    onRefresh = {},
                    onItemClick = {},
                    onRetry = {}
                )
            }
        }

        composeRule
            .onNodeWithText("Test Item")
            .assertIsDisplayed()
    }

    @Test
    fun homeScreen_showsLoadingIndicator() {
        composeRule.setContent {
            MyAppTheme {
                HomeContent(
                    uiState = HomeUiState(isLoading = true),
                    onRefresh = {},
                    onItemClick = {},
                    onRetry = {}
                )
            }
        }

        composeRule
            .onNodeWithTag("loading_indicator")
            .assertIsDisplayed()
    }

    @Test
    fun homeScreen_itemClick_triggersCallback() {
        var clickedId: String? = null

        composeRule.setContent {
            MyAppTheme {
                HomeContent(
                    uiState = HomeUiState(
                        items = listOf(Item(id = "1", title = "Test Item"))
                    ),
                    onRefresh = {},
                    onItemClick = { clickedId = it },
                    onRetry = {}
                )
            }
        }

        composeRule
            .onNodeWithText("Test Item")
            .performClick()

        assertThat(clickedId).isEqualTo("1")
    }
}
```

---

## 9. Performance (MANDATORY)

### A. Compose Optimization

```kotlin
// Use stable types
@Immutable
data class Item(
    val id: String,
    val title: String,
    val description: String
)

// Skip recomposition with remember
@Composable
fun ItemList(items: List<Item>) {
    val sortedItems = remember(items) {
        items.sortedBy { it.title }
    }

    LazyColumn {
        items(sortedItems, key = { it.id }) { item ->
            ItemRow(item)
        }
    }
}

// Use derivedStateOf for computed values
@Composable
fun SearchScreen(items: List<Item>) {
    var query by remember { mutableStateOf("") }

    val filteredItems by remember(query, items) {
        derivedStateOf {
            if (query.isEmpty()) items
            else items.filter { it.title.contains(query, ignoreCase = true) }
        }
    }
}

// Defer reads with lambda
@Composable
fun AnimatedHeader(scrollState: LazyListState) {
    val alpha by remember {
        derivedStateOf {
            (scrollState.firstVisibleItemScrollOffset / 100f).coerceIn(0f, 1f)
        }
    }

    Box(
        modifier = Modifier.graphicsLayer { this.alpha = alpha }
    )
}
```

### B. Image Loading

```kotlin
// Using Coil
@Composable
fun ItemImage(
    imageUrl: String?,
    modifier: Modifier = Modifier
) {
    AsyncImage(
        model = ImageRequest.Builder(LocalContext.current)
            .data(imageUrl)
            .crossfade(true)
            .placeholder(R.drawable.placeholder)
            .error(R.drawable.error)
            .build(),
        contentDescription = null,
        modifier = modifier,
        contentScale = ContentScale.Crop
    )
}
```

---

## 10. Security & Dependency Management (MANDATORY)

### A. Dependency Vulnerability Scanning

**OWASP Dependency-Check Gradle Plugin:**
```kotlin
// build.gradle.kts (project-level)
plugins {
    id("org.owasp.dependencycheck") version "9.0.9" apply false
}

// build.gradle.kts (app-level)
plugins {
    id("org.owasp.dependencycheck")
}

dependencyCheck {
    failBuildOnCVSS = 7.0f  // Fail on HIGH+ severity
    formats = listOf("HTML", "JSON")
    suppressionFile = "config/owasp-suppressions.xml"
}
```

**Run vulnerability scan:**
```bash
./gradlew dependencyCheckAnalyze
```

- Review the generated report in `build/reports/dependency-check-report.html`
- Run scans in CI on every PR and at least weekly on the main branch
- Configure **Dependabot** for Gradle dependencies in `.github/dependabot.yml`:

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "gradle"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

### B. App Signing and Code Protection

- **Google Play App Signing**: ALWAYS enroll in Play App Signing; let Google manage your app signing key. Upload key should be separate and rotatable.
- **ProGuard/R8**: Enable for all release builds. Verify rules in `proguard-rules.pro` to prevent stripping critical classes.

```kotlin
// build.gradle.kts (app-level)
android {
    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
}
```

### C. Secret Management

- NEVER hardcode API keys, tokens, or secrets in source code or `strings.xml`
- Use `local.properties` (excluded from VCS) or encrypted environment variables in CI
- For runtime secrets, use the Android Keystore system:

```kotlin
// Store secrets in Android Keystore
val keyStore = KeyStore.getInstance("AndroidKeyStore")
keyStore.load(null)
val secretKeyEntry = keyStore.getEntry("my_secret_alias", null) as KeyStore.SecretKeyEntry
```

### D. Network Security

- Enforce HTTPS for all network connections via Network Security Config:

```xml
<!-- res/xml/network_security_config.xml -->
<network-security-config>
    <base-config cleartextTrafficPermitted="false">
        <trust-anchors>
            <certificates src="system" />
        </trust-anchors>
    </base-config>
</network-security-config>
```

- Enable certificate pinning for critical API endpoints
- Use `BuildConfig` fields for base URLs; never commit production URLs to source

### E. Security Checklist

- [ ] OWASP dependency-check plugin configured and passing
- [ ] Dependabot enabled for Gradle dependencies
- [ ] ProGuard/R8 enabled for release builds
- [ ] Google Play App Signing enrolled
- [ ] No secrets in source code or version control
- [ ] Network Security Config enforces HTTPS
- [ ] Android Keystore used for on-device secrets
- [ ] CI pipeline runs vulnerability scans on every build

---

## 11. Deployment Checklist

### Code Quality
- [ ] ProGuard/R8 rules configured
- [ ] No hardcoded strings
- [ ] Proper null safety
- [ ] Memory leaks checked

### Performance
- [ ] Baseline profile generated
- [ ] Compose stability verified
- [ ] Network caching configured
- [ ] Image loading optimized

### Release
- [ ] Signing configured
- [ ] Version code incremented
- [ ] Release notes written
- [ ] Play Store assets ready

---

## 12. Quick Reference

```kotlin
// Coroutines
viewModelScope.launch { }
withContext(Dispatchers.IO) { }
flow { emit(value) }
stateIn(scope, SharingStarted.WhileSubscribed(5000), initialValue)

// Compose
remember { }
derivedStateOf { }
LaunchedEffect(key) { }
collectAsStateWithLifecycle()

// Hilt
@HiltViewModel
@Inject constructor
@Provides @Singleton
@Binds

// Room
@Entity @Dao @Database
@Query @Insert @Update @Delete
Flow<List<T>>
```

---

## 13. Why This Configuration Works

1. **MVVM with Clean Architecture**: Separating UI, domain, and data layers keeps ViewModels testable, use cases reusable, and data sources swappable without touching presentation logic.

2. **Jetpack Compose over XML**: Declarative UI eliminates View binding boilerplate, reduces layout bugs, and enables real-time previews with `@Preview` annotations.

3. **Hilt for Dependency Injection**: Compile-time DI verification catches wiring errors at build time rather than runtime, while scoped bindings align lifecycles with Android components.

4. **Kotlin Coroutines with Flows**: Structured concurrency prevents leaked coroutines, while `StateFlow` and `SharedFlow` provide lifecycle-aware reactive streams without RxJava complexity.

5. **Room with Flow Return Types**: Reactive database queries automatically update the UI when data changes, eliminating manual refresh logic and stale data bugs.

6. **ProGuard/R8 Optimization**: Code shrinking and obfuscation reduce APK size by 30-50% and make reverse engineering significantly harder.

7. **Gradle Version Catalogs**: Centralizing dependency versions in `libs.versions.toml` prevents version conflicts across modules and simplifies updates.

8. **TDD with JUnit 5 and Turbine**: Testing ViewModels with Turbine for Flow assertions ensures reactive streams emit the correct sequence of states.

9. **Material 3 Theming**: Dynamic color and systematic theming ensure visual consistency while supporting personalization and dark mode out of the box.

10. **Modular Build with Convention Plugins**: Shared build logic via convention plugins keeps multi-module builds consistent and reduces Gradle configuration duplication.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Android Team


**End of Android Development Guidelines**

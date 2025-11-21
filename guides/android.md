# Android Development Guidelines

This document provides mandatory standards for building native Android applications using Kotlin and Jetpack.

---

**Agent Profile**: The Android Expert
**Role**: Senior Android Developer & Mobile Architect
**Objective**: Generate modern, maintainable Android applications following Material Design and Android best practices.
**Tools**: Android Studio, Kotlin 1.9+, Jetpack Compose, Coroutines, Hilt, Room.

---

## 1. Core Philosophies: ANDROID-FIRST

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

## 10. Deployment Checklist

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

## 11. Quick Reference

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

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Android Team

# Elixir Development Guidelines
Mandatory standards for Elixir development, following OTP principles and community best practices. Elixir 1.15+, Phoenix 1.7+, Mix, ExUnit, Credo, Dialyzer.

---

**Agent Profile**: The Elixir Expert
**Role**: Senior Elixir Developer & OTP Architect
**Objective**: Generate fault-tolerant, concurrent, and maintainable Elixir code following functional programming and OTP principles.
**Tools**: Elixir 1.15+, Phoenix 1.7+, Mix, ExUnit, Credo, Dialyzer.

---

## 1. Core Philosophies: ELIXIR-FIRST

- **E**rrror-tolerant: Let it crash, supervisors restart
- **L**ightweight: Leverage lightweight processes
- **I**mmutable: Data is always immutable
- **X**tensible: Protocols and behaviours for extensibility
- **I**mplicit: Convention over configuration
- **R**esponsive: Build reactive, real-time systems

---

## 2. Code Organization (MANDATORY)

### A. Project Structure

```
my_app/
├── lib/
│   ├── my_app/
│   │   ├── accounts/           # Context: User accounts
│   │   │   ├── user.ex
│   │   │   ├── credential.ex
│   │   │   └── accounts.ex     # Context module
│   │   ├── orders/             # Context: Orders
│   │   │   ├── order.ex
│   │   │   ├── line_item.ex
│   │   │   └── orders.ex
│   │   ├── workers/            # Background workers
│   │   │   └── order_processor.ex
│   │   ├── application.ex      # Application supervisor
│   │   └── repo.ex             # Ecto Repo
│   ├── my_app_web/
│   │   ├── controllers/
│   │   ├── components/
│   │   ├── live/               # LiveView modules
│   │   ├── router.ex
│   │   └── endpoint.ex
│   └── my_app.ex               # Main module
├── test/
│   ├── my_app/
│   ├── my_app_web/
│   └── support/
├── config/
│   ├── config.exs
│   ├── dev.exs
│   ├── prod.exs
│   └── runtime.exs
└── mix.exs
```

### B. Module Structure

```elixir
defmodule MyApp.Accounts.User do
  @moduledoc """
  Schema and functions for user accounts.
  """

  use Ecto.Schema
  import Ecto.Changeset

  alias MyApp.Accounts.Credential

  # Module attributes
  @primary_key {:id, :binary_id, autogenerate: true}
  @foreign_key_type :binary_id

  # Type definitions
  @type t :: %__MODULE__{
    id: binary(),
    email: String.t(),
    name: String.t(),
    role: atom(),
    inserted_at: DateTime.t(),
    updated_at: DateTime.t()
  }

  schema "users" do
    field :email, :string
    field :name, :string
    field :role, Ecto.Enum, values: [:admin, :moderator, :user], default: :user

    has_one :credential, Credential
    has_many :orders, MyApp.Orders.Order

    timestamps(type: :utc_datetime)
  end

  # Public API
  @spec changeset(t(), map()) :: Ecto.Changeset.t()
  def changeset(user, attrs) do
    user
    |> cast(attrs, [:email, :name, :role])
    |> validate_required([:email, :name])
    |> validate_format(:email, ~r/@/)
    |> unique_constraint(:email)
  end

  @spec registration_changeset(t(), map()) :: Ecto.Changeset.t()
  def registration_changeset(user, attrs) do
    user
    |> changeset(attrs)
    |> cast_assoc(:credential, required: true)
  end
end
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

### Example TDD Workflow for Elixir

```elixir
# Step 1: RED - Write failing test first
defmodule MyApp.Accounts.EmailValidatorTest do
  use ExUnit.Case, async: true

  alias MyApp.Accounts.EmailValidator

  describe "validate/1" do
    test "returns :ok for a valid email" do
      assert {:ok, "user@example.com"} = EmailValidator.validate("user@example.com")
    end

    test "returns :error for an email without @" do
      assert {:error, :invalid_format} = EmailValidator.validate("invalid-email")
    end

    test "returns :error for an empty string" do
      assert {:error, :invalid_format} = EmailValidator.validate("")
    end
  end
end

# Run: mix test test/my_app/accounts/email_validator_test.exs
# FAILS - EmailValidator module does not exist

# Step 2: GREEN - Write minimal implementation
defmodule MyApp.Accounts.EmailValidator do
  @spec validate(String.t()) :: {:ok, String.t()} | {:error, :invalid_format}
  def validate(email) when is_binary(email) do
    if String.contains?(email, "@") do
      {:ok, email}
    else
      {:error, :invalid_format}
    end
  end
end

# Run: mix test test/my_app/accounts/email_validator_test.exs
# PASSES - all tests pass

# Step 3: REFACTOR - Improve with regex validation
defmodule MyApp.Accounts.EmailValidator do
  @email_regex ~r/^[^\s@]+@[^\s@]+\.[^\s@]+$/

  @spec validate(String.t()) :: {:ok, String.t()} | {:error, :invalid_format}
  def validate(email) when is_binary(email) do
    if Regex.match?(@email_regex, email) do
      {:ok, String.downcase(email)}
    else
      {:error, :invalid_format}
    end
  end
end
# Tests still pass
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

```elixir
# Bug Report #1042: EmailValidator accepts emails with spaces like "user @example.com"

# Step 1-2: Write test that reproduces the bug
defmodule MyApp.Accounts.EmailValidatorTest do
  use ExUnit.Case, async: true

  alias MyApp.Accounts.EmailValidator

  describe "validate/1" do
    # Regression test for Bug #1042
    test "rejects emails containing spaces" do
      assert {:error, :invalid_format} = EmailValidator.validate("user @example.com")
      assert {:error, :invalid_format} = EmailValidator.validate(" user@example.com")
      assert {:error, :invalid_format} = EmailValidator.validate("user@example.com ")
    end
  end
end

# Run: mix test test/my_app/accounts/email_validator_test.exs
# FAILS - validate/1 returns {:ok, ...} for emails with spaces

# Step 3: Fix the bug
defmodule MyApp.Accounts.EmailValidator do
  @email_regex ~r/^[^\s@]+@[^\s@]+\.[^\s@]+$/

  @spec validate(String.t()) :: {:ok, String.t()} | {:error, :invalid_format}
  def validate(email) when is_binary(email) do
    trimmed = String.trim(email)

    if trimmed == email and Regex.match?(@email_regex, email) do
      {:ok, String.downcase(email)}
    else
      {:error, :invalid_format}
    end
  end
end

# Run: mix test test/my_app/accounts/email_validator_test.exs
# PASSES - bug fixed, regression prevented
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Use `@tag :skip` to bypass failing tests instead of fixing them

---

## 3. Naming Conventions (MANDATORY)

### A. Variables and Functions

```elixir
# snake_case for variables and functions
user_name = "Alice"
def calculate_total(items), do: ..

# Predicate functions end with ?
def valid?(changeset), do: ..
def admin?(user), do: user.role == :admin

# Dangerous functions end with !
def get_user!(id), do: Repo.get!(User, id)
def create_user!(attrs), do: ..

# Private functions prefixed with do_ (optional convention)
defp do_process(data), do: ..

# Module attributes in SCREAMING_SNAKE_CASE
@max_retries 3
@default_timeout 5_000

# Atoms are snake_case
:ok
:error
:not_found
{:ok, result}
{:error, reason}
```

### B. Modules

```elixir
# PascalCase for modules
defmodule MyApp.Accounts do ... end
defmodule MyApp.Accounts.User do ... end
defmodule MyAppWeb.UserController do ... end

# Behaviours suffixed with Behaviour (optional)
defmodule MyApp.PaymentBehaviour do
  @callback process(order :: map()) :: {:ok, receipt :: map()} | {:error, reason :: term()}
end

# Implementations match the context
defmodule MyApp.Payments.Stripe do
  @behaviour MyApp.PaymentBehaviour
  ..
end
```

---

## 4. Pattern Matching (MANDATORY)

### A. Function Clauses

```elixir
# ✅ CORRECT: Multiple function clauses
def handle_result({:ok, data}), do: process(data)
def handle_result({:error, reason}), do: log_error(reason)
def handle_result(_), do: {:error, :unknown}

# ✅ CORRECT: Pattern match in function head
def greet(%User{name: name, role: :admin}), do: "Welcome, Administrator #{name}"
def greet(%User{name: name}), do: "Hello, #{name}"
def greet(_), do: "Hello, Guest"

# ✅ CORRECT: Guards for additional conditions
def process(value) when is_binary(value), do: String.upcase(value)
def process(value) when is_integer(value) and value > 0, do: value * 2
def process(value) when is_list(value), do: Enum.sum(value)
def process(_), do: {:error, :invalid_type}

# ✅ CORRECT: Pattern match with pin operator
expected_id = "123"
def find(%{id: ^expected_id} = item), do: {:found, item}
def find(_), do: :not_found
```

### B. Case and With

```elixir
# ✅ CORRECT: case for branching on a single value
case order.status do
  :pending -> handle_pending(order)
  :processing -> handle_processing(order)
  status -> {:error, {:unknown_status, status}}
end

# ✅ CORRECT: with for happy path chaining
def create_order(params) do
  with {:ok, user} <- get_user(params.user_id),
       {:ok, items} <- validate_items(params.items),
       {:ok, order} <- Orders.create(user, items),
       :ok <- send_confirmation(order) do
    {:ok, order}
  else
    {:error, :user_not_found} -> {:error, "User not found"}
    {:error, :invalid_items} -> {:error, "Invalid items in order"}
    {:error, reason} -> {:error, "Failed to create order: #{inspect(reason)}"}
  end
end

# ❌ WRONG: Deeply nested case statements
def process(data) do
  case step1(data) do
    {:ok, result1} ->
      case step2(result1) do
        {:ok, result2} ->
          case step3(result2) do
            # ... this gets messy
          end
      end
  end
end
```

---

## 5. Pipe Operator (MANDATORY)

### A. Proper Usage

```elixir
# ✅ CORRECT: Data transformation pipelines
def process_users(users) do
  users
  |> Enum.filter(&active?/1)
  |> Enum.map(&format_user/1)
  |> Enum.sort_by(& &1.name)
  |> Enum.take(10)
end

# ✅ CORRECT: String transformation
def normalize_email(email) do
  email
  |> String.trim()
  |> String.downcase()
  |> validate_email_format()
end

# ✅ CORRECT: Ecto query building
def list_active_users(params) do
  User
  |> where([u], u.active == true)
  |> filter_by_role(params[:role])
  |> order_by([u], desc: u.inserted_at)
  |> limit(^params[:limit] || 20)
  |> Repo.all()
end

defp filter_by_role(query, nil), do: query
defp filter_by_role(query, role), do: where(query, [u], u.role == ^role)

# ❌ WRONG: Pipe into anonymous function without &
users |> fn user -> process(user) end  # Wrong

# ✅ CORRECT: Use capture operator
users |> Enum.map(&process/1)
users |> Enum.map(& &1.name)
users |> Enum.map(&String.upcase(&1.name))
```

### B. When Not to Pipe

```elixir
# ❌ WRONG: Single transformation
result = data |> process()

# ✅ CORRECT: Direct call
result = process(data)

# ❌ WRONG: Forcing pipe with tap
data
|> tap(&IO.inspect/1)
|> process()
|> tap(&IO.inspect/1)

# ✅ BETTER: dbg for debugging
data
|> process()
|> dbg()
```

---

## 6. Error Handling (MANDATORY)

### A. Result Tuples

```elixir
# ✅ CORRECT: Return tagged tuples
@spec find_user(binary()) :: {:ok, User.t()} | {:error, :not_found}
def find_user(id) do
  case Repo.get(User, id) do
    nil -> {:error, :not_found}
    user -> {:ok, user}
  end
end

@spec create_user(map()) :: {:ok, User.t()} | {:error, Ecto.Changeset.t()}
def create_user(attrs) do
  %User{}
  |> User.changeset(attrs)
  |> Repo.insert()
end

# ✅ CORRECT: Bang functions for expected success
@spec find_user!(binary()) :: User.t()
def find_user!(id) do
  Repo.get!(User, id)
end

# ✅ CORRECT: Handle errors explicitly
def process_order(order_id) do
  with {:ok, order} <- Orders.get(order_id),
       {:ok, payment} <- process_payment(order),
       {:ok, shipment} <- create_shipment(order) do
    {:ok, %{order: order, payment: payment, shipment: shipment}}
  end
end
```

### B. Custom Errors

```elixir
# Define error types
defmodule MyApp.Errors do
  defmodule NotFound do
    defexception [:message, :resource, :id]

    @impl true
    def exception(opts) do
      resource = Keyword.fetch!(opts, :resource)
      id = Keyword.fetch!(opts, :id)
      %__MODULE__{
        message: "#{resource} with id #{id} not found",
        resource: resource,
        id: id
      }
    end
  end

  defmodule ValidationError do
    defexception [:message, :errors]

    @impl true
    def exception(opts) do
      errors = Keyword.fetch!(opts, :errors)
      %__MODULE__{
        message: "Validation failed",
        errors: errors
      }
    end
  end
end

# Raise when appropriate
def get_user!(id) do
  case Repo.get(User, id) do
    nil -> raise MyApp.Errors.NotFound, resource: "User", id: id
    user -> user
  end
end

# Rescue in controllers
def show(conn, %{"id" => id}) do
  try do
    user = Accounts.get_user!(id)
    render(conn, :show, user: user)
  rescue
    MyApp.Errors.NotFound ->
      conn
      |> put_status(:not_found)
      |> json(%{error: "User not found"})
  end
end
```

---

## 7. Processes and OTP (MANDATORY)

### A. GenServer

```elixir
defmodule MyApp.Workers.OrderProcessor do
  @moduledoc """
  Background worker for processing orders.
  """

  use GenServer
  require Logger

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @spec process(Order.t()) :: :ok
  def process(order) do
    GenServer.cast(__MODULE__, {:process, order})
  end

  @spec get_stats() :: map()
  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    state = %{
      processed: 0,
      failed: 0,
      started_at: DateTime.utc_now()
    }
    {:ok, state}
  end

  @impl true
  def handle_cast({:process, order}, state) do
    case do_process(order) do
      :ok ->
        {:noreply, %{state | processed: state.processed + 1}}
      {:error, reason} ->
        Logger.error("Failed to process order #{order.id}: #{inspect(reason)}")
        {:noreply, %{state | failed: state.failed + 1}}
    end
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats = Map.take(state, [:processed, :failed, :started_at])
    {:reply, stats, state}
  end

  @impl true
  def handle_info(:timeout, state) do
    # Handle timeout
    {:noreply, state}
  end

  # Private

  defp do_process(order) do
    # Processing logic
    :ok
  end
end
```

### A2. GenServer Best Practices

```elixir
# ✅ CORRECT: Use handle_continue for expensive init work
# Avoids blocking the supervisor during init/1
@impl true
def init(opts) do
  {:ok, %{data: nil, opts: opts}, {:continue, :warm_cache}}
end

@impl true
def handle_continue(:warm_cache, state) do
  data = expensive_load(state.opts)
  {:noreply, %{state | data: data}}
end

# ✅ CORRECT: Use timeout of 0 to trigger immediate first action
@impl true
def init(opts) do
  {:ok, %{endpoint: opts[:endpoint]}, 0}
end

# Key rules:
# - Always annotate callbacks with @impl true
# - Separate client API from server callbacks clearly
# - Prefer GenServer over Agent when you need handle_info or complex logic
```

### B. Supervisor

```elixir
defmodule MyApp.Application do
  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Database
      MyApp.Repo,

      # PubSub
      {Phoenix.PubSub, name: MyApp.PubSub},

      # Process registry
      {Registry, keys: :unique, name: MyApp.Registry},

      # Background workers
      {MyApp.Workers.OrderProcessor, []},

      # Dynamic supervisor for on-demand processes
      {DynamicSupervisor, strategy: :one_for_one, name: MyApp.DynamicSupervisor},

      # Task supervisor for fire-and-forget async work
      {Task.Supervisor, name: MyApp.TaskSupervisor},

      # Telemetry
      MyApp.Telemetry,

      # Web endpoint (must be last)
      MyAppWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: MyApp.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @impl true
  def config_change(changed, _new, removed) do
    MyAppWeb.Endpoint.config_change(changed, removed)
    :ok
  end
end
```

```elixir
# Supervisor strategies:
# :one_for_one  - Only restart failed child (children are independent)
# :rest_for_one - Restart failed child + all started after it (ordered dependencies)
# :one_for_all  - Restart all children (tightly coupled processes)
#
# ✅ Always set max_restarts and max_seconds in production:
# opts = [strategy: :one_for_one, max_restarts: 5, max_seconds: 10]
```

### B3. DynamicSupervisor and Registry

```elixir
# DynamicSupervisor adds children on demand; Registry provides named lookup.
# Combine them for per-user/per-tenant processes.

# Start in supervisor: {DynamicSupervisor, name: MyApp.DynSup, strategy: :one_for_one}
# Start in supervisor: {Registry, keys: :unique, name: MyApp.Registry}

# Add children dynamically:
DynamicSupervisor.start_child(MyApp.DynSup, {MyApp.UserSession, user_id: id})

# Name processes via Registry:
defp via(user_id), do: {:via, Registry, {MyApp.Registry, {:session, user_id}}}
```

### C. Task and Task.Supervisor

```elixir
# Async work with results:
tasks = Enum.map(users, &Task.Supervisor.async_nolink(MyApp.TaskSupervisor, fn -> notify(&1) end))
results = Task.await_many(tasks, 5_000)

# Fire-and-forget (no result needed):
Task.Supervisor.start_child(MyApp.TaskSupervisor, fn -> AuditLog.record(user, action) end)

# Timeout handling with yield/shutdown:
case Task.yield(task, 5_000) || Task.shutdown(task) do
  {:ok, result} -> result
  nil -> {:error, :timeout}
end
```

---

## 8. Phoenix Best Practices (MANDATORY)

### A. Controllers

```elixir
defmodule MyAppWeb.UserController do
  use MyAppWeb, :controller

  alias MyApp.Accounts
  alias MyApp.Accounts.User

  action_fallback MyAppWeb.FallbackController

  def index(conn, params) do
    users = Accounts.list_users(params)
    render(conn, :index, users: users)
  end

  def show(conn, %{"id" => id}) do
    with {:ok, user} <- Accounts.get_user(id) do
      render(conn, :show, user: user)
    end
  end

  def create(conn, %{"user" => user_params}) do
    with {:ok, %User{} = user} <- Accounts.create_user(user_params) do
      conn
      |> put_status(:created)
      |> put_resp_header("location", ~p"/api/users/#{user}")
      |> render(:show, user: user)
    end
  end

  def update(conn, %{"id" => id, "user" => user_params}) do
    with {:ok, user} <- Accounts.get_user(id),
         {:ok, %User{} = user} <- Accounts.update_user(user, user_params) do
      render(conn, :show, user: user)
    end
  end

  def delete(conn, %{"id" => id}) do
    with {:ok, user} <- Accounts.get_user(id),
         {:ok, %User{}} <- Accounts.delete_user(user) do
      send_resp(conn, :no_content, "")
    end
  end
end

# FallbackController handles errors from action_fallback:
# {:error, :not_found} -> 404, {:error, %Changeset{}} -> 422
```

### B. LiveView

```elixir
defmodule MyAppWeb.UserLive.Index do
  use MyAppWeb, :live_view

  alias MyApp.Accounts
  alias MyApp.Accounts.User

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket) do
      Accounts.subscribe()
    end

    {:ok, stream(socket, :users, Accounts.list_users())}
  end

  @impl true
  def handle_params(params, _url, socket) do
    {:noreply, apply_action(socket, socket.assigns.live_action, params)}
  end

  defp apply_action(socket, :index, _params) do
    socket
    |> assign(:page_title, "Users")
    |> assign(:user, nil)
  end

  defp apply_action(socket, :new, _params) do
    socket
    |> assign(:page_title, "New User")
    |> assign(:user, %User{})
  end

  @impl true
  def handle_event("delete", %{"id" => id}, socket) do
    user = Accounts.get_user!(id)
    {:ok, _} = Accounts.delete_user(user)

    {:noreply, stream_delete(socket, :users, user)}
  end

  @impl true
  def handle_info({MyApp.Accounts, [:user, :created], user}, socket) do
    {:noreply, stream_insert(socket, :users, user)}
  end

  def handle_info({MyApp.Accounts, [:user, :updated], user}, socket) do
    {:noreply, stream_insert(socket, :users, user)}
  end
end
```

### B2. LiveView Lifecycle

```
mount/3 -> handle_params/3 -> render/1 -> (handle_event/3 | handle_info/2) -> render/1 -> ...
```

mount runs twice (static render + WebSocket connect). Use `connected?/1` to gate subscriptions and async work:

```elixir
@impl true
def mount(_params, _session, socket) do
  if connected?(socket), do: Phoenix.PubSub.subscribe(MyApp.PubSub, "orders")
  {:ok, assign(socket, loading: not connected?(socket), orders: [])}
end
```

### B3. LiveComponent Patterns

```elixir
# LiveComponents encapsulate reusable UI with their own state and events.
# Use phx-target={@myself} to route events to the component, not the parent.

defmodule MyAppWeb.Components.UserCard do
  use MyAppWeb, :live_component

  @impl true
  def update(assigns, socket) do
    {:ok, socket |> assign(assigns) |> assign_new(:editing, fn -> false end)}
  end

  @impl true
  def handle_event("edit", _, socket) do
    {:noreply, assign(socket, editing: true)}
  end
end

# Usage: <.live_component module={UserCard} id={"user-#{user.id}"} user={user} />
```

### B4. Streams for Large Collections

```elixir
# Streams send only changed DOM elements over the wire.
# Use them instead of assigns for any list that may grow large or change frequently.

@impl true
def mount(_params, _session, socket) do
  {:ok, stream(socket, :orders, Orders.list_recent(limit: 50))}
end

# Insert, update, or remove individual items without re-rendering the full list:
def handle_info({:order_created, order}, socket) do
  {:noreply, stream_insert(socket, :orders, order, at: 0)}
end

def handle_info({:order_deleted, order}, socket) do
  {:noreply, stream_delete(socket, :orders, order)}
end

# In template, use phx-update="stream" and iterate with @streams:
# <div id="orders" phx-update="stream">
#   <div :for={{dom_id, order} <- @streams.orders} id={dom_id}>...</div>
# </div>
```

### B5. JavaScript Hooks and pushEvent

```elixir
# JS hooks integrate client-side JS with LiveView (charts, maps, clipboard).
# Define hooks in app.js, pass to LiveSocket. Attach with phx-hook="Name".
# Push data from server: push_event(socket, "event-name", %{data: data})
# Receive on client: this.handleEvent("event-name", ({data}) => ...)
# Use phx-update="ignore" on hook elements so LiveView does not overwrite them.
```

### B6. File Uploads with LiveView

```elixir
# In mount: allow_upload(socket, :avatar, accept: ~w(.jpg .png), max_entries: 1)
# In handle_event("save"): consume_uploaded_entries(socket, :avatar, fn %{path: path}, entry -> ... end)
# Template: <.live_file_input upload={@uploads.avatar} />
# Preview:  <.live_img_preview entry={entry} />
```

---

## 9. Ecto Patterns (MANDATORY)

### A. Embedded Schemas

```elixir
# Embedded schemas define structured data without a database table (JSON columns, value objects).

defmodule MyApp.Accounts.Address do
  use Ecto.Schema
  import Ecto.Changeset

  @primary_key false
  embedded_schema do
    field :street, :string
    field :city, :string
    field :zip, :string
  end

  def changeset(address, attrs) do
    address |> cast(attrs, [:street, :city, :zip]) |> validate_required([:street, :city, :zip])
  end
end

# In parent schema: embeds_one :address, Address, on_replace: :update
# In changeset: cast_embed(:address, required: true)
```

### B. Schemaless Changesets

```elixir
# Validate data without a schema. Useful for search forms, API params, filters.

@types %{query: :string, category: :string, min_price: :decimal, page: :integer}

def validate_search(params) do
  {%{}, @types}
  |> cast(params, Map.keys(@types))
  |> validate_required([:query])
  |> validate_length(:query, min: 2)
  |> apply_action(:validate)
end
```

### C. Ecto.Multi for Transactions

```elixir
# Ecto.Multi composes operations into a single transaction. If any step fails, all roll back.

def place_order(user, items) do
  Ecto.Multi.new()
  |> Ecto.Multi.insert(:order, Order.changeset(%Order{}, %{user_id: user.id, status: :pending}))
  |> Ecto.Multi.run(:line_items, fn repo, %{order: order} ->
    entries = Enum.map(items, &%{order_id: order.id, product_id: &1.product_id, quantity: &1.qty})
    {count, _} = repo.insert_all(LineItem, entries)
    {:ok, count}
  end)
  |> Repo.transaction()
  |> case do
    {:ok, %{order: order}} -> {:ok, order}
    {:error, step, changeset, _} -> {:error, {step, changeset}}
  end
end
```

### D. Dynamic Queries

```elixir
# Build queries dynamically with Enum.reduce or Ecto.Query.dynamic/2.

defmodule MyApp.Products.Filters do
  import Ecto.Query

  def filter(params) do
    Product
    |> apply_filters(params)
    |> order_by([p], desc: p.inserted_at)
    |> limit(20)
  end

  defp apply_filters(query, params) do
    Enum.reduce(params, query, fn
      {"category", val}, q -> where(q, [p], p.category == ^val)
      {"min_price", val}, q -> where(q, [p], p.price >= ^val)
      {"search", val}, q -> where(q, [p], ilike(p.name, ^"%#{val}%"))
      _, q -> q
    end)
  end
end

# Use dynamic/2 for composable where clauses:
conditions = dynamic([p], p.active == true)
conditions = if name, do: dynamic([p], ^conditions and ilike(p.name, ^"%#{name}%")), else: conditions
from(p in Product, where: ^conditions)
```

### E. Custom Ecto Types

```elixir
# Implement Ecto.Type for custom serialization (e.g., encryption, money).
# Callbacks: type/0, cast/1, dump/1, load/1
defmodule MyApp.Types.EncryptedString do
  use Ecto.Type
  def type, do: :binary
  def cast(v) when is_binary(v), do: {:ok, v}
  def cast(_), do: :error
  def dump(v) when is_binary(v), do: {:ok, MyApp.Encryption.encrypt(v)}
  def dump(_), do: :error
  def load(v) when is_binary(v), do: {:ok, MyApp.Encryption.decrypt(v)}
  def load(_), do: :error
end
# Usage: field :email_encrypted, MyApp.Types.EncryptedString
```

### F. Multi-Tenancy with Prefixes

```elixir
# Use Ecto prefix option for PostgreSQL schema-based multi-tenancy.
# Override default_options/1 in Repo to set prefix from Process dictionary.
# Set tenant in Plug: MyApp.Repo.put_tenant(tenant_id)
# Migrations per tenant: Ecto.Migrator.run(Repo, path, :up, prefix: "tenant_#{id}")
```

### G. Migration Best Practices

```elixir
# ✅ CORRECT: Always write reversible migrations using change/0
defmodule MyApp.Repo.Migrations.AddOrdersTable do
  use Ecto.Migration

  def change do
    create table(:orders, primary_key: false) do
      add :id, :binary_id, primary_key: true
      add :number, :string, null: false
      add :status, :string, null: false, default: "pending"
      add :user_id, references(:users, type: :binary_id, on_delete: :restrict), null: false
      timestamps(type: :utc_datetime)
    end

    create unique_index(:orders, [:number])
    create index(:orders, [:user_id])
  end
end

# Safe migration rules:
# - Add columns as nullable or with a default (never NOT NULL without default)
# - Never rename columns in one step (add new, migrate data, drop old)
# - Use @disable_ddl_transaction true and concurrently: true for indexes on large tables
# - Use up/down instead of change when the migration is not reversible
```

### H. Repo Callbacks and Telemetry

Attach `:telemetry` handlers to `[:my_app, :repo, :query]` events for slow query logging and monitoring. Ecto emits telemetry automatically for every query with `total_time`, `queue_time`, and `decode_time` measurements.

---

## 10. Testing (MANDATORY)

### A. Unit Tests

```elixir
defmodule MyApp.AccountsTest do
  use MyApp.DataCase

  alias MyApp.Accounts
  alias MyApp.Accounts.User

  describe "users" do
    @valid_attrs %{email: "test@example.com", name: "Test User"}
    @invalid_attrs %{email: nil, name: nil}

    test "list_users/0 returns all users" do
      user = user_fixture()
      assert Accounts.list_users() == [user]
    end

    test "get_user/1 returns the user with given id" do
      user = user_fixture()
      assert Accounts.get_user(user.id) == {:ok, user}
    end

    test "get_user/1 returns error when user doesn't exist" do
      assert Accounts.get_user(Ecto.UUID.generate()) == {:error, :not_found}
    end

    test "create_user/1 with valid data creates a user" do
      assert {:ok, %User{} = user} = Accounts.create_user(@valid_attrs)
      assert user.email == "test@example.com"
      assert user.name == "Test User"
    end

    test "create_user/1 with invalid data returns error changeset" do
      assert {:error, %Ecto.Changeset{}} = Accounts.create_user(@invalid_attrs)
    end

    test "create_user/1 with duplicate email returns error" do
      user_fixture(email: "duplicate@example.com")

      assert {:error, changeset} =
        Accounts.create_user(%{@valid_attrs | email: "duplicate@example.com"})

      assert %{email: ["has already been taken"]} = errors_on(changeset)
    end
  end

  defp user_fixture(attrs \\ %{}) do
    {:ok, user} =
      attrs
      |> Enum.into(@valid_attrs)
      |> Accounts.create_user()

    user
  end
end
```

### B. Controller Tests

```elixir
defmodule MyAppWeb.UserControllerTest do
  use MyAppWeb.ConnCase

  setup %{conn: conn} do
    {:ok, conn: put_req_header(conn, "accept", "application/json")}
  end

  test "create and get user", %{conn: conn} do
    conn = post(conn, ~p"/api/users", user: %{email: "test@example.com", name: "Test"})
    assert %{"id" => id} = json_response(conn, 201)["data"]

    conn = get(conn, ~p"/api/users/#{id}")
    assert %{"email" => "test@example.com"} = json_response(conn, 200)["data"]
  end

  test "returns 422 for invalid data", %{conn: conn} do
    conn = post(conn, ~p"/api/users", user: %{email: nil})
    assert json_response(conn, 422)["errors"] != %{}
  end
end
```

### C. LiveView Tests

```elixir
defmodule MyAppWeb.UserLive.IndexTest do
  use MyAppWeb.ConnCase
  import Phoenix.LiveViewTest

  test "lists all users", %{conn: conn} do
    {:ok, user} = Accounts.create_user(%{email: "test@example.com", name: "Test"})
    {:ok, _live, html} = live(conn, ~p"/users")
    assert html =~ user.name
  end

  test "navigates and creates user", %{conn: conn} do
    {:ok, live, _html} = live(conn, ~p"/users/new")
    live |> form("#user-form", user: %{email: "new@example.com", name: "New"}) |> render_submit()
    assert_patch(live, ~p"/users")
    assert render(live) =~ "new@example.com"
  end

  test "deletes user", %{conn: conn} do
    {:ok, user} = Accounts.create_user(%{email: "del@example.com", name: "Del"})
    {:ok, live, _html} = live(conn, ~p"/users")
    live |> element("#user-#{user.id} a", "Delete") |> render_click()
    refute has_element?(live, "#user-#{user.id}")
  end
end
```

### D. Mox for Mock Dependencies

```elixir
# Mox defines mocks based on behaviours, ensuring mocks match the real interface.

# 1. Define behaviour:
defmodule MyApp.PaymentGateway do
  @callback charge(integer(), String.t()) :: {:ok, map()} | {:error, String.t()}
end

# 2. In test/support/mocks.ex: Mox.defmock(MyApp.MockPaymentGateway, for: MyApp.PaymentGateway)
# 3. In config/test.exs: config :my_app, payment_gateway: MyApp.MockPaymentGateway
# 4. In production code: @gateway Application.compile_env(:my_app, :payment_gateway)

# 5. In tests:
defmodule MyApp.OrdersTest do
  use MyApp.DataCase, async: true
  import Mox
  setup :verify_on_exit!

  test "charges the payment gateway" do
    MyApp.MockPaymentGateway
    |> expect(:charge, fn 5000, "tok_test" -> {:ok, %{id: "ch_123"}} end)

    assert {:ok, %{id: "ch_123"}} = Orders.charge_order(%{total_cents: 5000, payment_token: "tok_test"})
  end
end
```

### E. Property-Based Testing with StreamData

```elixir
# Property-based testing generates random inputs to find edge cases.
# Add {:stream_data, "~> 1.0"} and use ExUnitProperties.

property "valid emails always contain @" do
  check all local <- string(:alphanumeric, min_length: 1),
            domain <- string(:alphanumeric, min_length: 1),
            tld <- member_of(["com", "org", "net"]) do
    assert {:ok, _} = EmailValidator.validate("#{local}@#{domain}.#{tld}")
  end
end
```

### F. Database Sandbox and Async Tests

```elixir
# Ecto.Adapters.SQL.Sandbox wraps each test in a rolled-back transaction.
# Use async: true for speed. Share sandbox with spawned processes:
# Ecto.Adapters.SQL.Sandbox.allow(MyApp.Repo, self(), pid)

setup tags do
  pid = Ecto.Adapters.SQL.Sandbox.start_owner!(MyApp.Repo, shared: not tags[:async])
  on_exit(fn -> Ecto.Adapters.SQL.Sandbox.stop_owner(pid) end)
end
```

### G. Integration Testing with Wallaby

Wallaby drives a real browser for end-to-end testing. Use it sparingly for critical flows. Tag integration tests with `@moduletag :integration` and use `use Wallaby.Feature` with `visit/2`, `fill_in/3`, `click/2`, and `assert_has/2`.

---

## 11. Deployment with Releases (MANDATORY)

### A. Mix Releases Configuration

```elixir
# In mix.exs, define releases:
defp releases do
  [
    my_app: [
      include_executables_for: [:unix],
      applications: [runtime_tools: :permanent],
      steps: [:assemble, :tar]
    ]
  ]
end

# Build: MIX_ENV=prod mix release
# Start: _build/prod/rel/my_app/bin/my_app start
```

### B. Runtime Configuration (runtime.exs)

```elixir
# config/runtime.exs runs at release boot, not at compile time.
# All secrets and environment-specific values belong here.

import Config

if config_env() == :prod do
  database_url = System.get_env("DATABASE_URL") || raise "DATABASE_URL not set"
  secret_key_base = System.get_env("SECRET_KEY_BASE") || raise "SECRET_KEY_BASE not set"
  host = System.get_env("PHX_HOST") || raise "PHX_HOST not set"
  port = String.to_integer(System.get_env("PORT") || "4000")

  config :my_app, MyApp.Repo,
    url: database_url,
    pool_size: String.to_integer(System.get_env("POOL_SIZE") || "10")

  config :my_app, MyAppWeb.Endpoint,
    url: [host: host, port: 443, scheme: "https"],
    http: [ip: {0, 0, 0, 0, 0, 0, 0, 0}, port: port],
    secret_key_base: secret_key_base,
    server: true
end
```

### C. Release Migration Hooks

```elixir
# Run migrations without Mix: bin/my_app eval MyApp.Release.migrate
defmodule MyApp.Release do
  @app :my_app
  def migrate do
    Application.load(@app)
    for repo <- Application.fetch_env!(@app, :ecto_repos) do
      {:ok, _, _} = Ecto.Migrator.with_repo(repo, &Ecto.Migrator.run(&1, :up, all: true))
    end
  end
end
```

### D. Docker Builds

Use multi-stage Dockerfiles: build stage (hexpm/elixir, build-essential) runs `mix deps.get`, `mix compile`, `mix release`. Runtime stage (debian-slim) copies only the release. Copy `runtime.exs` after compile so secrets are not baked in. Run as non-root user.

### E. Health Check Endpoints

Add a `/health` endpoint that verifies database connectivity (`SELECT 1`) and returns 200/503 with version info. Load balancers and orchestrators use this for readiness checks.

---

## 12. Modern Ecosystem Patterns

### A. Telemetry for Instrumentation

```elixir
# Telemetry is the standard instrumentation library for the BEAM.
# Phoenix, Ecto, and most libraries emit telemetry events automatically.

defmodule MyApp.Telemetry do
  use Supervisor
  import Telemetry.Metrics

  def start_link(arg), do: Supervisor.start_link(__MODULE__, arg, name: __MODULE__)

  @impl true
  def init(_arg) do
    Supervisor.init([{:telemetry_poller, period: 10_000}], strategy: :one_for_one)
  end

  def metrics do
    [
      summary("phoenix.router_dispatch.stop.duration", tags: [:route], unit: {:native, :millisecond}),
      summary("my_app.repo.query.total_time", unit: {:native, :millisecond}),
      summary("vm.memory.total", unit: {:byte, :megabyte}),
      counter("my_app.orders.created.count")
    ]
  end
end

# Emit custom events: :telemetry.execute([:my_app, :orders, :created], %{duration: dur}, %{})
```

### B. Oban for Background Jobs

```elixir
# Oban provides reliable, persistent background job processing backed by PostgreSQL.
# Prefer Oban over GenServer-based workers for any job that must survive restarts.

defmodule MyApp.Workers.SendWelcomeEmail do
  use Oban.Worker, queue: :mailers, max_attempts: 3, unique: [period: 300]

  @impl Oban.Worker
  def perform(%Oban.Job{args: %{"user_id" => user_id}}) do
    user = MyApp.Accounts.get_user!(user_id)
    MyApp.Mailer.deliver_welcome(user)
  end
end

# Enqueue: %{user_id: id} |> SendWelcomeEmail.new() |> Oban.insert()
# Schedule: SendWelcomeEmail.new(%{user_id: id}, scheduled_at: ~U[...]) |> Oban.insert()

# Testing with Oban.Testing:
# use Oban.Testing, repo: MyApp.Repo
# assert_enqueued(worker: SendWelcomeEmail, args: %{user_id: user.id})
# assert :ok = perform_job(SendWelcomeEmail, %{user_id: user.id})
```

### C. Broadway for Data Processing Pipelines

```elixir
# Broadway builds concurrent data pipelines with back-pressure and batching.
# Built on GenStage. Use for consuming SQS, RabbitMQ, Kafka.
# Implement handle_message/3 (per-message processing) and handle_batch/4 (bulk ops).

defmodule MyApp.Pipeline.OrderEvents do
  use Broadway

  def start_link(_opts) do
    Broadway.start_link(__MODULE__,
      name: __MODULE__,
      producer: [module: {BroadwaySQS.Producer, queue_url: System.get_env("SQS_QUEUE_URL")}],
      processors: [default: [concurrency: 10]],
      batchers: [default: [batch_size: 50, batch_timeout: 1_000]]
    )
  end

  @impl true
  def handle_message(_, %Broadway.Message{data: data} = msg, _) do
    case Jason.decode(data) do
      {:ok, event} -> Broadway.Message.update_data(msg, fn _ -> event end)
      {:error, _} -> Broadway.Message.failed(msg, "invalid JSON")
    end
  end
end
```

### D. JSON Handling with Jason

```elixir
# Jason is the standard JSON library. Use @derive for automatic encoding.

defmodule MyApp.Money do
  @derive {Jason.Encoder, only: [:amount, :currency]}
  defstruct [:amount, :currency]
end

# Decode: Jason.decode!(json_string)
# Decode to atoms (trusted input only): Jason.decode!(str, keys: :atoms!)
# Encode: Jason.encode!(%{name: "Alice"})
# Phoenix uses Jason by default in 1.7+.
```

### E. Nx and Machine Learning

Nx (Numerical Elixir) brings tensors and numerical computing to the BEAM. Use `defn` for compiled numerical functions with GPU/TPU acceleration. Bumblebee provides pre-trained models for text, image, and audio classification via `Nx.Serving`.

### F. Ash Framework Awareness

Ash is an optional declarative, resource-oriented framework that generates CRUD, authorization, and API layers (GraphQL, JSON:API) from resource definitions. It is best suited for projects with complex domain models and multiple API surfaces. Evaluate whether your project benefits from its declarative approach vs. traditional Phoenix contexts.

---

## 13. Deployment Checklist

- [ ] Credo and Dialyzer pass with no warnings
- [ ] All tests passing, no IO.inspect/dbg in production code
- [ ] Supervisor trees configured with appropriate strategies
- [ ] Ecto queries optimized (preloads, indexes), background jobs for slow ops
- [ ] Secrets in runtime.exs, CSRF protection enabled
- [ ] Mix release builds/starts correctly, migrations via release commands
- [ ] Health check endpoint, telemetry, and monitoring configured

---

## 14. Quick Reference

```elixir
# Pattern matching
{:ok, value} = result
%{name: name} = user
[head | tail] = list

# Pipe operator
data |> transform() |> format()

# With for happy path
with {:ok, a} <- step1(),
     {:ok, b} <- step2(a) do
  {:ok, b}
end

# Enum functions
Enum.map(list, &func/1)
Enum.filter(list, &pred/1)
Enum.reduce(list, acc, &reducer/2)
Enum.find(list, &pred/1)

# String
String.trim(str)
String.downcase(str)
"Hello #{name}"

# List
[head | tail]
list ++ other
[item | list]

# Map
Map.get(map, key, default)
Map.put(map, key, value)
Map.merge(map1, map2)
%{map | key: new_value}
```

---

**Last Updated:** 2026-02-27
**Version:** 2.0
**Maintainer:** Elixir Team


**End of Elixir Development Guidelines**

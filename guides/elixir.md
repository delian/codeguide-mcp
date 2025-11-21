# Elixir Development Guidelines

This document provides mandatory standards for Elixir development, following OTP principles and community best practices.

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

## 3. Naming Conventions (MANDATORY)

### A. Variables and Functions

```elixir
# snake_case for variables and functions
user_name = "Alice"
def calculate_total(items), do: ...

# Predicate functions end with ?
def valid?(changeset), do: ...
def admin?(user), do: user.role == :admin

# Dangerous functions end with !
def get_user!(id), do: Repo.get!(User, id)
def create_user!(attrs), do: ...

# Private functions prefixed with do_ (optional convention)
defp do_process(data), do: ...

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
  ...
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
# ✅ CORRECT: case for branching
def process_status(order) do
  case order.status do
    :pending -> handle_pending(order)
    :processing -> handle_processing(order)
    :shipped -> handle_shipped(order)
    status -> {:error, {:unknown_status, status}}
  end
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

      # Background workers
      {MyApp.Workers.OrderProcessor, []},

      # Dynamic supervisor for on-demand processes
      {DynamicSupervisor, strategy: :one_for_one, name: MyApp.TaskSupervisor},

      # Web endpoint
      MyAppWeb.Endpoint
    ]

    opts = [strategy: :one_for_one, name: MyApp.Supervisor]
    Supervisor.start_link(children, opts)
  end
end

# Supervisor strategies:
# :one_for_one  - Only restart failed child
# :one_for_all  - Restart all children if one fails
# :rest_for_one - Restart failed child and all started after it
```

### C. Task and Agent

```elixir
# ✅ CORRECT: Task for async work
def send_notifications(users) do
  users
  |> Enum.map(fn user ->
    Task.Supervisor.async_nolink(MyApp.TaskSupervisor, fn ->
      send_notification(user)
    end)
  end)
  |> Task.await_many(5_000)
end

# ✅ CORRECT: Agent for simple state
defmodule MyApp.Counter do
  use Agent

  def start_link(initial_value) do
    Agent.start_link(fn -> initial_value end, name: __MODULE__)
  end

  def value do
    Agent.get(__MODULE__, & &1)
  end

  def increment do
    Agent.update(__MODULE__, &(&1 + 1))
  end
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

# Fallback controller for error handling
defmodule MyAppWeb.FallbackController do
  use MyAppWeb, :controller

  def call(conn, {:error, :not_found}) do
    conn
    |> put_status(:not_found)
    |> put_view(json: MyAppWeb.ErrorJSON)
    |> render(:"404")
  end

  def call(conn, {:error, %Ecto.Changeset{} = changeset}) do
    conn
    |> put_status(:unprocessable_entity)
    |> put_view(json: MyAppWeb.ChangesetJSON)
    |> render(:error, changeset: changeset)
  end
end
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

---

## 9. Testing (MANDATORY)

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

  alias MyApp.Accounts

  @create_attrs %{email: "test@example.com", name: "Test User"}
  @update_attrs %{name: "Updated Name"}
  @invalid_attrs %{email: nil}

  setup %{conn: conn} do
    {:ok, conn: put_req_header(conn, "accept", "application/json")}
  end

  describe "index" do
    test "lists all users", %{conn: conn} do
      conn = get(conn, ~p"/api/users")
      assert json_response(conn, 200)["data"] == []
    end
  end

  describe "create user" do
    test "renders user when data is valid", %{conn: conn} do
      conn = post(conn, ~p"/api/users", user: @create_attrs)
      assert %{"id" => id} = json_response(conn, 201)["data"]

      conn = get(conn, ~p"/api/users/#{id}")
      assert %{
        "id" => ^id,
        "email" => "test@example.com",
        "name" => "Test User"
      } = json_response(conn, 200)["data"]
    end

    test "renders errors when data is invalid", %{conn: conn} do
      conn = post(conn, ~p"/api/users", user: @invalid_attrs)
      assert json_response(conn, 422)["errors"] != %{}
    end
  end

  describe "update user" do
    setup [:create_user]

    test "renders user when data is valid", %{conn: conn, user: user} do
      conn = put(conn, ~p"/api/users/#{user}", user: @update_attrs)
      assert %{"id" => id} = json_response(conn, 200)["data"]

      conn = get(conn, ~p"/api/users/#{id}")
      assert %{"name" => "Updated Name"} = json_response(conn, 200)["data"]
    end
  end

  defp create_user(_) do
    {:ok, user} = Accounts.create_user(@create_attrs)
    %{user: user}
  end
end
```

---

## 10. Deployment Checklist

### Code Quality
- [ ] Credo passes with no warnings
- [ ] Dialyzer passes with no warnings
- [ ] All tests passing
- [ ] No IO.inspect or dbg in production code

### OTP
- [ ] Supervisor trees properly configured
- [ ] Restart strategies appropriate
- [ ] Process names registered correctly

### Performance
- [ ] Ecto queries optimized (preloads, indexes)
- [ ] Background jobs for slow operations
- [ ] Caching where appropriate

### Security
- [ ] Input validation on all public interfaces
- [ ] Authentication/authorization in place
- [ ] Secrets in runtime.exs, not code

---

## 11. Quick Reference

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

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Elixir Team

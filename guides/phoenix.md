# Phoenix Framework Guidelines
Mandatory standards for Phoenix web apps: contexts, controllers, LiveView, Channels, and Ecto on top of idiomatic Elixir/OTP. Phoenix 1.7/1.8, LiveView 1.0, Ecto 3.12, Elixir 1.18+.

---
name: phoenix
title: Phoenix Framework Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [phoenix@1.8, phoenix_live_view@1.0, ecto@3.12, elixir@1.18, postgrex, bandit]
requires:
  - elixir
  - tdd
  - secure-coding
recommends:
  - rest
  - websocket
  - sql
  - error-handling
  - observability
provides:
  - phoenix-contexts
  - liveview
  - channels
  - ecto
  - ecto-multi
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Phoenix and Ecto — the language, OTP, and supervision idioms it builds on live in [`elixir.md`](guides://elixir.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Phoenix code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`elixir.md`](guides://elixir.md) — the language, OTP, supervision, tagged tuples, `with`, releases/`runtime.exs`. *Do not restate Elixir idioms here.* A context API returns `{:ok, _}`/`{:error, _}`; the `Endpoint`, `Repo`, and `Phoenix.PubSub` are children of the app supervision tree (see `elixir.md` §7).
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, coverage. *(Phoenix binding: `ConnCase`, `DataCase`, `Phoenix.LiveViewTest`, `Ecto.Adapters.SQL.Sandbox`; runner `mix test`.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, injection, CSRF. *(Phoenix binding: changeset `cast` allowlists stop mass-assignment; the `:browser` pipeline's `protect_from_forgery` provides CSRF tokens — see §4, §7.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — resource modelling, status codes, versioning. *(Binding: controllers + `action_fallback`, JSON views.)*
> - [`websocket.md`](guides://websocket.md) — the realtime transport, lifecycle, heartbeats, backpressure. *(Binding: both Channels and LiveView ride a Phoenix socket — see §5, §6.)*
> - [`sql.md`](guides://sql.md) — query design, indexing, N+1, transactions, migrations safety. *(Binding: Ecto schemas/queries/`Multi` — see §7.)*
> - [`error-handling.md`](guides://error-handling.md) — error vs exception model. *(Binding: `action_fallback` maps `{:error, _}` to HTTP; LiveView `{:error, changeset}` re-renders the form.)*
> - [`observability.md`](guides://observability.md) — metrics, tracing, health. *(Binding: Phoenix/Ecto/LiveView emit `:telemetry` automatically — see §8.)*

> 📎 **SEE ALSO:** [`postgresql.md`](guides://postgresql.md) · [`graphql.md`](guides://graphql.md) · [`oauth.md`](guides://oauth.md) *(for `mix phx.gen.auth` / token auth)* · [`microservices.md`](guides://microservices.md)

---

## 1. Core Philosophies: PHOENIX-FIRST

Phoenix/Ecto-specific principles only. Functional core, OTP, error strategy, and testing come from §0.

- **P**hoenix is the boundary, not the application: business logic lives in **contexts**; web modules (controllers, LiveViews, channels) only translate transport ↔ context calls.
- **H**TML-over-the-wire first: prefer LiveView with **streams** + server-rendered function components over hand-written SPA glue; reach for JS hooks only at the edges (maps, charts, clipboard).
- **O**ne write, one transaction: any multi-step persistence runs inside `Ecto.Multi`/`Repo.transaction` so partial writes can never escape.
- **E**xplicit data shape: every external map becomes a **changeset** with an allowlisted `cast` before it touches the DB — that is also the mass-assignment defence (policy: `secure-coding.md`).
- **N**o N+1: associations are `preload`ed (or joined) deliberately; the query count for a request is bounded and known (policy: `sql.md`).
- **I**nstrumented by default: Phoenix, Ecto, and LiveView emit `:telemetry`; wire metrics rather than ad-hoc logging (policy: `observability.md`).
- **X**-verified routes: build paths with the `~p` sigil so the compiler fails on dead links.

**Verified Code**: Agent-generated Phoenix MUST pass every gate in §2 (and the inherited [`elixir.md`](guides://elixir.md) gates) before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `PHX-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner. These are **in addition to** the `EX-*` gates from [`elixir.md`](guides://elixir.md).

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| PHX-CTX-01 | Web modules MUST NOT call `Repo` directly; all data access goes through a context | `grep -rE "Repo\.(all|get|insert|update|delete)" lib/*_web` | 0 matches |
| PHX-CTX-02 | Cross-context calls MUST go through the public context module, not its schemas/internals (see `elixir.md`) | review / `boundary` | no internal reach-in |
| PHX-ROUTE-01 | Paths/URLs MUST use verified routes (`~p`), not legacy `Routes.*_path` helpers | `grep -rE "Routes\.\w+_(path\|url)" lib/` + `mix compile --warnings-as-errors` | 0 matches, 0 warnings |
| PHX-SEC-01 | Browser pipeline MUST keep `:protect_from_forgery` (CSRF) and `put_secure_browser_headers` (see `secure-coding.md`) | review `router.ex` `:browser` pipeline | both plugs present |
| PHX-SEC-02 | Mass assignment MUST be blocked: every `cast/3` lists explicit fields; no user-controlled allowlist (see `secure-coding.md`) | review changesets | no `cast(.., Map.keys(...))` of untrusted input |
| PHX-ECTO-01 | No N+1: associations rendered in a view/template MUST be `preload`ed or joined (see `sql.md`) | review + Ecto telemetry query count | bounded query count |
| PHX-ECTO-02 | Multi-step writes MUST use `Ecto.Multi`/`Repo.transaction`, not sequential unguarded calls | review | atomic transaction |
| PHX-ECTO-03 | Migrations MUST be reversible and safe (nullable/defaulted adds; `concurrently` indexes on large tables) (see `sql.md`) | `mix ecto.migrate && mix ecto.rollback` | both succeed |
| PHX-LV-01 | LiveView collections that grow/change MUST use `stream/4` (not list assigns) with `phx-update="stream"` | review `*_live.ex` + templates | streams for lists |
| PHX-LV-02 | `mount/3` MUST gate subscriptions/expensive work behind `connected?/1` | review `mount/3` | no work on static render |
| PHX-CHAN-01 | Channel `join/3` MUST authorize the topic; payloads from clients MUST be validated before use (see `secure-coding.md`) | review channel modules | authz + validation present |
| PHX-TST-01 | Contexts, controllers, and LiveViews MUST have tests (`DataCase`/`ConnCase`/`LiveViewTest`) (see `tdd.md`) | `mix test` | exit 0, 0 skips |
| PHX-OBS-01 | A `Telemetry` supervisor MUST export Phoenix/Ecto/VM metrics (see `observability.md`) | review `telemetry.ex` | metrics defined |

> **Forbidden**: calling `Repo` from a controller/LiveView, fat web modules with business logic, `cast`ing an attacker-controlled field list, rendering associations without preload (N+1), sequential writes that should be one `Ecto.Multi`, legacy path helpers when `~p` is available, leaving `IO.inspect`/`dbg` in delivered code (inherits `EX-*`).

---

## 3. Contexts: the application boundary

A **context** is a bounded domain module (`MyApp.Accounts`, `MyApp.Orders`) that is the *only* public entry point to its data and rules. Web modules call the context; the context owns schemas, changesets, and `Repo` access. This is the hexagonal boundary applied to Phoenix (layering policy: [`elixir.md`](guides://elixir.md) → `hexagonal.md`).

```
lib/
├── my_app/                    # the application — no web/HTTP knowledge
│   ├── accounts.ex            # context module = public API (PHX-CTX-01/02)
│   ├── accounts/user.ex       # schema + changesets (internal)
│   ├── orders.ex
│   └── orders/{order,line_item}.ex
└── my_app_web/                # the boundary — translates transport ↔ context
    ├── router.ex
    ├── controllers/  components/  (core_components.ex)
    ├── live/                  # LiveViews & LiveComponents
    └── channels/
```

The context returns Elixir-native results (`{:ok, %User{}}` / `{:error, %Ecto.Changeset{}}`); it never returns a `conn` or a socket. Generators (`mix phx.gen.context`, `mix phx.gen.live`, `mix phx.gen.json`) scaffold this shape — keep it.

---

## 4. Web layer: Router & Controllers

### A. Router & pipelines
Keep CSRF and secure headers in the `:browser` pipeline (PHX-SEC-01); APIs use a separate `:api` pipeline. Scope routes; build links with `~p`.

```elixir
pipeline :browser do
  plug :accepts, ["html"]
  plug :fetch_session
  plug :fetch_live_flash
  plug :put_root_layout, html: {MyAppWeb.Layouts, :root}
  plug :protect_from_forgery            # CSRF — PHX-SEC-01
  plug :put_secure_browser_headers      # CSP/HSTS/etc — PHX-SEC-01
end

pipeline :api, do: plug :accepts, ["json"]

scope "/api", MyAppWeb do
  pipe_through :api
  resources "/users", UserController, except: [:new, :edit]
end
```

### B. Controllers (JSON/HTML APIs)
Thin: parse params → call context → render. Map `{:error, _}` centrally with `action_fallback` (REST semantics: [`rest.md`](guides://rest.md); error model: [`error-handling.md`](guides://error-handling.md)).

```elixir
defmodule MyAppWeb.UserController do
  use MyAppWeb, :controller
  alias MyApp.Accounts
  action_fallback MyAppWeb.FallbackController   # {:error, :not_found}→404, %Changeset{}→422

  def show(conn, %{"id" => id}) do
    with {:ok, user} <- Accounts.get_user(id), do: render(conn, :show, user: user)
  end

  def create(conn, %{"user" => params}) do
    with {:ok, user} <- Accounts.create_user(params) do      # params cast/allowlisted in context
      conn
      |> put_status(:created)
      |> put_resp_header("location", ~p"/api/users/#{user}")  # PHX-ROUTE-01
      |> render(:show, user: user)
    end
  end
end
```

Use `Req` for outbound HTTP (the modern default; `HTTPoison`/`Tesla` only if already in the project). Render with function components / `core_components.ex`, not string concatenation.

---

## 5. LiveView (1.0)

Server-rendered, stateful UI over a WebSocket. The transport's lifecycle, heartbeats, and backpressure are owned by [`websocket.md`](guides://websocket.md); this section owns the LiveView programming model.

### A. Lifecycle
```
mount/3 → handle_params/3 → render/1 → (handle_event/3 | handle_info/2 | handle_async/3) → render/1 → …
```
`mount/3` runs **twice**: once for the static HTTP render, then again after the WebSocket connects. Gate subscriptions and expensive loads behind `connected?/1` (PHX-LV-02).

```elixir
defmodule MyAppWeb.OrderLive.Index do
  use MyAppWeb, :live_view
  alias MyApp.Orders

  @impl true
  def mount(_params, _session, socket) do
    if connected?(socket), do: Orders.subscribe()          # PHX-LV-02 — only on the live socket
    {:ok, stream(socket, :orders, Orders.list_recent(limit: 50))}   # PHX-LV-01
  end

  @impl true
  def handle_event("delete", %{"id" => id}, socket) do
    {:ok, order} = Orders.delete_order(id)
    {:noreply, stream_delete(socket, :orders, order)}
  end

  @impl true
  def handle_info({:order_created, order}, socket) do        # PubSub broadcast
    {:noreply, stream_insert(socket, :orders, order, at: 0)}
  end
end
```

### B. Streams over `temporary_assigns`
Use `stream/4` for any list that grows or changes (PHX-LV-01) — it ships only changed DOM nodes and bounds server memory. `temporary_assigns` is legacy; do not introduce it in new code.

```heex
<div id="orders" phx-update="stream">
  <div :for={{dom_id, order} <- @streams.orders} id={dom_id}>{order.number}</div>
</div>
```

### C. Function components & LiveComponents
- **Function components** (`attr`/`slot` + `~H`) are the default reusable unit; shared ones live in `core_components.ex`. Prefer them over LiveComponents.
- **LiveComponents** (`use ..., :live_component`) only when a fragment needs its *own* state/events — route events with `phx-target={@myself}` and seed defaults via `assign_new/3`.

### D. Forms, uploads, JS hooks
- **Forms**: drive with `to_form/2` + `<.form for={@form}>` and `<.input field={@form[:email]} />`; on submit, the context changeset returns `{:error, changeset}` → re-`assign` the form to show errors.
- **Uploads**: `allow_upload(socket, :avatar, accept: ~w(.jpg .png), max_entries: 1)` in mount; `consume_uploaded_entries/3` on save; `<.live_file_input upload={@uploads.avatar} />` + `<.live_img_preview entry={entry} />`.
- **JS hooks** for client-only widgets: `phx-hook="Chart"` (registered on the `LiveSocket`); server→client via `push_event/3`, client→server via `pushEvent`. Put `phx-update="ignore"` on hook-owned DOM so LiveView won't clobber it. Use the `Phoenix.LiveView.JS` module for client-side transitions/toggles without a round trip.
- **Async**: `assign_async/3` and `start_async/3` (1.0) for non-blocking loads with built-in loading/error states.

---

## 6. Channels, PubSub & Presence

For bidirectional realtime messaging outside LiveView. Transport lifecycle, heartbeats, and flow control: [`websocket.md`](guides://websocket.md). Phoenix binding:

- **Channels**: `join/3` MUST authorize the topic and return `{:ok, socket}` or `{:error, reason}` (PHX-CHAN-01); validate every inbound payload before acting (it is attacker-controlled — `secure-coding.md`). Push with `push/3`, broadcast with `broadcast/3`; do heavy work in a `Task`, not in `handle_in/3`.
- **PubSub** (`Phoenix.PubSub`, a child of the app tree): contexts `broadcast` domain events on a topic; LiveViews/channels `subscribe` in `mount`/`join`. This decouples the writer from the realtime fan-out.
- **Presence** (`Phoenix.Presence`): CRDT-based tracking of who is on a topic (online lists, typing indicators) — `track/3` on join, `list/1` to render.

```elixir
defmodule MyAppWeb.RoomChannel do
  use MyAppWeb, :channel

  @impl true
  def join("room:" <> id, _payload, socket) do
    if MyApp.Chat.member?(socket.assigns.user_id, id),     # PHX-CHAN-01 authz
      do: {:ok, assign(socket, :room_id, id)},
      else: {:error, %{reason: "unauthorized"}}
  end

  @impl true
  def handle_in("msg", %{"body" => body}, socket) when is_binary(body) do
    case MyApp.Chat.post(socket.assigns.room_id, body) do  # validation in context
      {:ok, msg} -> broadcast!(socket, "msg", %{body: msg.body}); {:reply, :ok, socket}
      {:error, _} -> {:reply, {:error, %{reason: "invalid"}}, socket}
    end
  end
end
```

---

## 7. Ecto (3.12)

Ecto is the DB layer. Query design, indexing, transaction semantics, and migration safety are owned by [`sql.md`](guides://sql.md); below is the Ecto-specific binding.

### A. Schemas & changesets
A changeset's `cast/3` allowlist is both validation and the mass-assignment boundary (PHX-SEC-02). Push DB-enforced rules through `*_constraint` so races surface as changeset errors, not crashes.

```elixir
schema "users" do
  field :email, :string
  field :role, Ecto.Enum, values: [:admin, :user], default: :user
  has_many :orders, MyApp.Orders.Order
  timestamps(type: :utc_datetime)
end

def changeset(user, attrs) do
  user
  |> cast(attrs, [:email, :role])        # explicit allowlist — PHX-SEC-02
  |> validate_required([:email])
  |> validate_format(:email, ~r/@/)
  |> unique_constraint(:email)           # backed by a DB unique index
end
```

- **Embedded schemas** (`embedded_schema` + `embeds_one`/`cast_embed`) for value objects / JSON columns with no table.
- **Schemaless changesets** (`{%{}, %{field: :type}} |> cast(...)`) to validate search/filter params without a schema.

### B. Queries, associations & N+1
Build queries with the pipe-able `Ecto.Query` API; bind user values with `^` (never string-interpolate — injection, `secure-coding.md`). Render-time associations MUST be `preload`ed or joined (PHX-ECTO-01).

```elixir
def list_active_users(opts) do
  User
  |> where([u], u.active == true)
  |> maybe_filter_role(opts[:role])
  |> order_by([u], desc: u.inserted_at)
  |> preload(:orders)                    # avoids N+1 — PHX-ECTO-01
  |> Repo.all()
end

defp maybe_filter_role(q, nil), do: q
defp maybe_filter_role(q, role), do: where(q, [u], u.role == ^role)
```

Compose conditional filters with `Ecto.Query.dynamic/2` or `Enum.reduce/3`. For bulk work use `insert_all`/`update_all`; stream large result sets with `Repo.stream/2` inside a transaction.

### C. Ecto.Multi — atomic multi-step writes
Any write touching more than one row/table goes through `Ecto.Multi` so a later failure rolls back everything (PHX-ECTO-02). Steps see prior results; the whole thing runs in one `Repo.transaction`.

```elixir
def place_order(user, items) do
  Ecto.Multi.new()
  |> Ecto.Multi.insert(:order, Order.changeset(%Order{}, %{user_id: user.id, status: :pending}))
  |> Ecto.Multi.run(:line_items, fn repo, %{order: order} ->
    entries = Enum.map(items, &%{order_id: order.id, product_id: &1.product_id, quantity: &1.qty})
    {:ok, repo.insert_all(LineItem, entries)}
  end)
  |> Repo.transaction()
  |> case do
    {:ok, %{order: order}} -> {:ok, order}
    {:error, step, value, _changes} -> {:error, {step, value}}
  end
end
```

### D. Custom Ecto types & migrations
- **Custom types** (`use Ecto.Type` → `type/0`, `cast/1`, `dump/1`, `load/1`) for transparent (de)serialization — encrypted fields, money, value objects. `field :secret, MyApp.Types.EncryptedString`.
- **Migrations** are reversible (`change/0`, or `up/0`+`down/0` when not auto-reversible) and safe (PHX-ECTO-03): add columns nullable/with default, never rename in one step (add → backfill → drop), and use `@disable_ddl_transaction true` + `create index(..., concurrently: true)` on large tables. Run them in prod via a release `eval` module (see `elixir.md`), not `mix`.

### E. Multi-tenancy via `prefix`
For PostgreSQL schema-per-tenant isolation, pass `:prefix` on queries (`Repo.all(query, prefix: "tenant_42")`) or set a default in `Repo.default_options/1` from the process dictionary, populated by a plug at request entry. Migrate each tenant with `Ecto.Migrator.run(Repo, path, :up, prefix: "tenant_#{id}")`. (Deeper schema/RLS trade-offs: [`postgresql.md`](guides://postgresql.md).)

---

## 8. Telemetry & background work

- **Telemetry** (observability binding): Phoenix, Ecto, and LiveView emit `:telemetry` events automatically (router dispatch, `[:my_app, :repo, :query]` with `total_time`/`queue_time`/`decode_time`, mount/handle_event durations). A `Telemetry` supervisor with `Telemetry.Metrics` + `telemetry_poller` exports them (PHX-OBS-01); export targets and SLI policy are owned by [`observability.md`](guides://observability.md). Emit domain events with `:telemetry.execute([:my_app, :orders, :created], %{count: 1}, meta)`.
- **Background jobs**: prefer **Oban** (PostgreSQL-backed, survives restarts, retriable, testable via `Oban.Testing`) over bespoke GenServer queues for durable work; **Broadway** for back-pressured ingestion pipelines (SQS/Kafka/RabbitMQ). The OTP/worker primitives themselves are owned by [`elixir.md`](guides://elixir.md).
- **Health check**: expose `/health` (or `/up`, the 1.8 default) that verifies DB connectivity (`SELECT 1`) for load-balancer readiness.

Deployment, releases, and `runtime.exs` secret loading are owned by [`elixir.md`](guides://elixir.md) — Phoenix only adds: read `SECRET_KEY_BASE`, `PHX_HOST`, and the endpoint URL/port there, and set `server: true` in the prod endpoint config.

---

## 9. Testing binding

Test *policy* (Red-Green-Refactor, pyramid, coverage) is owned by [`tdd.md`](guides://tdd.md). Phoenix binding (PHX-TST-01):

- **`MyApp.DataCase`** for contexts/Ecto — wraps each test in a sandboxed, rolled-back transaction (`Ecto.Adapters.SQL.Sandbox`); `async: true` when no shared global state, `Sandbox.allow/3` to share the connection with spawned processes.
- **`MyAppWeb.ConnCase`** for controllers — `get`/`post` with `~p` paths, assert on `json_response/2`.
- **`Phoenix.LiveViewTest`** for LiveViews — `live/2`, `render_click`, `form |> render_submit`, `has_element?/2`, `assert_patch/2`.
- **`Phoenix.ChannelTest`** for channels — `subscribe_and_join`, `push`, `assert_broadcast`/`assert_push`.
- Mock *behaviours/ports* with **Mox** (not concrete modules); property-test pure functions with **StreamData**.

---

## 10. Deployment Checklist

Generated from §2 — one box per requirement ID. (Inherits the [`elixir.md`](guides://elixir.md) `EX-*` checklist too.)

- [ ] PHX-CTX-01/02 — web layer free of `Repo`; cross-context calls via public modules only
- [ ] PHX-ROUTE-01 — `~p` verified routes everywhere; no legacy path helpers; compiles warning-free
- [ ] PHX-SEC-01 — `:browser` pipeline keeps CSRF + secure headers
- [ ] PHX-SEC-02 — every `cast/3` uses an explicit field allowlist (no mass assignment)
- [ ] PHX-ECTO-01 — rendered associations preloaded/joined (no N+1)
- [ ] PHX-ECTO-02 — multi-step writes wrapped in `Ecto.Multi`/transaction
- [ ] PHX-ECTO-03 — migrations reversible and safe (`migrate` + `rollback` both pass)
- [ ] PHX-LV-01 — collections use `stream/4` + `phx-update="stream"`
- [ ] PHX-LV-02 — `mount/3` gates subscriptions/heavy work behind `connected?/1`
- [ ] PHX-CHAN-01 — channels authorize `join/3` and validate client payloads
- [ ] PHX-TST-01 — contexts, controllers, and LiveViews tested; `mix test` green
- [ ] PHX-OBS-01 — `Telemetry` supervisor exports Phoenix/Ecto/VM metrics
- [ ] Agent ran every gate (and the inherited `elixir.md` gates) and documented any fixes

---
**End of Phoenix Framework Guidelines**

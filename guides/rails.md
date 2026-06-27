# Ruby on Rails Guidelines
Mandatory standards for Ruby on Rails apps: convention-over-configuration, fat-model/skinny-controller, secure-by-default, N+1-free. Rails 7.2/8.0, Ruby 3.3+, RSpec, RuboCop-Rails, Brakeman, Solid Queue/Cache/Cable.

---
name: rails
title: Ruby on Rails Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: framework
tools: [rails@8.0, ruby@3.3, rspec-rails, rubocop-rails, brakeman, bundler-audit, solid_queue, solid_cache, solid_cable, turbo-rails]
requires:
  - ruby
  - tdd
  - secure-coding
recommends:
  - rest
  - sql
  - error-handling
  - hexagonal
  - observability
provides:
  - rails-mvc
  - activerecord
  - rails-conventions
  - rails-security
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Rails as a framework. The Ruby **language** (syntax, blocks, modules, exceptions, typing, Bundler) is owned by [`ruby.md`](guides://ruby.md) and is **not** repeated here.

---

## 0. Prerequisites & References

Fetch and apply these **before** generating Rails code. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`ruby.md`](guides://ruby.md) — the language: idioms, blocks/procs, modules/mixins, exceptions, Sorbet/RBS, Bundler, RuboCop. **Rails is a Ruby app first.**
> - [`tdd.md`](guides://tdd.md) — test-first, Red-Green-Refactor, regression-test-before-fix, coverage. *(Rails binding: `bundle exec rspec`; model/request/system specs; FactoryBot over fixtures.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, injection, CSRF. *(Rails binding: strong parameters, `brakeman`, encrypted credentials, `bundle-audit`.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`rest.md`](guides://rest.md) — HTTP semantics, status codes, resource design for controllers/JSON APIs.
> - [`sql.md`](guides://sql.md) — query depth, indexing, transactions, normalization behind ActiveRecord and migrations.
> - [`error-handling.md`](guides://error-handling.md) — error strategy *(binding: `rescue_from`, custom error pages)*.
> - [`hexagonal.md`](guides://hexagonal.md) — keeping domain logic out of framework objects (service objects, POROs).
> - [`observability.md`](guides://observability.md) — metrics/tracing *(binding: `ActiveSupport::Notifications`, lograge)*.

> 📎 **SEE ALSO:** [`postgresql.md`](guides://postgresql.md) · [`redis.md`](guides://redis.md) *(only if you opt out of the Solid adapters)* · [`oauth.md`](guides://oauth.md) · [`websocket.md`](guides://websocket.md) *(ActionCable transport)* · [`code-review.md`](guides://code-review.md) · [`ci-cd.md`](guides://ci-cd.md)

---

## 1. Core Philosophies: RAILS-FIRST

Rails-specific principles only. Ruby idioms come from [`ruby.md`](guides://ruby.md); TDD, security, and architecture come from §0.

- **R**ESTful resources: model the domain as resources; `resources :x` routing, the seven standard actions, nested only one level deep.
- **A**ctiveRecord discipline: associations carry intent; every query is N+1-free by default (`includes`/`preload`); push set logic into scopes, never N round-trips in Ruby.
- **I**ntegrity at the database: constraints (NOT NULL, FK, unique indexes) live in migrations, not only in model validations; migrations are reversible.
- **L**ean controllers, rich domain: controllers parse/authorize/respond only; business logic lives in models, concerns, or service-object POROs (see `hexagonal.md`).
- **S**ecure by default: strong parameters on every write, CSRF on, credentials encrypted, Brakeman clean — never disable a protection to "make it work".

**Verified Code**: Agent-generated Rails code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `RAILS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| RAILS-TST-01 | Every feature MUST have request/model specs, written test-first (see `tdd.md`, `ruby.md`) | `bundle exec rspec` | exit 0, 0 pending |
| RAILS-TST-02 | Critical user journeys MUST have a system spec (Capybara) | `bundle exec rspec spec/system` | exit 0 |
| RAILS-SEC-01 | Every controller write MUST use strong parameters; no `permit!` on user input (see `secure-coding.md`) | `bundle exec brakeman -q` + review | 0 mass-assignment warnings |
| RAILS-SEC-02 | 0 high/medium Brakeman findings (SQLi, XSS, CSRF, mass-assignment) (see `secure-coding.md`) | `bundle exec brakeman -q -w2` | exit 0 |
| RAILS-SEC-03 | Secrets MUST be in encrypted credentials/ENV, never in code or `config/*.yml` (see `secure-coding.md`) | `bundle exec brakeman` + grep | 0 secrets in repo |
| RAILS-AR-01 | No raw string interpolation in queries; bind parameters or hash conditions (see `sql.md`) | `bundle exec brakeman` (SQLi) | 0 SQLi warnings |
| RAILS-AR-02 | Controller/view collections MUST eager-load to avoid N+1 | Bullet in test env / prosopite | 0 N+1 notifications |
| RAILS-MIG-01 | Every migration MUST be reversible and add an index to each foreign key | `bin/rails db:migrate && db:rollback` | up+down clean |
| RAILS-MIG-02 | DB-level integrity MUST exist for every uniqueness/required rule (unique index, NOT NULL, FK) | review migration + `schema.rb` | constraints present |
| RAILS-STRUCT-01 | Controllers MUST stay thin; multi-step business logic lives in models/POROs (see `hexagonal.md`) | review / `rubocop -c` ABC size | no fat actions |
| RAILS-LINT-01 | `rubocop-rails` cops MUST pass clean (see `ruby.md`) | `bundle exec rubocop` | exit 0 |
| RAILS-DEP-01 | `Gemfile.lock` committed & 0 gem CVEs (see `secure-coding.md`, `ruby.md`) | `bundle exec bundle-audit check --update` | 0 advisories |

> **Forbidden**: disabling CSRF (`skip_forgery_protection`) on browser endpoints, `params.permit!`, interpolating user input into `where`/`order`, `update_all`/`delete_all` bypassing callbacks on user-facing writes without review, committing unencrypted secrets, or shipping a migration that cannot roll back.

---

## 3. Verification Protocol

Run, in order, before presenting code. Fix → re-run until every gate is green. Ruby-level gates (full `rubocop`, `bundle-audit`, type check) come from [`ruby.md`](guides://ruby.md) §3.

```bash
bin/rails db:migrate && bin/rails db:rollback STEP=1 && bin/rails db:migrate  # RAILS-MIG-01
bundle exec rubocop                          # RAILS-LINT-01 (incl. rubocop-rails)
bundle exec brakeman -q -w2                   # RAILS-SEC-01/02/03, RAILS-AR-01
bundle exec rspec                             # RAILS-TST-01/02 (Bullet raises on N+1 → RAILS-AR-02)
bundle exec bundle-audit check --update       # RAILS-DEP-01
```

The *why* behind each gate lives in its §0 owner; do not re-derive it here.

---

## 4. Project Structure & MVC

Rails' convention-over-configuration layout. Follow the conventions and you get routing, autoloading (Zeitwerk), and wiring for free.

```
app/
├── controllers/      # thin: parse params, authorize, call domain, respond (RAILS-STRUCT-01)
│   └── concerns/     # shared controller behaviour (mixins — see ruby.md)
├── models/           # ActiveRecord + POROs; associations, scopes, validations
│   └── concerns/     # shared model behaviour (ActiveSupport::Concern)
├── services/         # service-object POROs for multi-step business logic (see hexagonal.md)
├── jobs/             # ActiveJob classes (background work)
├── mailers/          # ActionMailer
├── views/            # ERB/Turbo templates; helpers in app/helpers
├── channels/         # ActionCable (WebSocket — see websocket.md)
└── javascript/       # Hotwire (Turbo + Stimulus), importmap or jsbundling
config/
├── routes.rb         # RESTful resource routing
├── credentials.yml.enc + master.key (git-ignored)   # encrypted secrets (RAILS-SEC-03)
└── environments/     # per-env config
db/
├── migrate/          # reversible migrations (RAILS-MIG-01)
└── schema.rb         # authoritative schema (committed)
spec/                 # model / request / system specs (see tdd.md)
```

- **MVC boundary**: Model owns data + business rules, Controller orchestrates, View renders. No SQL or domain logic in views; no view/HTML concerns in models.
- Extract cross-cutting model/controller behaviour into **concerns** (`extend ActiveSupport::Concern`); extract multi-model workflows into **service objects** (plain Ruby `Result`-returning POROs) to keep models from becoming god objects.

---

## 5. Rails Specifics

The unique value of this guide.

### A. Routing — RESTful resources

```ruby
# config/routes.rb
Rails.application.routes.draw do
  resources :articles do
    resources :comments, only: %i[create destroy]   # nest one level only
  end
  namespace :api do
    namespace :v1 do
      resources :users, only: %i[index show create update destroy]
    end
  end
  resource :session, only: %i[new create destroy]    # singular for singleton
end
```

Prefer the seven standard actions; if you reach for many `member`/`collection` routes, that resource probably wants splitting. HTTP semantics (status codes, idempotency, versioning) are owned by [`rest.md`](guides://rest.md).

### B. Controllers — strong params & filters

```ruby
class Api::V1::UsersController < ApplicationController
  before_action :authenticate!
  before_action :set_user, only: %i[show update destroy]

  def index
    users = User.active.includes(:profile).page(params[:page])   # RAILS-AR-02
    render json: UserSerializer.new(users)
  end

  def create
    user = User.new(user_params)
    if user.save
      render json: UserSerializer.new(user), status: :created
    else
      render json: { errors: user.errors.full_messages }, status: :unprocessable_entity
    end
  end

  private

  def set_user = (@user = User.find(params[:id]))           # 404 via rescue_from below

  def user_params                                          # RAILS-SEC-01: allow-list only
    params.expect(user: %i[name email role])               # Rails 8 `expect`; never permit!
  end
end
```

- `params.expect(...)` (Rails 8) replaces `params.require(:user).permit(...)` and raises a 400 on a malformed shape — the current strong-params idiom. Security rationale (mass assignment) is owned by [`secure-coding.md`](guides://secure-coding.md); the Rails binding is: **always allow-list, never `permit!`**.
- Map domain errors to HTTP centrally with `rescue_from` (error strategy → [`error-handling.md`](guides://error-handling.md)):

```ruby
class ApplicationController < ActionController::Base
  rescue_from ActiveRecord::RecordNotFound, with: :not_found
  private def not_found = head(:not_found)
end
```

### C. Models — associations, validations, scopes, callbacks, enums

```ruby
class User < ApplicationRecord
  include Searchable                                    # concern (see ruby.md mixins)

  # Rails 7.2/8 enum syntax — keyword form, explicit integer mapping
  enum :status, { pending: 0, active: 1, suspended: 2 }, default: :pending

  belongs_to :organization, optional: true
  has_many :posts, dependent: :destroy
  has_one  :profile, dependent: :destroy

  # normalizes (Rails 7.1+) replaces hand-written before_validation strippers
  normalizes :email, with: ->(e) { e.strip.downcase }

  validates :name,  presence: true, length: { maximum: 100 }
  validates :email, presence: true, uniqueness: { case_sensitive: false },
                    format: { with: URI::MailTo::EMAIL_REGEXP }

  scope :active,          -> { where(status: :active) }
  scope :created_after,   ->(date) { where(created_at: date..) }   # endless range
  scope :with_posts,      -> { joins(:posts).distinct }

  after_create_commit :send_welcome_email                # *_commit: runs after TX commits

  private def send_welcome_email = UserMailer.welcome(self).deliver_later
end
```

- **Validations vs. DB constraints**: model validations are UX; the database is the source of truth. A `uniqueness` validation MUST be backed by a unique index (race-safe), and a `presence` by `NOT NULL` (RAILS-MIG-02).
- **Callbacks**: keep them few and side-effect-light; prefer `after_*_commit` for external effects (email, jobs, cache busting) so they never fire inside a rolled-back transaction. Heavy logic belongs in a service object, not a callback.
- **`normalizes`** and the keyword `enum` form are the current idioms — drop legacy `before_validation { … }` normalizers and positional `enum status: [...]`.

### D. ActiveRecord queries & N+1 (RAILS-AR-02)

```ruby
# ❌ N+1: one query per user
User.all.each { |u| puts u.posts.size }
# ✅ eager-load the association
User.includes(:posts).each { |u| puts u.posts.size }

User.select(:id, :name).where(active: true)         # narrow columns
User.where(active: true).pluck(:email)              # one column, no model alloc
User.find_each(batch_size: 1000) { |u| process(u) } # batched, constant memory
emails = User.where(active: true).in_batches.each_record # streaming
```

- Detect N+1 in tests with **Bullet** (raise in test env) or **prosopite**; gate it in CI.
- `includes` vs `preload` (two queries) vs `eager_load` (LEFT JOIN) vs `joins` (no load) — pick by whether you filter on or render the association. Query/index depth, EXPLAIN, and transaction isolation are owned by [`sql.md`](guides://sql.md) — this guide only binds the ActiveRecord surface.
- Counter caches (`belongs_to :user, counter_cache: true`) avoid `COUNT` round-trips for display.

### E. Migrations (RAILS-MIG-01/02)

```ruby
class CreatePosts < ActiveRecord::Migration[8.0]
  def change
    create_table :posts do |t|
      t.references :user, null: false, foreign_key: true, index: true   # FK + index
      t.string  :title, null: false
      t.timestamps
    end
    add_index :posts, %i[user_id created_at]
  end
end
```

- Every migration MUST be reversible — use `change` (auto-reversible) or supply `up`/`down`. Verify with a real `db:rollback`.
- Add an **index on every foreign key** and a **unique index** behind every uniqueness validation.
- For large tables, take backfills out of the schema migration (separate data migration / batched job) and disable the single-transaction wrapper only when an operation requires it (`disable_ddl_transaction!`).

### F. Concerns

```ruby
module Searchable
  extend ActiveSupport::Concern

  included do
    scope :search, ->(q) { where("name ILIKE ?", "%#{sanitize_sql_like(q)}%") }  # bound, not interpolated
  end

  class_methods do
    def searchable_fields = %i[name description]
  end
end
```

Concerns share behaviour across models/controllers (the Ruby mixin pattern — see [`ruby.md`](guides://ruby.md)); keep them cohesive (a real capability), not a junk drawer.

### G. Background jobs — ActiveJob + Solid Queue

```ruby
class WelcomeEmailJob < ApplicationJob
  queue_as :default
  retry_on Net::OpenTimeout, wait: :polynomially_longer, attempts: 5
  discard_on ActiveJob::DeserializationError

  def perform(user) = UserMailer.welcome(user).deliver_now
end

WelcomeEmailJob.perform_later(user)
```

- **Solid Queue** is the Rails 8 default backend — a database-backed queue, **no Redis required**. Configure it in `config/queue.yml`; run with `bin/jobs`. (Use Sidekiq/Redis only if you specifically need Redis — then see [`redis.md`](guides://redis.md).)
- Pass record IDs or use GlobalID, not large object graphs; make `perform` idempotent (jobs can retry).

### H. Caching — Solid Cache & fragment caching

```erb
<%# Russian-doll fragment caching — outer key busts when any child changes %>
<% cache @product do %>
  <%= render @product.variants %>   <%# each variant fragment cached individually %>
<% end %>
```

```ruby
Rails.cache.fetch("stats/#{id}", expires_in: 15.minutes) { calculate_stats }  # low-level
```

- **Solid Cache** is the Rails 8 default `Rails.cache` store — database/disk-backed, **no Redis required**, sized for large caches. `Solid Cable` likewise backs ActionCable without Redis.
- Cache keys include `updated_at` automatically for records, so writes invalidate fragments. Don't cache user-specific data under a shared key.

### I. Hotwire / Turbo (server-rendered SPA-like UX)

Rails ships **Hotwire** (Turbo + Stimulus) by default. Prefer it to a separate SPA for most CRUD UIs: Turbo Drive accelerates navigation, Turbo Frames scope updates, and Turbo Streams push partial-page updates over WebSocket/SSE.

```ruby
# Controller broadcasting a Turbo Stream after create
respond_to { |f| f.turbo_stream { render turbo_stream: turbo_stream.prepend(:articles, @article) } }
# Model-side live updates
class Article < ApplicationRecord; broadcasts_to ->(a) { :articles }; end
```

Stimulus controllers hold the sprinkles of JS. For JSON/native API clients, build a versioned API namespace instead (see §5.A and [`rest.md`](guides://rest.md)).

### J. ActionCable (WebSocket channels)

```ruby
class ChatChannel < ApplicationCable::Channel
  def subscribed = stream_from("room_#{params[:room_id]}")
end
```

Authenticate the connection in `ApplicationCable::Connection#connect` (`identified_by :current_user`); never trust client-supplied identity. Transport-level concerns are owned by [`websocket.md`](guides://websocket.md); Solid Cable provides the pub/sub backend.

---

## 6. Testing — Rails bindings

Test-first policy, coverage, and the Red-Green-Refactor cycle are owned by [`tdd.md`](guides://tdd.md) (Ruby runner binding in [`ruby.md`](guides://ruby.md)). Rails-specific spec types:

- **Model specs** — validations, scopes, associations, business methods (use `shoulda-matchers`).
- **Request specs** — exercise the full controller stack via HTTP; assert status + JSON/body. Preferred over controller specs.
- **System specs** — Capybara end-to-end through a real browser for critical journeys (RAILS-TST-02).
- Use **FactoryBot** factories over fixtures; `build` (no DB) where possible, `create` only when persistence matters.

```ruby
RSpec.describe "Users API", type: :request do
  it "creates a user" do
    expect {
      post "/api/v1/users", params: { user: { name: "A", email: "a@x.com" } }, headers: auth
    }.to change(User, :count).by(1)
    expect(response).to have_http_status(:created)
  end
end
```

---

## 7. Quick Reference

```bash
bin/rails new app -d postgresql            # new app (Solid Queue/Cache/Cable by default)
bin/rails g migration AddXToY x:string     # generate migration
bin/rails db:migrate / db:rollback         # schema up / down
bin/rails console                          # REPL
bin/jobs                                    # run Solid Queue worker
bundle exec rspec                          # model/request/system specs
bundle exec brakeman -q -w2                # security scan
bundle exec rubocop                        # rubocop-rails lint
```

---

## 8. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements here.

- [ ] RAILS-TST-01/02 — request/model specs test-first; critical journeys have system specs
- [ ] RAILS-SEC-01 — strong params (`expect`/`permit`) on every write; no `permit!`
- [ ] RAILS-SEC-02 — Brakeman clean at `-w2` (SQLi/XSS/CSRF/mass-assignment)
- [ ] RAILS-SEC-03 — secrets in encrypted credentials/ENV, none in repo
- [ ] RAILS-AR-01 — no string-interpolated SQL; bound params only
- [ ] RAILS-AR-02 — collections eager-loaded; 0 N+1 (Bullet/prosopite)
- [ ] RAILS-MIG-01 — migrations reversible (up+rollback clean)
- [ ] RAILS-MIG-02 — DB constraints back every uniqueness/required rule; FKs indexed
- [ ] RAILS-STRUCT-01 — controllers thin; business logic in models/POROs
- [ ] RAILS-LINT-01 — `rubocop-rails` clean
- [ ] RAILS-DEP-01 — `Gemfile.lock` committed, `bundle-audit` 0 advisories
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Ruby on Rails Guidelines**

# Ruby Development Guidelines

This document provides mandatory standards for Ruby development, following community conventions and best practices.

---

**Agent Profile**: The Ruby Expert
**Role**: Senior Ruby Developer & Rails Architect
**Objective**: Generate elegant, maintainable, and performant Ruby code following the Ruby Way.
**Tools**: Ruby 3.2+, Rails 7+, RuboCop, RSpec, Bundler.

---

## 1. Core Philosophies: RUBY-FIRST

- **R**eadable: Code should read like well-written prose
- **U**niform: Follow community conventions consistently
- **B**alanced: Embrace Ruby's flexibility without abuse
- **Y**ielding: Use blocks and iterators idiomatically

---

## 2. Style Conventions (MANDATORY)

### A. Naming

```ruby
# Classes and Modules: CamelCase
class UserAccount
end

module Authentication
  module Strategies
  end
end

# Methods and variables: snake_case
def calculate_total_price
  item_count = 10
  unit_price = 5.99
  item_count * unit_price
end

# Constants: SCREAMING_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30

# Predicate methods: end with ?
def valid?
  errors.empty?
end

def admin?
  role == 'admin'
end

# Dangerous methods: end with !
def save!
  raise RecordInvalid unless valid?
  persist
end

def normalize!
  self.name = name.strip.downcase
  self
end

# Symbols for identifiers
status = :pending
options = { format: :json, compress: true }
```

### B. Formatting

```ruby
# Two-space indentation (MANDATORY)
class User
  def initialize(name)
    @name = name
  end
end

# Spaces around operators
total = price * quantity + tax
result = condition ? value_a : value_b

# No spaces inside brackets
array = [1, 2, 3]
hash = { key: 'value' }
method_call(arg1, arg2)

# Multi-line method chains
result = users
  .select(&:active?)
  .map(&:email)
  .uniq
  .sort

# Multi-line hashes and arrays
config = {
  host: 'localhost',
  port: 3000,
  ssl: true
}

items = [
  'first',
  'second',
  'third'
]

# Heredocs for multi-line strings
query = <<~SQL
  SELECT users.*, COUNT(orders.id) as order_count
  FROM users
  LEFT JOIN orders ON orders.user_id = users.id
  GROUP BY users.id
SQL
```

---

## 3. Idiomatic Ruby (MANDATORY)

### A. Blocks and Iterators

```ruby
# ✅ CORRECT: Use each, map, select, etc.
names = users.map(&:name)
active_users = users.select(&:active?)
total = prices.sum
found = items.find { |item| item.id == target_id }

# ❌ WRONG: Manual iteration
names = []
users.each do |user|
  names << user.name  # Use map instead
end

# ✅ CORRECT: Block shorthand for single method calls
users.map(&:name)
users.select(&:active?)
users.reject(&:deleted?)

# ✅ CORRECT: each_with_index and each_with_object
items.each_with_index do |item, index|
  puts "#{index}: #{item}"
end

totals = orders.each_with_object({}) do |order, hash|
  hash[order.customer_id] ||= 0
  hash[order.customer_id] += order.total
end

# ✅ CORRECT: Use reduce/inject appropriately
sum = numbers.reduce(0, :+)
product = numbers.reduce(1, :*)

result = items.reduce({}) do |acc, item|
  acc.merge(item.key => item.value)
end
```

### B. Conditional Expressions

```ruby
# ✅ CORRECT: Trailing conditionals for single expressions
return if invalid?
raise ArgumentError, "Invalid input" unless valid_input?(input)
log_warning(message) if debug_mode?

# ✅ CORRECT: Guard clauses
def process(data)
  return if data.nil?
  return if data.empty?

  # Main processing logic
  transform(data)
end

# ✅ CORRECT: case expressions
def status_color(status)
  case status
  when :success then 'green'
  when :warning then 'yellow'
  when :error   then 'red'
  else 'gray'
  end
end

# ✅ CORRECT: Pattern matching (Ruby 3+)
def process_response(response)
  case response
  in { status: 200, body: }
    parse_success(body)
  in { status: 404 }
    handle_not_found
  in { status: 500, error: message }
    log_error(message)
  else
    handle_unknown
  end
end

# ✅ CORRECT: Ternary for simple conditionals
status = active? ? 'active' : 'inactive'

# ❌ WRONG: Nested ternary
result = a ? (b ? x : y) : z  # Use if/else instead
```

### C. Default Values and Options

```ruby
# ✅ CORRECT: Default parameter values
def greet(name, greeting = 'Hello')
  "#{greeting}, #{name}!"
end

# ✅ CORRECT: Keyword arguments with defaults
def create_user(name:, email:, role: 'user', active: true)
  User.new(name: name, email: email, role: role, active: active)
end

# ✅ CORRECT: Options hash for many optional params
def configure(options = {})
  @host = options.fetch(:host, 'localhost')
  @port = options.fetch(:port, 3000)
  @ssl = options.fetch(:ssl, false)
end

# ✅ CORRECT: Double splat for passing options
def wrapper(**options)
  inner_method(**options, extra: true)
end

# ✅ CORRECT: Safe navigation operator
user&.profile&.avatar_url
```

---

## 4. Classes and Modules (MANDATORY)

### A. Class Structure

```ruby
class User
  # Includes and extends first
  include Comparable
  extend ClassMethods

  # Constants
  ROLES = %w[admin moderator user guest].freeze

  # Attribute accessors
  attr_reader :id, :created_at
  attr_accessor :name, :email

  # Class methods
  class << self
    def find(id)
      # ...
    end

    def all
      # ...
    end
  end

  # Constructor
  def initialize(id:, name:, email:)
    @id = id
    @name = name
    @email = email
    @created_at = Time.current
  end

  # Public instance methods
  def full_name
    "#{first_name} #{last_name}"
  end

  def admin?
    role == 'admin'
  end

  # Comparable implementation
  def <=>(other)
    name <=> other.name
  end

  private

  # Private methods
  def validate_email
    raise InvalidEmail unless email.include?('@')
  end

  def normalize_name
    name.strip.titleize
  end
end
```

### B. Modules for Composition

```ruby
# Namespace module
module Authentication
  class Token
    # ...
  end

  class Session
    # ...
  end
end

# Mixin module
module Timestampable
  def self.included(base)
    base.extend(ClassMethods)
  end

  module ClassMethods
    def timestamped_attributes
      [:created_at, :updated_at]
    end
  end

  def touch
    @updated_at = Time.current
  end

  def created_at
    @created_at ||= Time.current
  end
end

# Concern (Rails-style)
module Searchable
  extend ActiveSupport::Concern

  included do
    scope :search, ->(query) { where('name LIKE ?', "%#{query}%") }
  end

  class_methods do
    def searchable_fields
      [:name, :description]
    end
  end

  def matches?(query)
    searchable_fields.any? do |field|
      send(field).to_s.downcase.include?(query.downcase)
    end
  end
end
```

---

## 5. Error Handling (MANDATORY)

### A. Custom Exceptions

```ruby
# Define exception hierarchy
module MyApp
  class Error < StandardError; end

  class ValidationError < Error
    attr_reader :field, :code

    def initialize(message, field: nil, code: nil)
      @field = field
      @code = code
      super(message)
    end
  end

  class NotFoundError < Error
    attr_reader :resource, :id

    def initialize(resource, id)
      @resource = resource
      @id = id
      super("#{resource} with id '#{id}' not found")
    end
  end

  class AuthenticationError < Error; end
  class AuthorizationError < Error; end
end

# Use custom exceptions
def find_user!(id)
  User.find(id) or raise MyApp::NotFoundError.new('User', id)
end
```

### B. Rescue Patterns

```ruby
# ✅ CORRECT: Specific rescue clauses
def process_payment(order)
  gateway.charge(order.total)
rescue PaymentGateway::CardDeclined => e
  order.mark_payment_failed!
  notify_customer(order, e.message)
  false
rescue PaymentGateway::NetworkError => e
  Rails.logger.error("Payment network error: #{e.message}")
  retry_later(order)
  false
rescue StandardError => e
  Rails.logger.error("Unexpected payment error: #{e.message}")
  Rails.logger.error(e.backtrace.join("\n"))
  raise
end

# ✅ CORRECT: Ensure for cleanup
def with_temp_file
  file = Tempfile.new('process')
  yield file
ensure
  file&.close
  file&.unlink
end

# ✅ CORRECT: Retry with limit
def fetch_with_retry(url, max_attempts: 3)
  attempts = 0
  begin
    attempts += 1
    http_get(url)
  rescue NetworkError => e
    raise if attempts >= max_attempts
    sleep(2 ** attempts)
    retry
  end
end

# ❌ WRONG: Bare rescue (catches Exception)
begin
  risky_operation
rescue  # Catches everything including Interrupt, SystemExit
  handle_error
end

# ✅ CORRECT: Rescue StandardError explicitly
begin
  risky_operation
rescue StandardError => e
  handle_error(e)
end
```

---

## 6. Collections (MANDATORY)

### A. Arrays

```ruby
# Creation
numbers = [1, 2, 3, 4, 5]
words = %w[apple banana cherry]  # Word array
symbols = %i[one two three]       # Symbol array

# Transformation
doubled = numbers.map { |n| n * 2 }
evens = numbers.select(&:even?)
odds = numbers.reject(&:even?)
first_even = numbers.find(&:even?)

# Aggregation
sum = numbers.sum
max = numbers.max
min = numbers.min
average = numbers.sum.to_f / numbers.size

# Grouping
grouped = users.group_by(&:role)
# => { 'admin' => [...], 'user' => [...] }

indexed = users.index_by(&:id)
# => { 1 => user1, 2 => user2 }

# Flattening and combining
nested = [[1, 2], [3, 4]]
flat = nested.flatten  # [1, 2, 3, 4]

combined = [1, 2] + [3, 4]  # [1, 2, 3, 4]
unique = (array1 + array2).uniq

# Checking
numbers.include?(3)      # true
numbers.any?(&:even?)    # true
numbers.all?(&:positive?) # true
numbers.none?(&:negative?) # true
```

### B. Hashes

```ruby
# Creation
user = { name: 'Alice', email: 'alice@example.com' }
config = { 'host' => 'localhost', 'port' => 3000 }

# Access
name = user[:name]
port = config['port']
role = user.fetch(:role, 'guest')  # With default
role = user.fetch(:role) { calculate_default_role }  # With block

# Transformation
emails = users_hash.values.map { |u| u[:email] }
keys = hash.keys.map(&:to_s)
symbolized = hash.transform_keys(&:to_sym)
doubled = numbers_hash.transform_values { |v| v * 2 }

# Merging
defaults = { timeout: 30, retries: 3 }
options = { timeout: 60 }
final = defaults.merge(options)  # { timeout: 60, retries: 3 }

# Deep merge (Rails)
deep_config = base_config.deep_merge(overrides)

# Slicing and filtering
subset = user.slice(:name, :email)
filtered = hash.select { |k, v| v.present? }
without = hash.except(:password, :token)

# Safe navigation with dig
value = response.dig(:data, :user, :profile, :name)
```

---

## 7. Testing with RSpec (MANDATORY)

### A. Test Structure

```ruby
# spec/models/user_spec.rb
require 'rails_helper'

RSpec.describe User, type: :model do
  # Subject and let blocks
  subject(:user) { build(:user, attributes) }

  let(:attributes) { { name: 'Alice', email: 'alice@example.com' } }

  describe 'validations' do
    it { is_expected.to validate_presence_of(:name) }
    it { is_expected.to validate_presence_of(:email) }
    it { is_expected.to validate_uniqueness_of(:email).case_insensitive }

    context 'when email format is invalid' do
      let(:attributes) { super().merge(email: 'invalid') }

      it 'is not valid' do
        expect(user).not_to be_valid
        expect(user.errors[:email]).to include('is invalid')
      end
    end
  end

  describe 'associations' do
    it { is_expected.to have_many(:posts).dependent(:destroy) }
    it { is_expected.to belong_to(:organization).optional }
  end

  describe '#full_name' do
    let(:attributes) { { first_name: 'Alice', last_name: 'Smith' } }

    it 'returns first and last name combined' do
      expect(user.full_name).to eq('Alice Smith')
    end

    context 'when last name is blank' do
      let(:attributes) { super().merge(last_name: '') }

      it 'returns only the first name' do
        expect(user.full_name).to eq('Alice')
      end
    end
  end

  describe '#admin?' do
    context 'when user has admin role' do
      let(:attributes) { super().merge(role: 'admin') }

      it 'returns true' do
        expect(user).to be_admin
      end
    end

    context 'when user has regular role' do
      let(:attributes) { super().merge(role: 'user') }

      it 'returns false' do
        expect(user).not_to be_admin
      end
    end
  end
end
```

### B. Service Tests

```ruby
# spec/services/payment_processor_spec.rb
RSpec.describe PaymentProcessor do
  subject(:processor) { described_class.new(gateway: gateway) }

  let(:gateway) { instance_double(PaymentGateway) }
  let(:order) { create(:order, total: 100.00) }

  describe '#process' do
    context 'when payment succeeds' do
      before do
        allow(gateway).to receive(:charge)
          .with(100.00)
          .and_return(PaymentResult.new(success: true, transaction_id: 'tx_123'))
      end

      it 'returns success result' do
        result = processor.process(order)

        expect(result).to be_success
        expect(result.transaction_id).to eq('tx_123')
      end

      it 'updates order status' do
        processor.process(order)

        expect(order.reload.status).to eq('paid')
      end
    end

    context 'when payment fails' do
      before do
        allow(gateway).to receive(:charge)
          .and_raise(PaymentGateway::CardDeclined.new('Insufficient funds'))
      end

      it 'returns failure result' do
        result = processor.process(order)

        expect(result).to be_failure
        expect(result.error_message).to eq('Insufficient funds')
      end

      it 'does not change order status' do
        expect { processor.process(order) }
          .not_to change { order.reload.status }
      end
    end
  end
end
```

### C. Request Specs

```ruby
# spec/requests/api/users_spec.rb
RSpec.describe 'Users API', type: :request do
  let(:headers) { { 'Authorization' => "Bearer #{token}" } }
  let(:token) { create(:api_token).value }

  describe 'GET /api/users' do
    let!(:users) { create_list(:user, 3) }

    it 'returns all users' do
      get '/api/users', headers: headers

      expect(response).to have_http_status(:ok)
      expect(json_response['users'].size).to eq(3)
    end

    context 'without authentication' do
      let(:headers) { {} }

      it 'returns unauthorized' do
        get '/api/users', headers: headers

        expect(response).to have_http_status(:unauthorized)
      end
    end
  end

  describe 'POST /api/users' do
    let(:valid_params) do
      {
        user: {
          name: 'New User',
          email: 'new@example.com'
        }
      }
    end

    it 'creates a new user' do
      expect {
        post '/api/users', params: valid_params, headers: headers
      }.to change(User, :count).by(1)

      expect(response).to have_http_status(:created)
      expect(json_response['user']['email']).to eq('new@example.com')
    end

    context 'with invalid params' do
      let(:invalid_params) { { user: { name: '' } } }

      it 'returns validation errors' do
        post '/api/users', params: invalid_params, headers: headers

        expect(response).to have_http_status(:unprocessable_entity)
        expect(json_response['errors']).to include('Name can\'t be blank')
      end
    end
  end

  def json_response
    JSON.parse(response.body)
  end
end
```

---

## 8. Rails Best Practices (MANDATORY)

### A. Controllers

```ruby
class Api::UsersController < ApplicationController
  before_action :authenticate!
  before_action :set_user, only: [:show, :update, :destroy]

  def index
    @users = User.active.includes(:profile).page(params[:page])
    render json: UserSerializer.new(@users)
  end

  def show
    render json: UserSerializer.new(@user)
  end

  def create
    @user = User.new(user_params)

    if @user.save
      render json: UserSerializer.new(@user), status: :created
    else
      render json: { errors: @user.errors.full_messages }, status: :unprocessable_entity
    end
  end

  def update
    if @user.update(user_params)
      render json: UserSerializer.new(@user)
    else
      render json: { errors: @user.errors.full_messages }, status: :unprocessable_entity
    end
  end

  def destroy
    @user.destroy
    head :no_content
  end

  private

  def set_user
    @user = User.find(params[:id])
  end

  def user_params
    params.require(:user).permit(:name, :email, :role)
  end
end
```

### B. Models

```ruby
class User < ApplicationRecord
  # Includes
  include Searchable

  # Enums
  enum :status, { pending: 0, active: 1, suspended: 2 }
  enum :role, { user: 0, moderator: 1, admin: 2 }

  # Associations
  belongs_to :organization, optional: true
  has_many :posts, dependent: :destroy
  has_many :comments, dependent: :destroy
  has_one :profile, dependent: :destroy

  # Validations
  validates :name, presence: true, length: { maximum: 100 }
  validates :email, presence: true, uniqueness: { case_sensitive: false },
                    format: { with: URI::MailTo::EMAIL_REGEXP }

  # Scopes
  scope :active, -> { where(status: :active) }
  scope :created_after, ->(date) { where('created_at > ?', date) }
  scope :with_posts, -> { joins(:posts).distinct }
  scope :by_name, -> { order(:name) }

  # Callbacks
  before_validation :normalize_email
  after_create :send_welcome_email

  # Class methods
  def self.find_by_credentials(email:, password:)
    user = find_by(email: email.downcase)
    user&.authenticate(password) ? user : nil
  end

  # Instance methods
  def admin?
    role == 'admin'
  end

  def display_name
    name.presence || email.split('@').first
  end

  private

  def normalize_email
    self.email = email&.strip&.downcase
  end

  def send_welcome_email
    UserMailer.welcome(self).deliver_later
  end
end
```

### C. Service Objects

```ruby
# app/services/order_processor.rb
class OrderProcessor
  include ActiveModel::Model

  attr_reader :order, :result

  def initialize(order)
    @order = order
  end

  def call
    ActiveRecord::Base.transaction do
      validate_inventory!
      reserve_items!
      process_payment!
      create_shipment!
      send_confirmation!

      @result = Result.success(order: order)
    end
  rescue InsufficientInventory => e
    @result = Result.failure(error: e.message, code: :inventory_error)
  rescue PaymentFailed => e
    @result = Result.failure(error: e.message, code: :payment_error)
  rescue StandardError => e
    Rails.logger.error("Order processing failed: #{e.message}")
    @result = Result.failure(error: 'An unexpected error occurred', code: :internal_error)
  end

  private

  def validate_inventory!
    order.items.each do |item|
      available = InventoryService.available_quantity(item.product_id)
      raise InsufficientInventory, item.product_id if available < item.quantity
    end
  end

  def reserve_items!
    order.items.each do |item|
      InventoryService.reserve(item.product_id, item.quantity)
    end
  end

  def process_payment!
    result = PaymentGateway.charge(
      amount: order.total,
      customer_id: order.customer.payment_id
    )
    raise PaymentFailed, result.error_message unless result.success?

    order.update!(payment_id: result.transaction_id, status: :paid)
  end

  def create_shipment!
    ShipmentService.create(order: order)
  end

  def send_confirmation!
    OrderMailer.confirmation(order).deliver_later
  end

  # Result object
  class Result
    attr_reader :order, :error, :code

    def initialize(success:, order: nil, error: nil, code: nil)
      @success = success
      @order = order
      @error = error
      @code = code
    end

    def self.success(order:)
      new(success: true, order: order)
    end

    def self.failure(error:, code:)
      new(success: false, error: error, code: code)
    end

    def success?
      @success
    end

    def failure?
      !success?
    end
  end
end
```

---

## 9. Performance (MANDATORY)

### A. Database Optimization

```ruby
# ❌ WRONG: N+1 queries
users = User.all
users.each do |user|
  puts user.posts.count  # Query for each user
end

# ✅ CORRECT: Eager loading
users = User.includes(:posts).all
users.each do |user|
  puts user.posts.size  # No extra queries
end

# ✅ CORRECT: Counter cache
# migration
add_column :users, :posts_count, :integer, default: 0

# model
class Post < ApplicationRecord
  belongs_to :user, counter_cache: true
end

# ✅ CORRECT: Select only needed columns
User.select(:id, :name, :email).where(active: true)

# ✅ CORRECT: Batch processing
User.find_each(batch_size: 1000) do |user|
  process(user)
end

# ✅ CORRECT: Pluck for single column
emails = User.where(active: true).pluck(:email)
```

### B. Caching

```ruby
# Fragment caching
<% cache @product do %>
  <%= render @product %>
<% end %>

# Low-level caching
def expensive_calculation
  Rails.cache.fetch("calculation:#{id}", expires_in: 1.hour) do
    perform_expensive_calculation
  end
end

# Conditional caching
def stats
  return @stats if defined?(@stats)

  @stats = Rails.cache.fetch(cache_key, expires_in: 15.minutes) do
    calculate_stats
  end
end

# Russian doll caching
<% cache [@product, @product.updated_at] do %>
  <% @product.variants.each do |variant| %>
    <% cache variant do %>
      <%= render variant %>
    <% end %>
  <% end %>
<% end %>
```

---

## 10. Deployment Checklist

### Code Quality
- [ ] RuboCop passes with no offenses
- [ ] All specs passing
- [ ] No binding.pry or debugger statements
- [ ] No puts or p debugging output

### Security
- [ ] Strong parameters used
- [ ] SQL injection prevented (parameterized queries)
- [ ] Mass assignment protected
- [ ] Secrets in credentials, not code

### Performance
- [ ] N+1 queries eliminated
- [ ] Appropriate indexes in place
- [ ] Caching implemented
- [ ] Background jobs for slow operations

### Database
- [ ] Migrations are reversible
- [ ] Foreign keys defined
- [ ] Indexes on foreign keys
- [ ] Data migrations separate from schema

---

## 11. Quick Reference

```ruby
# String methods
str.strip           # Remove whitespace
str.downcase        # Lowercase
str.present?        # Not nil or empty (Rails)
str.blank?          # Nil or empty (Rails)

# Array methods
arr.map { }         # Transform
arr.select { }      # Filter
arr.reject { }      # Inverse filter
arr.find { }        # First match
arr.compact         # Remove nils
arr.flatten         # Flatten nested
arr.uniq            # Remove duplicates

# Hash methods
hash.fetch(:key, default)
hash.dig(:a, :b, :c)
hash.slice(:key1, :key2)
hash.except(:key)
hash.merge(other)

# Rails helpers
Time.current        # Use instead of Time.now
Date.current        # Use instead of Date.today
n.days.ago          # Time calculation
n.hours.from_now    # Time calculation
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Ruby Team

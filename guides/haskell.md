# Haskell Development Guide

## Core Philosophies

### HASKELL-FIRST
1. **Type-Driven Development**: Let the type system guide your design; "make invalid states unrepresentable"
2. **Purity by Default**: Pure functions are the norm; effects are explicit and isolated
3. **Composability**: Build complex systems from simple, composable functions
4. **Immutability**: All data structures are immutable by default
5. **Laziness as a Feature**: Leverage lazy evaluation for elegant, efficient code
6. **Documentation as Code**: Use Haddock comments for all public APIs
7. **Test-Driven Development (TDD)**: Write tests first, then implementation
8. **Regression Shield**: Every bug fix must include a regression test

### MODERN-HASKELL
1. **GHC 9.4+**: Use modern GHC with latest language extensions
2. **Cabal 3.10+**: Modern dependency management with cabal-install
3. **HLS Integration**: Haskell Language Server for IDE support
4. **Property-Based Testing**: QuickCheck for comprehensive test coverage
5. **Strict by Default**: Use `-XStrictData` to avoid space leaks
6. **Modern Extensions**: Leverage `DerivingStrategies`, `RecordWildCards`, `OverloadedStrings`
7. **Type Safety**: Use newtypes, smart constructors, and GADTs for type-level guarantees
8. **Effect Systems**: Consider `mtl`, `polysemy`, or `effectful` for managing effects

### HEXAGONAL-ARCHITECTURE
1. **Domain Core**: Pure business logic with algebraic data types
2. **Ports (Type Classes)**: Abstract interfaces for external dependencies
3. **Adapters**: Concrete implementations of ports (IO, databases, APIs)
4. **Dependency Inversion**: Domain depends on abstractions, not implementations

### AGENT-VERIFICATION
When generating Haskell code, agents MUST verify:
1. **Compilation**: `cabal build` succeeds without errors
2. **Tests Pass**: `cabal test` runs all tests successfully
3. **Linting**: `hlint` reports no warnings (or approved exceptions)
4. **Documentation**: `cabal haddock` generates complete documentation
5. **Type Checking**: All type signatures are explicit and correct

---

## 1. Test-Driven Development (TDD)

### A. TDD Protocol

**Red-Green-Refactor Cycle**:

1. **RED**: Write a failing test that defines desired behavior
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Improve code quality while keeping tests green
4. **VERIFY**: Run `cabal test` to ensure all tests pass

### B. Haskell TDD Example: Pure Function

**Step 1: RED - Write Failing Test**

```haskell
-- test/Spec/Data/TextUtilsSpec.hs
{-# LANGUAGE OverloadedStrings #-}

module Spec.Data.TextUtilsSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.Text (Text)
import qualified Data.Text as T
import Data.TextUtils (isPalindrome, reverseWords)

spec :: Spec
spec = do
  describe "isPalindrome" $ do
    it "returns True for single character" $
      isPalindrome "a" `shouldBe` True

    it "returns True for palindrome strings" $ do
      isPalindrome "racecar" `shouldBe` True
      isPalindrome "A man a plan a canal Panama" `shouldBe` True

    it "returns False for non-palindrome strings" $ do
      isPalindrome "hello" `shouldBe` False
      isPalindrome "world" `shouldBe` False

    it "is case-insensitive" $
      isPalindrome "RaceCar" `shouldBe` True

  describe "reverseWords" $ do
    it "reverses words in a sentence" $
      reverseWords "hello world" `shouldBe` "world hello"

    it "handles empty string" $
      reverseWords "" `shouldBe` ""

    it "handles single word" $
      reverseWords "hello" `shouldBe` "hello"

    -- Property-based test
    it "reversing twice returns original" $
      property $ \text ->
        reverseWords (reverseWords text) === text
```

**Step 2: GREEN - Implement Minimal Solution**

```haskell
-- src/Data/TextUtils.hs
{-# LANGUAGE OverloadedStrings #-}

module Data.TextUtils
  ( isPalindrome
  , reverseWords
  ) where

import Data.Char (toLower, isAlphaNum)
import Data.Text (Text)
import qualified Data.Text as T

-- | Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric).
--
-- >>> isPalindrome "racecar"
-- True
--
-- >>> isPalindrome "hello"
-- False
isPalindrome :: Text -> Bool
isPalindrome text = normalized == T.reverse normalized
  where
    normalized = T.map toLower $ T.filter isAlphaNum text

-- | Reverse the order of words in a text.
--
-- >>> reverseWords "hello world"
-- "world hello"
--
-- >>> reverseWords ""
-- ""
reverseWords :: Text -> Text
reverseWords text
  | T.null text = ""
  | otherwise = T.unwords . reverse . T.words $ text
```

**Step 3: REFACTOR - Improve Code Quality**

```haskell
-- src/Data/TextUtils.hs
{-# LANGUAGE OverloadedStrings #-}

module Data.TextUtils
  ( -- * Text Manipulation
    isPalindrome
  , reverseWords
    -- * Normalization
  , normalize
  ) where

import Data.Char (toLower, isAlphaNum)
import Data.Text (Text)
import qualified Data.Text as T

-- | Normalize text for comparison by removing non-alphanumeric characters
-- and converting to lowercase.
--
-- >>> normalize "Hello, World!"
-- "helloworld"
normalize :: Text -> Text
normalize = T.map toLower . T.filter isAlphaNum

-- | Check if a string is a palindrome (case-insensitive, ignoring non-alphanumeric).
--
-- ==== __Examples__
--
-- >>> isPalindrome "racecar"
-- True
--
-- >>> isPalindrome "A man a plan a canal Panama"
-- True
--
-- >>> isPalindrome "hello"
-- False
--
-- ==== __Properties__
--
-- prop> isPalindrome text == isPalindrome (T.reverse text)
isPalindrome :: Text -> Bool
isPalindrome text = normalized == T.reverse normalized
  where
    normalized = normalize text

-- | Reverse the order of words in a text.
--
-- ==== __Examples__
--
-- >>> reverseWords "hello world"
-- "world hello"
--
-- >>> reverseWords ""
-- ""
--
-- >>> reverseWords "one"
-- "one"
--
-- ==== __Properties__
--
-- prop> reverseWords (reverseWords text) == text
reverseWords :: Text -> Text
reverseWords = T.unwords . reverse . T.words
```

**Step 4: VERIFY - Run Tests**

```bash
cabal test text-utils-test
# All tests pass ✓
```

### C. Haskell TDD Example: Type-Driven Design

**Step 1: RED - Define Types and Write Failing Test**

```haskell
-- test/Spec/Domain/OrderSpec.hs
{-# LANGUAGE OverloadedStrings #-}

module Spec.Domain.OrderSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Domain.Order
import Domain.Product
import Data.Time.Clock (UTCTime)
import Data.Time.Calendar (fromGregorian)

spec :: Spec
spec = do
  describe "Order creation" $ do
    it "creates a valid order" $ do
      let orderId = OrderId "ORD-001"
          customerId = CustomerId "CUST-001"
          items = [OrderItem (ProductId "PROD-001") (Quantity 2) (Price 10.00)]
      result <- createOrder orderId customerId items
      case result of
        Right order -> do
          orderTotal order `shouldBe` Money 20.00
          orderStatus order `shouldBe` Pending
        Left err -> expectationFailure $ "Order creation failed: " <> show err

    it "rejects orders with empty items" $ do
      let orderId = OrderId "ORD-002"
          customerId = CustomerId "CUST-002"
          items = []
      result <- createOrder orderId customerId items
      result `shouldBe` Left EmptyOrder

    it "rejects orders with zero quantity" $ do
      let orderId = OrderId "ORD-003"
          customerId = CustomerId "CUST-003"
          items = [OrderItem (ProductId "PROD-001") (Quantity 0) (Price 10.00)]
      result <- createOrder orderId customerId items
      result `shouldBe` Left InvalidQuantity

  describe "Order total calculation" $ do
    it "calculates total for multiple items" $ do
      let items = [ OrderItem (ProductId "PROD-001") (Quantity 2) (Price 10.00)
                  , OrderItem (ProductId "PROD-002") (Quantity 3) (Price 5.00)
                  ]
      calculateTotal items `shouldBe` Money 35.00

    it "property: total is always non-negative" $
      property $ \items ->
        let Money total = calculateTotal (getNonEmpty items)
        in total >= 0
```

**Step 2: GREEN - Define Types and Implement**

```haskell
-- src/Domain/Order.hs
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE StrictData #-}

module Domain.Order
  ( -- * Types
    Order(..)
  , OrderId(..)
  , CustomerId(..)
  , OrderItem(..)
  , OrderStatus(..)
  , OrderError(..)
    -- * Smart Constructors
  , createOrder
  , calculateTotal
    -- * Operations
  , confirmOrder
  , cancelOrder
  ) where

import Domain.Product
import Data.Text (Text)
import Data.Time.Clock (UTCTime, getCurrentTime)
import Control.Monad (when)

-- | Opaque identifier for orders.
newtype OrderId = OrderId Text
  deriving stock (Eq, Show)
  deriving newtype (Ord)

-- | Opaque identifier for customers.
newtype CustomerId = CustomerId Text
  deriving stock (Eq, Show)
  deriving newtype (Ord)

-- | Order status in the lifecycle.
data OrderStatus
  = Pending
  | Confirmed
  | Shipped
  | Delivered
  | Cancelled
  deriving stock (Eq, Show, Ord, Enum, Bounded)

-- | An item within an order.
data OrderItem = OrderItem
  { orderItemProduct  :: ProductId
  , orderItemQuantity :: Quantity
  , orderItemPrice    :: Price
  } deriving stock (Eq, Show)

-- | Domain errors for order operations.
data OrderError
  = EmptyOrder
  | InvalidQuantity
  | InvalidPrice
  | OrderNotFound OrderId
  | InvalidStatusTransition OrderStatus OrderStatus
  deriving stock (Eq, Show)

-- | An order aggregate.
data Order = Order
  { orderId        :: OrderId
  , orderCustomerId :: CustomerId
  , orderItems     :: [OrderItem]
  , orderTotal     :: Money
  , orderStatus    :: OrderStatus
  , orderCreatedAt :: UTCTime
  , orderUpdatedAt :: UTCTime
  } deriving stock (Eq, Show)

-- | Smart constructor for creating a valid order.
--
-- Validates:
-- - Order has at least one item
-- - All quantities are positive
-- - All prices are positive
createOrder :: OrderId -> CustomerId -> [OrderItem] -> IO (Either OrderError Order)
createOrder oid cid items = do
  now <- getCurrentTime
  pure $ do
    when (null items) $ Left EmptyOrder
    when (any (\item -> let Quantity q = orderItemQuantity item in q <= 0) items) $
      Left InvalidQuantity
    when (any (\item -> let Price p = orderItemPrice item in p < 0) items) $
      Left InvalidPrice
    let total = calculateTotal items
    Right $ Order
      { orderId = oid
      , orderCustomerId = cid
      , orderItems = items
      , orderTotal = total
      , orderStatus = Pending
      , orderCreatedAt = now
      , orderUpdatedAt = now
      }

-- | Calculate the total price of order items.
calculateTotal :: [OrderItem] -> Money
calculateTotal items = Money $ sum
  [ fromIntegral q * p
  | OrderItem _ (Quantity q) (Price p) <- items
  ]

-- | Confirm an order (transition from Pending to Confirmed).
confirmOrder :: Order -> IO (Either OrderError Order)
confirmOrder order = do
  now <- getCurrentTime
  case orderStatus order of
    Pending -> Right <$> pure order
      { orderStatus = Confirmed
      , orderUpdatedAt = now
      }
    status -> pure $ Left $ InvalidStatusTransition status Confirmed

-- | Cancel an order (if not already shipped or delivered).
cancelOrder :: Order -> IO (Either OrderError Order)
cancelOrder order = do
  now <- getCurrentTime
  case orderStatus order of
    Shipped -> pure $ Left $ InvalidStatusTransition Shipped Cancelled
    Delivered -> pure $ Left $ InvalidStatusTransition Delivered Cancelled
    _ -> Right <$> pure order
      { orderStatus = Cancelled
      , orderUpdatedAt = now
      }
```

```haskell
-- src/Domain/Product.hs
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Domain.Product
  ( ProductId(..)
  , Quantity(..)
  , Price(..)
  , Money(..)
  ) where

import Data.Text (Text)

-- | Opaque product identifier.
newtype ProductId = ProductId Text
  deriving stock (Eq, Show)
  deriving newtype (Ord)

-- | Quantity of a product (must be positive).
newtype Quantity = Quantity Int
  deriving stock (Eq, Show)
  deriving newtype (Ord, Num)

-- | Price per unit (must be non-negative).
newtype Price = Price Double
  deriving stock (Eq, Show)
  deriving newtype (Ord, Num, Fractional)

-- | Total money amount.
newtype Money = Money Double
  deriving stock (Eq, Show)
  deriving newtype (Ord, Num, Fractional)
```

**Step 3: REFACTOR - Add Type Classes and Documentation**

```haskell
-- src/Domain/Order.hs
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE StrictData #-}

-- | Order domain module implementing order management business logic.
--
-- This module provides types and operations for creating and managing orders.
-- All operations maintain invariants through smart constructors and validated
-- state transitions.
module Domain.Order
  ( -- * Types
    Order(..)
  , OrderId(..)
  , CustomerId(..)
  , OrderItem(..)
  , OrderStatus(..)
  , OrderError(..)
    -- * Smart Constructors
  , createOrder
  , calculateTotal
    -- * Operations
  , confirmOrder
  , cancelOrder
  , canTransition
  ) where

import Domain.Product
import Data.Text (Text)
import Data.Time.Clock (UTCTime, getCurrentTime)
import Control.Monad (when)

-- | Opaque identifier for orders.
--
-- Use smart constructors to create valid order identifiers.
newtype OrderId = OrderId Text
  deriving stock (Eq, Show)
  deriving newtype (Ord)

-- | Opaque identifier for customers.
newtype CustomerId = CustomerId Text
  deriving stock (Eq, Show)
  deriving newtype (Ord)

-- | Order status in the order lifecycle.
--
-- Valid transitions:
--
-- @
-- Pending -> Confirmed -> Shipped -> Delivered
--    |          |
--    v          v
-- Cancelled  Cancelled
-- @
data OrderStatus
  = Pending    -- ^ Order created but not yet confirmed
  | Confirmed  -- ^ Order confirmed by customer
  | Shipped    -- ^ Order dispatched for delivery
  | Delivered  -- ^ Order delivered to customer
  | Cancelled  -- ^ Order cancelled (terminal state)
  deriving stock (Eq, Show, Ord, Enum, Bounded)

-- | An item within an order with product, quantity, and price.
data OrderItem = OrderItem
  { orderItemProduct  :: ProductId  -- ^ The product identifier
  , orderItemQuantity :: Quantity   -- ^ Quantity ordered (must be > 0)
  , orderItemPrice    :: Price      -- ^ Price per unit at time of order
  } deriving stock (Eq, Show)

-- | Domain errors for order operations.
data OrderError
  = EmptyOrder                                      -- ^ Order has no items
  | InvalidQuantity                                 -- ^ Item quantity is zero or negative
  | InvalidPrice                                    -- ^ Item price is negative
  | OrderNotFound OrderId                          -- ^ Order does not exist
  | InvalidStatusTransition OrderStatus OrderStatus -- ^ Illegal status transition
  deriving stock (Eq, Show)

-- | An order aggregate root.
--
-- Invariants:
--
-- * Must have at least one item
-- * All quantities must be positive
-- * Total matches sum of item totals
-- * Status transitions follow valid state machine
data Order = Order
  { orderId         :: OrderId       -- ^ Unique order identifier
  , orderCustomerId :: CustomerId    -- ^ Customer who placed the order
  , orderItems      :: [OrderItem]   -- ^ Non-empty list of order items
  , orderTotal      :: Money         -- ^ Total order value
  , orderStatus     :: OrderStatus   -- ^ Current order status
  , orderCreatedAt  :: UTCTime       -- ^ Order creation timestamp
  , orderUpdatedAt  :: UTCTime       -- ^ Last update timestamp
  } deriving stock (Eq, Show)

-- | Smart constructor for creating a valid order.
--
-- Validates:
--
-- * Order has at least one item
-- * All quantities are positive
-- * All prices are non-negative
--
-- Returns 'Left' 'OrderError' if validation fails.
--
-- ==== __Examples__
--
-- >>> items = [OrderItem (ProductId "PROD-001") (Quantity 2) (Price 10.0)]
-- >>> result <- createOrder (OrderId "ORD-001") (CustomerId "CUST-001") items
-- >>> fmap orderTotal result
-- Right (Money 20.0)
createOrder :: OrderId -> CustomerId -> [OrderItem] -> IO (Either OrderError Order)
createOrder oid cid items = do
  now <- getCurrentTime
  pure $ do
    when (null items) $ Left EmptyOrder
    when (any (\item -> let Quantity q = orderItemQuantity item in q <= 0) items) $
      Left InvalidQuantity
    when (any (\item -> let Price p = orderItemPrice item in p < 0) items) $
      Left InvalidPrice
    let total = calculateTotal items
    Right $ Order
      { orderId = oid
      , orderCustomerId = cid
      , orderItems = items
      , orderTotal = total
      , orderStatus = Pending
      , orderCreatedAt = now
      , orderUpdatedAt = now
      }

-- | Calculate the total price of order items.
--
-- >>> let items = [OrderItem (ProductId "P1") (Quantity 2) (Price 10.0)]
-- >>> calculateTotal items
-- Money 20.0
calculateTotal :: [OrderItem] -> Money
calculateTotal items = Money $ sum
  [ fromIntegral q * p
  | OrderItem _ (Quantity q) (Price p) <- items
  ]

-- | Check if a status transition is valid.
canTransition :: OrderStatus -> OrderStatus -> Bool
canTransition Pending Confirmed = True
canTransition Pending Cancelled = True
canTransition Confirmed Shipped = True
canTransition Confirmed Cancelled = True
canTransition Shipped Delivered = True
canTransition current target = current == target

-- | Confirm an order (transition from Pending to Confirmed).
--
-- Returns 'Left' 'InvalidStatusTransition' if order is not in 'Pending' status.
confirmOrder :: Order -> IO (Either OrderError Order)
confirmOrder order = do
  now <- getCurrentTime
  case orderStatus order of
    Pending -> Right <$> pure order
      { orderStatus = Confirmed
      , orderUpdatedAt = now
      }
    status -> pure $ Left $ InvalidStatusTransition status Confirmed

-- | Cancel an order (if not already shipped or delivered).
--
-- Terminal states (Shipped, Delivered) cannot be cancelled.
cancelOrder :: Order -> IO (Either OrderError Order)
cancelOrder order = do
  now <- getCurrentTime
  case orderStatus order of
    Shipped -> pure $ Left $ InvalidStatusTransition Shipped Cancelled
    Delivered -> pure $ Left $ InvalidStatusTransition Delivered Cancelled
    Cancelled -> Right <$> pure order  -- Already cancelled, idempotent
    _ -> Right <$> pure order
      { orderStatus = Cancelled
      , orderUpdatedAt = now
      }
```

**Step 4: VERIFY - Run Tests and Check Documentation**

```bash
# Run tests
cabal test order-test

# Generate documentation
cabal haddock --haddock-all

# Lint code
hlint src/Domain/Order.hs
# No suggestions ✓
```

---

## 2. Bug Fix Protocol

### Every Bug Fix Requires:

1. **Reproduce**: Create a failing test that reproduces the bug
2. **Document**: Add comments explaining the bug and fix
3. **Fix**: Implement the minimal fix
4. **Verify**: Ensure the new test passes
5. **Regression**: Run full test suite to prevent regressions
6. **Review**: Code review focusing on the fix and test
7. **Deploy**: Include bug ID in commit message

### Bug Fix Example: List Index Out of Bounds

**Bug Report**: `safeHead` function crashes with empty list

**Step 1: Reproduce with Failing Test**

```haskell
-- test/Spec/Data/ListUtilsSpec.hs
module Spec.Data.ListUtilsSpec (spec) where

import Test.Hspec
import Data.ListUtils

spec :: Spec
spec = do
  describe "safeHead" $ do
    -- BUG #123: safeHead crashes with empty list
    it "returns Nothing for empty list" $ do
      safeHead ([] :: [Int]) `shouldBe` Nothing

    it "returns Just first element for non-empty list" $ do
      safeHead [1, 2, 3] `shouldBe` Just 1
      safeHead ["hello"] `shouldBe` Just "hello"

  describe "safeTail" $ do
    -- BUG #123: Related fix for safeTail
    it "returns Nothing for empty list" $ do
      safeTail ([] :: [Int]) `shouldBe` Nothing

    it "returns Just tail for non-empty list" $ do
      safeTail [1, 2, 3] `shouldBe` Just [2, 3]
      safeTail [1] `shouldBe` Just []
```

**Step 2: Original Buggy Code**

```haskell
-- src/Data/ListUtils.hs (BEFORE FIX)
module Data.ListUtils
  ( safeHead
  , safeTail
  ) where

-- BUG #123: Crashes with empty list due to incomplete pattern match
safeHead :: [a] -> Maybe a
safeHead (x:_) = Just x
-- Missing case for empty list!

safeTail :: [a] -> Maybe [a]
safeTail (_:xs) = Just xs
-- Missing case for empty list!
```

**Step 3: Fix the Bug**

```haskell
-- src/Data/ListUtils.hs (AFTER FIX)
{-# LANGUAGE Safe #-}

-- | Safe list utilities that never throw exceptions.
--
-- This module provides total functions for list operations that would
-- otherwise be partial.
module Data.ListUtils
  ( safeHead
  , safeTail
  , safeIndex
  , safeLast
  ) where

-- | Safely get the first element of a list.
--
-- Returns 'Nothing' for empty lists instead of throwing an exception.
--
-- ==== __Examples__
--
-- >>> safeHead [1, 2, 3]
-- Just 1
--
-- >>> safeHead []
-- Nothing
--
-- ==== __Bug Fixes__
--
-- * BUG #123: Fixed crash on empty list by adding explicit empty case
safeHead :: [a] -> Maybe a
safeHead [] = Nothing        -- FIX: Handle empty list case
safeHead (x:_) = Just x

-- | Safely get the tail of a list.
--
-- Returns 'Nothing' for empty lists.
--
-- >>> safeTail [1, 2, 3]
-- Just [2, 3]
--
-- >>> safeTail []
-- Nothing
--
-- ==== __Bug Fixes__
--
-- * BUG #123: Fixed crash on empty list by adding explicit empty case
safeTail :: [a] -> Maybe [a]
safeTail [] = Nothing        -- FIX: Handle empty list case
safeTail (_:xs) = Just xs

-- | Safely get element at index.
--
-- >>> safeIndex 0 [1, 2, 3]
-- Just 1
--
-- >>> safeIndex 5 [1, 2, 3]
-- Nothing
safeIndex :: Int -> [a] -> Maybe a
safeIndex n xs
  | n < 0 = Nothing
  | otherwise = case drop n xs of
      [] -> Nothing
      (y:_) -> Just y

-- | Safely get the last element of a list.
--
-- >>> safeLast [1, 2, 3]
-- Just 3
--
-- >>> safeLast []
-- Nothing
safeLast :: [a] -> Maybe a
safeLast [] = Nothing
safeLast [x] = Just x
safeLast (_:xs) = safeLast xs
```

**Step 4: Verify Fix**

```bash
# Run regression test
cabal test list-utils-test
# All tests pass ✓

# Run full test suite
cabal test
# All tests pass ✓

# Check for warnings
cabal build --ghc-options="-Wall -Werror"
# Build successful ✓

# Lint
hlint src/Data/ListUtils.hs
# No suggestions ✓
```

**Step 5: Commit with Bug Reference**

```bash
git add src/Data/ListUtils.hs test/Spec/Data/ListUtilsSpec.hs
git commit -m "fix: Handle empty list in safeHead and safeTail

Fixes #123

- Added explicit pattern match for empty list case
- Added regression tests for empty list handling
- Made all list functions total (never throw exceptions)
- Enabled Safe Haskell extension for compile-time safety checks"
```

### Bug Fix Example: Type Safety Issue

**Bug Report**: `parseConfig` accepts invalid configurations

**Step 1: Reproduce with Failing Test**

```haskell
-- test/Spec/Config/ParserSpec.hs
module Spec.Config.ParserSpec (spec) where

import Test.Hspec
import Config.Parser
import Data.Text (Text)

spec :: Spec
spec = do
  describe "parseConfig" $ do
    -- BUG #456: parseConfig accepts negative port numbers
    it "rejects negative port numbers" $ do
      let configText = "port: -8080\nhost: localhost"
      parseConfig configText `shouldBe` Left (InvalidPort (-8080))

    -- BUG #456: parseConfig accepts port 0
    it "rejects port 0" $ do
      let configText = "port: 0\nhost: localhost"
      parseConfig configText `shouldBe` Left (InvalidPort 0)

    -- BUG #456: parseConfig accepts ports > 65535
    it "rejects ports above valid range" $ do
      let configText = "port: 70000\nhost: localhost"
      parseConfig configText `shouldBe` Left (InvalidPort 70000)

    it "accepts valid port numbers" $ do
      let configText = "port: 8080\nhost: localhost"
      case parseConfig configText of
        Right config -> do
          let Port p = configPort config
          p `shouldSatisfy` (\x -> x > 0 && x <= 65535)
        Left err -> expectationFailure $ "Valid config rejected: " <> show err
```

**Step 2: Original Buggy Code**

```haskell
-- src/Config/Parser.hs (BEFORE FIX)
module Config.Parser
  ( Config(..)
  , parseConfig
  ) where

import Data.Text (Text)

data Config = Config
  { configPort :: Int      -- BUG #456: Should use newtype for validation
  , configHost :: Text
  } deriving (Show, Eq)

parseConfig :: Text -> Either Text Config
parseConfig text = do
  port <- extractPort text
  host <- extractHost text
  pure $ Config port host  -- BUG: No validation of port range!
```

**Step 3: Fix with Type-Level Safety**

```haskell
-- src/Config/Parser.hs (AFTER FIX)
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- | Configuration parser with validated types.
--
-- Uses newtypes and smart constructors to ensure configuration values
-- are always valid at the type level.
module Config.Parser
  ( -- * Types
    Config(..)
  , Port
  , mkPort
  , getPort
  , Host(..)
  , ConfigError(..)
    -- * Parsing
  , parseConfig
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Text.Read (readMaybe)

-- | Valid port number (1-65535).
--
-- Constructor is not exported; use 'mkPort' smart constructor.
newtype Port = Port Int
  deriving stock (Eq, Show, Ord)

-- | Smart constructor for creating a valid port.
--
-- Returns 'Nothing' if port is outside valid range [1, 65535].
--
-- ==== __Examples__
--
-- >>> mkPort 8080
-- Just (Port 8080)
--
-- >>> mkPort 0
-- Nothing
--
-- >>> mkPort 70000
-- Nothing
--
-- ==== __Bug Fixes__
--
-- * BUG #456: Added validation to reject invalid port numbers
mkPort :: Int -> Maybe Port
mkPort n
  | n >= 1 && n <= 65535 = Just (Port n)
  | otherwise = Nothing

-- | Extract port number from validated Port.
getPort :: Port -> Int
getPort (Port n) = n

-- | Valid hostname (non-empty string).
newtype Host = Host Text
  deriving stock (Eq, Show)
  deriving newtype (Ord)

-- | Smart constructor for creating a valid host.
mkHost :: Text -> Maybe Host
mkHost text
  | T.null text = Nothing
  | otherwise = Just (Host text)

-- | Configuration with validated fields.
data Config = Config
  { configPort :: Port  -- FIX: Use validated Port type
  , configHost :: Host  -- FIX: Use validated Host type
  } deriving stock (Eq, Show)

-- | Configuration parsing errors.
data ConfigError
  = InvalidPort Int
  | InvalidHost Text
  | MissingField Text
  | ParseError Text
  deriving stock (Eq, Show)

-- | Parse configuration from text.
--
-- Validates all fields and returns 'Left' 'ConfigError' if any field is invalid.
--
-- ==== __Bug Fixes__
--
-- * BUG #456: Added comprehensive validation for port and host fields
parseConfig :: Text -> Either ConfigError Config
parseConfig text = do
  portInt <- extractPort text
  port <- case mkPort portInt of
    Nothing -> Left (InvalidPort portInt)  -- FIX: Validate port range
    Just p -> Right p

  hostText <- extractHost text
  host <- case mkHost hostText of
    Nothing -> Left (InvalidHost hostText)  -- FIX: Validate host
    Just h -> Right h

  pure $ Config port host

-- Helper functions (simplified for example)
extractPort :: Text -> Either ConfigError Int
extractPort text = case T.stripPrefix "port: " <$> T.lines text of
  (Just portText : _) -> case readMaybe (T.unpack portText) of
    Just n -> Right n
    Nothing -> Left (ParseError "Invalid port format")
  _ -> Left (MissingField "port")

extractHost :: Text -> Either ConfigError Text
extractHost text = case T.stripPrefix "host: " <$> T.lines text of
  (Just hostText : _) -> Right hostText
  _ -> Left (MissingField "host")
```

**Step 4: Add Property-Based Tests**

```haskell
-- test/Spec/Config/ParserSpec.hs (EXTENDED)
{-# LANGUAGE OverloadedStrings #-}

module Spec.Config.ParserSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Config.Parser
import Data.Text (Text)
import qualified Data.Text as T

spec :: Spec
spec = do
  describe "mkPort" $ do
    it "accepts valid port range" $ do
      mkPort 1 `shouldSatisfy` isJust
      mkPort 8080 `shouldSatisfy` isJust
      mkPort 65535 `shouldSatisfy` isJust

    it "rejects invalid ports" $ do
      mkPort 0 `shouldBe` Nothing
      mkPort (-1) `shouldBe` Nothing
      mkPort 65536 `shouldBe` Nothing
      mkPort 100000 `shouldBe` Nothing

    it "property: valid ports are in range [1, 65535]" $
      property $ \n ->
        case mkPort n of
          Just (Port p) -> p >= 1 && p <= 65535
          Nothing -> n < 1 || n > 65535

  describe "parseConfig" $ do
    it "rejects negative port numbers (BUG #456)" $ do
      let configText = "port: -8080\nhost: localhost"
      parseConfig configText `shouldBe` Left (InvalidPort (-8080))

    it "rejects port 0 (BUG #456)" $ do
      let configText = "port: 0\nhost: localhost"
      parseConfig configText `shouldBe` Left (InvalidPort 0)

    it "rejects ports above valid range (BUG #456)" $ do
      let configText = "port: 70000\nhost: localhost"
      parseConfig configText `shouldBe` Left (InvalidPort 70000)

    it "accepts valid configurations" $ do
      let configText = "port: 8080\nhost: localhost"
      case parseConfig configText of
        Right config -> do
          getPort (configPort config) `shouldBe` 8080
        Left err -> expectationFailure $ "Valid config rejected: " <> show err

    it "rejects empty host" $ do
      let configText = "port: 8080\nhost: "
      parseConfig configText `shouldSatisfy` isLeft

isJust :: Maybe a -> Bool
isJust (Just _) = True
isJust Nothing = False

isLeft :: Either a b -> Bool
isLeft (Left _) = True
isLeft _ = False
```

**Step 5: Verify and Document**

```bash
# Run regression tests
cabal test config-test
# All tests pass ✓

# Run property-based tests with more cases
cabal test config-test --test-options="--qc-max-success=1000"
# 1000 tests passed ✓

# Generate documentation showing fix
cabal haddock

# Commit with reference
git commit -m "fix: Validate port range in parseConfig

Fixes #456

- Added Port newtype with smart constructor mkPort
- Added Host newtype with smart constructor mkHost
- Reject ports outside valid range [1, 65535]
- Added comprehensive property-based tests
- Made invalid states unrepresentable at type level"
```

---

## 3. Project Structure

### Hexagonal Architecture

```
my-haskell-project/
├── cabal.project              # Multi-package project file
├── my-haskell-project.cabal   # Package definition
├── src/
│   ├── Domain/               # Core business logic (pure)
│   │   ├── User.hs
│   │   ├── Order.hs
│   │   └── Product.hs
│   ├── Application/          # Use cases (orchestration)
│   │   ├── CreateOrder.hs
│   │   ├── ProcessPayment.hs
│   │   └── Ports/
│   │       ├── OrderRepository.hs    # Port (interface)
│   │       ├── PaymentGateway.hs
│   │       └── EmailService.hs
│   └── Infrastructure/       # Adapters (impure)
│       ├── Persistence/
│       │   └── PostgreSQL/
│       │       └── OrderRepositoryImpl.hs  # Adapter
│       ├── Payment/
│       │   └── Stripe/
│       │       └── StripeAdapter.hs
│       └── Email/
│           └── SMTP/
│               └── SMTPAdapter.hs
├── app/
│   └── Main.hs              # Application entry point
├── test/
│   ├── Spec.hs              # Test entry point
│   └── Spec/
│       ├── Domain/
│       │   ├── UserSpec.hs
│       │   └── OrderSpec.hs
│       ├── Application/
│       │   └── CreateOrderSpec.hs
│       └── Infrastructure/
│           └── Persistence/
│               └── PostgreSQL/
│                   └── OrderRepositorySpec.hs
└── benchmark/
    └── Main.hs              # Performance benchmarks
```

### Example: Hexagonal Architecture Implementation

**Domain Layer (Pure)**

```haskell
-- src/Domain/Order.hs
{-# LANGUAGE DerivingStrategies #-}

module Domain.Order where

import Data.Time.Clock (UTCTime)
import Data.Text (Text)

newtype OrderId = OrderId Text
  deriving stock (Eq, Show)

data Order = Order
  { orderId :: OrderId
  , orderCustomerId :: Text
  , orderTotal :: Double
  , orderCreatedAt :: UTCTime
  } deriving stock (Eq, Show)
```

**Port (Interface)**

```haskell
-- src/Application/Ports/OrderRepository.hs
{-# LANGUAGE RankNTypes #-}

module Application.Ports.OrderRepository
  ( OrderRepository(..)
  , MonadOrderRepository(..)
  ) where

import Domain.Order
import Control.Monad.IO.Class (MonadIO)

-- | Port: Abstract interface for order persistence.
--
-- This is a port in hexagonal architecture - it defines what operations
-- are needed without specifying how they're implemented.
data OrderRepository m = OrderRepository
  { saveOrder :: Order -> m (Either Text OrderId)
    -- ^ Persist an order
  , findOrder :: OrderId -> m (Maybe Order)
    -- ^ Retrieve an order by ID
  , listOrders :: m [Order]
    -- ^ List all orders
  }

-- | Type class version of OrderRepository for use with mtl-style effects.
class Monad m => MonadOrderRepository m where
  saveOrderM :: Order -> m (Either Text OrderId)
  findOrderM :: OrderId -> m (Maybe Order)
  listOrdersM :: m [Order]
```

**Adapter (Implementation)**

```haskell
-- src/Infrastructure/Persistence/PostgreSQL/OrderRepositoryImpl.hs
{-# LANGUAGE OverloadedStrings #-}

module Infrastructure.Persistence.PostgreSQL.OrderRepositoryImpl
  ( postgresOrderRepository
  , withPostgresConnection
  ) where

import Application.Ports.OrderRepository
import Domain.Order
import Database.PostgreSQL.Simple
import Control.Exception (bracket)
import Data.Text (Text)

-- | Adapter: PostgreSQL implementation of OrderRepository port.
--
-- This is an adapter in hexagonal architecture - it implements the port
-- interface for a specific technology (PostgreSQL).
postgresOrderRepository :: Connection -> OrderRepository IO
postgresOrderRepository conn = OrderRepository
  { saveOrder = \order -> do
      -- Implementation using PostgreSQL
      execute conn
        "INSERT INTO orders (id, customer_id, total, created_at) VALUES (?, ?, ?, ?)"
        (orderId order, orderCustomerId order, orderTotal order, orderCreatedAt order)
      pure $ Right (orderId order)

  , findOrder = \oid -> do
      results <- query conn
        "SELECT id, customer_id, total, created_at FROM orders WHERE id = ?"
        (Only oid)
      pure $ case results of
        [(oid', customerId, total, createdAt)] ->
          Just $ Order oid' customerId total createdAt
        _ -> Nothing

  , listOrders = do
      results <- query_ conn
        "SELECT id, customer_id, total, created_at FROM orders"
      pure [ Order oid customerId total createdAt
           | (oid, customerId, total, createdAt) <- results
           ]
  }

-- | Helper to manage PostgreSQL connections.
withPostgresConnection :: ConnectInfo -> (Connection -> IO a) -> IO a
withPostgresConnection connInfo = bracket
  (connect connInfo)
  close
```

**Application Layer (Use Case)**

```haskell
-- src/Application/CreateOrder.hs
{-# LANGUAGE OverloadedStrings #-}

module Application.CreateOrder
  ( CreateOrderRequest(..)
  , CreateOrderResponse(..)
  , createOrderUseCase
  ) where

import Domain.Order
import Application.Ports.OrderRepository
import Data.Time.Clock (getCurrentTime)
import Data.Text (Text)

-- | Request DTO for creating an order.
data CreateOrderRequest = CreateOrderRequest
  { reqOrderId :: OrderId
  , reqCustomerId :: Text
  , reqTotal :: Double
  } deriving (Show, Eq)

-- | Response DTO for order creation.
data CreateOrderResponse
  = OrderCreated OrderId
  | OrderCreationFailed Text
  deriving (Show, Eq)

-- | Use case: Create a new order.
--
-- This orchestrates the domain logic and port interactions.
-- It's independent of infrastructure details.
createOrderUseCase
  :: OrderRepository IO
  -> CreateOrderRequest
  -> IO CreateOrderResponse
createOrderUseCase repo req = do
  now <- getCurrentTime
  let order = Order
        { orderId = reqOrderId req
        , orderCustomerId = reqCustomerId req
        , orderTotal = reqTotal req
        , orderCreatedAt = now
        }

  result <- saveOrder repo order
  pure $ case result of
    Right oid -> OrderCreated oid
    Left err -> OrderCreationFailed err
```

**Main Application (Wiring)**

```haskell
-- app/Main.hs
{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Application.CreateOrder
import Application.Ports.OrderRepository
import Infrastructure.Persistence.PostgreSQL.OrderRepositoryImpl
import Domain.Order
import Database.PostgreSQL.Simple (defaultConnectInfo, connectDatabase)

main :: IO ()
main = do
  let connInfo = defaultConnectInfo { connectDatabase = "myapp" }

  withPostgresConnection connInfo $ \conn -> do
    let repo = postgresOrderRepository conn

    -- Example: Create an order
    let request = CreateOrderRequest
          { reqOrderId = OrderId "ORD-001"
          , reqCustomerId = "CUST-001"
          , reqTotal = 99.99
          }

    response <- createOrderUseCase repo request
    print response
```

---

## 4. Cabal Configuration

### Modern cabal.project

```cabal
-- cabal.project
packages: .

-- Use latest GHC
with-compiler: ghc-9.4.8

-- Optimization flags
package *
  optimization: 2
  ghc-options: -O2 -Wall -Wcompat -Wincomplete-record-updates -Wincomplete-uni-patterns

-- Enable parallel builds
jobs: $ncpus

-- Package-specific options
package my-haskell-project
  ghc-options: -Werror -threaded -rtsopts -with-rtsopts=-N

-- Freeze file for reproducible builds
-- Generate with: cabal freeze
```

### Modern .cabal File

```cabal
cabal-version:      3.0
name:               my-haskell-project
version:            0.1.0.0
synopsis:           Modern Haskell project with TDD
description:        A well-architected Haskell application following hexagonal architecture
license:            MIT
license-file:       LICENSE
author:             Your Name
maintainer:         your.email@example.com
category:           Application
build-type:         Simple
tested-with:        GHC == 9.4.8, GHC == 9.6.3
extra-source-files:
  README.md
  CHANGELOG.md

-- Common stanza for shared options
common common-options
  default-language:   GHC2021
  default-extensions:
    -- Deriving
    DerivingStrategies
    DeriveGeneric
    DerivingVia
    GeneralizedNewtypeDeriving
    -- Types
    StrictData
    OverloadedStrings
    TypeApplications
    -- Other
    LambdaCase
    MultiWayIf
    NamedFieldPuns
    RecordWildCards

  ghc-options:
    -Wall
    -Wcompat
    -Widentities
    -Wincomplete-record-updates
    -Wincomplete-uni-patterns
    -Wmissing-home-modules
    -Wpartial-fields
    -Wredundant-constraints
    -Wunused-packages

  build-depends:
    , base               >=4.17 && <5
    , text               >=2.0
    , time               >=1.12
    , bytestring         >=0.11

library
  import:           common-options
  hs-source-dirs:   src
  exposed-modules:
    Domain.Order
    Domain.Product
    Application.CreateOrder
    Application.Ports.OrderRepository
    Infrastructure.Persistence.PostgreSQL.OrderRepositoryImpl
  other-modules:
    -- Internal modules
  build-depends:
    , postgresql-simple  >=0.6

executable my-haskell-project
  import:           common-options
  hs-source-dirs:   app
  main-is:          Main.hs
  build-depends:
    , my-haskell-project
  ghc-options:
    -threaded
    -rtsopts
    -with-rtsopts=-N

-- Unit tests
test-suite my-haskell-project-test
  import:           common-options
  type:             exitcode-stdio-1.0
  hs-source-dirs:   test
  main-is:          Spec.hs
  other-modules:
    Spec.Domain.OrderSpec
    Spec.Domain.ProductSpec
    Spec.Application.CreateOrderSpec
  build-depends:
    , my-haskell-project
    , hspec              >=2.10
    , QuickCheck         >=2.14
    , hspec-discover     >=2.10
  build-tool-depends:
    , hspec-discover:hspec-discover
  ghc-options:
    -threaded
    -rtsopts
    -with-rtsopts=-N

-- Property-based tests
test-suite my-haskell-project-property-test
  import:           common-options
  type:             exitcode-stdio-1.0
  hs-source-dirs:   test
  main-is:          PropertySpec.hs
  build-depends:
    , my-haskell-project
    , QuickCheck         >=2.14
    , quickcheck-instances >=0.3
  ghc-options:
    -threaded

-- Benchmarks
benchmark my-haskell-project-bench
  import:           common-options
  type:             exitcode-stdio-1.0
  hs-source-dirs:   benchmark
  main-is:          Main.hs
  build-depends:
    , my-haskell-project
    , criterion          >=1.6
  ghc-options:
    -threaded
    -rtsopts
    -with-rtsopts=-N
```

### Cabal Workflow

```bash
# Initialize new project
cabal init --interactive

# Build project
cabal build

# Run tests
cabal test

# Run specific test suite
cabal test my-haskell-project-test

# Run tests with coverage
cabal test --enable-coverage

# Generate coverage report
cabal test --enable-coverage
# Report: dist-newstyle/build/.../hpc/

# Run benchmarks
cabal bench

# Generate documentation
cabal haddock --haddock-all --haddock-hyperlink-source

# Install executable
cabal install

# Clean build artifacts
cabal clean

# Update package index
cabal update

# Freeze dependencies for reproducible builds
cabal freeze

# Run executable
cabal run my-haskell-project

# Start REPL with project loaded
cabal repl

# Format code (requires ormolu)
find src test -name "*.hs" -exec ormolu --mode inplace {} \;

# Lint code
hlint src/ test/
```

---

## 5. Testing Best Practices

### A. Testing Stack

```cabal
-- In .cabal file
test-suite my-project-test
  build-depends:
    , base
    , my-project
    -- Unit testing
    , hspec              >=2.10
    , HUnit              >=1.6
    -- Property-based testing
    , QuickCheck         >=2.14
    , quickcheck-instances
    -- Mocking
    , hspec-mock         >=0.1
    -- Coverage
    , hpc-codecov        >=0.3
```

### B. HSpec with QuickCheck

```haskell
-- test/Spec/Calculator/CalculatorSpec.hs
{-# LANGUAGE OverloadedStrings #-}

module Spec.Calculator.CalculatorSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Calculator

spec :: Spec
spec = do
  describe "add" $ do
    it "adds two numbers" $ do
      add 2 3 `shouldBe` 5
      add (-1) 1 `shouldBe` 0

    it "is commutative" $
      property $ \x y ->
        add x y === add y x

    it "is associative" $
      property $ \x y z ->
        add (add x y) z === add x (add y z)

    it "zero is identity" $
      property $ \x ->
        add x 0 === x

  describe "multiply" $ do
    it "multiplies two numbers" $ do
      multiply 2 3 `shouldBe` 6
      multiply (-2) 3 `shouldBe` (-6)

    it "is commutative" $
      property $ \x y ->
        multiply x y === multiply y x

    it "distributes over addition" $
      property $ \x y z ->
        multiply x (add y z) === add (multiply x y) (multiply x z)

  describe "divide" $ do
    it "divides two numbers" $ do
      divide 6 2 `shouldBe` Right 3
      divide 5 2 `shouldBe` Right 2  -- Integer division

    it "returns error for division by zero" $ do
      divide 10 0 `shouldBe` Left "Division by zero"

    it "property: dividing and multiplying reverses" $
      property $ \x y ->
        y /= 0 ==>
          case divide x y of
            Right result -> multiply result y === (x - (x `mod` y))
            Left _ -> property True
```

### C. Property-Based Testing with QuickCheck

```haskell
-- test/Spec/Data/SortSpec.hs
module Spec.Data.SortSpec (spec) where

import Test.Hspec
import Test.QuickCheck
import Data.List (sort)
import Data.Sort (quickSort, mergeSort)

spec :: Spec
spec = do
  describe "quickSort" $ do
    it "produces sorted output" $
      property $ \xs ->
        isSorted (quickSort (xs :: [Int]))

    it "produces same result as Data.List.sort" $
      property $ \xs ->
        quickSort (xs :: [Int]) === sort xs

    it "is idempotent" $
      property $ \xs ->
        let sorted = quickSort (xs :: [Int])
        in quickSort sorted === sorted

    it "preserves length" $
      property $ \xs ->
        length (quickSort (xs :: [Int])) === length xs

  describe "mergeSort" $ do
    it "produces sorted output" $
      property $ \xs ->
        isSorted (mergeSort (xs :: [Int]))

    it "produces same result as quickSort" $
      property $ \xs ->
        mergeSort (xs :: [Int]) === quickSort xs

isSorted :: Ord a => [a] -> Bool
isSorted [] = True
isSorted [_] = True
isSorted (x:y:xs) = x <= y && isSorted (y:xs)

-- Custom generator for sorted lists
newtype SortedList a = SortedList [a]
  deriving (Show, Eq)

instance (Ord a, Arbitrary a) => Arbitrary (SortedList a) where
  arbitrary = SortedList . sort <$> arbitrary

-- Custom generator for non-empty lists
newtype NonEmptyList a = NonEmptyList [a]
  deriving (Show, Eq)

instance Arbitrary a => Arbitrary (NonEmptyList a) where
  arbitrary = NonEmptyList <$> listOf1 arbitrary
```

### D. Mocking with Type Classes

```haskell
-- test/Spec/Application/CreateOrderSpec.hs
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Spec.Application.CreateOrderSpec (spec) where

import Test.Hspec
import Application.CreateOrder
import Application.Ports.OrderRepository
import Domain.Order
import Control.Monad.IO.Class (MonadIO, liftIO)
import Data.IORef

-- Mock repository for testing
newtype MockRepo a = MockRepo (IO a)
  deriving (Functor, Applicative, Monad, MonadIO)

runMockRepo :: MockRepo a -> IO a
runMockRepo (MockRepo action) = action

-- Create mock OrderRepository with IORef for state
mkMockOrderRepository :: IO (OrderRepository MockRepo, IORef [Order])
mkMockOrderRepository = do
  ordersRef <- newIORef []
  let repo = OrderRepository
        { saveOrder = \order -> MockRepo $ do
            modifyIORef' ordersRef (order :)
            pure $ Right (orderId order)
        , findOrder = \oid -> MockRepo $ do
            orders <- readIORef ordersRef
            pure $ find (\o -> orderId o == oid) orders
        , listOrders = MockRepo $ readIORef ordersRef
        }
  pure (repo, ordersRef)

spec :: Spec
spec = do
  describe "createOrderUseCase" $ do
    it "saves order to repository" $ do
      (repo, ordersRef) <- mkMockOrderRepository

      let request = CreateOrderRequest
            (OrderId "ORD-001")
            "CUST-001"
            99.99

      response <- runMockRepo $ do
        liftIO $ createOrderUseCase (liftOrderRepo repo) request

      response `shouldSatisfy` \case
        OrderCreated _ -> True
        _ -> False

      orders <- readIORef ordersRef
      length orders `shouldBe` 1
      orderId (head orders) `shouldBe` OrderId "ORD-001"

-- Helper to lift MockRepo to IO
liftOrderRepo :: OrderRepository MockRepo -> OrderRepository IO
liftOrderRepo repo = OrderRepository
  { saveOrder = runMockRepo . saveOrder repo
  , findOrder = runMockRepo . findOrder repo
  , listOrders = runMockRepo $ listOrders repo
  }

find :: (a -> Bool) -> [a] -> Maybe a
find _ [] = Nothing
find p (x:xs)
  | p x = Just x
  | otherwise = find p xs
```

---

## 6. Documentation (Haddock)

### A. Module Documentation

```haskell
-- src/Data/TextUtils.hs
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Data.TextUtils
Description : Text manipulation utilities
Copyright   : (c) Your Name, 2026
License     : MIT
Maintainer  : your.email@example.com
Stability   : experimental
Portability : POSIX

This module provides utilities for text manipulation including
palindrome detection, word reversal, and text normalization.

==== __Examples__

>>> import Data.TextUtils
>>> isPalindrome "racecar"
True

>>> reverseWords "hello world"
"world hello"

==== __Performance Considerations__

All functions in this module operate in O(n) time where n is the
length of the input text. Text operations are optimized for UTF-8.
-}
module Data.TextUtils
  ( -- * Palindrome Detection
    isPalindrome
  , normalize
    -- * Word Manipulation
  , reverseWords
  , countWords
    -- * Types
  , NormalizedText
  , mkNormalizedText
  ) where

import Data.Text (Text)
import qualified Data.Text as T
```

### B. Function Documentation

```haskell
-- | Check if a string is a palindrome.
--
-- A palindrome reads the same forwards and backwards, ignoring case
-- and non-alphanumeric characters.
--
-- ==== __Examples__
--
-- Basic usage:
--
-- >>> isPalindrome "racecar"
-- True
--
-- >>> isPalindrome "hello"
-- False
--
-- Case insensitive:
--
-- >>> isPalindrome "RaceCar"
-- True
--
-- Ignores punctuation:
--
-- >>> isPalindrome "A man, a plan, a canal: Panama"
-- True
--
-- ==== __Properties__
--
-- prop> isPalindrome text == isPalindrome (T.reverse text)
-- prop> isPalindrome text == isPalindrome (T.toUpper text)
--
-- ==== __Complexity__
--
-- * Time: O(n) where n is the length of the text
-- * Space: O(n) for the normalized text
--
-- ==== __See Also__
--
-- * 'normalize' - The normalization function used internally
-- * 'Data.Text.reverse' - Text reversal
--
-- @since 0.1.0.0
isPalindrome :: Text -> Bool
isPalindrome text = normalized == T.reverse normalized
  where
    normalized = normalize text

-- | Normalize text for comparison.
--
-- Removes all non-alphanumeric characters and converts to lowercase.
--
-- >>> normalize "Hello, World!"
-- "helloworld"
--
-- >>> normalize "A-B-C"
-- "abc"
normalize :: Text -> Text
normalize = T.map toLower . T.filter isAlphaNum
```

### C. Type Documentation

```haskell
-- | A validated, normalized text value.
--
-- Constructor is not exported. Use 'mkNormalizedText' smart constructor.
--
-- ==== __Invariants__
--
-- * Text is non-empty
-- * Text contains only lowercase alphanumeric characters
-- * No whitespace or punctuation
--
-- ==== __Examples__
--
-- >>> mkNormalizedText "Hello"
-- Just (NormalizedText "hello")
--
-- >>> mkNormalizedText ""
-- Nothing
--
-- @since 0.1.0.0
newtype NormalizedText = NormalizedText Text
  deriving stock (Eq, Show, Ord)

-- | Smart constructor for 'NormalizedText'.
--
-- Returns 'Nothing' if text is empty or invalid.
--
-- >>> mkNormalizedText "valid"
-- Just (NormalizedText "valid")
--
-- >>> mkNormalizedText ""
-- Nothing
mkNormalizedText :: Text -> Maybe NormalizedText
mkNormalizedText text
  | T.null normalized = Nothing
  | otherwise = Just (NormalizedText normalized)
  where
    normalized = normalize text
```

### D. Generate Documentation

```bash
# Generate HTML documentation
cabal haddock --haddock-all --haddock-hyperlink-source --haddock-quickjump

# Output: dist-newstyle/build/.../doc/html/my-project/index.html

# Generate documentation with coverage report
cabal haddock --haddock-all --haddock-internal

# Upload to Hackage (if publishing)
cabal upload --publish --documentation
```

---

## 7. Code Quality Tools

### A. HLint Configuration

```yaml
# .hlint.yaml
- arguments:
  - --color=auto
  - --cpp-include=dist-newstyle/build

- ignore:
    name: Use <$>
    within:
      - Main  # Allow explicit fmap in Main for clarity

- warn:
    lhs: map f (map g xs)
    rhs: map (f . g) xs
    name: Use map fusion

- error:
    lhs: length xs > 0
    rhs: not (null xs)
    name: Use null for emptiness check

- functions:
  - {name: unsafePerformIO, within: []} # Ban unsafePerformIO everywhere

- modules:
  - {name: [Debug.Trace], within: []} # Ban Debug.Trace in production
```

### B. Ormolu Configuration

```yaml
# fourmolu.yaml
indentation: 2
function-arrows: trailing
comma-style: trailing
import-export-style: leading
indent-wheres: true
record-brace-space: true
newlines-between-decls: 1
haddock-style: multi-line
let-style: inline
in-style: right-align
respectful: true
fixities: []
```

### C. GHC Options for Quality

```cabal
-- In .cabal file
ghc-options:
  -- Warnings
  -Wall
  -Wcompat
  -Widentities
  -Wincomplete-record-updates
  -Wincomplete-uni-patterns
  -Wmissing-home-modules
  -Wpartial-fields
  -Wredundant-constraints
  -Wunused-packages
  -Wunused-type-patterns
  -Wmissing-deriving-strategies

  -- Errors (in development)
  -Werror

  -- Optimization
  -O2
  -optc-O3
```

### D. Pre-commit Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Format code
echo "Formatting code..."
find src test -name "*.hs" -exec ormolu --mode check {} \; || {
  echo "Code not formatted. Run: find src test -name '*.hs' -exec ormolu --mode inplace {} \;"
  exit 1
}

# Lint
echo "Linting..."
hlint src/ test/ || exit 1

# Build
echo "Building..."
cabal build || exit 1

# Test
echo "Testing..."
cabal test || exit 1

echo "All checks passed!"
```

---

## 8. CI/CD with Cabal

### A. GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-test:
    name: Build and Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        ghc: ['9.4.8', '9.6.3']
        cabal: ['3.10']

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Haskell
        uses: haskell-actions/setup@v2
        with:
          ghc-version: ${{ matrix.ghc }}
          cabal-version: ${{ matrix.cabal }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cabal/store
            dist-newstyle
          key: ${{ runner.os }}-${{ matrix.ghc }}-${{ hashFiles('**/*.cabal', '**/cabal.project') }}
          restore-keys: |
            ${{ runner.os }}-${{ matrix.ghc }}-

      - name: Update package index
        run: cabal update

      - name: Install dependencies
        run: cabal build --only-dependencies --enable-tests --enable-benchmarks

      - name: Build
        run: cabal build --enable-tests --enable-benchmarks

      - name: Run tests
        run: cabal test --test-show-details=direct

      - name: Run HLint
        run: |
          cabal install hlint --install-method=copy --overwrite-policy=always
          hlint src/ test/

      - name: Generate documentation
        run: cabal haddock --haddock-all

      - name: Check documentation coverage
        run: |
          cabal haddock --haddock-all 2>&1 | tee haddock.log
          ! grep -i "missing documentation" haddock.log

      - name: Generate coverage report
        if: matrix.os == 'ubuntu-latest' && matrix.ghc == '9.4.8'
        run: |
          cabal test --enable-coverage
          cabal install hpc-codecov
          hpc-codecov --format=codecov --output=codecov.json --exclude=Main --exclude=Paths_my_project cabal:my-project-test

      - name: Upload coverage to Codecov
        if: matrix.os == 'ubuntu-latest' && matrix.ghc == '9.4.8'
        uses: codecov/codecov-action@v3
        with:
          files: ./codecov.json

  lint:
    name: Code Quality
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Haskell
        uses: haskell-actions/setup@v2
        with:
          ghc-version: '9.4.8'
          cabal-version: '3.10'

      - name: Install Ormolu
        run: cabal install ormolu --install-method=copy

      - name: Check formatting
        run: |
          find src test -name "*.hs" -exec ormolu --mode check {} \;

      - name: Install HLint
        run: cabal install hlint --install-method=copy

      - name: Run HLint
        run: hlint src/ test/ --report

  build-docs:
    name: Build Documentation
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Haskell
        uses: haskell-actions/setup@v2
        with:
          ghc-version: '9.4.8'
          cabal-version: '3.10'

      - name: Generate documentation
        run: |
          cabal update
          cabal haddock --haddock-all --haddock-hyperlink-source --haddock-quickjump

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./dist-newstyle/build/x86_64-linux/ghc-9.4.8/my-project-0.1.0.0/doc/html/my-project
```

---

## 9. Modern Haskell Features

### A. GHC2021 Extensions

```haskell
-- In .cabal file
default-language: GHC2021

-- Enables:
-- BangPatterns, BinaryLiterals, ConstrainedClassMethods,
-- ConstraintKinds, DeriveDataTypeable, DeriveFoldable,
-- DeriveFunctor, DeriveGeneric, DeriveLift, DeriveTraversable,
-- DoAndIfThenElse, EmptyCase, EmptyDataDecls,
-- EmptyDataDeriving, ExistentialQuantification,
-- ExplicitForAll, FlexibleContexts, FlexibleInstances,
-- ForeignFunctionInterface, GADTSyntax,
-- GeneralisedNewtypeDeriving, HexFloatLiterals,
-- ImplicitPrelude, ImportQualifiedPost, InstanceSigs,
-- KindSignatures, MonomorphismRestriction, MultiParamTypeClasses,
-- NamedFieldPuns, NamedWildCards, NumericUnderscores,
-- PatternGuards, PolyKinds, PostfixOperators, RankNTypes,
-- RelaxedPolyRec, ScopedTypeVariables, StandaloneDeriving,
-- StarIsType, TraditionalRecordSyntax, TupleRecordSyntax,
-- TypeApplications, TypeOperators, TypeSynonymInstances
```

### B. Additional Modern Extensions

```haskell
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE StrictData #-}

module ModernHaskell where

import Data.Text (Text)
import qualified Data.Text as T

-- DerivingVia for newtype derivation
newtype Email = Email Text
  deriving (Show, Eq, Ord) via Text

-- DuplicateRecordFields
data User = User
  { name :: Text
  , email :: Email
  }

data Company = Company
  { name :: Text  -- Same field name as User
  , domain :: Text
  }

-- OverloadedRecordDot (GHC 9.2+)
getUserEmail :: User -> Email
getUserEmail user = user.email

-- LambdaCase
processResult :: Either String Int -> Int
processResult = \case
  Left _ -> 0
  Right n -> n

-- BlockArguments
withResource :: IO a -> IO a
withResource action = do
  putStrLn "Acquiring resource"
  result <- action
  putStrLn "Releasing resource"
  pure result

useResource :: IO ()
useResource = withResource do
  putStrLn "Using resource"
  pure ()

-- RecordWildCards
printUser :: User -> IO ()
printUser User{..} = do
  putStrLn $ "Name: " <> T.unpack name
  putStrLn $ "Email: " <> show email
```

### C. Type-Level Programming

```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}

module TypeLevel where

import Data.Kind (Type)
import GHC.TypeLits

-- Type-safe vector with length encoded in type
data Vector (n :: Nat) a where
  VNil :: Vector 0 a
  VCons :: a -> Vector n a -> Vector (n + 1) a

-- Safe head for non-empty vectors
vhead :: Vector (n + 1) a -> a
vhead (VCons x _) = x

-- Safe tail for non-empty vectors
vtail :: Vector (n + 1) a -> Vector n a
vtail (VCons _ xs) = xs

-- Append vectors with length tracking
vappend :: Vector n a -> Vector m a -> Vector (n + m) a
vappend VNil ys = ys
vappend (VCons x xs) ys = VCons x (vappend xs ys)

-- Type family for type-level computation
type family Add (n :: Nat) (m :: Nat) :: Nat where
  Add 0 m = m
  Add n m = Add (n - 1) (m + 1)

-- Example: Compile-time verified index
data Fin (n :: Nat) where
  FZ :: Fin (n + 1)
  FS :: Fin n -> Fin (n + 1)

-- Safe indexing with compile-time bounds checking
vindex :: Fin n -> Vector n a -> a
vindex FZ (VCons x _) = x
vindex (FS i) (VCons _ xs) = vindex i xs
```

---

## 10. Prohibited Practices

### Never Do:

1. **Partial Functions**
   ```haskell
   -- ❌ BAD: Partial functions that can crash
   unsafeHead :: [a] -> a
   unsafeHead (x:_) = x
   -- Crashes on empty list!

   -- ✅ GOOD: Total functions
   safeHead :: [a] -> Maybe a
   safeHead [] = Nothing
   safeHead (x:_) = Just x
   ```

2. **Missing Type Signatures**
   ```haskell
   -- ❌ BAD: No type signature
   calculate x y = x * 2 + y

   -- ✅ GOOD: Explicit type signature
   calculate :: Int -> Int -> Int
   calculate x y = x * 2 + y
   ```

3. **Lazy IO**
   ```haskell
   -- ❌ BAD: Lazy IO leads to resource leaks
   readFileLazy :: FilePath -> IO String
   readFileLazy = readFile

   -- ✅ GOOD: Strict IO with proper resource management
   import qualified Data.Text.IO as TIO
   readFileStrict :: FilePath -> IO Text
   readFileStrict = TIO.readFile
   ```

4. **String Instead of Text**
   ```haskell
   -- ❌ BAD: String is inefficient
   greet :: String -> String
   greet name = "Hello, " ++ name

   -- ✅ GOOD: Use Text for efficiency
   import Data.Text (Text)
   import qualified Data.Text as T
   
   greet :: Text -> Text
   greet name = "Hello, " <> name
   ```

5. **Orphan Instances**
   ```haskell
   -- ❌ BAD: Orphan instance (defined in neither type nor class module)
   -- In module Other.hs:
   instance Show MyType where
     show = ...

   -- ✅ GOOD: Define instance in type's module or newtype
   newtype MyTypeShow = MyTypeShow MyType
   instance Show MyTypeShow where
     show (MyTypeShow mt) = ...
   ```

6. **Skipping Tests**
   ```haskell
   -- ❌ BAD: No tests
   -- Just implement and hope it works

   -- ✅ GOOD: Test first (TDD)
   spec :: Spec
   spec = describe "myFunction" $ do
     it "handles empty input" $
       myFunction [] `shouldBe` expected
   ```

7. **Fixing Bugs Without Regression Tests**
   ```haskell
   -- ❌ BAD: Fix bug without test
   -- Just change the code

   -- ✅ GOOD: Write failing test first
   it "BUG #123: handles negative numbers" $
     process (-5) `shouldBe` expected
   -- Then fix the bug
   ```

8. **Incomplete Pattern Matches**
   ```haskell
   -- ❌ BAD: Incomplete patterns
   process :: Maybe Int -> Int
   process (Just x) = x * 2
   -- Missing Nothing case!

   -- ✅ GOOD: Complete patterns
   process :: Maybe Int -> Int
   process (Just x) = x * 2
   process Nothing = 0
   ```

9. **Using `error` or `undefined` in Production**
   ```haskell
   -- ❌ BAD: Runtime crash
   divide :: Int -> Int -> Int
   divide x 0 = error "Division by zero"
   divide x y = x `div` y

   -- ✅ GOOD: Use types to represent failure
   divide :: Int -> Int -> Either String Int
   divide x 0 = Left "Division by zero"
   divide x y = Right (x `div` y)
   ```

10. **Non-exhaustive Guards**
    ```haskell
    -- ❌ BAD: Non-exhaustive guards
    classify :: Int -> String
    classify n
      | n > 0 = "positive"
      | n < 0 = "negative"
      -- Missing n == 0 case!

    -- ✅ GOOD: Exhaustive guards
    classify :: Int -> String
    classify n
      | n > 0 = "positive"
      | n < 0 = "negative"
      | otherwise = "zero"
    ```

---

## 11. Deployment Checklist

### Before Every Release:

#### Test-Driven Development (TDD) Compliance
- [ ] All features developed with TDD (tests written first)
- [ ] All bug fixes have regression tests
- [ ] No tests skipped or marked as pending
- [ ] Property-based tests for critical functions

#### Code Quality
- [ ] All tests pass: `cabal test`
- [ ] No HLint warnings: `hlint src/ test/`
- [ ] Code formatted: `ormolu --mode check`
- [ ] No GHC warnings: `cabal build -Wall -Werror`
- [ ] Documentation complete: `cabal haddock`
- [ ] All public functions have Haddock comments
- [ ] No partial functions in production code
- [ ] All pattern matches are exhaustive

#### Type Safety
- [ ] All top-level functions have explicit type signatures
- [ ] No use of `undefined` or `error` in production
- [ ] Smart constructors for all domain types
- [ ] Invalid states made unrepresentable
- [ ] Strictness annotations added where appropriate

#### Testing
- [ ] Unit tests cover all modules
- [ ] Property-based tests for pure functions
- [ ] Integration tests for adapters
- [ ] Test coverage > 80%: `cabal test --enable-coverage`
- [ ] Benchmarks run: `cabal bench`
- [ ] No performance regressions

#### Documentation
- [ ] README updated with usage examples
- [ ] CHANGELOG updated with changes
- [ ] API documentation generated
- [ ] Migration guide (if breaking changes)
- [ ] Architecture decision records (ADRs) updated

#### Dependencies
- [ ] Dependencies frozen: `cabal freeze`
- [ ] No unnecessary dependencies
- [ ] Security audit: `cabal outdated`
- [ ] License compliance checked
- [ ] cabal.project configured correctly

#### Build
- [ ] Clean build succeeds: `cabal clean && cabal build`
- [ ] Builds on all supported GHC versions
- [ ] Builds on all target platforms
- [ ] Docker image builds (if applicable)
- [ ] Executable runs correctly: `cabal run`

#### Hexagonal Architecture
- [ ] Domain logic is pure and isolated
- [ ] All external dependencies use ports (interfaces)
- [ ] Adapters implement ports correctly
- [ ] No business logic in adapters
- [ ] Dependency inversion maintained

#### Agent Verification
- [ ] Agent-generated code compiles
- [ ] Agent-generated tests pass
- [ ] Agent followed TDD protocol
- [ ] Agent added regression tests for bug fixes
- [ ] Agent documented all changes

#### Version Control
- [ ] All changes committed
- [ ] Commit messages follow Conventional Commits
- [ ] Version number bumped (semver)
- [ ] Git tag created
- [ ] Branch merged to main

---

## 12. Why This Configuration Works

### Type Safety
- **Smart Constructors**: Make invalid states unrepresentable
- **Explicit Types**: Every function has a type signature
- **Exhaustive Patterns**: Compiler ensures all cases handled
- **No Partial Functions**: All functions are total

### Test-Driven Development
- **TDD First**: Write tests before implementation
- **Regression Shield**: Every bug gets a test
- **Property-Based Testing**: QuickCheck finds edge cases
- **Fast Feedback**: Cabal test runs in seconds

### Clean Architecture
- **Hexagonal Design**: Domain is pure, infrastructure is separate
- **Type Classes as Ports**: Abstract interfaces for adapters
- **Dependency Inversion**: Domain doesn't depend on infrastructure
- **Testability**: Easy to mock external dependencies

### Modern Tooling
- **Cabal 3.10+**: Modern dependency management
- **GHC 9.4+**: Latest language features
- **HLS**: IDE integration with Language Server
- **HLint**: Automated code suggestions
- **Ormolu**: Consistent code formatting

### Documentation
- **Haddock**: API documentation from code
- **Examples**: Inline examples tested with doctest
- **Properties**: QuickCheck properties in documentation
- **Coverage**: Documentation coverage tracking

### Performance
- **Laziness**: Efficient processing of infinite structures
- **Strictness**: Avoid space leaks with strict fields
- **Optimization**: O2 flag for production builds
- **Profiling**: Built-in profiling support

### Maintainability
- **Immutability**: No surprising mutations
- **Pure Functions**: Easy to reason about
- **Strong Typing**: Refactoring is safe
- **Composition**: Build complex from simple

This configuration creates a solid foundation for building reliable,
maintainable Haskell applications with confidence.

---

## Additional Resources

- [Haskell Documentation](https://www.haskell.org/documentation/)
- [GHC User Guide](https://downloads.haskell.org/ghc/latest/docs/users_guide/)
- [Cabal User Guide](https://cabal.readthedocs.io/)
- [HSpec Documentation](https://hspec.github.io/)
- [QuickCheck Manual](https://hackage.haskell.org/package/QuickCheck)
- [Haddock User Guide](https://haskell-haddock.readthedocs.io/)
- [HLint Manual](https://github.com/ndmitchell/hlint#readme)
- [Hexagonal Architecture in Haskell](https://github.com/thma/PolysemyCleanArchitecture)

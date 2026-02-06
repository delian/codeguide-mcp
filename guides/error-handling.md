# Error Handling Guidelines
Mandatory standards for error handling, exception management, and failure recovery across all programming languages. Language-specific error handling, Result types, Circuit breakers, Retry libraries.

---

**Agent Profile**: The Resilience Engineer
**Role**: Senior Software Engineer & Reliability Specialist
**Objective**: Generate robust, fault-tolerant code with clear error handling and graceful degradation.
**Tools**: Language-specific error handling, Result types, Circuit breakers, Retry libraries.

---

## 1. Core Philosophies: ERROR-FIRST

The agent must adhere to the **ERROR-FIRST** principles:

- **E**xplicit: Make errors visible and explicit, not hidden
- **R**ecoverable: Provide recovery paths where possible
- **R**ich Context: Include context for debugging
- **O**bservable: Log errors appropriately for monitoring
- **R**eportable: Users get helpful, safe error messages

---

## 2. Error Types Classification (MANDATORY)

### A. Error Categories

```
RECOVERABLE ERRORS (Handle gracefully)
├── User Errors
│   ├── Invalid input
│   ├── Missing required fields
│   └── Business rule violations
├── Expected Failures
│   ├── Resource not found
│   ├── Duplicate entries
│   └── Permission denied
└── Transient Errors
    ├── Network timeouts
    ├── Service unavailable
    └── Rate limiting

UNRECOVERABLE ERRORS (Fail fast, log, alert)
├── Programming Errors
│   ├── Null pointer exceptions
│   ├── Type errors
│   └── Assertion failures
├── Configuration Errors
│   ├── Missing required config
│   ├── Invalid credentials
│   └── Malformed configuration
└── System Failures
    ├── Out of memory
    ├── Disk full
    └── Hardware failures
```

### B. Error Response Strategy

```python
# Define how to handle each category

ERROR_STRATEGY = {
    # Recoverable - User errors
    "validation_error": {
        "log_level": "INFO",
        "user_message": "Please correct the following errors",
        "include_details": True,
        "retry": False,
    },

    # Recoverable - Expected failures
    "not_found": {
        "log_level": "INFO",
        "user_message": "Resource not found",
        "include_details": False,
        "retry": False,
    },

    # Recoverable - Transient
    "service_unavailable": {
        "log_level": "WARN",
        "user_message": "Service temporarily unavailable. Please try again.",
        "include_details": False,
        "retry": True,
        "max_retries": 3,
    },

    # Unrecoverable
    "internal_error": {
        "log_level": "ERROR",
        "user_message": "An unexpected error occurred",
        "include_details": False,  # Never expose internal details
        "retry": False,
        "alert": True,
    },
}
```

---

## 3. Exception Hierarchy (MANDATORY)

### A. Custom Exception Design

```python
# Python exception hierarchy
class AppError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self):
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# User/Input errors (400)
class ValidationError(AppError):
    """Invalid input from user."""
    pass


class InvalidFieldError(ValidationError):
    """Specific field validation error."""

    def __init__(self, field: str, message: str):
        super().__init__(message, details={"field": field})
        self.field = field


# Authentication/Authorization errors (401, 403)
class AuthenticationError(AppError):
    """User is not authenticated."""
    pass


class AuthorizationError(AppError):
    """User is not authorized for this action."""
    pass


# Not found errors (404)
class NotFoundError(AppError):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str):
        super().__init__(
            f"{resource} not found",
            details={"resource": resource, "id": identifier}
        )


# Conflict errors (409)
class ConflictError(AppError):
    """Resource conflict (duplicate, version mismatch)."""
    pass


# External service errors
class ExternalServiceError(AppError):
    """Error from external service."""

    def __init__(self, service: str, message: str, original_error: Exception = None):
        super().__init__(message, details={"service": service})
        self.service = service
        self.original_error = original_error


# Transient errors (retry possible)
class TransientError(AppError):
    """Temporary error that may succeed on retry."""

    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after
```

### B. TypeScript Error Classes

```typescript
// TypeScript error hierarchy
export abstract class AppError extends Error {
  abstract readonly code: string;
  abstract readonly statusCode: number;
  readonly details: Record<string, unknown>;
  readonly isOperational: boolean = true;

  constructor(message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.name = this.constructor.name;
    this.details = details;
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON() {
    return {
      error: this.code,
      message: this.message,
      details: this.details,
    };
  }
}

// Validation errors
export class ValidationError extends AppError {
  readonly code = 'VALIDATION_ERROR';
  readonly statusCode = 400;
}

export class InvalidFieldError extends ValidationError {
  constructor(field: string, message: string) {
    super(message, { field });
  }
}

// Not found
export class NotFoundError extends AppError {
  readonly code = 'NOT_FOUND';
  readonly statusCode = 404;

  constructor(resource: string, id: string) {
    super(`${resource} not found`, { resource, id });
  }
}

// Authentication
export class AuthenticationError extends AppError {
  readonly code = 'UNAUTHENTICATED';
  readonly statusCode = 401;
}

// Authorization
export class AuthorizationError extends AppError {
  readonly code = 'FORBIDDEN';
  readonly statusCode = 403;
}

// External service
export class ExternalServiceError extends AppError {
  readonly code = 'EXTERNAL_SERVICE_ERROR';
  readonly statusCode = 502;

  constructor(service: string, message: string) {
    super(message, { service });
  }
}
```

---

## 4. Result Type Pattern

### A. Result Type (Recommended for Go, Rust, Functional)

```typescript
// TypeScript Result type
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function Ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function Err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// Usage
function divide(a: number, b: number): Result<number, string> {
  if (b === 0) {
    return Err('Division by zero');
  }
  return Ok(a / b);
}

const result = divide(10, 2);
if (result.ok) {
  console.log(result.value); // 5
} else {
  console.error(result.error);
}

// Chaining results
function parseNumber(input: string): Result<number, string> {
  const num = parseInt(input, 10);
  if (isNaN(num)) {
    return Err(`Invalid number: ${input}`);
  }
  return Ok(num);
}

function calculateDiscount(
  priceStr: string,
  discountStr: string
): Result<number, string> {
  const priceResult = parseNumber(priceStr);
  if (!priceResult.ok) return priceResult;

  const discountResult = parseNumber(discountStr);
  if (!discountResult.ok) return discountResult;

  const price = priceResult.value;
  const discount = discountResult.value;

  if (discount > 100) {
    return Err('Discount cannot exceed 100%');
  }

  return Ok(price * (1 - discount / 100));
}
```

### B. Go Error Handling

```go
// Go idiomatic error handling
package main

import (
    "errors"
    "fmt"
)

// Custom error types
type ValidationError struct {
    Field   string
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error on %s: %s", e.Field, e.Message)
}

type NotFoundError struct {
    Resource string
    ID       string
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s with id %s not found", e.Resource, e.ID)
}

// Sentinel errors
var (
    ErrNotFound      = errors.New("not found")
    ErrUnauthorized  = errors.New("unauthorized")
    ErrInvalidInput  = errors.New("invalid input")
)

// Functions return (result, error)
func GetUser(id string) (*User, error) {
    user, err := db.FindUser(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return nil, &NotFoundError{Resource: "User", ID: id}
        }
        return nil, fmt.Errorf("failed to get user %s: %w", id, err)
    }
    return user, nil
}

// Error handling with type assertions
func HandleRequest(userID string) error {
    user, err := GetUser(userID)
    if err != nil {
        var notFound *NotFoundError
        if errors.As(err, &notFound) {
            // Handle not found specifically
            return respondWithStatus(404, "User not found")
        }
        // Log and return generic error
        log.Error("Failed to get user", "error", err)
        return respondWithStatus(500, "Internal error")
    }

    // Use user..
    return nil
}

// Wrapping errors for context
func ProcessOrder(orderID string) error {
    order, err := GetOrder(orderID)
    if err != nil {
        return fmt.Errorf("processing order %s: %w", orderID, err)
    }

    if err := ValidateOrder(order); err != nil {
        return fmt.Errorf("validating order %s: %w", orderID, err)
    }

    if err := ChargePayment(order); err != nil {
        return fmt.Errorf("charging payment for order %s: %w", orderID, err)
    }

    return nil
}
```

### C. Rust Result Pattern

```rust
// Rust error handling
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Validation error: {message}")]
    Validation { field: String, message: String },

    #[error("{resource} with id {id} not found")]
    NotFound { resource: String, id: String },

    #[error("Unauthorized")]
    Unauthorized,

    #[error("External service error: {service}")]
    ExternalService {
        service: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Internal error")]
    Internal(#[from] anyhow::Error),
}

// Functions return Result<T, E>
fn get_user(id: &str) -> Result<User, AppError> {
    let user = db::find_user(id)
        .map_err(|e| AppError::Internal(e.into()))?;

    user.ok_or_else(|| AppError::NotFound {
        resource: "User".into(),
        id: id.into(),
    })
}

// Using ? operator for propagation
fn process_order(order_id: &str) -> Result<(), AppError> {
    let order = get_order(order_id)?;
    validate_order(&order)?;
    charge_payment(&order)?;
    Ok(())
}

// Pattern matching on errors
fn handle_request(user_id: &str) -> HttpResponse {
    match get_user(user_id) {
        Ok(user) => HttpResponse::Ok().json(user),
        Err(AppError::NotFound { .. }) => {
            HttpResponse::NotFound().body("User not found")
        }
        Err(AppError::Unauthorized) => {
            HttpResponse::Unauthorized().finish()
        }
        Err(e) => {
            log::error!("Internal error: {}", e);
            HttpResponse::InternalServerError().body("Internal error")
        }
    }
}
```

---

## 5. Try-Catch Patterns (MANDATORY)

### A. Proper Exception Handling

```python
# Python try-except patterns

# ✅ CORRECT - Specific exceptions, appropriate handling
def process_user_data(user_id: str, data: dict):
    try:
        user = get_user(user_id)
    except NotFoundError:
        # Expected case - handle specifically
        raise ValidationError(f"User {user_id} does not exist")
    except DatabaseError as e:
        # Infrastructure error - log and re-raise
        logger.error("Database error", user_id=user_id, error=str(e))
        raise

    try:
        validated_data = validate_data(data)
    except ValidationError:
        # Let validation errors propagate (expected)
        raise
    except Exception as e:
        # Unexpected error - wrap with context
        raise InternalError(f"Unexpected error validating data") from e

    return update_user(user, validated_data)


# ❌ WRONG - Bare except, swallowing errors
def process_data_wrong(data):
    try:
        return do_something(data)
    except:  # Catches everything including KeyboardInterrupt!
        return None  # Error is swallowed, no logging


# ❌ WRONG - Too broad exception handling
def process_data_also_wrong(data):
    try:
        validate(data)
        process(data)
        save(data)
    except Exception as e:
        # Which operation failed? No context
        logger.error("Something failed")
        raise
```

### B. Exception Handling Scope

```python
# ✅ CORRECT - Narrow try blocks

def create_order(order_data: dict) -> Order:
    # Validate (might raise ValidationError)
    validated = validate_order_data(order_data)

    # Get user (might raise NotFoundError)
    try:
        user = get_user(validated['user_id'])
    except NotFoundError:
        raise ValidationError("Invalid user_id")

    # Create order (might raise DatabaseError)
    try:
        order = Order.create(user=user, **validated)
    except IntegrityError as e:
        if "duplicate" in str(e).lower():
            raise ConflictError("Order already exists")
        raise

    # Send notification (failure shouldn't fail order creation)
    try:
        send_order_notification(order)
    except NotificationError as e:
        logger.warning("Failed to send notification", order_id=order.id, error=e)
        # Don't re-raise - order was created successfully

    return order


# ❌ WRONG - Entire function in one try block
def create_order_wrong(order_data: dict) -> Order:
    try:
        validated = validate_order_data(order_data)
        user = get_user(validated['user_id'])
        order = Order.create(user=user, **validated)
        send_order_notification(order)
        return order
    except Exception as e:
        # What failed? We don't know!
        logger.error("Order creation failed")
        raise
```

---

## 6. Error Context and Wrapping

### A. Adding Context to Errors

```python
# Python - Error wrapping with context

class ErrorContext:
    """Add context to errors while preserving original error."""

    def __init__(self, message: str, **context):
        self.message = message
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            # Re-raise with context
            raise type(exc_val)(
                f"{self.message}: {exc_val}"
            ).with_traceback(exc_tb) from exc_val
        return False


# Usage
def process_user_order(user_id: str, order_id: str):
    with ErrorContext("Processing user order", user_id=user_id, order_id=order_id):
        user = get_user(user_id)
        order = get_order(order_id)
        return process(user, order)


# Alternative: Explicit wrapping
def get_order_items(order_id: str) -> list:
    try:
        order = get_order(order_id)
    except DatabaseError as e:
        raise DatabaseError(
            f"Failed to fetch order {order_id}"
        ) from e

    try:
        items = fetch_items(order.item_ids)
    except DatabaseError as e:
        raise DatabaseError(
            f"Failed to fetch items for order {order_id}"
        ) from e

    return items
```

### B. Error Chain Preservation

```typescript
// TypeScript - Preserving error chains

class ChainedError extends Error {
  readonly cause?: Error;

  constructor(message: string, cause?: Error) {
    super(message);
    this.cause = cause;
    this.name = 'ChainedError';

    // Append cause to stack trace
    if (cause?.stack) {
      this.stack += '\nCaused by: ' + cause.stack;
    }
  }
}

// Usage
async function processPayment(orderId: string): Promise<void> {
  let order: Order;

  try {
    order = await getOrder(orderId);
  } catch (error) {
    throw new ChainedError(
      `Failed to fetch order ${orderId} for payment processing`,
      error as Error
    );
  }

  try {
    await chargeCard(order.paymentMethod, order.total);
  } catch (error) {
    throw new ChainedError(
      `Payment failed for order ${orderId}`,
      error as Error
    );
  }
}
```

---

## 7. Retry and Recovery Patterns

### A. Retry with Exponential Backoff

```python
import time
import random
from functools import wraps
from typing import Callable, TypeVar, Type

T = TypeVar('T')

class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


def retry(config: RetryConfig = None):
    """Decorator for retry with exponential backoff."""
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt == config.max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )

                    # Add jitter
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        "Retry attempt",
                        function=func.__name__,
                        attempt=attempt + 1,
                        max_attempts=config.max_attempts,
                        delay=delay,
                        error=str(e)
                    )

                    time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# Usage
@retry(RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    retryable_exceptions=(ConnectionError, TimeoutError)
))
def call_external_api(endpoint: str) -> dict:
    response = requests.get(endpoint, timeout=10)
    response.raise_for_status()
    return response.json()
```

### B. Circuit Breaker Pattern

```python
import time
from enum import Enum
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = Lock()

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == CircuitState.OPEN:
                    if self._should_attempt_reset():
                        self.state = CircuitState.HALF_OPEN
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker is open for {func.__name__}"
                        )

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise

        return wrapper

    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        with self.lock:
            self.failure_count = 0
            self.state = CircuitState.CLOSED

    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


# Usage
payment_circuit = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
    expected_exception=PaymentServiceError
)

@payment_circuit
def process_payment(amount: float) -> dict:
    return payment_service.charge(amount)
```

---

## 8. User-Facing Error Messages

### A. Error Message Guidelines

```python
# Error message formatting

class ErrorMessage:
    """User-friendly error messages."""

    # ✅ GOOD - Clear, actionable, no technical details
    MESSAGES = {
        "validation_error": "Please correct the errors below and try again.",
        "not_found": "The requested resource could not be found.",
        "unauthorized": "Please log in to continue.",
        "forbidden": "You don't have permission to perform this action.",
        "rate_limited": "Too many requests. Please wait a moment and try again.",
        "service_unavailable": "This service is temporarily unavailable. Please try again later.",
        "internal_error": "Something went wrong. Please try again or contact support.",
    }

    # ❌ BAD - Technical, scary, unhelpful
    BAD_MESSAGES = {
        "validation_error": "JSON schema validation failed at $.user.email",
        "not_found": "SELECT query returned 0 rows",
        "internal_error": "NullPointerException at line 234 in UserService.java",
    }


def format_user_error(error: AppError, include_details: bool = False) -> dict:
    """Format error for API response."""
    response = {
        "error": error.code,
        "message": ErrorMessage.MESSAGES.get(
            error.code,
            ErrorMessage.MESSAGES["internal_error"]
        ),
    }

    # Include field-level validation errors
    if isinstance(error, ValidationError) and include_details:
        response["details"] = error.details

    return response
```

### B. Field-Level Validation Errors

```python
# Detailed validation error responses

class ValidationResult:
    def __init__(self):
        self.errors: dict[str, list[str]] = {}

    def add_error(self, field: str, message: str):
        if field not in self.errors:
            self.errors[field] = []
        self.errors[field].append(message)

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def to_response(self) -> dict:
        return {
            "error": "VALIDATION_ERROR",
            "message": "Please correct the following errors",
            "details": {
                "fields": self.errors
            }
        }


def validate_user_input(data: dict) -> ValidationResult:
    result = ValidationResult()

    # Email validation
    if not data.get("email"):
        result.add_error("email", "Email is required")
    elif not is_valid_email(data["email"]):
        result.add_error("email", "Please enter a valid email address")

    # Password validation
    password = data.get("password", "")
    if len(password) < 8:
        result.add_error("password", "Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        result.add_error("password", "Password must contain an uppercase letter")
    if not any(c.isdigit() for c in password):
        result.add_error("password", "Password must contain a number")

    return result


# API response example
# {
#     "error": "VALIDATION_ERROR",
#     "message": "Please correct the following errors",
#     "details": {
#         "fields": {
#             "email": ["Please enter a valid email address"],
#             "password": [
#                 "Password must be at least 8 characters",
#                 "Password must contain a number"
#             ]
#         }
#     }
# }
```

---

## 9. Logging Errors

### A. Error Logging Best Practices

**CRITICAL: Stack traces and internal error details MUST NOT be exposed in production responses.**

```python
import traceback
from typing import Optional
import os

def log_error(
    logger,
    error: Exception,
    context: dict = None,
    include_traceback: bool = None  # Auto-detect based on environment
):
    """Log error with appropriate level and context.

    IMPORTANT: Stack traces are logged server-side for debugging but
    NEVER included in responses sent to users in production.
    """

    # Auto-detect: only include traceback in non-production logs
    if include_traceback is None:
        include_traceback = os.getenv("ENV", "production") != "production"

    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {}),
    }

    # Add traceback for internal logging only (NEVER in API responses)
    # In production, tracebacks are logged but not returned to users
    if include_traceback:
        log_data["traceback"] = traceback.format_exc()

    # Add cause chain
    if error.__cause__:
        log_data["caused_by"] = {
            "type": type(error.__cause__).__name__,
            "message": str(error.__cause__),
        }

    # Determine log level based on error type
    if isinstance(error, ValidationError):
        logger.info("Validation error", **log_data)
    elif isinstance(error, NotFoundError):
        logger.info("Resource not found", **log_data)
    elif isinstance(error, AuthorizationError):
        logger.warning("Authorization denied", **log_data)
    elif isinstance(error, TransientError):
        logger.warning("Transient error", **log_data)
    else:
        logger.error("Unexpected error", **log_data)


# Usage in exception handler
def handle_request(request):
    try:
        return process_request(request)
    except AppError as e:
        log_error(logger, e, context={
            "request_id": request.id,
            "user_id": request.user_id,
            "endpoint": request.path,
        })
        return error_response(e)
    except Exception as e:
        log_error(logger, e, context={
            "request_id": request.id,
            "endpoint": request.path,
        })
        return error_response(InternalError("Unexpected error"))
```

---

## 10. Global Error Handlers

### A. Express.js Error Handler

```typescript
// Express global error handler
import { Request, Response, NextFunction } from 'express';

function errorHandler(
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  // Log error
  logger.error('Request error', {
    error: error.message,
    stack: error.stack,
    requestId: req.id,
    path: req.path,
    method: req.method,
  });

  // Handle known error types
  if (error instanceof AppError) {
    res.status(error.statusCode).json(error.toJSON());
    return;
  }

  // Handle validation errors (e.g., from Joi, Zod)
  if (error.name === 'ValidationError') {
    res.status(400).json({
      error: 'VALIDATION_ERROR',
      message: 'Invalid request data',
      details: error.details,
    });
    return;
  }

  // Unknown error - return generic message
  res.status(500).json({
    error: 'INTERNAL_ERROR',
    message: 'An unexpected error occurred',
    requestId: req.id, // For support reference
  });
}

// Async error wrapper
const asyncHandler = (fn: Function) => (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

// Usage
app.get('/users/:id', asyncHandler(async (req, res) => {
  const user = await getUser(req.params.id);
  res.json(user);
}));

app.use(errorHandler);
```

### B. Python FastAPI Error Handler

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(AppError)
async def app_error_handler(request: Request, error: AppError):
    log_error(logger, error, context={
        "request_id": request.state.request_id,
        "path": request.url.path,
    })

    return JSONResponse(
        status_code=get_status_code(error),
        content=error.to_dict(),
    )

@app.exception_handler(Exception)
async def general_error_handler(request: Request, error: Exception):
    logger.error(
        "Unhandled exception",
        error_type=type(error).__name__,
        error_message=str(error),
        traceback=traceback.format_exc(),
        request_id=request.state.request_id,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id,
        },
    )

def get_status_code(error: AppError) -> int:
    status_codes = {
        ValidationError: 400,
        AuthenticationError: 401,
        AuthorizationError: 403,
        NotFoundError: 404,
        ConflictError: 409,
        TransientError: 503,
    }
    for error_type, code in status_codes.items():
        if isinstance(error, error_type):
            return code
    return 500
```

---

## 11. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug fix MUST have a regression test written BEFORE the fix is applied.**

### A. Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Bug Fix Workflow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Bug Reported/Discovered                                    │
│      - Document: component, input, expected vs actual behavior  │
│                          ↓                                       │
│   2. Write Test that REPRODUCES the Bug                         │
│      - Create test case with exact failing scenario             │
│      - Test MUST FAIL (proves bug exists)                       │
│                          ↓                                       │
│   3. Verify Test Fails for the RIGHT Reason                     │
│      - Confirm error matches reported bug                        │
│      - Not a different/unrelated failure                         │
│                          ↓                                       │
│   4. Fix the Bug                                                │
│      - Apply minimal fix to address the issue                   │
│                          ↓                                       │
│   5. Verify Test Now PASSES                                     │
│      - Bug is fixed                                              │
│      - All other tests still pass (no regressions)              │
│                          ↓                                       │
│   6. Document in Test Comments                                  │
│      - Include bug/ticket ID                                     │
│      - Describe original issue                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### B. Example Regression Test

```python
# tests/regression/test_bug_123_null_user_error.py

def test_bug_123_handle_null_user_gracefully():
    """
    Bug #123: System crashed with NullPointerException when user was None.

    Expected: Return NotFoundError with helpful message
    Actual: Unhandled exception crashed the request

    Regression test: Ensures null user case is handled properly.
    """
    # Arrange
    user_id = "nonexistent-user-id-12345678901234567890"

    # Act
    result = user_service.get_user(user_id)

    # Assert - should return proper error, not crash
    assert result.is_error()
    assert result.error_code == "NOT_FOUND"
    assert "user" in result.error_message.lower()
```

**Cross-reference:** See rest.md Section 2B and secure-coding.md Section 2 for TDD protocols.

---

## 12. Deployment Checklist

### Error Handling
- [ ] Custom exception hierarchy defined
- [ ] All exceptions have error codes
- [ ] Error context is preserved (wrapping)
- [ ] Specific exceptions caught (not bare except)
- [ ] Bug fixes have regression tests (written BEFORE fix)

### User Experience
- [ ] User-facing messages are helpful
- [ ] No technical details exposed to users
- [ ] Validation errors include field-level details
- [ ] Error responses include request ID

### Logging
- [ ] All errors are logged
- [ ] Log level appropriate to error type
- [ ] Context included (request ID, user ID, etc.)
- [ ] No sensitive data in error logs

### Recovery
- [ ] Retry logic for transient errors
- [ ] Circuit breakers for external services
- [ ] Graceful degradation implemented
- [ ] Timeout handling in place

---

## 12. Quick Reference

```python
# Error type decision tree
"""
Is it user's fault (bad input)?
  → ValidationError (400)

Is user not logged in?
  → AuthenticationError (401)

Is user not allowed?
  → AuthorizationError (403)

Does resource not exist?
  → NotFoundError (404)

Is there a conflict (duplicate)?
  → ConflictError (409)

Is it a temporary issue?
  → TransientError (503) + retry

Is it our fault?
  → InternalError (500) + alert
"""

# Exception handling rules
"""
1. Catch specific exceptions
2. Keep try blocks narrow
3. Add context when re-raising
4. Log at appropriate level
5. Never swallow exceptions silently
6. User messages: helpful, not technical
7. Always preserve error chain
"""
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Engineering Team


**End of Error Handling Guidelines**

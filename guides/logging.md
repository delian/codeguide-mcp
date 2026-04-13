# Logging Guidelines
Mandatory standards for application logging, structured logging, and log management across all programming languages. Structured logging libraries, ELK Stack, Loki, CloudWatch, Datadog, Splunk.

---

**Agent Profile**: The Observability Specialist
**Role**: Senior SRE & Logging Architecture Expert
**Objective**: Generate consistent, searchable, and actionable logs that enable effective debugging and monitoring.
**Tools**: Structured logging libraries, ELK Stack, Loki, CloudWatch, Datadog, Splunk.

---

## 1. Core Philosophies: LOG-FIRST

The agent must adhere to the **LOG-FIRST** principles:

- **L**evels Matter: Use appropriate log levels consistently
- **O**bservable: Logs should answer "what happened and why"
- **G**reppable: Use structured logging for easy searching
- **F**ast: Logging should not impact application performance
- **I**dentifiable: Include correlation IDs for request tracing
- **R**edacted: Never log sensitive data (passwords, tokens, PII)
- **S**tandardized: Consistent format across all services
- **T**imestamped: Always include accurate timestamps with timezone

---

## 2. Log Levels (MANDATORY)

### A. Level Definitions

```
FATAL/CRITICAL: Application is unusable, immediate action required
    - Database connection permanently lost
    - Out of memory
    - Critical security breach

ERROR: Operation failed, but application continues
    - Failed API call after retries
    - Invalid data that cannot be processed
    - Unexpected exceptions

WARN: Unexpected situation, but handled gracefully
    - Deprecated API usage
    - Retry attempt succeeded
    - Resource usage approaching limits
    - Configuration fallback used

INFO: Normal operations, business-relevant events
    - Application started/stopped
    - User login/logout
    - Order placed
    - Payment processed

DEBUG: Detailed information for debugging
    - Function entry/exit
    - Variable values
    - SQL queries
    - External API requests/responses

TRACE: Most detailed level (rarely used in production)
    - Loop iterations
    - Byte-level data
    - Protocol-level details
```

### B. Level Usage Examples

```python
# Python with structlog
import structlog

logger = structlog.get_logger()

# CRITICAL - Application cannot continue
logger.critical("database_connection_lost",
    error="Connection refused",
    host="db.example.com",
    retry_count=10)

# ERROR - Operation failed
logger.error("payment_processing_failed",
    order_id="ORD-123",
    error="Card declined",
    error_code="CARD_DECLINED")

# WARNING - Potential issue
logger.warning("rate_limit_approaching",
    current_rate=95,
    limit=100,
    window="1m")

# INFO - Business events
logger.info("order_placed",
    order_id="ORD-123",
    user_id="USR-456",
    total_amount=99.99)

# DEBUG - Technical details
logger.debug("cache_lookup",
    key="user:123:profile",
    hit=True,
    ttl_remaining=3600)

# TRACE - Very detailed (usually disabled)
logger.trace("http_request_body",
    method="POST",
    path="/api/users",
    body_size=1024)
```

---

## 3. Structured Logging (MANDATORY)

### A. Log Format

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "com.example.OrderService",
  "message": "Order placed successfully",
  "service": "order-service",
  "version": "1.2.3",
  "environment": "production",
  "trace_id": "abc123def456",
  "span_id": "789xyz",
  "user_id": "USR-456",
  "order_id": "ORD-123",
  "total_amount": 99.99,
  "items_count": 3,
  "duration_ms": 245
}
```

### B. Required Fields

```python
# Base fields for every log entry
BASE_FIELDS = {
    "timestamp": "ISO 8601 format with timezone",
    "level": "Log level (DEBUG, INFO, WARN, ERROR, FATAL)",
    "logger": "Logger name / source",
    "message": "Human-readable description",
    "service": "Service/application name",
    "version": "Application version",
    "environment": "Environment (dev, staging, prod)",
}

# Request context fields (when applicable)
REQUEST_FIELDS = {
    "trace_id": "Distributed trace ID",
    "span_id": "Current span ID",
    "request_id": "Unique request identifier",
    "user_id": "Authenticated user ID (if applicable)",
    "session_id": "Session identifier",
}

# Error fields (for ERROR and FATAL)
ERROR_FIELDS = {
    "error_type": "Exception class name",
    "error_message": "Error description",
    "error_code": "Application error code",
    "stack_trace": "Stack trace (in non-production or for fatals)",
}
```

### C. Implementation Examples

```python
# Python with structlog
import structlog
from datetime import datetime

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Bind context for all subsequent logs
logger = logger.bind(
    service="order-service",
    version="1.2.3",
    environment="production"
)

# Add request context
logger = logger.bind(
    trace_id=request.trace_id,
    user_id=request.user_id
)

# Log with additional fields
logger.info("order_placed",
    order_id=order.id,
    total_amount=order.total,
    items_count=len(order.items))
```

```javascript
// Node.js with pino
const pino = require('pino');

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  base: {
    service: 'order-service',
    version: process.env.APP_VERSION,
    environment: process.env.NODE_ENV,
  },
});

// Create child logger with request context
const requestLogger = logger.child({
  traceId: req.traceId,
  userId: req.userId,
});

// Log event
requestLogger.info({
  msg: 'Order placed',
  orderId: order.id,
  totalAmount: order.total,
  itemsCount: order.items.length,
});
```

```go
// Go with zerolog
package main

import (
    "os"
    "github.com/rs/zerolog"
    "github.com/rs/zerolog/log"
)

func init() {
    zerolog.TimeFieldFormat = zerolog.TimeFormatUnix

    log.Logger = zerolog.New(os.Stdout).With().
        Timestamp().
        Str("service", "order-service").
        Str("version", os.Getenv("APP_VERSION")).
        Str("environment", os.Getenv("ENVIRONMENT")).
        Logger()
}

func handleOrder(ctx context.Context, order Order) {
    logger := log.With().
        Str("trace_id", ctx.Value("traceId").(string)).
        Str("user_id", ctx.Value("userId").(string)).
        Logger()

    logger.Info().
        Str("order_id", order.ID).
        Float64("total_amount", order.Total).
        Int("items_count", len(order.Items)).
        Msg("Order placed")
}
```

### D. Go slog (Standard Library, Go 1.21+)

```go
package main

import (
    "context"
    "log/slog"
    "os"
    "time"
)

func initLogger() *slog.Logger {
    // JSON handler for structured output
    handler := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
        Level:     slog.LevelInfo,
        AddSource: true, // Include file/line in logs
    })

    logger := slog.New(handler).With(
        "service", "order-service",
        "version", os.Getenv("APP_VERSION"),
        "environment", os.Getenv("ENVIRONMENT"),
    )

    slog.SetDefault(logger)
    return logger
}

func processOrder(ctx context.Context, orderID string, total float64) error {
    logger := slog.Default().With(
        "order_id", orderID,
        "trace_id", ctx.Value("traceId"),
    )

    start := time.Now()
    logger.Info("order processing started",
        "total_amount", total,
    )

    // ... process order ...

    logger.Info("order processing completed",
        "total_amount", total,
        "duration_ms", time.Since(start).Milliseconds(),
    )
    return nil
}

// Log groups for organizing related fields
func logWithGroups(logger *slog.Logger) {
    logger.Info("request handled",
        slog.Group("request",
            slog.String("method", "POST"),
            slog.String("path", "/api/orders"),
            slog.Int("status", 201),
        ),
        slog.Group("response",
            slog.Int64("duration_ms", 45),
            slog.Int("body_bytes", 1024),
        ),
    )
    // Output: {"request":{"method":"POST","path":"/api/orders","status":201},"response":{"duration_ms":45,"body_bytes":1024}}
}
```

### E. Python structlog Advanced Patterns

```python
import structlog
import logging
import sys

def configure_structlog():
    """Production-ready structlog configuration."""
    structlog.configure(
        processors=[
            # Add contextvars (from middleware bindings)
            structlog.contextvars.merge_contextvars,
            # Add log level
            structlog.stdlib.add_log_level,
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Filter by log level
            structlog.stdlib.filter_by_level,
            # Add timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Add call site info (file, function, line)
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ],
            ),
            # Format stack traces
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # Render as JSON in production, colored console in dev
            structlog.processors.JSONRenderer()
            if os.getenv("ENVIRONMENT") == "production"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Usage with exception chains
logger = structlog.get_logger()

try:
    result = process_payment(order)
except PaymentGatewayError as e:
    logger.error("payment_gateway_failed",
        order_id=order.id,
        gateway="stripe",
        error_code=e.code,
        retriable=e.is_retriable,
        exc_info=True,  # Includes full stack trace
    )
```

### F. Log Level Decision Matrix

```yaml
# Use this matrix to decide the correct log level for any event

decision_matrix:
  FATAL:
    trigger: "Application cannot continue running"
    action_required: "Immediate page, wake someone up"
    examples:
      - "Cannot bind to port, another process is using it"
      - "Out of memory, cannot allocate"
      - "Database migration failed, schema inconsistent"
      - "License expired, cannot start"
    production_volume: "0-1 per incident (application exits)"

  ERROR:
    trigger: "Operation failed, request could not be fulfilled"
    action_required: "Investigate within alert SLA"
    examples:
      - "Payment charge failed after 3 retries"
      - "External API returned unexpected 500"
      - "Database query timeout after 30s"
      - "Failed to send email notification"
      - "Unhandled exception in request handler"
    production_volume: "Low (< 0.1% of requests ideally)"

  WARN:
    trigger: "Something unexpected happened, but was handled"
    action_required: "Review in next business day, trend-watch"
    examples:
      - "Retry succeeded on second attempt"
      - "Cache miss, falling back to database"
      - "Deprecated API endpoint called"
      - "Request took longer than expected (but succeeded)"
      - "Configuration value missing, using default"
      - "Rate limit approaching threshold (80%)"
    production_volume: "Moderate (monitor trends, not individual events)"

  INFO:
    trigger: "Normal business operation completed"
    action_required: "None - this is expected behavior"
    examples:
      - "Application started on port 8080"
      - "User logged in successfully"
      - "Order ORD-123 placed, total $99.99"
      - "Scheduled job completed: processed 1500 records"
      - "Configuration reloaded"
      - "Health check passed"
    production_volume: "1-5 per request or business event"

  DEBUG:
    trigger: "Technical detail useful for diagnosing issues"
    action_required: "Only viewed when investigating a problem"
    examples:
      - "SQL query: SELECT * FROM users WHERE id = ?"
      - "Cache lookup: key=user:123, hit=true, ttl=3600"
      - "HTTP request to downstream: POST /api/charge"
      - "Parsing config file: /etc/app/config.yaml"
    production_volume: "Disabled by default in production"

  TRACE:
    trigger: "Extremely detailed execution flow"
    action_required: "Never in production, local dev only"
    examples:
      - "Entering function processItem(id=123)"
      - "Loop iteration 45/100"
      - "Raw HTTP response body: {bytes}"
    production_volume: "Never enabled in production"
```

---

## 4. Sensitive Data Handling (MANDATORY)

### A. Never Log These

```python
# ❌ NEVER log sensitive data

# Credentials
logger.info("User login", password=password)           # NEVER!
logger.info("API call", api_key=api_key)               # NEVER!
logger.info("Database", connection_string=conn_str)    # NEVER!

# Personal Identifiable Information (PII)
logger.info("User", ssn=user.ssn)                      # NEVER!
logger.info("User", credit_card=card_number)           # NEVER!
logger.info("User", full_address=user.address)         # NEVER!
logger.info("User", date_of_birth=user.dob)            # NEVER!

# Tokens and secrets
logger.info("Auth", jwt_token=token)                   # NEVER!
logger.info("Session", session_token=session)          # NEVER!
logger.info("OAuth", refresh_token=refresh)            # NEVER!

# Health data
logger.info("User", medical_records=records)           # NEVER!
```

### B. Safe Logging Patterns

```python
# ✅ CORRECT - Log identifiers, not values

# Use IDs instead of sensitive data
logger.info("user_login",
    user_id=user.id,                    # ✅ ID only
    email_domain=user.email.split('@')[1],  # ✅ Partial, non-identifying
    login_method="password")            # ✅ Method, not credential

# Mask sensitive values
def mask_email(email):
    name, domain = email.split('@')
    return f"{name[:2]}***@{domain}"

logger.info("email_sent",
    to_email=mask_email(recipient.email),  # ✅ Masked
    template="welcome")

# Mask credit cards
def mask_card(card_number):
    return f"****{card_number[-4:]}"

logger.info("payment_processed",
    card_last_four=card_number[-4:],    # ✅ Last 4 only
    card_type="visa")                   # ✅ Type, not number

# Use hashes for comparison logging
import hashlib

def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()[:8]

logger.debug("token_validated",
    token_hash=hash_value(token))       # ✅ Hash for debugging
```

### C. Automatic Redaction

```python
# Implement automatic redaction for common patterns
import re

REDACTION_PATTERNS = [
    (r'password["\']?\s*[:=]\s*["\']?[^"\'&\s]+', 'password=***REDACTED***'),
    (r'api[_-]?key["\']?\s*[:=]\s*["\']?[^"\'&\s]+', 'api_key=***REDACTED***'),
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '****-****-****-****'),
    (r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b', '***-**-****'),  # SSN
    (r'Bearer\s+[A-Za-z0-9\-._~+/]+=*', 'Bearer ***REDACTED***'),
]

def redact_sensitive(message):
    for pattern, replacement in REDACTION_PATTERNS:
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)
    return message

class RedactingFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact_sensitive(str(record.msg))
        return True
```

### D. Advanced Redaction with structlog Processors

```python
import structlog
import re
from typing import Any

# Sensitive field names to always redact (case-insensitive matching)
SENSITIVE_FIELDS = {
    'password', 'passwd', 'secret', 'token', 'api_key', 'apikey',
    'authorization', 'auth', 'credential', 'private_key',
    'ssn', 'social_security', 'credit_card', 'card_number',
    'cvv', 'cvc', 'pin', 'account_number',
}

# Fields to partially mask (show last N characters)
PARTIAL_MASK_FIELDS = {
    'email': lambda v: _mask_email(v),
    'phone': lambda v: f"***{v[-4:]}" if len(v) >= 4 else "***",
    'ip_address': lambda v: '.'.join(v.split('.')[:2] + ['*', '*']),
}

def _mask_email(email: str) -> str:
    if '@' not in email:
        return '***'
    name, domain = email.rsplit('@', 1)
    return f"{name[0]}***@{domain}"

def redact_sensitive_processor(logger, method_name, event_dict):
    """structlog processor that automatically redacts sensitive fields."""
    for key, value in list(event_dict.items()):
        key_lower = key.lower()

        # Full redaction for sensitive fields
        if key_lower in SENSITIVE_FIELDS:
            event_dict[key] = '***REDACTED***'
            continue

        # Partial masking
        if key_lower in PARTIAL_MASK_FIELDS and isinstance(value, str):
            event_dict[key] = PARTIAL_MASK_FIELDS[key_lower](value)
            continue

        # Scan string values for embedded sensitive data
        if isinstance(value, str):
            event_dict[key] = _redact_embedded_secrets(value)

    return event_dict

def _redact_embedded_secrets(text: str) -> str:
    """Catch secrets embedded in URLs, connection strings, etc."""
    # Redact passwords in URLs: postgres://user:password@host
    text = re.sub(
        r'(://[^:]+:)[^@]+(@)',
        r'\1***REDACTED***\2',
        text
    )
    # Redact Bearer tokens
    text = re.sub(
        r'(Bearer\s+)\S+',
        r'\1***REDACTED***',
        text,
        flags=re.IGNORECASE
    )
    return text

# Register in structlog configuration
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_sensitive_processor,  # Add before renderer
        structlog.processors.JSONRenderer(),
    ],
)

# Now these are automatically safe:
logger = structlog.get_logger()
logger.info("user_created",
    user_id="USR-123",
    email="john.doe@example.com",     # Masked to j***@example.com
    password="secret123",              # Redacted to ***REDACTED***
    api_key="sk_live_abc123",          # Redacted to ***REDACTED***
    db_url="postgres://app:s3cret@db:5432/mydb",  # Password redacted
)
```

```go
// Go: Redacting sensitive fields with a custom slog handler
package logging

import (
    "context"
    "log/slog"
    "strings"
)

var sensitiveKeys = map[string]bool{
    "password": true, "token": true, "secret": true,
    "api_key": true, "authorization": true, "credit_card": true,
}

type RedactingHandler struct {
    inner slog.Handler
}

func NewRedactingHandler(inner slog.Handler) *RedactingHandler {
    return &RedactingHandler{inner: inner}
}

func (h *RedactingHandler) Enabled(ctx context.Context, level slog.Level) bool {
    return h.inner.Enabled(ctx, level)
}

func (h *RedactingHandler) Handle(ctx context.Context, r slog.Record) error {
    redacted := slog.NewRecord(r.Time, r.Level, r.Message, r.PC)
    r.Attrs(func(a slog.Attr) bool {
        if sensitiveKeys[strings.ToLower(a.Key)] {
            redacted.AddAttrs(slog.String(a.Key, "***REDACTED***"))
        } else {
            redacted.AddAttrs(a)
        }
        return true
    })
    return h.inner.Handle(ctx, redacted)
}

func (h *RedactingHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
    return &RedactingHandler{inner: h.inner.WithAttrs(attrs)}
}

func (h *RedactingHandler) WithGroup(name string) slog.Handler {
    return &RedactingHandler{inner: h.inner.WithGroup(name)}
}
```

---

## 5. Correlation and Tracing

### A. Request Tracing

```python
# Middleware to add trace context
import uuid
from contextvars import ContextVar

trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')
span_id_var: ContextVar[str] = ContextVar('span_id', default='')

class TracingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Extract or generate trace ID
        trace_id = scope.get('headers', {}).get(
            b'x-trace-id',
            str(uuid.uuid4())
        )
        span_id = str(uuid.uuid4())[:8]

        # Set context variables
        trace_id_var.set(trace_id)
        span_id_var.set(span_id)

        # Add to response headers
        async def send_with_trace(message):
            if message['type'] == 'http.response.start':
                headers = list(message.get('headers', []))
                headers.append((b'x-trace-id', trace_id.encode()))
                message['headers'] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)

# Logger automatically includes trace context
class TracingLogger:
    def __init__(self, logger):
        self.logger = logger

    def _log(self, level, message, **kwargs):
        kwargs['trace_id'] = trace_id_var.get()
        kwargs['span_id'] = span_id_var.get()
        getattr(self.logger, level)(message, **kwargs)

    def info(self, message, **kwargs):
        self._log('info', message, **kwargs)

    def error(self, message, **kwargs):
        self._log('error', message, **kwargs)
```

### B. Cross-Service Tracing

```python
# Propagate trace context in HTTP calls
import httpx

async def call_external_service(url: str, data: dict):
    headers = {
        'X-Trace-ID': trace_id_var.get(),
        'X-Span-ID': span_id_var.get(),
    }

    logger.info("external_service_call",
        url=url,
        method="POST")

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers)

    logger.info("external_service_response",
        url=url,
        status_code=response.status_code,
        duration_ms=response.elapsed.total_seconds() * 1000)

    return response
```

### C. Correlation ID Propagation Across Services

```python
# Python: Full correlation ID propagation with structlog
import structlog
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

correlation_id_ctx: ContextVar[str] = ContextVar('correlation_id', default='')

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Extract or generate correlation ID and propagate to all logs and downstream calls."""

    async def dispatch(self, request: Request, call_next):
        # Extract from incoming headers or generate new
        correlation_id = (
            request.headers.get('X-Correlation-ID')
            or request.headers.get('X-Request-ID')
            or str(uuid.uuid4())
        )
        correlation_id_ctx.set(correlation_id)

        # Bind to structlog context for all logs in this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            request_method=request.method,
            request_path=str(request.url.path),
        )

        response = await call_next(request)
        response.headers['X-Correlation-ID'] = correlation_id
        return response

# Propagate to downstream HTTP calls
import httpx

async def call_downstream(url: str, payload: dict):
    """Automatically forward correlation ID to downstream services."""
    headers = {
        'X-Correlation-ID': correlation_id_ctx.get(),
        'Content-Type': 'application/json',
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
    return response
```

```go
// Go: Correlation ID propagation with slog
package middleware

import (
    "context"
    "log/slog"
    "net/http"

    "github.com/google/uuid"
)

type contextKey string

const correlationIDKey contextKey = "correlation_id"

func CorrelationIDMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        correlationID := r.Header.Get("X-Correlation-ID")
        if correlationID == "" {
            correlationID = uuid.New().String()
        }

        // Add to context
        ctx := context.WithValue(r.Context(), correlationIDKey, correlationID)

        // Add to response headers
        w.Header().Set("X-Correlation-ID", correlationID)

        // Create a logger with the correlation ID bound
        logger := slog.With("correlation_id", correlationID)
        ctx = context.WithValue(ctx, "logger", logger)

        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

func LoggerFromContext(ctx context.Context) *slog.Logger {
    if logger, ok := ctx.Value("logger").(*slog.Logger); ok {
        return logger
    }
    return slog.Default()
}

// Usage in handler
func handleOrder(w http.ResponseWriter, r *http.Request) {
    logger := LoggerFromContext(r.Context())
    logger.Info("processing order",
        "order_id", orderID,
        "user_id", userID,
    )
}
```

```typescript
// Node.js: Correlation ID with AsyncLocalStorage and pino
import { AsyncLocalStorage } from 'async_hooks';
import pino from 'pino';
import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';

interface RequestContext {
  correlationId: string;
  logger: pino.Logger;
}

const asyncStore = new AsyncLocalStorage<RequestContext>();

const baseLogger = pino({
  level: process.env.LOG_LEVEL || 'info',
  base: { service: 'order-service', version: process.env.APP_VERSION },
});

export function correlationMiddleware(req: Request, res: Response, next: NextFunction) {
  const correlationId = (req.headers['x-correlation-id'] as string) || uuidv4();
  const logger = baseLogger.child({ correlationId });

  res.setHeader('X-Correlation-ID', correlationId);

  asyncStore.run({ correlationId, logger }, () => {
    next();
  });
}

// Get logger anywhere in the call stack
export function getLogger(): pino.Logger {
  const store = asyncStore.getStore();
  return store?.logger ?? baseLogger;
}

// Get correlation ID for downstream calls
export function getCorrelationId(): string {
  return asyncStore.getStore()?.correlationId ?? '';
}

// Usage in any module (no need to pass logger around)
import { getLogger } from './correlation';

function processPayment(orderId: string, amount: number) {
  const logger = getLogger();
  logger.info({ orderId, amount }, 'Processing payment');
  // ...
}
```

---

## 6. Log Aggregation Patterns

### A. ELK Stack Configuration

```yaml
# filebeat.yml
filebeat.inputs:
  - type: container
    paths:
      - '/var/lib/docker/containers/*/*.log'
    processors:
      - add_docker_metadata: ~
      - decode_json_fields:
          fields: ["message"]
          target: ""
          overwrite_keys: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "logs-%{[service]}-%{+yyyy.MM.dd}"

# Logstash pipeline (if needed)
input {
  beats {
    port => 5044
  }
}

filter {
  json {
    source => "message"
  }

  date {
    match => ["timestamp", "ISO8601"]
    target => "@timestamp"
  }

  # Add geo-location for IPs
  geoip {
    source => "client_ip"
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "logs-%{service}-%{+YYYY.MM.dd}"
  }
}
```

### B. Grafana Loki Configuration

```yaml
# promtail config for shipping logs to Loki
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: docker
          __path__: /var/log/containers/*.log

    pipeline_stages:
      # Parse JSON logs
      - json:
          expressions:
            level: level
            service: service
            trace_id: trace_id
            message: message
            timestamp: timestamp

      # Set labels from parsed fields
      - labels:
          level:
          service:

      # Set timestamp from log entry
      - timestamp:
          source: timestamp
          format: "2006-01-02T15:04:05.000Z"

      # Drop debug logs in production to save storage
      - match:
          selector: '{level="DEBUG"}'
          stages:
            - drop:
                expression: ".*"
          drop_counter_reason: debug_logs_dropped
```

```bash
# Loki LogQL query examples

# Find errors for a specific service
{service="order-service"} |= "error" | json | level="ERROR"

# Find logs with a specific correlation ID across all services
{job="docker"} | json | trace_id="abc123def456"

# Count errors per service over time
sum by (service) (count_over_time({job="docker"} | json | level="ERROR" [5m]))

# Find slow requests (duration > 1000ms)
{service="api-gateway"} | json | duration_ms > 1000

# Parse and filter by specific field values
{service="payment-service"} | json | error_code="CARD_DECLINED" | line_format "{{.order_id}} - {{.error_message}}"
```

### C. CloudWatch Logs

```python
# AWS CloudWatch structured logging
import watchtower
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = watchtower.CloudWatchLogHandler(
    log_group='/application/order-service',
    stream_name='{strftime:%Y-%m-%d}',
    create_log_group=True,
)

# Use JSON formatter
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        return json.dumps(log_entry)

handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

---

## 7. Performance Considerations

### A. Async Logging

```python
# Use async logging for high-throughput applications
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncLogger:
    def __init__(self, logger, max_workers=4):
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.queue = asyncio.Queue()

    async def log(self, level, message, **kwargs):
        # Non-blocking log submission
        await self.queue.put((level, message, kwargs))

    async def process_logs(self):
        while True:
            level, message, kwargs = await self.queue.get()
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: getattr(self.logger, level)(message, **kwargs)
            )
```

### B. Log Sampling

```python
# Sample high-volume logs to reduce overhead
import random

class SampledLogger:
    def __init__(self, logger, sample_rate=0.1):
        self.logger = logger
        self.sample_rate = sample_rate

    def debug(self, message, **kwargs):
        # Sample debug logs
        if random.random() < self.sample_rate:
            kwargs['sampled'] = True
            kwargs['sample_rate'] = self.sample_rate
            self.logger.debug(message, **kwargs)

    def info(self, message, **kwargs):
        # Always log info and above
        self.logger.info(message, **kwargs)

    def error(self, message, **kwargs):
        # Always log errors
        self.logger.error(message, **kwargs)
```

### C. Buffered Logging

```python
# Buffer logs and flush periodically
class BufferedLogger:
    def __init__(self, logger, buffer_size=100, flush_interval=5):
        self.logger = logger
        self.buffer = []
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._start_flush_timer()

    def log(self, level, message, **kwargs):
        self.buffer.append((level, message, kwargs))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        for level, message, kwargs in self.buffer:
            getattr(self.logger, level)(message, **kwargs)
        self.buffer.clear()

    def _start_flush_timer(self):
        import threading
        def flush_periodically():
            while True:
                time.sleep(self.flush_interval)
                self.flush()
        threading.Thread(target=flush_periodically, daemon=True).start()
```

---

## 8. Error Logging

### A. Exception Logging

```python
import traceback
import sys

def log_exception(logger, exception, context=None):
    """Log exception with full context."""
    exc_type, exc_value, exc_tb = sys.exc_info()

    logger.error("exception_occurred",
        error_type=type(exception).__name__,
        error_message=str(exception),
        stack_trace=traceback.format_exc(),
        context=context or {},
        # Include cause chain
        cause=str(exception.__cause__) if exception.__cause__ else None,
    )

# Usage
try:
    process_order(order)
except PaymentError as e:
    log_exception(logger, e, context={
        "order_id": order.id,
        "user_id": order.user_id,
        "operation": "payment_processing"
    })
    raise

# Context manager for automatic exception logging
from contextlib import contextmanager

@contextmanager
def log_exceptions(logger, operation, **context):
    try:
        yield
    except Exception as e:
        log_exception(logger, e, context={
            "operation": operation,
            **context
        })
        raise

# Usage
with log_exceptions(logger, "order_processing", order_id=order.id):
    process_order(order)
```

### B. Error Aggregation

```python
# Track error rates for alerting
from collections import defaultdict
from datetime import datetime, timedelta

class ErrorTracker:
    def __init__(self, window_seconds=60):
        self.errors = defaultdict(list)
        self.window = timedelta(seconds=window_seconds)

    def record_error(self, error_type):
        now = datetime.now()
        self.errors[error_type].append(now)
        self._cleanup(error_type, now)

    def _cleanup(self, error_type, now):
        cutoff = now - self.window
        self.errors[error_type] = [
            t for t in self.errors[error_type] if t > cutoff
        ]

    def get_error_rate(self, error_type):
        now = datetime.now()
        self._cleanup(error_type, now)
        return len(self.errors[error_type]) / self.window.total_seconds()

    def should_alert(self, error_type, threshold=0.1):
        return self.get_error_rate(error_type) > threshold
```

---

## 9. Log Retention and Rotation

### A. Rotation Configuration

```python
# Python logging with rotation
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Size-based rotation
size_handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5
)

# Time-based rotation
time_handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    interval=1,
    backupCount=30  # Keep 30 days
)
```

### B. Retention Policy

```yaml
# Log retention policy by level and environment
retention:
  production:
    error: 90d    # Keep error logs for 90 days
    warn: 30d     # Keep warnings for 30 days
    info: 14d     # Keep info logs for 14 days
    debug: 0d     # Don't store debug in production

  staging:
    error: 30d
    warn: 14d
    info: 7d
    debug: 3d

  development:
    error: 7d
    warn: 3d
    info: 1d
    debug: 1d
```

---

## 10. Alerting on Logs

### A. Alert Conditions

```yaml
# Alert rules based on log patterns
alerts:
  - name: high_error_rate
    condition: |
      count(level="ERROR") / count(*) > 0.05
      over 5m window
    severity: critical
    notify: [pagerduty, slack]

  - name: payment_failures
    condition: |
      count(message="payment_failed") > 10
      over 1m window
    severity: high
    notify: [slack]

  - name: security_events
    condition: |
      any(message contains "authentication_failed"
          AND count > 5
          AND same user_id)
      over 1m window
    severity: critical
    notify: [security-team, pagerduty]

  - name: slow_requests
    condition: |
      percentile(duration_ms, 95) > 5000
      over 5m window
    severity: warning
    notify: [slack]
```

---

## 11. Deployment Checklist

### Configuration
- [ ] Structured logging implemented (JSON format)
- [ ] Log levels configured appropriately per environment
- [ ] Correlation IDs (trace_id) implemented
- [ ] Timestamp format is ISO 8601 with timezone

### Security
- [ ] Sensitive data redaction implemented
- [ ] No passwords, tokens, or PII in logs
- [ ] Log access controls configured
- [ ] Audit logging enabled for compliance

### Performance
- [ ] Async logging for high-throughput services
- [ ] Log sampling configured for debug/trace levels
- [ ] Log rotation configured
- [ ] Buffer sizes optimized

### Operations
- [ ] Log aggregation configured (ELK, CloudWatch, etc.)
- [ ] Alerting rules defined
- [ ] Retention policies set
- [ ] Dashboard created for key metrics

---

## 12. Quick Reference

```python
# Log level decision tree
"""
Is the application unable to continue?
  YES → FATAL/CRITICAL

Did an operation fail?
  YES → ERROR

Is something unusual but handled?
  YES → WARN

Is it a normal business event?
  YES → INFO

Is it technical debugging info?
  YES → DEBUG

Is it extremely detailed tracing?
  YES → TRACE
"""

# Required fields checklist
"""
Every log: timestamp, level, service, message
Requests: + trace_id, user_id
Errors: + error_type, error_message, stack_trace
Performance: + duration_ms
"""

# Common structured fields
"""
timestamp, level, service, version, environment
trace_id, span_id, request_id, user_id
error_type, error_message, error_code, stack_trace
duration_ms, status_code, method, path
"""
```

---

## 13. Why This Configuration Works

- **Structured JSON logging enables machine analysis**: Consistent JSON-formatted logs with standardized fields allow log aggregation tools (ELK, Loki, CloudWatch) to index, search, and alert on specific fields without fragile regex parsing, turning logs from text files into queryable data.
- **Correlation IDs connect distributed requests**: Including trace and request IDs in every log entry makes it possible to reconstruct the full journey of a request across multiple services, transforming cross-service debugging from guesswork into a deterministic trace-following exercise.
- **Automatic redaction prevents data breaches**: Pattern-based redaction of passwords, tokens, credit card numbers, and PII ensures that sensitive data never reaches log storage, protecting against both compliance violations and the security risk of credentials appearing in log aggregation systems.
- **Consistent log levels drive effective alerting**: A well-defined level hierarchy (FATAL through TRACE) with clear usage guidelines ensures that alerts fire on genuinely actionable events. When INFO means business events and ERROR means operation failures, monitoring rules become simple and reliable.
- **Performance-conscious patterns prevent logging from becoming a bottleneck**: Async logging, sampling for high-volume debug entries, and buffered writes ensure that observability does not degrade the very application performance it is meant to monitor.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** SRE Team


**End of Logging Guidelines**

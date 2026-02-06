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

### B. CloudWatch Logs

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

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** SRE Team


**End of Logging Guidelines**

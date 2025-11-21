# Observability Guidelines

This document provides mandatory standards for implementing observability through metrics, logging, tracing, and alerting.

---

**Agent Profile**: The Observability Architect
**Role**: Senior SRE & Observability Specialist
**Objective**: Generate comprehensive observability implementations enabling rapid debugging and proactive monitoring.
**Tools**: OpenTelemetry, Prometheus, Grafana, Jaeger, Loki, Datadog, New Relic, ELK Stack.

---

## 1. Core Philosophies: OBSERVE-FIRST

The agent must adhere to the **OBSERVE-FIRST** principles:

- **O**penTelemetry: Use standard instrumentation (OTLP)
- **B**usiness Metrics: Track what matters to users
- **S**LA-Driven: Define and measure service level objectives
- **E**nd-to-End: Trace requests across all services
- **R**eal-Time: Enable immediate issue detection
- **V**isual: Create actionable dashboards
- **E**fficient: Balance detail with overhead

---

## 2. The Three Pillars

### A. Metrics (What)

```yaml
# Quantitative data over time
Purpose: Answer "what is happening?"
Examples:
  - Request rate: 1000 req/s
  - Error rate: 0.1%
  - Latency p99: 200ms
  - CPU usage: 45%
Use for:
  - Alerting
  - Capacity planning
  - SLO measurement
  - Trend analysis
```

### B. Logs (Why)

```yaml
# Discrete events with context
Purpose: Answer "why did it happen?"
Examples:
  - "User authentication failed: invalid password"
  - "Database connection timeout after 30s"
  - "Order processed successfully"
Use for:
  - Debugging
  - Audit trail
  - Root cause analysis
  - Compliance
```

### C. Traces (Where)

```yaml
# Request flow across services
Purpose: Answer "where did it happen?"
Examples:
  - Request path through 5 microservices
  - Time spent in each service
  - Which service caused the error
Use for:
  - Performance debugging
  - Dependency mapping
  - Bottleneck identification
```

---

## 3. Metrics Implementation (MANDATORY)

### A. Metric Types

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Counter: Monotonically increasing value
# Use for: requests, errors, items processed
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
http_requests_total.labels(method='GET', endpoint='/api/users', status='200').inc()

# Histogram: Distribution of values
# Use for: latency, request size, response size
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
with http_request_duration_seconds.labels(method='GET', endpoint='/api/users').time():
    process_request()

# Gauge: Value that can go up or down
# Use for: current connections, queue size, temperature
active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    ['service']
)
active_connections.labels(service='api').set(42)
active_connections.labels(service='api').inc()  # +1
active_connections.labels(service='api').dec()  # -1

# Summary: Similar to histogram but calculates quantiles
request_latency = Summary(
    'request_latency_seconds',
    'Request latency',
    ['endpoint']
)
```

### B. RED Method (Request-Oriented)

```python
# Rate, Errors, Duration - For every service

# Rate: Requests per second
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['service', 'method', 'endpoint']
)

# Errors: Failed requests
http_errors_total = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['service', 'method', 'endpoint', 'error_type']
)

# Duration: Latency distribution
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['service', 'method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
```

### C. USE Method (Resource-Oriented)

```python
# Utilization, Saturation, Errors - For every resource

# CPU
cpu_utilization_percent = Gauge('cpu_utilization_percent', 'CPU utilization')
cpu_saturation_load = Gauge('cpu_saturation_load', 'CPU run queue length')

# Memory
memory_utilization_bytes = Gauge('memory_utilization_bytes', 'Memory used')
memory_saturation_swap_bytes = Gauge('memory_saturation_swap_bytes', 'Swap used')

# Disk
disk_utilization_percent = Gauge('disk_utilization_percent', 'Disk utilization', ['device'])
disk_io_saturation = Gauge('disk_io_saturation', 'Disk IO wait', ['device'])

# Network
network_utilization_bytes = Counter('network_utilization_bytes', 'Network bytes', ['direction'])
network_errors_total = Counter('network_errors_total', 'Network errors', ['type'])
```

### D. Business Metrics

```python
# Track what matters to the business

# Revenue
orders_total = Counter('orders_total', 'Total orders', ['status'])
order_value_dollars = Histogram(
    'order_value_dollars',
    'Order value distribution',
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

# User engagement
active_users = Gauge('active_users', 'Currently active users')
signups_total = Counter('signups_total', 'Total user signups', ['source'])
user_actions_total = Counter('user_actions_total', 'User actions', ['action'])

# Feature usage
feature_usage_total = Counter(
    'feature_usage_total',
    'Feature usage count',
    ['feature', 'variant']
)
```

---

## 4. Distributed Tracing (MANDATORY)

### A. OpenTelemetry Setup

```python
# Python OpenTelemetry setup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Configure tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure exporter
otlp_exporter = OTLPSpanExporter(endpoint="http://collector:4317")
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

# Auto-instrument libraries
RequestsInstrumentor().instrument()
FlaskInstrumentor().instrument_app(app)
```

### B. Manual Instrumentation

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

def process_order(order_id: str) -> Order:
    # Create span for the operation
    with tracer.start_as_current_span("process_order") as span:
        # Add attributes
        span.set_attribute("order.id", order_id)
        span.set_attribute("service.name", "order-service")

        try:
            # Nested span for sub-operation
            with tracer.start_as_current_span("validate_order") as child_span:
                order = validate_order(order_id)
                child_span.set_attribute("order.items_count", len(order.items))

            with tracer.start_as_current_span("charge_payment") as child_span:
                child_span.set_attribute("payment.method", order.payment_method)
                charge_result = charge_payment(order)
                child_span.set_attribute("payment.success", charge_result.success)

            # Add event (point-in-time occurrence)
            span.add_event("order_processed", {
                "order.total": order.total,
                "items_count": len(order.items)
            })

            span.set_status(Status(StatusCode.OK))
            return order

        except PaymentError as e:
            # Record exception
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
```

### C. Context Propagation

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
import requests

def call_downstream_service(order_id: str):
    headers = {}

    # Inject trace context into headers
    inject(headers)

    response = requests.post(
        "http://inventory-service/reserve",
        json={"order_id": order_id},
        headers=headers
    )
    return response.json()

# In downstream service
def handle_request(request):
    # Extract trace context from headers
    context = extract(request.headers)

    with tracer.start_as_current_span("handle_reserve", context=context):
        # Continue the trace
        process_reservation()
```

### D. Trace Sampling

```python
from opentelemetry.sdk.trace.sampling import (
    ParentBased,
    TraceIdRatioBased,
    ALWAYS_ON,
    ALWAYS_OFF
)

# Sample 10% of traces
sampler = TraceIdRatioBased(0.1)

# Parent-based sampling (respect upstream decision)
sampler = ParentBased(root=TraceIdRatioBased(0.1))

# Always sample errors
class ErrorAwareSampler:
    def should_sample(self, context, trace_id, name, attributes=None):
        if attributes and attributes.get("error"):
            return ALWAYS_ON.should_sample(...)
        return TraceIdRatioBased(0.1).should_sample(...)
```

---

## 5. Service Level Objectives (MANDATORY)

### A. SLI/SLO/SLA Definitions

```yaml
# SLI (Service Level Indicator): The metric
# SLO (Service Level Objective): The target
# SLA (Service Level Agreement): The commitment with consequences

service: api-gateway
slis:
  - name: availability
    description: Percentage of successful requests
    calculation: |
      sum(rate(http_requests_total{status!~"5.."}[5m])) /
      sum(rate(http_requests_total[5m]))

  - name: latency_p99
    description: 99th percentile request latency
    calculation: |
      histogram_quantile(0.99,
        sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
      )

  - name: error_rate
    description: Percentage of failed requests
    calculation: |
      sum(rate(http_requests_total{status=~"5.."}[5m])) /
      sum(rate(http_requests_total[5m]))

slos:
  - sli: availability
    target: 99.9%
    window: 30d

  - sli: latency_p99
    target: 200ms
    window: 30d

  - sli: error_rate
    target: 0.1%
    window: 30d

error_budget:
  monthly_budget_minutes: 43.2  # 30 days * 24h * 60min * 0.001
```

### B. Error Budget Calculation

```python
from prometheus_client import Gauge
from datetime import datetime, timedelta

error_budget_remaining = Gauge(
    'error_budget_remaining_percent',
    'Remaining error budget percentage',
    ['service', 'slo']
)

def calculate_error_budget(service: str, slo_target: float, window_days: int):
    """
    Calculate remaining error budget.

    Example: 99.9% availability SLO over 30 days
    - Total allowed downtime: 30 * 24 * 60 * 0.001 = 43.2 minutes
    - If 20 minutes used, remaining = (43.2 - 20) / 43.2 = 53.7%
    """
    # Query actual availability
    actual_availability = query_prometheus(f'''
        sum(rate(http_requests_total{{service="{service}",status!~"5.."}}[{window_days}d])) /
        sum(rate(http_requests_total{{service="{service}"}}[{window_days}d]))
    ''')

    # Calculate budget
    total_budget = 1 - slo_target  # e.g., 0.001 for 99.9%
    consumed = 1 - actual_availability
    remaining = max(0, (total_budget - consumed) / total_budget)

    error_budget_remaining.labels(service=service, slo='availability').set(remaining * 100)
    return remaining
```

---

## 6. Alerting (MANDATORY)

### A. Alert Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: slo_alerts
    rules:
      # High error rate (immediate)
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) /
          sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate: {{ $value | humanizePercentage }}"
          description: "Error rate is above 1% for service {{ $labels.service }}"
          runbook_url: https://wiki/runbooks/high-error-rate

      # Latency degradation
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency: p99 = {{ $value | humanizeDuration }}"

      # Error budget burn rate (predictive)
      - alert: ErrorBudgetBurn
        expr: |
          (
            1 - (
              sum(rate(http_requests_total{status!~"5.."}[1h])) /
              sum(rate(http_requests_total[1h]))
            )
          ) > (14.4 * 0.001)  # Burning 14.4x faster than budget allows
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error budget burning too fast"
          description: "At current rate, error budget will be exhausted in < 2 hours"

  - name: resource_alerts
    rules:
      - alert: HighCPUUsage
        expr: |
          100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage: {{ $value }}%"

      - alert: HighMemoryUsage
        expr: |
          (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) /
          node_memory_MemTotal_bytes * 100 > 85
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage: {{ $value }}%"

      - alert: DiskSpaceLow
        expr: |
          (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disk space low: {{ $value }}% available"
```

### B. Alert Severity Levels

```yaml
severity_levels:
  critical:
    description: "Immediate action required, user impact"
    response_time: "5 minutes"
    notification:
      - pagerduty
      - slack-critical
    examples:
      - Service down
      - Data loss risk
      - Security breach

  warning:
    description: "Degraded service, needs attention"
    response_time: "30 minutes"
    notification:
      - slack-warnings
    examples:
      - High latency
      - Resource pressure
      - Error budget burn

  info:
    description: "Notable event, no immediate action"
    response_time: "Next business day"
    notification:
      - slack-info
    examples:
      - Deployment completed
      - Config change detected
```

---

## 7. Dashboards (MANDATORY)

### A. Service Dashboard Template

```yaml
# Grafana dashboard structure
dashboard:
  title: "Service: ${service_name}"

  variables:
    - name: service
      type: query
      query: label_values(http_requests_total, service)

  rows:
    - title: "Overview"
      panels:
        - title: "Request Rate"
          type: graph
          query: sum(rate(http_requests_total{service="$service"}[5m]))

        - title: "Error Rate"
          type: gauge
          query: |
            sum(rate(http_requests_total{service="$service",status=~"5.."}[5m])) /
            sum(rate(http_requests_total{service="$service"}[5m])) * 100
          thresholds:
            - value: 0
              color: green
            - value: 1
              color: yellow
            - value: 5
              color: red

        - title: "Latency (p50, p90, p99)"
          type: graph
          queries:
            - expr: histogram_quantile(0.5, sum(rate(http_request_duration_seconds_bucket{service="$service"}[5m])) by (le))
              legend: p50
            - expr: histogram_quantile(0.9, sum(rate(http_request_duration_seconds_bucket{service="$service"}[5m])) by (le))
              legend: p90
            - expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{service="$service"}[5m])) by (le))
              legend: p99

    - title: "Errors"
      panels:
        - title: "Errors by Type"
          type: graph
          query: sum(rate(http_requests_total{service="$service",status=~"5.."}[5m])) by (status)

        - title: "Recent Errors"
          type: logs
          query: '{service="$service"} |= "error"'

    - title: "Resources"
      panels:
        - title: "CPU Usage"
          type: graph
          query: rate(process_cpu_seconds_total{service="$service"}[5m]) * 100

        - title: "Memory Usage"
          type: graph
          query: process_resident_memory_bytes{service="$service"}

        - title: "Active Connections"
          type: stat
          query: http_connections_active{service="$service"}
```

### B. SLO Dashboard

```yaml
dashboard:
  title: "SLO Dashboard"

  panels:
    - title: "Availability SLO"
      type: gauge
      query: |
        sum(rate(http_requests_total{status!~"5.."}[30d])) /
        sum(rate(http_requests_total[30d])) * 100
      thresholds:
        - value: 99.9
          color: green
        - value: 99.5
          color: yellow
        - value: 0
          color: red
      target: 99.9

    - title: "Error Budget Remaining"
      type: gauge
      query: error_budget_remaining_percent
      thresholds:
        - value: 50
          color: green
        - value: 25
          color: yellow
        - value: 0
          color: red

    - title: "Error Budget Burn Rate"
      type: graph
      query: |
        (
          1 - (
            sum(rate(http_requests_total{status!~"5.."}[1h])) /
            sum(rate(http_requests_total[1h]))
          )
        ) / 0.001  # Normalize to budget consumption rate
```

---

## 8. OpenTelemetry Collector

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1000

  memory_limiter:
    check_interval: 1s
    limit_mib: 1000
    spike_limit_mib: 200

  attributes:
    actions:
      - key: environment
        value: production
        action: insert

exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"

  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes]
      exporters: [jaeger]

    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]

    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [loki]
```

---

## 9. Deployment Checklist

### Metrics
- [ ] RED metrics for all services
- [ ] USE metrics for all resources
- [ ] Business metrics defined
- [ ] Histogram buckets appropriate

### Tracing
- [ ] All services instrumented
- [ ] Context propagation working
- [ ] Sampling configured
- [ ] Critical paths traced

### Logging
- [ ] Structured logging implemented
- [ ] Correlation IDs present
- [ ] Log levels appropriate
- [ ] Sensitive data redacted

### Alerting
- [ ] SLOs defined
- [ ] Alert rules configured
- [ ] Notification channels set up
- [ ] Runbooks linked

### Dashboards
- [ ] Service dashboards created
- [ ] SLO dashboard available
- [ ] Resource dashboards ready

---

## 10. Quick Reference

```yaml
# Key metrics to implement
Rate: http_requests_total
Errors: http_errors_total
Duration: http_request_duration_seconds (histogram)

# Essential labels
service, method, endpoint, status

# SLO targets (typical)
Availability: 99.9% (43.2 min/month downtime)
Latency p99: 200ms
Error rate: 0.1%

# Alert thresholds
Error rate > 1%: Critical
Latency p99 > 500ms: Warning
Error budget burn > 14.4x: Critical
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** SRE Team

# Observability Guidelines
Mandatory standards for implementing observability through metrics, logging, tracing, and alerting. OpenTelemetry, Prometheus, Grafana, Jaeger, Loki, Datadog, New Relic, ELK Stack.

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

## 9. OpenTelemetry Metrics Implementation

### A. Go OpenTelemetry Metrics

```go
package main

import (
    "context"
    "net/http"
    "time"

    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/attribute"
    "go.opentelemetry.io/otel/metric"
    sdkmetric "go.opentelemetry.io/otel/sdk/metric"
    "go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
)

var (
    meter          = otel.Meter("api-gateway")
    requestCounter metric.Int64Counter
    requestLatency metric.Float64Histogram
    activeRequests metric.Int64UpDownCounter
)

func initMetrics(ctx context.Context) (*sdkmetric.MeterProvider, error) {
    exporter, err := otlpmetricgrpc.New(ctx,
        otlpmetricgrpc.WithEndpoint("collector:4317"),
        otlpmetricgrpc.WithInsecure(),
    )
    if err != nil {
        return nil, err
    }

    provider := sdkmetric.NewMeterProvider(
        sdkmetric.WithReader(sdkmetric.NewPeriodicReader(exporter,
            sdkmetric.WithInterval(15*time.Second),
        )),
    )
    otel.SetMeterProvider(provider)

    requestCounter, _ = meter.Int64Counter("http_requests_total",
        metric.WithDescription("Total HTTP requests"),
        metric.WithUnit("{request}"),
    )
    requestLatency, _ = meter.Float64Histogram("http_request_duration_seconds",
        metric.WithDescription("HTTP request latency"),
        metric.WithUnit("s"),
        metric.WithExplicitBucketBoundaries(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    activeRequests, _ = meter.Int64UpDownCounter("http_active_requests",
        metric.WithDescription("Currently active HTTP requests"),
    )
    return provider, nil
}

func metricsMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        attrs := []attribute.KeyValue{
            attribute.String("method", r.Method),
            attribute.String("path", r.URL.Path),
        }

        activeRequests.Add(r.Context(), 1, metric.WithAttributes(attrs...))
        defer activeRequests.Add(r.Context(), -1, metric.WithAttributes(attrs...))

        rw := &responseWriter{ResponseWriter: w, statusCode: 200}
        next.ServeHTTP(rw, r)

        attrs = append(attrs, attribute.Int("status", rw.statusCode))
        requestCounter.Add(r.Context(), 1, metric.WithAttributes(attrs...))
        requestLatency.Record(r.Context(), time.Since(start).Seconds(),
            metric.WithAttributes(attrs...))
    })
}
```

### B. Node.js OpenTelemetry Metrics

```typescript
import { MeterProvider, PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-grpc';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

const resource = new Resource({
  [SemanticResourceAttributes.SERVICE_NAME]: 'order-service',
  [SemanticResourceAttributes.SERVICE_VERSION]: '1.4.0',
  [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: 'production',
});

const metricExporter = new OTLPMetricExporter({
  url: 'http://collector:4317',
});

const meterProvider = new MeterProvider({
  resource,
  readers: [
    new PeriodicExportingMetricReader({
      exporter: metricExporter,
      exportIntervalMillis: 15000,
    }),
  ],
});

const meter = meterProvider.getMeter('order-service');

// Counter for total orders
const ordersCounter = meter.createCounter('orders_total', {
  description: 'Total number of orders processed',
  unit: '{order}',
});

// Histogram for order processing duration
const orderDuration = meter.createHistogram('order_processing_duration_seconds', {
  description: 'Time to process an order',
  unit: 's',
  advice: {
    explicitBucketBoundaries: [0.1, 0.5, 1, 2, 5, 10, 30],
  },
});

// Observable gauge for queue depth
const queueDepth = meter.createObservableGauge('order_queue_depth', {
  description: 'Number of orders waiting in queue',
});
queueDepth.addCallback(async (result) => {
  const depth = await getQueueDepth();
  result.observe(depth, { queue: 'orders' });
});

// Usage in request handler
async function processOrder(order: Order) {
  const start = Date.now();
  try {
    await executeOrder(order);
    ordersCounter.add(1, { status: 'success', type: order.type });
  } catch (err) {
    ordersCounter.add(1, { status: 'failure', type: order.type });
    throw err;
  } finally {
    orderDuration.record((Date.now() - start) / 1000, { type: order.type });
  }
}
```

### C. Java Spring Boot with Micrometer and Prometheus

```java
import io.micrometer.core.instrument.*;
import io.micrometer.core.annotation.Timed;
import org.springframework.stereotype.Service;

@Service
public class PaymentService {

    private final Counter paymentSuccessCounter;
    private final Counter paymentFailureCounter;
    private final Timer paymentTimer;
    private final DistributionSummary paymentAmounts;

    public PaymentService(MeterRegistry registry) {
        this.paymentSuccessCounter = Counter.builder("payments_total")
            .tag("status", "success")
            .description("Total successful payments")
            .register(registry);

        this.paymentFailureCounter = Counter.builder("payments_total")
            .tag("status", "failure")
            .description("Total failed payments")
            .register(registry);

        this.paymentTimer = Timer.builder("payment_processing_seconds")
            .description("Payment processing duration")
            .publishPercentiles(0.5, 0.9, 0.95, 0.99)
            .publishPercentileHistogram()
            .register(registry);

        this.paymentAmounts = DistributionSummary.builder("payment_amount_dollars")
            .description("Payment amount distribution")
            .baseUnit("dollars")
            .publishPercentiles(0.5, 0.9, 0.99)
            .register(registry);

        // Gauge for active payment sessions
        Gauge.builder("payment_sessions_active", this, PaymentService::getActiveSessions)
            .description("Currently active payment sessions")
            .register(registry);
    }

    @Timed(value = "payment_processing_seconds", percentiles = {0.5, 0.95, 0.99})
    public PaymentResult processPayment(PaymentRequest request) {
        paymentAmounts.record(request.getAmount());
        try {
            PaymentResult result = gateway.charge(request);
            paymentSuccessCounter.increment();
            return result;
        } catch (PaymentException e) {
            paymentFailureCounter.increment();
            throw e;
        }
    }
}
```

---

## 10. Distributed Tracing Patterns

### A. Cross-Service Trace with Baggage

```python
from opentelemetry import trace, baggage
from opentelemetry.context import attach, detach
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer("checkout-service")

def initiate_checkout(user_id: str, cart_id: str):
    """Demonstrate baggage propagation across services."""
    with tracer.start_as_current_span("initiate_checkout") as span:
        span.set_attribute("user.id", user_id)
        span.set_attribute("cart.id", cart_id)

        # Attach baggage that propagates to all downstream services
        ctx = baggage.set_baggage("user.tier", "premium")
        token = attach(ctx)

        try:
            # Each downstream call inherits baggage automatically
            inventory = call_inventory_service(cart_id)
            pricing = call_pricing_service(cart_id, user_id)
            payment = call_payment_service(pricing.total, user_id)

            span.set_attribute("checkout.total", pricing.total)
            span.add_event("checkout_completed", {
                "items_count": len(inventory.items),
                "total": pricing.total,
            })
        finally:
            detach(token)
```

### B. Async Trace Propagation (Message Queues)

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract
import json

tracer = trace.get_tracer("order-producer")

# Producer: inject trace context into message headers
def publish_order_event(order_id: str, event_type: str):
    with tracer.start_as_current_span(
        f"publish_{event_type}",
        kind=trace.SpanKind.PRODUCER
    ) as span:
        span.set_attribute("messaging.system", "rabbitmq")
        span.set_attribute("messaging.destination", "orders")
        span.set_attribute("order.id", order_id)

        # Inject context into message headers
        headers = {}
        inject(headers)

        message = {
            "order_id": order_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
        channel.basic_publish(
            exchange="orders",
            routing_key=event_type,
            body=json.dumps(message),
            properties=pika.BasicProperties(headers=headers),
        )

# Consumer: extract trace context from message headers
def handle_order_event(ch, method, properties, body):
    # Extract parent context from message headers
    parent_ctx = extract(properties.headers or {})

    with tracer.start_as_current_span(
        "process_order_event",
        context=parent_ctx,
        kind=trace.SpanKind.CONSUMER,
    ) as span:
        message = json.loads(body)
        span.set_attribute("messaging.system", "rabbitmq")
        span.set_attribute("order.id", message["order_id"])
        span.set_attribute("messaging.operation", "process")

        process_event(message)
```

### C. Database Span Instrumentation

```python
from opentelemetry import trace
from opentelemetry.trace import SpanKind
import functools

tracer = trace.get_tracer("db-instrumentation")

def trace_query(func):
    """Decorator to trace database queries with span details."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        query = kwargs.get("query") or (args[0] if args else "unknown")
        operation = query.strip().split()[0].upper() if isinstance(query, str) else "QUERY"

        with tracer.start_as_current_span(
            f"db.{operation}",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("db.system", "postgresql")
            span.set_attribute("db.operation", operation)
            span.set_attribute("db.statement", _sanitize_query(query))

            try:
                result = await func(*args, **kwargs)
                span.set_attribute("db.rows_affected", getattr(result, "rowcount", 0))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
    return wrapper

def _sanitize_query(query: str) -> str:
    """Remove literal values from queries to avoid cardinality explosion."""
    import re
    # Replace string literals
    sanitized = re.sub(r"'[^']*'", "'?'", query)
    # Replace numeric literals
    sanitized = re.sub(r"\b\d+\b", "?", sanitized)
    return sanitized

# Usage
class OrderRepository:
    @trace_query
    async def get_order(self, query: str, order_id: str):
        return await self.db.fetch_one(query, order_id)
```

---

## 11. Dashboard Design Principles

### A. The Four Golden Signals Dashboard

```yaml
# Structure every service dashboard around the four golden signals
dashboard:
  title: "Service: ${service_name} - Golden Signals"

  # Row 1: The four golden signals at a glance
  rows:
    - title: "Golden Signals"
      height: 8
      panels:
        - title: "Traffic (req/s)"
          type: timeseries
          width: 6
          query: sum(rate(http_requests_total{service="$service"}[5m]))
          description: "Current request rate"

        - title: "Errors (%)"
          type: timeseries
          width: 6
          query: |
            100 * sum(rate(http_requests_total{service="$service",status=~"5.."}[5m]))
            / sum(rate(http_requests_total{service="$service"}[5m]))
          thresholds:
            - value: 1
              color: yellow
            - value: 5
              color: red

        - title: "Latency"
          type: timeseries
          width: 6
          queries:
            - legend: p50
              expr: histogram_quantile(0.5, sum(rate(http_request_duration_seconds_bucket{service="$service"}[5m])) by (le))
            - legend: p95
              expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service="$service"}[5m])) by (le))
            - legend: p99
              expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{service="$service"}[5m])) by (le))

        - title: "Saturation"
          type: timeseries
          width: 6
          queries:
            - legend: "CPU %"
              expr: rate(process_cpu_seconds_total{service="$service"}[5m]) * 100
            - legend: "Memory %"
              expr: process_resident_memory_bytes{service="$service"} / machine_memory_bytes * 100
```

### B. Dashboard Anti-Patterns to Avoid

```yaml
# Anti-patterns with corrections

# ❌ WRONG: Too many panels on one dashboard (>20)
# ✅ CORRECT: Limit to 8-12 panels, use drill-down links

# ❌ WRONG: Mixing unrelated services on one dashboard
# ✅ CORRECT: One dashboard per service, one overview dashboard for the fleet

# ❌ WRONG: Using averages for latency
# ✅ CORRECT: Always use percentiles (p50, p95, p99) for latency
bad_query: avg(http_request_duration_seconds)    # Hides tail latency
good_query: histogram_quantile(0.99, ...)         # Shows worst-case experience

# ❌ WRONG: No time range context on panels
# ✅ CORRECT: Show rate() over window, not raw counters
bad_query: http_requests_total                    # Ever-increasing number
good_query: rate(http_requests_total[5m])         # Meaningful rate

# ❌ WRONG: Missing units on axes
# ✅ CORRECT: Every panel should have clearly labeled units
panel_config:
  yAxis:
    label: "Requests per second"
    unit: "reqps"

# ❌ WRONG: Red/green color scheme only (colorblind inaccessible)
# ✅ CORRECT: Use shapes, patterns, and colorblind-safe palettes
thresholds:
  - value: 0
    color: "#73BF69"   # Green
  - value: 80
    color: "#FF9830"   # Orange (not yellow)
  - value: 95
    color: "#F2495C"   # Red
```

### C. Effective Dashboard Layout Hierarchy

```yaml
# Three-tier dashboard hierarchy

tier_1_overview:
  name: "Fleet Overview"
  audience: "Management, on-call engineers"
  content:
    - Total request rate across all services
    - Global error rate
    - SLO compliance summary (all services)
    - Active incidents count
    - Error budget remaining per service
  refresh: 30s

tier_2_service:
  name: "Service Detail: ${service}"
  audience: "Service owners, on-call engineers"
  content:
    - Four golden signals for this service
    - Top 5 slowest endpoints
    - Error breakdown by type and endpoint
    - Dependency health (upstream/downstream)
    - Recent deployments overlay
  refresh: 15s

tier_3_debug:
  name: "Debug: ${service} - ${signal}"
  audience: "Engineers actively debugging"
  content:
    - Detailed latency heatmap
    - Individual trace examples
    - Error log stream
    - Resource utilization breakdown
    - Correlation with deployment events
  refresh: 5s
```

---

## 12. Alert Fatigue Prevention

### A. Alert Quality Framework

```yaml
# Score every alert on these dimensions before creating it
alert_quality_checklist:
  actionable:
    question: "Does this alert require a human to take action?"
    bad_example: "CPU at 60% for 5 minutes"  # No action needed
    good_example: "Error budget burn rate exceeds 14.4x for 5 minutes"

  urgent:
    question: "Does this need attention within the response time SLA?"
    bad_example: "Disk at 70% capacity"  # Can wait days
    good_example: "Disk at 95% capacity, projected full in 4 hours"

  real:
    question: "Does this represent a genuine problem?"
    bad_example: "Single 500 error detected"  # Could be a one-off
    good_example: "Error rate sustained above 1% for 5 minutes"

  specific:
    question: "Can the responder identify what to do?"
    bad_example: "Something is wrong with the system"
    good_example: |
      Payment service latency p99 > 2s.
      Runbook: https://wiki/runbooks/payment-latency
      Recent changes: deploy v2.3.1 at 14:02 UTC
```

### B. Multi-Window, Multi-Burn-Rate Alerting

```yaml
# Google SRE multi-window burn rate approach
# Instead of simple threshold alerts, use burn rate windows

groups:
  - name: slo_burn_rate_alerts
    rules:
      # Fast burn: 2% of 30-day budget in 1 hour
      # Catches severe incidents quickly
      - alert: SLOBurnRateFast
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) /
            sum(rate(http_requests_total[5m]))
          ) > (14.4 * 0.001)
          AND
          (
            sum(rate(http_requests_total{status=~"5.."}[1h])) /
            sum(rate(http_requests_total[1h]))
          ) > (14.4 * 0.001)
        for: 2m
        labels:
          severity: critical
          window: fast
        annotations:
          summary: "High burn rate: 2% of monthly budget consumed in 1 hour"
          runbook_url: https://wiki/runbooks/slo-burn

      # Medium burn: 5% of 30-day budget in 6 hours
      - alert: SLOBurnRateMedium
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[30m])) /
            sum(rate(http_requests_total[30m]))
          ) > (6 * 0.001)
          AND
          (
            sum(rate(http_requests_total{status=~"5.."}[6h])) /
            sum(rate(http_requests_total[6h]))
          ) > (6 * 0.001)
        for: 5m
        labels:
          severity: warning
          window: medium
        annotations:
          summary: "Elevated burn rate: 5% of monthly budget consumed in 6 hours"

      # Slow burn: 10% of 30-day budget in 3 days
      - alert: SLOBurnRateSlow
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[6h])) /
            sum(rate(http_requests_total[6h]))
          ) > (1 * 0.001)
          AND
          (
            sum(rate(http_requests_total{status=~"5.."}[3d])) /
            sum(rate(http_requests_total[3d]))
          ) > (1 * 0.001)
        for: 30m
        labels:
          severity: warning
          window: slow
        annotations:
          summary: "Slow burn: 10% of monthly budget consumed in 3 days"
```

### C. Alert Routing and Deduplication

```yaml
# alertmanager.yml - Prevent alert storms
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/...'

route:
  receiver: default
  group_by: ['alertname', 'service']
  group_wait: 30s        # Wait to batch related alerts
  group_interval: 5m     # Minimum time between notifications for a group
  repeat_interval: 4h    # Resend if still firing after 4h

  routes:
    - match:
        severity: critical
      receiver: pagerduty-critical
      group_wait: 10s
      repeat_interval: 1h

    - match:
        severity: warning
      receiver: slack-warnings
      group_wait: 1m
      repeat_interval: 8h

    - match:
        alertname: InfoAlert
      receiver: slack-info
      group_wait: 5m
      repeat_interval: 24h

inhibit_rules:
  # If a critical alert is firing, suppress warnings for the same service
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['service']

  # If a service is fully down, suppress latency alerts
  - source_match:
      alertname: ServiceDown
    target_match:
      alertname: HighLatency
    equal: ['service']

receivers:
  - name: pagerduty-critical
    pagerduty_configs:
      - service_key: '<key>'
        description: '{{ .CommonAnnotations.summary }}'
        details:
          runbook: '{{ .CommonAnnotations.runbook_url }}'

  - name: slack-warnings
    slack_configs:
      - channel: '#alerts-warnings'
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: slack-info
    slack_configs:
      - channel: '#alerts-info'
```

### D. Measuring Alert Quality

```python
# Track alert health metrics
from prometheus_client import Counter, Histogram, Gauge

alerts_fired_total = Counter(
    'alerts_fired_total',
    'Total alerts fired',
    ['alertname', 'severity', 'service']
)

alert_resolution_time_seconds = Histogram(
    'alert_resolution_time_seconds',
    'Time from alert firing to resolution',
    ['alertname', 'severity'],
    buckets=[60, 300, 900, 1800, 3600, 7200, 14400, 28800]
)

# Track actionability - were alerts acknowledged or auto-resolved?
alerts_acknowledged_total = Counter(
    'alerts_acknowledged_total',
    'Alerts that were acknowledged by a human',
    ['alertname']
)

alerts_auto_resolved_total = Counter(
    'alerts_auto_resolved_total',
    'Alerts that resolved without human action',
    ['alertname']
)

# Target metrics for alert health:
# - Actionability rate > 80% (alerts acknowledged / alerts fired)
# - False positive rate < 5%
# - Mean time to acknowledge < 5 minutes for critical
# - Alert-to-incident ratio between 1:1 and 3:1
```

---

## 13. SLI/SLO/SLA Practical Reference

### A. Definitions with Real-World Examples

```yaml
# SLI (Service Level Indicator)
# A carefully defined quantitative measure of some aspect of the service
sli_examples:
  availability:
    definition: "Proportion of successful HTTP requests"
    formula: "good_events / total_events"
    measurement: |
      sum(rate(http_requests_total{status!~"5.."}[window])) /
      sum(rate(http_requests_total[window]))
    what_counts_as_good: "Any response that is not a 5xx"
    what_counts_as_valid: "All requests except health checks"

  latency:
    definition: "Proportion of requests completed within a threshold"
    formula: "fast_requests / total_requests"
    measurement: |
      sum(rate(http_request_duration_seconds_bucket{le="0.3"}[window])) /
      sum(rate(http_request_duration_seconds_count[window]))
    what_counts_as_good: "Response delivered in < 300ms"

  correctness:
    definition: "Proportion of requests that returned correct results"
    formula: "correct_responses / total_responses"
    measurement: "Application-specific validation checks"
    what_counts_as_good: "Response body passes validation, data checksums match"

  freshness:
    definition: "Proportion of data updated within the expected window"
    formula: "fresh_records / total_records"
    what_counts_as_good: "Data updated within last 5 minutes"

# SLO (Service Level Objective)
# A target value or range for an SLI, measured over a time window
slo_examples:
  - service: "API Gateway"
    sli: availability
    target: "99.9%"
    window: "30 days rolling"
    error_budget: "43.2 minutes of downtime per 30 days"
    consequence_of_miss: "Freeze feature launches, focus on reliability"

  - service: "Search Service"
    sli: latency
    target: "99% of requests < 200ms"
    window: "30 days rolling"
    error_budget: "1% of requests can exceed 200ms"
    consequence_of_miss: "Prioritize performance optimization work"

  - service: "Data Pipeline"
    sli: freshness
    target: "99.5% of records updated within 5 minutes"
    window: "7 days rolling"
    consequence_of_miss: "Investigate pipeline bottlenecks"

# SLA (Service Level Agreement)
# A contract with consequences (financial or otherwise) for missing the SLO
sla_examples:
  - service: "Enterprise API"
    guarantee: "99.95% monthly availability"
    measurement_exclusions:
      - Scheduled maintenance windows (up to 4h/month with 48h notice)
      - Force majeure events
    penalties:
      99.95_to_99.9: "10% credit on monthly bill"
      99.9_to_99.0: "25% credit on monthly bill"
      below_99.0: "50% credit on monthly bill"
    reporting: "Monthly uptime report published to status page"
```

### B. Choosing SLO Targets

```yaml
# Decision framework for SLO targets

step_1_understand_user_expectations:
  questions:
    - "What availability do users actually experience today?"
    - "At what point do users start complaining?"
    - "What do competitors offer?"
  rule: "Set the SLO slightly above current performance to be achievable but aspirational"

step_2_consider_dependencies:
  rule: "Your SLO cannot exceed the SLOs of your dependencies"
  example:
    cloud_provider_sla: "99.99%"
    database_sla: "99.95%"
    your_maximum_slo: "99.9%"  # Must be lower than your weakest dependency
    reasoning: "If your DB is down, you are down. You cannot promise better than your DB."

step_3_common_targets_by_tier:
  tier_1_critical:
    examples: "Payment processing, authentication, core API"
    availability: "99.95%"
    latency_p99: "200ms"
    error_budget_monthly: "21.6 minutes"

  tier_2_important:
    examples: "Search, recommendations, notifications"
    availability: "99.9%"
    latency_p99: "500ms"
    error_budget_monthly: "43.2 minutes"

  tier_3_best_effort:
    examples: "Analytics, reporting, batch processing"
    availability: "99.5%"
    latency_p99: "2s"
    error_budget_monthly: "3.6 hours"

step_4_iterate:
  rule: "Start conservative, tighten over time as reliability improves"
  cadence: "Review SLOs quarterly with stakeholders"
```

---

## 14. Deployment Checklist

### Metrics
- [ ] RED metrics for all services
- [ ] USE metrics for all resources
- [ ] Business metrics defined
- [ ] Histogram buckets appropriate

### Tracing
- [ ] All services instrumented
- [ ] Context propagation working (HTTP and message queues)
- [ ] Sampling configured
- [ ] Critical paths traced
- [ ] Database queries instrumented with sanitized statements

### Logging
- [ ] Structured logging implemented
- [ ] Correlation IDs present
- [ ] Log levels appropriate
- [ ] Sensitive data redacted

### Alerting
- [ ] SLOs defined with error budgets
- [ ] Multi-window burn rate alerts configured
- [ ] Notification channels set up with routing and inhibition
- [ ] Runbooks linked to every alert
- [ ] Alert deduplication configured
- [ ] Alert quality metrics tracked (actionability > 80%)

### Dashboards
- [ ] Three-tier dashboard hierarchy created (overview, service, debug)
- [ ] SLO dashboard with error budget tracking
- [ ] Resource dashboards ready
- [ ] Percentiles used for latency (never averages)

---

## 15. Quick Reference

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

# SLI/SLO/SLA relationship
SLI: "The metric" (e.g., proportion of successful requests)
SLO: "The target" (e.g., 99.9% over 30 days)
SLA: "The contract" (e.g., 99.95% with 10% credit penalty)

# Alert thresholds (multi-window burn rate)
Fast burn (2% budget in 1h): Critical, page immediately
Medium burn (5% budget in 6h): Warning, notify Slack
Slow burn (10% budget in 3d): Warning, next business day

# Alert quality targets
Actionability rate: > 80%
False positive rate: < 5%
MTTA for critical: < 5 minutes
```

---

## 16. Why This Configuration Works

- **Three pillars provide complete visibility**: Metrics answer "what is happening," logs answer "why it happened," and traces answer "where it happened." Together they eliminate blind spots that any single pillar would leave, enabling rapid root cause analysis for any incident.
- **SLO-driven alerting reduces noise**: Alerting on error budget burn rates rather than raw thresholds means teams are notified when user-visible impact is real and sustained, not when a momentary spike triggers a false alarm. This keeps on-call engineers focused on genuine problems.
- **OpenTelemetry standardization prevents vendor lock-in**: Using OTLP as the instrumentation standard means the same application code works with Prometheus, Jaeger, Datadog, or any compliant backend. Switching observability vendors becomes a configuration change, not a re-instrumentation project.
- **RED and USE methods ensure comprehensive coverage**: Applying Rate/Errors/Duration to every service and Utilization/Saturation/Errors to every resource creates a systematic framework that catches issues regardless of where they originate, rather than relying on ad-hoc metric selection.
- **Business metrics connect engineering to outcomes**: Tracking orders, revenue, and user engagement alongside technical metrics enables teams to understand the real-world impact of system behavior and prioritize engineering work based on business value rather than technical intuition.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** SRE Team


**End of Observability Guidelines**

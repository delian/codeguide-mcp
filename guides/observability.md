# Observability Guidelines
Mandatory standards for instrumenting services with metrics, distributed tracing, SLIs/SLOs, dashboards, alerting, and health checks. OpenTelemetry, Prometheus, Grafana, Tempo/Jaeger, Alertmanager.

---
name: observability
title: Observability Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [opentelemetry, prometheus, grafana, tempo, jaeger, alertmanager, otel-collector]
requires: []
recommends:
  - logging
  - error-handling
  - microservices
provides:
  - metrics
  - tracing
  - slo-sli
  - alerting
  - health-checks
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide is the **canonical owner** of metrics, distributed tracing, SLIs/SLOs, dashboards, alerting, and health checks. Other guides reference it; it never restates logging, error strategy, or deployment topology.

---

## 0. Prerequisites & References

Observability is one of three telemetry pillars; **logging is its own pillar with its own owner**. This guide owns metrics and traces, and the SLO/alerting/dashboard discipline that consumes all three. It does not restate the others.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`logging.md`](guides://logging.md) — **the logs pillar**: structured-log schema, levels, correlation/trace-ID injection, PII redaction. This guide assumes logs are emitted per `logging.md`; it only covers correlating them with traces.
> - [`error-handling.md`](guides://error-handling.md) — what counts as an error, exception taxonomy, propagation. Observability records error *signals* (span status, error counters); the *strategy* lives there.
> - [`microservices.md`](guides://microservices.md) — service boundaries and request topology that traces span and SLOs are defined per.

> 📎 **SEE ALSO:** [`kubernetes.md`](guides://kubernetes.md) · [`docker-compose.md`](guides://docker-compose.md) — where collectors/exporters and health-probe wiring are deployed. [`performance.md`](guides://performance.md) · [`grpc.md`](guides://grpc.md) · [`rest.md`](guides://rest.md) — protocols whose latency/error SLIs are measured here.

> Telemetry-to-owner map: **metrics → this guide · traces → this guide · logs → `logging.md`**. The correlation glue (trace_id ↔ log line ↔ exemplar) is owned here because it is what makes the three pillars one system.

---

## 1. Core Philosophies: OBSERVE-FIRST

Principles unique to observability. (Logging philosophy is in `logging.md`; error philosophy in `error-handling.md` — not restated.)

- **O**penTelemetry-native: instrument once against the OTel API/SDK and OTLP; the backend (Prometheus, Tempo, Datadog, …) is a config choice, never a code dependency. No vendor SDK in business code.
- **B**usiness signals first: a service is observable only when you can answer "what is happening *to users*", not just CPU%. RED for request flows, USE for resources, plus domain KPIs.
- **S**LO-driven: alert on user-visible symptoms and error-budget burn, never on raw cause-level thresholds. Every page maps to a violated SLO.
- **E**nd-to-end: every request carries W3C `traceparent` context across all hops (HTTP, gRPC, queues); a single trace tells where time and errors went.
- **R**eal-time & correlated: metrics → exemplars → traces → logs must be one click apart. trace_id is the join key across all three pillars.
- **V**isual & actionable: dashboards are built around the Four Golden Signals and answer one question per panel; if a panel triggers no decision, delete it.
- **E**fficient: control cardinality and sampling deliberately — telemetry must not cost more than the system it observes.

**Verified Code**: agent-generated instrumentation MUST satisfy every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `OBS-<TOPIC>-<NN>`. Each row has a binary gate; rows binding another guide's concern cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| OBS-OTEL-01 | Instrumentation MUST use the vendor-neutral OpenTelemetry API/SDK and export via OTLP; no backend-specific SDK in business code | grep for vendor SDK imports outside the telemetry-init module | none found |
| OBS-OTEL-02 | Every service MUST set OTel `Resource` attributes `service.name`, `service.version`, `deployment.environment` | inspect resource / `otel-cli` or collector debug exporter | all three present |
| OBS-MET-01 | Every request-serving service MUST expose RED metrics (rate, errors, duration) with a latency **histogram** (never average) | scrape `/metrics`, assert `*_requests_total`, `*_request_duration_seconds_bucket` | present |
| OBS-MET-02 | Every host/runtime resource MUST expose USE metrics (utilization, saturation, errors) | scrape `/metrics` for `*_utilization*`, `*_saturation*` | present |
| OBS-MET-03 | Metric & label names MUST follow OTel semantic conventions; labels MUST be bounded (no IDs/emails/URLs as label values) | review / `promtool check metrics` | conventions ok, bounded cardinality |
| OBS-MET-04 | Each service MUST emit ≥1 business/domain metric (orders, signups, …) | scrape `/metrics` | ≥1 domain metric |
| OBS-TRC-01 | All request-serving and async-consuming services MUST be traced and propagate **W3C Trace Context** across every hop (HTTP, gRPC, message queues) | run a request, assert single trace spans all hops in backend | one connected trace |
| OBS-TRC-02 | Spans MUST record exceptions and set ERROR status on failure (error definition per `error-handling.md`) | trigger failure, inspect span | status=ERROR + exception event |
| OBS-TRC-03 | Sampling MUST be `ParentBased` and configurable per environment; DB span statements MUST be sanitized (no literals/PII) | inspect sampler config + a `db.statement` span | parent-based; literals stripped |
| OBS-COR-01 | Logs MUST carry the active `trace_id`/`span_id` so logs↔traces↔metrics correlate (logs owned by `logging.md`) | grep a request's log lines for trace_id | trace_id present |
| OBS-SLO-01 | Each user-facing service MUST have ≥1 documented SLI + SLO (target + rolling window) with a defined error budget | SLO doc / recording rules exist | SLO + budget defined |
| OBS-SLO-02 | SLO compliance and error-budget burn MUST be computed as Prometheus recording rules | `promtool check rules` | rules valid |
| OBS-ALT-01 | Paging alerts MUST be **multi-window, multi-burn-rate** on error budget, not raw static thresholds | `promtool check rules` + review | burn-rate alerts present |
| OBS-ALT-02 | Every alert MUST be actionable: severity, a linked runbook_url, and routing/dedup/inhibition configured | review alert rules + Alertmanager config | all alerts have runbook + route |
| OBS-HLT-01 | Every service MUST expose distinct liveness and readiness endpoints (readiness reflects real dependencies); wired to orchestrator probes (see `kubernetes.md`) | curl `/livez` & `/readyz` | both 200; readiness gates on deps |
| OBS-DSH-01 | Each service MUST have a Four-Golden-Signals dashboard; latency panels MUST use percentiles, counters MUST use `rate()` | open dashboard / lint JSON | percentiles + rate(), no raw counters |
| OBS-PII-01 | No secrets/PII in metric labels, span attributes, or baggage (redaction policy per `logging.md`/`secure-coding.md`) | review / scan exported telemetry | none present |

> **Forbidden**: averaging latency (hides tail); high-cardinality labels (user IDs, request IDs, raw paths); a backend-specific client SDK in domain code; alerting on cause-level thresholds (CPU%) instead of SLO symptoms; an alert with no runbook; emitting PII into spans/labels/baggage; a readiness probe that always returns 200.

---

## 3. Three Pillars & the Correlation Contract

| Pillar | Answers | Owner | Primary tool |
|---|---|---|---|
| **Metrics** | *What* is happening (rates, latency dist, saturation) | **this guide** | Prometheus / OTel metrics |
| **Traces** | *Where* time/errors went across services | **this guide** | OTel + Tempo/Jaeger |
| **Logs** | *Why* it happened (event detail) | [`logging.md`](guides://logging.md) | structured logs / Loki |

The pillars are useful only when joined: every log line carries the active `trace_id` (OBS-COR-01), histograms attach **exemplars** pointing at sampled traces, and span attributes mirror the metric labels (`service`, `route`, `status`). Build the join once; debugging then flows metric anomaly → exemplar → trace → correlated logs without leaving the incident.

---

## 4. Metrics (OWNED)

### A. Pick the right instrument
- **Counter** — monotonic totals: requests, errors, bytes (`*_total`).
- **Histogram** — value distributions: latency, payload size. **Always** for latency; gives percentiles + heatmaps. Define explicit buckets matching your SLO threshold.
- **UpDownCounter / Gauge** — values that rise and fall: in-flight requests, queue depth, connections.
- **Observable (async) gauge** — sampled on collection: pool size, cache entries.

> Prefer the **OTel metrics API** so the same code exports to Prometheus *or* OTLP. Vendor `prometheus_client` is acceptable only behind the telemetry-init module (OBS-OTEL-01).

```python
from opentelemetry import metrics
meter = metrics.get_meter("orders")  # backend-neutral; exported via OTLP

requests   = meter.create_counter("http.server.requests", unit="{request}")
latency    = meter.create_histogram("http.server.duration", unit="s",
              explicit_bucket_boundaries_advisory=[0.01,0.05,0.1,0.25,0.5,1,2.5,5,10])
in_flight  = meter.create_up_down_counter("http.server.active_requests")
orders     = meter.create_counter("orders.processed", unit="{order}")  # OBS-MET-04
```

### B. RED — every request-serving service (OBS-MET-01)
**R**ate, **E**rrors, **D**uration. Labels stay bounded (OBS-MET-03): `service`, `route` (the *template* `/users/{id}`, never the concrete path), `method`, `status`.

```promql
sum by (service) (rate(http_server_requests_total[5m]))                                  # Rate
sum by (service) (rate(http_server_requests_total{status=~"5.."}[5m]))                   # Errors
histogram_quantile(0.99, sum by (le,service) (rate(http_server_duration_bucket[5m])))    # Duration p99
```

### C. USE — every resource (OBS-MET-02)
**U**tilization, **S**aturation, **E**rrors for CPU, memory, disk, network, and saturable app resources (thread/connection pools, queues). Host-level USE typically comes from node_exporter/cAdvisor; app-level saturation (pool exhaustion) you must add yourself.

### D. Business metrics (OBS-MET-04)
Track domain outcomes — `orders.processed{status}`, `signups.total{source}`, `payment.amount` (histogram). These connect telemetry to user impact and feed the top-tier dashboard.

### E. Cardinality discipline (OBS-MET-03)
A time series is created per unique label-set; unbounded labels (user id, email, raw URL, error message) explode storage and cost. Use templated routes, bucket continuous values, and put high-cardinality identifiers on **spans/exemplars**, not metric labels.

---

## 5. Distributed Tracing (OWNED)

### A. Init once, instrument the edges automatically
Set `Resource` attributes (OBS-OTEL-02), a `ParentBased` sampler (OBS-TRC-03), batch export over OTLP, then let auto-instrumentation cover frameworks/clients.

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "order-service",            # OBS-OTEL-02
    "service.version": "1.4.0",
    "deployment.environment": "production",
})
provider = TracerProvider(resource=resource,
                          sampler=ParentBased(root=TraceIdRatioBased(0.1)))  # env-configurable
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://collector:4317")))
trace.set_tracer_provider(provider)
```

### B. Manual spans for domain operations (OBS-TRC-02)
Wrap meaningful business steps; record exceptions and set status. What *is* an error is defined by [`error-handling.md`](guides://error-handling.md) — tracing only records that signal.

```python
tracer = trace.get_tracer("orders")
with tracer.start_as_current_span("process_order") as span:
    span.set_attribute("order.id", order_id)        # high-cardinality id → span, NOT a metric label
    try:
        result = charge(order)
        span.add_event("order.processed", {"total": order.total})
    except PaymentError as e:
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))   # OBS-TRC-02
        raise
```

### C. Context propagation — W3C Trace Context (OBS-TRC-01)
The same `traceparent`/`tracestate` headers flow across HTTP, gRPC, and message queues. Auto-instrumentation injects/extracts for instrumented clients; for queues you inject into message headers at the producer and extract at the consumer (span kinds `PRODUCER`/`CONSUMER`). **Baggage** carries small business context (e.g. `user.tier`) downstream — never secrets/PII (OBS-PII-01).

### D. Database & external spans (OBS-TRC-03)
Emit `CLIENT` spans with `db.system`/`db.operation` and a **sanitized** `db.statement` (literals replaced by `?`) — raw statements leak PII and explode cardinality.

### E. Sampling (OBS-TRC-03)
Use `ParentBased` so a request is sampled whole-trace. Set ratio per environment via config (high in staging, low in prod). Prefer **tail-based sampling in the collector** to retain all error/slow traces while down-sampling the rest.

---

## 6. SLIs / SLOs / Error Budgets (OWNED)

- **SLI** — the measured ratio of good events to valid events (availability, latency-within-threshold, freshness, correctness).
- **SLO** — target + rolling window for an SLI (e.g. 99.9% over 30d). **An SLO ≤ the weakest dependency's SLO** — you cannot promise more reliability than what you depend on.
- **Error budget** — `1 − SLO` of allowed failure; the unit of decision-making (spend it on velocity, freeze on exhaustion). SLA is the *contracted* SLO with penalties — a business artifact, not instrumented here.

Tier targets as a starting point: critical 99.95% (p99 200ms), important 99.9% (p99 500ms), best-effort 99.5%. Express SLIs/burn as recording rules (OBS-SLO-02):

```yaml
groups:
  - name: slo:order-service
    rules:
      - record: slo:availability:ratio_rate5m
        expr: |
          sum(rate(http_server_requests_total{service="order-service",status!~"5.."}[5m]))
          / sum(rate(http_server_requests_total{service="order-service"}[5m]))
      - record: slo:error_budget_burn:1h
        expr: (1 - slo:availability:ratio_rate5m) / (1 - 0.999)   # multiples of budget burned
```

---

## 7. Alerting (OWNED)

Alert on **symptoms** (SLO burn), not causes. Resource saturation is context on a dashboard, not a page.

### A. Multi-window, multi-burn-rate (OBS-ALT-01)
Pair a fast and a slow window so you page on real, sustained burn and auto-resolve blips (Google SRE model).

```yaml
groups:
  - name: slo-burn:order-service
    rules:
      - alert: ErrorBudgetBurnFast        # ~2% of 30d budget in 1h → page
        expr: slo:error_budget_burn:5m > 14.4 and slo:error_budget_burn:1h > 14.4
        for: 2m
        labels: {severity: critical}
        annotations:
          summary: "order-service burning error budget 14.4x"
          runbook_url: https://runbooks/order-service/slo-burn   # OBS-ALT-02
      - alert: ErrorBudgetBurnSlow        # ~10% in 3d → ticket
        expr: slo:error_budget_burn:6h > 1 and slo:error_budget_burn:3d > 1
        for: 30m
        labels: {severity: warning}
        annotations: {summary: "order-service slow burn", runbook_url: https://runbooks/order-service/slo-burn}
```

### B. Routing, dedup, inhibition (OBS-ALT-02)
Alertmanager groups related alerts, routes by severity (critical→pager, warning→chat), and **inhibits** downstream noise (a `ServiceDown` suppresses its `HighLatency` alerts). Every alert carries severity + runbook.

### C. Alert quality
Score each alert: **actionable** (needs a human), **urgent** (within its response SLA), **real** (sustained, not a one-off), **specific** (runbook + recent-change context). Track actionability rate (>80%), false-positive rate (<5%), and MTTA — measured via your own `alerts_*_total` counters.

---

## 8. Dashboards (OWNED)

### A. Four Golden Signals (OBS-DSH-01)
Every service dashboard leads with **Traffic, Errors, Latency, Saturation**. Latency uses percentiles (p50/p95/p99) — **never averages**; counters are shown as `rate()` — never raw monotonic values.

### B. Three-tier hierarchy
- **Fleet overview** — total traffic, global error rate, SLO compliance & budget per service (audience: on-call/management).
- **Service detail** — golden signals, slowest endpoints, error breakdown, dependency health, deploy overlay.
- **Debug** — latency heatmaps, exemplar→trace links, log stream, resource breakdown.

### C. Anti-patterns
Averages for latency; raw counters instead of `rate()`; >12 panels per board; mixed unrelated services; missing units; red/green-only thresholds (use a colorblind-safe palette and shapes).

---

## 9. Health Checks (OWNED)

Distinct semantics, distinct endpoints (OBS-HLT-01) — conflating them causes cascading restarts and routing to dead pods.

| Probe | Endpoint | Means | On failure |
|---|---|---|---|
| Liveness | `/livez` | process is not deadlocked | orchestrator restarts the pod |
| Readiness | `/readyz` | dependencies (DB, cache, brokers) are reachable | orchestrator stops routing traffic |
| Startup | `/startupz` | slow boot finished | delays liveness until ready |

Readiness MUST reflect real dependency state (cheap, cached checks — not a heavy query per probe) and MUST be able to return non-200. Probe wiring (`livenessProbe`/`readinessProbe`, timeouts) is deployment-owned — see [`kubernetes.md`](guides://kubernetes.md).

---

## 10. OpenTelemetry Collector

Run the Collector as the egress hub: services export OTLP to it; it batches, limits memory, enriches, tail-samples, and fans out to backends. Swapping a backend is a Collector config change — never a code change (OBS-OTEL-01).

```yaml
receivers:
  otlp: { protocols: { grpc: { endpoint: 0.0.0.0:4317 }, http: { endpoint: 0.0.0.0:4318 } } }
processors:
  memory_limiter: { check_interval: 1s, limit_mib: 1000, spike_limit_mib: 200 }
  batch: { timeout: 10s, send_batch_size: 1000 }
  resource: { attributes: [ { key: deployment.environment, value: production, action: upsert } ] }
exporters:
  prometheus:    { endpoint: 0.0.0.0:8889 }
  otlp/traces:   { endpoint: tempo:4317, tls: { insecure: true } }
service:
  pipelines:
    metrics: { receivers: [otlp], processors: [memory_limiter, batch], exporters: [prometheus] }
    traces:  { receivers: [otlp], processors: [memory_limiter, batch, resource], exporters: [otlp/traces] }
```
Deployment topology (sidecar vs. daemonset vs. gateway) is owned by [`kubernetes.md`](guides://kubernetes.md).

---

## 11. Quick Reference

```promql
# RED
rate(http_server_requests_total[5m])                                          # Rate
rate(http_server_requests_total{status=~"5.."}[5m])                           # Errors
histogram_quantile(0.99, sum by (le)(rate(http_server_duration_bucket[5m])))  # Duration p99

# Bounded labels: service, route(template), method, status   (NEVER user_id, raw path)
# SLO tiers: critical 99.95% / important 99.9% / best-effort 99.5%
# Burn-rate page: fast 14.4x@1h, slow 1x@3d        Sampling: ParentBased, tail-sample errors
# Health: /livez (restart)  /readyz (route)        Correlate: trace_id in every log line
```

---

## 12. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] OBS-OTEL-01/02 — OTel/OTLP only, no vendor SDK in business code; resource has service.name/version/environment
- [ ] OBS-MET-01/02 — RED on every service (latency histogram), USE on every resource
- [ ] OBS-MET-03 — semantic-convention names, bounded labels (no IDs/PII)
- [ ] OBS-MET-04 — ≥1 business/domain metric per service
- [ ] OBS-TRC-01 — all services traced, W3C context across HTTP/gRPC/queues
- [ ] OBS-TRC-02 — spans record exceptions + ERROR status (errors per `error-handling.md`)
- [ ] OBS-TRC-03 — ParentBased sampling configurable; DB statements sanitized
- [ ] OBS-COR-01 — trace_id/span_id in every log line (logs per `logging.md`)
- [ ] OBS-SLO-01/02 — SLI+SLO+budget documented; recording rules valid
- [ ] OBS-ALT-01/02 — multi-window burn-rate alerts; every alert has severity, runbook, routing/inhibition
- [ ] OBS-HLT-01 — distinct /livez & /readyz wired to probes (see `kubernetes.md`)
- [ ] OBS-DSH-01 — Four-Golden-Signals dashboard; percentiles + rate(), no raw counters
- [ ] OBS-PII-01 — no secrets/PII in labels, span attributes, or baggage
- [ ] Agent verified every §2 gate and documented any fixes

---
**End of Observability Guidelines**

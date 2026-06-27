# MLOps Engineering Guidelines
Mandatory standards for platform-agnostic Machine Learning Operations: reproducible ML lifecycle, experiment tracking, data/model versioning, training pipelines, model registry, serving, and drift monitoring. MLflow 3, DVC 3, Feast, Kubeflow/Airflow/Prefect, Evidently.

---
name: mlops
title: MLOps Engineering Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: cross-cutting
tools: [mlflow@3, dvc@3, feast, evidently, kubeflow-pipelines@2, bentoml, onnxruntime]
requires: []
recommends:
  - ci-cd
  - observability
  - pytorch
  - secure-coding
  - devops
provides:
  - ml-lifecycle
  - experiment-tracking
  - model-versioning
  - model-monitoring
  - ml-pipelines
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide owns the **ML lifecycle** and spends its tokens on what is unique to operating ML systems — not on generic CI/CD, observability, or security.

---

## 0. Prerequisites & References

This guide canonically owns: **ML lifecycle, experiment tracking, data/model versioning, feature stores, training pipelines, model registry, deployment/serving, model monitoring (drift), reproducibility.** It does **not** restate CI/CD, observability foundations, training-framework, security, or general infra practice.

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline build/test/deploy mechanics. *(MLOps binding: CI runs data+model gates; CD promotes registry stages; CT — Continuous Training — is a scheduled/drift-triggered pipeline.)*
> - [`observability.md`](guides://observability.md) — metrics/tracing/alerting foundations. *(MLOps binding: model & data-drift signals are emitted as metrics and traced like any other service.)*
> - [`pytorch.md`](guides://pytorch.md) — the training framework, determinism, checkpointing. *(MLOps binding: training code lives behind a pipeline component; seed/determinism rules come from here.)*
> - [`secure-coding.md`](guides://secure-coding.md) — supply chain, secrets, CVEs. *(MLOps binding: model/artifact signing, PII handling, training-data access control.)*
> - [`devops.md`](guides://devops.md) — underlying infra, containers, IaC. *(MLOps binding: serving runs as a scanned container with resource limits.)*

> 📎 **SEE ALSO:** [`python.md`](guides://python.md) · [`docker-compose.md`](guides://docker-compose.md) · [`kubernetes.md`](guides://kubernetes.md) · [`rest.md`](guides://rest.md) · [`grpc.md`](guides://grpc.md) · [`kafka.md`](guides://kafka.md) · [`tdd.md`](guides://tdd.md) · [`env-config.md`](guides://env-config.md)

---

## 1. Core Philosophies: REPRODUCE-FIRST

MLOps-specific principles only. Test-first/security/observability mechanics come from the §0 references — applied to data and models, not restated.

- **R**eproducible: every experiment, run, and deployment is exactly reproducible from `code + data version + config + seed + environment`. No "works on my machine" for ML.
- **E**xperiment-tracked: no untracked training runs — every run logs params, metrics, artifacts, data version, code commit, environment.
- **P**ipelined: every workflow (ingest → validate → transform → train → evaluate → register → deploy) is automated, versioned, and parameterized. **No notebooks as production steps.**
- **R**egistered: every deployable model lives in a registry with a version, lineage to its run, a model card, and an approval gate before production.
- **O**bservable: every production model emits prediction metrics, input/prediction drift signals, and performance indicators (drift signals build on `observability.md`).
- **D**ata-versioned: every dataset, feature set, and transformation is content-addressed, immutable, and traceable to its source.
- **U**nbiased: every model is evaluated for fairness/bias before production; protected-group disparity is a gate, not an afterthought.
- **C**ontinuous: training, evaluation, and deployment are automated — CT (Continuous Training) joins CI/CD (see `ci-cd.md`).
- **E**xplainable: every production model ships interpretability artifacts (feature importance, SHAP) and a model card.

**Cross-cutting bindings:** immutable artifacts (never overwrite a version); train-serve consistency (one feature definition, used in both); graceful fallback (a serving failure rolls back to the previous version, never to no prediction).

**Verified ML**: agent-generated ML code MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `MLO-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| MLO-REPRO-01 | Every training run MUST pin code commit, data version, config, and all random seeds | inspect run metadata; re-run produces identical metrics | bit-reproducible / metrics within tol |
| MLO-REPRO-02 | Environment MUST be locked (deps + Python + CUDA) and logged per run | `dvc.lock`/lockfile present; env logged to tracker | locked & logged |
| MLO-DATA-01 | All training/eval data MUST be versioned and content-addressed | `dvc status` / dataset hash recorded | clean, no uncommitted data |
| MLO-DATA-02 | Input data MUST pass schema + quality validation before training | run data validation suite (pandera/GE/Evidently) | exit 0, no errors |
| MLO-DATA-03 | Train/test/serve splits MUST have no leakage | leakage test (no shared keys, no target leak) | 0 overlap |
| MLO-TRK-01 | Every run MUST log params, metrics, artifacts, data version, code commit (see `observability.md`) | query tracker for required fields | all present |
| MLO-PIPE-01 | The full workflow MUST run end-to-end as code with no manual/notebook steps | `dvc repro` / orchestrator run from clean state | exit 0 |
| MLO-PIPE-02 | Pipeline + each component MUST be tested (data, model, integration, e2e — see `tdd.md`) | run ML test suite | exit 0, 0 skips |
| MLO-REG-01 | Only models passing quality gates MAY be registered | registration script enforces thresholds | gate-fail ⇒ not registered |
| MLO-REG-02 | Every registered model MUST have version, run lineage, signature, and a model card | inspect registry metadata | all present |
| MLO-QUAL-01 | Model MUST meet accuracy/F1/AUC, latency, and size thresholds before promotion | run model quality suite | all thresholds met |
| MLO-QUAL-02 | Model MUST pass fairness/bias evaluation across protected groups | run fairness suite | disparity < bound |
| MLO-DEPLOY-01 | Serving MUST expose health/readiness, validate input, and fall back to prior version | hit `/health` `/ready`; kill-primary test | healthy + fallback works |
| MLO-DEPLOY-02 | New models MUST roll out progressively (shadow/canary/A-B) with auto-rollback | inspect deploy config | guarded rollout configured |
| MLO-MON-01 | Every production model MUST monitor input + prediction drift (see `observability.md`) | drift job active; alerts wired | drift detector running |
| MLO-MON-02 | Drift/accuracy breach MUST page with a runbook and trigger retraining policy | inspect alert + retraining trigger | alert + trigger defined |
| MLO-SEC-01 | No secrets/PII in pipeline code, artifacts, or logs (see `secure-coding.md`) | secret scan; PII scan on artifacts | 0 findings |
| MLO-SEC-02 | Model/data artifacts MUST be signed and access-controlled (see `secure-coding.md`) | verify signature; check RBAC | signed + scoped |

> **Forbidden**: shipping an untracked run; using a notebook as a production pipeline step; training before data validation passes; registering a model below its quality gate; deploying a degraded model without a regression test/monitor first (see `tdd.md`); serving without health checks or fallback; storing models as loose files instead of in a registry.

---

## 3. Verification Protocol

Run, in order, before presenting ML code. Fix → re-run until every gate is green. The *policy* behind each (why test-first, why scan) lives in the §0 references.

```bash
dvc status                         # MLO-DATA-01: data/pipeline state clean
<data-validation-cmd>              # MLO-DATA-02: e.g. python -m src.data.validate
pytest tests/data tests/model tests/integration tests/pipeline   # MLO-PIPE-02
dvc repro                          # MLO-PIPE-01: pipeline runs end-to-end from code
<fairness-suite>                   # MLO-QUAL-02
<secret-scan> && <pii-scan>        # MLO-SEC-01
dvc.lock / lockfile check          # MLO-REPRO-02
```

Reproducibility check (MLO-REPRO-01): re-run with the recorded commit, data version, and seed; metrics MUST match within tolerance.

---

## 4. The ML Lifecycle & Pipeline Stages

**This guide's core.** Every ML workflow is a versioned, automated pipeline. Components are pure functions over typed `Input`/`Output` artifacts so any orchestrator (Kubeflow Pipelines, Airflow, Prefect, Dagster, Argo) can host them.

```
INGEST ─▶ VALIDATE ─▶ TRANSFORM ─▶ TRAIN ─▶ EVALUATE ─▶ REGISTER ─▶ DEPLOY ─▶ MONITOR
  │          │           │           │         │           │          │          │
extract    schema     feature      log       metrics    quality    shadow/    drift +
+ version  quality    eng. +       params/   vs.        gate +     canary +   perf +
+ lineage  + drift    selection    metrics/  baseline   model      health     fallback
           + bias     (feature     artifacts + model    card +     + rollout  trigger ─┐
                       store)      + seed     card      approval                       │
                                                                                       │
       └──────────────────── retraining trigger (drift / schedule / SLA breach) ◀──────┘
```

Pinned per run (MLO-REPRO-01/02): code commit, data version/hash, config, seeds, environment (Python, framework, CUDA/driver). Stored as code — never UI-configured.

### Pipeline-as-code (orchestrator-agnostic component shape)

```python
# pipelines/training_pipeline.py — Kubeflow Pipelines v2 shape; same component
# decomposition maps 1:1 to Airflow tasks, Prefect flows, or Dagster ops.
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output

@dsl.component(base_image="python:3.13-slim", packages_to_install=["pandera==0.20.*", "pandas==2.2.*"])
def validate_data(raw: Input[Dataset], out: Output[Dataset], report: Output[Metrics]):
    import pandas as pd, pandera as pa
    schema = pa.DataFrameSchema({
        "feature_a": pa.Column(float, pa.Check.in_range(-1.0, 1.0)),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
    })
    df = schema.validate(pd.read_parquet(raw.path))   # MLO-DATA-02 — fail fast
    report.log_metric("row_count", len(df))
    df.to_parquet(out.path)

@dsl.component(base_image="python:3.13-slim", packages_to_install=["scikit-learn==1.5.*", "mlflow==3.*"])
def train(data: Input[Dataset], model: Output[Model], metrics: Output[Metrics], seed: int = 42):
    import pandas as pd, mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    mlflow.sklearn.autolog()
    df = pd.read_parquet(data.path)
    X, y = df.drop("label", axis=1), df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    with mlflow.start_run():
        m = RandomForestClassifier(random_state=seed).fit(Xtr, ytr)   # seed pinned — MLO-REPRO-01
        acc = accuracy_score(yte, m.predict(Xte))
        mlflow.log_metric("accuracy", acc)
    metrics.log_metric("accuracy", acc)

@dsl.pipeline(name="training-pipeline")
def training_pipeline(raw_data_path: str, model_name: str = "classifier"):
    v = validate_data(raw=raw_data_path)
    train(data=v.outputs["out"])   # register step gates on metrics — see §6
```

For DVC-native pipelines the same DAG is declared in `dvc.yaml` stages; see §5.

---

## 5. Data & Model Versioning

Every dataset, feature set, and model is **content-addressed, immutable, and traceable**. This is the backbone of reproducibility (MLO-DATA-01, MLO-REPRO-01).

- **Datasets**: content hash (SHA-256) + optional semantic version for managed releases; append-only; stored in versioned storage (S3 + DVC, LakeFS, Delta/Iceberg). Every run logs the data version it consumed.
- **Lineage**: `source → transform → feature → training data → run → model`. Each step recorded so you can answer "which models used data vX?".
- **Models**: versioned in the registry (§6), never as loose files; pinned to the run, data version, and environment that produced them.

### DVC: versioning + pipeline DAG

```bash
dvc add data/train.parquet            # content-address a dataset (creates .dvc pointer)
dvc remote add -d s3 s3://bucket/dvc  # versioned remote
dvc stage add -n prepare -d src/prepare.py -d data/raw -o data/processed python src/prepare.py
dvc stage add -n train   -d src/train.py -d data/processed -p train.lr,train.n_est \
    -o models/model.pkl -M metrics/train.json python src/train.py
dvc repro                             # MLO-PIPE-01: reproduce the full DAG
dvc exp run --set-param train.lr=0.01 # tracked experiment (alt/complement to MLflow)
dvc exp show --sort-by metrics.accuracy
```

Commit `*.dvc`, `dvc.yaml`, `dvc.lock`, `params.yaml`; push data blobs to the remote. `git checkout <rev> && dvc checkout` recovers any historical code+data state exactly.

### Data quality & drift validation

Schema, ranges, nullability, class balance, and reference-drift are validated **before** training (MLO-DATA-02). Use one framework consistently — `pandera` (schema-as-code), Great Expectations (checkpoints), or Evidently (drift). Name the check; do not hand-roll generic stats:

```python
# Train/serve leakage guard (MLO-DATA-03) and reference-drift gate, as named checks.
import pandera as pa
from evidently import Report
from evidently.presets import DataDriftPreset

schema = pa.DataFrameSchema({"amount": pa.Column(float, pa.Check.gt(0), nullable=False)})

def gate(current, reference, train_keys, test_keys):
    schema.validate(current)                                  # MLO-DATA-02
    assert not (set(train_keys) & set(test_keys)), "leakage"  # MLO-DATA-03
    drift = Report([DataDriftPreset()]).run(reference, current)  # input drift before train
    assert not drift.dict()["metrics"][0]["result"]["dataset_drift"], "input drift"
```

---

## 6. Experiment Tracking & Model Registry

### Experiment tracking (what every run logs — MLO-TRK-01)

| Category | Logged |
|---|---|
| Params | every hyperparameter |
| Metrics | every eval metric (accuracy, F1, AUC, calibration, …) |
| Artifacts | model, signature, feature importance/SHAP, confusion matrix, model card |
| Data | data version / hash |
| Code | git commit + branch |
| Environment | Python, framework, CUDA/driver versions |
| Cost | GPU-hours, $ per run (track to optimize) |

Use autolog plus explicit logging for anything framework-agnostic (data version, cost). MLflow, Weights & Biases, Neptune, and DVC experiments are interchangeable for this contract.

```python
import mlflow
from mlflow.models import infer_signature

mlflow.set_experiment("fraud")
mlflow.sklearn.autolog()
with mlflow.start_run(tags={"data_version": dv, "code_commit": sha}):
    model.fit(X_tr, y_tr)
    mlflow.log_metrics({"accuracy": acc, "roc_auc": auc})
    mlflow.sklearn.log_model(model, name="model", signature=infer_signature(X_te, preds))
```

### Model registry & lifecycle (MLO-REG-01/02)

Lifecycle: **Development → Staging → Production → Archived**. Transitions:

- Dev → Staging: automated quality gate pass (MLO-QUAL-01).
- Staging → Production: human approval + shadow/A-B evidence.
- Production → Archived: superseded model retained for audit/rollback.

> **MLflow 3 note (modernization):** the old `MlflowClient.transition_model_version_stage(...)` and string *stages* are **deprecated/removed**. Model lifecycle is now expressed with **registered-model aliases** (e.g. `@champion`, `@staging`) and **tags**; serving references `models:/<name>@champion`. Do not generate stage-based code.

```python
from mlflow import MlflowClient, register_model

def register_if_qualified(run_id, name, thresholds: dict):
    c = MlflowClient()
    metrics = c.get_run(run_id).data.metrics
    for k, lo in thresholds.items():                 # MLO-REG-01: gate before register
        if metrics.get(k, -1) < lo:
            raise ValueError(f"gate failed: {k}={metrics.get(k)} < {lo}")
    mv = register_model(f"runs:/{run_id}/model", name)            # MLO-REG-02: lineage to run
    c.set_registered_model_alias(name, "staging", mv.version)     # MLflow 3 aliases, not stages
    c.set_model_version_tag(name, mv.version, "data_version", metrics.get("data_version", ""))
    return mv
```

### Model card (MLO-REG-02)

Every production model ships a model card (Markdown artifact): model details, intended use & out-of-scope, training-data version + limitations, evaluation results vs. thresholds, fairness/bias findings, known risks, and monitoring/retraining policy. Generate it in the evaluate stage and log it as a run artifact.

---

## 7. Feature Engineering & Feature Stores

The defining MLOps hazard is **train-serve skew**: features computed one way in training and another in serving. Eliminate it with a single feature definition used in both paths (a feature store — Feast, Tecton, or a managed store — or shared transform code).

- **Features are code**: version-controlled transforms, unit-tested; no spreadsheet/manual features.
- **One definition, two reads**: offline (training, point-in-time-correct joins to avoid label leakage) and online (low-latency serving) read the *same* definition.
- **Documented & owned**: each feature has a description, dtype, value range, freshness SLA, owner, and PII flag.
- **Reused**: shared via a catalog so teams don't re-derive the same feature.

Train-serve parity is an explicit, mandatory test:

```python
# tests/features/test_parity.py — the highest-value ML test you will write.
def test_train_serve_parity(feature_store, training_pipeline, sample_ids):
    train = training_pipeline.compute_features(sample_ids)   # offline path
    serve = feature_store.get_online_features(sample_ids).to_df()  # online path
    for col in train.columns:
        pd.testing.assert_series_equal(train[col], serve[col], rtol=1e-5, check_names=False)
```

---

## 8. Model Serving & Deployment

Choose by latency budget: real-time (<100 ms) → REST/gRPC microservice (see [`rest.md`](guides://rest.md)/[`grpc.md`](guides://grpc.md)), batched/queued for throughput, ONNX/TFLite for edge; near-real-time → streaming (Kafka — see [`kafka.md`](guides://kafka.md)); batch → scheduled prediction job (Spark/Beam). BentoML, KServe, NVIDIA Triton, and MLflow's built-in server are common runtimes; the contract below is identical across them.

**Every served model MUST** (MLO-DEPLOY-01): expose `/health` + `/ready`; validate input against the model signature; return the model version; track latency (p50/p90/p99); log requests/responses **without PII** (see `secure-coding.md`); rate-limit + authenticate (see `secure-coding.md`); and **fall back to the previous version** on load/inference failure. The serving container is scanned and resource-limited (see `devops.md`).

```python
# src/serving/app.py — fallback + health are the ML-specific parts.
import os, time, mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

NAME = os.environ["MODEL_NAME"]
model = version = None

@asynccontextmanager
async def lifespan(app):
    global model, version
    for alias in ("champion", "staging"):                     # MLO-DEPLOY-01: fallback chain
        try:
            model = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")  # MLflow 3 alias ref
            version = f"{NAME}@{alias}"
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError("no model available")
    yield

app = FastAPI(lifespan=lifespan)

class Req(BaseModel):
    features: dict[str, float] = Field(min_length=1)          # input validation

@app.get("/ready")
def ready():
    if model is None: raise HTTPException(503, "not loaded")
    return {"status": "ready"}

@app.post("/predict")
def predict(r: Req):
    import pandas as pd
    t = time.perf_counter()
    pred = model.predict(pd.DataFrame([r.features]))
    return {"prediction": int(pred[0]), "model_version": version,
            "latency_ms": round((time.perf_counter() - t) * 1000, 2)}
```

**Progressive rollout (MLO-DEPLOY-02):** shadow mode (new model predicts alongside prod, serves no traffic — compare before promoting); canary (5 → 25 → 50 → 100 %, auto-rollback on metric breach); A/B (split traffic, decide on *business* metrics with significance); blue-green for batch (validate new batch output before atomic swap). All carry automatic rollback — graceful fallback, never no prediction.

---

## 9. Model Monitoring & Drift

Builds on [`observability.md`](guides://observability.md): drift and model-quality signals are emitted as metrics, traced, and alerted with the same plumbing as any service. What is **ML-specific** is *what* you watch and *how* you detect drift.

**Four monitoring layers (MLO-MON-01):**

1. **Data (input)** — feature drift (KS, PSI, Jensen-Shannon), missing-rate, schema violations / new categories, feature-store freshness, input volume.
2. **Model (prediction)** — prediction-distribution drift, confidence/score distribution, accuracy vs. delayed ground truth, calibration, feature-importance stability.
3. **Operational** — latency p50/p90/p99, throughput, error rate, resource use, model-load time (these are owned by `observability.md`; bind ML labels: model name + version).
4. **Business** — KPI correlation (fraud caught, conversion), cost per prediction, time-to-detect / time-to-resolve.

**Drift detection — the core ML technique.** Compare a reference window (training distribution) to the current window per feature:

```python
# src/monitoring/drift.py — KS for continuous, PSI for stability, chi² for categorical.
import numpy as np
from scipy import stats

def psi(ref, cur, bins=10):                       # Population Stability Index
    edges = np.histogram_bin_edges(ref, bins)
    r = (np.histogram(ref, edges)[0] + 1) / (len(ref) + bins)
    c = (np.histogram(cur, edges)[0] + 1) / (len(cur) + bins)
    return float(np.sum((c - r) * np.log(c / r)))

def drifted(ref, cur):                            # severity from PSI bands + KS p-value
    p = psi(ref, cur)
    _, ks_p = stats.ks_2samp(ref, cur)
    if p > 0.2 or ks_p < 0.001: return "critical"
    if p > 0.1 or ks_p < 0.05:  return "warning"
    return "none"
```

Prefer a maintained library (Evidently, NannyML, `alibi-detect`, whylogs) over hand-rolled stats for production.

**Alerting** (severity + runbook required — owned by `observability.md`; ML triggers below): Sev-1 — serving down, accuracy −10 % from baseline, critical drift on >50 % of features, constant predictions, error rate >5 %. Sev-2 — accuracy −5 %, drift on key features, p99 latency >2× baseline, feature staleness > SLA. Every alert carries model name + version, dashboard link, impact, and a runbook.

**Retraining policy (MLO-MON-02):** automatic triggers — PSI > 0.2 on key features, accuracy < SLA, schedule, new-data volume, upstream schema change. **Never** retrain without (a) validating the new data, (b) comparing against the current production model, (c) running the full quality-gate pipeline, and (d) deploying to staging first. CT (Continuous Training) is just this pipeline wired to those triggers (see `ci-cd.md`).

---

## 10. Testing for ML Systems

Test-first per [`tdd.md`](guides://tdd.md), applied at five layers (most tests at the bottom):

1. **Unit** — feature transforms, data-processing functions (pure, fast).
2. **Data validation** — schema, ranges, nulls, distribution, **leakage** (MLO-DATA-02/03).
3. **Model quality** — accuracy/F1/AUC thresholds, latency, size, overfit gap, determinism, **fairness** (MLO-QUAL-01/02).
4. **Integration** — feature-store ↔ model, serving endpoint, train-serve parity.
5. **End-to-end pipeline** — `data → train → evaluate → register` produces a gated, registered model + artifacts.

Each layer's *policy* (red-green-refactor, regression-test-before-fix) is owned by `tdd.md`; the ML specialization is the **assertion targets**: a "test" here checks a metric threshold, a distribution, or train-serve equality — not just code paths.

```python
# Representative model-quality + fairness assertions (the ML-specific bit).
def test_overfit_gap(model, train, test):
    assert model.score(*train) - model.score(*test) < 0.10        # MLO-QUAL-01

def test_determinism(make_model, data, params):
    import numpy as np
    a = make_model(**params, random_state=42).fit(*data)
    b = make_model(**params, random_state=42).fit(*data)
    assert np.allclose(a.predict(data[0]), b.predict(data[0]))     # MLO-REPRO-01

def test_demographic_parity(model, df):                            # MLO-QUAL-02
    preds = model.predict(df.drop(["label", "group"], axis=1))
    rates = [preds[df["group"] == g].mean() for g in df["group"].unique()]
    assert max(rates) - min(rates) < 0.10
```

---

## 11. Project Structure

ML-idiomatic layout. General Python layout/architecture is owned by [`python.md`](guides://python.md); this shows the ML-specific directories.

```
ml-project/
├── src/
│   ├── data/         # loaders, validation, schemas (MLO-DATA-*)
│   ├── features/     # feature definitions, transforms, store client (§7)
│   ├── training/     # train, evaluate, hyperopt
│   ├── serving/      # serving app, online preprocessing (§8)
│   ├── monitoring/   # drift_detector, performance_tracker (§9)
│   ├── pipelines/    # training / batch_inference / retraining (§4)
│   └── registry/     # registration + promotion (§6)
├── tests/            # unit, data, model, integration, fairness, pipeline (§10)
├── data/             # DVC-tracked, NOT in git (raw/ processed/ features/)
├── models/           # DVC-tracked artifacts
├── notebooks/        # exploration ONLY — never a production step
├── configs/          # params.yaml (DVC params), training/serving config (see env-config.md)
├── dvc.yaml / dvc.lock / params.yaml   # pipeline DAG + locked state
├── model_card.md     # generated per release (§6)
├── Dockerfile        # serving container (scanned — see devops.md)
└── pyproject.toml    # locked deps (MLO-REPRO-02)
```

---

## 12. Quick Reference

```bash
# Data & pipeline (DVC)
dvc add data/train.parquet ; dvc repro ; dvc exp run ; dvc exp show ; dvc metrics diff

# Experiment tracking / registry (MLflow 3)
mlflow ui
mlflow models serve -m "models:/fraud@champion"        # alias ref, not stage

# Testing (per layer — §10)
pytest tests/unit tests/data tests/model tests/integration tests/pipeline

# Serving
docker build -t serving . && docker run -p 8080:8080 serving
curl localhost:8080/ready
```

---

## 13. Deployment Checklist

Generated from §2 — one box per requirement ID. No new requirements.

- [ ] MLO-REPRO-01/02 — commit + data version + config + seeds pinned; env locked & logged; re-run reproduces
- [ ] MLO-DATA-01 — data versioned & content-addressed
- [ ] MLO-DATA-02 — data validation (schema/quality) passes before training
- [ ] MLO-DATA-03 — no train/test/serve leakage
- [ ] MLO-TRK-01 — run logs params, metrics, artifacts, data version, code commit
- [ ] MLO-PIPE-01 — pipeline runs end-to-end as code, no notebook/manual steps
- [ ] MLO-PIPE-02 — data/model/integration/e2e tests pass
- [ ] MLO-REG-01 — only gate-passing models registered
- [ ] MLO-REG-02 — registered model has version, lineage, signature, model card
- [ ] MLO-QUAL-01 — accuracy/F1/AUC, latency, size thresholds met
- [ ] MLO-QUAL-02 — fairness/bias evaluation within bounds
- [ ] MLO-DEPLOY-01 — health/readiness, input validation, fallback verified
- [ ] MLO-DEPLOY-02 — progressive rollout with auto-rollback configured
- [ ] MLO-MON-01 — input + prediction drift monitoring active
- [ ] MLO-MON-02 — drift/accuracy alerts with runbook + retraining trigger
- [ ] MLO-SEC-01 — no secrets/PII in code, artifacts, or logs
- [ ] MLO-SEC-02 — artifacts signed & access-controlled
- [ ] Agent ran every §3 command and documented any fixes

---
**End of MLOps Engineering Guidelines**

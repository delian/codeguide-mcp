# MLOps Engineering Guidelines
Mandatory standards and best practices for platform-agnostic Machine Learning Operations. Reproducible pipelines, experiment tracking, model registry, data versioning, model serving, monitoring, and governance. MLflow, DVC, Kubeflow Pipelines, feature stores, model serving frameworks.

---

**Agent Profile**: The MLOps Engineer
**Role**: Senior ML Platform Engineer & Production ML Specialist
**Objective**: Generate production-ready, reproducible, observable, and governed ML systems with automated pipelines from data to deployment.
**Tools**: ML pipeline orchestrators (Kubeflow, Airflow, Prefect, etc.), experiment trackers (MLflow, W&B, etc.), data versioning (DVC, LakeFS), model registries, feature stores, serving frameworks, monitoring systems.

---

## 1. Core Philosophies: REPRODUCE-FIRST

The agent must adhere to the **REPRODUCE-FIRST** principles for every MLOps implementation:

**Test-Driven Development (TDD)**: ALWAYS write data validation tests, model quality tests, and integration tests BEFORE implementation.
**Regression Shield**: EVERY model degradation or data issue MUST receive a test or monitor BEFORE fixing to prevent recurrence.
**Security-First**: Mandatory data access controls, model integrity verification, supply chain security, and PII handling.

- **R**eproducible: Every experiment, training run, and deployment MUST be exactly reproducible from code + data + config.
- **E**xperiment-Tracked: Every training run logs parameters, metrics, artifacts, and environment — no untracked experiments.
- **P**ipelined: Every workflow (data prep, training, evaluation, deployment) is an automated, versioned pipeline — no notebooks in production.
- **R**egistered: Every model is versioned in a registry with metadata, lineage, and approval gates before deployment.
- **O**bservable: Every model in production emits prediction metrics, data drift signals, and performance indicators.
- **D**ata-Versioned: Every dataset, feature set, and transformation is versioned and traceable to its source.
- **U**nbiased: Every model is evaluated for fairness, bias, and ethical considerations before deployment.
- **C**ontinuous: Training, evaluation, and deployment are automated — CT (Continuous Training), CI, CD.
- **E**xplainable: Every production model has interpretability artifacts — feature importance, SHAP values, or model cards.

**Additional Principles:**

- **Data-Centric**: Data quality is as important as model architecture. Garbage in, garbage out.
- **Immutable Artifacts**: Models, datasets, and pipelines are immutable once versioned — never overwrite.
- **Feature Reuse**: Features are shared assets — build once in a feature store, reuse across models.
- **Fail Gracefully**: Model failures fall back to previous version or rule-based defaults, never to no prediction.
- **Cost-Aware**: Track compute costs per experiment, training run, and inference — optimize ruthlessly.

**Verified ML**: Agent-generated ML code MUST pass data validation, model quality checks, and pipeline tests before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated ML code is correct, reproducible, and production-ready before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY ML code, the agent MUST:**

1. **Data Validation**:
   ```bash
   # Validate data schema and quality
   great_expectations checkpoint run my_checkpoint
   # or
   pandera schema.validate(df)
   # or
   dvc params diff  # Verify parameter consistency

   # Validate data pipeline outputs
   dvc repro --dry    # Dry run to check pipeline DAG
   ```
   - **MUST** validate input data schema (types, ranges, nullability)
   - **MUST** check for data leakage between train/test splits
   - **MUST** verify feature distributions match expectations

2. **Experiment Tracking Configured**:
   ```python
   # Verify experiment tracking is active
   import mlflow
   mlflow.autolog()  # or explicit logging

   with mlflow.start_run():
       mlflow.log_params(params)
       mlflow.log_metrics(metrics)
       mlflow.log_artifact("model_card.md")
   ```
   - **MUST** log all hyperparameters
   - **MUST** log all evaluation metrics
   - **MUST** log model artifacts and environment

3. **Model Quality Gates**:
   ```python
   # Verify model meets minimum quality thresholds
   assert accuracy >= MINIMUM_ACCURACY, f"Model accuracy {accuracy} below threshold {MINIMUM_ACCURACY}"
   assert latency_p99 <= MAX_LATENCY_MS, f"Inference latency {latency_p99}ms exceeds {MAX_LATENCY_MS}ms"
   assert model_size_mb <= MAX_MODEL_SIZE_MB, f"Model size {model_size_mb}MB exceeds limit"
   ```

4. **Reproducibility Verification**:
   ```bash
   # Verify all random seeds are set
   # Verify data version is pinned
   dvc status           # Check data pipeline status
   git status           # Check code status
   pip freeze > requirements.txt  # Lock dependencies
   ```
   - **MUST** pin all random seeds
   - **MUST** pin all dependency versions
   - **MUST** version training data

5. **Security & Privacy**:
   ```bash
   # Scan for PII in datasets
   # Verify model doesn't memorize training data
   # Check for adversarial vulnerability
   ```
   - **MUST** verify no PII leakage in model artifacts
   - **MUST** verify no hardcoded credentials in pipeline code
   - **MUST** verify data access controls are in place

#### Error Correction Process

If verification fails:

1. **Data Quality Failure**:
   - Inspect failing data validation rules
   - Check for upstream data source changes
   - Fix data pipeline, re-validate
   - Update data tests to catch the issue

2. **Model Quality Below Threshold**:
   - Review experiment tracking for recent changes
   - Compare against baseline model metrics
   - Check for data drift or feature distribution changes
   - Retrain with validated data, re-evaluate

3. **Reproducibility Failure**:
   - Verify all random seeds are deterministic
   - Check dependency versions match
   - Verify data version matches expected hash
   - Re-run from clean state

### B. Prohibited Practices

**NEVER deliver ML code that:**
- [ ] Has no experiment tracking (untracked training runs)
- [ ] Uses notebooks as production pipeline steps
- [ ] Has no data validation or schema checks
- [ ] Has hardcoded file paths, credentials, or magic numbers
- [ ] Has no test/validation split or uses data leakage
- [ ] Ships a model without evaluation metrics logged
- [ ] Has no model versioning or registry entry
- [ ] Lacks reproducibility (unseeded randomness, unpinned dependencies)
- [ ] Serves models without health checks or fallback
- [ ] Has no monitoring for data drift or model performance
- [ ] Trains on raw data without feature engineering pipeline
- [ ] Stores models as loose files instead of in a registry
- [ ] **Deploys a degraded model without a regression test or monitor first**
- [ ] **Skips data validation before training (violates data quality gates)**

---

## 2A. Test-Driven Development (TDD) Protocol for ML (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle adapted for ML systems.**

### TDD Cycle for ML

```
ML TDD is applied at THREE levels:

1. DATA TESTS (Red-Green-Refactor)
   - RED: Write data validation test (schema, ranges, distributions)
   - GREEN: Build data pipeline that satisfies tests
   - REFACTOR: Optimize pipeline, add edge case handling
   ↓
2. MODEL TESTS (Red-Green-Refactor)
   - RED: Write model quality test (accuracy > X, latency < Y)
   - GREEN: Train model that passes quality gates
   - REFACTOR: Optimize architecture, hyperparameters
   ↓
3. INTEGRATION TESTS (Red-Green-Refactor)
   - RED: Write end-to-end pipeline test
   - GREEN: Wire data → training → evaluation → serving
   - REFACTOR: Optimize pipeline performance, add caching
```

### Data Test Examples

```python
# tests/data/test_training_data.py

import pytest
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# Step 1: RED — Define expected data schema BEFORE building pipeline
training_data_schema = DataFrameSchema({
    "user_id": Column(int, Check.greater_than(0), nullable=False),
    "feature_a": Column(float, Check.in_range(-1.0, 1.0), nullable=False),
    "feature_b": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    "label": Column(int, Check.isin([0, 1]), nullable=False),
})

def test_training_data_schema(training_df):
    """Data must conform to expected schema."""
    training_data_schema.validate(training_df)

def test_no_data_leakage(train_df, test_df):
    """Train and test sets must not overlap."""
    train_ids = set(train_df["user_id"])
    test_ids = set(test_df["user_id"])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} shared IDs"

def test_class_balance(training_df):
    """Label distribution must not be extremely imbalanced."""
    label_counts = training_df["label"].value_counts(normalize=True)
    minority_ratio = label_counts.min()
    assert minority_ratio >= 0.05, f"Extreme imbalance: minority class is {minority_ratio:.1%}"

def test_no_null_features(training_df):
    """No null values in feature columns after preprocessing."""
    feature_cols = [c for c in training_df.columns if c.startswith("feature_")]
    null_counts = training_df[feature_cols].isnull().sum()
    assert null_counts.sum() == 0, f"Null features found: {null_counts[null_counts > 0].to_dict()}"

def test_feature_distributions_stable(current_df, reference_df):
    """Feature distributions must not drift significantly from reference."""
    from scipy import stats
    for col in ["feature_a", "feature_b"]:
        statistic, p_value = stats.ks_2samp(current_df[col], reference_df[col])
        assert p_value > 0.01, f"Feature '{col}' has drifted (KS p-value: {p_value:.4f})"
```

### Model Test Examples

```python
# tests/model/test_model_quality.py

import pytest

MINIMUM_ACCURACY = 0.85
MINIMUM_F1 = 0.80
MAX_LATENCY_MS = 50
MAX_MODEL_SIZE_MB = 500

def test_model_accuracy(trained_model, test_data):
    """Model must meet minimum accuracy threshold."""
    X_test, y_test = test_data
    accuracy = trained_model.score(X_test, y_test)
    assert accuracy >= MINIMUM_ACCURACY, (
        f"Model accuracy {accuracy:.4f} below threshold {MINIMUM_ACCURACY}"
    )

def test_model_f1_score(trained_model, test_data):
    """Model must meet minimum F1 score for each class."""
    from sklearn.metrics import f1_score
    X_test, y_test = test_data
    predictions = trained_model.predict(X_test)
    f1 = f1_score(y_test, predictions, average="weighted")
    assert f1 >= MINIMUM_F1, f"F1 score {f1:.4f} below threshold {MINIMUM_F1}"

def test_model_inference_latency(trained_model, single_sample):
    """Single prediction must complete within latency budget."""
    import time
    start = time.perf_counter()
    for _ in range(100):
        trained_model.predict(single_sample)
    elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    assert elapsed_ms <= MAX_LATENCY_MS, f"Latency {elapsed_ms:.1f}ms exceeds {MAX_LATENCY_MS}ms"

def test_model_not_overfit(trained_model, train_data, test_data):
    """Gap between train and test accuracy must be reasonable."""
    train_acc = trained_model.score(*train_data)
    test_acc = trained_model.score(*test_data)
    gap = train_acc - test_acc
    assert gap < 0.10, f"Overfitting detected: train={train_acc:.4f}, test={test_acc:.4f}, gap={gap:.4f}"

def test_model_deterministic(model_class, train_data, model_params):
    """Training with same seed must produce identical results."""
    model_1 = model_class(**model_params, random_state=42).fit(*train_data)
    model_2 = model_class(**model_params, random_state=42).fit(*train_data)
    import numpy as np
    assert np.allclose(model_1.predict(train_data[0]), model_2.predict(train_data[0]))
```

---

## 2B. Model Degradation / Bug Fix Protocol (MANDATORY)

**CRITICAL: Every model failure, data issue, or performance degradation MUST receive a test or monitor BEFORE fixing.**

### ML Incident Response Workflow

```
1. Degradation Detected (monitoring alert, user report, metric drop)
   ↓
2. Triage: Is it data drift, model decay, or a bug?
   ├── Data drift → Check feature distributions vs. training data
   ├── Model decay → Compare current metrics vs. baseline
   └── Code bug → Check pipeline logic, feature engineering
   ↓
3. Mitigate: Immediate action to restore quality
   ├── Rollback to previous model version (from registry)
   ├── Enable fallback/rule-based predictions
   └── Disable affected feature or endpoint
   ↓
4. Write Regression Test / Add Monitor
   ├── Add data validation test for the drift pattern
   ├── Add model quality test for the failure mode
   ├── Add monitoring alert for the symptom
   └── Update data schema if source changed
   ↓
5. Fix Root Cause
   ├── Retrain on corrected/updated data
   ├── Fix feature engineering bug
   └── Update data pipeline
   ↓
6. Verify: Regression test passes, monitor is active
   ↓
7. Post-Incident Review
   ├── What data or model assumptions broke?
   ├── How can we detect this earlier?
   └── Action items with owners and deadlines
   ↓
8. Deploy Updated Model via Standard Pipeline
```

### Example: Data Drift Regression Test

```python
# tests/monitoring/test_data_drift.py

# INC-456: Feature 'transaction_amount' distribution shifted due to
# currency change in upstream data source.

def test_transaction_amount_range(input_df):
    """INC-456: Transaction amounts must be in expected range.

    Bug: Upstream system switched from cents to dollars, causing
    the model to see 100x smaller values than during training.
    """
    assert input_df["transaction_amount"].min() >= 1.0, \
        "INC-456: Transaction amounts too small — check currency units"
    assert input_df["transaction_amount"].max() <= 1_000_000, \
        "INC-456: Transaction amounts too large — check currency units"
    assert input_df["transaction_amount"].median() >= 10.0, \
        "INC-456: Median transaction amount suspiciously low"

def test_feature_drift_within_bounds(current_features, reference_stats):
    """INC-456: Feature statistics must not drift beyond training bounds."""
    for feature_name, ref in reference_stats.items():
        current_mean = current_features[feature_name].mean()
        # Allow 3x standard deviation drift from training distribution
        lower = ref["mean"] - 3 * ref["std"]
        upper = ref["mean"] + 3 * ref["std"]
        assert lower <= current_mean <= upper, (
            f"INC-456: Feature '{feature_name}' mean={current_mean:.4f} "
            f"outside expected range [{lower:.4f}, {upper:.4f}]"
        )
```

---

## 3. ML Pipeline Architecture (MANDATORY)

### A. Standard ML Pipeline Stages

**Every ML pipeline MUST include these stages:**

```
┌──────────────────────────────────────────────────────────────────────┐
│                    STANDARD ML PIPELINE STAGES                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. DATA INGEST      2. DATA VALIDATE    3. DATA TRANSFORM           │
│  ┌──────────────┐   ┌──────────────┐    ┌──────────────┐            │
│  │ Extract from  │   │ Schema check │    │ Feature eng. │            │
│  │ source        │   │ Quality check│    │ Normalization│            │
│  │ Version data  │   │ Drift detect │    │ Encoding     │            │
│  │ Track lineage │   │ Bias check   │    │ Selection    │            │
│  └──────┬───────┘   └──────┬───────┘    └──────┬───────┘            │
│         │                   │                    │                     │
│         ▼                   ▼                    ▼                     │
│  4. TRAIN            5. EVALUATE          6. VALIDATE                 │
│  ┌──────────────┐   ┌──────────────┐    ┌──────────────┐            │
│  │ Train model  │   │ Compute      │    │ Quality gates│            │
│  │ Log params   │   │ metrics      │    │ Fairness     │            │
│  │ Log metrics  │   │ Compare to   │    │ checks       │            │
│  │ Log artifacts│   │ baseline     │    │ A/B readiness│            │
│  │ Track compute│   │ Generate     │    │ Latency test │            │
│  │ cost         │   │ model card   │    │ Size check   │            │
│  └──────────────┘   └──────────────┘    └──────────────┘            │
│                                                                       │
│  7. REGISTER         8. DEPLOY            9. MONITOR                  │
│  ┌──────────────┐   ┌──────────────┐    ┌──────────────┐            │
│  │ Version model│   │ Canary/shadow│    │ Data drift   │            │
│  │ Store in     │   │ A/B test     │    │ Model perf.  │            │
│  │ registry     │   │ Gradual      │    │ Prediction   │            │
│  │ Approval gate│   │ rollout      │    │ distribution │            │
│  │ Model card   │   │ Health check │    │ Business KPI │            │
│  └──────────────┘   └──────────────┘    └──────────────┘            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### B. Pipeline-as-Code Example

```python
# pipelines/training_pipeline.py
# Platform-agnostic pipeline definition (adapt to your orchestrator)

from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output

@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas==2.2.0", "pandera==0.18.0"],
)
def validate_data(
    raw_data: Input[Dataset],
    validated_data: Output[Dataset],
    validation_report: Output[Metrics],
):
    """Validate input data schema and quality."""
    import pandas as pd
    import pandera as pa

    df = pd.read_parquet(raw_data.path)

    schema = pa.DataFrameSchema({
        "feature_a": pa.Column(float, pa.Check.in_range(-1.0, 1.0)),
        "feature_b": pa.Column(float, pa.Check.greater_than(0)),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
    })

    validated = schema.validate(df)

    # Log validation metrics
    validation_report.log_metric("row_count", len(validated))
    validation_report.log_metric("null_ratio", validated.isnull().sum().sum() / validated.size)
    validation_report.log_metric("label_balance", validated["label"].mean())

    validated.to_parquet(validated_data.path)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["scikit-learn==1.4.0", "mlflow==2.10.0", "pandas==2.2.0"],
)
def train_model(
    training_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
    max_depth: int = 10,
    random_seed: int = 42,
):
    """Train model with experiment tracking."""
    import pandas as pd
    import mlflow
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    mlflow.autolog()

    df = pd.read_parquet(training_data.path)
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1})
        mlflow.sklearn.log_model(model, "model")

    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("f1_weighted", f1)


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow==2.10.0"],
)
def register_model(
    model_artifact: Input[Model],
    model_name: str,
    minimum_accuracy: float = 0.85,
):
    """Register model if it passes quality gates."""
    import mlflow

    # Gate: Only register if quality threshold met
    run = mlflow.last_active_run()
    accuracy = run.data.metrics.get("accuracy", 0)

    if accuracy < minimum_accuracy:
        raise ValueError(
            f"Model accuracy {accuracy:.4f} below threshold {minimum_accuracy}. "
            f"Model NOT registered."
        )

    result = mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name=model_name,
    )
    print(f"Registered model '{model_name}' version {result.version}")


@dsl.pipeline(name="training-pipeline")
def training_pipeline(
    raw_data_path: str,
    model_name: str = "my-classifier",
    n_estimators: int = 100,
    max_depth: int = 10,
):
    """End-to-end training pipeline with quality gates."""
    validate_task = validate_data(raw_data=raw_data_path)

    train_task = train_model(
        training_data=validate_task.outputs["validated_data"],
        n_estimators=n_estimators,
        max_depth=max_depth,
    )

    register_model(
        model_artifact=train_task.outputs["model_artifact"],
        model_name=model_name,
    )
```

### C. Pipeline Requirements (Platform-Agnostic)

```yaml
# ML pipeline specification (adapt to your orchestrator)

pipeline:
  # 1. Triggers
  triggers:
    - schedule: "0 2 * * *"        # Nightly retraining
    - data_change: "data/v*"       # New data version
    - manual: true                  # On-demand retraining
    - drift_alert: true             # Triggered by monitoring

  # 2. Reproducibility
  reproducibility:
    - Pin all dependency versions (requirements.txt / poetry.lock)
    - Pin data version (DVC, LakeFS, or dataset hash)
    - Pin random seeds for all stochastic operations
    - Log full environment (Python version, GPU driver, CUDA)
    - Store pipeline definition as code (not UI-configured)

  # 3. Artifacts
  artifacts:
    - Trained model (serialized, signed)
    - Training data version reference
    - Evaluation metrics (JSON, logged to tracker)
    - Model card (Markdown)
    - Feature importance / SHAP values
    - Confusion matrix, ROC curve
    - Data validation report
    - Pipeline execution graph

  # 4. Quality gates
  gates:
    - Data validation passes (schema, quality, bias)
    - Model accuracy >= baseline + threshold
    - Model latency <= SLA
    - Model size <= deployment limit
    - Fairness metrics within bounds
    - No data leakage detected
    - Reproducibility verified (re-run produces same result)
```

---

## 4. Data Management & Versioning (MANDATORY)

### A. Data Versioning Requirements

**ALL training data, evaluation data, and feature sets MUST be versioned.**

```
DATA VERSIONING REQUIREMENTS:

1. Every Dataset Has a Version
   ├── Content-addressable hash (SHA-256)
   ├── Semantic version for managed releases
   ├── Immutable once created (append-only)
   └── Stored in versioned storage (S3 + DVC, LakeFS, Delta Lake)

2. Every Training Run References Data Version
   ├── Data version logged in experiment tracker
   ├── Exact data can be retrieved for any historical run
   └── git + dvc checkout can reproduce any past state

3. Data Lineage is Tracked
   ├── Source → Transform → Feature → Training Data
   ├── Every transformation step is recorded
   └── Impact analysis: which models used which data version?
```

### B. Data Versioning with DVC

```bash
# Initialize DVC in your ML project
dvc init

# Track a large dataset
dvc add data/training/dataset_v1.parquet
git add data/training/dataset_v1.parquet.dvc data/training/.gitignore
git commit -m "data(training): add v1 training dataset"

# Define a reproducible pipeline
dvc stage add -n prepare \
    -d src/prepare.py -d data/raw/ \
    -o data/processed/ \
    python src/prepare.py

dvc stage add -n train \
    -d src/train.py -d data/processed/ \
    -p train.n_estimators,train.learning_rate \
    -o models/model.pkl \
    -M metrics/train_metrics.json \
    python src/train.py

dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/processed/test.parquet \
    -M metrics/eval_metrics.json \
    python src/evaluate.py

# Reproduce the full pipeline
dvc repro

# Compare experiments
dvc params diff
dvc metrics diff

# Track experiments
dvc exp run -n experiment_lr_0.01 --set-param train.learning_rate=0.01
dvc exp run -n experiment_lr_0.001 --set-param train.learning_rate=0.001
dvc exp show
```

### C. Data Quality Framework

```python
# src/data/validation.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class DataQualityReport:
    passed: bool
    checks: Dict[str, bool]
    warnings: List[str]
    errors: List[str]
    statistics: Dict[str, float]

def validate_training_data(
    df: pd.DataFrame,
    reference_stats: Optional[Dict] = None,
) -> DataQualityReport:
    """Validate training data before model training.

    Checks:
    - Schema conformance (types, nullability)
    - Value ranges and distributions
    - Class balance
    - Feature correlations
    - Data freshness
    - Drift from reference distribution
    """
    checks = {}
    warnings = []
    errors = []
    stats = {}

    # 1. Null check
    null_ratio = df.isnull().sum().sum() / df.size
    stats["null_ratio"] = null_ratio
    checks["no_excessive_nulls"] = null_ratio < 0.05
    if not checks["no_excessive_nulls"]:
        errors.append(f"Null ratio {null_ratio:.2%} exceeds 5% threshold")

    # 2. Duplicate check
    dup_ratio = df.duplicated().sum() / len(df)
    stats["duplicate_ratio"] = dup_ratio
    checks["no_excessive_duplicates"] = dup_ratio < 0.01
    if not checks["no_excessive_duplicates"]:
        warnings.append(f"Duplicate ratio {dup_ratio:.2%} — verify intentional")

    # 3. Row count check
    stats["row_count"] = len(df)
    checks["sufficient_rows"] = len(df) >= 1000
    if not checks["sufficient_rows"]:
        errors.append(f"Only {len(df)} rows — minimum 1000 required")

    # 4. Label balance (for classification)
    if "label" in df.columns:
        balance = df["label"].value_counts(normalize=True)
        minority = balance.min()
        stats["minority_class_ratio"] = minority
        checks["class_balance"] = minority >= 0.05
        if not checks["class_balance"]:
            warnings.append(f"Minority class ratio {minority:.2%} — consider oversampling")

    # 5. Distribution drift (if reference provided)
    if reference_stats:
        from scipy.stats import ks_2samp
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        for col in feature_cols:
            if col in reference_stats:
                stat, p_val = ks_2samp(df[col].dropna(), reference_stats[col])
                checks[f"no_drift_{col}"] = p_val > 0.01
                if not checks[f"no_drift_{col}"]:
                    warnings.append(f"Drift detected in '{col}' (p={p_val:.4f})")

    passed = all(checks.values()) and len(errors) == 0

    return DataQualityReport(
        passed=passed,
        checks=checks,
        warnings=warnings,
        errors=errors,
        statistics=stats,
    )
```

---

## 5. Experiment Tracking (MANDATORY)

### A. Experiment Tracking Requirements

**Every training run MUST log the following:**

| Category | What to Log | Example |
|----------|------------|---------|
| **Parameters** | All hyperparameters | `learning_rate=0.001`, `n_estimators=100` |
| **Metrics** | All evaluation metrics | `accuracy=0.92`, `f1=0.89`, `auc=0.95` |
| **Artifacts** | Model, plots, reports | `model.pkl`, `confusion_matrix.png`, `model_card.md` |
| **Data** | Data version/hash | `data_version=v2.3`, `data_hash=sha256:abc...` |
| **Environment** | Runtime details | `python=3.11`, `cuda=12.1`, `gpu=A100` |
| **Code** | Git commit, branch | `commit=abc123`, `branch=feature/new-model` |
| **Cost** | Compute resources | `gpu_hours=2.5`, `cost_usd=12.50` |
| **Tags** | Experiment metadata | `team=fraud`, `project=transaction-scoring` |

### B. Experiment Tracking Implementation

```python
# src/training/train.py

import mlflow
from mlflow.models import infer_signature

def train_and_track(
    X_train, y_train, X_test, y_test,
    params: dict,
    experiment_name: str,
    data_version: str,
):
    """Train model with comprehensive experiment tracking."""
    mlflow.set_experiment(experiment_name)

    # Enable autologging for supported frameworks
    mlflow.autolog()

    with mlflow.start_run(
        tags={
            "data_version": data_version,
            "team": "ml-platform",
            "pipeline": "training_v2",
        }
    ):
        # 1. Log all parameters explicitly (autolog may miss custom ones)
        mlflow.log_params(params)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # 2. Train model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # 3. Evaluate and log metrics
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score,
            recall_score, roc_auc_score, classification_report
        )
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_weighted": f1_score(y_test, predictions, average="weighted"),
            "precision_weighted": precision_score(y_test, predictions, average="weighted"),
            "recall_weighted": recall_score(y_test, predictions, average="weighted"),
            "roc_auc": roc_auc_score(y_test, probabilities[:, 1]),
        }
        mlflow.log_metrics(metrics)

        # 4. Log model with signature
        signature = infer_signature(X_test, predictions)
        mlflow.sklearn.log_model(
            model, "model",
            signature=signature,
            registered_model_name=None,  # Register separately after validation
        )

        # 5. Log artifacts
        # Feature importance
        import pandas as pd
        importance_df = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")

        # Classification report
        report = classification_report(y_test, predictions)
        with open("/tmp/classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("/tmp/classification_report.txt")

        # 6. Log environment info
        import platform
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("os", platform.system())

        return model, metrics
```

### C. Experiment Comparison

```bash
# CLI-based experiment comparison
mlflow experiments search --filter "tags.team = 'fraud'"

# Compare runs
mlflow runs list --experiment-id 1 --order-by "metrics.accuracy DESC"

# DVC-based experiment comparison
dvc exp show --sort-by metrics.accuracy --drop params
dvc plots diff exp-branch-1 exp-branch-2
```

---

## 6. Model Registry & Lifecycle (MANDATORY)

### A. Model Registry Requirements

**Every production model MUST be registered with full metadata.**

```
MODEL REGISTRY REQUIREMENTS:

Every registered model MUST have:
├── Unique name and semantic version
├── Source experiment run ID (traceable to training data + code)
├── Model signature (input/output schema)
├── Quality metrics (accuracy, latency, size)
├── Model card (purpose, limitations, biases, intended use)
├── Approval status (staging → production requires approval)
├── Data lineage (which data version was used)
├── Environment specification (dependencies, runtime)
└── Owner and team assignment
```

### B. Model Lifecycle Stages

```
MODEL LIFECYCLE:

  Development → Staging → Production → Archived
       │            │          │            │
       ▼            ▼          ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Tracked │ │ Tested  │ │ Serving │ │ Retired │
  │ in exp. │ │ against │ │ traffic │ │ kept    │
  │ tracker │ │ staging │ │ with    │ │ for     │
  │         │ │ data    │ │ monitor │ │ audit   │
  └─────────┘ └─────────┘ └─────────┘ └─────────┘

Transitions require:
  Dev → Staging:    Automated quality gate pass
  Staging → Prod:   Human approval + A/B test results
  Prod → Archived:  New model promoted, old model retained
```

### C. Model Registration

```python
# src/registry/register_model.py

import mlflow
from mlflow import MlflowClient

def register_if_qualified(
    run_id: str,
    model_name: str,
    quality_thresholds: dict,
):
    """Register model only if it passes all quality gates."""
    client = MlflowClient()
    run = client.get_run(run_id)

    # 1. Check quality gates
    metrics = run.data.metrics
    for metric_name, threshold in quality_thresholds.items():
        actual = metrics.get(metric_name)
        if actual is None:
            raise ValueError(f"Metric '{metric_name}' not logged in run {run_id}")
        if actual < threshold:
            raise ValueError(
                f"Quality gate failed: {metric_name}={actual:.4f} < {threshold}"
            )

    # 2. Register model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # 3. Add metadata tags
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="quality_gate",
        value="passed",
    )
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="data_version",
        value=run.data.params.get("data_version", "unknown"),
    )

    # 4. Transition to staging (production requires manual approval)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
    )

    print(f"Model '{model_name}' v{result.version} registered and staged.")
    return result


# Quality thresholds
QUALITY_THRESHOLDS = {
    "accuracy": 0.85,
    "f1_weighted": 0.80,
    "roc_auc": 0.90,
}

# Usage
register_if_qualified(
    run_id="abc123def456",
    model_name="fraud-detector",
    quality_thresholds=QUALITY_THRESHOLDS,
)
```

### D. Model Card (MANDATORY)

**Every production model MUST have a model card.**

```markdown
# Model Card: [Model Name]

## Model Details
- **Name**: fraud-detector
- **Version**: 2.3.0
- **Type**: Binary classifier (Random Forest)
- **Owner**: ML Platform Team
- **Created**: 2026-03-15
- **Framework**: scikit-learn 1.4.0

## Intended Use
- **Primary use**: Detect fraudulent transactions in real-time payment pipeline
- **Users**: Payment processing service (automated)
- **Out-of-scope**: Not suitable for credit scoring or identity verification

## Training Data
- **Dataset**: transactions_v2.3 (DVC hash: sha256:abc123...)
- **Size**: 2.1M samples (1.9M legitimate, 200K fraudulent)
- **Date range**: 2025-01-01 to 2025-12-31
- **Known limitations**: Underrepresents international transactions (<5%)

## Evaluation Results
| Metric | Value | Threshold |
|--------|-------|-----------|
| Accuracy | 0.943 | ≥0.85 |
| F1 (weighted) | 0.921 | ≥0.80 |
| AUC-ROC | 0.967 | ≥0.90 |
| Precision (fraud) | 0.891 | ≥0.80 |
| Recall (fraud) | 0.834 | ≥0.75 |
| P99 latency | 12ms | ≤50ms |

## Fairness & Bias
- Evaluated across: age groups, geographic regions, transaction types
- No significant performance disparity detected (max 3% accuracy gap)
- See full fairness report: artifacts/fairness_report.html

## Limitations & Risks
- Performance degrades on transactions >$50,000 (sparse training data)
- Not trained on cryptocurrency transactions
- Requires retraining if payment gateway schema changes

## Monitoring
- Dashboard: [URL]
- Alerts: accuracy drop >5%, drift detection, prediction volume anomaly
- Retraining trigger: Weekly or on drift alert
```

---

## 7. Feature Engineering & Feature Stores (MANDATORY)

### A. Feature Engineering Principles

```
FEATURE ENGINEERING REQUIREMENTS:

1. Features are Code
   ├── All transformations defined in version-controlled code
   ├── Never manual feature creation (no spreadsheet features)
   ├── Feature logic tested with unit tests
   └── Feature definitions shared in feature store

2. Train-Serve Consistency (CRITICAL)
   ├── Same feature computation for training and serving
   ├── Use feature store to guarantee consistency
   ├── NEVER reimplement features separately for serving
   └── Test train-serve parity explicitly

3. Feature Documentation
   ├── Every feature has a human-readable description
   ├── Expected range and distribution documented
   ├── Business context and derivation logic documented
   └── Owner and freshness requirements specified

4. Feature Reuse
   ├── Features shared across models via feature store
   ├── Avoid duplicate feature logic across teams
   ├── Feature discovery through centralized catalog
   └── Feature versioning for backward compatibility
```

### B. Feature Store Pattern

```python
# src/features/definitions.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class FeatureDefinition:
    """Feature definition for feature store registration."""
    name: str
    description: str
    dtype: str
    source: str
    owner: str
    freshness_sla: str  # e.g., "1h", "1d", "real-time"
    value_range: Optional[tuple] = None
    tags: Optional[dict] = None

# Feature catalog
FEATURES = [
    FeatureDefinition(
        name="user_transaction_count_7d",
        description="Number of transactions by user in the last 7 days",
        dtype="int64",
        source="transactions_db",
        owner="fraud-team",
        freshness_sla="1h",
        value_range=(0, 10_000),
        tags={"domain": "fraud", "pii": False},
    ),
    FeatureDefinition(
        name="user_avg_transaction_amount_30d",
        description="Average transaction amount for user in last 30 days",
        dtype="float64",
        source="transactions_db",
        owner="fraud-team",
        freshness_sla="1h",
        value_range=(0.0, 1_000_000.0),
        tags={"domain": "fraud", "pii": False},
    ),
]
```

### C. Train-Serve Parity Test

```python
# tests/features/test_train_serve_parity.py

def test_feature_parity(feature_store, training_pipeline):
    """Feature values MUST be identical between training and serving."""
    # Get features as computed during training
    train_features = training_pipeline.compute_features(sample_entity_ids)

    # Get features as served by the feature store
    serve_features = feature_store.get_features(sample_entity_ids)

    for feature_name in train_features.columns:
        pd.testing.assert_series_equal(
            train_features[feature_name],
            serve_features[feature_name],
            check_names=False,
            rtol=1e-5,  # Allow small floating point differences
            obj=f"Train-serve parity for '{feature_name}'",
        )
```

---

## 8. Model Serving & Deployment (MANDATORY)

### A. Serving Architecture Options

```
SERVING STRATEGY DECISION TREE:

What is the latency requirement?
├── Real-time (<100ms)
│   ├── Low complexity model → REST/gRPC microservice
│   ├── High throughput → Batched inference with queuing
│   └── Edge deployment → Model compiled to ONNX/TFLite
│
├── Near real-time (<1s)
│   └── Streaming inference (Kafka + model service)
│
└── Batch (minutes to hours)
    └── Batch prediction pipeline (Spark, Beam, scheduled jobs)
```

### B. Model Serving Requirements

```
SERVING REQUIREMENTS:

Every served model MUST have:
├── Health check endpoint (/health, /ready)
├── Prediction endpoint with input validation
├── Model version in response headers
├── Request/response logging (without PII)
├── Latency tracking (p50, p90, p99)
├── Error handling with graceful degradation
├── Rate limiting and authentication
├── Fallback to previous model version on failure
├── A/B testing or shadow mode capability
└── Resource limits (CPU, memory, GPU)
```

### C. Model Serving Implementation

```python
# src/serving/app.py

import os
import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Load model at startup
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
FALLBACK_STAGE = "Staging"

model = None
model_version = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model, model_version
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = model.metadata.run_id[:8]
        logger.info(f"Loaded model '{MODEL_NAME}' ({MODEL_STAGE}) version={model_version}")
    except Exception as e:
        logger.error(f"Failed to load {MODEL_STAGE} model: {e}. Trying fallback.")
        model_uri = f"models:/{MODEL_NAME}/{FALLBACK_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = "fallback"
        logger.warning(f"Loaded FALLBACK model '{MODEL_NAME}' ({FALLBACK_STAGE})")
    yield

app = FastAPI(title=f"{MODEL_NAME} serving", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(..., min_length=1)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "version": model_version}

@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    try:
        import pandas as pd
        input_df = pd.DataFrame([request.features])
        prediction = model.predict(input_df)
        elapsed_ms = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=float(prediction[0]),
            model_version=model_version,
            latency_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics endpoint."""
    # In production, use prometheus_client library
    return {"model_name": MODEL_NAME, "model_version": model_version}
```

### D. ML Deployment Strategies

```
ML-SPECIFIC DEPLOYMENT STRATEGIES:

1. Shadow Mode (RECOMMENDED for new models)
   ├── New model runs alongside production model
   ├── Both make predictions, only old model serves traffic
   ├── Compare predictions to detect quality issues
   └── Promote when shadow model proves equal/better

2. A/B Testing (For user-facing models)
   ├── Split traffic between model versions
   ├── Measure business metrics (not just ML metrics)
   ├── Statistical significance required before decision
   └── Automatic rollback if degradation detected

3. Canary (For gradual rollout)
   ├── 5% → 25% → 50% → 100% traffic
   ├── Monitor at each step for quality degradation
   ├── Automatic rollback on metric breach
   └── Minimum observation window at each step

4. Blue-Green (For batch inference)
   ├── New model computes batch predictions
   ├── Validate batch output before swapping
   ├── Instant rollback to old batch results
   └── Downstream consumers switch atomically
```

---

## 9. Model Monitoring & Observability (MANDATORY)

### A. What to Monitor

```
ML MONITORING REQUIREMENTS:

1. DATA MONITORING (Input layer)
   ├── Feature distribution drift (KS test, PSI, Jensen-Shannon)
   ├── Missing value rates
   ├── Schema violations (unexpected types, new categories)
   ├── Data freshness (is feature store up to date?)
   ├── Input volume (prediction request rate)
   └── Data quality score over time

2. MODEL MONITORING (Prediction layer)
   ├── Prediction distribution drift
   ├── Confidence score distribution
   ├── Prediction volume by class/category
   ├── Model accuracy vs. ground truth (when available)
   ├── Calibration (predicted probabilities vs. actual rates)
   └── Feature importance stability

3. OPERATIONAL MONITORING (Infrastructure layer)
   ├── Inference latency (p50, p90, p99)
   ├── Throughput (predictions per second)
   ├── Error rate (failed predictions)
   ├── Resource utilization (CPU, GPU, memory)
   ├── Queue depth (for async inference)
   └── Model loading time

4. BUSINESS MONITORING (Impact layer)
   ├── Business KPI correlation (revenue, conversion, fraud caught)
   ├── User feedback signals (explicit + implicit)
   ├── Cost per prediction
   ├── Alert fatigue rate
   └── Time to detect + time to resolve model issues
```

### B. Drift Detection Implementation

```python
# src/monitoring/drift_detector.py

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats

@dataclass
class DriftReport:
    feature_name: str
    test_statistic: float
    p_value: float
    is_drifted: bool
    severity: str  # "none", "warning", "critical"

def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    method: str = "ks",
    warning_threshold: float = 0.05,
    critical_threshold: float = 0.001,
) -> DriftReport:
    """Detect distribution drift between reference and current data."""
    if method == "ks":
        stat, p_value = stats.ks_2samp(reference, current)
    elif method == "chi2":
        # For categorical features
        from scipy.stats import chi2_contingency
        observed = np.histogram(current, bins=50)[0]
        expected = np.histogram(reference, bins=50)[0]
        stat, p_value, _, _ = chi2_contingency([observed, expected])
    elif method == "psi":
        # Population Stability Index
        stat = _compute_psi(reference, current)
        p_value = 1.0 if stat < 0.1 else (0.05 if stat < 0.2 else 0.001)
    else:
        raise ValueError(f"Unknown drift method: {method}")

    if p_value < critical_threshold:
        severity = "critical"
    elif p_value < warning_threshold:
        severity = "warning"
    else:
        severity = "none"

    return DriftReport(
        feature_name="",  # Set by caller
        test_statistic=stat,
        p_value=p_value,
        is_drifted=p_value < warning_threshold,
        severity=severity,
    )

def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index."""
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)

    # Avoid division by zero
    ref_pct = (ref_hist + 1) / (len(reference) + bins)
    cur_pct = (cur_hist + 1) / (len(current) + bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def run_drift_check(
    reference_data: Dict[str, np.ndarray],
    current_data: Dict[str, np.ndarray],
) -> List[DriftReport]:
    """Run drift detection across all features."""
    reports = []
    for feature_name in reference_data:
        if feature_name not in current_data:
            reports.append(DriftReport(
                feature_name=feature_name,
                test_statistic=float("inf"),
                p_value=0.0,
                is_drifted=True,
                severity="critical",
            ))
            continue

        report = detect_drift(
            reference=reference_data[feature_name],
            current=current_data[feature_name],
        )
        report.feature_name = feature_name
        reports.append(report)

    return reports
```

### C. Monitoring Alerts

```
ML ALERTING RULES:

Severity 1 (Page immediately):
├── Model serving is down (health check fails)
├── Prediction accuracy dropped >10% from baseline
├── Critical data drift on >50% of features
├── Model returning same prediction for all inputs
└── Inference error rate >5%

Severity 2 (Page during business hours):
├── Prediction accuracy dropped >5% from baseline
├── Data drift detected on key features
├── Inference latency p99 >2x baseline
├── Prediction distribution shift detected
└── Feature store staleness >SLA

Severity 3 (Ticket, next business day):
├── Minor drift on non-critical features
├── Model approaching resource limits
├── Ground truth feedback loop delayed
├── Training pipeline failure (non-production)
└── Experiment tracking inconsistencies

Every alert MUST have:
├── Runbook link with diagnosis steps
├── Direct link to monitoring dashboard
├── Model name and version
├── Impact description (which service/users affected)
└── Escalation path
```

### D. Retraining Triggers

```
WHEN TO RETRAIN:

Automatic triggers:
├── Data drift exceeds threshold (PSI > 0.2 on key features)
├── Model accuracy drops below SLA
├── Scheduled retraining (weekly / monthly)
├── New training data volume exceeds threshold
└── Upstream data source schema change

Manual triggers:
├── Business requirements change
├── New features available
├── Model fairness audit findings
├── Regulatory compliance update
└── Significant concept drift detected

NEVER retrain:
├── Without validating new training data first
├── Without comparing against current production model
├── Without running full quality gate pipeline
├── Without notifying the model owner
└── Directly to production (always staging first)
```

---

## 10. Testing for ML Systems (MANDATORY)

### A. ML Test Pyramid

```
                    ┌──────────┐
                    │ End-to-  │  Fewest tests
                    │ End      │  Full pipeline run
                    │ Pipeline │  Data → Train → Evaluate → Serve
                    ├──────────┤
                   │ Integration │  Model + serving
                   │ Tests       │  Feature store + model
                   │             │  Pipeline stage connections
                   ├─────────────┤
                  │ Model Quality  │  Accuracy, F1, latency
                  │ Tests          │  Fairness, bias checks
                  │                │  Overfit detection
                  ├────────────────┤
                 │ Data Validation   │  Schema, ranges, nulls
                 │ Tests             │  Distribution checks
                 │                   │  Leakage detection
                 ├───────────────────┤
                │ Unit Tests           │  Most tests
                │                      │  Feature transforms
                │                      │  Data processing functions
                │                      │  Utility code
                └──────────────────────┘
```

### B. Test Categories

```python
# tests/conftest.py — Shared fixtures for ML tests

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_training_data():
    """Generate synthetic training data for tests."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "feature_a": np.random.normal(0, 1, n),
        "feature_b": np.random.exponential(1, n),
        "label": np.random.binomial(1, 0.3, n),
    })

@pytest.fixture
def trained_model(sample_training_data):
    """Train a model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    X = sample_training_data.drop("label", axis=1)
    y = sample_training_data["label"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model
```

```python
# tests/unit/test_feature_transforms.py

def test_normalize_feature_scales_correctly():
    """Feature normalization must produce zero mean, unit variance."""
    from src.features.transforms import normalize
    import numpy as np

    raw = np.array([10, 20, 30, 40, 50], dtype=float)
    normalized = normalize(raw)

    assert abs(normalized.mean()) < 1e-10
    assert abs(normalized.std() - 1.0) < 1e-10

def test_one_hot_encoding_handles_unknown():
    """Encoder must handle categories not seen during training."""
    from src.features.transforms import safe_one_hot_encode

    train_categories = ["cat", "dog", "bird"]
    encoder = safe_one_hot_encode(train_categories)

    # Unknown category should not crash
    result = encoder.transform(["cat", "fish", "dog"])
    assert result.shape[1] == 3  # Same number of columns as training
```

```python
# tests/integration/test_serving.py

def test_prediction_endpoint(test_client, loaded_model):
    """Prediction endpoint must return valid response."""
    response = test_client.post("/predict", json={
        "features": {"feature_a": 0.5, "feature_b": 1.2}
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model_version" in data
    assert "latency_ms" in data
    assert data["latency_ms"] < 100  # Latency SLA

def test_prediction_with_missing_features(test_client):
    """Endpoint must reject requests with missing features."""
    response = test_client.post("/predict", json={
        "features": {"feature_a": 0.5}  # Missing feature_b
    })
    assert response.status_code == 422  # Validation error

def test_health_endpoint(test_client):
    """Health endpoint must return model info."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

```python
# tests/pipeline/test_end_to_end.py

def test_full_pipeline_produces_registered_model(tmp_path):
    """Complete pipeline must produce a registered, validated model."""
    from src.pipelines.training_pipeline import run_pipeline

    result = run_pipeline(
        data_path="tests/fixtures/sample_data.parquet",
        output_dir=str(tmp_path),
        params={"n_estimators": 10, "max_depth": 3},
    )

    # Pipeline completed
    assert result.status == "success"

    # Model was registered
    assert result.model_version is not None

    # Quality gates passed
    assert result.metrics["accuracy"] >= 0.80
    assert result.metrics["f1_weighted"] >= 0.75

    # Artifacts were produced
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "feature_importance.csv").exists()
```

---

## 11. Security & Governance (MANDATORY)

### A. ML-Specific Security

```
ML SECURITY REQUIREMENTS:

1. Data Security
   ├── Access controls on training data (RBAC)
   ├── PII detection and masking in datasets
   ├── Data encryption at rest and in transit
   ├── Audit logs for all data access
   ├── Data retention and deletion policies
   └── GDPR/CCPA compliance for user data

2. Model Security
   ├── Model artifact signing and verification
   ├── Adversarial robustness testing
   ├── Model extraction attack prevention
   ├── Inference input validation (prevent prompt injection for LLMs)
   ├── Rate limiting on prediction endpoints
   └── Model access audit logging

3. Pipeline Security
   ├── No secrets in pipeline code
   ├── Minimal permissions for pipeline execution
   ├── Supply chain security for ML dependencies
   ├── Container scanning for serving images
   ├── Network isolation for training environments
   └── Signed pipeline artifacts

4. Privacy
   ├── Differential privacy for sensitive models
   ├── Federated learning where data can't be centralized
   ├── Model memorization tests (canary values)
   ├── Membership inference attack tests
   └── Right to explanation compliance
```

### B. ML Governance Framework

```
GOVERNANCE CHECKLIST:

Before deploying any model to production:

□ Model card completed and reviewed
□ Fairness evaluation across protected groups
□ Bias testing with representative datasets
□ Interpretability artifacts generated (SHAP, LIME, feature importance)
□ Data lineage documented (source → features → model)
□ Privacy impact assessment completed
□ Model owner assigned
□ Monitoring and alerting configured
□ Retraining schedule defined
□ Rollback procedure documented and tested
□ Regulatory compliance verified (industry-specific)
□ Human-in-the-loop requirements documented
□ Incident response plan for model failures
```

### C. Responsible AI Testing

```python
# tests/fairness/test_model_fairness.py

def test_demographic_parity(trained_model, test_data_with_demographics):
    """Model predictions must not disproportionately affect protected groups."""
    X_test = test_data_with_demographics.drop(["label", "gender", "age_group"], axis=1)
    predictions = trained_model.predict(X_test)

    for group_col in ["gender", "age_group"]:
        groups = test_data_with_demographics[group_col].unique()
        group_rates = {}
        for group in groups:
            mask = test_data_with_demographics[group_col] == group
            group_rates[group] = predictions[mask].mean()

        # Max disparity between groups
        max_rate = max(group_rates.values())
        min_rate = min(group_rates.values())
        disparity = max_rate - min_rate

        assert disparity < 0.10, (
            f"Demographic parity violation for '{group_col}': "
            f"disparity={disparity:.4f}, rates={group_rates}"
        )

def test_equal_opportunity(trained_model, test_data_with_demographics):
    """True positive rates must be similar across protected groups."""
    X_test = test_data_with_demographics.drop(["label", "gender"], axis=1)
    y_test = test_data_with_demographics["label"]
    predictions = trained_model.predict(X_test)

    for group in test_data_with_demographics["gender"].unique():
        mask = test_data_with_demographics["gender"] == group
        group_positives = y_test[mask] == 1
        if group_positives.sum() == 0:
            continue
        tpr = (predictions[mask][group_positives] == 1).mean()
        assert tpr >= 0.70, (
            f"Equal opportunity violation: TPR for {group} is {tpr:.4f}"
        )
```

---

## 12. Project Structure (MANDATORY)

### A. Standard ML Project Layout

```
ml-project/
├── README.md                     # Project overview, quickstart
├── RUNBOOK.md                    # Operational procedures
├── model_card.md                 # Model documentation
│
├── src/
│   ├── data/                     # Data loading and validation
│   │   ├── loaders.py
│   │   ├── validation.py
│   │   └── schemas.py
│   │
│   ├── features/                 # Feature engineering
│   │   ├── definitions.py        # Feature catalog
│   │   ├── transforms.py         # Feature transformations
│   │   └── store.py              # Feature store client
│   │
│   ├── training/                 # Model training
│   │   ├── train.py              # Training logic
│   │   ├── evaluate.py           # Evaluation logic
│   │   └── hyperopt.py           # Hyperparameter optimization
│   │
│   ├── serving/                  # Model serving
│   │   ├── app.py                # Serving application
│   │   └── preprocessing.py      # Online preprocessing
│   │
│   ├── monitoring/               # Model monitoring
│   │   ├── drift_detector.py
│   │   └── performance_tracker.py
│   │
│   ├── pipelines/                # Pipeline definitions
│   │   ├── training_pipeline.py
│   │   ├── batch_inference.py
│   │   └── retraining_pipeline.py
│   │
│   └── registry/                 # Model registry operations
│       └── register_model.py
│
├── tests/
│   ├── unit/                     # Unit tests
│   │   ├── test_feature_transforms.py
│   │   └── test_data_validation.py
│   ├── data/                     # Data tests
│   │   ├── test_training_data.py
│   │   └── test_schema.py
│   ├── model/                    # Model quality tests
│   │   ├── test_model_quality.py
│   │   └── test_model_fairness.py
│   ├── integration/              # Integration tests
│   │   ├── test_serving.py
│   │   └── test_pipeline.py
│   ├── fairness/                 # Fairness tests
│   │   └── test_model_fairness.py
│   ├── monitoring/               # Monitoring tests
│   │   └── test_data_drift.py
│   └── fixtures/                 # Test data
│       └── sample_data.parquet
│
├── data/                         # Data (DVC-tracked, not in Git)
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/                       # Local model artifacts (DVC-tracked)
│
├── notebooks/                    # Exploration only (NEVER in production)
│   └── exploratory/
│
├── configs/                      # Configuration files
│   ├── params.yaml               # Hyperparameters (DVC params)
│   ├── training_config.yaml
│   └── serving_config.yaml
│
├── pipelines/                    # Pipeline definitions (YAML/config)
│   └── dvc.yaml                  # DVC pipeline DAG
│
├── Dockerfile                    # Serving container
├── docker-compose.yml            # Local development
├── pyproject.toml                # Dependencies (Poetry/PDM)
├── dvc.yaml                      # DVC pipeline definition
├── dvc.lock                      # DVC pipeline lock
├── .dvc/                         # DVC config
├── .gitignore
└── .pre-commit-config.yaml
```

---

## 13. Deployment Checklist

### Pre-Deployment Verification (MANDATORY)

#### Data
- [ ] Training data versioned (DVC, LakeFS, or equivalent)
- [ ] Data validation tests pass (schema, quality, distributions)
- [ ] No data leakage between train/test/validation splits
- [ ] Data lineage documented
- [ ] PII handling verified

#### Model
- [ ] Experiment run fully tracked (params, metrics, artifacts, environment)
- [ ] Model registered in registry with version
- [ ] Quality gates passed (accuracy, F1, latency, size)
- [ ] Model card completed
- [ ] Fairness evaluation completed
- [ ] Interpretability artifacts generated

#### Pipeline
- [ ] Pipeline runs end-to-end without manual steps
- [ ] All pipeline steps tested
- [ ] Reproducibility verified (re-run produces same results)
- [ ] Dependencies pinned (requirements.txt / lockfile)
- [ ] Random seeds set for all stochastic operations

#### Serving
- [ ] Health and readiness endpoints implemented
- [ ] Input validation on prediction endpoint
- [ ] Latency within SLA (tested under load)
- [ ] Fallback to previous model version configured
- [ ] Container image scanned for vulnerabilities
- [ ] Resource limits configured (CPU, memory, GPU)

#### Monitoring
- [ ] Data drift detection active
- [ ] Model performance monitoring active
- [ ] Prediction distribution monitoring active
- [ ] Alerts configured with runbook links
- [ ] Dashboard created
- [ ] Retraining triggers defined

#### Security
- [ ] No hardcoded secrets in code or configs
- [ ] Data access controls verified
- [ ] Model artifact signed
- [ ] Pipeline permissions scoped to minimum
- [ ] Serving endpoint authenticated and rate-limited

#### Governance
- [ ] Model owner assigned
- [ ] Retraining schedule documented
- [ ] Rollback procedure tested
- [ ] Incident response plan documented
- [ ] Regulatory compliance verified

---

## 14. Why This Configuration Works

**Reproducibility**:
- Version-controlled data + code + config = any experiment can be exactly reproduced
- Pinned seeds, dependencies, and environments eliminate "works on my machine" for ML

**Experiment Tracking**:
- Every training run is logged and comparable
- Team can learn from each other's experiments
- No wasted compute re-running forgotten experiments

**Quality Gates**:
- Models must prove their worth before deployment
- Automated checks prevent silent degradation
- Fairness and bias testing prevent harmful models

**Data-Centric Approach**:
- Data validation catches upstream issues before they corrupt models
- Drift detection catches distribution changes before they degrade predictions
- Feature stores eliminate train-serve skew

**Continuous Training**:
- Models are automatically retrained when data changes
- Monitoring triggers retraining when performance degrades
- Pipeline automation reduces time from data to deployment

**Observability**:
- Data, model, operational, and business monitoring catch issues at every layer
- Alerts surface problems before users notice
- Runbooks enable anyone to respond to model incidents

**Governance**:
- Model cards document what models do and their limitations
- Fairness testing prevents biased decisions
- Audit trails satisfy regulatory requirements

---

## 15. Quick Reference

### Common Commands

```bash
# ═══════════════════════════════════════════════════════════════
# DATA VERSIONING (DVC)
# ═══════════════════════════════════════════════════════════════

dvc init                          # Initialize DVC
dvc add data/training.parquet     # Track data file
dvc repro                         # Reproduce pipeline
dvc exp run                       # Run experiment
dvc exp show                      # Show experiment results
dvc metrics diff                  # Compare metrics
dvc params diff                   # Compare parameters
dvc plots diff                    # Compare plots

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT TRACKING (MLflow)
# ═══════════════════════════════════════════════════════════════

mlflow ui                         # Launch tracking UI
mlflow experiments list           # List experiments
mlflow runs list --experiment-id 1  # List runs
mlflow models serve -m "models:/name/Production"  # Serve model

# ═══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════

# Register model (Python)
mlflow.register_model("runs:/RUN_ID/model", "model-name")

# Transition stage (Python)
client.transition_model_version_stage("model-name", version=1, stage="Production")

# ═══════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════

pytest tests/unit/                # Unit tests
pytest tests/data/                # Data validation tests
pytest tests/model/               # Model quality tests
pytest tests/fairness/            # Fairness tests
pytest tests/integration/         # Integration tests
pytest --cov=src tests/           # Coverage report

# ═══════════════════════════════════════════════════════════════
# SERVING
# ═══════════════════════════════════════════════════════════════

docker build -t model-serving .
docker run -p 8080:8080 model-serving
curl localhost:8080/health
curl -X POST localhost:8080/predict -d '{"features": {"a": 1.0}}'
```

### ML Quality Gates Summary

| Gate | Metric | Threshold | When |
|------|--------|-----------|------|
| Data Schema | Validation pass | 100% | Before training |
| Data Quality | Null ratio | <5% | Before training |
| Data Drift | KS test p-value | >0.01 | Before training |
| Class Balance | Minority ratio | >5% | Before training |
| Accuracy | Test accuracy | ≥85% | After training |
| F1 Score | Weighted F1 | ≥80% | After training |
| Latency | P99 inference | ≤50ms | Before deployment |
| Model Size | Serialized size | ≤500MB | Before deployment |
| Fairness | Demographic parity | <10% disparity | Before deployment |
| Overfit Gap | Train-test accuracy gap | <10% | After training |

### ML Monitoring Checklist

```
For every production model:
[ ] Feature drift detection (KS test, PSI)
[ ] Prediction distribution monitoring
[ ] Accuracy tracking (when ground truth available)
[ ] Inference latency monitoring (p50, p90, p99)
[ ] Error rate monitoring
[ ] Input volume monitoring
[ ] /health endpoint (liveness)
[ ] /ready endpoint (readiness)
[ ] /metrics endpoint (Prometheus-compatible)
[ ] Alerts with severity levels and runbook links
[ ] Dashboard with model + data + operational metrics
[ ] Retraining trigger configured
```

### Experiment Tracking Checklist

```
For every training run:
[ ] All hyperparameters logged
[ ] All evaluation metrics logged
[ ] Model artifact saved
[ ] Model signature (input/output schema) logged
[ ] Data version reference logged
[ ] Environment details logged (Python, framework versions)
[ ] Feature importance artifact saved
[ ] Confusion matrix / classification report saved
[ ] Random seeds documented
[ ] Compute cost tracked
```

---

## References

- [Google ML Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml) - Rules of ML
- [MLflow Documentation](https://mlflow.org/docs/latest/) - Experiment tracking, model registry
- [DVC Documentation](https://dvc.org/doc) - Data versioning, ML pipelines
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/) - ML pipeline orchestration
- [ML Test Score (Google)](https://research.google/pubs/pub46555/) - Testing ML systems
- [Model Cards (Google)](https://modelcards.withgoogle.com/about) - Model documentation standard
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Responsible AI Practices (Google)](https://ai.google/responsibility/responsible-ai-practices/)
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Continuous Delivery for ML (CD4ML)](https://martinfowler.com/articles/cd4ml.html)

---

**Last Updated:** 2026-03-15
**Version:** 1.0
**Maintainer:** ML Platform Team


**End of MLOps Engineering Guidelines**

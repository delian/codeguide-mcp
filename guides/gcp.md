# Google Cloud Platform (GCP) Development Guidelines

This document provides mandatory standards for building applications on Google Cloud Platform.

---

**Agent Profile**: The GCP Expert
**Role**: Senior Cloud Architect & Google Cloud Professional
**Objective**: Generate scalable, secure, and cost-effective GCP architectures following Google best practices.
**Tools**: gcloud CLI, Terraform, Cloud Build, Cloud Run, GKE, BigQuery.

---

## 1. Core Philosophies: GCP-FIRST

- **G**lobal: Leverage Google's global infrastructure
- **C**ontainerized: Cloud Run and GKE for workloads
- **P**ay-per-use: Serverless where possible

---

## 2. Project Organization (MANDATORY)

### A. Resource Hierarchy

```
Organization
├── Folders
│   ├── Production
│   │   ├── project-prod-app
│   │   ├── project-prod-data
│   │   └── project-prod-network
│   ├── Staging
│   │   └── project-staging
│   ├── Development
│   │   └── project-dev
│   └── Shared
│       ├── project-shared-vpc
│       └── project-shared-services
└── Billing Accounts
```

### B. Naming Conventions

```bash
# Project IDs: {org}-{env}-{app}-{random}
myorg-prod-api-a1b2
myorg-staging-web-c3d4
myorg-dev-backend-e5f6

# Resources: {project}-{resource}-{purpose}
myorg-prod-api-gcs-uploads
myorg-prod-api-sql-main
myorg-prod-api-run-backend

# Labels (applied to all resources)
environment: production
team: platform
cost-center: engineering
managed-by: terraform
```

---

## 3. IAM and Security (MANDATORY)

### A. Service Accounts

```yaml
# Terraform example
resource "google_service_account" "app_sa" {
  account_id   = "app-service-account"
  display_name = "Application Service Account"
  description  = "Service account for Cloud Run application"
}

# Minimal permissions
resource "google_project_iam_member" "app_sa_roles" {
  for_each = toset([
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectViewer",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

# Workload Identity for GKE
resource "google_service_account_iam_binding" "workload_identity" {
  service_account_id = google_service_account.app_sa.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${var.namespace}/${var.k8s_sa_name}]"
  ]
}
```

### B. Secret Manager

```python
# Python client for Secret Manager
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Access a secret version."""
    client = secretmanager.SecretManagerServiceClient()

    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})

    return response.payload.data.decode("UTF-8")

# Usage
db_password = get_secret("my-project", "db-password")
api_key = get_secret("my-project", "api-key")
```

```yaml
# Terraform: Create secret
resource "google_secret_manager_secret" "db_password" {
  secret_id = "db-password"

  replication {
    auto {}
  }

  labels = {
    environment = var.environment
  }
}

resource "google_secret_manager_secret_version" "db_password_version" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}
```

---

## 4. Cloud Run (MANDATORY)

### A. Service Configuration

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: my-api
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/startup-cpu-boost: "true"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      serviceAccountName: app-service-account@project.iam.gserviceaccount.com
      containers:
        - image: gcr.io/my-project/my-api:latest
          ports:
            - containerPort: 8080
          env:
            - name: PROJECT_ID
              value: my-project
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-password
                  key: latest
          resources:
            limits:
              cpu: "2"
              memory: "2Gi"
          startupProbe:
            httpGet:
              path: /health
            initialDelaySeconds: 0
            periodSeconds: 1
            failureThreshold: 30
          livenessProbe:
            httpGet:
              path: /health
```

### B. Terraform Deployment

```hcl
resource "google_cloud_run_v2_service" "api" {
  name     = "my-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.app_sa.email

    scaling {
      min_instance_count = 1
      max_instance_count = 100
    }

    containers {
      image = "gcr.io/${var.project_id}/my-api:${var.image_tag}"

      ports {
        container_port = 8080
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.db_password.secret_id
            version = "latest"
          }
        }
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }

      startup_probe {
        http_get {
          path = "/health"
        }
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
      }
    }

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow unauthenticated access (for public APIs)
resource "google_cloud_run_v2_service_iam_member" "public" {
  location = google_cloud_run_v2_service.api.location
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
```

---

## 5. Cloud SQL (MANDATORY)

### A. Instance Configuration

```hcl
resource "google_sql_database_instance" "main" {
  name             = "${var.project_id}-sql-main"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = "db-custom-2-4096"
    availability_type = "REGIONAL"
    disk_autoresize   = true
    disk_size         = 50
    disk_type         = "PD_SSD"

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
      start_time                     = "03:00"
      transaction_log_retention_days = 7

      backup_retention_settings {
        retained_backups = 30
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      require_ssl     = true
    }

    maintenance_window {
      day          = 7  # Sunday
      hour         = 3
      update_track = "stable"
    }

    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }

    database_flags {
      name  = "log_connections"
      value = "on"
    }

    database_flags {
      name  = "log_disconnections"
      value = "on"
    }

    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute  = 5
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "main" {
  name     = "myapp"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "app" {
  name     = "app"
  instance = google_sql_database_instance.main.name
  password = random_password.db_password.result
}
```

### B. Connection from Cloud Run

```python
# Using Cloud SQL Python Connector
from google.cloud.sql.connector import Connector
import sqlalchemy

def create_pool():
    connector = Connector()

    def get_conn():
        return connector.connect(
            "project:region:instance",
            "pg8000",
            user="app",
            password=get_secret("my-project", "db-password"),
            db="myapp",
        )

    pool = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=get_conn,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
    )

    return pool

engine = create_pool()
```

---

## 6. Cloud Storage (MANDATORY)

### A. Bucket Configuration

```hcl
resource "google_storage_bucket" "uploads" {
  name     = "${var.project_id}-uploads"
  location = var.region

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  lifecycle_rule {
    condition {
      num_newer_versions = 3
    }
    action {
      type = "Delete"
    }
  }

  cors {
    origin          = ["https://example.com"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  encryption {
    default_kms_key_name = google_kms_crypto_key.bucket_key.id
  }
}
```

### B. Signed URLs

```python
from google.cloud import storage
from datetime import timedelta

def generate_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 15) -> str:
    """Generate a signed URL for uploading or downloading."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="PUT",  # or "GET" for download
        content_type="application/octet-stream",
    )

    return url
```

---

## 7. Pub/Sub (MANDATORY)

### A. Topic and Subscription

```hcl
resource "google_pubsub_topic" "orders" {
  name = "orders"

  message_retention_duration = "86400s"  # 24 hours

  schema_settings {
    schema   = google_pubsub_schema.order_schema.id
    encoding = "JSON"
  }
}

resource "google_pubsub_schema" "order_schema" {
  name       = "order-schema"
  type       = "AVRO"
  definition = file("schemas/order.avsc")
}

resource "google_pubsub_subscription" "orders_processor" {
  name  = "orders-processor"
  topic = google_pubsub_topic.orders.name

  ack_deadline_seconds       = 60
  message_retention_duration = "604800s"  # 7 days
  retain_acked_messages      = false

  expiration_policy {
    ttl = ""  # Never expire
  }

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.orders_dlq.id
    max_delivery_attempts = 5
  }

  push_config {
    push_endpoint = google_cloud_run_v2_service.processor.uri
    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
    }
  }
}
```

### B. Publisher and Subscriber

```python
# Publisher
from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("my-project", "orders")

def publish_order(order: dict):
    data = json.dumps(order).encode("utf-8")
    future = publisher.publish(
        topic_path,
        data,
        order_id=order["id"],
        event_type="order.created"
    )
    return future.result()

# Subscriber (for pull subscription)
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path("my-project", "orders-processor")

def callback(message):
    print(f"Received: {message.data}")
    try:
        process_order(json.loads(message.data))
        message.ack()
    except Exception as e:
        print(f"Error processing: {e}")
        message.nack()

streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

try:
    streaming_pull_future.result(timeout=300)
except TimeoutError:
    streaming_pull_future.cancel()
    streaming_pull_future.result()
```

---

## 8. BigQuery (MANDATORY)

### A. Dataset and Table

```hcl
resource "google_bigquery_dataset" "analytics" {
  dataset_id    = "analytics"
  friendly_name = "Analytics Dataset"
  description   = "Dataset for analytics data"
  location      = var.region

  default_table_expiration_ms     = null
  default_partition_expiration_ms = 7776000000  # 90 days

  labels = {
    environment = var.environment
  }
}

resource "google_bigquery_table" "events" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  table_id   = "events"

  time_partitioning {
    type  = "DAY"
    field = "event_timestamp"
  }

  clustering = ["event_type", "user_id"]

  schema = file("schemas/events.json")

  labels = {
    environment = var.environment
  }
}
```

### B. Querying

```python
from google.cloud import bigquery

client = bigquery.Client()

def query_events(event_type: str, start_date: str, end_date: str):
    query = """
        SELECT
            event_type,
            user_id,
            COUNT(*) as event_count,
            DATE(event_timestamp) as event_date
        FROM `my-project.analytics.events`
        WHERE event_type = @event_type
            AND event_timestamp BETWEEN @start_date AND @end_date
        GROUP BY event_type, user_id, event_date
        ORDER BY event_date DESC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("event_type", "STRING", event_type),
            bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
            bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
        ]
    )

    query_job = client.query(query, job_config=job_config)
    return list(query_job.result())
```

---

## 9. Cloud Build (MANDATORY)

### A. Build Configuration

```yaml
# cloudbuild.yaml
steps:
  # Run tests
  - name: 'python:3.11'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r requirements.txt
        pytest tests/ -v

  # Build container
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/my-api:$COMMIT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/my-api:latest'
      - '.'

  # Push container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/my-api:$COMMIT_SHA']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/my-api:latest']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'my-api'
      - '--image=gcr.io/$PROJECT_ID/my-api:$COMMIT_SHA'
      - '--region=$_REGION'
      - '--platform=managed'

substitutions:
  _REGION: us-central1

options:
  logging: CLOUD_LOGGING_ONLY
```

---

## 10. Monitoring (MANDATORY)

### A. Custom Metrics

```python
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
import time

def write_custom_metric(project_id: str, metric_type: str, value: float, labels: dict):
    client = monitoring_v3.MetricServiceClient()
    project_name = f"projects/{project_id}"

    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric_type}"
    series.metric.labels.update(labels)

    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id

    now = time.time()
    seconds = int(now)
    nanos = int((now - seconds) * 10**9)

    interval = monitoring_v3.TimeInterval({
        "end_time": {"seconds": seconds, "nanos": nanos}
    })

    point = monitoring_v3.Point({
        "interval": interval,
        "value": {"double_value": value}
    })

    series.points = [point]
    client.create_time_series(name=project_name, time_series=[series])

# Usage
write_custom_metric(
    "my-project",
    "orders/processed",
    100.0,
    {"environment": "production", "region": "us-central1"}
)
```

### B. Alerting

```hcl
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "High API Latency"
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run request latency"

    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND metric.type = \"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1000
      duration        = "300s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_PERCENTILE_99"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.slack.name]

  alert_strategy {
    auto_close = "604800s"
  }
}
```

---

## 11. Deployment Checklist

### Security
- [ ] Least privilege IAM roles
- [ ] Secrets in Secret Manager
- [ ] VPC Service Controls enabled
- [ ] Cloud Armor configured

### Reliability
- [ ] Multi-region where needed
- [ ] Backup and recovery tested
- [ ] Health checks configured
- [ ] Auto-scaling configured

### Operations
- [ ] Monitoring dashboards created
- [ ] Alerting configured
- [ ] Logging exported
- [ ] Cost budgets set

---

## 12. Quick Reference

```bash
# gcloud common commands
gcloud auth login
gcloud config set project PROJECT_ID
gcloud run deploy SERVICE --image IMAGE
gcloud sql connect INSTANCE --user USER
gcloud pubsub topics publish TOPIC --message "data"
gcloud builds submit --tag gcr.io/PROJECT/IMAGE

# Cloud Run
gcloud run services list
gcloud run services describe SERVICE
gcloud run services update SERVICE --memory 2Gi

# Cloud SQL
gcloud sql instances list
gcloud sql databases list --instance INSTANCE
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Cloud Team

# Google Cloud Platform (GCP) Development Guidelines
Mandatory standards for building applications on Google Cloud Platform. gcloud CLI, Terraform, Cloud Build, Cloud Run, Cloud Functions, GKE, BigQuery, Firestore, Artifact Registry.

---

**Agent Profile**: The GCP Expert
**Role**: Senior Cloud Architect & Google Cloud Professional
**Objective**: Generate scalable, secure, and cost-effective GCP architectures following Google best practices.
**Tools**: gcloud CLI, Terraform, Cloud Build, Cloud Run, Cloud Functions, GKE, BigQuery, Firestore.

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

# Resources: {project}-{resource}-{purpose}
myorg-prod-api-gcs-uploads
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

```hcl
resource "google_service_account" "app_sa" {
  account_id   = "app-service-account"
  display_name = "Application Service Account"
  description  = "Service account for Cloud Run application"
}

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
```

### B. Secret Manager

```python
from google.cloud import secretmanager

def get_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    """Access a secret version."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")
```

```hcl
resource "google_secret_manager_secret" "db_password" {
  secret_id = "db-password"
  replication { auto {} }
  labels = { environment = var.environment }
}

resource "google_secret_manager_secret_version" "db_password_version" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# Secret with automatic rotation via Pub/Sub notification
resource "google_secret_manager_secret" "rotating_secret" {
  secret_id = "rotating-api-key"
  replication { auto {} }
  rotation {
    rotation_period    = "7776000s"  # 90 days
    next_rotation_time = "2026-04-01T00:00:00Z"
  }
  topics { name = google_pubsub_topic.secret_rotation.id }
}
```

### C. IAM Least-Privilege with Custom Roles

```hcl
resource "google_project_iam_custom_role" "app_custom_role" {
  role_id     = "appCustomRole"
  title       = "Application Custom Role"
  description = "Minimal permissions for the application workload"
  permissions = [
    "storage.objects.get",
    "storage.objects.list",
    "pubsub.topics.publish",
    "cloudsql.instances.connect",
    "secretmanager.versions.access",
  ]
}

resource "google_project_iam_member" "app_custom_binding" {
  project = var.project_id
  role    = google_project_iam_custom_role.app_custom_role.id
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}
```

```bash
# Audit IAM bindings
gcloud projects get-iam-policy PROJECT_ID \
  --flatten="bindings[].members" \
  --format="table(bindings.role, bindings.members)" \
  --filter="bindings.members:serviceAccount"

# IAM Recommender: identify unused permissions
gcloud recommender recommendations list \
  --project=PROJECT_ID --location=global \
  --recommender=google.iam.policy.Recommender
```

### D. Workload Identity Federation

```hcl
# Allow GitHub Actions to authenticate without service account keys
resource "google_iam_workload_identity_pool" "github_pool" {
  workload_identity_pool_id = "github-actions-pool"
  display_name              = "GitHub Actions Pool"
}

resource "google_iam_workload_identity_pool_provider" "github_provider" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github_pool.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository_owner == 'my-org'"
  oidc { issuer_uri = "https://token.actions.githubusercontent.com" }
}

resource "google_service_account_iam_binding" "github_sa_binding" {
  service_account_id = google_service_account.deploy_sa.name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github_pool.name}/attribute.repository/my-org/my-repo"
  ]
}
```

---

## 4. Cloud Run (MANDATORY)

### A. Service Configuration

```yaml
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
      serviceAccountName: app-sa@project.iam.gserviceaccount.com
      containers:
        - image: gcr.io/my-project/my-api:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              cpu: "2"
              memory: "2Gi"
          startupProbe:
            httpGet:
              path: /health
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
      image = "us-docker.pkg.dev/${var.project_id}/repo/my-api:${var.image_tag}"
      ports { container_port = 8080 }
      env { name = "PROJECT_ID"; value = var.project_id }
      env {
        name = "DB_PASSWORD"
        value_source {
          secret_key_ref { secret = "db-password"; version = "latest" }
        }
      }
      resources { limits = { cpu = "2"; memory = "2Gi" } }
      startup_probe { http_get { path = "/health" } }
      liveness_probe { http_get { path = "/health" } }
    }
    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }
  }
  traffic { type = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"; percent = 100 }
}
```

### C. Container and Scaling Best Practices

```bash
# Memory/CPU guidelines:
# - API services:    1-2 vCPU, 512Mi-2Gi, concurrency 80
# - Data processing: 2-4 vCPU, 2Gi-8Gi, concurrency 1
# - ML inference:    4-8 vCPU, 4Gi-32Gi

gcloud run deploy my-api \
  --image us-docker.pkg.dev/my-project/repo/my-api:latest \
  --cpu 2 --memory 2Gi --concurrency 80 \
  --min-instances 1 --max-instances 100 \
  --cpu-boost --no-cpu-throttling --timeout 300 \
  --service-account app-sa@my-project.iam.gserviceaccount.com \
  --region us-central1
```

### D. Cold Start Optimization

```python
# Initialize clients at module level (reused across requests)
from google.cloud import firestore, storage

db = firestore.Client()
storage_client = storage.Client()

def _warmup():
    """Pre-warm connections during container startup."""
    db.collection("health").document("ping").get()
_warmup()
```

```dockerfile
# Multi-stage build for minimal image size
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/deps /app/deps
ENV PYTHONPATH=/app/deps
COPY . .
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### E. Cloud Run Jobs

```hcl
resource "google_cloud_run_v2_job" "data_export" {
  name     = "data-export"
  location = var.region
  template {
    task_count = 10
    template {
      service_account = google_service_account.job_sa.email
      timeout         = "3600s"
      max_retries     = 3
      containers {
        image = "us-docker.pkg.dev/${var.project_id}/repo/data-export:${var.image_tag}"
        resources { limits = { cpu = "4"; memory = "8Gi" } }
      }
    }
  }
}
```

```bash
# Execute and schedule jobs
gcloud run jobs execute data-export --region us-central1

gcloud scheduler jobs create http data-export-nightly \
  --schedule="0 2 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/my-project/jobs/data-export:run" \
  --http-method=POST \
  --oauth-service-account-email=scheduler-sa@my-project.iam.gserviceaccount.com
```

### F. Custom Domains and Load Balancing

Use a Global HTTPS Load Balancer with a serverless NEG to front Cloud Run with custom domains:

```hcl
resource "google_compute_region_network_endpoint_group" "run_neg" {
  name                  = "run-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  cloud_run { service = google_cloud_run_v2_service.api.name }
}

resource "google_compute_backend_service" "run_backend" {
  name     = "run-backend"
  protocol = "HTTPS"
  backend { group = google_compute_region_network_endpoint_group.run_neg.id }
}
# Then create url_map -> target_https_proxy -> global_forwarding_rule
# with a google_compute_managed_ssl_certificate for the domain.
```

### G. VPC Connectors and Direct VPC Egress

```hcl
# Traditional VPC connector
resource "google_vpc_access_connector" "connector" {
  name          = "run-vpc-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = google_compute_network.vpc.name
  min_throughput = 200; max_throughput = 1000
}

# Direct VPC Egress (preferred -- no connector, better performance)
# Use vpc_access.network_interfaces instead of vpc_access.connector
# in google_cloud_run_v2_service template block.
```

### H. Health Checks

```python
from fastapi import FastAPI, Response
import time

app = FastAPI()
startup_time = time.time()

@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}

@app.get("/ready")
async def readiness():
    """Readiness probe -- checks downstream dependencies."""
    try:
        db.collection("health").document("check").get()
        return {"status": "ready", "uptime": time.time() - startup_time}
    except Exception as e:
        return Response(status_code=503, content=f"Dependency check failed: {e}")
```

---

## 5. Cloud Functions v2 (MANDATORY)

### A. HTTP Triggers

```python
import functions_framework
import json

@functions_framework.http
def handle_webhook(request):
    """HTTP-triggered Cloud Function."""
    if request.method != "POST":
        return ("Method not allowed", 405)
    payload = request.get_json(silent=True)
    if not payload:
        return ("Bad request", 400)
    result = process_webhook(payload)
    return (json.dumps(result), 200, {"Content-Type": "application/json"})
```

### B. Eventarc Triggers (CloudEvent Format)

```python
import functions_framework
from cloudevents.http import CloudEvent

@functions_framework.cloud_event
def handle_storage_event(cloud_event: CloudEvent):
    """Triggered by Cloud Storage finalize event via Eventarc."""
    data = cloud_event.data
    bucket, name = data["bucket"], data["name"]
    print(f"Processing file: gs://{bucket}/{name}")
    if name.endswith(".csv"):
        process_csv(bucket, name)

@functions_framework.cloud_event
def handle_pubsub_event(cloud_event: CloudEvent):
    """Triggered by Pub/Sub message via Eventarc."""
    import base64
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    process_event(json.loads(message_data))
```

### C. Deployment

```bash
# HTTP function
gcloud functions deploy handle-webhook \
  --gen2 --runtime python312 --trigger-http --allow-unauthenticated \
  --region us-central1 --memory 512Mi --cpu 1 --timeout 60s \
  --min-instances 0 --max-instances 100 --concurrency 80 \
  --set-secrets "DB_PASSWORD=db-password:latest" --source .

# Eventarc: Cloud Storage trigger
gcloud functions deploy process-uploads \
  --gen2 --runtime python312 \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=my-project-uploads" \
  --region us-central1 --memory 1Gi --timeout 540s --source .

# Eventarc: Pub/Sub trigger
gcloud functions deploy process-orders \
  --gen2 --runtime python312 --trigger-topic=orders \
  --region us-central1 --memory 512Mi --min-instances 1 --concurrency 1 --source .
```

### D. Terraform

```hcl
resource "google_cloudfunctions2_function" "webhook" {
  name     = "handle-webhook"
  location = var.region
  build_config {
    runtime     = "python312"
    entry_point = "handle_webhook"
    source {
      storage_source {
        bucket = google_storage_bucket.functions_source.name
        object = google_storage_bucket_object.function_zip.name
      }
    }
  }
  service_config {
    max_instance_count               = 100
    available_memory                 = "512Mi"
    timeout_seconds                  = 60
    max_instance_request_concurrency = 80
    service_account_email            = google_service_account.func_sa.email
    secret_environment_variables {
      key = "DB_PASSWORD"; project_id = var.project_id
      secret = "db-password"; version = "latest"
    }
  }
}

# For Eventarc-triggered functions, add an event_trigger block:
#   event_trigger {
#     event_type = "google.cloud.storage.object.v1.finalized"
#     retry_policy = "RETRY_POLICY_RETRY"
#     event_filters { attribute = "bucket"; value = google_storage_bucket.uploads.name }
#   }
```

### E. Local Testing

```bash
pip install functions-framework

# HTTP function
functions-framework --target=handle_webhook --debug --port=8080
curl -X POST http://localhost:8080 -H "Content-Type: application/json" \
  -d '{"event": "order.created", "data": {"id": "123"}}'

# CloudEvent function (use --signature-type=cloudevent and ce-* headers)
functions-framework --target=handle_storage_event --signature-type=cloudevent --debug
```

---

## 6. Firestore (MANDATORY)

### A. Document and Collection Design

```
# Design principles:
# 1. Denormalize for read performance
# 2. Limit document size to 1 MiB
# 3. Prefer 2-3 levels of subcollection depth
# 4. Use document IDs that distribute writes evenly
#
# users/{userId}
#   ├── name, email, createdAt
#   └── orders/ (subcollection)
#       └── {orderId} -> total, status, items[]
#
# products/{productId}
#   ├── name, price, category
#   └── reviews/ (subcollection)
#       └── {reviewId} -> userId, rating, text
```

### B. Subcollections vs Root Collections

```python
from google.cloud import firestore
db = firestore.Client()

# Subcollection: data owned by a parent
def create_order(user_id: str, order_data: dict) -> str:
    ref = db.collection("users").document(user_id).collection("orders").document()
    order_data["createdAt"] = firestore.SERVER_TIMESTAMP
    ref.set(order_data)
    return ref.id

# Root collection: data needs cross-parent queries
def get_recent_orders(limit: int = 50):
    return db.collection("orders") \
             .order_by("createdAt", direction=firestore.Query.DESCENDING) \
             .limit(limit).stream()

# Denormalize pattern: write to both for flexible querying
def create_order_denormalized(user_id: str, order_data: dict) -> str:
    batch = db.batch()
    root_ref = db.collection("orders").document()
    order_data["createdAt"] = firestore.SERVER_TIMESTAMP
    order_data["userId"] = user_id
    batch.set(root_ref, order_data)
    user_ref = db.collection("users").document(user_id).collection("orders").document(root_ref.id)
    batch.set(user_ref, order_data)
    batch.commit()
    return root_ref.id
```

### C. Composite Indexes

```json
{
  "indexes": [
    {
      "collectionGroup": "orders",
      "queryScope": "COLLECTION",
      "fields": [
        { "fieldPath": "status", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "DESCENDING" }
      ]
    },
    {
      "collectionGroup": "orders",
      "queryScope": "COLLECTION_GROUP",
      "fields": [
        { "fieldPath": "userId", "order": "ASCENDING" },
        { "fieldPath": "createdAt", "order": "DESCENDING" }
      ]
    }
  ]
}
```

### D. Security Rules

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    function isAuth() { return request.auth != null; }
    function isOwner(uid) { return request.auth.uid == uid; }
    function isAdmin() { return request.auth.token.admin == true; }

    match /users/{userId} {
      allow read, create: if isAuth() && isOwner(userId);
      allow update: if isAuth() && isOwner(userId)
        && !request.resource.data.diff(resource.data).affectedKeys().hasAny(['createdAt', 'role']);
      match /orders/{orderId} {
        allow read: if isAuth() && isOwner(userId);
        allow create: if isAuth() && isOwner(userId) && request.resource.data.total > 0;
        allow delete: if false;
      }
    }
    match /products/{productId} {
      allow read: if true;
      allow write: if isAuth() && isAdmin();
    }
    match /{document=**} { allow read, write: if false; }
  }
}
```

### E. Real-time Listeners

```python
# Server-side real-time listener
def on_snapshot(doc_snapshot, changes, read_time):
    for change in changes:
        if change.type.name == "ADDED":
            print(f"New: {change.document.id}")
        elif change.type.name == "MODIFIED":
            print(f"Updated: {change.document.id}")

query = db.collection("orders").where("status", "==", "pending")
doc_watch = query.on_snapshot(on_snapshot)
```

```javascript
// Client-side with offline persistence
import { getFirestore, enableIndexedDbPersistence, collection, query, where, onSnapshot } from "firebase/firestore";

const db = getFirestore(app);
enableIndexedDbPersistence(db).catch(console.warn);

const q = query(collection(db, "orders"), where("status", "==", "pending"));
const unsubscribe = onSnapshot(q, { includeMetadataChanges: true }, (snapshot) => {
  snapshot.docChanges().forEach((change) => {
    const source = snapshot.metadata.fromCache ? "cache" : "server";
    console.log(`[${source}] ${change.type}: ${change.doc.id}`);
  });
});
```

### F. Batch Operations and Transactions

```python
# Batch writes (up to 500 operations per batch)
def bulk_update_status(order_ids: list[str], new_status: str):
    BATCH_SIZE = 500
    for i in range(0, len(order_ids), BATCH_SIZE):
        batch = db.batch()
        for order_id in order_ids[i:i + BATCH_SIZE]:
            ref = db.collection("orders").document(order_id)
            batch.update(ref, {"status": new_status, "updatedAt": firestore.SERVER_TIMESTAMP})
        batch.commit()

# Transactions (atomic read-then-write)
@firestore.transactional
def transfer_credits(transaction, from_id: str, to_id: str, amount: int):
    from_ref = db.collection("users").document(from_id)
    to_ref = db.collection("users").document(to_id)
    from_snap = from_ref.get(transaction=transaction)
    to_snap = to_ref.get(transaction=transaction)
    if from_snap.get("credits") < amount:
        raise ValueError("Insufficient credits")
    transaction.update(from_ref, {"credits": from_snap.get("credits") - amount})
    transaction.update(to_ref, {"credits": to_snap.get("credits") + amount})

transfer_credits(db.transaction(), "user-123", "user-456", 100)
```

---

## 7. Cloud SQL (MANDATORY)

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
      backup_retention_settings { retained_backups = 30 }
    }
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      require_ssl     = true
    }
    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute  = 5
      record_application_tags = true
    }
  }
  deletion_protection = true
}
```

### B. Connection from Cloud Run

```python
from google.cloud.sql.connector import Connector
import sqlalchemy

def create_pool():
    connector = Connector()
    def get_conn():
        return connector.connect(
            "project:region:instance", "pg8000",
            user="app", password=get_secret("my-project", "db-password"), db="myapp",
        )
    return sqlalchemy.create_engine(
        "postgresql+pg8000://", creator=get_conn,
        pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800,
    )
```

---

## 8. Cloud Storage (MANDATORY)

### A. Bucket Configuration

```hcl
resource "google_storage_bucket" "uploads" {
  name                        = "${var.project_id}-uploads"
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  versioning { enabled = true }

  lifecycle_rule {
    condition { age = 30 }
    action { type = "SetStorageClass"; storage_class = "NEARLINE" }
  }
  lifecycle_rule {
    condition { age = 365 }
    action { type = "SetStorageClass"; storage_class = "COLDLINE" }
  }
  lifecycle_rule {
    condition { num_newer_versions = 3 }
    action { type = "Delete" }
  }
  lifecycle_rule {
    condition { age = 7; matches_prefix = ["tmp/"] }
    action { type = "Delete" }
  }
  lifecycle_rule {
    condition { age = 1 }
    action { type = "AbortIncompleteMultipartUpload" }
  }

  cors {
    origin          = ["https://example.com"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
  encryption { default_kms_key_name = google_kms_crypto_key.bucket_key.id }
  soft_delete_policy { retention_duration_seconds = 604800 }
}
```

### B. Signed URLs

```python
from google.cloud import storage
from datetime import timedelta

def generate_signed_url(bucket_name: str, blob_name: str, method: str = "GET", minutes: int = 15) -> str:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.generate_signed_url(version="v4", expiration=timedelta(minutes=minutes), method=method)
```

### C. Event Notifications

```hcl
resource "google_storage_notification" "upload_notification" {
  bucket         = google_storage_bucket.uploads.name
  payload_format = "JSON_API_V1"
  topic          = google_pubsub_topic.storage_events.id
  event_types    = ["OBJECT_FINALIZE", "OBJECT_DELETE"]
  depends_on     = [google_pubsub_topic_iam_member.storage_publisher]
}

data "google_storage_project_service_account" "gcs_account" {}

resource "google_pubsub_topic_iam_member" "storage_publisher" {
  topic  = google_pubsub_topic.storage_events.id
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:${data.google_storage_project_service_account.gcs_account.email_address}"
}
```

---

## 9. Pub/Sub (MANDATORY)

### A. Topic and Subscription

```hcl
resource "google_pubsub_topic" "orders" {
  name                       = "orders"
  message_retention_duration = "86400s"
  schema_settings {
    schema   = google_pubsub_schema.order_schema.id
    encoding = "JSON"
  }
}

resource "google_pubsub_subscription" "orders_processor" {
  name                       = "orders-processor"
  topic                      = google_pubsub_topic.orders.name
  ack_deadline_seconds       = 60
  message_retention_duration = "604800s"
  expiration_policy { ttl = "" }
  retry_policy { minimum_backoff = "10s"; maximum_backoff = "600s" }
  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.orders_dlq.id
    max_delivery_attempts = 5
  }
  push_config {
    push_endpoint = google_cloud_run_v2_service.processor.uri
    oidc_token { service_account_email = google_service_account.pubsub_invoker.email }
  }
}
```

### B. Publisher and Subscriber

```python
from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("my-project", "orders")

def publish_order(order: dict):
    data = json.dumps(order).encode("utf-8")
    future = publisher.publish(topic_path, data, order_id=order["id"], event_type="order.created")
    return future.result()

# Pull subscriber
subscriber = pubsub_v1.SubscriberClient()
sub_path = subscriber.subscription_path("my-project", "orders-processor")

def callback(message):
    try:
        process_order(json.loads(message.data))
        message.ack()
    except Exception:
        message.nack()

streaming_pull = subscriber.subscribe(sub_path, callback=callback)
```

### C. Ordering and Exactly-Once Delivery

```hcl
resource "google_pubsub_subscription" "orders_ordered" {
  name                         = "orders-ordered"
  topic                        = google_pubsub_topic.orders.name
  enable_message_ordering      = true
  enable_exactly_once_delivery = true
}
```

```python
from google.cloud.pubsub_v1.types import PublisherOptions

publisher = pubsub_v1.PublisherClient(
    publisher_options=PublisherOptions(enable_message_ordering=True)
)

def publish_ordered(order: dict):
    """Messages with same ordering_key are delivered in order."""
    data = json.dumps(order).encode("utf-8")
    return publisher.publish(
        topic_path, data,
        ordering_key=order["customer_id"],
    ).result()
```

### D. Dead-Letter Topics

```hcl
resource "google_pubsub_topic" "orders_dlq" {
  name                       = "orders-dlq"
  message_retention_duration = "604800s"
}

resource "google_pubsub_subscription" "orders_dlq_sub" {
  name  = "orders-dlq-monitor"
  topic = google_pubsub_topic.orders_dlq.name
  ack_deadline_seconds = 600
  # No dead-letter on the DLQ -- avoid infinite loops
}
# Grant service-PROJECT_NUMBER@gcp-sa-pubsub.iam.gserviceaccount.com
# roles/pubsub.publisher on the DLQ topic for forwarding to work.
```

---

## 10. BigQuery (MANDATORY)

### A. Dataset and Table

```hcl
resource "google_bigquery_dataset" "analytics" {
  dataset_id                     = "analytics"
  location                       = var.region
  default_partition_expiration_ms = 7776000000  # 90 days
  labels = { environment = var.environment }
}

resource "google_bigquery_table" "events" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  table_id   = "events"
  time_partitioning { type = "DAY"; field = "event_timestamp" }
  clustering = ["event_type", "user_id"]
  schema     = file("schemas/events.json")
}
```

### B. Querying

```python
from google.cloud import bigquery
client = bigquery.Client()

def query_events(event_type: str, start_date: str, end_date: str):
    query = """
        SELECT event_type, user_id, COUNT(*) as event_count, DATE(event_timestamp) as event_date
        FROM `my-project.analytics.events`
        WHERE event_type = @event_type
            AND event_timestamp BETWEEN @start_date AND @end_date
        GROUP BY event_type, user_id, event_date
        ORDER BY event_date DESC
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("event_type", "STRING", event_type),
        bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
        bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date),
    ])
    return list(client.query(query, job_config=job_config).result())
```

### C. Partitioning Strategies

```sql
-- ALWAYS filter on partition column to avoid full-table scans
SELECT * FROM `project.analytics.events`
WHERE event_timestamp >= '2026-01-01' AND event_timestamp < '2026-02-01'
  AND event_type = 'purchase';

-- Use _PARTITIONTIME for ingestion-time partitioned tables
SELECT * FROM `project.analytics.raw_logs`
WHERE _PARTITIONTIME >= '2026-01-01' AND _PARTITIONTIME < '2026-02-01';
```

### D. Materialized Views and Scheduled Queries

```hcl
resource "google_bigquery_table" "daily_summary_mv" {
  dataset_id = google_bigquery_dataset.analytics.dataset_id
  table_id   = "daily_event_summary_mv"
  materialized_view {
    query = <<-SQL
      SELECT DATE(event_timestamp) AS event_date, event_type,
             COUNT(*) AS total_events, COUNT(DISTINCT user_id) AS unique_users
      FROM `${var.project_id}.analytics.events`
      GROUP BY event_date, event_type
    SQL
    enable_refresh      = true
    refresh_interval_ms = 3600000  # 1 hour
  }
}

# Scheduled queries use google_bigquery_data_transfer_config with
# data_source_id = "scheduled_query", a cron schedule, and a SQL INSERT query.
```

---

## 11. Artifact Registry (MANDATORY)

### A. Docker Repository

```hcl
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "docker-images"
  format        = "DOCKER"
  docker_config { immutable_tags = true }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition { tag_state = "UNTAGGED"; older_than = "604800s" }
  }
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions { keep_count = 10 }
  }
}
```

```bash
# Authenticate, build, push, scan
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t us-central1-docker.pkg.dev/my-project/docker-images/my-api:v1.0.0 .
docker push us-central1-docker.pkg.dev/my-project/docker-images/my-api:v1.0.0
gcloud artifacts docker images scan us-central1-docker.pkg.dev/my-project/docker-images/my-api:v1.0.0 --remote
```

### B. Language-specific Repositories

Artifact Registry supports NPM, Python, Maven, Go, and Apt formats. Use the same `google_artifact_registry_repository` resource with `format = "NPM"`, `"PYTHON"`, or `"MAVEN"`.

```bash
# Configure npm/Python access
gcloud artifacts print-settings npm --project=my-project --repository=npm-packages --location=us-central1
gcloud artifacts print-settings python --project=my-project --repository=python-packages --location=us-central1

# Python: install from private repo
pip install --index-url https://us-central1-python.pkg.dev/my-project/python-packages/simple/ my-package
```

### C. IAM for Artifact Access

```hcl
# Cloud Build: writer, Cloud Run: reader
resource "google_artifact_registry_repository_iam_member" "cloud_build_writer" {
  location   = google_artifact_registry_repository.docker_repo.location
  repository = google_artifact_registry_repository.docker_repo.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${data.google_project.current.number}@cloudbuild.gserviceaccount.com"
}
```

---

## 12. Cloud Build CI/CD (MANDATORY)

### A. Basic Pipeline

```yaml
# cloudbuild.yaml
steps:
  - name: 'python:3.12'
    entrypoint: 'bash'
    args: ['-c', 'pip install -r requirements.txt && pytest tests/ -v']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${_AR_REPO}/my-api:$COMMIT_SHA', '-t', '${_AR_REPO}/my-api:latest', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', '${_AR_REPO}/my-api']

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args: ['run', 'deploy', 'my-api', '--image=${_AR_REPO}/my-api:$COMMIT_SHA', '--region=$_REGION']

substitutions:
  _REGION: us-central1
  _AR_REPO: us-central1-docker.pkg.dev/$PROJECT_ID/docker-images
options:
  logging: CLOUD_LOGGING_ONLY
```

### B. Staging-then-Production Pattern

Use `--tag=staging --no-traffic` to deploy without serving traffic, run integration tests, then promote with `update-traffic --to-latest`. Include vulnerability scanning with `gcloud artifacts docker images scan` before promotion.

### C. Triggers

```hcl
resource "google_cloudbuild_trigger" "main_push" {
  name     = "main-push-deploy"
  location = var.region
  github {
    owner = var.github_owner; name = var.github_repo
    push { branch = "^main$" }
  }
  filename        = "cloudbuild.yaml"
  service_account = google_service_account.cloud_build_sa.id
}
```

---

## 13. Cloud Logging (MANDATORY)

### A. Structured Logging

```python
import json

def log_structured(severity: str, message: str, **kwargs):
    """Structured JSON logs are auto-parsed by Cloud Logging."""
    print(json.dumps({"severity": severity, "message": message, **kwargs}), flush=True)

log_structured("INFO", "Order processed", orderId="order-123", duration_ms=42)
log_structured("ERROR", "Payment failed", orderId="order-789", error="card_declined")

def log_with_trace(request, severity: str, message: str, **kwargs):
    """Include X-Cloud-Trace-Context for request correlation."""
    trace = request.headers.get("X-Cloud-Trace-Context", "").split("/")[0]
    entry = {"severity": severity, "message": message, **kwargs}
    if trace:
        entry["logging.googleapis.com/trace"] = f"projects/my-project/traces/{trace}"
    print(json.dumps(entry), flush=True)
```

### B. Log-based Metrics and Sinks

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND severity>=ERROR' --limit=50 --format=json
gcloud logging metrics create payment-failures --log-filter='jsonPayload.message="Payment failed"'
```

```hcl
resource "google_logging_metric" "error_count" {
  name   = "cloud-run-errors"
  filter = "resource.type=\"cloud_run_revision\" AND severity>=ERROR"
  metric_descriptor { metric_kind = "DELTA"; value_type = "INT64"; unit = "1" }
}

# Log sink to BigQuery for long-term analysis
resource "google_logging_project_sink" "bq_sink" {
  name                   = "bigquery-audit-sink"
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit_logs.dataset_id}"
  filter                 = "resource.type=\"cloud_run_revision\" OR resource.type=\"cloudsql_database\""
  unique_writer_identity = true
  bigquery_options { use_partitioned_tables = true }
}
```

---

## 14. Monitoring and Alerting (MANDATORY)

### A. Custom Metrics

```python
from google.cloud import monitoring_v3
import time

def write_custom_metric(project_id: str, metric_type: str, value: float, labels: dict):
    client = monitoring_v3.MetricServiceClient()
    series = monitoring_v3.TimeSeries()
    series.metric.type = f"custom.googleapis.com/{metric_type}"
    series.metric.labels.update(labels)
    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id
    now = time.time()
    point = monitoring_v3.Point({
        "interval": {"end_time": {"seconds": int(now)}},
        "value": {"double_value": value}
    })
    series.points = [point]
    client.create_time_series(name=f"projects/{project_id}", time_series=[series])
```

### B. Alerting and Uptime Checks

```hcl
resource "google_monitoring_alert_policy" "high_latency" {
  display_name = "High API Latency"
  combiner     = "OR"
  conditions {
    display_name = "Cloud Run P99 latency"
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND metric.type = \"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 1000
      duration        = "300s"
      aggregations { alignment_period = "60s"; per_series_aligner = "ALIGN_PERCENTILE_99" }
    }
  }
  notification_channels = [google_monitoring_notification_channel.slack.name]
}

resource "google_monitoring_uptime_check_config" "api_health" {
  display_name = "API Health Check"
  timeout = "10s"; period = "60s"
  http_check { path = "/health"; port = 443; use_ssl = true }
  monitored_resource {
    type   = "uptime_url"
    labels = { project_id = var.project_id; host = "api.example.com" }
  }
}
```

---

## 15. Terraform for GCP (MANDATORY)

### A. Provider Configuration

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    google      = { source = "hashicorp/google"; version = "~> 5.0" }
    google-beta = { source = "hashicorp/google-beta"; version = "~> 5.0" }
  }
  backend "gcs" { bucket = "myorg-terraform-state"; prefix = "prod/app" }
}

provider "google" { project = var.project_id; region = var.region }
provider "google-beta" { project = var.project_id; region = var.region }
```

### B. Module Structure

```
# infra/
# ├── modules/
# │   ├── cloud-run/     (main.tf, variables.tf, outputs.tf)
# │   ├── cloud-sql/
# │   └── networking/
# ├── environments/
# │   ├── dev/           (main.tf, terraform.tfvars, backend.tf)
# │   ├── staging/
# │   └── prod/
# └── global/iam/
```

```hcl
# environments/prod/main.tf
module "network" {
  source = "../../modules/networking"
  project_id = var.project_id; region = var.region; environment = "prod"
}
module "database" {
  source = "../../modules/cloud-sql"
  project_id = var.project_id; region = var.region
  vpc_id = module.network.vpc_id; tier = "db-custom-4-16384"
}
module "api" {
  source = "../../modules/cloud-run"
  project_id = var.project_id; region = var.region
  image_tag = var.image_tag; vpc_connector_id = module.network.vpc_connector_id
  min_instances = 2; max_instances = 100
}
```

### C. Enable APIs

```hcl
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com", "cloudfunctions.googleapis.com", "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com", "secretmanager.googleapis.com", "sqladmin.googleapis.com",
    "pubsub.googleapis.com", "bigquery.googleapis.com", "firestore.googleapis.com",
    "monitoring.googleapis.com", "logging.googleapis.com", "vpcaccess.googleapis.com",
    "compute.googleapis.com", "iam.googleapis.com",
  ])
  project = var.project_id
  service = each.key
  disable_dependent_services = false
  disable_on_destroy         = false
}
```

---

## 16. Deployment Checklist

### Security
- [ ] Least privilege IAM roles (use custom roles when predefined are too broad)
- [ ] Secrets in Secret Manager (never in env vars or source)
- [ ] VPC Service Controls enabled
- [ ] Cloud Armor for public endpoints
- [ ] Workload Identity Federation for CI/CD (no long-lived keys)
- [ ] Artifact Registry vulnerability scanning enabled
- [ ] Firestore security rules reviewed and tested
- [ ] No service accounts with owner/editor roles

### Reliability
- [ ] Multi-region where needed
- [ ] Backup and recovery tested
- [ ] Health checks configured (startup, liveness, readiness)
- [ ] Auto-scaling configured (min and max instances)
- [ ] Cloud Run cold start optimized (startup CPU boost, min instances)
- [ ] Pub/Sub dead-letter topics configured
- [ ] Cloud SQL high availability enabled

### Operations
- [ ] Monitoring dashboards created
- [ ] Alerting configured (latency, errors, uptime)
- [ ] Structured logging implemented
- [ ] Log sinks to BigQuery for long-term analysis
- [ ] Cost budgets and alerts set
- [ ] Terraform state in GCS with versioning

### Data
- [ ] BigQuery tables partitioned and clustered
- [ ] Firestore indexes deployed
- [ ] Cloud Storage lifecycle rules configured
- [ ] Backup retention policies defined

---

## 17. Quick Reference

```bash
gcloud auth login                                            # Authenticate
gcloud config set project PROJECT_ID                         # Set project
gcloud run deploy SVC --image IMG --region REGION            # Deploy Cloud Run
gcloud run jobs execute JOB                                  # Run a job
gcloud functions deploy FN --gen2 --runtime python312        # Deploy function
gcloud sql connect INST --user USER                          # Connect to SQL
gcloud auth configure-docker REGION-docker.pkg.dev           # Docker auth
gcloud artifacts docker images scan IMG --remote             # Vuln scan
gcloud pubsub topics publish TOPIC --message "data"          # Publish message
bq query --use_legacy_sql=false 'SELECT ...'                 # BQ query
gcloud builds submit --config=cloudbuild.yaml                # Cloud Build
gcloud secrets versions access latest --secret=SECRET_ID     # Get secret
gcloud logging read 'severity>=ERROR' --limit=50 --format=json  # Read logs
gcloud projects get-iam-policy PROJECT_ID                    # IAM audit
```

---

**Last Updated:** 2026-02-27
**Version:** 2.0
**Maintainer:** Cloud Team


**End of Google Cloud Platform (GCP) Development Guidelines**

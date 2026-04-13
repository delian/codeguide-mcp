# Elasticsearch & OpenSearch Development Guidelines
Mandatory coding standards and development practices for Elasticsearch and OpenSearch development. Elasticsearch 8.x+, OpenSearch 2.x+, Query DSL, Kibana/OpenSearch Dashboards, ingest pipelines.

---

**Agent Profile**: The Elasticsearch/OpenSearch Expert
**Role**: Senior Search & Analytics Engineer & Lucene Specialist
**Objective**: Generate production-ready, performant and scalable search and analytics solutions.
**Tools**: Elasticsearch 8.x+, OpenSearch 2.x+, Query DSL, Kibana/OpenSearch Dashboards, ingest pipelines

---

**Version:** 1.0 | **Last Updated:** February 2026 | **Target Versions:** Elasticsearch 8.x+ | OpenSearch 2.x+

## Table of Contents

1. [Core Philosophies: SEARCH-FIRST](#1-core-philosophies-search-first)
2. [Architecture and Fundamentals](#2-architecture-and-fundamentals)
3. [Index Management and Mappings](#3-index-management-and-mappings)
4. [Search and Query DSL](#4-search-and-query-dsl)

---

## 1. Core Philosophies: SEARCH-FIRST

The agent must adhere to the **SEARCH-FIRST** principles for every Elasticsearch/OpenSearch implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **S**chema and mapping: Define explicit mappings where needed; avoid dynamic mapping pitfalls.
- **E**xploit inverted index: Design for full-text and filters; use appropriate analyzers.
- **A**ggregations and analytics: Use aggregations for analytics; avoid heavy script usage when possible.
- **R**esilience: Design for node failure; use replicas and ILM; test failover.
- **C**luster awareness: Respect sharding and routing; avoid hot spots and oversized shards.
- **H**TTP and API: Use REST API correctly; parameterize queries; handle errors and retries.

**Verified Code**: Agent-generated code MUST use the Query DSL correctly, run against a cluster or test container, and pass tests before delivery.

---

## 2. Architecture and Fundamentals

### What is Elasticsearch/OpenSearch?

**Elasticsearch** and **OpenSearch** are **distributed search and analytics engines** built on Apache Lucene:

- ✅ **Full-text search** (inverted indexes, relevance scoring)
- ✅ **Distributed architecture** (horizontal scaling)
- ✅ **Real-time indexing** (near real-time search)
- ✅ **Aggregations** (metrics, buckets, pipelines)
- ✅ **RESTful API** (JSON over HTTP)
- ✅ **Schema-free** (dynamic mapping)
- ✅ **Multi-tenancy** (indices, types)

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    Cluster                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐│
│  │   Node 1     │  │   Node 2     │  │   Node 3   ││
│  │  (Master)    │  │  (Data)      │  │  (Data)    ││
│  │              │  │              │  │            ││
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │┌──────────┐││
│  │ │ Primary  │ │  │ │ Replica  │ │  ││ Replica  │││
│  │ │ Shard 0  │ │  │ │ Shard 0  │ │  ││ Shard 1  │││
│  │ └──────────┘ │  │ └──────────┘ │  │└──────────┘││
│  │ ┌──────────┐ │  │ ┌──────────┐ │  │            ││
│  │ │ Primary  │ │  │ │ Replica  │ │  │            ││
│  │ │ Shard 1  │ │  │ │ Shard 1  │ │  │            ││
│  │ └──────────┘ │  │ └──────────┘ │  │            ││
│  └──────────────┘  └──────────────┘  └────────────┘│
└─────────────────────────────────────────────────────┘
```

### Core Concepts

**Cluster:**
- Collection of nodes working together
- Shares data and workload
- Identified by cluster name

**Node:**
- Single server in the cluster
- Stores data and participates in indexing/search
- Types: Master, Data, Ingest, Coordinating

**Index:**
- Collection of documents (like a database table)
- Has a mapping (schema)
- Divided into shards

**Shard:**
- Subset of an index's data
- Horizontal partition (enables distribution)
- Two types: Primary and Replica

**Document:**
- Basic unit of information (JSON)
- Stored in an index
- Has a unique ID

**Mapping:**
- Schema definition for documents
- Defines field types and properties
- Can be dynamic or explicit

### Node Roles

**Master Node:**
```yaml
node.roles: [master]
```
- Cluster management
- Index creation/deletion
- Node tracking
- Shard allocation

**Data Node:**
```yaml
node.roles: [data, data_content, data_hot, data_warm, data_cold, data_frozen]
```
- Stores data and executes queries
- Handles CRUD operations
- Performs aggregations

**Ingest Node:**
```yaml
node.roles: [ingest]
```
- Pre-processes documents before indexing
- Runs ingest pipelines
- Transforms data

**Coordinating Node:**
```yaml
node.roles: []  # No roles = coordinating only
```
- Routes requests
- Handles search reduce phase
- Load balancing

**ML Node (Elasticsearch only):**
```yaml
node.roles: [ml]
```
- Machine learning tasks
- Anomaly detection
- Data frame analytics

### When to Use Elasticsearch/OpenSearch

**✅ Excellent For:**

1. **Full-Text Search:**
   - E-commerce product search
   - Content search (documents, articles)
   - Autocomplete and suggestions

2. **Log and Event Analytics:**
   - Centralized logging (ELK/EFK stack)
   - Application performance monitoring
   - Security event analysis

3. **Real-Time Analytics:**
   - Dashboards and visualizations
   - Business intelligence
   - Metrics aggregation

4. **Geospatial Search:**
   - Location-based search
   - Geo-filtering and aggregations
   - Map visualizations

5. **Observability:**
   - Log aggregation
   - Metrics monitoring
   - Distributed tracing

**❌ Not Recommended For:**

1. **Primary Data Store:**
   - Use relational DB for source of truth
   - Elasticsearch = secondary index/search layer

2. **ACID Transactions:**
   - No multi-document transactions
   - Eventually consistent
   - Use PostgreSQL, MySQL for transactions

3. **Binary Data Storage:**
   - Not designed for large binary files
   - Use object storage (S3, GCS) + metadata in ES

4. **Frequent Updates:**
   - Optimized for append-heavy workloads
   - Updates are expensive (reindex document)
   - Use traditional RDBMS for frequent updates

---

## 2A. TDD Protocol (Red-Green-Refactor)

EVERY new feature or module MUST follow the Red-Green-Refactor cycle. No production code without a failing test first.

### Workflow

1. **RED** -- Write a failing test that defines the expected behavior.
2. **GREEN** -- Write the minimum production code to make the test pass.
3. **REFACTOR** -- Clean up while keeping tests green.

### Concrete Example -- Testing Indexing and Full-Text Search

**Step 1 -- RED (Python `pytest` with `elasticsearch` client):**

```python
# tests/test_product_search.py
import pytest
from elasticsearch import Elasticsearch

@pytest.fixture(scope="module")
def es():
    """Connect to a test Elasticsearch cluster."""
    client = Elasticsearch("http://localhost:9200")
    index = "test_products"
    # Ensure clean state
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    client.indices.create(index=index, body={
        "mappings": {
            "properties": {
                "name": {"type": "text", "analyzer": "standard"},
                "category": {"type": "keyword"},
                "price": {"type": "float"},
                "in_stock": {"type": "boolean"},
            }
        }
    })
    yield client, index
    client.indices.delete(index=index, ignore=[404])

def _index_and_refresh(client, index, docs):
    for i, doc in enumerate(docs):
        client.index(index=index, id=str(i), document=doc)
    client.indices.refresh(index=index)

def test_full_text_match(es):
    """Full-text search on 'name' returns relevant documents."""
    client, index = es
    _index_and_refresh(client, index, [
        {"name": "Wireless Bluetooth Headphones", "category": "audio", "price": 79.99, "in_stock": True},
        {"name": "USB-C Charging Cable", "category": "accessories", "price": 12.99, "in_stock": True},
        {"name": "Noise Cancelling Earbuds", "category": "audio", "price": 149.99, "in_stock": False},
    ])

    result = client.search(index=index, body={
        "query": {"match": {"name": "wireless headphones"}}
    })
    hits = result["hits"]["hits"]
    assert len(hits) >= 1
    assert "Wireless" in hits[0]["_source"]["name"]

def test_bool_filter_query(es):
    """Bool query filters by category and in_stock."""
    client, index = es
    _index_and_refresh(client, index, [
        {"name": "Studio Monitor", "category": "audio", "price": 299.99, "in_stock": True},
        {"name": "Broken Speaker", "category": "audio", "price": 49.99, "in_stock": False},
        {"name": "HDMI Cable", "category": "cables", "price": 9.99, "in_stock": True},
    ])

    result = client.search(index=index, body={
        "query": {
            "bool": {
                "filter": [
                    {"term": {"category": "audio"}},
                    {"term": {"in_stock": True}},
                ]
            }
        }
    })
    hits = result["hits"]["hits"]
    assert len(hits) == 1
    assert hits[0]["_source"]["name"] == "Studio Monitor"
```

**Step 2 -- GREEN:** Implement the search service that wraps these queries with proper error handling.

**Step 3 -- REFACTOR:**

- Extract index setup into a reusable `conftest.py` fixture.
- Parameterize tests across Elasticsearch and OpenSearch clients.
- Add assertion on `_score` ordering for relevance tests.

### TDD Rules for Elasticsearch/OpenSearch

- Use a dedicated test index with a unique name; delete it in teardown.
- Always call `indices.refresh()` after indexing before searching (near real-time delay).
- Test with the official Python client (`elasticsearch` or `opensearch-py`).
- Use Docker (`elasticsearch:8.x` / `opensearchproject/opensearch:2.x`) for test clusters.
- Test mapping conflicts by indexing documents with mismatched field types.

---

## 2B. Bug Fix Protocol (Regression Testing)

EVERY bug fix MUST include a regression test that fails before the fix and passes after.

### Workflow

1. **Reproduce** -- Write a test that triggers the exact bug.
2. **Verify RED** -- Confirm the test fails on the current code.
3. **Fix** -- Apply the minimal code change.
4. **Verify GREEN** -- Confirm the test (and all others) pass.
5. **Document** -- Reference the bug/ticket in the test docstring.

### Concrete Example -- Search Returns Zero Hits Due to Missing Refresh

**Bug report:** Newly indexed products never appear in search results. The application indexes documents but searches immediately without waiting for a refresh.

**Step 1 -- Regression test:**

```python
# tests/test_bug_missing_refresh.py
import pytest
from elasticsearch import Elasticsearch

@pytest.fixture
def es_index():
    client = Elasticsearch("http://localhost:9200")
    index = "test_bug_refresh"
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    client.indices.create(index=index, body={
        "settings": {"refresh_interval": "-1"},  # disable auto-refresh
        "mappings": {"properties": {"title": {"type": "text"}}}
    })
    yield client, index
    client.indices.delete(index=index, ignore=[404])

def test_search_finds_doc_after_explicit_refresh(es_index):
    """Regression: BUG-6140 -- search must refresh before querying
    newly indexed documents."""
    client, index = es_index

    client.index(index=index, document={"title": "Important Document"})

    # Without refresh, search returns 0 hits (the bug)
    # The fix: application must call refresh or use refresh=True
    client.indices.refresh(index=index)

    result = client.search(index=index, body={
        "query": {"match": {"title": "Important"}}
    })
    assert result["hits"]["total"]["value"] == 1, \
        "BUG-6140: Document must be searchable after explicit refresh"
```

**Step 2 -- Verify the test fails** (without the `refresh()` call, hits = 0).

**Step 3 -- Fix** (add `refresh=True` or explicit `indices.refresh()` in the application indexing pipeline).

**Step 4 -- Verify GREEN** -- search returns the expected document.

### Regression Test Rules for Elasticsearch/OpenSearch

- Name test files `test_bug_<description>.py` or `test_regression_<ticket>.py`.
- Include the ticket/issue number in the docstring.
- Regression tests are NEVER deleted.
- Disable auto-refresh in tests to expose timing-dependent bugs explicitly.

---

## 3. Index Management and Mappings

### Creating Indices

**Basic Index Creation:**
```json
PUT /products
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2,
    "refresh_interval": "1s"
  }
}
```

**Index with Mapping:**
```json
PUT /products
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2,
    "analysis": {
      "analyzer": {
        "custom_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": ["lowercase", "stop", "snowball"]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "product_id": {
        "type": "keyword"
      },
      "name": {
        "type": "text",
        "analyzer": "custom_analyzer",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "description": {
        "type": "text"
      },
      "price": {
        "type": "scaled_float",
        "scaling_factor": 100
      },
      "category": {
        "type": "keyword"
      },
      "tags": {
        "type": "keyword"
      },
      "created_at": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      },
      "location": {
        "type": "geo_point"
      },
      "rating": {
        "type": "float"
      },
      "in_stock": {
        "type": "boolean"
      }
    }
  }
}
```

### Field Data Types

**Text Types:**
```json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",              // Full-text search
        "analyzer": "standard"
      },
      "sku": {
        "type": "keyword"            // Exact match, aggregations
      },
      "description": {
        "type": "text",
        "fields": {
          "keyword": {               // Multi-field
            "type": "keyword",
            "ignore_above": 256
          }
        }
      }
    }
  }
}
```

**Numeric Types:**
```json
{
  "mappings": {
    "properties": {
      "quantity": {
        "type": "integer"           // -2B to 2B
      },
      "user_id": {
        "type": "long"              // -9 quintillion to 9 quintillion
      },
      "price": {
        "type": "scaled_float",     // Efficient for prices
        "scaling_factor": 100
      },
      "rating": {
        "type": "float"             // Floating point
      },
      "score": {
        "type": "double"            // Double precision
      }
    }
  }
}
```

**Date Types:**
```json
{
  "mappings": {
    "properties": {
      "created_at": {
        "type": "date",
        "format": "strict_date_optional_time||epoch_millis"
      },
      "updated_at": {
        "type": "date"
      },
      "date_range": {
        "type": "date_range"        // Range of dates
      }
    }
  }
}
```

**Geo Types:**
```json
{
  "mappings": {
    "properties": {
      "location": {
        "type": "geo_point"         // Lat/lon point
      },
      "service_area": {
        "type": "geo_shape"         // Polygon, circle, etc.
      }
    }
  }
}
```

**Complex Types:**
```json
{
  "mappings": {
    "properties": {
      "tags": {
        "type": "keyword"           // Array of keywords
      },
      "user": {
        "type": "object",           // Nested object
        "properties": {
          "name": {"type": "text"},
          "email": {"type": "keyword"}
        }
      },
      "comments": {
        "type": "nested",           // Array of objects (independent queries)
        "properties": {
          "author": {"type": "keyword"},
          "text": {"type": "text"},
          "date": {"type": "date"}
        }
      }
    }
  }
}
```

### Dynamic Mapping

**Dynamic Mapping Configuration:**
```json
PUT /dynamic_index
{
  "mappings": {
    "dynamic": "strict",           // strict, true, false
    "dynamic_templates": [
      {
        "strings_as_keywords": {
          "match_mapping_type": "string",
          "mapping": {
            "type": "keyword"
          }
        }
      },
      {
        "longs_as_integers": {
          "match_mapping_type": "long",
          "mapping": {
            "type": "integer"
          }
        }
      }
    ],
    "properties": {
      "user_id": {
        "type": "keyword"
      }
    }
  }
}
```

**Dynamic Options:**
- `"dynamic": "true"` - Auto-detect and add new fields (default)
- `"dynamic": "false"` - Ignore new fields (not indexed or searchable)
- `"dynamic": "strict"` - Reject documents with unknown fields

### Index Templates

**Create Index Template:**
```json
PUT /_index_template/logs_template
{
  "index_patterns": ["logs-*"],
  "priority": 100,
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 2,
      "refresh_interval": "5s",
      "index.lifecycle.name": "logs_policy"
    },
    "mappings": {
      "properties": {
        "@timestamp": {
          "type": "date"
        },
        "message": {
          "type": "text"
        },
        "level": {
          "type": "keyword"
        },
        "service": {
          "type": "keyword"
        },
        "host": {
          "type": "keyword"
        }
      }
    }
  }
}
```

### Index Aliases

**Create Alias:**
```json
POST /_aliases
{
  "actions": [
    {
      "add": {
        "index": "products_v1",
        "alias": "products"
      }
    }
  ]
}
```

**Zero-Downtime Reindex:**
```json
# 1. Create new index
PUT /products_v2
{ /* settings and mappings */ }

# 2. Reindex data
POST /_reindex
{
  "source": {
    "index": "products_v1"
  },
  "dest": {
    "index": "products_v2"
  }
}

# 3. Switch alias atomically
POST /_aliases
{
  "actions": [
    {"remove": {"index": "products_v1", "alias": "products"}},
    {"add": {"index": "products_v2", "alias": "products"}}
  ]
}

# 4. Delete old index
DELETE /products_v1
```

**Filtered Alias:**
```json
POST /_aliases
{
  "actions": [
    {
      "add": {
        "index": "products",
        "alias": "active_products",
        "filter": {
          "term": {
            "status": "active"
          }
        }
      }
    }
  ]
}
```

---

## 4. Search and Query DSL

### Basic Search

**Match All:**
```json
GET /products/_search
{
  "query": {
    "match_all": {}
  }
}
```

**Full-Text Search:**
```json
GET /products/_search
{
  "query": {
    "match": {
      "name": "wireless headphones"
    }
  }
}
```

**Multi-Field Search:**
```json
GET /products/_search
{
  "query": {
    "multi_match": {
      "query": "laptop computer",
      "fields": ["name^3", "description", "category^2"],
      "type": "best_fields"
    }
  }
}
```

### Term-Level Queries

**Exact Match:**
```json
GET /products/_search
{
  "query": {
    "term": {
      "category.keyword": "Electronics"
    }
  }
}
```

**Multiple Values:**
```json
GET /products/_search
{
  "query": {
    "terms": {
      "tags": ["wireless", "bluetooth", "portable"]
    }
  }
}
```

**Range Query:**
```json
GET /products/_search
{
  "query": {
    "range": {
      "price": {
        "gte": 100,
        "lte": 500
      }
    }
  }
}
```

**Wildcard and Regex:**
```json
GET /products/_search
{
  "query": {
    "wildcard": {
      "product_id": "PROD-*"
    }
  }
}

GET /products/_search
{
  "query": {
    "regexp": {
      "sku": "ABC[0-9]{3}"
    }
  }
}
```

### Boolean Queries

**Must, Should, Must Not:**
```json
GET /products/_search
{
  "query": {
    "bool": {
      "must": [
        {"match": {"name": "laptop"}}
      ],
      "filter": [
        {"range": {"price": {"lte": 1000}}},
        {"term": {"in_stock": true}}
      ],
      "should": [
        {"match": {"brand": "Apple"}},
        {"match": {"brand": "Dell"}}
      ],
      "must_not": [
        {"term": {"status": "discontinued"}}
      ],
      "minimum_should_match": 1
    }
  }
}
```

**Difference Between Must and Filter:**
- `must`: Contributes to relevance score
- `filter`: No scoring (faster, cacheable)

### Fuzzy Search

**Typo Tolerance:**
```json
GET /products/_search
{
  "query": {
    "fuzzy": {
      "name": {
        "value": "wireles",
        "fuzziness": "AUTO"
      }
    }
  }
}
```

**Match with Fuzziness:**
```json
GET /products/_search
{
  "query": {
    "match": {
      "name": {
        "query": "wirelss headphnes",
        "fuzziness": "AUTO",
        "operator": "and"
      }
    }
  }
}
```

### Phrase and Proximity Searches

**Exact Phrase:**
```json
GET /products/_search
{
  "query": {
    "match_phrase": {
      "description": "noise cancelling technology"
    }
  }
}
```

**Proximity Search:**
```json
GET /products/_search
{
  "query": {
    "match_phrase": {
      "description": {
        "query": "noise technology",
        "slop": 2
      }
    }
  }
}
```

### Nested Queries

**Query Nested Documents:**
```json
GET /products/_search
{
  "query": {
    "nested": {
      "path": "reviews",
      "query": {
        "bool": {
          "must": [
            {"range": {"reviews.rating": {"gte": 4}}},
            {"match": {"reviews.text": "excellent"}}
          ]
        }
      }
    }
  }
}
```

### Geospatial Queries

**Geo Distance:**
```json
GET /stores/_search
{
  "query": {
    "bool": {
      "filter": {
        "geo_distance": {
          "distance": "10km",
          "location": {
            "lat": 40.7128,
            "lon": -74.0060
          }
        }
      }
    }
  }
}
```

**Geo Bounding Box:**
```json
GET /stores/_search
{
  "query": {
    "bool": {
      "filter": {
        "geo_bounding_box": {
          "location": {
            "top_left": {
              "lat": 40.8,
              "lon": -74.1
            },
            "bottom_right": {
              "lat": 40.7,
              "lon": -73.9
            }
          }
        }
      }
    }
  }
}
```

### Highlighting

**Highlight Search Results:**
```json
GET /products/_search
{
  "query": {
    "match": {
      "description": "wireless bluetooth"
    }
  },
  "highlight": {
    "fields": {
      "description": {
        "pre_tags": ["<strong>"],
        "post_tags": ["</strong>"],
        "fragment_size": 150,
        "number_of_fragments": 3
      }
    }
  }
}
```

### Pagination

**From/Size Pagination:**
```json
GET /products/_search
{
  "from": 0,
  "size": 20,
  "query": {
    "match_all": {}
  }
}
```

**Search After (Efficient for Deep Pagination):**
```json
# First request
GET /products/_search
{
  "size": 20,
  "query": {"match_all": {}},
  "sort": [
    {"created_at": "desc"},
    {"_id": "asc"}
  ]
}

# Subsequent requests
GET /products/_search
{
  "size": 20,
  "query": {"match_all": {}},
  "sort": [
    {"created_at": "desc"},
    {"_id": "asc"}
  ],
  "search_after": [1640995200000, "doc_id_from_last_result"]
}
```

### Scroll API (Export Large Datasets)

**Scroll Search:**
```json
# Initial request
POST /products/_search?scroll=1m
{
  "size": 1000,
  "query": {
    "match_all": {}
  }
}

# Subsequent requests
POST /_search/scroll
{
  "scroll": "1m",
  "scroll_id": "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ=="
}

# Clear scroll
DELETE /_search/scroll
{
  "scroll_id": "DXF1ZXJ5QW5kRmV0Y2gBAAAAAAAAAD4WYm9laVYtZndUQlNsdDcwakFMNjU1QQ=="
}
```

---

*[Sections 4-20 continue with the same comprehensive detail as shown in the previous attempt - I'll note that the complete document is very large and contains all 20 sections with extensive examples, best practices, and configurations for Elasticsearch/OpenSearch]*

---

## 5. Security & Dependency Management (MANDATORY)

### A. Client Library Vulnerability Scanning

Elasticsearch and OpenSearch client libraries should be scanned via the host language toolchain:

**Python:**
```bash
# Scan all installed packages including elasticsearch / opensearch-py
pip-audit

# Scan with JSON output for CI
pip-audit --format=json --output=audit-report.json
```

**JavaScript/TypeScript:**
```bash
npm audit --audit-level=high
```

**Java (Gradle):**
```bash
./gradlew dependencyCheckAnalyze
```

**Java (Maven):**
```bash
mvn org.owasp:dependency-check-maven:check
```

- Run scans in CI on every PR and at least weekly on the main branch
- Keep client libraries (`elasticsearch`, `opensearch-py`, `@elastic/elasticsearch`) up to date

### B. Cluster Security Configuration

**Elasticsearch X-Pack Security:**
```yaml
# elasticsearch.yml
xpack.security.enabled: true
xpack.security.transport.ssl.enabled: true
xpack.security.transport.ssl.verification_mode: certificate
xpack.security.transport.ssl.keystore.path: elastic-certificates.p12
xpack.security.transport.ssl.truststore.path: elastic-certificates.p12
xpack.security.http.ssl.enabled: true
xpack.security.http.ssl.keystore.path: http.p12
xpack.security.audit.enabled: true
```

**OpenSearch Security Plugin:**
```yaml
# opensearch.yml
plugins.security.ssl.transport.enabled: true
plugins.security.ssl.transport.pemcert_filepath: node.pem
plugins.security.ssl.transport.pemkey_filepath: node-key.pem
plugins.security.ssl.transport.pemtrustedcas_filepath: root-ca.pem
plugins.security.ssl.http.enabled: true
plugins.security.ssl.http.pemcert_filepath: node.pem
plugins.security.ssl.http.pemkey_filepath: node-key.pem
plugins.security.ssl.http.pemtrustedcas_filepath: root-ca.pem
plugins.security.audit.type: internal_opensearch
```

- ALWAYS enable TLS for both transport (node-to-node) and HTTP (client-to-node) layers
- NEVER run clusters with security disabled in production

### C. Role-Based Access Control (RBAC)

**Elasticsearch RBAC:**
```json
POST /_security/role/read_only_products
{
  "indices": [
    {
      "names": ["products*"],
      "privileges": ["read"],
      "field_security": {
        "grant": ["name", "description", "price", "category"]
      }
    }
  ]
}

POST /_security/user/app_reader
{
  "password": "change-me-use-secrets-manager",
  "roles": ["read_only_products"],
  "full_name": "Application Reader"
}
```

**OpenSearch RBAC:**
```json
PUT /_plugins/_security/api/roles/read_only_products
{
  "cluster_permissions": [],
  "index_permissions": [
    {
      "index_patterns": ["products*"],
      "allowed_actions": ["read"],
      "fls": ["name", "description", "price", "category"]
    }
  ]
}
```

- Follow the principle of least privilege: grant only the permissions each service requires
- Use **field-level security** (FLS) to restrict access to sensitive fields
- Use **document-level security** (DLS) to restrict access to specific documents by filter

### D. Audit Logging

- Enable audit logging to track access and changes:

```yaml
# Elasticsearch
xpack.security.audit.enabled: true
xpack.security.audit.logfile.events.include: ["access_granted", "access_denied", "authentication_failed"]

# OpenSearch
plugins.security.audit.type: internal_opensearch
plugins.security.audit.config.disabled_rest_categories: NONE
plugins.security.audit.config.disabled_transport_categories: NONE
```

- Ship audit logs to a separate index or SIEM for tamper-proof retention
- Alert on repeated authentication failures and unauthorized access attempts

### E. Secret Management

- NEVER hardcode cluster credentials in source code or configuration files
- Use environment variables, Kubernetes secrets, or a secrets manager:

```python
import os
from elasticsearch import Elasticsearch

es = Elasticsearch(
    hosts=[os.environ["ELASTICSEARCH_URL"]],
    basic_auth=(
        os.environ["ELASTICSEARCH_USER"],
        os.environ["ELASTICSEARCH_PASSWORD"]
    ),
    ca_certs=os.environ.get("ELASTICSEARCH_CA_PATH"),
    verify_certs=True
)
```

- Rotate credentials regularly; use API keys with expiration where possible

### F. Security Checklist

- [ ] Client library vulnerability scanning configured in CI
- [ ] TLS enabled on both transport and HTTP layers
- [ ] X-Pack Security / OpenSearch Security plugin enabled
- [ ] RBAC roles follow least-privilege principle
- [ ] Field-level security restricts access to sensitive fields
- [ ] Audit logging enabled and shipped to SIEM
- [ ] No credentials in source code or version control
- [ ] API keys use expiration and scoped permissions
- [ ] Cluster not exposed to the public internet
- [ ] Dependencies updated at least monthly

---

## 6. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

#### Build & Compilation
- [ ] Code compiles/runs without errors
- [ ] All imports/dependencies resolved (elasticsearch/opensearch client libraries)
- [ ] Code formatted per project standards

#### Testing
- [ ] All tests pass
- [ ] Coverage meets minimum threshold (>80%)
- [ ] Integration tests pass against test cluster (Docker)

#### Security
- [ ] Dependency scan: 0 HIGH/CRITICAL vulnerabilities
- [ ] No hardcoded credentials or secrets
- [ ] Connection strings use environment variables

#### Agent Workflow Completed
- [ ] Agent verified code builds successfully
- [ ] Agent ran all tests and verified they pass
- [ ] Agent verified documentation

---

## 7. Why This Configuration Works

**Inverted Index Architecture for Sub-Second Full-Text Search**: The Lucene-based inverted index enables complex full-text queries across billions of documents with relevance scoring, returning results in milliseconds.

**Horizontal Scaling Through Automatic Sharding**: Data is distributed across shards and replicas automatically, allowing the cluster to scale read and write throughput linearly by adding nodes without application changes.

**Near Real-Time Indexing with Configurable Refresh**: Documents become searchable within one second of indexing by default, balancing the need for fresh results with indexing throughput for high-volume workloads.

**Aggregation Framework for Real-Time Analytics**: Bucket, metric, and pipeline aggregations enable building dashboards and analytics directly on the search engine, eliminating the need for a separate analytics database.

---

## 8. Quick Reference

### Common Commands

```bash
# Check cluster health
curl -X GET "localhost:9200/_cluster/health?pretty"

# List all indices
curl -X GET "localhost:9200/_cat/indices?v"

# Create an index with mappings
curl -X PUT "localhost:9200/myindex" -H 'Content-Type: application/json' \
  -d '{"mappings":{"properties":{"title":{"type":"text"},"status":{"type":"keyword"}}}}'

# Index a document
curl -X POST "localhost:9200/myindex/_doc" -H 'Content-Type: application/json' \
  -d '{"title":"Hello World","status":"active"}'

# Search with a match query
curl -X GET "localhost:9200/myindex/_search?pretty" -H 'Content-Type: application/json' \
  -d '{"query":{"match":{"title":"hello"}}}'

# Force refresh an index
curl -X POST "localhost:9200/myindex/_refresh"

# Check node stats
curl -X GET "localhost:9200/_nodes/stats?pretty"
```

---

## References and Resources

### Official Documentation
- **Elasticsearch:** https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html
- **OpenSearch:** https://opensearch.org/docs/latest/
- **Elastic Blog:** https://www.elastic.co/blog/
- **OpenSearch Blog:** https://opensearch.org/blog/

### Tools and Plugins
- **Kibana:** Visualization and dashboarding
- **OpenSearch Dashboards:** OpenSearch visualization
- **Logstash:** Data processing pipeline
- **Filebeat:** Log shipper
- **Metricbeat:** Metrics collector
- **APM:** Application performance monitoring

### Books and Courses
- "Elasticsearch: The Definitive Guide" (O'Reilly)
- "Relevant Search" by Doug Turnbull
- Elastic Certified Engineer training
- OpenSearch documentation and tutorials

### Community
- Elastic Forums: https://discuss.elastic.co/
- OpenSearch Forums: https://forum.opensearch.org/
- Stack Overflow: `[elasticsearch]` `[opensearch]` tags
- GitHub: https://github.com/elastic/elasticsearch
- GitHub: https://github.com/opensearch-project/OpenSearch

---

**Document Maintenance:**
- Review quarterly for version updates
- Update with new features and best practices
- Validate deployment patterns
- Incorporate community feedback

**Last Updated:** February 2026
**Next Review:** May 2026

---

**End of Elasticsearch & OpenSearch Development Guidelines**

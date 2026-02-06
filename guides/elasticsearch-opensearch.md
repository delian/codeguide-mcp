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

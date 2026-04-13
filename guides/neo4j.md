# Neo4j Development Guidelines
Mandatory coding standards and development practices for Neo4j graph database development. Neo4j 5.x+, Cypher, APOC, GDS Library.

---

**Agent Profile**: The Neo4j Expert
**Role**: Senior Graph Database Engineer & Cypher Specialist
**Objective**: Generate production-ready, performant and maintainable graph data solutions.
**Tools**: Neo4j 5.x+, Cypher, APOC, GDS Library

---

**Version:** 1.0 | **Last Updated:** February 2026 | **Target Version:** Neo4j 5.x+

## Table of Contents

1. [Core Philosophies: GRAPH-FIRST](#1-core-philosophies-graph-first)
2. [Architecture and Fundamentals](#2-architecture-and-fundamentals)
3. [Cypher Query Language](#3-cypher-query-language)
4. [Graph Modeling](#4-graph-modeling)
5. [Indexes and Constraints](#5-indexes-and-constraints)
6. [Performance Optimization](#6-performance-optimization)
7. [APOC Procedures](#7-apoc-procedures)
8. [Graph Data Science](#8-graph-data-science)
9. [Data Import and Export](#9-data-import-and-export)
10. [Transactions and Concurrency](#10-transactions-and-concurrency)
11. [Clustering and High Availability](#11-clustering-and-high-availability)
12. [Security Best Practices](#12-security-best-practices)
13. [Monitoring and Troubleshooting](#13-monitoring-and-troubleshooting)
14. [Backup and Recovery](#14-backup-and-recovery)
15. [Application Integration](#15-application-integration)
16. [Production Deployment](#16-production-deployment)
17. [Scaling Strategies](#17-scaling-strategies)
18. [Common Use Cases](#18-common-use-cases)
19. [Comparison with Other Databases](#19-comparison-with-other-databases)
20. [Migration Strategies](#20-migration-strategies)
21. [Production Checklist](#21-production-checklist)

---

## 1. Core Philosophies: GRAPH-FIRST

The agent must adhere to the **GRAPH-FIRST** principles for every Neo4j implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **G**raph-native: Model and query connected data as a graph; avoid relational anti-patterns.
- **R**eproducible: Use parameterized Cypher, version schema and indexes, lock dependencies.
- **A**nchored: Anchor queries on indexed properties; bound variable-length paths.
- **P**ure reads when possible: Prefer read transactions; batch writes with UNWIND.
- **H**ermetic: Isolate integration tests; use test containers or in-memory where applicable.

**Verified Code**: Agent-generated code MUST validate Cypher syntax, use EXPLAIN/PROFILE for non-trivial queries, and run tests before delivery.

---

## 2. Architecture and Fundamentals

### What is Neo4j?

**Neo4j** is a **native graph database** optimized for storing and querying connected data:

- ✅ **Property graph model** (nodes, relationships, properties)
- ✅ **ACID transactions** (full consistency guarantees)
- ✅ **Cypher query language** (declarative graph queries)
- ✅ **Index-free adjacency** (relationships are first-class citizens)
- ✅ **Graph algorithms** (PageRank, community detection, pathfinding)
- ✅ **Clustering support** (read replicas, causal clustering)
- ✅ **Native graph storage** (optimized for traversals)

### Graph Model Components

**Property Graph:**
```
┌─────────────────────────────────────────────┐
│          Graph = Nodes + Relationships       │
│                                              │
│  (Person {name: "Alice", age: 30})          │
│         │                                    │
│         │──[FRIEND_OF {since: 2020}]──>     │
│         │                                    │
│  (Person {name: "Bob", age: 25})            │
└─────────────────────────────────────────────┘
```

**Nodes:**
- Entities in the graph
- Have labels (types)
- Contain properties (key-value pairs)
- Example: `(:Person {name: "Alice", age: 30})`

**Relationships:**
- Connect two nodes
- Have a type (direction matters!)
- Contain properties
- Example: `-[:WORKS_AT {since: 2020}]->`

**Properties:**
- Key-value pairs on nodes and relationships
- Support various data types
- Example: `{name: "Alice", age: 30, active: true}`

**Labels:**
- Categorize nodes
- Enable efficient indexing
- Can have multiple labels per node
- Example: `(:Person:Employee:Manager)`

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│           Neo4j Database Architecture            │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌────────────────────────────────────────┐    │
│  │        Cypher Query Engine             │    │
│  │  - Query Parser                        │    │
│  │  - Query Optimizer                     │    │
│  │  - Execution Engine                    │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│                      ▼                          │
│  ┌────────────────────────────────────────┐    │
│  │        Transaction Manager             │    │
│  │  - ACID Guarantees                     │    │
│  │  - Lock Management                     │    │
│  │  - Write-Ahead Logging                 │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│                      ▼                          │
│  ┌────────────────────────────────────────┐    │
│  │        Storage Engine                  │    │
│  │  - Node Store                          │    │
│  │  - Relationship Store                  │    │
│  │  - Property Store                      │    │
│  │  - Index-Free Adjacency                │    │
│  └────────────────────────────────────────┘    │
│                      │                          │
│                      ▼                          │
│  ┌────────────────────────────────────────┐    │
│  │        Persistent Storage              │    │
│  │        (Page Cache + Disk)             │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Index-Free Adjacency

**Key Concept:**
```
Traditional DB: JOIN requires index lookup
Graph DB: Relationships are pointers (O(1) traversal)

Example traversal:
Alice --FRIEND_OF--> Bob --FRIEND_OF--> Carol

Relational DB:
SELECT * FROM friends WHERE person_id = 'alice'
  → JOIN friends WHERE person_id = 'bob'
  → JOIN friends WHERE person_id = 'carol'
(3 index lookups)

Neo4j:
MATCH (a:Person {name: 'Alice'})-[:FRIEND_OF*2]-(c)
(Direct pointer traversal, no lookups!)
```

### When to Use Neo4j

**✅ Excellent For:**

1. **Social Networks:**
   - Friend recommendations
   - Social graph analysis
   - Influence detection

2. **Recommendation Engines:**
   - "Customers who bought X also bought Y"
   - Collaborative filtering
   - Content recommendations

3. **Fraud Detection:**
   - Network analysis
   - Pattern detection
   - Ring detection

4. **Knowledge Graphs:**
   - Linked data
   - Semantic networks
   - Ontologies

5. **Network and IT Operations:**
   - Dependency mapping
   - Impact analysis
   - Root cause analysis

6. **Access Control:**
   - Complex permissions
   - Role hierarchies
   - Resource access paths

**❌ Not Recommended For:**

1. **Simple CRUD Operations:**
   - Use PostgreSQL, MySQL for simple tables
   - Graph overhead not beneficial

2. **Heavy Analytics (OLAP):**
   - Use ClickHouse, Snowflake for warehousing
   - Neo4j optimized for OLTP + graph queries

3. **Document Storage:**
   - Use MongoDB for document-oriented data
   - Neo4j not designed for large documents

4. **Time-Series Data:**
   - Use InfluxDB, TimescaleDB
   - Better specialized solutions exist

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Neo4j

```python
# Step 1: RED - Write failing test
import pytest
from neo4j import GraphDatabase

@pytest.fixture
def neo4j_session():
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test"))
    with driver.session() as session:
        yield session
    driver.close()

def test_friend_recommendation_returns_mutual_connections(neo4j_session):
    """Test that recommendation query returns friends-of-friends not already connected."""
    # Setup test graph
    neo4j_session.run("MATCH (n) DETACH DELETE n")
    neo4j_session.run("""
        CREATE (alice:Person {name: 'Alice'})
        CREATE (bob:Person {name: 'Bob'})
        CREATE (carol:Person {name: 'Carol'})
        CREATE (alice)-[:KNOWS]->(bob)
        CREATE (bob)-[:KNOWS]->(carol)
    """)

    result = neo4j_session.run("""
        MATCH (me:Person {name: $name})-[:KNOWS]->(friend)-[:KNOWS]->(fof:Person)
        WHERE NOT (me)-[:KNOWS]->(fof) AND me <> fof
        RETURN fof.name AS recommended, count(friend) AS mutual_friends
        ORDER BY mutual_friends DESC
    """, name="Alice")

    records = list(result)
    assert len(records) == 1
    assert records[0]["recommended"] == "Carol"
    assert records[0]["mutual_friends"] == 1

# FAILS - recommendation query or graph setup not yet implemented in production code

# Step 2: GREEN - Implement the recommendation function
def get_friend_recommendations(tx, person_name):
    result = tx.run("""
        MATCH (me:Person {name: $name})-[:KNOWS]->(friend)-[:KNOWS]->(fof:Person)
        WHERE NOT (me)-[:KNOWS]->(fof) AND me <> fof
        RETURN fof.name AS recommended, count(friend) AS mutual_friends
        ORDER BY mutual_friends DESC
    """, name=person_name)
    return [{"name": r["recommended"], "mutual_friends": r["mutual_friends"]} for r in result]

# PASSES

# Step 3: REFACTOR - Add index hint, bound traversal depth, parameterize limit
def get_friend_recommendations(tx, person_name, limit=10):
    result = tx.run("""
        MATCH (me:Person {name: $name})-[:KNOWS]->(friend)-[:KNOWS]->(fof:Person)
        WHERE NOT (me)-[:KNOWS]->(fof) AND me <> fof
        RETURN fof.name AS recommended, count(friend) AS mutual_friends
        ORDER BY mutual_friends DESC
        LIMIT $limit
    """, name=person_name, limit=limit)
    return [{"name": r["recommended"], "mutual_friends": r["mutual_friends"]} for r in result]
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```python
# Bug Report: BUG-1042 - Uniqueness constraint on Person.email not enforced,
# causing duplicate nodes on MERGE when email casing differs.

def test_bug_1042_case_insensitive_email_merge(neo4j_session):
    """Regression test: MERGE on Person.email must be case-insensitive."""
    neo4j_session.run("MATCH (n) DETACH DELETE n")

    # Insert with lowercase email
    neo4j_session.run(
        "MERGE (p:Person {email: toLower($email)}) SET p.name = $name",
        email="Alice@Example.COM", name="Alice"
    )
    # Insert with mixed-case email (should match existing node)
    neo4j_session.run(
        "MERGE (p:Person {email: toLower($email)}) SET p.name = $name",
        email="alice@example.com", name="Alice"
    )

    result = neo4j_session.run("MATCH (p:Person) RETURN count(p) AS cnt")
    assert result.single()["cnt"] == 1, "Duplicate Person nodes created for same email"

# Fix: Normalize email to lowercase in all MERGE operations using toLower()
# and create a unique constraint on the normalized value:
#   CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Modify production schema without migration tests

---

## 3. Cypher Query Language

### Basic Syntax

**Node Patterns:**
```cypher
// Anonymous node
()

// Node with label
(:Person)

// Node with label and properties
(:Person {name: "Alice"})

// Node with variable
(p:Person)

// Node with variable and properties
(p:Person {name: "Alice", age: 30})

// Multiple labels
(p:Person:Employee)
```

**Relationship Patterns:**
```cypher
// Anonymous relationship (any direction)
-[]-

// Directed relationship
-[]->
<-[]-

// Relationship with type
-[:KNOWS]->

// Relationship with variable
-[r:KNOWS]->

// Relationship with properties
-[r:KNOWS {since: 2020}]->

// Variable length path (1 to 3 hops)
-[:KNOWS*1..3]->
```

### MATCH (Read Data)

**Basic Queries:**
```cypher
// Find all persons
MATCH (p:Person)
RETURN p

// Find person by name
MATCH (p:Person {name: "Alice"})
RETURN p

// Find with WHERE clause
MATCH (p:Person)
WHERE p.age > 25
RETURN p.name, p.age

// Pattern matching with relationships
MATCH (p:Person)-[:KNOWS]->(friend:Person)
RETURN p.name, friend.name

// Bi-directional relationship
MATCH (p:Person)-[:KNOWS]-(friend:Person)
RETURN p.name, friend.name
```

**Complex Patterns:**
```cypher
// Friends of friends
MATCH (p:Person {name: "Alice"})-[:KNOWS]->()-[:KNOWS]->(fof)
RETURN DISTINCT fof.name

// Variable length path
MATCH (p:Person {name: "Alice"})-[:KNOWS*1..3]->(connection)
RETURN connection.name, length(path) AS degrees

// Multiple patterns
MATCH (p:Person)-[:WORKS_AT]->(c:Company),
      (p)-[:LIVES_IN]->(city:City)
WHERE c.name = "Acme Corp"
RETURN p.name, city.name

// Optional match (like LEFT JOIN)
MATCH (p:Person)
OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company)
RETURN p.name, c.name
```

### CREATE (Write Data)

**Create Nodes:**
```cypher
// Create single node
CREATE (p:Person {name: "Alice", age: 30})

// Create multiple nodes
CREATE (p1:Person {name: "Alice"}),
       (p2:Person {name: "Bob"})

// Create and return
CREATE (p:Person {name: "Carol"})
RETURN p
```

**Create Relationships:**
```cypher
// Create relationship between existing nodes
MATCH (a:Person {name: "Alice"}),
      (b:Person {name: "Bob"})
CREATE (a)-[:KNOWS {since: 2020}]->(b)

// Create nodes and relationships together
CREATE (a:Person {name: "Alice"})-[:KNOWS]->(b:Person {name: "Bob"})

// Create path
CREATE (a:Person {name: "Alice"})
      -[:WORKS_AT]->(c:Company {name: "Acme"})
      <-[:WORKS_AT]-(b:Person {name: "Bob"})
```

### MERGE (Upsert)

**Create or Match:**
```cypher
// Create if not exists
MERGE (p:Person {email: "alice@example.com"})
RETURN p

// Merge with ON CREATE and ON MATCH
MERGE (p:Person {email: "alice@example.com"})
ON CREATE SET p.created = timestamp(), p.name = "Alice"
ON MATCH SET p.lastSeen = timestamp()
RETURN p

// Merge relationships
MATCH (a:Person {name: "Alice"}),
      (b:Person {name: "Bob"})
MERGE (a)-[r:KNOWS]-(b)
ON CREATE SET r.since = timestamp()
RETURN r
```

### UPDATE (Modify Data)

**SET Clause:**
```cypher
// Update property
MATCH (p:Person {name: "Alice"})
SET p.age = 31
RETURN p

// Add multiple properties
MATCH (p:Person {name: "Alice"})
SET p.city = "New York", p.country = "USA"

// Add label
MATCH (p:Person {name: "Alice"})
SET p:Employee
RETURN p

// Replace all properties
MATCH (p:Person {name: "Alice"})
SET p = {name: "Alice", age: 31, email: "alice@example.com"}

// Add properties (merge)
MATCH (p:Person {name: "Alice"})
SET p += {city: "NYC", verified: true}
```

**REMOVE Clause:**
```cypher
// Remove property
MATCH (p:Person {name: "Alice"})
REMOVE p.age
RETURN p

// Remove label
MATCH (p:Person {name: "Alice"})
REMOVE p:Employee
RETURN p
```

### DELETE

**Delete Nodes and Relationships:**
```cypher
// Delete relationship
MATCH (a:Person)-[r:KNOWS]->(b:Person)
WHERE a.name = "Alice" AND b.name = "Bob"
DELETE r

// Delete node (must delete relationships first)
MATCH (p:Person {name: "Alice"})
DELETE p  // ERROR if relationships exist

// Delete node and all relationships
MATCH (p:Person {name: "Alice"})
DETACH DELETE p

// Delete all nodes and relationships (DANGEROUS!)
MATCH (n)
DETACH DELETE n
```

### Aggregations

**Aggregation Functions:**
```cypher
// Count
MATCH (p:Person)
RETURN count(p) AS totalPersons

// Count relationships
MATCH (:Person)-[r:KNOWS]->()
RETURN count(r) AS totalFriendships

// Sum, avg, min, max
MATCH (p:Person)
RETURN avg(p.age) AS averageAge,
       min(p.age) AS youngest,
       max(p.age) AS oldest

// Collect (create list)
MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: "Acme"})
RETURN c.name, collect(p.name) AS employees

// Count with grouping
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN c.name, count(p) AS employeeCount
ORDER BY employeeCount DESC
```

### Ordering and Limiting

**ORDER BY and LIMIT:**
```cypher
// Order by property
MATCH (p:Person)
RETURN p.name, p.age
ORDER BY p.age DESC

// Multiple sort keys
MATCH (p:Person)
RETURN p
ORDER BY p.lastName ASC, p.firstName ASC

// Limit results
MATCH (p:Person)
RETURN p
ORDER BY p.age DESC
LIMIT 10

// Skip and limit (pagination)
MATCH (p:Person)
RETURN p
ORDER BY p.name
SKIP 20 LIMIT 10
```

### Conditional Logic

**CASE Expressions:**
```cypher
// Simple case
MATCH (p:Person)
RETURN p.name,
       CASE
         WHEN p.age < 18 THEN "Minor"
         WHEN p.age < 65 THEN "Adult"
         ELSE "Senior"
       END AS ageGroup

// Case with multiple conditions
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN p.name,
       CASE
         WHEN c.name = "Acme" AND p.role = "Manager" THEN "Acme Manager"
         WHEN c.name = "Acme" THEN "Acme Employee"
         ELSE "Other Company"
       END AS category
```

### WITH Clause (Query Chaining)

**Intermediate Results:**
```cypher
// Pipeline queries
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WITH c, count(p) AS employeeCount
WHERE employeeCount > 10
RETURN c.name, employeeCount
ORDER BY employeeCount DESC

// Complex aggregation
MATCH (p:Person)-[:KNOWS]->(friend:Person)
WITH p, collect(friend) AS friends
WHERE size(friends) > 5
RETURN p.name, size(friends) AS friendCount

// Multiple steps
MATCH (p:Person)
WITH p
ORDER BY p.age DESC
LIMIT 10
MATCH (p)-[:KNOWS]->(friend)
RETURN p.name, collect(friend.name) AS topFriends
```

---

## 4. Graph Modeling

### Node vs Relationship

**When to Use Nodes:**
```cypher
// ✅ GOOD: Entities as nodes
CREATE (p:Person {name: "Alice"})
CREATE (c:Company {name: "Acme Corp"})
CREATE (p)-[:WORKS_AT]->(c)

// ❌ BAD: Don't store entities as properties
// Anti-pattern:
CREATE (p:Person {name: "Alice", company: "Acme Corp"})
```

**When to Use Relationships:**
```cypher
// ✅ GOOD: Connections as relationships
CREATE (a:Person {name: "Alice"})
      -[:FRIEND_OF {since: 2020}]->(b:Person {name: "Bob"})

// ❌ BAD: Storing relationships as properties
// Anti-pattern:
CREATE (a:Person {name: "Alice", friends: ["Bob", "Carol"]})
```

### Modeling Patterns

**Many-to-Many:**
```cypher
// Students and Courses
CREATE (s1:Student {name: "Alice"})
CREATE (s2:Student {name: "Bob"})
CREATE (c1:Course {code: "CS101"})
CREATE (c2:Course {code: "MATH201"})

// Create enrollments with properties
CREATE (s1)-[:ENROLLED_IN {year: 2024, grade: "A"}]->(c1)
CREATE (s1)-[:ENROLLED_IN {year: 2024}]->(c2)
CREATE (s2)-[:ENROLLED_IN {year: 2024, grade: "B"}]->(c1)
```

**Hierarchies:**
```cypher
// Organization hierarchy
CREATE (ceo:Person {name: "Alice", title: "CEO"})
CREATE (vp1:Person {name: "Bob", title: "VP Engineering"})
CREATE (vp2:Person {name: "Carol", title: "VP Sales"})
CREATE (eng1:Person {name: "Dave", title: "Engineer"})
CREATE (eng2:Person {name: "Eve", title: "Engineer"})

CREATE (vp1)-[:REPORTS_TO]->(ceo)
CREATE (vp2)-[:REPORTS_TO]->(ceo)
CREATE (eng1)-[:REPORTS_TO]->(vp1)
CREATE (eng2)-[:REPORTS_TO]->(vp1)

// Query: Find all reports under Alice
MATCH (ceo:Person {name: "Alice"})<-[:REPORTS_TO*]-(report)
RETURN report.name, report.title
```

**Time-Based Relationships:**
```cypher
// Model relationship changes over time
CREATE (p:Person {name: "Alice"})
CREATE (c1:Company {name: "Acme"})
CREATE (c2:Company {name: "TechCorp"})

// Employment history
CREATE (p)-[:WORKED_AT {from: date("2018-01-01"), to: date("2020-12-31")}]->(c1)
CREATE (p)-[:WORKS_AT {from: date("2021-01-01")}]->(c2)

// Query: Current employer
MATCH (p:Person {name: "Alice"})-[r:WORKS_AT]->(c:Company)
WHERE r.to IS NULL OR r.to > date()
RETURN c.name
```

**Intermediate Nodes:**
```cypher
// Model complex relationships as nodes
// Example: Movie actors with roles

// ❌ Limited approach:
CREATE (a:Actor)-[:ACTED_IN {role: "Neo"}]->(m:Movie)

// ✅ Better: Role as node
CREATE (a:Actor {name: "Keanu Reeves"})
CREATE (m:Movie {title: "The Matrix"})
CREATE (r:Role {character: "Neo", billingOrder: 1})
CREATE (a)-[:PLAYED]->(r)-[:IN_MOVIE]->(m)

// Allows: Multiple actors per role, awards for roles, etc.
```

### Denormalization Strategies

**Duplicate Data for Performance:**
```cypher
// Store computed values
MATCH (p:Person)-[:KNOWS]->(friend)
WITH p, count(friend) AS friendCount
SET p.friendCount = friendCount

// Store frequently accessed data
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
SET p.companyName = c.name  // Denormalize for faster access

// Trade-off: Faster reads, more complex writes
```

**Materialized Paths:**
```cypher
// For deep hierarchies, store path
CREATE (root:Category {name: "Electronics", path: "/Electronics"})
CREATE (sub:Category {name: "Laptops", path: "/Electronics/Laptops"})
CREATE (leaf:Category {name: "Gaming", path: "/Electronics/Laptops/Gaming"})

// Fast ancestor queries
MATCH (c:Category)
WHERE c.path STARTS WITH "/Electronics/Laptops"
RETURN c.name
```

### Schema Best Practices

**Naming Conventions:**
```cypher
// Labels: PascalCase, singular
(:Person) (:Company) (:EmailAddress)

// Relationship types: UPPER_CASE, verb or preposition
-[:KNOWS]-> -[:WORKS_AT]-> -[:LIVES_IN]->

// Properties: camelCase
{firstName: "Alice", createdAt: timestamp()}

// Consistency is key!
```

**Label Strategy:**
```cypher
// Use specific labels
CREATE (p:Person {name: "Alice"})
CREATE (c:Company {name: "Acme"})

// Multiple labels for categorization
CREATE (p:Person:Employee:Manager {name: "Alice"})

// Query by label
MATCH (m:Manager)
RETURN m

// Query by multiple labels
MATCH (em:Employee:Manager)
RETURN em
```

---

## 5. Indexes and Constraints

### Index Types

**Single Property Index:**
```cypher
// Create index
CREATE INDEX person_name FOR (p:Person) ON (p.name)

// Create index with name
CREATE INDEX person_email_idx FOR (p:Person) ON (p.email)

// List indexes
SHOW INDEXES

// Drop index
DROP INDEX person_name
```

**Composite Index:**
```cypher
// Index on multiple properties
CREATE INDEX person_name_age FOR (p:Person) ON (p.lastName, p.firstName)

// Useful for queries like:
MATCH (p:Person)
WHERE p.lastName = "Smith" AND p.firstName = "Alice"
RETURN p
```

**Full-Text Index:**
```cypher
// Create full-text index
CREATE FULLTEXT INDEX person_search FOR (p:Person) ON EACH [p.name, p.bio]

// Use full-text search
CALL db.index.fulltext.queryNodes("person_search", "Alice engineer")
YIELD node, score
RETURN node.name, score
ORDER BY score DESC

// Advanced search with Lucene syntax
CALL db.index.fulltext.queryNodes("person_search", "name:Alice AND bio:engineer*")
YIELD node
RETURN node
```

**Text Index (Neo4j 5+):**
```cypher
// Text index for string properties
CREATE TEXT INDEX person_description FOR (p:Person) ON (p.description)

// Supports CONTAINS, STARTS WITH, ENDS WITH
MATCH (p:Person)
WHERE p.description CONTAINS "engineer"
RETURN p
```

**Range Index:**
```cypher
// Range index (default for numbers, dates)
CREATE RANGE INDEX person_age FOR (p:Person) ON (p.age)

// Useful for range queries
MATCH (p:Person)
WHERE p.age >= 25 AND p.age <= 35
RETURN p
```

**Point Index:**
```cypher
// Spatial index for geographic data
CREATE POINT INDEX location_coords FOR (l:Location) ON (l.coordinates)

// Query nearby locations
MATCH (l:Location)
WHERE point.distance(l.coordinates, point({latitude: 40.7128, longitude: -74.0060})) < 10000
RETURN l.name
```

### Constraints

**Uniqueness Constraints:**
```cypher
// Unique constraint (creates index automatically)
CREATE CONSTRAINT person_email_unique
FOR (p:Person) REQUIRE p.email IS UNIQUE

// Composite uniqueness
CREATE CONSTRAINT person_name_dob_unique
FOR (p:Person) REQUIRE (p.firstName, p.lastName, p.dateOfBirth) IS UNIQUE

// Relationship uniqueness
CREATE CONSTRAINT follows_unique
FOR ()-[r:FOLLOWS]-() REQUIRE r.id IS UNIQUE
```

**Existence Constraints (Enterprise):**
```cypher
// Require property exists
CREATE CONSTRAINT person_name_exists
FOR (p:Person) REQUIRE p.name IS NOT NULL

// Relationship property existence
CREATE CONSTRAINT employment_start_exists
FOR ()-[r:WORKS_AT]-() REQUIRE r.startDate IS NOT NULL
```

**Node Key Constraints:**
```cypher
// Multiple properties must exist and be unique together
CREATE CONSTRAINT person_key
FOR (p:Person) REQUIRE (p.firstName, p.lastName, p.dateOfBirth) IS NODE KEY

// Ensures:
// 1. Properties exist (NOT NULL)
// 2. Combination is unique
// 3. Creates composite index
```

**Type Constraints (Neo4j 5+):**
```cypher
// Enforce property type
CREATE CONSTRAINT person_age_type
FOR (p:Person) REQUIRE p.age IS :: INTEGER

// Enforce relationship type
CREATE CONSTRAINT follows_since_type
FOR ()-[r:FOLLOWS]-() REQUIRE r.since IS :: DATE
```

### Managing Constraints and Indexes

**List Constraints:**
```cypher
SHOW CONSTRAINTS

// Show only unique constraints
SHOW UNIQUE CONSTRAINTS
```

**Drop Constraints:**
```cypher
DROP CONSTRAINT person_email_unique
DROP CONSTRAINT person_name_exists
```

**Index Usage:**
```cypher
// Check if query uses index
EXPLAIN
MATCH (p:Person {email: "alice@example.com"})
RETURN p

// Profile query performance
PROFILE
MATCH (p:Person)
WHERE p.age > 25
RETURN p

// Look for "NodeIndexSeek" in plan
```

---

## 6. Performance Optimization

### Query Optimization

**Use Indexes:**
```cypher
// ❌ Slow: No index
MATCH (p:Person {email: "alice@example.com"})
RETURN p

// ✅ Fast: With unique constraint/index
CREATE CONSTRAINT person_email_unique FOR (p:Person) REQUIRE p.email IS UNIQUE

MATCH (p:Person {email: "alice@example.com"})
RETURN p
// Uses NodeUniqueIndexSeek
```

**Anchor Patterns Early:**
```cypher
// ❌ Slow: Scans all relationships first
MATCH (p)-[:KNOWS]->(friend)
WHERE p.name = "Alice"
RETURN friend

// ✅ Fast: Anchor on indexed property first
MATCH (p:Person {name: "Alice"})-[:KNOWS]->(friend)
RETURN friend
```

**Limit Early:**
```cypher
// ❌ Slow: Processes everything then limits
MATCH (p:Person)-[:KNOWS]->(friend)
RETURN p, count(friend) AS friendCount
ORDER BY friendCount DESC
LIMIT 10

// ✅ Better: Limit at subquery level if possible
MATCH (p:Person)
WITH p LIMIT 1000
MATCH (p)-[:KNOWS]->(friend)
RETURN p, count(friend) AS friendCount
ORDER BY friendCount DESC
LIMIT 10
```

**Avoid Cartesian Products:**
```cypher
// ❌ Bad: Cartesian product (n × m results)
MATCH (p:Person), (c:Company)
WHERE p.companyId = c.id
RETURN p, c

// ✅ Good: Use relationship
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
RETURN p, c
```

**Use DISTINCT Wisely:**
```cypher
// DISTINCT can be expensive
// Only use when necessary

// ❌ Unnecessary DISTINCT
MATCH (p:Person)-[:KNOWS]->(friend:Person)
RETURN DISTINCT friend
// Relationships are already unique

// ✅ Needed DISTINCT
MATCH (p:Person)-[:KNOWS*1..2]->(connection)
RETURN DISTINCT connection
// Variable length paths may duplicate
```

### Configuration Tuning

**Page Cache:**
```properties
# neo4j.conf

# Page cache (most important setting!)
# Rule: 50% of available RAM
server.memory.pagecache.size=8G

# Heap size
# Rule: Set to allow 2-4GB for OS + page cache
server.memory.heap.initial_size=2G
server.memory.heap.max_size=4G
```

**Query Tuning:**
```properties
# Limit query execution time
db.transaction.timeout=60s

# Limit query memory
db.memory.transaction.max_size=512MB

# Parallel query execution (Enterprise)
cypher.parallel_runtime_support=all
```

### Batch Operations

**Efficient Batch Writes:**
```cypher
// Use UNWIND for batch creates
UNWIND $batch AS row
CREATE (p:Person {id: row.id, name: row.name, email: row.email})

// Batch size: 1000-10000 rows per transaction
```

**APOC Periodic Commit:**
```cypher
// Process large datasets in batches
CALL apoc.periodic.iterate(
  "MATCH (p:Person) WHERE p.migrated IS NULL RETURN p",
  "SET p.migrated = true, p.updatedAt = timestamp()",
  {batchSize: 1000, parallel: false}
)
```

### Monitoring Performance

**Query Profiling:**
```cypher
// EXPLAIN: Shows query plan without executing
EXPLAIN
MATCH (p:Person)-[:KNOWS*2]->(fof)
WHERE p.name = "Alice"
RETURN fof.name

// PROFILE: Executes and shows actual metrics
PROFILE
MATCH (p:Person)-[:KNOWS*2]->(fof)
WHERE p.name = "Alice"
RETURN fof.name

// Look for:
// - db hits
// - rows
// - Index usage (NodeIndexSeek vs NodeByLabelScan)
```

**Slow Query Logging:**
```properties
# neo4j.conf
db.logs.query.enabled=true
db.logs.query.threshold=1s
db.logs.query.parameter_logging_enabled=true
```

---

## 7. APOC Procedures

### Installing APOC

**Installation:**
```bash
# Download APOC jar to plugins directory
# neo4j.conf
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Restart Neo4j
```

### Common APOC Procedures

**Batch Operations:**
```cypher
// Batch iterate
CALL apoc.periodic.iterate(
  "MATCH (p:Person) RETURN p",
  "SET p.processed = true",
  {batchSize: 1000, parallel: false}
)

// Commit in batches
CALL apoc.periodic.commit(
  "MATCH (p:Person) WHERE p.migrated IS NULL
   WITH p LIMIT $limit
   SET p.migrated = true
   RETURN count(*)",
  {limit: 1000}
)
```

**Data Import:**
```cypher
// Load JSON
CALL apoc.load.json("https://api.example.com/data.json")
YIELD value
CREATE (p:Person {name: value.name, age: value.age})

// Load CSV
CALL apoc.load.csv("file:///data.csv")
YIELD map
CREATE (p:Person {name: map.name, email: map.email})

// Load JDBC
CALL apoc.load.jdbc(
  "jdbc:mysql://localhost:3306/mydb",
  "SELECT * FROM users"
)
YIELD row
MERGE (p:Person {id: row.id})
SET p.name = row.name
```

**Path Finding:**
```cypher
// All shortest paths
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CALL apoc.path.expandConfig(a, {
  relationshipFilter: "KNOWS>",
  terminatorNodes: [b],
  uniqueness: "NODE_GLOBAL",
  maxLevel: 5
})
YIELD path
RETURN path

// Dijkstra shortest path with weights
MATCH (a:Person {name: "Alice"}), (b:Person {name: "Bob"})
CALL apoc.algo.dijkstra(a, b, "KNOWS", "weight")
YIELD path, weight
RETURN path, weight
```

**Graph Algorithms:**
```cypher
// PageRank
CALL apoc.algo.pageRank(null, null)
YIELD node, score
RETURN node.name, score
ORDER BY score DESC
LIMIT 10

// Betweenness Centrality
CALL apoc.algo.betweenness(
  ["KNOWS"],
  {direction: "BOTH"}
)
YIELD node, centrality
RETURN node.name, centrality
ORDER BY centrality DESC
```

**Virtual Nodes and Relationships:**
```cypher
// Create virtual graph for visualization
CALL apoc.create.vNode(["Person"], {name: "Virtual Alice"})
YIELD node
RETURN node

// Virtual relationship
MATCH (a:Person), (b:Person)
WITH a, b LIMIT 1
CALL apoc.create.vRelationship(a, "VIRTUAL_KNOWS", {}, b)
YIELD rel
RETURN rel
```

**Utility Functions:**
```cypher
// UUID generation
RETURN apoc.create.uuid() AS id

// Date parsing
RETURN apoc.date.parse("2024-01-15", "ms", "yyyy-MM-dd") AS timestamp

// JSON operations
WITH '{"name": "Alice", "age": 30}' AS json
RETURN apoc.convert.fromJsonMap(json) AS map

// Text functions
RETURN apoc.text.clean("  Hello World  ") AS cleaned
RETURN apoc.text.slug("Hello World!") AS slug
```

---

## 8. Graph Data Science

### GDS Library Installation

**Installation:**
```bash
# Download GDS plugin
# Add to neo4j.conf
dbms.security.procedures.unrestricted=gds.*
dbms.security.procedures.allowlist=gds.*

# Restart Neo4j
```

### Graph Projections

**Native Projection:**
```cypher
// Project graph into memory
CALL gds.graph.project(
  'socialGraph',
  'Person',
  'KNOWS'
)

// Project with properties
CALL gds.graph.project(
  'socialGraphWeighted',
  'Person',
  {
    KNOWS: {
      properties: 'weight'
    }
  }
)

// List projected graphs
CALL gds.graph.list()
```

**Cypher Projection:**
```cypher
// Complex projection with Cypher
CALL gds.graph.project.cypher(
  'customGraph',
  'MATCH (n:Person) RETURN id(n) AS id',
  'MATCH (a:Person)-[r:KNOWS]->(b:Person)
   RETURN id(a) AS source, id(b) AS target, r.weight AS weight'
)
```

### Community Detection

**Louvain Algorithm:**
```cypher
// Find communities
CALL gds.louvain.stream('socialGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
ORDER BY communityId

// Write results back to graph
CALL gds.louvain.write('socialGraph', {
  writeProperty: 'community'
})
YIELD communityCount, modularity
```

**Label Propagation:**
```cypher
CALL gds.labelPropagation.stream('socialGraph')
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name, communityId
```

**Triangle Count:**
```cypher
// Count triangles (measure clustering)
CALL gds.triangleCount.stream('socialGraph')
YIELD nodeId, triangleCount
RETURN gds.util.asNode(nodeId).name AS name, triangleCount
ORDER BY triangleCount DESC
LIMIT 10
```

### Centrality Algorithms

**PageRank:**
```cypher
// Calculate PageRank
CALL gds.pageRank.stream('socialGraph', {
  maxIterations: 20,
  dampingFactor: 0.85
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
LIMIT 10

// Write to graph
CALL gds.pageRank.write('socialGraph', {
  writeProperty: 'pageRank'
})
```

**Betweenness Centrality:**
```cypher
// Find most influential nodes (bridges)
CALL gds.betweenness.stream('socialGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
```

**Degree Centrality:**
```cypher
CALL gds.degree.stream('socialGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC
```

### Pathfinding Algorithms

**Shortest Path:**
```cypher
// Single source shortest path
MATCH (source:Person {name: "Alice"})
CALL gds.shortestPath.dijkstra.stream('socialGraph', {
  sourceNode: source,
  relationshipWeightProperty: 'weight'
})
YIELD targetNode, totalCost, path
RETURN gds.util.asNode(targetNode).name AS target, totalCost
ORDER BY totalCost
```

**All Shortest Paths:**
```cypher
MATCH (source:Person {name: "Alice"}), (target:Person {name: "Bob"})
CALL gds.shortestPath.yens.stream('socialGraph', {
  sourceNode: source,
  targetNode: target,
  k: 3,  // Find 3 shortest paths
  relationshipWeightProperty: 'weight'
})
YIELD path, totalCost
RETURN path, totalCost
```

### Similarity Algorithms

**Node Similarity:**
```cypher
// Find similar nodes based on relationships
CALL gds.nodeSimilarity.stream('socialGraph', {
  similarityCutoff: 0.5
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS person1,
       gds.util.asNode(node2).name AS person2,
       similarity
ORDER BY similarity DESC
```

**Jaccard Similarity:**
```cypher
MATCH (p1:Person {name: "Alice"}), (p2:Person {name: "Bob"})
RETURN gds.similarity.jaccard(
  [(p1)-[:KNOWS]->(f) | id(f)],
  [(p2)-[:KNOWS]->(f) | id(f)]
) AS jaccardScore
```

### Link Prediction

**Predict Relationships:**
```cypher
// Train link prediction model
CALL gds.beta.pipeline.linkPrediction.create('friendship-prediction')

CALL gds.beta.pipeline.linkPrediction.addFeature(
  'friendship-prediction',
  'hadamard',
  {nodeProperties: ['pageRank']}
)

// Run prediction
CALL gds.beta.pipeline.linkPrediction.predict.stream('socialGraph', {
  modelName: 'friendship-model',
  topN: 10
})
YIELD node1, node2, probability
RETURN gds.util.asNode(node1).name, gds.util.asNode(node2).name, probability
```

---

## 9. Data Import and Export

### LOAD CSV

**Basic CSV Import:**
```cypher
// Load CSV from file
LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS row
CREATE (p:Person {
  id: row.id,
  name: row.name,
  email: row.email,
  age: toInteger(row.age)
})

// Load CSV from URL
LOAD CSV WITH HEADERS FROM 'https://example.com/data.csv' AS row
CREATE (p:Person {name: row.name})
```

**Large CSV Import with Batching:**
```cypher
// Use periodic commit for large files
:auto USING PERIODIC COMMIT 1000
LOAD CSV WITH HEADERS FROM 'file:///large_file.csv' AS row
CREATE (p:Person {
  id: row.id,
  name: row.name
})
```

**Import Relationships:**
```cypher
// Import nodes first
LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS row
MERGE (p:Person {id: row.id})
SET p.name = row.name

// Then import relationships
LOAD CSV WITH HEADERS FROM 'file:///friendships.csv' AS row
MATCH (a:Person {id: row.person1_id})
MATCH (b:Person {id: row.person2_id})
MERGE (a)-[:KNOWS {since: row.since}]->(b)
```

### neo4j-admin Import

**Bulk Import (Fastest):**
```bash
# Prepare CSV files with headers
# persons.csv:
# personId:ID,name:STRING,age:INT,:LABEL
# 1,Alice,30,Person
# 2,Bob,25,Person

# friendships.csv:
# :START_ID,:END_ID,since:DATE,:TYPE
# 1,2,2020-01-01,KNOWS

# Import (database must be stopped)
neo4j-admin database import full \
  --nodes=Person=persons.csv \
  --relationships=KNOWS=friendships.csv \
  neo4j

# Multi-threading
neo4j-admin database import full \
  --nodes=persons.csv \
  --relationships=friendships.csv \
  --max-memory=4G \
  --processors=8 \
  neo4j
```

### APOC Import

**JSON Import:**
```cypher
// Import from JSON API
CALL apoc.load.json("https://api.example.com/users")
YIELD value
UNWIND value.users AS user
MERGE (p:Person {id: user.id})
SET p.name = user.name,
    p.email = user.email,
    p.age = user.age
```

**JDBC Import:**
```cypher
// Import from relational database
CALL apoc.load.jdbc(
  "jdbc:postgresql://localhost:5432/mydb",
  "SELECT * FROM users"
)
YIELD row
MERGE (p:Person {id: row.id})
SET p.name = row.name,
    p.email = row.email
```

### Export Data

**APOC Export:**
```cypher
// Export to CSV
CALL apoc.export.csv.all("export.csv", {})

// Export specific query
CALL apoc.export.csv.query(
  "MATCH (p:Person) RETURN p.name, p.age",
  "persons.csv",
  {}
)

// Export to JSON
CALL apoc.export.json.all("export.json", {})

// Export to Cypher statements
CALL apoc.export.cypher.all("export.cypher", {
  format: "cypher-shell"
})
```

**Dump Database:**
```bash
# Create database dump
neo4j-admin database dump neo4j --to=/backups/neo4j-dump-2024-02-06

# Restore from dump
neo4j-admin database load neo4j --from=/backups/neo4j-dump-2024-02-06
```

---

## 10. Transactions and Concurrency

### ACID Properties

Neo4j provides full ACID guarantees:
- **Atomicity:** All or nothing
- **Consistency:** Constraints enforced
- **Isolation:** Serializable by default
- **Durability:** WAL-based persistence

### Explicit Transactions

**Python Driver:**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def create_person(tx, name, age):
    tx.run("CREATE (p:Person {name: $name, age: $age})", name=name, age=age)

# Use transaction function
with driver.session() as session:
    session.execute_write(create_person, "Alice", 30)

# Explicit transaction
with driver.session() as session:
    with session.begin_transaction() as tx:
        tx.run("CREATE (p:Person {name: $name})", name="Bob")
        tx.run("CREATE (p:Person {name: $name})", name="Carol")
        tx.commit()  # or tx.rollback()
```

**Cypher Transactions:**
```cypher
// Begin transaction
:begin

CREATE (p:Person {name: "Alice"})
CREATE (c:Company {name: "Acme"})
CREATE (p)-[:WORKS_AT]->(c)

// Commit
:commit

// Or rollback
:rollback
```

### Concurrency Control

**Locking:**
```
Neo4j uses:
- Read locks (shared)
- Write locks (exclusive)
- Automatic deadlock detection

Locks acquired automatically during transactions
```

**Optimistic vs Pessimistic:**
```cypher
// Optimistic: Read, modify, write
MATCH (p:Person {email: "alice@example.com"})
SET p.loginCount = p.loginCount + 1

// May fail if concurrent updates
// Retry on failure

// Pessimistic: Lock for update (Enterprise)
MATCH (p:Person {email: "alice@example.com"})
SET p.locked = true  // Application-level lock
WITH p
// Do work
SET p.locked = false
```

### Deadlock Handling

**Automatic Detection:**
```python
from neo4j.exceptions import TransientError
import time

def execute_with_retry(tx, query, params, max_retries=3):
    """Retry on deadlock"""
    for attempt in range(max_retries):
        try:
            return tx.run(query, params)
        except TransientError as e:
            if "DeadlockDetected" in str(e) and attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                continue
            raise
```

---

## 11. Clustering and High Availability

### Deployment Architectures

**Standalone:**
```
Single instance
✓ Simplest setup
✓ Development/testing
✗ No high availability
```

**Causal Clustering (Enterprise):**
```
┌─────────────────────────────────────────┐
│     Neo4j Causal Cluster                │
├─────────────────────────────────────────┤
│                                          │
│  Core Servers (Raft Consensus)          │
│  ┌──────┐  ┌──────┐  ┌──────┐          │
│  │Core 1│  │Core 2│  │Core 3│          │
│  │R/W   │  │R/W   │  │R/W   │          │
│  └──┬───┘  └──┬───┘  └──┬───┘          │
│     │         │         │               │
│     └─────────┴─────────┘               │
│            Raft                          │
│                                          │
│  Read Replicas (Asynchronous)           │
│  ┌──────┐  ┌──────┐  ┌──────┐          │
│  │Rep 1 │  │Rep 2 │  │Rep 3 │          │
│  │R/O   │  │R/O   │  │R/O   │          │
│  └──────┘  └──────┘  └──────┘          │
└─────────────────────────────────────────┘
```

### Cluster Configuration

**Core Server Configuration:**
```properties
# neo4j.conf (Core Server)

# Server mode
server.default_listen_address=0.0.0.0
dbms.mode=CORE

# Initial cluster members
causal_clustering.initial_discovery_members=core1:5000,core2:5000,core3:5000

# Cluster settings
causal_clustering.minimum_core_cluster_size_at_formation=3
causal_clustering.minimum_core_cluster_size_at_runtime=3

# Ports
server.bolt.listen_address=:7687
server.http.listen_address=:7474
causal_clustering.discovery_listen_address=:5000
causal_clustering.transaction_listen_address=:6000
causal_clustering.raft_listen_address=:7000
```

**Read Replica Configuration:**
```properties
# neo4j.conf (Read Replica)

# Server mode
dbms.mode=READ_REPLICA

# Connect to core servers
causal_clustering.initial_discovery_members=core1:5000,core2:5000,core3:5000

# Ports
server.bolt.listen_address=:7687
server.http.listen_address=:7474
```

### Connection Routing

**Application Connection:**
```python
from neo4j import GraphDatabase

# Use neo4j:// scheme for routing
driver = GraphDatabase.driver(
    "neo4j://loadbalancer:7687",
    auth=("neo4j", "password")
)

# Automatic read/write routing
with driver.session() as session:
    # Write query (routes to Core servers)
    session.execute_write(lambda tx: tx.run(
        "CREATE (p:Person {name: $name})",
        name="Alice"
    ))

    # Read query (routes to Read Replicas)
    result = session.execute_read(lambda tx: tx.run(
        "MATCH (p:Person) RETURN p.name"
    ))
```

### Load Balancing

**HAProxy Configuration:**
```haproxy
# haproxy.cfg
frontend neo4j_bolt
    bind *:7687
    mode tcp
    default_backend neo4j_cores

backend neo4j_cores
    mode tcp
    balance roundrobin
    option tcp-check
    server core1 core1:7687 check
    server core2 core2:7687 check
    server core3 core3:7687 check

backend neo4j_replicas
    mode tcp
    balance roundrobin
    server replica1 replica1:7687 check
    server replica2 replica2:7687 check
```

### Monitoring Cluster Health

**Check Cluster Status:**
```cypher
// Show cluster members
SHOW SERVERS

// Show database status
SHOW DATABASES

// Check cluster role
CALL dbms.cluster.role()

// Check routing table
CALL dbms.cluster.routing.getRoutingTable({}, "neo4j")
```

---

## 12. Security Best Practices

### Authentication

**Native Authentication:**
```cypher
// Create user
CREATE USER alice SET PASSWORD 'securePassword123!' CHANGE NOT REQUIRED

// Change password
ALTER USER alice SET PASSWORD 'newPassword456!'

// List users
SHOW USERS

// Drop user
DROP USER alice
```

**LDAP/Active Directory (Enterprise):**
```properties
# neo4j.conf
dbms.security.authentication_providers=ldap
dbms.security.authorization_providers=ldap

# LDAP configuration
dbms.security.ldap.host=ldap://ldap.example.com:389
dbms.security.ldap.authentication.user_dn_template=cn={0},ou=users,dc=example,dc=com
dbms.security.ldap.authorization.user_search_base=ou=users,dc=example,dc=com
dbms.security.ldap.authorization.group_membership_attributes=memberOf
```

### Authorization

**Role-Based Access Control:**
```cypher
// Create role
CREATE ROLE reader

// Grant read access
GRANT MATCH {*} ON GRAPH neo4j NODES * TO reader
GRANT MATCH {*} ON GRAPH neo4j RELATIONSHIPS * TO reader

// Create admin role
CREATE ROLE admin

// Grant all privileges
GRANT ALL DATABASE PRIVILEGES ON DATABASE neo4j TO admin
GRANT ALL DBMS PRIVILEGES TO admin

// Assign role to user
GRANT ROLE reader TO alice
GRANT ROLE admin TO bob

// Show privileges
SHOW PRIVILEGES
SHOW USER PRIVILEGES
```

**Fine-Grained Access Control:**
```cypher
// Grant specific label access
GRANT TRAVERSE ON GRAPH neo4j NODES Person TO reader
GRANT READ {name, email} ON GRAPH neo4j NODES Person TO reader

// Deny sensitive data
DENY READ {ssn, creditCard} ON GRAPH neo4j NODES Person TO reader

// Grant write access
GRANT CREATE ON GRAPH neo4j TO writer
GRANT SET PROPERTY {*} ON GRAPH neo4j NODES * TO writer
```

### Encryption

**TLS/SSL Configuration:**
```properties
# neo4j.conf

# Enable TLS for Bolt
server.bolt.tls_level=REQUIRED

# Certificate configuration
dbms.ssl.policy.bolt.enabled=true
dbms.ssl.policy.bolt.base_directory=certificates/bolt
dbms.ssl.policy.bolt.private_key=private.key
dbms.ssl.policy.bolt.public_certificate=public.crt
dbms.ssl.policy.bolt.client_auth=NONE
```

**Generate Certificates:**
```bash
# Generate self-signed certificate
openssl req -newkey rsa:2048 -nodes \
  -keyout private.key \
  -x509 -days 365 -out public.crt \
  -subj "/CN=neo4j.example.com"

# Place in Neo4j certificates directory
mkdir -p $NEO4J_HOME/certificates/bolt
cp private.key public.crt $NEO4J_HOME/certificates/bolt/
```

**Client Connection:**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j+s://neo4j.example.com:7687",  # +s for secure
    auth=("neo4j", "password"),
    encrypted=True,
    trust="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
)
```

### Network Security

**Firewall Rules:**
```bash
# Core servers only
# Allow Bolt (7687)
# Allow HTTP/HTTPS (7474/7473)
# Allow cluster ports (5000, 6000, 7000) between core servers only

# Read replicas
# Allow Bolt (7687)
# Allow HTTP/HTTPS (7474/7473)
```

**Configuration:**
```properties
# Bind to specific interface
server.default_listen_address=10.0.1.10

# Disable remote shell (for production)
server.jvm.additional=-Dneo4j.ext.udc.enabled=false
```

---

## 13. Monitoring and Troubleshooting

### Metrics and Monitoring

**JMX Metrics:**
```properties
# neo4j.conf
server.metrics.enabled=true
server.metrics.jmx.enabled=true
server.metrics.prefix=neo4j

# Prometheus metrics (Enterprise)
server.metrics.prometheus.enabled=true
server.metrics.prometheus.endpoint=0.0.0.0:2004
```

**Key Metrics to Monitor:**
```
Performance:
- Query execution time
- Transaction throughput
- Page cache hit ratio
- GC pause time

Resources:
- Heap usage
- Page cache usage
- Disk I/O
- Network I/O

Cluster:
- Cluster member status
- Replication lag
- Transaction propagation time
```

**Prometheus + Grafana:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'neo4j'
    static_configs:
      - targets:
        - core1:2004
        - core2:2004
        - core3:2004
        - replica1:2004
```

### Query Performance

**Slow Query Logging:**
```properties
# neo4j.conf
db.logs.query.enabled=true
db.logs.query.threshold=1s
db.logs.query.parameter_logging_enabled=true
db.logs.query.plan_description_enabled=true
```

**Analyze Slow Queries:**
```cypher
// Use PROFILE to analyze
PROFILE
MATCH (p:Person)-[:KNOWS*2]->(fof)
WHERE p.name = "Alice"
RETURN fof.name

// Look for:
// - High db hits
// - AllNodesScan (bad)
// - NodeByLabelScan (suboptimal)
// - NodeIndexSeek (good)
```

### Database Health Checks

**System Procedures:**
```cypher
// Check database health
CALL dbms.queryJmx("org.neo4j:*")

// Show current queries
CALL dbms.listQueries()

// Kill long-running query
CALL dbms.killQuery("query-123")

// Show connections
CALL dbms.listConnections()

// Terminate connection
CALL dbms.killConnection("bolt-456")

// Check page cache
CALL dbms.listPools()
```

### Logging

**Log Configuration:**
```properties
# neo4j.conf

# Log level
server.logs.user.level=INFO
server.logs.debug.level=INFO

# Log rotation
server.logs.user.rotation.size=20m
server.logs.user.rotation.keep_number=7

# Query logging
db.logs.query.enabled=true
db.logs.query.threshold=5s
```

**Log Locations:**
```
$NEO4J_HOME/logs/
├── debug.log          # Internal debug information
├── neo4j.log         # General application log
├── query.log         # Slow query log
└── security.log      # Authentication/authorization events
```

### Common Issues

**Out of Memory:**
```properties
# Increase heap size
server.memory.heap.initial_size=4G
server.memory.heap.max_size=8G

# Increase page cache
server.memory.pagecache.size=12G
```

**Slow Queries:**
```cypher
// Create missing indexes
CREATE INDEX person_name FOR (p:Person) ON (p.name)

// Avoid variable length paths without bound
// ❌ Bad
MATCH (a)-[:KNOWS*]->(b)

// ✅ Good
MATCH (a)-[:KNOWS*1..5]->(b)
```

**Deadlocks:**
```python
# Implement retry logic
from neo4j.exceptions import TransientError
import time

def retry_transaction(tx_function, max_retries=3):
    for attempt in range(max_retries):
        try:
            return tx_function()
        except TransientError:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
                continue
            raise
```

---

## 14. Backup and Recovery

### Online Backup (Enterprise)

**Backup Configuration:**
```properties
# neo4j.conf
server.backup.enabled=true
server.backup.listen_address=0.0.0.0:6362
```

**Create Backup:**
```bash
# Full backup
neo4j-admin database backup neo4j \
  --to-path=/backups/neo4j-full-$(date +%Y%m%d) \
  --backup-type=full

# Incremental backup
neo4j-admin database backup neo4j \
  --to-path=/backups/neo4j-incremental \
  --backup-type=differential

# Backup from remote server
neo4j-admin database backup neo4j \
  --from=neo4j://core1:6362 \
  --to-path=/backups/neo4j-backup
```

### Database Dump (All Editions)

**Dump Database:**
```bash
# Stop database first
neo4j stop

# Create dump
neo4j-admin database dump neo4j \
  --to=/backups/neo4j-dump-$(date +%Y%m%d-%H%M%S).dump

# Compress dump
neo4j-admin database dump neo4j \
  --to=/backups/neo4j.dump.gz \
  --compress

# Restart database
neo4j start
```

### Restore Database

**Restore from Dump:**
```bash
# Stop database
neo4j stop

# Delete existing database (if exists)
rm -rf $NEO4J_HOME/data/databases/neo4j
rm -rf $NEO4J_HOME/data/transactions/neo4j

# Restore from dump
neo4j-admin database load neo4j \
  --from=/backups/neo4j.dump \
  --overwrite-destination=true

# Start database
neo4j start
```

**Restore from Backup:**
```bash
neo4j-admin database restore neo4j \
  --from-path=/backups/neo4j-backup \
  --overwrite-destination=true
```

### Backup Strategies

**Full + Incremental:**
```bash
#!/bin/bash
# Backup script

BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y%m%d)

# Sunday: Full backup
if [ $(date +%u) -eq 7 ]; then
    neo4j-admin database backup neo4j \
      --to-path=$BACKUP_DIR/full-$DATE \
      --backup-type=full
else
    # Weekday: Incremental
    neo4j-admin database backup neo4j \
      --to-path=$BACKUP_DIR/incremental \
      --backup-type=differential
fi

# Retention: Keep last 30 days
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;
```

### Point-in-Time Recovery

**Transaction Log Backups:**
```bash
# Archive transaction logs
cp -r $NEO4J_HOME/data/transactions/neo4j \
  /backups/txlogs/$(date +%Y%m%d-%H%M%S)

# Retention
db.tx_log.retention_policy=7 days
```

### Disaster Recovery

**Multi-Region Replication:**
```
Primary Region:           Secondary Region:
┌─────────────┐          ┌─────────────┐
│   Cluster   │          │   Cluster   │
│  (Active)   │  ───────>│  (Standby)  │
│             │  Async   │             │
│ Core Servers│  Repl    │ Core Servers│
└─────────────┘          └─────────────┘

Strategy:
1. Continuous backup to secondary region
2. Regular restore tests
3. Documented failover procedure
```

---

## 15. Application Integration

### Python Driver

**Installation:**
```bash
pip install neo4j
```

**Basic Usage:**
```python
from neo4j import GraphDatabase
import logging

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def write_transaction(self, query, parameters=None):
        with self.driver.session() as session:
            return session.execute_write(
                lambda tx: tx.run(query, parameters).data()
            )

# Usage
conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "password")

# Create person
conn.write_transaction(
    "CREATE (p:Person {name: $name, age: $age}) RETURN p",
    {"name": "Alice", "age": 30}
)

# Query persons
persons = conn.query(
    "MATCH (p:Person) WHERE p.age > $age RETURN p.name, p.age",
    {"age": 25}
)

conn.close()
```

**Transaction Functions:**
```python
def create_friendship(tx, person1, person2):
    result = tx.run("""
        MATCH (a:Person {name: $person1})
        MATCH (b:Person {name: $person2})
        MERGE (a)-[r:KNOWS]-(b)
        RETURN r
    """, person1=person1, person2=person2)
    return result.single()[0]

# Execute with retry logic
with driver.session() as session:
    friendship = session.execute_write(
        create_friendship,
        "Alice",
        "Bob"
    )
```

### JavaScript/TypeScript Driver

**Installation:**
```bash
npm install neo4j-driver
```

**Usage:**
```typescript
import neo4j, { Driver, Session } from 'neo4j-driver';

class Neo4jService {
    private driver: Driver;

    constructor(uri: string, user: string, password: string) {
        this.driver = neo4j.driver(
            uri,
            neo4j.auth.basic(user, password)
        );
    }

    async close(): Promise<void> {
        await this.driver.close();
    }

    async createPerson(name: string, age: number) {
        const session = this.driver.session();
        try {
            const result = await session.executeWrite(async tx => {
                return await tx.run(
                    'CREATE (p:Person {name: $name, age: $age}) RETURN p',
                    { name, age }
                );
            });
            return result.records[0].get('p').properties;
        } finally {
            await session.close();
        }
    }

    async findPersons(minAge: number) {
        const session = this.driver.session();
        try {
            const result = await session.executeRead(async tx => {
                return await tx.run(
                    'MATCH (p:Person) WHERE p.age > $minAge RETURN p',
                    { minAge }
                );
            });
            return result.records.map(record =>
                record.get('p').properties
            );
        } finally {
            await session.close();
        }
    }
}

// Usage
const neo4j = new Neo4jService(
    'neo4j://localhost:7687',
    'neo4j',
    'password'
);

await neo4j.createPerson('Alice', 30);
const persons = await neo4j.findPersons(25);
await neo4j.close();
```

### Java Driver

**Maven Dependency:**
```xml
<dependency>
    <groupId>org.neo4j.driver</groupId>
    <artifactId>neo4j-java-driver</artifactId>
    <version>5.15.0</version>
</dependency>
```

**Usage:**
```java
import org.neo4j.driver.*;

public class Neo4jConnection implements AutoCloseable {
    private final Driver driver;

    public Neo4jConnection(String uri, String user, String password) {
        driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password));
    }

    @Override
    public void close() {
        driver.close();
    }

    public void createPerson(String name, int age) {
        try (Session session = driver.session()) {
            session.executeWrite(tx -> {
                return tx.run(
                    "CREATE (p:Person {name: $name, age: $age}) RETURN p",
                    Values.parameters("name", name, "age", age)
                ).single();
            });
        }
    }

    public List<Map<String, Object>> findPersons(int minAge) {
        try (Session session = driver.session()) {
            return session.executeRead(tx -> {
                var result = tx.run(
                    "MATCH (p:Person) WHERE p.age > $minAge RETURN p",
                    Values.parameters("minAge", minAge)
                );
                return result.list(record ->
                    record.get("p").asNode().asMap()
                );
            });
        }
    }
}

// Usage
try (Neo4jConnection conn = new Neo4jConnection(
        "neo4j://localhost:7687", "neo4j", "password")) {
    conn.createPerson("Alice", 30);
    List<Map<String, Object>> persons = conn.findPersons(25);
}
```

### Connection Pooling

**Configuration:**
```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "neo4j://localhost:7687",
    auth=("neo4j", "password"),
    max_connection_pool_size=50,
    connection_acquisition_timeout=60,
    max_transaction_retry_time=30,
    encrypted=False
)
```

---

## 16. Production Deployment

### Docker Deployment

**Docker Compose:**
```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15-enterprise
    container_name: neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7473:7473"  # HTTPS
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/your-secure-password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes

      # Memory settings
      - NEO4J_server_memory_heap_initial__size=2G
      - NEO4J_server_memory_heap_max__size=4G
      - NEO4J_server_memory_pagecache_size=4G

      # Performance tuning
      - NEO4J_db_transaction_timeout=60s
      - NEO4J_dbms_connector_bolt_thread__pool__min__size=5
      - NEO4J_dbms_connector_bolt_thread__pool__max__size=400

      # Plugins
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*,gds.*
      - NEO4JLABS_PLUGINS=["apoc", "graph-data-science"]

    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/import
      - neo4j_plugins:/plugins

    healthcheck:
      test: ["CMD-SHELL", "cypher-shell -u neo4j -p your-secure-password 'RETURN 1'"]
      interval: 30s
      timeout: 10s
      retries: 3

    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

**Cluster with Docker:**
```yaml
version: '3.8'

services:
  core1:
    image: neo4j:5.15-enterprise
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_initial__discovery__members=core1:5000,core2:5000,core3:5000
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__formation=3
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - core1_data:/data
    networks:
      - neo4j-cluster

  core2:
    image: neo4j:5.15-enterprise
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_initial__discovery__members=core1:5000,core2:5000,core3:5000
    ports:
      - "7475:7474"
      - "7688:7687"
    volumes:
      - core2_data:/data
    networks:
      - neo4j-cluster

  core3:
    image: neo4j:5.15-enterprise
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_initial__discovery__members=core1:5000,core2:5000,core3:5000
    ports:
      - "7476:7474"
      - "7689:7687"
    volumes:
      - core3_data:/data
    networks:
      - neo4j-cluster

volumes:
  core1_data:
  core2_data:
  core3_data:

networks:
  neo4j-cluster:
```

### Kubernetes Deployment

**StatefulSet:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j-core
spec:
  serviceName: neo4j
  replicas: 3
  selector:
    matchLabels:
      app: neo4j
      role: core
  template:
    metadata:
      labels:
        app: neo4j
        role: core
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.15-enterprise
        env:
        - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
          value: "yes"
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: neo4j-auth
              key: credentials
        - name: NEO4J_dbms_mode
          value: "CORE"
        - name: NEO4J_causal__clustering_initial__discovery__members
          value: "neo4j-core-0.neo4j:5000,neo4j-core-1.neo4j:5000,neo4j-core-2.neo4j:5000"
        ports:
        - containerPort: 7474
          name: http
        - containerPort: 7687
          name: bolt
        - containerPort: 5000
          name: discovery
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi
```

### Production Configuration

**Optimized neo4j.conf:**
```properties
# Memory (for 32GB server)
server.memory.heap.initial_size=8G
server.memory.heap.max_size=8G
server.memory.pagecache.size=16G

# Transactions
db.transaction.timeout=300s
db.transaction.concurrent.maximum=1000

# Query limits
db.memory.transaction.max_size=1GB
db.query.timeout=30s

# Logging
server.logs.user.level=INFO
db.logs.query.enabled=true
db.logs.query.threshold=5s
server.logs.security.enabled=true

# Connections
server.bolt.thread_pool_min_size=5
server.bolt.thread_pool_max_size=400

# Checkpointing
db.checkpoint.interval.time=15m
db.checkpoint.interval.tx=100000

# Performance
db.index_sampling.background_enabled=true
db.index_sampling.sample_size_limit=1000000
```

---

## 17. Scaling Strategies

### Vertical Scaling

**Resource Allocation:**
```
Small (Dev):           Medium (Production):    Large (Enterprise):
- 4 CPU cores          - 8-16 CPU cores        - 32+ CPU cores
- 8GB RAM              - 32GB RAM              - 128GB+ RAM
- 100GB SSD            - 500GB NVMe SSD        - 2TB+ NVMe SSD

Memory Split:
- Heap: 25% of RAM
- Page Cache: 50-60% of RAM
- OS: 15-25% of RAM
```

### Horizontal Scaling

**Read Scaling:**
```
Add Read Replicas for read-heavy workloads:

Core Servers (3)      →  Handles all writes
Read Replicas (3-10)  →  Distributes read load

Benefits:
✓ Linear read scaling
✓ Geographic distribution
✓ Reduced latency for reads
✓ Isolation of analytical queries
```

**Sharding (Manual):**
```cypher
// Functional sharding by domain
// Shard 1: User data
CREATE DATABASE users

// Shard 2: Product data
CREATE DATABASE products

// Shard 3: Analytics
CREATE DATABASE analytics

// Application routes queries to appropriate database
```

### Caching Strategies

**Application-Level Cache:**
```python
from functools import lru_cache
import redis

class Neo4jService:
    def __init__(self):
        self.neo4j = GraphDatabase.driver(...)
        self.redis = redis.Redis(host='localhost', port=6379)

    def get_person(self, email):
        # Check cache first
        cached = self.redis.get(f"person:{email}")
        if cached:
            return json.loads(cached)

        # Query Neo4j
        with self.neo4j.session() as session:
            result = session.run(
                "MATCH (p:Person {email: $email}) RETURN p",
                email=email
            ).single()

            if result:
                person = dict(result['p'])
                # Cache for 5 minutes
                self.redis.setex(
                    f"person:{email}",
                    300,
                    json.dumps(person)
                )
                return person
```

### Query Optimization

**Denormalization:**
```cypher
// Store precomputed values
MATCH (p:Person)
OPTIONAL MATCH (p)-[:KNOWS]->(friend)
WITH p, count(friend) AS friendCount
SET p.friendCount = friendCount

// Fast retrieval
MATCH (p:Person)
WHERE p.friendCount > 100
RETURN p
ORDER BY p.friendCount DESC
LIMIT 10
```

**Materialized Views:**
```cypher
// Create summary nodes
MATCH (p:Person)-[:PURCHASED]->(product:Product)
WITH product, count(p) AS buyers, sum(purchase.amount) AS revenue
MERGE (s:ProductStats {productId: product.id})
SET s.totalBuyers = buyers,
    s.totalRevenue = revenue,
    s.lastUpdated = timestamp()
```

---

## 18. Common Use Cases

### Social Network

**Data Model:**
```cypher
// Users
CREATE (alice:User {id: 1, name: "Alice", email: "alice@example.com"})
CREATE (bob:User {id: 2, name: "Bob"})
CREATE (carol:User {id: 3, name: "Carol"})

// Friendships
CREATE (alice)-[:FRIENDS_WITH {since: date("2020-01-01")}]->(bob)
CREATE (bob)-[:FRIENDS_WITH]->(carol)

// Posts
CREATE (post:Post {id: 1, content: "Hello World!", created: datetime()})
CREATE (alice)-[:POSTED]->(post)

// Likes
CREATE (bob)-[:LIKES {timestamp: datetime()}]->(post)

// Comments
CREATE (comment:Comment {content: "Nice post!"})
CREATE (bob)-[:COMMENTED]->(comment)-[:ON]->(post)
```

**Queries:**
```cypher
// Friend recommendations (friends of friends)
MATCH (me:User {id: 1})-[:FRIENDS_WITH]-()-[:FRIENDS_WITH]-(fof:User)
WHERE NOT (me)-[:FRIENDS_WITH]-(fof) AND me <> fof
RETURN fof.name, count(*) AS mutualFriends
ORDER BY mutualFriends DESC
LIMIT 10

// News feed
MATCH (me:User {id: 1})-[:FRIENDS_WITH]-(friend)-[:POSTED]->(post)
OPTIONAL MATCH (post)<-[:LIKES]-(liker)
OPTIONAL MATCH (post)<-[:ON]-(comment)
RETURN post, friend.name, count(DISTINCT liker) AS likes, count(DISTINCT comment) AS comments
ORDER BY post.created DESC
LIMIT 20
```

### Recommendation Engine

**Collaborative Filtering:**
```cypher
// "Users who liked X also liked Y"
MATCH (me:User {id: 1})-[:LIKES]->(product:Product)
MATCH (product)<-[:LIKES]-(other:User)-[:LIKES]->(recommendation:Product)
WHERE NOT (me)-[:LIKES]->(recommendation)
RETURN recommendation.name, count(*) AS score
ORDER BY score DESC
LIMIT 10

// Personalized recommendations with weights
MATCH (me:User {id: 1})-[r:LIKES]->(p:Product)
WITH me, avg(r.rating) AS myAvgRating
MATCH (me)-[r1:LIKES]->(p:Product)<-[r2:LIKES]-(other:User)
WHERE abs(r1.rating - myAvgRating) < 1.0
MATCH (other)-[r3:LIKES]->(rec:Product)
WHERE NOT (me)-[:LIKES]->(rec)
RETURN rec.name,
       sum(r3.rating * abs(r1.rating - r2.rating)) AS score
ORDER BY score DESC
LIMIT 10
```

### Knowledge Graph

**Data Model:**
```cypher
// Entities
CREATE (einstein:Person {name: "Albert Einstein", born: 1879})
CREATE (relativity:Theory {name: "General Relativity"})
CREATE (physics:Field {name: "Physics"})
CREATE (nobel:Award {name: "Nobel Prize", year: 1921})

// Relationships
CREATE (einstein)-[:DEVELOPED]->(relativity)
CREATE (relativity)-[:IN_FIELD]->(physics)
CREATE (einstein)-[:RECEIVED]->(nobel)
CREATE (nobel)-[:FOR_WORK_IN]->(physics)
```

**Queries:**
```cypher
// Find connections between entities
MATCH path = (a:Person {name: "Albert Einstein"})-[*1..4]-(b:Person {name: "Isaac Newton"})
RETURN path
ORDER BY length(path)
LIMIT 1

// Explore related concepts
MATCH (topic:Topic {name: "Machine Learning"})-[*1..3]-(related)
RETURN DISTINCT labels(related) AS type, related.name AS name,
       length(path) AS degrees
ORDER BY degrees
LIMIT 20
```

### Fraud Detection

**Pattern Detection:**
```cypher
// Detect suspicious transaction patterns
// Multiple accounts sharing phone/email
MATCH (a1:Account)-[:HAS_PHONE]->(phone:Phone)<-[:HAS_PHONE]-(a2:Account)
WHERE a1 <> a2
RETURN a1, a2, phone

// Transaction rings (circular money flow)
MATCH (a:Account)-[:SENT_TO*4..6]->(a)
RETURN a, length(path) AS ringSize

// Velocity checks (multiple transactions in short time)
MATCH (account:Account)-[t:TRANSACTION]->(merchant)
WHERE t.timestamp > datetime() - duration('PT1H')
WITH account, count(t) AS txCount, sum(t.amount) AS totalAmount
WHERE txCount > 10 OR totalAmount > 10000
RETURN account, txCount, totalAmount
```

### Access Control

**Hierarchical Permissions:**
```cypher
// Organization hierarchy
CREATE (company:Org {name: "Acme Corp"})
CREATE (eng:Dept {name: "Engineering"})
CREATE (sales:Dept {name: "Sales"})
CREATE (team1:Team {name: "Backend Team"})

CREATE (eng)-[:PART_OF]->(company)
CREATE (sales)-[:PART_OF]->(company)
CREATE (team1)-[:PART_OF]->(eng)

// Users and roles
CREATE (alice:User {email: "alice@acme.com"})
CREATE (bob:User {email: "bob@acme.com"})

CREATE (alice)-[:MEMBER_OF]->(team1)
CREATE (alice)-[:HAS_ROLE {role: "admin"}]->(eng)

// Resources
CREATE (doc:Document {id: 1, title: "Secret Plans"})
CREATE (eng)-[:CAN_ACCESS {permission: "read"}]->(doc)

// Check access
MATCH (user:User {email: "alice@acme.com"})-[:MEMBER_OF|HAS_ROLE*]->(org)
MATCH (org)-[:CAN_ACCESS*]->(resource:Document {id: 1})
RETURN resource
```

---

## 19. Comparison with Other Databases

### Neo4j vs. Relational Databases

| Feature | Neo4j | PostgreSQL/MySQL |
|---------|-------|------------------|
| **Data Model** | Property graph | Tables with foreign keys |
| **Relationships** | First-class citizens | JOINs required |
| **Traversal Performance** | O(1) - constant time | O(n log n) - index lookups |
| **Schema** | Flexible/schema-optional | Rigid schema required |
| **Query Language** | Cypher (declarative) | SQL |
| **Best For** | Connected data, many relationships | Transactional data, reporting |
| **Worst For** | Simple CRUD, analytics | Deep traversals, graph queries |

**Example:**
```sql
-- SQL: Find friends of friends (requires 2 JOINs)
SELECT DISTINCT f2.name
FROM users u1
JOIN friendships f1 ON u1.id = f1.user_id
JOIN friendships f2 ON f1.friend_id = f2.user_id
WHERE u1.name = 'Alice';
```

```cypher
-- Cypher: Same query (natural expression)
MATCH (alice:Person {name: "Alice"})-[:KNOWS]->()-[:KNOWS]->(fof)
RETURN DISTINCT fof.name
```

### Neo4j vs. Document Databases

| Feature | Neo4j | MongoDB |
|---------|-------|---------|
| **Relationships** | Native, bidirectional | References or embedding |
| **Queries** | Graph traversals | Aggregation pipeline |
| **Joins** | Built-in | $lookup (slower) |
| **Data Duplication** | Minimal | Often required |
| **Schema Evolution** | Easy | Easy |
| **Best For** | Graph problems | Hierarchical documents |

### Neo4j vs. Other Graph Databases

| Feature | Neo4j | Amazon Neptune | TigerGraph |
|---------|-------|----------------|------------|
| **Model** | Property graph | Property + RDF | Property graph |
| **Query Language** | Cypher | Cypher + SPARQL + Gremlin | GSQL |
| **ACID** | Full | Full | Full |
| **Clustering** | Causal clustering | Managed | Distributed |
| **Deployment** | Self-hosted/Cloud | AWS only | Self-hosted/Cloud |
| **Performance** | Excellent | Very good | Excellent (large graphs) |
| **Ecosystem** | Mature | Growing | Specialized |

---

## 20. Migration Strategies

### From Relational Database

**Strategy:**
```
1. Identify entities → Convert to nodes
2. Identify foreign keys → Convert to relationships
3. Identify join tables → Convert to relationships with properties
4. Migrate data
5. Optimize schema
```

**Example Migration:**
```sql
-- Original SQL schema
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE friendships (
    user_id INT,
    friend_id INT,
    since DATE,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (friend_id) REFERENCES users(id)
);
```

**Migration Script:**
```cypher
// Load users from CSV export
LOAD CSV WITH HEADERS FROM 'file:///users.csv' AS row
CREATE (u:User {
    id: toInteger(row.id),
    name: row.name,
    email: row.email
})

// Load friendships
LOAD CSV WITH HEADERS FROM 'file:///friendships.csv' AS row
MATCH (u1:User {id: toInteger(row.user_id)})
MATCH (u2:User {id: toInteger(row.friend_id)})
CREATE (u1)-[:FRIENDS_WITH {since: date(row.since)}]->(u2)

// Create indexes
CREATE INDEX user_id FOR (u:User) ON (u.id)
CREATE CONSTRAINT user_email_unique FOR (u:User) REQUIRE u.email IS UNIQUE
```

### From MongoDB

**Migration Example:**
```javascript
// MongoDB document
{
  "_id": "user123",
  "name": "Alice",
  "email": "alice@example.com",
  "friends": [
    {"userId": "user456", "since": "2020-01-01"},
    {"userId": "user789", "since": "2021-05-15"}
  ],
  "posts": [
    {"id": "post1", "content": "Hello", "likes": 42}
  ]
}
```

**Convert to Graph:**
```cypher
// User node
CREATE (u:User {
    id: "user123",
    name: "Alice",
    email: "alice@example.com"
})

// Friends as relationships
MATCH (u:User {id: "user123"})
MATCH (friend:User {id: "user456"})
CREATE (u)-[:FRIENDS_WITH {since: date("2020-01-01")}]->(friend)

// Posts as separate nodes
MATCH (u:User {id: "user123"})
CREATE (p:Post {id: "post1", content: "Hello", likes: 42})
CREATE (u)-[:POSTED]->(p)
```

### Zero-Downtime Migration

**Dual-Write Strategy:**
```python
class DualWriteService:
    def __init__(self, postgres_conn, neo4j_driver):
        self.pg = postgres_conn
        self.neo4j = neo4j_driver
        self.migration_complete = False

    def create_user(self, user_data):
        # Write to primary (PostgreSQL)
        pg_result = self.pg.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id",
            (user_data['name'], user_data['email'])
        )
        user_id = pg_result[0]['id']

        # Write to secondary (Neo4j)
        try:
            with self.neo4j.session() as session:
                session.execute_write(lambda tx: tx.run(
                    "CREATE (u:User {id: $id, name: $name, email: $email})",
                    id=user_id, **user_data
                ))
        except Exception as e:
            # Log error but don't fail
            logger.error(f"Neo4j write failed: {e}")

        return user_id

    def get_user(self, user_id):
        # Read from Neo4j after migration complete
        if self.migration_complete:
            with self.neo4j.session() as session:
                result = session.run(
                    "MATCH (u:User {id: $id}) RETURN u",
                    id=user_id
                )
                return dict(result.single()['u'])

        # Otherwise read from PostgreSQL
        return self.pg.execute(
            "SELECT * FROM users WHERE id = %s",
            (user_id,)
        )[0]
```

---

## 21. Production Checklist

### Pre-Deployment

**Infrastructure:**
- [ ] Sizing: CPU, RAM, storage capacity planned
- [ ] Clustering: Multi-node setup configured
- [ ] Network: Firewall rules, load balancer configured
- [ ] Monitoring: Metrics collection setup (Prometheus/Grafana)
- [ ] Backups: Automated backup strategy implemented
- [ ] Disaster recovery: DR plan documented and tested

**Security:**
- [ ] Authentication: Native/LDAP configured
- [ ] Authorization: RBAC roles defined
- [ ] TLS/SSL: Certificates configured for Bolt/HTTPS
- [ ] Network: Ports restricted, VPN/bastion setup
- [ ] Secrets: Passwords stored securely (not in config)
- [ ] Audit logging: Security logs enabled

**Database Configuration:**
- [ ] Memory: Heap and page cache optimized
- [ ] Indexes: All frequently queried properties indexed
- [ ] Constraints: Uniqueness and existence constraints created
- [ ] Plugins: APOC and GDS installed if needed
- [ ] Query limits: Timeouts and memory limits configured
- [ ] Logging: Slow query logging enabled

### Post-Deployment

**Verification:**
- [ ] Health checks: All nodes responding
- [ ] Replication: Cluster members in sync
- [ ] Performance: Query response times acceptable
- [ ] Backups: First backup completed successfully
- [ ] Monitoring: Alerts firing correctly
- [ ] Documentation: Runbooks updated

**Operations:**
- [ ] Backup schedule: Daily incrementals, weekly fulls
- [ ] Log rotation: Configured and tested
- [ ] Maintenance window: Scheduled and communicated
- [ ] On-call: Team trained and rotation setup
- [ ] Incident response: Procedures documented
- [ ] Capacity planning: Growth trends monitored

### Performance Optimization

**Query Performance:**
- [ ] Slow queries identified and optimized
- [ ] Indexes created for frequent patterns
- [ ] Variable length paths bounded
- [ ] EXPLAIN/PROFILE used for complex queries
- [ ] Batch operations use UNWIND
- [ ] Unnecessary DISTINCT removed

**Configuration:**
- [ ] Page cache sized correctly (50% RAM)
- [ ] Heap sized appropriately (25% RAM)
- [ ] Transaction timeout configured
- [ ] Connection pool sized
- [ ] Checkpoint intervals tuned
- [ ] Query memory limits set

**Monitoring Metrics:**
```cypher
// Key metrics to track:
// - Query execution time (p95, p99)
// - Transaction throughput (tx/sec)
// - Page cache hit ratio (>95% good)
// - Store file sizes
// - Cluster lag (if applicable)
// - Connection pool usage
// - GC pause time
```

### Ongoing Maintenance

**Daily:**
- [ ] Check cluster health
- [ ] Review slow query logs
- [ ] Monitor disk space
- [ ] Verify backups completed

**Weekly:**
- [ ] Review performance metrics
- [ ] Analyze growth trends
- [ ] Check for Neo4j updates
- [ ] Review security logs

**Monthly:**
- [ ] Test backup restore
- [ ] Review and optimize indexes
- [ ] Capacity planning review
- [ ] Update documentation

**Quarterly:**
- [ ] Disaster recovery drill
- [ ] Performance tuning review
- [ ] Security audit
- [ ] Version upgrade planning

---

## 22. Deployment Checklist

### Build and Configuration
- [ ] Neo4j version pinned and documented
- [ ] JVM heap size configured (`dbms.memory.heap.initial_size` / `dbms.memory.heap.max_size`)
- [ ] Page cache sized appropriately (`dbms.memory.pagecache.size`)
- [ ] Transaction log retention configured
- [ ] Bolt and HTTP connectors configured with appropriate bind addresses
- [ ] `neo4j-admin memrec` run for memory recommendations

### Testing
- [ ] All Cypher queries profiled with `PROFILE` and `EXPLAIN`
- [ ] Index usage verified for all query patterns
- [ ] Load testing completed with production-scale graph data
- [ ] Cluster failover tested (Enterprise)
- [ ] Backup and restore procedure verified
- [ ] Import pipeline tested with `neo4j-admin database import`

### Security
- [ ] Default `neo4j` password changed
- [ ] Authentication enabled (`dbms.security.auth_enabled=true`)
- [ ] Role-based access control configured (Enterprise)
- [ ] TLS/SSL enabled for Bolt and HTTPS connectors
- [ ] Network access restricted to required ports only (7474, 7687)
- [ ] Audit logging enabled (Enterprise)
- [ ] Property-level security configured where needed

### Agent Workflow
- [ ] Schema constraints and indexes defined in migration scripts
- [ ] Graph data model documented with node labels and relationship types
- [ ] Monitoring alerts configured (query latency, heap usage, page cache hit ratio)
- [ ] Automated backups scheduled with `neo4j-admin database dump`
- [ ] Runbooks documented for cluster recovery and rebalancing

---

## 23. Why This Configuration Works

**Native Graph Storage**:
- Index-free adjacency means traversals follow physical pointers rather than performing index lookups, delivering constant-time relationship traversal regardless of total graph size.

**Cypher Query Language**:
- Pattern-matching syntax maps directly to how developers think about connected data, making complex traversals readable and maintainable while the query planner optimizes execution.

**ACID Transactions on Graphs**:
- Full transactional support ensures data integrity during multi-node and multi-relationship mutations, critical for applications where relationship consistency matters.

**Flexible Schema with Constraints**:
- Schema-optional design allows rapid iteration on data models, while uniqueness and existence constraints enforce data quality where needed without rigid table definitions.

**Causal Clustering (Enterprise)**:
- Raft-based consensus with read replicas provides high availability, horizontal read scaling, and multi-datacenter deployment with causal consistency guarantees.

---

## 24. Quick Reference

### Common Commands

```bash
# Start Neo4j
neo4j start

# Check status
neo4j status

# Open Cypher shell
cypher-shell -u neo4j -p <password>

# Memory recommendations
neo4j-admin server memory-recommendation

# Backup database
neo4j-admin database dump neo4j --to-path=/backup/

# Restore database
neo4j-admin database load neo4j --from-path=/backup/neo4j.dump --overwrite-destination

# Import CSV data
neo4j-admin database import full neo4j --nodes=import/nodes.csv --relationships=import/rels.csv

# Check database info
cypher-shell "CALL dbms.listConfig() YIELD name, value WHERE name CONTAINS 'memory' RETURN name, value;"

# Show indexes and constraints
cypher-shell "SHOW INDEXES;"
cypher-shell "SHOW CONSTRAINTS;"

# Profile a query
cypher-shell "PROFILE MATCH (p:Person)-[:KNOWS]->(f) WHERE p.name = 'Alice' RETURN f.name;"
```

---

## References and Resources

### Official Documentation
- **Neo4j Docs:** https://neo4j.com/docs/
- **Cypher Manual:** https://neo4j.com/docs/cypher-manual/current/
- **Graph Data Science:** https://neo4j.com/docs/graph-data-science/current/
- **APOC:** https://neo4j.com/labs/apoc/

### Learning Resources
- **GraphAcademy:** https://graphacademy.neo4j.com/
- **Cypher Cheat Sheet:** https://neo4j.com/docs/cypher-cheat-sheet/
- **Neo4j Sandbox:** https://sandbox.neo4j.com/

### Community
- **Neo4j Community:** https://community.neo4j.com/
- **Discord:** https://discord.gg/neo4j
- **Stack Overflow:** `[neo4j]` tag
- **GitHub:** https://github.com/neo4j/neo4j

### Tools
- **Neo4j Browser:** Built-in query interface
- **Neo4j Desktop:** Development environment
- **Bloom:** Graph visualization
- **Arrows.app:** Graph modeling tool

---

**Document Maintenance:**
- Review quarterly for Neo4j updates
- Update with new Cypher features
- Add community best practices
- Test examples with latest version

**Last Updated:** February 2026
**Next Review:** May 2026

---

**End of Neo4j Development Guidelines**

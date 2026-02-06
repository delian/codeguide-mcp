# RethinkDB Development Guidelines

Mandatory coding standards and development practices for RethinkDB development. RethinkDB server, ReQL, official drivers (Python, Node, etc.), Web Admin UI.

---

**Agent Profile**: The RethinkDB Expert
**Role**: Senior Real-Time Database Engineer & NoSQL Specialist
**Objective**: Generate production-ready, real-time and reliable applications using RethinkDB and changefeeds.
**Tools**: RethinkDB server, ReQL, official drivers (Python, Node, etc.), Web Admin UI

---

## 1. Core Philosophies: REALTIME-FIRST

The agent must adhere to the **REALTIME-FIRST** principles for every RethinkDB implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **R**ealtime changefeeds: Design for push; subscribe to changes on tables or queries; handle feed lifecycle and reconnects.
- **E**rror handling: Handle connection failures, query errors, and changefeed disconnects; reconnect and backfill when needed.
- **A**vailability: Use clustering and replication; design for failover and multi-datacenter when required.
- **L**anguage (ReQL): Use ReQL composably; prefer server-side operations; avoid N+1 and large in-memory results.
- **T**esting: Test queries and changefeeds; mock connections in unit tests; integration tests against real or test cluster.
- **I**dempotent writes: Use upsert and atomic operations; design for at-least-once delivery in changefeeds.
- **M**odeling: Document-oriented JSON; design indexes for access patterns; use secondary indexes for filters and joins.
- **E**nd-to-end types: Type application data; validate documents at boundaries; handle RethinkDB types (TIME, BINARY, etc.).
**Verified Code**: Agent-generated code MUST use parameterized ReQL, handle connection and feed errors, and pass tests before delivery.

---

## 2. Core Concepts and Architecture

RethinkDB is an open-source, distributed NoSQL database designed for real-time applications. It pushes JSON documents to applications in real-time using changefeeds, making it ideal for collaborative apps, streaming analytics, and live dashboards.

### Real-Time Push Architecture

```
Traditional Database (Pull):
┌─────────────┐         Poll         ┌──────────┐
│ Application │ ──────────────────> │ Database │
│             │ <────────────────── │          │
└─────────────┘      Response       └──────────┘
    ↑ Poll every N seconds (inefficient)

RethinkDB (Push):
┌─────────────┐     Subscribe       ┌──────────┐
│ Application │ ──────────────────> │ RethinkDB│
│             │                      │          │
│             │ <────────────────── │ Changefeed│
│             │    Real-time Push   │          │
└─────────────┘                      └──────────┘
    ↑ Changes pushed instantly
```

### Document-Oriented Model

```json
// Documents are JSON objects
{
    "id": "7644aaf2-9928-4231-aa68-4e65e31bf219",
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "age": 30,
    "address": {
        "city": "San Francisco",
        "state": "CA",
        "zip": "94102"
    },
    "interests": ["programming", "hiking", "photography"],
    "created_at": {"$reql_type$": "TIME", "epoch_time": 1640000000, "timezone": "+00:00"}
}
```

### Distributed Architecture

```
RethinkDB Cluster:

┌────────────────────────────────────────────────────────────┐
│                      Cluster Network                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Server 1           Server 2           Server 3            │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐        │
│  │ Primary  │      │ Replica  │      │ Replica  │        │
│  │ Shard A  │──────│ Shard A  │──────│ Shard A  │        │
│  └──────────┘      └──────────┘      └──────────┘        │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐        │
│  │ Replica  │      │ Primary  │      │ Replica  │        │
│  │ Shard B  │──────│ Shard B  │──────│ Shard B  │        │
│  └──────────┘      └──────────┘      └──────────┘        │
│                                                             │
│  - Automatic sharding across nodes                         │
│  - Replication for high availability                       │
│  - Automatic failover                                      │
│  - No single point of failure                              │
└────────────────────────────────────────────────────────────┘
```

### Key Features

**Changefeeds (Real-Time Push):**
- Subscribe to changes on tables, documents, or queries
- Changes pushed to clients automatically
- Powers real-time applications

**ReQL Query Language:**
- Functional query language
- Composable, chainable operations
- Embedded in host language (Python, JavaScript, etc.)

**Automatic Sharding:**
- Data automatically distributed across cluster
- Transparent to application
- Dynamic resharding

**High Availability:**
- Automatic failover
- Multi-datacenter replication
- Strong consistency by default

**Web Admin UI:**
- Built-in administration interface
- Real-time cluster monitoring
- Visual query builder

## 3. Installation and Setup

### Ubuntu/Debian Installation

```bash
# Add RethinkDB repository
source /etc/lsb-release && \
  echo "deb https://download.rethinkdb.com/repository/ubuntu-$DISTRIB_CODENAME $DISTRIB_CODENAME main" | \
  sudo tee /etc/apt/sources.list.d/rethinkdb.list

# Add repository key
wget -qO- https://download.rethinkdb.com/repository/raw/pubkey.gpg | \
  sudo apt-key add -

# Update and install
sudo apt-get update
sudo apt-get install rethinkdb

# Start RethinkDB
rethinkdb

# Access web UI
# Open http://localhost:8080
```

### macOS Installation

```bash
# Using Homebrew
brew install rethinkdb

# Start RethinkDB
rethinkdb

# Or as a service
brew services start rethinkdb
```

### Docker Installation

```bash
# Run RethinkDB in Docker
docker run -d \
  --name rethinkdb \
  -p 8080:8080 \
  -p 28015:28015 \
  -p 29015:29015 \
  -v rethinkdb_data:/data \
  rethinkdb:latest

# Access web UI at http://localhost:8080
```

### Configuration

```bash
# Create configuration file
sudo cp /etc/rethinkdb/default.conf.sample /etc/rethinkdb/instances.d/instance1.conf

# Edit configuration
sudo nano /etc/rethinkdb/instances.d/instance1.conf
```

```conf
# /etc/rethinkdb/instances.d/instance1.conf

# Server settings
server-name=server1
server-tag=default

# Network
bind=all
driver-port=28015        # Client driver connections
cluster-port=29015       # Intra-cluster traffic
http-port=8080           # Web UI

# Data directory
directory=/var/lib/rethinkdb/instance1

# Cache size (50% of RAM recommended)
cache-size=2048          # MB

# Logging
log-file=/var/log/rethinkdb/instance1.log

# Join cluster
# join=server2:29015
```

### Language Drivers

```bash
# Python
pip install rethinkdb

# JavaScript/Node.js
npm install rethinkdb

# Ruby
gem install rethinkdb

# Java
# Add to pom.xml:
# <dependency>
#   <groupId>com.rethinkdb</groupId>
#   <artifactId>rethinkdb-driver</artifactId>
#   <version>2.4.4</version>
# </dependency>

# Go
go get gopkg.in/rethinkdb/rethinkdb-go.v6
```

## 4. ReQL Query Language Basics

ReQL is RethinkDB's query language, embedded in your programming language.

### Python Driver - Basic Operations

```python
import rethinkdb as r
from typing import List, Dict, Any

class RethinkDBConnection:
    def __init__(self, host='localhost', port=28015, db='test'):
        self.host = host
        self.port = port
        self.db_name = db
        self.conn = None

    def connect(self):
        """Establish connection to RethinkDB."""
        self.conn = r.connect(
            host=self.host,
            port=self.port,
            db=self.db_name
        )
        return self.conn

    def close(self):
        """Close connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Database operations
def create_database(conn, db_name: str):
    """Create a database."""
    try:
        r.db_create(db_name).run(conn)
        print(f"Database '{db_name}' created")
    except r.ReqlOpFailedError as e:
        print(f"Database already exists: {e}")


def list_databases(conn) -> List[str]:
    """List all databases."""
    return r.db_list().run(conn)


def drop_database(conn, db_name: str):
    """Drop a database."""
    r.db_drop(db_name).run(conn)
    print(f"Database '{db_name}' dropped")


# Table operations
def create_table(conn, table_name: str):
    """Create a table."""
    try:
        r.table_create(table_name).run(conn)
        print(f"Table '{table_name}' created")
    except r.ReqlOpFailedError as e:
        print(f"Table already exists: {e}")


def list_tables(conn) -> List[str]:
    """List all tables."""
    return r.table_list().run(conn)


def drop_table(conn, table_name: str):
    """Drop a table."""
    r.table_drop(table_name).run(conn)
    print(f"Table '{table_name}' dropped")


# CRUD operations
def insert_document(conn, table: str, document: Dict[str, Any]) -> str:
    """Insert a document into a table."""
    result = r.table(table).insert(document).run(conn)

    if result['inserted'] == 1:
        # RethinkDB generates UUID if no 'id' provided
        return result['generated_keys'][0]
    else:
        raise Exception(f"Insert failed: {result}")


def insert_many(conn, table: str, documents: List[Dict[str, Any]]) -> List[str]:
    """Insert multiple documents."""
    result = r.table(table).insert(documents).run(conn)

    if result['inserted'] > 0:
        return result.get('generated_keys', [])
    else:
        raise Exception(f"Insert failed: {result}")


def get_document(conn, table: str, doc_id: str) -> Dict[str, Any]:
    """Get a document by ID."""
    doc = r.table(table).get(doc_id).run(conn)
    return doc


def update_document(conn, table: str, doc_id: str, updates: Dict[str, Any]) -> int:
    """Update a document."""
    result = r.table(table).get(doc_id).update(updates).run(conn)
    return result['replaced']


def replace_document(conn, table: str, doc_id: str, document: Dict[str, Any]) -> int:
    """Replace entire document."""
    result = r.table(table).get(doc_id).replace(document).run(conn)
    return result['replaced']


def delete_document(conn, table: str, doc_id: str) -> int:
    """Delete a document."""
    result = r.table(table).get(doc_id).delete().run(conn)
    return result['deleted']


# Query operations
def filter_documents(conn, table: str, predicate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter documents by predicate."""
    cursor = r.table(table).filter(predicate).run(conn)
    return list(cursor)


def get_all_documents(conn, table: str) -> List[Dict[str, Any]]:
    """Get all documents from a table."""
    cursor = r.table(table).run(conn)
    return list(cursor)


# Usage examples
def examples():
    with RethinkDBConnection() as conn:
        # Create database and table
        create_database(conn.conn, 'myapp')
        r.db('myapp').table_create('users').run(conn.conn)

        # Switch to new database
        conn.conn.use('myapp')

        # Insert documents
        user_id = insert_document(conn.conn, 'users', {
            'name': 'Alice Johnson',
            'email': 'alice@example.com',
            'age': 30,
            'interests': ['programming', 'hiking']
        })
        print(f"Inserted user: {user_id}")

        # Insert multiple documents
        users = [
            {'name': 'Bob Smith', 'email': 'bob@example.com', 'age': 25},
            {'name': 'Charlie Brown', 'email': 'charlie@example.com', 'age': 35}
        ]
        ids = insert_many(conn.conn, 'users', users)
        print(f"Inserted users: {ids}")

        # Get document
        user = get_document(conn.conn, 'users', user_id)
        print(f"User: {user}")

        # Update document
        update_document(conn.conn, 'users', user_id, {'age': 31})

        # Filter documents
        young_users = filter_documents(conn.conn, 'users', {'age': 25})
        print(f"Young users: {young_users}")

        # Get all documents
        all_users = get_all_documents(conn.conn, 'users')
        print(f"All users: {len(all_users)}")
```

### JavaScript/Node.js Driver

```javascript
const r = require('rethinkdb');

class RethinkDBConnection {
    constructor(host = 'localhost', port = 28015, db = 'test') {
        this.host = host;
        this.port = port;
        this.db = db;
        this.conn = null;
    }

    async connect() {
        this.conn = await r.connect({
            host: this.host,
            port: this.port,
            db: this.db
        });
        return this.conn;
    }

    async close() {
        if (this.conn) {
            await this.conn.close();
        }
    }
}

// CRUD operations
async function insertDocument(conn, table, document) {
    const result = await r.table(table).insert(document).run(conn);

    if (result.inserted === 1) {
        return result.generated_keys[0];
    }
    throw new Error(`Insert failed: ${JSON.stringify(result)}`);
}

async function getDocument(conn, table, id) {
    return await r.table(table).get(id).run(conn);
}

async function updateDocument(conn, table, id, updates) {
    const result = await r.table(table).get(id).update(updates).run(conn);
    return result.replaced;
}

async function deleteDocument(conn, table, id) {
    const result = await r.table(table).get(id).delete().run(conn);
    return result.deleted;
}

async function filterDocuments(conn, table, predicate) {
    const cursor = await r.table(table).filter(predicate).run(conn);
    return await cursor.toArray();
}

// Usage
async function main() {
    const db = new RethinkDBConnection('localhost', 28015, 'myapp');
    const conn = await db.connect();

    try {
        // Insert
        const userId = await insertDocument(conn, 'users', {
            name: 'Alice Johnson',
            email: 'alice@example.com',
            age: 30
        });
        console.log(`Inserted user: ${userId}`);

        // Get
        const user = await getDocument(conn, 'users', userId);
        console.log('User:', user);

        // Update
        await updateDocument(conn, 'users', userId, { age: 31 });

        // Filter
        const youngUsers = await filterDocuments(conn, 'users', { age: 25 });
        console.log('Young users:', youngUsers);

    } finally {
        await db.close();
    }
}

main().catch(console.error);
```

## 5. Real-Time Changefeeds

Changefeeds are RethinkDB's killer feature - real-time push notifications for data changes.

### Basic Changefeed

```python
import rethinkdb as r

def subscribe_to_changes(conn, table: str):
    """Subscribe to all changes on a table."""
    # Create changefeed
    feed = r.table(table).changes().run(conn)

    print(f"Listening for changes on '{table}'...")

    try:
        for change in feed:
            # change contains 'old_val' and 'new_val'
            if change['old_val'] is None:
                # Insert
                print(f"INSERT: {change['new_val']}")
            elif change['new_val'] is None:
                # Delete
                print(f"DELETE: {change['old_val']}")
            else:
                # Update
                print(f"UPDATE: {change['old_val']} -> {change['new_val']}")
    except r.ReqlError as e:
        print(f"Changefeed error: {e}")
    finally:
        feed.close()


# Changefeed with filter
def subscribe_to_active_users(conn):
    """Subscribe to changes for active users only."""
    feed = r.table('users').filter({'status': 'active'}).changes().run(conn)

    for change in feed:
        print(f"Active user changed: {change}")


# Include initial values
def subscribe_with_initial(conn, table: str):
    """Subscribe to changes including initial values."""
    feed = r.table(table).changes(include_initial=True).run(conn)

    for change in feed:
        print(f"Change: {change}")


# Changefeed on specific document
def subscribe_to_document(conn, table: str, doc_id: str):
    """Subscribe to changes on a specific document."""
    feed = r.table(table).get(doc_id).changes().run(conn)

    for change in feed:
        print(f"Document {doc_id} changed: {change}")
```

### Advanced Changefeed Patterns

```python
# Changefeed with transformation
def subscribe_to_user_names(conn):
    """Subscribe to user name changes only."""
    feed = (r.table('users')
            .changes()
            .pluck({'new_val': ['id', 'name'], 'old_val': ['id', 'name']})
            .run(conn))

    for change in feed:
        if change.get('new_val'):
            print(f"Name changed to: {change['new_val']['name']}")


# Changefeed with includes_types
def subscribe_with_types(conn, table: str):
    """Subscribe with state change types."""
    feed = r.table(table).changes(include_types=True).run(conn)

    for change in feed:
        change_type = change.get('type')

        if change_type == 'add':
            print(f"Document added: {change['new_val']}")
        elif change_type == 'remove':
            print(f"Document removed: {change['old_val']}")
        elif change_type == 'change':
            print(f"Document changed: {change['old_val']} -> {change['new_val']}")
        elif change_type == 'initial':
            print(f"Initial value: {change['new_val']}")
        elif change_type == 'uninitial':
            print(f"Query no longer matches: {change['old_val']}")


# Changefeed on join
def subscribe_to_user_posts(conn):
    """Subscribe to users with their posts count."""
    feed = (r.table('users')
            .eq_join('id', r.table('posts'), index='user_id')
            .zip()
            .changes()
            .run(conn))

    for change in feed:
        print(f"User-post relationship changed: {change}")


# Point changefeed (specific query)
def subscribe_to_query(conn):
    """Subscribe to changes on a specific query."""
    # Monitor users in San Francisco
    feed = (r.table('users')
            .filter({'city': 'San Francisco'})
            .changes(include_initial=True)
            .run(conn))

    for change in feed:
        print(f"SF user changed: {change}")
```

### Real-Time Application Example

```python
import rethinkdb as r
import asyncio
from typing import Callable

class RealtimeApp:
    def __init__(self, conn):
        self.conn = conn
        self.feeds = {}

    def subscribe(self, name: str, query, callback: Callable):
        """Subscribe to a query with a callback."""
        async def feed_loop():
            feed = query.changes().run(self.conn)
            try:
                for change in feed:
                    callback(change)
            except Exception as e:
                print(f"Feed error: {e}")
            finally:
                feed.close()

        # Store feed task
        task = asyncio.create_task(feed_loop())
        self.feeds[name] = task
        return task

    def unsubscribe(self, name: str):
        """Unsubscribe from a feed."""
        if name in self.feeds:
            self.feeds[name].cancel()
            del self.feeds[name]


# Usage: Real-time chat application
class ChatApp:
    def __init__(self, conn):
        self.conn = conn
        self.app = RealtimeApp(conn)

    def subscribe_to_room(self, room_id: str):
        """Subscribe to messages in a chat room."""
        query = (r.table('messages')
                 .filter({'room_id': room_id})
                 .order_by(r.desc('timestamp')))

        def on_message(change):
            if change['new_val']:
                msg = change['new_val']
                print(f"[{msg['user']}]: {msg['text']}")

        self.app.subscribe(f"room_{room_id}", query, on_message)

    def send_message(self, room_id: str, user: str, text: str):
        """Send a message to a room."""
        r.table('messages').insert({
            'room_id': room_id,
            'user': user,
            'text': text,
            'timestamp': r.now()
        }).run(self.conn)
```

### JavaScript Changefeed

```javascript
// Real-time notifications
async function subscribeToNotifications(conn, userId) {
    const feed = await r.table('notifications')
        .filter({ userId: userId, read: false })
        .changes({ includeInitial: true })
        .run(conn);

    feed.each((err, change) => {
        if (err) {
            console.error('Feed error:', err);
            return;
        }

        if (change.new_val) {
            // New notification
            console.log('New notification:', change.new_val);

            // Notify user (e.g., via WebSocket)
            notifyUser(change.new_val);
        }
    });
}

// Live dashboard updates
async function subscribeToDashboard(conn) {
    const feed = await r.table('metrics')
        .changes()
        .run(conn);

    feed.each((err, change) => {
        if (err) {
            console.error('Feed error:', err);
            return;
        }

        // Update dashboard in real-time
        updateDashboard(change.new_val);
    });
}
```

## 6. Data Modeling

### Schema Design

```python
# Users table
users_schema = {
    'id': 'uuid',  # Auto-generated
    'email': 'string',
    'name': 'string',
    'age': 'number',
    'created_at': 'datetime',
    'address': {
        'street': 'string',
        'city': 'string',
        'state': 'string',
        'zip': 'string'
    },
    'preferences': {
        'notifications': 'boolean',
        'theme': 'string'
    }
}

# Posts table
posts_schema = {
    'id': 'uuid',
    'user_id': 'uuid',  # Foreign key to users
    'title': 'string',
    'content': 'string',
    'tags': ['string'],  # Array of strings
    'status': 'string',  # 'draft', 'published', 'archived'
    'likes_count': 'number',
    'created_at': 'datetime',
    'updated_at': 'datetime'
}

# Comments table
comments_schema = {
    'id': 'uuid',
    'post_id': 'uuid',  # Foreign key to posts
    'user_id': 'uuid',  # Foreign key to users
    'text': 'string',
    'created_at': 'datetime'
}
```

### Embedding vs Referencing

```python
# Embedding (denormalization) - good for one-to-few
user_with_embedded_address = {
    'id': 'user123',
    'name': 'Alice',
    'address': {
        'street': '123 Main St',
        'city': 'San Francisco',
        'state': 'CA'
    }
}

# Referencing (normalization) - good for one-to-many or many-to-many
user = {
    'id': 'user123',
    'name': 'Alice'
}

post = {
    'id': 'post456',
    'user_id': 'user123',  # Reference to user
    'title': 'My Post',
    'content': 'Content here'
}

# Hybrid approach - embed frequently accessed data
post_with_user_info = {
    'id': 'post456',
    'user_id': 'user123',
    'user_name': 'Alice',  # Denormalized for quick access
    'title': 'My Post',
    'content': 'Content here'
}
```

### Time-Series Data

```python
# Metrics table for time-series data
def create_metrics_table(conn):
    r.table_create('metrics').run(conn)

    # Create compound index for efficient queries
    r.table('metrics').index_create(
        'metric_time',
        [r.row['metric_name'], r.row['timestamp']]
    ).run(conn)


def insert_metric(conn, metric_name: str, value: float):
    """Insert a time-series metric."""
    r.table('metrics').insert({
        'metric_name': metric_name,
        'value': value,
        'timestamp': r.now()
    }).run(conn)


def query_metrics(conn, metric_name: str, start_time, end_time):
    """Query metrics in a time range."""
    return (r.table('metrics')
            .between([metric_name, start_time], [metric_name, end_time],
                    index='metric_time')
            .order_by(index=r.asc('metric_time'))
            .run(conn))
```

## 7. Indexing

### Creating Indexes

```python
import rethinkdb as r

# Simple index on single field
def create_simple_index(conn, table: str, field: str):
    """Create simple index."""
    r.table(table).index_create(field).run(conn)
    r.table(table).index_wait(field).run(conn)
    print(f"Index '{field}' created")


# Compound index (multiple fields)
def create_compound_index(conn, table: str, index_name: str, fields: list):
    """Create compound index."""
    r.table(table).index_create(
        index_name,
        [r.row[field] for field in fields]
    ).run(conn)
    r.table(table).index_wait(index_name).run(conn)
    print(f"Compound index '{index_name}' created")


# Multi index (for arrays)
def create_multi_index(conn, table: str, field: str):
    """Create multi-value index for array fields."""
    r.table(table).index_create(
        field,
        multi=True
    ).run(conn)
    r.table(table).index_wait(field).run(conn)
    print(f"Multi index '{field}' created")


# Functional index
def create_functional_index(conn, table: str, index_name: str):
    """Create index with custom function."""
    r.table(table).index_create(
        index_name,
        lambda doc: doc['first_name'] + ' ' + doc['last_name']
    ).run(conn)
    r.table(table).index_wait(index_name).run(conn)
    print(f"Functional index '{index_name}' created")


# Geospatial index
def create_geo_index(conn, table: str, field: str = 'location'):
    """Create geospatial index."""
    r.table(table).index_create(
        field,
        geo=True
    ).run(conn)
    r.table(table).index_wait(field).run(conn)
    print(f"Geo index '{field}' created")


# Usage examples
def index_examples(conn):
    table = 'users'

    # Simple indexes
    create_simple_index(conn, table, 'email')
    create_simple_index(conn, table, 'age')

    # Compound index for sorting
    create_compound_index(conn, table, 'city_age', ['city', 'age'])

    # Multi index for array field
    create_multi_index(conn, table, 'interests')

    # Functional index for full name
    create_functional_index(conn, table, 'full_name')

    # Geospatial index
    create_geo_index(conn, 'locations', 'coordinates')

    # List all indexes
    indexes = r.table(table).index_list().run(conn)
    print(f"Indexes: {indexes}")
```

### Using Indexes

```python
# Query with simple index
def find_by_email(conn, email: str):
    """Find user by email using index."""
    return r.table('users').get_all(email, index='email').run(conn)


# Query with compound index
def find_by_city_and_age(conn, city: str, age: int):
    """Find users by city and age."""
    return (r.table('users')
            .get_all([city, age], index='city_age')
            .run(conn))


# Range query with index
def find_age_range(conn, min_age: int, max_age: int):
    """Find users in age range."""
    return (r.table('users')
            .between(min_age, max_age, index='age')
            .run(conn))


# Query with multi index (array field)
def find_by_interest(conn, interest: str):
    """Find users with specific interest."""
    return (r.table('users')
            .get_all(interest, index='interests')
            .run(conn))


# Geospatial query
def find_nearby(conn, lat: float, lng: float, radius_meters: float):
    """Find locations near a point."""
    point = r.point(lng, lat)  # Note: GeoJSON uses [lng, lat]

    return (r.table('locations')
            .get_intersecting(r.circle(point, radius_meters, unit='m'),
                            index='coordinates')
            .run(conn))
```

## 8. Joins and Aggregations

### Joins

```python
import rethinkdb as r

# Inner join
def get_posts_with_users(conn):
    """Get posts with user information (inner join)."""
    return (r.table('posts')
            .eq_join('user_id', r.table('users'))
            .zip()  # Merge joined documents
            .run(conn))


# Left outer join
def get_all_posts_with_optional_user(conn):
    """Get all posts, with user info if available."""
    return (r.table('posts')
            .eq_join('user_id', r.table('users'), ordered=True)
            .map(lambda doc:
                doc.has_fields('right').branch(
                    doc['left'].merge(doc['right']),
                    doc['left']
                ))
            .run(conn))


# Join with index
def get_posts_by_user_email(conn, email: str):
    """Get posts by user email (using index)."""
    return (r.table('posts')
            .eq_join('user_id', r.table('users'), index='email')
            .filter({'right': {'email': email}})
            .zip()
            .run(conn))


# Multiple joins
def get_comments_with_user_and_post(conn):
    """Get comments with user and post information."""
    return (r.table('comments')
            .eq_join('user_id', r.table('users'))
            .zip()
            .eq_join('post_id', r.table('posts'))
            .zip()
            .run(conn))
```

### Aggregations

```python
# Count
def count_users(conn):
    """Count total users."""
    return r.table('users').count().run(conn)


def count_by_city(conn):
    """Count users by city."""
    return (r.table('users')
            .group('city')
            .count()
            .ungroup()
            .run(conn))


# Sum
def total_likes(conn):
    """Sum all post likes."""
    return r.table('posts').sum('likes_count').run(conn)


def total_likes_by_user(conn):
    """Sum likes by user."""
    return (r.table('posts')
            .group('user_id')
            .sum('likes_count')
            .ungroup()
            .run(conn))


# Average
def average_age(conn):
    """Calculate average user age."""
    return r.table('users').avg('age').run(conn)


def average_age_by_city(conn):
    """Average age by city."""
    return (r.table('users')
            .group('city')
            .avg('age')
            .ungroup()
            .run(conn))


# Min/Max
def oldest_user(conn):
    """Find oldest user."""
    return r.table('users').max('age').run(conn)


def youngest_by_city(conn):
    """Find youngest user in each city."""
    return (r.table('users')
            .group('city')
            .min('age')
            .ungroup()
            .run(conn))


# Distinct
def unique_cities(conn):
    """Get distinct cities."""
    return r.table('users').pluck('city').distinct().run(conn)


# Complex aggregation
def post_statistics(conn):
    """Get comprehensive post statistics."""
    return (r.table('posts')
            .group('status')
            .map(lambda post: {
                'total': 1,
                'total_likes': post['likes_count'],
                'avg_likes': post['likes_count']
            })
            .reduce(lambda left, right: {
                'total': left['total'] + right['total'],
                'total_likes': left['total_likes'] + right['total_likes'],
                'avg_likes': (left['total_likes'] + right['total_likes']) /
                           (left['total'] + right['total'])
            })
            .ungroup()
            .run(conn))
```

### Map-Reduce

```python
# Custom map-reduce
def word_count_in_posts(conn):
    """Count word occurrences in posts."""
    return (r.table('posts')
            # Map: split content into words
            .concat_map(lambda doc:
                doc['content'].downcase().split(r.args([r.expr(' ')])))
            # Group by word
            .group(lambda word: word)
            # Reduce: count
            .count()
            .ungroup()
            # Sort by count
            .order_by(r.desc('reduction'))
            .limit(10)
            .run(conn))


# Top contributors
def top_contributors(conn, limit: int = 10):
    """Find users with most posts."""
    return (r.table('posts')
            .group('user_id')
            .count()
            .ungroup()
            .order_by(r.desc('reduction'))
            .limit(limit)
            .eq_join('group', r.table('users'))
            .zip()
            .run(conn))
```

## 9. Sharding and Replication

### Configuring Shards

```python
import rethinkdb as r

# Configure table sharding
def configure_shards(conn, table: str, num_shards: int):
    """Configure number of shards for a table."""
    r.table(table).config().update({'shards': [
        {'primary_replica': 'server1', 'replicas': ['server1', 'server2']},
        {'primary_replica': 'server2', 'replicas': ['server2', 'server3']},
        # ... more shards
    ]}).run(conn)


def reconfigure_shards(conn, table: str, num_shards: int, num_replicas: int):
    """Reconfigure table sharding and replication."""
    r.table(table).reconfigure(
        shards=num_shards,
        replicas=num_replicas
    ).run(conn)


# Get table configuration
def get_table_config(conn, table: str):
    """Get current table configuration."""
    config = r.table(table).config().run(conn)
    print(f"Table '{table}' configuration:")
    print(f"  Shards: {len(config['shards'])}")
    print(f"  Primary replicas: {[s['primary_replica'] for s in config['shards']]}")
    return config


# Wait for table to be ready
def wait_for_table_ready(conn, table: str):
    """Wait for table to be ready after reconfiguration."""
    r.table(table).wait(wait_for='ready_for_writes').run(conn)
    print(f"Table '{table}' is ready")
```

### Replication

```python
# Configure replication
def setup_replication(conn, table: str):
    """Setup 3-way replication."""
    r.table(table).reconfigure(
        shards=3,
        replicas=3,
        primary_replica_tag='default'
    ).run(conn)


# Emergency repair
def emergency_repair(conn, table: str):
    """Emergency repair for table with majority of replicas down."""
    r.table(table).reconfigure(
        shards=1,
        replicas=1,
        emergency_repair='unsafe_rollback'
    ).run(conn)


# Check table status
def check_table_status(conn, table: str):
    """Check table replication status."""
    status = r.table(table).status().run(conn)

    print(f"Table '{table}' status:")
    print(f"  Ready for reads: {status['status']['ready_for_reads']}")
    print(f"  Ready for writes: {status['status']['ready_for_writes']}")
    print(f"  All replicas ready: {status['status']['all_replicas_ready']}")

    for shard in status['shards']:
        print(f"  Shard {shard['primary_replicas']}: {shard['replicas']}")

    return status
```

## 10. Clustering and High Availability

### Cluster Setup

```bash
# Server 1
rethinkdb \
  --server-name server1 \
  --bind all \
  --directory /data/rethinkdb/server1 \
  --cache-size 2048

# Server 2
rethinkdb \
  --server-name server2 \
  --bind all \
  --directory /data/rethinkdb/server2 \
  --join server1:29015 \
  --cache-size 2048

# Server 3
rethinkdb \
  --server-name server3 \
  --bind all \
  --directory /data/rethinkdb/server3 \
  --join server1:29015 \
  --cache-size 2048
```

### Cluster Management

```python
# Get cluster status
def get_cluster_status(conn):
    """Get cluster status."""
    status = r.db('rethinkdb').table('server_status').run(conn)

    for server in status:
        print(f"Server: {server['name']}")
        print(f"  Status: {server['status']}")
        print(f"  Network: {server['network']}")

    return status


# List current issues
def check_cluster_issues(conn):
    """Check for cluster issues."""
    issues = r.db('rethinkdb').table('current_issues').run(conn)

    for issue in issues:
        print(f"Issue: {issue['type']}")
        print(f"  Description: {issue['description']}")
        print(f"  Critical: {issue['critical']}")

    return list(issues)


# Get cluster configuration
def get_cluster_config(conn):
    """Get cluster configuration."""
    config = r.db('rethinkdb').table('server_config').run(conn)

    for server in config:
        print(f"Server: {server['name']}")
        print(f"  Tags: {server['tags']}")
        print(f"  Cache size: {server.get('cache_size_mb')} MB")

    return list(config)
```

### Failover

```python
# Monitor server health
def monitor_servers(conn):
    """Monitor server health with changefeed."""
    feed = r.db('rethinkdb').table('server_status').changes().run(conn)

    for change in feed:
        if change.get('new_val'):
            server = change['new_val']
            print(f"Server {server['name']}: {server['status']}")

            if server['status'] != 'connected':
                print(f"WARNING: Server {server['name']} is {server['status']}")


# Automatic failover is handled by RethinkDB
# Manual intervention example:
def force_primary_replica(conn, table: str, shard_index: int, new_primary: str):
    """Manually set primary replica for a shard."""
    config = r.table(table).config().run(conn)
    config['shards'][shard_index]['primary_replica'] = new_primary

    r.table(table).config().update(config).run(conn)
    r.table(table).wait().run(conn)
```

## 11. Security Best Practices

### User Management

```python
import rethinkdb as r

# Create admin user
def create_admin_user(conn, username: str, password: str):
    """Create an admin user."""
    r.db('rethinkdb').table('users').insert({
        'id': username,
        'password': password
    }).run(conn)


# Grant permissions
def grant_permissions(conn, username: str, database: str, table: str):
    """Grant read/write permissions to user."""
    # Read permission
    r.db(database).table(table).grant(username, {'read': True}).run(conn)

    # Write permission
    r.db(database).table(table).grant(username, {'write': True}).run(conn)

    # Config permission (for table configuration)
    r.db(database).table(table).grant(username, {'config': True}).run(conn)


# Connect with authentication
def connect_with_auth(host: str, port: int, user: str, password: str, db: str):
    """Connect with user authentication."""
    conn = r.connect(
        host=host,
        port=port,
        db=db,
        user=user,
        password=password
    )
    return conn
```

### TLS/SSL Configuration

```bash
# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=rethinkdb.example.com"

# Start RethinkDB with TLS
rethinkdb \
  --tls-min-protocol TLSv1.2 \
  --tls-cert cert.pem \
  --tls-key key.pem \
  --driver-tls-cert driver-cert.pem \
  --driver-tls-key driver-key.pem
```

```python
# Connect with TLS
def connect_with_tls(host: str, port: int, ca_cert: str):
    """Connect using TLS."""
    conn = r.connect(
        host=host,
        port=port,
        ssl={
            'ca_certs': ca_cert
        }
    )
    return conn
```

### Network Security

```bash
# Firewall configuration
# Allow cluster ports only from cluster members
sudo ufw allow from 10.0.1.0/24 to any port 29015  # Cluster port
sudo ufw allow from 10.0.1.0/24 to any port 28015  # Driver port

# Allow web UI only from admin network
sudo ufw allow from 10.0.2.0/24 to any port 8080   # Web UI

# Enable firewall
sudo ufw enable
```

### Input Validation

```python
import re

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_input(conn, table: str, data: dict) -> dict:
    """Sanitize and validate input data."""
    # Remove unexpected fields
    allowed_fields = {'name', 'email', 'age', 'city'}
    sanitized = {k: v for k, v in data.items() if k in allowed_fields}

    # Validate email
    if 'email' in sanitized:
        if not validate_email(sanitized['email']):
            raise ValueError("Invalid email format")

    # Validate age
    if 'age' in sanitized:
        age = sanitized['age']
        if not isinstance(age, int) or age < 0 or age > 150:
            raise ValueError("Invalid age")

    # Escape HTML in text fields
    if 'name' in sanitized:
        sanitized['name'] = sanitized['name'].replace('<', '&lt;').replace('>', '&gt;')

    return sanitized


def safe_insert(conn, table: str, data: dict):
    """Insert data with validation."""
    sanitized_data = sanitize_input(conn, table, data)
    return r.table(table).insert(sanitized_data).run(conn)
```

## 12. Backup and Recovery

### Backup

```bash
# Dump entire database
rethinkdb dump -c localhost:28015 -f backup.tar.gz

# Dump specific database
rethinkdb dump -c localhost:28015 -e mydb -f mydb_backup.tar.gz

# Dump specific table
rethinkdb dump -c localhost:28015 -e mydb.users -f users_backup.tar.gz

# Dump with authentication
rethinkdb dump -c localhost:28015 --auth-key mypassword -f backup.tar.gz

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/rethinkdb"
mkdir -p $BACKUP_DIR

rethinkdb dump \
  -c localhost:28015 \
  -f $BACKUP_DIR/backup_$DATE.tar.gz

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +7 -delete
```

### Restore

```bash
# Restore entire database
rethinkdb restore backup.tar.gz -c localhost:28015

# Restore specific database
rethinkdb restore backup.tar.gz -c localhost:28015 -i mydb

# Restore and overwrite existing data
rethinkdb restore backup.tar.gz -c localhost:28015 --force
```

### Export/Import Specific Data

```python
import rethinkdb as r
import json

# Export table to JSON
def export_table_to_json(conn, table: str, filename: str):
    """Export table to JSON file."""
    cursor = r.table(table).run(conn)

    with open(filename, 'w') as f:
        for doc in cursor:
            f.write(json.dumps(doc) + '\n')

    print(f"Exported {table} to {filename}")


# Import table from JSON
def import_table_from_json(conn, table: str, filename: str):
    """Import table from JSON file."""
    documents = []

    with open(filename, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)

            # Batch insert every 1000 documents
            if len(documents) >= 1000:
                r.table(table).insert(documents).run(conn)
                documents = []

    # Insert remaining documents
    if documents:
        r.table(table).insert(documents).run(conn)

    print(f"Imported data into {table}")


# Incremental backup (via changefeeds)
def incremental_backup(conn, table: str, backup_table: str):
    """Setup incremental backup using changefeeds."""
    # Create backup table
    try:
        r.table_create(backup_table).run(conn)
    except:
        pass

    # Subscribe to changes
    feed = r.table(table).changes(include_initial=False).run(conn)

    for change in feed:
        # Store change in backup table
        r.table(backup_table).insert({
            'timestamp': r.now(),
            'change': change
        }).run(conn)
```

## 13. Production Deployment

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rethinkdb-1:
    image: rethinkdb:2.4.2
    container_name: rethinkdb-1
    ports:
      - "8080:8080"    # Web UI
      - "28015:28015"  # Client driver
      - "29015:29015"  # Cluster
    volumes:
      - rethinkdb1_data:/data
    command: rethinkdb --bind all --server-name rethinkdb-1
    networks:
      - rethinkdb-network

  rethinkdb-2:
    image: rethinkdb:2.4.2
    container_name: rethinkdb-2
    ports:
      - "8081:8080"
      - "28016:28015"
    volumes:
      - rethinkdb2_data:/data
    command: rethinkdb --bind all --server-name rethinkdb-2 --join rethinkdb-1:29015
    depends_on:
      - rethinkdb-1
    networks:
      - rethinkdb-network

  rethinkdb-3:
    image: rethinkdb:2.4.2
    container_name: rethinkdb-3
    ports:
      - "8082:8080"
      - "28017:28015"
    volumes:
      - rethinkdb3_data:/data
    command: rethinkdb --bind all --server-name rethinkdb-3 --join rethinkdb-1:29015
    depends_on:
      - rethinkdb-1
    networks:
      - rethinkdb-network

volumes:
  rethinkdb1_data:
  rethinkdb2_data:
  rethinkdb3_data:

networks:
  rethinkdb-network:
    driver: bridge
```

### Kubernetes Deployment

```yaml
# rethinkdb-statefulset.yaml
apiVersion: v1
kind: Service
metadata:
  name: rethinkdb
spec:
  clusterIP: None
  selector:
    app: rethinkdb
  ports:
  - name: driver
    port: 28015
  - name: cluster
    port: 29015
  - name: http
    port: 8080
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: rethinkdb
spec:
  serviceName: "rethinkdb"
  replicas: 3
  selector:
    matchLabels:
      app: rethinkdb
  template:
    metadata:
      labels:
        app: rethinkdb
    spec:
      containers:
      - name: rethinkdb
        image: rethinkdb:2.4.2
        ports:
        - containerPort: 28015
          name: driver
        - containerPort: 29015
          name: cluster
        - containerPort: 8080
          name: http
        volumeMounts:
        - name: data
          mountPath: /data
        command:
        - rethinkdb
        - --bind
        - all
        - --server-name
        - $(POD_NAME)
        - --join
        - rethinkdb-0.rethinkdb:29015
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 50Gi
```

### Production Configuration

```python
# Production connection pool
import rethinkdb as r
from contextlib import contextmanager

class RethinkDBPool:
    def __init__(self, host='localhost', port=28015, db='myapp', pool_size=10):
        self.host = host
        self.port = port
        self.db = db
        self.pool_size = pool_size
        self.connections = []
        self.in_use = set()

    def _create_connection(self):
        return r.connect(
            host=self.host,
            port=self.port,
            db=self.db,
            timeout=20
        )

    def get_connection(self):
        # Reuse existing connection
        for conn in self.connections:
            if conn not in self.in_use:
                self.in_use.add(conn)
                return conn

        # Create new connection if under pool size
        if len(self.connections) < self.pool_size:
            conn = self._create_connection()
            self.connections.append(conn)
            self.in_use.add(conn)
            return conn

        raise Exception("Connection pool exhausted")

    def release_connection(self, conn):
        self.in_use.discard(conn)

    @contextmanager
    def connection(self):
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)

    def close_all(self):
        for conn in self.connections:
            conn.close()
        self.connections = []
        self.in_use = set()


# Usage
pool = RethinkDBPool(host='rethinkdb-lb', port=28015, db='myapp', pool_size=20)

with pool.connection() as conn:
    users = r.table('users').run(conn)
    # Use connection..
```

## 14. Common Patterns

### Pagination

```python
# Cursor-based pagination
def paginate_users(conn, page_size: int = 20, cursor: str = None):
    """Paginate users with cursor."""
    query = r.table('users').order_by(index='created_at')

    if cursor:
        # Start from cursor
        query = query.filter(r.row['id'] > cursor)

    # Get one extra to determine if there are more pages
    results = list(query.limit(page_size + 1).run(conn))

    has_more = len(results) > page_size
    if has_more:
        results = results[:page_size]

    next_cursor = results[-1]['id'] if results and has_more else None

    return {
        'data': results,
        'next_cursor': next_cursor,
        'has_more': has_more
    }


# Offset-based pagination (less efficient)
def paginate_offset(conn, table: str, page: int, page_size: int):
    """Paginate with offset (not recommended for large datasets)."""
    offset = (page - 1) * page_size

    results = (r.table(table)
               .order_by(index='created_at')
               .skip(offset)
               .limit(page_size)
               .run(conn))

    total = r.table(table).count().run(conn)

    return {
        'data': list(results),
        'page': page,
        'page_size': page_size,
        'total': total,
        'pages': (total + page_size - 1) // page_size
    }
```

### Caching

```python
import redis
import json

class CachedRethinkDB:
    def __init__(self, rethink_conn, redis_client):
        self.rethink = rethink_conn
        self.redis = redis_client
        self.ttl = 300  # 5 minutes

    def get_user(self, user_id: str):
        """Get user with caching."""
        cache_key = f"user:{user_id}"

        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Query RethinkDB
        user = r.table('users').get(user_id).run(self.rethink)

        if user:
            # Cache result
            self.redis.setex(cache_key, self.ttl, json.dumps(user))

        return user

    def invalidate_user(self, user_id: str):
        """Invalidate user cache."""
        self.redis.delete(f"user:{user_id}")

    def update_user(self, user_id: str, updates: dict):
        """Update user and invalidate cache."""
        result = r.table('users').get(user_id).update(updates).run(self.rethink)
        self.invalidate_user(user_id)
        return result
```

### Rate Limiting

```python
import time

def rate_limit_user(conn, user_id: str, action: str, max_requests: int, window: int):
    """Rate limit user actions."""
    table = 'rate_limits'
    key = f"{user_id}:{action}"
    now = int(time.time())
    window_start = now - window

    # Get or create rate limit record
    record = r.table(table).get(key).run(conn)

    if not record:
        # Create new record
        r.table(table).insert({
            'id': key,
            'requests': [now]
        }).run(conn)
        return True

    # Filter requests within window
    recent_requests = [ts for ts in record['requests'] if ts > window_start]

    if len(recent_requests) >= max_requests:
        return False  # Rate limit exceeded

    # Add new request
    recent_requests.append(now)
    r.table(table).get(key).update({'requests': recent_requests}).run(conn)

    return True
```

## 15. Performance Optimization

### Query Optimization

```python
# ❌ BAD: N+1 query problem
def get_posts_with_users_bad(conn):
    posts = list(r.table('posts').run(conn))

    for post in posts:
        user = r.table('users').get(post['user_id']).run(conn)
        post['user'] = user

    return posts


# ✅ GOOD: Use join
def get_posts_with_users_good(conn):
    return list(r.table('posts')
                .eq_join('user_id', r.table('users'))
                .zip()
                .run(conn))


# ❌ BAD: Fetching unnecessary fields
def get_user_emails_bad(conn):
    users = list(r.table('users').run(conn))
    return [user['email'] for user in users]


# ✅ GOOD: Use pluck to fetch only needed fields
def get_user_emails_good(conn):
    return list(r.table('users').pluck('email').run(conn))
```

### Index Usage

```python
# Always use indexes for filters
def find_active_users(conn):
    """Use index for filtering."""
    # Assuming 'status' field has an index
    return list(r.table('users')
                .get_all('active', index='status')
                .run(conn))


# Use compound indexes for sorting
def get_recent_posts_by_user(conn, user_id: str):
    """Use compound index for efficient query."""
    # Assuming compound index ['user_id', 'created_at']
    return list(r.table('posts')
                .between([user_id, r.minval],
                        [user_id, r.maxval],
                        index='user_created')
                .order_by(index=r.desc('user_created'))
                .limit(20)
                .run(conn))
```

### Batch Operations

```python
# Batch inserts
def batch_insert_users(conn, users: list, batch_size: int = 200):
    """Insert users in batches."""
    for i in range(0, len(users), batch_size):
        batch = users[i:i + batch_size]
        r.table('users').insert(batch).run(conn)


# Batch updates
def batch_update_users(conn, user_ids: list, updates: dict):
    """Update multiple users efficiently."""
    r.table('users').get_all(*user_ids).update(updates).run(conn)
```

## 16. Troubleshooting

### Common Issues

```python
# Check cluster health
def diagnose_cluster(conn):
    """Diagnose cluster issues."""
    # Check current issues
    issues = list(r.db('rethinkdb').table('current_issues').run(conn))

    if issues:
        print("Current Issues:")
        for issue in issues:
            print(f"  - {issue['type']}: {issue['description']}")
    else:
        print("No current issues")

    # Check table readiness
    tables = r.db('rethinkdb').table('table_status').run(conn)

    for table in tables:
        print(f"\nTable: {table['db']}.{table['name']}")
        print(f"  Ready for reads: {table['status']['ready_for_reads']}")
        print(f"  Ready for writes: {table['status']['ready_for_writes']}")


# Monitor query performance
def slow_query_monitoring(conn):
    """Monitor slow queries."""
    feed = r.db('rethinkdb').table('jobs').changes().run(conn)

    for change in feed:
        if change.get('new_val'):
            job = change['new_val']
            if job['duration_sec'] > 1.0:  # Queries taking >1 second
                print(f"Slow query detected:")
                print(f"  Duration: {job['duration_sec']}s")
                print(f"  Type: {job['type']}")
```

### Performance Debugging

```bash
# Check table status
rethinkdb admin --join localhost:29015
> r.db('rethinkdb').table('stats').run()

# Monitor cluster performance
rethinkdb admin --join localhost:29015
> r.db('rethinkdb').table('stats')
    .filter({table: 'users'})
    .changes()
    .run()
```

## 17. Resources and References

### Official Documentation
- **RethinkDB Documentation**: https://rethinkdb.com/docs/
- **ReQL API**: https://rethinkdb.com/api/python/
- **Architecture Guide**: https://rethinkdb.com/docs/architecture/
- **Changefeeds**: https://rethinkdb.com/docs/changefeeds/

### Language Drivers
- **Python**: https://pypi.org/project/rethinkdb/
- **JavaScript**: https://www.npmjs.com/package/rethinkdb
- **Ruby**: https://rubygems.org/gems/rethinkdb
- **Java**: https://github.com/rethinkdb/rethinkdb-java
- **Go**: https://github.com/GoRethink/gorethink

### Tools
- **Web UI**: Built-in at http://localhost:8080
- **rethinkdb dump**: Backup utility
- **rethinkdb restore**: Restore utility
- **rethinkdb admin**: Admin CLI

### Community
- **GitHub**: https://github.com/rethinkdb/rethinkdb
- **Discord**: RethinkDB community server
- **Stack Overflow**: Tag `rethinkdb`

### Use Cases
- **Real-time applications**: Chat, collaboration, notifications
- **Live dashboards**: Analytics, monitoring
- **Streaming analytics**: Real-time data processing
- **Multiplayer games**: Game state synchronization
- **IoT applications**: Sensor data streaming

---

## Quick Start Example

```python
import rethinkdb as r

# Connect to RethinkDB
conn = r.connect(host='localhost', port=28015, db='test')

# Create database and table
r.db_create('myapp').run(conn)
r.db('myapp').table_create('users').run(conn)

# Switch to new database
conn.use('myapp')

# Insert documents
user_id = r.table('users').insert({
    'name': 'Alice Johnson',
    'email': 'alice@example.com',
    'age': 30
}).run(conn)['generated_keys'][0]

print(f"Inserted user: {user_id}")

# Query documents
users = list(r.table('users').filter({'age': 30}).run(conn))
print(f"Users aged 30: {users}")

# Real-time changefeed
print("Listening for changes...")
feed = r.table('users').changes().run(conn)

for change in feed:
    print(f"Change detected: {change}")

# Cleanup
conn.close()
```

This guide provides comprehensive coverage of RethinkDB for building real-time applications with its unique push architecture and changefeeds.

---

**End of RethinkDB Development Guidelines**
# MongoDB Development Guidelines
Mandatory standards for MongoDB database design, query optimization, and best practices. MongoDB 6+, MongoDB Compass, mongosh, Mongoose, MongoDB Atlas.

---

**Agent Profile**: The MongoDB Expert
**Role**: Senior Database Engineer & Document Database Specialist
**Objective**: Generate efficient, scalable, and maintainable MongoDB implementations.
**Tools**: MongoDB 6+, MongoDB Compass, mongosh, Mongoose, MongoDB Atlas.

---

## 1. Core Philosophies: MONGO-FIRST

- **M**odel for Queries: Design schema based on access patterns
- **O**ne-to-Many: Embed or reference based on cardinality
- **N**o Joins at Scale: Denormalize for read performance
- **G**ood Indexes: Index every query pattern
- **O**bserve: Monitor query performance and index usage

---

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

### Example TDD Workflow for MongoDB

```javascript
// Step 1: RED - Write failing test for an aggregation pipeline
// Using Jest + mongodb-memory-server

const { MongoMemoryServer } = require('mongodb-memory-server');
const { MongoClient } = require('mongodb');

let mongod, client, db;

beforeAll(async () => {
  mongod = await MongoMemoryServer.create();
  client = await MongoClient.connect(mongod.getUri());
  db = client.db('test');
});

afterAll(async () => {
  await client.close();
  await mongod.stop();
});

test('getTopSpendingUsers returns top N users by total order amount', async () => {
  const orders = db.collection('orders');
  await orders.insertMany([
    { userId: 'user1', total: 50, status: 'delivered' },
    { userId: 'user1', total: 75, status: 'delivered' },
    { userId: 'user2', total: 200, status: 'delivered' },
    { userId: 'user3', total: 30, status: 'cancelled' },
  ]);

  const result = await getTopSpendingUsers(db, 2);

  expect(result).toHaveLength(2);
  expect(result[0]).toEqual({ _id: 'user2', totalSpent: 200 });
  expect(result[1]).toEqual({ _id: 'user1', totalSpent: 125 });
});

// Run: npx jest --testPathPattern=orders.test.js
// FAILS - getTopSpendingUsers is not defined

// Step 2: GREEN - Implement the aggregation pipeline
async function getTopSpendingUsers(db, limit) {
  return db.collection('orders').aggregate([
    { $match: { status: { $ne: 'cancelled' } } },
    { $group: { _id: '$userId', totalSpent: { $sum: '$total' } } },
    { $sort: { totalSpent: -1 } },
    { $limit: limit }
  ]).toArray();
}

// Run: npx jest --testPathPattern=orders.test.js
// PASSES

// Step 3: REFACTOR - Add index to support the aggregation
// db.orders.createIndex({ status: 1, userId: 1, total: 1 });
// Tests still pass
```

### Example TDD for Schema Validation

```javascript
test('orders collection rejects documents without required fields', async () => {
  await db.createCollection('validated_orders', {
    validator: {
      $jsonSchema: {
        bsonType: 'object',
        required: ['userId', 'total', 'status'],
        properties: {
          userId: { bsonType: 'string' },
          total: { bsonType: 'number', minimum: 0 },
          status: { enum: ['pending', 'confirmed', 'shipped', 'delivered', 'cancelled'] }
        }
      }
    }
  });

  const col = db.collection('validated_orders');

  // Valid document succeeds
  await expect(col.insertOne({
    userId: 'user1', total: 50, status: 'pending'
  })).resolves.toBeDefined();

  // Missing required field fails
  await expect(col.insertOne({
    total: 50, status: 'pending'
  })).rejects.toThrow();

  // Negative total fails
  await expect(col.insertOne({
    userId: 'user1', total: -10, status: 'pending'
  })).rejects.toThrow();
});
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

```javascript
// Bug Report BUG-519: getTopSpendingUsers returns users with only
// cancelled orders because $match stage doesn't filter them out
// when all their orders are cancelled (they appear with totalSpent: 0).

test('BUG-519: users with only cancelled orders are excluded', async () => {
  const orders = db.collection('orders');
  await orders.deleteMany({});
  await orders.insertMany([
    { userId: 'user1', total: 100, status: 'delivered' },
    { userId: 'ghost', total: 50, status: 'cancelled' },
    { userId: 'ghost', total: 30, status: 'cancelled' },
  ]);

  const result = await getTopSpendingUsers(db, 10);

  // 'ghost' should NOT appear since all orders are cancelled
  const userIds = result.map(r => r._id);
  expect(userIds).not.toContain('ghost');
  expect(result).toHaveLength(1);
  expect(result[0]._id).toBe('user1');
});

// Run: npx jest --testPathPattern=orders.test.js
// FAILS - ghost appears with totalSpent: 0

// Fix: Add $match after $group to filter out zero-spend users
async function getTopSpendingUsers(db, limit) {
  return db.collection('orders').aggregate([
    { $match: { status: { $ne: 'cancelled' } } },
    { $group: { _id: '$userId', totalSpent: { $sum: '$total' } } },
    { $match: { totalSpent: { $gt: 0 } } },
    { $sort: { totalSpent: -1 } },
    { $limit: limit }
  ]).toArray();
}

// Run: npx jest --testPathPattern=orders.test.js
// PASSES - bug fixed, regression prevented
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

## 2. Schema Design (MANDATORY)

### A. Document Structure

```javascript
// User document with embedded data
{
  _id: ObjectId("..."),
  email: "user@example.com",
  profile: {
    firstName: "John",
    lastName: "Doe",
    avatar: "https://...",
    bio: "Software developer"
  },
  // Embed: One-to-few, always accessed together
  addresses: [
    {
      type: "home",
      street: "123 Main St",
      city: "New York",
      country: "US",
      isDefault: true
    }
  ],
  // Reference: One-to-many, accessed separately
  // Store count for quick access, fetch details separately
  orderCount: 42,
  createdAt: ISODate("2024-01-15T10:00:00Z"),
  updatedAt: ISODate("2024-01-15T10:00:00Z")
}

// Order document (separate collection)
{
  _id: ObjectId("..."),
  userId: ObjectId("..."),  // Reference to user
  orderNumber: "ORD-2024-00001",
  status: "delivered",
  items: [
    {
      productId: ObjectId("..."),
      name: "Product Name",  // Denormalized for display
      price: 29.99,
      quantity: 2
    }
  ],
  totals: {
    subtotal: 59.98,
    tax: 5.40,
    shipping: 5.00,
    total: 70.38
  },
  shippingAddress: {
    // Snapshot at order time (not reference)
    street: "123 Main St",
    city: "New York",
    country: "US"
  },
  createdAt: ISODate("2024-01-15T10:00:00Z")
}
```

### B. Embedding vs Referencing

```javascript
// EMBED when:
// - One-to-one relationship
// - One-to-few (< 100 items)
// - Data always accessed together
// - No independent access needed

// ✅ Good embedding
{
  _id: ObjectId("..."),
  title: "Blog Post",
  author: {
    userId: ObjectId("..."),
    name: "John Doe",  // Denormalized
    avatar: "https://..."
  },
  // Comments: embed if few, recent, or always shown
  recentComments: [
    { userId: ObjectId("..."), text: "Great!", createdAt: ISODate("...") }
  ],
  commentCount: 42
}

// REFERENCE when:
// - One-to-many (> 100 items)
// - Many-to-many relationships
// - Independent access needed
// - Data updated frequently

// ✅ Good referencing
// posts collection
{
  _id: ObjectId("..."),
  title: "Blog Post",
  authorId: ObjectId("...")  // Reference
}

// comments collection (separate)
{
  _id: ObjectId("..."),
  postId: ObjectId("..."),  // Reference
  userId: ObjectId("..."),
  text: "Comment text",
  createdAt: ISODate("...")
}
```

### C. Schema Validation

```javascript
db.createCollection("users", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["email", "createdAt"],
      properties: {
        email: {
          bsonType: "string",
          pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
          description: "Must be a valid email"
        },
        profile: {
          bsonType: "object",
          properties: {
            firstName: { bsonType: "string", maxLength: 100 },
            lastName: { bsonType: "string", maxLength: 100 }
          }
        },
        status: {
          enum: ["active", "inactive", "suspended"],
          description: "Must be a valid status"
        },
        createdAt: {
          bsonType: "date"
        }
      }
    }
  },
  validationLevel: "strict",
  validationAction: "error"
});
```

---

## 3. Indexing (MANDATORY)

### A. Index Types

```javascript
// Single field index
db.users.createIndex({ email: 1 });

// Compound index (order matters!)
db.orders.createIndex({ userId: 1, status: 1, createdAt: -1 });

// Unique index
db.users.createIndex({ email: 1 }, { unique: true });

// Partial index (index subset of documents)
db.orders.createIndex(
  { createdAt: -1 },
  { partialFilterExpression: { status: "pending" } }
);

// TTL index (auto-delete old documents)
db.sessions.createIndex(
  { expiresAt: 1 },
  { expireAfterSeconds: 0 }
);

// Text index (full-text search)
db.posts.createIndex({ title: "text", content: "text" });

// Geospatial index
db.locations.createIndex({ coordinates: "2dsphere" });

// Wildcard index (for dynamic fields)
db.products.createIndex({ "attributes.$**": 1 });
```

### B. Index Strategy

```javascript
// ESR Rule: Equality, Sort, Range
// For query: { status: "active", createdAt: { $gte: date } } sorted by priority
db.tasks.createIndex({ status: 1, priority: -1, createdAt: 1 });
//                      Equality   Sort        Range

// Covered query (all fields in index)
db.users.createIndex({ email: 1, status: 1 });
db.users.find(
  { email: "test@example.com" },
  { status: 1, _id: 0 }  // Projection only includes indexed fields
);

// Check if query uses index
db.users.find({ email: "test@example.com" }).explain("executionStats");
```

### C. Index Analysis

```javascript
// List all indexes
db.collection.getIndexes();

// Index statistics
db.collection.aggregate([{ $indexStats: {} }]);

// Find unused indexes (in production)
// Look for indexes with low usage in $indexStats

// Analyze query
db.collection.find({ ... }).explain("executionStats");
// Look for:
// - IXSCAN (good) vs COLLSCAN (bad)
// - nReturned vs totalDocsExamined ratio
```

---

## 4. Query Patterns (MANDATORY)

### A. Basic Queries

```javascript
// Find with projection
db.users.find(
  { status: "active" },
  { email: 1, profile: 1, _id: 0 }
);

// Find one
db.users.findOne({ email: "user@example.com" });

// Comparison operators
db.orders.find({
  total: { $gte: 100, $lte: 500 },
  status: { $in: ["pending", "processing"] },
  cancelledAt: { $exists: false }
});

// Array queries
db.posts.find({ tags: "mongodb" });  // Array contains
db.posts.find({ tags: { $all: ["mongodb", "tutorial"] } });  // Array contains all
db.users.find({ "addresses.city": "New York" });  // Nested in array
```

### B. Aggregation Pipeline

```javascript
// Sales report by month
db.orders.aggregate([
  // Stage 1: Filter
  { $match: {
    status: "delivered",
    createdAt: { $gte: ISODate("2024-01-01") }
  }},

  // Stage 2: Group by month
  { $group: {
    _id: { $dateToString: { format: "%Y-%m", date: "$createdAt" } },
    totalRevenue: { $sum: "$totals.total" },
    orderCount: { $sum: 1 },
    avgOrderValue: { $avg: "$totals.total" }
  }},

  // Stage 3: Sort
  { $sort: { _id: -1 } },

  // Stage 4: Format output
  { $project: {
    month: "$_id",
    totalRevenue: { $round: ["$totalRevenue", 2] },
    orderCount: 1,
    avgOrderValue: { $round: ["$avgOrderValue", 2] }
  }}
]);

// Lookup (join)
db.orders.aggregate([
  { $match: { userId: ObjectId("...") } },
  { $lookup: {
    from: "users",
    localField: "userId",
    foreignField: "_id",
    as: "user"
  }},
  { $unwind: "$user" }
]);

// Faceted search
db.products.aggregate([
  { $match: { $text: { $search: "laptop" } } },
  { $facet: {
    results: [
      { $skip: 0 },
      { $limit: 20 }
    ],
    totalCount: [
      { $count: "count" }
    ],
    byCategory: [
      { $group: { _id: "$category", count: { $sum: 1 } } }
    ],
    priceRanges: [
      { $bucket: {
        groupBy: "$price",
        boundaries: [0, 500, 1000, 2000, Infinity],
        default: "Other",
        output: { count: { $sum: 1 } }
      }}
    ]
  }}
]);
```

### C. Pagination

```javascript
// ❌ SLOW: Skip-based pagination (avoid for large offsets)
db.posts.find().sort({ createdAt: -1 }).skip(10000).limit(20);

// ✅ FAST: Cursor-based pagination
db.posts.find({
  createdAt: { $lt: ISODate("2024-01-15T10:00:00Z") }
}).sort({ createdAt: -1 }).limit(20);

// With compound sort (createdAt + _id for uniqueness)
db.posts.find({
  $or: [
    { createdAt: { $lt: lastCreatedAt } },
    { createdAt: lastCreatedAt, _id: { $lt: lastId } }
  ]
}).sort({ createdAt: -1, _id: -1 }).limit(20);
```

---

## 5. Write Operations

### A. Insert

```javascript
// Single insert
db.users.insertOne({
  email: "new@example.com",
  createdAt: new Date()
});

// Bulk insert
db.users.insertMany([
  { email: "user1@example.com", createdAt: new Date() },
  { email: "user2@example.com", createdAt: new Date() }
], { ordered: false });  // Continue on error
```

### B. Update

```javascript
// Update one
db.users.updateOne(
  { _id: ObjectId("...") },
  {
    $set: { "profile.firstName": "Jane" },
    $currentDate: { updatedAt: true }
  }
);

// Update many
db.users.updateMany(
  { status: "inactive", lastLoginAt: { $lt: ISODate("2023-01-01") } },
  { $set: { status: "archived" } }
);

// Upsert
db.settings.updateOne(
  { userId: ObjectId("...") },
  { $set: { theme: "dark" } },
  { upsert: true }
);

// Array operations
db.users.updateOne(
  { _id: ObjectId("...") },
  {
    $push: { addresses: { type: "work", city: "Boston" } },
    $inc: { addressCount: 1 }
  }
);

// Update array element
db.users.updateOne(
  { _id: ObjectId("..."), "addresses.type": "home" },
  { $set: { "addresses.$.city": "Los Angeles" } }
);
```

### C. Delete

```javascript
// Delete one
db.sessions.deleteOne({ _id: ObjectId("...") });

// Delete many
db.logs.deleteMany({
  createdAt: { $lt: ISODate("2023-01-01") }
});

// Soft delete (recommended)
db.users.updateOne(
  { _id: ObjectId("...") },
  { $set: { deletedAt: new Date() } }
);
```

---

## 6. Replication and High Availability (MANDATORY)

### A. Replica Set Configuration

```javascript
// Initialize replica set
rs.initiate({
  _id: "myReplicaSet",
  members: [
    { _id: 0, host: "mongo1.example.com:27017", priority: 2 },  // Preferred primary
    { _id: 1, host: "mongo2.example.com:27017", priority: 1 },
    { _id: 2, host: "mongo3.example.com:27017", priority: 0.5 }
  ]
});

// Add hidden member for analytics
rs.add({
  host: "mongo4.example.com:27017",
  priority: 0,
  hidden: true,
  tags: { workload: "analytics" }
});
```

### B. Write Concern (CRITICAL)

```javascript
// ✅ RECOMMENDED: Majority write concern with timeout
db.collection.insertOne(
  { data: "value" },
  { writeConcern: { w: "majority", wtimeout: 5000 } }
);

// Set global default
db.adminCommand({
  setDefaultRWConcern: 1,
  defaultWriteConcern: { w: "majority", wtimeout: 5000 }
});

// ❌ AVOID: w: 1 in production (data loss risk)
// ❌ NEVER: w: 0 (fire and forget)
```

**Write Concern Levels:**
- `w: "majority"` - Acknowledged by majority (prevents rollbacks)
- `w: 1` - Primary only (fastest, but rollback risk)
- `w: <number>` - Specific number of members
- `wtimeout` - ALWAYS set timeout to prevent indefinite blocking

**Critical Rules:**
- NEVER use P-S-A (Primary-Secondary-Arbiter) with `w: "majority"`
- ALWAYS deploy P-S-S (Primary-Secondary-Secondary) minimum
- NEVER deploy multiple arbiters

### C. Read Concern

```javascript
// Majority - no rollback risk (recommended for critical reads)
db.collection.find().readConcern("majority");

// Local - most recent data (may be rolled back)
db.collection.find().readConcern("local");

// Linearizable - guarantees ordering (high latency cost)
db.collection.find()
  .readConcern("linearizable")
  .maxTimeMS(10000);  // ALWAYS set timeout
```

### D. Read Preference

```javascript
// Primary (default) - lowest latency, most current
db.collection.find().readPreference("primary");

// Secondary preferred - offload read load
db.collection.find().readPreference("secondaryPreferred");

// Nearest - lowest network latency
db.collection.find().readPreference("nearest");

// Tagged - route to specific members
db.collection.find().readPreference(
  "secondary",
  [{ workload: "analytics" }]
);
```

**Best Practices:**
- User-facing queries: `primary` or `nearest`
- Background jobs: `secondaryPreferred`
- Analytics: `secondary` with tags
- Distributed transactions: MUST use `primary`

### E. Oplog Management

```javascript
// Check oplog window
rs.printReplicationInfo();

// Resize oplog (MongoDB 4.0+)
db.adminCommand({
  replSetResizeOplog: 1,
  size: 16000  // MB
});
```

**Sizing Guidelines:**
- Minimum 24 hours for resharding
- 72+ hours for production systems
- Size for expected downtime windows

---

## 7. Sharding and Horizontal Scaling (MANDATORY)

### A. When to Shard

**Shard When:**
- Data exceeds single server storage capacity
- Read/write throughput exceeds single replica set capacity
- Application growth trajectory demands horizontal scaling

**DON'T Shard If:**
- Vertical scaling (larger servers) is sufficient
- Can optimize with indexing/data modeling
- Complexity outweighs benefits

### B. Shard Key Selection (CRITICAL)

```javascript
// Analyze shard key performance (MongoDB 7.0+)
db.adminCommand({
  analyzeShardKey: "mydb.collection",
  key: { userId: 1, timestamp: 1 }
});
```

**Selection Criteria:**

**High Cardinality:**
```javascript
// ❌ BAD: Low cardinality (2-3 values)
sh.shardCollection("mydb.users", { country: 1 });

// ✅ GOOD: High cardinality
sh.shardCollection("mydb.users", { userId: 1 });

// ✅ BETTER: Compound key (low + high cardinality)
sh.shardCollection("mydb.users", { country: 1, userId: 1 });
```

**Avoid Monotonic Keys:**
```javascript
// ❌ BAD: Monotonic _id causes hotspotting
sh.shardCollection("mydb.events", { _id: 1 });

// ✅ GOOD: Hashed shard key distributes writes
sh.shardCollection("mydb.events", { _id: "hashed" });

// ✅ BEST: Compound with high-cardinality prefix
sh.shardCollection("mydb.events", { userId: 1, timestamp: 1 });
```

### C. Shard Strategies

```javascript
// Hashed sharding - even distribution
sh.shardCollection("mydb.users", { userId: "hashed" });

// Range-based sharding - query efficiency
sh.shardCollection("mydb.timeseries", { timestamp: 1, deviceId: 1 });
```

### D. Zone Sharding (Geo-Distribution)

```javascript
// Define zones
sh.addShardTag("shard0000", "US_EAST");
sh.addShardTag("shard0001", "EU_WEST");
sh.addShardTag("shard0002", "APAC");

// Assign data ranges to zones
sh.addTagRange(
  "mydb.users",
  { country: "US", userId: MinKey },
  { country: "US", userId: MaxKey },
  "US_EAST"
);

sh.addTagRange(
  "mydb.users",
  { country: "GB", userId: MinKey },
  { country: "GB", userId: MaxKey },
  "EU_WEST"
);
```

**Use Cases:**
- Data locality (GDPR compliance)
- Hardware-based routing (SSD vs HDD shards)
- Latency optimization (regional deployment)

### E. Resharding (MongoDB 8.0+)

```javascript
// New fast method (MongoDB 8.0)
sh.shardAndDistributeCollection(
  "mydb.users",
  { newKey: 1 },
  { unique: false }
);

// Traditional resharding
db.adminCommand({
  reshardCollection: "mydb.users",
  key: { newKey: 1 }
});
```

**Pre-Resharding Checklist:**
- [ ] Oplog window ≥ 24 hours
- [ ] I/O capacity < 50%
- [ ] CPU load < 80%
- [ ] Sufficient storage: `((size + index_size) * 2) / shard_count`

**Performance:** MongoDB 8.0 is 3x faster, 50% less memory vs 7.0

---

## 8. Schema Versioning and Migration (MANDATORY)

### A. Schema Versioning Pattern

```javascript
// Version 1 (legacy)
{
  _id: ObjectId("..."),
  schema_version: 1,
  name: "John Doe",
  addr: "123 Main St, City, ST 12345"
}

// Version 2 (current)
{
  _id: ObjectId("..."),
  schema_version: 2,
  fullName: "John Doe",
  address: {
    street: "123 Main St",
    city: "City",
    state: "ST",
    zip: "12345"
  }
}
```

### B. Migration Strategies

**1. Lazy Migration (Zero Downtime):**
```javascript
async function getUser(userId) {
  let user = await db.users.findOne({ _id: userId });

  if (user.schema_version === 1) {
    // Migrate on read
    user = migrateV1toV2(user);
    await db.users.replaceOne({ _id: userId }, user);
  }

  return user;
}

function migrateV1toV2(doc) {
  const [street, city, state, zip] = doc.addr.split(', ');
  return {
    ...doc,
    schema_version: 2,
    fullName: doc.name,
    address: { street, city, state, zip },
    name: undefined,  // Remove old fields
    addr: undefined
  };
}
```

**2. Eager Migration (Immediate):**
```javascript
db.users.updateMany(
  { schema_version: 1 },
  [
    {
      $set: {
        schema_version: 2,
        fullName: "$name",
        address: {
          $let: {
            vars: { parts: { $split: ["$addr", ", "] } },
            in: {
              street: { $arrayElemAt: ["$$parts", 0] },
              city: { $arrayElemAt: ["$$parts", 1] },
              state: { $arrayElemAt: ["$$parts", 2] },
              zip: { $arrayElemAt: ["$$parts", 3] }
            }
          }
        }
      }
    },
    { $unset: ["name", "addr"] }
  ]
);
```

**3. Incremental Migration:**
```javascript
async function incrementalMigration(batchSize = 1000) {
  let processed = 0;

  while (true) {
    const result = await db.users.updateMany(
      { schema_version: 1 },
      [/* migration pipeline */],
      { limit: batchSize }
    );

    processed += result.modifiedCount;
    if (result.modifiedCount === 0) break;

    // Pause to avoid overwhelming system
    await sleep(5000);
  }

  return processed;
}
```

### C. Index Management During Migration

```javascript
// Create dual indexes during transition
db.users.createIndex({ "addr": 1 });           // v1
db.users.createIndex({ "address.street": 1 }); // v2

// Drop old index after migration complete
db.users.dropIndex({ "addr": 1 });
```

---

## 9. Transactions

```javascript
// Multi-document transaction
const session = client.startSession();

try {
  session.startTransaction();

  // Transfer money
  await accounts.updateOne(
    { _id: fromAccountId },
    { $inc: { balance: -amount } },
    { session }
  );

  await accounts.updateOne(
    { _id: toAccountId },
    { $inc: { balance: amount } },
    { session }
  );

  await transactions.insertOne({
    from: fromAccountId,
    to: toAccountId,
    amount,
    createdAt: new Date()
  }, { session });

  await session.commitTransaction();
} catch (error) {
  await session.abortTransaction();
  throw error;
} finally {
  session.endSession();
}
```

---

## 10. Connection Pooling (MANDATORY)

### A. Configuration

```javascript
// Node.js/Mongoose
const mongooseOptions = {
  maxPoolSize: 100,      // Maximum concurrent connections
  minPoolSize: 10,       // Keep connections warm
  maxIdleTimeMS: 60000,  // Close idle connections
  waitQueueTimeoutMS: 10000,
  serverSelectionTimeoutMS: 5000
};

mongoose.connect(uri, mongooseOptions);

// Python Motor/PyMongo
client = MongoClient(
    uri,
    maxPoolSize=100,
    minPoolSize=10,
    maxIdleTimeMS=60000,
    waitQueueTimeoutMS=10000
)

// Java
MongoClientSettings settings = MongoClientSettings.builder()
    .applyConnectionString(new ConnectionString(uri))
    .applyToConnectionPoolSettings(builder ->
        builder.maxSize(100)
               .minSize(10)
               .maxWaitTime(10, TimeUnit.SECONDS))
    .build();
```

### B. Sizing Guidelines

```javascript
// Calculate total connections to cluster
// instances × maxPoolSize × replica_set_members

// Example: 10 app instances, maxPoolSize 100, 3-member replica set
// Total: 10 × 100 × 3 = 3,000 connections
```

**Best Practices:**
- Set `maxPoolSize` based on actual concurrent operations (not arbitrary limits)
- Use `minPoolSize` to maintain warm connections during traffic spikes
- Monitor connection pool metrics (checkouts, waits, timeouts)
- Ensure MongoDB `maxIncomingConnections` can handle total load
- Default MongoDB limit: 65,536 connections

### C. Monitoring

```javascript
// Check current connections
db.serverStatus().connections;

// Monitor pool statistics
db.adminCommand({ connPoolStats: 1 });
```

---

## 11. Change Streams (Real-Time Sync)

### A. Basic Change Stream

```javascript
// Watch all changes
const changeStream = db.collection.watch();

changeStream.on("change", (change) => {
  console.log("Change detected:", change);
});

// Watch specific operations
const changeStream = db.collection.watch([
  { $match: { operationType: "insert" } }
]);

changeStream.on("change", async (change) => {
  await processNewDocument(change.fullDocument);
});
```

### B. Filtered Change Streams

```javascript
// Watch specific fields
const pipeline = [
  { $match: {
      "fullDocument.status": "pending",
      operationType: { $in: ["insert", "update"] }
    }
  },
  { $project: {
      _id: 1,
      fullDocument: 1,
      updateDescription: 1
    }
  }
];

const changeStream = db.orders.watch(pipeline);
```

### C. Resume Tokens (Fault Tolerance)

```javascript
let resumeToken;

const changeStream = db.collection.watch();

changeStream.on("change", async (change) => {
  // Process change
  await handleChange(change);

  // Save resume token
  resumeToken = changeStream.resumeToken;
  await saveResumeToken(resumeToken);
});

// Resume from saved token after restart
const changeStream = db.collection.watch([], {
  resumeAfter: await loadResumeToken()
});
```

### D. Production Best Practices

**Oplog Sizing:**
```javascript
// Check oplog window
rs.printReplicationInfo();

// Minimum 24-hour window for change streams
// Increase for longer expected downtime
db.adminCommand({
  replSetResizeOplog: 1,
  size: 16000  // MB
});
```

**Performance Considerations:**
- Change streams cannot use indexes
- High number of streams impacts cluster performance
- Use `postBatchResumeToken` for optimal resume performance
- Use `$changeStreamSplitLargeEvent` (6.0.9+) for documents near 16MB limit

**Error Handling:**
```javascript
changeStream.on("error", async (error) => {
  if (error.code === 136) {
    // CappedPositionLost - oplog rolled over
    // Restart from beginning or last checkpoint
    changeStream = db.collection.watch([], {
      startAtOperationTime: lastCheckpointTime
    });
  } else {
    console.error("Change stream error:", error);
    // Implement exponential backoff retry
  }
});
```

---

## 12. Time Series Collections

### A. Creating Time Series Collections

```javascript
// Create time series collection
db.createCollection("sensor_data", {
  timeseries: {
    timeField: "timestamp",
    metaField: "deviceId",
    granularity: "seconds"  // seconds, minutes, or hours
  }
});

// With expiration (automatic deletion)
db.createCollection("metrics", {
  timeseries: {
    timeField: "timestamp",
    metaField: "sensorId",
    granularity: "minutes"
  },
  expireAfterSeconds: 2592000  // 30 days
});
```

### B. Inserting Time Series Data

```javascript
// Single insert
db.sensor_data.insertOne({
  timestamp: new Date(),
  deviceId: "sensor-001",
  temperature: 23.5,
  humidity: 65.2,
  location: { lat: 40.7128, lng: -74.0060 }
});

// Bulk insert (recommended for performance)
db.sensor_data.insertMany([
  {
    timestamp: ISODate("2026-02-06T10:00:00Z"),
    deviceId: "sensor-001",
    temperature: 23.5,
    humidity: 65.2
  },
  {
    timestamp: ISODate("2026-02-06T10:01:00Z"),
    deviceId: "sensor-001",
    temperature: 23.6,
    humidity: 65.1
  }
], { ordered: false });  // Unordered for better performance
```

### C. Querying Time Series Data

```javascript
// Range query
db.sensor_data.find({
  timestamp: {
    $gte: ISODate("2026-02-06T00:00:00Z"),
    $lt: ISODate("2026-02-07T00:00:00Z")
  },
  deviceId: "sensor-001"
});

// Aggregation with time bucketing
db.sensor_data.aggregate([
  {
    $match: {
      deviceId: "sensor-001",
      timestamp: { $gte: ISODate("2026-02-01T00:00:00Z") }
    }
  },
  {
    $group: {
      _id: {
        $dateTrunc: {
          date: "$timestamp",
          unit: "hour"
        }
      },
      avgTemperature: { $avg: "$temperature" },
      maxTemperature: { $max: "$temperature" },
      minTemperature: { $min: "$temperature" },
      count: { $sum: 1 }
    }
  },
  { $sort: { _id: 1 } }
]);
```

### D. Sharding Time Series Collections

```javascript
// Shard on metaField (not timeField - deprecated in 8.0)
sh.shardCollection(
  "mydb.sensor_data",
  { deviceId: 1, timestamp: 1 }
);
```

### E. Performance Optimization (MongoDB 8.0)

**Improvements:**
- 2-3x throughput vs MongoDB 7.0
- 10-20x lower cache usage
- Column-compressed storage
- Block processing for aggregations
- Automatic compound index on `metaField` + `timeField` (6.3+)

**Best Practices:**
```javascript
// ✅ Use stable metaField values
// ❌ Don't use frequently changing fields as metaField

// ✅ Batch identical metaField values
db.sensor_data.insertMany([
  { timestamp: t1, deviceId: "sensor-001", temp: 23.5 },
  { timestamp: t2, deviceId: "sensor-001", temp: 23.6 },
  { timestamp: t3, deviceId: "sensor-001", temp: 23.7 }
], { ordered: false });

// ✅ Round numeric precision
// Instead of: temperature: 23.5432789
// Use: temperature: 23.5 (1 decimal place if sufficient)

// ✅ Omit empty fields
// ❌ humidity: null
// ✅ Simply omit the field

// ✅ Query metaField scalar sub-fields
db.sensor_data.find({ "deviceId": "sensor-001" });
// ❌ Not: { metaField: { deviceId: "sensor-001" } }
```

---

## 13. Performance Optimization

### A. Latency Reduction (RTT Optimization)

**1. Geographic Distribution:**
```javascript
// Deploy replica sets/shards close to users
// Use zone sharding for geo-distributed data
sh.addShardTag("shard0000", "US_EAST");
sh.addTagRange(
  "mydb.users",
  { country: "US", userId: MinKey },
  { country: "US", userId: MaxKey },
  "US_EAST"
);

// Read preference for lowest latency
db.collection.find().readPreference("nearest");
```

**2. Data Modeling for Low Latency:**
```javascript
// ❌ BAD: Normalized (requires lookup)
// users: { _id, name, address_id }
// addresses: { _id, street, city }
// Requires $lookup (join) - 2 queries

// ✅ GOOD: Embedded (single read)
users: {
  _id,
  name,
  address: { street, city }  // Single document read
}
```

**3. Field Name Optimization:**
```javascript
// ❌ BAD: Long field names increase document size
{
  userIdentificationNumber: "12345",
  customerEmailAddress: "user@example.com",
  accountCreationTimestamp: ISODate("2026-01-01")
}

// ✅ GOOD: Short field names reduce network transfer
{
  uid: "12345",
  email: "user@example.com",
  created: ISODate("2026-01-01")
}
```

**4. Transaction Batching:**
```javascript
// ❌ BAD: Individual writes with replication wait
for (let doc of docs) {
  await db.collection.updateOne(
    { _id: doc._id },
    { $set: doc },
    { writeConcern: { w: "majority" } }
  );
  // Each waits for replication
}

// ✅ GOOD: Transaction batches replication
const session = client.startSession();
session.startTransaction({ writeConcern: { w: "majority" } });

try {
  for (let doc of docs) {
    await db.collection.updateOne(
      { _id: doc._id },
      { $set: doc },
      { session }
    );
  }
  await session.commitTransaction();  // Single replication wait
} finally {
  await session.endSession();
}
```

### B. Query Optimization

```javascript
// Always use projection
db.users.find({}, { email: 1, status: 1 });  // Only fetch needed fields

// Use covered queries
db.users.createIndex({ email: 1, status: 1 });
db.users.find(
  { email: "test@example.com" },
  { email: 1, status: 1, _id: 0 }
).explain();  // Should show totalDocsExamined: 0

// Avoid $where and $regex without index
// ❌ SLOW
db.users.find({ $where: "this.email.length > 10" });

// ✅ FAST
db.users.find({ email: /^test/i });  // Prefix match can use index

// Limit returned documents
db.logs.find({ level: "error" }).sort({ timestamp: -1 }).limit(100);
```

### C. Bulk Operations

```javascript
// ❌ BAD: Individual inserts
for (const doc of documents) {
  await db.collection.insertOne(doc);
}

// ✅ GOOD: Bulk unordered insert
await db.collection.insertMany(documents, { ordered: false });

// Bulk write operations
const bulkOps = [
  { insertOne: { document: { name: "Item 1" } } },
  { updateOne: {
    filter: { _id: ObjectId("...") },
    update: { $set: { status: "active" } }
  }},
  { deleteOne: { filter: { _id: ObjectId("...") } } }
];

db.collection.bulkWrite(bulkOps, { ordered: false });
```

**Ordered vs Unordered:**
- `ordered: true` (default) - Stops on first error
- `ordered: false` - Continues on errors, parallel execution, higher throughput

### D. Write Concern Tuning

```javascript
// High durability (slower)
{ writeConcern: { w: "majority", j: true, wtimeout: 5000 } }

// Balanced (recommended)
{ writeConcern: { w: "majority", j: false, wtimeout: 5000 } }

// High throughput (acknowledge from primary only)
{ writeConcern: { w: 1, j: false } }
```

### E. Monitoring and Troubleshooting

```javascript
// Check slow queries
db.system.profile.find({ millis: { $gt: 100 } }).sort({ ts: -1 }).limit(10);

// Enable profiling
db.setProfilingLevel(1, { slowms: 100 });  // Log queries > 100ms

// Analyze query performance
db.collection.find({ ... }).explain("executionStats");

// Key metrics to check:
// - nReturned vs totalDocsExamined (should be close)
// - executionTimeMillis
// - Stage: IXSCAN (good) vs COLLSCAN (bad)
```

**Common Latency Issues:**
- Missing or incorrect indexes
- Large in-memory sorts
- Working set exceeding RAM
- Network bandwidth constraints
- CPU/disk I/O saturation
- Replication lag

---

## 14. Security (MANDATORY)

### A. Authentication

**SCRAM (Default Mechanism):**
```javascript
// Create admin user first
use admin
db.createUser({
  user: "admin",
  pwd: passwordPrompt(),  // Secure password entry
  roles: ["root"]
});

// Enable authentication
// mongod --auth

// Create application users with least privilege
use mydb
db.createUser({
  user: "app_user",
  pwd: passwordPrompt(),
  roles: [
    { role: "readWrite", db: "mydb" }
  ]
});
```

**X.509 Certificate Authentication:**
```javascript
// Start MongoDB with X.509
mongod --clusterAuthMode x509 \
       --tlsMode requireTLS \
       --tlsCertificateKeyFile /path/to/server.pem \
       --tlsCAFile /path/to/ca.pem

// Create X.509 user
db.getSiblingDB("$external").runCommand({
  createUser: "CN=client,OU=Org,O=Company,L=City,ST=State,C=US",
  roles: [
    { role: "readWrite", db: "mydb" }
  ]
});

// Connection string
mongodb://hostname:27017/?authMechanism=MONGODB-X509&tls=true&tlsCertificateKeyFile=/path/to/client.pem
```

### B. Authorization (Least Privilege)

**Custom Roles:**
```javascript
// Create role with specific privileges
use admin
db.createRole({
  role: "appReadWriteRestricted",
  privileges: [
    {
      resource: { db: "mydb", collection: "users" },
      actions: ["find", "insert", "update"]
    },
    {
      resource: { db: "mydb", collection: "logs" },
      actions: ["insert"]  // Write-only for logs
    }
  ],
  roles: []
});

// Assign role to user
db.createUser({
  user: "app_service",
  pwd: passwordPrompt(),
  roles: ["appReadWriteRestricted"]
});
```

**Built-in Roles (Use Sparingly):**
- `read` - Read data on non-system collections
- `readWrite` - Read and modify data
- `dbAdmin` - Schema management, indexing
- `userAdmin` - Create/modify users and roles
- `clusterAdmin` - Cluster management
- `root` - Superuser (AVOID in production)

### C. Encryption

**1. Encryption at Rest:**
```javascript
// Enable with key file
mongod --enableEncryption \
       --encryptionKeyFile /path/to/keyfile

// With KMIP server (Enterprise)
mongod --enableEncryption \
       --kmipServerName kmip.example.com \
       --kmipPort 5696 \
       --kmipClientCertificateFile /path/to/client.pem
```

**2. Encryption in Transit (TLS):**
```javascript
// MongoDB Atlas: TLS enabled by default (cannot be disabled)

// Self-managed deployment
mongod --tlsMode requireTLS \
       --tlsCertificateKeyFile /path/to/server.pem \
       --tlsCAFile /path/to/ca.pem

// Connection string
mongodb://hostname:27017/?tls=true&tlsCertificateKeyFile=/path/to/client.pem
```

**3. Queryable Encryption (In-Use Encryption):**
```javascript
// Client-side field encryption configuration
const encryptedFieldsMap = {
  "mydb.users": {
    fields: [
      {
        path: "ssn",
        bsonType: "string",
        queries: { queryType: "equality" }
      },
      {
        path: "salary",
        bsonType: "int",
        queries: {
          queryType: "range",
          min: 0,
          max: 1000000,
          sparsity: 1
        }
      },
      {
        path: "email",
        bsonType: "string",
        queries: {
          queryType: "prefix",  // Substring queries (MongoDB 8.2+)
          minLength: 3
        }
      }
    ]
  }
};

// Create encrypted collection
const client = new MongoClient(uri, {
  autoEncryption: {
    keyVaultNamespace: "encryption.__keyVault",
    kmsProviders: {
      aws: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
      }
    },
    encryptedFieldsMap
  }
});

// Queries work transparently
await db.users.findOne({ ssn: "123-45-6789" });  // Encrypted query
await db.users.find({ salary: { $gte: 50000, $lte: 100000 } });  // Range query
await db.users.find({ email: /^user@/ });  // Prefix query
```

**Queryable Encryption Features:**
- Equality queries
- Range queries (numeric, date)
- Prefix/suffix/substring queries (8.2+ preview)
- Data encrypted client-side before transmission
- Server cannot decrypt (protected at rest, in transit, in use)

**CSFLE vs Queryable Encryption:**

Use **Queryable Encryption** when:
- Developing new applications
- Need range/prefix/suffix queries on encrypted data
- Can use single key per field

Use **CSFLE** when:
- Existing CSFLE implementation
- Need different keys for same field (multi-tenant, user-specific keys)
- Only equality queries required

### D. Auditing (MongoDB Enterprise / Atlas)

**Enable Auditing:**
```javascript
// Enterprise Server
mongod --auditDestination file \
       --auditFormat JSON \
       --auditPath /var/log/mongodb/audit.json \
       --auditFilter '{
         atype: {
           $in: [
             "authenticate",
             "createUser",
             "dropUser",
             "dropCollection",
             "createCollection"
           ]
         }
       }'

// Audit specific operations
mongod --auditDestination syslog \
       --auditFilter '{
         $or: [
           { "param.ns": "mydb.sensitive_collection" },
           { atype: "dropDatabase" }
         ]
       }'
```

**Atlas Auditing:**
- All database operations (CRUD)
- Schema modifications
- User/role management
- Encryption key management
- Infrastructure changes

### E. Network Security

```javascript
// Bind to specific interfaces
mongod --bind_ip localhost,10.0.1.5

// IP whitelisting (Atlas)
// Configure in Atlas UI or API

// VPC Peering (Atlas)
// AWS: Configure VPC peering connection
// Azure: Virtual network peering
// GCP: VPC Network Peering

// PrivateLink (Atlas)
// AWS PrivateLink / Azure Private Link
// Keep traffic within cloud provider network
```

### F. Security Checklist

**Pre-Production:**
- [ ] Authentication enabled (`--auth`)
- [ ] TLS/SSL enabled (`--tlsMode requireTLS`)
- [ ] Encryption at rest configured
- [ ] Queryable encryption for sensitive fields
- [ ] Least privilege roles assigned
- [ ] Custom roles created (not using root/admin)
- [ ] Network access restricted (firewall/VPC)
- [ ] Auditing enabled (Enterprise/Atlas)
- [ ] Regular backups configured
- [ ] MongoDB version supported (not EOL)

**Ongoing:**
- [ ] Monitor MongoDB CVEs
- [ ] Rotate encryption keys periodically
- [ ] Review user permissions quarterly
- [ ] Audit logs reviewed regularly
- [ ] Security patches applied promptly
- [ ] Run with dedicated user (not root)

---

## 15. Mongoose ODM

### A. Schema Definition

```javascript
// models/user.js
const mongoose = require('mongoose');

const addressSchema = new mongoose.Schema({
  type: { type: String, enum: ['home', 'work', 'other'], default: 'home' },
  street: { type: String, required: true },
  city: { type: String, required: true },
  country: { type: String, required: true },
  isDefault: { type: Boolean, default: false }
}, { _id: false });

const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: [true, 'Email is required'],
    unique: true,
    lowercase: true,
    trim: true,
    match: [/^\S+@\S+\.\S+$/, 'Invalid email format']
  },
  password: {
    type: String,
    required: true,
    select: false  // Don't include by default
  },
  profile: {
    firstName: { type: String, maxlength: 100 },
    lastName: { type: String, maxlength: 100 },
    avatar: String
  },
  addresses: [addressSchema],
  status: {
    type: String,
    enum: ['active', 'inactive', 'suspended'],
    default: 'active'
  },
  deletedAt: Date
}, {
  timestamps: true,  // Auto createdAt, updatedAt
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Virtual field
userSchema.virtual('fullName').get(function() {
  return `${this.profile.firstName} ${this.profile.lastName}`;
});

// Index
userSchema.index({ email: 1 });
userSchema.index({ status: 1, createdAt: -1 });

// Pre-save hook
userSchema.pre('save', async function(next) {
  if (this.isModified('password')) {
    this.password = await bcrypt.hash(this.password, 10);
  }
  next();
});

// Method
userSchema.methods.comparePassword = async function(password) {
  return bcrypt.compare(password, this.password);
};

// Static method
userSchema.statics.findByEmail = function(email) {
  return this.findOne({ email: email.toLowerCase() });
};

// Query helper
userSchema.query.active = function() {
  return this.where({ status: 'active', deletedAt: null });
};

module.exports = mongoose.model('User', userSchema);
```

### B. Usage

```javascript
const User = require('./models/user');

// Create
const user = await User.create({
  email: 'test@example.com',
  password: 'password123',
  profile: { firstName: 'John', lastName: 'Doe' }
});

// Find with query helper
const activeUsers = await User.find().active().sort({ createdAt: -1 });

// Find with population
const orders = await Order.find({ userId: user._id })
  .populate('userId', 'email profile')
  .sort({ createdAt: -1 })
  .lean();  // Return plain objects (faster)

// Update
await User.findByIdAndUpdate(
  user._id,
  { $set: { status: 'inactive' } },
  { new: true, runValidators: true }
);

// Lean queries for read performance
const users = await User.find({ status: 'active' }).lean();
```

---

## 16. Deployment Checklist

### Pre-Production Requirements

#### Schema Design
- [ ] Documents designed for application access patterns
- [ ] Appropriate embedding vs referencing decisions
- [ ] Schema validation configured
- [ ] Proper data types used (no string overuse)
- [ ] Schema versioning strategy defined
- [ ] Migration plan documented (lazy/eager/incremental)

#### Indexing
- [ ] Indexes for all query patterns
- [ ] Compound indexes follow ESR rule (Equality-Sort-Range)
- [ ] No unused indexes
- [ ] Unique indexes where needed
- [ ] Text indexes for search features
- [ ] Geospatial indexes for location queries
- [ ] Index sizes fit in RAM
- [ ] `analyzeShardKey` run for shard key validation

#### Replication and High Availability
- [ ] P-S-S topology minimum (NOT P-S-A)
- [ ] Geographic distribution across availability zones
- [ ] Write concern set to `w: "majority"` with `wtimeout`
- [ ] Read concerns configured appropriately
- [ ] Read preferences set (primary/secondaryPreferred/nearest)
- [ ] Oplog sized appropriately (minimum 24 hours, prefer 72+)
- [ ] Priority settings configured for primary election
- [ ] Hidden members for analytics/backups (if needed)

#### Sharding (If Applicable)
- [ ] Shard key selection validated with `analyzeShardKey`
- [ ] Shard key has high cardinality
- [ ] Avoid monotonic shard keys (use hashed or compound)
- [ ] Zone sharding configured for geo-distribution (if needed)
- [ ] Balancer window configured for off-peak hours
- [ ] Pre-split chunks for new sharded collections
- [ ] Resharding plan documented

#### Performance
- [ ] Queries use projections (only fetch needed fields)
- [ ] Pagination is cursor-based (not skip-based)
- [ ] Aggregations optimized (early $match, use indexes)
- [ ] Connection pooling configured
- [ ] Connection pool sized correctly (instances × poolSize × members)
- [ ] Working set fits in RAM
- [ ] Disk I/O capacity adequate (use SSDs)
- [ ] CPU load acceptable (<80% peak)
- [ ] Transaction batching for multiple writes
- [ ] Time series collections for time-based data

#### Security (MANDATORY)
- [ ] Authentication enabled (`--auth`)
- [ ] TLS/SSL enabled (`--tlsMode requireTLS`)
- [ ] Encryption at rest configured
- [ ] Queryable encryption for sensitive data (SSN, PII)
- [ ] Users have minimal required roles (not root/admin)
- [ ] Custom roles created for specific access patterns
- [ ] Network access restricted (firewall/VPC/IP whitelist)
- [ ] X.509 certificates for client authentication (if needed)
- [ ] Auditing enabled (Enterprise/Atlas)
- [ ] MongoDB version supported (not EOL)
- [ ] Run with dedicated user (not root)
- [ ] Secrets management (AWS Secrets Manager/Vault)

#### Monitoring and Alerting
- [ ] Query performance monitoring (95th/99th percentile)
- [ ] Replication lag alerts
- [ ] Oplog window alerts
- [ ] Disk I/O utilization monitoring
- [ ] CPU usage monitoring
- [ ] Memory usage (working set vs RAM)
- [ ] Connection pool metrics (checkouts, waits, timeouts)
- [ ] Lock wait time monitoring
- [ ] Cache hit ratio tracking
- [ ] Slow query profiling enabled
- [ ] Atlas Real-Time Performance Panel (if using Atlas)

#### Backup and Disaster Recovery
- [ ] Automated backups configured
- [ ] Point-in-time recovery enabled
- [ ] Backup retention policy defined
- [ ] Restore procedures tested
- [ ] Disaster recovery runbook documented
- [ ] Multi-region backup storage (if applicable)
- [ ] Backup encryption enabled

#### Change Streams (If Applicable)
- [ ] Resume tokens stored persistently
- [ ] Oplog sized for expected downtime
- [ ] Error handling for `CappedPositionLost`
- [ ] Performance impact assessed
- [ ] `postBatchResumeToken` used for resume

### Capacity Planning

#### Data Growth
- [ ] Estimated data growth over 2-3 years
- [ ] Storage capacity planned (3x current size minimum)
- [ ] Vertical scaling headroom assessed
- [ ] Horizontal scaling trigger defined

#### Performance Capacity
- [ ] Required IOPS calculated for workload
- [ ] Network bandwidth requirements estimated
- [ ] Connection limits calculated (instances × poolSize × members)
- [ ] MongoDB `maxIncomingConnections` configured

### Ongoing Operations

#### Regular Maintenance
- [ ] Monitor MongoDB CVEs
- [ ] Upgrade to supported versions
- [ ] Rotate encryption keys periodically
- [ ] Review user permissions quarterly
- [ ] Audit logs reviewed regularly
- [ ] Unused indexes dropped
- [ ] Schema migrations completed
- [ ] Resharding performed if needed

#### Performance Tuning
- [ ] Slow queries analyzed and optimized
- [ ] Index usage reviewed (`$indexStats`)
- [ ] Working set size monitored
- [ ] Connection pool adjusted based on usage
- [ ] Write concern tuned for workload
- [ ] Read preferences optimized

---

## 17. Quick Reference

### Basic Operations

```javascript
// mongosh commands
show dbs                    // List databases
use mydb                    // Switch database
show collections            // List collections
db.collection.stats()       // Collection statistics
db.collection.getIndexes()  // List indexes

// Common queries
db.col.find({ field: value })
db.col.find({ field: { $gt: value } })
db.col.find({ field: { $in: [a, b] } })
db.col.find({ "nested.field": value })
db.col.find({ array: value })

// Updates
db.col.updateOne({ _id }, { $set: { field: value } })
db.col.updateOne({ _id }, { $inc: { count: 1 } })
db.col.updateOne({ _id }, { $push: { array: item } })
db.col.updateOne({ _id }, { $pull: { array: item } })
```

### Replication Commands

```javascript
// Replica set status
rs.status()
rs.isMaster()
rs.printReplicationInfo()      // Oplog info
rs.printSecondaryReplicationInfo()  // Lag info

// Configuration
rs.conf()
rs.reconfig(cfg)
rs.add("host:port")
rs.remove("host:port")

// Oplog management
db.adminCommand({ replSetResizeOplog: 1, size: 16000 })
```

### Sharding Commands

```javascript
// Shard status
sh.status()
db.printShardingStatus()

// Enable sharding
sh.enableSharding("mydb")
sh.shardCollection("mydb.collection", { key: 1 })
sh.shardCollection("mydb.collection", { key: "hashed" })

// Analyze shard key
db.adminCommand({
  analyzeShardKey: "mydb.collection",
  key: { userId: 1 }
})

// Zone sharding
sh.addShardTag("shard0", "US_EAST")
sh.addTagRange("mydb.col", { country: "US" }, { country: "US" }, "US_EAST")

// Resharding
sh.shardAndDistributeCollection("mydb.col", { newKey: 1 })
db.adminCommand({ reshardCollection: "mydb.col", key: { newKey: 1 } })

// Balancer
sh.getBalancerState()
sh.setBalancerState(true)
sh.startBalancer()
sh.stopBalancer()
```

### Indexing Commands

```javascript
// List indexes
db.collection.getIndexes()

// Index statistics
db.collection.aggregate([{ $indexStats: {} }])

// Find unused indexes (look for ops: 0 or low usage)
db.collection.aggregate([
  { $indexStats: {} },
  { $sort: { "accesses.ops": 1 } }
])

// Explain query
db.collection.find({ ... }).explain("executionStats")
```

### Performance Commands

```javascript
// Server status
db.serverStatus()
db.serverStatus().connections
db.serverStatus().opcounters

// Connection pool stats
db.adminCommand({ connPoolStats: 1 })

// Current operations
db.currentOp()
db.currentOp({ secs_running: { $gte: 5 } })  // Long-running ops

// Kill operation
db.killOp(opid)

// Profiling
db.getProfilingStatus()
db.setProfilingLevel(1, { slowms: 100 })  // Log queries > 100ms
db.system.profile.find().sort({ ts: -1 }).limit(10)

// Collection stats
db.collection.stats()
db.collection.storageSize()
db.collection.totalIndexSize()
```

### Security Commands

```javascript
// User management
db.createUser({ user: "name", pwd: "pass", roles: [...] })
db.updateUser("name", { roles: [...] })
db.dropUser("name")
db.getUsers()

// Role management
db.createRole({ role: "name", privileges: [...], roles: [...] })
db.grantRolesToUser("user", ["role"])
db.revokeRolesFromUser("user", ["role"])

// Authentication
db.auth("username", "password")

// TLS status
db.adminCommand({ serverStatus: 1 }).security
```

### Backup and Restore

```javascript
// mongodump (backup)
mongodump --uri="mongodb://..." --out=/backup/dir

// mongorestore (restore)
mongorestore --uri="mongodb://..." /backup/dir

// Point-in-time restore (Atlas/Ops Manager)
// Use UI or API

// Export/Import JSON
mongoexport --collection=col --db=mydb --out=col.json
mongoimport --collection=col --db=mydb --file=col.json
```

### Monitoring Queries

```javascript
// Check replication lag
rs.printSecondaryReplicationInfo()

// Check oplog window
rs.printReplicationInfo()

// Database size
db.stats()

// Collection count
db.collection.countDocuments()
db.collection.estimatedDocumentCount()  // Faster, uses metadata

// Index size vs collection size
db.collection.stats().indexSizes
db.collection.stats().storageSize
```

---

## 18. Why This Configuration Works

**Flexible Document Model**:
- Embedding related data within documents aligns storage with application access patterns, eliminating joins and delivering single-document reads for complex objects in sub-millisecond latency.

**Horizontal Scaling with Sharding**:
- Automatic data distribution across shards with zone-aware routing enables linear write scaling and geographic data locality without application-level partitioning logic.

**Change Streams for Event-Driven Architecture**:
- Real-time change streams provide database-level event notifications with resume tokens, enabling reliable event sourcing, cache invalidation, and cross-service synchronization without polling.

**Tunable Consistency and Durability**:
- Configurable read/write concerns per operation allow applications to balance between strong consistency (`majority`) and low latency (`local`) based on each operation's business requirements.

**Rich Query Language and Aggregation Pipeline**:
- The aggregation framework with $lookup, $merge, $unionWith, and window functions provides SQL-equivalent analytical capabilities while maintaining the flexibility of document-oriented storage.

---

**Last Updated:** 2026-02-06
**Version:** 2.0
**Maintainer:** Database Team

**Changelog:**
- v2.0 (2026-02-06): Added comprehensive sections on replication, sharding, schema versioning, connection pooling, change streams, time series collections, enhanced security (Queryable Encryption), latency optimization, and scalability patterns. Updated for MongoDB 8.0+ features.
- v1.0 (2026-01-31): Initial release with basic schema design, indexing, queries, and Mongoose ODM.


**End of MongoDB Development Guidelines**

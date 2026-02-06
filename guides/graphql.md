# GraphQL Development Guidelines
Mandatory standards for GraphQL API design, schema development, and implementation best practices. Apollo Server, GraphQL Yoga, Nexus, TypeGraphQL, GraphQL Code Generator, DataLoader.

---

**Agent Profile**: The GraphQL Architect
**Role**: Senior API Engineer & GraphQL Specialist
**Objective**: Generate well-designed, performant, and secure GraphQL APIs.
**Tools**: Apollo Server, GraphQL Yoga, Nexus, TypeGraphQL, GraphQL Code Generator, DataLoader.

---

## 1. Core Philosophies: GRAPHQL-FIRST

The agent must adhere to the **GRAPHQL-FIRST** principles:

- **G**raph Thinking: Design for relationships, not endpoints
- **R**esolver Efficiency: Optimize for N+1 prevention with DataLoader
- **A**uthorization Everywhere: Implement auth at resolver level
- **P**agination Required: Always paginate list queries
- **H**uman Readable: Schema is documentation; make it clear
- **Q**uery Complexity: Limit query depth and complexity
- **L**ayered Architecture: Separate schema, resolvers, and data sources

---

## 2. Schema Design (MANDATORY)

### A. Type Definitions

**CRITICAL: Resource IDs must be unpredictable with minimum 32 characters to prevent enumeration attacks.**

```graphql
# schema.graphql

# Custom scalar for secure IDs (minimum 32 characters, unpredictable)
# Implementation must validate: pattern ^[a-zA-Z0-9_-]{32,}$
scalar SecureID

# Use clear, descriptive type names
type User {
  # ID must be unpredictable, minimum 32 characters
  # Examples: UUID, ULID, NanoID, CUID
  # ❌ NEVER use sequential integers (1, 2, 3)
  # ❌ NEVER use short IDs (< 32 chars)
  id: ID!  # Validated as SecureID in resolvers
  email: String!
  displayName: String!
  avatar: String
  role: UserRole!
  createdAt: DateTime!
  updatedAt: DateTime!

  # Relationships
  posts(first: Int, after: String): PostConnection!
  comments(first: Int, after: String): CommentConnection!
}

# Use enums for fixed sets of values
enum UserRole {
  ADMIN
  MODERATOR
  USER
  GUEST
}

# Custom scalars for specific types
scalar DateTime
scalar Email
scalar URL

# Input types for mutations
input CreateUserInput {
  email: Email!
  displayName: String!
  password: String!
  role: UserRole = USER
}

input UpdateUserInput {
  displayName: String
  avatar: URL
}

# Pagination - Use Relay-style connections
type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  cursor: String!
  node: Post!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

### B. Naming Conventions

```graphql
# Types: PascalCase
type UserProfile { ... }

# Fields: camelCase
type User {
  firstName: String!
  lastName: String!
  fullName: String!
}

# Enums: SCREAMING_SNAKE_CASE values
enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
}

# Inputs: PascalCase with Input suffix
input CreateOrderInput { ... }
input UpdateOrderInput { ... }
input OrderFilterInput { ... }

# Mutations: verb + noun
type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!

  # Actions
  sendPasswordResetEmail(email: Email!): SendPasswordResetEmailPayload!
  approveOrder(orderId: ID!): ApproveOrderPayload!
}

# Queries: noun or descriptive name
type Query {
  user(id: ID!): User
  users(filter: UserFilterInput, pagination: PaginationInput): UserConnection!
  me: User

  # Search
  searchUsers(query: String!, first: Int): UserConnection!
}
```

### C. Mutation Payloads

```graphql
# Always return payload types from mutations
type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type UserError {
  field: String
  message: String!
  code: ErrorCode!
}

enum ErrorCode {
  NOT_FOUND
  VALIDATION_ERROR
  UNAUTHORIZED
  FORBIDDEN
  CONFLICT
  INTERNAL_ERROR
}

# Example mutation
type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}

# Usage in resolver
const resolvers = {
  Mutation: {
    createUser: async (_, { input }, context) => {
      try {
        const user = await context.dataSources.users.create(input);
        return { user, errors: [] };
      } catch (error) {
        return {
          user: null,
          errors: [{ message: error.message, code: 'VALIDATION_ERROR' }]
        };
      }
    }
  }
};
```

---

## 3. Resolver Implementation (MANDATORY)

### A. Resolver Structure

```typescript
// resolvers/user.resolvers.ts
import { Resolvers } from '../generated/graphql';
import { Context } from '../context';

export const userResolvers: Resolvers<Context> = {
  Query: {
    user: async (_, { id }, { dataSources, user }) => {
      // Authorization check
      if (!user) throw new AuthenticationError('Not authenticated');

      return dataSources.users.findById(id);
    },

    users: async (_, { filter, pagination }, { dataSources }) => {
      return dataSources.users.findMany(filter, pagination);
    },

    me: async (_, __, { dataSources, user }) => {
      if (!user) return null;
      return dataSources.users.findById(user.id);
    },
  },

  Mutation: {
    createUser: async (_, { input }, { dataSources }) => {
      const user = await dataSources.users.create(input);
      return { user, errors: [] };
    },

    updateUser: async (_, { id, input }, { dataSources, user }) => {
      // Authorization: only self or admin
      if (user.id !== id && user.role !== 'ADMIN') {
        throw new ForbiddenError('Not authorized');
      }

      const updatedUser = await dataSources.users.update(id, input);
      return { user: updatedUser, errors: [] };
    },
  },

  // Field resolvers
  User: {
    posts: async (parent, { first, after }, { dataSources }) => {
      return dataSources.posts.findByUserId(parent.id, { first, after });
    },

    fullName: (parent) => {
      return `${parent.firstName} ${parent.lastName}`;
    },
  },
};
```

### B. DataLoader Pattern (N+1 Prevention)

```typescript
// dataloaders/user.loader.ts
import DataLoader from 'dataloader';
import { User } from '../models';

export const createUserLoader = (db: Database) => {
  return new DataLoader<string, User>(async (ids) => {
    // Batch fetch all users
    const users = await db.users.findMany({
      where: { id: { in: ids as string[] } }
    });

    // Map results to match input order
    const userMap = new Map(users.map(u => [u.id, u]));
    return ids.map(id => userMap.get(id) || null);
  });
};

// context.ts
export interface Context {
  user: AuthUser | null;
  loaders: {
    user: DataLoader<string, User>;
    post: DataLoader<string, Post>;
  };
  dataSources: DataSources;
}

export const createContext = async ({ req }): Promise<Context> => {
  const user = await authenticateRequest(req);

  return {
    user,
    loaders: {
      user: createUserLoader(db),
      post: createPostLoader(db),
    },
    dataSources: new DataSources(db),
  };
};

// Usage in resolver
const resolvers = {
  Post: {
    author: (parent, _, { loaders }) => {
      return loaders.user.load(parent.authorId);
    },
  },

  Comment: {
    author: (parent, _, { loaders }) => {
      return loaders.user.load(parent.authorId);
    },
  },
};
```

---

## 4. Authentication & Authorization (MANDATORY)

### A. Authentication

```typescript
// middleware/auth.ts
import { AuthenticationError } from 'apollo-server-core';
import jwt from 'jsonwebtoken';

export async function authenticateRequest(req: Request): Promise<AuthUser | null> {
  const authHeader = req.headers.authorization;

  if (!authHeader?.startsWith('Bearer ')) {
    return null;
  }

  const token = authHeader.substring(7);

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    return decoded as AuthUser;
  } catch {
    return null;
  }
}

// Directive-based authentication
import { mapSchema, getDirective, MapperKind } from '@graphql-tools/utils';

export function authDirectiveTransformer(schema: GraphQLSchema) {
  return mapSchema(schema, {
    [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
      const authDirective = getDirective(schema, fieldConfig, 'auth')?.[0];

      if (authDirective) {
        const { resolve = defaultFieldResolver } = fieldConfig;
        const requiredRole = authDirective.requires;

        fieldConfig.resolve = async (source, args, context, info) => {
          if (!context.user) {
            throw new AuthenticationError('Not authenticated');
          }

          if (requiredRole && context.user.role !== requiredRole) {
            throw new ForbiddenError('Insufficient permissions');
          }

          return resolve(source, args, context, info);
        };
      }

      return fieldConfig;
    },
  });
}
```

### B. Authorization Schema

```graphql
# Directive definition
directive @auth(requires: UserRole) on FIELD_DEFINITION

type Query {
  # Public
  post(id: ID!): Post

  # Requires authentication
  me: User @auth

  # Requires specific role
  users: [User!]! @auth(requires: ADMIN)
  adminDashboard: AdminDashboard! @auth(requires: ADMIN)
}

type Mutation {
  # Requires authentication
  createPost(input: CreatePostInput!): CreatePostPayload! @auth

  # Requires admin
  deleteUser(id: ID!): DeleteUserPayload! @auth(requires: ADMIN)
}
```

### C. Field-Level Authorization

```typescript
// resolvers/user.resolvers.ts
const resolvers = {
  User: {
    email: (parent, _, { user }) => {
      // Only show email to self or admin
      if (user?.id === parent.id || user?.role === 'ADMIN') {
        return parent.email;
      }
      return null;
    },

    privateNotes: (parent, _, { user }) => {
      // Only show to self
      if (user?.id !== parent.id) {
        throw new ForbiddenError('Not authorized to view private notes');
      }
      return parent.privateNotes;
    },
  },
};
```

---

## 5. Error Handling (MANDATORY)

### Protocol-Specific Design Note

**Why GraphQL error format differs from REST/gRPC:**

| Aspect | GraphQL | REST | gRPC |
|--------|---------|------|------|
| **Error format** | `errors[]` with `extensions.code` | JSON with `error`, `message` | Status codes with `errdetails` |
| **Pagination** | Relay connections (`first`, `after`) | URL params or cursor | Request message fields |
| **Naming** | camelCase (fields) | snake_case | snake_case (proto) |
| **Rate limiting** | Query complexity limits | HTTP headers | Interceptor-based |

These differences are **intentional and appropriate** for each protocol:
- GraphQL returns partial data + errors in a single response
- GraphQL uses camelCase per JavaScript/TypeScript conventions
- Query complexity limits prevent expensive nested queries

**Cross-API services** should use API gateway transformations to convert between formats.

---

### A. Error Types

```typescript
// errors/index.ts
import { GraphQLError } from 'graphql';

export class NotFoundError extends GraphQLError {
  constructor(resource: string, id: string) {
    super(`${resource} with id ${id} not found`, {
      extensions: {
        code: 'NOT_FOUND',
        resource,
        id,
      },
    });
  }
}

export class ValidationError extends GraphQLError {
  constructor(message: string, field?: string) {
    super(message, {
      extensions: {
        code: 'VALIDATION_ERROR',
        field,
      },
    });
  }
}

export class AuthenticationError extends GraphQLError {
  constructor(message = 'Not authenticated') {
    super(message, {
      extensions: { code: 'UNAUTHENTICATED' },
    });
  }
}

export class ForbiddenError extends GraphQLError {
  constructor(message = 'Not authorized') {
    super(message, {
      extensions: { code: 'FORBIDDEN' },
    });
  }
}
```

### B. Error Formatting

```typescript
// server.ts
import { ApolloServer } from '@apollo/server';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  formatError: (formattedError, error) => {
    // Log internal errors
    if (formattedError.extensions?.code === 'INTERNAL_SERVER_ERROR') {
      console.error('Internal error:', error);

      // Don't expose internal error details in production
      if (process.env.NODE_ENV === 'production') {
        return {
          message: 'An unexpected error occurred',
          extensions: { code: 'INTERNAL_SERVER_ERROR' },
        };
      }
    }

    return formattedError;
  },
});
```

---

## 6. Pagination (MANDATORY)

### A. Relay-Style Cursor Pagination

```graphql
# schema.graphql
type Query {
  posts(
    first: Int
    after: String
    last: Int
    before: String
    filter: PostFilterInput
  ): PostConnection!
}

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  cursor: String!
  node: Post!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

input PostFilterInput {
  authorId: ID
  status: PostStatus
  createdAfter: DateTime
  search: String
}
```

### B. Pagination Implementation

```typescript
// utils/pagination.ts
import { Prisma } from '@prisma/client';

interface PaginationArgs {
  first?: number;
  after?: string;
  last?: number;
  before?: string;
}

interface ConnectionResult<T> {
  edges: Array<{ cursor: string; node: T }>;
  pageInfo: {
    hasNextPage: boolean;
    hasPreviousPage: boolean;
    startCursor: string | null;
    endCursor: string | null;
  };
  totalCount: number;
}

export async function paginate<T extends { id: string }>(
  model: any,
  args: PaginationArgs,
  where: any = {}
): Promise<ConnectionResult<T>> {
  const { first, after, last, before } = args;

  // Validate args
  if (first && last) {
    throw new Error('Cannot use both first and last');
  }

  const take = first || last || 20;
  const cursor = after || before;

  // Get total count
  const totalCount = await model.count({ where });

  // Build query
  const queryArgs: any = {
    where,
    take: take + 1, // Fetch one extra to check hasNextPage
    orderBy: { createdAt: 'desc' },
  };

  if (cursor) {
    queryArgs.cursor = { id: decodeCursor(cursor) };
    queryArgs.skip = 1; // Skip the cursor item
  }

  if (last) {
    queryArgs.orderBy = { createdAt: 'asc' };
  }

  let items = await model.findMany(queryArgs);

  // Check for extra item (indicates more pages)
  const hasMore = items.length > take;
  if (hasMore) {
    items = items.slice(0, take);
  }

  // Reverse if using last
  if (last) {
    items = items.reverse();
  }

  const edges = items.map((item: T) => ({
    cursor: encodeCursor(item.id),
    node: item,
  }));

  return {
    edges,
    pageInfo: {
      hasNextPage: first ? hasMore : !!before,
      hasPreviousPage: first ? !!after : hasMore,
      startCursor: edges[0]?.cursor || null,
      endCursor: edges[edges.length - 1]?.cursor || null,
    },
    totalCount,
  };
}

function encodeCursor(id: string): string {
  return Buffer.from(`cursor:${id}`).toString('base64');
}

function decodeCursor(cursor: string): string {
  const decoded = Buffer.from(cursor, 'base64').toString('utf8');
  return decoded.replace('cursor:', '');
}
```

---

## 7. Performance Optimization (MANDATORY)

### A. Query Complexity Analysis

```typescript
// plugins/complexity.ts
import { ApolloServerPlugin } from '@apollo/server';
import {
  getComplexity,
  simpleEstimator,
  fieldExtensionsEstimator,
} from 'graphql-query-complexity';

const MAX_COMPLEXITY = 1000;
const MAX_DEPTH = 10;

export const complexityPlugin: ApolloServerPlugin = {
  async requestDidStart() {
    return {
      async didResolveOperation({ request, document, schema }) {
        const complexity = getComplexity({
          schema,
          operationName: request.operationName,
          query: document,
          variables: request.variables,
          estimators: [
            fieldExtensionsEstimator(),
            simpleEstimator({ defaultComplexity: 1 }),
          ],
        });

        if (complexity > MAX_COMPLEXITY) {
          throw new GraphQLError(
            `Query too complex: ${complexity}. Maximum allowed: ${MAX_COMPLEXITY}`
          );
        }
      },
    };
  },
};

// Schema with complexity hints
type Query {
  users(first: Int): UserConnection! @complexity(value: 10, multipliers: ["first"])
  user(id: ID!): User @complexity(value: 1)
}

type User {
  posts(first: Int): PostConnection! @complexity(value: 5, multipliers: ["first"])
}
```

### B. Depth Limiting

```typescript
// plugins/depth-limit.ts
import depthLimit from 'graphql-depth-limit';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [depthLimit(10)],
});
```

### C. Persisted Queries

```typescript
// Apollo Client setup
import { createPersistedQueryLink } from '@apollo/client/link/persisted-queries';
import { sha256 } from 'crypto-hash';

const link = createPersistedQueryLink({ sha256 });

// Server-side allowlist
const server = new ApolloServer({
  typeDefs,
  resolvers,
  persistedQueries: {
    cache: new KeyValueCache(),
  },
});
```

---

## 8. Subscriptions (Real-time)

### A. Subscription Schema

```graphql
type Subscription {
  # Subscribe to new messages in a channel
  messageCreated(channelId: ID!): Message!

  # Subscribe to user status changes
  userStatusChanged(userId: ID!): UserStatus!

  # Subscribe to order updates
  orderUpdated(orderId: ID!): Order!
}

type Message {
  id: ID!
  content: String!
  author: User!
  channel: Channel!
  createdAt: DateTime!
}
```

### B. Subscription Implementation

```typescript
// resolvers/subscription.resolvers.ts
import { PubSub, withFilter } from 'graphql-subscriptions';

const pubsub = new PubSub();

// Event names
const EVENTS = {
  MESSAGE_CREATED: 'MESSAGE_CREATED',
  USER_STATUS_CHANGED: 'USER_STATUS_CHANGED',
  ORDER_UPDATED: 'ORDER_UPDATED',
};

export const subscriptionResolvers = {
  Subscription: {
    messageCreated: {
      subscribe: withFilter(
        () => pubsub.asyncIterator([EVENTS.MESSAGE_CREATED]),
        (payload, variables, context) => {
          // Filter: only send to subscribers of this channel
          return payload.messageCreated.channelId === variables.channelId;
        }
      ),
    },

    orderUpdated: {
      subscribe: withFilter(
        () => pubsub.asyncIterator([EVENTS.ORDER_UPDATED]),
        (payload, variables, context) => {
          // Authorization: only order owner can subscribe
          const order = payload.orderUpdated;
          return order.id === variables.orderId &&
                 order.userId === context.user?.id;
        }
      ),
    },
  },

  Mutation: {
    createMessage: async (_, { input }, { dataSources, user }) => {
      const message = await dataSources.messages.create({
        ...input,
        authorId: user.id,
      });

      // Publish event
      pubsub.publish(EVENTS.MESSAGE_CREATED, { messageCreated: message });

      return { message, errors: [] };
    },
  },
};
```

---

## 9. Testing (MANDATORY)

### A. Schema Testing

```typescript
// __tests__/schema.test.ts
import { buildSchema, printSchema } from 'graphql';
import { readFileSync } from 'fs';

describe('GraphQL Schema', () => {
  const schemaString = readFileSync('./schema.graphql', 'utf8');

  it('should be valid', () => {
    expect(() => buildSchema(schemaString)).not.toThrow();
  });

  it('should have required types', () => {
    const schema = buildSchema(schemaString);
    expect(schema.getType('User')).toBeDefined();
    expect(schema.getType('Query')).toBeDefined();
    expect(schema.getType('Mutation')).toBeDefined();
  });
});
```

### B. Resolver Testing

```typescript
// __tests__/resolvers/user.test.ts
import { createTestClient } from 'apollo-server-testing';
import { ApolloServer, gql } from 'apollo-server';
import { createTestContext } from '../test-utils';

const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      email
      displayName
    }
  }
`;

describe('User Resolvers', () => {
  let server: ApolloServer;

  beforeAll(() => {
    server = new ApolloServer({
      typeDefs,
      resolvers,
      context: createTestContext,
    });
  });

  it('should return user by id', async () => {
    const { query } = createTestClient(server);

    const result = await query({
      query: GET_USER,
      variables: { id: 'user-1' },
    });

    expect(result.errors).toBeUndefined();
    expect(result.data.user).toEqual({
      id: 'user-1',
      email: 'test@example.com',
      displayName: 'Test User',
    });
  });

  it('should return null for non-existent user', async () => {
    const { query } = createTestClient(server);

    const result = await query({
      query: GET_USER,
      variables: { id: 'non-existent' },
    });

    expect(result.errors).toBeUndefined();
    expect(result.data.user).toBeNull();
  });
});
```

### C. Integration Testing

```typescript
// __tests__/integration/auth.test.ts
import request from 'supertest';
import { app } from '../app';

describe('Authentication Flow', () => {
  it('should login and access protected query', async () => {
    // Login
    const loginResponse = await request(app)
      .post('/graphql')
      .send({
        query: `
          mutation Login($input: LoginInput!) {
            login(input: $input) {
              token
              user { id email }
            }
          }
        `,
        variables: {
          input: { email: 'test@example.com', password: 'password123' }
        }
      });

    const { token } = loginResponse.body.data.login;

    // Access protected query
    const meResponse = await request(app)
      .post('/graphql')
      .set('Authorization', `Bearer ${token}`)
      .send({
        query: `query { me { id email } }`
      });

    expect(meResponse.body.data.me.email).toBe('test@example.com');
  });
});
```

---

## 10. Distributed Tracing (MANDATORY)

**CRITICAL: GraphQL APIs MUST propagate trace IDs for observability.**

### A. Trace ID in Context

```typescript
// context.ts - Include trace ID in GraphQL context
export interface Context {
  user: AuthUser | null;
  loaders: DataLoaders;
  dataSources: DataSources;
  traceId: string;  // MANDATORY: Trace ID for distributed tracing
  requestId: string;
}

export const createContext = async ({ req }): Promise<Context> => {
  // Extract or generate trace ID
  const traceId = req.headers['x-trace-id'] ||
                  req.headers['traceparent']?.split('-')[1] ||
                  generateTraceId();

  return {
    user: await authenticateRequest(req),
    loaders: createLoaders(db),
    dataSources: new DataSources(db),
    traceId,
    requestId: req.headers['x-request-id'] || generateRequestId(),
  };
};

// Include trace ID in response headers
const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [{
    async requestDidStart({ contextValue }) {
      return {
        async willSendResponse({ response }) {
          response.http?.headers.set('x-trace-id', contextValue.traceId);
        },
      };
    },
  }],
});
```

### B. Trace ID in Logging

```typescript
// All log entries MUST include trace ID
function logResolver(resolverName: string, context: Context, data: any) {
  logger.info({
    resolver: resolverName,
    traceId: context.traceId,  // MANDATORY
    requestId: context.requestId,
    userId: context.user?.id,
    ...data,
  });
}
```

**Cross-reference:** See logging.md Section 5 for trace ID implementation patterns.

---

## 11. Code Generation

### A. GraphQL Code Generator

```yaml
# codegen.yml
schema: "./schema/**/*.graphql"
documents: "./src/**/*.graphql"
generates:
  ./src/generated/graphql.ts:
    plugins:
      - typescript
      - typescript-resolvers
      - typescript-operations
    config:
      contextType: ../context#Context
      mappers:
        User: ../models#UserModel
        Post: ../models#PostModel
      scalars:
        DateTime: Date
        Email: string
        URL: string
```

### B. Generated Types Usage

```typescript
// resolvers/user.resolvers.ts
import { Resolvers, UserResolvers } from '../generated/graphql';

// Fully typed resolvers
export const userResolvers: Resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      // id is typed as string
      // return type must match User
      return context.dataSources.users.findById(id);
    },
  },

  User: {
    // Field resolver with proper typing
    posts: async (parent, args, context) => {
      // parent is UserModel
      // args is { first?: number, after?: string }
      return context.dataSources.posts.findByUserId(parent.id, args);
    },
  },
};
```

---

## 11. Security Best Practices

### A. Query Allowlisting (Production)

```typescript
// Automatic Persisted Queries with allowlist
const allowedQueries = new Map<string, DocumentNode>([
  ['GetUser', gql`query GetUser($id: ID!) { user(id: $id) { id email } }`],
  ['ListPosts', gql`query ListPosts($first: Int) { posts(first: $first) { edges { node { id title } } } }`],
]);

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    {
      async requestDidStart() {
        return {
          async didResolveOperation({ request }) {
            if (process.env.NODE_ENV === 'production') {
              const hash = request.extensions?.persistedQuery?.sha256Hash;
              if (!hash || !allowedQueries.has(hash)) {
                throw new ForbiddenError('Query not allowed');
              }
            }
          },
        };
      },
    },
  ],
});
```

### B. Input Validation

```typescript
// validation/user.validation.ts
import { z } from 'zod';

// Secure ID validation - minimum 32 characters, alphanumeric with - and _
// Matches REST API and gRPC ID requirements for consistency
export const secureIdSchema = z.string()
  .min(32, 'ID must be at least 32 characters')
  .max(64, 'ID must be at most 64 characters')
  .regex(/^[a-zA-Z0-9_-]+$/, 'ID must be alphanumeric with underscores and hyphens');

export const createUserSchema = z.object({
  email: z.string().email(),
  displayName: z.string().min(2).max(100),
  password: z.string().min(8).regex(/[A-Z]/).regex(/[0-9]/),
});

// Validate IDs in resolvers
export const getUserSchema = z.object({
  id: secureIdSchema,
});

// In resolver
const resolvers = {
  Mutation: {
    createUser: async (_, { input }, context) => {
      // Validate input
      const validation = createUserSchema.safeParse(input);

      if (!validation.success) {
        return {
          user: null,
          errors: validation.error.errors.map(e => ({
            field: e.path.join('.'),
            message: e.message,
            code: 'VALIDATION_ERROR',
          })),
        };
      }

      return context.dataSources.users.create(validation.data);
    },
  },
};
```

---

## 12. Deployment Checklist

### Schema Design
- [ ] Clear, descriptive type names
- [ ] Consistent naming conventions
- [ ] Proper nullability (! where appropriate)
- [ ] Relay-style pagination for lists
- [ ] Payload types for mutations

### Security
- [ ] Authentication implemented
- [ ] Field-level authorization
- [ ] Query complexity limits
- [ ] Depth limiting
- [ ] Input validation

### Performance
- [ ] DataLoader for N+1 prevention
- [ ] Pagination on all list queries
- [ ] Persisted queries in production
- [ ] Caching strategy implemented

### Testing
- [ ] Schema validation tests
- [ ] Resolver unit tests
- [ ] Integration tests
- [ ] E2E tests for critical flows

---

## 13. Quick Reference

```graphql
# Common patterns

# Pagination
query {
  posts(first: 10, after: "cursor") {
    edges {
      cursor
      node { id title }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}

# Mutation with payload
mutation {
  createPost(input: { title: "Hello", content: "World" }) {
    post { id title }
    errors { field message code }
  }
}

# Subscription
subscription {
  messageCreated(channelId: "123") {
    id
    content
    author { displayName }
  }
}
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** API Team


**End of GraphQL Development Guidelines**

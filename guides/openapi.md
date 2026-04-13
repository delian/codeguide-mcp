# OpenAPI Specification Guidelines
Mandatory standards for designing and documenting REST APIs using OpenAPI (formerly Swagger). OpenAPI 3.1, Swagger UI, Redoc, Stoplight, Spectral.

---

**Agent Profile**: The OpenAPI Expert
**Role**: Senior API Designer & Documentation Specialist
**Objective**: Generate comprehensive, accurate, and developer-friendly API specifications.
**Tools**: OpenAPI 3.1, Swagger UI, Redoc, Stoplight, Spectral.

---

## 1. Core Philosophies: OPENAPI-FIRST

- **O**rganized: Logical structure and consistent naming
- **P**recise: Accurate schemas with validation rules
- **E**xamples: Rich examples for every endpoint
- **N**avigable: Easy to browse and understand
- **A**utomated: Enable code generation and testing
- **P**roduction-ready: Versioned and maintained
- **I**nteractive: Try-it-out functionality

---

## 2. Document Structure (MANDATORY)

### A. Basic Structure

```yaml
# openapi.yaml
openapi: 3.1.0

info:
  title: My API
  version: 1.0.0
  description: |
    A comprehensive API for managing resources.

    ## Authentication
    All endpoints require a Bearer token in the Authorization header.

    ## Rate Limiting
    - 1000 requests per minute for authenticated users
    - 100 requests per minute for unauthenticated users

    ## Pagination
    List endpoints support cursor-based pagination using `cursor` and `limit` parameters.
  termsOfService: https://example.com/terms
  contact:
    name: API Support
    email: api-support@example.com
    url: https://example.com/support
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server
  - url: http://localhost:3000/v1
    description: Development server

tags:
  - name: Users
    description: User management operations
  - name: Orders
    description: Order processing operations
  - name: Products
    description: Product catalog operations

paths:
  # Path definitions..

components:
  # Reusable components..

security:
  - BearerAuth: []
```

### B. File Organization

```
api/
├── openapi.yaml              # Main entry point
├── paths/
│   ├── users.yaml            # /users endpoints
│   ├── orders.yaml           # /orders endpoints
│   └── products.yaml         # /products endpoints
├── components/
│   ├── schemas/
│   │   ├── user.yaml
│   │   ├── order.yaml
│   │   └── common.yaml
│   ├── parameters/
│   │   └── common.yaml
│   ├── responses/
│   │   └── errors.yaml
│   └── securitySchemes.yaml
└── examples/
    ├── users.yaml
    └── orders.yaml
```

```yaml
# Main openapi.yaml with $ref
openapi: 3.1.0
info:
  title: My API
  version: 1.0.0

paths:
  /users:
    $ref: './paths/users.yaml#/users'
  /users/{id}:
    $ref: './paths/users.yaml#/users~1{id}'
  /orders:
    $ref: './paths/orders.yaml#/orders'

components:
  schemas:
    User:
      $ref: './components/schemas/user.yaml'
    Order:
      $ref: './components/schemas/order.yaml'
```

---

## 2A. TDD Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL API specification changes.**

### Red-Green-Refactor Cycle with Spectral Linting

```yaml
# ═══════════════════════════════════════════════════════════════
# STEP 1: RED - Write failing Spectral rules first
# ═══════════════════════════════════════════════════════════════

# .spectral.yml - Custom linting rules
extends: ["spectral:oas", "spectral:asyncapi"]

rules:
  operation-operationId:
    severity: error
    description: Every operation must have an operationId
  operation-description:
    severity: error
    description: Every operation must have a description
  path-params:
    severity: error
  oas3-valid-media-example:
    severity: error
  require-pagination:
    severity: warn
    description: List endpoints must support pagination
    given: "$.paths[*].get"
    then:
      field: parameters
      function: schema
      functionOptions:
        schema:
          type: array
          contains:
            type: object
            properties:
              name:
                enum: ["limit", "cursor", "page"]

# Run: npx @stoplight/spectral-cli lint openapi.yaml
# ❌ FAILS - spec is missing operationIds, descriptions, pagination
```

```bash
# test/validate-spec.sh
#!/bin/bash
set -euo pipefail

# Lint with Spectral
npx @stoplight/spectral-cli lint openapi.yaml --fail-severity=error || {
  echo "FAIL: Spectral linting errors found"
  exit 1
}

# Validate schema is valid OpenAPI 3.1
npx swagger-cli validate openapi.yaml || {
  echo "FAIL: Invalid OpenAPI document"
  exit 1
}

echo "PASS: OpenAPI specification is valid"

# ═══════════════════════════════════════════════════════════════
# STEP 2: GREEN - Fix spec to pass all Spectral rules
# ═══════════════════════════════════════════════════════════════

# Add operationIds, descriptions, and pagination parameters

# Run: bash test/validate-spec.sh
# ✅ PASSES - all rules satisfied

# ═══════════════════════════════════════════════════════════════
# STEP 3: REFACTOR - Improve examples, add more schemas, keep valid
# ═══════════════════════════════════════════════════════════════
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every spec bug MUST receive a validation test BEFORE fixing.**

### Bug Fix Workflow Example

```bash
# ═══════════════════════════════════════════════════════════════
# Bug Report #108: Breaking API change shipped because response
# schema was modified without version bump
# ═══════════════════════════════════════════════════════════════

# STEP 1: Write test that detects breaking changes
# test/detect-breaking-changes.sh

#!/bin/bash
set -euo pipefail

# Compare current spec against the last released version
npx openapi-diff \
  https://api.example.com/v1/openapi.yaml \
  openapi.yaml \
  --fail-on-incompatible || {
  echo "FAIL Bug #108: Breaking changes detected without version bump"
  echo "Either bump the API version or make the change backward-compatible"
  exit 1
}

echo "PASS: No breaking changes found"

# Run: bash test/detect-breaking-changes.sh
# ❌ FAILS - breaking change detected in response schema

# STEP 2: Fix the bug - Revert schema change or bump API version

# Run: bash test/detect-breaking-changes.sh
# ✅ PASSES - spec is backward-compatible or properly versioned
```

---

## 3. Path Definitions (MANDATORY)

### A. Resource Endpoints

```yaml
paths:
  /users:
    get:
      operationId: listUsers
      summary: List all users
      description: |
        Returns a paginated list of users.
        Results are sorted by creation date in descending order.
      tags:
        - Users
      parameters:
        - $ref: '#/components/parameters/LimitParam'
        - $ref: '#/components/parameters/CursorParam'
        - name: status
          in: query
          description: Filter by user status
          schema:
            type: string
            enum: [active, inactive, pending]
        - name: role
          in: query
          description: Filter by user role
          schema:
            type: string
            enum: [admin, user, guest]
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserList'
              examples:
                default:
                  $ref: '#/components/examples/UserListExample'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '500':
          $ref: '#/components/responses/InternalError'

    post:
      operationId: createUser
      summary: Create a new user
      description: Creates a new user account and sends a welcome email.
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            examples:
              basic:
                summary: Basic user creation
                value:
                  email: user@example.com
                  name: John Doe
              withRole:
                summary: User with admin role
                value:
                  email: admin@example.com
                  name: Admin User
                  role: admin
      responses:
        '201':
          description: User created successfully
          headers:
            Location:
              description: URL of the created user
              schema:
                type: string
                format: uri
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'
        '422':
          $ref: '#/components/responses/ValidationError'

  /users/{id}:
    parameters:
      - $ref: '#/components/parameters/UserIdParam'

    get:
      operationId: getUser
      summary: Get a user by ID
      tags:
        - Users
      responses:
        '200':
          description: User details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          $ref: '#/components/responses/NotFound'

    patch:
      operationId: updateUser
      summary: Update a user
      description: |
        Partially updates a user. Only provided fields will be updated.
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateUserRequest'
      responses:
        '200':
          description: User updated
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          $ref: '#/components/responses/BadRequest'
        '404':
          $ref: '#/components/responses/NotFound'

    delete:
      operationId: deleteUser
      summary: Delete a user
      description: |
        Soft-deletes a user. The user data will be retained for 30 days
        before permanent deletion.
      tags:
        - Users
      responses:
        '204':
          description: User deleted successfully
        '404':
          $ref: '#/components/responses/NotFound'
```

### B. Nested Resources

```yaml
paths:
  /users/{userId}/orders:
    parameters:
      - name: userId
        in: path
        required: true
        schema:
          type: string
          format: uuid

    get:
      operationId: listUserOrders
      summary: List orders for a user
      tags:
        - Orders
      parameters:
        - $ref: '#/components/parameters/LimitParam'
        - name: status
          in: query
          schema:
            type: string
            enum: [pending, processing, shipped, delivered, cancelled]
      responses:
        '200':
          description: List of orders
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/OrderList'

  /users/{userId}/orders/{orderId}:
    parameters:
      - name: userId
        in: path
        required: true
        schema:
          type: string
          format: uuid
      - name: orderId
        in: path
        required: true
        schema:
          type: string
          format: uuid

    get:
      operationId: getUserOrder
      summary: Get a specific order for a user
      tags:
        - Orders
      responses:
        '200':
          description: Order details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Order'
        '404':
          $ref: '#/components/responses/NotFound'
```

---

## 4. Schema Definitions (MANDATORY)

### A. Object Schemas

```yaml
components:
  schemas:
    User:
      type: object
      description: Represents a user account
      required:
        - id
        - email
        - name
        - createdAt
      properties:
        id:
          type: string
          format: uuid
          description: Unique identifier
          readOnly: true
          example: '123e4567-e89b-12d3-a456-426614174000'
        email:
          type: string
          format: email
          description: User's email address
          maxLength: 255
          example: user@example.com
        name:
          type: string
          description: User's display name
          minLength: 1
          maxLength: 100
          example: John Doe
        role:
          type: string
          description: User's role in the system
          enum: [admin, user, guest]
          default: user
          example: user
        status:
          type: string
          description: Account status
          enum: [active, inactive, pending]
          default: pending
          example: active
        avatarUrl:
          type: string
          format: uri
          description: URL to user's avatar image
          nullable: true
          example: 'https://example.com/avatars/123.jpg'
        metadata:
          type: object
          description: Additional user metadata
          additionalProperties: true
          example:
            theme: dark
            language: en
        createdAt:
          type: string
          format: date-time
          description: When the user was created
          readOnly: true
          example: '2024-01-15T10:30:00Z'
        updatedAt:
          type: string
          format: date-time
          description: When the user was last updated
          readOnly: true
          example: '2024-01-15T10:30:00Z'

    CreateUserRequest:
      type: object
      description: Request body for creating a user
      required:
        - email
        - name
      properties:
        email:
          type: string
          format: email
          maxLength: 255
        name:
          type: string
          minLength: 1
          maxLength: 100
        role:
          type: string
          enum: [admin, user, guest]
          default: user
        password:
          type: string
          format: password
          minLength: 8
          maxLength: 128
          writeOnly: true

    UpdateUserRequest:
      type: object
      description: Request body for updating a user
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        role:
          type: string
          enum: [admin, user, guest]
        status:
          type: string
          enum: [active, inactive, pending]
        avatarUrl:
          type: string
          format: uri
          nullable: true
      minProperties: 1
```

### B. Collection Schemas

```yaml
components:
  schemas:
    UserList:
      type: object
      required:
        - data
        - pagination
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/User'
        pagination:
          $ref: '#/components/schemas/Pagination'

    Pagination:
      type: object
      required:
        - total
        - limit
        - hasMore
      properties:
        total:
          type: integer
          description: Total number of items
          minimum: 0
          example: 150
        limit:
          type: integer
          description: Maximum items per page
          minimum: 1
          maximum: 100
          example: 20
        cursor:
          type: string
          description: Cursor for the current page
          nullable: true
          example: 'eyJpZCI6MTIzfQ'
        nextCursor:
          type: string
          description: Cursor for the next page
          nullable: true
          example: 'eyJpZCI6MTQzfQ'
        hasMore:
          type: boolean
          description: Whether more items exist
          example: true
```

### C. Polymorphic Schemas

```yaml
components:
  schemas:
    Notification:
      oneOf:
        - $ref: '#/components/schemas/EmailNotification'
        - $ref: '#/components/schemas/SMSNotification'
        - $ref: '#/components/schemas/PushNotification'
      discriminator:
        propertyName: type
        mapping:
          email: '#/components/schemas/EmailNotification'
          sms: '#/components/schemas/SMSNotification'
          push: '#/components/schemas/PushNotification'

    EmailNotification:
      type: object
      required:
        - type
        - to
        - subject
        - body
      properties:
        type:
          type: string
          enum: [email]
        to:
          type: string
          format: email
        subject:
          type: string
        body:
          type: string

    SMSNotification:
      type: object
      required:
        - type
        - phoneNumber
        - message
      properties:
        type:
          type: string
          enum: [sms]
        phoneNumber:
          type: string
          pattern: '^\+[1-9]\d{1,14}$'
        message:
          type: string
          maxLength: 160

    PushNotification:
      type: object
      required:
        - type
        - deviceToken
        - title
        - body
      properties:
        type:
          type: string
          enum: [push]
        deviceToken:
          type: string
        title:
          type: string
        body:
          type: string
```

---

## 5. Parameters (MANDATORY)

### A. Reusable Parameters

```yaml
components:
  parameters:
    # Path parameters
    UserIdParam:
      name: id
      in: path
      required: true
      description: User's unique identifier
      schema:
        type: string
        format: uuid

    # Query parameters
    LimitParam:
      name: limit
      in: query
      description: Maximum number of items to return
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 20

    CursorParam:
      name: cursor
      in: query
      description: Pagination cursor from previous response
      schema:
        type: string

    SortParam:
      name: sort
      in: query
      description: |
        Sort order. Prefix with `-` for descending.
        Example: `-createdAt` for newest first.
      schema:
        type: string
      examples:
        ascending:
          value: createdAt
          summary: Oldest first
        descending:
          value: -createdAt
          summary: Newest first

    FieldsParam:
      name: fields
      in: query
      description: Comma-separated list of fields to include
      schema:
        type: string
      example: id,name,email

    # Header parameters
    IdempotencyKeyHeader:
      name: Idempotency-Key
      in: header
      description: Unique key for idempotent requests
      schema:
        type: string
        format: uuid
```

### B. Filter Parameters

```yaml
components:
  parameters:
    DateRangeFilter:
      name: createdAt
      in: query
      description: |
        Filter by creation date. Supports operators:
        - `gte:2024-01-01` - Greater than or equal
        - `lte:2024-12-31` - Less than or equal
        - `2024-01-01..2024-12-31` - Range
      schema:
        type: string
      examples:
        after:
          value: 'gte:2024-01-01'
          summary: Created on or after date
        before:
          value: 'lte:2024-12-31'
          summary: Created on or before date
        range:
          value: '2024-01-01..2024-12-31'
          summary: Created within date range
```

---

## 6. Responses (MANDATORY)

### A. Success Responses

```yaml
components:
  responses:
    Created:
      description: Resource created successfully
      headers:
        Location:
          description: URL of the created resource
          schema:
            type: string
            format: uri

    NoContent:
      description: Request successful, no content returned

    Accepted:
      description: Request accepted for processing
      content:
        application/json:
          schema:
            type: object
            properties:
              jobId:
                type: string
                description: ID to track the async operation
              status:
                type: string
                enum: [pending, processing]
              statusUrl:
                type: string
                format: uri
                description: URL to check operation status
```

### B. Error Responses

```yaml
components:
  schemas:
    Error:
      type: object
      required:
        - code
        - message
      properties:
        code:
          type: string
          description: Machine-readable error code
          example: VALIDATION_ERROR
        message:
          type: string
          description: Human-readable error message
          example: Invalid request parameters
        details:
          type: array
          description: Additional error details
          items:
            type: object
            properties:
              field:
                type: string
                example: email
              message:
                type: string
                example: Must be a valid email address
              code:
                type: string
                example: INVALID_FORMAT
        requestId:
          type: string
          description: Request ID for support reference
          example: req_abc123

  responses:
    BadRequest:
      description: Invalid request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: BAD_REQUEST
            message: Invalid request format
            requestId: req_abc123

    Unauthorized:
      description: Authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: UNAUTHORIZED
            message: Authentication required
            requestId: req_abc123
      headers:
        WWW-Authenticate:
          schema:
            type: string
          example: Bearer realm="api"

    Forbidden:
      description: Insufficient permissions
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: FORBIDDEN
            message: You do not have permission to perform this action
            requestId: req_abc123

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: NOT_FOUND
            message: The requested resource was not found
            requestId: req_abc123

    Conflict:
      description: Resource conflict
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: CONFLICT
            message: A user with this email already exists
            requestId: req_abc123

    ValidationError:
      description: Validation failed
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: VALIDATION_ERROR
            message: Validation failed
            details:
              - field: email
                message: Must be a valid email address
                code: INVALID_FORMAT
              - field: name
                message: Required field
                code: REQUIRED
            requestId: req_abc123

    TooManyRequests:
      description: Rate limit exceeded
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: RATE_LIMITED
            message: Too many requests. Please try again later.
            requestId: req_abc123
      headers:
        X-RateLimit-Limit:
          schema:
            type: integer
          description: Request limit per window
        X-RateLimit-Remaining:
          schema:
            type: integer
          description: Remaining requests in window
        X-RateLimit-Reset:
          schema:
            type: integer
          description: Unix timestamp when the limit resets
        Retry-After:
          schema:
            type: integer
          description: Seconds to wait before retrying

    InternalError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            code: INTERNAL_ERROR
            message: An unexpected error occurred
            requestId: req_abc123
```

---

## 7. Security Schemes (MANDATORY)

```yaml
components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWT token authentication. Include the token in the Authorization header:
        ```
        Authorization: Bearer <token>
        ```

    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for server-to-server communication

    OAuth2:
      type: oauth2
      description: OAuth 2.0 authentication
      flows:
        authorizationCode:
          authorizationUrl: https://auth.example.com/oauth/authorize
          tokenUrl: https://auth.example.com/oauth/token
          refreshUrl: https://auth.example.com/oauth/refresh
          scopes:
            read:users: Read user information
            write:users: Create and modify users
            read:orders: Read order information
            write:orders: Create and modify orders
        clientCredentials:
          tokenUrl: https://auth.example.com/oauth/token
          scopes:
            admin: Full administrative access

# Apply security globally
security:
  - BearerAuth: []

# Override for specific endpoints
paths:
  /public/status:
    get:
      security: []  # No auth required
      # ..

  /admin/users:
    get:
      security:
        - OAuth2: [read:users, admin]
      # ..
```

---

## 8. Webhooks (OpenAPI 3.1)

```yaml
webhooks:
  orderCreated:
    post:
      summary: Order created webhook
      description: Triggered when a new order is created
      operationId: orderCreatedWebhook
      tags:
        - Webhooks
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WebhookPayload'
            example:
              id: evt_123
              type: order.created
              timestamp: '2024-01-15T10:30:00Z'
              data:
                orderId: ord_456
                customerId: cust_789
                total: 99.99
      responses:
        '200':
          description: Webhook processed successfully
        '400':
          description: Invalid payload

components:
  schemas:
    WebhookPayload:
      type: object
      required:
        - id
        - type
        - timestamp
        - data
      properties:
        id:
          type: string
          description: Unique event identifier
        type:
          type: string
          description: Event type
          enum:
            - order.created
            - order.updated
            - order.cancelled
            - user.created
            - user.updated
        timestamp:
          type: string
          format: date-time
        data:
          type: object
          description: Event-specific data
```

---

## 9. Validation with Spectral

```yaml
# .spectral.yaml
extends: spectral:oas
rules:
  # Naming conventions
  operation-operationId-camelCase:
    severity: error
    given: "$.paths[*][*]"
    then:
      field: operationId
      function: casing
      functionOptions:
        type: camel

  # Require descriptions
  operation-description:
    severity: warn
    given: "$.paths[*][*]"
    then:
      field: description
      function: truthy

  # Require examples
  schema-examples:
    severity: warn
    given: "$.components.schemas[*]"
    then:
      - field: example
        function: truthy

  # Error response format
  error-response-format:
    severity: error
    given: "$.paths[*][*].responses[?(@property >= 400)]"
    then:
      field: content.application/json.schema.$ref
      function: pattern
      functionOptions:
        match: "#/components/schemas/Error"
```

---

## 10. Deployment Checklist

### Documentation Quality
- [ ] All endpoints have descriptions
- [ ] All parameters documented
- [ ] Examples for all schemas
- [ ] Error responses documented

### Schema Completeness
- [ ] Required fields marked
- [ ] Validation rules defined
- [ ] Formats specified (email, uri, etc.)
- [ ] Enums documented

### Security
- [ ] Authentication schemes defined
- [ ] Security applied to endpoints
- [ ] Sensitive fields marked writeOnly

### Versioning
- [ ] Version in info.version
- [ ] Breaking changes documented
- [ ] Deprecation notices added

---

## 11. Quick Reference

```yaml
# Common types
type: string
type: integer
type: number
type: boolean
type: array
type: object

# String formats
format: date          # 2024-01-15
format: date-time     # 2024-01-15T10:30:00Z
format: email
format: uri
format: uuid
format: password

# Validation
minLength: 1
maxLength: 255
minimum: 0
maximum: 100
pattern: '^[a-z]+$'
enum: [a, b, c]

# Object validation
required: [field1, field2]
additionalProperties: false
minProperties: 1

# Array validation
minItems: 1
maxItems: 100
uniqueItems: true

# Modifiers
readOnly: true
writeOnly: true
nullable: true
deprecated: true
```

---

## 12. Why This Configuration Works

1. **Design-First Approach**: Writing the OpenAPI spec before implementation aligns frontend and backend teams on the contract, catching design issues before any code is written.

2. **Reusable Component Schemas**: Defining schemas in `components/schemas` and referencing them with `$ref` eliminates duplication and ensures consistent data structures across endpoints.

3. **Rich Examples on Every Endpoint**: Concrete request/response examples enable instant "try it out" testing in Swagger UI and serve as living documentation that never goes stale.

4. **Spectral Linting Rules**: Automated style enforcement catches inconsistent naming, missing descriptions, and schema violations in CI before specs are merged.

5. **Semantic Versioning with URL Paths**: Including the major version in the URL (`/v1/`, `/v2/`) allows breaking changes without disrupting existing consumers.

6. **Standardized Error Responses**: Consistent error schemas with `type`, `title`, `status`, and `detail` fields (RFC 7807) enable clients to build generic error handling.

7. **Security Scheme Declarations**: Explicit security definitions document authentication requirements and enable code generators to produce clients with built-in auth support.

8. **Pagination with Cursor-Based Patterns**: Cursor pagination in the spec prevents offset-based performance degradation and provides stable page boundaries during concurrent writes.

9. **Code Generation from Spec**: Generating server stubs and client SDKs from the OpenAPI spec guarantees implementation matches the contract and reduces hand-written boilerplate.

10. **Webhook Definitions (OpenAPI 3.1)**: Documenting webhooks alongside REST endpoints gives consumers a complete integration picture in a single specification file.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** API Team


**End of OpenAPI Specification Guidelines**

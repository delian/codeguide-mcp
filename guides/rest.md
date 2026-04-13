# Modern REST API Development Guidelines
Mandatory coding standards and development practices for modern REST API design with emphasis on OpenAPI specifications, security, validation, documentation, and hexagonal architecture principles. This guide is language-agnostic and focuses on API design rather than implementation. OpenAPI 3.1+, JSON Schema, JWT, OAuth 2.0, API Gateway patterns, API documentation tools.

---

**Agent Profile**: The REST API Architect  
**Role**: Senior API Design Engineer & Integration Specialist  
**Objective**: Generate production-ready, well-documented, secure, and maintainable REST API designs using OpenAPI specifications with hexagonal architecture, comprehensive validation, and best practices.  
**Tools**: OpenAPI 3.1+, JSON Schema, JWT, OAuth 2.0, API Gateway patterns, API documentation tools.

---

## 1. Core Philosophies: REST-FIRST

The agent must adhere to the **REST-FIRST** principles for every REST API design:

- **M**inimalistic Design: Clean, concise, well-structured API endpoints
- **O**penAPI First: Always use OpenAPI for API specification
- **D**ocumentation as Code: API documentation auto-generatable from OpenAPI
- **E**rror Handling: Explicit error responses, proper HTTP status codes
- **R**esource-Oriented: RESTful resource design, proper HTTP methods
- **N**ative Standards: Follow HTTP/HTTPS standards, REST principles

- **R**obust Validation: JSON Schema validation, input sanitization
- **E**xplicit Security: Authentication, authorization, secure defaults
- **S**tandard Patterns: Follow REST conventions, consistent naming
- **T**esting First: API test examples, contract testing
- **A**rchitectural: Hexagonal architecture, clear separation
- **P**erformance: Efficient endpoints, proper caching headers
- **I**dempotent Operations: Safe to retry, proper HTTP methods

**V**erified APIs: Agent-generated API designs MUST be valid OpenAPI, documented, and testable
- **E**xplicit Versioning: API versioning strategy, backward compatibility
- **R**obust Validation: JSON Schema, request/response validation
- **I**mmutable IDs: Unpredictable resource IDs, no sensitive data in URLs
- **F**unctional Design: Resource-oriented, stateless operations
- **I**dempotent Operations: Safe to retry, proper error handling
- **E**fficient Execution: Performance-optimized, proper HTTP caching

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. OpenAPI Specification Verification Protocol

**CRITICAL: Agents MUST ALWAYS verify that all generated/modified REST API designs are valid OpenAPI specifications, properly documented, and include test examples. Verification is MANDATORY for every API change.**

#### Verification Checklist

**Before delivering ANY REST API design, the agent MUST:**

1. **OpenAPI Validation (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: API specification MUST be valid OpenAPI 3.1+. This is non-negotiable.**
   ```bash
   # Validate OpenAPI specification
   npx @apidevtools/swagger-cli validate api/openapi.yaml
   # Exit code MUST be 0
   
   # OR using spectral
   npx @stoplight/spectral-cli lint api/openapi.yaml
   # Exit code MUST be 0
   
   # Check for required components
   # - Must have info, paths, components
   # - Must have security schemes
   # - Must have schemas for all request/response bodies
   ```
   - **MUST** be valid OpenAPI 3.1+ specification
   - All paths properly defined
   - All schemas valid JSON Schema
   - No validation errors

2. **JSON Schema Validation (MANDATORY - ALWAYS REQUIRED)**:
   **CRITICAL: All request/response bodies MUST have JSON Schema definitions.**
   ```bash
   # Validate JSON Schema
   npx ajv-cli validate -s schema.json -d data.json
   # Exit code MUST be 0
   ```
   - **MUST** have JSON Schema for all request/response bodies
   - Schemas must include validation rules
   - Optional fields and defaults properly defined

3. **Documentation Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # Generate API documentation
   npx redoc-cli bundle api/openapi.yaml -o docs/api.html
   # Exit code MUST be 0
   
   # Verify documentation includes:
   # - All endpoints documented
   # - Request/response examples
   # - Authentication information
   # - Error responses
   ```
   - **MUST** generate documentation without errors
   - All endpoints have descriptions
   - All request/response bodies have examples
   - Authentication flows documented

4. **Test Examples Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # Verify test examples exist
   # Check that each endpoint has example requests/responses
   # Verify examples match schemas
   ```
   - **MUST** have test examples for all endpoints
   - Examples must match JSON Schema
   - Examples must be realistic and usable

5. **Security Verification (MANDATORY - ALWAYS REQUIRED)**:
   ```bash
   # Verify security requirements
   # - Authentication schemes defined
   # - Authorization requirements specified
   # - No sensitive data in URL paths
   # - Proper HTTP methods for operations
   ```
   - **MUST** have security schemes defined
   - All protected endpoints have security requirements
   - No passwords or sensitive data in URLs
   - Proper authentication/authorization

6. **Post-Modification Verification (MANDATORY)**:
   ```bash
   # After ANY modification, ALWAYS run:
   # 1. Validate OpenAPI
   npx @apidevtools/swagger-cli validate api/openapi.yaml
   # Exit code MUST be 0
   
   # 2. Validate JSON Schemas
   # Check all schemas are valid
   
   # 3. Generate documentation
   npx redoc-cli bundle api/openapi.yaml
   # Exit code MUST be 0
   
   # 4. Verify examples
   # Check all examples match schemas
   ```

### B. Error Correction Process

If verification fails:

1. **Read the error message** - OpenAPI validation errors, schema errors, missing documentation
2. **Identify the root cause** - invalid schema, missing required field, security issue
3. **Fix the issue** in the generated API specification
4. **Re-verify** by running checks again
5. **Repeat until successful** - iterate as many times as needed
6. **Only present working, documented APIs** to the user

**CRITICAL**: Never provide REST API designs that don't validate or are missing documentation. Always verify first, fix issues, then present the working solution.

**MANDATORY RULES:**
1. **OpenAPI validation is ALWAYS required** - API MUST be valid OpenAPI 3.1+
2. **JSON Schema is ALWAYS required** - All request/response bodies MUST have schemas
3. **Documentation is ALWAYS required** - All endpoints MUST be documented with examples
4. **Security is ALWAYS required** - Authentication/authorization MUST be properly defined

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new REST API endpoints and features.**

### TDD Cycle for REST APIs

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD Cycle for REST APIs                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. RED: Write a failing API test first                        │
│      - Define expected request/response                          │
│      - Write contract test for endpoint                          │
│      - Test MUST fail (endpoint doesn't exist yet)              │
│                          ↓                                       │
│   2. GREEN: Implement minimal endpoint to pass test             │
│      - Create OpenAPI specification                              │
│      - Implement handler with minimal logic                      │
│      - Validate against JSON Schema                              │
│                          ↓                                       │
│   3. REFACTOR: Improve while keeping tests green                │
│      - Add proper error handling                                 │
│      - Optimize response structure                               │
│      - Enhance documentation                                     │
│                          ↓                                       │
│                      Repeat                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for REST API Endpoint

#### Step 1: RED - Write Failing Contract Test First

```yaml
# tests/contracts/users_api_test.yaml
# Contract test for POST /v1/users endpoint

test_create_user:
  description: "Create a new user via REST API"
  request:
    method: POST
    path: /v1/users
    headers:
      Content-Type: application/json
      Authorization: Bearer ${TEST_TOKEN}
    body:
      email: "newuser@example.com"
      name: "John Doe"
      role: "user"
  expected_response:
    status: 201
    headers:
      Content-Type: application/json
    body:
      id: "${json-schema: string, minLength: 32}"
      email: "newuser@example.com"
      name: "John Doe"
      role: "user"
      created_at: "${json-schema: string, format: date-time}"
      updated_at: "${json-schema: string, format: date-time}"

test_create_user_validation_error:
  description: "Reject invalid email format"
  request:
    method: POST
    path: /v1/users
    headers:
      Content-Type: application/json
      Authorization: Bearer ${TEST_TOKEN}
    body:
      email: "invalid-email"
      name: "John Doe"
  expected_response:
    status: 422
    body:
      error: "VALIDATION_ERROR"
      message: "${json-schema: string}"
      details:
        - field: "email"
          message: "${json-schema: string}"
```

```bash
# Run contract test - MUST FAIL (endpoint doesn't exist)
npx dredd api/openapi.yaml http://localhost:3000 --dry-run
# Expected: FAIL - endpoint not implemented

# OR using Postman/Newman
newman run tests/contracts/users_api_test.json
# Expected: FAIL - 404 Not Found
```

#### Step 2: GREEN - Write Minimal OpenAPI Specification

```yaml
# api/openapi.yaml - Minimal implementation to pass test
paths:
  /v1/users:
    post:
      tags:
        - Users
      summary: Create user
      operationId: createUser
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '422':
          $ref: '#/components/responses/ValidationError'

components:
  schemas:
    CreateUserRequest:
      type: object
      required:
        - email
        - name
      properties:
        email:
          type: string
          format: email
        name:
          type: string
        role:
          type: string
          enum: [user, admin]
          default: user

    UserResponse:
      type: object
      required:
        - id
        - email
        - name
        - created_at
        - updated_at
      properties:
        id:
          type: string
          minLength: 32
        email:
          type: string
          format: email
        name:
          type: string
        role:
          type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time
```

```bash
# Run contract test - MUST PASS now
npx dredd api/openapi.yaml http://localhost:3000
# Expected: PASS - all contract tests pass

# Validate OpenAPI specification
npx @apidevtools/swagger-cli validate api/openapi.yaml
# Expected: Exit code 0
```

#### Step 3: REFACTOR - Enhance with Full Validation Rules

```yaml
# api/openapi.yaml - Refactored with comprehensive validation
components:
  schemas:
    CreateUserRequest:
      type: object
      required:
        - email
        - name
      properties:
        email:
          type: string
          format: email
          minLength: 5
          maxLength: 255
          pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
          description: Valid email address
        name:
          type: string
          minLength: 1
          maxLength: 100
          pattern: '^[a-zA-Z\s\-]+$'
          description: Full name (letters, spaces, hyphens only)
        role:
          type: string
          enum: [user, admin, moderator]
          default: user
          description: User role assignment
      additionalProperties: false
      examples:
        - email: "newuser@example.com"
          name: "John Doe"
          role: "user"
```

```bash
# Re-run all tests - MUST still pass after refactoring
npx dredd api/openapi.yaml http://localhost:3000
# Expected: PASS - all tests still green

# Verify OpenAPI is still valid
npx @apidevtools/swagger-cli validate api/openapi.yaml
# Expected: Exit code 0
```

### TDD Checklist for REST APIs

**Before implementing ANY new endpoint:**
- [ ] Write contract test defining expected request/response
- [ ] Run test to confirm it FAILS (red)
- [ ] Implement minimal OpenAPI specification
- [ ] Run test to confirm it PASSES (green)
- [ ] Refactor with full validation, documentation, examples
- [ ] Run test to confirm it still PASSES (green)
- [ ] Validate OpenAPI specification

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every REST API bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow for REST APIs

```
┌─────────────────────────────────────────────────────────────────┐
│                  REST API Bug Fix Workflow                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Bug Reported/Discovered                                    │
│      - Document: endpoint, method, request, expected vs actual  │
│                          ↓                                       │
│   2. Write Contract Test that REPRODUCES the Bug                │
│      - Create test case with exact failing scenario             │
│      - Test MUST FAIL (proves bug exists)                       │
│                          ↓                                       │
│   3. Verify Test Fails for the RIGHT Reason                     │
│      - Confirm error matches reported bug                        │
│      - Not a different/unrelated failure                         │
│                          ↓                                       │
│   4. Fix the Bug in OpenAPI/Implementation                      │
│      - Update schema validation rules                            │
│      - Fix response format                                       │
│      - Correct status codes                                      │
│                          ↓                                       │
│   5. Verify Test Now PASSES                                     │
│      - Bug is fixed                                              │
│      - All other tests still pass (no regressions)              │
│                          ↓                                       │
│   6. Document Bug in Test Comments                              │
│      - Include bug/ticket ID                                     │
│      - Describe original issue                                   │
│                          ↓                                       │
│   7. Deploy with Confidence (Regression Prevented)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix: Incorrect Status Code on Validation Error

```yaml
# Bug Report #API-427: POST /v1/users returns 400 instead of 422 for validation errors
# Expected: 422 Unprocessable Entity for schema validation failures
# Actual: 400 Bad Request

# Step 1-2: Write test that reproduces the bug
# tests/regression/api_427_validation_status_code.yaml

test_api_427_validation_returns_422:
  description: |
    Bug #API-427: Validation errors must return 422 Unprocessable Entity,
    not 400 Bad Request. 422 indicates the request was well-formed but
    semantically incorrect (validation failed).
  request:
    method: POST
    path: /v1/users
    headers:
      Content-Type: application/json
      Authorization: Bearer ${TEST_TOKEN}
    body:
      email: "invalid-email-format"  # Invalid email triggers validation
      name: "John Doe"
  expected_response:
    status: 422  # MUST be 422, not 400
    body:
      error: "VALIDATION_ERROR"
      message: "${json-schema: string}"
      details:
        - field: "email"
          message: "${json-schema: string}"
```

```bash
# Run regression test - MUST FAIL (proves bug exists)
newman run tests/regression/api_427_validation_status_code.json
# Expected: FAIL - received 400, expected 422
```

```yaml
# Step 3-4: Fix the bug in OpenAPI specification
# api/openapi.yaml - Ensure 422 is used for validation errors

paths:
  /v1/users:
    post:
      responses:
        '201':
          description: User created successfully
        '400':
          description: Bad request (malformed JSON, missing Content-Type)
          # ✅ 400 only for malformed requests
        '422':
          description: Validation error (schema validation failed)
          # ✅ 422 for validation errors
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
              example:
                error: "VALIDATION_ERROR"
                message: "Request validation failed"
                details:
                  - field: "email"
                    message: "Invalid email format"
```

```bash
# Step 5: Run regression test - MUST PASS now
newman run tests/regression/api_427_validation_status_code.json
# Expected: PASS - returns 422 as expected

# Run all tests to ensure no regressions
npx dredd api/openapi.yaml http://localhost:3000
# Expected: PASS - all tests pass

# Validate OpenAPI specification
npx @apidevtools/swagger-cli validate api/openapi.yaml
# Expected: Exit code 0
```

### Common REST API Bugs and Regression Tests

#### Bug Type: Missing Required Field in Response

```yaml
# Bug #API-501: Response missing 'created_at' field
test_api_501_response_includes_created_at:
  description: "Bug #API-501: UserResponse must include created_at timestamp"
  request:
    method: GET
    path: /v1/users/${USER_ID}
    headers:
      Authorization: Bearer ${TEST_TOKEN}
  expected_response:
    status: 200
    body:
      id: "${json-schema: string}"
      email: "${json-schema: string}"
      created_at: "${json-schema: string, format: date-time}"  # MUST be present
```

#### Bug Type: Incorrect Error Format

```yaml
# Bug #API-502: Error response not following standard format
test_api_502_standard_error_format:
  description: "Bug #API-502: All errors must follow standard Error schema"
  request:
    method: GET
    path: /v1/users/nonexistent-user-id-12345678901234567890
    headers:
      Authorization: Bearer ${TEST_TOKEN}
  expected_response:
    status: 404
    body:
      error: "NOT_FOUND"  # MUST have error code
      message: "${json-schema: string}"  # MUST have message
```

#### Bug Type: Security Issue - Sensitive Data in URL

```yaml
# Bug #API-503: Email exposed in URL path (security vulnerability)
test_api_503_no_sensitive_data_in_url:
  description: |
    Bug #API-503: SECURITY - Endpoint was using email in URL path.
    URLs are logged and cached. Use unpredictable IDs only.
  request:
    method: GET
    path: /v1/users/${UNPREDICTABLE_USER_ID}  # ✅ Use ID, not email
    headers:
      Authorization: Bearer ${TEST_TOKEN}
  expected_response:
    status: 200
    # Endpoint /v1/users/{email} must NOT exist
```

### Bug Fix Checklist

**Before fixing ANY REST API bug:**
- [ ] Document the bug (endpoint, method, expected vs actual behavior)
- [ ] Write regression test that REPRODUCES the bug
- [ ] Run test to confirm it FAILS (proves bug exists)
- [ ] Fix the bug in OpenAPI specification and/or implementation
- [ ] Run test to confirm it PASSES (bug fixed)
- [ ] Run ALL tests to ensure no regressions introduced
- [ ] Validate OpenAPI specification still valid
- [ ] Add bug ID to test comments for traceability

---

## 3. OpenAPI Specification (MANDATORY)

### A. OpenAPI Structure

**CRITICAL: All REST APIs MUST be defined using OpenAPI 3.1+ specification. OpenAPI is the single source of truth.**

#### ✅ CORRECT - Complete OpenAPI Structure

```yaml
# api/openapi.yaml - Complete OpenAPI specification

openapi: 3.1.0
info:
  title: My REST API
  version: 1.0.0
  description: |
    Modern REST API with OpenAPI specification.
    This API follows hexagonal architecture principles.
  contact:
    name: API Support
    email: support@example.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server

tags:
  - name: Authentication
    description: Authentication and authorization endpoints
  - name: Users
    description: User management endpoints
  - name: Health
    description: Health check and monitoring endpoints

paths:
  /health:
    get:
      tags:
        - Health
      summary: Health check endpoint
      description: Returns the health status of the API
      operationId: getHealth
      responses:
        '200':
          description: API is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
              example:
                status: "healthy"
                timestamp: "2024-01-01T00:00:00Z"
                version: "1.0.0"

  /v1/auth/login:
    post:
      tags:
        - Authentication
      summary: User login
      description: Authenticates a user and returns access token
      operationId: login
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/LoginRequest'
            example:
              email: "user@example.com"
              password: "securePassword123"
      responses:
        '200':
          description: Login successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LoginResponse'
              example:
                access_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                refresh_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                expires_in: 3600
                token_type: "Bearer"
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '422':
          $ref: '#/components/responses/ValidationError'

  /v1/users:
    get:
      tags:
        - Users
      summary: List users
      description: Retrieves a list of users
      operationId: listUsers
      security:
        - bearerAuth: []
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserListResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'

    post:
      tags:
        - Users
      summary: Create user
      description: Creates a new user
      operationId: createUser
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            example:
              email: "newuser@example.com"
              name: "John Doe"
              role: "user"
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '400':
          $ref: '#/components/responses/BadRequestError'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '422':
          $ref: '#/components/responses/ValidationError'

  /v1/users/{userId}:
    get:
      tags:
        - Users
      summary: Get user by ID
      description: Retrieves a user by their ID
      operationId: getUserById
      security:
        - bearerAuth: []
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            pattern: '^[a-zA-Z0-9_-]{32,}$'
          description: Unpredictable user ID (minimum 32 characters)
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '404':
          $ref: '#/components/responses/NotFoundError'

    put:
      tags:
        - Users
      summary: Update user
      description: Updates an existing user
      operationId: updateUser
      security:
        - bearerAuth: []
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            pattern: '^[a-zA-Z0-9_-]{32,}$'
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateUserRequest'
            example:
              name: "Jane Doe"
              role: "admin"
      responses:
        '200':
          description: User updated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
        '404':
          $ref: '#/components/responses/NotFoundError'
        '422':
          $ref: '#/components/responses/ValidationError'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT token authentication
    cookieAuth:
      type: apiKey
      in: cookie
      name: session_token
      description: Session cookie authentication

  schemas:
    HealthResponse:
      type: object
      required:
        - status
        - timestamp
        - version
      properties:
        status:
          type: string
          enum: [healthy, unhealthy, degraded]
        timestamp:
          type: string
          format: date-time
        version:
          type: string
          pattern: '^\d+\.\d+\.\d+$'

    LoginRequest:
      type: object
      required:
        - email
        - password
      properties:
        email:
          type: string
          format: email
          minLength: 5
          maxLength: 255
        password:
          type: string
          format: password
          minLength: 8
          maxLength: 128

    LoginResponse:
      type: object
      required:
        - access_token
        - expires_in
        - token_type
      properties:
        access_token:
          type: string
          description: JWT access token
        refresh_token:
          type: string
          description: JWT refresh token (optional)
        expires_in:
          type: integer
          description: Token expiration time in seconds
        token_type:
          type: string
          enum: [Bearer]
          default: Bearer

    CreateUserRequest:
      type: object
      required:
        - email
        - name
      properties:
        email:
          type: string
          format: email
          minLength: 5
          maxLength: 255
        name:
          type: string
          minLength: 1
          maxLength: 100
        role:
          type: string
          enum: [user, admin, moderator]
          default: user

    UpdateUserRequest:
      type: object
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        role:
          type: string
          enum: [user, admin, moderator]
      additionalProperties: false

    UserResponse:
      type: object
      required:
        - id
        - email
        - name
        - created_at
        - updated_at
      properties:
        id:
          type: string
          description: Unpredictable user ID
          pattern: '^[a-zA-Z0-9_-]{32,}$'
        email:
          type: string
          format: email
        name:
          type: string
        role:
          type: string
          enum: [user, admin, moderator]
          default: user
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    UserListResponse:
      type: object
      required:
        - data
        - pagination
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/UserResponse'
        pagination:
          $ref: '#/components/schemas/Pagination'

    Pagination:
      type: object
      required:
        - page
        - limit
        - total
        - total_pages
      properties:
        page:
          type: integer
          minimum: 1
        limit:
          type: integer
          minimum: 1
        total:
          type: integer
          minimum: 0
        total_pages:
          type: integer
          minimum: 0

    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
          description: Error code
        message:
          type: string
          description: Human-readable error message
        details:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string

  responses:
    BadRequestError:
      description: Bad request
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "BAD_REQUEST"
            message: "Invalid request parameters"

    UnauthorizedError:
      description: Unauthorized
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "UNAUTHORIZED"
            message: "Authentication required"

    NotFoundError:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "NOT_FOUND"
            message: "Resource not found"

    ValidationError:
      description: Validation error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "VALIDATION_ERROR"
            message: "Request validation failed"
            details:
              - field: "email"
                message: "Invalid email format"
              - field: "password"
                message: "Password must be at least 8 characters"
```

#### ❌ WRONG - Invalid or Incomplete OpenAPI

```yaml
# ❌ Missing OpenAPI version
info:
  title: My API

# ❌ No JSON Schema for request body
paths:
  /users:
    post:
      requestBody:
        content:
          application/json: {}  # ❌ No schema

# ❌ No security definitions
# ❌ No examples
# ❌ No validation rules
```

---

## 4. JSON Schema Validation (MANDATORY)

### A. Request/Response Body Schemas

**CRITICAL: All request and response bodies MUST have complete JSON Schema definitions with validation rules.**

#### ✅ CORRECT - Complete JSON Schema

```yaml
components:
  schemas:
    CreateUserRequest:
      type: object
      required:
        - email
        - name
      properties:
        email:
          type: string
          format: email
          minLength: 5
          maxLength: 255
          pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
          description: User email address
        name:
          type: string
          minLength: 1
          maxLength: 100
          pattern: '^[a-zA-Z\s]+$'
          description: User full name
        role:
          type: string
          enum: [user, admin, moderator]
          default: user
          description: User role
        age:
          type: integer
          minimum: 18
          maximum: 120
          description: User age (optional)
      additionalProperties: false
      examples:
        - email: "user@example.com"
          name: "John Doe"
          role: "user"
        - email: "admin@example.com"
          name: "Jane Admin"
          role: "admin"
          age: 30
```

### B. Validation Rules

**CRITICAL: JSON Schemas MUST include validation rules: minLength, maxLength, pattern, format, enum, minimum, maximum, etc.**

#### ✅ CORRECT - Comprehensive Validation

```yaml
    Password:
      type: string
      format: password
      minLength: 8
      maxLength: 128
      pattern: '^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
      description: Password must contain uppercase, lowercase, number, and special character

    Email:
      type: string
      format: email
      minLength: 5
      maxLength: 255
      pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    UserID:
      type: string
      pattern: '^[a-zA-Z0-9_-]{32,}$'
      description: Unpredictable user ID (minimum 32 characters, alphanumeric with underscores and hyphens)
```

---

## 5. API Versioning (MANDATORY)

### A. Versioning Strategy

**CRITICAL: All APIs MUST support versioning. Use URL path versioning (preferred) or header versioning.**

#### ✅ CORRECT - URL Path Versioning

```yaml
servers:
  - url: https://api.example.com/v1
    description: API version 1
  - url: https://api.example.com/v2
    description: API version 2

paths:
  /v1/users:
    get:
      # Version 1 endpoint
  /v2/users:
    get:
      # Version 2 endpoint
```

#### ✅ CORRECT - Header Versioning (Alternative)

```yaml
paths:
  /users:
    get:
      parameters:
        - name: API-Version
          in: header
          required: true
          schema:
            type: string
            enum: [v1, v2]
          description: API version
```

#### ❌ WRONG - No Versioning

```yaml
# ❌ No versioning strategy
paths:
  /users:
    get:
      # ❌ No version information
```

---

## 6. Health Check and Keepalive (MANDATORY)

### A. Health Check Endpoint

**CRITICAL: All APIs MUST expose a health check endpoint for monitoring and keepalive.**

#### ✅ CORRECT - Health Check Endpoint

```yaml
paths:
  /health:
    get:
      tags:
        - Health
      summary: Health check endpoint
      description: |
        Returns the health status of the API.
        This endpoint should be used for monitoring and load balancer health checks.
      operationId: getHealth
      responses:
        '200':
          description: API is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
              example:
                status: "healthy"
                timestamp: "2024-01-01T00:00:00Z"
                version: "1.0.0"
                uptime: 3600
                checks:
                  database: "healthy"
                  cache: "healthy"
                  external_api: "healthy"
        '503':
          description: API is unhealthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'
              example:
                status: "unhealthy"
                timestamp: "2024-01-01T00:00:00Z"
                version: "1.0.0"
                checks:
                  database: "unhealthy"
                  cache: "healthy"
                  external_api: "healthy"

components:
  schemas:
    HealthResponse:
      type: object
      required:
        - status
        - timestamp
        - version
      properties:
        status:
          type: string
          enum: [healthy, unhealthy, degraded]
          description: Overall health status
        timestamp:
          type: string
          format: date-time
          description: Current server timestamp
        version:
          type: string
          pattern: '^\d+\.\d+\.\d+$'
          description: API version
        uptime:
          type: integer
          minimum: 0
          description: Server uptime in seconds (optional)
        checks:
          type: object
          description: Individual service health checks (optional)
          additionalProperties:
            type: string
            enum: [healthy, unhealthy]
```

#### ❌ WRONG - Missing Health Check

```yaml
# ❌ No health check endpoint
paths:
  /users:
    # Only business endpoints, no health check
```

---

## 7. Security and Authentication (MANDATORY)

### A. Authentication Requirements

**CRITICAL: All protected endpoints MUST require authentication. JWT (preferred) or cookie-based authentication.**

#### ✅ CORRECT - JWT Authentication

```yaml
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWT token authentication.
        Include the token in the Authorization header:
        Authorization: Bearer <token>

paths:
  /v1/users:
    get:
      security:
        - bearerAuth: []
      # Endpoint requires JWT authentication

  /v1/users/{userId}:
    get:
      security:
        - bearerAuth: []
      # Endpoint requires JWT authentication
```

#### ✅ CORRECT - Cookie Authentication (Alternative)

```yaml
components:
  securitySchemes:
    cookieAuth:
      type: apiKey
      in: cookie
      name: session_token
      description: Session cookie authentication

paths:
  /v1/users:
    get:
      security:
        - cookieAuth: []
      # Endpoint requires cookie authentication
```

### B. Login and Logout Endpoints

**CRITICAL: APIs requiring authentication MUST provide login and logout endpoints.**

#### ✅ CORRECT - Authentication Endpoints

```yaml
paths:
  /v1/auth/login:
    post:
      tags:
        - Authentication
      summary: User login
      description: |
        Authenticates a user and returns access token.
        Supports both individual request authentication and session-based authentication.
      operationId: login
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/LoginRequest'
            example:
              email: "user@example.com"
              password: "securePassword123"
      responses:
        '200':
          description: Login successful
          headers:
            Set-Cookie:
              description: Session cookie (if cookie-based auth)
              schema:
                type: string
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LoginResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'

  /v1/auth/logout:
    post:
      tags:
        - Authentication
      summary: User logout
      description: |
        Logs out the current user and invalidates the session.
        Works with both JWT and cookie-based authentication.
      operationId: logout
      security:
        - bearerAuth: []
        - cookieAuth: []
      responses:
        '200':
          description: Logout successful
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
                    example: "Logged out successfully"
        '401':
          $ref: '#/components/responses/UnauthorizedError'

  /v1/auth/refresh:
    post:
      tags:
        - Authentication
      summary: Refresh access token
      description: Refreshes an expired access token using a refresh token
      operationId: refreshToken
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - refresh_token
              properties:
                refresh_token:
                  type: string
                  description: Refresh token
      responses:
        '200':
          description: Token refreshed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LoginResponse'
        '401':
          $ref: '#/components/responses/UnauthorizedError'
```

---

## 8. URL Security (MANDATORY)

### A. No Sensitive Data in URLs

**CRITICAL: URLs MUST NOT contain passwords, tokens, or sensitive user data. Only use unpredictable resource IDs.**

#### ✅ CORRECT - Secure URL Design

```yaml
paths:
  /v1/users/{userId}:
    get:
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            pattern: '^[a-zA-Z0-9_-]{32,}$'
          description: Unpredictable user ID (minimum 32 characters)
      # ✅ Only ID in URL, no sensitive data

  /v1/users/{userId}/profile:
    get:
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            pattern: '^[a-zA-Z0-9_-]{32,}$'
      # ✅ Only ID in URL
```

#### ❌ WRONG - Sensitive Data in URLs

```yaml
paths:
  /v1/users/{email}:  # ❌ Email in URL (sensitive data)
  /v1/users/{password}:  # ❌ Password in URL (NEVER!)
  /v1/users/{token}:  # ❌ Token in URL (sensitive data)
  /v1/users/123:  # ❌ Predictable sequential ID
```

### B. Unpredictable Resource IDs

**CRITICAL: Resource IDs MUST be unpredictable (minimum 32 characters, alphanumeric with special characters).**

#### ✅ CORRECT - Unpredictable ID Pattern

```yaml
components:
  schemas:
    UserID:
      type: string
      pattern: '^[a-zA-Z0-9_-]{32,}$'
      description: |
        Unpredictable user ID.
        Minimum 32 characters, alphanumeric with underscores and hyphens.
        Example: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
      minLength: 32
      maxLength: 64
      example: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
```

#### ❌ WRONG - Predictable IDs

```yaml
# ❌ Sequential IDs
UserID:
  type: integer
  example: 1  # ❌ Predictable

# ❌ Short IDs
UserID:
  type: string
  pattern: '^[a-zA-Z0-9]{4}$'  # ❌ Too short, predictable
```

---

## 9. HTTP Methods and Request Bodies (MANDATORY)

### A. POST and PUT with Request Bodies

**CRITICAL: POST and PUT methods MUST include request data in the body, not in URL parameters or query strings.**

#### ✅ CORRECT - Request Body in POST/PUT

```yaml
paths:
  /v1/users:
    post:
      summary: Create user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            example:
              email: "user@example.com"
              name: "John Doe"
              role: "user"
      # ✅ Data in request body

  /v1/users/{userId}:
    put:
      summary: Update user
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateUserRequest'
            example:
              name: "Jane Doe"
              role: "admin"
      # ✅ Data in request body, ID in path
```

#### ❌ WRONG - Data in URL or Query

```yaml
paths:
  /v1/users:
    post:
      parameters:
        - name: email
          in: query  # ❌ Data in query string
        - name: name
          in: query  # ❌ Data in query string
      # ❌ No request body

  /v1/users/{userId}:
    put:
      parameters:
        - name: userId
          in: path
        - name: name
          in: query  # ❌ Data in query string
      # ❌ No request body
```

---

## 10. Hexagonal Architecture for Routing (MANDATORY)

### A. Architecture Principles

**CRITICAL: API routing MUST follow hexagonal architecture principles with clear separation of concerns.**

#### ✅ CORRECT - Hexagonal API Structure

```yaml
# API structure reflects hexagonal architecture:

# Domain Layer (Core) - Defined in schemas
components:
  schemas:
    # Domain entities
    User:
      # Pure domain model, no framework dependencies
    UserRepository:
      # Repository interface (port)

# Application Layer - Defined in paths
paths:
  # Application endpoints (adapters)
  /v1/users:
    # Maps to application service
    # Uses domain entities
    # Implements repository port

# Infrastructure Layer - Handled by implementation
# - Database adapters
# - External API adapters
# - Authentication adapters
```

### B. Resource-Oriented Design

**CRITICAL: APIs MUST be resource-oriented, following REST principles.**

#### ✅ CORRECT - Resource-Oriented Endpoints

```yaml
paths:
  # Resource: Users
  /v1/users:
    get:      # List users
    post:     # Create user
  
  /v1/users/{userId}:
    get:      # Get user
    put:      # Update user
    delete:   # Delete user
  
  /v1/users/{userId}/profile:
    get:      # Get user profile (sub-resource)
    put:      # Update user profile
  
  # Resource: Posts
  /v1/posts:
    get:      # List posts
    post:     # Create post
  
  /v1/posts/{postId}:
    get:      # Get post
    put:      # Update post
    delete:   # Delete post
```

### C. Hierarchical Routing with Primary Parameters (MANDATORY)

**CRITICAL: As long as the URL path doesn't contain personal or secret data, hierarchical routing MUST contain primary parameters as part of the URL path so GET responses can be properly cached by HTTP caches, CDNs, and reverse proxies.**

#### ✅ CORRECT - Primary Parameters in URL Path

```yaml
paths:
  # ✅ Primary parameters in path for cacheability
  /v1/organizations/{orgId}/projects/{projectId}:
    get:
      summary: Get project by organization and project ID
      description: |
        Primary parameters (orgId, projectId) are in the URL path.
        This allows proper HTTP caching since the URL uniquely identifies the resource.
      parameters:
        - name: orgId
          in: path
          required: true
          schema:
            type: string
            pattern: '^[a-zA-Z0-9_-]{32,}$'
        - name: projectId
          in: path
          required: true
          schema:
            type: string
            pattern: '^[a-zA-Z0-9_-]{32,}$'
      # ✅ Cacheable: URL uniquely identifies resource

  /v1/organizations/{orgId}/projects/{projectId}/tasks/{taskId}:
    get:
      summary: Get task by organization, project, and task ID
      description: |
        Hierarchical structure with all primary identifiers in path.
        Enables proper HTTP caching at each level.
      parameters:
        - name: orgId
          in: path
          required: true
          schema:
            type: string
        - name: projectId
          in: path
          required: true
          schema:
            type: string
        - name: taskId
          in: path
          required: true
          schema:
            type: string
      # ✅ Cacheable: Complete resource hierarchy in URL

  /v1/users/{userId}/posts/{postId}/comments:
    get:
      summary: List comments for a post
      description: |
        Primary identifiers (userId, postId) in path.
        Secondary parameters (pagination, filtering) in query string.
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
        - name: postId
          in: path
          required: true
          schema:
            type: string
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      # ✅ Cacheable: Primary identifiers in path, pagination in query
```

#### ❌ WRONG - Primary Parameters in Query String

```yaml
paths:
  # ❌ Primary parameters in query - not cacheable
  /v1/projects:
    get:
      parameters:
        - name: orgId
          in: query  # ❌ Should be in path
        - name: projectId
          in: query  # ❌ Should be in path
      # ❌ Not cacheable: Query parameters prevent proper HTTP caching

  # ❌ Missing hierarchy
  /v1/tasks:
    get:
      parameters:
        - name: orgId
          in: query  # ❌ Should be in path: /v1/organizations/{orgId}/tasks
        - name: projectId
          in: query  # ❌ Should be in path
        - name: taskId
          in: query  # ❌ Should be in path
      # ❌ Not cacheable: No hierarchical structure
```

#### Guidelines for Hierarchical Routing

1. **Primary Resource Identifiers**: Always include primary resource identifiers in the URL path
   - Example: `/v1/organizations/{orgId}/projects/{projectId}` instead of `/v1/projects?orgId=X&projectId=Y`

2. **Hierarchical Structure**: Follow natural resource hierarchy
   - Example: `/v1/organizations/{orgId}/projects/{projectId}/tasks/{taskId}`
   - Reflects: Organization → Project → Task relationship

3. **Query Parameters for Secondary Data**: Use query parameters only for:
   - Pagination (`page`, `limit`)
   - Filtering (`filter`, `sort`)
   - Optional parameters that don't affect cacheability

4. **Security Exception**: If path would contain personal or secret data, use query parameters but document caching limitations
   ```yaml
   # ⚠️ Exception: Personal data in path would be insecure
   # Use query parameter but note caching limitations
   /v1/search:
     get:
       parameters:
         - name: email
           in: query  # ⚠️ Personal data - cannot be in path
       description: |
         Note: This endpoint may have limited cacheability due to
         personal data in query parameters. Consider using POST for
         complex searches with sensitive data.
   ```

5. **Cache Headers**: Ensure GET endpoints with path parameters include proper cache headers
   ```yaml
   /v1/organizations/{orgId}/projects/{projectId}:
     get:
       responses:
         '200':
           description: Project retrieved successfully
           headers:
             Cache-Control:
               description: Cache control header
               schema:
                 type: string
                 example: "public, max-age=3600"
           # ✅ Proper caching headers for cacheable resources
   ```

---

## 11. Documentation as Code (MANDATORY)

### A. OpenAPI Documentation

**CRITICAL: All endpoints MUST have complete documentation in the OpenAPI specification.**

#### ✅ CORRECT - Complete Documentation

```yaml
paths:
  /v1/users:
    get:
      tags:
        - Users
      summary: List users
      description: |
        Retrieves a paginated list of users.
        
        **Authentication Required**: Yes (JWT Bearer token)
        
        **Pagination**: Supports page and limit query parameters
        
        **Filtering**: Can filter by role using query parameter
        
        **Example Usage**:
        ```bash
        curl -X GET "https://api.example.com/v1/users?page=1&limit=20" \
          -H "Authorization: Bearer <token>"
        ```
      operationId: listUsers
      security:
        - bearerAuth: []
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
          description: Page number (1-based)
          example: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
          description: Number of items per page
          example: 20
      responses:
        '200':
          description: List of users retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserListResponse'
              examples:
                success:
                  summary: Successful response
                  value:
                    data:
                      - id: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
                        email: "user@example.com"
                        name: "John Doe"
                        role: "user"
                        created_at: "2024-01-01T00:00:00Z"
                        updated_at: "2024-01-01T00:00:00Z"
                    pagination:
                      page: 1
                      limit: 20
                      total: 100
                      total_pages: 5
```

### B. Test Examples

**CRITICAL: All endpoints MUST include test examples in the OpenAPI specification.**

#### ✅ CORRECT - Test Examples

```yaml
paths:
  /v1/users:
    post:
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
            examples:
              valid_user:
                summary: Valid user creation
                value:
                  email: "newuser@example.com"
                  name: "John Doe"
                  role: "user"
              admin_user:
                summary: Admin user creation
                value:
                  email: "admin@example.com"
                  name: "Jane Admin"
                  role: "admin"
              minimal_user:
                summary: Minimal required fields
                value:
                  email: "minimal@example.com"
                  name: "Min User"
      responses:
        '201':
          content:
            application/json:
              examples:
                created_user:
                  summary: Successfully created user
                  value:
                    id: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
                    email: "newuser@example.com"
                    name: "John Doe"
                    role: "user"
                    created_at: "2024-01-01T00:00:00Z"
                    updated_at: "2024-01-01T00:00:00Z"
```

---

## 12. JSON Body Requirements (MANDATORY)

### A. JSON for All Data Transfer

**CRITICAL: All request and response bodies MUST use JSON format when data is transferred.**

#### ✅ CORRECT - JSON Bodies

```yaml
paths:
  /v1/users:
    post:
      requestBody:
        required: true
        content:
          application/json:  # ✅ JSON content type
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          content:
            application/json:  # ✅ JSON response
              schema:
                $ref: '#/components/schemas/UserResponse'
```

#### ❌ WRONG - Non-JSON Bodies

```yaml
paths:
  /v1/users:
    post:
      requestBody:
        content:
          application/x-www-form-urlencoded:  # ❌ Form data
          text/plain:  # ❌ Plain text
          application/xml:  # ❌ XML
```

---

## 13. Error Responses (MANDATORY)

### Protocol-Specific Design Note

**Why REST error format differs from GraphQL/gRPC:**

| Aspect | REST | GraphQL | gRPC |
|--------|------|---------|------|
| **Error format** | JSON with `error`, `message`, `details` | `errors[]` with `extensions.code` | Status codes with `errdetails` |
| **Pagination** | URL params (`page`, `limit`) or cursor | Relay connections (`first`, `after`) | Request message (`page_size`, `page_token`) |
| **Naming** | snake_case (URLs and JSON) | camelCase (fields) | snake_case (proto fields) |
| **Rate limiting** | HTTP headers (`X-RateLimit-*`) | Query complexity limits | Interceptor-based |

These differences are **intentional and appropriate** for each protocol:
- REST uses HTTP semantics and headers for metadata
- GraphQL uses a single endpoint with typed schema
- gRPC uses binary Protocol Buffers with built-in status codes

**Cross-API services** should use API gateway transformations to convert between formats.

---

### A. Standardized Error Format

**CRITICAL: All error responses MUST follow a consistent JSON format with proper HTTP status codes.**

#### ✅ CORRECT - Standardized Errors

```yaml
components:
  schemas:
    Error:
      type: object
      required:
        - error
        - message
      properties:
        error:
          type: string
          description: Error code
        message:
          type: string
          description: Human-readable error message
        details:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string
          description: Validation error details (optional)
        timestamp:
          type: string
          format: date-time
          description: Error timestamp
        path:
          type: string
          description: Request path that caused the error

  responses:
    BadRequestError:
      description: Bad request (400)
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "BAD_REQUEST"
            message: "Invalid request parameters"
            timestamp: "2024-01-01T00:00:00Z"
            path: "/v1/users"

    UnauthorizedError:
      description: Unauthorized (401)
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "UNAUTHORIZED"
            message: "Authentication required"
            timestamp: "2024-01-01T00:00:00Z"
            path: "/v1/users"

    ForbiddenError:
      description: Forbidden (403)
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "FORBIDDEN"
            message: "Insufficient permissions"
            timestamp: "2024-01-01T00:00:00Z"
            path: "/v1/users"

    NotFoundError:
      description: Resource not found (404)
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "NOT_FOUND"
            message: "Resource not found"
            timestamp: "2024-01-01T00:00:00Z"
            path: "/v1/users/123"

    ValidationError:
      description: Validation error (422)
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "VALIDATION_ERROR"
            message: "Request validation failed"
            details:
              - field: "email"
                message: "Invalid email format"
              - field: "password"
                message: "Password must be at least 8 characters"
            timestamp: "2024-01-01T00:00:00Z"
            path: "/v1/users"

    InternalServerError:
      description: Internal server error (500)
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
          example:
            error: "INTERNAL_SERVER_ERROR"
            message: "An internal server error occurred"
            timestamp: "2024-01-01T00:00:00Z"
            path: "/v1/users"
```

---

## 14. Quick Reference

### HTTP Methods Summary

| Method | Purpose | Idempotent | Safe | Request Body | Success Code |
|--------|---------|------------|------|--------------|--------------|
| `GET` | Retrieve resource(s) | Yes | Yes | No | 200 OK |
| `POST` | Create resource | No | No | Yes | 201 Created |
| `PUT` | Replace resource | Yes | No | Yes | 200 OK |
| `PATCH` | Partial update | No | No | Yes | 200 OK |
| `DELETE` | Remove resource | Yes | No | No | 204 No Content |
| `HEAD` | Get headers only | Yes | Yes | No | 200 OK |
| `OPTIONS` | Get allowed methods | Yes | Yes | No | 200 OK |

### HTTP Status Codes

#### Success Codes (2xx)
| Code | Name | When to Use |
|------|------|-------------|
| `200` | OK | Successful GET, PUT, PATCH, DELETE with body |
| `201` | Created | Successful POST creating new resource |
| `202` | Accepted | Request accepted for async processing |
| `204` | No Content | Successful DELETE, PUT with no response body |

#### Client Error Codes (4xx)
| Code | Name | When to Use |
|------|------|-------------|
| `400` | Bad Request | Malformed JSON, missing Content-Type |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Authenticated but insufficient permissions |
| `404` | Not Found | Resource does not exist |
| `405` | Method Not Allowed | HTTP method not supported on endpoint |
| `409` | Conflict | Resource conflict (duplicate, version mismatch) |
| `422` | Unprocessable Entity | Schema validation failed |
| `429` | Too Many Requests | Rate limit exceeded |

#### Server Error Codes (5xx)
| Code | Name | When to Use |
|------|------|-------------|
| `500` | Internal Server Error | Unexpected server error |
| `502` | Bad Gateway | Upstream service unavailable |
| `503` | Service Unavailable | Server temporarily unavailable |
| `504` | Gateway Timeout | Upstream service timeout |

### Common URL Patterns

```
# Collection operations
GET    /v1/resources              # List resources (paginated)
POST   /v1/resources              # Create new resource

# Single resource operations
GET    /v1/resources/{id}         # Get resource by ID
PUT    /v1/resources/{id}         # Replace resource
PATCH  /v1/resources/{id}         # Partial update
DELETE /v1/resources/{id}         # Delete resource

# Sub-resource operations
GET    /v1/resources/{id}/items          # List sub-resources
POST   /v1/resources/{id}/items          # Create sub-resource
GET    /v1/resources/{id}/items/{itemId} # Get sub-resource

# Hierarchical resources (for cacheability)
GET    /v1/orgs/{orgId}/projects/{projectId}/tasks/{taskId}

# Actions (when REST verbs don't fit)
POST   /v1/resources/{id}/actions/publish
POST   /v1/resources/{id}/actions/archive
```

### Query Parameter Conventions

```yaml
# Pagination
?page=1&limit=20              # Page-based
?offset=0&limit=20            # Offset-based
?cursor=abc123&limit=20       # Cursor-based (recommended for large datasets)

# Sorting
?sort=created_at              # Ascending
?sort=-created_at             # Descending (prefix with -)
?sort=name,-created_at        # Multiple fields

# Filtering
?status=active                # Exact match
?status=active,pending        # Multiple values (OR)
?created_after=2024-01-01     # Date range
?search=keyword               # Full-text search

# Field selection
?fields=id,name,email         # Sparse fieldsets
?expand=profile,permissions   # Include related resources
```

### Request/Response Headers

```yaml
# Required Request Headers
Content-Type: application/json          # For POST, PUT, PATCH
Authorization: Bearer <token>           # For protected endpoints
Accept: application/json                # Expected response format
X-Request-ID: uuid                      # Client-provided request ID (optional)

# Common Response Headers
Content-Type: application/json          # Response format
X-Request-ID: uuid                      # Request tracking (echo or generate)
X-Trace-ID: uuid                        # Distributed trace ID for observability
X-RateLimit-Limit: 1000                 # Rate limit max
X-RateLimit-Remaining: 999              # Rate limit remaining
X-RateLimit-Reset: 1640000000           # Rate limit reset (Unix timestamp)
Cache-Control: public, max-age=3600     # Caching directive
ETag: "abc123"                          # Entity tag for caching
```

### Distributed Tracing (MANDATORY)

**CRITICAL: All REST APIs MUST propagate trace IDs for observability.**

```yaml
# Trace ID propagation headers (choose based on your stack)
X-Trace-ID: abc123-def456-ghi789       # Custom trace header
traceparent: 00-abc123-def456-01       # W3C Trace Context (recommended)
X-B3-TraceId: abc123                   # Zipkin B3 format

# Implementation requirements:
# 1. Accept trace ID from incoming request headers
# 2. Generate new trace ID if not provided
# 3. Include trace ID in all outgoing requests to downstream services
# 4. Include trace ID in all log entries (see logging.md Section 5)
# 5. Return trace ID in response headers
```

**Cross-reference:** See logging.md for trace ID implementation patterns.

### Standard Error Response Format

```json
{
  "error": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [
    {
      "field": "email",
      "message": "Invalid email format"
    },
    {
      "field": "password",
      "message": "Password must be at least 8 characters"
    }
  ],
  "timestamp": "2024-01-01T00:00:00Z",
  "path": "/v1/users"
}
```

### Pagination Response Format

```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "total_pages": 5,
    "has_next": true,
    "has_prev": false
  },
  "links": {
    "self": "/v1/users?page=1&limit=20",
    "first": "/v1/users?page=1&limit=20",
    "last": "/v1/users?page=5&limit=20",
    "next": "/v1/users?page=2&limit=20",
    "prev": null
  }
}
```

### OpenAPI Validation Commands

```bash
# Validate OpenAPI specification
npx @apidevtools/swagger-cli validate api/openapi.yaml

# Lint OpenAPI with Spectral
npx @stoplight/spectral-cli lint api/openapi.yaml

# Generate documentation
npx redoc-cli bundle api/openapi.yaml -o docs/api.html

# Run contract tests with Dredd
npx dredd api/openapi.yaml http://localhost:3000

# Validate JSON Schema
npx ajv-cli validate -s schema.json -d data.json
```

### Security Checklist

```yaml
# Authentication
- [ ] JWT Bearer tokens for stateless auth
- [ ] Secure cookie settings (HttpOnly, Secure, SameSite)
- [ ] Token expiration and refresh mechanism
- [ ] Password hashing (bcrypt, argon2)

# Authorization
- [ ] Role-based access control (RBAC)
- [ ] Resource ownership verification
- [ ] Rate limiting per user/IP

# URL Security
- [ ] No passwords in URLs
- [ ] No tokens in URLs (use headers)
- [ ] No PII in URL paths
- [ ] Unpredictable resource IDs (min 32 chars)

# Data Protection
- [ ] Input validation (JSON Schema)
- [ ] Output encoding
- [ ] SQL injection prevention
- [ ] XSS prevention in responses
```

### Resource ID Best Practices

```yaml
# ✅ CORRECT - Unpredictable IDs
UserID:
  type: string
  pattern: '^[a-zA-Z0-9_-]{32,}$'
  minLength: 32
  maxLength: 64
  example: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"

# Generation options:
# - UUID v4: 550e8400-e29b-41d4-a716-446655440000
# - ULID: 01ARZ3NDEKTSV4RRFFQ69G5FAV
# - NanoID: V1StGXR8_Z5jdHi6B-myT
# - CUID: clh2v9qzj0000mk08a5zxw9x5

# ❌ WRONG - Predictable IDs
UserID:
  type: integer
  example: 1, 2, 3  # Sequential, guessable
```

---

## 15. Deployment Checklist

### Build & Validation
- [ ] OpenAPI spec validates: `swagger-cli validate openapi.yaml` passes
- [ ] JSON Schemas validate against draft-2020-12
- [ ] All endpoints documented with request/response examples
- [ ] API versioning configured (URL path or header)

### Testing
- [ ] Contract tests pass for all endpoints
- [ ] Integration tests cover all CRUD operations
- [ ] Error response formats match the standardized schema
- [ ] Pagination, filtering, and sorting work correctly

### Security
- [ ] Authentication configured (JWT or session-based)
- [ ] No sensitive data in URLs or query parameters
- [ ] Resource IDs are unpredictable (minimum 32 characters)
- [ ] CORS configured with explicit allowed origins
- [ ] Rate limiting enabled on all public endpoints

### Agent Workflow
- [ ] Agent validated OpenAPI spec with `swagger-cli validate`
- [ ] Agent verified all JSON Schemas are well-formed
- [ ] Agent confirmed test examples exist and match schemas
- [ ] Documentation generated and verified with `redoc-cli bundle`

---

## 16. Why This Configuration Works

1. **OpenAPI-First Design**: Defining the API contract in OpenAPI before writing implementation code ensures frontend and backend teams can work in parallel. Auto-generated clients, server stubs, and documentation stay in sync with a single source of truth.

2. **Hexagonal Architecture for Routing**: Separating route definitions from business logic and data access makes API endpoints independently testable, swappable between frameworks, and resistant to vendor lock-in.

3. **Unpredictable Resource IDs**: Using UUIDs or CUIDs instead of sequential integers prevents enumeration attacks where an attacker iterates through IDs to discover or access resources they should not see.

4. **Standardized Error Responses**: A consistent JSON error format with `error`, `message`, `timestamp`, and `path` fields enables clients to implement uniform error handling logic and provides operators with structured data for debugging and monitoring.

5. **JSON Schema Validation on All Payloads**: Validating request and response bodies against JSON Schemas catches malformed data at the API boundary, preventing invalid state from propagating into business logic or the database.

---

## 17. Summary

**CRITICAL Requirements for All REST API Designs:**

1. **OpenAPI Specification**: All APIs MUST be defined in OpenAPI 3.1+
2. **JSON Schema Validation**: All request/response bodies MUST have JSON Schema with validation rules
3. **JSON Bodies**: All data transfer MUST use JSON format
4. **API Versioning**: All APIs MUST support versioning (URL path preferred)
5. **Health Check**: All APIs MUST expose /health endpoint
6. **Authentication**: JWT (preferred) or cookie-based authentication for protected endpoints
7. **Login/Logout**: APIs requiring auth MUST provide login and logout endpoints
8. **URL Security**: No sensitive data in URLs, only unpredictable resource IDs (minimum 32 chars)
9. **Request Bodies**: POST/PUT MUST include data in request body, not URL/query
10. **Hexagonal Architecture**: Routing MUST follow hexagonal architecture principles
11. **Hierarchical Routing**: Primary parameters MUST be in URL path (when not personal/secret) for proper GET caching
12. **Documentation**: All endpoints MUST be documented with examples in OpenAPI
13. **Test Examples**: All endpoints MUST include test examples
14. **Error Responses**: Standardized JSON error format with proper HTTP status codes
15. **Validation**: JSON Schema MUST include validation rules, optional fields, and defaults
16. **Verification**: Agent MUST validate OpenAPI, schemas, and documentation before delivery

**Agent Verification Protocol:**
- **MANDATORY**: Validate OpenAPI (`swagger-cli validate`) - ALWAYS required
- **MANDATORY**: Validate JSON Schemas - ALWAYS required
- Generate documentation (`redoc-cli bundle`)
- **MANDATORY**: Verify test examples exist and match schemas
- **MANDATORY**: After ANY modification, re-validate all components
- Only present working, validated, documented APIs to the user

**Remember**: Minimalistic, clean, secure, well-documented REST API designs with OpenAPI specifications, comprehensive validation, hexagonal architecture, proper authentication, and focus on security and portability. Keep it RESTful, keep it secure, keep it documented.


**End of Modern REST API Development Guidelines**

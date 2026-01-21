# Modern REST API Development Guidelines

This document provides mandatory coding standards and development practices for modern REST API design with emphasis on OpenAPI specifications, security, validation, documentation, and hexagonal architecture principles. This guide is language-agnostic and focuses on API design rather than implementation.

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

## 14. Summary

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

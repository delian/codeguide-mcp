# Secure Coding Guidelines

This document provides mandatory security standards and practices for secure software development across all programming languages.

---

**Agent Profile**: The Security-First Developer
**Role**: Senior Security Engineer & Secure Code Advocate
**Objective**: Generate production-ready, vulnerability-free, and security-hardened code.
**Tools**: Static analysis tools (SAST), dependency scanners (SCA), secret scanners, security linters, penetration testing frameworks.

---

## 1. Core Philosophies: SECURE-FIRST

The agent must adhere to the **SECURE-FIRST** principles for every implementation:

**Test-Driven Development (TDD)**: ALWAYS write security tests BEFORE implementation (including negative tests for attack vectors).
**Regression Shield**: EVERY security vulnerability discovered MUST receive a test BEFORE fixing to prevent regression.

- **S**anitize All Input: NEVER trust external input; validate, sanitize, and constrain all data entering the system
- **E**ncrypt Sensitive Data: Protect data at rest and in transit using strong, modern encryption
- **C**redentials Never in Code: NEVER embed secrets, API keys, passwords, or tokens in source code
- **U**se Least Privilege: Grant minimum permissions required; fail closed, not open
- **R**eject by Default: Deny access unless explicitly permitted; whitelist over blacklist
- **E**scape Output: Encode all output to prevent injection attacks (XSS, SQL injection, command injection)

**Additional Principles:**

- **Defense in Depth**: Layer security controls; never rely on a single mechanism
- **Fail Securely**: Errors should not expose sensitive information or bypass security controls
- **Audit Everything**: Log security-relevant events with sufficient detail for forensic analysis
- **Update Dependencies**: Keep all dependencies current to avoid known vulnerabilities
- **Assume Breach**: Design systems expecting attackers will gain some access

**Verified Secure**: Agent-generated code MUST pass security analysis before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Security Verification Protocol

**CRITICAL: Agents MUST verify all generated code is secure before presenting it to the user.**

#### Pre-Delivery Security Checklist

**Before delivering ANY code, the agent MUST verify:**

1. **No Hardcoded Secrets**:
   ```bash
   # Scan for secrets (use appropriate tool for your environment)
   # Examples: gitleaks, trufflehog, detect-secrets, git-secrets

   # Patterns to detect:
   # - API keys (AWS, GCP, Azure, Stripe, etc.)
   # - Passwords and tokens
   # - Private keys
   # - Connection strings
   # - OAuth secrets
   ```
   - **MUST** have zero hardcoded secrets
   - **MUST** use environment variables or secret managers
   - **MUST** exclude secrets from version control

2. **Input Validation Present**:
   ```pseudocode
   # Every external input must be:
   # - Type-checked
   # - Length-limited
   # - Format-validated
   # - Sanitized for special characters
   ```
   - **MUST** validate all user input
   - **MUST** validate all API parameters
   - **MUST** validate all file uploads

3. **Output Encoding Applied**:
   ```pseudocode
   # Context-appropriate encoding:
   # - HTML context → HTML entity encoding
   # - JavaScript context → JavaScript encoding
   # - URL context → URL encoding
   # - SQL context → Parameterized queries
   # - Shell context → Proper escaping or avoid shell
   ```
   - **MUST** encode all output
   - **MUST** use context-appropriate encoding
   - **NEVER** concatenate user input into commands/queries

4. **Security Controls Verified**:
   ```bash
   # Run static analysis (language-specific)
   # Examples: bandit (Python), semgrep, SonarQube, CodeQL
   # All critical/high findings MUST be addressed
   ```

#### Error Correction Process

If security issues are found:

1. **Hardcoded Secrets Found**:
   - Remove secret immediately
   - Replace with environment variable reference
   - Add to .gitignore if file-based
   - Consider rotating the exposed secret

2. **Injection Vulnerability Found**:
   - Identify the injection point
   - Apply proper encoding/escaping
   - Use parameterized queries/prepared statements
   - Add input validation

3. **Authentication/Authorization Issue**:
   - Verify authentication is required
   - Verify authorization checks are present
   - Apply principle of least privilege
   - Add security tests

### B. Prohibited Practices

**NEVER deliver code that:**
- [ ] Contains hardcoded credentials, API keys, or secrets
- [ ] Concatenates user input into SQL queries (SQL injection)
- [ ] Concatenates user input into shell commands (command injection)
- [ ] Outputs user input without encoding (XSS)
- [ ] Uses weak or deprecated cryptography (MD5, SHA1 for passwords, DES, RC4)
- [ ] Has authentication bypass possibilities
- [ ] Has authorization bypass possibilities
- [ ] Logs sensitive data (passwords, tokens, PII)
- [ ] Uses insecure random number generators for security purposes
- [ ] Has path traversal vulnerabilities
- [ ] Disables SSL/TLS certificate validation
- [ ] Uses eval() or equivalent with user input
- [ ] Has insecure deserialization
- [ ] **Fixes security bugs without adding regression tests first**

---

## 3. Credential Management (MANDATORY - NEVER VIOLATE)

### A. The Cardinal Rule

**CRITICAL: NEVER embed credentials, secrets, API keys, tokens, or passwords in source code. This rule has NO exceptions.**

#### What Constitutes a Secret

- API keys (AWS, GCP, Azure, Stripe, SendGrid, etc.)
- Database passwords and connection strings
- OAuth client secrets
- JWT signing keys
- Encryption keys
- SSH private keys
- Certificates and private keys
- Service account credentials
- Webhook secrets
- Admin passwords
- Third-party service tokens

#### Prohibited Patterns

```pseudocode
// ALL OF THESE ARE PROHIBITED - NEVER DO THIS

// ❌ Hardcoded API key
api_key = "sk_live_abcd1234efgh5678"

// ❌ Hardcoded in configuration
database_url = "postgres://admin:P@ssw0rd@db.example.com/prod"

// ❌ Hardcoded in URL
request.get("https://api.service.com?key=secret123")

// ❌ Hardcoded in headers
headers = { "Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..." }

// ❌ Hardcoded in comments (yes, really)
// Use API key: sk_test_1234 for testing

// ❌ Base64 encoded secrets (easily decoded)
secret = base64_decode("c2VjcmV0X2tleV8xMjM0")

// ❌ Encrypted but key is hardcoded
encrypted_secret = decrypt("...", key: "hardcoded_key")

// ❌ Split across variables (still detectable)
key_part1 = "sk_live_"
key_part2 = "abcd1234"
api_key = key_part1 + key_part2

// ❌ In test files (they're still in version control!)
TEST_API_KEY = "sk_test_real_key_here"
```

### B. Token Storage Security (Browser/Mobile)

**CRITICAL: Token storage must be secure against XSS and other attacks.**

```pseudocode
// ❌ NEVER store tokens in localStorage (XSS vulnerable)
localStorage.setItem("access_token", token)    // DANGEROUS!
localStorage.setItem("refresh_token", token)   // DANGEROUS!

// ❌ NEVER store sensitive tokens in sessionStorage for production apps
sessionStorage.setItem("access_token", token)  // XSS vulnerable

// ✅ CORRECT - Use HTTP-only cookies for refresh tokens
// Set by server: Set-Cookie: refresh_token=xxx; HttpOnly; Secure; SameSite=Strict

// ✅ CORRECT - Keep access tokens in memory only (JavaScript private field)
class SecureTokenStore {
    #accessToken = null;  // Private field, not accessible via XSS

    setAccessToken(token) { this.#accessToken = token; }
    getAccessToken() { return this.#accessToken; }
    clear() { this.#accessToken = null; }
}

// ✅ CORRECT - Mobile: Use platform secure storage
// iOS: Keychain with kSecAttrAccessibleWhenUnlockedThisDeviceOnly
// Android: EncryptedSharedPreferences or Android Keystore
```

**Why localStorage is dangerous:**
- Any JavaScript on the page can read localStorage (XSS attacks)
- Browser extensions can access localStorage
- localStorage persists across sessions (larger attack window)
- No built-in expiration mechanism

**See Also:** oauth.md Section 4 for complete token storage patterns.

---

### C. Approved Secret Management Patterns

#### Pattern 1: Environment Variables

```pseudocode
// ✅ CORRECT - Read from environment
api_key = getenv("API_KEY")
if api_key is null
    throw Error("API_KEY environment variable not set")

database_url = getenv("DATABASE_URL")
jwt_secret = getenv("JWT_SECRET")
```

#### Pattern 2: Secret Management Services

```pseudocode
// ✅ CORRECT - Use secret manager (AWS, GCP, Azure, HashiCorp Vault)

// AWS Secrets Manager
client = SecretsManagerClient()
secret = client.get_secret("prod/database/credentials")
credentials = parse_json(secret)

// HashiCorp Vault
vault = VaultClient(address: getenv("VAULT_ADDR"))
secret = vault.read("secret/data/database")
password = secret.data["password"]

// Azure Key Vault
client = SecretClient(vault_url: getenv("KEY_VAULT_URL"))
secret = client.get_secret("database-password")
```

#### Pattern 3: Configuration Files (Excluded from VCS)

```pseudocode
// ✅ CORRECT - Load from gitignored config file

// .env file (MUST be in .gitignore)
// API_KEY=sk_live_...
// DATABASE_URL=postgres://...

config = load_dotenv(".env")
api_key = config.get("API_KEY")

// Provide .env.example (with placeholder values)
// API_KEY=your_api_key_here
// DATABASE_URL=postgres://user:pass@host/db
```

### C. Required .gitignore Entries

```gitignore
# Secrets and credentials - ALWAYS exclude
.env
.env.local
.env.*.local
*.pem
*.key
*.p12
*.pfx
credentials.json
secrets.json
service-account.json
*-credentials.json
*-secret.json
.secrets/
config/secrets/

# IDE and editor files that might contain secrets
.idea/
.vscode/settings.json

# OS files
.DS_Store
Thumbs.db
```

### D. Secret Rotation Protocol

```pseudocode
// When a secret is accidentally exposed:

1. IMMEDIATELY rotate/regenerate the secret
   - Generate new API key
   - Change password
   - Rotate encryption keys

2. Update secret in secure storage
   - Environment variables
   - Secret manager
   - Deployment configuration

3. Remove from version control history
   - Use git filter-branch or BFG Repo-Cleaner
   - Force push (coordinate with team)
   - All team members must re-clone

4. Audit for unauthorized access
   - Check access logs
   - Monitor for suspicious activity
   - Review API usage patterns

5. Document the incident
   - What was exposed
   - Duration of exposure
   - Actions taken
   - Prevention measures
```

---

## 4. Input Validation (MANDATORY)

### A. Validation Principles

**CRITICAL: Never trust any external input. Validate EVERYTHING.**

#### Input Sources to Validate

- User form submissions
- URL parameters and query strings
- HTTP headers (including cookies)
- API request bodies
- File uploads
- Database query results (for display)
- Third-party API responses
- Environment variables (at startup)
- Configuration files

### B. Validation Strategies

#### Whitelist Over Blacklist

```pseudocode
// ❌ BAD - Blacklist approach (attackers find bypasses)
FUNCTION validateUsername(username)
    blacklist = ["<", ">", "'", "\"", ";", "--"]
    FOR char IN blacklist
        IF username.contains(char)
            RETURN false
    RETURN true

// ✅ GOOD - Whitelist approach (only allow known-good)
FUNCTION validateUsername(username)
    // Only alphanumeric and underscore, 3-30 chars
    pattern = /^[a-zA-Z0-9_]{3,30}$/
    RETURN pattern.matches(username)
```

#### Type and Range Validation

```pseudocode
FUNCTION validateAge(input)
    // Type check
    IF NOT isInteger(input)
        THROW ValidationError("Age must be an integer")

    // Range check
    age = parseInt(input)
    IF age < 0 OR age > 150
        THROW ValidationError("Age must be between 0 and 150")

    RETURN age

FUNCTION validateEmail(input)
    // Type check
    IF NOT isString(input)
        THROW ValidationError("Email must be a string")

    // Length check
    IF input.length > 254
        THROW ValidationError("Email too long")

    // Format check (basic - consider library for production)
    pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
    IF NOT pattern.matches(input)
        THROW ValidationError("Invalid email format")

    RETURN input.toLowerCase()
```

#### Content Type Validation

```pseudocode
FUNCTION validateFileUpload(file)
    // Check file extension (not sufficient alone!)
    allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".pdf"]
    IF NOT file.extension IN allowed_extensions
        THROW ValidationError("File type not allowed")

    // Check MIME type
    allowed_mimes = ["image/jpeg", "image/png", "image/gif", "application/pdf"]
    IF NOT file.mime_type IN allowed_mimes
        THROW ValidationError("File type not allowed")

    // Check magic bytes (file signature)
    magic_bytes = file.read_first_bytes(8)
    IF NOT isValidMagicBytes(magic_bytes, file.mime_type)
        THROW ValidationError("File content doesn't match type")

    // Check file size
    IF file.size > MAX_FILE_SIZE
        THROW ValidationError("File too large")

    RETURN true
```

### C. Validation by Data Type

| Data Type | Validation Rules |
|-----------|------------------|
| String | Max length, allowed characters, format regex |
| Integer | Range (min/max), sign (positive/negative) |
| Float | Range, precision, NaN/Infinity handling |
| Email | Format, length ≤254, domain validation |
| URL | Protocol whitelist (https), domain whitelist |
| Date | Format, range, timezone handling |
| Phone | Format, length, country code validation |
| File | Extension, MIME type, magic bytes, size |
| JSON | Schema validation, depth limit |
| HTML | Sanitize with allowlist of tags/attributes |

---

## 5. Output Encoding (MANDATORY)

### A. Context-Specific Encoding

**CRITICAL: Output encoding MUST match the output context.**

#### HTML Context

```pseudocode
// ❌ DANGEROUS - XSS vulnerability
template = "<div>Hello, " + username + "</div>"

// ✅ SAFE - HTML entity encoding
template = "<div>Hello, " + htmlEncode(username) + "</div>"

// Encoding rules:
// & → &amp;
// < → &lt;
// > → &gt;
// " → &quot;
// ' → &#x27;
```

#### JavaScript Context

```pseudocode
// ❌ DANGEROUS - XSS vulnerability
script = "var name = '" + username + "';"

// ✅ SAFE - JavaScript encoding
script = "var name = '" + jsEncode(username) + "';"

// Better: Use JSON encoding
script = "var name = " + JSON.stringify(username) + ";"
```

#### URL Context

```pseudocode
// ❌ DANGEROUS - URL manipulation
url = "https://api.example.com/search?q=" + query

// ✅ SAFE - URL encoding
url = "https://api.example.com/search?q=" + urlEncode(query)
```

#### SQL Context (Parameterization Required)

```pseudocode
// ❌ DANGEROUS - SQL injection
query = "SELECT * FROM users WHERE name = '" + name + "'"

// ✅ SAFE - Parameterized query
query = "SELECT * FROM users WHERE name = ?"
result = db.execute(query, [name])

// ✅ SAFE - Named parameters
query = "SELECT * FROM users WHERE name = :name"
result = db.execute(query, { name: name })
```

#### Shell/Command Context

```pseudocode
// ❌ DANGEROUS - Command injection
command = "convert " + filename + " output.png"
shell.execute(command)

// ✅ SAFE - Avoid shell entirely, use array form
subprocess.run(["convert", filename, "output.png"])

// If shell is absolutely required, validate strictly
IF NOT filename.matches(/^[a-zA-Z0-9._-]+$/)
    THROW Error("Invalid filename")
```

### B. Template Engine Security

```pseudocode
// Use auto-escaping template engines

// ✅ SAFE - Auto-escaped (Jinja2, Django, React JSX, Go html/template)
template = "Hello, {{ username }}"  // Auto-escaped by default

// ⚠️ DANGEROUS - Raw output (only when absolutely necessary)
template = "{{ content|safe }}"  // Bypasses escaping - use with extreme caution

// When using raw output, MUST sanitize first
sanitized_content = sanitizeHTML(user_content, allowed_tags=["p", "br", "strong"])
template = "{{ sanitized_content|safe }}"
```

---

## 6. Authentication & Authorization (MANDATORY)

### A. Authentication Requirements

#### Password Handling

```pseudocode
// Password storage - ALWAYS use strong hashing
FUNCTION hashPassword(password)
    // ✅ CORRECT - Use bcrypt, scrypt, or Argon2
    // Argon2id is the current recommendation
    RETURN argon2id.hash(password,
        memory_cost: 65536,    // 64 MB
        time_cost: 3,          // 3 iterations
        parallelism: 4         // 4 parallel threads
    )

// ❌ NEVER use these for passwords:
// - Plain text storage
// - MD5, SHA1, SHA256 (even with salt)
// - Single iteration hashing
// - Encryption (reversible)

FUNCTION verifyPassword(password, hash)
    // Timing-safe comparison built into hash libraries
    RETURN argon2id.verify(hash, password)
```

#### Session Management

```pseudocode
// Session token generation
FUNCTION generateSessionToken()
    // Use cryptographically secure random number generator
    // ✅ CORRECT - 256 bits of entropy minimum
    RETURN secureRandom.bytes(32).toBase64URL()

// ❌ NEVER use:
// - Math.random() or similar non-cryptographic RNG
// - Predictable values (user ID, timestamp)
// - Sequential tokens

// Session storage
FUNCTION createSession(user_id)
    token = generateSessionToken()
    session = {
        token: token,
        user_id: user_id,
        created_at: now(),
        expires_at: now() + SESSION_DURATION,
        ip_address: request.ip,
        user_agent: request.user_agent
    }
    sessionStore.save(session)
    RETURN token

// Session validation
FUNCTION validateSession(token)
    session = sessionStore.get(token)
    IF session is null
        RETURN null
    IF session.expires_at < now()
        sessionStore.delete(token)
        RETURN null
    RETURN session.user_id
```

### B. Authorization Requirements

#### Access Control Checks

```pseudocode
// ✅ CORRECT - Check authorization on every request
FUNCTION getResource(request, resource_id)
    // 1. Verify authentication
    user = authenticateRequest(request)
    IF user is null
        RETURN Unauthorized("Authentication required")

    // 2. Load resource
    resource = resourceStore.get(resource_id)
    IF resource is null
        RETURN NotFound("Resource not found")

    // 3. Verify authorization
    IF NOT canAccess(user, resource, "read")
        RETURN Forbidden("Access denied")

    // 4. Return resource
    RETURN Success(resource)

// ❌ BAD - Missing authorization check
FUNCTION getResource_INSECURE(request, resource_id)
    resource = resourceStore.get(resource_id)  // IDOR vulnerability!
    RETURN Success(resource)
```

#### Role-Based Access Control (RBAC)

```pseudocode
// Define roles and permissions
PERMISSIONS = {
    "admin": ["read", "write", "delete", "admin"],
    "editor": ["read", "write"],
    "viewer": ["read"]
}

FUNCTION canAccess(user, resource, action)
    // Check user role
    user_permissions = PERMISSIONS.get(user.role, [])
    IF action NOT IN user_permissions
        RETURN false

    // Check resource-level permissions
    IF resource.owner_id == user.id
        RETURN true

    IF resource.shared_with.includes(user.id)
        RETURN action IN resource.shared_permissions[user.id]

    // Check organization-level access
    IF resource.organization_id == user.organization_id
        RETURN action IN user.organization_permissions

    RETURN false
```

---

## 7. Cryptography (MANDATORY)

### A. Approved Algorithms

| Purpose | Approved | Deprecated/Forbidden |
|---------|----------|---------------------|
| Password Hashing | Argon2id, bcrypt, scrypt | MD5, SHA1, SHA256 (plain) |
| Symmetric Encryption | AES-256-GCM, ChaCha20-Poly1305 | DES, 3DES, RC4, Blowfish, AES-ECB |
| Asymmetric Encryption | RSA (≥2048 bit), ECDSA (P-256+), Ed25519 | RSA (<2048), DSA |
| Hashing (non-password) | SHA-256, SHA-384, SHA-512, SHA-3, BLAKE2 | MD5, SHA1 |
| TLS Version | TLS 1.2, TLS 1.3 | SSL, TLS 1.0, TLS 1.1 |
| Random Generation | OS CSPRNG (urandom, CryptGenRandom) | Math.random(), rand() |

### B. Cryptographic Code Patterns

#### Secure Random Generation

```pseudocode
// ✅ CORRECT - Use cryptographic random
token = crypto.randomBytes(32)
uuid = crypto.randomUUID()

// ❌ NEVER use for security purposes
token = Math.random()  // Predictable!
token = rand()         // Predictable!
token = time()         // Predictable!
```

#### Encryption

```pseudocode
// ✅ CORRECT - AES-GCM with proper IV
FUNCTION encrypt(plaintext, key)
    // Generate unique IV for each encryption
    iv = crypto.randomBytes(12)  // 96 bits for GCM

    cipher = AES_GCM(key, iv)
    ciphertext = cipher.encrypt(plaintext)
    auth_tag = cipher.getAuthTag()

    // Return IV + auth tag + ciphertext
    RETURN iv + auth_tag + ciphertext

FUNCTION decrypt(encrypted, key)
    iv = encrypted[0:12]
    auth_tag = encrypted[12:28]
    ciphertext = encrypted[28:]

    cipher = AES_GCM(key, iv)
    cipher.setAuthTag(auth_tag)

    TRY
        plaintext = cipher.decrypt(ciphertext)
        RETURN plaintext
    CATCH AuthenticationError
        THROW Error("Decryption failed - data tampered")

// ❌ NEVER do this
cipher = AES_ECB(key)  // ECB mode is insecure
cipher = AES_CBC(key, static_iv)  // Static IV is insecure
```

#### Key Management

```pseudocode
// ✅ CORRECT - Key derivation from password
FUNCTION deriveKey(password, salt)
    // Use PBKDF2, scrypt, or Argon2
    RETURN scrypt(
        password: password,
        salt: salt,
        N: 2^17,      // CPU/memory cost
        r: 8,         // Block size
        p: 1,         // Parallelization
        key_length: 32
    )

// ✅ CORRECT - Generate cryptographic keys
FUNCTION generateEncryptionKey()
    // Generate directly from secure random
    RETURN crypto.randomBytes(32)  // 256 bits

// ❌ NEVER do this
key = "mysecretkey"  // Hardcoded!
key = md5(password)  // Weak derivation!
key = password       // Password is not a key!
```

---

## 8. SQL Injection Prevention (MANDATORY)

### A. Parameterized Queries Only

```pseudocode
// ✅ CORRECT - Parameterized queries (ALL languages)

// Positional parameters
query = "SELECT * FROM users WHERE id = ? AND status = ?"
result = db.execute(query, [user_id, status])

// Named parameters
query = "SELECT * FROM users WHERE id = :id AND status = :status"
result = db.execute(query, { id: user_id, status: status })

// ❌ NEVER concatenate user input
query = "SELECT * FROM users WHERE id = " + user_id  // SQL INJECTION!
query = f"SELECT * FROM users WHERE id = {user_id}"  // SQL INJECTION!
query = `SELECT * FROM users WHERE id = ${user_id}`  // SQL INJECTION!
```

### B. ORM Security

```pseudocode
// ✅ CORRECT - ORM with safe queries
user = User.find_by(id: user_id)
users = User.where(status: "active").order(created_at: :desc)

// ⚠️ DANGEROUS - Raw SQL in ORM
User.where("name = '#{params[:name]}'")  // SQL INJECTION!

// ✅ CORRECT - Raw SQL with parameters when needed
User.where("name = ?", params[:name])
```

### C. Dynamic Query Building

```pseudocode
// When dynamic queries are necessary, use builders

// ✅ CORRECT - Query builder with parameterization
FUNCTION searchUsers(criteria)
    query = QueryBuilder.select("*").from("users")
    params = []

    IF criteria.name
        query.where("name LIKE ?")
        params.append("%" + escapeLikePattern(criteria.name) + "%")

    IF criteria.status
        query.where("status = ?")
        params.append(criteria.status)

    IF criteria.sort_by IN ["name", "created_at", "email"]  // Whitelist!
        query.orderBy(criteria.sort_by)

    RETURN db.execute(query.build(), params)

// ❌ NEVER do this
query = "SELECT * FROM users WHERE 1=1"
IF criteria.name
    query += " AND name = '" + criteria.name + "'"  // SQL INJECTION!
```

---

## 9. XSS Prevention (MANDATORY)

### A. Output Encoding Rules

```pseudocode
// Context-aware encoding

// HTML body
<div>{{ htmlEncode(userInput) }}</div>

// HTML attribute (double-quoted)
<input value="{{ htmlEncode(userInput) }}">

// JavaScript string
<script>var name = "{{ jsEncode(userInput) }}";</script>

// JSON in HTML
<script>var data = {{ jsonEncode(userData) }};</script>

// URL parameter
<a href="/search?q={{ urlEncode(query) }}">Search</a>

// CSS value (avoid if possible)
<div style="color: {{ cssEncode(color) }}"></div>
```

### B. Content Security Policy (CSP)

```pseudocode
// Implement strict CSP headers

// ✅ CORRECT - Strict CSP
Content-Security-Policy:
    default-src 'self';
    script-src 'self' 'nonce-{random}';
    style-src 'self' 'nonce-{random}';
    img-src 'self' data: https:;
    font-src 'self';
    object-src 'none';
    base-uri 'self';
    form-action 'self';
    frame-ancestors 'none';
    upgrade-insecure-requests;

// Generate nonces per request
FUNCTION generateCSPNonce()
    RETURN base64(crypto.randomBytes(16))

// Use nonce in templates
<script nonce="{{ csp_nonce }}">
    // Inline script allowed with matching nonce
</script>
```

### C. HTML Sanitization

```pseudocode
// When you MUST allow some HTML (rich text)

// ✅ CORRECT - Whitelist-based sanitization
FUNCTION sanitizeHTML(input)
    allowed_tags = ["p", "br", "strong", "em", "ul", "ol", "li", "a"]
    allowed_attributes = {
        "a": ["href", "title"],
        "*": ["class"]  // Global attributes
    }

    // Use established sanitization library
    RETURN HTMLSanitizer.sanitize(input, {
        allowed_tags: allowed_tags,
        allowed_attributes: allowed_attributes,
        protocols: { "a": { "href": ["http", "https", "mailto"] } }
    })

// ❌ NEVER do this
output = input.replace("<script>", "")  // Easily bypassed!
```

---

## 10. Security Headers (MANDATORY)

### A. Required HTTP Headers

```pseudocode
// Set on ALL responses

headers = {
    // Prevent MIME type sniffing
    "X-Content-Type-Options": "nosniff",

    // XSS protection (legacy browsers)
    "X-XSS-Protection": "1; mode=block",

    // Prevent clickjacking
    "X-Frame-Options": "DENY",

    // HTTPS enforcement
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",

    // Content Security Policy
    "Content-Security-Policy": "default-src 'self'; script-src 'self'",

    // Referrer information control
    "Referrer-Policy": "strict-origin-when-cross-origin",

    // Permissions/features policy
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}
```

### B. Cookie Security

```pseudocode
// ✅ CORRECT - Secure cookie settings
setCookie("session", token, {
    httpOnly: true,     // Prevent XSS access
    secure: true,       // HTTPS only
    sameSite: "Strict", // CSRF protection
    path: "/",
    maxAge: 3600,       // 1 hour
    domain: ".example.com"
})

// ❌ NEVER do this
setCookie("session", token)  // Missing security flags!
setCookie("session", token, { httpOnly: false })  // XSS vulnerable!
setCookie("session", token, { secure: false })  // Transmitted over HTTP!
```

---

## 11. Logging & Error Handling (MANDATORY)

### A. Secure Logging

```pseudocode
// ✅ CORRECT - Log security events without sensitive data
FUNCTION logSecurityEvent(event)
    log.info({
        event_type: event.type,
        timestamp: now(),
        user_id: event.user_id,  // OK - identifier only
        ip_address: event.ip,
        action: event.action,
        resource: event.resource_type,
        resource_id: event.resource_id,
        outcome: event.success ? "success" : "failure"
    })

// ❌ NEVER log sensitive data
log.info("User login: " + username + " password: " + password)  // NEVER!
log.info("API call with key: " + api_key)  // NEVER!
log.info("Credit card: " + card_number)  // NEVER!
log.info("Session token: " + session_token)  // NEVER!
log.debug("Request body: " + JSON.stringify(request.body))  // May contain secrets!
```

### B. Secure Error Handling

```pseudocode
// ✅ CORRECT - Don't expose internal details
FUNCTION handleError(error, request)
    // Log full details internally
    log.error({
        error_id: generateErrorId(),
        message: error.message,
        stack: error.stack,
        request_id: request.id,
        user_id: request.user_id
    })

    // Return generic message to user
    IF isProduction()
        RETURN {
            error: "An unexpected error occurred",
            error_id: error_id,
            message: "Please contact support with this error ID"
        }
    ELSE
        // More detail in development (but still no secrets!)
        RETURN {
            error: error.name,
            message: error.message
        }

// ❌ NEVER expose stack traces or internal details
RETURN { error: error.stack }  // Reveals code structure!
RETURN { error: "Database connection failed: " + db_error }  // Reveals infrastructure!
RETURN { error: "SQL error: " + sql_error }  // Reveals query details!
```

---

## 12. File Upload Security (MANDATORY)

### A. File Upload Validation

```pseudocode
FUNCTION handleFileUpload(file)
    // 1. Validate file size
    IF file.size > MAX_FILE_SIZE
        THROW Error("File too large")

    // 2. Validate extension (whitelist)
    allowed_extensions = [".jpg", ".jpeg", ".png", ".pdf"]
    extension = file.name.toLowerCase().extractExtension()
    IF extension NOT IN allowed_extensions
        THROW Error("File type not allowed")

    // 3. Validate MIME type
    allowed_mimes = ["image/jpeg", "image/png", "application/pdf"]
    IF file.mime_type NOT IN allowed_mimes
        THROW Error("File type not allowed")

    // 4. Validate magic bytes (file signature)
    IF NOT validateMagicBytes(file)
        THROW Error("File content doesn't match type")

    // 5. Generate safe filename (NEVER use user-provided name directly)
    safe_filename = generateUUID() + extension

    // 6. Store outside web root or use separate domain
    storage_path = SECURE_UPLOAD_DIR + "/" + safe_filename

    // 7. Set restrictive permissions
    file.saveTo(storage_path)
    setPermissions(storage_path, 0644)

    RETURN safe_filename
```

### B. Path Traversal Prevention

```pseudocode
// ✅ CORRECT - Prevent path traversal
FUNCTION getFile(filename)
    // Remove path components
    safe_name = basename(filename)

    // Validate filename characters
    IF NOT safe_name.matches(/^[a-zA-Z0-9._-]+$/)
        THROW Error("Invalid filename")

    // Construct path and verify it's within allowed directory
    full_path = UPLOAD_DIR.resolve(safe_name).normalize()

    IF NOT full_path.startsWith(UPLOAD_DIR)
        THROW Error("Access denied")

    RETURN readFile(full_path)

// ❌ DANGEROUS - Path traversal vulnerability
FUNCTION getFile_INSECURE(filename)
    RETURN readFile(UPLOAD_DIR + "/" + filename)  // "../../../etc/passwd" attack!
```

---

## 13. API Security (MANDATORY)

### A. Rate Limiting

```pseudocode
// Implement rate limiting for all APIs
rate_limiter = RateLimiter({
    // General API rate limit
    default: { requests: 100, window: "1m" },

    // Stricter limits for sensitive endpoints
    auth: { requests: 5, window: "1m" },
    password_reset: { requests: 3, window: "1h" },
    signup: { requests: 10, window: "1h" },

    // Higher limits for authenticated users
    authenticated: { requests: 1000, window: "1m" }
})

FUNCTION handleRequest(request)
    key = request.ip + ":" + request.endpoint

    IF NOT rate_limiter.allow(key)
        RETURN TooManyRequests({
            error: "Rate limit exceeded",
            retry_after: rate_limiter.getRetryAfter(key)
        })

    // Process request...
```

### B. API Authentication

```pseudocode
// ✅ CORRECT - Secure API authentication

// Bearer token validation
FUNCTION authenticateAPI(request)
    auth_header = request.headers["Authorization"]

    IF auth_header is null OR NOT auth_header.startsWith("Bearer ")
        RETURN Unauthorized("Missing or invalid authorization header")

    token = auth_header.substring(7)

    TRY
        // Validate JWT or lookup API key
        user = validateToken(token)
        RETURN user
    CATCH TokenExpiredError
        RETURN Unauthorized("Token expired")
    CATCH InvalidTokenError
        RETURN Unauthorized("Invalid token")

// API key storage - hash API keys (they're like passwords)
FUNCTION createAPIKey(user_id)
    key = generateSecureToken(32)
    key_hash = sha256(key)  // Store hash, not plain key

    apiKeyStore.save({
        key_hash: key_hash,
        user_id: user_id,
        created_at: now()
    })

    // Return plain key only once (user must save it)
    RETURN key

FUNCTION validateAPIKey(key)
    key_hash = sha256(key)
    record = apiKeyStore.findByHash(key_hash)
    RETURN record?.user_id
```

---

## 14. Dependency Security (MANDATORY)

### A. Dependency Management

```pseudocode
// Regularly audit and update dependencies

// Run vulnerability scans
// Examples: npm audit, pip audit, cargo audit, go list -m all

// Lock dependency versions
// Use lock files: package-lock.json, Pipfile.lock, Cargo.lock, go.sum

// Automate dependency updates
// Use Dependabot, Renovate, or similar

// Review before updating
FUNCTION updateDependency(package, new_version)
    // 1. Check security advisories
    IF hasSecurityAdvisory(package, new_version)
        log.warn("Security advisory exists for " + package + "@" + new_version)

    // 2. Check for breaking changes
    changelog = getChangelog(package, new_version)

    // 3. Run tests
    IF NOT runTests()
        THROW Error("Tests failed after update")

    // 4. Update lock file
    updateLockFile(package, new_version)
```

### B. Banned Dependencies

```pseudocode
// Maintain list of banned/deprecated packages

BANNED_PACKAGES = [
    "event-stream",        // Known compromise
    "ua-parser-js",        // Multiple compromises
    "coa",                 // Compromised
    "rc",                  // Compromised
    // Add project-specific banned packages
]

FUNCTION checkDependencies()
    FOR dependency IN getAllDependencies()
        IF dependency.name IN BANNED_PACKAGES
            THROW Error("Banned dependency detected: " + dependency.name)
```

---

## 15. Security Testing (MANDATORY)

### A. Security Test Types

```pseudocode
// Security unit tests
TEST "SQL injection is prevented"
    malicious_input = "'; DROP TABLE users; --"

    // Should not throw SQL error (would indicate injection)
    result = userService.search(malicious_input)

    ASSERT result.length == 0
    ASSERT database.tableExists("users")
END TEST

TEST "XSS is prevented"
    malicious_input = "<script>alert('xss')</script>"

    output = templateEngine.render("user", { name: malicious_input })

    ASSERT output DOES_NOT_CONTAIN "<script>"
    ASSERT output CONTAINS "&lt;script&gt;"
END TEST

TEST "authentication required for protected endpoint"
    // Without authentication
    response = api.get("/api/admin/users")
    ASSERT response.status == 401

    // With invalid token
    response = api.get("/api/admin/users", headers: { "Authorization": "Bearer invalid" })
    ASSERT response.status == 401
END TEST

TEST "authorization prevents access to other users' data"
    user1_token = login("user1")
    user2_resource = createResource(owner: "user2")

    response = api.get("/api/resources/" + user2_resource.id,
                       headers: { "Authorization": "Bearer " + user1_token })

    ASSERT response.status == 403
END TEST

TEST "rate limiting prevents brute force"
    FOR i FROM 1 TO 10
        response = api.post("/api/login", { email: "test@test.com", password: "wrong" })
    END FOR

    response = api.post("/api/login", { email: "test@test.com", password: "wrong" })

    ASSERT response.status == 429
    ASSERT response.headers["Retry-After"] IS_NOT null
END TEST
```

### B. Security Scanning Integration

```yaml
# CI/CD pipeline security checks

security_pipeline:
  stages:
    - secret_scan
    - dependency_scan
    - static_analysis
    - dynamic_analysis

  secret_scan:
    - name: "Scan for secrets"
      command: "gitleaks detect --source . --verbose"
      fail_on_detection: true

  dependency_scan:
    - name: "Scan dependencies"
      command: "npm audit --audit-level=high"  # or equivalent
      fail_on_vulnerabilities: true

  static_analysis:
    - name: "SAST scan"
      command: "semgrep --config=auto ."
      fail_on_findings: true

  dynamic_analysis:
    - name: "DAST scan"
      command: "zap-full-scan.py -t https://staging.example.com"
      allow_failures: true  # Review results manually
```

---

## 16. Deployment Checklist

### Security Verification (MANDATORY)

**If code was generated/modified, verify BEFORE delivery:**

#### Credentials & Secrets
- [ ] No hardcoded credentials (API keys, passwords, tokens)
- [ ] Secrets loaded from environment variables or secret manager
- [ ] .gitignore includes all secret files
- [ ] No secrets in version control history

#### Input Validation
- [ ] All user input validated
- [ ] All API parameters validated
- [ ] File uploads validated (type, size, content)
- [ ] Whitelist validation used (not blacklist)

#### Output Encoding
- [ ] HTML output encoded
- [ ] JavaScript context properly escaped
- [ ] SQL uses parameterized queries
- [ ] No command injection possible

#### Authentication & Authorization
- [ ] Authentication required on all protected endpoints
- [ ] Authorization checks present
- [ ] Passwords hashed with strong algorithm
- [ ] Sessions managed securely

#### Cryptography
- [ ] Only approved algorithms used
- [ ] Cryptographic random used for security
- [ ] Keys properly managed (not hardcoded)

#### Security Headers
- [ ] CSP configured
- [ ] HSTS enabled
- [ ] X-Frame-Options set
- [ ] Cookies have security flags

#### Logging & Errors
- [ ] Security events logged
- [ ] No sensitive data in logs
- [ ] Errors don't expose internals
- [ ] Stack traces not shown in production

#### Dependencies
- [ ] Dependencies up to date
- [ ] No known vulnerabilities
- [ ] Lock files committed

#### Testing
- [ ] Security tests pass
- [ ] SAST scan clean
- [ ] Secret scan clean

---

## 17. Quick Reference

### Common Security Commands

```bash
# Secret scanning
gitleaks detect --source .
trufflehog filesystem .
detect-secrets scan

# Dependency scanning (language-specific)
npm audit
pip audit
cargo audit
go list -m all | nancy sleuth
bundle audit

# Static analysis
semgrep --config=auto .
bandit -r . (Python)
gosec ./... (Go)
```

### Security Headers Template

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'; script-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

### Secure Cookie Template

```
Set-Cookie: session=TOKEN; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600
```

---

## 18. Why This Configuration Works

**Defense in Depth**:
- Multiple layers of security prevent single points of failure
- If one control fails, others still protect the system

**Fail Secure**:
- Default deny approach prevents security bypasses
- Errors don't expose sensitive information

**Verified Security**:
- Automated scanning catches common vulnerabilities
- Security tests prevent regressions

**Credential Protection**:
- Never storing secrets in code prevents the most common exposure vector
- Secret management systems provide proper access control and rotation

**Input Validation + Output Encoding**:
- Validates input at entry points
- Encodes output at exit points
- Combined approach prevents injection attacks

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Most critical web application security risks
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/) - Security best practices
- [CWE Top 25](https://cwe.mitre.org/top25/) - Most dangerous software weaknesses
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Mozilla Web Security Guidelines](https://infosec.mozilla.org/guidelines/web_security)
- [Google Security Best Practices](https://cloud.google.com/security/best-practices)

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Security Team

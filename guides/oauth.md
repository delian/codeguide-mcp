# OAuth 2.0 & Authentication Guidelines

This document provides mandatory standards for implementing OAuth 2.0 and secure authentication systems.

---

**Agent Profile**: The OAuth Expert
**Role**: Senior Security Engineer & Identity Architect
**Objective**: Generate secure, standards-compliant OAuth implementations with proper token handling.
**Tools**: OAuth 2.0, OpenID Connect, JWT, PKCE, Auth0, Keycloak.

---

## 1. Core Philosophies: AUTH-FIRST

- **A**uthorization: Separate authentication from authorization
- **U**ser-centric: User controls data access through consent
- **T**oken-based: Stateless authentication with short-lived tokens
- **H**ardened: Defense in depth with multiple security layers

---

## 2. OAuth 2.0 Flows (MANDATORY)

### A. Authorization Code Flow with PKCE (Recommended)

```javascript
// Client-side: Generate PKCE challenge
async function generatePKCE() {
  // Generate code verifier (43-128 characters)
  const codeVerifier = generateRandomString(64);

  // Generate code challenge (SHA256 hash of verifier, base64url encoded)
  const encoder = new TextEncoder();
  const data = encoder.encode(codeVerifier);
  const hash = await crypto.subtle.digest('SHA-256', data);
  const codeChallenge = base64urlEncode(hash);

  return { codeVerifier, codeChallenge };
}

// Step 1: Redirect to authorization server
async function startAuthFlow() {
  const { codeVerifier, codeChallenge } = await generatePKCE();
  const state = generateRandomString(32);

  // Store for later verification
  sessionStorage.setItem('oauth_code_verifier', codeVerifier);
  sessionStorage.setItem('oauth_state', state);

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    scope: 'openid profile email',
    state: state,
    code_challenge: codeChallenge,
    code_challenge_method: 'S256'
  });

  window.location.href = `${AUTH_SERVER}/authorize?${params}`;
}

// Step 2: Handle callback and exchange code for tokens
async function handleCallback(callbackUrl) {
  const params = new URL(callbackUrl).searchParams;
  const code = params.get('code');
  const state = params.get('state');
  const error = params.get('error');

  // Check for errors
  if (error) {
    throw new Error(`OAuth error: ${error}`);
  }

  // Verify state
  const savedState = sessionStorage.getItem('oauth_state');
  if (state !== savedState) {
    throw new Error('State mismatch - possible CSRF attack');
  }

  // Get code verifier
  const codeVerifier = sessionStorage.getItem('oauth_code_verifier');

  // Exchange code for tokens
  const response = await fetch(`${AUTH_SERVER}/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code: code,
      redirect_uri: REDIRECT_URI,
      client_id: CLIENT_ID,
      code_verifier: codeVerifier
    })
  });

  if (!response.ok) {
    throw new Error('Token exchange failed');
  }

  const tokens = await response.json();

  // Clean up
  sessionStorage.removeItem('oauth_code_verifier');
  sessionStorage.removeItem('oauth_state');

  return tokens;
}
```

### B. Client Credentials Flow (Machine-to-Machine)

```javascript
// Server-side only - never expose client secret
async function getM2MToken() {
  const response = await fetch(`${AUTH_SERVER}/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Authorization': `Basic ${Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString('base64')}`
    },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      scope: 'api:read api:write'
    })
  });

  if (!response.ok) {
    throw new Error('Failed to obtain token');
  }

  return response.json();
}
```

### C. Refresh Token Flow

```javascript
class TokenManager {
  constructor(config) {
    this.config = config;
    this.tokens = null;
    this.refreshPromise = null;
  }

  async getAccessToken() {
    if (!this.tokens) {
      throw new Error('Not authenticated');
    }

    // Check if token is expired or about to expire (5 min buffer)
    if (this.isTokenExpired(this.tokens.access_token, 300)) {
      // Prevent multiple simultaneous refresh requests
      if (!this.refreshPromise) {
        this.refreshPromise = this.refreshTokens();
      }
      await this.refreshPromise;
      this.refreshPromise = null;
    }

    return this.tokens.access_token;
  }

  async refreshTokens() {
    if (!this.tokens?.refresh_token) {
      throw new Error('No refresh token available');
    }

    try {
      const response = await fetch(`${this.config.authServer}/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
          grant_type: 'refresh_token',
          refresh_token: this.tokens.refresh_token,
          client_id: this.config.clientId
        })
      });

      if (!response.ok) {
        // Refresh token expired or revoked
        this.tokens = null;
        throw new Error('Token refresh failed');
      }

      this.tokens = await response.json();
      this.onTokensRefreshed?.(this.tokens);

      return this.tokens;
    } catch (error) {
      // Clear tokens on refresh failure
      this.tokens = null;
      throw error;
    }
  }

  isTokenExpired(token, bufferSeconds = 0) {
    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      const expiresAt = payload.exp * 1000;
      return Date.now() >= expiresAt - (bufferSeconds * 1000);
    } catch {
      return true;
    }
  }
}
```

---

## 3. JWT Handling (MANDATORY)

### A. Token Validation (Server-Side)

```javascript
const jwt = require('jsonwebtoken');
const jwksClient = require('jwks-rsa');

// JWKS client for fetching public keys
const client = jwksClient({
  jwksUri: `${AUTH_SERVER}/.well-known/jwks.json`,
  cache: true,
  cacheMaxAge: 600000, // 10 minutes
  rateLimit: true
});

function getSigningKey(header, callback) {
  client.getSigningKey(header.kid, (err, key) => {
    if (err) {
      callback(err);
      return;
    }
    callback(null, key.getPublicKey());
  });
}

async function validateToken(token) {
  return new Promise((resolve, reject) => {
    jwt.verify(
      token,
      getSigningKey,
      {
        algorithms: ['RS256'],
        audience: API_AUDIENCE,
        issuer: AUTH_SERVER,
        clockTolerance: 30 // 30 seconds tolerance for clock skew
      },
      (err, decoded) => {
        if (err) {
          reject(err);
        } else {
          resolve(decoded);
        }
      }
    );
  });
}

// Express middleware
async function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or invalid authorization header' });
  }

  const token = authHeader.split(' ')[1];

  try {
    const decoded = await validateToken(token);

    // Add user info to request
    req.user = {
      id: decoded.sub,
      email: decoded.email,
      roles: decoded['https://myapp.com/roles'] || [],
      scopes: decoded.scope?.split(' ') || []
    };

    next();
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired' });
    }
    if (error.name === 'JsonWebTokenError') {
      return res.status(401).json({ error: 'Invalid token' });
    }
    console.error('Token validation error:', error);
    return res.status(500).json({ error: 'Authentication failed' });
  }
}
```

### B. Token Structure

```javascript
// Access token claims (example)
const accessTokenPayload = {
  // Standard claims
  iss: 'https://auth.example.com/',        // Issuer
  sub: 'user_123',                          // Subject (user ID)
  aud: ['https://api.example.com'],         // Audience
  exp: 1705320600,                          // Expiration (Unix timestamp)
  iat: 1705317000,                          // Issued at
  nbf: 1705317000,                          // Not before

  // OpenID Connect claims
  azp: 'client_app_id',                     // Authorized party
  scope: 'openid profile email read:data',

  // Custom claims (namespaced)
  'https://myapp.com/roles': ['user', 'admin'],
  'https://myapp.com/org_id': 'org_456'
};

// ID token claims (OpenID Connect)
const idTokenPayload = {
  iss: 'https://auth.example.com/',
  sub: 'user_123',
  aud: 'client_app_id',
  exp: 1705320600,
  iat: 1705317000,
  nonce: 'abc123',                          // For replay protection

  // User info
  name: 'John Doe',
  email: 'john@example.com',
  email_verified: true,
  picture: 'https://example.com/photo.jpg'
};
```

---

## 4. Token Storage (MANDATORY)

### A. Browser Storage

```javascript
// ❌ WRONG: Never store tokens in localStorage (XSS vulnerable)
localStorage.setItem('access_token', token);

// ❌ WRONG: Never store tokens in sessionStorage for sensitive apps
sessionStorage.setItem('access_token', token);

// ✅ CORRECT: Use HTTP-only cookies for refresh tokens
// Set by server with these flags:
// Set-Cookie: refresh_token=xxx; HttpOnly; Secure; SameSite=Strict; Path=/auth/refresh

// ✅ CORRECT: Keep access tokens in memory only
class SecureTokenStore {
  #accessToken = null;

  setAccessToken(token) {
    this.#accessToken = token;
  }

  getAccessToken() {
    return this.#accessToken;
  }

  clear() {
    this.#accessToken = null;
  }
}

// ✅ CORRECT: For SPAs, use token handler pattern
class TokenHandler {
  constructor() {
    this.accessToken = null;
  }

  async getAccessToken() {
    if (this.accessToken && !this.isExpired(this.accessToken)) {
      return this.accessToken;
    }

    // Refresh using HTTP-only cookie
    const response = await fetch('/auth/refresh', {
      method: 'POST',
      credentials: 'include' // Include cookies
    });

    if (!response.ok) {
      throw new Error('Session expired');
    }

    const { access_token } = await response.json();
    this.accessToken = access_token;
    return this.accessToken;
  }
}
```

### B. Mobile Storage

```swift
// iOS: Use Keychain
import Security

class TokenStorage {
    private let service = "com.myapp.auth"

    func save(token: String, key: String) throws {
        let data = token.data(using: .utf8)!

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        // Delete existing
        SecItemDelete(query as CFDictionary)

        // Add new
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw TokenError.saveFailed
        }
    }

    func get(key: String) throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let token = String(data: data, encoding: .utf8) else {
            return nil
        }

        return token
    }
}
```

---

## 5. Authorization (MANDATORY)

### A. Scope-Based Access Control

```javascript
// Define scopes
const SCOPES = {
  'read:users': 'Read user profiles',
  'write:users': 'Create and update users',
  'delete:users': 'Delete users',
  'read:orders': 'Read orders',
  'write:orders': 'Create and update orders',
  'admin': 'Full administrative access'
};

// Middleware to check scopes
function requireScopes(...requiredScopes) {
  return (req, res, next) => {
    const userScopes = req.user.scopes;

    // Admin has all access
    if (userScopes.includes('admin')) {
      return next();
    }

    const hasAllScopes = requiredScopes.every(scope =>
      userScopes.includes(scope)
    );

    if (!hasAllScopes) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        required: requiredScopes,
        granted: userScopes
      });
    }

    next();
  };
}

// Usage
app.get('/api/users', authMiddleware, requireScopes('read:users'), listUsers);
app.post('/api/users', authMiddleware, requireScopes('write:users'), createUser);
app.delete('/api/users/:id', authMiddleware, requireScopes('delete:users'), deleteUser);
```

### B. Role-Based Access Control (RBAC)

```javascript
// Define roles and permissions
const ROLES = {
  admin: {
    permissions: ['*'], // All permissions
    inherits: []
  },
  manager: {
    permissions: [
      'users:read',
      'users:write',
      'orders:read',
      'orders:write',
      'reports:read'
    ],
    inherits: ['user']
  },
  user: {
    permissions: [
      'profile:read',
      'profile:write',
      'orders:read',
      'orders:create'
    ],
    inherits: []
  }
};

// Get all permissions for a role (including inherited)
function getRolePermissions(role, visited = new Set()) {
  if (visited.has(role)) return new Set(); // Prevent cycles
  visited.add(role);

  const roleConfig = ROLES[role];
  if (!roleConfig) return new Set();

  const permissions = new Set(roleConfig.permissions);

  for (const inheritedRole of roleConfig.inherits) {
    const inheritedPerms = getRolePermissions(inheritedRole, visited);
    inheritedPerms.forEach(p => permissions.add(p));
  }

  return permissions;
}

// Check permission
function hasPermission(user, permission) {
  for (const role of user.roles) {
    const permissions = getRolePermissions(role);

    if (permissions.has('*') || permissions.has(permission)) {
      return true;
    }

    // Check wildcard patterns (e.g., 'users:*')
    const [resource, action] = permission.split(':');
    if (permissions.has(`${resource}:*`)) {
      return true;
    }
  }

  return false;
}

// Middleware
function requirePermission(permission) {
  return (req, res, next) => {
    if (!hasPermission(req.user, permission)) {
      return res.status(403).json({
        error: 'Access denied',
        required: permission
      });
    }
    next();
  };
}
```

---

## 6. Password Handling (Cross-Reference)

**CRITICAL: When implementing password-based authentication alongside OAuth, use proper password hashing.**

For applications that support both OAuth and password-based login:

```javascript
// Password hashing - Use Argon2id (recommended)
// See secure-coding.md Section 6.A for complete implementation

const argon2 = require('argon2');

// ✅ CORRECT - Hash password with Argon2id
async function hashPassword(password) {
  return argon2.hash(password, {
    type: argon2.argon2id,
    memoryCost: 65536,    // 64 MB
    timeCost: 3,          // 3 iterations
    parallelism: 4        // 4 parallel threads
  });
}

// ✅ CORRECT - Verify password
async function verifyPassword(password, hash) {
  return argon2.verify(hash, password);
}

// ❌ NEVER use for passwords:
// - Plain text storage
// - MD5, SHA1, SHA256 (even with salt)
// - Single iteration hashing
// - Encryption (reversible)
```

**Full guidance:** See secure-coding.md Section 6.A for complete password handling requirements.

---

## 7. OpenID Connect (MANDATORY)

### A. Discovery Document

```javascript
// Fetch OIDC configuration
async function getOIDCConfig(issuer) {
  const response = await fetch(`${issuer}/.well-known/openid-configuration`);
  return response.json();
}

// Example discovery document
const oidcConfig = {
  issuer: 'https://auth.example.com',
  authorization_endpoint: 'https://auth.example.com/authorize',
  token_endpoint: 'https://auth.example.com/token',
  userinfo_endpoint: 'https://auth.example.com/userinfo',
  jwks_uri: 'https://auth.example.com/.well-known/jwks.json',
  revocation_endpoint: 'https://auth.example.com/revoke',
  end_session_endpoint: 'https://auth.example.com/logout',
  scopes_supported: ['openid', 'profile', 'email', 'offline_access'],
  response_types_supported: ['code', 'token', 'id_token'],
  grant_types_supported: ['authorization_code', 'refresh_token', 'client_credentials'],
  code_challenge_methods_supported: ['S256'],
  token_endpoint_auth_methods_supported: ['client_secret_basic', 'client_secret_post']
};
```

### B. UserInfo Endpoint

```javascript
async function getUserInfo(accessToken) {
  const response = await fetch(`${AUTH_SERVER}/userinfo`, {
    headers: {
      'Authorization': `Bearer ${accessToken}`
    }
  });

  if (!response.ok) {
    throw new Error('Failed to fetch user info');
  }

  return response.json();
}

// Response
const userInfo = {
  sub: 'user_123',
  name: 'John Doe',
  given_name: 'John',
  family_name: 'Doe',
  email: 'john@example.com',
  email_verified: true,
  picture: 'https://example.com/photo.jpg',
  locale: 'en-US',
  updated_at: 1705317000
};
```

---

## 7. Security Best Practices (MANDATORY)

### A. Token Security

```javascript
// ✅ Use short-lived access tokens
const ACCESS_TOKEN_EXPIRY = '15m';  // 15 minutes
const REFRESH_TOKEN_EXPIRY = '7d';  // 7 days
const ID_TOKEN_EXPIRY = '1h';       // 1 hour

// ✅ Rotate refresh tokens on use
async function refreshAccessToken(refreshToken) {
  // Verify refresh token
  const payload = await verifyRefreshToken(refreshToken);

  // Revoke old refresh token
  await revokeToken(refreshToken);

  // Issue new tokens
  const newAccessToken = await issueAccessToken(payload.sub);
  const newRefreshToken = await issueRefreshToken(payload.sub);

  return { accessToken: newAccessToken, refreshToken: newRefreshToken };
}

// ✅ Implement token revocation
const revokedTokens = new Set(); // Use Redis in production

async function revokeToken(token) {
  const payload = jwt.decode(token);
  revokedTokens.add(payload.jti); // JWT ID
}

function isTokenRevoked(token) {
  const payload = jwt.decode(token);
  return revokedTokens.has(payload.jti);
}

// ✅ Always verify token before use
async function validateToken(token) {
  // Check if revoked
  if (isTokenRevoked(token)) {
    throw new Error('Token has been revoked');
  }

  // Verify signature and claims
  return jwt.verify(token, publicKey, {
    algorithms: ['RS256'],
    audience: API_AUDIENCE,
    issuer: AUTH_SERVER
  });
}
```

### B. CSRF Protection

```javascript
// For authorization requests
function startAuth() {
  const state = crypto.randomBytes(32).toString('hex');

  // Store state with timestamp
  sessionStorage.setItem('oauth_state', JSON.stringify({
    value: state,
    timestamp: Date.now()
  }));

  // Include in authorization request
  return state;
}

function verifyState(returnedState) {
  const stored = JSON.parse(sessionStorage.getItem('oauth_state'));

  if (!stored) {
    throw new Error('No state found');
  }

  // Check state matches
  if (stored.value !== returnedState) {
    throw new Error('State mismatch');
  }

  // Check state isn't too old (10 minutes max)
  if (Date.now() - stored.timestamp > 600000) {
    throw new Error('State expired');
  }

  sessionStorage.removeItem('oauth_state');
}
```

### C. Secure Session Management

```javascript
// Server-side session handling
const session = require('express-session');
const RedisStore = require('connect-redis').default;
const redis = require('redis');

const redisClient = redis.createClient();

app.use(session({
  store: new RedisStore({ client: redisClient }),
  secret: process.env.SESSION_SECRET,
  name: 'sid', // Don't use default 'connect.sid'
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: true,          // HTTPS only
    httpOnly: true,        // No JS access
    sameSite: 'strict',    // CSRF protection
    maxAge: 86400000,      // 24 hours
    domain: '.example.com' // Only if needed for subdomains
  }
}));

// Logout - properly invalidate session
app.post('/logout', async (req, res) => {
  // Revoke refresh token
  if (req.session.refreshToken) {
    await revokeToken(req.session.refreshToken);
  }

  // Destroy session
  req.session.destroy((err) => {
    if (err) {
      console.error('Session destruction error:', err);
    }
    res.clearCookie('sid');
    res.json({ success: true });
  });
});
```

---

## 8. Error Handling

```javascript
// Standard OAuth error responses
const OAuthErrors = {
  invalid_request: {
    status: 400,
    description: 'The request is missing a required parameter or malformed'
  },
  invalid_client: {
    status: 401,
    description: 'Client authentication failed'
  },
  invalid_grant: {
    status: 400,
    description: 'The authorization code or refresh token is invalid or expired'
  },
  unauthorized_client: {
    status: 401,
    description: 'The client is not authorized for this grant type'
  },
  unsupported_grant_type: {
    status: 400,
    description: 'The grant type is not supported'
  },
  invalid_scope: {
    status: 400,
    description: 'The requested scope is invalid or unknown'
  },
  access_denied: {
    status: 403,
    description: 'The resource owner denied the request'
  }
};

// Token endpoint error response
app.post('/token', async (req, res) => {
  try {
    const tokens = await handleTokenRequest(req.body);
    res.json(tokens);
  } catch (error) {
    const oauthError = OAuthErrors[error.code] || OAuthErrors.invalid_request;

    res.status(oauthError.status).json({
      error: error.code || 'invalid_request',
      error_description: error.message || oauthError.description
    });
  }
});
```

---

## 9. Deployment Checklist

### Security
- [ ] HTTPS required everywhere
- [ ] Tokens have appropriate expiration
- [ ] Refresh tokens rotated on use
- [ ] State parameter validated
- [ ] PKCE implemented for public clients

### Token Handling
- [ ] Access tokens short-lived
- [ ] Refresh tokens secure and rotatable
- [ ] Token revocation implemented
- [ ] JWTs properly validated

### Configuration
- [ ] Client secrets secured
- [ ] Redirect URIs validated strictly
- [ ] Scopes properly defined
- [ ] CORS configured correctly

### Monitoring
- [ ] Failed auth attempts logged
- [ ] Token usage monitored
- [ ] Anomaly detection in place
- [ ] Alerts for security events

---

## 10. Quick Reference

```javascript
// OAuth 2.0 Grant Types
'authorization_code' // User authentication (with PKCE)
'refresh_token'      // Token refresh
'client_credentials' // Machine-to-machine
'urn:ietf:params:oauth:grant-type:device_code' // Device flow

// Common Scopes
'openid'        // Required for OIDC
'profile'       // Basic profile
'email'         // Email address
'offline_access' // Request refresh token

// Token Endpoints
GET  /authorize     // Authorization endpoint
POST /token         // Token endpoint
POST /revoke        // Revocation endpoint
GET  /userinfo      // User info endpoint
GET  /.well-known/openid-configuration // Discovery

// JWT Claims
iss  // Issuer
sub  // Subject (user ID)
aud  // Audience
exp  // Expiration
iat  // Issued at
jti  // JWT ID (unique identifier)
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Security Team

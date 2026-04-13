# Environment Configuration Guidelines
Mandatory standards for managing environment variables and configuration across different deployment environments. dotenv, docker-compose, Kubernetes ConfigMaps/Secrets, AWS SSM, HashiCorp Vault.

---

**Agent Profile**: The Configuration Expert
**Role**: Senior Platform Engineer & Security Specialist
**Objective**: Generate secure, maintainable configuration management strategies for multi-environment deployments.
**Tools**: dotenv, docker-compose, Kubernetes ConfigMaps/Secrets, AWS SSM, HashiCorp Vault.

---

## 1. Core Philosophies: CONFIG-FIRST

- **C**entralized: Single source of truth for configuration
- **O**verrideable: Environment-specific values
- **N**ever hardcoded: No secrets in code
- **F**ail-fast: Validate configuration at startup
- **I**mmutable: Config changes trigger deployments
- **G**uarded: Secrets encrypted and access-controlled

---

## 2. Configuration Hierarchy (MANDATORY)

### A. Priority Order (Lowest to Highest)

```markdown
1. Default values in code
2. Configuration files (.env.defaults)
3. Environment-specific files (.env.development)
4. Environment variables
5. Command-line arguments
6. Runtime overrides (feature flags)
```

### B. File Structure

```
project/
├── config/
│   ├── default.js           # Default values
│   ├── development.js       # Development overrides
│   ├── test.js              # Test environment
│   ├── staging.js           # Staging environment
│   ├── production.js        # Production environment
│   └── custom-environment-variables.js  # Env var mapping
├── .env.example             # Template with all variables
├── .env                     # Local overrides (gitignored)
├── .env.development         # Development defaults
├── .env.test                # Test defaults
└── docker-compose.yml       # Container environment
```

---

## 3. Environment Variables (MANDATORY)

### A. Naming Convention

```bash
# Format: [APP]_[CATEGORY]_[NAME]
# All uppercase, underscore separated

# Database
MYAPP_DB_HOST=localhost
MYAPP_DB_PORT=5432
MYAPP_DB_NAME=myapp
MYAPP_DB_USER=postgres
MYAPP_DB_PASSWORD=secret
MYAPP_DB_POOL_SIZE=10
MYAPP_DB_SSL_ENABLED=true

# Redis
MYAPP_REDIS_URL=redis://localhost:6379
MYAPP_REDIS_PASSWORD=

# API Keys (external services)
MYAPP_STRIPE_API_KEY=sk_test_xxx
MYAPP_SENDGRID_API_KEY=SG.xxx

# Feature Flags
MYAPP_FEATURE_NEW_CHECKOUT=true
MYAPP_FEATURE_BETA_UI=false

# Application
MYAPP_PORT=3000
MYAPP_HOST=0.0.0.0
MYAPP_LOG_LEVEL=info
MYAPP_NODE_ENV=development

# URLs
MYAPP_API_URL=https://api.example.com
MYAPP_WEB_URL=https://www.example.com
MYAPP_CDN_URL=https://cdn.example.com
```

### B. Variable Types

```typescript
// config/schema.ts
import { z } from 'zod';

const envSchema = z.object({
  // Required strings
  NODE_ENV: z.enum(['development', 'test', 'staging', 'production']),
  DATABASE_URL: z.string().url(),

  // Optional with defaults
  PORT: z.coerce.number().default(3000),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),

  // Booleans (various formats)
  ENABLE_CACHE: z.preprocess(
    (val) => val === 'true' || val === '1' || val === 'yes',
    z.boolean()
  ).default(true),

  // Numbers with validation
  DB_POOL_SIZE: z.coerce.number().min(1).max(100).default(10),
  REQUEST_TIMEOUT: z.coerce.number().positive().default(30000),

  // Optional secrets
  API_KEY: z.string().min(1).optional(),
  SECRET_KEY: z.string().min(32),

  // Arrays (comma-separated)
  ALLOWED_ORIGINS: z.preprocess(
    (val) => typeof val === 'string' ? val.split(',').map(s => s.trim()) : [],
    z.array(z.string().url())
  ).default(['http://localhost:3000']),

  // JSON values
  FEATURE_FLAGS: z.preprocess(
    (val) => typeof val === 'string' ? JSON.parse(val) : val,
    z.record(z.boolean())
  ).default({}),
});

export type Env = z.infer<typeof envSchema>;
```

---

## 4. Configuration Loading (MANDATORY)

### A. Node.js Configuration Module

```typescript
// config/index.ts
import dotenv from 'dotenv';
import path from 'path';
import { z } from 'zod';

// Load environment-specific .env file
const envFile = process.env.NODE_ENV === 'test'
  ? '.env.test'
  : process.env.NODE_ENV === 'production'
    ? '.env.production'
    : '.env';

dotenv.config({ path: path.resolve(process.cwd(), envFile) });

// Schema definition
const configSchema = z.object({
  env: z.enum(['development', 'test', 'staging', 'production']),
  port: z.coerce.number(),
  host: z.string(),

  database: z.object({
    url: z.string().url(),
    poolSize: z.coerce.number(),
    ssl: z.boolean(),
  }),

  redis: z.object({
    url: z.string(),
    password: z.string().optional(),
  }),

  auth: z.object({
    jwtSecret: z.string().min(32),
    jwtExpiresIn: z.string(),
    bcryptRounds: z.coerce.number(),
  }),

  logging: z.object({
    level: z.enum(['debug', 'info', 'warn', 'error']),
    format: z.enum(['json', 'pretty']),
  }),

  features: z.object({
    newCheckout: z.boolean(),
    betaUI: z.boolean(),
  }),
});

// Parse and validate
function loadConfig() {
  const raw = {
    env: process.env.NODE_ENV || 'development',
    port: process.env.PORT || 3000,
    host: process.env.HOST || '0.0.0.0',

    database: {
      url: process.env.DATABASE_URL,
      poolSize: process.env.DB_POOL_SIZE || 10,
      ssl: process.env.DB_SSL === 'true',
    },

    redis: {
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      password: process.env.REDIS_PASSWORD,
    },

    auth: {
      jwtSecret: process.env.JWT_SECRET,
      jwtExpiresIn: process.env.JWT_EXPIRES_IN || '1h',
      bcryptRounds: process.env.BCRYPT_ROUNDS || 10,
    },

    logging: {
      level: process.env.LOG_LEVEL || 'info',
      format: process.env.LOG_FORMAT || 'json',
    },

    features: {
      newCheckout: process.env.FEATURE_NEW_CHECKOUT === 'true',
      betaUI: process.env.FEATURE_BETA_UI === 'true',
    },
  };

  const result = configSchema.safeParse(raw);

  if (!result.success) {
    console.error('Invalid configuration:');
    console.error(result.error.format());
    process.exit(1);
  }

  return result.data;
}

export const config = loadConfig();
export type Config = z.infer<typeof configSchema>;
```

### B. Fail-Fast Validation

```typescript
// Validate at startup
import { config } from './config';

// This runs immediately when imported
// If validation fails, process.exit(1) is called

// Additional runtime checks
function validateConfig(config: Config): void {
  const errors: string[] = [];

  // Check required secrets in production
  if (config.env === 'production') {
    if (!config.auth.jwtSecret || config.auth.jwtSecret.length < 32) {
      errors.push('JWT_SECRET must be at least 32 characters in production');
    }

    if (config.database.url.includes('localhost')) {
      errors.push('DATABASE_URL should not point to localhost in production');
    }

    if (!config.database.ssl) {
      errors.push('Database SSL should be enabled in production');
    }
  }

  // Check for placeholder values
  const placeholders = ['your-api-key', 'change-me', 'xxx', 'TODO'];
  const checkPlaceholder = (value: string, name: string) => {
    if (placeholders.some(p => value.toLowerCase().includes(p))) {
      errors.push(`${name} appears to contain a placeholder value`);
    }
  };

  if (config.auth.jwtSecret) {
    checkPlaceholder(config.auth.jwtSecret, 'JWT_SECRET');
  }

  if (errors.length > 0) {
    console.error('Configuration validation failed:');
    errors.forEach(e => console.error(`  - ${e}`));
    process.exit(1);
  }
}

validateConfig(config);
```

---

## 5. Secrets Management (MANDATORY)

### A. Never Commit Secrets

```gitignore
# .gitignore
.env
.env.local
.env.*.local
*.pem
*.key
credentials.json
secrets/
```

```yaml
# .env.example - Template for developers
# Copy to .env and fill in values

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/myapp

# Authentication
JWT_SECRET=generate-a-secure-random-string-at-least-32-chars

# External Services
STRIPE_API_KEY=sk_test_your_key_here
SENDGRID_API_KEY=SG.your_key_here

# Feature Flags
FEATURE_NEW_CHECKOUT=false
FEATURE_BETA_UI=false
```

### B. Secrets in Production

```typescript
// Using AWS Secrets Manager
import { SecretsManagerClient, GetSecretValueCommand } from '@aws-sdk/client-secrets-manager';

async function loadSecrets(secretName: string): Promise<Record<string, string>> {
  const client = new SecretsManagerClient({ region: process.env.AWS_REGION });

  const response = await client.send(
    new GetSecretValueCommand({ SecretId: secretName })
  );

  if (response.SecretString) {
    return JSON.parse(response.SecretString);
  }

  throw new Error('Secret not found');
}

// Load secrets before starting app
async function bootstrap() {
  if (process.env.NODE_ENV === 'production') {
    const secrets = await loadSecrets('myapp/production');

    // Inject into environment
    process.env.DATABASE_URL = secrets.DATABASE_URL;
    process.env.JWT_SECRET = secrets.JWT_SECRET;
    process.env.STRIPE_API_KEY = secrets.STRIPE_API_KEY;
  }

  // Now load config
  const { config } = await import('./config');
  return config;
}
```

### C. Kubernetes Secrets

```yaml
# k8s/secrets.yaml (use external-secrets or sealed-secrets in practice)
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
type: Opaque
stringData:
  DATABASE_URL: postgresql://user:pass@db:5432/myapp
  JWT_SECRET: your-secret-key

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
data:
  NODE_ENV: production
  LOG_LEVEL: info
  PORT: "3000"

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  template:
    spec:
      containers:
        - name: myapp
          envFrom:
            - configMapRef:
                name: myapp-config
            - secretRef:
                name: myapp-secrets
          env:
            - name: DATABASE_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: DATABASE_PASSWORD
```

---

## 6. Environment-Specific Configuration (MANDATORY)

### A. Development

```typescript
// config/development.ts
export default {
  database: {
    url: 'postgresql://postgres:postgres@localhost:5432/myapp_dev',
    logging: true,
    synchronize: true, // Auto-sync schema (dev only!)
  },

  redis: {
    url: 'redis://localhost:6379',
  },

  logging: {
    level: 'debug',
    format: 'pretty',
  },

  auth: {
    jwtExpiresIn: '7d', // Longer for dev convenience
  },

  cors: {
    origins: ['http://localhost:3000', 'http://localhost:3001'],
  },

  features: {
    // Enable all features in dev
    newCheckout: true,
    betaUI: true,
  },
};
```

### B. Testing

```typescript
// config/test.ts
export default {
  database: {
    url: process.env.TEST_DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/myapp_test',
    logging: false,
    dropSchema: true, // Clean slate for each test run
  },

  redis: {
    url: 'redis://localhost:6379/1', // Different DB index
  },

  logging: {
    level: 'error', // Quiet during tests
  },

  auth: {
    jwtSecret: 'test-secret-not-for-production',
    jwtExpiresIn: '1h',
  },

  // Mock external services
  external: {
    stripe: { useMock: true },
    sendgrid: { useMock: true },
  },
};
```

### C. Production

```typescript
// config/production.ts
export default {
  database: {
    // URL from secrets manager
    ssl: true,
    poolSize: 20,
    logging: false,
    synchronize: false, // Never auto-sync in production!
  },

  redis: {
    // URL from secrets manager
    tls: true,
  },

  logging: {
    level: 'info',
    format: 'json',
  },

  auth: {
    jwtExpiresIn: '15m', // Short-lived tokens
    bcryptRounds: 12, // Higher security
  },

  cors: {
    origins: ['https://www.example.com', 'https://app.example.com'],
  },

  rateLimit: {
    windowMs: 60000,
    max: 100,
  },
};
```

---

## 7. Docker Configuration (MANDATORY)

### A. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "${PORT:-3000}:3000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/myapp
      - REDIS_URL=redis://redis:6379
    env_file:
      - .env
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:

# docker-compose.override.yml (dev-specific, gitignored)
version: '3.8'
services:
  app:
    volumes:
      - .:/app
      - /app/node_modules
    command: npm run dev
```

### B. Dockerfile

```dockerfile
FROM node:20-alpine AS base

# Production dependencies
FROM base AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Build
FROM base AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production image
FROM base AS production
WORKDIR /app

# Security: Run as non-root
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

COPY --from=deps --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=build --chown=nodejs:nodejs /app/dist ./dist
COPY --from=build --chown=nodejs:nodejs /app/package.json ./

USER nodejs

# Config via environment
ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

CMD ["node", "dist/index.js"]
```

---

## 8. Multi-Environment Deployment (MANDATORY)

### A. Environment Matrix

```markdown
| Variable          | Development | Staging           | Production         |
|-------------------|-------------|-------------------|-------------------|
| NODE_ENV          | development | staging           | production        |
| LOG_LEVEL         | debug       | info              | info              |
| DB_SSL            | false       | true              | true              |
| DB_POOL_SIZE      | 5           | 10                | 20                |
| JWT_EXPIRES_IN    | 7d          | 1h                | 15m               |
| RATE_LIMIT_MAX    | 1000        | 100               | 100               |
| FEATURE_BETA      | true        | true              | false             |
```

### B. CI/CD Environment Variables

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main, staging]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.ref == 'refs/heads/main' && 'production' || 'staging' }}
    steps:
      - uses: actions/checkout@v4

      - name: Deploy
        env:
          # From GitHub environment secrets
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          JWT_SECRET: ${{ secrets.JWT_SECRET }}
          # From GitHub environment variables
          NODE_ENV: ${{ vars.NODE_ENV }}
          LOG_LEVEL: ${{ vars.LOG_LEVEL }}
        run: ./deploy.sh
```

---

## 9. Configuration Documentation (MANDATORY)

### A. Generate Docs from Schema

```typescript
// scripts/generate-env-docs.ts
import { configSchema } from '../config/schema';

function generateDocs(schema: z.ZodObject<any>, prefix = ''): string {
  let docs = '';

  for (const [key, value] of Object.entries(schema.shape)) {
    const envKey = prefix ? `${prefix}_${key}`.toUpperCase() : key.toUpperCase();

    if (value instanceof z.ZodObject) {
      docs += generateDocs(value, envKey);
    } else {
      const description = value._def.description || 'No description';
      const defaultValue = value._def.defaultValue?.() ?? 'Required';

      docs += `### ${envKey}\n`;
      docs += `${description}\n\n`;
      docs += `- Type: ${getZodType(value)}\n`;
      docs += `- Default: \`${defaultValue}\`\n\n`;
    }
  }

  return docs;
}

console.log('# Environment Variables\n\n');
console.log(generateDocs(configSchema));
```

### B. Example Documentation Output

```markdown
# Environment Variables

## Required Variables

### DATABASE_URL
PostgreSQL connection string.

- Type: string (URL)
- Required: Yes
- Example: `postgresql://user:pass@host:5432/dbname`

### JWT_SECRET
Secret key for signing JWT tokens. Must be at least 32 characters.

- Type: string
- Required: Yes
- Minimum length: 32

## Optional Variables

### PORT
HTTP server port.

- Type: number
- Default: `3000`

### LOG_LEVEL
Application log level.

- Type: enum
- Values: `debug`, `info`, `warn`, `error`
- Default: `info`
```

---

## 10. Deployment Checklist

### Development Setup
- [ ] `.env.example` is up to date
- [ ] All developers have `.env` configured
- [ ] Local services (DB, Redis) documented

### Before Deployment
- [ ] All required secrets configured
- [ ] No placeholder values in production
- [ ] SSL/TLS enabled for databases
- [ ] Secrets not logged

### Production
- [ ] Secrets in secure storage (SSM, Vault)
- [ ] Config validation at startup
- [ ] Sensitive values redacted in logs
- [ ] Rotation plan for secrets

---

## 11. Quick Reference

```bash
# Common patterns
MYAPP_CATEGORY_NAME=value

# Types
STRING=hello
NUMBER=42
BOOLEAN=true
URL=https://example.com
JSON='{"key":"value"}'
ARRAY=a,b,c

# Files
.env              # Local (gitignored)
.env.example      # Template (committed)
.env.development  # Dev defaults
.env.production   # Prod defaults (no secrets!)

# Never commit
- Passwords
- API keys
- Private keys
- Connection strings with credentials
```

---

## 12. Why This Configuration Works

- **Fail-fast validation prevents runtime surprises**: Validating all configuration at startup with Zod schemas ensures misconfigurations are caught immediately during deployment, not at 2 AM when a code path finally touches an unset variable.
- **Secrets never touch version control**: The strict separation between committed templates (.env.example) and gitignored local overrides (.env), combined with production secrets management (Vault, SSM, Kubernetes Secrets), eliminates the most common vector for credential leaks.
- **Hierarchical overrides support all environments**: The layered configuration approach (defaults, environment files, environment variables, CLI arguments) allows a single codebase to run correctly across development, test, staging, and production without code changes.
- **Schema-driven documentation stays in sync**: Generating configuration documentation from the same schema used for validation guarantees that docs always reflect reality, eliminating stale or incomplete environment variable documentation.
- **Typed configuration prevents subtle bugs**: Parsing environment strings into proper types (numbers, booleans, arrays, URLs) at the configuration boundary means application code works with correct types throughout, avoiding string comparison bugs and type coercion surprises.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Platform Team


**End of Environment Configuration Guidelines**

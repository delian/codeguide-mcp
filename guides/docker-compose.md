# Docker Compose Guidelines
This document provides mandatory coding style and practices for creation of docker-compose files

---
Agent Profile: The Orchestration Architect
Role: Senior DevOps Engineer & Container Orchestration Specialist
Objective: Generate production-ready, maintainable, and highly optimized docker-compose configurations.
Tools: Docker Compose v2.x+, Docker Engine 24.x+, YAML 1.2 Standards.

## 1. Core Philosophies
The agent must adhere to the "SIX PILLARS" standard for every docker-compose.yml generated:

**Declarative**: Define desired state, not imperative commands.
**Reproducible**: Same compose file = same environment (dev/staging/prod parity).
**Secure**: Secrets management, least privilege, network isolation.
**Maintainable**: Clear structure, comments, version control friendly.
**Modular**: Separate components in individual files, composed with `include`.
**Hexagonal**: Layer separation (domain, application, infrastructure) reflected in compose structure.

## 2. Hexagonal Architecture & Modular Composition (MANDATORY)

### A. Architecture Principles

Docker Compose configurations MUST follow hexagonal (ports & adapters) architecture:

- **Domain Layer**: Core business services (APIs, workers, schedulers)
- **Infrastructure Layer**: External dependencies (databases, caches, message queues)
- **Adapters Layer**: Gateways, proxies, load balancers, API gateways
- **Observability Layer**: Monitoring, logging, tracing services

### B. Modular File Structure

**CRITICAL: Each architectural layer MUST be in a separate compose file, combined using `include`.**

```
project/
├── docker-compose.yml           # Main entry point (uses include)
├── compose/
│   ├── domain/
│   │   ├── api-services.yml      # Core API services
│   │   ├── worker-services.yml   # Background workers
│   │   └── scheduler-services.yml # Cron/scheduled jobs
│   ├── infrastructure/
│   │   ├── databases.yml         # PostgreSQL, MongoDB, etc.
│   │   ├── caches.yml            # Redis, Memcached
│   │   ├── messaging.yml         # RabbitMQ, Kafka, NATS
│   │   └── storage.yml           # MinIO, S3-compatible storage
│   ├── adapters/
│   │   ├── gateway.yml           # API Gateway (Traefik, Kong)
│   │   ├── proxy.yml             # Reverse proxy (Nginx)
│   │   └── auth.yml              # Auth service (Keycloak, Auth0)
│   ├── observability/
│   │   ├── monitoring.yml        # Prometheus, Grafana
│   │   ├── logging.yml           # Loki, Elasticsearch
│   │   └── tracing.yml           # Jaeger, Tempo
│   └── shared/
│       ├── networks.yml          # Shared network definitions
│       ├── volumes.yml           # Shared volume definitions
│       └── secrets.yml           # Shared secrets
├── docker-compose.dev.yml       # Development overrides
├── docker-compose.prod.yml      # Production overrides
└── .env
```

### C. Using `include` Directive

The main `docker-compose.yml` uses `include` to compose modular files:

```yaml
# docker-compose.yml - Main composition file
name: myapp-stack

# Include modular compose files
include:
  # Shared resources (must be first)
  - compose/shared/networks.yml
  - compose/shared/volumes.yml
  - compose/shared/secrets.yml
  
  # Infrastructure layer (databases, queues, caches)
  - compose/infrastructure/databases.yml
  - compose/infrastructure/caches.yml
  - compose/infrastructure/messaging.yml
  
  # Domain layer (core business services)
  - compose/domain/api-services.yml
  - compose/domain/worker-services.yml
  
  # Adapters layer (gateways, proxies)
  - compose/adapters/gateway.yml
  - compose/adapters/proxy.yml
  
  # Observability layer (optional for production)
  - path: compose/observability/monitoring.yml
    env_file: .env.monitoring
  - path: compose/observability/logging.yml
    env_file: .env.logging

# Global defaults (optional)
services:
  x-common-logging: &default-logging
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"
```

### D. Modular File Examples

#### 1. Infrastructure Layer - Databases

```yaml
# compose/infrastructure/databases.yml
name: infrastructure-databases

services:
  postgres:
    image: postgres:16.1-alpine3.19
    restart: unless-stopped
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d:ro
    networks:
      - database
    secrets:
      - db_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 1G
    labels:
      com.example.layer: "infrastructure"
      com.example.component: "database"
```

#### 2. Infrastructure Layer - Caches

```yaml
# compose/infrastructure/caches.yml
name: infrastructure-caches

services:
  redis:
    image: redis:7.2-alpine3.19
    restart: unless-stopped
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    networks:
      - cache
      - backend
    healthcheck:
      test: ["CMD", "redis-cli", "--pass", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 768M
    labels:
      com.example.layer: "infrastructure"
      com.example.component: "cache"
```

#### 3. Domain Layer - API Services

```yaml
# compose/domain/api-services.yml
name: domain-api

services:
  user-service:
    image: mycompany/user-service:${USER_SERVICE_VERSION}
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file:
      - .env
    environment:
      - SERVICE_NAME=user-service
      - PORT=3001
      - DB_HOST=postgres
      - CACHE_HOST=redis
    expose:
      - "3001"
    networks:
      - backend
      - database
      - cache
    secrets:
      - db_password
      - jwt_secret
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    labels:
      com.example.layer: "domain"
      com.example.component: "api"
      com.example.service: "user-service"

  order-service:
    image: mycompany/order-service:${ORDER_SERVICE_VERSION}
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file:
      - .env
    environment:
      - SERVICE_NAME=order-service
      - PORT=3002
      - DB_HOST=postgres
      - CACHE_HOST=redis
    expose:
      - "3002"
    networks:
      - backend
      - database
      - cache
    secrets:
      - db_password
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3002/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    labels:
      com.example.layer: "domain"
      com.example.component: "api"
      com.example.service: "order-service"
```

#### 4. Domain Layer - Workers

```yaml
# compose/domain/worker-services.yml
name: domain-workers

services:
  email-worker:
    image: mycompany/email-worker:${WORKER_VERSION}
    restart: unless-stopped
    depends_on:
      rabbitmq:
        condition: service_healthy
    env_file:
      - .env
    environment:
      - WORKER_TYPE=email
      - QUEUE_HOST=rabbitmq
    networks:
      - backend
      - messaging
    secrets:
      - smtp_credentials
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    labels:
      com.example.layer: "domain"
      com.example.component: "worker"
      com.example.worker-type: "email"

  notification-worker:
    image: mycompany/notification-worker:${WORKER_VERSION}
    restart: unless-stopped
    depends_on:
      rabbitmq:
        condition: service_healthy
    env_file:
      - .env
    environment:
      - WORKER_TYPE=notification
      - QUEUE_HOST=rabbitmq
    networks:
      - backend
      - messaging
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    labels:
      com.example.layer: "domain"
      com.example.component: "worker"
      com.example.worker-type: "notification"
```

#### 5. Adapters Layer - Gateway

```yaml
# compose/adapters/gateway.yml
name: adapters-gateway

services:
  traefik:
    image: traefik:v3.0
    restart: unless-stopped
    command:
      - --api.dashboard=true
      - --providers.docker=true
      - --providers.docker.exposedbydefault=false
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.letsencrypt.acme.email=${ACME_EMAIL}
      - --certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json
      - --certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web
      - --metrics.prometheus=true
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - traefik-certs:/letsencrypt
    networks:
      - public
      - backend
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
    labels:
      com.example.layer: "adapter"
      com.example.component: "gateway"
      # Traefik dashboard
      traefik.enable: "true"
      traefik.http.routers.dashboard.rule: "Host(`traefik.${DOMAIN}`)"
      traefik.http.routers.dashboard.service: "api@internal"
```

#### 6. Shared - Networks

```yaml
# compose/shared/networks.yml
name: shared-networks

networks:
  # Public-facing network
  public:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: app-public
    labels:
      com.example.layer: "adapter"
  
  # Backend services network
  backend:
    driver: bridge
    internal: false
    labels:
      com.example.layer: "domain"
  
  # Database network (isolated)
  database:
    driver: bridge
    internal: true
    labels:
      com.example.layer: "infrastructure"
      com.example.isolated: "true"
  
  # Cache network
  cache:
    driver: bridge
    internal: false
    labels:
      com.example.layer: "infrastructure"
  
  # Messaging network
  messaging:
    driver: bridge
    internal: true
    labels:
      com.example.layer: "infrastructure"
  
  # Monitoring network
  monitoring:
    driver: bridge
    labels:
      com.example.layer: "observability"
```

#### 7. Shared - Volumes

```yaml
# compose/shared/volumes.yml
name: shared-volumes

volumes:
  # Database volumes
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/postgres
    labels:
      com.example.layer: "infrastructure"
      com.example.component: "database"
  
  # Cache volumes
  redis-data:
    driver: local
    labels:
      com.example.layer: "infrastructure"
      com.example.component: "cache"
  
  # Gateway volumes
  traefik-certs:
    driver: local
    labels:
      com.example.layer: "adapter"
      com.example.component: "gateway"
  
  # Monitoring volumes
  prometheus-data:
    driver: local
    labels:
      com.example.layer: "observability"
  
  grafana-data:
    driver: local
    labels:
      com.example.layer: "observability"
```

#### 8. Shared - Secrets

```yaml
# compose/shared/secrets.yml
name: shared-secrets

secrets:
  db_password:
    file: ./secrets/db_password.txt
  
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  
  smtp_credentials:
    file: ./secrets/smtp_credentials.txt
  
  api_key:
    external: true
    name: production_api_key
```

### E. Environment-Specific Composition

Use environment-specific files that also leverage modular structure:

```yaml
# docker-compose.dev.yml - Development overrides
include:
  - compose/shared/networks.yml
  - compose/shared/volumes.yml
  - compose/infrastructure/databases.yml
  - compose/infrastructure/caches.yml
  - compose/domain/api-services.yml
  # Note: No observability layer in dev

services:
  # Override user-service for development
  user-service:
    build:
      context: ./services/user-service
      dockerfile: Dockerfile.dev
    volumes:
      - ./services/user-service:/app:cached
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - DEBUG=app:*
    ports:
      - "3001:3001"
      - "9229:9229"  # Debugger
```

```yaml
# docker-compose.prod.yml - Production overrides
include:
  - compose/shared/networks.yml
  - compose/shared/volumes.yml
  - compose/shared/secrets.yml
  - compose/infrastructure/databases.yml
  - compose/infrastructure/caches.yml
  - compose/infrastructure/messaging.yml
  - compose/domain/api-services.yml
  - compose/domain/worker-services.yml
  - compose/adapters/gateway.yml
  - compose/observability/monitoring.yml
  - compose/observability/tracing.yml

services:
  # Production-specific overrides
  user-service:
    image: mycompany/user-service:${USER_SERVICE_VERSION}@sha256:${USER_SERVICE_DIGEST}
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    restart: always
```

### F. Commands for Modular Composition

```bash
# Development (limited layers)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production (all layers)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Only infrastructure layer
docker compose -f compose/infrastructure/databases.yml up -d

# Specific service layers
docker compose -f docker-compose.yml up postgres redis user-service

# Validate entire composition
docker compose config

# View merged configuration
docker compose config --no-interpolate > merged-compose.yml
```

### G. Benefits of Modular Hexagonal Approach

1. **Layer Isolation**: Infrastructure changes don't affect domain services
2. **Independent Scaling**: Scale only the layers that need it
3. **Team Ownership**: Different teams own different compose files
4. **Environment Flexibility**: Include only needed layers per environment
5. **Testing Simplicity**: Test each layer independently
6. **Deployment Control**: Deploy layers in correct dependency order
7. **Maintenance**: Smaller, focused files are easier to understand and modify
8. **Reusability**: Share infrastructure layer across multiple projects

### H. Mandatory Labels for Layers

All services MUST include layer identification labels:

```yaml
labels:
  com.example.layer: "domain|infrastructure|adapter|observability"
  com.example.component: "api|worker|database|cache|gateway|etc"
  com.example.service: "service-name"
```

## 3. Mandatory Structure Requirements

### A. File Format & Versioning
* **File Name**: Use `docker-compose.yml` (preferred) or `docker-compose.yaml`.

* **Compose Spec**: Use modern Compose Specification format (no legacy `version:` field for Compose V2).

* **YAML Style**: Use 2-space indentation, no tabs. Always use explicit quotes for strings containing special characters.

```yaml
# ✅ CORRECT - Modern Compose Spec (no version field needed)
name: myapp-stack

services:
  web:
    image: nginx:1.25-alpine
    
# ❌ WRONG - Legacy version format
version: '3.8'
services:
  web:
    image: nginx:1.25-alpine
```

### B. Top-Level Organization
Structure files in this exact order:

1. `name:` - Stack name (optional but recommended)
2. `services:` - Container definitions
3. `networks:` - Custom network definitions
4. `volumes:` - Named volume definitions
5. `configs:` - Configuration file definitions
6. `secrets:` - Secrets definitions

### C. Service Definition Order
Within each service, follow this property order:

1. `image:` or `build:`
2. `container_name:` (use sparingly)
3. `hostname:`
4. `depends_on:` (with health checks)
5. `environment:` or `env_file:`
6. `ports:` (external:internal)
7. `expose:` (internal only)
8. `volumes:`
9. `networks:`
10. `healthcheck:`
11. `restart:` policy
12. `deploy:` (resources, replicas)
13. `security_opt:`, `cap_add:`, `cap_drop:`
14. `labels:`
15. `command:` or `entrypoint:`

## 3. Mandatory Best Practices

### A. Image Management
* **Pin Versions**: NEVER use `:latest`. Always use specific semantic versions.

```yaml
# ✅ CORRECT
services:
  db:
    image: postgres:16.1-alpine3.19
    
# ❌ WRONG
services:
  db:
    image: postgres:latest
```

* **Digests for Production**: Pin critical services by SHA256 digest.

```yaml
services:
  api:
    image: myapp:v2.3.1@sha256:abcd1234...
```

### B. Environment Variables & Secrets
* **NEVER** hardcode secrets in compose files.

* **Use `.env` files** for non-sensitive configuration.

* **Use Docker Secrets** for sensitive data (passwords, tokens, keys).

```yaml
# ✅ CORRECT - Using environment variables
services:
  app:
    env_file:
      - .env
      - .env.local
    environment:
      - NODE_ENV=production
      - LOG_LEVEL=${LOG_LEVEL:-info}
    secrets:
      - db_password
      - api_key

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    external: true
    
# ❌ WRONG - Hardcoded secrets
services:
  app:
    environment:
      - DB_PASSWORD=super_secret_123
```

### C. Health Checks & Dependencies
* **Always define health checks** for services that others depend on.

* **Use `depends_on` with conditions** to ensure proper startup order.

```yaml
services:
  backend:
    image: myapp:1.0.0
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
        
  db:
    image: postgres:16.1-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
```

### D. Networking
* **Always use custom networks** instead of default bridge.

* **Use network aliases** for service discovery.

* **Isolate services** into multiple networks based on trust boundaries.

```yaml
services:
  frontend:
    image: nginx:1.25-alpine
    networks:
      - public
      - backend
      
  api:
    image: myapi:1.0
    networks:
      - backend
      - database
      
  db:
    image: postgres:16.1-alpine
    networks:
      - database  # Only accessible to api, not frontend

networks:
  public:
    driver: bridge
  backend:
    driver: bridge
    internal: false
  database:
    driver: bridge
    internal: true  # No external access
```

### E. Volume Management
* **Use named volumes** for persistent data.

* **Use bind mounts** only for development.

* **Always specify mount options** for security.

```yaml
services:
  app:
    volumes:
      # Named volume (production)
      - app-data:/var/lib/app:rw
      # Bind mount (development only)
      - ./src:/app/src:ro
      # tmpfs for temporary data
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 100M
          
volumes:
  app-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/app
```

### F. Resource Constraints
* **Always set resource limits** to prevent resource exhaustion.

* **Use `deploy` section** for resource reservations and limits.

```yaml
services:
  api:
    image: myapi:1.0
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
```

### G. Security Hardening
* **Run as non-root** user whenever possible.

* **Drop unnecessary capabilities**.

* **Use read-only root filesystem** where applicable.

```yaml
services:
  app:
    image: myapp:1.0
    user: "1001:1001"
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE  # Only if needed
    tmpfs:
      - /tmp  # Writable temp directory
```

## 4. Gold Standard Examples

### Example 1: Full-Stack Web Application (Production)

```yaml
name: webapp-production

services:
  # Frontend - Nginx serving static files
  frontend:
    image: nginx:1.25-alpine@sha256:a5127daff3d6f4606be3100a252419bfa84fd6ee5cd74d0feaca1a5068f97dcf
    container_name: webapp-frontend
    restart: unless-stopped
    depends_on:
      backend:
        condition: service_healthy
    environment:
      - NGINX_HOST=${DOMAIN:-localhost}
      - NGINX_PORT=80
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - static-content:/usr/share/nginx/html:ro
    networks:
      - public
      - backend
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 128M
    labels:
      com.example.description: "Frontend web server"
      com.example.department: "engineering"

  # Backend API
  backend:
    image: mycompany/api:2.5.1
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    env_file:
      - .env
      - .env.production
    environment:
      - NODE_ENV=production
      - PORT=3000
      - DB_HOST=db
      - REDIS_HOST=redis
    expose:
      - "3000"
    volumes:
      - ./uploads:/app/uploads:rw
      - type: tmpfs
        target: /app/tmp
        tmpfs:
          size: 256M
    networks:
      - backend
      - database
    secrets:
      - db_password
      - jwt_secret
      - api_key
    healthcheck:
      test: ["CMD", "node", "healthcheck.js"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 45s
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    user: "1001:1001"
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    labels:
      com.example.description: "Backend API service"

  # PostgreSQL Database
  db:
    image: postgres:16.1-alpine3.19
    restart: unless-stopped
    environment:
      - POSTGRES_DB=${DB_NAME}
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
      - POSTGRES_INITDB_ARGS=--encoding=UTF8 --locale=en_US.UTF-8
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-db:/docker-entrypoint-initdb.d:ro
    networks:
      - database
    secrets:
      - db_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER} -d ${DB_NAME}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 1G
    security_opt:
      - no-new-privileges:true
    labels:
      com.example.description: "PostgreSQL database"

  # Redis Cache
  redis:
    image: redis:7.2-alpine3.19
    restart: unless-stopped
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --save 60 1000
      --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - database
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 768M
        reservations:
          cpus: '0.1'
          memory: 256M
    security_opt:
      - no-new-privileges:true
    labels:
      com.example.description: "Redis cache"

networks:
  public:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: webapp-public
  backend:
    driver: bridge
    internal: false
  database:
    driver: bridge
    internal: true  # Database network has no internet access

volumes:
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/postgres
  redis-data:
    driver: local
  static-content:
    driver: local

secrets:
  db_password:
    file: ./secrets/db_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt
  api_key:
    external: true
    name: production_api_key
```

### Example 2: Development Environment (with Live Reload)

```yaml
name: webapp-dev

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
      target: development
    volumes:
      - ./frontend/src:/app/src:ro
      - ./frontend/public:/app/public:ro
      - node_modules:/app/node_modules
    ports:
      - "3000:3000"
      - "3001:3001"  # Vite HMR
    environment:
      - NODE_ENV=development
      - VITE_API_URL=http://localhost:8000
      - CHOKIDAR_USEPOLLING=true  # For file watching in Docker
    networks:
      - dev
    command: npm run dev

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    volumes:
      - ./backend:/app:cached
      - /app/node_modules  # Anonymous volume to prevent overwriting
    ports:
      - "8000:8000"
      - "9229:9229"  # Node.js debugger
    environment:
      - NODE_ENV=development
      - DEBUG=app:*
      - DB_HOST=db
    env_file:
      - .env.development
    networks:
      - dev
    depends_on:
      - db
    command: npm run dev

  db:
    image: postgres:16.1-alpine
    environment:
      - POSTGRES_DB=devdb
      - POSTGRES_USER=devuser
      - POSTGRES_PASSWORD=devpass
    ports:
      - "5432:5432"  # Exposed for local development tools
    volumes:
      - postgres-dev-data:/var/lib/postgresql/data
    networks:
      - dev

networks:
  dev:
    driver: bridge

volumes:
  postgres-dev-data:
  node_modules:
```

### Example 3: Microservices with Observability

```yaml
name: microservices-stack

services:
  # API Gateway
  gateway:
    image: traefik:v3.0
    command:
      - --api.insecure=true
      - --providers.docker=true
      - --providers.docker.exposedbydefault=false
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --metrics.prometheus=true
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"  # Traefik dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - public
      - services
    labels:
      com.example.description: "API Gateway"

  # User Service
  user-service:
    image: mycompany/user-service:1.2.3
    restart: unless-stopped
    environment:
      - SERVICE_NAME=user-service
      - JAEGER_AGENT_HOST=jaeger
      - PROMETHEUS_PORT=9090
    networks:
      - services
      - database
    depends_on:
      db:
        condition: service_healthy
    labels:
      traefik.enable: "true"
      traefik.http.routers.users.rule: "PathPrefix(`/api/users`)"
      com.example.service: "user-service"

  # Order Service
  order-service:
    image: mycompany/order-service:1.2.3
    restart: unless-stopped
    environment:
      - SERVICE_NAME=order-service
      - JAEGER_AGENT_HOST=jaeger
      - KAFKA_BROKERS=kafka:9092
    networks:
      - services
      - database
      - messaging
    depends_on:
      - kafka
      - db
    labels:
      traefik.enable: "true"
      traefik.http.routers.orders.rule: "PathPrefix(`/api/orders`)"

  # Database
  db:
    image: postgres:16.1-alpine
    environment:
      - POSTGRES_USER=microservices
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password
      - POSTGRES_DB=microservices
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - database
    secrets:
      - db_password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Message Queue
  kafka:
    image: confluentinc/cp-kafka:7.5.3
    environment:
      - KAFKA_BROKER_ID=1
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
    networks:
      - messaging
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.3
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181
      - ZOOKEEPER_TICK_TIME=2000
    networks:
      - messaging

  # Observability Stack
  prometheus:
    image: prom/prometheus:v2.48.0
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
      - --web.console.libraries=/usr/share/prometheus/console_libraries
      - --web.console.templates=/usr/share/prometheus/consoles
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - services
      - monitoring

  grafana:
    image: grafana/grafana:10.2.2
    environment:
      - GF_SECURITY_ADMIN_PASSWORD__FILE=/run/secrets/grafana_password
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3001:3000"
    networks:
      - monitoring
    secrets:
      - grafana_password
    depends_on:
      - prometheus

  jaeger:
    image: jaegertracing/all-in-one:1.52
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # HTTP collector
    networks:
      - services
      - monitoring

networks:
  public:
    driver: bridge
  services:
    driver: bridge
  database:
    driver: bridge
    internal: true
  messaging:
    driver: bridge
  monitoring:
    driver: bridge

volumes:
  postgres-data:
  prometheus-data:
  grafana-data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
  grafana_password:
    file: ./secrets/grafana_password.txt
```

## 5. Environment-Specific Overrides

Use multiple compose files with the `-f` flag for different environments:

```bash
# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Staging
docker compose -f docker-compose.yml -f docker-compose.staging.yml up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

**Base file: `docker-compose.yml`** (shared configuration)

**Override file: `docker-compose.prod.yml`**
```yaml
services:
  api:
    image: myapp:${VERSION}
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
    restart: always
```

## 6. Essential Companion Files

### `.env` Template
```bash
# Application
NODE_ENV=production
LOG_LEVEL=info
DOMAIN=example.com

# Database
DB_NAME=myapp
DB_USER=appuser
DB_PORT=5432

# Redis
REDIS_PASSWORD=change_this_in_env_file

# Versions
APP_VERSION=1.2.3
POSTGRES_VERSION=16.1-alpine3.19
```

### `.dockerignore`
```
# Version control
.git
.gitignore
.github

# Dependencies
node_modules
venv
__pycache__

# Development
.env.local
.env.development
*.log
.vscode
.idea

# Testing
coverage
.pytest_cache

# Build artifacts
dist
build
*.pyc
```

### `Makefile` for Common Operations
```makefile
.PHONY: help up down logs ps restart clean

help:
	@echo "Available commands:"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs"
	@echo "  make restart  - Restart services"
	@echo "  make clean    - Remove all containers, volumes, and networks"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

restart:
	docker compose restart

clean:
	docker compose down -v --remove-orphans
	docker system prune -f
```

## 7. Validation & Testing

### Pre-Deployment Checklist

#### Modular Architecture (MANDATORY)
- [ ] **Separate compose files per layer** (domain, infrastructure, adapters, observability)
- [ ] **Main docker-compose.yml uses `include`** to compose modular files
- [ ] **Layer labels present** on all services (`com.example.layer`)
- [ ] **Component labels present** on all services (`com.example.component`)
- [ ] **Directory structure follows hexagonal pattern** (`compose/domain/`, `compose/infrastructure/`, etc.)
- [ ] **Shared resources in dedicated files** (networks.yml, volumes.yml, secrets.yml)
- [ ] **Environment overrides are modular** (dev/staging/prod files also use include)

#### Security & Configuration
- [ ] All images have specific version tags (no `:latest`)
- [ ] Secrets use `secrets:` or external secret management
- [ ] Health checks defined for all critical services
- [ ] Resource limits set for all services
- [ ] Networks properly isolated (internal networks for databases)
- [ ] Volumes use named volumes (not bind mounts) in production
- [ ] All services run as non-root users
- [ ] `depends_on` uses health check conditions
- [ ] Restart policies configured appropriately
- [ ] Labels added for documentation and filtering

#### Hexagonal Architecture Verification
- [ ] **Infrastructure layer isolated** (databases, caches, messaging in separate networks)
- [ ] **Domain services don't expose ports** directly (use expose, not ports)
- [ ] **Adapters handle external traffic** (gateways, proxies with public network)
- [ ] **Observability layer is optional** (can be excluded from dev environment)
- [ ] **Dependencies flow inward** (domain depends on infrastructure, not vice versa)

### Validation Commands

```bash
# Validate main compose file with all includes
docker compose config

# Validate and view final merged configuration
docker compose config --quiet && echo "✅ Valid"

# View merged configuration without interpolation (for debugging)
docker compose config --no-interpolate

# Check for configuration issues with image digests
docker compose config --resolve-image-digests

# Validate specific modular file
docker compose -f compose/infrastructure/databases.yml config

# List all services from merged composition
docker compose config --services

# List all networks from merged composition
docker compose config --volumes

# Dry-run to check all images are accessible
docker compose pull --dry-run

# Validate modular structure (check that files exist)
test -f docker-compose.yml && \
test -d compose/infrastructure && \
test -d compose/domain && \
test -d compose/adapters && \
test -d compose/shared && \
echo "✅ Modular structure valid" || \
echo "❌ Missing required directories"

# Verify all services have layer labels
docker compose config --format json | \
jq -r '.services | to_entries[] | select(.value.labels["com.example.layer"] == null) | .key' | \
(grep . && echo "❌ Services missing layer labels" || echo "✅ All services have layer labels")
```

## 8. Interaction Protocol

**User Input:** "Create a docker-compose setup for a Django app with PostgreSQL and Redis."

**Agent Response Strategy:**

1. **Analyze Context**: Django = Python web framework, needs WSGI server, static files, database migrations.

2. **Design Architecture**: Identify layers:
   - Infrastructure: PostgreSQL, Redis
   - Domain: Django API/WSGI service, worker processes
   - Adapters: Nginx reverse proxy
   - Observability: (optional) Prometheus, logging

3. **Create Modular Structure**: 
   - `compose/infrastructure/databases.yml` (PostgreSQL)
   - `compose/infrastructure/caches.yml` (Redis)
   - `compose/domain/api-services.yml` (Django/Gunicorn)
   - `compose/adapters/proxy.yml` (Nginx)
   - `compose/shared/networks.yml`, `volumes.yml`, `secrets.yml`
   - Main `docker-compose.yml` with `include` directives

4. **Apply Best Practices**: Security hardening, health checks, proper networking, resource limits.

5. **Environment Separation**: Provide environment-specific overrides (dev/staging/prod) that also use modular includes.

6. **Review Against Six Pillars**: 
   - Declarative ✓
   - Reproducible ✓ (pinned versions)
   - Secure ✓ (secrets, non-root, isolated networks)
   - Maintainable ✓ (clear structure, comments)
   - Modular ✓ (separate files per layer, include composition)
   - Hexagonal ✓ (infrastructure/domain/adapter separation)

7. **Output**: Return modular compose files + directory structure + .env template + brief explanation of design decisions.

## 9. Common Anti-Patterns to AVOID

### ❌ WRONG: Using `container_name` everywhere
```yaml
services:
  api-1:
    container_name: my-api-1  # Prevents scaling
    image: myapi:1.0
```

### ✅ CORRECT: Let Docker Compose generate names (or use deploy.replicas)
```yaml
services:
  api:
    image: myapi:1.0
    deploy:
      replicas: 3  # Now you can scale
```

### ❌ WRONG: Building images in production compose
```yaml
services:
  app:
    build: .  # Should be pre-built image
```

### ✅ CORRECT: Use pre-built, tagged images
```yaml
services:
  app:
    image: registry.company.com/app:v1.2.3
```

### ❌ WRONG: Exposing unnecessary ports
```yaml
services:
  db:
    ports:
      - "5432:5432"  # Database exposed to host
```

### ✅ CORRECT: Use internal networking only
```yaml
services:
  db:
    expose:
      - "5432"  # Only accessible to other services
    networks:
      - database
```

### ❌ WRONG: No resource limits
```yaml
services:
  app:
    image: myapp:1.0
    # No limits = can consume all host resources
```

### ✅ CORRECT: Always set limits
```yaml
services:
  app:
    image: myapp:1.0
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
```

## 10. Advanced Patterns

### A. Using Profiles for Optional Services
```yaml
services:
  app:
    image: myapp:1.0
    
  debug-tools:
    image: nicolaka/netshoot
    profiles:
      - debug
    command: sleep infinity

# Run normally: docker compose up
# Run with debug: docker compose --profile debug up
```

### B. Extension Fields (DRY principle)
```yaml
x-common-healthcheck: &common-healthcheck
  interval: 10s
  timeout: 5s
  retries: 3
  start_period: 30s

x-logging: &default-logging
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"

services:
  app:
    image: myapp:1.0
    healthcheck:
      <<: *common-healthcheck
      test: ["CMD", "curl", "-f", "http://localhost/health"]
    logging: *default-logging
```

### C. CI/CD Integration
```yaml
# docker-compose.ci.yml
services:
  test:
    build:
      context: .
      target: test
    environment:
      - CI=true
    command: npm test
    
  integration-tests:
    build: .
    depends_on:
      db-test:
        condition: service_healthy
    command: npm run test:integration
    
  db-test:
    image: postgres:16-alpine
    environment:
      - POSTGRES_DB=testdb
      - POSTGRES_USER=test
      - POSTGRES_PASSWORD=test
    tmpfs:
      - /var/lib/postgresql/data  # Fast, ephemeral storage for tests
```

## 11. Performance Optimization

### A. Build Cache Optimization
```yaml
services:
  app:
    build:
      context: .
      cache_from:
        - myapp:latest
        - myapp:${GIT_BRANCH}
      args:
        BUILDKIT_INLINE_CACHE: 1
```

### B. Parallel Service Startup
```yaml
# Use healthchecks instead of sleep delays
services:
  app:
    depends_on:
      db:
        condition: service_healthy  # Better than sleep 10
```

### C. Volume Performance (Development)
```yaml
services:
  app:
    volumes:
      - ./src:/app/src:cached  # Optimize for Mac/Windows
      # Options: consistent, cached, delegated
```

## 12. Documentation Requirements

Every docker-compose.yml MUST include:

1. **Inline Comments**: Explain non-obvious configurations
2. **README.md**: Document environment variables, secrets setup, and deployment steps
3. **Diagram**: Optional but recommended for complex microservices

```yaml
services:
  # Frontend: Serves static assets and proxies API requests
  # Security: Runs as nginx user (non-root)
  # Performance: Uses Alpine for minimal size (10MB vs 140MB)
  frontend:
    image: nginx:1.25-alpine
    # ... configuration ...
```

---

## Why This Configuration Standard Works

1. **Compose Specification Format**: Modern Docker Compose v2 uses the Compose Specification, making `version:` field obsolete and improving forward compatibility.

2. **Health Check Dependencies**: Using `condition: service_healthy` prevents race conditions and eliminates the need for retry logic in application code.

3. **Network Isolation**: Internal networks for databases prevent accidental exposure while allowing service-to-service communication.

4. **Resource Constraints**: Prevents noisy neighbor problems and ensures predictable performance in production.

5. **Secrets Management**: Using `secrets:` with external files or Docker Swarm secrets keeps sensitive data out of version control and environment variables.

6. **Extension Fields**: YAML anchors and extensions reduce duplication and make configs more maintainable.

7. **Environment Overrides**: Multiple compose files allow reusing base configuration across dev/staging/prod while customizing as needed.

8. **Modular Composition with `include`**: Breaking compose configurations into separate files per architectural layer provides:
   - **Team Autonomy**: Different teams can own and modify their layers independently without conflicts
   - **Selective Deployment**: Deploy only the layers needed for specific environments (e.g., skip observability in dev)
   - **Easier Testing**: Test each layer in isolation before integration
   - **Reduced Complexity**: Smaller, focused files are easier to understand, review, and maintain
   - **Reusability**: Share common infrastructure layers across multiple projects
   - **Version Control**: Clearer git history and easier code reviews with focused file changes

9. **Hexagonal Architecture**: Mapping docker-compose structure to hexagonal architecture principles ensures:
   - **Clear Boundaries**: Infrastructure (databases, caches) isolated from domain logic (APIs, workers)
   - **Dependency Direction**: Domain services depend on infrastructure, never the reverse
   - **Port Isolation**: Domain services use `expose` (internal), adapters use `ports` (external)
   - **Network Segmentation**: Each layer has appropriate network access (database network is internal-only)
   - **Independent Scaling**: Scale domain services without affecting infrastructure capacity
   - **Technology Flexibility**: Swap infrastructure components (e.g., PostgreSQL → MySQL) without touching domain services
   - **Security by Design**: Adapters act as security boundaries, domain core never exposed directly

---

## References & Further Reading

- [Compose Specification](https://docs.docker.com/compose/compose-file/)
- [Docker Compose `include` directive](https://docs.docker.com/compose/compose-file/14-include/)
- [Docker Compose Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [12-Factor App Methodology](https://12factor.net/)
- [Hexagonal Architecture (Ports & Adapters)](https://alistair.cockburn.us/hexagonal-architecture/)
- [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

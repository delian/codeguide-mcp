# Architecture Decision Records (ADR) Guidelines

This document provides mandatory standards for documenting architecture decisions using ADRs.

---

**Agent Profile**: The ADR Expert
**Role**: Senior Software Architect & Technical Documentation Specialist
**Objective**: Generate well-structured ADRs that capture architectural decisions and their context for future reference.
**Tools**: Markdown, ADR Tools (adr-tools), Docusaurus, GitHub/GitLab.

---

## 1. Core Philosophies: ADR-FIRST

- **A**rchived: Decisions are immutable records
- **D**ocumented: Context and reasoning captured
- **R**eviewable: Part of the code review process

---

## 2. ADR Structure (MANDATORY)

### A. Standard Template

```markdown
# ADR-NNNN: Title of Decision

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-XXXX]

## Date
YYYY-MM-DD

## Context
What is the issue that we're seeing that is motivating this decision or change?
Describe the forces at play (technical, political, social, project-specific).
Include any constraints that must be considered.

## Decision
What is the change that we're proposing and/or doing?
State the decision in full sentences with active voice.
Be specific about what we will and won't do.

## Consequences
What becomes easier or more difficult to do because of this change?

### Positive
- Benefit 1
- Benefit 2

### Negative
- Drawback 1
- Drawback 2

### Neutral
- Side effect that is neither positive nor negative

## Alternatives Considered
What other options were considered and why were they rejected?

### Alternative 1: [Name]
- Description
- Reason rejected

### Alternative 2: [Name]
- Description
- Reason rejected

## References
- Link to related documents
- Link to relevant discussions
- Link to related ADRs
```

### B. Complete Example

```markdown
# ADR-0012: Use PostgreSQL as Primary Database

## Status
Accepted

## Date
2024-01-15

## Context

Our application needs a reliable database for storing user data, transactions,
and application state. We need to choose between several database options that
can support our requirements:

**Requirements:**
- Strong consistency for financial transactions
- Support for complex queries and reporting
- Horizontal read scaling
- Proven track record in production
- Strong ecosystem and tooling
- Team familiarity

**Current situation:**
- Team has experience with PostgreSQL and MySQL
- Expected initial load: 10,000 users, 1M transactions/month
- Expected growth: 10x over 2 years
- Mix of OLTP and some analytical queries

**Constraints:**
- Budget for managed database service: $500/month initially
- Must support ACID transactions
- Need full-text search capabilities
- Prefer open-source to avoid vendor lock-in

## Decision

We will use **PostgreSQL** as our primary database, deployed on AWS RDS.

Specifically:
- PostgreSQL 15 on AWS RDS (db.r6g.large initially)
- Multi-AZ deployment for high availability
- Read replicas for reporting queries
- Use native JSONB for flexible schema portions
- Use pg_trgm extension for full-text search

We will NOT use:
- MySQL (lacks advanced features we need)
- MongoDB (need ACID guarantees)
- CockroachDB (overkill for current scale)

## Consequences

### Positive
- **Reliability**: PostgreSQL has excellent ACID compliance and data integrity
- **Features**: Rich feature set including JSONB, full-text search, CTEs, window functions
- **Ecosystem**: Excellent tooling (pgAdmin, pg_dump, extensions)
- **Scaling**: Read replicas handle reporting without impacting OLTP
- **Team skill**: Team already familiar with PostgreSQL
- **Cost**: RDS PostgreSQL is cost-effective for our scale

### Negative
- **Vendor coupling**: Some RDS-specific features may create AWS dependency
- **Write scaling**: Vertical scaling for writes has limits (acceptable for now)
- **Operational overhead**: Need to manage backups, upgrades, monitoring

### Neutral
- **Migration path**: Can migrate to Aurora PostgreSQL if needed
- **Learning curve**: Some advanced features (JSONB, CTEs) need learning

## Alternatives Considered

### Alternative 1: MySQL / Aurora MySQL
- **Description**: Popular RDBMS with AWS Aurora option
- **Pros**: Team familiarity, Aurora scaling capabilities
- **Rejected because**:
  - Weaker support for JSONB-style flexible schemas
  - Less powerful analytical features (window functions, CTEs)
  - PostgreSQL extensions are more valuable for our use case

### Alternative 2: MongoDB
- **Description**: Document database with flexible schema
- **Pros**: Schema flexibility, easy horizontal scaling
- **Rejected because**:
  - Weaker consistency guarantees (need ACID for transactions)
  - Team less familiar with MongoDB
  - Complex queries more difficult

### Alternative 3: CockroachDB
- **Description**: Distributed SQL database
- **Pros**: Horizontal write scaling, strong consistency
- **Rejected because**:
  - Overkill for our current scale
  - Higher operational complexity
  - Higher cost for our volume

### Alternative 4: SQLite
- **Description**: Embedded database
- **Rejected because**:
  - Single-machine limitation
  - Not suitable for multi-server deployment

## References
- [PostgreSQL vs MySQL Comparison](https://example.com/pg-vs-mysql)
- [AWS RDS PostgreSQL Pricing](https://aws.amazon.com/rds/postgresql/pricing/)
- [Internal RFC: Database Selection](https://internal.example.com/rfc/database)
- Related: ADR-0003 (Cloud Provider Selection)
- Supersedes: ADR-0005 (Initial Database Decision)
```

---

## 3. ADR Types (MANDATORY)

### A. Technology Selection ADR

```markdown
# ADR-0015: Use React for Frontend Framework

## Status
Accepted

## Date
2024-01-20

## Context

We need to select a frontend framework for our new customer portal.
The application will have:
- Complex interactive forms
- Real-time data updates
- Integration with existing REST APIs
- Mobile-responsive design

Team composition:
- 3 frontend developers (2 with React experience, 1 with Vue)
- Need to hire 2 more developers

## Decision

We will use **React 18** with TypeScript for the frontend.

Supporting decisions:
- State management: Zustand (simple) or React Query (server state)
- Routing: React Router v6
- Styling: Tailwind CSS
- Build tool: Vite

## Consequences

### Positive
- Large talent pool for hiring
- Extensive component ecosystem
- Team already has React experience
- TypeScript support is excellent

### Negative
- React has a steeper learning curve than Vue
- Need to choose from many state management options
- Bundle size considerations with dependencies

## Alternatives Considered

### Vue 3
- Easier learning curve
- Rejected: Smaller team expertise, smaller talent pool

### Angular
- Full-featured framework
- Rejected: Heavier, more opinionated than needed

### Svelte
- Excellent performance
- Rejected: Smaller ecosystem, hiring concerns
```

### B. Architecture Pattern ADR

```markdown
# ADR-0018: Adopt Event-Driven Architecture for Order Processing

## Status
Accepted

## Date
2024-02-01

## Context

Our order processing system currently uses synchronous API calls between services.
This causes:
- Tight coupling between services
- Cascading failures when downstream services are unavailable
- Difficulty scaling individual components
- Long response times for complex orders

## Decision

We will adopt an **event-driven architecture** for order processing using:
- Amazon EventBridge as the event bus
- Event schemas defined in JSON Schema
- At-least-once delivery with idempotent consumers
- Dead letter queues for failed events

Event flow:
1. Order Service publishes `OrderCreated` event
2. Inventory Service consumes, publishes `InventoryReserved`
3. Payment Service consumes, publishes `PaymentProcessed`
4. Fulfillment Service consumes, begins shipping

## Consequences

### Positive
- Services are loosely coupled
- Better fault isolation
- Easier to add new consumers
- Natural audit trail of events

### Negative
- Eventual consistency (not immediate)
- More complex debugging
- Need event versioning strategy
- Learning curve for team

## Alternatives Considered

### Synchronous Orchestration (Saga Pattern)
- Central orchestrator coordinates calls
- Rejected: Still has coupling, complex error handling

### Direct Service-to-Service Events
- Services publish directly to each other
- Rejected: Point-to-point coupling, harder to add consumers
```

### C. Coding Standard ADR

```markdown
# ADR-0022: Adopt Conventional Commits

## Status
Accepted

## Date
2024-02-15

## Context

Our commit messages are inconsistent, making it difficult to:
- Generate changelogs automatically
- Understand the nature of changes
- Trigger appropriate CI/CD pipelines
- Perform semantic versioning

## Decision

We will adopt **Conventional Commits** specification for all commit messages.

Format:
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature (MINOR version)
- `fix`: Bug fix (PATCH version)
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code change, no feature/fix
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

Breaking changes: Add `!` after type or `BREAKING CHANGE:` in footer.

## Consequences

### Positive
- Automated changelog generation
- Semantic versioning automation
- Clearer commit history
- Better PR reviews

### Negative
- Learning curve for team
- Need commit linting setup
- Slightly more effort per commit

## Implementation

1. Add commitlint to pre-commit hooks
2. Update contributing guidelines
3. Team training session
4. Retrofit recent commits (optional)
```

---

## 4. ADR Lifecycle (MANDATORY)

### A. Status Transitions

```markdown
## ADR Statuses

### Proposed
- Initial state when ADR is created
- Under discussion and review
- May be modified based on feedback

### Accepted
- Decision has been approved
- Implementation can proceed
- ADR is now immutable (except status)

### Deprecated
- Decision is no longer recommended
- Kept for historical reference
- New code should not follow this decision

### Superseded
- Replaced by a newer ADR
- Link to superseding ADR required
- Original reasoning preserved for context

## Status Change Rules

1. **Proposed → Accepted**: Requires team/architect approval
2. **Accepted → Deprecated**: When decision becomes outdated
3. **Accepted → Superseded**: When new ADR replaces this one
4. **Never delete ADRs**: History must be preserved
```

### B. Superseding an ADR

```markdown
# ADR-0025: Migrate from REST to GraphQL for Mobile API

## Status
Accepted

## Date
2024-03-01

## Supersedes
ADR-0010: Use REST API for Mobile Clients

## Context

ADR-0010 established REST as our mobile API protocol. After 18 months, we've
encountered issues:
- Over-fetching: Mobile clients receive unnecessary data
- Under-fetching: Multiple round trips for related data
- Versioning complexity with many endpoints

## Decision

We will migrate mobile APIs to **GraphQL** while maintaining REST for:
- Third-party integrations
- Simple CRUD operations
- Backward compatibility

[... rest of ADR ...]
```

---

## 5. ADR Organization (MANDATORY)

### A. File Structure

```
docs/
└── architecture/
    └── decisions/
        ├── README.md           # Index of all ADRs
        ├── 0001-record-architecture-decisions.md
        ├── 0002-use-typescript.md
        ├── 0003-cloud-provider-selection.md
        ├── 0004-authentication-strategy.md
        └── templates/
            ├── adr-template.md
            └── adr-template-short.md
```

### B. Index File

```markdown
# Architecture Decision Records

This directory contains the Architecture Decision Records (ADRs) for this project.

## What is an ADR?

An ADR is a document that captures an important architectural decision made
along with its context and consequences.

## ADR Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-0001](0001-record-architecture-decisions.md) | Record Architecture Decisions | Accepted | 2024-01-01 |
| [ADR-0002](0002-use-typescript.md) | Use TypeScript | Accepted | 2024-01-05 |
| [ADR-0003](0003-cloud-provider-selection.md) | Select AWS as Cloud Provider | Accepted | 2024-01-10 |
| [ADR-0004](0004-authentication-strategy.md) | Use OAuth 2.0 with PKCE | Accepted | 2024-01-15 |
| [ADR-0005](0005-initial-database.md) | Use MySQL | Superseded | 2024-01-08 |

## Creating a New ADR

1. Copy `templates/adr-template.md`
2. Rename to `NNNN-title-with-dashes.md`
3. Fill in all sections
4. Submit PR for review
5. Update this index after approval
```

---

## 6. ADR Review Process (MANDATORY)

### A. Review Checklist

```markdown
## ADR Review Checklist

### Content Quality
- [ ] Context clearly explains the problem
- [ ] Decision is specific and actionable
- [ ] Consequences are balanced (pros and cons)
- [ ] Alternatives are genuinely considered
- [ ] Technical accuracy verified

### Format
- [ ] Follows standard template
- [ ] Proper markdown formatting
- [ ] Links are valid
- [ ] Date is correct
- [ ] Status is appropriate

### Impact Assessment
- [ ] Affected systems identified
- [ ] Migration path considered (if applicable)
- [ ] Cost implications noted
- [ ] Timeline feasible
- [ ] Security implications addressed

### Stakeholder Review
- [ ] Technical lead approved
- [ ] Affected team members consulted
- [ ] Security team reviewed (if applicable)
- [ ] Product owner informed (if applicable)
```

### B. PR Template for ADRs

```markdown
## ADR Submission

### Summary
Brief description of the decision being proposed.

### Checklist
- [ ] ADR follows template structure
- [ ] All sections completed
- [ ] Alternatives genuinely evaluated
- [ ] Impact assessment complete
- [ ] Related ADRs linked

### Reviewers
- [ ] @tech-lead
- [ ] @affected-team-member
- [ ] @security-team (if applicable)

### Discussion Points
List any specific points that need discussion or clarification.
```

---

## 7. ADR Tools (MANDATORY)

### A. adr-tools Commands

```bash
# Install adr-tools
brew install adr-tools

# Initialize ADR directory
adr init docs/architecture/decisions

# Create new ADR
adr new "Use PostgreSQL as Primary Database"
# Creates: docs/architecture/decisions/0012-use-postgresql-as-primary-database.md

# Link ADRs
adr link 12 "Supersedes" 5 "Superseded by"
# Adds links between ADRs

# Generate TOC
adr generate toc > docs/architecture/decisions/README.md

# List all ADRs
adr list
```

### B. Custom Script

```bash
#!/bin/bash
# scripts/new-adr.sh

# Get next ADR number
LAST_NUM=$(ls docs/architecture/decisions/*.md 2>/dev/null | \
  grep -oP '\d{4}' | sort -n | tail -1)
NEXT_NUM=$(printf "%04d" $((10#${LAST_NUM:-0} + 1)))

# Get title from argument
TITLE="$1"
if [ -z "$TITLE" ]; then
  echo "Usage: ./new-adr.sh 'Title of Decision'"
  exit 1
fi

# Create filename
FILENAME=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
FILEPATH="docs/architecture/decisions/${NEXT_NUM}-${FILENAME}.md"

# Create ADR from template
cat > "$FILEPATH" << EOF
# ADR-${NEXT_NUM}: ${TITLE}

## Status
Proposed

## Date
$(date +%Y-%m-%d)

## Context
[Describe the context and problem]

## Decision
[Describe the decision]

## Consequences

### Positive
- [Benefit]

### Negative
- [Drawback]

## Alternatives Considered

### Alternative 1: [Name]
- Description
- Reason rejected

## References
- [Links]
EOF

echo "Created: $FILEPATH"
```

---

## 8. Deployment Checklist

### Creating ADRs
- [ ] Use standard template
- [ ] Include all sections
- [ ] Be specific in decision
- [ ] Consider alternatives fairly
- [ ] Link related ADRs

### Reviewing ADRs
- [ ] Verify technical accuracy
- [ ] Check completeness
- [ ] Validate alternatives
- [ ] Assess impact

### Maintaining ADRs
- [ ] Update status when needed
- [ ] Never delete ADRs
- [ ] Keep index current
- [ ] Review annually

---

## 9. Quick Reference

```markdown
## ADR Statuses
- Proposed → Under discussion
- Accepted → Approved for implementation
- Deprecated → No longer recommended
- Superseded → Replaced by newer ADR

## Essential Sections
1. Title (ADR-NNNN: Brief Title)
2. Status
3. Date
4. Context (Why?)
5. Decision (What?)
6. Consequences (So what?)
7. Alternatives (What else?)

## Naming Convention
NNNN-brief-title-with-dashes.md
Example: 0015-use-react-for-frontend.md

## Key Principles
- ADRs are immutable after acceptance
- Document decisions, not just outcomes
- Include rejected alternatives
- Link related decisions
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Architecture Team

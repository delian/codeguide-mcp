# deduplicate_code
description: Find and safely refactor duplicated/abstractable code across the whole repo (any language or infra), then record the lessons.
role: assistant
type: text
You are an expert Senior Engineer. Deduplicate and abstract code across the ENTIRE repository to minimize the surface area that can produce bugs. Check every part of the code — application and infrastructure alike, no matter the stack: TypeScript, Python, Go, shell, Dockerfiles, docker-compose, CI/CD pipelines, IaC, and config. Follow these steps:
1. Analyze the full source, architecture and infrastructure stack; map every language and artifact type in play.
2. (Recommended) Run the `bug_hunt` prompt first so existing bugs are double-checked and cannot be carried into — or masked by — the deduplication effort.
3. Fetch and apply the relevant coding guides from this MCP before changing code: read `guides://list`, then the matching guides for each stack (the language guide plus e.g. `designpatterns.md`, `hexagonal.md`, `comments.md`). Honor CLAUDE.md and CLAUDE.local.md.
4. Identify every instance of duplicated, near-duplicated, copy-pasted, or otherwise abstractable code across ALL of it — not just app code, but Dockerfiles, docker-compose, CI/CD, IaC, scripts, and config too.
5. Refactor to remove the duplication: extract shared functions, modules, libraries, base images, or templates. You are free to refactor the application and move code into new modules or libraries during this session to create a single source of truth.
6. Keep every change behavior-preserving: never break the build, compilation, runtime behavior, lints, types, or tests. Where a duplicate is not yet covered, add a test before collapsing it.
7. After each change verify the build/compile succeeds, linters and type checks pass, and the full unit/integration suite is green; revert anything that cannot be proven safe.
8. Keep the result compact, readable, and maintainable; for each deduplication, briefly explain how it improves the codebase and note any trade-offs, referencing the applicable coding guide.
9. After the deduplication completes, update (per CLAUDE.md and CLAUDE.local.md) `docs/lessons.md` and `lessons-summary.md` with the lessons learned, so the same duplication is not reintroduced.

description: Provide the duplication-and-abstraction request.
role: user
type: text
Deduplicate and abstract the codebase. Scope (paths/areas, or leave blank for the whole repo):

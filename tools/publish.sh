#!/usr/bin/env bash
#
# Package and publish codeguide-mcp to the MCP Registry, which is what feeds the
# VS Code Extensions "@mcp" list (via the GitHub MCP Registry downstream of it).
#
# Steps performed:
#   1. Preflight  — required tools, docker login, server.json/pyproject version agreement
#   2. Build      — container image tagged :VERSION and :latest
#   3. Push       — both tags to Docker Hub (the registry pulls :VERSION to verify ownership)
#   4. Validate   — server.json against the live registry schema
#   5. Publish    — mcp-publisher publish
#   6. Verify     — read the entry back out of the registry API
#
# Authentication is interactive and deliberately NOT automated here: run
#   mcp-publisher login github
# once beforehand. It uses a device-code flow that needs a browser.
#
# Usage:
#   tools/publish.sh                  # build, push, publish the version in server.json
#   tools/publish.sh --version 0.2.0  # override the version (updates server.json + pyproject)
#   tools/publish.sh --dry-run        # preflight + build + validate, no push, no publish
#   tools/publish.sh --skip-build     # reuse images already on Docker Hub
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="delian/codeguide-mcp"
SERVER_JSON="server.json"
VERSION=""
DRY_RUN=0
SKIP_BUILD=0

die() { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:-}"; [[ -n "$VERSION" ]] || die "--version needs a value"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

# --- 1. Preflight ------------------------------------------------------------
step "Preflight"

for cmd in docker jq mcp-publisher; do
  command -v "$cmd" >/dev/null 2>&1 || die "$cmd is not installed.
  docker:         https://docs.docker.com/get-docker/
  jq:             your package manager
  mcp-publisher:  https://modelcontextprotocol.io/registry/quickstart#step-3-install-mcp-publisher"
done

[[ -f "$SERVER_JSON" ]] || die "$SERVER_JSON not found in $REPO_ROOT"

SERVER_NAME="$(jq -r '.name' "$SERVER_JSON")"
[[ "$SERVER_NAME" != "null" && -n "$SERVER_NAME" ]] || die "$SERVER_JSON has no .name"

if [[ -n "$VERSION" ]]; then
  note "Setting version to $VERSION in $SERVER_JSON and pyproject.toml"
  tmp="$(mktemp)"
  jq --arg v "$VERSION" \
    '.version = $v
     | (.packages // []) |= map(
         if .registryType == "oci"
         then .identifier = ((.identifier | split(":")[0]) + ":" + $v)
         else . end)' \
    "$SERVER_JSON" > "$tmp" && mv "$tmp" "$SERVER_JSON"
  # Only the first version= line, which is the [project] one.
  sed -i "0,/^version = /s/^version = .*/version = \"$VERSION\"/" pyproject.toml
else
  VERSION="$(jq -r '.version' "$SERVER_JSON")"
fi
[[ "$VERSION" != "null" && -n "$VERSION" ]] || die "could not determine version"

PYPROJECT_VERSION="$(sed -n '0,/^version = /s/^version = "\(.*\)"/\1/p' pyproject.toml)"
[[ "$PYPROJECT_VERSION" == "$VERSION" ]] \
  || die "version mismatch: $SERVER_JSON has $VERSION, pyproject.toml has $PYPROJECT_VERSION.
Re-run with --version $VERSION to sync them."

# The registry proves image ownership via this label; a mismatch fails publish.
LABEL_IN_DOCKERFILE="$(grep -oP 'io\.modelcontextprotocol\.server\.name="\K[^"]+' Dockerfile || true)"
[[ "$LABEL_IN_DOCKERFILE" == "$SERVER_NAME" ]] \
  || die "Dockerfile LABEL io.modelcontextprotocol.server.name is '${LABEL_IN_DOCKERFILE:-missing}'
but $SERVER_JSON .name is '$SERVER_NAME'. They must match exactly."

OCI_IDENTIFIER="$(jq -r '(.packages // [])[] | select(.registryType=="oci") | .identifier' "$SERVER_JSON")"
if [[ -n "$OCI_IDENTIFIER" ]]; then
  EXPECTED="docker.io/$IMAGE:$VERSION"
  [[ "$OCI_IDENTIFIER" == "$EXPECTED" ]] \
    || die "OCI identifier is '$OCI_IDENTIFIER' but expected '$EXPECTED'"
fi

if (( ! DRY_RUN )); then
  docker system info 2>/dev/null | grep -q 'Username' \
    || jq -e '.auths["https://index.docker.io/v1/"]' ~/.docker/config.json >/dev/null 2>&1 \
    || die "not logged in to Docker Hub — run: docker login"
fi

note "server:  $SERVER_NAME"
note "version: $VERSION"
note "image:   $IMAGE:$VERSION (and :latest)"
(( DRY_RUN )) && note "mode:    DRY RUN (no push, no publish)"

# --- 2/3. Build and push -----------------------------------------------------
if (( SKIP_BUILD )); then
  step "Build/push skipped (--skip-build)"
else
  step "Building $IMAGE:$VERSION"
  docker build . -t "$IMAGE:$VERSION" -t "$IMAGE:latest"

  BUILT_LABEL="$(docker inspect "$IMAGE:$VERSION" \
    --format '{{index .Config.Labels "io.modelcontextprotocol.server.name"}}')"
  [[ "$BUILT_LABEL" == "$SERVER_NAME" ]] \
    || die "built image label is '$BUILT_LABEL', expected '$SERVER_NAME'"
  note "ownership label verified in image"

  if (( DRY_RUN )); then
    step "Push skipped (--dry-run)"
  else
    step "Pushing to Docker Hub"
    docker push "$IMAGE:$VERSION"
    docker push "$IMAGE:latest"
  fi
fi

# --- 4. Validate -------------------------------------------------------------
step "Validating $SERVER_JSON against the registry"
mcp-publisher validate

# --- 5. Publish --------------------------------------------------------------
if (( DRY_RUN )); then
  step "Publish skipped (--dry-run)"
  note "Re-run without --dry-run to publish."
  exit 0
fi

step "Publishing to the MCP Registry"
if ! mcp-publisher publish; then
  die "publish failed.
If the error mentions authentication, run:  mcp-publisher login github
If it mentions permissions, your namespace must match your GitHub account:
  '$SERVER_NAME' requires logging in as the owner of 'io.github.<username>'."
fi

# --- 6. Verify ---------------------------------------------------------------
step "Verifying the registry entry"
sleep 2
FOUND="$(curl -fsS "https://registry.modelcontextprotocol.io/v0.1/servers?search=$SERVER_NAME" \
  | jq -r --arg n "$SERVER_NAME" '[.servers[]? | select(.name == $n)] | length')"
if [[ "$FOUND" == "0" ]]; then
  note "Not visible in search yet — indexing can lag. Check again with:"
  note "  curl 'https://registry.modelcontextprotocol.io/v0.1/servers?search=$SERVER_NAME' | jq"
else
  note "Published: $SERVER_NAME v$VERSION is live in the registry"
fi

cat <<EOF

Next:
  * The GitHub MCP Registry ingests from the official registry, which is what
    populates the VS Code Extensions "@mcp" list. If the entry does not appear
    there after ingestion, request inclusion: partnerships@github.com
  * Verify a client can reach it:
      uv run python verify_server.py --http <your-remote-url>
EOF

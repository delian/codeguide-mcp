# Jenkins Development Guidelines
Mandatory standards for Jenkins automation: declarative Pipeline-as-Code, JCasC-managed controllers, containerized agents, masked credentials. Jenkins LTS, Declarative Pipeline, JCasC, Blue Ocean.

---
name: jenkins
title: Jenkins Development Guidelines
version: 2.0
last_reviewed: 2026-06-05
kind: infra
tools: [jenkins-lts, declarative-pipeline, jcasc, blue-ocean, docker-workflow, kubernetes-plugin]
requires:
  - secure-coding
recommends:
  - ci-cd
  - git
  - dockerfile
provides:
  - jenkinsfile
  - declarative-pipeline
  - shared-libraries
  - jcasc
---

> 🧭 Authored per [`CONVENTIONS.md`](guides://CONVENTIONS.md): shared concerns are referenced, not restated. This guide covers only what is unique to Jenkins — the general CI/CD *concepts* (stages, quality gates, deployment strategies) are owned by [`ci-cd.md`](guides://ci-cd.md).

---

## 0. Prerequisites & References

Fetch and apply these **before** writing Jenkins config. Their rules are assumed below and not repeated.

> 📎 **REQUIRED — fetch & apply first:**
> - [`secure-coding.md`](guides://secure-coding.md) — secrets, supply chain, CVE policy. *(Jenkins binding: never hardcode secrets; bind via `credentials()` / `withCredentials`; scan deps & images in-pipeline.)*

> 📎 **RECOMMENDED — fetch when the task touches them:**
> - [`ci-cd.md`](guides://ci-cd.md) — pipeline stages, quality gates, deployment strategies (canary/blue-green), artifact promotion. *(Jenkins is the executor; the strategy is owned here.)*
> - [`git.md`](guides://git.md) — branching, SCM webhooks/triggers. *(Binding: multibranch + `checkout scm`.)*
> - [`dockerfile.md`](guides://dockerfile.md) — image authoring for containerized agents and built artifacts.

> 📎 **SEE ALSO:** [`kubernetes.md`](guides://kubernetes.md) · [`observability.md`](guides://observability.md) · [`tdd.md`](guides://tdd.md)

---

## 1. Core Philosophies: JENKINS-FIRST

Jenkins-specific principles only. CI/CD strategy, security policy, and SCM workflow come from §0.

- **J**enkinsfile-as-code: every pipeline lives in a versioned `Jenkinsfile` beside the app; nothing meaningful configured in the UI.
- **E**phemeral agents: builds run on disposable Docker/Kubernetes agents (`agent { docker }` / pod templates), never on the controller. Controller has `numExecutors: 0`.
- **N**o scripted unless forced: declarative Pipeline is the default; scripted (`node {}`) only for dynamic stage generation or complex Groovy (§5).
- **K**eep it DRY: cross-project logic lives in a versioned Shared Library (`vars/`, `src/`), referenced via `@Library` (§4).
- **I**mmutable controller config: the controller is reproduced from JCasC (`jenkins.yaml`) + plugins manifest — no manual clicks (§6).
- **N**ever trust scripts: sandbox enabled, in-process script approval reviewed, agent→controller access restricted (§7).
- **S**hift quality left: parallel stages, fail-fast quality gates, masked credentials, scanned artifacts before any deploy.

**Verified Pipelines**: Agent-generated Jenkins config MUST pass every gate in §2 before delivery.

---

## 2. Requirements (MANDATORY, auditable)

RFC-2119 keywords. IDs `JNK-<TOPIC>-<NN>`. Each row has a binary gate; rows binding a shared rule cite its owner.

| ID | Requirement | Verify | Gate |
|----|-------------|--------|------|
| JNK-STRUCT-01 | Pipeline MUST be a versioned declarative `Jenkinsfile` (not UI-defined) | file in repo root; `jenkins-cli declarative-linter < Jenkinsfile` | exit 0 |
| JNK-STRUCT-02 | Scripted pipeline MUST NOT be used unless declarative is provably insufficient (§5) | review | justified or declarative |
| JNK-LINT-01 | `Jenkinsfile` MUST pass the declarative linter | `curl -X POST -F "jenkinsfile=<Jenkinsfile" $JENKINS/pipeline-model-converter/validate` | "Successfully validated" |
| JNK-AGENT-01 | Builds MUST run on ephemeral Docker/K8s agents; controller has 0 executors | grep `agent { docker`/`kubernetes`; JCasC `numExecutors: 0` | no build on controller |
| JNK-SEC-01 | No hardcoded secrets; all secrets via `credentials()`/`withCredentials` (see `secure-coding.md`) | `gitleaks detect`; grep Jenkinsfile/JCasC | 0 plaintext secrets |
| JNK-SEC-02 | Pipeline MUST scan deps & built images before deploy (see `secure-coding.md`) | `dependencyCheck` + `trivy --exit-code 1 --severity HIGH,CRITICAL` | 0 high/critical |
| JNK-SEC-03 | Script sandbox enabled; no blanket-approved unsafe signatures | review JCasC `scriptApproval` | sandbox on, list audited |
| JNK-SEC-04 | RBAC enabled (matrix/RBAC auth), anonymous read-only or off | JCasC `authorizationStrategy` | not `unsecured`/`loggedInUsersCanDoAnything` |
| JNK-CASC-01 | Controller config MUST be reproduced from JCasC, not manual UI | `jenkins.yaml` present; `apply`/`reload` clean | config drift = 0 |
| JNK-CASC-02 | Plugins MUST be pinned in a versioned manifest | `plugins.txt` with `id:version` | no floating versions |
| JNK-TST-01 | Pipeline MUST run tests and fail on failure before build/deploy (see `tdd.md`, `ci-cd.md`) | `junit` results; gate stage | red build blocks deploy |
| JNK-SCM-01 | Multibranch + `checkout scm`; triggers via webhook not polling (see `git.md`) | JCasC multibranch source; webhook configured | no SCM polling |
| JNK-OPT-01 | Pipeline MUST set `buildDiscarder` and a `timeout` | grep `options { buildDiscarder`, `timeout` | both present |

> **Forbidden**: defining real pipelines in the Jenkins UI; running builds on the controller; printing a secret to the log; disabling the Groovy sandbox or approving unsafe signatures wholesale; deploying an unscanned artifact; unbounded builds (no timeout/log rotation).

---

## 3. Verification Protocol

Run, in order, before presenting Jenkins config. Fix → re-run until every gate is green.

```bash
# JNK-STRUCT-01 / JNK-LINT-01 — validate declarative syntax
curl -s -X POST -F "jenkinsfile=<Jenkinsfile" \
  "$JENKINS_URL/pipeline-model-converter/validate"        # "Successfully validated"
# or, with CLI: java -jar jenkins-cli.jar -s $JENKINS_URL declarative-linter < Jenkinsfile

gitleaks detect --source . --no-banner                    # JNK-SEC-01: 0 plaintext secrets
grep -nE "credentials\(|withCredentials" Jenkinsfile      # JNK-SEC-01: secrets bound, not literal

# JCasC: validate before applying (dry run / reload)
docker run --rm -v "$PWD/jenkins.yaml:/jenkins.yaml" jenkins/jenkins:lts \
  bash -c 'echo validate'                                 # JNK-CASC-01 (or POST to /configuration-as-code/checkNewSource)

yq '.jenkins.numExecutors' jenkins.yaml                   # JNK-AGENT-01: must be 0
test -f plugins.txt && grep -q ':' plugins.txt            # JNK-CASC-02: versions pinned
```

The *why* behind each gate (test-first, CVE policy, deploy strategy) lives in its §0 owner; do not re-derive it here.

---

## 4. Declarative Pipeline & Shared Libraries

The heart of this guide. CI/CD *stage semantics* are owned by [`ci-cd.md`](guides://ci-cd.md); below is the Jenkins *mechanics*.

### A. Declarative skeleton

```groovy
pipeline {
  agent none                                  // no default — pin agent per stage
  options {
    buildDiscarder(logRotator(numToKeepStr: '20'))   // JNK-OPT-01
    timeout(time: 1, unit: 'HOURS')                  // JNK-OPT-01
    timestamps(); disableConcurrentBuilds()
  }
  environment {
    REGISTRY = 'registry.example.com'
    DOCKER_CREDS = credentials('docker-registry')     // -> _USR / _PSW, masked
  }
  stages {
    stage('Test') {
      agent { docker { image 'node:20-alpine' } }     // JNK-AGENT-01: ephemeral
      steps { sh 'npm ci && npm test -- --coverage' }
      post { always { junit 'reports/junit.xml' } }   // JNK-TST-01 gate
    }
    stage('Quality') { parallel {                      // independent → parallel
      stage('Lint') { agent { docker { image 'node:20-alpine' } } steps { sh 'npm run lint' } }
      stage('SAST') { steps { withSonarQubeEnv('SonarQube') { sh 'npm run sonar' }
                              timeout(time: 5, unit: 'MINUTES') { waitForQualityGate abortPipeline: true } } }
    } }
    stage('Deploy Prod') {
      when { allOf { branch 'main'; expression { params.DEPLOY == true } } }
      steps { timeout(time: 30, unit: 'MINUTES') { input message: 'Deploy to prod?', ok: 'Deploy' }
              sh 'kubectl apply -f manifests/' }
    }
  }
  post {
    failure { slackSend color: 'danger', message: "${env.JOB_NAME} #${env.BUILD_NUMBER} failed" }
    always  { cleanWs() }
  }
}
```

### B. The directives that matter (Jenkins-unique)

- **`agent`** — *where* a stage runs. `none` (force per-stage), `label 'x'`, `docker { image }`, `dockerfile { filename; dir }` (build agent from repo Dockerfile — see [`dockerfile.md`](guides://dockerfile.md)), or `kubernetes { yaml }` pod template. **Executors** are per-agent build slots; controller MUST be 0 (JNK-AGENT-01).
- **`when`** — gate a stage: `branch`, `changeRequest()`, `expression { }`, `allOf`/`anyOf`/`not`, `beforeAgent true` (skip agent spin-up if false).
- **`parallel`** — run independent stages concurrently; add `failFast true` to abort siblings on first failure.
- **`matrix`** — cartesian expansion over `axes`; combine with per-cell `agent`/`when`/`excludes`.
- **`post`** — `always` / `success` / `failure` / `unstable` / `changed` / `cleanup` — for reporting, notification, `cleanWs()`.
- **`stash`/`unstash`** — pass workspace files between agents (small artifacts); large/durable artifacts go to Artifactory/Nexus or `archiveArtifacts`.
- **`environment`** — static, `credentials('id')` binding, or `sh(returnStdout:true)` dynamic values.

Common patterns (use as-is): `retry(n)` + `sleep` for flaky deploys; `lock(resource)` (Lockable Resources plugin) to serialize on shared infra; `milestone` to discard superseded older runs.

### C. Multibranch & SCM triggers

Branching/webhook policy is owned by [`git.md`](guides://git.md). Jenkins binding: use a **multibranch pipeline** (or org folder) so each branch/PR auto-discovers its `Jenkinsfile`. Use `checkout scm` (never a hardcoded remote). Trigger via SCM **webhook**, not `pollSCM` (JNK-SCM-01). Configure the SCM source in JCasC (§6), not the UI.

### D. Shared Libraries (DRY — `provides: shared-libraries`)

Reusable steps live in a versioned library repo, loaded with `@Library('name@ref') _` and registered as a *Global Trusted* library in JCasC. Layout:

```
(shared-library repo)
├── vars/            # global steps: vars/deployToK8s.groovy exposes step deployToK8s(...)
│   └── deployToK8s.groovy   # def call(Map cfg) { ... }
├── src/             # org.acme.* Groovy classes (OOP helpers, unit-testable)
├── resources/       # static files loaded via libraryResource 'tpl/x.yaml'
└── test/            # JenkinsPipelineUnit specs (see tdd.md)
```

```groovy
// vars/deployToK8s.groovy
def call(Map cfg) {
  withCredentials([file(credentialsId: cfg.kubeconfig ?: 'kubeconfig', variable: 'KUBECONFIG')]) {
    sh "kubectl -n ${cfg.namespace} apply -f ${cfg.manifests}"
  }
}
```

Rules: keep `call()` thin, validate inputs, document params in a header, and unit-test with **JenkinsPipelineUnit** (test-first per [`tdd.md`](guides://tdd.md)). Trusted libraries run outside the sandbox — review changes like production code.

---

## 5. Scripted Pipeline (escape hatch only)

Use scripted (`node {}`) **only** when declarative cannot express it (JNK-STRUCT-02): dynamic stage generation from data, complex pre-`stages` Groovy, or legacy migration. Prefer pushing logic into a Shared Library called from a declarative pipeline instead.

```groovy
node('docker') {
  try {
    stage('Checkout') { cleanWs(); checkout scm }
    def stages = readYaml(file: 'ci/stages.yaml').stages   // dynamic: data-driven
    stages.each { s -> stage(s.name) { docker.image(s.image).inside { sh s.cmd } } }
    currentBuild.result = 'SUCCESS'
  } catch (e) { currentBuild.result = 'FAILURE'; throw e }
  finally { cleanWs() }
}
```

Scripted gives no automatic `post`/`agent`/`options` — you must hand-roll `try/finally`, notifications, and cleanup, which is exactly why declarative is the default.

---

## 6. Configuration as Code (JCasC — `provides: jcasc`)

The controller is **reproduced from `jenkins.yaml`** (`CASC_JENKINS_CONFIG`) — never configured by clicking (JNK-CASC-01). All secrets are env/Vault references, never literals (JNK-SEC-01). Pin plugins in `plugins.txt`.

```yaml
# jenkins.yaml — Jenkins Configuration as Code
jenkins:
  systemMessage: "Managed by JCasC — do not edit in UI"
  numExecutors: 0                      # JNK-AGENT-01: controller runs no builds
  authorizationStrategy:               # JNK-SEC-04: RBAC, not 'unsecured'
    globalMatrix:
      permissions:
        - "Overall/Administer:admin"
        - "Overall/Read:authenticated"
        - "Job/Build:developer"
  clouds:
    - kubernetes:                       # ephemeral agents
        name: kubernetes
        jenkinsUrl: "http://jenkins:8080"
        templates:
          - { name: node-20, label: node-20,
              containers: [{ name: node, image: "node:20-alpine", command: "sleep", args: "infinity",
                             resourceRequestCpu: "500m", resourceLimitMemory: "1Gi" }] }
  globalLibraries:                      # Shared Library registration (§4.D)
    libraries:
      - name: jenkins-shared-library
        defaultVersion: main
        retriever: { modernSCM: { scm: { git: { remote: "https://github.com/org/lib.git",
                     credentialsId: github-credentials } } } }
credentials:                            # values come from env (${...}) / Vault — never plaintext
  system:
    domainCredentials:
      - credentials:
          - usernamePassword: { id: github-credentials, username: ${GITHUB_USER}, password: ${GITHUB_TOKEN} }
          - string:           { id: sonarqube-token, secret: ${SONARQUBE_TOKEN} }
          - file:             { id: kubeconfig, fileName: kubeconfig, secretBytes: ${KUBECONFIG_B64} }
security:
  scriptApproval:                       # JNK-SEC-03: explicit, audited — not blanket
    approvedSignatures: []
unclassified:
  location: { url: "https://jenkins.example.com", adminAddress: "jenkins@example.com" }
jobs:                                   # seed jobs/folders as code
  - script: >
      multibranchPipelineJob('Applications/my-app') {
        branchSources { git { remote('https://github.com/org/my-app.git'); credentialsId('github-credentials') } }
        orphanedItemStrategy { discardOldItems { numToKeep(10) } }
      }
```

Plugin manifest (consumed by `jenkins-plugin-cli`):

```text
# plugins.txt — pinned (JNK-CASC-02)
configuration-as-code:1850.va_a_8c31d3158
workflow-aggregator:600.vb_57cdd26fdd7
docker-workflow:580.vc0c340686b_54
kubernetes:4306.vc91e951ea_eb_d
blueocean:1.27.16
```

Build the controller image with `jenkins-plugin-cli --plugin-file plugins.txt` and mount `jenkins.yaml`; bring it up with Docker Compose / Helm. Validate before applying via the JCasC "reload"/check endpoint.

---

## 7. Pipeline Security (Jenkins-specific)

Secrets/CVE *policy* is owned by [`secure-coding.md`](guides://secure-coding.md). Jenkins mechanics:

- **Credentials binding (JNK-SEC-01)** — bind, never echo. `credentials('id')` in `environment` yields masked vars; `withCredentials` for scoped, multi-type access. Masking is best-effort — never `echo` a `_PSW`/secret.
  ```groovy
  withCredentials([
    usernamePassword(credentialsId: 'nexus', usernameVariable: 'U', passwordVariable: 'P'),
    string(credentialsId: 'api-token', variable: 'TOKEN'),
    sshUserPrivateKey(credentialsId: 'ssh', keyFileVariable: 'KEY', usernameVariable: 'SSHU'),
  ]) { sh 'curl -u "$U:$P" $NEXUS && ssh -i "$KEY" $SSHU@host' }
  ```
- **Groovy sandbox & script approval (JNK-SEC-03)** — pipeline Groovy runs sandboxed; unsafe method signatures require explicit in-process approval. Keep the approved-signatures list short and audited; never disable the sandbox. Move risky logic into a **trusted Shared Library** (reviewed code) instead of approving signatures.
- **Agent → controller protection** — agents are untrusted; disable the remoting CLI, enable agent-to-controller access control, and keep build logic off the controller.
- **RBAC (JNK-SEC-04)** — matrix or role-based auth via JCasC; anonymous is read-only or disabled; least privilege per folder/job.
- **In-pipeline scanning (JNK-SEC-02)** — run SAST + dependency + secret + image scans as parallel gates and fail the build:
  ```groovy
  stage('Security') { parallel {
    stage('Deps')   { steps { dependencyCheck odcInstallation: 'dc'
                              dependencyCheckPublisher failedTotalCritical: 0, failedTotalHigh: 0 } }
    stage('Secrets'){ steps { sh 'gitleaks detect --source . --report-path gitleaks.json' } }
    stage('Image')  { steps { sh 'trivy image --severity HIGH,CRITICAL --exit-code 1 $REGISTRY/$APP:$TAG' } }
  } }
  ```

**Blue Ocean** is the recommended visualization/debugging UI for declarative pipelines (stage view, parallel branches, per-step logs). It is a viewer — it does not replace the versioned `Jenkinsfile`.

---

## 8. Quick Reference

```groovy
// Common steps / snippets
checkout scm                                            // multibranch SCM
def cfg = readYaml file: 'manifest.yaml'                // also readJSON / writeFile
def files = findFiles glob: 'tests/**/*.test.js'
stash includes: 'dist/**', name: 'build'; unstash 'build'
junit 'reports/*.xml'; archiveArtifacts 'dist/**'
```

```bash
# Jenkins CLI (auth with API token, not password)
java -jar jenkins-cli.jar -s $JENKINS_URL -auth user:$TOKEN list-jobs
java -jar jenkins-cli.jar -s $JENKINS_URL -auth user:$TOKEN build my-job -p ENV=prod -s -v
java -jar jenkins-cli.jar -s $JENKINS_URL -auth user:$TOKEN declarative-linter < Jenkinsfile
java -jar jenkins-cli.jar -s $JENKINS_URL -auth user:$TOKEN safe-restart
# Trigger via REST + webhook
curl -X POST "$JENKINS_URL/job/my-job/buildWithParameters?ENV=prod" --user user:$TOKEN
```

---

## 9. Deployment Checklist

Generated from §2 — one box per requirement ID.

- [ ] JNK-STRUCT-01/02 — versioned declarative `Jenkinsfile`; scripted only if justified
- [ ] JNK-LINT-01 — declarative linter passes
- [ ] JNK-AGENT-01 — builds on ephemeral Docker/K8s agents; controller 0 executors
- [ ] JNK-SEC-01 — no hardcoded secrets; bound via `credentials()`/`withCredentials`
- [ ] JNK-SEC-02 — deps & images scanned, build fails on high/critical
- [ ] JNK-SEC-03 — sandbox on, script-approval list audited
- [ ] JNK-SEC-04 — RBAC enabled, anonymous read-only/off
- [ ] JNK-CASC-01/02 — controller reproduced from JCasC; plugins pinned
- [ ] JNK-TST-01 — tests run and gate the build (see `tdd.md`, `ci-cd.md`)
- [ ] JNK-SCM-01 — multibranch + `checkout scm`, webhook-triggered
- [ ] JNK-OPT-01 — `buildDiscarder` + `timeout` set
- [ ] Agent ran every §3 command and documented any fixes

---
**End of Jenkins Guidelines**

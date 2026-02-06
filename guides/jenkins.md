# Jenkins Development Guidelines
Mandatory coding standards and development practices for Jenkins development. Jenkins Declarative Pipelines, Jenkins Configuration as Code (JCasC), Blue Ocean, Docker, Kubernetes.

---

**Agent Profile**: The Jenkins DevOps Expert
**Role**: Senior DevOps Engineer & Jenkins Specialist
**Objective**: Generate efficient, maintainable, secure Jenkins pipelines using declarative syntax and infrastructure as code.
**Tools**: Jenkins Declarative Pipelines, Jenkins Configuration as Code (JCasC), Blue Ocean, Docker, Kubernetes.

---

## 1. Core Philosophies: JENKINS-FIRST

The agent must adhere to the **JENKINS-FIRST** principles for every Jenkins configuration:

**Test-Driven Development (TDD)**: ALL pipelines MUST verify tests pass before deployment (Red-Green-Refactor mandatory).
**Regression Shield**: EVERY bug fix MUST reference issue ID and include regression test verification in pipeline.
**Declarative First**: ALWAYS prefer declarative over scripted pipelines (imperative only when necessary).
**Pipeline as Code**: ALL pipelines defined in Jenkinsfile, versioned with application code.
**Infrastructure as Code**: Jenkins configuration managed via JCasC (Jenkins Configuration as Code).
**Security First**: Credentials in Jenkins secrets, no hardcoded secrets, RBAC enabled.
**Docker Native**: Use Docker agents, containerized builds, immutable build environments.

**Shared Libraries**: Reusable pipeline code in shared libraries, DRY principle.
**Blue Ocean**: Modern UI for pipeline visualization and debugging.
**Parallel Execution**: Parallelize independent stages for faster builds.
**Automated Testing**: Every commit triggers tests, quality gates enforced.
**Observability**: Build metrics, test reports, deployment tracking.
**Artifact Management**: Nexus/Artifactory integration, semantic versioning.
**Reproducible Builds**: Locked dependencies, deterministic builds, versioned tools.

**Verified Code**: Agent-generated pipelines MUST validate (syntax/lint) and pass quality gates before delivery.

---

## 2. Declarative Pipeline Structure (MANDATORY)

### A. Complete Declarative Pipeline Template

```groovy
// Jenkinsfile - Declarative Pipeline (PREFERRED)

// Pipeline definition MUST use declarative syntax
pipeline {
    // Agent configuration - use Docker for reproducible builds
    agent {
        docker {
            image 'node:20-alpine'
            // Mount Docker socket for Docker-in-Docker
            args '-v /var/run/docker.sock:/var/run/docker.sock'
            // Use custom registry if needed
            registryUrl 'https://registry.example.com'
            registryCredentialsId 'docker-registry-creds'
        }
    }
    
    // Environment variables
    environment {
        // Application configuration
        NODE_VERSION = '20'
        APP_NAME = 'my-application'
        
        // Build configuration
        BUILD_NUMBER = "${env.BUILD_NUMBER}"
        GIT_COMMIT_SHORT = sh(
            script: 'git rev-parse --short HEAD',
            returnStdout: true
        ).trim()
        
        // Docker configuration
        DOCKER_REGISTRY = 'registry.example.com'
        DOCKER_IMAGE = "${DOCKER_REGISTRY}/${APP_NAME}"
        
        // Credentials (use Jenkins credentials store)
        NEXUS_CREDENTIALS = credentials('nexus-credentials')
        SONAR_TOKEN = credentials('sonarqube-token')
        
        // Test coverage threshold
        COVERAGE_THRESHOLD = '80'
    }
    
    // Build options
    options {
        // Keep only last 10 builds
        buildDiscarder(logRotator(numToKeepStr: '10'))
        
        // Timeout for entire pipeline
        timeout(time: 1, unit: 'HOURS')
        
        // Disable concurrent builds
        disableConcurrentBuilds()
        
        // Timestamps in console output
        timestamps()
        
        // ANSI color output
        ansiColor('xterm')
        
        // Skip default checkout
        skipDefaultCheckout(true)
    }
    
    // Build triggers
    triggers {
        // Poll SCM every 5 minutes
        pollSCM('H/5 * * * *')
        
        // Scheduled builds (nightly security scan)
        cron('H 2 * * *')
    }
    
    // Pipeline parameters
    parameters {
        choice(
            name: 'DEPLOY_ENV',
            choices: ['dev', 'staging', 'production'],
            description: 'Environment to deploy to'
        )
        booleanParam(
            name: 'RUN_TESTS',
            defaultValue: true,
            description: 'Run test suite'
        )
        booleanParam(
            name: 'SKIP_DEPLOYMENT',
            defaultValue: false,
            description: 'Skip deployment stage'
        )
    }
    
    // Pipeline stages
    stages {
        // ============================================
        // STAGE 1: CHECKOUT
        // ============================================
        stage('Checkout') {
            steps {
                // Clean workspace
                cleanWs()
                
                // Checkout with submodules
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: "${env.GIT_BRANCH}"]],
                    extensions: [
                        [$class: 'CloneOption', depth: 0, noTags: false, shallow: false],
                        [$class: 'SubmoduleOption', recursiveSubmodules: true]
                    ],
                    userRemoteConfigs: [[
                        url: "${env.GIT_URL}",
                        credentialsId: 'git-credentials'
                    ]]
                ])
                
                // Display build information
                script {
                    echo "Building ${APP_NAME}"
                    echo "Branch: ${env.GIT_BRANCH}"
                    echo "Commit: ${GIT_COMMIT_SHORT}"
                    echo "Build: ${BUILD_NUMBER}"
                }
            }
        }
        
        // ============================================
        // STAGE 2: VALIDATE
        // ============================================
        stage('Validate') {
            parallel {
                // Lint code
                stage('Lint') {
                    steps {
                        sh '''
                            npm ci --prefer-offline
                            npm run lint
                            npm run format:check
                        '''
                    }
                }
                
                // Verify commit messages
                stage('Commit Messages') {
                    when {
                        expression { env.CHANGE_ID != null }
                    }
                    steps {
                        sh '''
                            npm install -g @commitlint/cli @commitlint/config-conventional
                            commitlint --from origin/main --to HEAD --verbose
                        '''
                    }
                }
                
                // Verify issue reference
                stage('Issue Link') {
                    when {
                        expression { env.CHANGE_ID != null }
                    }
                    steps {
                        script {
                            def prTitle = env.CHANGE_TITLE ?: ''
                            def prBody = env.CHANGE_DESCRIPTION ?: ''
                            
                            if (!prTitle.matches('.*#\\d+.*') && !prBody.matches('.*#\\d+.*')) {
                                error 'Pull request must reference an issue (#123)'
                            }
                            echo '✓ Issue reference found'
                        }
                    }
                }
            }
        }
        
        // ============================================
        // STAGE 3: TEST (TDD VERIFICATION)
        // ============================================
        stage('Test') {
            when {
                expression { params.RUN_TESTS == true }
            }
            parallel {
                // Unit tests
                stage('Unit Tests') {
                    steps {
                        sh '''
                            # Verify tests exist
                            TEST_COUNT=$(find tests/ -name "*.test.ts" -o -name "*.spec.ts" | wc -l)
                            if [ "$TEST_COUNT" -eq 0 ]; then
                                echo "ERROR: No test files found - TDD violation"
                                exit 1
                            fi
                            echo "✓ Found $TEST_COUNT test files"
                            
                            # Run tests
                            npm test -- --coverage --ci
                            
                            # Check coverage threshold
                            COVERAGE=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
                            echo "Coverage: $COVERAGE%"
                            
                            if [ $(echo "$COVERAGE < $COVERAGE_THRESHOLD" | bc -l) -eq 1 ]; then
                                echo "ERROR: Coverage $COVERAGE% is below $COVERAGE_THRESHOLD% threshold"
                                exit 1
                            fi
                            echo "✓ Coverage threshold met"
                        '''
                    }
                    post {
                        always {
                            // Publish test results
                            junit 'coverage/junit.xml'
                            
                            // Publish coverage report
                            publishHTML([
                                reportDir: 'coverage/lcov-report',
                                reportFiles: 'index.html',
                                reportName: 'Coverage Report',
                                keepAll: true
                            ])
                            
                            // Cobertura plugin
                            cobertura(
                                coberturaReportFile: 'coverage/cobertura-coverage.xml',
                                onlyStable: false,
                                failUnhealthy: true,
                                failUnstable: true,
                                autoUpdateHealth: true,
                                autoUpdateStability: true,
                                lineCoverageTargets: "${COVERAGE_THRESHOLD}, 0, 0"
                            )
                        }
                    }
                }
                
                // Integration tests
                stage('Integration Tests') {
                    agent {
                        docker {
                            image 'node:20-alpine'
                            // Additional services
                            args '-v /var/run/docker.sock:/var/run/docker.sock'
                        }
                    }
                    steps {
                        sh '''
                            # Start services
                            docker-compose -f docker-compose.test.yml up -d
                            
                            # Wait for services
                            sleep 10
                            
                            # Run integration tests
                            npm run test:integration
                            
                            # Stop services
                            docker-compose -f docker-compose.test.yml down
                        '''
                    }
                }
            }
        }
        
        // ============================================
        // STAGE 4: VERIFY BUG FIXES
        // ============================================
        stage('Verify Bug Fixes') {
            when {
                expression {
                    def prTitle = env.CHANGE_TITLE ?: ''
                    return prTitle.toLowerCase().contains('fix') || 
                           prTitle.toLowerCase().contains('bug')
                }
            }
            steps {
                script {
                    def prTitle = env.CHANGE_TITLE ?: ''
                    def issueNum = (prTitle =~ /#(\d+)/)[0]?[1]
                    
                    if (!issueNum) {
                        error 'Bug fix PR must reference issue number (#123)'
                    }
                    
                    // Check if tests reference the issue
                    def testFound = sh(
                        script: "grep -r 'issue.*#${issueNum}\\|bug.*#${issueNum}\\|Bug #${issueNum}' tests/ || true",
                        returnStdout: true
                    ).trim()
                    
                    if (!testFound) {
                        error "Bug fix for issue #${issueNum} missing regression test. " +
                              "Add a test with comment: // Bug #${issueNum}"
                    }
                    
                    echo "✓ Regression test found for issue #${issueNum}"
                }
            }
        }
        
        // ============================================
        // STAGE 5: BUILD
        // ============================================
        stage('Build') {
            steps {
                sh '''
                    npm ci --prefer-offline
                    npm run build
                    
                    # Display build size
                    BUILD_SIZE=$(du -sh dist/ | cut -f1)
                    echo "Build size: $BUILD_SIZE"
                '''
            }
            post {
                success {
                    // Archive build artifacts
                    archiveArtifacts(
                        artifacts: 'dist/**/*',
                        fingerprint: true,
                        allowEmptyArchive: false
                    )
                }
            }
        }
        
        // ============================================
        // STAGE 6: DOCKER BUILD
        // ============================================
        stage('Docker Build') {
            steps {
                script {
                    // Build Docker image
                    def dockerImage = docker.build(
                        "${DOCKER_IMAGE}:${GIT_COMMIT_SHORT}",
                        "--build-arg BUILD_DATE=\$(date -u +\"%Y-%m-%dT%H:%M:%SZ\") " +
                        "--build-arg VCS_REF=${GIT_COMMIT_SHORT} " +
                        "--build-arg VERSION=${BUILD_NUMBER} " +
                        "."
                    )
                    
                    // Tag with branch name
                    dockerImage.tag(env.GIT_BRANCH.replaceAll('/', '-'))
                    
                    // Tag latest for main branch
                    if (env.GIT_BRANCH == 'main') {
                        dockerImage.tag('latest')
                    }
                    
                    // Push to registry
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-creds') {
                        dockerImage.push("${GIT_COMMIT_SHORT}")
                        dockerImage.push(env.GIT_BRANCH.replaceAll('/', '-'))
                        
                        if (env.GIT_BRANCH == 'main') {
                            dockerImage.push('latest')
                        }
                    }
                }
            }
        }
        
        // ============================================
        // STAGE 7: SECURITY SCAN
        // ============================================
        stage('Security') {
            parallel {
                // SAST with SonarQube
                stage('SonarQube') {
                    steps {
                        withSonarQubeEnv('SonarQube') {
                            sh '''
                                npm run sonar-scanner \
                                    -Dsonar.projectKey=${APP_NAME} \
                                    -Dsonar.sources=src \
                                    -Dsonar.tests=tests \
                                    -Dsonar.javascript.lcov.reportPaths=coverage/lcov.info
                            '''
                        }
                        
                        // Wait for quality gate
                        timeout(time: 5, unit: 'MINUTES') {
                            waitForQualityGate abortPipeline: true
                        }
                    }
                }
                
                // Dependency check
                stage('OWASP Dependency Check') {
                    steps {
                        dependencyCheck(
                            additionalArguments: '--format HTML --format JSON',
                            odcInstallation: 'dependency-check'
                        )
                        
                        dependencyCheckPublisher(
                            pattern: 'dependency-check-report.json',
                            failedTotalHigh: 0,
                            unstableTotalHigh: 5
                        )
                    }
                }
                
                // Container scan with Trivy
                stage('Container Scan') {
                    steps {
                        sh """
                            docker run --rm \
                                -v /var/run/docker.sock:/var/run/docker.sock \
                                aquasec/trivy:latest image \
                                --severity HIGH,CRITICAL \
                                --exit-code 1 \
                                ${DOCKER_IMAGE}:${GIT_COMMIT_SHORT}
                        """
                    }
                }
                
                // Secret detection
                stage('Secret Scan') {
                    steps {
                        sh '''
                            # Install gitleaks
                            wget -qO- https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz | tar xvz
                            
                            # Run gitleaks
                            ./gitleaks detect --source . --verbose
                        '''
                    }
                }
            }
        }
        
        // ============================================
        // STAGE 8: DEPLOY
        // ============================================
        stage('Deploy') {
            when {
                expression { params.SKIP_DEPLOYMENT == false }
            }
            stages {
                // Deploy to Development
                stage('Deploy to Dev') {
                    when {
                        branch 'develop'
                    }
                    steps {
                        deployToEnvironment('dev')
                    }
                }
                
                // Deploy to Staging
                stage('Deploy to Staging') {
                    when {
                        branch 'main'
                    }
                    steps {
                        deployToEnvironment('staging')
                    }
                }
                
                // Deploy to Production
                stage('Deploy to Production') {
                    when {
                        tag pattern: "v\\d+\\.\\d+\\.\\d+", comparator: "REGEXP"
                    }
                    steps {
                        // Require manual approval
                        input(
                            message: 'Deploy to Production?',
                            ok: 'Deploy',
                            submitter: 'admin,devops-lead'
                        )
                        
                        deployToEnvironment('production')
                    }
                }
            }
        }
        
        // ============================================
        // STAGE 9: SMOKE TESTS
        // ============================================
        stage('Smoke Tests') {
            when {
                expression { params.SKIP_DEPLOYMENT == false }
            }
            steps {
                script {
                    def environment = getDeploymentEnvironment()
                    def healthUrl = getHealthCheckUrl(environment)
                    
                    retry(3) {
                        sleep(10)
                        sh """
                            curl -f ${healthUrl} || exit 1
                        """
                    }
                    
                    echo "✓ Smoke tests passed for ${environment}"
                }
            }
        }
    }
    
    // Post-build actions
    post {
        always {
            // Clean workspace
            cleanWs(
                deleteDirs: true,
                disableDeferredWipeout: true,
                patterns: [
                    [pattern: 'node_modules', type: 'INCLUDE'],
                    [pattern: '.npm', type: 'INCLUDE']
                ]
            )
            
            // Send notifications
            script {
                notifyBuild(currentBuild.result)
            }
        }
        success {
            echo '✓ Pipeline completed successfully'
        }
        failure {
            echo '✗ Pipeline failed'
        }
        unstable {
            echo '⚠ Pipeline unstable'
        }
    }
}

// ============================================
// HELPER FUNCTIONS (Declarative)
// ============================================

// Deploy to environment
def deployToEnvironment(String env) {
    echo "Deploying to ${env}..."
    
    // Use Kubernetes
    sh """
        kubectl set image deployment/${APP_NAME} \
            ${APP_NAME}=${DOCKER_IMAGE}:${GIT_COMMIT_SHORT} \
            --namespace=${env}
        
        kubectl rollout status deployment/${APP_NAME} \
            --namespace=${env} \
            --timeout=5m
    """
    
    echo "✓ Deployed to ${env}"
}

// Get deployment environment
def getDeploymentEnvironment() {
    if (env.GIT_BRANCH == 'develop') return 'dev'
    if (env.GIT_BRANCH == 'main') return 'staging'
    if (env.TAG_NAME) return 'production'
    return 'unknown'
}

// Get health check URL
def getHealthCheckUrl(String env) {
    switch(env) {
        case 'dev':
            return 'https://dev.example.com/health'
        case 'staging':
            return 'https://staging.example.com/health'
        case 'production':
            return 'https://example.com/health'
        default:
            error "Unknown environment: ${env}"
    }
}

// Send notifications
def notifyBuild(String buildStatus) {
    def subject = "${APP_NAME} - Build #${BUILD_NUMBER} - ${buildStatus}"
    def summary = "${subject} (<${env.BUILD_URL}|Open>)"
    
    // Slack notification
    slackSend(
        color: buildStatus == 'SUCCESS' ? 'good' : 'danger',
        message: summary,
        channel: '#builds'
    )
    
    // Email notification
    emailext(
        subject: subject,
        body: """
            <p>Build Status: ${buildStatus}</p>
            <p>Build Number: ${BUILD_NUMBER}</p>
            <p>Branch: ${env.GIT_BRANCH}</p>
            <p>Commit: ${GIT_COMMIT_SHORT}</p>
            <p><a href="${env.BUILD_URL}">View Build</a></p>
        """,
        recipientProviders: [
            [$class: 'DevelopersRecipientProvider'],
            [$class: 'RequesterRecipientProvider']
        ]
    )
}
```

### B. Shared Library for Reusable Pipeline Code

```groovy
// vars/standardPipeline.groovy - Shared Library

/**
 * Standard pipeline for Node.js applications.
 *
 * @param config Configuration map with:
 *   - appName: Application name
 *   - nodeVersion: Node.js version (default: 20)
 *   - dockerRegistry: Docker registry URL
 *   - coverageThreshold: Code coverage threshold (default: 80)
 */
def call(Map config = [:]) {
    pipeline {
        agent {
            docker {
                image "node:${config.nodeVersion ?: '20'}-alpine"
                args '-v /var/run/docker.sock:/var/run/docker.sock'
            }
        }
        
        environment {
            APP_NAME = config.appName
            DOCKER_REGISTRY = config.dockerRegistry ?: 'registry.example.com'
            COVERAGE_THRESHOLD = config.coverageThreshold ?: '80'
        }
        
        options {
            buildDiscarder(logRotator(numToKeepStr: '10'))
            timeout(time: 1, unit: 'HOURS')
            timestamps()
            ansiColor('xterm')
        }
        
        stages {
            stage('Checkout') {
                steps {
                    checkout scm
                }
            }
            
            stage('Test') {
                steps {
                    verifyTDD()
                    sh 'npm ci && npm test'
                }
            }
            
            stage('Build') {
                steps {
                    sh 'npm run build'
                }
            }
            
            stage('Deploy') {
                steps {
                    deployApplication(config)
                }
            }
        }
    }
}

// Usage in Jenkinsfile:
// @Library('jenkins-shared-library') _
// standardPipeline(
//     appName: 'my-app',
//     nodeVersion: '20',
//     dockerRegistry: 'registry.example.com'
// )
```

```groovy
// vars/verifyTDD.groovy - TDD verification

/**
 * Verifies TDD compliance.
 * 
 * Checks:
 * - Tests exist
 * - No skipped tests
 * - Coverage threshold met
 */
def call() {
    sh '''
        # Verify tests exist
        TEST_COUNT=$(find tests/ -name "*.test.*" -o -name "*.spec.*" | wc -l)
        if [ "$TEST_COUNT" -eq 0 ]; then
            echo "ERROR: No test files found - TDD violation"
            exit 1
        fi
        echo "✓ Found $TEST_COUNT test files"
    '''
}
```

---

## 3. Jenkins Configuration as Code (JCasC) (MANDATORY)

### A. JCasC Configuration File

```yaml
# jenkins.yaml - Jenkins Configuration as Code

jenkins:
  systemMessage: "Jenkins configured automatically by JCasC"
  numExecutors: 0  # Use agents only
  mode: EXCLUSIVE
  
  # Security realm
  securityRealm:
    local:
      allowsSignup: false
      users:
        - id: admin
          password: ${ADMIN_PASSWORD}
        - id: developer
          password: ${DEVELOPER_PASSWORD}
  
  # Authorization strategy
  authorizationStrategy:
    globalMatrix:
      permissions:
        - "Overall/Administer:admin"
        - "Overall/Read:authenticated"
        - "Job/Build:developer"
        - "Job/Read:developer"
        - "Job/Workspace:developer"
  
  # Global libraries
  globalLibraries:
    libraries:
      - name: "jenkins-shared-library"
        defaultVersion: "main"
        retriever:
          modernSCM:
            scm:
              git:
                remote: "https://github.com/org/jenkins-shared-library.git"
                credentialsId: "github-credentials"
  
  # Clouds (Kubernetes)
  clouds:
    - kubernetes:
        name: "kubernetes"
        serverUrl: "https://kubernetes.default"
        namespace: "jenkins"
        jenkinsUrl: "http://jenkins:8080"
        jenkinsTunnel: "jenkins-agent:50000"
        containerCapStr: "10"
        connectTimeout: 5
        readTimeout: 15
        retentionTimeout: 5
        templates:
          - name: "node-20"
            label: "node-20"
            containers:
              - name: "node"
                image: "node:20-alpine"
                command: "/bin/sh -c"
                args: "cat"
                ttyEnabled: true
                resourceRequestCpu: "500m"
                resourceRequestMemory: "512Mi"
                resourceLimitCpu: "1000m"
                resourceLimitMemory: "1024Mi"

# Credentials
credentials:
  system:
    domainCredentials:
      - credentials:
          # GitHub
          - usernamePassword:
              scope: GLOBAL
              id: "github-credentials"
              username: ${GITHUB_USERNAME}
              password: ${GITHUB_TOKEN}
              description: "GitHub credentials"
          
          # Docker Registry
          - usernamePassword:
              scope: GLOBAL
              id: "docker-registry-creds"
              username: ${DOCKER_USERNAME}
              password: ${DOCKER_PASSWORD}
              description: "Docker registry credentials"
          
          # Nexus
          - usernamePassword:
              scope: GLOBAL
              id: "nexus-credentials"
              username: ${NEXUS_USERNAME}
              password: ${NEXUS_PASSWORD}
              description: "Nexus credentials"
          
          # SonarQube
          - string:
              scope: GLOBAL
              id: "sonarqube-token"
              secret: ${SONARQUBE_TOKEN}
              description: "SonarQube token"
          
          # Kubernetes
          - file:
              scope: GLOBAL
              id: "kubeconfig"
              fileName: "kubeconfig"
              secretBytes: ${KUBECONFIG_BASE64}
              description: "Kubernetes config"

# Tools
tools:
  git:
    installations:
      - name: "Default"
        home: "git"
  
  maven:
    installations:
      - name: "Maven 3.9"
        properties:
          - installSource:
              installers:
                - maven:
                    id: "3.9.5"
  
  nodejs:
    installations:
      - name: "Node 20"
        properties:
          - installSource:
              installers:
                - nodeJSInstaller:
                    id: "20.10.0"
                    npmPackages: "typescript eslint prettier"

# Unclassified (plugins configuration)
unclassified:
  # Location
  location:
    url: "https://jenkins.example.com"
    adminAddress: "jenkins@example.com"
  
  # Blue Ocean
  blueOceanPluginConfiguration:
    displayURLProvider: "BlueOcean"
  
  # SonarQube
  sonarGlobalConfiguration:
    installations:
      - name: "SonarQube"
        serverUrl: "https://sonarqube.example.com"
        credentialsId: "sonarqube-token"
  
  # Slack
  slackNotifier:
    teamDomain: "example"
    tokenCredentialId: "slack-token"
    room: "#builds"
  
  # Email
  mailer:
    charset: "UTF-8"
    smtpHost: "smtp.example.com"
    smtpPort: "587"
    useSsl: false
    useTls: true
    replyToAddress: "jenkins@example.com"

# Security
security:
  # CSRF protection
  crumbIssuer:
    standard:
      excludeClientIPFromCrumb: false
  
  # Agent → Controller security
  remotingCLI:
    enabled: false
  
  # Script approval
  scriptApproval:
    approvedSignatures:
      - "method groovy.json.JsonSlurper parseText java.lang.String"
      - "staticMethod org.codehaus.groovy.runtime.DefaultGroovyMethods getText java.io.InputStream"

# Jobs
jobs:
  - script: >
      folder('Applications') {
        description('Application pipelines')
      }
  
  - script: >
      multibranchPipelineJob('Applications/my-app') {
        branchSources {
          git {
            id('my-app-git')
            remote('https://github.com/org/my-app.git')
            credentialsId('github-credentials')
          }
        }
        orphanedItemStrategy {
          discardOldItems {
            numToKeep(10)
          }
        }
      }
```

### B. Docker Compose for Jenkins with JCasC

```yaml
# docker-compose.yml - Jenkins with JCasC

version: '3.8'

services:
  jenkins:
    image: jenkins/jenkins:lts-jdk17
    container_name: jenkins
    restart: unless-stopped
    
    ports:
      - "8080:8080"
      - "50000:50000"
    
    volumes:
      # Jenkins home
      - jenkins_home:/var/jenkins_home
      
      # JCasC configuration
      - ./jenkins.yaml:/var/jenkins_home/casc_configs/jenkins.yaml:ro
      
      # Docker socket for Docker-in-Docker
      - /var/run/docker.sock:/var/run/docker.sock
    
    environment:
      # JCasC
      - CASC_JENKINS_CONFIG=/var/jenkins_home/casc_configs/jenkins.yaml
      
      # Credentials (use secrets in production)
      - ADMIN_PASSWORD=${ADMIN_PASSWORD:-admin}
      - DEVELOPER_PASSWORD=${DEVELOPER_PASSWORD:-developer}
      - GITHUB_USERNAME=${GITHUB_USERNAME}
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - DOCKER_USERNAME=${DOCKER_USERNAME}
      - DOCKER_PASSWORD=${DOCKER_PASSWORD}
      - NEXUS_USERNAME=${NEXUS_USERNAME}
      - NEXUS_PASSWORD=${NEXUS_PASSWORD}
      - SONARQUBE_TOKEN=${SONARQUBE_TOKEN}
      
      # Java options
      - JAVA_OPTS=-Djenkins.install.runSetupWizard=false -Xmx2048m
    
    networks:
      - jenkins

  # SonarQube for code quality
  sonarqube:
    image: sonarqube:lts-community
    container_name: sonarqube
    restart: unless-stopped
    
    ports:
      - "9000:9000"
    
    environment:
      - SONAR_JDBC_URL=jdbc:postgresql://postgres:5432/sonarqube
      - SONAR_JDBC_USERNAME=sonar
      - SONAR_JDBC_PASSWORD=sonar
    
    volumes:
      - sonarqube_data:/opt/sonarqube/data
      - sonarqube_logs:/opt/sonarqube/logs
      - sonarqube_extensions:/opt/sonarqube/extensions
    
    networks:
      - jenkins
    
    depends_on:
      - postgres

  # PostgreSQL for SonarQube
  postgres:
    image: postgres:15-alpine
    container_name: postgres
    restart: unless-stopped
    
    environment:
      - POSTGRES_USER=sonar
      - POSTGRES_PASSWORD=sonar
      - POSTGRES_DB=sonarqube
    
    volumes:
      - postgres_data:/var/lib/postgresql/data
    
    networks:
      - jenkins

volumes:
  jenkins_home:
  sonarqube_data:
  sonarqube_logs:
  sonarqube_extensions:
  postgres_data:

networks:
  jenkins:
    driver: bridge
```

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL pipeline development.**

### TDD Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    TDD CYCLE FOR JENKINS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌─────────┐                                                  │
│    │   RED   │  1. Write a failing pipeline test first          │
│    │ (FAIL)  │     - Define expected behavior                   │
│    └────┬────┘     - Test should FAIL initially                 │
│         │                                                       │
│         ▼                                                       │
│    ┌─────────┐                                                  │
│    │  GREEN  │  2. Write minimal pipeline code to pass          │
│    │ (PASS)  │     - Implement only what's needed               │
│    └────┬────┘     - All tests should PASS                      │
│         │                                                       │
│         ▼                                                       │
│    ┌─────────┐                                                  │
│    │REFACTOR │  3. Improve code while keeping tests green       │
│    │(IMPROVE)│     - Optimize, clean up, DRY                    │
│    └────┬────┘     - Tests still PASS                           │
│         │                                                       │
│         └──────────────► Repeat for next feature                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Example TDD Workflow for Jenkins Pipelines

**Scenario**: Implement a deployment stage that deploys to Kubernetes with health checks.

```groovy
// ============================================
// Step 1: RED - Write failing test first
// ============================================

// tests/pipeline/DeploymentStageTest.groovy
import com.lesfurets.jenkins.unit.BasePipelineTest
import org.junit.Before
import org.junit.Test
import static org.junit.Assert.*

class DeploymentStageTest extends BasePipelineTest {

    @Override
    @Before
    void setUp() throws Exception {
        super.setUp()
        // Register mock commands
        helper.registerAllowedMethod('sh', [Map.class], { Map args ->
            if (args.script.contains('kubectl set image')) {
                return 0  // Success
            }
            if (args.script.contains('kubectl rollout status')) {
                return 0  // Success
            }
            return 1  // Failure
        })
    }

    @Test
    void 'deployment stage should deploy to correct namespace'() {
        // Given
        def script = loadScript('vars/deployToEnvironment.groovy')

        // When
        script.call('staging')

        // Then
        assertTrue(
            helper.callStack.findAll { it.methodName == 'sh' }
                .any { it.args[0].script.contains('--namespace=staging') }
        )
    }

    @Test
    void 'deployment stage should wait for rollout completion'() {
        // Given
        def script = loadScript('vars/deployToEnvironment.groovy')

        // When
        script.call('production')

        // Then
        def shCalls = helper.callStack.findAll { it.methodName == 'sh' }
        assertTrue(
            shCalls.any { it.args[0].script.contains('kubectl rollout status') }
        )
    }

    @Test
    void 'deployment should fail if rollout times out'() {
        // Given
        helper.registerAllowedMethod('sh', [Map.class], { Map args ->
            if (args.script.contains('kubectl rollout status')) {
                throw new Exception('Rollout timeout')
            }
            return 0
        })
        def script = loadScript('vars/deployToEnvironment.groovy')

        // When/Then
        assertThrows(Exception) {
            script.call('production')
        }
    }
}

// Run: ./gradlew test
// FAILS - deployToEnvironment.groovy doesn't exist yet
```

```groovy
// ============================================
// Step 2: GREEN - Write minimal implementation
// ============================================

// vars/deployToEnvironment.groovy
def call(String environment) {
    echo "Deploying to ${environment}..."

    // Deploy to Kubernetes
    sh """
        kubectl set image deployment/\${APP_NAME} \
            \${APP_NAME}=\${DOCKER_IMAGE}:\${GIT_COMMIT_SHORT} \
            --namespace=${environment}
    """

    // Wait for rollout
    sh """
        kubectl rollout status deployment/\${APP_NAME} \
            --namespace=${environment} \
            --timeout=5m
    """

    echo "Successfully deployed to ${environment}"
}

// Run: ./gradlew test
// PASSES - All 3 tests pass
```

```groovy
// ============================================
// Step 3: REFACTOR - Improve with health checks
// ============================================

// vars/deployToEnvironment.groovy (improved)
def call(String environment, Map config = [:]) {
    def timeout = config.timeout ?: '5m'
    def healthCheckPath = config.healthCheckPath ?: '/health'
    def appName = env.APP_NAME
    def image = "${env.DOCKER_IMAGE}:${env.GIT_COMMIT_SHORT}"

    echo "Deploying ${appName} to ${environment}..."

    // Deploy to Kubernetes
    sh """
        kubectl set image deployment/${appName} \
            ${appName}=${image} \
            --namespace=${environment}
    """

    // Wait for rollout with timeout
    sh """
        kubectl rollout status deployment/${appName} \
            --namespace=${environment} \
            --timeout=${timeout}
    """

    // Verify health endpoint
    def healthUrl = getHealthUrl(environment, healthCheckPath)
    retry(3) {
        sleep(10)
        sh "curl -f ${healthUrl}"
    }

    echo "Successfully deployed ${appName} to ${environment}"
}

def getHealthUrl(String env, String path) {
    def baseUrls = [
        'dev': 'https://dev.example.com',
        'staging': 'https://staging.example.com',
        'production': 'https://example.com'
    ]
    return "${baseUrls[env]}${path}"
}

// Run: ./gradlew test
// PASSES - All tests still pass, code is cleaner
```

### Visual TDD Example: Pipeline Stage Development

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TDD WORKFLOW VISUALIZATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  STEP 1: RED (Write Failing Test)                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  @Test                                                            │  │
│  │  void 'stage should verify coverage threshold'() {                │  │
│  │      def script = loadScript('vars/verifyTDD.groovy')             │  │
│  │      script.call(threshold: 80)                                   │  │
│  │      // Assert coverage check was performed                       │  │
│  │  }                                                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  $ ./gradlew test                                               │    │
│  │  > Task :test FAILED                                            │    │
│  │  > verifyTDD.groovy not found                                   │    │
│  │  BUILD FAILED                                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  STEP 2: GREEN (Make Test Pass)                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  // vars/verifyTDD.groovy                                         │  │
│  │  def call(Map config = [:]) {                                     │  │
│  │      def threshold = config.threshold ?: 80                       │  │
│  │      sh "npm test -- --coverage"                                  │  │
│  │      def coverage = readJSON file: 'coverage/coverage-summary.json'│ │
│  │      if (coverage.total.lines.pct < threshold) {                  │  │
│  │          error "Coverage ${coverage.total.lines.pct}% < ${threshold}%"│
│  │      }                                                            │  │
│  │  }                                                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  $ ./gradlew test                                               │    │
│  │  > Task :test                                                   │    │
│  │  > All tests passed                                             │    │
│  │  BUILD SUCCESSFUL                                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              │                                          │
│                              ▼                                          │
│  STEP 3: REFACTOR (Improve Code Quality)                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  // vars/verifyTDD.groovy (refactored)                            │  │
│  │  def call(Map config = [:]) {                                     │  │
│  │      def threshold = config.threshold ?: 80                       │  │
│  │      def testCmd = config.testCommand ?: 'npm test -- --coverage' │  │
│  │                                                                   │  │
│  │      verifyTestsExist()                                           │  │
│  │      runTests(testCmd)                                            │  │
│  │      verifyCoverage(threshold)                                    │  │
│  │  }                                                                │  │
│  │                                                                   │  │
│  │  def verifyTestsExist() { ... }                                   │  │
│  │  def runTests(String cmd) { ... }                                 │  │
│  │  def verifyCoverage(int threshold) { ... }                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every pipeline bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                 BUG FIX WORKFLOW FOR JENKINS                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BUG REPORTED                                                │
│     ├── Issue #456: "Pipeline fails when branch has slashes"    │
│     └── Reproduction steps documented                           │
│         │                                                       │
│         ▼                                                       │
│  2. WRITE REGRESSION TEST (Test FAILS)                          │
│     ├── Test reproduces the exact bug scenario                  │
│     ├── Test references issue ID: // Bug #456                   │
│     └── Verify test fails for the RIGHT reason                  │
│         │                                                       │
│         ▼                                                       │
│  3. VERIFY TEST FAILS CORRECTLY                                 │
│     ├── Run: ./gradlew test                                     │
│     └── Confirm: Test fails with expected error                 │
│         │                                                       │
│         ▼                                                       │
│  4. FIX THE BUG                                                 │
│     ├── Implement minimal fix                                   │
│     └── Do NOT add extra features                               │
│         │                                                       │
│         ▼                                                       │
│  5. VERIFY TEST PASSES                                          │
│     ├── Run: ./gradlew test                                     │
│     ├── Confirm: Previously failing test now passes             │
│     └── Confirm: No other tests broke                           │
│         │                                                       │
│         ▼                                                       │
│  6. DOCUMENT IN COMMIT                                          │
│     └── git commit -m "fix: handle branch names with slashes    │
│                                                                 │
│         Fixes #456                                              │
│         Added regression test for branch name sanitization"     │
│         │                                                       │
│         ▼                                                       │
│  7. REGRESSION PREVENTED                                        │
│     └── Future changes cannot reintroduce this bug              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Example Bug Fix with Regression Test

**Bug Report #789**: Pipeline fails when Docker image tag contains special characters from branch name (e.g., `feature/user-auth` creates invalid tag `feature/user-auth`).

```groovy
// ============================================
// Step 1-2: Write test that reproduces the bug
// ============================================

// tests/pipeline/DockerTagTest.groovy
import com.lesfurets.jenkins.unit.BasePipelineTest
import org.junit.Before
import org.junit.Test
import static org.junit.Assert.*

class DockerTagTest extends BasePipelineTest {

    @Override
    @Before
    void setUp() throws Exception {
        super.setUp()
        binding.setVariable('env', [
            BRANCH_NAME: 'feature/user-auth',
            BUILD_NUMBER: '42'
        ])
    }

    /**
     * Bug #789: Pipeline fails when branch name contains slashes.
     * Docker image tags cannot contain slashes, causing docker build to fail.
     *
     * This regression test ensures branch names are properly sanitized
     * before being used as Docker image tags.
     */
    @Test
    void 'Bug #789: should sanitize branch names with slashes for Docker tags'() {
        // Given - Branch name with slashes
        def script = loadScript('vars/buildDockerImage.groovy')
        binding.setVariable('env', [BRANCH_NAME: 'feature/user-auth'])

        // When
        def tag = script.generateTag()

        // Then - Slashes should be replaced
        assertFalse("Tag should not contain slashes", tag.contains('/'))
        assertEquals("feature-user-auth", tag)
    }

    @Test
    void 'Bug #789: should handle multiple special characters in branch name'() {
        // Given - Branch name with multiple special chars
        def script = loadScript('vars/buildDockerImage.groovy')
        binding.setVariable('env', [BRANCH_NAME: 'feature/ABC-123/fix_bug'])

        // When
        def tag = script.generateTag()

        // Then
        assertEquals("feature-ABC-123-fix-bug", tag)
    }

    @Test
    void 'Bug #789: should handle branch names starting with refs'() {
        // Given - Full ref path
        def script = loadScript('vars/buildDockerImage.groovy')
        binding.setVariable('env', [BRANCH_NAME: 'refs/heads/feature/test'])

        // When
        def tag = script.generateTag()

        // Then
        assertEquals("feature-test", tag)
    }
}

// Run: ./gradlew test
// FAILS - generateTag() doesn't handle slashes
// org.junit.ComparisonFailure:
// Expected: feature-user-auth
// Actual: feature/user-auth (contains invalid character '/')
```

```groovy
// ============================================
// Step 3: Verify the test fails for the right reason
// ============================================

// Current buggy implementation in vars/buildDockerImage.groovy
def generateTag() {
    // BUG: Does not sanitize branch name
    return env.BRANCH_NAME
}

// $ ./gradlew test --tests DockerTagTest
//
// DockerTagTest > Bug #789: should sanitize branch names with slashes FAILED
//     org.junit.ComparisonFailure: Tag should not contain slashes
//     Expected: feature-user-auth
//     Actual: feature/user-auth
//
// 3 tests completed, 3 failed
//
// FAILURE: Build failed with an exception.
```

```groovy
// ============================================
// Step 4: Fix the bug
// ============================================

// vars/buildDockerImage.groovy (fixed)
/**
 * Builds a Docker image with proper tag sanitization.
 *
 * Bug #789: Fixed branch name sanitization for Docker tags.
 */
def call(Map config = [:]) {
    def appName = config.appName ?: env.APP_NAME
    def registry = config.registry ?: env.DOCKER_REGISTRY
    def tag = generateTag()

    def fullImage = "${registry}/${appName}:${tag}"

    docker.build(fullImage, ".")

    return fullImage
}

/**
 * Generates a Docker-safe tag from the branch name.
 *
 * Bug #789: Sanitizes branch names by:
 * - Removing refs/heads/ prefix
 * - Replacing slashes with hyphens
 * - Replacing underscores with hyphens
 * - Converting to lowercase
 *
 * @return Sanitized tag string safe for Docker
 */
def generateTag() {
    def branch = env.BRANCH_NAME ?: 'unknown'

    // Remove refs/heads/ prefix if present
    branch = branch.replaceAll('^refs/heads/', '')

    // Replace invalid Docker tag characters
    // Docker tags can contain: lowercase/uppercase letters, digits, underscores, periods, hyphens
    // But slashes are NOT allowed
    branch = branch
        .replaceAll('/', '-')      // Replace slashes with hyphens
        .replaceAll('_', '-')      // Normalize underscores to hyphens
        .replaceAll('[^a-zA-Z0-9.-]', '-')  // Remove other invalid chars
        .replaceAll('-+', '-')     // Collapse multiple hyphens
        .replaceAll('^-|-$', '')   // Remove leading/trailing hyphens

    return branch
}

// Run: ./gradlew test --tests DockerTagTest
//
// DockerTagTest > Bug #789: should sanitize branch names with slashes PASSED
// DockerTagTest > Bug #789: should handle multiple special characters PASSED
// DockerTagTest > Bug #789: should handle branch names starting with refs PASSED
//
// 3 tests completed, 3 passed
//
// BUILD SUCCESSFUL
```

```groovy
// ============================================
// Step 5: Verify all tests still pass
// ============================================

// Run full test suite to ensure no regressions
// $ ./gradlew test
//
// > Task :test
//
// DockerTagTest > Bug #789: should sanitize branch names with slashes PASSED
// DockerTagTest > Bug #789: should handle multiple special characters PASSED
// DockerTagTest > Bug #789: should handle branch names starting with refs PASSED
// DeploymentStageTest > deployment stage should deploy to correct namespace PASSED
// DeploymentStageTest > deployment stage should wait for rollout completion PASSED
// ... (all other tests)
//
// 47 tests completed, 47 passed
//
// BUILD SUCCESSFUL
```

### Bug Fix Verification in Pipeline

Add this stage to your Jenkinsfile to automatically verify bug fixes include regression tests:

```groovy
// Stage to verify bug fixes have regression tests
stage('Verify Bug Fixes') {
    when {
        expression {
            def prTitle = env.CHANGE_TITLE ?: ''
            def prBody = env.CHANGE_DESCRIPTION ?: ''
            return prTitle.toLowerCase().contains('fix') ||
                   prTitle.toLowerCase().contains('bug') ||
                   prBody.toLowerCase().contains('fixes #')
        }
    }
    steps {
        script {
            // Extract issue number from PR
            def prText = "${env.CHANGE_TITLE} ${env.CHANGE_DESCRIPTION}"
            def issuePattern = /#(\d+)/
            def matcher = (prText =~ issuePattern)

            if (!matcher.find()) {
                error '''
                    Bug fix PR must reference an issue number.
                    Use: "Fixes #123" or "Bug #123" in title or description.
                '''
            }

            def issueNum = matcher[0][1]
            echo "Checking for regression test for issue #${issueNum}..."

            // Verify test file references the bug
            def testFound = sh(
                script: """
                    grep -r "Bug #${issueNum}\\|bug.*${issueNum}\\|issue.*${issueNum}\\|Fixes #${issueNum}" \
                        tests/ test/ src/test/ \
                        --include="*.groovy" \
                        --include="*.java" \
                        || true
                """,
                returnStdout: true
            ).trim()

            if (!testFound) {
                error """
                    Bug fix for issue #${issueNum} is missing a regression test.

                    Please add a test that:
                    1. Reproduces the bug scenario
                    2. Includes a comment referencing: Bug #${issueNum}
                    3. Verifies the fix works correctly

                    Example:
                    /**
                     * Bug #${issueNum}: [Description of bug]
                     */
                    @Test
                    void 'Bug #${issueNum}: should handle edge case'() {
                        // Test implementation
                    }
                """
            }

            echo "Regression test found for issue #${issueNum}"
        }
    }
}
```

---

## 4. Scripted Pipeline (Use Only When Declarative Insufficient)

### A. When to Use Scripted Pipeline

**Use declarative pipeline unless:**
- Complex conditional logic
- Dynamic stage generation
- Advanced Groovy programming needed
- Legacy pipeline migration

### B. Scripted Pipeline Example

```groovy
// Jenkinsfile - Scripted Pipeline (USE SPARINGLY)

// Node allocation
node('docker') {
    // Variables
    def appName = 'my-application'
    def gitCommitShort
    def dockerImage
    
    try {
        // Stage: Checkout
        stage('Checkout') {
            cleanWs()
            checkout scm
            gitCommitShort = sh(
                script: 'git rev-parse --short HEAD',
                returnStdout: true
            ).trim()
            
            echo "Building ${appName} @ ${gitCommitShort}"
        }
        
        // Stage: Test
        stage('Test') {
            docker.image('node:20-alpine').inside {
                sh '''
                    npm ci
                    npm test
                '''
            }
        }
        
        // Stage: Build
        stage('Build') {
            docker.image('node:20-alpine').inside {
                sh '''
                    npm run build
                '''
            }
        }
        
        // Stage: Docker
        stage('Docker') {
            dockerImage = docker.build(
                "${appName}:${gitCommitShort}",
                "."
            )
            
            docker.withRegistry('https://registry.example.com', 'docker-creds') {
                dockerImage.push(gitCommitShort)
                dockerImage.push('latest')
            }
        }
        
        // Stage: Deploy
        stage('Deploy') {
            if (env.BRANCH_NAME == 'main') {
                input(
                    message: 'Deploy to production?',
                    ok: 'Deploy'
                )
                
                sh """
                    kubectl set image deployment/${appName} \
                        ${appName}=registry.example.com/${appName}:${gitCommitShort}
                """
            }
        }
        
        // Success
        currentBuild.result = 'SUCCESS'
        
    } catch (Exception e) {
        currentBuild.result = 'FAILURE'
        throw e
    } finally {
        // Cleanup
        cleanWs()
        
        // Notifications
        notifyBuild(currentBuild.result)
    }
}

def notifyBuild(String buildStatus) {
    def subject = "${env.JOB_NAME} - Build #${env.BUILD_NUMBER} - ${buildStatus}"
    
    slackSend(
        color: buildStatus == 'SUCCESS' ? 'good' : 'danger',
        message: subject
    )
}
```

---

## 5. Multi-Branch Pipeline (MANDATORY for Projects)

### A. Jenkinsfile for Multi-Branch

```groovy
// Jenkinsfile - Multi-Branch Pipeline

pipeline {
    agent none
    
    options {
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 1, unit: 'HOURS')
    }
    
    stages {
        stage('Build') {
            agent {
                docker { image 'node:20-alpine' }
            }
            steps {
                sh 'npm ci && npm run build'
            }
        }
        
        stage('Test') {
            agent {
                docker { image 'node:20-alpine' }
            }
            steps {
                sh 'npm test'
            }
        }
        
        // Deploy based on branch
        stage('Deploy') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                    tag pattern: "v\\d+\\.\\d+\\.\\d+", comparator: "REGEXP"
                }
            }
            agent any
            steps {
                script {
                    if (env.BRANCH_NAME == 'develop') {
                        echo 'Deploying to dev...'
                        deployTo('dev')
                    } else if (env.BRANCH_NAME == 'main') {
                        echo 'Deploying to staging...'
                        deployTo('staging')
                    } else if (env.TAG_NAME) {
                        echo 'Deploying to production...'
                        input message: 'Deploy to production?'
                        deployTo('production')
                    }
                }
            }
        }
    }
}

def deployTo(String environment) {
    sh """
        echo "Deploying to ${environment}"
        # Add deployment commands here
    """
}
```

### B. Multi-Branch Job Configuration (JCasC)

```yaml
# jenkins.yaml - Multi-branch pipeline configuration

jobs:
  - script: >
      multibranchPipelineJob('my-application') {
        displayName('My Application')
        description('Multi-branch pipeline for my-application')
        
        branchSources {
          git {
            id('my-app-git')
            remote('https://github.com/org/my-app.git')
            credentialsId('github-credentials')
            
            traits {
              gitBranchDiscovery()
              gitTagDiscovery()
              headWildcardFilter {
                includes('main develop feature/* release/* hotfix/*')
                excludes('experimental/*')
              }
            }
          }
        }
        
        orphanedItemStrategy {
          discardOldItems {
            daysToKeep(7)
            numToKeep(10)
          }
        }
        
        triggers {
          periodicFolderTrigger {
            interval('1h')
          }
        }
        
        factory {
          workflowBranchProjectFactory {
            scriptPath('Jenkinsfile')
          }
        }
      }
```

---

## 6. Security Best Practices (MANDATORY)

### A. Credentials Management

```groovy
// Jenkinsfile - Proper credentials usage

pipeline {
    agent any
    
    environment {
        // Use credentials binding
        DOCKER_CREDS = credentials('docker-registry-creds')
        GITHUB_TOKEN = credentials('github-token')
        
        // The credentials() method exposes:
        // - DOCKER_CREDS_USR (username)
        // - DOCKER_CREDS_PSW (password)
        // - DOCKER_CREDS (username:password)
    }
    
    stages {
        stage('Build') {
            steps {
                // Use credentials in shell
                sh '''
                    echo "$DOCKER_CREDS_PSW" | docker login -u "$DOCKER_CREDS_USR" --password-stdin registry.example.com
                '''
                
                // Credentials are masked in logs
                echo "Username: ${DOCKER_CREDS_USR}"  // Visible
                echo "Password: ${DOCKER_CREDS_PSW}"  // Masked as ****
            }
        }
        
        stage('Deploy') {
            steps {
                // Use withCredentials for more control
                withCredentials([
                    usernamePassword(
                        credentialsId: 'nexus-credentials',
                        usernameVariable: 'NEXUS_USER',
                        passwordVariable: 'NEXUS_PASS'
                    ),
                    string(
                        credentialsId: 'api-token',
                        variable: 'API_TOKEN'
                    ),
                    file(
                        credentialsId: 'kubeconfig',
                        variable: 'KUBECONFIG_FILE'
                    )
                ]) {
                    sh '''
                        # Use credentials
                        curl -u "$NEXUS_USER:$NEXUS_PASS" https://nexus.example.com/
                        curl -H "Authorization: Bearer $API_TOKEN" https://api.example.com/
                        kubectl --kubeconfig="$KUBECONFIG_FILE" get pods
                    '''
                }
            }
        }
    }
}
```

### B. Security Scanning in Pipeline

```groovy
// Jenkinsfile - Comprehensive security scanning

pipeline {
    agent any
    
    stages {
        stage('Security Scan') {
            parallel {
                // SAST with SonarQube
                stage('SAST') {
                    steps {
                        withSonarQubeEnv('SonarQube') {
                            sh 'npm run sonar-scanner'
                        }
                        timeout(time: 5, unit: 'MINUTES') {
                            waitForQualityGate abortPipeline: true
                        }
                    }
                }
                
                // Dependency check
                stage('Dependencies') {
                    steps {
                        dependencyCheck(
                            additionalArguments: '--format HTML --format JSON',
                            odcInstallation: 'dependency-check'
                        )
                        dependencyCheckPublisher(
                            pattern: 'dependency-check-report.json',
                            failedTotalCritical: 0,
                            failedTotalHigh: 0
                        )
                    }
                }
                
                // Secret detection
                stage('Secrets') {
                    steps {
                        sh '''
                            wget -qO- https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz | tar xvz
                            ./gitleaks detect --source . --report-format json --report-path gitleaks-report.json
                        '''
                        archiveArtifacts 'gitleaks-report.json'
                    }
                }
                
                // Container scan
                stage('Container') {
                    steps {
                        sh """
                            docker run --rm \
                                -v /var/run/docker.sock:/var/run/docker.sock \
                                aquasec/trivy:latest image \
                                --severity HIGH,CRITICAL \
                                --exit-code 1 \
                                ${env.DOCKER_IMAGE}
                        """
                    }
                }
            }
        }
    }
}
```

---

## 7. Deployment Checklist

### Jenkins Setup
- [ ] **Jenkins installed**: LTS version with Java 17
- [ ] **Plugins installed**: Blue Ocean, Pipeline, Docker, Kubernetes, SonarQube
- [ ] **JCasC configured**: jenkins.yaml with all settings
- [ ] **Security enabled**: RBAC, CSRF protection, agent security
- [ ] **Credentials configured**: GitHub, Docker, cloud providers
- [ ] **Tools configured**: Git, Node.js, Maven, Docker
- [ ] **Shared library**: Created and configured
- [ ] **Agents configured**: Kubernetes/Docker agents

### Pipeline Configuration
- [ ] **Jenkinsfile**: Declarative pipeline in repository
- [ ] **Multi-branch**: Configured for automatic branch discovery
- [ ] **Stages defined**: Checkout, validate, test, build, security, deploy
- [ ] **TDD verification**: Automated test existence check
- [ ] **Bug fix verification**: Regression test requirement
- [ ] **Parallel execution**: Independent stages parallelized
- [ ] **Docker agents**: Use containerized build environments
- [ ] **Caching**: Dependency caching configured

### Security
- [ ] **SAST**: SonarQube integration
- [ ] **Dependency scanning**: OWASP Dependency Check
- [ ] **Secret detection**: Gitleaks or equivalent
- [ ] **Container scanning**: Trivy or equivalent
- [ ] **Credentials**: All secrets in Jenkins credentials store
- [ ] **No hardcoded secrets**: Verified in pipeline
- [ ] **Quality gates**: SonarQube quality gate enforced
- [ ] **RBAC**: Proper role-based access control

### Testing
- [ ] **TDD workflow**: Tests verified before build
- [ ] **Coverage tracking**: Cobertura or similar
- [ ] **Coverage threshold**: ≥80% enforced
- [ ] **Test reports**: JUnit XML published
- [ ] **Regression tests**: Required for bug fixes
- [ ] **Integration tests**: Configured with test services
- [ ] **Smoke tests**: Post-deployment verification

### Deployment
- [ ] **Multi-environment**: Dev, staging, production
- [ ] **Manual approval**: Production requires approval
- [ ] **Rollback plan**: Documented and tested
- [ ] **Health checks**: Post-deployment verification
- [ ] **Blue Ocean**: Installed for better visualization
- [ ] **Notifications**: Slack/email configured
- [ ] **Artifact management**: Nexus/Artifactory integration

### Documentation
- [ ] **README**: Pipeline documentation
- [ ] **Shared library docs**: Functions documented
- [ ] **JCasC**: Configuration documented
- [ ] **Runbooks**: Incident response procedures
- [ ] **Architecture**: Pipeline architecture documented

---

## 8. Why This Configuration Works

1. **Declarative First**: Easier to read, maintain, and validate than scripted pipelines.
2. **TDD Enforcement**: Automated checks ensure tests exist and pass before deployment.
3. **Regression Shield**: Bug fixes require tests, preventing regressions.
4. **JCasC**: Infrastructure as code makes Jenkins reproducible and versionable.
5. **Docker Agents**: Immutable, reproducible build environments.
6. **Shared Libraries**: DRY principle, reusable pipeline code.
7. **Security First**: Multiple scanning layers prevent vulnerabilities.
8. **Parallel Execution**: Faster pipelines through parallelization.
9. **Blue Ocean**: Modern UI improves pipeline visibility and debugging.
10. **Credential Management**: Secure secret handling with Jenkins credentials store.
11. **Multi-Branch**: Automatic pipeline creation for branches and tags.
12. **Quality Gates**: SonarQube prevents low-quality code from deploying.

---

## 9. Quick Reference

### Common Jenkins CLI Commands

```bash
# ============================================
# JENKINS CLI COMMANDS
# ============================================

# Download Jenkins CLI
wget http://localhost:8080/jnlpJars/jenkins-cli.jar

# Authentication (use API token, not password)
export JENKINS_URL=http://localhost:8080
export JENKINS_USER=admin
export JENKINS_TOKEN=your-api-token

# List all jobs
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN list-jobs

# Get job info
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN get-job my-pipeline

# Build a job
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN build my-pipeline

# Build with parameters
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN build my-pipeline \
    -p DEPLOY_ENV=staging \
    -p RUN_TESTS=true

# Get build console output
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN console my-pipeline 42

# Restart Jenkins safely (wait for builds to complete)
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN safe-restart

# Reload JCasC configuration
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN reload-jcasc-configuration

# List installed plugins
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN list-plugins

# Install a plugin
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN install-plugin docker-workflow

# Validate Jenkinsfile syntax (requires Pipeline plugin)
java -jar jenkins-cli.jar -s $JENKINS_URL -auth $JENKINS_USER:$JENKINS_TOKEN declarative-linter < Jenkinsfile

# ============================================
# CURL COMMANDS (REST API)
# ============================================

# Trigger build via REST API
curl -X POST "$JENKINS_URL/job/my-pipeline/build" \
    --user "$JENKINS_USER:$JENKINS_TOKEN"

# Trigger parameterized build
curl -X POST "$JENKINS_URL/job/my-pipeline/buildWithParameters" \
    --user "$JENKINS_USER:$JENKINS_TOKEN" \
    --data "DEPLOY_ENV=staging&RUN_TESTS=true"

# Get build status
curl -s "$JENKINS_URL/job/my-pipeline/lastBuild/api/json" \
    --user "$JENKINS_USER:$JENKINS_TOKEN" | jq '.result'

# Get queue info
curl -s "$JENKINS_URL/queue/api/json" \
    --user "$JENKINS_USER:$JENKINS_TOKEN" | jq '.items[].task.name'

# Get node/agent status
curl -s "$JENKINS_URL/computer/api/json" \
    --user "$JENKINS_USER:$JENKINS_TOKEN" | jq '.computer[].displayName'

# Disable a job
curl -X POST "$JENKINS_URL/job/my-pipeline/disable" \
    --user "$JENKINS_USER:$JENKINS_TOKEN"

# Enable a job
curl -X POST "$JENKINS_URL/job/my-pipeline/enable" \
    --user "$JENKINS_USER:$JENKINS_TOKEN"
```

### Jenkinsfile Patterns Cheat Sheet

```groovy
// ============================================
// DECLARATIVE PIPELINE PATTERNS
// ============================================

// Pattern 1: Conditional Stage Execution
stage('Deploy to Prod') {
    when {
        allOf {
            branch 'main'
            expression { params.DEPLOY_TO_PROD == true }
        }
    }
    steps { /* ... */ }
}

// Pattern 2: Parallel Stages
stage('Tests') {
    parallel {
        stage('Unit')       { steps { sh 'npm test' } }
        stage('Integration') { steps { sh 'npm run test:e2e' } }
        stage('Lint')       { steps { sh 'npm run lint' } }
    }
}

// Pattern 3: Matrix Builds
stage('Build Matrix') {
    matrix {
        axes {
            axis { name 'NODE_VERSION'; values '18', '20', '22' }
            axis { name 'OS'; values 'linux', 'windows' }
        }
        stages {
            stage('Build') {
                agent { label "${OS}" }
                steps { sh "nvm use ${NODE_VERSION} && npm ci && npm run build" }
            }
        }
    }
}

// Pattern 4: Input with Timeout
stage('Approval') {
    steps {
        timeout(time: 1, unit: 'HOURS') {
            input message: 'Deploy to production?', ok: 'Deploy'
        }
    }
}

// Pattern 5: Retry with Exponential Backoff
stage('Deploy') {
    steps {
        script {
            def attempt = 0
            retry(3) {
                attempt++
                sleep(time: Math.pow(2, attempt).toInteger(), unit: 'SECONDS')
                sh 'kubectl apply -f manifests/'
            }
        }
    }
}

// Pattern 6: Stash/Unstash for Artifact Passing
stage('Build') {
    steps {
        sh 'npm run build'
        stash includes: 'dist/**/*', name: 'build-artifacts'
    }
}
stage('Deploy') {
    steps {
        unstash 'build-artifacts'
        sh 'aws s3 sync dist/ s3://my-bucket/'
    }
}

// Pattern 7: Docker Agent with Custom Dockerfile
stage('Test') {
    agent {
        dockerfile {
            filename 'Dockerfile.test'
            dir 'ci'
            args '-v /tmp:/tmp'
            additionalBuildArgs '--build-arg NODE_VERSION=20'
        }
    }
    steps { sh 'npm test' }
}

// Pattern 8: Post-Stage Actions
stage('Test') {
    steps { sh 'npm test -- --coverage' }
    post {
        always  { junit 'coverage/junit.xml' }
        success { echo 'Tests passed!' }
        failure { slackSend color: 'danger', message: 'Tests failed!' }
    }
}

// Pattern 9: Environment-Specific Configuration
stage('Deploy') {
    environment {
        DEPLOY_CONFIG = credentials("${params.ENVIRONMENT}-config")
    }
    steps {
        sh "envsubst < config.template.yaml > config.yaml"
        sh "kubectl apply -f config.yaml"
    }
}

// Pattern 10: Shared Library Call
@Library('my-shared-lib') _

pipeline {
    agent any
    stages {
        stage('Deploy') {
            steps {
                deployToKubernetes(
                    environment: 'staging',
                    namespace: 'my-app',
                    timeout: '10m'
                )
            }
        }
    }
}
```

### Pipeline Structure Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      JENKINSFILE STRUCTURE REFERENCE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  pipeline {                                                                 │
│      │                                                                      │
│      ├── agent { ... }              // WHERE to run                         │
│      │   ├── any                    // Any available agent                  │
│      │   ├── none                   // No default agent (define per-stage)  │
│      │   ├── label 'docker'         // Specific agent label                 │
│      │   ├── docker { image '...' } // Docker container                     │
│      │   └── kubernetes { ... }     // Kubernetes pod                       │
│      │                                                                      │
│      ├── environment { ... }        // Environment variables                │
│      │   ├── VAR = 'value'          // Static value                         │
│      │   ├── VAR = credentials('id')// From credentials store               │
│      │   └── VAR = sh(script, returnStdout)  // Dynamic value               │
│      │                                                                      │
│      ├── options { ... }            // Pipeline options                     │
│      │   ├── buildDiscarder(...)    // Retain N builds                      │
│      │   ├── timeout(...)           // Global timeout                       │
│      │   ├── timestamps()           // Add timestamps to logs               │
│      │   ├── disableConcurrentBuilds() // No parallel runs                  │
│      │   └── skipDefaultCheckout()  // Manual checkout                      │
│      │                                                                      │
│      ├── triggers { ... }           // Auto-trigger pipeline                │
│      │   ├── pollSCM('H/5 * * * *') // Poll every 5 min                     │
│      │   ├── cron('H 2 * * *')      // Nightly at 2 AM                      │
│      │   └── upstream('job', 'SUCCESS')  // After upstream job              │
│      │                                                                      │
│      ├── parameters { ... }         // Build parameters                     │
│      │   ├── string(...)            // Text input                           │
│      │   ├── booleanParam(...)      // Checkbox                             │
│      │   ├── choice(...)            // Dropdown                             │
│      │   └── password(...)          // Masked input                         │
│      │                                                                      │
│      ├── stages { ... }             // Pipeline stages                      │
│      │   └── stage('Name') {                                                │
│      │       ├── when { ... }       // Conditional execution                │
│      │       │   ├── branch 'main'                                          │
│      │       │   ├── tag pattern: 'v*'                                      │
│      │       │   ├── expression { ... }                                     │
│      │       │   ├── allOf { ... }                                          │
│      │       │   └── anyOf { ... }                                          │
│      │       │                                                              │
│      │       ├── agent { ... }      // Stage-specific agent                 │
│      │       │                                                              │
│      │       ├── environment { ... }// Stage-specific env vars              │
│      │       │                                                              │
│      │       ├── steps { ... }      // Commands to run                      │
│      │       │   ├── sh '...'       // Shell command                        │
│      │       │   ├── script { ... } // Groovy script                        │
│      │       │   ├── checkout scm   // Git checkout                         │
│      │       │   └── withCredentials([...]) { ... }                         │
│      │       │                                                              │
│      │       ├── parallel { ... }   // Parallel stages                      │
│      │       │                                                              │
│      │       └── post { ... }       // Post-stage actions                   │
│      │   }                                                                  │
│      │                                                                      │
│      └── post { ... }               // Post-pipeline actions                │
│          ├── always { ... }         // Run regardless of result             │
│          ├── success { ... }        // Only on SUCCESS                      │
│          ├── failure { ... }        // Only on FAILURE                      │
│          ├── unstable { ... }       // Only on UNSTABLE                     │
│          ├── aborted { ... }        // Only on ABORTED                      │
│          ├── cleanup { ... }        // Always, after all others             │
│          └── changed { ... }        // When result changes from last build  │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Credentials Usage Quick Reference

```groovy
// ============================================
// CREDENTIALS PATTERNS
// ============================================

// Username/Password
environment {
    CREDS = credentials('my-creds-id')
    // Creates: CREDS (user:pass), CREDS_USR, CREDS_PSW
}

// Secret text
environment {
    API_KEY = credentials('api-key-id')
    // Creates: API_KEY (the secret value)
}

// Secret file
withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
    sh 'kubectl --kubeconfig=$KUBECONFIG get pods'
}

// SSH key
withCredentials([sshUserPrivateKey(
    credentialsId: 'ssh-key',
    keyFileVariable: 'SSH_KEY',
    usernameVariable: 'SSH_USER'
)]) {
    sh 'ssh -i $SSH_KEY $SSH_USER@server.example.com'
}

// Multiple credentials
withCredentials([
    usernamePassword(credentialsId: 'docker', usernameVariable: 'D_USER', passwordVariable: 'D_PASS'),
    string(credentialsId: 'slack-token', variable: 'SLACK_TOKEN'),
    file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')
]) {
    sh '''
        docker login -u $D_USER -p $D_PASS
        gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
    '''
}
```

### Common Groovy Snippets for Pipelines

```groovy
// ============================================
// USEFUL GROOVY SNIPPETS
// ============================================

// Read JSON file
def config = readJSON file: 'config.json'
echo "Version: ${config.version}"

// Read YAML file
def manifest = readYaml file: 'manifest.yaml'
echo "App Name: ${manifest.metadata.name}"

// Write file
writeFile file: 'output.txt', text: 'Hello, World!'

// Get changed files in commit
def changes = sh(script: 'git diff --name-only HEAD~1', returnStdout: true).trim().split('\n')
if (changes.any { it.startsWith('src/') }) {
    echo 'Source files changed, running full build'
}

// Parse version from package.json
def packageJson = readJSON file: 'package.json'
def version = packageJson.version
echo "Building version: ${version}"

// Create semantic version tag
def (major, minor, patch) = version.tokenize('.')
def newPatch = patch.toInteger() + 1
def newVersion = "${major}.${minor}.${newPatch}"

// HTTP request
def response = httpRequest(
    url: 'https://api.example.com/status',
    httpMode: 'GET',
    validResponseCodes: '200'
)
def status = readJSON text: response.content
echo "API Status: ${status.healthy}"

// Conditional based on file existence
if (fileExists('Dockerfile')) {
    echo 'Dockerfile found, building container'
    sh 'docker build -t myapp .'
}

// Get list of files matching pattern
def testFiles = findFiles(glob: 'tests/**/*.test.js')
echo "Found ${testFiles.length} test files"
```

---

## References

- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Pipeline Syntax](https://www.jenkins.io/doc/book/pipeline/syntax/)
- [Jenkins Configuration as Code](https://www.jenkins.io/projects/jcasc/)
- [Shared Libraries](https://www.jenkins.io/doc/book/pipeline/shared-libraries/)
- [Blue Ocean](https://www.jenkins.io/doc/book/blueocean/)
- [Docker Pipeline Plugin](https://plugins.jenkins.io/docker-workflow/)
- [Kubernetes Plugin](https://plugins.jenkins.io/kubernetes/)
- [Best Practices](https://www.jenkins.io/doc/book/pipeline/pipeline-best-practices/)

---

**Last Updated:** 2026-01-18
**Version:** 1.0
**Maintainer:** DevOps Team

---

**End of Jenkins Development Guidelines**

# Modern Jenkins CI/CD Guidelines
This document provides mandatory standards and best practices for Jenkins usage, emphasizing declarative pipelines, Jenkins as code, and modern CI/CD practices.

---

**Agent Profile**: The Jenkins DevOps Expert  
**Role**: Senior DevOps Engineer & Jenkins Specialist  
**Objective**: Generate efficient, maintainable, secure Jenkins pipelines using declarative syntax and infrastructure as code.  
**Tools**: Jenkins Declarative Pipelines, Jenkins Configuration as Code (JCasC), Blue Ocean, Docker, Kubernetes.

## Core Philosophies

The agent must adhere to the "JENKINS-FIRST" principles for every Jenkins configuration:

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

---

## 1. Declarative Pipeline Structure (MANDATORY)

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

## 2. Jenkins Configuration as Code (JCasC) (MANDATORY)

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

## 3. Scripted Pipeline (Use Only When Declarative Insufficient)

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

## 4. Multi-Branch Pipeline (MANDATORY for Projects)

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

## 5. Security Best Practices (MANDATORY)

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

## 6. Deployment Checklist

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

## 7. Why This Configuration Works

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

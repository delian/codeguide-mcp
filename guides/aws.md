# AWS Development Guidelines
Mandatory standards for building applications on Amazon Web Services. AWS CLI, CloudFormation, CDK, Terraform, SAM, Serverless Framework.

---

**Agent Profile**: The AWS Expert
**Role**: Senior Cloud Architect & AWS Solutions Architect
**Objective**: Generate secure, scalable, and cost-effective AWS architectures following Well-Architected Framework principles.
**Tools**: AWS CLI, CloudFormation, CDK, Terraform, SAM, Serverless Framework.

---

## 1. Core Philosophies: AWS-FIRST

- **A**utomated: Infrastructure as Code for everything
- **W**ell-Architected: Follow the five pillars
- **S**ecure: Least privilege and defense in depth

---

## 2. Well-Architected Framework (MANDATORY)

### A. The Five Pillars

```yaml
# Operational Excellence
- Automate operations with IaC
- Make frequent, small, reversible changes
- Refine procedures frequently
- Anticipate and learn from failure

# Security
- Implement strong identity foundation
- Enable traceability
- Apply security at all layers
- Automate security best practices
- Protect data in transit and at rest

# Reliability
- Automatically recover from failure
- Test recovery procedures
- Scale horizontally
- Stop guessing capacity

# Performance Efficiency
- Use serverless architectures
- Go global in minutes
- Use the right tool for the job
- Experiment more often

# Cost Optimization
- Implement cloud financial management
- Analyze and attribute expenditure
- Use cost-effective resources
- Optimize over time
```

---

## 3. IAM and Security (MANDATORY)

### A. IAM Policies

```json
// ✅ CORRECT: Least privilege policy
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::my-app-bucket/*"
      ],
      "Condition": {
        "StringEquals": {
          "s3:x-amz-acl": "bucket-owner-full-control"
        }
      }
    },
    {
      "Sid": "AllowS3BucketList",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-app-bucket"
      ]
    }
  ]
}

// ❌ WRONG: Overly permissive
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": "*"
    }
  ]
}
```

### B. IAM Roles for Services

```yaml
Resources:
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${AWS::StackName}-lambda-role'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal: { Service: lambda.amazonaws.com }
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: DynamoDBAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action: [dynamodb:GetItem, dynamodb:PutItem, dynamodb:UpdateItem, dynamodb:Query]
                Resource: [!GetAtt UsersTable.Arn, !Sub '${UsersTable.Arn}/index/*']
```

### C. Secrets Manager vs Parameter Store

```yaml
# Secrets Manager: credentials, API keys, tokens (supports auto-rotation)
# Parameter Store: config values, feature flags, hierarchical settings (/app/prod/db/host)
```

```python
import boto3, json
from functools import lru_cache

# Secrets Manager
def get_secret(secret_name: str) -> dict:
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Parameter Store with caching
ssm = boto3.client('ssm')

@lru_cache(maxsize=32)
def get_parameter(name: str, decrypt: bool = True) -> str:
    """Cached parameter retrieval (cache cleared on cold start)."""
    return ssm.get_parameter(Name=name, WithDecryption=decrypt)['Parameter']['Value']
```

### D. IAM Least-Privilege Patterns

```json
// ✅ Scope to specific resources + conditions
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": ["arn:aws:secretsmanager:us-east-1:123456789012:secret:prod/myapp/*"],
      "Condition": { "StringEquals": { "aws:RequestedRegion": "us-east-1" } }
    }
  ]
}
```

```yaml
# Permission boundaries: guardrails for delegated role creation
Resources:
  DevBoundary:
    Type: AWS::IAM::ManagedPolicy
    Properties:
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Action: ['lambda:*', 'dynamodb:*', 's3:*', 'logs:*', 'sqs:*']
            Resource: '*'
          - Effect: Deny
            Action: ['iam:CreateUser', 'iam:CreateRole']
            Resource: '*'
```

---

## 4. Lambda Functions (MANDATORY)

### A. Function Structure

```python
import json, os, boto3
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger()
tracer = Tracer()
metrics = Metrics()

# Initialize OUTSIDE handler for connection reuse across warm invocations
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])

@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def handler(event: dict, context: LambdaContext) -> dict:
    try:
        body = json.loads(event.get('body', '{}'))
        result = process_request(body)
        return {'statusCode': 200, 'headers': {'Content-Type': 'application/json'},
                'body': json.dumps(result)}
    except json.JSONDecodeError:
        return error_response(400, 'Invalid JSON')
    except Exception:
        logger.exception("Unexpected error")
        return error_response(500, 'Internal server error')

@tracer.capture_method
def process_request(data: dict) -> dict:
    table.put_item(Item={'pk': data['id'], 'sk': 'METADATA', 'data': data})
    return {'id': data['id'], 'status': 'created'}

def error_response(code: int, msg: str) -> dict:
    return {'statusCode': code, 'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': msg})}
```

### B. Lambda Configuration (SAM)

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Runtime: python3.11
    Timeout: 30
    MemorySize: 256
    Tracing: Active
    Environment:
      Variables:
        LOG_LEVEL: INFO
        POWERTOOLS_SERVICE_NAME: my-service

Resources:
  ProcessFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: !Sub '${AWS::StackName}-process'
      Handler: lambda_function.handler
      CodeUri: src/
      Environment:
        Variables:
          TABLE_NAME: !Ref DataTable
      Policies:
        - DynamoDBCrudPolicy: { TableName: !Ref DataTable }
      Events:
        ApiEvent: { Type: Api, Properties: { Path: /process, Method: POST } }
```

### C. Cold Start Optimization

```python
# ✅ CORRECT: Initialize SDK clients and connections OUTSIDE the handler
# These persist across warm invocations
import boto3
import os

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])
s3_client = boto3.client('s3')

def handler(event, context):
    # Reuses existing connections on warm starts
    table.get_item(Key={'pk': event['id'], 'sk': 'METADATA'})

# ❌ WRONG: Creating clients inside handler
def handler_bad(event, context):
    dynamodb = boto3.resource('dynamodb')  # New connection every invocation
    table = dynamodb.Table(os.environ['TABLE_NAME'])
    table.get_item(Key={'pk': event['id'], 'sk': 'METADATA'})
```

```yaml
# Cold start mitigation strategies:
# 1. ARM64 architecture (faster cold starts, lower cost)
# 2. Right-size memory (more memory = more CPU = faster init)
# 3. SnapStart for Java (eliminates cold start)
# 4. Keep deployment package small (use layers for deps)
# 5. Avoid VPC unless required (adds cold start latency)
Resources:
  OptimizedFunction:
    Type: AWS::Serverless::Function
    Properties:
      Architectures: [arm64]
      MemorySize: 512
      CodeUri: src/
```

```python
# Lazy initialization - only pay init cost when needed
_heavy_client = None

def get_heavy_client():
    global _heavy_client
    if _heavy_client is None:
        _heavy_client = SomeHeavyClient()
    return _heavy_client
```

### D. Lambda Layers for Shared Code

```yaml
Resources:
  SharedUtilsLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: !Sub '${AWS::StackName}-shared-utils'
      ContentUri: layers/shared-utils/
      CompatibleRuntimes: [python3.11, python3.12]
      CompatibleArchitectures: [x86_64, arm64]
      RetentionPolicy: Retain
    Metadata:
      BuildMethod: python3.11

  # Reference layers in functions
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.handler
      CodeUri: src/
      Layers:
        - !Ref SharedUtilsLayer
        # Or use the official Powertools managed layer:
        # - arn:aws:lambda:us-east-1:017000801446:layer:AWSLambdaPowertoolsPythonV3:7
```

```bash
# Layer directory structure
# Python: layers/shared-utils/python/shared/{__init__.py, models.py, utils.py}
# Node.js: layers/shared-utils/nodejs/{node_modules/, package.json}
```

### E. Powertools for AWS Lambda

```python
# Powertools provides: structured logging, X-Ray tracing, custom metrics,
# idempotency, batch processing, event handler routing, validation
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.utilities.idempotency import DynamoDBPersistenceLayer, idempotent
from aws_lambda_powertools.utilities.batch import BatchProcessor, EventType, batch_processor

logger = Logger(service="order-service")
tracer = Tracer(service="order-service")
metrics = Metrics(service="order-service", namespace="MyApp")

# Idempotency: prevent duplicate processing on retries
persistence = DynamoDBPersistenceLayer(table_name="IdempotencyTable")

@idempotent(persistence_store=persistence)
def process_payment(data: dict) -> dict:
    return {"payment_id": "pay_123", "status": "completed"}

# API routing
app = APIGatewayRestResolver()

@app.get("/orders/<order_id>")
@tracer.capture_method
def get_order(order_id: str):
    return {"order": fetch_order(order_id)}

@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def handler(event, context):
    return app.resolve(event, context)
```

### F. Event Source Mapping Patterns

```yaml
Resources:
  # SQS with partial batch failure reporting
  OrderProcessor:
    Type: AWS::Serverless::Function
    Properties:
      Handler: processor.handler
      CodeUri: src/
      Timeout: 300  # Must be <= SQS visibility timeout
      Events:
        SQSEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt OrderQueue.Arn
            BatchSize: 10
            FunctionResponseTypes: [ReportBatchItemFailures]  # Only retry failed records
            ScalingConfig: { MaximumConcurrency: 10 }

  # DynamoDB Streams with event filtering (reduces Lambda invocations)
  ChangeProcessor:
    Type: AWS::Serverless::Function
    Properties:
      Handler: changes.handler
      CodeUri: src/
      Events:
        DDBStream:
          Type: DynamoDB
          Properties:
            Stream: !GetAtt MainTable.StreamArn
            StartingPosition: TRIM_HORIZON
            BatchSize: 100
            BisectBatchOnFunctionError: true
            FilterCriteria:
              Filters: [{ Pattern: '{"eventName": ["INSERT", "MODIFY"]}' }]
```

### G. Lambda Function URLs

```yaml
# Function URL - direct HTTPS endpoint without API Gateway
# Use for: webhooks, simple APIs, internal service-to-service calls
Resources:
  WebhookFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: webhook.handler
      CodeUri: src/
      FunctionUrlConfig:
        AuthType: NONE  # Public endpoint (use AWS_IAM for private)
        Cors:
          AllowOrigins: ['https://example.com']
          AllowMethods: [POST]
        InvokeMode: BUFFERED  # or RESPONSE_STREAM for streaming
```

### H. Provisioned Concurrency

```yaml
# Eliminate cold starts for latency-sensitive functions
Resources:
  CriticalFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: critical.handler
      CodeUri: src/
      AutoPublishAlias: live
      ProvisionedConcurrencyConfig:
        ProvisionedConcurrentExecutions: 10
  # Auto-scale provisioned concurrency (5-50) at 70% utilization
  ScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      MaxCapacity: 50
      MinCapacity: 5
      ResourceId: !Sub 'function:${CriticalFunction}:live'
      ScalableDimension: lambda:function:ProvisionedConcurrency
      ServiceNamespace: lambda
```

---

## 5. DynamoDB (MANDATORY)

### A. Table Design

```python
# Single-table design pattern
"""
Entity Types and Key Schema:
- User:    PK=USER#<id>,       SK=METADATA
- Order:   PK=USER#<user_id>,  SK=ORDER#<order_id>
- Product: PK=PRODUCT#<id>,    SK=METADATA

GSI1 (Inverted Index): GSI1PK=SK, GSI1SK=PK (query orders by order_id)
GSI2 (Sparse Index):   GSI2PK=status, GSI2SK=created_at (query by status)
"""
import boto3
from datetime import datetime

class DynamoDBRepository:
    def __init__(self, table_name: str):
        self.table = boto3.resource('dynamodb').Table(table_name)

    def create_user(self, user_id: str, email: str, name: str) -> dict:
        item = {
            'pk': f'USER#{user_id}', 'sk': 'METADATA',
            'gsi1pk': f'EMAIL#{email}', 'gsi1sk': f'USER#{user_id}',
            'entity_type': 'USER', 'email': email, 'name': name,
            'created_at': datetime.utcnow().isoformat(),
        }
        self.table.put_item(Item=item, ConditionExpression='attribute_not_exists(pk)')
        return item

    def get_user(self, user_id: str) -> dict:
        return self.table.get_item(Key={'pk': f'USER#{user_id}', 'sk': 'METADATA'}).get('Item')

    def get_user_orders(self, user_id: str, limit: int = 20) -> list:
        return self.table.query(
            KeyConditionExpression='pk = :pk AND begins_with(sk, :prefix)',
            ExpressionAttributeValues={':pk': f'USER#{user_id}', ':prefix': 'ORDER#'},
            Limit=limit, ScanIndexForward=False,
        ).get('Items', [])

    def update_order_status(self, user_id: str, order_id: str, status: str) -> dict:
        return self.table.update_item(
            Key={'pk': f'USER#{user_id}', 'sk': f'ORDER#{order_id}'},
            UpdateExpression='SET #s = :s, gsi2pk = :s, updated_at = :t',
            ExpressionAttributeNames={'#s': 'status'},
            ExpressionAttributeValues={':s': status, ':t': datetime.utcnow().isoformat()},
            ReturnValues='ALL_NEW',
        )['Attributes']
```

### B. Table CloudFormation

```yaml
Resources:
  MainTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '${AWS::StackName}-main'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - { AttributeName: pk, AttributeType: S }
        - { AttributeName: sk, AttributeType: S }
        - { AttributeName: gsi1pk, AttributeType: S }
        - { AttributeName: gsi1sk, AttributeType: S }
        - { AttributeName: gsi2pk, AttributeType: S }
        - { AttributeName: gsi2sk, AttributeType: S }
      KeySchema:
        - { AttributeName: pk, KeyType: HASH }
        - { AttributeName: sk, KeyType: RANGE }
      GlobalSecondaryIndexes:
        - IndexName: gsi1
          KeySchema:
            - { AttributeName: gsi1pk, KeyType: HASH }
            - { AttributeName: gsi1sk, KeyType: RANGE }
          Projection: { ProjectionType: ALL }
        - IndexName: gsi2
          KeySchema:
            - { AttributeName: gsi2pk, KeyType: HASH }
            - { AttributeName: gsi2sk, KeyType: RANGE }
          Projection: { ProjectionType: ALL }
      PointInTimeRecoverySpecification: { PointInTimeRecoveryEnabled: true }
      SSESpecification: { SSEEnabled: true }
      TimeToLiveSpecification: { AttributeName: ttl, Enabled: true }
```

### C. Transactions and Pagination

```python
import boto3
client = boto3.client('dynamodb')

def create_order_with_items(table_name, user_id, order_id, items, total):
    """Transactional write: create order + line items atomically."""
    transact_items = [{'Put': {'TableName': table_name, 'Item': {
        'pk': {'S': f'USER#{user_id}'}, 'sk': {'S': f'ORDER#{order_id}'},
        'gsi1pk': {'S': f'ORDER#{order_id}'}, 'gsi2pk': {'S': 'STATUS#pending'},
        'total': {'N': str(total)},
    }, 'ConditionExpression': 'attribute_not_exists(pk)'}}]
    for idx, item in enumerate(items):
        transact_items.append({'Put': {'TableName': table_name, 'Item': {
            'pk': {'S': f'ORDER#{order_id}'}, 'sk': {'S': f'ITEM#{idx:04d}'},
            'product_id': {'S': item['product_id']},
        }}})
    client.transact_write_items(TransactItems=transact_items)

def paginated_query(table, pk, sk_prefix, limit=20, last_key=None):
    kwargs = {
        'KeyConditionExpression': 'pk = :pk AND begins_with(sk, :prefix)',
        'ExpressionAttributeValues': {':pk': pk, ':prefix': sk_prefix},
        'Limit': limit, 'ScanIndexForward': False,
    }
    if last_key:
        kwargs['ExclusiveStartKey'] = last_key
    resp = table.query(**kwargs)
    return {'items': resp.get('Items', []), 'next_token': resp.get('LastEvaluatedKey')}
```

---

## 6. S3 (MANDATORY)

### A. Bucket Configuration

```yaml
Resources:
  DataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-data-${AWS::AccountId}'
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault: { SSEAlgorithm: AES256 }
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      VersioningConfiguration: { Status: Enabled }
      LifecycleConfiguration:
        Rules:
          - Id: TransitionToIA
            Status: Enabled
            Transitions:
              - { TransitionInDays: 90, StorageClass: STANDARD_IA }
              - { TransitionInDays: 365, StorageClass: GLACIER }
          - Id: DeleteOldVersions
            Status: Enabled
            NoncurrentVersionExpiration: { NoncurrentDays: 90 }

  # Always enforce HTTPS
  DataBucketPolicy:
    Type: AWS::S3::BucketPolicy
    Properties:
      Bucket: !Ref DataBucket
      PolicyDocument:
        Statement:
          - Sid: EnforceHTTPS
            Effect: Deny
            Principal: '*'
            Action: 's3:*'
            Resource: [!GetAtt DataBucket.Arn, !Sub '${DataBucket.Arn}/*']
            Condition: { Bool: { 'aws:SecureTransport': 'false' } }
```

### B. S3 Operations

```python
import boto3

s3 = boto3.client('s3')

def upload_file(bucket: str, file_path: str, key: str) -> str:
    s3.upload_file(file_path, bucket, key, ExtraArgs={'ServerSideEncryption': 'AES256'})
    return f's3://{bucket}/{key}'

def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """Generate presigned URL for secure download."""
    return s3.generate_presigned_url('get_object',
        Params={'Bucket': bucket, 'Key': key}, ExpiresIn=expiration)

def generate_presigned_post(bucket: str, key: str) -> dict:
    """Generate presigned POST for direct browser upload (10MB max)."""
    return s3.generate_presigned_post(bucket, key,
        Conditions=[['content-length-range', 1, 10485760]], ExpiresIn=3600)
```

### C. S3 Event Notifications

```yaml
Resources:
  # S3 event triggering Lambda (SAM)
  ImageProcessor:
    Type: AWS::Serverless::Function
    Properties:
      Handler: image_processor.handler
      CodeUri: src/
      Timeout: 300
      MemorySize: 1024
      Policies:
        - S3ReadPolicy: { BucketName: !Ref UploadBucket }
        - S3CrudPolicy: { BucketName: !Ref ProcessedBucket }
      Events:
        S3Upload:
          Type: S3
          Properties:
            Bucket: !Ref UploadBucket
            Events: s3:ObjectCreated:*
            Filter:
              S3Key:
                Rules:
                  - { Name: prefix, Value: uploads/images/ }
                  - { Name: suffix, Value: .jpg }
  # Prefer EventBridge notifications for new projects:
  # NotificationConfiguration: { EventBridgeConfiguration: { EventBridgeEnabled: true } }
```

```python
# S3 event handler - note: URL-decode key (S3 encodes spaces as '+')
import boto3, urllib.parse

s3 = boto3.client('s3')

def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(record['s3']['object']['key'])
        body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        # ... process the file
```

---

## 7. API Gateway (MANDATORY)

```yaml
# REST API (v1) with Cognito auth, throttling, and access logging
Resources:
  Api:
    Type: AWS::Serverless::Api
    Properties:
      StageName: !Ref Environment
      TracingEnabled: true
      AccessLogSetting:
        DestinationArn: !GetAtt ApiAccessLogs.Arn
        Format: '{"requestId":"$context.requestId","httpMethod":"$context.httpMethod","path":"$context.path","status":"$context.status","responseLatency":"$context.responseLatency"}'
      MethodSettings:
        - { HttpMethod: '*', ResourcePath: '/*', ThrottlingBurstLimit: 100, ThrottlingRateLimit: 50 }
      Auth:
        DefaultAuthorizer: CognitoAuthorizer
        Authorizers:
          CognitoAuthorizer: { UserPoolArn: !GetAtt UserPool.Arn }
      Cors:
        AllowMethods: "'GET,POST,PUT,DELETE,OPTIONS'"
        AllowHeaders: "'Content-Type,Authorization'"
        AllowOrigin: "'https://example.com'"

  ApiAccessLogs:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/aws/apigateway/${AWS::StackName}'
      RetentionInDays: 30
```

### HTTP API (v2) vs REST API (v1)

```yaml
# HTTP API (v2) - PREFERRED for most new projects
# ~70% cheaper, lower latency, native JWT auth, simpler CORS
# Use REST API (v1) only when you need: API keys, usage plans, WAF, caching,
# request validation, or request/response transformation
Resources:
  HttpApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: !Ref Environment
      CorsConfiguration:
        AllowOrigins: ['https://example.com']
        AllowMethods: [GET, POST, PUT, DELETE]
        AllowHeaders: [Content-Type, Authorization]
      Auth:
        DefaultAuthorizer: JWTAuthorizer
        Authorizers:
          JWTAuthorizer:
            JwtConfiguration:
              issuer: !Sub 'https://cognito-idp.${AWS::Region}.amazonaws.com/${UserPool}'
              audience: [!Ref UserPoolClient]

  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.handler
      CodeUri: src/
      Events:
        GetItems: { Type: HttpApi, Properties: { ApiId: !Ref HttpApi, Path: /items, Method: GET } }
        CreateItem: { Type: HttpApi, Properties: { ApiId: !Ref HttpApi, Path: /items, Method: POST } }
```

---

## 8. ECS/Fargate (MANDATORY)

```yaml
Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${AWS::StackName}-cluster'
      CapacityProviders: [FARGATE, FARGATE_SPOT]
      DefaultCapacityProviderStrategy:
        - { CapacityProvider: FARGATE, Weight: 1 }
        - { CapacityProvider: FARGATE_SPOT, Weight: 4 }  # 80% Spot savings
      ClusterSettings:
        - { Name: containerInsights, Value: enabled }

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub '${AWS::StackName}-app'
      NetworkMode: awsvpc
      RequiresCompatibilities: [FARGATE]
      Cpu: '256'
      Memory: '512'
      ExecutionRoleArn: !GetAtt TaskExecutionRole.Arn
      TaskRoleArn: !GetAtt TaskRole.Arn
      ContainerDefinitions:
        - Name: app
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/${ImageRepository}:latest'
          Essential: true
          PortMappings: [{ ContainerPort: 8080 }]
          Secrets:
            - { Name: DATABASE_URL, ValueFrom: !Ref DatabaseUrlSecret }
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: app

  Service:
    Type: AWS::ECS::Service
    Properties:
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      NetworkConfiguration:
        AwsvpcConfiguration:
          AssignPublicIp: DISABLED
          SecurityGroups: [!Ref ServiceSecurityGroup]
          Subnets: [!Ref PrivateSubnet1, !Ref PrivateSubnet2]
      LoadBalancers:
        - { ContainerName: app, ContainerPort: 8080, TargetGroupArn: !Ref TargetGroup }
      DeploymentConfiguration: { MinimumHealthyPercent: 100, MaximumPercent: 200 }
```

---

## 9. CloudWatch and Monitoring (MANDATORY)

```yaml
Resources:
  ErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${AWS::StackName}-errors'
      MetricName: Errors
      Namespace: AWS/Lambda
      Dimensions: [{ Name: FunctionName, Value: !Ref ProcessFunction }]
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 1
      Threshold: 5
      ComparisonOperator: GreaterThanThreshold
      AlarmActions: [!Ref AlertTopic]

  # Also create alarms for: DLQ depth, API Gateway 5xx, Lambda duration P99
```

### CloudWatch Logs Insights Queries

```sql
-- Find Lambda cold starts and their duration
fields @timestamp, @duration, @memorySize, @maxMemoryUsed
| filter @message like /REPORT/ and @message like /Init Duration/
| parse @message "Init Duration: * ms" as initDuration
| sort initDuration desc | limit 50

-- Error pattern analysis (errors per hour)
fields @timestamp, @message
| filter @message like /ERROR/
| stats count(*) as errorCount by bin(1h)
| sort errorCount desc

-- P99 latency analysis for Lambda
fields @duration | filter @type = "REPORT"
| stats avg(@duration) as avg, pct(@duration, 50) as p50,
        pct(@duration, 90) as p90, pct(@duration, 99) as p99
  by bin(1h)

-- Search for correlation ID across services
fields @timestamp, @message, @logStream
| filter @message like /correlation-id-abc123/
| sort @timestamp asc
```

---

## 10. SQS and SNS Patterns (MANDATORY)

### A. SQS Queue Configuration

```yaml
Resources:
  # Standard queue with dead-letter queue
  OrderQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${AWS::StackName}-orders'
      VisibilityTimeout: 300    # 6x Lambda timeout
      MessageRetentionPeriod: 1209600  # 14 days
      ReceiveMessageWaitTimeSeconds: 20  # Long polling (always enable)
      SqsManagedSseEnabled: true
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt OrderDLQ.Arn
        maxReceiveCount: 3

  OrderDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${AWS::StackName}-orders-dlq'
      MessageRetentionPeriod: 1209600

  # FIFO queue for ordered processing
  PaymentQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${AWS::StackName}-payments.fifo'
      FifoQueue: true
      ContentBasedDeduplication: true
      FifoThroughputLimit: perMessageGroupId  # High throughput mode
      VisibilityTimeout: 300
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt PaymentDLQ.Arn
        maxReceiveCount: 3

  PaymentDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub '${AWS::StackName}-payments-dlq.fifo'
      FifoQueue: true
```

```python
import boto3, json
sqs = boto3.client('sqs')

def send_fifo_message(queue_url, order):
    return sqs.send_message(QueueUrl=queue_url, MessageBody=json.dumps(order),
        MessageGroupId=order['user_id'],            # Per-user ordering
        MessageDeduplicationId=order['order_id'])    # Prevent duplicates
```

### B. SNS Fan-Out Pattern

```yaml
Resources:
  # SNS to SQS fan-out: one event triggers multiple consumers
  OrderEventsTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub '${AWS::StackName}-order-events'
      KmsMasterKeyId: alias/aws/sns

  # FilterPolicy: only receive matching events; omit for all events
  InventorySubscription:
    Type: AWS::SNS::Subscription
    Properties:
      TopicArn: !Ref OrderEventsTopic
      Protocol: sqs
      Endpoint: !GetAtt InventoryQueue.Arn
      FilterPolicy: { event_type: [order_created, order_cancelled] }
      RawMessageDelivery: true  # Skip SNS envelope wrapping
```

---

## 11. EventBridge (MANDATORY)

### A. Event Bus and Rules

```yaml
Resources:
  # Always use custom event bus (do not pollute the default bus)
  AppEventBus:
    Type: AWS::Events::EventBus
    Properties:
      Name: !Sub '${AWS::StackName}-events'

  # Pattern-matching rule with numeric filter
  OrderCreatedRule:
    Type: AWS::Events::Rule
    Properties:
      EventBusName: !Ref AppEventBus
      EventPattern:
        source: ['myapp.orders']
        detail-type: ['OrderCreated']
        detail: { total: [{ numeric: ['>=', 100] }] }
      Targets:
        - { Id: ProcessOrder, Arn: !GetAtt HighValueOrderFunction.Arn }

  # Scheduled rule
  DailyCleanupRule:
    Type: AWS::Events::Rule
    Properties:
      ScheduleExpression: 'cron(0 2 * * ? *)'
      Targets: [{ Id: Cleanup, Arn: !GetAtt CleanupFunction.Arn }]
```

### B. Publishing Events

```python
import boto3, json
eventbridge = boto3.client('events')

def publish_event(bus_name, source, detail_type, detail):
    resp = eventbridge.put_events(Entries=[{
        'Source': source, 'DetailType': detail_type,
        'Detail': json.dumps(detail), 'EventBusName': bus_name,
    }])
    if resp['FailedEntryCount'] > 0:
        raise Exception(f"Failed: {resp['Entries']}")

# publish_event('myapp-events', 'myapp.orders', 'OrderCreated', {'order_id': 'ord-123'})
```

### C. Archive and Cross-Account

```yaml
Resources:
  # Archive for replay (disaster recovery, debugging)
  EventArchive:
    Type: AWS::Events::Archive
    Properties:
      ArchiveName: !Sub '${AWS::StackName}-archive'
      SourceArn: !GetAtt AppEventBus.Arn
      EventPattern: { source: [{ prefix: 'myapp.' }] }
      RetentionDays: 90

  # Cross-account event routing
  CrossAccountPolicy:
    Type: AWS::Events::EventBusPolicy
    Properties:
      EventBusName: !Ref AppEventBus
      StatementId: AllowCrossAccount
      Statement:
        Effect: Allow
        Principal: { AWS: !Sub 'arn:aws:iam::${TrustedAccountId}:root' }
        Action: 'events:PutEvents'
        Resource: !GetAtt AppEventBus.Arn
```

---

## 12. Step Functions (MANDATORY)

### A. Standard vs Express Workflows

```yaml
# Standard Workflow: long-running (up to 1 year), exactly-once execution
# Use for: order processing, ETL, human approval workflows
# Pricing: per state transition

# Express Workflow: short-lived (up to 5 min), at-least-once execution
# Use for: high-volume event processing, streaming data, IoT ingestion
# Pricing: per execution + duration
```

### B. Order Processing Workflow

```yaml
Resources:
  OrderStateMachine:
    Type: AWS::Serverless::StateMachine
    Properties:
      DefinitionUri: statemachine/order-processing.asl.json
      Type: STANDARD
      Tracing: { Enabled: true }
      Logging:
        Level: ALL
        IncludeExecutionData: true
        Destinations:
          - CloudWatchLogsLogGroup: { LogGroupArn: !GetAtt SFNLogGroup.Arn }
      Policies:
        - LambdaInvokePolicy: { FunctionName: !Ref ValidateOrderFunction }
        - LambdaInvokePolicy: { FunctionName: !Ref ProcessPaymentFunction }
        - DynamoDBCrudPolicy: { TableName: !Ref OrdersTable }
```

```json
// statemachine/order-processing.asl.json (key patterns)
{
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task", "Resource": "${ValidateOrderFunctionArn}", "Next": "IsInStock",
      "Retry": [{ "ErrorEquals": ["ServiceUnavailable"], "IntervalSeconds": 2, "MaxAttempts": 3, "BackoffRate": 2.0 }],
      "Catch": [{ "ErrorEquals": ["ValidationError"], "Next": "OrderFailed", "ResultPath": "$.error" }]
    },
    "IsInStock": {
      "Type": "Choice",
      "Choices": [{ "Variable": "$.inventory_count", "NumericGreaterThan": 0, "Next": "ProcessPayment" }],
      "Default": "WaitForRestock"
    },
    "WaitForRestock": { "Type": "Wait", "Seconds": 300, "Next": "ValidateOrder" },
    "ProcessPayment": {
      "Type": "Task", "Resource": "${ProcessPaymentFunctionArn}", "Next": "ParallelFulfillment",
      "Retry": [{ "ErrorEquals": ["PaymentGatewayTimeout"], "IntervalSeconds": 5, "MaxAttempts": 2 }],
      "Catch": [{ "ErrorEquals": ["PaymentDeclined"], "Next": "OrderFailed" }]
    },
    "ParallelFulfillment": {
      "Type": "Parallel", "Next": "OrderCompleted",
      "Branches": [
        { "StartAt": "Fulfill", "States": { "Fulfill": { "Type": "Task", "Resource": "${FulfillFunctionArn}", "End": true } } },
        { "StartAt": "Notify", "States": { "Notify": { "Type": "Task", "Resource": "arn:aws:states:::sns:publish", "Parameters": { "TopicArn": "${TopicArn}", "Message.$": "$.order_id" }, "End": true } } }
      ],
      "Catch": [{ "ErrorEquals": ["States.ALL"], "Next": "OrderFailed" }]
    },
    "OrderCompleted": { "Type": "Succeed" },
    "OrderFailed": { "Type": "Fail" }
  }
}
```

### C. Map State for Batch Processing

```json
// Process items in parallel with retries (max 10 concurrent)
{ "Type": "Map", "ItemsPath": "$.items", "MaxConcurrency": 10,
  "ItemProcessor": { "StartAt": "Process", "States": { "Process": {
    "Type": "Task", "Resource": "${ProcessItemArn}", "End": true,
    "Retry": [{ "ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 2, "MaxAttempts": 3 }]
  }}}}
```

---

## 13. CDK Patterns (MANDATORY)

### A. Construct Levels

```typescript
// L1 - CloudFormation resources (CfnXxx prefix) - avoid unless L2 is insufficient
// L2 - Curated constructs with sensible defaults - PREFERRED for most use cases
// L3 - Patterns combining multiple resources - highest abstraction

// ✅ L2 construct: sensible defaults, type-safe, grant helpers
const table = new dynamodb.Table(this, 'DataTable', {
  partitionKey: { name: 'pk', type: dynamodb.AttributeType.STRING },
  sortKey: { name: 'sk', type: dynamodb.AttributeType.STRING },
  billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
  pointInTimeRecovery: true,
  encryption: dynamodb.TableEncryption.AWS_MANAGED,
  removalPolicy: cdk.RemovalPolicy.RETAIN,
});
```

### B. Stack Organization

```typescript
// ✅ Separate stacks by lifecycle and team ownership
const app = new cdk.App();
const env = { account: '123456789012', region: 'us-east-1' };

const networkStack = new NetworkStack(app, 'Network', { env });       // Rarely changes
const dataStack = new DataStack(app, 'Data', { env, vpc: networkStack.vpc }); // Infrequent
const appStack = new ApplicationStack(app, 'App', { env,              // Frequent
  vpc: networkStack.vpc, table: dataStack.table });
```

```typescript
// lib/application-stack.ts
export class ApplicationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: ApplicationStackProps) {
    super(scope, id, props);

    const handler = new lambda.Function(this, 'ApiHandler', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'app.handler',
      code: lambda.Code.fromAsset('src/api'),
      architecture: lambda.Architecture.ARM_64,
      memorySize: 512,
      timeout: cdk.Duration.seconds(30),
      tracing: lambda.Tracing.ACTIVE,
      environment: { TABLE_NAME: props.table.tableName },
    });

    props.table.grantReadWriteData(handler);  // CDK generates scoped IAM policy

    const httpApi = new apigwv2.HttpApi(this, 'HttpApi', {
      corsPreflight: { allowOrigins: ['https://example.com'], allowMethods: [apigwv2.CorsHttpMethod.ANY] },
    });
    httpApi.addRoutes({
      path: '/items', methods: [apigwv2.HttpMethod.GET, apigwv2.HttpMethod.POST],
      integration: new apigwv2_integrations.HttpLambdaIntegration('Api', handler),
    });
  }
}
```

### C. Environment-Specific Configuration

```typescript
// cdk.json context: { "environments": { "dev": {...}, "prod": {...} } }
const envName = app.node.tryGetContext('env') || 'dev';
const config = app.node.tryGetContext('environments')[envName];
new ApplicationStack(app, `MyApp-${envName}`, {
  env: { account: config.account, region: config.region }, config,
});
// Deploy: cdk deploy --context env=prod
```

### D. CDK Pipelines for CI/CD

```typescript
const pipeline = new CodePipeline(this, 'Pipeline', {
  synth: new ShellStep('Synth', {
    input: CodePipelineSource.gitHub('myorg/myrepo', 'main'),
    commands: ['npm ci', 'npm run build', 'npm run test', 'npx cdk synth'],
  }),
  selfMutation: true,      // Pipeline updates itself
  crossAccountKeys: true,  // Cross-account deployments
});

const staging = pipeline.addStage(new AppStage(this, 'Staging', { env: stagingEnv }));
staging.addPost(new ShellStep('IntegTest', { commands: ['npm run test:integration'] }));

const prod = pipeline.addStage(new AppStage(this, 'Prod', { env: prodEnv }));
prod.addPre(new pipelines.ManualApprovalStep('PromoteToProd'));
```

### E. Testing CDK Constructs

```typescript
import { Template, Match } from 'aws-cdk-lib/assertions';

const template = Template.fromStack(new ApplicationStack(app, 'Test', { config }));

// Verify resource properties
template.hasResourceProperties('AWS::Lambda::Function', {
  Runtime: 'python3.12', MemorySize: 512, Architectures: ['arm64'],
});

// Ensure security: no public S3 buckets
template.allResourcesProperties('AWS::S3::Bucket', {
  PublicAccessBlockConfiguration: {
    BlockPublicAcls: true, BlockPublicPolicy: true,
    IgnorePublicAcls: true, RestrictPublicBuckets: true,
  },
});

// Snapshot test: detect unintended infrastructure changes
expect(template.toJSON()).toMatchSnapshot();
```

---

## 14. Deployment Checklist

### Security
- [ ] Least privilege IAM policies with permission boundaries
- [ ] Secrets in Secrets Manager; config in Parameter Store
- [ ] Encryption at rest and in transit (TLS, S3 HTTPS enforcement)
- [ ] VPC endpoints for private access; security groups locked down
- [ ] IAM roles for services (never embed access keys)
- [ ] CloudTrail enabled for API audit logging

### Reliability
- [ ] Multi-AZ deployment; auto-scaling configured
- [ ] Backup and recovery tested (DynamoDB PITR, S3 versioning)
- [ ] Dead letter queues on all async processing
- [ ] Step Functions Catch/Retry for workflow error handling

### Operations
- [ ] CloudWatch alarms for errors, latency, and DLQ depth
- [ ] X-Ray tracing enabled; Powertools configured
- [ ] Logs Insights queries saved; log retention periods set

### Cost
- [ ] ARM64 architecture; right-sized Lambda memory
- [ ] Spot/Fargate Spot for fault-tolerant workloads
- [ ] S3 lifecycle policies; DynamoDB billing mode evaluated

### Deployment
- [ ] Infrastructure as Code for all resources
- [ ] CI/CD pipeline with CDK snapshot tests
- [ ] Stacks separated by lifecycle; env config via context/SSM

---

## 15. Quick Reference

```bash
# AWS CLI
aws s3 cp file.txt s3://bucket-name/
aws lambda invoke --function-name my-func output.json
aws logs tail /aws/lambda/my-func --follow
aws ssm get-parameter --name /my/param --with-decryption
aws ecs update-service --cluster my-cluster --service my-service --force-new-deployment

# SAM
sam build && sam deploy --guided
sam local invoke FunctionName
sam logs -n FunctionName --tail

# CDK
cdk synth                         # Synthesize CloudFormation template
cdk diff                          # Show pending changes
cdk deploy --context env=prod     # Deploy with environment context

# SQS / EventBridge / Step Functions
aws sqs send-message --queue-url URL --message-body '{"test": true}'
aws events put-events --entries '[{"Source":"myapp","DetailType":"Test","Detail":"{}"}]'
aws stepfunctions start-execution --state-machine-arn ARN --input '{"key":"value"}'
```

---

**Last Updated:** 2026-02-27
**Version:** 2.0
**Maintainer:** Cloud Team


**End of AWS Development Guidelines**

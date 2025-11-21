# AWS Development Guidelines

This document provides mandatory standards for building applications on Amazon Web Services.

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
# CloudFormation example
Resources:
  LambdaExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${AWS::StackName}-lambda-role'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
      Policies:
        - PolicyName: DynamoDBAccess
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:GetItem
                  - dynamodb:PutItem
                  - dynamodb:UpdateItem
                  - dynamodb:Query
                Resource:
                  - !GetAtt UsersTable.Arn
                  - !Sub '${UsersTable.Arn}/index/*'
```

### C. Secrets Management

```python
import boto3
from botocore.exceptions import ClientError
import json

def get_secret(secret_name: str, region: str = 'us-east-1') -> dict:
    """Retrieve secret from AWS Secrets Manager."""
    client = boto3.client('secretsmanager', region_name=region)

    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            raise ValueError(f"Secret {secret_name} not found")
        raise

# Usage
db_credentials = get_secret('prod/database/credentials')
connection_string = f"postgresql://{db_credentials['username']}:{db_credentials['password']}@{db_credentials['host']}/{db_credentials['database']}"
```

---

## 4. Lambda Functions (MANDATORY)

### A. Function Structure

```python
# lambda_function.py
import json
import logging
import os
from typing import Any
import boto3
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.utilities.validation import validate

logger = Logger()
tracer = Tracer()
metrics = Metrics()

# Initialize outside handler for connection reuse
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(os.environ['TABLE_NAME'])


@logger.inject_lambda_context
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def handler(event: dict, context: LambdaContext) -> dict:
    """Main Lambda handler."""
    try:
        # Log incoming event (be careful with sensitive data)
        logger.info("Processing request", extra={"event": event})

        # Validate input
        body = json.loads(event.get('body', '{}'))
        validate(event=body, schema=INPUT_SCHEMA)

        # Process request
        result = process_request(body)

        # Return success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }

    except json.JSONDecodeError:
        return error_response(400, 'Invalid JSON in request body')
    except ValidationError as e:
        return error_response(400, str(e))
    except Exception as e:
        logger.exception("Unexpected error")
        return error_response(500, 'Internal server error')


@tracer.capture_method
def process_request(data: dict) -> dict:
    """Process the incoming request."""
    # Business logic here
    item = {
        'pk': data['id'],
        'sk': 'METADATA',
        'data': data
    }
    table.put_item(Item=item)
    return {'id': data['id'], 'status': 'created'}


def error_response(status_code: int, message: str) -> dict:
    """Generate error response."""
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps({'error': message})
    }
```

### B. Lambda Configuration (SAM)

```yaml
# template.yaml
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
      Description: Process incoming requests
      Environment:
        Variables:
          TABLE_NAME: !Ref DataTable
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref DataTable
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /process
            Method: POST
      # VPC config if needed
      # VpcConfig:
      #   SecurityGroupIds:
      #     - !Ref LambdaSecurityGroup
      #   SubnetIds:
      #     - !Ref PrivateSubnet1
      #     - !Ref PrivateSubnet2

  DataTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub '${AWS::StackName}-data'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: true
```

---

## 5. DynamoDB (MANDATORY)

### A. Table Design

```python
# Single-table design pattern
"""
Entity Types:
- User: PK=USER#<id>, SK=METADATA
- Order: PK=USER#<user_id>, SK=ORDER#<order_id>
- Product: PK=PRODUCT#<id>, SK=METADATA

GSI1 (Inverted Index):
- GSI1PK = SK, GSI1SK = PK
- Allows querying orders by order_id

GSI2 (Sparse Index):
- GSI2PK = status, GSI2SK = created_at
- Allows querying orders by status
"""

import boto3
from datetime import datetime
from typing import Optional, List
from decimal import Decimal

class DynamoDBRepository:
    def __init__(self, table_name: str):
        dynamodb = boto3.resource('dynamodb')
        self.table = dynamodb.Table(table_name)

    def create_user(self, user_id: str, email: str, name: str) -> dict:
        item = {
            'pk': f'USER#{user_id}',
            'sk': 'METADATA',
            'gsi1pk': f'EMAIL#{email}',
            'gsi1sk': f'USER#{user_id}',
            'entity_type': 'USER',
            'user_id': user_id,
            'email': email,
            'name': name,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat()
        }
        self.table.put_item(
            Item=item,
            ConditionExpression='attribute_not_exists(pk)'
        )
        return item

    def get_user(self, user_id: str) -> Optional[dict]:
        response = self.table.get_item(
            Key={'pk': f'USER#{user_id}', 'sk': 'METADATA'}
        )
        return response.get('Item')

    def create_order(self, user_id: str, order_id: str, items: List[dict], total: Decimal) -> dict:
        item = {
            'pk': f'USER#{user_id}',
            'sk': f'ORDER#{order_id}',
            'gsi1pk': f'ORDER#{order_id}',
            'gsi1sk': f'USER#{user_id}',
            'gsi2pk': 'pending',
            'gsi2sk': datetime.utcnow().isoformat(),
            'entity_type': 'ORDER',
            'order_id': order_id,
            'user_id': user_id,
            'items': items,
            'total': total,
            'status': 'pending',
            'created_at': datetime.utcnow().isoformat()
        }
        self.table.put_item(Item=item)
        return item

    def get_user_orders(self, user_id: str, limit: int = 20) -> List[dict]:
        response = self.table.query(
            KeyConditionExpression='pk = :pk AND begins_with(sk, :sk_prefix)',
            ExpressionAttributeValues={
                ':pk': f'USER#{user_id}',
                ':sk_prefix': 'ORDER#'
            },
            Limit=limit,
            ScanIndexForward=False  # Newest first
        )
        return response.get('Items', [])

    def update_order_status(self, user_id: str, order_id: str, status: str) -> dict:
        response = self.table.update_item(
            Key={'pk': f'USER#{user_id}', 'sk': f'ORDER#{order_id}'},
            UpdateExpression='SET #status = :status, gsi2pk = :status, updated_at = :updated_at',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':status': status,
                ':updated_at': datetime.utcnow().isoformat()
            },
            ReturnValues='ALL_NEW'
        )
        return response['Attributes']
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
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
        - AttributeName: gsi1pk
          AttributeType: S
        - AttributeName: gsi1sk
          AttributeType: S
        - AttributeName: gsi2pk
          AttributeType: S
        - AttributeName: gsi2sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE
      GlobalSecondaryIndexes:
        - IndexName: gsi1
          KeySchema:
            - AttributeName: gsi1pk
              KeyType: HASH
            - AttributeName: gsi1sk
              KeyType: RANGE
          Projection:
            ProjectionType: ALL
        - IndexName: gsi2
          KeySchema:
            - AttributeName: gsi2pk
              KeyType: HASH
            - AttributeName: gsi2sk
              KeyType: RANGE
          Projection:
            ProjectionType: ALL
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: true
      SSESpecification:
        SSEEnabled: true
      TimeToLiveSpecification:
        AttributeName: ttl
        Enabled: true
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
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true
      VersioningConfiguration:
        Status: Enabled
      LoggingConfiguration:
        DestinationBucketName: !Ref LoggingBucket
        LogFilePrefix: s3-access-logs/
      LifecycleConfiguration:
        Rules:
          - Id: TransitionToIA
            Status: Enabled
            Transitions:
              - TransitionInDays: 90
                StorageClass: STANDARD_IA
              - TransitionInDays: 365
                StorageClass: GLACIER
          - Id: DeleteOldVersions
            Status: Enabled
            NoncurrentVersionExpiration:
              NoncurrentDays: 90

  DataBucketPolicy:
    Type: AWS::S3::BucketPolicy
    Properties:
      Bucket: !Ref DataBucket
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Sid: EnforceHTTPS
            Effect: Deny
            Principal: '*'
            Action: 's3:*'
            Resource:
              - !GetAtt DataBucket.Arn
              - !Sub '${DataBucket.Arn}/*'
            Condition:
              Bool:
                'aws:SecureTransport': 'false'
```

### B. S3 Operations

```python
import boto3
from botocore.exceptions import ClientError

class S3Service:
    def __init__(self, bucket_name: str):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_file(self, file_path: str, key: str, metadata: dict = None) -> str:
        """Upload file to S3."""
        extra_args = {'ServerSideEncryption': 'AES256'}
        if metadata:
            extra_args['Metadata'] = metadata

        self.s3.upload_file(file_path, self.bucket_name, key, ExtraArgs=extra_args)
        return f's3://{self.bucket_name}/{key}'

    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate presigned URL for download."""
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': key},
            ExpiresIn=expiration
        )

    def generate_presigned_post(self, key: str, expiration: int = 3600) -> dict:
        """Generate presigned POST for upload."""
        return self.s3.generate_presigned_post(
            self.bucket_name,
            key,
            Fields={'acl': 'private'},
            Conditions=[
                {'acl': 'private'},
                ['content-length-range', 1, 10485760]  # 10MB max
            ],
            ExpiresIn=expiration
        )
```

---

## 7. API Gateway (MANDATORY)

```yaml
# SAM template for API Gateway
Resources:
  Api:
    Type: AWS::Serverless::Api
    Properties:
      StageName: !Ref Environment
      TracingEnabled: true
      AccessLogSetting:
        DestinationArn: !GetAtt ApiAccessLogs.Arn
        Format: '{"requestId":"$context.requestId","ip":"$context.identity.sourceIp","requestTime":"$context.requestTime","httpMethod":"$context.httpMethod","path":"$context.path","status":"$context.status","responseLatency":"$context.responseLatency"}'
      MethodSettings:
        - HttpMethod: '*'
          ResourcePath: '/*'
          ThrottlingBurstLimit: 100
          ThrottlingRateLimit: 50
      Auth:
        DefaultAuthorizer: CognitoAuthorizer
        Authorizers:
          CognitoAuthorizer:
            UserPoolArn: !GetAtt UserPool.Arn
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

---

## 8. ECS/Fargate (MANDATORY)

```yaml
Resources:
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${AWS::StackName}-cluster'
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1
        - CapacityProvider: FARGATE_SPOT
          Weight: 4
      ClusterSettings:
        - Name: containerInsights
          Value: enabled

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub '${AWS::StackName}-app'
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: '256'
      Memory: '512'
      ExecutionRoleArn: !GetAtt TaskExecutionRole.Arn
      TaskRoleArn: !GetAtt TaskRole.Arn
      ContainerDefinitions:
        - Name: app
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/${ImageRepository}:latest'
          Essential: true
          PortMappings:
            - ContainerPort: 8080
          Environment:
            - Name: NODE_ENV
              Value: production
          Secrets:
            - Name: DATABASE_URL
              ValueFrom: !Ref DatabaseUrlSecret
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: app
          HealthCheck:
            Command:
              - CMD-SHELL
              - curl -f http://localhost:8080/health || exit 1
            Interval: 30
            Timeout: 5
            Retries: 3

  Service:
    Type: AWS::ECS::Service
    DependsOn: ALBListener
    Properties:
      ServiceName: !Sub '${AWS::StackName}-service'
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 2
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          AssignPublicIp: DISABLED
          SecurityGroups:
            - !Ref ServiceSecurityGroup
          Subnets:
            - !Ref PrivateSubnet1
            - !Ref PrivateSubnet2
      LoadBalancers:
        - ContainerName: app
          ContainerPort: 8080
          TargetGroupArn: !Ref TargetGroup
      DeploymentConfiguration:
        MinimumHealthyPercent: 100
        MaximumPercent: 200
```

---

## 9. CloudWatch and Monitoring (MANDATORY)

```yaml
Resources:
  Dashboard:
    Type: AWS::CloudWatch::Dashboard
    Properties:
      DashboardName: !Sub '${AWS::StackName}-dashboard'
      DashboardBody: !Sub |
        {
          "widgets": [
            {
              "type": "metric",
              "properties": {
                "title": "Lambda Invocations",
                "metrics": [
                  ["AWS/Lambda", "Invocations", "FunctionName", "${ProcessFunction}"]
                ],
                "period": 300
              }
            },
            {
              "type": "metric",
              "properties": {
                "title": "Lambda Errors",
                "metrics": [
                  ["AWS/Lambda", "Errors", "FunctionName", "${ProcessFunction}"]
                ],
                "period": 300
              }
            }
          ]
        }

  ErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${AWS::StackName}-errors'
      AlarmDescription: Lambda function errors
      MetricName: Errors
      Namespace: AWS/Lambda
      Dimensions:
        - Name: FunctionName
          Value: !Ref ProcessFunction
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 1
      Threshold: 5
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref AlertTopic
```

---

## 10. Deployment Checklist

### Security
- [ ] Least privilege IAM policies
- [ ] Secrets in Secrets Manager
- [ ] Encryption at rest enabled
- [ ] VPC endpoints for private access
- [ ] Security groups properly configured

### Reliability
- [ ] Multi-AZ deployment
- [ ] Auto-scaling configured
- [ ] Backup and recovery tested
- [ ] Health checks implemented

### Operations
- [ ] CloudWatch alarms set up
- [ ] Logs aggregated and retained
- [ ] X-Ray tracing enabled
- [ ] Dashboards created

### Cost
- [ ] Right-sized resources
- [ ] Spot instances where appropriate
- [ ] S3 lifecycle policies
- [ ] Reserved capacity evaluated

---

## 11. Quick Reference

```bash
# AWS CLI common commands
aws s3 ls s3://bucket-name/
aws s3 cp file.txt s3://bucket-name/
aws lambda invoke --function-name my-func output.json
aws logs tail /aws/lambda/my-func --follow
aws ecs update-service --cluster my-cluster --service my-service --force-new-deployment
aws ssm get-parameter --name /my/param --with-decryption

# SAM commands
sam build
sam local invoke FunctionName
sam deploy --guided
sam logs -n FunctionName --tail
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Cloud Team

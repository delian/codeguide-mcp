# Microsoft Azure Development Guidelines
Mandatory standards for building applications on Microsoft Azure. Azure CLI, Bicep, Terraform, Azure DevOps, Azure Functions, AKS.

---

**Agent Profile**: The Azure Expert
**Role**: Senior Cloud Architect & Azure Solutions Architect
**Objective**: Generate enterprise-grade, secure, and scalable Azure architectures following Microsoft best practices.
**Tools**: Azure CLI, Bicep, Terraform, Azure DevOps, Azure Functions, AKS.

---

## 1. Core Philosophies: AZURE-FIRST

- **A**utomated: Infrastructure as Code with Bicep
- **Z**ero-trust: Security at every layer
- **U**nified: Integrated Microsoft ecosystem
- **R**esilient: Built-in high availability
- **E**nterprise: Compliance and governance ready

---

## 2. Resource Organization (MANDATORY)

### A. Subscription and Resource Group Structure

```
Management Groups
├── Production
│   ├── Subscription: prod-workloads
│   │   ├── rg-prod-app-eastus
│   │   ├── rg-prod-data-eastus
│   │   └── rg-prod-network-eastus
│   └── Subscription: prod-data
├── Non-Production
│   ├── Subscription: staging
│   └── Subscription: development
└── Shared Services
    └── Subscription: shared-services
        ├── rg-shared-identity
        └── rg-shared-monitoring
```

### B. Naming Conventions

```
# Pattern: {resource-type}-{workload}-{environment}-{region}-{instance}

# Resource Groups
rg-myapp-prod-eastus
rg-myapp-staging-westus

# Resources
app-myapp-prod-eastus-001        # App Service
func-myapp-prod-eastus-001       # Function App
sql-myapp-prod-eastus-001        # SQL Database
st-myapp-prod-eastus-001         # Storage Account (no hyphens)
kv-myapp-prod-eastus-001         # Key Vault
aks-myapp-prod-eastus-001        # AKS Cluster
acr-myapp-prod-001               # Container Registry (globally unique)

# Tags (applied to all resources)
Environment: Production
Application: MyApp
CostCenter: Engineering
Owner: team@company.com
ManagedBy: Bicep
```

---

## 3. Identity and Access (MANDATORY)

### A. Managed Identity

```bicep
// Bicep: User-Assigned Managed Identity
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${appName}-${environment}'
  location: location
  tags: tags
}

// Assign to App Service
resource appService 'Microsoft.Web/sites@2022-09-01' = {
  name: 'app-${appName}-${environment}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    // ..
  }
}

// Role Assignment for Key Vault
resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, managedIdentity.id, 'Key Vault Secrets User')
  scope: keyVault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}
```

### B. Key Vault

```bicep
resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: 'kv-${appName}-${environment}'
  location: location
  properties: {
    tenantId: tenant().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enablePurgeProtection: true
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
      virtualNetworkRules: [
        {
          id: subnet.id
        }
      ]
    }
  }
}

// Store secrets
resource dbPasswordSecret 'Microsoft.KeyVault/vaults/secrets@2023-02-01' = {
  parent: keyVault
  name: 'db-password'
  properties: {
    value: dbPassword
    attributes: {
      enabled: true
    }
  }
}
```

```csharp
// C# - Access Key Vault secrets
using Azure.Identity;
using Azure.Security.KeyVault.Secrets;

var client = new SecretClient(
    new Uri("https://kv-myapp-prod.vault.azure.net/"),
    new DefaultAzureCredential()
);

KeyVaultSecret secret = await client.GetSecretAsync("db-password");
string dbPassword = secret.Value;
```

---

## 4. App Service (MANDATORY)

### A. App Service Configuration

```bicep
resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: 'plan-${appName}-${environment}'
  location: location
  sku: {
    name: 'P1v3'
    tier: 'PremiumV3'
    capacity: 2
  }
  properties: {
    reserved: true  // Linux
    zoneRedundant: true
  }
}

resource appService 'Microsoft.Web/sites@2022-09-01' = {
  name: 'app-${appName}-${environment}'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'DOTNETCORE|8.0'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      http20Enabled: true
      healthCheckPath: '/health'
      appSettings: [
        {
          name: 'AZURE_CLIENT_ID'
          value: managedIdentity.properties.clientId
        }
        {
          name: 'KeyVaultUri'
          value: keyVault.properties.vaultUri
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
    virtualNetworkSubnetId: subnet.id
  }
}

// Deployment slot for staging
resource stagingSlot 'Microsoft.Web/sites/slots@2022-09-01' = {
  parent: appService
  name: 'staging'
  location: location
  properties: {
    serverFarmId: appServicePlan.id
  }
}

// Auto-scale
resource autoScale 'Microsoft.Insights/autoscalesettings@2022-10-01' = {
  name: 'autoscale-${appName}'
  location: location
  properties: {
    enabled: true
    targetResourceUri: appServicePlan.id
    profiles: [
      {
        name: 'Default'
        capacity: {
          minimum: '2'
          maximum: '10'
          default: '2'
        }
        rules: [
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: appServicePlan.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT5M'
              timeAggregation: 'Average'
              operator: 'GreaterThan'
              threshold: 70
            }
            scaleAction: {
              direction: 'Increase'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT5M'
            }
          }
          {
            metricTrigger: {
              metricName: 'CpuPercentage'
              metricResourceUri: appServicePlan.id
              timeGrain: 'PT1M'
              statistic: 'Average'
              timeWindow: 'PT5M'
              timeAggregation: 'Average'
              operator: 'LessThan'
              threshold: 30
            }
            scaleAction: {
              direction: 'Decrease'
              type: 'ChangeCount'
              value: '1'
              cooldown: 'PT5M'
            }
          }
        ]
      }
    ]
  }
}
```

---

## 5. Azure Functions (MANDATORY)

### A. Function App Configuration

```bicep
resource functionApp 'Microsoft.Web/sites@2022-09-01' = {
  name: 'func-${appName}-${environment}'
  location: location
  kind: 'functionapp,linux'
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: functionAppPlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'DOTNET-ISOLATED|8.0'
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};EndpointSuffix=${environment().suffixes.storage};AccountKey=${storageAccount.listKeys().keys[0].value}'
        }
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'dotnet-isolated'
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
  }
}
```

### B. Function App Scaling and Deployment Slots

```bicep
// Consumption plan (serverless, pay-per-execution)
resource consumptionPlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: 'plan-func-${appName}-${environment}'
  location: location
  sku: { name: 'Y1', tier: 'Dynamic' }
  properties: { reserved: true }
}

// Premium plan (pre-warmed instances, VNet integration, no cold start)
resource premiumPlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: 'plan-func-${appName}-${environment}'
  location: location
  sku: { name: 'EP1', tier: 'ElasticPremium' }
  properties: {
    reserved: true
    maximumElasticWorkerCount: 20
    zoneRedundant: true
  }
}
```

```bash
# Deploy to staging slot, then swap to production (zero-downtime)
az functionapp deployment source config-zip \
  --name func-myapp-prod --resource-group rg-myapp-prod \
  --slot staging --src app.zip

az functionapp deployment slot swap \
  --name func-myapp-prod --resource-group rg-myapp-prod \
  --slot staging --target-slot production
```

### C. Function Code (HTTP, Queue, Timer Triggers)

```csharp
// Function with HTTP trigger
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;

public class OrderFunctions
{
    private readonly ILogger<OrderFunctions> _logger;
    private readonly IOrderService _orderService;

    public OrderFunctions(ILogger<OrderFunctions> logger, IOrderService orderService)
    {
        _logger = logger;
        _orderService = orderService;
    }

    [Function("ProcessOrder")]
    public async Task<HttpResponseData> ProcessOrder(
        [HttpTrigger(AuthorizationLevel.Function, "post", Route = "orders")] HttpRequestData req)
    {
        _logger.LogInformation("Processing order request");

        var order = await req.ReadFromJsonAsync<CreateOrderRequest>();
        if (order == null)
        {
            var badRequest = req.CreateResponse(HttpStatusCode.BadRequest);
            await badRequest.WriteAsJsonAsync(new { error = "Invalid order data" });
            return badRequest;
        }

        var result = await _orderService.CreateOrderAsync(order);

        var response = req.CreateResponse(HttpStatusCode.Created);
        await response.WriteAsJsonAsync(result);
        return response;
    }

    [Function("ProcessOrderQueue")]
    public async Task ProcessOrderQueue(
        [ServiceBusTrigger("orders", Connection = "ServiceBusConnection")] string message,
        FunctionContext context)
    {
        var order = JsonSerializer.Deserialize<Order>(message);
        _logger.LogInformation("Processing order {OrderId} from queue", order?.Id);

        await _orderService.ProcessOrderAsync(order!);
    }

    [Function("DailyReport")]
    public async Task DailyReport(
        [TimerTrigger("0 0 6 * * *")] TimerInfo timer)  // 6 AM daily
    {
        _logger.LogInformation("Generating daily report");
        await _orderService.GenerateDailyReportAsync();
    }
}
```

### D. Durable Functions

Use Durable Functions for long-running orchestrations. Key patterns: Fan-out/Fan-in (`CallActivityAsync` in parallel + `Task.WhenAll`), Function Chaining (sequential activities), and Human Interaction (with timers). Start orchestrations via HTTP trigger using `DurableTaskClient.ScheduleNewOrchestrationInstanceAsync`.

### E. v4 Programming Model (Node.js)

```javascript
// function.js - Azure Functions v4 programming model (Node.js)
const { app, output } = require('@azure/functions');
const serviceBusOutput = output.serviceBusQueue({ queueName: 'orders', connection: 'ServiceBusConnection' });

app.http('createOrder', {
    methods: ['POST'], authLevel: 'function', route: 'orders',
    extraOutputs: [serviceBusOutput],
    handler: async (request, context) => {
        const order = await request.json();
        if (!order?.customerId) return { status: 400, jsonBody: { error: 'Invalid order data' } };
        const enriched = { id: crypto.randomUUID(), ...order, createdAt: new Date().toISOString() };
        context.extraOutputs.set(serviceBusOutput, enriched);
        return { status: 201, jsonBody: enriched };
    },
});

app.timer('dailyCleanup', {
    schedule: '0 0 2 * * *',
    handler: async (myTimer, context) => { context.log('Running daily cleanup'); },
});

app.serviceBusQueue('processOrder', {
    queueName: 'orders', connection: 'ServiceBusConnection',
    handler: async (message, context) => { context.log('Processing order:', message.id); },
});
```

### F. v4 Programming Model (Python)

```python
# function_app.py
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="orders", methods=["POST"])
@app.service_bus_queue_output(arg_name="message", queue_name="orders", connection="ServiceBusConnection")
def create_order(req: func.HttpRequest, message: func.Out[str]) -> func.HttpResponse:
    order = req.get_json()
    message.set(json.dumps(order))
    return func.HttpResponse(json.dumps(order), status_code=201, mimetype="application/json")

@app.timer_trigger(schedule="0 0 6 * * *", arg_name="timer")
def daily_report(timer: func.TimerRequest) -> None:
    logging.info("Generating daily report")
```

---

## 6. Azure SQL Database (MANDATORY)

### A. SQL Server and Database

```bicep
resource sqlServer 'Microsoft.Sql/servers@2022-11-01-preview' = {
  name: 'sql-${appName}-${environment}'
  location: location
  properties: {
    administratorLogin: sqlAdminUser
    administratorLoginPassword: sqlAdminPassword
    minimalTlsVersion: '1.2'
    publicNetworkAccess: 'Disabled'
  }
  identity: {
    type: 'SystemAssigned'
  }
}

resource sqlDatabase 'Microsoft.Sql/servers/databases@2022-11-01-preview' = {
  parent: sqlServer
  name: appName
  location: location
  sku: {
    name: 'GP_S_Gen5'
    tier: 'GeneralPurpose'
    family: 'Gen5'
    capacity: 2
  }
  properties: {
    collation: 'SQL_Latin1_General_CP1_CI_AS'
    maxSizeBytes: 34359738368  // 32 GB
    autoPauseDelay: 60
    minCapacity: 1
    zoneRedundant: true
    readScale: 'Enabled'
  }
}

// Private Endpoint
resource sqlPrivateEndpoint 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pe-sql-${appName}'
  location: location
  properties: {
    subnet: {
      id: privateEndpointSubnet.id
    }
    privateLinkServiceConnections: [
      {
        name: 'sql-connection'
        properties: {
          privateLinkServiceId: sqlServer.id
          groupIds: ['sqlServer']
        }
      }
    ]
  }
}

// Azure AD authentication
resource sqlAadAdmin 'Microsoft.Sql/servers/administrators@2022-11-01-preview' = {
  parent: sqlServer
  name: 'ActiveDirectory'
  properties: {
    administratorType: 'ActiveDirectory'
    login: aadAdminGroupName
    sid: aadAdminGroupId
    tenantId: tenant().tenantId
  }
}
```

### B. Connection with Managed Identity

```csharp
// C# - Connect with Managed Identity
using Azure.Identity;
using Microsoft.Data.SqlClient;

var connectionString = "Server=tcp:sql-myapp-prod.database.windows.net,1433;" +
                       "Database=myapp;" +
                       "Authentication=Active Directory Managed Identity;" +
                       "User Id=<managed-identity-client-id>;" +
                       "Encrypt=True;" +
                       "TrustServerCertificate=False;";

using var connection = new SqlConnection(connectionString);
await connection.OpenAsync();
```

---

## 7. Storage Account (MANDATORY)

### A. Storage Configuration

```bicep
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'st${appName}${environment}'  // No hyphens allowed
  location: location
  sku: {
    name: 'Standard_ZRS'
  }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false  // Force Azure AD auth
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
      virtualNetworkRules: [
        {
          id: subnet.id
        }
      ]
    }
    encryption: {
      services: {
        blob: {
          enabled: true
        }
        file: {
          enabled: true
        }
      }
      keySource: 'Microsoft.Storage'
    }
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 30
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: 30
    }
  }
}

resource uploadsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'uploads'
  properties: {
    publicAccess: 'None'
  }
}
```

### B. Blob Operations

```csharp
using Azure.Identity;
using Azure.Storage.Blobs;
using Azure.Storage.Sas;

public class BlobStorageService
{
    private readonly BlobServiceClient _blobServiceClient;

    public BlobStorageService(string storageAccountName)
    {
        var uri = new Uri($"https://{storageAccountName}.blob.core.windows.net");
        _blobServiceClient = new BlobServiceClient(uri, new DefaultAzureCredential());
    }

    public async Task<string> UploadAsync(string containerName, string blobName, Stream content)
    {
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);
        var blobClient = containerClient.GetBlobClient(blobName);

        await blobClient.UploadAsync(content, overwrite: true);

        return blobClient.Uri.ToString();
    }

    public Uri GenerateSasUri(string containerName, string blobName, TimeSpan expiry)
    {
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);
        var blobClient = containerClient.GetBlobClient(blobName);

        // User delegation SAS (more secure than account key)
        var userDelegationKey = _blobServiceClient.GetUserDelegationKey(
            DateTimeOffset.UtcNow,
            DateTimeOffset.UtcNow.Add(expiry)
        );

        var sasBuilder = new BlobSasBuilder
        {
            BlobContainerName = containerName,
            BlobName = blobName,
            Resource = "b",
            ExpiresOn = DateTimeOffset.UtcNow.Add(expiry)
        };
        sasBuilder.SetPermissions(BlobSasPermissions.Read);

        var sasUri = new BlobUriBuilder(blobClient.Uri)
        {
            Sas = sasBuilder.ToSasQueryParameters(userDelegationKey, _blobServiceClient.AccountName)
        };

        return sasUri.ToUri();
    }
}
```

---

## 8. Service Bus (MANDATORY)

### A. Service Bus Configuration

```bicep
resource serviceBusNamespace 'Microsoft.ServiceBus/namespaces@2022-10-01-preview' = {
  name: 'sb-${appName}-${environment}'
  location: location
  sku: {
    name: 'Premium'
    tier: 'Premium'
    capacity: 1
  }
  properties: {
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Disabled'
    zoneRedundant: true
  }
}

resource ordersQueue 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = {
  parent: serviceBusNamespace
  name: 'orders'
  properties: {
    maxDeliveryCount: 5
    lockDuration: 'PT1M'
    defaultMessageTimeToLive: 'P7D'
    deadLetteringOnMessageExpiration: true
    requiresDuplicateDetection: true
    duplicateDetectionHistoryTimeWindow: 'PT10M'
  }
}

resource notificationsTopic 'Microsoft.ServiceBus/namespaces/topics@2022-10-01-preview' = {
  parent: serviceBusNamespace
  name: 'notifications'
  properties: {
    defaultMessageTimeToLive: 'P1D'
    maxSizeInMegabytes: 1024
  }
}

resource emailSubscription 'Microsoft.ServiceBus/namespaces/topics/subscriptions@2022-10-01-preview' = {
  parent: notificationsTopic
  name: 'email'
  properties: {
    maxDeliveryCount: 3
    lockDuration: 'PT30S'
    deadLetteringOnMessageExpiration: true
  }
}
```

### B. Service Bus Client

```csharp
using Azure.Identity;
using Azure.Messaging.ServiceBus;

public class ServiceBusService
{
    private readonly ServiceBusClient _client;

    public ServiceBusService(string fullyQualifiedNamespace)
    {
        _client = new ServiceBusClient(fullyQualifiedNamespace, new DefaultAzureCredential());
    }

    public async Task SendMessageAsync(string queueName, object message)
    {
        await using var sender = _client.CreateSender(queueName);

        var serviceBusMessage = new ServiceBusMessage(JsonSerializer.SerializeToUtf8Bytes(message))
        {
            ContentType = "application/json",
            MessageId = Guid.NewGuid().ToString()
        };

        await sender.SendMessageAsync(serviceBusMessage);
    }

    public async Task ProcessMessagesAsync(string queueName, Func<string, Task> handler, CancellationToken cancellationToken)
    {
        await using var processor = _client.CreateProcessor(queueName, new ServiceBusProcessorOptions
        {
            AutoCompleteMessages = false,
            MaxConcurrentCalls = 10
        });

        processor.ProcessMessageAsync += async args =>
        {
            try
            {
                await handler(args.Message.Body.ToString());
                await args.CompleteMessageAsync(args.Message);
            }
            catch (Exception ex)
            {
                await args.AbandonMessageAsync(args.Message);
                throw;
            }
        };

        processor.ProcessErrorAsync += args =>
        {
            Console.WriteLine($"Error: {args.Exception}");
            return Task.CompletedTask;
        };

        await processor.StartProcessingAsync(cancellationToken);
    }
}
```

### C. Session-Enabled Queues and Dead-Letter Handling

Use `requiresSession: true` on queues for ordered processing per group (e.g., per customer). Set `SessionId` on messages, process with `CreateSessionProcessor` with `MaxConcurrentCallsPerSession: 1`.

Monitor dead-letter queues at `{queueName}/$deadletterqueue` for poison messages. Log `DeadLetterReason` and `DeadLetterErrorDescription`, then complete or resubmit.

### D. Topic Subscriptions with Filters

Use `CorrelationFilter` for exact property matching and `SqlFilter` for expression-based routing on topic subscriptions. Always enable `deadLetteringOnMessageExpiration` on subscriptions.

---

## 9. Application Insights (MANDATORY)

### A. Application Insights Configuration

```bicep
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'log-${appName}-${environment}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 90
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${appName}-${environment}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}
```

### B. Integration in .NET

```csharp
// Program.cs - Add telemetry
builder.Services.AddApplicationInsightsTelemetry(options =>
{
    options.ConnectionString = builder.Configuration["APPLICATIONINSIGHTS_CONNECTION_STRING"];
});

// Custom telemetry: TrackEvent, TrackException, StartOperation
_telemetry.TrackEvent("OrderCreated", new Dictionary<string, string>
    { ["OrderId"] = order.Id }, new Dictionary<string, double> { ["Total"] = (double)order.Total });
_telemetry.TrackException(ex);
```

### C. KQL Queries for Azure Monitor

```kql
// Failed requests in the last 24 hours
requests
| where timestamp > ago(24h) and success == false
| summarize count() by resultCode, name, bin(timestamp, 1h)

// Slow API endpoints (>2s response time)
requests
| where timestamp > ago(1h) and duration > 2000
| summarize avg(duration), max(duration), count() by name
| order by avg_duration desc

// Dependency failures (SQL, HTTP, Service Bus)
dependencies
| where timestamp > ago(1h) and success == false
| summarize count() by type, target, resultCode

// End-to-end transaction tracing
union requests, dependencies, exceptions, traces
| where operation_Id == "specific-operation-id"
| order by timestamp asc

// Availability and P95 latency by endpoint
requests
| where timestamp > ago(24h)
| summarize availability = countif(success) * 100.0 / count(), p95 = percentile(duration, 95) by name
```

### D. Alert Rules

Configure `Microsoft.Insights/metricAlerts` for key metrics (Http5xx, response time, CPU). Create `Microsoft.Insights/actionGroups` for email/SMS/webhook notifications. Set `evaluationFrequency: PT5M` and `windowSize: PT15M` for production alerts.

---

## 10. Azure DevOps Pipeline (MANDATORY)

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
      - develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  buildConfiguration: 'Release'
  azureSubscription: 'MyAzureConnection'

stages:
  - stage: Build
    jobs:
      - job: Build
        steps:
          - task: UseDotNet@2
            inputs:
              version: '8.0.x'

          - task: DotNetCoreCLI@2
            displayName: 'Restore'
            inputs:
              command: 'restore'

          - task: DotNetCoreCLI@2
            displayName: 'Build'
            inputs:
              command: 'build'
              arguments: '--configuration $(buildConfiguration)'

          - task: DotNetCoreCLI@2
            displayName: 'Test'
            inputs:
              command: 'test'
              arguments: '--configuration $(buildConfiguration) --collect:"XPlat Code Coverage"'

          - task: DotNetCoreCLI@2
            displayName: 'Publish'
            inputs:
              command: 'publish'
              publishWebProjects: true
              arguments: '--configuration $(buildConfiguration) --output $(Build.ArtifactStagingDirectory)'

          - publish: $(Build.ArtifactStagingDirectory)
            artifact: drop

  - stage: DeployStaging
    dependsOn: Build
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/develop'))
    jobs:
      - deployment: Deploy
        environment: 'staging'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureWebApp@1
                  inputs:
                    azureSubscription: $(azureSubscription)
                    appName: 'app-myapp-staging'
                    package: '$(Pipeline.Workspace)/drop/**/*.zip'

  - stage: DeployProduction
    dependsOn: Build
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - deployment: Deploy
        environment: 'production'
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureWebApp@1
                  inputs:
                    azureSubscription: $(azureSubscription)
                    appName: 'app-myapp-prod'
                    deployToSlotOrASE: true
                    slotName: 'staging'
                    package: '$(Pipeline.Workspace)/drop/**/*.zip'

                - task: AzureAppServiceManage@0
                  inputs:
                    azureSubscription: $(azureSubscription)
                    Action: 'Swap Slots'
                    WebAppName: 'app-myapp-prod'
                    SourceSlot: 'staging'
```

### B. Pipeline Templates

Use YAML templates for reusable multi-environment deployments. Each template accepts parameters for environment, subscription, app name, and resource group. Include Bicep `az deployment group create` and `AzureWebApp@1` tasks.

---

## 11. Bicep Patterns (MANDATORY)

### A. Module Organization

```
infra/
├── main.bicep                  # Entry point
├── parameters/
│   ├── dev.bicepparam          # Environment parameters
│   ├── staging.bicepparam
│   └── prod.bicepparam
├── modules/
│   ├── networking/vnet.bicep, nsg.bicep, private-endpoints.bicep
│   ├── compute/app-service.bicep, function-app.bicep, container-app.bicep
│   ├── data/sql-database.bicep, cosmos-db.bicep, storage-account.bicep
│   ├── security/key-vault.bicep, managed-identity.bicep
│   └── monitoring/app-insights.bicep, log-analytics.bicep
└── bicepconfig.json
```

### B. Main Entry Point

```bicep
// main.bicep
targetScope = 'resourceGroup'
param appName string
@allowed(['dev', 'staging', 'prod'])
param environment string
param location string = resourceGroup().location
param tags object = { Application: appName, Environment: environment, ManagedBy: 'Bicep' }

module networking 'modules/networking/vnet.bicep' = {
  name: 'networking-${uniqueString(deployment().name)}'
  params: { appName: appName, environment: environment, location: location, tags: tags }
}
module identity 'modules/security/managed-identity.bicep' = {
  name: 'identity-${uniqueString(deployment().name)}'
  params: { appName: appName, environment: environment, location: location, tags: tags }
}
module keyVault 'modules/security/key-vault.bicep' = {
  name: 'keyvault-${uniqueString(deployment().name)}'
  params: {
    appName: appName, environment: environment, location: location, tags: tags
    subnetId: networking.outputs.privateEndpointSubnetId
    managedIdentityPrincipalId: identity.outputs.principalId
  }
}
output appServiceUrl string = appService.outputs.defaultHostname
```

### C. Parameter Files, Types, Conditionals, and Loops

```bicep
// parameters/prod.bicepparam
using '../main.bicep'
param appName = 'myapp'
param environment = 'prod'

// User-defined types (types.bicep)
@export()
type environmentType = 'dev' | 'staging' | 'prod'

// Conditional: Deploy resource only in production
resource pe 'Microsoft.Network/privateEndpoints@2023-05-01' = if (environment == 'prod') {
  name: 'pe-st-${appName}'
  // ...
}

// Loop: Create multiple resources from array
resource queues 'Microsoft.ServiceBus/namespaces/queues@2022-10-01-preview' = [
  for queueName in ['orders', 'notifications', 'audit-log']: {
    parent: serviceBusNamespace
    name: queueName
    properties: { maxDeliveryCount: 5, deadLetteringOnMessageExpiration: true }
  }
]
```

### D. Linting and Migration

```json
// bicepconfig.json - key rules
{ "analyzers": { "core": { "rules": {
  "no-hardcoded-env-urls": { "level": "error" },
  "no-unused-params": { "level": "error" },
  "secure-parameter-default": { "level": "error" },
  "no-hardcoded-location": { "level": "error" },
  "use-recent-api-versions": { "level": "warning", "configuration": { "maxAgeInDays": 730 } }
}}}}
```

```bash
az bicep decompile --file azuredeploy.json          # Convert ARM to Bicep
az deployment group what-if --resource-group rg-myapp --template-file main.bicep
```

---

## 12. Container Apps (MANDATORY)

### A. Container App Environment and App

```bicep
resource containerAppEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: 'cae-${appName}-${environment}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: { customerId: logAnalytics.properties.customerId, sharedKey: logAnalytics.listKeys().primarySharedKey }
    }
    vnetConfiguration: { infrastructureSubnetId: containerAppSubnet.id }
    zoneRedundant: environment == 'prod'
  }
}

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ca-${appName}-${environment}'
  location: location
  identity: { type: 'UserAssigned', userAssignedIdentities: { '${managedIdentity.id}': {} } }
  properties: {
    managedEnvironmentId: containerAppEnv.id
    configuration: {
      activeRevisionsMode: 'Multiple'
      ingress: { external: true, targetPort: 8080, traffic: [{ latestRevision: true, weight: 100 }] }
      secrets: [{ name: 'db-conn', keyVaultUrl: '${keyVault.properties.vaultUri}secrets/db-conn', identity: managedIdentity.id }]
      registries: [{ server: '${acr.name}.azurecr.io', identity: managedIdentity.id }]
    }
    template: {
      containers: [{
        name: 'api'
        image: '${acr.name}.azurecr.io/${appName}:latest'
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: [{ name: 'ConnectionStrings__Db', secretRef: 'db-conn' }]
        probes: [{ type: 'Liveness', httpGet: { path: '/health', port: 8080 } }]
      }]
      scale: {
        minReplicas: environment == 'prod' ? 2 : 0
        maxReplicas: 20
        rules: [
          { name: 'http', http: { metadata: { concurrentRequests: '50' } } }
          { name: 'queue', custom: { type: 'azure-servicebus', metadata: { queueName: 'orders', messageCount: '10' } } }
        ]
      }
    }
  }
}
```

### B. Revision Management and Dapr

```bash
# Canary: split traffic between revisions
az containerapp ingress traffic set --name ca-myapp-prod --resource-group rg-myapp-prod \
  --revision-weight ca-myapp-prod--v1=80 ca-myapp-prod--v2=20
# Promote: az containerapp ingress traffic set ... --revision-weight ca-myapp-prod--v2=100
```

Enable Dapr sidecar with `dapr: { enabled: true, appId: appName, appPort: 8080 }` in configuration. Register Dapr components as `managedEnvironments/daprComponents` resources.

---

## 13. Cosmos DB (MANDATORY)

### A. Account, Container, and Partition Key

```bicep
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: 'cosmos-${appName}-${environment}'
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    consistencyPolicy: { defaultConsistencyLevel: 'Session' }
    locations: [
      { locationName: location, failoverPriority: 0, isZoneRedundant: true }
      { locationName: secondaryLocation, failoverPriority: 1, isZoneRedundant: true }
    ]
    enableAutomaticFailover: true
    publicNetworkAccess: 'Disabled'
    disableLocalAuth: true
  }
}

resource ordersContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2023-04-15' = {
  parent: cosmosDb
  name: 'orders'
  properties: {
    resource: {
      id: 'orders'
      partitionKey: { paths: ['/customerId'], kind: 'Hash', version: 2 }
      indexingPolicy: {
        includedPaths: [{ path: '/customerId/?' }, { path: '/status/?' }]
        excludedPaths: [{ path: '/description/?' }]
      }
    }
    options: { autoscaleSettings: { maxThroughput: 4000 } }
  }
}
```

### B. Client and Change Feed

```csharp
var client = new CosmosClient(endpoint, new DefaultAzureCredential(),
    new CosmosClientOptions { ConnectionMode = ConnectionMode.Direct, ConsistencyLevel = ConsistencyLevel.Session });
var container = client.GetContainer(dbName, "orders");

// Point read (most efficient), parameterized queries, Change Feed
var item = await container.ReadItemAsync<Order>(id, new PartitionKey(custId));

// Change Feed for event-driven processing
var processor = monitoredContainer
    .GetChangeFeedProcessorBuilder<Order>("proc", async (ctx, changes, ct) => {
        foreach (var order in changes) await HandleChangeAsync(order);
    })
    .WithInstanceName(Environment.MachineName)
    .WithLeaseContainer(leaseContainer).Build();
await processor.StartAsync();
```

```
PARTITION KEY: High cardinality, even distribution, aligned with queries.
GOOD: /tenantId, /customerId    BAD: /status, /createdDate
CONSISTENCY: Strong > Bounded Staleness > Session (DEFAULT) > Consistent Prefix > Eventual
```

---

## 14. Networking (MANDATORY)

### A. VNet and NSG

```bicep
resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: 'vnet-${appName}-${environment}'
  location: location
  properties: {
    addressSpace: { addressPrefixes: ['10.0.0.0/16'] }
    subnets: [
      { name: 'snet-app', properties: { addressPrefix: '10.0.1.0/24', delegations: [{ name: 'web', properties: { serviceName: 'Microsoft.Web/serverFarms' } }], networkSecurityGroup: { id: appNsg.id } } }
      { name: 'snet-pe', properties: { addressPrefix: '10.0.2.0/24', privateEndpointNetworkPolicies: 'Disabled' } }
      { name: 'snet-aca', properties: { addressPrefix: '10.0.16.0/21', delegations: [{ name: 'aca', properties: { serviceName: 'Microsoft.App/environments' } }] } }
    ]
  }
}

// NSG: AllowHTTPS (100), AllowAzureLB (110), DenyAllInbound (4096)
// PE NSG: AllowVNetInbound (100), DenyAllInbound (4096)
```

### B. Private Endpoints and DNS

```bicep
resource kvPe 'Microsoft.Network/privateEndpoints@2023-05-01' = {
  name: 'pe-kv-${appName}'
  location: location
  properties: {
    subnet: { id: peSubnet.id }
    privateLinkServiceConnections: [{ name: 'kv', properties: { privateLinkServiceId: keyVault.id, groupIds: ['vault'] } }]
  }
}
// Create privateDnsZone + virtualNetworkLink + privateDnsZoneGroup for each PE
// DNS zones: Key Vault=privatelink.vaultcore.azure.net, SQL=privatelink.database.windows.net
// Storage=privatelink.blob.core.windows.net, Cosmos=privatelink.documents.azure.com
```

---

## 15. Front Door, WAF, and Policy (MANDATORY)

Use Azure Front Door Premium for global load balancing with WAF. Enable `Microsoft_DefaultRuleSet` 2.1, `Microsoft_BotManagerRuleSet`, and rate limiting custom rules. Use private link origins to keep backends private.

```bash
az afd profile create --profile-name afd-myapp --resource-group rg-myapp --sku Premium_AzureFrontDoor
az policy assignment create --name "require-tls-12" \
  --policy "/providers/Microsoft.Authorization/policyDefinitions/f0e6e85b-9b9f-4a4b-b67b-f730d42f1b0b" \
  --scope "/subscriptions/${SUBSCRIPTION_ID}"
az policy state summarize --subscription "${SUBSCRIPTION_ID}" --output table
```

---

## 16. Managed Identity Best Practices (MANDATORY)

```
System-Assigned: Tied to resource lifecycle. User-Assigned (RECOMMENDED): Independent, shared.
NEVER: Store keys in app settings, use account keys, hardcode credentials.
ALWAYS: Use DefaultAzureCredential, least-privilege RBAC, user-assigned identity.
```

```csharp
var credential = new DefaultAzureCredential(new DefaultAzureCredentialOptions {
    ManagedIdentityClientId = Environment.GetEnvironmentVariable("AZURE_CLIENT_ID")
});
// Use with: SecretClient, BlobServiceClient, ServiceBusClient, CosmosClient
```

Key role IDs: Key Vault Secrets User (`4633458b`), Storage Blob Data Contributor (`ba92f5b4`), Service Bus Data Sender (`69a216fc`), AcrPull (`7f951dda`).

---

## 17. Key Vault Advanced Patterns (MANDATORY)

Use `@Microsoft.KeyVault` references in App Service settings so secrets load as environment variables without code changes. Set `keyVaultReferenceIdentity` to the managed identity.

```bicep
{ name: 'DbPassword', value: '@Microsoft.KeyVault(VaultName=${kv.name};SecretName=db-password)' }
```

Cache secrets with short TTL (15 minutes) using `IMemoryCache` for automatic rotation support.

---

## 18. Deployment Checklist

### Security
- [ ] Managed Identity (user-assigned), Key Vault for secrets, private endpoints
- [ ] Entra ID auth (disable local auth), NSGs on all subnets, WAF enabled
- [ ] Azure Policy compliant, TLS 1.2 minimum, least-privilege RBAC

### Reliability
- [ ] Zone redundancy, multi-region failover, backups configured
- [ ] Health probes (liveness + readiness), auto-scaling, deployment slots
- [ ] Dead-letter queue monitoring, circuit breaker patterns

### Networking & Operations
- [ ] VNet with subnet segmentation, private endpoints + DNS for all PaaS
- [ ] Application Insights + Log Analytics, alerts for errors/latency, cost tags
- [ ] Bicep modules per environment, what-if validation, pipeline templates

---

## 19. Quick Reference

```bash
az login && az account set --subscription "Name"
az group create --name rg-myapp --location eastus
az webapp up --name app-myapp --resource-group rg-myapp
az functionapp deployment source config-zip --name func-myapp --src app.zip
az containerapp create --name ca-myapp --resource-group rg-myapp --environment cae-myapp --image myacr.azurecr.io/myapp:latest
az keyvault secret show --vault-name kv-myapp --name secret-name
az bicep build --file main.bicep
az deployment group create --resource-group rg-myapp --template-file main.bicep --parameters @parameters/prod.bicepparam
az deployment group what-if --resource-group rg-myapp --template-file main.bicep
az network private-endpoint list --resource-group rg-myapp --output table
az policy state summarize --subscription "${SUBSCRIPTION_ID}"
```

---

**Last Updated:** 2026-02-27
**Version:** 2.0
**Maintainer:** Cloud Team


**End of Microsoft Azure Development Guidelines**

# Microsoft Azure Development Guidelines

This document provides mandatory standards for building applications on Microsoft Azure.

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
    // ...
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

### B. Function Code

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
// Program.cs
builder.Services.AddApplicationInsightsTelemetry(options =>
{
    options.ConnectionString = builder.Configuration["APPLICATIONINSIGHTS_CONNECTION_STRING"];
});

// Custom telemetry
public class OrderService
{
    private readonly TelemetryClient _telemetry;

    public OrderService(TelemetryClient telemetry)
    {
        _telemetry = telemetry;
    }

    public async Task<Order> CreateOrderAsync(CreateOrderRequest request)
    {
        using var operation = _telemetry.StartOperation<RequestTelemetry>("CreateOrder");

        try
        {
            var order = new Order { /* ... */ };

            _telemetry.TrackEvent("OrderCreated", new Dictionary<string, string>
            {
                ["OrderId"] = order.Id,
                ["CustomerId"] = order.CustomerId
            }, new Dictionary<string, double>
            {
                ["OrderTotal"] = (double)order.Total
            });

            return order;
        }
        catch (Exception ex)
        {
            _telemetry.TrackException(ex);
            throw;
        }
    }
}
```

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

---

## 11. Deployment Checklist

### Security
- [ ] Managed Identity configured
- [ ] Key Vault for secrets
- [ ] Private endpoints enabled
- [ ] Azure AD authentication

### Reliability
- [ ] Zone redundancy enabled
- [ ] Backup configured
- [ ] Health probes set up
- [ ] Auto-scaling configured

### Operations
- [ ] Application Insights enabled
- [ ] Log Analytics workspace
- [ ] Alerts configured
- [ ] Cost management tags

---

## 12. Quick Reference

```bash
# Azure CLI common commands
az login
az account set --subscription "Name"
az group create --name rg-myapp --location eastus
az webapp up --name app-myapp --resource-group rg-myapp
az keyvault secret show --vault-name kv-myapp --name secret-name
az functionapp deployment source config-zip --name func-myapp --src app.zip

# Bicep
az deployment group create --resource-group rg-myapp --template-file main.bicep
az bicep build --file main.bicep
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Cloud Team

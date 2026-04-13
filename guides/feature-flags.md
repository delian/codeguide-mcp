# Feature Flags Guidelines
Mandatory standards for implementing and managing feature flags in production systems. LaunchDarkly, Split.io, Unleash, Flagsmith, custom implementations.

---

**Agent Profile**: The Feature Flags Expert
**Role**: Senior Release Engineer & Product Delivery Specialist
**Objective**: Generate safe, scalable feature flag implementations that enable progressive delivery and experimentation.
**Tools**: LaunchDarkly, Split.io, Unleash, Flagsmith, custom implementations.

---

## 1. Core Philosophies: FLAGS-FIRST

- **F**lexible: Enable runtime configuration changes
- **L**imited: Flags have a lifecycle, remove when done
- **A**uditable: Track all flag changes and usage
- **G**radual: Roll out features incrementally
- **S**afe: Kill switches for instant rollback

---

## 2. Flag Types (MANDATORY)

### A. Flag Categories

```typescript
// Define clear flag types
enum FlagType {
  // Release flags: Control feature rollout
  // Short-lived, should be removed after full rollout
  RELEASE = 'release',

  // Experiment flags: A/B testing
  // Time-boxed, removed after experiment concludes
  EXPERIMENT = 'experiment',

  // Ops flags: Operational controls
  // Can be long-lived, for system behavior tuning
  OPS = 'ops',

  // Permission flags: Entitlement-based features
  // Long-lived, tied to pricing/plans
  PERMISSION = 'permission',

  // Kill switch: Emergency disable
  // Always present, default to enabled
  KILL_SWITCH = 'kill_switch'
}

interface FeatureFlag {
  key: string;
  type: FlagType;
  description: string;
  owner: string;
  createdAt: Date;
  expiresAt?: Date; // Required for RELEASE and EXPERIMENT
  defaultValue: boolean | string | number | object;
  tags: string[];
}

// Example flag definitions
const flags: FeatureFlag[] = [
  {
    key: 'new-checkout-flow',
    type: FlagType.RELEASE,
    description: 'New streamlined checkout experience',
    owner: 'checkout-team',
    createdAt: new Date('2024-01-15'),
    expiresAt: new Date('2024-03-15'), // 2 month max
    defaultValue: false,
    tags: ['checkout', 'frontend']
  },
  {
    key: 'pricing-experiment-v2',
    type: FlagType.EXPERIMENT,
    description: 'Test new pricing display format',
    owner: 'growth-team',
    createdAt: new Date('2024-01-20'),
    expiresAt: new Date('2024-02-20'), // 1 month experiment
    defaultValue: 'control',
    tags: ['pricing', 'experiment']
  },
  {
    key: 'rate-limit-threshold',
    type: FlagType.OPS,
    description: 'Requests per minute limit',
    owner: 'platform-team',
    createdAt: new Date('2024-01-01'),
    defaultValue: 100,
    tags: ['ops', 'rate-limiting']
  },
  {
    key: 'payments-enabled',
    type: FlagType.KILL_SWITCH,
    description: 'Kill switch for payment processing',
    owner: 'payments-team',
    createdAt: new Date('2024-01-01'),
    defaultValue: true, // Enabled by default
    tags: ['payments', 'critical']
  }
];
```

---

## 3. Implementation Patterns (MANDATORY)

### A. Client SDK

```typescript
// feature-flags.ts
interface FlagContext {
  userId?: string;
  email?: string;
  userAttributes?: Record<string, any>;
  sessionId?: string;
  environment: 'development' | 'staging' | 'production';
}

interface FlagValue<T> {
  value: T;
  variation: string;
  reason: string;
}

class FeatureFlagClient {
  private flags: Map<string, any> = new Map();
  private context: FlagContext;
  private eventQueue: FlagEvaluationEvent[] = [];

  constructor(config: { apiKey: string; context: FlagContext }) {
    this.context = config.context;
    this.initialize(config.apiKey);
  }

  private async initialize(apiKey: string): Promise<void> {
    // Fetch initial flag values
    const response = await fetch('/api/flags', {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    });
    const data = await response.json();
    data.flags.forEach((flag: any) => {
      this.flags.set(flag.key, flag);
    });

    // Set up real-time updates (SSE or WebSocket)
    this.subscribeToUpdates(apiKey);
  }

  // Boolean flag evaluation
  isEnabled(flagKey: string, defaultValue: boolean = false): boolean {
    return this.evaluate(flagKey, defaultValue).value;
  }

  // String variation evaluation
  getVariation(flagKey: string, defaultValue: string = 'control'): string {
    return this.evaluate(flagKey, defaultValue).value;
  }

  // Numeric flag evaluation
  getNumber(flagKey: string, defaultValue: number = 0): number {
    return this.evaluate(flagKey, defaultValue).value;
  }

  // JSON flag evaluation
  getJSON<T>(flagKey: string, defaultValue: T): T {
    return this.evaluate(flagKey, defaultValue).value;
  }

  private evaluate<T>(flagKey: string, defaultValue: T): FlagValue<T> {
    const flag = this.flags.get(flagKey);

    if (!flag) {
      this.trackEvaluation(flagKey, 'not-found', defaultValue);
      return {
        value: defaultValue,
        variation: 'default',
        reason: 'FLAG_NOT_FOUND'
      };
    }

    // Evaluate targeting rules
    const result = this.evaluateRules(flag, this.context);

    this.trackEvaluation(flagKey, result.variation, result.value);

    return result;
  }

  private evaluateRules<T>(flag: any, context: FlagContext): FlagValue<T> {
    // Check kill switch
    if (flag.killed) {
      return {
        value: flag.offVariation,
        variation: 'off',
        reason: 'KILLED'
      };
    }

    // Check user targeting
    if (flag.targets && context.userId) {
      for (const target of flag.targets) {
        if (target.userIds.includes(context.userId)) {
          return {
            value: target.variation,
            variation: target.name,
            reason: 'TARGETED'
          };
        }
      }
    }

    // Check percentage rollout
    if (flag.rollout && context.userId) {
      const bucket = this.hashUser(context.userId, flag.key) % 100;
      if (bucket < flag.rollout.percentage) {
        return {
          value: flag.onVariation,
          variation: 'on',
          reason: 'ROLLOUT'
        };
      }
    }

    // Default variation
    return {
      value: flag.defaultVariation,
      variation: 'default',
      reason: 'DEFAULT'
    };
  }

  private hashUser(userId: string, flagKey: string): number {
    // Consistent hashing for stable bucketing
    const str = `${flagKey}:${userId}`;
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return Math.abs(hash);
  }

  private trackEvaluation(flagKey: string, variation: string, value: any): void {
    this.eventQueue.push({
      flagKey,
      variation,
      value,
      userId: this.context.userId,
      timestamp: new Date()
    });

    // Batch send events
    if (this.eventQueue.length >= 10) {
      this.flushEvents();
    }
  }

  private async flushEvents(): Promise<void> {
    const events = [...this.eventQueue];
    this.eventQueue = [];

    await fetch('/api/flags/events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events })
    });
  }
}

// Usage
const flags = new FeatureFlagClient({
  apiKey: process.env.FLAG_API_KEY!,
  context: {
    userId: currentUser.id,
    email: currentUser.email,
    environment: 'production'
  }
});

if (flags.isEnabled('new-checkout-flow')) {
  renderNewCheckout();
} else {
  renderLegacyCheckout();
}
```

### B. Server-Side Implementation

```typescript
// flag-service.ts
import Redis from 'ioredis';

interface TargetingRule {
  attribute: string;
  operator: 'equals' | 'contains' | 'in' | 'gt' | 'lt';
  values: any[];
  variation: string;
}

interface FlagConfig {
  key: string;
  enabled: boolean;
  defaultVariation: string;
  variations: Record<string, any>;
  rules: TargetingRule[];
  rollout?: {
    percentage: number;
    variation: string;
  };
  killSwitch: boolean;
}

class FeatureFlagService {
  private redis: Redis;
  private localCache: Map<string, FlagConfig> = new Map();
  private cacheExpiry: number = 60000; // 1 minute
  private lastRefresh: number = 0;

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
    this.startCacheRefresh();
  }

  async evaluate(
    flagKey: string,
    context: Record<string, any>,
    defaultValue: any = null
  ): Promise<any> {
    const flag = await this.getFlag(flagKey);

    if (!flag) {
      return defaultValue;
    }

    // Kill switch check
    if (flag.killSwitch) {
      return flag.variations[flag.defaultVariation];
    }

    // Not enabled
    if (!flag.enabled) {
      return flag.variations[flag.defaultVariation];
    }

    // Evaluate targeting rules
    for (const rule of flag.rules) {
      if (this.matchesRule(rule, context)) {
        return flag.variations[rule.variation];
      }
    }

    // Percentage rollout
    if (flag.rollout && context.userId) {
      const inRollout = this.isInRollout(
        context.userId,
        flagKey,
        flag.rollout.percentage
      );
      if (inRollout) {
        return flag.variations[flag.rollout.variation];
      }
    }

    // Default
    return flag.variations[flag.defaultVariation];
  }

  private matchesRule(rule: TargetingRule, context: Record<string, any>): boolean {
    const value = context[rule.attribute];

    switch (rule.operator) {
      case 'equals':
        return value === rule.values[0];
      case 'contains':
        return String(value).includes(rule.values[0]);
      case 'in':
        return rule.values.includes(value);
      case 'gt':
        return value > rule.values[0];
      case 'lt':
        return value < rule.values[0];
      default:
        return false;
    }
  }

  private isInRollout(userId: string, flagKey: string, percentage: number): boolean {
    const hash = this.consistentHash(`${flagKey}:${userId}`);
    return hash % 100 < percentage;
  }

  private consistentHash(str: string): number {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) + hash) + str.charCodeAt(i);
    }
    return Math.abs(hash);
  }

  private async getFlag(key: string): Promise<FlagConfig | null> {
    // Try local cache first
    if (this.localCache.has(key) && Date.now() - this.lastRefresh < this.cacheExpiry) {
      return this.localCache.get(key)!;
    }

    // Fetch from Redis
    const data = await this.redis.get(`flag:${key}`);
    if (data) {
      const flag = JSON.parse(data);
      this.localCache.set(key, flag);
      return flag;
    }

    return null;
  }

  private startCacheRefresh(): void {
    // Subscribe to flag updates
    const subscriber = this.redis.duplicate();
    subscriber.subscribe('flag-updates');
    subscriber.on('message', (channel, message) => {
      const { key, flag } = JSON.parse(message);
      this.localCache.set(key, flag);
    });

    // Periodic full refresh
    setInterval(() => {
      this.refreshAllFlags();
    }, 60000);
  }

  private async refreshAllFlags(): Promise<void> {
    const keys = await this.redis.keys('flag:*');
    const pipeline = this.redis.pipeline();
    keys.forEach(key => pipeline.get(key));
    const results = await pipeline.exec();

    results?.forEach((result, index) => {
      if (result[1]) {
        const flag = JSON.parse(result[1] as string);
        this.localCache.set(flag.key, flag);
      }
    });
    this.lastRefresh = Date.now();
  }

  // Admin API
  async updateFlag(key: string, updates: Partial<FlagConfig>): Promise<void> {
    const flag = await this.getFlag(key);
    if (!flag) throw new Error(`Flag ${key} not found`);

    const updated = { ...flag, ...updates };
    await this.redis.set(`flag:${key}`, JSON.stringify(updated));
    await this.redis.publish('flag-updates', JSON.stringify({ key, flag: updated }));
  }

  async setRolloutPercentage(key: string, percentage: number): Promise<void> {
    await this.updateFlag(key, {
      rollout: { percentage, variation: 'on' }
    });
  }

  async enableKillSwitch(key: string): Promise<void> {
    await this.updateFlag(key, { killSwitch: true });
  }
}
```

---

## 4. Rollout Strategies (MANDATORY)

### A. Percentage Rollout

```typescript
// Gradual rollout example
class RolloutManager {
  constructor(private flagService: FeatureFlagService) {}

  // Progressive rollout schedule
  async executeRollout(flagKey: string, schedule: RolloutSchedule): Promise<void> {
    for (const stage of schedule.stages) {
      // Set percentage
      await this.flagService.setRolloutPercentage(flagKey, stage.percentage);

      // Wait and monitor
      await this.waitAndMonitor(flagKey, stage.duration, stage.criteria);

      // Check if we should proceed
      const metrics = await this.getMetrics(flagKey);
      if (!this.meetsSuccessCriteria(metrics, stage.criteria)) {
        // Rollback
        await this.flagService.setRolloutPercentage(flagKey, 0);
        throw new Error(`Rollout failed at ${stage.percentage}%`);
      }
    }
  }

  private async waitAndMonitor(
    flagKey: string,
    duration: number,
    criteria: SuccessCriteria
  ): Promise<void> {
    const startTime = Date.now();
    while (Date.now() - startTime < duration) {
      const metrics = await this.getMetrics(flagKey);

      // Check for critical failures
      if (metrics.errorRate > criteria.maxErrorRate) {
        throw new Error('Error rate exceeded threshold');
      }

      await sleep(60000); // Check every minute
    }
  }
}

// Usage
const rollout = new RolloutManager(flagService);

await rollout.executeRollout('new-feature', {
  stages: [
    { percentage: 1, duration: 3600000, criteria: { maxErrorRate: 0.01 } },   // 1% for 1 hour
    { percentage: 10, duration: 86400000, criteria: { maxErrorRate: 0.01 } }, // 10% for 1 day
    { percentage: 50, duration: 86400000, criteria: { maxErrorRate: 0.01 } }, // 50% for 1 day
    { percentage: 100, duration: 0, criteria: { maxErrorRate: 0.01 } }        // 100%
  ]
});
```

### B. Ring-Based Rollout

```typescript
// Deploy to groups in sequence
interface RolloutRing {
  name: string;
  targeting: {
    attribute: string;
    values: string[];
  };
  duration: number;
}

const rolloutRings: RolloutRing[] = [
  {
    name: 'internal',
    targeting: { attribute: 'email', values: ['@company.com'] },
    duration: 86400000 // 1 day
  },
  {
    name: 'beta-users',
    targeting: { attribute: 'tier', values: ['beta'] },
    duration: 259200000 // 3 days
  },
  {
    name: 'early-adopters',
    targeting: { attribute: 'signupDate', values: ['<2023-01-01'] },
    duration: 604800000 // 1 week
  },
  {
    name: 'all-users',
    targeting: { attribute: 'any', values: ['*'] },
    duration: 0
  }
];

async function ringRollout(flagKey: string, rings: RolloutRing[]): Promise<void> {
  for (const ring of rings) {
    console.log(`Rolling out to ${ring.name}`);

    // Add targeting rule for this ring
    await flagService.addTargetingRule(flagKey, {
      attribute: ring.targeting.attribute,
      operator: ring.targeting.attribute === 'email' ? 'contains' : 'in',
      values: ring.targeting.values,
      variation: 'on'
    });

    // Wait and monitor
    if (ring.duration > 0) {
      await monitorRollout(flagKey, ring.duration);
    }
  }
}
```

---

## 5. A/B Testing (MANDATORY)

### A. Experiment Configuration

```typescript
interface Experiment {
  key: string;
  name: string;
  hypothesis: string;
  startDate: Date;
  endDate: Date;
  trafficPercentage: number;
  variations: {
    key: string;
    name: string;
    weight: number; // Percentage of traffic
  }[];
  metrics: {
    primary: string;
    secondary: string[];
  };
  minimumSampleSize: number;
}

const experiment: Experiment = {
  key: 'checkout-button-color',
  name: 'Checkout Button Color Test',
  hypothesis: 'A green checkout button will increase conversion by 5%',
  startDate: new Date('2024-02-01'),
  endDate: new Date('2024-02-14'),
  trafficPercentage: 100,
  variations: [
    { key: 'control', name: 'Blue Button', weight: 50 },
    { key: 'treatment', name: 'Green Button', weight: 50 }
  ],
  metrics: {
    primary: 'checkout_conversion_rate',
    secondary: ['cart_abandonment_rate', 'revenue_per_visitor']
  },
  minimumSampleSize: 10000
};

class ExperimentService {
  async assignVariation(
    experimentKey: string,
    userId: string
  ): Promise<string> {
    const experiment = await this.getExperiment(experimentKey);

    // Check if experiment is active
    if (!this.isExperimentActive(experiment)) {
      return 'control';
    }

    // Consistent assignment based on user ID
    const hash = this.hashUser(userId, experimentKey);

    // Check if user is in experiment traffic
    if (hash % 100 >= experiment.trafficPercentage) {
      return 'control'; // Not in experiment
    }

    // Assign to variation based on weights
    const variationHash = this.hashUser(userId, `${experimentKey}:variation`);
    let cumulative = 0;
    for (const variation of experiment.variations) {
      cumulative += variation.weight;
      if (variationHash % 100 < cumulative) {
        return variation.key;
      }
    }

    return 'control';
  }

  async trackConversion(
    experimentKey: string,
    userId: string,
    metricName: string,
    value: number = 1
  ): Promise<void> {
    const variation = await this.assignVariation(experimentKey, userId);

    await this.analytics.track({
      event: 'experiment_conversion',
      properties: {
        experiment: experimentKey,
        variation,
        metric: metricName,
        value,
        userId
      }
    });
  }

  async getResults(experimentKey: string): Promise<ExperimentResults> {
    const experiment = await this.getExperiment(experimentKey);
    const results: ExperimentResults = {
      experiment: experimentKey,
      variations: {}
    };

    for (const variation of experiment.variations) {
      const metrics = await this.calculateMetrics(experimentKey, variation.key);
      results.variations[variation.key] = {
        sampleSize: metrics.sampleSize,
        conversionRate: metrics.conversionRate,
        confidence: metrics.confidence,
        improvement: metrics.improvement
      };
    }

    // Statistical significance calculation
    results.isSignificant = this.calculateSignificance(results);

    return results;
  }
}
```

---

## 6. Lifecycle Management (MANDATORY)

### A. Flag Cleanup

```typescript
// flag-lifecycle.ts
interface FlagMetadata {
  key: string;
  type: FlagType;
  createdAt: Date;
  expiresAt?: Date;
  lastEvaluated?: Date;
  evaluationCount: number;
  owner: string;
  status: 'active' | 'stale' | 'expired';
}

class FlagLifecycleManager {
  async auditFlags(): Promise<FlagAuditReport> {
    const flags = await this.getAllFlags();
    const report: FlagAuditReport = {
      total: flags.length,
      active: [],
      stale: [],
      expired: [],
      recommendations: []
    };

    for (const flag of flags) {
      const status = this.assessFlagStatus(flag);

      switch (status) {
        case 'active':
          report.active.push(flag.key);
          break;
        case 'stale':
          report.stale.push(flag.key);
          report.recommendations.push({
            flag: flag.key,
            action: 'review',
            reason: `Not evaluated in ${this.daysSince(flag.lastEvaluated!)} days`
          });
          break;
        case 'expired':
          report.expired.push(flag.key);
          report.recommendations.push({
            flag: flag.key,
            action: 'remove',
            reason: `Expired on ${flag.expiresAt?.toISOString()}`
          });
          break;
      }
    }

    return report;
  }

  private assessFlagStatus(flag: FlagMetadata): 'active' | 'stale' | 'expired' {
    // Check expiration
    if (flag.expiresAt && new Date() > flag.expiresAt) {
      return 'expired';
    }

    // Check staleness (no evaluations in 30 days)
    if (flag.lastEvaluated && this.daysSince(flag.lastEvaluated) > 30) {
      return 'stale';
    }

    // Release flags at 100% for 14 days should be removed
    if (flag.type === FlagType.RELEASE) {
      const rollout = this.getRolloutPercentage(flag.key);
      if (rollout === 100 && this.daysSince(flag.lastEvaluated!) > 14) {
        return 'expired';
      }
    }

    return 'active';
  }

  async cleanupFlag(flagKey: string): Promise<void> {
    // 1. Document the removal
    await this.documentFlagRemoval(flagKey);

    // 2. Remove from code (generate PR)
    await this.generateCleanupPR(flagKey);

    // 3. Archive flag data
    await this.archiveFlag(flagKey);

    // 4. Delete flag configuration
    await this.deleteFlag(flagKey);
  }

  private async generateCleanupPR(flagKey: string): Promise<void> {
    // Find all usages in code
    const usages = await this.findFlagUsages(flagKey);

    // Generate code changes
    const changes = usages.map(usage => ({
      file: usage.file,
      line: usage.line,
      before: usage.code,
      after: this.generateCleanedCode(usage)
    }));

    // Create PR
    await this.createPullRequest({
      title: `Remove feature flag: ${flagKey}`,
      description: `
        ## Flag Removal

        Removing feature flag \`${flagKey}\` as it has been at 100% rollout for 14+ days.

        ### Files Changed
        ${changes.map(c => `- ${c.file}`).join('\n')}

        ### Verification
        - [ ] All tests pass
        - [ ] No runtime errors in staging
        - [ ] Monitoring shows no issues
      `,
      changes
    });
  }
}
```

### B. Flag Documentation

```typescript
// Generate flag documentation
async function generateFlagDocumentation(): Promise<string> {
  const flags = await flagService.getAllFlags();

  let doc = '# Feature Flags\n\n';

  // Group by type
  const grouped = groupBy(flags, 'type');

  for (const [type, typeFlags] of Object.entries(grouped)) {
    doc += `## ${type} Flags\n\n`;

    for (const flag of typeFlags as FlagMetadata[]) {
      doc += `### ${flag.key}\n\n`;
      doc += `**Description:** ${flag.description}\n`;
      doc += `**Owner:** ${flag.owner}\n`;
      doc += `**Created:** ${flag.createdAt.toISOString()}\n`;
      if (flag.expiresAt) {
        doc += `**Expires:** ${flag.expiresAt.toISOString()}\n`;
      }
      doc += `**Default:** ${JSON.stringify(flag.defaultValue)}\n`;
      doc += '\n';
    }
  }

  return doc;
}
```

---

## 7. React Integration

```tsx
// feature-flag-provider.tsx
import React, { createContext, useContext, useEffect, useState } from 'react';

interface FlagContextType {
  isEnabled: (key: string) => boolean;
  getVariation: (key: string) => string;
  loading: boolean;
}

const FlagContext = createContext<FlagContextType | null>(null);

export function FeatureFlagProvider({
  children,
  userId
}: {
  children: React.ReactNode;
  userId: string;
}) {
  const [flags, setFlags] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadFlags() {
      const response = await fetch(`/api/flags?userId=${userId}`);
      const data = await response.json();
      setFlags(data.flags);
      setLoading(false);
    }
    loadFlags();
  }, [userId]);

  const value: FlagContextType = {
    isEnabled: (key) => flags[key]?.enabled ?? false,
    getVariation: (key) => flags[key]?.variation ?? 'control',
    loading
  };

  return (
    <FlagContext.Provider value={value}>
      {children}
    </FlagContext.Provider>
  );
}

export function useFeatureFlag(key: string): boolean {
  const context = useContext(FlagContext);
  if (!context) throw new Error('Must be used within FeatureFlagProvider');
  return context.isEnabled(key);
}

export function useVariation(key: string): string {
  const context = useContext(FlagContext);
  if (!context) throw new Error('Must be used within FeatureFlagProvider');
  return context.getVariation(key);
}

// Component wrapper
export function Feature({
  flag,
  children,
  fallback = null
}: {
  flag: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  const enabled = useFeatureFlag(flag);
  return <>{enabled ? children : fallback}</>;
}

// Usage
function App() {
  return (
    <FeatureFlagProvider userId={currentUser.id}>
      <Feature flag="new-dashboard" fallback={<OldDashboard />}>
        <NewDashboard />
      </Feature>
    </FeatureFlagProvider>
  );
}
```

---

## 8. Deployment Checklist

### Flag Design
- [ ] Clear naming convention followed
- [ ] Type and lifecycle defined
- [ ] Owner assigned
- [ ] Expiration date set (for release/experiment)
- [ ] Default value is safe

### Implementation
- [ ] Server-side evaluation for security-sensitive flags
- [ ] Consistent hashing for stable assignment
- [ ] Caching with appropriate TTL
- [ ] Fallback values defined

### Operations
- [ ] Monitoring dashboard set up
- [ ] Kill switch tested
- [ ] Audit logging enabled
- [ ] Cleanup process scheduled

### Testing
- [ ] Both flag states tested
- [ ] Integration tests include flag variations
- [ ] Performance impact measured

---

## 9. Quick Reference

```typescript
// Common patterns
flags.isEnabled('feature-key')
flags.getVariation('experiment-key')
flags.getNumber('rate-limit')
flags.getJSON('config')

// Targeting
{ attribute: 'email', operator: 'contains', values: ['@company.com'] }
{ attribute: 'tier', operator: 'in', values: ['premium', 'enterprise'] }
{ attribute: 'signupDate', operator: 'lt', values: ['2024-01-01'] }

// Rollout percentages
1%   → Internal testing
10%  → Early adopters
50%  → Half traffic
100% → Full rollout

// Flag lifecycle
RELEASE    → 2 months max
EXPERIMENT → 1 month max
OPS        → Review quarterly
PERMISSION → Permanent (until plan changes)
```

---

## 10. Why This Configuration Works

- **Progressive delivery reduces deployment risk**: Percentage-based rollouts and ring-based deployment allow teams to expose new features to small user populations first, catching issues before they affect the entire user base. This turns risky big-bang releases into controlled, reversible experiments.
- **Kill switches enable instant rollback**: Having pre-configured kill switches for critical features means any production issue can be mitigated in seconds without a code deployment, dramatically reducing incident duration.
- **Typed flag categories enforce lifecycle discipline**: Classifying flags as release, experiment, ops, or permission types with explicit expiration dates prevents the accumulation of stale flags that create technical debt and make code increasingly difficult to reason about.
- **Consistent hashing ensures stable user experience**: Using deterministic hash-based bucketing ensures individual users always see the same flag variation across sessions and page loads, preventing confusing experience inconsistencies during rollouts.
- **Automated cleanup keeps the codebase clean**: The lifecycle management process with audit reports, automated cleanup PRs, and staleness detection ensures feature flags remain a temporary deployment mechanism rather than a permanent source of branching complexity.

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Platform Team


**End of Feature Flags Guidelines**

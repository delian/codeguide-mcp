# Zod Development Guidelines
Mandatory coding standards and development practices for Zod schema validation in TypeScript. Type-safe validation, OpenAI structured outputs, runtime data integrity. Zod 3.x/4.x, TypeScript 5.x, Vitest, OpenAI SDK.

---

**Agent Profile**: The Zod Validation Expert
**Role**: Senior TypeScript Validation Engineer & AI Integration Specialist
**Objective**: Generate production-ready, type-safe, rigorously validated code with OpenAI structured output integration.
**Tools**: Zod 3.24+/4.x, TypeScript 5.x, Vitest 3.x, OpenAI Node SDK 4.x+, ESLint, Prettier

---

## 1. Core Philosophies: VALID-FIRST

The agent must adhere to the **VALID-FIRST** principles for every Zod implementation:

**Test-Driven Development (TDD)**: ALWAYS write tests BEFORE implementation (Red-Green-Refactor cycle mandatory).
**Regression Shield**: EVERY bug discovered MUST receive a test BEFORE fixing to prevent regression.

- **V**alidate at boundaries: Parse ALL external data (API responses, user input, env vars, file reads) with Zod schemas at system boundaries. Never trust unvalidated data.
- **A**utomate types: ALWAYS infer TypeScript types from Zod schemas with `z.infer<>`. Never maintain parallel type definitions — the schema IS the source of truth.
- **L**ayer schemas: Compose complex schemas from small, reusable building blocks. Prefer `.extend()`, `.merge()`, `.pick()`, `.omit()` over duplication.
- **I**ntegrate with AI: Use `zodResponseFormat()`, `zodTextFormat()`, and `zodFunction()` for OpenAI structured outputs. Let Zod enforce the contract between your app and the LLM.
- **D**efend with parsing: Use `schema.parse()` for fail-fast or `schema.safeParse()` for graceful error handling. NEVER use type assertions (`as`) to bypass validation.

**Additional Principles:**

- Schema-first design: Define Zod schemas before writing business logic
- Coerce at the edge: Use `z.coerce.*` for external string inputs (query params, env vars, form data)
- Fail loudly: Validation errors should surface immediately with actionable messages

**Verified Code**: Agent-generated code MUST pass `tsc --noEmit`, all Vitest tests, and ESLint checks before delivery.

---

## 2. Agent Code Generation Requirements (MANDATORY)

### A. Verification Protocol

**CRITICAL: Agents MUST verify that all generated Zod code compiles with strict TypeScript and passes all tests before presenting it to the user.**

#### Pre-Delivery Checklist

**Before delivering ANY Zod code, the agent MUST:**

1. **Type Check**:
   ```bash
   # Verify TypeScript compilation with strict mode
   npx tsc --noEmit --strict
   # Exit code MUST be 0

   # Verify no implicit any
   npx tsc --noEmit --noImplicitAny
   ```
   - **MUST** have zero type errors
   - All `z.infer<>` types must resolve correctly
   - No `any` types in schema-derived code

2. **Test Verification**:
   ```bash
   # Run all tests
   npx vitest run

   # Run with coverage
   npx vitest run --coverage
   ```
   - **MUST** have all tests passing
   - Schema validation tests cover valid AND invalid inputs

3. **Lint Check**:
   ```bash
   # Run linter
   npx eslint src/ --ext .ts

   # Format check
   npx prettier --check "src/**/*.ts"
   ```
   - No lint errors
   - Code formatted consistently

#### Error Correction Process

If verification fails:

1. **Type Errors**:
   - Read full error message
   - Check schema definition matches expected shape
   - Verify `z.infer<>` usage is correct
   - Re-verify

2. **Test Failures**:
   - Run failing test in isolation
   - Check test expectations vs actual parsed output
   - Fix schema logic or test expectations
   - Re-run all tests to ensure no regressions

3. **Schema Mismatch with OpenAI**:
   - Verify schema uses only JSON Schema-compatible types
   - Remove `.transform()` and `.refine()` from schemas sent to OpenAI
   - Use `.describe()` or `.meta()` for field documentation
   - Re-verify with `z.toJSONSchema()`

### B. Agent Workflow Example

**Complete Zod generation workflow:**

1. **Generate Code Structure**:
   ```
   src/
   ├── schemas/
   │   ├── user.schema.ts
   │   └── index.ts
   ├── services/
   │   └── user.service.ts
   └── __tests__/
       └── user.schema.test.ts
   ```

2. **Generate Schema**:
   ```typescript
   import { z } from "zod";

   export const UserSchema = z.object({
     id: z.string().uuid(),
     email: z.string().email(),
     name: z.string().min(1).max(100),
     role: z.enum(["admin", "user", "viewer"]),
     createdAt: z.coerce.date(),
   });

   export type User = z.infer<typeof UserSchema>;
   ```

3. **Verify**:
   ```bash
   npx tsc --noEmit
   # ✓ Verification successful
   ```

4. **Add Tests**:
   ```typescript
   import { describe, it, expect } from "vitest";
   import { UserSchema } from "../schemas/user.schema";

   describe("UserSchema", () => {
     it("parses valid user", () => {
       const result = UserSchema.safeParse({
         id: "550e8400-e29b-41d4-a716-446655440000",
         email: "user@example.com",
         name: "Alice",
         role: "admin",
         createdAt: "2024-01-15T10:30:00Z",
       });
       expect(result.success).toBe(true);
     });

     it("rejects invalid email", () => {
       const result = UserSchema.safeParse({
         id: "550e8400-e29b-41d4-a716-446655440000",
         email: "not-an-email",
         name: "Alice",
         role: "admin",
         createdAt: "2024-01-15T10:30:00Z",
       });
       expect(result.success).toBe(false);
     });
   });
   ```

5. **Run Tests**:
   ```bash
   npx vitest run
   # ✓ All tests pass
   ```

6. **Final Verification**:
   ```bash
   npx tsc --noEmit && npx vitest run && npx eslint src/
   # ✓ All checks passed
   ```

7. **Present Code**: Only after ALL checks pass

### C. Prohibited Practices

**NEVER deliver Zod code that:**
- [ ] Fails TypeScript strict compilation
- [ ] Has failing tests
- [ ] Lacks tests for both valid and invalid inputs
- [ ] Uses `as` type assertions instead of Zod parsing
- [ ] Maintains separate TypeScript interfaces alongside Zod schemas
- [ ] Uses `z.any()` or `z.unknown()` without subsequent refinement
- [ ] Catches and silently swallows ZodError
- [ ] Sends schemas with `.transform()` or `.refine()` to OpenAI structured outputs
- [ ] **Fixes bugs without adding regression tests first**
- [ ] **Writes implementation before writing tests (violates TDD)**
- [ ] **Skips Red-Green-Refactor cycle for new features**

---

## 2A. Test-Driven Development (TDD) Protocol (MANDATORY)

**CRITICAL: Follow the Red-Green-Refactor cycle for ALL new code.**

### TDD Cycle

```
1. RED: Write a failing test first
   ↓
2. GREEN: Write minimal code to make it pass
   ↓
3. REFACTOR: Improve code while keeping tests green
   ↓
   Repeat
```

### Example TDD Workflow for Zod

```typescript
// Step 1: RED - Write failing test first
import { describe, it, expect } from "vitest";
import { AddressSchema } from "../schemas/address.schema";

describe("AddressSchema", () => {
  it("parses a valid US address", () => {
    const result = AddressSchema.safeParse({
      street: "123 Main St",
      city: "Springfield",
      state: "IL",
      zip: "62701",
      country: "US",
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.state).toBe("IL");
    }
  });

  it("rejects zip code with wrong format", () => {
    const result = AddressSchema.safeParse({
      street: "123 Main St",
      city: "Springfield",
      state: "IL",
      zip: "abc",
      country: "US",
    });
    expect(result.success).toBe(false);
  });
});

// Run: npx vitest run
// FAILS - AddressSchema doesn't exist yet

// Step 2: GREEN - Write minimal implementation
import { z } from "zod";

export const AddressSchema = z.object({
  street: z.string().min(1),
  city: z.string().min(1),
  state: z.string().length(2),
  zip: z.string().regex(/^\d{5}(-\d{4})?$/),
  country: z.string().length(2),
});

export type Address = z.infer<typeof AddressSchema>;

// Run: npx vitest run
// PASSES - tests pass

// Step 3: REFACTOR - Extract reusable parts
const USStateCode = z.string().length(2).toUpperCase();
const USZipCode = z.string().regex(/^\d{5}(-\d{4})?$/, "Invalid ZIP code format");
const CountryCode = z.string().length(2).toUpperCase();

export const AddressSchema = z.object({
  street: z.string().min(1, "Street is required"),
  city: z.string().min(1, "City is required"),
  state: USStateCode,
  zip: USZipCode,
  country: CountryCode,
});
// Tests still pass
```

---

## 2B. Bug Fix Protocol (MANDATORY)

**CRITICAL: Every bug MUST receive a regression test BEFORE fixing.**

### Bug Fix Workflow

```
1. Bug Reported/Discovered
   ↓
2. Write a test that REPRODUCES the bug (test will FAIL)
   ↓
3. Verify the test fails for the right reason
   ↓
4. Fix the bug (make the test pass)
   ↓
5. Verify the test now PASSES
   ↓
6. Document the bug in test comments (include bug ID)
   ↓
7. Deploy with confidence (regression prevented)
```

### Example Bug Fix

```typescript
// Bug Report #412: Schema accepts negative age values

// Step 1-2: Write test that reproduces the bug
import { describe, it, expect } from "vitest";
import { ProfileSchema } from "../schemas/profile.schema";

describe("ProfileSchema - Bug #412", () => {
  it("rejects negative age values", () => {
    // Bug: ProfileSchema.parse({ name: "Alice", age: -5 }) succeeded
    // Expected: should reject negative ages
    const result = ProfileSchema.safeParse({ name: "Alice", age: -5 });
    expect(result.success).toBe(false);
  });

  it("rejects age of zero", () => {
    const result = ProfileSchema.safeParse({ name: "Alice", age: 0 });
    expect(result.success).toBe(false);
  });
});

// Run: npx vitest run
// FAILS - negative age is accepted

// Step 3: Fix the bug
import { z } from "zod";

export const ProfileSchema = z.object({
  name: z.string().min(1),
  // FIX: Add .positive() to reject zero and negative values
  age: z.number().int().positive().max(150),
});

// Run: npx vitest run
// PASSES - bug fixed, regression prevented
```

### Prohibited Practices for Bug Fixes

**NEVER:**
- Fix a bug without adding a regression test first
- Write implementation before writing tests (violates TDD)
- Skip the Red-Green-Refactor cycle
- Commit code with failing tests
- Remove tests to make code pass
- Use `z.any()` to paper over type mismatches

---

## 3. Project Structure & Organization (MANDATORY)

### A. Standard Project Layout

**Follow this layout for Zod-based projects:**

```
src/
├── schemas/                  # All Zod schemas
│   ├── common/               # Shared primitives and base schemas
│   │   ├── primitives.ts     # Email, UUID, dates, etc.
│   │   └── pagination.ts     # Pagination schemas
│   ├── user.schema.ts        # Domain schemas
│   ├── order.schema.ts
│   ├── ai/                   # OpenAI-specific schemas
│   │   ├── responses.ts      # zodResponseFormat schemas
│   │   └── tools.ts          # zodFunction schemas
│   └── index.ts              # Barrel exports
├── services/                 # Business logic consuming schemas
│   └── user.service.ts
├── api/                      # API route handlers
│   └── user.routes.ts
├── __tests__/                # Tests
│   ├── schemas/              # Schema unit tests
│   │   └── user.schema.test.ts
│   └── integration/          # Integration tests
│       └── ai-parsing.test.ts
├── lib/                      # Utilities
│   └── validation.ts         # Shared validation helpers
└── types/                    # ONLY for types NOT derivable from Zod
    └── externals.d.ts
```

### B. Schema Organization Principles

**Follow these principles:**

1. **One schema per domain entity**:
   ```typescript
   // CORRECT - user.schema.ts
   export const UserSchema = z.object({ ... });
   export const CreateUserSchema = UserSchema.omit({ id: true, createdAt: true });
   export const UpdateUserSchema = UserSchema.partial().required({ id: true });

   // WRONG - schemas.ts (dumping ground)
   export const UserSchema = z.object({ ... });
   export const OrderSchema = z.object({ ... });
   export const ProductSchema = z.object({ ... });
   ```

2. **Extract shared primitives**:
   ```typescript
   // schemas/common/primitives.ts
   export const Email = z.string().email().toLowerCase().trim();
   export const UUID = z.string().uuid();
   export const NonEmptyString = z.string().min(1).trim();
   export const PositiveInt = z.number().int().positive();
   export const ISODate = z.coerce.date();
   export const Slug = z.string().regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/);
   ```

3. **Derive, don't duplicate**:
   ```typescript
   // Base schema
   const BaseUser = z.object({
     id: UUID,
     email: Email,
     name: NonEmptyString,
     role: z.enum(["admin", "user"]),
     createdAt: ISODate,
   });

   // Derived schemas
   export const CreateUserInput = BaseUser.omit({ id: true, createdAt: true });
   export const UserResponse = BaseUser.extend({ postCount: z.number() });
   export const UserListItem = BaseUser.pick({ id: true, name: true, email: true });
   ```

---

## 4. Schema Design Patterns (MANDATORY)

### A. Primitive Schemas and Coercion

**Use built-in validators and coercion at system boundaries:**

```typescript
import { z } from "zod";

// String validations
const Name = z.string().min(1).max(100).trim();
const Email = z.string().email().toLowerCase().trim();
const URL = z.string().url();
const Slug = z.string().regex(/^[a-z0-9-]+$/);

// Number validations
const Port = z.number().int().min(1).max(65535);
const Percentage = z.number().min(0).max(100);
const Price = z.number().positive().multipleOf(0.01);

// Coercion for external string inputs (query params, env vars, forms)
const QueryLimit = z.coerce.number().int().min(1).max(100).default(20);
const QueryPage = z.coerce.number().int().positive().default(1);
const IsActive = z.coerce.boolean();
const StartDate = z.coerce.date();

// Enums
const Status = z.enum(["active", "inactive", "pending"]);
type Status = z.infer<typeof Status>; // "active" | "inactive" | "pending"

// Native enums (for existing TypeScript enums)
enum Direction { Up = "UP", Down = "DOWN" }
const DirectionSchema = z.nativeEnum(Direction);
```

### B. Object Schemas and Composition

**Compose complex schemas from building blocks:**

```typescript
// Base entity with common fields
const BaseEntity = z.object({
  id: z.string().uuid(),
  createdAt: z.coerce.date(),
  updatedAt: z.coerce.date(),
});

// Domain schema extending base
const UserSchema = BaseEntity.extend({
  email: z.string().email(),
  name: z.string().min(1).max(100),
  role: z.enum(["admin", "user", "viewer"]),
  profile: z.object({
    bio: z.string().max(500).optional(),
    avatarUrl: z.string().url().optional(),
  }).optional(),
});

// CRUD variants derived from base
const CreateUserInput = UserSchema.omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

const UpdateUserInput = UserSchema.pick({
  name: true,
  role: true,
  profile: true,
}).partial(); // All fields optional for PATCH

const UserListItem = UserSchema.pick({
  id: true,
  email: true,
  name: true,
  role: true,
});

// Strict mode - reject unknown keys (default in Zod v4)
const StrictInput = z.object({ name: z.string() }).strict();

// Passthrough mode - preserve unknown keys
const FlexibleInput = z.object({ name: z.string() }).passthrough();

// Type inference
type User = z.infer<typeof UserSchema>;
type CreateUserInput = z.infer<typeof CreateUserInput>;
```

### C. Discriminated Unions

**Use discriminated unions for polymorphic data:**

```typescript
const TextBlock = z.object({
  type: z.literal("text"),
  content: z.string(),
});

const ImageBlock = z.object({
  type: z.literal("image"),
  url: z.string().url(),
  alt: z.string(),
  width: z.number().positive().optional(),
  height: z.number().positive().optional(),
});

const CodeBlock = z.object({
  type: z.literal("code"),
  language: z.string(),
  source: z.string(),
});

// Discriminated union - efficient O(1) dispatch on "type" field
const ContentBlock = z.discriminatedUnion("type", [
  TextBlock,
  ImageBlock,
  CodeBlock,
]);

type ContentBlock = z.infer<typeof ContentBlock>;
// { type: "text"; content: string } | { type: "image"; url: string; ... } | ...

// Parsing automatically narrows the type
const block = ContentBlock.parse(data);
if (block.type === "image") {
  console.log(block.url); // TypeScript knows this is an ImageBlock
}
```

### D. Recursive Schemas

**Use `z.lazy()` for recursive/self-referencing structures:**

```typescript
interface Category {
  name: string;
  children: Category[];
}

const CategorySchema: z.ZodType<Category> = z.object({
  name: z.string(),
  children: z.lazy(() => CategorySchema.array()),
});

// JSON-like recursive structure
type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };

const JSONValueSchema: z.ZodType<JSONValue> = z.lazy(() =>
  z.union([
    z.string(),
    z.number(),
    z.boolean(),
    z.null(),
    z.array(JSONValueSchema),
    z.record(JSONValueSchema),
  ])
);
```

### E. Transform and Pipe

**Use `.transform()` for post-parse data shaping and `.pipe()` for multi-step validation:**

```typescript
// Transform: parse then reshape
const UserFromAPI = z.object({
  first_name: z.string(),
  last_name: z.string(),
  email_address: z.string().email(),
}).transform((data) => ({
  firstName: data.first_name,
  lastName: data.last_name,
  email: data.email_address,
  fullName: `${data.first_name} ${data.last_name}`,
}));

type UserFromAPI = z.infer<typeof UserFromAPI>;
// { firstName: string; lastName: string; email: string; fullName: string }

// Pipe: chain validation steps
const StringToNumber = z.string()
  .transform((s) => parseInt(s, 10))
  .pipe(z.number().int().positive());

// Coerce string → validate as number → ensure positive int
StringToNumber.parse("42");   // 42
StringToNumber.parse("-5");   // throws: not positive
StringToNumber.parse("abc");  // throws: NaN is not a number

// Branded types for nominal typing
const UserId = z.string().uuid().brand<"UserId">();
const OrderId = z.string().uuid().brand<"OrderId">();

type UserId = z.infer<typeof UserId>;   // string & { __brand: "UserId" }
type OrderId = z.infer<typeof OrderId>; // string & { __brand: "OrderId" }

function getUser(id: UserId) { /* ... */ }
// getUser(orderId) — compile error! Types are incompatible
```

---

## 5. OpenAI Structured Outputs Integration (MANDATORY)

### A. Response Format with zodResponseFormat

**Use `zodResponseFormat()` to enforce typed responses from OpenAI:**

```typescript
import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod";

// Define the response schema
// IMPORTANT: Only use JSON Schema-compatible types
// NO .transform(), .refine(), .preprocess(), .pipe() — only pure shapes
const SentimentAnalysis = z.object({
  sentiment: z.enum(["positive", "negative", "neutral"]),
  confidence: z.number(),
  reasoning: z.string(),
  keywords: z.array(z.string()),
});

const client = new OpenAI();

async function analyzeSentiment(text: string) {
  const completion = await client.chat.completions.parse({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "Analyze the sentiment of the given text.",
      },
      { role: "user", content: text },
    ],
    response_format: zodResponseFormat(SentimentAnalysis, "sentiment_analysis"),
  });

  const result = completion.choices[0]?.message.parsed;
  if (!result) throw new Error("No parsed response");

  // result is fully typed as z.infer<typeof SentimentAnalysis>
  return result;
}
```

### B. Responses API with zodTextFormat

**Use `zodTextFormat()` with the newer Responses API:**

```typescript
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";

const ExtractedData = z.object({
  people: z.array(z.object({
    name: z.string(),
    role: z.string(),
    company: z.string().optional(),
  })),
  topics: z.array(z.string()),
  action_items: z.array(z.object({
    description: z.string(),
    assignee: z.string().optional(),
    deadline: z.string().optional(),
  })),
});

const client = new OpenAI();

async function extractMeetingData(transcript: string) {
  const response = await client.responses.parse({
    model: "gpt-4o",
    input: `Extract structured data from this meeting transcript:\n\n${transcript}`,
    text: {
      format: zodTextFormat(ExtractedData, "meeting_data"),
    },
  });

  // response.output_parsed is typed as z.infer<typeof ExtractedData>
  return response.output_parsed;
}
```

### C. Tool Calling with zodFunction

**Use `zodFunction()` for type-safe function/tool calling:**

```typescript
import OpenAI from "openai";
import { zodFunction } from "openai/helpers/zod";
import { z } from "zod";

// Define tool parameter schemas
const SearchParams = z.object({
  query: z.string(),
  category: z.enum(["docs", "code", "issues"]),
  limit: z.number().int().min(1).max(50).optional(),
});

const CreateTicketParams = z.object({
  title: z.string(),
  description: z.string(),
  priority: z.enum(["low", "medium", "high", "critical"]),
  assignee: z.string().optional(),
});

const client = new OpenAI();

async function agentLoop(userMessage: string) {
  const completion = await client.chat.completions.parse({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: "You help users search documentation and create tickets.",
      },
      { role: "user", content: userMessage },
    ],
    tools: [
      zodFunction({ name: "search", parameters: SearchParams }),
      zodFunction({ name: "create_ticket", parameters: CreateTicketParams }),
    ],
  });

  const toolCalls = completion.choices[0]?.message.tool_calls;
  if (!toolCalls) return completion.choices[0]?.message.content;

  for (const toolCall of toolCalls) {
    switch (toolCall.function.name) {
      case "search": {
        // parsed_arguments is typed as z.infer<typeof SearchParams>
        const args = toolCall.function.parsed_arguments as z.infer<typeof SearchParams>;
        console.log(`Searching ${args.category} for: ${args.query}`);
        break;
      }
      case "create_ticket": {
        const args = toolCall.function.parsed_arguments as z.infer<typeof CreateTicketParams>;
        console.log(`Creating ${args.priority} ticket: ${args.title}`);
        break;
      }
    }
  }
}
```

### D. OpenAI Schema Constraints

**Schemas sent to OpenAI MUST follow these rules:**

```typescript
// ALLOWED in OpenAI schemas (pure JSON Schema types)
z.string()              // string
z.number()              // number
z.boolean()             // boolean
z.null()                // null
z.literal("value")      // const
z.enum(["a", "b"])      // enum
z.object({})            // object
z.array(z.string())     // array
z.union([...])          // anyOf
z.discriminatedUnion()  // anyOf with discriminator
z.string().optional()   // with nullable
z.string().nullable()   // with nullable
z.string().describe()   // description metadata (Zod v3)
z.string().meta()       // metadata (Zod v4)

// NOT ALLOWED in OpenAI schemas (runtime-only features)
z.string().transform()  // transforms don't serialize to JSON Schema
z.string().refine()     // refinements are runtime-only
z.string().regex()      // regex not supported in structured outputs
z.preprocess()          // preprocessing is runtime-only
z.pipe()                // pipes don't serialize
z.string().brand()      // brands are TypeScript-only
z.lazy()                // recursive schemas need careful handling

// Pattern: separate AI schema from runtime schema
const AIResponseSchema = z.object({
  name: z.string(),
  age: z.number(),
  email: z.string(),
});

// Runtime schema adds validations after AI parsing
const ValidatedResponse = AIResponseSchema.extend({
  email: z.string().email(),
  age: z.number().int().positive().max(150),
});

// Usage: parse with AI schema, then validate with runtime schema
const aiResult = completion.choices[0]?.message.parsed;  // AIResponseSchema
const validated = ValidatedResponse.parse(aiResult);      // Full validation
```

### E. Streaming Structured Outputs

**Handle partial results during streaming:**

```typescript
import OpenAI from "openai";
import { zodResponseFormat } from "openai/helpers/zod";
import { z } from "zod";

const AnalysisSchema = z.object({
  summary: z.string(),
  key_points: z.array(z.string()),
  sentiment: z.enum(["positive", "negative", "neutral"]),
});

const client = new OpenAI();

async function streamAnalysis(text: string) {
  const stream = client.beta.chat.completions.stream({
    model: "gpt-4o",
    messages: [
      { role: "system", content: "Analyze the following text." },
      { role: "user", content: text },
    ],
    response_format: zodResponseFormat(AnalysisSchema, "analysis"),
  });

  // Listen for partial parsed snapshots
  stream.on("content.delta", ({ snapshot }) => {
    // snapshot is a partial string as it builds up
    console.log("Partial:", snapshot);
  });

  const completion = await stream.finalChatCompletion();
  const parsed = completion.choices[0]?.message.parsed;
  return parsed; // Fully typed AnalysisSchema
}
```

---

## 6. Configuration & Environment Validation (MANDATORY)

### A. Environment Variable Validation

**ALWAYS validate env vars at application startup with Zod:**

```typescript
import { z } from "zod";

const EnvSchema = z.object({
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
  DATABASE_URL: z.string().url(),
  OPENAI_API_KEY: z.string().min(1, "OPENAI_API_KEY is required"),
  REDIS_URL: z.string().url().optional(),
  LOG_LEVEL: z.enum(["debug", "info", "warn", "error"]).default("info"),
  CORS_ORIGINS: z.string().transform((s) => s.split(",").map((o) => o.trim())),
  MAX_REQUEST_SIZE: z.coerce.number().default(10_485_760), // 10MB
});

// Parse once at startup — fail fast if invalid
export const env = EnvSchema.parse(process.env);

// Type is automatically inferred
// env.PORT is number, env.CORS_ORIGINS is string[], etc.
```

### B. Feature Flags

```typescript
const FeatureFlagsSchema = z.object({
  enableNewUI: z.coerce.boolean().default(false),
  maxUploadSizeMB: z.coerce.number().default(50),
  allowedModels: z.string()
    .default("gpt-4o,gpt-4o-mini")
    .transform((s) => s.split(",")),
  maintenanceMode: z.coerce.boolean().default(false),
});

export const flags = FeatureFlagsSchema.parse(process.env);
```

---

## 7. API Validation Patterns (MANDATORY)

### A. Request/Response Validation

**Validate all API boundaries:**

```typescript
import { z } from "zod";

// Pagination (reusable)
const PaginationSchema = z.object({
  page: z.coerce.number().int().positive().default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sortBy: z.string().optional(),
  sortOrder: z.enum(["asc", "desc"]).default("desc"),
});

// API response wrapper
function createListResponse<T extends z.ZodType>(itemSchema: T) {
  return z.object({
    data: z.array(itemSchema),
    pagination: z.object({
      page: z.number(),
      limit: z.number(),
      total: z.number(),
      totalPages: z.number(),
    }),
  });
}

const UserListResponse = createListResponse(UserSchema);
type UserListResponse = z.infer<typeof UserListResponse>;

// Error response
const APIErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.array(z.object({
      field: z.string(),
      message: z.string(),
    })).optional(),
  }),
});
```

### B. Express/Hono Middleware Pattern

```typescript
import { z, ZodType } from "zod";
import type { Request, Response, NextFunction } from "express";

function validate<T extends ZodType>(schema: T, source: "body" | "query" | "params") {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req[source]);
    if (!result.success) {
      return res.status(400).json({
        error: {
          code: "VALIDATION_ERROR",
          message: "Invalid request",
          details: result.error.issues.map((issue) => ({
            field: issue.path.join("."),
            message: issue.message,
          })),
        },
      });
    }
    req[source] = result.data;
    next();
  };
}

// Usage
app.post(
  "/users",
  validate(CreateUserInput, "body"),
  async (req, res) => {
    const data = req.body as z.infer<typeof CreateUserInput>; // Already validated
    // ...
  }
);
```

### C. tRPC Integration

```typescript
import { z } from "zod";
import { router, publicProcedure } from "./trpc";

export const userRouter = router({
  getById: publicProcedure
    .input(z.object({ id: z.string().uuid() }))
    .query(async ({ input }) => {
      // input is typed as { id: string }
      return db.user.findUnique({ where: { id: input.id } });
    }),

  create: publicProcedure
    .input(CreateUserInput)
    .mutation(async ({ input }) => {
      // input is typed as z.infer<typeof CreateUserInput>
      return db.user.create({ data: input });
    }),

  list: publicProcedure
    .input(PaginationSchema)
    .query(async ({ input }) => {
      const { page, limit, sortBy, sortOrder } = input;
      // ...
    }),
});
```

---

## 8. Testing (MANDATORY)

### A. Schema Unit Tests

**Test both valid inputs (happy path) and invalid inputs (error path):**

```typescript
import { describe, it, expect } from "vitest";
import { UserSchema, CreateUserInput } from "../schemas/user.schema";

describe("UserSchema", () => {
  const validUser = {
    id: "550e8400-e29b-41d4-a716-446655440000",
    email: "alice@example.com",
    name: "Alice",
    role: "admin" as const,
    createdAt: "2024-01-15T10:30:00Z",
    updatedAt: "2024-01-15T10:30:00Z",
  };

  describe("valid inputs", () => {
    it("parses a complete valid user", () => {
      const result = UserSchema.safeParse(validUser);
      expect(result.success).toBe(true);
    });

    it("coerces date strings to Date objects", () => {
      const result = UserSchema.parse(validUser);
      expect(result.createdAt).toBeInstanceOf(Date);
    });

    it("accepts all valid roles", () => {
      for (const role of ["admin", "user", "viewer"]) {
        const result = UserSchema.safeParse({ ...validUser, role });
        expect(result.success).toBe(true);
      }
    });
  });

  describe("invalid inputs", () => {
    it("rejects invalid email", () => {
      const result = UserSchema.safeParse({ ...validUser, email: "not-email" });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.error.issues[0].path).toEqual(["email"]);
      }
    });

    it("rejects unknown role", () => {
      const result = UserSchema.safeParse({ ...validUser, role: "superadmin" });
      expect(result.success).toBe(false);
    });

    it("rejects empty name", () => {
      const result = UserSchema.safeParse({ ...validUser, name: "" });
      expect(result.success).toBe(false);
    });

    it("rejects missing required fields", () => {
      const result = UserSchema.safeParse({});
      expect(result.success).toBe(false);
    });
  });
});
```

### B. OpenAI Integration Tests

```typescript
import { describe, it, expect } from "vitest";
import { z } from "zod";
import { zodResponseFormat, zodFunction } from "openai/helpers/zod";

describe("OpenAI schema compatibility", () => {
  const ResponseSchema = z.object({
    answer: z.string(),
    confidence: z.number(),
    sources: z.array(z.string()),
  });

  it("zodResponseFormat produces valid format object", () => {
    const format = zodResponseFormat(ResponseSchema, "response");
    expect(format.type).toBe("json_schema");
    expect(format.json_schema.name).toBe("response");
    expect(format.json_schema.schema).toBeDefined();
  });

  it("zodFunction produces valid tool definition", () => {
    const SearchParams = z.object({
      query: z.string(),
      limit: z.number().optional(),
    });

    const tool = zodFunction({ name: "search", parameters: SearchParams });
    expect(tool.type).toBe("function");
    expect(tool.function.name).toBe("search");
    expect(tool.function.parameters).toBeDefined();
  });

  it("rejects schemas with transforms for AI use", () => {
    // Document this pattern: transforms break OpenAI serialization
    const SchemaWithTransform = z.object({
      name: z.string().transform((s) => s.toUpperCase()),
    });

    // This will serialize but the transform won't run server-side
    // Ensure team understands this limitation
    const format = zodResponseFormat(SchemaWithTransform, "test");
    expect(format.json_schema.schema).toBeDefined();
    // Transform is stripped in JSON Schema — AI won't apply it
  });
});
```

### C. Test Coverage Requirements

- Minimum coverage: 90% for all schema files
- Every schema must have tests for valid AND invalid inputs
- Every discriminated union variant must be tested
- Transform outputs must be tested with expected values
- OpenAI schemas must have serialization tests
- All public APIs must have tests

---

## 9. Error Handling (MANDATORY)

### A. SafeParse Pattern

**Use `safeParse()` for graceful error handling:**

```typescript
import { z, ZodError } from "zod";

// Pattern 1: safeParse with early return
function processInput(raw: unknown) {
  const result = InputSchema.safeParse(raw);
  if (!result.success) {
    return { ok: false, errors: formatZodError(result.error) } as const;
  }
  // result.data is fully typed
  return { ok: true, data: result.data } as const;
}

// Pattern 2: parse with try/catch for fail-fast
function processInputStrict(raw: unknown) {
  try {
    const data = InputSchema.parse(raw); // throws ZodError on failure
    return data;
  } catch (error) {
    if (error instanceof ZodError) {
      throw new ValidationError("Invalid input", error.issues);
    }
    throw error;
  }
}
```

### B. Error Formatting

```typescript
import { ZodError } from "zod";

// Flat error map for forms
function formatZodError(error: ZodError): Record<string, string> {
  const formatted: Record<string, string> = {};
  for (const issue of error.issues) {
    const path = issue.path.join(".");
    if (!formatted[path]) {
      formatted[path] = issue.message;
    }
  }
  return formatted;
}

// Usage
const result = UserSchema.safeParse(formData);
if (!result.success) {
  const errors = formatZodError(result.error);
  // { "email": "Invalid email", "name": "String must contain at least 1 character(s)" }
}

// Built-in flatten (Zod v3.x)
const flat = result.error.flatten();
// { formErrors: [...], fieldErrors: { email: [...], name: [...] } }

// Built-in format (nested)
const formatted = result.error.format();
// { email: { _errors: ["Invalid email"] }, name: { _errors: ["Required"] } }
```

### C. Custom Error Messages

```typescript
const UserSchema = z.object({
  email: z.string({
    required_error: "Email is required",
    invalid_type_error: "Email must be a string",
  }).email("Please provide a valid email address"),

  age: z.number()
    .int("Age must be a whole number")
    .min(13, "Must be at least 13 years old")
    .max(150, "Invalid age"),

  password: z.string()
    .min(8, "Password must be at least 8 characters")
    .regex(/[A-Z]/, "Password must contain an uppercase letter")
    .regex(/[0-9]/, "Password must contain a number"),
});
```

### D. Common Errors

| Error Type | Description | Handling |
|------------|-------------|----------|
| `invalid_type` | Wrong type provided | Show field type requirement |
| `invalid_string` | Failed string validation (email, url, uuid) | Show format requirement |
| `too_small` | Below minimum length/value | Show minimum requirement |
| `too_big` | Exceeds maximum length/value | Show maximum allowed |
| `invalid_enum_value` | Value not in enum | Show allowed values |
| `invalid_union` | No union member matched | Show expected variants |
| `custom` | Custom refinement failed | Show custom message |

---

## 10. Zod v4 Features (RECOMMENDED)

### A. JSON Schema Conversion

**Zod v4 provides first-party JSON Schema conversion:**

```typescript
import * as z from "zod";

const UserSchema = z.object({
  name: z.string(),
  email: z.string().email(),
  age: z.number().int().positive(),
});

// Convert to JSON Schema
const jsonSchema = z.toJSONSchema(UserSchema);
// {
//   type: "object",
//   properties: {
//     name: { type: "string" },
//     email: { type: "string" },
//     age: { type: "integer" }
//   },
//   required: ["name", "email", "age"],
//   additionalProperties: false
// }

// With options
z.toJSONSchema(UserSchema, {
  target: "draft-07",   // JSON Schema draft version
  reused: "inline",     // How to handle reused schemas
});
```

### B. Global Registry and Metadata

```typescript
import * as z from "zod";

// Add metadata via .meta()
const EmailSchema = z.string().email().meta({
  title: "Email Address",
  description: "A valid email address",
  examples: ["user@example.com"],
});

// Global registry for cross-cutting metadata
z.globalRegistry.add(EmailSchema, {
  id: "email_address",
  title: "Email Address",
  description: "User's email address",
});

// Metadata appears in JSON Schema output
z.toJSONSchema(EmailSchema);
// { type: "string", title: "Email Address", description: "..." }
```

### C. z.interface() for Open Objects

```typescript
import * as z from "zod";

// z.interface() allows additional properties (like TypeScript interfaces)
const Config = z.interface({
  host: z.string(),
  port: z.number(),
});
// Allows { host: "localhost", port: 3000, debug: true }

// vs z.object() which strips unknown properties by default
const StrictConfig = z.object({
  host: z.string(),
  port: z.number(),
});
// Strips unknown: { host: "localhost", port: 3000 }
```

---

## 11. Dependencies & Package Management (MANDATORY)

### A. Dependency Management

**Install Zod and related packages:**

```bash
# Core Zod (pick one version)
npm install zod               # Latest (v3.x or v4.x)

# OpenAI SDK (includes Zod helpers)
npm install openai

# Testing
npm install -D vitest @vitest/coverage-v8

# Optional ecosystem
npm install zod-validation-error  # Better error messages
npm install @anatine/zod-mock     # Generate mock data from schemas
```

### B. TypeScript Configuration

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "moduleResolution": "bundler",
    "module": "ESNext",
    "target": "ES2022",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist"]
}
```

---

## 12. Security & Dependency Management (MANDATORY)

### A. Automated Dependency Management

```bash
# npm: install dependencies from package.json
npm install

# npm: update all dependencies to latest compatible versions
npm update

# npm: check for outdated packages
npm outdated

# npm: install exact versions (no ^ or ~ ranges)
npm install --save-exact zod@3.24.4

# Regenerate lockfile from scratch
rm -rf node_modules package-lock.json && npm install
```

**`package-lock.json`** is the lockfile. Always commit it to version control for reproducible builds.

### B. Vulnerability Scanning & Security

```bash
# npm audit: scan dependencies for known vulnerabilities
npm audit

# npm audit: automatically fix vulnerabilities where possible
npm audit fix

# npm audit: full report in JSON format (for CI/CD)
npm audit --json

# Snyk: deeper vulnerability scanning
snyk test

# Check for unused or duplicate dependencies
npx depcheck
```

**Security best practices:**
- **Zod-specific**: Always use `.safeParse()` at API boundaries — never trust external input
- **Zod-specific**: Use `.strict()` on schemas processing untrusted data to reject unexpected fields
- **Zod-specific**: Avoid `.passthrough()` on schemas handling user input — it allows unvalidated fields through
- Run `npm audit` in CI/CD pipelines and fail builds on high/critical vulnerabilities
- Use `package-lock.json` and `npm ci` (not `npm install`) in CI for deterministic installs
- Prefer `--save-exact` for production dependencies to prevent supply chain attacks via semver ranges
- Review changelogs before upgrading Zod major versions — schema behavior may change
- Never use `z.any()` without subsequent refinement in security-sensitive schemas
- Validate environment variables at startup using Zod schemas (fail fast on misconfiguration)

### C. Dependency File

```json
{
  "name": "my-zod-project",
  "version": "1.0.0",
  "private": true,
  "engines": {
    "node": ">=20.0.0"
  },
  "dependencies": {
    "zod": "3.24.4"
  },
  "devDependencies": {
    "typescript": "5.7.3",
    "vitest": "3.1.1",
    "@types/node": "22.13.14",
    "prettier": "3.5.3",
    "eslint": "9.23.0"
  },
  "scripts": {
    "build": "tsc",
    "test": "vitest run",
    "lint": "eslint src/ --ext .ts",
    "format": "prettier --write \"src/**/*.ts\"",
    "audit": "npm audit --audit-level=high",
    "typecheck": "tsc --noEmit --strict"
  }
}
```

---

## 13. Deployment Checklist

### Agent-Generated Code Verification (MANDATORY)

**If code was generated/modified by an agent, verify BEFORE delivery:**

#### Build & Compilation
- [ ] Code compiles: `npx tsc --noEmit --strict` returns exit code 0
- [ ] No TypeScript errors or warnings
- [ ] All `z.infer<>` types resolve correctly
- [ ] Code formatted: `npx prettier --check "src/**/*.ts"` produces no changes

#### Testing
- [ ] All tests pass: `npx vitest run` returns exit code 0
- [ ] Schema tests cover valid AND invalid inputs
- [ ] Coverage above 90% for schema files
- [ ] OpenAI schema compatibility verified

#### Code Quality
- [ ] Linter passes: `npx eslint src/ --ext .ts`
- [ ] No `z.any()` without refinement
- [ ] No duplicate type definitions alongside Zod schemas
- [ ] No `as` assertions bypassing Zod parsing

#### Zod-Specific
- [ ] Types inferred from schemas, not manually defined
- [ ] Shared primitives extracted to `schemas/common/`
- [ ] CRUD variants use `.pick()`, `.omit()`, `.partial()`, `.extend()`
- [ ] `.safeParse()` used at API boundaries with proper error handling
- [ ] OpenAI schemas contain only JSON Schema-compatible types
- [ ] Environment variables validated with Zod at startup

#### Agent Workflow Completed
- [ ] Agent verified code compiles with strict TypeScript
- [ ] Agent ran all tests and verified they pass
- [ ] Agent ran formatters and linters
- [ ] Agent verified OpenAI schema serialization where applicable
- [ ] Agent documented any fixes made during verification

---

## 14. Why This Configuration Works

**Type Safety Without Duplication**:
- Zod schemas serve as the single source of truth for both runtime validation and TypeScript types. No drift between interfaces and validators.

**AI-Ready Validation**:
- OpenAI's structured outputs natively support Zod via `zodResponseFormat()`, `zodTextFormat()`, and `zodFunction()`. Your validation layer doubles as your AI contract.

**Composable by Design**:
- `.extend()`, `.pick()`, `.omit()`, `.partial()`, and `.merge()` let you derive all CRUD variants from a single base schema without duplication.

**Fail-Fast at Boundaries**:
- Coercion + parsing at system boundaries (env vars, API inputs, AI responses) catches problems early and guarantees that all downstream code works with validated types.

---

## 15. Quick Reference

### Common Schema Patterns

```typescript
import { z } from "zod";

// Primitives
z.string().email()                    // Email validation
z.string().uuid()                     // UUID validation
z.string().url()                      // URL validation
z.string().min(1).max(255)            // Bounded string
z.number().int().positive()           // Positive integer
z.coerce.number()                     // String → number coercion
z.coerce.date()                       // String → Date coercion
z.coerce.boolean()                    // String → boolean coercion

// Objects
z.object({}).extend({})               // Add fields
z.object({}).pick({ a: true })        // Keep only listed fields
z.object({}).omit({ a: true })        // Remove listed fields
z.object({}).partial()                // All fields optional
z.object({}).required()               // All fields required
z.object({}).strict()                 // Reject unknown keys
z.object({}).passthrough()            // Preserve unknown keys

// Collections
z.array(z.string())                   // Array of strings
z.array(z.string()).nonempty()        // At least one element
z.tuple([z.string(), z.number()])     // Fixed-length typed array
z.record(z.string(), z.number())      // Record<string, number>
z.map(z.string(), z.number())         // Map<string, number>
z.set(z.string())                     // Set<string>

// Unions & Enums
z.enum(["a", "b", "c"])              // String enum
z.nativeEnum(MyEnum)                  // TypeScript enum
z.union([z.string(), z.number()])     // string | number
z.discriminatedUnion("type", [...])   // Tagged union

// Modifiers
z.string().optional()                 // string | undefined
z.string().nullable()                 // string | null
z.string().nullish()                  // string | null | undefined
z.string().default("hello")           // Default value
z.string().catch("fallback")          // Fallback on parse failure
z.string().brand<"MyBrand">()        // Nominal typing
z.string().describe("A description")  // Metadata (v3)
z.string().meta({ title: "..." })     // Metadata (v4)

// Transforms
z.string().transform(s => s.trim())   // Post-parse transform
z.string().pipe(z.number())           // Multi-step validation
z.preprocess(val => String(val), z.string()) // Pre-parse transform

// Type inference
type User = z.infer<typeof UserSchema>;
type Input = z.input<typeof UserSchema>;   // Before transforms
type Output = z.output<typeof UserSchema>; // After transforms
```

### OpenAI Integration Commands

```typescript
import { zodResponseFormat, zodTextFormat, zodFunction } from "openai/helpers/zod";

// Structured response (Chat Completions API)
zodResponseFormat(MySchema, "schema_name")

// Structured response (Responses API)
zodTextFormat(MySchema, "schema_name")

// Tool/function calling
zodFunction({ name: "tool_name", parameters: ParamsSchema })

// Parse completion
client.chat.completions.parse({ model, messages, response_format })

// Parse response
client.responses.parse({ model, input, text: { format } })
```

### Common Commands

```bash
# Type check
npx tsc --noEmit --strict

# Test
npx vitest run

# Test with coverage
npx vitest run --coverage

# Lint
npx eslint src/ --ext .ts

# Format
npx prettier --write "src/**/*.ts"

# Watch mode
npx vitest watch
```

---

**End of Zod Guidelines**

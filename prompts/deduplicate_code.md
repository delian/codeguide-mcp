# deduplicate_code
description: Check the code for duplication and propose deduplication steps.
role: assistant
type: text
You are an expert Senior Engineer. Follow these steps:
1. Analyze the source code, architecture and infrastructure stack.
2. Identify any duplicated code across the codebase, including TypeScript, Python, Dockerfiles, docker-compose, etc.
3. Propose specific steps to abstract and deduplicate the identified code while ensuring that the changes do not break the application build and behavior.
4. Make sure all fixes are not breaking the build, compilation and operation of the code and application, all the code passes lints, could be build, unit tests are passing.
5. For each proposed deduplication step, provide a brief explanation of how it improves the codebase and any potential trade-offs.
6. All the changes should produce nice, compact, readable and maintainable code.
7. Create or update a Git issue, commit for the deduplication task with clear instructions and references to the relevant coding guides if applicable.

description: Provide duplication check input.
role: user
type: text
Here is the check for code duplication:

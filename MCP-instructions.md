This is a MCP server for managing coding guides and practices.
It extends or replaces AGENTS.md file with new coding practices and guidelines.
It provides a simple API for listing and retrieving the content of coding guidelines.

## Usage Workflow

**IMPORTANT: Always start by listing available guides before selecting any specific guide.**

1. **First Step - List Guides**: Always begin by requesting `guides://list` to see all available coding guides and their descriptions.

2. **List Output Format**: The list output follows this format:
   ```
   guide_name.md - brief description extracted from the guide
   ```
   Each line contains:
   - The guide filename (e.g., `python_style.md`)
   - A dash separator
   - A brief description (up to 150 characters) explaining what the guide provides

3. **Select Guides Based on Software Stack**: After reviewing the list, select one or more guides that are relevant to the current software stack, programming languages, frameworks, or technologies being used in the project.

4. **Retrieve Guide Content**: Once you've identified the relevant guides, request their full content using `guides://{guide_name}`.

## API Endpoints

- `guides://list`: List all available coding guides with brief descriptions.
  - **Output format**: Each line is `guide_name.md - brief description`
  - **Always use this first** to discover available guides
  
- `guides://{guide_name}`: Get the full content of a specific coding guide.
  - **Example**: `guides://python_style.md`
  - **Use after** identifying relevant guides from the list

---
name: knowledge-glean-search
description: >
  Search and retrieve information from enterprise datasources via Glean
  (Confluence, SharePoint, Google Drive, OneDrive, etc.). Covers glean_search,
  glean_get_file, glean_chat, and glean_get_metadata MCP tools, including
  search strategies, pagination, and structured output. Use when the
  conversation involves internal or proprietary information, or when exploring
  code that references internal systems or undocumented design decisions.
  Triggers: internal docs, Confluence, SharePoint, Google Drive, enterprise
  search, design docs, runbooks, oncall, internal wiki, knowledge base, Glean,
  internal documentation, proprietary information, team processes.
tags:
  - glean
  - enterprise-search
  - internal-docs
  - knowledge-retrieval
  - confluence
  - sharepoint
  - google-drive
---

# Glean Enterprise Knowledge Retrieval

Search, retrieve, and synthesize information from enterprise datasources via the Glean platform.

## When to Use

- The conversation involves internal or proprietary information not available in public sources
- You need to find design docs, runbooks, architecture decisions, or internal wiki pages
- Code references internal systems, APIs, or configurations that are undocumented in the repo
- You need context on team processes, project status, or internal benchmarks

Do NOT use when:
- The information is publicly available (open-source docs, public APIs)
- You already have the answer from reading the codebase or git history
- The user is asking about code that is fully self-documented

## Available Tools

| Tool | Purpose |
|------|---------|
| `glean_search` | Keyword-based search across all connected datasources |
| `glean_get_file` | Fetch full document content (requires `system` parameter) |
| `glean_chat` | Conversational Q&A with cited sources (supports multi-turn via `chat_id`) |
| `glean_get_metadata` | Quick metadata lookup (title, author, date, size) |
| `health_check` | Verify Glean service connectivity |

## Workflow

### Step 1: Classify the Request

| Request Type | Primary Tool | Strategy |
|-------------|-------------|----------|
| Find documents about a topic | `glean_search` | Search with relevant keywords, refine if needed |
| Get contents of a known document | `glean_get_file` | Fetch directly if system and URL/ID are known |
| Answer a factual question | `glean_chat` | Use conversational AI for cited answers |
| Research a broad topic | `glean_search` → `glean_get_file` | Search first, then fetch top results |
| Check document details | `glean_get_metadata` | Quick metadata lookup before full retrieval |

### Step 2: Execute Search and Retrieval

#### Searching

Use `glean_search` for keyword-based discovery:

- Start with specific, targeted queries
- If results are sparse, broaden the query or try alternative terms
- Keep `page_size` small (1-5) to avoid token limits
- Use `cursor` from the response for pagination when more results are needed

#### Fetching Content

Use `glean_get_file` to retrieve full document content:

- **Requires `system` parameter** — use the `datasource` value from search or chat results
- Content is paginated via MCP protocol — use `pageToken` from the response to fetch subsequent pages
- Content is truncated at `max_length` per page (default 10000) — adjust if needed
- For large documents, fetch incrementally and summarize as you go

#### Conversational Queries

Use `glean_chat` for questions that benefit from AI synthesis:

- Best for "how", "why", and "what" questions where you need a synthesized answer
- Returns answers with citations to source documents
- Supports multi-turn conversation via `chat_id` — pass the `chat_id` from a previous response to continue a conversation thread

#### Metadata Checks

Use `glean_get_metadata` before fetching large files:

- Confirms the document exists and is accessible
- Shows title, author, last modified date, size
- Requires the `system` parameter — use `datasource` from search results

## Search Strategy

### Iterative Refinement

1. **First pass**: Use the user's original terms
2. **If insufficient**: Try synonyms, acronyms, or related terms
3. **If still insufficient**: Broaden the scope or use `glean_chat` for a conversational approach
4. **Cross-reference**: When a search result mentions related documents, follow up with additional searches

### Query Tips

- Use specific technical terms when searching for engineering content
- Include project names, team names, or product names to narrow results
- For recent content, mention timeframes if the search supports it
- Combine multiple related searches to build a comprehensive picture

## Structured Output

Return research results in this format:

```
## Glean Research Summary

**Query**: <original user question or topic>
**Sources searched**: <number of searches performed>
**Documents found**: <number of relevant documents>

### Key Findings

1. <finding with source citation>
2. <finding with source citation>
...

### Sources

| # | Title | System | URL |
|---|-------|--------|-----|
| 1 | Document title | Confluence/SharePoint/etc. | link |
| ... | | | |

### Additional Context

<any synthesized insights, connections between sources, or caveats>
```

## Error Handling

| Symptom | Cause | Resolution |
|---------|-------|------------|
| Any tool call fails unexpectedly | Glean service may be unavailable | Run `health_check` to verify connectivity; if it fails, inform user and suggest checking network/VPN |
| Empty search results | Query too specific or no matching content | Broaden query, try alternative terms |
| `glean_get_file` fails | Wrong `system` parameter or access denied | Verify `system` matches the `datasource` from search results |
| Token limit exceeded | Response too large | Reduce `page_size`, reduce `max_snippet_size`, or paginate |
| Missing `system` param | Required for get_file and get_metadata | Extract `datasource` from prior search or chat results |

## Scope Boundaries

**In scope**: Searching, retrieving, and synthesizing information from Glean-connected datasources. Summarizing findings. Following up on references across documents.

**Out of scope** (delegate to the user or other agents):
- Modifying or uploading documents to datasources
- Managing Glean configuration or datasource connections
- Actions that require write access to any connected system

Always cite your sources. Never fabricate document content or URLs — only report
what is returned by the Glean tools.

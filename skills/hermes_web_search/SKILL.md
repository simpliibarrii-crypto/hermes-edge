---
title: Hermes Web Search
description: Search the web and return concise, summarized results for a query. Use when the user asks about current events, facts, or anything that may require up-to-date information beyond the on-device model's knowledge.
type: javascript
version: 1.0.0
author: hermes-edge
---

# Hermes Web Search

A JavaScript Agent Skill for the Google AI Edge Gallery. Hermes invokes this
skill via a tool call; the Gallery runs the implementation inside its hidden
WebView and returns the result as a `<tool_response>`.

## Tool schema

```json
{
  "name": "web_search",
  "description": "Search the web for up-to-date information and return the top results.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "The search query." },
      "num_results": { "type": "integer", "description": "Max results to return (1-5).", "default": 3 }
    },
    "required": ["query"]
  }
}
```

## When to use

- The user asks about recent events, prices, schedules, or live data.
- A factual question where the on-device model is likely stale or unsure.

## Implementation

```javascript
// Runs in the Gallery WebView sandbox. Receives `args` (parsed tool arguments)
// and must call `resolve(result)` / `reject(error)`.
async function run(args) {
  const query = (args.query || "").trim();
  if (!query) {
    return { error: "Empty query." };
  }
  const n = Math.min(Math.max(parseInt(args.num_results, 10) || 3, 1), 5);

  // DuckDuckGo Instant Answer API: no API key, CORS-friendly, privacy-respecting.
  const url =
    "https://api.duckduckgo.com/?q=" +
    encodeURIComponent(query) +
    "&format=json&no_html=1&skip_disambig=1";

  try {
    const resp = await fetch(url, { method: "GET" });
    if (!resp.ok) {
      return { error: "Search request failed with status " + resp.status };
    }
    const data = await resp.json();

    const results = [];
    if (data.AbstractText) {
      results.push({ title: data.Heading || query, snippet: data.AbstractText, url: data.AbstractURL });
    }
    for (const topic of data.RelatedTopics || []) {
      if (results.length >= n) break;
      if (topic.Text && topic.FirstURL) {
        results.push({ title: topic.Text.split(" - ")[0], snippet: topic.Text, url: topic.FirstURL });
      }
    }

    if (results.length === 0) {
      return { results: [], note: "No instant answer found for query." };
    }
    return { query, results: results.slice(0, n) };
  } catch (e) {
    return { error: "Web search error: " + String(e) };
  }
}
```

## Returns

A JSON object the model consumes as the tool response:

```json
{ "query": "...", "results": [ { "title": "...", "snippet": "...", "url": "..." } ] }
```

## Sharing

Share this skill with another Gallery user by URL (raw GitHub link to this
`SKILL.md`) or by importing the local file from `/sdcard/Download/`.

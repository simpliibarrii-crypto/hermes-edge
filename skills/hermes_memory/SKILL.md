---
title: Hermes Memory
description: Offline note-taking / key-value memory. Use whenever the user asks to remember something for later, recall what they told you earlier, or forget a stored note ("remember that my project is due Friday", "what did I tell you about X", "forget my address").
type: javascript
version: 1.0.0
author: hermes-edge
---

# Hermes Memory

An offline JavaScript Agent Skill for the Google AI Edge Gallery. No network
access — it persists small key/value notes in the WebView's `localStorage`, so
memories survive across chat sessions and app restarts on the same device. This
gives the small on-device model a durable scratchpad it would otherwise lack.

## Tool schema

```json
{
  "name": "memory",
  "description": "Persist and retrieve small key/value notes across sessions, fully offline. Supports remember, recall, forget, and list_memories.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["remember", "recall", "forget", "list_memories"],
        "description": "Which memory operation to perform."
      },
      "key": { "type": "string", "description": "Memory key (required for remember/recall/forget)." },
      "value": { "type": "string", "description": "Value to store (required for remember)." }
    },
    "required": ["action"]
  }
}
```

## When to use

- "Remember that ..." / "Note that ..." / "Don't forget ..." → `remember`.
- "What did I tell you about ...?" / "What's my ...?" → `recall`.
- "Forget ..." / "Delete the note about ..." → `forget`.
- "What do you remember?" / "List my notes" → `list_memories`.

## Implementation

```javascript
// Offline key/value memory backed by localStorage. All keys are namespaced
// under "hermes_memory:" so the skill never collides with other Gallery state.
async function run(args) {
  const NS = "hermes_memory:";
  const action = (args.action || "").trim();

  // localStorage may be unavailable in some embeddings; degrade to an in-memory
  // store for the lifetime of the page so the skill never hard-crashes.
  const store = (() => {
    try {
      const t = "__hermes_probe__";
      window.localStorage.setItem(t, "1");
      window.localStorage.removeItem(t);
      return window.localStorage;
    } catch (e) {
      if (!globalThis.__hermesMemFallback) globalThis.__hermesMemFallback = new Map();
      const m = globalThis.__hermesMemFallback;
      return {
        getItem: (k) => (m.has(k) ? m.get(k) : null),
        setItem: (k, v) => m.set(k, v),
        removeItem: (k) => m.delete(k),
        key: (i) => Array.from(m.keys())[i] ?? null,
        get length() { return m.size; },
      };
    }
  })();

  const allKeys = () => {
    const keys = [];
    for (let i = 0; i < store.length; i++) {
      const k = store.key(i);
      if (k && k.startsWith(NS)) keys.push(k.slice(NS.length));
    }
    return keys.sort();
  };

  switch (action) {
    case "remember": {
      if (!args.key) return { error: "remember requires a 'key'." };
      if (args.value === undefined || args.value === null)
        return { error: "remember requires a 'value'." };
      store.setItem(NS + args.key, String(args.value));
      return { ok: true, action, key: args.key, value: String(args.value) };
    }
    case "recall": {
      if (!args.key) return { error: "recall requires a 'key'." };
      const v = store.getItem(NS + args.key);
      if (v === null) return { found: false, key: args.key };
      return { found: true, key: args.key, value: v };
    }
    case "forget": {
      if (!args.key) return { error: "forget requires a 'key'." };
      const existed = store.getItem(NS + args.key) !== null;
      store.removeItem(NS + args.key);
      return { ok: true, action, key: args.key, existed };
    }
    case "list_memories": {
      const keys = allKeys();
      const memories = keys.map((k) => ({ key: k, value: store.getItem(NS + k) }));
      return { count: memories.length, memories };
    }
    default:
      return { error: "Unknown action. Use remember | recall | forget | list_memories." };
  }
}
```

## Returns

```json
{ "ok": true, "action": "remember", "key": "project_due", "value": "Friday" }
{ "found": true, "key": "project_due", "value": "Friday" }
{ "count": 1, "memories": [{ "key": "project_due", "value": "Friday" }] }
```

## Sharing

Share by raw GitHub URL to this `SKILL.md`, or place the file in
`/sdcard/Download/` and import it in the Gallery's Agent Skills screen.

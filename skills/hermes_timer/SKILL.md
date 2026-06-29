---
title: Hermes Timer
description: Offline timers, stopwatches, and alarms. Use when the user asks to set a countdown ("set a 5-minute timer"), start/stop a stopwatch ("start timing my workout", "how long has it been since I started X"), or list active timers. Fully offline.
type: javascript
version: 1.0.0
author: hermes-edge
---

# Hermes Timer

An offline JavaScript Agent Skill for the Google AI Edge Gallery. No network
access — it uses `setTimeout` and `Date.now()` inside the Gallery WebView to run
countdown timers and stopwatches. Active timers/stopwatches are tracked in a
`Map` that lives on `globalThis` so they persist for the lifetime of the page.

## Tool schema

```json
{
  "name": "timer",
  "description": "Set countdown timers and run stopwatches, fully offline. Supports set_timer, start_stopwatch, stop_stopwatch, and list_timers.",
  "parameters": {
    "type": "object",
    "properties": {
      "action": {
        "type": "string",
        "enum": ["set_timer", "start_stopwatch", "stop_stopwatch", "list_timers"],
        "description": "Which timer operation to perform."
      },
      "seconds": { "type": "number", "description": "Countdown duration in seconds (required for set_timer)." },
      "label": { "type": "string", "description": "Name for the timer/stopwatch (required for set_timer, start_stopwatch, stop_stopwatch)." }
    },
    "required": ["action"]
  }
}
```

## When to use

- "Set a 5-minute timer" / "Remind me in 30 seconds" → `set_timer` (seconds=300/30).
- "Start timing my workout" / "Start a stopwatch called X" → `start_stopwatch`.
- "Stop the workout timer" / "How long has it been since I started X" → `stop_stopwatch`.
- "What timers are running?" → `list_timers`.

## Implementation

```javascript
// Offline timers + stopwatches. State is kept on globalThis so repeated skill
// invocations within the same WebView session share the same Map.
async function run(args) {
  const G = globalThis;
  if (!G.__hermesTimers) {
    G.__hermesTimers = { timers: new Map(), stopwatches: new Map() };
  }
  const { timers, stopwatches } = G.__hermesTimers;
  const action = (args.action || "").trim();

  const fmt = (ms) => {
    const s = Math.max(0, Math.round(ms / 1000));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return m > 0 ? `${m}m ${r}s` : `${r}s`;
  };

  switch (action) {
    case "set_timer": {
      const seconds = Number(args.seconds);
      if (!Number.isFinite(seconds) || seconds <= 0)
        return { error: "set_timer requires a positive 'seconds'." };
      const label = args.label || `timer_${timers.size + 1}`;
      const startedAt = Date.now();
      const firesAt = startedAt + seconds * 1000;
      // Fire-and-forget: clears itself from the Map when it elapses.
      const handle = setTimeout(() => {
        const t = timers.get(label);
        if (t) t.fired = true;
      }, seconds * 1000);
      timers.set(label, { label, seconds, startedAt, firesAt, handle, fired: false });
      return { ok: true, action, label, seconds, fires_in: fmt(seconds * 1000) };
    }
    case "start_stopwatch": {
      const label = args.label || `stopwatch_${stopwatches.size + 1}`;
      if (stopwatches.has(label))
        return { error: `Stopwatch '${label}' already running.` };
      stopwatches.set(label, { label, startedAt: Date.now() });
      return { ok: true, action, label, started: true };
    }
    case "stop_stopwatch": {
      const label = args.label;
      if (!label || !stopwatches.has(label))
        return { error: `No running stopwatch named '${label}'.` };
      const sw = stopwatches.get(label);
      const elapsedMs = Date.now() - sw.startedAt;
      stopwatches.delete(label);
      return { ok: true, action, label, elapsed_ms: elapsedMs, elapsed: fmt(elapsedMs) };
    }
    case "list_timers": {
      const now = Date.now();
      const activeTimers = Array.from(timers.values()).map((t) => ({
        label: t.label,
        remaining: t.fired ? "elapsed" : fmt(t.firesAt - now),
        fired: t.fired,
      }));
      const activeStopwatches = Array.from(stopwatches.values()).map((s) => ({
        label: s.label,
        elapsed: fmt(now - s.startedAt),
      }));
      return {
        timers: activeTimers,
        stopwatches: activeStopwatches,
        count: activeTimers.length + activeStopwatches.length,
      };
    }
    default:
      return { error: "Unknown action. Use set_timer | start_stopwatch | stop_stopwatch | list_timers." };
  }
}
```

## Returns

```json
{ "ok": true, "action": "set_timer", "label": "tea", "seconds": 300, "fires_in": "5m 0s" }
{ "ok": true, "action": "stop_stopwatch", "label": "workout", "elapsed_ms": 92000, "elapsed": "1m 32s" }
{ "timers": [{ "label": "tea", "remaining": "4m 12s", "fired": false }], "stopwatches": [], "count": 1 }
```

## Sharing

Share by raw GitHub URL to this `SKILL.md`, or place the file in
`/sdcard/Download/` and import it in the Gallery's Agent Skills screen.

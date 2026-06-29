---
title: Hermes Calculator
description: Evaluate arithmetic and math expressions fully offline. Use whenever the user asks for a calculation, conversion, or any precise numeric result the model should not guess.
type: javascript
version: 1.0.0
author: hermes-edge
---

# Hermes Calculator

An offline JavaScript Agent Skill for the Google AI Edge Gallery. No network
access — it evaluates a math expression locally inside the Gallery WebView and
returns the numeric result. This keeps Hermes accurate on arithmetic, which
small language models otherwise get wrong.

## Tool schema

```json
{
  "name": "calculator",
  "description": "Evaluate an arithmetic/math expression and return the numeric result. Offline.",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": { "type": "string", "description": "A math expression, e.g. \"(12 * 9) + sqrt(81)\"." }
    },
    "required": ["expression"]
  }
}
```

## When to use

- Any arithmetic: addition, multiplication, percentages, powers, roots.
- Unit-free numeric reasoning where exactness matters.

## Implementation

```javascript
// Offline evaluator. Tokenizes and parses a safe arithmetic grammar — does NOT
// use eval(), so arbitrary code cannot run. Supports + - * / % ^, parentheses,
// and the functions sqrt, abs, sin, cos, tan, log, ln, exp, and constants pi, e.
async function run(args) {
  const expr = (args.expression || "").trim();
  if (!expr) return { error: "Empty expression." };

  const FUNCS = {
    sqrt: Math.sqrt, abs: Math.abs, sin: Math.sin, cos: Math.cos,
    tan: Math.tan, log: Math.log10, ln: Math.log, exp: Math.exp,
    floor: Math.floor, ceil: Math.ceil, round: Math.round,
  };
  const CONSTS = { pi: Math.PI, e: Math.E };

  // Tokenizer
  const tokens = [];
  const re = /\s*([A-Za-z_]+|\d+\.?\d*|\.\d+|[()+\-*/%^,])/g;
  let m, last = 0;
  while ((m = re.exec(expr)) !== null) {
    if (m.index !== last) return { error: "Unexpected character in expression." };
    tokens.push(m[1]);
    last = re.lastIndex;
  }
  if (last !== expr.length) return { error: "Trailing characters in expression." };

  let pos = 0;
  const peek = () => tokens[pos];
  const next = () => tokens[pos++];

  function parseExpr() { return parseAddSub(); }
  function parseAddSub() {
    let v = parseMulDiv();
    while (peek() === "+" || peek() === "-") {
      const op = next();
      const r = parseMulDiv();
      v = op === "+" ? v + r : v - r;
    }
    return v;
  }
  function parseMulDiv() {
    let v = parsePow();
    while (peek() === "*" || peek() === "/" || peek() === "%") {
      const op = next();
      const r = parsePow();
      if (op === "*") v *= r;
      else if (op === "/") v /= r;
      else v %= r;
    }
    return v;
  }
  function parsePow() {
    let v = parseUnary();
    if (peek() === "^") { next(); v = Math.pow(v, parsePow()); }
    return v;
  }
  function parseUnary() {
    if (peek() === "-") { next(); return -parseUnary(); }
    if (peek() === "+") { next(); return parseUnary(); }
    return parseAtom();
  }
  function parseAtom() {
    const t = next();
    if (t === "(") {
      const v = parseExpr();
      if (next() !== ")") throw new Error("Missing closing parenthesis.");
      return v;
    }
    if (/^[A-Za-z_]+$/.test(t)) {
      if (t in CONSTS) return CONSTS[t];
      if (t in FUNCS) {
        if (next() !== "(") throw new Error("Expected '(' after " + t);
        const arg = parseExpr();
        if (next() !== ")") throw new Error("Missing ')' for " + t);
        return FUNCS[t](arg);
      }
      throw new Error("Unknown identifier: " + t);
    }
    const num = parseFloat(t);
    if (Number.isNaN(num)) throw new Error("Unexpected token: " + t);
    return num;
  }

  try {
    const result = parseExpr();
    if (pos !== tokens.length) return { error: "Could not fully parse expression." };
    if (!Number.isFinite(result)) return { error: "Result is not finite." };
    return { expression: expr, result };
  } catch (e) {
    return { error: String(e.message || e) };
  }
}
```

## Returns

```json
{ "expression": "(12 * 9) + sqrt(81)", "result": 117 }
```

## Sharing

Share by raw GitHub URL to this `SKILL.md`, or place the file in
`/sdcard/Download/` and import it in the Gallery's Agent Skills screen.

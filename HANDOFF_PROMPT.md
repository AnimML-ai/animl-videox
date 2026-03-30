# Handoff Prompt

Paste this into the generator session before closing it.
Claude will write `HANDOFF.md` to the repo root.

---

Before we close this session, write a file called `HANDOFF.md` at the repo root.
Be factual and terse. Do not summarize what went well. Focus on what a skeptical reviewer needs to know.

Structure it exactly as follows:

```markdown
## What was built
[One paragraph. What exists now that didn't before.]

## Files changed
[List each file and one sentence on why it was touched.]

## Known uncertainties
[The most important section. List every place you made an assumption,
took a shortcut, weren't sure about correctness, or left something
implicit. Be honest — this is not a self-review, it's a handoff.]

## Explicit non-goals
[What was deliberately deferred or out of scope for this session.]

## How to test
[The minimal command or sequence to verify the core behavior works.]
```

Do not add any other sections. Do not be encouraging about the output quality.

# Exo Cluster Rules

You are running on a local AI cluster with limited decode speed. Context size directly impacts performance — every additional 1K tokens slows decode by ~0.33ms per token. At 100K+ context, tool calls take minutes.

## Critical Performance Rules

1. **Use subagents for research.** When you need to read multiple files or explore the codebase, spawn Agent subagents. Each subagent starts with fresh, small context and runs fast.

2. **Spawn multiple focused subagents, not one big one.** Instead of one agent that reads 20 files, spawn 3-4 agents that each research one module. They run sequentially but each stays fast.

3. **Never read entire files when you can search.** Use Grep to find specific patterns. Use Glob to find files. Only Read the specific lines you need (use offset/limit).

4. **Keep your main context lean.** Don't paste large file contents into your response. Summarize findings concisely.

5. **Subagents have a 40K token context limit.** They will hit context_window_exceeded if they grow too large. Design subagent prompts to be focused — read specific files, answer specific questions, return concise results.

6. **Batch tool calls.** When you need to do multiple independent operations, emit them all in one response rather than one per turn. Each round-trip costs a full decode cycle.

7. **Start writing early.** Don't read the entire codebase before generating output. Read what you need, then start producing results incrementally.

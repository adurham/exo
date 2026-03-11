# MANDATORY RULES — LOCAL AI CLUSTER

You are running on a LOCAL AI CLUSTER. Decode speed degrades with context size. These rules prevent you from wasting minutes per response.

## RULE 1: USE SUBAGENTS — NOT DIRECT TOOL CALLS

ALWAYS use the Agent tool to spawn subagents for reading files, searching code, and exploring the codebase. Do NOT call Read, Grep, or Glob directly in your main conversation. The ONLY exception is a single targeted Read when you already know the exact file and line numbers.

Why: Each subagent starts with a fresh, small context and runs 5-10x faster than your main conversation. A subagent doing 5 tool calls takes less time than you doing 1 tool call at high context.

How:
- Spawn 2-4 subagents IN PARALLEL, each answering ONE specific question
- Each subagent prompt must be 1-3 sentences — just the question
- Subagents have a 20K token limit. Tell them: "Do at most 5 tool calls. Return a 1-3 sentence summary."
- While subagents research, you can write code or plan — do NOT sit idle

## RULE 2: NEVER BLOAT YOUR CONTEXT

- NEVER read entire files in your main conversation
- NEVER paste file contents into your response — summarize in 1-3 sentences
- NEVER repeat back code the user can already see
- If a subagent returns detailed results, extract only what matters

## RULE 3: BATCH EVERYTHING

Every response you generate costs a full decode cycle. When you need multiple operations, emit them ALL in one response. Never do sequential tool calls when you could do parallel ones.

## RULE 4: ACT FAST

Do NOT over-research. Read the minimum needed, then start writing code. You can always ask a subagent for more info later.

# MANDATORY PERFORMANCE RULES — READ CAREFULLY

YOU ARE RUNNING ON A LOCAL AI CLUSTER WITH EXTREMELY LIMITED DECODE SPEED. These rules are NON-NEGOTIABLE. Violating them wastes minutes of the user's time per tool call.

Context size = decode speed. Every 1K tokens added to your context costs ~0.33ms per output token. At 100K context, a single tool call response takes 5+ MINUTES. You MUST keep context small.

## RULE 1: SMART TOOL CALL ROUTING (MANDATORY)

Your decode speed degrades linearly with context size. Subagents start with fresh, small context and run much faster. Use this decision framework:

**Use subagents when ANY of these are true:**
- You need 3+ tool calls for a research task (file reads, searches, exploration)
- Your conversation is long (many prior messages/tool results)
- You have parallel work to do while waiting (e.g., write code while subagent researches)
- The task is exploratory — you don't know exactly which files/lines you need

**Do it directly (no subagent) when ALL of these are true:**
- You need 1-2 targeted lookups (you know exactly which file and what to find)
- Your conversation is still short (early in the session)
- You have nothing else to do while waiting — sitting idle is worse than just reading

**Subagent best practices:**
- Spawn 2-4 FOCUSED subagents in parallel, each answering ONE specific question
- Subagents have a ~20K token context limit — keep their prompts focused
- Tell subagents to return CONCISE summaries, not raw file contents
- In the subagent prompt, instruct them: "Use Grep to search, Glob to find files, and Read with offset/limit for specific lines. Do NOT use Bash for ls or cat. Return only a brief summary."
- ALWAYS prefer parallel subagents over sequential tool calls at high context

## RULE 2: NEVER READ ENTIRE FILES (MANDATORY)

- Use Grep with specific patterns to find what you need
- Use Glob to find files by name
- When you must Read, ALWAYS use offset/limit to read only the relevant lines
- NEVER use Bash to run ls, cat, head, tail, or find — use Glob, Grep, and Read instead
- NEVER paste large file contents into your response — summarize

## RULE 3: BATCH ALL TOOL CALLS (MANDATORY)

Every round-trip costs a full decode cycle (minutes at high context). When you need multiple independent operations, you MUST emit them ALL in a single response. Never do one tool call per turn when you could do five.

## RULE 4: START PRODUCING OUTPUT EARLY (MANDATORY)

Do NOT read the entire codebase before writing code. Read the minimum needed, then start. You can always read more later if needed. Every extra read at high context wastes minutes.

## RULE 5: KEEP MAIN CONTEXT LEAN (MANDATORY)

- Do not repeat back large blocks of code or file contents
- Summarize findings in 1-3 sentences
- If a subagent returns detailed results, extract only what you need

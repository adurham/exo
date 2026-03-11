# MANDATORY PERFORMANCE RULES — READ CAREFULLY

YOU ARE RUNNING ON A LOCAL AI CLUSTER WITH EXTREMELY LIMITED DECODE SPEED. These rules are NON-NEGOTIABLE. Violating them wastes minutes of the user's time per tool call.

Context size = decode speed. Every 1K tokens added to your context costs ~0.33ms per output token. At 100K context, a single tool call response takes 5+ MINUTES. You MUST keep context small.

## RULE 1: USE SUBAGENTS FOR ALL RESEARCH (MANDATORY)

When you need to read files, search code, or explore the codebase, you MUST spawn Agent subagents. NEVER do research in your main conversation. Each subagent starts with fresh, small context and runs 3-5x faster than your bloated main context.

- Spawn 2-4 FOCUSED subagents, each answering ONE specific question
- Subagents have a ~50K token context limit — keep their prompts focused
- Tell subagents to return CONCISE summaries, not raw file contents

## RULE 2: NEVER READ ENTIRE FILES (MANDATORY)

- Use Grep with specific patterns to find what you need
- Use Glob to find files by name
- When you must Read, ALWAYS use offset/limit to read only the relevant lines
- NEVER paste large file contents into your response — summarize

## RULE 3: BATCH ALL TOOL CALLS (MANDATORY)

Every round-trip costs a full decode cycle (minutes at high context). When you need multiple independent operations, you MUST emit them ALL in a single response. Never do one tool call per turn when you could do five.

## RULE 4: START PRODUCING OUTPUT EARLY (MANDATORY)

Do NOT read the entire codebase before writing code. Read the minimum needed, then start. You can always read more later if needed. Every extra read at high context wastes minutes.

## RULE 5: KEEP MAIN CONTEXT LEAN (MANDATORY)

- Do not repeat back large blocks of code or file contents
- Summarize findings in 1-3 sentences
- If a subagent returns detailed results, extract only what you need

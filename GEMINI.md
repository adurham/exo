# Gemini Interaction Rules

*   **Plan and Approval:** Before making sweeping codebase changes, running background processes (like benchmarks or stress tests), or executing a complex sequence of commands, you MUST explicitly present a general "plan of attack" to the user.
*   **Wait for Confirmation:** After presenting the plan, pause and wait for the user to explicitly approve it before proceeding with execution. You do not need to ask for approval for every single command (like simple file reads or greps during the research phase), but any significant action or state change requires prior consent.
*   **Explain Actions:** Continue to briefly explain *why* you are taking a specific action before doing it, so the user can easily follow along with your logic.

# Dev Loop

Autonomous dev loop. Runs GATE → SELECT → DISPATCH → SAVE → ROUTE in a cycle.

Launch the dev workflow:
- Use the Workflow tool: `Workflow({ name: "dev", args: { userMessage: "$ARGUMENTS" } })`
- If args contain "brainstorm", a brainstorm will be triggered at GATE
- The workflow runs in the background. Check progress with `/workflows`
- When complete, review the returned result and report to the user

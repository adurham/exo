Unified Agentic & Project Guidelines

SYSTEM INSTRUCTION: You must follow these rules strictly. If you encounter code violations in files you are not actively editing, do not fix them ad-hoc; instead, raise a GitHub Issue or inform the user directly.

1. Development & Operations

Package Management:

ALWAYS use uv for dependency management and script execution (uv run, uv pip install, uv sync).

NEVER use pyenv directly; rely on uv for Python version management.

Deployment (STRICT NO-DEPLOY):

NEVER deploy code, run deploy_workers.sh, or suggest deployment commands.

NEVER attempt to auto-create instances or trigger model loads.

All deployment and remote sync actions are manual user steps.

Dependencies:

Introduce new dependencies only after explicit approval.

Restrict requests to libraries that are ubiquitous in production environments.

2. Cluster & Networking Topology

Architecture: 3-node cluster.

macstudio-m4 (128GB RAM)

macbook-m4 (36GB RAM)

work-macbook-m4 (48GB RAM)

Network Triangle (Direct Links):

Network A (Studio ↔ MB M4 Max): 192.168.201.x

Network B (Studio ↔ MB M4 Pro): 192.168.202.x

Network C (MB M4 Max ↔ MB M4 Pro): 192.168.205.x

Routing Rules:

Priority: ALWAYS use Thunderbolt interfaces/IPs when available.

Worker-to-Worker: MUST use direct links (Network C); do not route through the Master.

Resource Management:

Full Utilization: Do not artificially cap memory or resources unless configured.

No Hardcoding: Never hardcode memory limits or hostname checks in source logic.

3. Code Discipline & Philosophy

Purity & State:

Referential Transparency: Every function must return the same output for the same input (no hidden state).

Effect Handlers: Core logic must be pure. Push side-effects (I/O, database writes) to injectable "effect handlers."

Classes: Use classes only to wrap fixed state and prevent unsafe mutation.

Immutability: Mark classes, methods, and variables as @final or immutable wherever applicable.

Naming Conventions:

Descriptive: No 3-letter acronyms or non-standard contractions.

Self-Documenting: Function signatures must explain the function's purpose without needing the body code.

Comments:

Delete redundant comments after editing.

Retain comments only for complex logic context.

4. Typing & Pydantic

Strict Typing:

Maintain exhaustive strict typing. Never bypass the type checker.

Use Literal[...] for enum-like sets.

Use typing.NewType for primitives (e.g., distinguishing two str types) to enforce semantic separation with zero runtime cost.

Pydantic Standards:

Centralize a ConfigDict with frozen=True and strict=True. Use this everywhere.

Polymorphism: Use discriminated unions (Annotated[Base, Field(discriminator='variant')]) for BaseModel hierarchies.

Serialization: Add a type: str field to serializable objects to explicitly state identity.

IDs & UUIDs:

Subclass Pydantic's UUID4 for custom IDs.

Generation: Use uuid.uuid4() for fresh IDs.

Idempotency: Generate keys by hashing persisted state + function-specific salt to prevent collisions after crashes.

5. Error Handling

Minimalism: Eliminate superfluous try/catch and if branches by using strict typing and static analysis.

Intentionality: Catch exceptions only if you can handle or transform them meaningfully.

Documentation: Docstrings must specify where and why an exception is expected to be handled.

6. Commit Messages

Format: <type>: <imperative subject> (Max 50 chars, Capitalized, No trailing period).

Types:

feature: New feature.

bugfix: Bug fix.

refactor: Code change (no behavior change).

documentation: Docs only.

test: Adding/correcting tests.

chore: Maintenance/tooling.

Body: Separate from subject with a blank line.
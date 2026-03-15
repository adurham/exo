# Exo Cluster Guidelines for Claude Code

## Core Architecture

exo connects Apple Silicon devices into a distributed AI inference cluster via libp2p, RDMA over Thunderbolt 5, and MLX.

- **Pattern:** Event-sourced, master-worker architecture. A pure functional state machine in `src/exo/shared/apply.py` processes `Event`s to produce `State` transitions. The `State` type (`src/exo/shared/types/state.py`) is the single source of truth for instances, runners, downloads, tasks, and topology.
- **Parallelism:** Supports Pipeline Parallel (PP) and Tensor Parallel (TP) via `mlx.distributed`. PP splits layers across nodes (minimal network syncs); TP splits attention heads across nodes (requires per-layer syncs). Hybrid TP+PP is used for multi-node clusters.
- **Networking:** Rust-based libp2p networking layer (`rust/networking/`) with PyO3 bindings (`rust/exo_pyo3_bindings/`). Nodes discover peers via mDNS/libp2p and communicate over pub/sub topics (`src/exo/routing/topics.py`).
- **RDMA:** Direct GPU-to-GPU memory transfers over Thunderbolt 5 for distributed MLX operations. Tuned via `MLX_JACCL_NUM_BUFFERS`, `MLX_JACCL_FRAME_SIZE` env vars.

## Project Structure

```
src/exo/
├── main.py                  # Entry point — Node setup, election, master/worker lifecycle
├── master/                  # Master node logic
│   ├── main.py              # Master orchestrator
│   ├── api.py               # FastAPI/Hypercorn HTTP API server
│   ├── placement.py         # Shard placement across cluster nodes
│   ├── event_log.py         # Event log persistence
│   └── adapters/            # API adapters (OpenAI, Claude, Ollama, Responses API)
├── worker/                  # Worker node logic
│   ├── main.py              # Worker orchestrator
│   ├── engines/mlx/         # MLX inference engine (LLM generation, KV cache)
│   ├── engines/image/       # Image generation engine (mflux)
│   └── runner/              # Runner supervisor and bootstrap
├── shared/                  # Shared types and logic
│   ├── apply.py             # Pure state machine — Event → State transitions
│   ├── election.py          # Leader election protocol
│   ├── topology.py          # Network topology graph (rustworkx)
│   ├── models/model_cards.py # Supported model definitions
│   └── types/               # Pydantic type definitions
│       ├── state.py         # Global State model
│       ├── events.py        # Event types (all cluster events)
│       ├── tasks.py         # Task lifecycle types
│       ├── common.py        # NodeId, SessionId, etc.
│       └── worker/          # Worker-specific types (instances, runners, shards)
├── routing/                 # libp2p event routing
│   ├── router.py            # Rust networking bridge
│   ├── event_router.py      # Event pub/sub dispatcher
│   └── topics.py            # libp2p topic definitions
├── download/                # Model download coordination
│   ├── coordinator.py       # Distributed download orchestration
│   └── huggingface_utils.py # HuggingFace Hub integration
└── utils/                   # Utilities
    ├── channels.py          # Typed async channels
    ├── task_group.py         # anyio task group wrapper
    ├── info_gatherer/        # System info (macmon GPU metrics, Thunderbolt, network)
    └── pydantic_ext.py       # CamelCaseModel base class

rust/
├── networking/              # libp2p swarm, mDNS discovery
├── exo_pyo3_bindings/       # PyO3 Python ↔ Rust bridge
│   ├── src/lib.rs           # Binding entry point
│   └── exo_pyo3_bindings.pyi # Generated type stubs
└── util/                    # Rust utilities (WakerDeque)

dashboard/                   # SvelteKit web dashboard
app/                         # macOS native app (EXO)
scripts/                     # Benchmarking and analysis scripts
tests/                       # Integration/stress tests (separate from unit tests in src/)
```

## Development & Commands

Use the `justfile` for all standard workflows:
- `just check` — basedpyright strict type checking
- `just lint` — ruff linting with auto-fix
- `just test` — pytest on `src/`
- `just fmt` — Nix treefmt formatting
- `just rust-rebuild` — regenerate PyO3 stubs + rebuild Rust bindings
- `just sync` — `uv sync --all-packages`
- `just sync-clean` — force reinstall all packages (no cache)
- `just build-dashboard` — build SvelteKit dashboard
- `just clean` — remove __pycache__, target/, .venv, dashboard build artifacts

**Environment:** Nix flake (`flake.nix`) provides the dev shell with Rust toolchain (via fenix/crane), Python, and system dependencies. Use `direnv` (`.envrc`) for automatic shell activation.

**Package manager:** `uv` (required ≥0.8.6). The workspace includes `rust/exo_pyo3_bindings` and `bench` as uv workspace members.

### Testing

- `pytest` uses `asyncio_mode = "auto"` and sets `EXO_TESTS=1` automatically.
- Slow tests are excluded by default via `@pytest.mark.slow` marker.
- Run specific tests: `uv run pytest path/to/file.py::test_name -v`
- Tests ignore `mlx/`, `mlx-lm/` submodule dirs and `tests/start_distributed_test.py`.
- Unit tests live alongside source code in `src/exo/**/tests/` directories.
- Integration/stress tests live in top-level `tests/`.

### Rust Development

- Workspace: 3 crates — `networking` (libp2p), `exo_pyo3_bindings` (PyO3), `util`.
- Edition 2024, strict Clippy lints (pedantic, nursery, cargo + restriction subset).
- After modifying Rust code: run `just rust-rebuild` to regenerate stubs and reinstall bindings.
- Stub generation: `cargo run --bin stub_gen` generates `exo_pyo3_bindings.pyi`.

## CI & Baseline State (DO NOT attempt to fix these proactively)

CI runs on GitHub Actions (`.github/workflows/pipeline.yml`):
- Builds Nix outputs on `aarch64-darwin`, `x86_64-linux`, `aarch64-linux`.
- Runs pytest on macOS only (requires Metal GPU access).
- Uses Cachix for Nix binary cache.

**Pre-existing issues — do not increase these counts:**
- basedpyright has ~190 pre-existing errors.
- ruff has ~23 pre-existing errors.
- Rust binding test `test_sleep_on_multiple_items` fails currently (pytest ignores Rust inherently).

## Deployment Rule (MANDATORY)

**Before telling the user code is ready to test or deploy, it MUST be committed and pushed to `origin/main`.** The cluster nodes deploy from `origin/main` — uncommitted or unpushed code does not exist as far as the cluster is concerned. Never say "ready to test" or "ready to deploy" with unpushed changes in the working tree. No exceptions.

## Cluster Operation

The cluster is a 3-node M4 setup (2x Mac Studio 128GB + 1x MacBook Pro 36GB) managed by `start_cluster.sh`. Key environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `EXO_FAST_SYNCH` | GPU sync mode (1=fast, 0=safe) | `1` |
| `EXO_DISABLE_METAL_TIMEOUT` | Bypass GPU watchdog for long contexts | `1` |
| `EXO_PREFILL_STEP_SIZE` | Chunked prefill token count | `524288` |
| `EXO_ADAPTIVE_THROTTLE` | Adaptive prefill throttle (ms) | `100` |
| `EXO_LIBP2P_NAMESPACE` | Peer discovery namespace | `MAC_STUDIO_CLUSTER` |
| `EXO_KV_BITS` | KV cache quantization bits | `16` |
| `EXO_COMPILE_DECODE` | Enable mx.compile for decode | `0` |
| `MLX_JACCL_NUM_BUFFERS` | RDMA buffer count | `2` |
| `MLX_JACCL_FRAME_SIZE` | RDMA frame size (bytes) | `4096` |

## Coding Conventions

- **Purity:** Pure functions preferred; use classes only to safely encapsulate mutable state.
- **Typing (Strict):** Python ≥3.13. Use `NewType` for structural disambiguation, `Literal` over enums, exhaustive `match`. basedpyright in strict mode.
- **Data:** Use Pydantic models with `frozen=True` and `strict=True`. Extend `CamelCaseModel` for JSON-serialized types.
- **Error Handling:** No unnecessary try/catch blocks; know exactly where exceptions will be caught.
- **Async:** `anyio` with `uvloop` backend. Use `TaskGroup` wrapper from `utils/task_group.py`.
- **Logging:** `loguru` logger (not stdlib `logging`).
- **Serialization:** `msgspec` for high-performance encoding; `zstandard` for compression.
- **Commits:** Imperative mood, prefix with `feature:`, `bugfix:`, `refactor:`, `chore:`, `documentation:`, `test:`. (≤50 chars subject).
- **Submodules:** `mlx` and `mlx-lm` are local editable git submodules (forked from `adurham/mlx` and `adurham/mlx-lm`), NOT PyPI packages. Changes to distributed C++ code in `mlx/` are common.

## Key Dependencies

| Package | Purpose |
|---|---|
| `mlx` / `mlx-lm` | Apple ML framework + LLM tooling (editable submodules) |
| `pydantic` | Data validation, frozen models, strict types |
| `fastapi` + `hypercorn` | HTTP API server (OpenAI-compatible) |
| `aiohttp` | Async HTTP client |
| `libp2p` (via Rust) | P2P networking, peer discovery |
| `rustworkx` | Graph data structure for topology |
| `huggingface-hub` | Model downloading |
| `mflux` | Image generation (Flux models on Apple Silicon) |
| `loguru` | Structured logging |
| `msgspec` | Fast serialization |
| `tiktoken` | Tokenization (Kimi K2) |
| `openai-harmony` | OpenAI API compatibility layer |

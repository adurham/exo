# Exo Cluster Guidelines for Claude Code

## Core Architecture
exo connects Apple Silicon devices into a distributed AI inference cluster via libp2p, RDMA over Thunderbolt 5, and MLX. 
- **Pattern:** Event-sourced, master-worker architecture. Pure functional state machine (`shared/apply.py`).
- **Parallelism:** Supports Pipeline Parallel (PP) and Tensor Parallel (TP) via `mlx.distributed`.

## Development & Commands
Use the `justfile` for all standard workflows:
- `just check` (basedpyright strict type checking)
- `just lint` (ruff)
- `just test` (pytest `src/`)
- `just fmt` (Nix treefmt)
- `just rust-rebuild` (rebuilds PyO3 bindings after Rust changes)
- `just sync` (uv sync --all-packages)

**Testing Quirks:**
- `pytest` uses `asyncio_mode = "auto"` and `EXO_TESTS=1`.
- Exclude slow tests (default behavior via `@pytest.mark.slow`).
- Run specific tests like: `uv run pytest path/to/file.py::test_name -v`

## CI & Baseline State (DO NOT attempt to fix these proactively)
- basedpyright has ~190 pre-existing errors. Do not increase this count.
- ruff has ~23 pre-existing errors.
- Rust binding test `test_sleep_on_multiple_items` fails currently (pytest ignores Rust inherently).

## Coding Conventions
- **Purity:** Pure functions preferred; use classes only to safely encapsulate mutable state.
- **Typing (Strict):** Python ≥3.13. Use `NewType` for structural disambiguation, `Literal` over enums, exhaustive `match`.
- **Data:** Use Pydantic models with `frozen=True` and `strict=True`.
- **Error Handling:** No unnecessary try/catch blocks; know exactly where exceptions will be caught.
- **Commits:** Imperative mood, prefix with `feature:`, `bugfix:`, `refactor:`, `chore:`, `documentation:`, `test:`. (≤50 chars subject).
- **Submodules:** `mlx` and `mlx-lm` are local editable git submodules (forked), NOT PyPI packages.
fmt:
    nix fmt

lint:
    uv run ruff check --fix

test:
    uv run pytest src

test-deployed:
    #!/usr/bin/env bash
    # Run integration tests against deployed nodes
    # Usage: just test-deployed [API_URL]
    API_URL="${1:-http://100.93.253.67:52415}"
    echo "Running integration tests against deployed cluster at $API_URL"
    EXO_API_URL="$API_URL" uv run pytest src/exo/tests/test_deployed_integration.py -v -s

check:
    uv run basedpyright --project pyproject.toml

sync:
    uv sync --all-packages

sync-clean:
    uv sync --all-packages --force-reinstall --no-cache

rust-rebuild:
    cargo run --bin stub_gen
    just sync-clean

build-dashboard:
    #!/usr/bin/env bash
    cd dashboard
    npm install
    npm run build

package:
    uv run pyinstaller packaging/pyinstaller/exo.spec

clean:
    rm -rf **/__pycache__
    rm -rf target/
    rm -rf .venv
    rm -rf dashboard/node_modules
    rm -rf dashboard/.svelte-kit
    rm -rf dashboard/build

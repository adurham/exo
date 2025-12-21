#!/usr/bin/env bash
set -euo pipefail

NODES=("macstudio-m4" "macbook-m4" "work-macbook-m4")
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-$HOME/repos/exo}"
LOG_FILE="${LOG_FILE:-$HOME/exo.log}"
REPO_URL="${REPO_URL:-git@github.com:adurham/exo.git}"
BRANCH="${BRANCH:-m4-rdma-only}"
DASHBOARD_NODES=(${DASHBOARD_NODES:-${NODES[0]}})
SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=2
)

success_nodes=()
failed_nodes=()

deploy_node() {
  local node="$1"
  local build_dashboard=0
  local force_arg="--force-worker"
  if [[ "$node" == "${NODES[0]}" ]]; then
    force_arg="--force-master"
  fi
  for dnode in "${DASHBOARD_NODES[@]}"; do
    if [[ "$dnode" == "$node" ]]; then
      build_dashboard=1
      break
    fi
  done
  echo "--------------------------------------------------"
  echo "🔌 Connecting to ${node}..."

  if ssh "${SSH_OPTS[@]}" "$node" \
    "REMOTE_REPO_DIR='${REMOTE_REPO_DIR}' LOG_FILE='${LOG_FILE}' REPO_URL='${REPO_URL}' BRANCH='${BRANCH}' BUILD_DASHBOARD='${build_dashboard}' FORCE_ARG='${force_arg}' bash -s" <<'EOF'
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ ! -d "${REMOTE_REPO_DIR}/.git" ]]; then
  echo "   📂 Cloning repository..."
  mkdir -p "${REMOTE_REPO_DIR%/*}"
  git clone "${REPO_URL}" "${REMOTE_REPO_DIR}"
fi

cd "${REMOTE_REPO_DIR}"

echo "   🧹 Cleaning and pulling..."
git fetch origin "${BRANCH}"
git checkout -B "${BRANCH}" "origin/${BRANCH}"
git reset --hard "origin/${BRANCH}"

DASHBOARD_DIR="${DASHBOARD_DIR:-${REMOTE_REPO_DIR}/dashboard/build}"
if [[ ! -d "${DASHBOARD_DIR}" ]]; then
  if [[ "${BUILD_DASHBOARD}" -eq 1 ]]; then
    echo "   🧰 Building dashboard assets..."
    if ! command -v npm >/dev/null 2>&1; then
      echo "   ❌ npm not found; install Node or set DASHBOARD_DIR to a prebuilt dashboard."
      exit 1
    fi
    pushd "${REMOTE_REPO_DIR}/dashboard" >/dev/null
    npm ci --no-progress
    npm run build
    popd >/dev/null
  else
    echo "   ℹ️  Skipping dashboard build; creating placeholder assets..."
    DASHBOARD_DIR="${REMOTE_REPO_DIR}/.dashboard_placeholder"
    mkdir -p "${DASHBOARD_DIR}"
    if [[ ! -f "${DASHBOARD_DIR}/index.html" ]]; then
      cat > "${DASHBOARD_DIR}/index.html" <<'HTML'
<!doctype html><title>EXO Dashboard Disabled</title><h1>Dashboard not deployed on this node.</h1>
HTML
    fi
  fi
fi
export DASHBOARD_DIR

echo "   🛠️  Rebuilding Rust bindings..."
uv sync --reinstall-package exo-pyo3-bindings

echo "   🛑 Stopping old instance..."
pkill -f 'uv run exo' || true
sleep 1

echo "   🚀 Starting Exo..."
EXO_USE_RDMA=1 RUST_BACKTRACE=1 nohup uv run exo ${FORCE_ARG} > "${LOG_FILE}" 2>&1 < /dev/null &

for attempt in {1..6}; do
  sleep 2
  if lsof -iTCP:5678 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "   ✅ Port 5678 confirmed."
    exit 0
  fi
done

echo "   ⚠️ Port 5678 not listening; recent log:"
tail -n 40 "${LOG_FILE}" || true
exit 1
EOF
  then
    echo "   ✨ Finished deployment step for ${node}."
    success_nodes+=("$node")
  else
    echo "   ❌ Deployment failed for ${node}."
    failed_nodes+=("$node")
  fi
}

echo "🚀 Starting Cluster Deployment (Full Rebuild Mode)..."
for node in "${NODES[@]}"; do
  deploy_node "$node"
done

echo "--------------------------------------------------"
if [[ ${#failed_nodes[@]} -eq 0 ]]; then
  echo "🎉 Deployment cycle finished successfully."
else
  echo "⚠️ Deployment completed with failures: ${failed_nodes[*]}"
  exit 1
fi


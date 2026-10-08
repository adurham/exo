# `bench/phase20_guard.py` — API contract (shared by the guard, ladder and GPU-busy tools)

Importable as `phase20_guard` when `bench/` is on `sys.path`; also a CLI. Pure stdlib + `ssh`/`scp` subprocesses + read-only sqlite.
Nothing in this module may ever send a generation request, start/stop a process on a node, or write to state.db.

```python
NODES: dict[str, str] = {"studio1": "m4-1", "studio2": "m4-2"}        # ssh aliases -> short tags
API_BASE = "http://192.168.86.48:52415"                                # cluster API (studio1 is master/API node)
STATE_DB_URI = "file:/Users/adam.durham/.hermes/state.db?mode=ro"

class GuardFailure(RuntimeError): ...
class ChunkAborted(GuardFailure): ...          # raised by ChunkGuard.check() after an abort

@dataclass
class IdleReport:
    ok: bool
    reasons: list[str]                         # human-readable, empty when ok
    detail: dict                               # per node: last_non_own_post_age_s, active_tasks, ...; state_db: {...}

def idle_check(own_requests: Sequence[float] | None = None, *, min_idle_s: int = 600) -> IdleReport   # R1 (see PREREG D1)
def wait_for_idle(*, poll_s: float = 30, max_wait_s: float = 3600, own_requests=None) -> IdleReport    # blocks, GuardFailure on timeout

@dataclass
class CanaryReport:
    ok: bool
    state: str                                 # "healthy" (median>=10) | "marginal" (5..10) | "degraded" (<5)
    per_node: dict[str, list[float]]           # the 3 TFLOPS numbers the canary prints
    median: dict[str, float]

def canary(nodes: Sequence[str] = ("studio1", "studio2"), *, timeout_s: float = 90) -> CanaryReport      # R4
# Runs /tmp/gpu_canary2.py (scp'd from /Users/adam.durham/.hermes/cache/scratch/gpu_canary2.py if missing) with the node's
# ~/repos/exo/.venv/bin/python on both nodes CONCURRENTLY? NO — sequentially (one node's GPU at a time is fine, the cluster is idle).

class ChunkGuard:                              # R2 + R3, context manager
    def __init__(self, label: str, *, max_wall_s: float = 900, poll_s: float = 15,
                 registry_path: str | None = None, interrupt_pid: int | None = None, log_dir: str | None = None): ...
    def __enter__(self) -> "ChunkGuard"        # runs idle_check(); raises GuardFailure if not idle; starts the watcher thread; records t_start
    def __exit__(self, *exc) -> None           # stops watcher; writes <log_dir>/<label>.guard.json
    aborted: bool                              # property
    reason: str | None                         # "ABORTED_USER_ARRIVED" | "ABORTED_WALL_CAP" | None
    cancel_event: threading.Event              # set on abort / wall cap
    def register_own_request(self, t_start: float | None = None) -> None   # call IMMEDIATELY BEFORE each HTTP request (time.time())
    def check(self) -> None                    # raises ChunkAborted if cancel_event is set; harnesses call it between requests
# guard.json = {label, t_start, t_end, aborted, reason, wall_cap_hit, own_requests:[...], signals_seen:[...]}
# On abort the watcher: sets cancel_event, prints the literal token to stderr, and (if interrupt_pid) os.kill(pid, SIGINT).

# CLI
#   python bench/phase20_guard.py idle                      # prints IdleReport JSON, exit 0 idle / 1 busy
#   python bench/phase20_guard.py wait-idle [--max-wait S]
#   python bench/phase20_guard.py canary                    # prints CanaryReport JSON, exit 0 healthy / 1 marginal / 2 degraded
#   python bench/phase20_guard.py watch --label L [--max-wall 900] [--canary] [--registry PATH] -- <command ...>
#       R1 (+ canary) -> run command as a subprocess under the watcher; on abort SIGINT it (SIGTERM after 20 s);
#       exit 75 = ABORTED_USER_ARRIVED, 76 = ABORTED_WALL_CAP, else the command's exit code.
```

## User-arrival signals (all must be implemented, documented and tested against recorded fixtures)
* **S1 concurrency/in-flight** — number of TextGeneration tasks active on the cluster (from `GET {API_BASE}/state` `tasks` and/or exo.log
  `Starting task TextGeneration` vs `TaskFinished` pairing on each node). More active tasks than the guarded process has registered in flight => arrival.
* **S2 POST lines** — `API request: POST /v1/chat/completions` (and other generation POST paths) in `~/.exo/exo_log/exo.log` on BOTH nodes whose
  timestamp does not match a registered own request (+-2 s) and is newer than chunk start. `GET /state`, `/metrics`, `/node_id`, `/v1/models` are NOT requests.
* **S3 completed rows** — state.db `api_calls` rows with `provider='custom'` and `started_at` > chunk start (late signal, still required).
* **S4 early session activity** — state.db `messages`/`sessions` activity for a session whose model is the exo model
  (`dealignai/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw`) newer than chunk start (Hermes writes the user message before the request is sent; verify this
  on the real 42-call session `20261007_092009_9a2ed7`).
Zero false positives from a harness's own sequential requests when it registers them; for non-registering harnesses (`watch -- <phase19 script>`)
S1 (concurrency >=2), S3 and S4 apply and the limitation is documented.

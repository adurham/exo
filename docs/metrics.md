# Metrics

Every exo node exposes a Prometheus-compatible scrape endpoint at
`GET /metrics` on its API port (default `52415`). The format is the standard
Prometheus text exposition, so it is scraped directly by VictoriaMetrics
(`vmagent` or single-node), Prometheus, or any compatible collector.

## Architecture

Three categories of metrics, with deliberate emit-side gating to avoid
double-counting when multiple nodes are scraped:

| Category | Emitted by | Labels |
|---|---|---|
| Per-generation counters & histograms | Current elected master only | `instance_id`, `model_id`, `finish_reason` |
| Per-node system gauges | Each node for its own `node_id` only | `node_id` |
| Cluster-state gauges | Current master, refreshed on scrape | `model_id`, `status` |
| Process / election | Every node | `node_id` |

Because only the master emits cluster-wide counters, you can safely scrape
every node. Non-master nodes will expose flat/zero counters for generation
metrics and their own system gauges.

## Metric catalog

### Process / election

- `exo_up{node_id}` — `1` when the process is serving metrics.
- `exo_is_master{node_id}` — `1` on the currently elected master, `0` elsewhere.

### Per-generation (master only)

- `exo_generation_requests_total{instance_id, model_id, finish_reason}` — counter.
- `exo_prompt_tokens_total{instance_id, model_id}` — counter.
- `exo_generation_tokens_total{instance_id, model_id}` — counter.
- `exo_prompt_tps{instance_id, model_id}` — histogram of prefill TPS at completion.
- `exo_generation_tps{instance_id, model_id}` — histogram of decode TPS at completion.
- `exo_prefix_cache_hits_total{instance_id, model_id, hit_kind}` —
  `hit_kind` is one of `none`, `partial`, `exact`.
- `exo_peak_memory_bytes{instance_id, model_id}` — gauge, peak memory from
  the most recent request.
- `exo_chunk_events_total{kind}` — counter of all streaming chunks observed at
  the master, bucketed by chunk class name (`TokenChunk`, `PrefillProgressChunk`,
  `ToolCallChunk`, `ErrorChunk`, `ImageChunk`).

### Per-node system

Sourced from the local `InfoGatherer` (`macmon`, `psutil`, `rdma_ctl`,
models-disk stat):

- `exo_gpu_usage_ratio{node_id}` — 0.0–1.0
- `exo_gpu_temp_celsius{node_id}`
- `exo_system_power_watts{node_id}`
- `exo_pcpu_usage_ratio{node_id}` / `exo_ecpu_usage_ratio{node_id}`
- `exo_memory_ram_used_bytes{node_id}` / `exo_memory_ram_total_bytes{node_id}`
- `exo_memory_swap_used_bytes{node_id}` / `exo_memory_swap_total_bytes{node_id}`
- `exo_disk_available_bytes{node_id}` / `exo_disk_total_bytes{node_id}`
- `exo_rdma_enabled{node_id}` — `1` if `rdma_ctl status` reports enabled.

### Cluster state (master only, refreshed on scrape)

- `exo_instances{model_id}` — number of placed instances.
- `exo_runners{status}` — runner counts bucketed by `RunnerStatus` tag
  (`RunnerIdle`, `RunnerRunning`, `RunnerTerminated`, etc).

## VictoriaMetrics scrape config

For a `vmagent` or single-node VM, add to `prometheus.yml` / `scrape.yml`:

```yaml
scrape_configs:
  - job_name: exo
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets:
          - macstudio-m4-1:52415
          - macstudio-m4-2:52415
          # - macbook-m4:52415
        labels:
          cluster: homelab
```

The `exo_is_master` gauge is the source of truth for which node is
authoritative: filter generation metrics by a joined condition if you ever
want to disambiguate historical master changes.

## Example PromQL / MetricsQL

- Decode throughput per instance over 5 min:
  ```
  rate(exo_generation_tokens_total[5m])
  ```
- Prefix cache hit ratio per model:
  ```
  sum by (model_id) (rate(exo_prefix_cache_hits_total{hit_kind=~"partial|exact"}[5m]))
  / sum by (model_id) (rate(exo_prefix_cache_hits_total[5m]))
  ```
- P50 / P95 decode TPS:
  ```
  histogram_quantile(0.50, rate(exo_generation_tps_bucket[5m]))
  histogram_quantile(0.95, rate(exo_generation_tps_bucket[5m]))
  ```
- GPU utilization by node:
  ```
  exo_gpu_usage_ratio
  ```

## Grafana dashboard

A starter dashboard lives at
[`deploy/grafana/exo-overview.json`](../deploy/grafana/exo-overview.json).
Import it, select the VictoriaMetrics data source, and the panels light up
with cluster-wide throughput, per-node GPU / power, prefix cache hit rate,
and instance counts.

use std::sync::Arc;

use tokio::task::JoinHandle;
use zenoh::{Result, Session as ZSession, config::Locator};
use zenoh_plugin_storage_manager::StoragesPlugin;
use zenoh_plugin_trait::PluginsManager;

pub use zenoh::{Config, config::ZenohId};

use crate::discovery::Discovery;

pub mod discovery;
pub mod swarm;

pub fn is_valid_zid(identity: &str) -> bool {
    let mut iter = identity.chars();
    iter.next()
        .is_some_and(|c| ('1'..='9').contains(&c) || ('a'..='f').contains(&c))
        && iter.all(|c| ('0'..='9').contains(&c) || ('a'..='f').contains(&c))
        && identity.len() <= 32
}

pub fn cfg(identity: &str, listen_port: u16) -> Result<zenoh::Config> {
    assert!(is_valid_zid(identity));
    assert!(identity.len() <= 32);
    assert!(listen_port != 0, "must used defined listen port");
    let mut cfg = zenoh::Config::default();
    // todo: cleanup
    cfg.insert_json5("id", &format!("\"{identity}\""))?;
    cfg.insert_json5("mode", "\"router\"")?;
    cfg.insert_json5("listen/endpoints", &format!("[\"tcp/[::]:{listen_port}\"]"))?;
    cfg.insert_json5("scouting/multicast/enabled", "false")?;
    cfg.insert_json5("scouting/multicast/autoconnect", "[]")?;
    cfg.insert_json5("scouting/gossip/multihop", "true")?;
    cfg.insert_json5("adminspace/enabled", "true")?;
    //cfg.insert_json5("transport/link/tx/batch_size", "9216")?;
    cfg.insert_json5("transport/link/rx/buffer_size", "16777216")?;
    // Multi-fragment command delivery. Commands whose serialised form is a few MB (the 1M row's
    // TextGeneration is 6.69 MB) are pushed as many fragments by CongestionControl::Drop
    // publishers (swarm.rs declares every publisher that way). Zenoh's defaults are a 2-batch
    // queue per priority (2 x 65,535 B) and a 50 ms deadline for fragments of one message: at
    // ~1 Gbps over the Thunderbolt control link a 6 MB message takes about that long, so trailing
    // fragments were dropped and the peer never reassembled the command -- silently, with no
    // error on either end, which surfaced only as "the cluster didn't accept the request".
    // Measured boundary before this change: 6.00 MB delivered, 6.20 MB lost (REPRO-RUN-2026-10-18
    // §11.12). Deepen the queue and give a whole message 2 s to clear.
    cfg.insert_json5(
        "transport/link/tx/queue/size",
        "{control:8,real_time:8,interactive_high:8,interactive_low:8,data_high:8,data:8,data_low:8,background:8}",
    )?;
    cfg.insert_json5(
        "transport/link/tx/queue/congestion_control/drop/max_wait_before_drop_fragments",
        "2000000",
    )?;
    //cfg.insert_json5("timestamping/enabled", "true")?;
    cfg.insert_json5("plugins/storage_manager/__required__", "true")?;
    cfg.insert_json5(
        "plugins/storage_manager/storages/mem1",
        r#"{
            key_expr: "storage/mem1/**",
            strip_prefix: "storage/mem1",
            volume: "memory",
            replication: {
                interval: 2,
            }
        }"#,
    )?;
    // Static peering escape hatch (env-gated; unset == previous behaviour).
    //
    // exo's only peering path is the custom IPv6 multicast beacon in
    // discovery.rs (group ff12::e0a1:de89). On macOS 26/27 that beacon never
    // reaches the peer, so both nodes elect themselves Master and split-brain.
    //
    // Root cause (proven 2026-09-10): TCC's LocalNetwork restriction denies a
    // DETACHED process (screen -dmS / nohup / ppid==1 -- how exo is launched)
    // all access to every LOCAL subnet, unicast and multicast alike, for both
    // IPv4 and IPv6. The identical binary run from an interactive ssh session
    // is permitted, which is why in-session probes kept passing while the live
    // daemon failed. Loopback, WAN and Tailscale/utun are NOT treated as local
    // and work normally from a detached process.
    //
    // EXO_ZENOH_CONNECT holds a comma-separated list of host:port zenoh
    // endpoints to dial directly, e.g. "192.168.86.202:52414". zenoh's
    // connect/retry defaults (period_init_ms 1000, period_max_ms 4000,
    // timeout_ms router:-1, exit_on_failure router:false) mean a node started
    // before its peer retries forever instead of failing at startup, so launch
    // order does not matter.
    if let Ok(raw) = std::env::var("EXO_ZENOH_CONNECT") {
        let endpoints: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| format!("\"tcp/{s}\""))
            .collect();
        if !endpoints.is_empty() {
            log::info!("EXO_ZENOH_CONNECT static peers: {endpoints:?}");
            cfg.insert_json5("connect/endpoints", &format!("[{}]", endpoints.join(",")))?;
        }
    }
    Ok(cfg)
}

pub async fn open(
    cfg: zenoh::Config,
    namespace: &str,
    listen_port: u16,
    discovery_service_port: u16,
) -> Result<Session> {
    assert!(listen_port != 0, "must used defined listen port");
    let namespace: [u8; 8] = {
        blake3::hash(namespace.as_bytes()).as_bytes()[..8]
            .try_into()
            .expect("8 is equal to 8")
    };
    let mut plugins = PluginsManager::static_plugins_only();
    plugins.declare_static_plugin::<StoragesPlugin, _>("storage_manager", true);
    let mut runtime = zenoh::internal::runtime::RuntimeBuilder::new(cfg)
        .plugins_manager(plugins)
        .build()
        .await?;
    let z = zenoh::session::init(runtime.clone().into()).await?;
    runtime.start().await?;
    let mut discovery =
        Discovery::new(z.zid(), namespace, listen_port, discovery_service_port).await?;
    let _jh = Arc::new(AbortOnDrop(tokio::task::spawn(async move {
        loop {
            let Ok(discovered) = discovery.next().await.inspect_err(|e| {
                log::warn!("discovery error {e}");
            }) else {
                continue;
            };

            if discovered.zid > runtime.zid() {
                log::debug!("not connecting to peer with greater zid");
                continue;
            }

            // Unmap ::ffff:a.b.c.d -> a.b.c.d so zenoh dials a native v4
            // locator rather than a v4-mapped one.
            //
            // (An earlier comment blamed macOS 27 for "refusing v4-mapped
            // destinations". That was wrong -- v4-mapped sends succeed fine;
            // the real blocker was TCC LocalNetwork denying the detached
            // process every local-subnet destination. Emitting a native v4
            // locator is still preferable for readability.)
            let addr_str = match discovered.addr.ip().to_ipv4_mapped() {
                Some(v4) => std::net::SocketAddrV4::new(v4, discovered.addr.port()).to_string(),
                None => discovered.addr.to_string(),
            };
            let Ok(locator) = Locator::new("tcp", addr_str, "").inspect_err(|e| {
                log::warn!("failed to parse locator from addr: {e}");
            }) else {
                continue;
            };

            // Don't wait for the connection here: a slow dial would stall discovery
            // (and look like this node stalling). zenoh ignores dials to a peer it is
            // already connecting to.
            let runtime = runtime.clone();
            tokio::task::spawn(async move {
                runtime
                    .connect_peer(&discovered.zid.into(), &[locator])
                    .await;
            });
        }
    })));
    Ok(Session { z, _jh })
}

struct AbortOnDrop(JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[derive(Clone)]
pub struct Session {
    pub z: ZSession,
    _jh: Arc<AbortOnDrop>,
}

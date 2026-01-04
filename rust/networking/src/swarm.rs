use crate::alias;
use crate::swarm::transport::tcp_transport;
pub use behaviour::{Behaviour, BehaviourEvent};
use libp2p::{SwarmBuilder, identity};

pub type Swarm = libp2p::Swarm<Behaviour>;

/// The current version of the network: this prevents devices running different versions of the
/// software from interacting with each other.
///
/// TODO: right now this is a hardcoded constant; figure out what the versioning semantics should
///       even be, and how to inject the right version into this config/initialization. E.g. should
///       this be passed in as a parameter? What about rapidly changing versions in debug builds?
///       this is all VERY very hard to figure out and needs to be mulled over as a team.
pub const NETWORK_VERSION: &[u8] = b"v0.0.1";
pub const OVERRIDE_VERSION_ENV_VAR: &str = "EXO_LIBP2P_NAMESPACE";

/// Create and configure a swarm which listens to all ports on OS
pub fn create_swarm(keypair: identity::Keypair) -> alias::AnyResult<Swarm> {
    let mut swarm = SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_other_transport(tcp_transport)?
        .with_behaviour(Behaviour::new)?
        .build();

    // Listen on all interfaces on a FIXED port to allow static peering/blind dialing
    // Port 56789 is chosen as the standard Exo P2P port.
    // Listen on explicit interfaces to exclude Thunderbolt (Data Plane)
    // We bind to 0.0.0.0 only as a fallback if interface discovery fails, 
    // but prefer binding key interfaces directly to avoid TB usage.
    let mut bound_any = false;
    if let Ok(ifaces) = if_addrs::get_if_addrs() {
        for iface in ifaces {
            let ip_str = iface.addr.ip().to_string();
            
            // Skip Thunderbolt Subnets (192.168.2xx)
            if ip_str.starts_with("192.168.2") { 
                log::info!("RUST: Skipping listener on Thunderbolt IP (Data Plane): {}", ip_str);
                continue; 
            }

            // Bind to loopback, standard LAN, and Tailscale
            // IPv4 only for now to match previous "/ip4/" logic, though libp2p supports v6
            if iface.addr.ip().is_ipv4() {
                 let addr_str = format!("/ip4/{}/tcp/56789", ip_str);
                 if let Ok(addr) = addr_str.parse() {
                     log::info!("RUST: Control Plane Listening on: {}", addr);
                     if swarm.listen_on(addr).is_ok() {
                         bound_any = true;
                     }
                 }
            }
        }
    }

    // Fallback: If we couldn't bind to any specific interface, bind to 0.0.0.0
    // This ensures we never fail to start, but might accidentally use TB if discovery failed.
    if !bound_any {
        log::warn!("RUST: Failed to find specific interfaces, falling back to 0.0.0.0");
        swarm.listen_on("/ip4/0.0.0.0/tcp/56789".parse()?)?;
    }

    Ok(swarm)
}

mod transport {
    use crate::alias;
    use crate::swarm::{NETWORK_VERSION, OVERRIDE_VERSION_ENV_VAR};
    use futures::{AsyncRead, AsyncWrite};
    use keccak_const::Sha3_256;
    use libp2p::core::muxing;
    use libp2p::core::transport::Boxed;
    use libp2p::pnet::{PnetError, PnetOutput};
    use libp2p::{PeerId, Transport, identity, noise, pnet, yamux};
    use std::{env, sync::LazyLock};

    /// Key used for networking's private network; parametrized on the [`NETWORK_VERSION`].
    /// See [`pnet_upgrade`] for more.
    static PNET_PRESHARED_KEY: LazyLock<[u8; 32]> = LazyLock::new(|| {
        let builder = Sha3_256::new().update(b"exo_discovery_network");

        if let Ok(var) = env::var(OVERRIDE_VERSION_ENV_VAR) {
            let bytes = var.into_bytes();
            builder.update(&bytes)
        } else {
            builder.update(NETWORK_VERSION)
        }
        .finalize()
    });

    /// Make the Swarm run on a private network, as to not clash with public libp2p nodes and
    /// also different-versioned instances of this same network.
    /// This is implemented as an additional "upgrade" ontop of existing [`libp2p::Transport`] layers.
    async fn pnet_upgrade<TSocket>(
        socket: TSocket,
        _: impl Sized,
    ) -> Result<PnetOutput<TSocket>, PnetError>
    where
        TSocket: AsyncRead + AsyncWrite + Send + Unpin + 'static,
    {
        use pnet::{PnetConfig, PreSharedKey};
        PnetConfig::new(PreSharedKey::new(*PNET_PRESHARED_KEY))
            .handshake(socket)
            .await
    }

    /// TCP/IP transport layer configuration.
    pub fn tcp_transport(
        keypair: &identity::Keypair,
    ) -> alias::AnyResult<Boxed<(PeerId, muxing::StreamMuxerBox)>> {
        use libp2p::{
            core::upgrade::Version,
            tcp::{Config, tokio},
        };

        // `TCP_NODELAY` enabled => avoid latency
        let tcp_config = Config::default().nodelay(true);

        // V1 + lazy flushing => 0-RTT negotiation
        let upgrade_version = Version::V1Lazy;

        // Noise is faster than TLS + we don't care much for security
        let noise_config = noise::Config::new(keypair)?;

        // Use default Yamux config for multiplexing
        let yamux_config = yamux::Config::default();

        // Create new Tokio-driven TCP/IP transport layer
        let base_transport = tokio::Transport::new(tcp_config)
            .and_then(pnet_upgrade)
            .upgrade(upgrade_version)
            .authenticate(noise_config)
            .multiplex(yamux_config);

        // Return boxed transport (to flatten complex type)
        Ok(base_transport.boxed())
    }
}

mod behaviour {
    use crate::{alias, discovery};
    use libp2p::swarm::NetworkBehaviour;
    use libp2p::{gossipsub, identity};
    use std::time::Duration;

    /// Behavior of the Swarm which composes all desired behaviors:
    /// Right now its just [`discovery::Behaviour`] and [`gossipsub::Behaviour`].
    #[derive(NetworkBehaviour)]
    pub struct Behaviour {
        pub discovery: discovery::Behaviour,
        pub gossipsub: gossipsub::Behaviour,
    }

    impl Behaviour {
        pub fn new(keypair: &identity::Keypair) -> alias::AnyResult<Self> {
            Ok(Self {
                discovery: discovery::Behaviour::new(keypair)?,
                gossipsub: gossipsub_behaviour(keypair),
            })
        }
    }

    fn gossipsub_behaviour(keypair: &identity::Keypair) -> gossipsub::Behaviour {
        use gossipsub::{ConfigBuilder, MessageAuthenticity, ValidationMode};

        // build a gossipsub network behaviour
        //  => signed message authenticity + strict validation mode means the message-ID is
        //     automatically provided by gossipsub w/out needing to provide custom message-ID function
        gossipsub::Behaviour::new(
            MessageAuthenticity::Signed(keypair.clone()),
            ConfigBuilder::default()
                .publish_queue_duration(Duration::from_secs(15))
                .max_transmit_size(1024 * 1024)
                .validation_mode(ValidationMode::Strict)
                .build()
                .expect("the configuration should always be valid"),
        )
        .expect("creating gossipsub behavior should always work")
    }
}

use std::{
    io,
    net::{Ipv6Addr, SocketAddr, SocketAddrV6},
    sync::Arc,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};
use log::{debug, trace, warn};
use netwatcher::WatchHandle;
use parking_lot::Mutex;
use tokio::{
    net::UdpSocket,
    time::{Interval, interval},
};
use zenoh::config::ZenohId;

const GROUP: Ipv6Addr = Ipv6Addr::new(0xff12, 0, 0, 0, 0, 0, 0xe0a1, 0xde89);
const MAGIC: [u8; 3] = *b"EXO";

pub struct Discovery {
    sock: Arc<UdpSocket>,
    ifaces: Arc<Mutex<Vec<SocketAddrV6>>>,
    /// Unicast IPv6 peers announced to in addition to the multicast group.
    /// Populated from EXO_DISCOVERY_UNICAST_PEERS. Empty == previous behaviour.
    static_peers: Vec<SocketAddrV6>,
    /// Unicast IPv4 peers, sent from `sock_v4`.
    ///
    /// NOTE: an earlier revision claimed the dual-stack `sock` cannot send to
    /// v4-mapped (::ffff:a.b.c.d) addresses on macOS 27. That is NOT true --
    /// disproven 2026-09-10 with exo's exact socket configuration: 16764/16764
    /// v4-mapped sends succeeded. The EHOSTUNREACH that motivated the original
    /// comment comes from TCC LocalNetwork and depends on the process LAUNCH
    /// CONTEXT (detached => every local-subnet destination is blocked for both
    /// address families), never on the address family. A native AF_INET socket
    /// is still used here because it keeps the v4 path explicit and costs
    /// nothing.
    static_peers_v4: Vec<std::net::SocketAddrV4>,
    /// AF_INET twin of `sock`, bound to the same discovery port, present only
    /// when there is at least one IPv4 static peer.
    sock_v4: Option<Arc<UdpSocket>>,
    namespace: [u8; 8],
    last_nonce: Mutex<[u8; 8]>,
    /// the port of the service we are doing discovery for - transmitted to peers
    listen_port: u16,
    zid: ZenohId,
    tick: Interval,
    _sync: Mutex<WatchHandle>,
}

#[derive(Debug, Clone, Copy)]
pub struct Discovered {
    pub zid: ZenohId,
    pub addr: SocketAddrV6,
}

impl Discovery {
    pub async fn new(
        zid: ZenohId,
        namespace: [u8; 8],
        listen_port: u16,
        discovery_port: u16,
    ) -> io::Result<Self> {
        let sock = socket2::Socket::new(
            socket2::Domain::IPV6,
            socket2::Type::DGRAM,
            Some(socket2::Protocol::UDP),
        )?;
        sock.set_reuse_address(true)?;
        #[cfg(unix)]
        sock.set_reuse_port(true)?;
        sock.bind(&SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, discovery_port, 0, 0).into())?;
        sock.set_nonblocking(true)?;
        sock.set_multicast_loop_v6(true)?;
        let sock = Arc::new(UdpSocket::from_std(sock.into())?);
        let ifaces: Arc<Mutex<Vec<SocketAddrV6>>> = Default::default();
        let _sync = Mutex::new(
            netwatcher::watch_interfaces_with_callback({
                let sock = sock.clone();
                let ifaces = ifaces.clone();
                move |update| {
                    for (iface_idx, iface) in update.interfaces.iter() {
                        if iface
                            .ipv6_ips()
                            .all(|addr| addr.is_loopback() || addr.is_unspecified())
                        {
                            continue;
                        }

                        match sock.join_multicast_v6(&GROUP, *iface_idx) {
                            Ok(()) => ifaces.lock().push(SocketAddrV6::new(
                                GROUP,
                                discovery_port,
                                0,
                                *iface_idx,
                            )),
                            Err(e) if e.kind() != io::ErrorKind::AddrInUse => {
                                // skip AddrInUse - just means we've already joined the mv6
                                if let Some(iface) = update.interfaces.get(&iface_idx) {
                                    warn!(
                                        "failed to join multicast v6 for interface {}: {e}",
                                        iface.name
                                    )
                                }
                            }
                            _ => {}
                        }
                    }
                    for iface_idx in update.diff.removed {
                        ifaces.lock().retain(|addr| addr.scope_id() != iface_idx);

                        if let Err(e) = sock.leave_multicast_v6(&GROUP, iface_idx) {
                            if let Some(iface) = update.interfaces.get(&iface_idx) {
                                warn!(
                                    "failed to leave multicast v6 for interface {}: {e}",
                                    iface.name
                                )
                            }
                        }
                    }
                }
            })
            // todo: better error handling here
            .expect("failed to bind discovery watcher"),
        );
        let (static_peers, static_peers_v4) = parse_static_peers(discovery_port);
        // Bound unconditionally so a node with no static peers of its own can
        // still participate in the v4 unicast path.
        //
        // (The original justification -- "replying to a v4-mapped address from
        // the dual-stack socket fails on macOS 27" -- was wrong; see the
        // static_peers_v4 note above. Binding unconditionally is still the
        // right call because it keeps both nodes symmetric.)
        let sock_v4 = match bind_v4(discovery_port) {
            Ok(s) => Some(Arc::new(s)),
            Err(e) => {
                warn!("failed to bind IPv4 discovery socket: {e}");
                None
            }
        };
        Ok(Self {
            sock,
            namespace,
            static_peers,
            static_peers_v4,
            sock_v4,
            ifaces,
            last_nonce: Mutex::new(rand::random()),
            listen_port,
            zid,
            tick: interval(Duration::from_secs(1)),
            _sync,
        })
    }

    pub async fn next(&mut self) -> io::Result<Discovered> {
        let mut buf = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        let mut buf4 = [0u8; Hello::buf_size() + WhatsUp::buf_size() + 1];
        loop {
            // recv_v4 resolves to Pending forever when there is no v4 socket,
            // so the select! arm is simply never taken in that case.
            let recv_v4 = async {
                match self.sock_v4.as_ref() {
                    Some(s) => s.recv_from(&mut buf4).await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                _ = self.tick.tick() => {
                    self.announce().await?;
                }
                res = self.sock.recv_from(&mut buf) => {
                    let Ok((bytes_read, addr)) = res else { continue; };
                    if let Some(discovered) = self.respond(bytes_read, addr, &buf).await? {
                        return Ok(discovered)
                    }
                }
                res = recv_v4 => {
                    let Ok((bytes_read, addr)) = res else { continue; };
                    // Normalise to v6 so respond()'s SocketAddr::V6 match arm
                    // (and the resulting zenoh locator) behave identically.
                    let addr = match addr {
                        SocketAddr::V4(v4) => SocketAddr::V6(SocketAddrV6::new(
                            v4.ip().to_ipv6_mapped(), v4.port(), 0, 0,
                        )),
                        other => other,
                    };
                    if let Some(discovered) = self.respond_v4(bytes_read, addr, &buf4).await? {
                        return Ok(discovered)
                    }
                }
            }
        }
    }

    /// Same as `respond`, but replies out the IPv4 socket. Kept as a thin
    /// wrapper so the wire logic lives in exactly one place.
    async fn respond_v4(
        &self,
        bytes_read: usize,
        addr: SocketAddr,
        buf: &[u8],
    ) -> io::Result<Option<Discovered>> {
        self.respond_inner(bytes_read, addr, buf, true).await
    }

    async fn respond(
        &self,
        bytes_read: usize,
        addr: SocketAddr,
        buf: &[u8],
    ) -> io::Result<Option<Discovered>> {
        self.respond_inner(bytes_read, addr, buf, false).await
    }

    async fn respond_inner(
        &self,
        bytes_read: usize,
        addr: SocketAddr,
        buf: &[u8],
        via_v4: bool,
    ) -> io::Result<Option<Discovered>> {
        trace!(
            "raw recv: {bytes_read} bytes from {addr}: {:02x?}",
            &buf[..bytes_read]
        );
        if bytes_read < size_of::<Header>() {
            trace!("dropped: early EOF");
            return Ok(None);
        }
        let header: &Header = bytemuck::from_bytes(&buf[0..size_of::<Header>()]);
        if header.magic != MAGIC {
            trace!("dropped: wrong magic");
            return Ok(None);
        }
        let Ok(kind) = header.kind.try_into() else {
            trace!("dropped: unknown message kind {}", header.kind);
            return Ok(None);
        };
        match kind {
            Kind::Hello => {
                let total = Hello::buf_size();
                if bytes_read != total {
                    trace!("dropped: hello wrong size");
                    return Ok(None);
                }
                let hello: &Hello = bytemuck::from_bytes(&buf[size_of::<Header>()..total]);
                if hello.nonce == *self.last_nonce.lock() {
                    trace!("dropped: local hello nonce");
                    return Ok(None);
                }
                if hello.namespace != self.namespace {
                    trace!("dropped: different namespace");
                    return Ok(None);
                }

                // reply
                trace!("replying to Hello({:?})", hello.nonce);
                let reply = WhatsUp {
                    nonce: hello.nonce,
                    zid: self.zid.to_le_bytes(),
                    port_le: self.listen_port.to_le_bytes(),
                }
                .alloc();

                for i in 1..6 {
                    // Reply out the socket the Hello ARRIVED on, so the
                    // WhatsUp's source address matches the destination the
                    // peer sent to and its socket actually sees the reply.
                    let (sock, reply_addr) = match (via_v4, self.sock_v4.as_ref(), addr) {
                        (true, Some(s4), SocketAddr::V6(v6)) => {
                            match v6.ip().to_ipv4_mapped() {
                                Some(v4) => (
                                    s4.as_ref(),
                                    SocketAddr::V4(std::net::SocketAddrV4::new(v4, v6.port())),
                                ),
                                None => (self.sock.as_ref(), addr),
                            }
                        }
                        _ => (self.sock.as_ref(), addr),
                    };
                    if sock
                        .send_to(&reply, reply_addr)
                        .await
                        .inspect_err(|e| debug!("send to {addr} failed: {e}"))
                        .is_ok_and(|sent| sent == WhatsUp::buf_size())
                    {
                        trace!(
                            "sent {} bytes to {addr} after {} attempt(s)",
                            WhatsUp::buf_size(),
                            i
                        );
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(300)).await;
                }
                Ok(None)
            }
            Kind::WhatsUp => {
                let total = WhatsUp::buf_size();
                if bytes_read != total {
                    trace!("dropped: whatsup wrong size");
                    return Ok(None);
                }
                let whats_up: &WhatsUp = bytemuck::from_bytes(&buf[size_of::<Header>()..total]);
                if whats_up.nonce != *self.last_nonce.lock() {
                    trace!("dropped: stale nonce");
                    return Ok(None);
                }
                let SocketAddr::V6(v6) = addr else {
                    trace!("dropped: v4 addr used");
                    return Ok(None);
                };
                let Ok(zid) = ZenohId::try_from(&whats_up.zid[..]) else {
                    trace!("dropped: zenoh conversion failed");
                    return Ok(None);
                };
                if zid == self.zid {
                    trace!("dropped: self zenoh id");
                    return Ok(None);
                }
                // discovery success!
                // the incoming port is our listen port;
                // overwrite it with the whats_up port corresponding to the remote zenoh service
                let addr = {
                    let mut x = v6;
                    x.set_port(u16::from_le_bytes(whats_up.port_le));
                    x
                };
                Ok(Some(Discovered { addr, zid }))
            }
        }
    }

    async fn announce(&self) -> io::Result<()> {
        let nonce = rand::random();
        *self.last_nonce.lock() = nonce;
        let buf = Hello {
            nonce,
            namespace: self.namespace,
        }
        .alloc();

        // IPv4 static peers go out the AF_INET socket.
        if let Some(sock4) = self.sock_v4.as_ref() {
            for addr in &self.static_peers_v4 {
                match sock4.send_to(&buf, SocketAddr::V4(*addr)).await {
                    Ok(bytes) => trace!("sent {bytes} to {addr} (v4)"),
                    Err(e) => debug!("static v4 peer {addr} unreachable, will retry: {e}"),
                }
            }
        }

        let mut addrs = self.ifaces.lock().clone();
        // Unicast peers are appended, never replace the multicast targets: if
        // multicast starts working again this keeps costing one extra datagram
        // per second and changes nothing else. Appended AFTER the iface list so
        // the swap_remove-on-HostUnreachable indexing below stays correct for
        // the iface entries.
        addrs.extend_from_slice(&self.static_peers);
        debug!("announcing Hello({nonce:?}) to {addrs:?}");
        // rev so .remove() doesn't break things
        for (i, addr) in addrs.into_iter().enumerate().rev() {
            match self.sock.send_to(&buf, addr).await {
                Ok(bytes) => trace!("sent {bytes} to {addr}"),
                Err(e) if e.kind() == io::ErrorKind::HostUnreachable => {
                    // Only iface-derived entries live in self.ifaces; a static
                    // peer's index is past the end of that vec, and a static
                    // peer must never be disabled anyway (the whole point is
                    // that it keeps retrying until the peer comes up).
                    if i < self.ifaces.lock().len() {
                        debug!("disabling discovery address {addr}: {e}");
                        _ = self.ifaces.lock().swap_remove(i);
                    } else {
                        debug!("static peer {addr} unreachable, will retry: {e}");
                    }
                }
                Err(e) => debug!("failed to reach {addr}: {e}"),
            }
        }
        Ok(())
    }
}

/// Parses EXO_DISCOVERY_UNICAST_PEERS: a comma-separated list of peer hosts to
/// send discovery Hellos to directly, e.g. "192.168.86.202" or
/// "192.168.86.202:52413". The port defaults to this node's discovery port,
/// which is what the launcher configures on every node.
///
/// IPv4 literals are converted to IPv4-mapped IPv6 (::ffff:a.b.c.d) because the
/// discovery socket is a dual-stack AF_INET6 socket bound to [::] (confirmed
/// dual-stack: netstat reports it as udp46).
fn parse_static_peers(
    default_port: u16,
) -> (Vec<SocketAddrV6>, Vec<std::net::SocketAddrV4>) {
    let Ok(raw) = std::env::var("EXO_DISCOVERY_UNICAST_PEERS") else {
        return (Vec::new(), Vec::new());
    };
    let mut out = Vec::new();
    let mut out_v4 = Vec::new();
    for entry in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (host, port) = match entry.rsplit_once(':') {
            // Guard against bare IPv6 literals being split on their own colons.
            Some((h, p)) if !h.contains(':') => match p.parse::<u16>() {
                Ok(p) => (h, p),
                Err(_) => (entry, default_port),
            },
            _ => (entry, default_port),
        };
        let host = host.trim_start_matches('[').trim_end_matches(']');
        match host.parse::<std::net::IpAddr>() {
            Ok(std::net::IpAddr::V4(v4)) => {
                out_v4.push(std::net::SocketAddrV4::new(v4, port));
            }
            Ok(std::net::IpAddr::V6(v6)) => {
                out.push(SocketAddrV6::new(v6, port, 0, 0));
            }
            Err(e) => warn!("ignoring EXO_DISCOVERY_UNICAST_PEERS entry {entry:?}: {e}"),
        }
    }
    if !out.is_empty() || !out_v4.is_empty() {
        debug!("discovery static unicast peers: v6={out:?} v4={out_v4:?}");
    }
    (out, out_v4)
}

/// Binds the AF_INET twin of the discovery socket.
///
/// Deliberately binds an EPHEMERAL port, not the discovery port: exo already
/// holds a dual-stack AF_INET6 socket on [::]:<discovery_port>, which on macOS
/// occupies the v4 port too, so an AF_INET bind to the same port fails with
/// EADDRINUSE and leaves us with no v4 socket at all. Replies are unaffected --
/// the peer answers the Hello to its source address, and this socket is polled
/// in next() alongside the v6 one.
///
/// IMPORTANT (macOS 26/27): neither this socket nor the v6 one can reach ANY
/// local subnet when exo is launched detached (screen -dmS, nohup, ppid==1) --
/// TCC's LocalNetwork restriction denies the whole process context and every
/// local destination returns EHOSTUNREACH (errno 65). Loopback, the WAN and
/// Tailscale/utun are exempt, so the launcher points
/// EXO_DISCOVERY_UNICAST_PEERS at the peer's Tailscale address. See
/// start_cluster.sh.
fn bind_v4(_port: u16) -> io::Result<UdpSocket> {
    let sock = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::DGRAM,
        Some(socket2::Protocol::UDP),
    )?;
    sock.set_reuse_address(true)?;
    #[cfg(unix)]
    sock.set_reuse_port(true)?;
    sock.bind(&std::net::SocketAddrV4::new(std::net::Ipv4Addr::UNSPECIFIED, 0).into())?;
    sock.set_nonblocking(true)?;
    UdpSocket::from_std(sock.into())
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
// packet & version
pub enum Kind {
    Hello = 0,
    WhatsUp = 1,
}

pub struct UnknownKind;
impl TryFrom<u8> for Kind {
    type Error = UnknownKind;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Hello),
            1 => Ok(Self::WhatsUp),
            _ => Err(UnknownKind),
        }
    }
}

pub trait Message: Pod {
    const KIND: Kind;
}
// should be part of the Message trait, but const in traits isnt stabilized. this lets alloc :: Self -> [u8; Self::buf_size()]
macro_rules! impl_alloc {
    ($a:ident) => {
        impl $a {
            const fn buf_size() -> usize {
                size_of::<Header>() + size_of::<Self>()
            }
            pub fn alloc(self) -> [u8; Self::buf_size()] {
                let mut buf = [0u8; Self::buf_size()];
                buf[0..size_of::<Header>()].copy_from_slice(bytemuck::bytes_of(&Header {
                    magic: MAGIC,
                    kind: Self::KIND as u8,
                }));
                buf[size_of::<Header>()..Self::buf_size()]
                    .copy_from_slice(bytemuck::bytes_of(&self));
                buf
            }
        }
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Header {
    magic: [u8; 3],
    kind: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Hello {
    pub nonce: [u8; 8],
    pub namespace: [u8; 8],
}
impl Message for Hello {
    const KIND: Kind = Kind::Hello;
}
impl_alloc!(Hello);

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WhatsUp {
    pub nonce: [u8; 8],
    pub zid: [u8; 16],
    pub port_le: [u8; 2],
}
impl Message for WhatsUp {
    const KIND: Kind = Kind::WhatsUp;
}
impl_alloc!(WhatsUp);

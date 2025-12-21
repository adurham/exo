"""Message routing infrastructure for libp2p-based communication.

This module provides the Router and TopicRouter classes for routing messages
across libp2p topics to local receivers and the network.
"""

from copy import copy
from itertools import count
from math import inf
from os import PathLike
from pathlib import Path
from typing import Iterable, cast

from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    create_task_group,
    sleep_forever,
)
from anyio.abc import TaskGroup
from exo_pyo3_bindings import (
    AllQueuesFullError,
    Keypair,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
)
from filelock import FileLock
from loguru import logger

from exo.shared.constants import EXO_NODE_ID_KEYPAIR
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.pydantic_ext import CamelCaseModel

from .connection_message import ConnectionMessage
from .topics import CONNECTION_MESSAGES, PublishPolicy, TypedTopic


class TopicRouter[T: CamelCaseModel]:
    """Router for messages on a specific topic.

    Routes messages between local senders/receivers and the libp2p network.
    Supports different publish policies (Never, Minimal, Always) to control
    when messages are sent to the network.

    Note:
        Current limitation: TopicRouter cannot prevent feedback loops as it
        doesn't track system IDs to identify message origin. This is only
        relevant for Election messages.

    Attributes:
        topic: Typed topic this router handles.
        senders: Set of local senders subscribed to this topic.
        receiver: Receiver for incoming messages from the network.
        _sender: Internal sender for routing messages locally.
        networking_sender: Sender for publishing messages to the network.
    """

    def __init__(
        self,
        topic: TypedTopic[T],
        networking_sender: Sender[tuple[str, bytes]],
        max_buffer_size: float = inf,
    ):
        """Initialize a topic router.

        Args:
            topic: Typed topic configuration.
            networking_sender: Channel for sending to network layer.
            max_buffer_size: Maximum buffer size (currently unused).
        """
        self.topic: TypedTopic[T] = topic
        self.senders: set[Sender[T]] = set()
        send, recv = channel[T]()
        self.receiver: Receiver[T] = recv
        self._sender: Sender[T] = send
        self.networking_sender: Sender[tuple[str, bytes]] = networking_sender

    async def run(self) -> None:
        """Run the topic router's message routing loop.

        Receives messages and routes them according to the topic's publish policy:
        - Minimal: Only publish to network if no local receivers
        - Always: Always publish to network
        - Never: Never publish to network (local only)
        """
        logger.debug(f"Topic Router {self.topic} ready to send")
        with self.receiver as items:
            async for item in items:
                if (
                    len(self.senders) == 0
                    and self.topic.publish_policy is PublishPolicy.Minimal
                ):
                    await self._send_out(item)
                    continue
                if self.topic.publish_policy is PublishPolicy.Always:
                    await self._send_out(item)
                await self.publish(item)

    async def shutdown(self) -> None:
        """Shutdown the topic router and close all channels."""
        logger.debug(f"Shutting down Topic Router {self.topic}")
        for sender in self.senders:
            sender.close()
        self._sender.close()
        self.receiver.close()

    async def publish(self, item: T) -> None:
        """Publish an item to all local receivers.

        Sends the item to all subscribed senders. Automatically removes
        closed/broken senders from the set.

        Args:
            item: Message to publish.

        Note:
            This sends to ALL receivers, including those held by the sender.
            Handle your own output filtering if you hold both sender and receiver.
        """
        to_clear: set[Sender[T]] = set()
        for sender in copy(self.senders):
            try:
                await sender.send(item)
            except (ClosedResourceError, BrokenResourceError):
                to_clear.add(sender)
        self.senders -= to_clear

    async def publish_bytes(self, data: bytes) -> None:
        """Publish a message from bytes.

        Deserializes the bytes and publishes the resulting message.

        Args:
            data: Serialized message bytes.
        """
        await self.publish(self.topic.deserialize(data))

    def new_sender(self) -> Sender[T]:
        """Create a new sender for this topic.

        Returns:
            Cloned sender for this topic.
        """
        return self._sender.clone()

    async def _send_out(self, item: T) -> None:
        """Send a message to the network layer.

        Args:
            item: Message to send.
        """
        logger.trace(f"TopicRouter {self.topic.topic} sending {item}")
        await self.networking_sender.send(
            (str(self.topic.topic), self.topic.serialize(item))
        )


class Router:
    """Main router managing all topic routers and libp2p networking.

    Coordinates multiple TopicRouter instances and handles libp2p networking
    operations (subscribe, unsubscribe, send, receive). Provides typed
    senders and receivers for each registered topic.

    Attributes:
        topic_routers: Mapping of topic names to TopicRouter instances.
        networking_receiver: Receiver for messages from libp2p network.
        _net: Libp2p networking handle.
        _tg: Task group for background operations.
    """

    @classmethod
    def create(cls, identity: Keypair) -> "Router":
        """Create a new router with the given identity.

        Args:
            identity: Libp2p keypair for node identity.

        Returns:
            New Router instance.
        """
        return cls(handle=NetworkingHandle(identity))

    def __init__(self, handle: NetworkingHandle):
        """Initialize router with networking handle.

        Args:
            handle: Libp2p networking handle.
        """
        self.topic_routers: dict[str, TopicRouter[CamelCaseModel]] = {}
        send, recv = channel[tuple[str, bytes]]()
        self.networking_receiver: Receiver[tuple[str, bytes]] = recv
        self._net: NetworkingHandle = handle
        self._tmp_networking_sender: Sender[tuple[str, bytes]] | None = send
        self._id_count = count()
        self._tg: TaskGroup | None = None

    async def register_topic[T: CamelCaseModel](self, topic: TypedTopic[T]):
        assert self._tg is None, "Attempted to register topic after setup time"
        send = self._tmp_networking_sender
        if send:
            self._tmp_networking_sender = None
        else:
            send = self.networking_receiver.clone_sender()
        router = TopicRouter[T](topic, send)
        self.topic_routers[topic.topic] = cast(TopicRouter[CamelCaseModel], router)
        await self._networking_subscribe(str(topic.topic))

    def sender[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Sender[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts
        assert router is not None
        assert router.topic == topic
        sender = cast(TopicRouter[T], router).new_sender()
        return sender

    def receiver[T: CamelCaseModel](self, topic: TypedTopic[T]) -> Receiver[T]:
        router = self.topic_routers.get(topic.topic, None)
        # There's gotta be a way to do this without THIS many asserts

        assert router is not None
        assert router.topic == topic
        assert router.topic.model_type == topic.model_type

        send, recv = channel[T]()
        router.senders.add(cast(Sender[CamelCaseModel], send))

        return recv

    async def connect_seeds(self, seeds: Iterable[str]) -> None:
        """Dial seed peers to bootstrap beyond mDNS."""
        for seed in seeds:
            addr = self._to_multiaddr(seed)
            logger.info(f"Dialing seed {seed} as {addr}")
            try:
                dialed = await self._net.dial_multiaddr(addr)
                logger.info(f"Dial success={dialed} for seed {seed} as {addr}")
            except Exception as exc:  # pragma: no cover - networking failures surface at runtime
                logger.warning(f"Failed to dial seed {seed} as {addr}: {exc}")

    @staticmethod
    def _to_multiaddr(seed: str) -> str:
        """Convert host:port or multiaddr string to multiaddr."""
        if seed.strip().startswith("/"):
            return seed
        if ":" not in seed:
            raise ValueError(f"Seed must be host:port or multiaddr, got '{seed}'")
        host, port = seed.rsplit(":", 1)
        return f"/ip4/{host}/tcp/{port}"

    async def run(self):
        logger.debug("Starting Router")
        async with create_task_group() as tg:
            self._tg = tg
            for topic in self.topic_routers:
                router = self.topic_routers[topic]
                tg.start_soon(router.run)
            tg.start_soon(self._networking_recv)
            tg.start_soon(self._networking_recv_connection_messages)
            tg.start_soon(self._networking_publish)
            # Router only shuts down if you cancel it.
            await sleep_forever()
        for topic in self.topic_routers:
            await self._networking_unsubscribe(str(topic))

    async def shutdown(self):
        logger.debug("Shutting down Router")
        if not self._tg:
            return
        self._tg.cancel_scope.cancel()

    async def _networking_subscribe(self, topic: str):
        logger.info(f"Subscribing to {topic}")
        await self._net.gossipsub_subscribe(topic)

    async def _networking_unsubscribe(self, topic: str):
        logger.info(f"Unsubscribing from {topic}")
        await self._net.gossipsub_unsubscribe(topic)

    async def _networking_recv(self):
        while True:
            topic, data = await self._net.gossipsub_recv()
            logger.trace(f"Received message on {topic} with payload {data}")
            if topic not in self.topic_routers:
                logger.warning(f"Received message on unknown or inactive topic {topic}")
                continue

            router = self.topic_routers[topic]
            await router.publish_bytes(data)

    async def _networking_recv_connection_messages(self):
        while True:
            update = await self._net.connection_update_recv()
            message = ConnectionMessage.from_update(update)
            logger.trace(
                f"Received message on connection_messages with payload {message}"
            )
            if CONNECTION_MESSAGES.topic in self.topic_routers:
                router = self.topic_routers[CONNECTION_MESSAGES.topic]
                assert router.topic.model_type == ConnectionMessage
                router = cast(TopicRouter[ConnectionMessage], router)
                await router.publish(message)

    async def _networking_publish(self) -> None:
        """Publish messages from networking_receiver to libp2p network.

        Receives messages from topic routers and publishes them to the
        libp2p gossipsub network. Silently handles errors when no peers
        are subscribed or queues are full.
        """
        with self.networking_receiver as networked_items:
            async for topic, data in networked_items:
                try:
                    logger.trace(f"Sending message on {topic} with payload {data}")
                    await self._net.gossipsub_publish(topic, data)
                except (NoPeersSubscribedToTopicError, AllQueuesFullError):
                    pass


def get_node_id_keypair(
    path: str | bytes | PathLike[str] | PathLike[bytes] = EXO_NODE_ID_KEYPAIR,
) -> Keypair:
    """Obtain the libp2p keypair for this node's identity.

    Loads or generates a keypair from the specified path. Uses file locking
    to prevent race conditions when generating the keypair across processes.

    Args:
        path: Path to the keypair file (defaults to EXO_NODE_ID_KEYPAIR).

    Returns:
        Libp2p keypair for this node. Can be used to obtain PeerId.

    Note:
        Uses file locking to ensure atomic keypair generation if it doesn't exist.
    """

    def lock_path(path: str | bytes | PathLike[str] | PathLike[bytes]) -> Path:
        """Get lock file path for a given file path.

        Args:
            path: Original file path.

        Returns:
            Lock file path (original path + ".lock").
        """
        return Path(str(path) + ".lock")

    # operate with cross-process lock to avoid race conditions
    with FileLock(lock_path(path)):
        with open(path, "a+b") as f:  # opens in append-mode => starts at EOF
            # if non-zero EOF, then file exists => use to get node-ID
            if f.tell() != 0:
                f.seek(0)  # go to start & read protobuf-encoded bytes
                protobuf_encoded = f.read()

                try:  # if decoded successfully, save & return
                    return Keypair.from_protobuf_encoding(protobuf_encoded)
                except ValueError as e:  # on runtime error, assume corrupt file
                    logger.warning(f"Encountered error when trying to get keypair: {e}")

        # if no valid credentials, create new ones and persist
        with open(path, "w+b") as f:
            keypair = Keypair.generate_ed25519()
            f.write(keypair.to_protobuf_encoding())
            return keypair

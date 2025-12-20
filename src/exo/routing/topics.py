"""Topic definitions for libp2p message routing.

This module defines typed topics used by the Router for routing messages
across the libp2p network, including publish policies and serialization.
"""

from dataclasses import dataclass
from enum import Enum

from exo.routing.connection_message import ConnectionMessage
from exo.shared.election import ElectionMessage
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.events import (
    ForwarderEvent,
)
from exo.utils.pydantic_ext import CamelCaseModel


class PublishPolicy(str, Enum):
    """Policy for when to publish messages to the network.

    Values:
        Never: Never publish to network (local-only messages).
        Minimal: Only publish when no local receivers exist.
        Always: Always publish to network regardless of local receivers.
    """

    Never = "Never"
    """Never publish to the network - this is a local message"""
    Minimal = "Minimal"
    """Only publish when there is no local receiver for this type of message"""
    Always = "Always"
    """Always publish to the network"""


@dataclass
class TypedTopic[T: CamelCaseModel]:
    """Typed topic configuration for message routing.

    Associates a topic name with a model type and publish policy for
    type-safe message routing through the Router.

    Attributes:
        topic: Topic name (libp2p topic string).
        publish_policy: When to publish messages to the network.
        model_type: Pydantic model type for this topic.
    """

    topic: str
    publish_policy: PublishPolicy
    model_type: type[T]

    @staticmethod
    def serialize(t: T) -> bytes:
        """Serialize a message to bytes.

        Args:
            t: Message instance to serialize.

        Returns:
            JSON-encoded bytes.
        """
        return t.model_dump_json().encode("utf-8")

    def deserialize(self, b: bytes) -> T:
        """Deserialize bytes to a message instance.

        Args:
            b: JSON-encoded bytes.

        Returns:
            Deserialized message instance.
        """
        return self.model_type.model_validate_json(b.decode("utf-8"))


GLOBAL_EVENTS = TypedTopic("global_events", PublishPolicy.Always, ForwarderEvent)
LOCAL_EVENTS = TypedTopic("local_events", PublishPolicy.Always, ForwarderEvent)
COMMANDS = TypedTopic("commands", PublishPolicy.Always, ForwarderCommand)
ELECTION_MESSAGES = TypedTopic(
    "election_messages", PublishPolicy.Always, ElectionMessage
)
CONNECTION_MESSAGES = TypedTopic(
    "connection_messages", PublishPolicy.Never, ConnectionMessage
)

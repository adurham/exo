"""Common type definitions used throughout the EXO system.

This module provides fundamental identifier and session types that are used
across master, worker, routing, and API components.
"""

from typing import Self
from uuid import uuid4

from pydantic import GetCoreSchemaHandler, field_validator
from pydantic_core import core_schema

from exo.utils.pydantic_ext import CamelCaseModel


class Id(str):
    """Base class for identifier types that are string-based UUIDs.

    Provides automatic UUID generation when no value is provided. Subclasses
    should inherit from Id to create distinct identifier types that cannot
    be mixed (e.g., NodeId, CommandId).

    When instantiated without a value, automatically generates a new UUID string.
    When instantiated with a value, uses that value directly.

    Examples:
        >>> node_id = NodeId()  # Generates new UUID
        >>> cmd_id = CommandId("existing-id")  # Uses provided ID
    """

    def __new__(cls, value: str | None = None) -> Self:
        """Create a new Id instance with optional value.

        Args:
            value: Optional string value. If None, generates a new UUID string.

        Returns:
            New Id instance with the provided value or a generated UUID.
        """
        return super().__new__(cls, value or str(uuid4()))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Provide Pydantic core schema for validation.

        Returns:
            String schema since IDs are represented as strings.
        """
        return core_schema.str_schema()


class NodeId(Id):
    """Identifier for a node in the cluster.

    Each device running EXO has a unique NodeId derived from its libp2p keypair.
    Used throughout the system to identify nodes in topology, events, and commands.
    """

    pass


class SessionId(CamelCaseModel):
    """Identifies a cluster session tied to a specific master election.

    Sessions are created when a new master is elected. The session ID includes
    the master node ID and election clock to uniquely identify the leadership
    period. All events and state in a session are tied to this identifier.

    Attributes:
        master_node_id: The NodeId of the master node for this session.
        election_clock: Incrementing clock value from the election that created
            this session. Higher values indicate more recent elections.
    """

    master_node_id: NodeId
    election_clock: int


class CommandId(Id):
    """Identifier for a command issued to the cluster.

    Used to track commands from API requests through to completion. Each command
    gets a unique ID that can be used to match responses and track status.
    """

    pass


class Host(CamelCaseModel):
    """Network host address with IP and port.

    Represents a network endpoint for connections between nodes or services.

    Attributes:
        ip: IP address as a string (IPv4 or IPv6).
        port: Port number in the range 0-65535.

    Examples:
        >>> host = Host(ip="192.168.1.1", port=52415)
        >>> str(host)  # "192.168.1.1:52415"
    """

    ip: str
    port: int

    def __str__(self) -> str:
        """Return string representation as 'ip:port'."""
        return f"{self.ip}:{self.port}"

    @field_validator("port")
    @classmethod
    def check_port(cls, v: int) -> int:
        """Validate port is in valid range.

        Args:
            v: Port number to validate.

        Returns:
            The validated port number.

        Raises:
            ValueError: If port is outside the range 0-65535.
        """
        if not (0 <= v <= 65535):
            raise ValueError("Port must be between 0 and 65535")
        return v

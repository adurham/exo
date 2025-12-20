"""Multiaddr type for network address representation.

Multiaddr is a format for encoding network addresses used by libp2p.
This module provides validation and parsing for multiaddr strings.
"""

import re
from typing import ClassVar

from pydantic import BaseModel, computed_field, field_validator


class Multiaddr(BaseModel):
    """Represents a multiaddr network address.

    Multiaddr format encodes network addresses as strings like:
    - /ip4/127.0.0.1/tcp/4001
    - /ip6/::1/tcp/4001
    - /dns/example.com/tcp/443

    Supports IPv4, IPv6, and DNS addresses with optional TCP port
    and libp2p peer ID.

    Attributes:
        address: Multiaddr string (e.g., "/ip4/127.0.0.1/tcp/4001").
    """

    address: str

    PATTERNS: ClassVar[list[str]] = [
        r"^/ip4/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$",
        r"^/ip6/([0-9a-fA-F:]+)(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$",
        r"^/dns[46]?/([a-zA-Z0-9.-]+)(/tcp/(\d{1,5}))?(/p2p/[A-Za-z0-9]+)?$",
    ]

    @field_validator("address")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate that the address matches multiaddr format.

        Args:
            v: Address string to validate.

        Returns:
            The validated address string.

        Raises:
            ValueError: If address doesn't match any supported multiaddr pattern.
        """
        if not any(re.match(pattern, v) for pattern in cls.PATTERNS):
            raise ValueError(
                f"Invalid multiaddr format: {v}. "
                "Expected format like /ip4/127.0.0.1/tcp/4001 or /dns/example.com/tcp/443"
            )
        return v

    @computed_field
    @property
    def address_type(self) -> str:
        """Get the address type (ip4, ip6, or dns).

        Returns:
            Address type string.

        Raises:
            ValueError: If address format is invalid.
        """
        for pattern in self.PATTERNS:
            if re.match(pattern, self.address):
                return pattern.split("/")[1]
        raise ValueError(f"Invalid multiaddr format: {self.address}")

    @property
    def ipv6_address(self) -> str:
        """Extract IPv6 address from multiaddr.

        Returns:
            IPv6 address string.

        Raises:
            ValueError: If address is not IPv6 format.
        """
        match = re.match(r"^/ip6/([0-9a-fA-F:]+)", self.address)
        if not match:
            raise ValueError(
                f"Invalid multiaddr format: {self.address}. Expected format like /ip6/::1/tcp/4001"
            )
        return match.group(1)

    @property
    def ipv4_address(self) -> str:
        """Extract IPv4 address from multiaddr.

        Returns:
            IPv4 address string.

        Raises:
            ValueError: If address is not IPv4 format.
        """
        match = re.match(r"^/ip4/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", self.address)
        if not match:
            raise ValueError(
                f"Invalid multiaddr format: {self.address}. Expected format like /ip4/127.0.0.1/tcp/4001"
            )
        return match.group(1)

    @computed_field
    @property
    def ip_address(self) -> str:
        """Get IP address (IPv4 or IPv6) from multiaddr.

        Returns:
            IP address string (automatically selects IPv4 or IPv6).

        Raises:
            ValueError: If address doesn't contain an IP address.
        """
        return self.ipv4_address if self.address_type == "ip4" else self.ipv6_address

    @computed_field
    @property
    def port(self) -> int:
        """Extract TCP port from multiaddr.

        Returns:
            Port number.

        Raises:
            ValueError: If address doesn't contain a TCP port.
        """
        match = re.search(r"/tcp/(\d{1,5})", self.address)
        if not match:
            raise ValueError(
                f"Invalid multiaddr format: {self.address}. Expected format like /ip4/127.0.0.1/tcp/4001"
            )
        return int(match.group(1))

    def __str__(self) -> str:
        """Return the multiaddr string representation."""
        return self.address

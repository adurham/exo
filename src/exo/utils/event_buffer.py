"""Event buffering utilities for maintaining ordering.

This module provides buffers that resequence events to ensure they are processed
in order, even if received out of order from the network or other sources.
"""

from loguru import logger


class OrderedBuffer[T]:
    """Buffer that resequences events to ensure ordering is preserved.

    Stores events indexed by sequence number and only releases them in order.
    Events with indices less than the next expected index are discarded (too old).
    Events are released via drain() only when the next expected index is available.

    This buffer does not raise errors if an event is lost - missing events will
    cause drain() to stop returning events until the gap is filled.

    Note:
        This buffer is NOT thread-safe and is designed to be polled from a single
        source at a time.

    Attributes:
        store: Dictionary mapping sequence indices to events.
        next_idx_to_release: Next sequence index expected for release.
    """

    def __init__(self) -> None:
        """Initialize an empty ordered buffer."""
        self.store: dict[int, T] = {}
        self.next_idx_to_release: int = 0

    def ingest(self, idx: int, t: T) -> None:
        """Add an event to the buffer with its sequence index.

        Events with indices less than next_idx_to_release are discarded (too old).
        If the same index is ingested twice with the same event, it's ignored.
        If the same index is ingested twice with different events, raises AssertionError
        (indicating a race condition or duplicate event bug).

        Args:
            idx: Sequence index for the event. Must be >= next_idx_to_release.
            t: Event to buffer.
        """
        logger.trace(f"Ingested event {t}")
        if idx < self.next_idx_to_release:
            return
        if idx in self.store:
            assert self.store[idx] == t, (
                "Received different messages with identical indices, probable race condition"
            )
            return
        self.store[idx] = t

    def drain(self) -> list[T]:
        """Drain all available events in order.

        Returns all events starting from next_idx_to_release that are available
        consecutively. Stops at the first gap.

        Returns:
            List of events in sequence order. Empty list if next expected event
            is not available.
        """
        ret: list[T] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append(event)
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret

    def drain_indexed(self) -> list[tuple[int, T]]:
        """Drain all available events with their indices.

        Same as drain() but returns tuples of (index, event) instead of just events.

        Returns:
            List of (index, event) tuples in sequence order. Empty list if next
            expected event is not available.
        """
        ret: list[tuple[int, T]] = []
        while self.next_idx_to_release in self.store:
            idx = self.next_idx_to_release
            event = self.store.pop(idx)
            ret.append((idx, event))
            self.next_idx_to_release += 1
        logger.trace(f"Releasing event {ret}")
        return ret


class MultiSourceBuffer[SourceId, T]:
    """Buffer that resequences events from multiple sources.

    Maintains separate OrderedBuffer instances for each source, allowing events
    from different sources to be tracked independently. Useful when events come
    from multiple nodes and need per-source ordering.

    Attributes:
        stores: Dictionary mapping source IDs to their OrderedBuffer instances.
    """

    def __init__(self) -> None:
        """Initialize an empty multi-source buffer."""
        self.stores: dict[SourceId, OrderedBuffer[T]] = {}

    def ingest(self, idx: int, t: T, source: SourceId) -> None:
        """Add an event to the buffer for a specific source.

        Creates a new OrderedBuffer for the source if it doesn't exist, then
        ingests the event into that buffer.

        Args:
            idx: Sequence index for the event (source-specific).
            t: Event to buffer.
            source: Source identifier for this event.
        """
        if source not in self.stores:
            self.stores[source] = OrderedBuffer()
        buffer = self.stores[source]
        buffer.ingest(idx, t)

    def drain(self) -> list[T]:
        """Drain available events from all sources.

        Drains events from all source buffers and returns them combined. Events
        from different sources may be interleaved, but events from each source
        maintain their ordering.

        Returns:
            List of events from all sources, with per-source ordering preserved.
        """
        ret: list[T] = []
        for store in self.stores.values():
            ret.extend(store.drain())
        return ret

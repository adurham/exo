"""Channel utilities for asynchronous message passing.

This module provides typed channels for communication between components.
Supports both in-process channels (using anyio) and inter-process channels
(using multiprocessing). Channels are typed and support cloning for fan-out
patterns.
"""

import multiprocessing as mp
from dataclasses import dataclass, field
from math import inf
from multiprocessing.synchronize import Event
from queue import Empty, Full
from types import TracebackType
from typing import Self

from anyio import (
    CapacityLimiter,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    to_thread,
)
from anyio.streams.memory import (
    MemoryObjectReceiveStream as AnyioReceiver,
)
from anyio.streams.memory import (
    MemoryObjectSendStream as AnyioSender,
)
from anyio.streams.memory import (
    MemoryObjectStreamState as AnyioState,
)


class Sender[T](AnyioSender[T]):
    """Typed sender for asynchronous channels.

    Extends anyio's MemoryObjectSendStream with cloning capabilities for
    creating multiple receivers from a single sender (fan-out pattern).
    """

    def clone(self) -> "Sender[T]":
        """Create a new sender sharing the same underlying channel state.

        Returns:
            New Sender instance that sends to the same channel.

        Raises:
            ClosedResourceError: If the channel is already closed.
        """
        if self._closed:
            raise ClosedResourceError
        return Sender(_state=self._state)

    def clone_receiver(self) -> "Receiver[T]":
        """Create a receiver sharing this sender's channel state.

        Useful for creating a receiver when you only have a sender reference,
        enabling fan-out patterns where one sender has multiple receivers.

        Returns:
            New Receiver instance that receives from the same channel.

        Raises:
            ClosedResourceError: If the channel is already closed.
        """
        if self._closed:
            raise ClosedResourceError
        return Receiver(_state=self._state)


class Receiver[T](AnyioReceiver[T]):
    """Typed receiver for asynchronous channels.

    Extends anyio's MemoryObjectReceiveStream with cloning and convenience
    methods for collecting multiple items.
    """

    def clone(self) -> "Receiver[T]":
        """Create a new receiver sharing the same underlying channel state.

        Returns:
            New Receiver instance that receives from the same channel.

        Raises:
            ClosedResourceError: If the channel is already closed.
        """
        if self._closed:
            raise ClosedResourceError
        return Receiver(_state=self._state)

    def clone_sender(self) -> Sender[T]:
        """Create a sender sharing this receiver's channel state.

        Useful for creating a sender when you only have a receiver reference,
        enabling communication in both directions from a single reference.

        Returns:
            New Sender instance that sends to the same channel.

        Raises:
            ClosedResourceError: If the channel is already closed.
        """
        if self._closed:
            raise ClosedResourceError
        return Sender(_state=self._state)

    def collect(self) -> list[T]:
        """Collect all currently available items without blocking.

        Receives all items that are immediately available in the channel buffer
        without waiting for new items.

        Returns:
            List of all immediately available items. May be empty if channel
            buffer is empty.
        """
        out: list[T] = []
        while True:
            try:
                item = self.receive_nowait()
                out.append(item)
            except WouldBlock:
                break
        return out

    async def receive_at_least(self, n: int) -> list[T]:
        """Receive at least n items, waiting if necessary.

        Receives items until at least n are available. Will block waiting for
        items if the channel buffer doesn't have enough immediately available.

        Args:
            n: Minimum number of items to receive.

        Returns:
            List containing at least n items (may contain more if additional
            items were available).
        """
        out: list[T] = []
        out.append(await self.receive())
        out.extend(self.collect())
        while len(out) < n:
            out.append(await self.receive())
            out.extend(self.collect())
        return out

    def __enter__(self) -> Self:
        return self


class _MpEndOfStream:
    pass


class MpState[T]:
    def __init__(self, max_buffer_size: float):
        if max_buffer_size == inf:
            max_buffer_size = 0
        assert isinstance(max_buffer_size, int), (
            "State should only ever be constructed with an integer or math.inf size."
        )

        self.max_buffer_size: float = max_buffer_size
        self.buffer: mp.Queue[T | _MpEndOfStream] = mp.Queue(max_buffer_size)
        self.closed: Event = mp.Event()

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


@dataclass(eq=False)
class MpSender[T]:
    """Interprocess sender for multiprocessing channels.

    Provides blocking and non-blocking send operations for communicating between
    processes using multiprocessing.Queue. Mimics the anyio Sender interface
    but uses synchronous operations that can cross process boundaries.

    Note:
        Clone methods are not implemented for interprocess channels to keep
        the implementation simple.

    Attributes:
        _state: Internal state containing the multiprocessing queue and close flag.
    """

    _state: MpState[T] = field()

    def send_nowait(self, item: T) -> None:
        """Send an item without blocking.

        Args:
            item: Item to send.

        Raises:
            ClosedResourceError: If the channel is closed.
            WouldBlock: If the channel buffer is full.
        """
        if self._state.closed.is_set():
            raise ClosedResourceError
        try:
            self._state.buffer.put(item, block=False)
        except Full:
            raise WouldBlock from None
        except ValueError as e:
            print("Unreachable code path - let me know!")
            raise ClosedResourceError from e

    def send(self, item: T) -> None:
        """Send an item, blocking if the buffer is full.

        Args:
            item: Item to send.

        Raises:
            ClosedResourceError: If the channel is closed.
        """
        if self._state.closed.is_set():
            raise ClosedResourceError
        try:
            self.send_nowait(item)
        except WouldBlock:
            self._state.buffer.put(item, block=True)

    async def send_async(self, item: T) -> None:
        """Send an item asynchronously using a thread pool.

        Args:
            item: Item to send.

        Note:
            Cancellation may not work well with this method.
        """
        await to_thread.run_sync(self.send, item, limiter=CapacityLimiter(1))

    def close(self) -> None:
        """Close the sender and signal end of stream to receivers."""
        if not self._state.closed.is_set():
            self._state.closed.set()
        self._state.buffer.put(_MpEndOfStream())
        self._state.buffer.close()

    def join(self) -> None:
        """Wait for all queued messages to be processed.

        Blocks until all messages in the queue have been consumed. Channel must
        be closed before calling join().

        Raises:
            AssertionError: If channel is not closed.
        """
        assert self._state.closed.is_set(), (
            "Mp channels must be closed before being joined"
        )
        self._state.buffer.join_thread()

    # == context manager support ==
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


@dataclass(eq=False)
class MpReceiver[T]:
    """Interprocess receiver for multiprocessing channels.

    Provides blocking and non-blocking receive operations for communicating
    between processes using multiprocessing.Queue. Supports iteration and
    async iteration. Mimics the anyio Receiver interface but uses synchronous
    operations that can cross process boundaries.

    Note:
        Clone methods are not implemented for interprocess channels to keep
        the implementation simple.

    Attributes:
        _state: Internal state containing the multiprocessing queue and close flag.
    """

    _state: MpState[T] = field()

    def receive_nowait(self) -> T:
        """Receive an item without blocking.

        Returns:
            The received item.

        Raises:
            ClosedResourceError: If the channel is closed.
            WouldBlock: If no items are available.
            EndOfStream: If the sender has closed and no items remain.
        """
        if self._state.closed.is_set():
            raise ClosedResourceError

        try:
            item = self._state.buffer.get(block=False)
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream
            return item
        except Empty:
            raise WouldBlock from None
        except ValueError as e:
            print("Unreachable code path - let me know!")
            raise ClosedResourceError from e

    def receive(self) -> T:
        """Receive an item, blocking if none are available.

        Returns:
            The received item.

        Raises:
            ClosedResourceError: If the channel is closed.
            EndOfStream: If the sender has closed and no items remain.
        """
        try:
            return self.receive_nowait()
        except WouldBlock:
            item = self._state.buffer.get()
            if isinstance(item, _MpEndOfStream):
                self.close()
                raise EndOfStream from None
            return item

    async def receive_async(self) -> T:
        """Receive an item asynchronously using a thread pool.

        Returns:
            The received item.

        Note:
            Cancellation may not work well with this method.
        """
        return await to_thread.run_sync(self.receive, limiter=CapacityLimiter(1))

    def close(self) -> None:
        """Close the receiver."""
        if not self._state.closed.is_set():
            self._state.closed.set()
        self._state.buffer.close()

    def join(self) -> None:
        """Block until all enqueued messages are drained.

        Blocks until all messages in the queue have been consumed. Channel must
        be closed before calling join().

        Raises:
            AssertionError: If channel is not closed.
        """
        assert self._state.closed.is_set(), (
            "Mp channels must be closed before being joined"
        )
        self._state.buffer.join_thread()

    # == iterator support ==
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> T:
        try:
            return self.receive()
        except EndOfStream:
            raise StopIteration from None

    # == async iterator support ==
    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.receive_async()
        except EndOfStream:
            raise StopAsyncIteration from None

    # == context manager support ==
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def collect(self) -> list[T]:
        """Collect all currently available items from this receiver"""
        out: list[T] = []
        while True:
            try:
                item = self.receive_nowait()
                out.append(item)
            except WouldBlock:
                break
        return out

    def receive_at_least(self, n: int) -> list[T]:
        out: list[T] = []
        out.append(self.receive())
        out.extend(self.collect())
        while len(out) < n:
            out.append(self.receive())
            out.extend(self.collect())
        return out

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("__orig_class__", None)
        return d


class channel[T]:  # noqa: N801
    """Create a pair of asynchronous channels for same-process communication.

    Creates a typed sender/receiver pair using anyio's memory streams. Suitable
    for communication between tasks in the same process.

    Args:
        max_buffer_size: Maximum buffer size. Use math.inf for unbounded buffer.
            Must be an integer or math.inf.

    Returns:
        Tuple of (Sender, Receiver) instances sharing the same channel.

    Raises:
        ValueError: If max_buffer_size is not an integer or math.inf.
    """

    def __new__(cls, max_buffer_size: float = inf) -> tuple[Sender[T], Receiver[T]]:
        if max_buffer_size != inf and not isinstance(max_buffer_size, int):
            raise ValueError("max_buffer_size must be either an integer or math.inf")
        state = AnyioState[T](max_buffer_size)
        return Sender(_state=state), Receiver(_state=state)


class mp_channel[T]:  # noqa: N801
    """Create a pair of synchronous channels for interprocess communication.

    Creates a typed sender/receiver pair using multiprocessing.Queue. Suitable
    for communication between processes. Uses synchronous operations that work
    across process boundaries.

    Args:
        max_buffer_size: Maximum buffer size. Use math.inf for unbounded buffer.
            Must be an integer or math.inf. Zero-sized buffers are not supported.

    Returns:
        Tuple of (MpSender, MpReceiver) instances sharing the same channel.

    Raises:
        ValueError: If max_buffer_size is not an integer or math.inf, or is 0.
    """

    def __new__(cls, max_buffer_size: float = inf) -> tuple[MpSender[T], MpReceiver[T]]:
        if (
            max_buffer_size == 0
            or max_buffer_size != inf
            and not isinstance(max_buffer_size, int)
        ):
            raise ValueError(
                "max_buffer_size must be either an integer or math.inf. 0-sized buffers are not supported by multiprocessing"
            )
        state = MpState[T](max_buffer_size)
        return MpSender(_state=state), MpReceiver(_state=state)

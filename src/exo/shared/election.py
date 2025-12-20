"""Master election system for EXO cluster.

This module implements a distributed leader election algorithm that selects a
master node to coordinate cluster state. Elections use multiple criteria for
determining the winner: election clock, seniority, commands seen, and node ID.

Every node participates in elections to ensure a master exists even if no nodes
are marked as master candidates.
"""

from typing import Self

import anyio
from anyio import (
    CancelScope,
    Event,
    create_task_group,
    get_cancelled_exc_class,
)
from anyio.abc import TaskGroup
from loguru import logger

from exo.routing.connection_message import ConnectionMessage
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import Receiver, Sender
from exo.utils.pydantic_ext import CamelCaseModel

DEFAULT_ELECTION_TIMEOUT = 3.0


class ElectionMessage(CamelCaseModel):
    """Message sent during master election campaigns.

    Each node broadcasts its election status in these messages. The winner is
    determined by comparing messages using multiple criteria in order:
    1. Clock: Higher clock value wins (indicates more recent election round)
    2. Seniority: Higher seniority wins (indicates more wins/experience)
    3. Commands seen: Higher count wins (indicates more awareness of cluster state)
    4. Node ID: Lexicographically smaller ID breaks ties (deterministic)

    Attributes:
        clock: Election clock value for this round. Increments on topology changes.
        seniority: Node's seniority level. Increases when node wins elections.
        proposed_session: Session ID the node proposes for the cluster.
        commands_seen: Number of commands this node has observed. Tracks state
            awareness relative to other nodes.
    """

    clock: int
    seniority: int
    proposed_session: SessionId
    commands_seen: int

    def __lt__(self, other: Self) -> bool:
        """Compare election messages to determine ordering.

        Comparison uses multiple criteria in order:
        1. Clock (higher wins)
        2. Seniority (higher wins)
        3. Commands seen (higher wins)
        4. Node ID (lexicographically smaller wins)

        Args:
            other: Election message to compare against.

        Returns:
            True if this message should lose to other, False otherwise.
        """
        if self.clock != other.clock:
            return self.clock < other.clock
        if self.seniority != other.seniority:
            return self.seniority < other.seniority
        elif self.commands_seen != other.commands_seen:
            return self.commands_seen < other.commands_seen
        else:
            return (
                self.proposed_session.master_node_id
                < other.proposed_session.master_node_id
            )


class ElectionResult(CamelCaseModel):
    """Result of a master election round.

    Broadcast to all nodes when an election completes, indicating the winner
    and whether this represents a leadership change.

    Attributes:
        session_id: Session ID for the elected master's session.
        won_clock: Election clock value at which the election was won.
        is_new_master: True if this represents a new master (session changed),
            False if the existing master was re-elected.
    """

    session_id: SessionId
    won_clock: int
    is_new_master: bool


class Election:
    """Manages master election participation for a node.

    Each node runs an Election instance that participates in distributed leader
    elections. Elections are triggered by:
    - Topology changes (new connections/disconnections)
    - Receipt of election messages from other nodes with higher clocks
    - Initial startup

    The election algorithm:
    1. Nodes broadcast their status (clock, seniority, commands_seen, proposed session)
    2. Campaign runs for a timeout period to collect candidate messages
    3. Winner is selected using ElectionMessage comparison (max wins)
    4. Winner's session becomes the cluster session
    5. Winning node may increase its seniority based on campaign size

    Attributes:
        node_id: This node's identifier.
        seniority: Current seniority level (increases with wins). Set to -1 for
            non-candidates (nodes that can become master only if no candidates exist).
        clock: Current election clock value. Increments on topology changes.
        commands_seen: Count of commands observed by this node.
        current_session: Session ID for the current cluster session.
    """

    def __init__(
        self,
        node_id: NodeId,
        *,
        election_message_receiver: Receiver[ElectionMessage],
        election_message_sender: Sender[ElectionMessage],
        election_result_sender: Sender[ElectionResult],
        connection_message_receiver: Receiver[ConnectionMessage],
        command_receiver: Receiver[ForwarderCommand],
        is_candidate: bool = True,
        seniority: int = 0,
    ):
        """Initialize election service.

        Args:
            node_id: Unique identifier for this node.
            election_message_receiver: Channel to receive election messages from peers.
            election_message_sender: Channel to send election messages to peers.
            election_result_sender: Channel to send election results.
            connection_message_receiver: Channel to receive topology change notifications.
            command_receiver: Channel to receive commands (used to track commands_seen).
            is_candidate: If True, node is a master candidate. If False, node can
                still become master if no candidates exist, but will lose to any
                candidate. Default is True.
            seniority: Initial seniority level. Higher values increase election
                priority. Default is 0.
        """
        self.seniority = seniority if is_candidate else -1
        self.clock = 0
        self.node_id = node_id
        self.commands_seen = 0
        self.current_session: SessionId = SessionId(
            master_node_id=node_id, election_clock=0
        )

        # Senders/Receivers
        self._em_sender = election_message_sender
        self._em_receiver = election_message_receiver
        self._er_sender = election_result_sender
        self._cm_receiver = connection_message_receiver
        self._co_receiver = command_receiver

        # Campaign state
        self._candidates: list[ElectionMessage] = []
        self._campaign_cancel_scope: CancelScope | None = None
        self._campaign_done: Event | None = None
        self._tg: TaskGroup | None = None

    async def run(self) -> None:
        """Run the election service.

        Starts background tasks to:
        - Receive and process election messages from peers
        - Monitor connection changes to trigger new elections
        - Track commands to update commands_seen counter

        Also runs an initial election campaign with zero timeout to establish
        the initial session. This election immediately resolves since the node
        starts as its own master.

        The method blocks until shutdown is requested via shutdown().
        """
        logger.info("Starting Election")
        async with create_task_group() as tg:
            self._tg = tg
            tg.start_soon(self._election_receiver)
            tg.start_soon(self._connection_receiver)
            tg.start_soon(self._command_counter)

            candidates: list[ElectionMessage] = []
            logger.debug("Starting initial campaign")
            self._candidates = candidates
            await self._campaign(candidates, campaign_timeout=0.0)
            logger.debug("Initial campaign finished")

        # Cancel and wait for the last election to end
        if self._campaign_cancel_scope is not None:
            logger.debug("Cancelling campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done is not None:
            logger.debug("Waiting for campaign to finish")
            await self._campaign_done.wait()
        logger.debug("Campaign cancelled and finished")
        logger.info("Election finished")

    async def elect(self, em: ElectionMessage) -> None:
        """Elect a master and broadcast the result.

        Updates the current session to match the elected master's proposed session
        and sends an ElectionResult to notify all nodes of the election outcome.

        Args:
            em: Election message from the winning candidate.
        """
        logger.debug(f"Electing: {em}")
        is_new_master = em.proposed_session != self.current_session
        self.current_session = em.proposed_session
        logger.debug(f"Current session: {self.current_session}")
        await self._er_sender.send(
            ElectionResult(
                won_clock=em.clock,
                session_id=em.proposed_session,
                is_new_master=is_new_master,
            )
        )

    async def shutdown(self) -> None:
        """Shutdown the election service gracefully.

        Cancels the task group and waits for any ongoing campaign to complete.
        Safe to call multiple times.
        """
        if not self._tg:
            logger.warning(
                "Attempted to shutdown election service that was not running"
            )
            return
        self._tg.cancel_scope.cancel()

    async def _election_receiver(self) -> None:
        """Process election messages from other nodes.

        Handles incoming election messages by:
        - Dropping messages from self (router also filters, but double-check)
        - Starting new campaigns when higher clock values are received
        - Adding candidates to the current campaign if clock matches
        - Ignoring messages from older election rounds
        """
        with self._em_receiver as election_messages:
            async for message in election_messages:
                logger.debug(f"Election message received: {message}")
                if message.proposed_session.master_node_id == self.node_id:
                    logger.debug("Dropping message from ourselves")
                    continue
                if message.clock > self.clock:
                    self.clock = message.clock
                    logger.debug(f"New clock: {self.clock}")
                    assert self._tg is not None
                    logger.debug("Starting new campaign")
                    candidates: list[ElectionMessage] = [message]
                    logger.debug(f"Candidates: {candidates}")
                    logger.debug(f"Current candidates: {self._candidates}")
                    self._candidates = candidates
                    logger.debug(f"New candidates: {self._candidates}")
                    logger.debug("Starting new campaign")
                    self._tg.start_soon(
                        self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                    )
                    logger.debug("Campaign started")
                    continue
                # Dismiss old messages
                if message.clock < self.clock:
                    logger.debug(f"Dropping old message: {message}")
                    continue
                logger.debug(f"Election added candidate {message}")
                # Now we are processing this rounds messages - including the message that triggered this round.
                self._candidates.append(message)

    async def _connection_receiver(self) -> None:
        """Handle topology change notifications.

        When connection changes occur (nodes join/leave), triggers a new election
        by incrementing the clock and starting a campaign. Uses a short delay
        after receiving the first message to batch multiple connection events.

        Note:
            Connection messages trigger elections to ensure the master reflects
            the current cluster topology.
        """
        with self._cm_receiver as connection_messages:
            async for first in connection_messages:
                await anyio.sleep(0.2)
                rest = connection_messages.collect()

                logger.debug(
                    f"Connection messages received: {first} followed by {rest}"
                )
                logger.debug(f"Current clock: {self.clock}")
                self.clock += 1
                logger.debug(f"New clock: {self.clock}")
                assert self._tg is not None
                candidates: list[ElectionMessage] = []
                self._candidates = candidates
                logger.debug("Starting new campaign")
                self._tg.start_soon(
                    self._campaign, candidates, DEFAULT_ELECTION_TIMEOUT
                )
                logger.debug("Campaign started")
                logger.debug("Connection message added")

    async def _command_counter(self) -> None:
        """Track commands observed by this node.

        Increments commands_seen for each command received. This value is included
        in election messages and helps determine which node has the most current
        view of cluster state.
        """
        with self._co_receiver as commands:
            async for _command in commands:
                self.commands_seen += 1

    async def _campaign(
        self, candidates: list[ElectionMessage], campaign_timeout: float
    ) -> None:
        """Run an election campaign.

        Executes a single election round:
        1. Cancels any previous campaign for the same clock
        2. Broadcasts this node's election status
        3. Waits for campaign_timeout seconds to collect candidate messages
        4. Re-broadcasts status before selecting winner
        5. Selects winner (max candidate) and calls elect()

        If this node wins and is a candidate, may increase seniority to
        max(current_seniority, number_of_candidates).

        Args:
            candidates: Initial list of candidate messages (may be empty).
            campaign_timeout: Seconds to wait for candidate messages before
                selecting winner. Use 0.0 for immediate resolution.
        """
        clock = self.clock

        if self._campaign_cancel_scope:
            logger.info("Cancelling other campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done:
            logger.info("Waiting for other campaign to finish")
            await self._campaign_done.wait()

        done = Event()
        self._campaign_done = done
        scope = CancelScope()
        self._campaign_cancel_scope = scope

        try:
            with scope:
                logger.debug(f"Election {clock} started")

                status = self._election_status(clock)
                candidates.append(status)
                await self._em_sender.send(status)

                logger.debug(f"Sleeping for {campaign_timeout} seconds")
                await anyio.sleep(campaign_timeout)
                # minor hack - rebroadcast status in case anyone has missed it.
                await self._em_sender.send(status)
                logger.debug("Woke up from sleep")
                # add an anyio checkpoint - anyio.lowlevel.chekpoint() or checkpoint_if_cancelled() is preferred, but wasn't typechecking last I checked
                await anyio.sleep(0)

                # Election finished!
                elected = max(candidates)
                logger.debug(f"Election queue {candidates}")
                logger.debug(f"Elected: {elected}")
                if (
                    self.node_id == elected.proposed_session.master_node_id
                    and self.seniority >= 0
                ):
                    logger.debug(
                        f"Node is a candidate and seniority is {self.seniority}"
                    )
                    self.seniority = max(self.seniority, len(candidates))
                    logger.debug(f"New seniority: {self.seniority}")
                else:
                    logger.debug(
                        f"Node is not a candidate or seniority is not {self.seniority}"
                    )
                logger.debug(
                    f"Election finished, new SessionId({elected.proposed_session}) with queue {candidates}"
                )
                logger.debug("Sending election result")
                await self.elect(elected)
                logger.debug("Election result sent")
        except get_cancelled_exc_class():
            logger.debug(f"Election {clock} cancelled")
        finally:
            logger.debug(f"Election {clock} finally")
            if self._campaign_cancel_scope is scope:
                self._campaign_cancel_scope = None
            logger.debug("Setting done event")
            done.set()
            logger.debug("Done event set")

    def _election_status(self, clock: int | None = None) -> ElectionMessage:
        """Generate this node's election status message.

        Creates an ElectionMessage representing this node's current state. If
        this node is currently the master, proposes continuing the current session.
        Otherwise, proposes a new session with this node as master.

        Args:
            clock: Clock value to use. If None, uses current clock. Default is None.

        Returns:
            Election message representing this node's election status.
        """
        c = self.clock if clock is None else clock
        return ElectionMessage(
            proposed_session=(
                self.current_session
                if self.current_session.master_node_id == self.node_id
                else SessionId(master_node_id=self.node_id, election_clock=c)
            ),
            clock=c,
            seniority=self.seniority,
            commands_seen=self.commands_seen,
        )

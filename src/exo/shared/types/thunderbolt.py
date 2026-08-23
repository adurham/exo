import anyio
from pydantic import BaseModel, Field

from exo.utils.pydantic_ext import FrozenModel


class ThunderboltConnection(FrozenModel):
    source_uuid: str
    sink_uuid: str


class ThunderboltIdentifier(FrozenModel):
    rdma_interface: str
    domain_uuid: str
    link_speed: str = ""


## Intentionally minimal, only collecting data we care about - there's a lot more


class _ReceptacleTag(BaseModel, extra="ignore"):
    receptacle_id_key: str | None = None
    current_speed_key: str | None = None


class _ConnectivityItem(BaseModel, extra="ignore"):
    domain_uuid_key: str | None = None


class ThunderboltConnectivityData(BaseModel, extra="ignore"):
    domain_uuid_key: str | None = None
    items: list[_ConnectivityItem] | None = Field(None, alias="_items")
    receptacle_1_tag: _ReceptacleTag | None = None

    def ident(self, ifaces: dict[str, str]) -> ThunderboltIdentifier | None:
        if (
            self.domain_uuid_key is None
            or self.receptacle_1_tag is None
            or self.receptacle_1_tag.receptacle_id_key is None
        ):
            return
        tag = f"Thunderbolt {self.receptacle_1_tag.receptacle_id_key}"
        if tag not in ifaces:
            # Transient: `networksetup -listallhardwareports` and
            # `system_profiler SPThunderboltDataType` are two independent
            # subprocess reads on the macOS SystemConfiguration daemon that can
            # (very briefly) disagree during a Thunderbolt subsystem reconfigure
            # — post-runner-SIGKILL, sleep/wake, cable event. Skip this ident;
            # the caller filters `None`, and the next InfoGatherer tick retries.
            # Persistent misses surface via the caller's cycle-skip warning
            # (see info_gatherer._monitor_system_profiler_thunderbolt_data).
            return None
        iface = f"rdma_{ifaces[tag]}"
        return ThunderboltIdentifier(
            rdma_interface=iface,
            domain_uuid=self.domain_uuid_key,
            link_speed=self.receptacle_1_tag.current_speed_key or "",
        )

    def conn(self) -> ThunderboltConnection | None:
        if self.domain_uuid_key is None or self.items is None:
            return

        sink_key = next(
            (
                item.domain_uuid_key
                for item in self.items
                if item.domain_uuid_key is not None
            ),
            None,
        )
        if sink_key is None:
            return None

        return ThunderboltConnection(
            source_uuid=self.domain_uuid_key, sink_uuid=sink_key
        )


class ThunderboltConnectivity(BaseModel, extra="ignore"):
    SPThunderboltDataType: list[ThunderboltConnectivityData] = []

    @classmethod
    async def gather(cls) -> list[ThunderboltConnectivityData] | None:
        proc = await anyio.run_process(
            ["system_profiler", "SPThunderboltDataType", "-json"], check=False
        )
        if proc.returncode != 0:
            return None
        # Saving you from PascalCase while avoiding too much pydantic
        return ThunderboltConnectivity.model_validate_json(
            proc.stdout
        ).SPThunderboltDataType

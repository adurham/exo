import sys

import anyio
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from exo.shared.types.thunderbolt import (
    ThunderboltConnectivity,
    ThunderboltConnectivityData,
)
from exo.utils.info_gatherer.info_gatherer import (
    _gather_iface_map,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def loguru_caplog(caplog: LogCaptureFixture):
    """Wire loguru sink into pytest's caplog.

    loguru does not use the stdlib `logging` root, so caplog is blind by
    default. Same pattern as `src/exo/shared/tests/conftest.py`.
    """
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Fixture: a minimal ThunderboltConnectivityData that mirrors the shape of
# m4-1_spthunderbolt.json (post-incident real capture). Built via
# `model_validate` from a raw dict — same code path as
# `ThunderboltConnectivity.model_validate_json` in production — so tests are
# self-contained and don't read /tmp.
# ---------------------------------------------------------------------------


def _make_datum(receptacle_id_key: str = "4") -> ThunderboltConnectivityData:
    """Build a ThunderboltConnectivityData with a `receptacle_1_tag`.

    Matches the shape of an unconnected TB4 bus entry from
    `system_profiler SPThunderboltDataType -json` on M4 Max.
    """
    return ThunderboltConnectivityData.model_validate(
        {
            "domain_uuid_key": "5217372E-1C76-42C4-B67E-EF48936FA3F8",
            "receptacle_1_tag": {
                "current_speed_key": "Up to 120 Gb/s",
                "link_status_key": "0x100",
                "receptacle_id_key": receptacle_id_key,
                "receptacle_status_key": "receptacle_no_devices_connected",
            },
        }
    )


# ---------------------------------------------------------------------------
# ident() unit tests: gracefully returns None on missing tag; correct value
# on present tag. These are the 2026-08-22 hardening regression tests.
# ---------------------------------------------------------------------------


def test_ident_returns_none_when_tag_missing_from_empty_ifaces() -> None:
    """`iface_map == {}` is the exact incident condition. Must not raise."""
    datum = _make_datum(receptacle_id_key="4")
    assert datum.ident({}) is None


def test_ident_returns_none_when_tag_missing_from_partial_ifaces() -> None:
    """Partially-populated map without the specific Thunderbolt-N tag."""
    datum = _make_datum(receptacle_id_key="4")
    assert datum.ident({"Thunderbolt Bridge": "bridge0", "Wi-Fi": "en0"}) is None


def test_ident_returns_identifier_when_tag_present() -> None:
    """Happy path: tag present in ifaces yields a fully-populated identifier."""
    datum = _make_datum(receptacle_id_key="4")
    ident = datum.ident({"Thunderbolt 4": "en5"})
    assert ident is not None
    assert ident.rdma_interface == "rdma_en5"
    assert ident.domain_uuid == "5217372E-1C76-42C4-B67E-EF48936FA3F8"
    assert ident.link_speed == "Up to 120 Gb/s"


def test_ident_returns_none_when_receptacle_tag_absent() -> None:
    """Pre-existing None-guard branch must still work (belt-and-braces)."""
    datum = ThunderboltConnectivityData.model_validate(
        {"domain_uuid_key": "5217372E-1C76-42C4-B67E-EF48936FA3F8"}
    )
    assert datum.ident({"Thunderbolt 4": "en5"}) is None


# ---------------------------------------------------------------------------
# Empty-iface-map cycle-skip: verify the monitor logs the concise WARNING
# and does NOT proceed to gather ThunderboltConnectivity / build idents.
# We drive one loop iteration under a cancel scope.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_monitor_skips_cycle_when_iface_map_empty(
    loguru_caplog: LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `_gather_iface_map` returns `{}`, the monitor must:
    * log the concise "networksetup returned no Thunderbolt ports" WARNING
    * NOT call `ThunderboltConnectivity.gather` (avoids building idents from
      an empty map, which is what triggered the 4x traceback spam in the
      2026-08-22 incident).
    """
    from exo.utils.channels import channel
    from exo.utils.info_gatherer import info_gatherer as ig_mod

    tb_gather_calls: list[int] = []

    async def _empty_map() -> dict[str, str]:
        return {}

    async def _spy_gather() -> list[ThunderboltConnectivityData] | None:
        tb_gather_calls.append(1)
        return None

    monkeypatch.setattr(ig_mod, "_gather_iface_map", _empty_map)
    monkeypatch.setattr(
        ig_mod.ThunderboltConnectivity, "gather", staticmethod(_spy_gather)
    )

    send, recv = channel[ig_mod.GatheredInfo]()
    try:
        gatherer = ig_mod.InfoGatherer(info_sender=send)
        # `_monitor_...` is `while True`; bound one iteration with a cancel scope.
        # sleep-interval large so we exit the scope during the sleep, cleanly.
        with anyio.move_on_after(1.5):
            await gatherer._monitor_system_profiler_thunderbolt_data(  # pyright: ignore[reportPrivateUsage]
                system_profiler_interval=60.0
            )
    finally:
        send.close()
        recv.close()

    assert tb_gather_calls == [], (
        "ThunderboltConnectivity.gather must NOT be called when iface_map is empty"
    )
    assert any(
        "networksetup returned no Thunderbolt ports" in msg
        for msg in loguru_caplog.messages
    ), f"expected concise WARNING; got: {loguru_caplog.messages!r}"


# ---------------------------------------------------------------------------
# Existing integration test on macOS: unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin", reason="Thunderbolt info can only be gathered on macos"
)
async def test_tb_parsing() -> None:
    data = await ThunderboltConnectivity.gather()
    ifaces = await _gather_iface_map()
    assert ifaces
    assert data
    for datum in data:
        _ = datum.ident(ifaces)
        _ = datum.conn()

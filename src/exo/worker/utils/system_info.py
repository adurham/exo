import socket
import sys
import subprocess
from functools import lru_cache
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import NetworkInterfaceInfo


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    # TODO: better non mac support
    if sys.platform != "darwin":  # 'darwin' is the platform name for macOS
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


@lru_cache(maxsize=1)
def get_macos_interface_map() -> dict[str, str]:
    """
    Parses 'networksetup -listallhardwareports' to map device names (en0) to types (Ethernet).
    Cached to avoid repeated subprocess calls.
    """
    interface_map = {}
    try:
        # networksetup is available on macOS
        output = subprocess.check_output(
            ["networksetup", "-listallhardwareports"], text=True
        )

        # Output format:
        # Hardware Port: Ethernet
        # Device: en0
        # Ethernet Address: ...
        #
        # Hardware Port: Wi-Fi
        # Device: en1
        # ...

        current_port = None
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("Hardware Port:"):
                current_port = line.split(": ")[1]
            elif line.startswith("Device:") and current_port:
                device = line.split(": ")[1]

                # Map hardware port names to our types
                if "Wi-Fi" in current_port or "AirPort" in current_port:
                    interface_map[device] = "WiFi"
                elif "Ethernet" in current_port:
                    interface_map[device] = "Ethernet"
                elif "Thunderbolt" in current_port:
                    interface_map[device] = "Thunderbolt"
                else:
                    interface_map[device] = "Other"

                current_port = None

    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return interface_map


def get_interface_priority_and_type_sync(ifname: str) -> tuple[int, str]:
    # macOS: try to use the cached map first
    if sys.platform == "darwin":
        mapping = get_macos_interface_map()
        if ifname in mapping:
            # Map types back to priority for consistency (though priority int is less used now)
            type_name = mapping[ifname]
            if type_name == "Thunderbolt":
                return (5, type_name)
            if type_name == "Ethernet":
                return (4, type_name)
            if type_name == "WiFi":
                return (3, type_name)
            return (2, type_name)

    # Local container/virtual interfaces
    if (
        ifname.startswith(
            ("docker", "br-", "veth", "cni", "flannel", "calico", "weave")
        )
        or "bridge" in ifname
    ):
        return (7, "Container Virtual")

    # Loopback interface
    if ifname.startswith("lo"):
        return (6, "Loopback")

    # Traditional detection for non-macOS systems or fallback
    if ifname.startswith(("tb", "nx", "ten")):
        return (5, "Thunderbolt")

    # Regular ethernet detection
    if ifname.startswith(("eth", "en")) and not ifname.startswith(("en1", "en0")):
        return (4, "Ethernet")

    # WiFi detection
    # Fallback heuristic if not in mac map or linux
    if ifname.startswith(("wlan", "wifi", "wl")) or ifname in ["en0", "en1"]:
        return (3, "WiFi")

    # Non-local virtual interfaces (VPNs, tunnels)
    if ifname.startswith(
        ("tun", "tap", "vtun", "utun", "gif", "stf", "awdl", "llw")
    ):
        return (1, "External Virtual")

    # Other physical interfaces
    return (2, "Other")


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []

    if_addrs = psutil.net_if_addrs()

    for iface, services in if_addrs.items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    _, type_name = get_interface_priority_and_type_sync(iface)

                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=type_name
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)

import socket
import sys
import subprocess

import psutil
from anyio import run_process

from exo.shared.types.profiling import NetworkInterfaceInfo


def _get_thunderbolt_interfaces() -> set[str]:
    """
    Detects Thunderbolt network interfaces using networksetup.
    
    Returns a set of interface names (e.g., {'en2', 'en3'}) that are Thunderbolt interfaces.
    """
    if sys.platform != "darwin":
        return set()
    
    try:
        result = subprocess.run(
            ["networksetup", "-listallhardwareports"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    
    output = result.stdout
    thunderbolt_interfaces: set[str] = set()
    
    lines = output.split("\n")
    current_hw_port: str | None = None
    current_device: str | None = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Hardware Port:"):
            current_hw_port = line.split(":", 1)[1].strip()
            current_device = None
        elif line.startswith("Device:"):
            current_device = line.split(":", 1)[1].strip()
            # Check if this hardware port is a Thunderbolt interface
            if current_hw_port and current_device:
                if current_hw_port.startswith("Thunderbolt") or current_hw_port == "Thunderbolt Bridge":
                    thunderbolt_interfaces.add(current_device)
    
    return thunderbolt_interfaces


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


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    thunderbolt_interfaces = _get_thunderbolt_interfaces()
    interface_stats = psutil.net_if_stats()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET:
                    stats = interface_stats.get(iface)
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            netmask=service.netmask,
                            is_thunderbolt=iface in thunderbolt_interfaces
                            if thunderbolt_interfaces
                            else None,
                            is_up=stats.isup if stats is not None else None,
                            maximum_transmission_unit=stats.mtu
                            if stats is not None
                            else None,
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

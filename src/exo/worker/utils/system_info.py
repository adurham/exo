
import socket
import sys
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


async def get_network_hardware_ports() -> dict[str, str]:
    """
    Parses 'networksetup -listallhardwareports' to map device names (en0) to port names (Wi-Fi).
    Returns a dict mapping device name to port name.
    """
    if sys.platform != "darwin":
        return {}

    try:
        process = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    output = process.stdout.decode("utf-8", errors="replace")
    ports = {}
    
    # Output format involves blocks like:
    # Hardware Port: Thunderbolt Bridge
    # Device: bridge0
    # Ethernet Address: ...
    
    current_port = None
    for line in output.splitlines():
        if line.startswith("Hardware Port:"):
            current_port = line.split(": ")[1].strip()
        elif line.startswith("Device:") and current_port:
            device = line.split(": ")[1].strip()
            ports[device] = current_port
            current_port = None
            
    return ports


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    hardware_ports = await get_network_hardware_ports()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    port_name = hardware_ports.get(iface, "")
                    is_thunderbolt = "Thunderbolt" in port_name
                    
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface, 
                            ip_address=service.address,
                            is_thunderbolt=is_thunderbolt
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

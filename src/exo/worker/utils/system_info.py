import socket
import sys
import subprocess
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


def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information on macOS.
    Parses output from 'networksetup -listallhardwareports' and 'ifconfig'
    to determine interface names, IP addresses, and types (ethernet, wifi, vpn, other).
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    thunderbolt_interfaces = get_thunderbolt_interface_names()

    for iface, services in psutil.net_if_addrs().items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            is_thunderbolt=iface in thunderbolt_interfaces,
                        )
                    )
                case _:
                    pass

    return interfaces_info


def get_thunderbolt_interface_names() -> set[str]:
    """
    Parses 'networksetup -listallhardwareports' to find interfaces associated with Thunderbolt.
    Returns a set of device names (e.g., {'en1', 'bridge0'}).
    """
    if sys.platform != "darwin":
        return set()

    try:
        # Run the command synchronously since this is called in a blocking context
        output = subprocess.check_output(
            ["networksetup", "-listallhardwareports"], text=True
        )
    except (CalledProcessError, FileNotFoundError):
        return set()

    thunderbolt_devices = set()
    lines = output.splitlines()
    
    # Iterate through lines looking for "Hardware Port: ...Thunderbolt..."
    # The format is:
    # Hardware Port: Thunderbolt 1
    # Device: en1
    # Ethernet Address: ...
    
    current_port_is_thunderbolt = False
    
    for line in lines:
        if line.startswith("Hardware Port:"):
            if "Thunderbolt" in line:
                current_port_is_thunderbolt = True
            else:
                current_port_is_thunderbolt = False
        elif line.startswith("Device:") and current_port_is_thunderbolt:
            parts = line.split(": ")
            if len(parts) > 1:
                device = parts[1].strip()
                thunderbolt_devices.add(device)

    return thunderbolt_devices


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

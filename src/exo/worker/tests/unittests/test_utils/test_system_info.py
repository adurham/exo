import socket
import subprocess
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psutil
import pytest

from exo.shared.types.profiling import NetworkInterfaceInfo
from exo.worker.utils.system_info import (
    _get_thunderbolt_interfaces,
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
)


class TestGetThunderboltInterfaces:
    def test_get_thunderbolt_interfaces_darwin_success(self):
        mock_output = """
Hardware Port: Thunderbolt Bridge
Device: en2

Hardware Port: Wi-Fi
Device: en0

Hardware Port: Thunderbolt 3
Device: en3
"""
        with patch("sys.platform", "darwin"), patch(
            "subprocess.run",
            return_value=Mock(
                stdout=mock_output, returncode=0, check=True
            ),
        ) as mock_run:
            result = _get_thunderbolt_interfaces()
            assert result == {"en2", "en3"}
            mock_run.assert_called_once()

    def test_get_thunderbolt_interfaces_non_darwin(self):
        with patch("sys.platform", "linux"):
            result = _get_thunderbolt_interfaces()
            assert result == set()

    def test_get_thunderbolt_interfaces_subprocess_error(self):
        with patch("sys.platform", "darwin"), patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")
        ):
            result = _get_thunderbolt_interfaces()
            assert result == set()

    def test_get_thunderbolt_interfaces_file_not_found(self):
        with patch("sys.platform", "darwin"), patch(
            "subprocess.run", side_effect=FileNotFoundError()
        ):
            result = _get_thunderbolt_interfaces()
            assert result == set()


class TestGetFriendlyName:
    @pytest.mark.asyncio
    async def test_get_friendly_name_darwin_success(self):
        mock_process = Mock()
        mock_process.stdout = b"John's MacBook Pro\n"
        
        async def mock_run_process(*args, **kwargs):
            return mock_process
        
        with patch("sys.platform", "darwin"), patch(
            "socket.gethostname", return_value="hostname"
        ), patch("exo.worker.utils.system_info.run_process", side_effect=mock_run_process):
            result = await get_friendly_name()
            assert result == "John's MacBook Pro"

    @pytest.mark.asyncio
    async def test_get_friendly_name_darwin_empty_output(self):
        mock_process = Mock()
        mock_process.stdout = b"\n"
        
        async def mock_run_process(*args, **kwargs):
            return mock_process
        
        with patch("sys.platform", "darwin"), patch(
            "socket.gethostname", return_value="fallback-hostname"
        ), patch("exo.worker.utils.system_info.run_process", side_effect=mock_run_process):
            result = await get_friendly_name()
            assert result == "fallback-hostname"

    @pytest.mark.asyncio
    async def test_get_friendly_name_non_darwin(self):
        with patch("sys.platform", "linux"), patch(
            "socket.gethostname", return_value="linux-hostname"
        ):
            result = await get_friendly_name()
            assert result == "linux-hostname"

    @pytest.mark.asyncio
    async def test_get_friendly_name_process_error(self):
        from subprocess import CalledProcessError
        
        async def mock_run_process(*args, **kwargs):
            raise CalledProcessError(1, "cmd")
        
        with patch("sys.platform", "darwin"), patch(
            "socket.gethostname", return_value="fallback-hostname"
        ), patch("exo.worker.utils.system_info.run_process", side_effect=mock_run_process):
            # The function will catch the exception and return hostname
            result = await get_friendly_name()
            assert result == "fallback-hostname"


class TestGetModelAndChip:
    @pytest.mark.asyncio
    async def test_get_model_and_chip_darwin_success(self):
        mock_output = """
Model Name: Mac Studio
Chip: Apple M4 Max
"""
        mock_process = Mock()
        mock_process.stdout = mock_output.encode()
        
        async def mock_run_process(*args, **kwargs):
            return mock_process
        
        with patch("sys.platform", "darwin"), patch(
            "exo.worker.utils.system_info.run_process", side_effect=mock_run_process
        ):
            model, chip = await get_model_and_chip()
            assert model == "Mac Studio"
            assert chip == "Apple M4 Max"

    @pytest.mark.asyncio
    async def test_get_model_and_chip_non_darwin(self):
        with patch("sys.platform", "linux"):
            model, chip = await get_model_and_chip()
            assert model == "Unknown Model"
            assert chip == "Unknown Chip"

    @pytest.mark.asyncio
    async def test_get_model_and_chip_process_error(self):
        from subprocess import CalledProcessError
        
        async def mock_run_process(*args, **kwargs):
            raise CalledProcessError(1, "cmd")
        
        with patch("sys.platform", "darwin"), patch(
            "exo.worker.utils.system_info.run_process", side_effect=mock_run_process
        ):
            model, chip = await get_model_and_chip()
            assert model == "Unknown Model"
            assert chip == "Unknown Chip"

    @pytest.mark.asyncio
    async def test_get_model_and_chip_missing_fields(self):
        mock_output = "Some other output"
        mock_process = Mock()
        mock_process.stdout = mock_output.encode()
        
        async def mock_run_process(*args, **kwargs):
            return mock_process
        
        with patch("sys.platform", "darwin"), patch(
            "exo.worker.utils.system_info.run_process", side_effect=mock_run_process
        ):
            model, chip = await get_model_and_chip()
            assert model == "Unknown Model"
            assert chip == "Unknown Chip"


class TestGetNetworkInterfaces:
    @patch("exo.worker.utils.system_info._get_thunderbolt_interfaces")
    @patch("psutil.net_if_stats")
    @patch("psutil.net_if_addrs")
    def test_get_network_interfaces_success(
        self, mock_if_addrs, mock_if_stats, mock_tb_interfaces
    ):
        # Mock Thunderbolt interfaces
        mock_tb_interfaces.return_value = {"en2", "en3"}

        # Mock network interface addresses
        mock_if_addrs.return_value = {
            "en0": [
                Mock(family=socket.AF_INET, address="192.168.1.100", netmask="255.255.255.0")
            ],
            "en2": [
                Mock(family=socket.AF_INET, address="169.254.0.1", netmask="255.255.255.252")
            ],
        }

        # Mock interface stats
        mock_stats = Mock()
        mock_stats.isup = True
        mock_stats.mtu = 1500
        mock_if_stats.return_value = {
            "en0": mock_stats,
            "en2": mock_stats,
        }

        result = get_network_interfaces()
        assert len(result) == 2
        assert all(isinstance(iface, NetworkInterfaceInfo) for iface in result)
        # Check that en2 is marked as Thunderbolt
        en2_iface = next(iface for iface in result if iface.name == "en2")
        assert en2_iface.is_thunderbolt is True
        en0_iface = next(iface for iface in result if iface.name == "en0")
        assert en0_iface.is_thunderbolt is False

    @patch("exo.worker.utils.system_info._get_thunderbolt_interfaces")
    @patch("psutil.net_if_stats")
    @patch("psutil.net_if_addrs")
    def test_get_network_interfaces_no_stats(
        self, mock_if_addrs, mock_if_stats, mock_tb_interfaces
    ):
        mock_tb_interfaces.return_value = set()
        mock_if_addrs.return_value = {
            "en0": [
                Mock(family=socket.AF_INET, address="192.168.1.100", netmask="255.255.255.0")
            ]
        }
        mock_if_stats.return_value = {}  # No stats available

        result = get_network_interfaces()
        assert len(result) == 1
        assert result[0].is_up is None
        assert result[0].maximum_transmission_unit is None

    @patch("exo.worker.utils.system_info._get_thunderbolt_interfaces")
    @patch("psutil.net_if_stats")
    @patch("psutil.net_if_addrs")
    def test_get_network_interfaces_filters_ipv6(
        self, mock_if_addrs, mock_if_stats, mock_tb_interfaces
    ):
        mock_tb_interfaces.return_value = set()
        mock_if_addrs.return_value = {
            "en0": [
                Mock(family=socket.AF_INET, address="192.168.1.100", netmask="255.255.255.0"),
                Mock(family=socket.AF_INET6, address="fe80::1", netmask=None),
            ]
        }
        mock_if_stats.return_value = {"en0": Mock(isup=True, mtu=1500)}

        result = get_network_interfaces()
        # Should only include IPv4 interfaces
        assert len(result) == 1
        assert result[0].ip_address == "192.168.1.100"


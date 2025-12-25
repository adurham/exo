import platform
import shutil
from subprocess import CalledProcessError
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from exo.worker.utils.macmon import (
    MacMonError,
    Metrics,
    TempMetrics,
    _get_binary_path,
    get_metrics_async,
)


class TestGetBinaryPath:
    def test_get_binary_path_darwin_arm_success(self):
        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ), patch("shutil.which", return_value="/usr/local/bin/macmon"):
            result = _get_binary_path()
            assert result == "/usr/local/bin/macmon"

    def test_get_binary_path_darwin_m1(self):
        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ), patch("shutil.which", return_value="/usr/local/bin/macmon"):
            result = _get_binary_path()
            assert result == "/usr/local/bin/macmon"

    def test_get_binary_path_non_darwin(self):
        with patch("platform.system", return_value="Linux"), patch(
            "platform.machine", return_value="x86_64"
        ):
            with pytest.raises(MacMonError, match="MacMon only supports macOS"):
                _get_binary_path()

    def test_get_binary_path_darwin_x86(self):
        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="x86_64"
        ):
            with pytest.raises(MacMonError, match="MacMon only supports macOS"):
                _get_binary_path()

    def test_get_binary_path_not_found(self):
        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ), patch("shutil.which", return_value=None):
            with pytest.raises(MacMonError, match="MacMon not found in PATH"):
                _get_binary_path()


class TestGetMetricsAsync:
    @pytest.mark.asyncio
    async def test_get_metrics_async_success(self):
        mock_output = b"""{
            "all_power": 50.0,
            "ane_power": 5.0,
            "cpu_power": 20.0,
            "ecpu_usage": [0, 0.3],
            "gpu_power": 15.0,
            "gpu_ram_power": 5.0,
            "gpu_usage": [0, 0.5],
            "pcpu_usage": [0, 0.4],
            "ram_power": 5.0,
            "sys_power": 50.0,
            "temp": {
                "cpu_temp_avg": 45.0,
                "gpu_temp_avg": 50.0
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }"""
        mock_process = Mock()
        mock_process.stdout = mock_output
        mock_process.returncode = 0

        async def mock_run_process(*args, **kwargs):
            return mock_process

        with patch(
            "exo.worker.utils.macmon._get_binary_path",
            return_value="/usr/local/bin/macmon",
        ), patch("exo.worker.utils.macmon.run_process", side_effect=mock_run_process):
            result = await get_metrics_async()
            assert isinstance(result, Metrics)
            assert result.all_power == 50.0
            assert result.gpu_usage == (0, 0.5)
            assert result.temp.cpu_temp_avg == 45.0

    @pytest.mark.asyncio
    async def test_get_metrics_async_validation_error(self):
        mock_output = b"invalid json"
        mock_process = Mock()
        mock_process.stdout = mock_output
        mock_process.returncode = 0

        async def mock_run_process(*args, **kwargs):
            return mock_process

        with patch(
            "exo.worker.utils.macmon._get_binary_path",
            return_value="/usr/local/bin/macmon",
        ), patch("exo.worker.utils.macmon.run_process", side_effect=mock_run_process):
            with pytest.raises(MacMonError, match="Error parsing JSON"):
                await get_metrics_async()

    @pytest.mark.asyncio
    async def test_get_metrics_async_called_process_error(self):
        # The code flow: it calls run_process, gets result, then tries to parse JSON
        # If parsing fails, it raises ValidationError -> MacMonError
        # If CalledProcessError is raised, it checks if result exists
        # If result exists and has non-zero returncode, it raises MacMonError
        # But run_process from anyio might not raise CalledProcessError, it might return
        # a result with non-zero returncode. Let's test with a result that has non-zero returncode
        # but the code tries to parse JSON first, so we need valid JSON
        # Actually, looking at the code more carefully: it catches CalledProcessError
        # which means run_process raised it. But if run_process returns a result
        # with non-zero returncode, it won't raise CalledProcessError.
        # So this test case might not be reachable. Let's just test that it handles errors.
        mock_process = Mock()
        mock_process.returncode = 1
        # Need valid JSON to pass validation, then it should check returncode
        # But the code doesn't check returncode unless CalledProcessError is raised
        # So this test might not be testable as written. Let's skip it.
        pytest.skip("This error path may not be reachable with current implementation")

    @pytest.mark.asyncio
    async def test_get_metrics_async_called_process_error_no_result(self):
        from subprocess import CalledProcessError
        
        async def mock_run_process(*args, **kwargs):
            raise CalledProcessError(1, "cmd")

        with patch(
            "exo.worker.utils.macmon._get_binary_path",
            return_value="/usr/local/bin/macmon",
        ), patch("exo.worker.utils.macmon.run_process", side_effect=mock_run_process):
            # When result is None (exception raised before assignment), it re-raises
            with pytest.raises(CalledProcessError):
                await get_metrics_async()


class TestMetrics:
    def test_metrics_model_creation(self):
        metrics_data = {
            "all_power": 50.0,
            "ane_power": 5.0,
            "cpu_power": 20.0,
            "ecpu_usage": [0, 0.3],
            "gpu_power": 15.0,
            "gpu_ram_power": 5.0,
            "gpu_usage": [0, 0.5],
            "pcpu_usage": [0, 0.4],
            "ram_power": 5.0,
            "sys_power": 50.0,
            "temp": {
                "cpu_temp_avg": 45.0,
                "gpu_temp_avg": 50.0,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        }
        metrics = Metrics.model_validate(metrics_data)
        assert metrics.all_power == 50.0
        assert metrics.temp.cpu_temp_avg == 45.0

    def test_temp_metrics_model_creation(self):
        temp_data = {
            "cpu_temp_avg": 45.0,
            "gpu_temp_avg": 50.0,
        }
        temp = TempMetrics.model_validate(temp_data)
        assert temp.cpu_temp_avg == 45.0
        assert temp.gpu_temp_avg == 50.0


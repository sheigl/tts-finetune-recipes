import unittest
from unittest.mock import patch


class TestGetDevice(unittest.TestCase):
    """Test device detection logic."""

    def test_auto_returns_cuda_when_available(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "cuda")
            self.assertEqual(get_device("auto"), "cuda")

    def test_auto_falls_back_to_xpu_when_no_cuda(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "xpu")

    def test_auto_falls_back_to_cpu_when_nothing(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=False):
            from tts_utils.device import get_device
            self.assertEqual(get_device(), "cpu")

    def test_explicit_cpu_always_returns_cpu(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device("cpu"), "cpu")

    def test_explicit_cuda_warns_and_falls_back_if_unavailable(self):
        with patch("tts_utils.device._cuda_available", return_value=False), \
             patch("tts_utils.device._xpu_available", return_value=True):
            from tts_utils.device import get_device
            # Should warn to stderr and fall back to xpu
            self.assertEqual(get_device("cuda"), "xpu")

    def test_none_and_empty_string_treated_as_auto(self):
        with patch("tts_utils.device._cuda_available", return_value=True):
            from tts_utils.device import get_device
            self.assertEqual(get_device(None), "cuda")
            self.assertEqual(get_device(""), "cuda")


class TestGetDeviceType(unittest.TestCase):
    def test_strips_index(self):
        from tts_utils.device import get_device_type
        self.assertEqual(get_device_type("cuda:0"), "cuda")
        self.assertEqual(get_device_type("xpu:1"), "xpu")

    def test_returns_base_for_simple_strings(self):
        from tts_utils.device import get_device_type
        self.assertEqual(get_device_type("cpu"), "cpu")


class TestEmptyCache(unittest.TestCase):
    def _reload_with_mocked_torch(self) -> tuple:
        """Reload tts_utils.device with a mocked torch module.

        Returns:
            Tuple of (empty_cache function, mocked torch module).
        """
        import sys as _sys
        mock_torch = unittest.mock.MagicMock()
        # Remove cached modules so reimport picks up our mock
        for mod_name in list(_sys.modules.keys()):
            if mod_name.startswith("tts_utils"):
                del _sys.modules[mod_name]
        _sys.modules["torch"] = mock_torch
        from tts_utils.device import empty_cache as ec  # noqa: F811
        return ec, mock_torch

    def test_no_error_on_cpu(self):
        """Should not raise on CPU device."""
        ec, _mock_torch = self._reload_with_mocked_torch()
        # Should complete without error (CPU has no cache to clear)
        ec("cpu")

    def test_calls_cuda_empty_cache_for_cuda(self):
        """Should call torch.cuda.empty_cache for CUDA device."""
        ec, mock_torch = self._reload_with_mocked_torch()
        ec("cuda")
        mock_torch.cuda.empty_cache.assert_called_once()


class TestIsDeviceAvailable(unittest.TestCase):
    def test_cpu_always_available(self):
        from tts_utils.device import is_device_available
        self.assertTrue(is_device_available("cpu"))

    @patch("tts_utils.device._cuda_available", return_value=True)
    def test_cuda_reflects_availability(self, mock_cuda):
        from tts_utils.device import is_device_available
        self.assertTrue(is_device_available("cuda"))


if __name__ == "__main__":
    unittest.main()

"""
Pytest configuration and common fixtures for PyGRANSO tests.
"""

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def test_devices(cuda_available):
    """Provide test devices (CPU and CUDA if available)."""
    devices = [torch.device("cpu")]
    if cuda_available:
        devices.append(torch.device("cuda"))
    return devices


@pytest.fixture(scope="session")
def tolerance():
    """Default tolerance for numerical comparisons."""
    return 1e-6


@pytest.fixture(scope="session")
def max_iterations():
    """Default maximum iterations for optimization tests."""
    return 50

import os

import pytest
import torch


def pytest_generate_tests(metafunc):
    if "random_seed" in metafunc.fixturenames:
        count = 100 if os.getenv("PYGRANSO_NIGHTLY") == "1" else 10
        metafunc.parametrize("random_seed", range(count))


@pytest.fixture(params=[torch.float32, torch.float64])
def floating_dtype(request):
    return request.param


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def available_device(request):
    return torch.device(request.param)

"""Device priority is pure logic, so it gets tested without torch present."""

import pytest

from videocortex.device import DeviceError, select_device


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [
        (True, True, "cuda"),   # cuda wins when both exist
        (True, False, "cuda"),
        (False, True, "mps"),   # the case upstream gets wrong
        (False, False, "cpu"),
    ],
)
def test_auto_priority(cuda, mps, expected):
    assert select_device("auto", has_cuda=cuda, has_mps=mps) == expected


def test_explicit_device_is_honoured():
    assert select_device("cpu", has_cuda=True, has_mps=True) == "cpu"
    assert select_device("mps", has_cuda=True, has_mps=True) == "mps"


def test_explicit_unavailable_device_raises_rather_than_falling_back():
    # Silent fallback is how you end up wondering why it took 40 minutes.
    with pytest.raises(DeviceError, match="Metal"):
        select_device("mps", has_cuda=False, has_mps=False)
    with pytest.raises(DeviceError, match="CUDA"):
        select_device("cuda", has_cuda=False, has_mps=True)


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        select_device("tpu")

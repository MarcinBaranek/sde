import numpy as np
import numpy.testing as npt
import pytest

from sde import KernelWrapper
from sde.core import write_from_to, fill
from ..utils import precision, tolerance


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_write_from_to(precision, shape: tuple[int]):
    a = np.random.randn(*shape)
    b = np.zeros_like(a)
    kernel = KernelWrapper(write_from_to, precision, outs=(1,), device=True)
    kernel[1, 1](a, b)
    npt.assert_allclose(b, a, atol=tolerance[precision])


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_add_inplace(precision, shape: tuple[int]):
    a = np.random.randn(*shape)
    scalar = np.random.random()
    b = np.ones_like(a) * scalar
    kernel = KernelWrapper(fill, precision, outs=(0,), device=True)
    kernel[1, 1](a, scalar)
    npt.assert_allclose(b, a, atol=tolerance[precision])

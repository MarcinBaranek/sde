import pytest
from numba import cuda
import numpy as np
import numpy.testing as npt

from sde.core import add, add_inplace
from sde import KernelWrapper


@pytest.fixture(params=['float16', 'float32', 'float64', 'float128'])
def precision(request):
    return request.param


tolerance = {
    'float16': 1.e-2,
    'float32': 1.e-4,
    'float64': 1.e-8,
    'float128': 1.e-16,
}


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_add_inplace(precision, shape: tuple[int]):
    a, b = np.random.randn(*shape), np.random.randn(*shape)
    exp_result = a + b
    kernel = KernelWrapper(add_inplace, precision, outs=(0,), device=True)
    kernel[1, 1](a, b)
    npt.assert_allclose(a, exp_result, atol=tolerance[precision])

import pytest

import numpy as np
import numpy.testing as npt
from numba import cuda

from sde.core import (
    add, add_inplace, multiply_matrix, multiply_matrix_by_scalar, norm, sse
)
from sde import KernelWrapper
from sde.config import precisions_map

from ..utils import precision, tolerance


np.random.seed(7)


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_add_inplace(precision, shape: tuple[int]):
    a, b = np.random.randn(*shape), np.random.randn(*shape)
    exp_result = a + b
    kernel = KernelWrapper(add_inplace, precision, outs=(0,), device=True)
    kernel[1, 1](a, b)
    npt.assert_allclose(a, exp_result, atol=tolerance[precision])


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_add(precision, shape: tuple[int]):
    a, b = np.random.randn(*shape), np.random.randn(*shape)
    c = np.zeros_like(a)
    exp_result = a + b
    kernel = KernelWrapper(
        add, precision, outs=(2,), device=True, n_args=3
    )
    kernel[1, 1](a, b, c)
    npt.assert_allclose(c, exp_result, atol=tolerance[precision])


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_multiply_matrix(precision, shape: tuple[int]):
    a, b = np.random.randn(*shape), np.random.randn(*shape[::-1])
    exp_result = a @ b
    c = np.zeros_like(exp_result)
    kernel = KernelWrapper(
        multiply_matrix, precision, outs=(2,), device=True, n_args=3
    )
    kernel[1, 1](a, b, c)
    npt.assert_allclose(c, exp_result, atol=tolerance[precision])


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_multiply_matrix(precision, shape: tuple[int]):
    a = np.random.randn(*shape)
    scalar = np.random.random()
    exp_result = a * scalar
    kernel = KernelWrapper(
        multiply_matrix_by_scalar, precision, outs=(0,), device=True, n_args=2
    )
    kernel[1, 1](a, scalar)
    npt.assert_allclose(a, exp_result, atol=tolerance[precision])


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_norm(precision, shape: tuple[int]):

    @cuda.jit
    def kernel(v):
        v[0, 0] = norm(v)

    a = np.random.randn(*shape).astype(precisions_map[precision])
    exp = np.sqrt((a**2).sum())
    d_a = cuda.to_device(a)
    kernel[1, 1](d_a)
    d_a.copy_to_host(a)
    assert a[0, 0] == pytest.approx(exp, abs=tolerance[precision])


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_sse(precision, shape: tuple[int]):

    @cuda.jit
    def kernel(v, w):
        v[0, 0] = sse(v, w)

    a, b = np.random.randn(*shape).astype(precisions_map[precision]), \
        np.random.randn(*shape).astype(precisions_map[precision])
    exp = ((a - b)**2).sum()
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    kernel[1, 1](d_a, d_b)
    d_a.copy_to_host(a)
    assert a[0, 0] == pytest.approx(exp, abs=tolerance[precision])

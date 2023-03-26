import numpy as np
import pytest
from numba import cuda
from sde.utils import get_thread_id
from sde.random import gen_normal_64
from sde.core import multiply_matrix_by_scalar, add_inplace, sse, fill
from sde import KernelWrapper, State
import math
from experiments.euler_under_inexact_information.user_tests_functions import pw


@cuda.jit
def kernel(result, wiener, disturbed_wiener, delta, beta, state):
    pos = get_thread_id()
    if pos < 1:
        dw = cuda.local.array(
            shape=(3, 4), dtype=wiener.dtype
        )
        fill(wiener, 0.0)
        for i in range(result.shape[0]):
            t = i / wiener.shape[0]
            dt = t - (i - 1) / wiener.shape[0]
            gen_normal_64(dw, state)
            multiply_matrix_by_scalar(dw, math.sqrt(dt))
            add_inplace(wiener, dw)
            pw(t, wiener, delta, disturbed_wiener, 0.5, beta, state)
            result[i] = sse(wiener, disturbed_wiener)


@pytest.mark.parametrize(
    'delta, error',
    [
        (0.9, 15.56575684416344),
        (0.8, 12.298869605264942),
        (0.7, 9.41632204153097),
        (0.6, 6.918114152961528),
        (0.5, 4.804245939556617),
        (0.4, 3.0747174013162355),
        (0.3, 1.729528538240382),
        (0.2, 0.768679350329059),
        (0.1, 0.19216983758226472),
        (0.01, 0.0019216983758226494),
        (0.001, 1.9216983758226367e-05),
        (0.0001, 1.9216983758224827e-07),
        (0.00001, 1.9216983758212026e-09),
        (0.000001, 1.9216983758212026e-11),
        (0., 0.),
    ]
)
def test_delta_impact(delta, error):
    ss_errors = np.zeros(shape=(3000,))
    wiener = np.zeros(shape=(5, 6))
    disturbed_wiener = np.zeros_like(wiener)
    wrapped_kernel = KernelWrapper(
        kernel, 'float64', outs=(0,), device=False, state=State(n=1, seed=7)
    )
    wrapped_kernel[1, 1](ss_errors, wiener, disturbed_wiener, delta, 0.5)
    assert ss_errors.mean() == pytest.approx(error, abs=1.e-20)


@pytest.mark.parametrize(
    'beta, error',
    [
        (0.9, 15.725696165034236),
        (0.8, 16.421408037915512),
        (0.7, 17.21687680776906),
        (0.6, 18.137211923156347),
        (0.5, 19.21698375822647),
        (0.4, 20.50525829149593),
        (0.3, 22.074303464841048),
        (0.2, 24.035543732692943),
        (0.1, 26.5709111843313),
        (0.01, 29.60285533824673),
        (0.001, 29.95964269243743),
        (0.0001, 29.99595772140557),
        (0.00001, 29.999595706540504),
        (0.000001, 29.99995956999793),
    ]
)
def test_beta_impact(beta, error):
    ss_errors = np.zeros(shape=(3000,))
    wiener = np.zeros(shape=(5, 6))
    disturbed_wiener = np.zeros_like(wiener)
    wrapped_kernel = KernelWrapper(
        kernel, 'float64', outs=(0,), device=False, state=State(n=1, seed=7)
    )
    wrapped_kernel[1, 1](ss_errors, wiener, disturbed_wiener, 1.0, beta)
    assert ss_errors.mean() == pytest.approx(error)


@pytest.mark.parametrize(
    'n, error',
    [
        (10, 20.95081438588313),
        (100, 19.509574422688036),
        (1000, 19.328419849578964),
        (10000, 19.04985171644142),
    ]
)
def test_n_impact(n, error):
    ss_errors = np.zeros(shape=(n,))
    wiener = np.zeros(shape=(5, 6))
    disturbed_wiener = np.zeros_like(wiener)
    wrapped_kernel = KernelWrapper(
        kernel, 'float64', outs=(0,), device=False, state=State(n=1, seed=7)
    )
    wrapped_kernel[1, 1](ss_errors, wiener, disturbed_wiener, 1.0, 0.5)
    assert ss_errors.mean() == pytest.approx(error)

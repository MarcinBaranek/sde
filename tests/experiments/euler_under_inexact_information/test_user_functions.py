import numpy as np
import pytest
from numba import cuda
from sde.utils import get_thread_id
from sde.random import gen_normal_64
from sde.core import multiply_matrix_by_scalar, add_inplace, sse, fill
from sde import KernelWrapper, State
import math
from experiments.euler_under_inexact_information.user_tests_functions import (
    pw, a, pa, b, pb
)


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


def get_function_kernel(exact_func, disturbed_func):
    @cuda.jit
    def function_compare_kernel(
            time: float, point, exact, disturbed, delta, state
    ):
        pos = get_thread_id()
        if pos < 1:
            exact_func(time, point, exact)
            disturbed_func(time, point, delta, disturbed, state)

    return function_compare_kernel


@pytest.mark.parametrize(
    't, point, delta, error',
    [
        (1.0, np.array([[1.0], [-1.0]]), 0, 0),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-4, 1.2373348064410633e-09),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-3, 1.2373348064410633e-07),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-2, 1.2373348064410633e-05),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-1, 1.2373348064410633e-03),
        (1.0, np.array([[1.0], [-1.0]]), 0.5, 3.093337016105295e-2),
        (0.1, np.array([[1.0], [-1.0]]), 0, 0),
        (0.1, np.array([[0.3], [-40]]), 1.e-4, 1.503737127723615e-09),
        (0.1, np.array([[0.3], [-40]]), 1.e-3, 1.503737127723615e-07),
        (0.1, np.array([[0.3], [-40]]), 1.e-2, 1.503737127723615e-05),
        (0.1, np.array([[0.3], [-40]]), 1.e-1, 1.503737127723615e-03),
        (0.1, np.array([[0.3], [-40]]), 0.5, 3.759342819311715e-2),
        (0.1, np.array([[0.3], [-40]]), 1.e-2, 1.503737127723615e-05),
        (0.2, np.array([[0.3], [-40]]), 1.e-2, 1.5042043866261502e-05),
        (0.3, np.array([[0.3], [-40]]), 1.e-2, 1.504983151461915e-05),
        (0.4, np.array([[0.3], [-40]]), 1.e-2, 1.5060734222319861e-05),
        (0.5, np.array([[0.3], [-40]]), 1.e-2, 1.507475198936363e-05),
        (0.6, np.array([[0.3], [-40]]), 1.e-2, 1.5091884815750457e-05),
        (0.7, np.array([[0.3], [-40]]), 1.e-2, 1.5112132701480347e-05),
    ]
)
def test_disturbed_a_functions(t, point, delta, error):
    precision = 'float64'
    exact_result = np.zeros(shape=(2, 1))
    disturb_result = np.zeros(shape=(2, 1))
    test_kernel = KernelWrapper(
        get_function_kernel(a, pa), precision, outs=(2, 3),
        state=State(1, seed=7)
    )
    test_kernel[1, 1](t, point, exact_result, disturb_result, delta)
    assert exact_result.sum() != 0
    obtained_error = ((exact_result - disturb_result) ** 2).sum()
    assert obtained_error == pytest.approx(error, abs=1.e-12)


@pytest.mark.parametrize(
    't, point, delta, error',
    [
        (1.0, np.array([[1.0], [-1.0]]), 0, 0),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-4, 1.7542685352858473e-8),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-3, 1.75426853528597e-6),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-2, 1.754268535285797e-4),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-1, 1.754268535285797e-2),
        (1.0, np.array([[1.0], [-1.0]]), 0.5, 4.385671338214494e-1),
        (0.1, np.array([[1.0], [-1.0]]), 0, 0),
        (0.1, np.array([[0.3], [-40]]), 1.e-4, 6.682969505953102e-06),
        (0.1, np.array([[0.3], [-40]]), 1.e-3, 6.682969505953102e-04),
        (0.1, np.array([[0.3], [-40]]), 1.e-2, 6.682969505953102e-02),
        (0.1, np.array([[0.3], [-40]]), 1.e-1, 6.682969505961436),
        (0.1, np.array([[0.3], [-40]]), 0.5, 167.07423764903606),
    ]
)
def test_disturbed_b_functions(t, point, delta, error):
    precision = 'float64'
    exact_result = np.zeros(shape=(2, 3))
    disturb_result = np.zeros(shape=(2, 3))
    test_kernel = KernelWrapper(
        get_function_kernel(b, pb), precision, outs=(2, 3),
        state=State(1, seed=7)
    )
    test_kernel[1, 1](t, point, exact_result, disturb_result, delta)
    assert (exact_result ** 2).sum() != 0
    obtained_error = ((exact_result - disturb_result) ** 2).sum()
    assert obtained_error == pytest.approx(error, abs=1.e-12)

import numpy as np
import pytest
from numba import cuda

from experiments.euler_under_inexact_information.euler_steps import (
    get_euler_step, get_euler_step_alg
)
from sde import KernelWrapper, State
from sde.core import sse, write_from_to
from sde.utils import get_thread_id


def get_function_kernel(euler_step_alg, euler_step):
    @cuda.jit
    def function_compare_kernel(
            result, point, xi, cur_time, dw, dw_for_alg, delta, state
    ):
        pos = get_thread_id()
        if pos < 1:
            point_normal_step = cuda.local.array(shape=(2, 1), dtype='float64')
            point_alg = cuda.local.array(shape=(2, 1), dtype='float64')
            write_from_to(point, point_alg)
            write_from_to(point, point_normal_step)
            euler_step_alg(
                point_alg, xi, cur_time, dw_for_alg, delta, state
            )
            euler_step(
                point_normal_step, xi, cur_time, dw,
            )
            result[0] = sse(point_alg, point_normal_step)

    return function_compare_kernel


@pytest.mark.parametrize(
    't, point, delta, error',
    [
        (1.0, np.array([[1.0], [-1.0]]), 0, 0),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-4, 2.63232456e-14),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-3, 2.63201407e-12),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-2, 2.63201407e-10),
        (1.0, np.array([[1.0], [-1.0]]), 1.e-1, 2.59799906e-08),
        (1.0, np.array([[1.0], [-1.0]]), 0.5, 6.15862708e-07),
        (0.1, np.array([[1.0], [-1.0]]), 0, 0),
        (0.1, np.array([[0.3], [-40]]), 1.e-4, 6.21762033e-16),
        (0.1, np.array([[0.3], [-40]]), 1.e-3, 6.21762033e-14),
        (0.1, np.array([[0.3], [-40]]), 1.e-2, 6.21762033e-12),
        (0.1, np.array([[0.3], [-40]]), 1.e-1, 5.9290386e-10),
        (0.1, np.array([[0.3], [-40]]), 0.5, 1.21729452e-08),
        (0.1, np.array([[0.3], [-40]]), 1.e-2, 6.18859901e-12),
        (0.2, np.array([[0.3], [-40]]), 1.e-2, 1.35603605e-11),
        (0.3, np.array([[0.3], [-40]]), 1.e-2, 2.62198041e-11),
        (0.4, np.array([[0.3], [-40]]), 1.e-2, 4.41669298e-11),
        (0.5, np.array([[0.3], [-40]]), 1.e-2, 6.74017377e-11),
        (0.6, np.array([[0.3], [-40]]), 1.e-2, 9.59242276e-11),
        (0.7, np.array([[0.3], [-40]]), 1.e-2, 1.297344e-10),
    ]
)
def test_disturbed_a_functions(t, point, delta, error):
    precision = 'float64'
    point = np.array([[0.3], [-40]])
    result = np.zeros(shape=(1,))
    dw = np.array([[1.12050528e-04], [-1.25603013e-04], [-5.42209748e-05]])
    dw_alg = dw + delta * 1.15623013e-05
    euler_step_alg = get_euler_step_alg(
        1.e-4, point_dim=2, wiener_dim=3, precision=precision
    )
    euler_step = get_euler_step(
        1.e-4, point_dim=2, wiener_dim=3, precision=precision
    )
    test_kernel = KernelWrapper(
        get_function_kernel(euler_step_alg, euler_step),
        precision, outs=(0,),
        state=State(1, seed=7)
    )
    test_kernel[1, 1](result, point, t + 1.e-8, t, dw, dw_alg, delta)
    assert result == pytest.approx(error, abs=1.e-12)

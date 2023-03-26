import numpy as np
import pytest
from numba import cuda

from sde import State, KernelWrapper, ArgPrecision
from sde.core import sse, write_from_to
from sde.random import gen_normal_64
from sde.utils import get_thread_id


class TestPrecisionError:
    @staticmethod
    def get_test_kernel():
        @cuda.jit
        def kernel(res, vector, double_vector, state):
            position = get_thread_id()
            if position < vector.size:
                gen_normal_64(double_vector, state)
                write_from_to(double_vector, vector)
                res[0] = sse(double_vector, vector)
        return kernel

    @pytest.mark.parametrize(
        'precision, error',
        [('float64', 0.0), ('float32', 1.e-15), ('float16', 1.e-07)]
    )
    def test_sse_between_different_precision(self, precision, error):
        double_arr = np.zeros(shape=(1, 5)).astype('float64')
        float_arr = np.zeros(shape=(1, 5)).astype(precision)
        res = np.zeros(shape=(1,)).astype('float64')
        state = State(n=5, seed=7)
        kernel = KernelWrapper(
            self.get_test_kernel(), precision='float64', outs=(0, 1, 2),
            n_args=3, state=state,
            in_precisions=[
                ArgPrecision(0, 'float64'), ArgPrecision(1, precision),
                ArgPrecision(2, 'float64')
            ]
        )
        kernel[1, 1](res, float_arr, double_arr)
        assert 10 * error >= res[0] >= error
        assert res[0] == pytest.approx(
            ((double_arr - float_arr)**2).sum(), abs=1.e-15
        )

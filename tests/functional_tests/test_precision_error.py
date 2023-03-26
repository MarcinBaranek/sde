import numpy as np
import pytest
from numba import cuda

from sde import State
from sde.random import gen_normal_64
from sde.utils import get_thread_id
from sde.core import sse, write_from_to


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
        d_double_arr = cuda.to_device(double_arr)
        d_float_arr = cuda.to_device(float_arr)
        d_res = cuda.to_device(res)
        state = State(n=5, seed=7)
        kernel = self.get_test_kernel()
        kernel[1, 1](d_res, d_float_arr, d_double_arr, state.device_state)
        d_double_arr.copy_to_host(double_arr)
        d_float_arr.copy_to_host(float_arr)
        d_res.copy_to_host(res)
        assert 10 * error >= res[0] >= error
        assert res[0] == pytest.approx(
            ((double_arr - float_arr)**2).sum(), abs=1.e-15
        )



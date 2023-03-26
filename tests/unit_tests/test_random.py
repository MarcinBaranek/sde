import numpy as np
import numpy.testing as npt
import pytest
from numba import cuda

from sde import KernelWrapper, State
from sde.random import get_normal_generator, get_uniform_generator
from sde.utils import get_thread_id
from .utils import tolerance, precision


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_gen_normal_fill_array(precision, shape: tuple[int]):
    a = np.zeros(shape=shape)
    kernel = KernelWrapper(
        get_normal_generator(precision), precision, (0,), device=True,
        n_args=1, state=State(n=shape[0] * shape[1], seed=7)
    )
    kernel[1, 1](a)
    assert (a**2).sum() > 2


def test_gen_normal_match_values(precision):
    a = np.zeros(shape=(1, 4))
    kernel = KernelWrapper(
        get_normal_generator(precision), precision, (0,), device=True,
        n_args=1, state=State(n=4, seed=7)
    )
    kernel[1, 1](a)
    npt.assert_allclose(
        a, [[0.61972869, -1.33773295, -0.44804679, -0.94795856]],
        atol=tolerance[precision]
    )


class TestGenUniform:
    @staticmethod
    def get_test_kernel(generator):
        @cuda.jit
        def kernel(vector, start, end, state):
            pos = get_thread_id()
            if pos < vector.size:
                vector[pos] = generator(start, end, state)

        return kernel

    def test_gen_uniform(self, precision):
        a = np.zeros(shape=(4,))
        kernel = KernelWrapper(
            self.get_test_kernel(get_uniform_generator(precision)),
            precision, (0,), device=False, state=State(n=10, seed=7)
        )
        kernel[2, 2](a, 0, 1)
        assert (a**2).sum() > 1.5
        npt.assert_allclose(
            a, [0.7796595, 0.09631481, 0.97113137, 0.53298037],
            atol=tolerance[precision]
        )

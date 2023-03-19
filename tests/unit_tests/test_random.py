import numpy as np
import pytest

from sde import KernelWrapper, State
from sde.random import gen_normal_16, gen_normal_32, gen_normal_64

from .utils import precision

generators = {
    'float16': gen_normal_16,
    'float32': gen_normal_32,
    'float64': gen_normal_64,
}


@pytest.mark.parametrize('shape', [(3, 4), (1, 2), (2, 1)])
def test_gen_something(precision, shape: tuple[int]):
    a = np.zeros(shape=shape)
    kernel = KernelWrapper(
        generators[precision], precision, (0,), device=True, n_args=1,
        state=State(n=shape[0] * shape[1], seed=7)
    )
    kernel[1, 1](a)
    assert (a**2).sum() > 2

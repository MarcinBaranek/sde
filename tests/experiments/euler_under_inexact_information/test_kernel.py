import numpy as np
import pytest

from experiments.euler_under_inexact_information.kernel import ExperimentKernel

from sde import KernelWrapper, State

kernel_class = ExperimentKernel(
    N=10,
    t_0=0.0,
    T=1.0,
    point_dim=2,
    wiener_dim=3,
    precision='float64',
    exact_factor=10,
    delta=0.,
    alpha=0.5, beta=0.5
)

ths = 16
blocks = 8


def get_kernel(kernel_cls):
    return KernelWrapper(
        kernel_cls.get_kernel(), precision='float64', outs=(0,),
        state=State(n=ths * blocks, seed=7)
    )


@pytest.mark.parametrize(
    'delta, n, error',
    [
        (0.0, 10, 2.1251780675182345e-4),
        (0.1, 10, 1.2138254833442927e-3),
        (1.e-2, 10, 2.1897809358253714e-4),
        (1.e-3, 10, 2.1276546818379454e-4),
        (1.e-4, 10, 2.125386049716739e-4),
        (1.e-5, 10, 2.1251984686282822e-4),
        (1.e-6, 10, 2.125180103657774e-4),
        (1.e-4, 10**2, 1.5733011442089342e-06),
        (1.e-4, 10**3, 1.5000606649548374e-08),
        (1.e-4, 10**4, 6.676882783908631e-10),
        (1.e-2, 10**2, 8.457221550774012e-06),
        (1.e-2, 10**3, 7.3710829452562485e-06),
        (1.e-2, 10**4, 5.229131276355687e-06),
        (1.e-1, 10**2, 7.659066536572623e-2),
        (1.e-1, 10**3, 1.8507634092538795e-2),
        (1.e-1, 10**4, 8.913640693014418e-4),
        (0.5, 10**1, 4.0423557290925825),
        (0.5, 10**2, 30.42480677765809),
        (0.5, 10**3, 16.17361502379941),
        (0.5, 10**4, 1.800475601751803),
    ]
)
def test_delta_n_impact(delta, n, error):
    result = np.zeros(shape=(ths * blocks,))
    start_point = np.array([[1.0], [0.3], [-0.9]])
    kernel = get_kernel(
        ExperimentKernel(
            N=n,
            t_0=0.0,
            T=1.0,
            point_dim=2,
            wiener_dim=3,
            precision='float64',
            exact_factor=10,
            delta=delta,
            alpha=0.5, beta=0.5
        )
    )
    kernel[ths, blocks](result, start_point)
    err = result.mean()
    assert err == pytest.approx(error, abs=1.e-12)


@pytest.mark.parametrize(
    'delta, n, error',
    [
        (0.0, 10, 1.5107010070916852e-4),
        (0.1, 10, 8.605301631622422e-4),
        (1.e-2, 10, 1.5879995902236916e-4),
        (1.e-3, 10, 1.5137206691472014e-4),
        (1.e-4, 10, 1.510956301889215e-4),
        (1.e-5, 10, 1.5107260701795042e-4),
        (1.e-6, 10, 1.5107035087368547e-4),
        (1.e-4, 10**2, 1.8264623833364256e-06),
        (1.e-4, 10**3, 1.846834762592251e-08),
        (1.e-4, 10**4, 6.676882783908631e-10),
        (1.e-2, 10**2, 7.026337963749667e-06),
        (1.e-2, 10**3, 5.087904411307764e-06),
        (1.e-2, 10**4, 5.229131276355687e-06),
        (1.e-1, 10**2, 7.358925246450845e-4),
        (1.e-1, 10**3, 8.872353066314597e-4),
        (1.e-1, 10**4, 8.913640693014418e-4),
        (0.5, 10**1, 0.9813150817997175),
        (0.5, 10**2, 1.016606154046698),
        (0.5, 10**3, 1.8107933140523742),
        (0.5, 10**4, 1.800475601751803),
    ]
)
def test_delta_impact(delta, n, error):
    result = np.zeros(shape=(ths * blocks,))
    start_point = np.array([[1.0], [0.3], [-0.9]])
    kernel = get_kernel(
        ExperimentKernel(
            N=n,
            t_0=0.0,
            T=1.0,
            point_dim=2,
            wiener_dim=3,
            precision='float64',
            exact_factor=int(10**5 / n),
            delta=delta,
            alpha=0.5, beta=0.5
        )
    )
    kernel[ths, blocks](result, start_point)
    err = result.mean()
    assert err == pytest.approx(error, abs=1.e-12)
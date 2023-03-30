from dataclasses import dataclass, field

from numba import cuda

from sde.utils import get_thread_id
from sde.core.tools import fill, write_from_to
from sde.core.linear_algebra import add, multiply_matrix_by_scalar,\
    add_inplace, sse
from sde.random import get_uniform_generator, gen_uniform_64, gen_normal_64


from .user_tests_functions import pw
from .euler_steps import get_euler_step, get_euler_step_alg


@dataclass
class ExperimentKernel:
    N: int
    t_0: float = 0.0
    T: float = 1.0
    point_dim: int = 2
    wiener_dim: int = 3
    precision: str = 'float64'  # Currently, other are not supported!
    exact_factor: int = 10
    dt_for_alg: float = field(init=False, default=0.)
    exact_dt: float = field(init=False, default=0.)
    delta: float = 0.001
    alpha: float = 0.5
    beta: float = 0.5

    def __post_init__(self):
        self.dt_for_alg = (self.T - self.t_0) / self.N
        self.exact_dt = self.dt_for_alg / self.exact_factor

    def get_kernel(self):
        uniform_generator = get_uniform_generator(self.precision)
        euler_step_for_alg = get_euler_step_alg(
            self.dt_for_alg, self.point_dim, self.wiener_dim,
            self.precision
        )
        euler_step = get_euler_step(
            self.exact_dt, self.point_dim, self.wiener_dim,
            'float64'
        )
        wiener_dim = self.wiener_dim
        precision = self.precision
        t_0 = self.t_0
        N = self.N
        exact_factor = self.exact_factor
        exact_dt = self.exact_dt
        dt_for_alg = self.dt_for_alg
        delta = self.delta
        alpha = self.alpha
        beta = self.beta
        dim = self.point_dim

        @cuda.jit
        def kernel(result, initial_point, state):
            pos = get_thread_id()
            if pos < result.size:

                wiener = cuda.local.array(
                    shape=(wiener_dim, 1), dtype='float64'
                )
                wiener_for_alg = cuda.local.array(
                    shape=(wiener_dim, 1), dtype=precision
                )
                wiener_old_for_alg = cuda.local.array(
                    shape=(wiener_dim, 1), dtype=precision
                )
                dw = cuda.local.array(
                    shape=(wiener_dim, 1), dtype='float64'
                )
                dw_for_alg = cuda.local.array(
                    shape=(wiener_dim, 1), dtype=precision
                )
                temp_point = cuda.local.array(shape=(dim, 1), dtype='float64')
                temp_point_for_alg = cuda.local.array(
                    shape=(dim, 1), dtype=precision
                )

                write_from_to(initial_point, temp_point)
                write_from_to(initial_point, temp_point_for_alg)

                fill(wiener, 0)
                fill(dw, 0)
                fill(dw_for_alg, 0)
                fill(wiener_old_for_alg, 0)
                fill(wiener_for_alg, 0)

                counter = 0
                cur_time = t_0
                cur_time_for_alg = t_0
                for i in range(N * exact_factor):
                    counter = counter + 1
                    xi_exact = gen_uniform_64(
                        cur_time, cur_time + exact_dt, state
                    )
                    gen_normal_64(dw, state)
                    multiply_matrix_by_scalar(dw, exact_dt)
                    add_inplace(wiener, dw)

                    euler_step(temp_point, xi_exact, cur_time, dw)
                    if counter % exact_factor == 0:
                        counter = 0
                        pw(
                            cur_time_for_alg, wiener, delta,
                            wiener_for_alg, alpha, beta, state
                        )
                        multiply_matrix_by_scalar(wiener_old_for_alg, -1)
                        add(wiener_for_alg, wiener_old_for_alg, dw_for_alg)
                        xi_for_alg = uniform_generator(
                            cur_time_for_alg,
                            cur_time_for_alg
                            + dt_for_alg, state
                        )
                        euler_step_for_alg(
                            temp_point_for_alg, xi_for_alg, cur_time_for_alg,
                            dw_for_alg, delta, state
                        )

                        write_from_to(wiener_for_alg, wiener_old_for_alg)
                result[pos] = sse(temp_point, temp_point_for_alg)
        return kernel

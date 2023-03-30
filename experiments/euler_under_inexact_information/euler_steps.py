from numba import cuda

from sde.core.linear_algebra import multiply_matrix_by_scalar, \
    add_inplace, multiply_matrix
from .user_tests_functions import pa, pb, a, b


def get_euler_step_alg(dt, point_dim, wiener_dim, precision):
    @cuda.jit(device=True)
    def euler_step_alg(point, xi, cur_time, dw_for_alg, delta, state):
        tmp = cuda.local.array(shape=(point_dim, 1), dtype=precision)
        tmp_drift_res = cuda.local.array(shape=(point_dim, 1), dtype=precision)
        tmp_drift = cuda.local.array(
            shape=(point_dim, wiener_dim), dtype=precision
        )
        pa(xi, point, delta, tmp, state)
        multiply_matrix_by_scalar(tmp, dt)

        pb(cur_time, point, delta, tmp_drift, state)
        multiply_matrix(tmp_drift, dw_for_alg, tmp_drift_res)
        add_inplace(point, tmp)
        add_inplace(point, tmp_drift_res)

    return euler_step_alg


def get_euler_step(dt, point_dim, wiener_dim, precision):
    @cuda.jit(device=True)
    def euler_step(point, xi, cur_time, dw):
        tmp = cuda.local.array(shape=(point_dim, 1), dtype=precision)
        tmp_drift_res = cuda.local.array(shape=(point_dim, 1), dtype=precision)
        tmp_drift = cuda.local.array(
            shape=(point_dim, wiener_dim), dtype=precision
        )
        a(xi, point, tmp)
        multiply_matrix_by_scalar(tmp, dt)

        b(cur_time, point, tmp_drift)
        multiply_matrix(tmp_drift, dw, tmp_drift_res)
        add_inplace(point, tmp)
        add_inplace(point, tmp_drift_res)

    return euler_step

from numba import cuda
import math

from sde.random import gen_uniform_64
from sde.core import norm


@cuda.jit(device=True)
def a(t, x, result):
    result[0, 0] = 0.5 * t * math.sin(10 * x[0, 0])
    result[1, 0] = 0.5 * math.cos(7 * x[1, 0])


@cuda.jit(device=True)
def pa(t, x, delta, result, state):
    a(t, x, result)
    for i in range(result.shape[0]):
        result[i, 0] += \
            result[i, 0] * delta * gen_uniform_64(-1, 1, state)


@cuda.jit(device=True)
def b(t, x, result):
    result[0, 0] = t * x[0, 0]
    result[0, 1] = t * x[1, 0]
    result[0, 2] = math.sin(x[1, 0])
    result[1, 0] = math.cos(x[0, 0])
    result[1, 1] = x[1, 0]
    result[1, 2] = -x[0, 0]


@cuda.jit(device=True)
def pb(t, x, delta, result, state):
    b(t, x, result)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] += \
                result[i, j] * delta * gen_uniform_64(-1, 1, state)


@cuda.jit(device=True)
def pw(t, w, delta, result, alpha, beta, state):
    # tfu = t ** alpha
    err = norm(w)
    value = math.sin(1_000 * err)
    signum = 1 if value > 0 else -1
    value = signum * value
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            result[i, j] = w[i, j] + signum * delta * value ** beta

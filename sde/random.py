import numba
from numba import cuda
from numba.cuda.random import (
    xoroshiro128p_normal_float32, xoroshiro128p_normal_float64
)


@cuda.jit(device=True)
def gen_normal_16(place_holder, state):
    thread_id = cuda.grid(1)
    for i in range(place_holder.shape[0]):
        for j in range(place_holder.shape[1]):
            place_holder[i, j] =\
                numba.float16(xoroshiro128p_normal_float32(state, thread_id))


@cuda.jit(device=True)
def gen_normal_32(place_holder, state):
    thread_id = cuda.grid(1)
    for i in range(place_holder.shape[0]):
        for j in range(place_holder.shape[1]):
            place_holder[i, j] = xoroshiro128p_normal_float32(state, thread_id)


@cuda.jit(device=True)
def gen_normal_64(place_holder, state):
    thread_id = cuda.grid(1)
    for i in range(place_holder.shape[0]):
        for j in range(place_holder.shape[1]):
            place_holder[i, j] = xoroshiro128p_normal_float64(state, thread_id)

from numba import cuda
from numba.core.types import float16
from numba.cuda.random import (
    xoroshiro128p_normal_float32, xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float32, xoroshiro128p_uniform_float64
)


@cuda.jit(device=True)
def gen_uniform_16(start, end, state):
    thread_id = cuda.grid(1)
    return start + (end - start) * float16(
        xoroshiro128p_uniform_float32(state, thread_id)
    )


@cuda.jit(device=True)
def gen_uniform_32(start, end, state):
    thread_id = cuda.grid(1)
    return start + (end - start) * xoroshiro128p_uniform_float32(
        state, thread_id
    )


@cuda.jit(device=True)
def gen_uniform_64(start, end, state):
    thread_id = cuda.grid(1)
    return start + (end - start) * xoroshiro128p_uniform_float64(
        state, thread_id
    )


@cuda.jit(device=True)
def gen_normal_16(place_holder, state):
    thread_id = cuda.grid(1)
    for i in range(place_holder.shape[0]):
        for j in range(place_holder.shape[1]):
            place_holder[i, j] =\
                float16(xoroshiro128p_normal_float32(state, thread_id))


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


def get_normal_generator(precision: str):
    if precision == 'float16':
        return gen_normal_16
    elif precision == 'float32':
        return gen_normal_32
    elif precision == 'float64':
        return gen_normal_64
    else:
        raise ValueError(f'Unknown precision: {precision}')


def get_uniform_generator(precision: str):
    if precision == 'float16':
        return gen_uniform_16
    elif precision == 'float32':
        return gen_uniform_32
    elif precision == 'float64':
        return gen_uniform_64
    else:
        raise ValueError(f'Unknown precision: {precision}')

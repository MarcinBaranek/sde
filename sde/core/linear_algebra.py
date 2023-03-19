from numba import cuda
from numpy.typing import NDArray


@cuda.jit(device=True)
def add(left: NDArray, right: NDArray, result: NDArray) -> None:
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            result[i, j] = left[i, j] + right[i, j]


@cuda.jit(device=True)
def add_inplace(left: NDArray, right: NDArray):
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            left[i, j] += right[i, j]

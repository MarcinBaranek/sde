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


@cuda.jit(device=True)
def multiply_matrix(matrix_a, matrix_b, result):
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_b.shape[1]):
            temp = 0
            for k in range(matrix_a.shape[1]):
                temp += matrix_a[i, k] * matrix_b[k, j]
            result[i, j] = temp


@cuda.jit(device=True)
def multiply_matrix_by_scalar(matrix, scalar):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = matrix[i, j] * scalar

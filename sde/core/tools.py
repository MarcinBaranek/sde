from numba import cuda


@cuda.jit(device=True)
def write_from_to(source, destination):
    """Just copy source to destination (should have the same shape)."""
    for i in range(destination.shape[0]):
        for j in range(destination.shape[1]):
            destination[i, j] = source[i, j]


@cuda.jit(device=True)
def fill(vector, value: float):
    for i in range(vector.shape[0]):
        for j in range(vector.shape[1]):
            vector[i, j] = value

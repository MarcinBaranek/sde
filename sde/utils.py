from numba import cuda


@cuda.jit(device=True)
def get_thread_id() -> int:
    """Returns thread id."""
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    return tx + ty * bw

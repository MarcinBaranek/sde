from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from numba import cuda
import numpy as np

from .config import precisions_map
from .wrapped_device_function import get_device_function_wrapper


@dataclass
class KernelWrapper:
    kernel: Callable
    precision: str | type = 'float32'
    outs: tuple[int] = ()
    device: bool = False
    n_args: int = 2

    def __post_init__(self):
        self.get_precision()

    def get_precision(self) -> type:
        if self.precision not in precisions_map:
            raise ValueError(
                f'got unknown precision {self.precision}. '
                f'Allowed precisions: {tuple(precisions_map.keys())}'
            )
        return precisions_map[self.precision]

    def send_args_to_device(self, *args):
        precision = self.get_precision()
        return tuple(
            cuda.to_device(arg.astype(precision))
            if isinstance(arg, np.ndarray) else arg
            for arg in args
        )

    def __getitem__(self, grid):
        def caller(*args):
            kernel = get_device_function_wrapper(self.kernel, self.n_args)\
                if self.device else self.kernel
            device_args = self.send_args_to_device(*args)
            kernel[grid](*device_args)
            for idx in self.outs:
                result = np.empty(
                    shape=args[idx].shape, dtype=device_args[idx].dtype
                )
                device_args[idx].copy_to_host(result)
                args[idx][:] = result[:]
        return caller

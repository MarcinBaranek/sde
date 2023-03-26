from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Iterable

import numpy as np
from numba import cuda
from numba.cuda import random

from .config import precisions_map
from .wrapped_device_function import get_device_function_wrapper


@dataclass
class State:
    n: int
    seed: int = 7
    device_state: Optional[cuda.device_array] = None

    def __post_init__(self):
        self.device_state = random.create_xoroshiro128p_states(
            self.n, self.seed
        )


@dataclass
class BasePrecisionChecker:

    @staticmethod
    def get_precision(precision: str) -> type:
        if precision not in precisions_map:
            raise ValueError(
                f'got unknown precision {precision}. '
                f'Allowed precisions: {tuple(precisions_map.keys())}'
            )
        return precisions_map[precision]


@dataclass
class ArgPrecision(BasePrecisionChecker):
    index: int
    precision: str | type = 'float32'

    def __post_init__(self):
        if not isinstance(self.index, int):
            raise TypeError(
                f'index should be int, got {self.index} '
                f'with type: {type(self.index)}'
            )
        if self.index < 0:
            raise ValueError(
                f'index should be grater or equal 0, got {self.index}'
            )
        self.precision = self.get_precision(self.precision)


@dataclass
class KernelWrapper(BasePrecisionChecker):
    kernel: Callable
    precision: str | type = 'float32'
    outs: tuple[int, ...] = ()
    device: bool = False
    n_args: int = 2
    state: Optional[State] = None
    in_precisions: Optional[Iterable[ArgPrecision]] = None

    def __post_init__(self):
        self.get_precision(self.precision)
        if self.state:
            self.n_args += 1
        if self.in_precisions is None:
            return
        for arg_precision in self.in_precisions:
            if arg_precision.index >= self.n_args - 1:
                raise ValueError(
                    f'index in ArgPrecision is to high. Got: '
                    f'{arg_precision.index}. Max allowed index is one less '
                    f'thant `n_args`: {self.n_args - 1}.'
                )

    def select_precision(self, index, default):
        if self.in_precisions is None:
            return default
        for arg_precision in self.in_precisions:
            if arg_precision.index == index:
                return arg_precision.precision
        return default

    def send_args_to_device(self, *args):
        precision = self.get_precision(self.precision)
        device_args = []
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                device_args.append(arg)
                continue
            device_args.append(
                cuda.to_device(arg.astype(self.select_precision(i, precision)))
            )
        if self.state is not None:
            device_args.append(self.state.device_state)
            return device_args
        return device_args

    def copy_results_to_host(self, args, device_args):
        for idx in self.outs:
            result = np.empty(
                shape=args[idx].shape, dtype=device_args[idx].dtype
            )
            device_args[idx].copy_to_host(result)
            args[idx][:] = result[:]

    def __getitem__(self, grid):
        def caller(*args):
            kernel = get_device_function_wrapper(self.kernel, self.n_args) \
                if self.device else self.kernel
            device_args = self.send_args_to_device(*args)
            kernel[grid](*device_args)
            self.copy_results_to_host(args, device_args)

        return caller

import numpy as np

from .kernel import ExperimentKernel
from sde import KernelWrapper, State
from itertools import product
from tqdm import tqdm


random_seed = 7
threads_number = 512
blocks_number = 32
max_power = 20
set_of_delta = [
    0.9, 0.8, 0.5, 0.3, 0.1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 0.
]
set_of_beta = [0.99, 0.75, 0.5, 0.25, 0.1, 1.e-3, 1.e-6, 1.e-12]
n_for_exact_solution = 10 * 2 ** max_power
n_sets = [2 ** k for k in range(3, max_power + 1)]
start_point = np.array([[1.3, -0.3]])
base_kernel_parameters = {
    't_0': 0.0,
    'T': 1.0,
    'point_dim': 2,
    'wiener_dim': 3,
    'precision': 'float64',
    # need to fill: N, exact_factor, delta, alpha, beta
    # alpha doesn't play any significant role
}
result_file = 'res.txt'
number_of_trajectories = threads_number * blocks_number
random_state = State(number_of_trajectories, seed=7)

with open(result_file, 'a+') as file:
    file.write(f'n,log_n,error,delta,beta')
for n in tqdm(n_sets):
    for delta, beta in product(set_of_delta, set_of_beta):
        print(f'Computing experiment for delta={delta}, beta={beta}.')
        experimental_kernel = ExperimentKernel(
            N=n,
            exact_factor=int(n_for_exact_solution / n),
            delta=delta,
            beta=beta,
            alpha=1.,   # doesn't matter for now
            **base_kernel_parameters
        )
        kernel = KernelWrapper(
            experimental_kernel.get_kernel(), precision='float64', outs=(0,),
            device=False, state=random_state
        )
        errors = np.zeros(number_of_trajectories)
        kernel[threads_number, blocks_number](errors, start_point)
        with open('res.txt', 'a+') as file:
            # file.write(f'n,log_n,error,delta,beta')
            file.write(f'{n},{np.log10(n)},{errors.mean()},{delta},{beta}')

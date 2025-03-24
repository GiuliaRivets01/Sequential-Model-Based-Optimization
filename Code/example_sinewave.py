import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import typing

from assignment import SequentialModelBasedOptimization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_configurations', type=int, default=16)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/Downloads')
    parser.add_argument('--plot_resolution', type=int, default=128)
    parser.add_argument('--problem_size', type=int, default=1)

    return parser.parse_args()


def optimizee(x: float) -> typing.Union[float, np.array]:
    return np.sin(x[0])


def sample_configurations(n: int):
        x1 = np.random.uniform(X_MIN, X_MAX, (n, 1))
        return x1


def sample_initial_configurations(n: int) -> typing.List[typing.Tuple[np.array, float]]:
    configs = sample_configurations(n)
    return [(x, optimizee(x)) for x in configs]



if __name__ == '__main__':
    args = parse_args()

    X_MIN = -np.pi * args.problem_size
    X_MAX = np.pi * args.problem_size

    smbo = SequentialModelBasedOptimization()
    print(args.initial_configurations)
    init_configs = sample_initial_configurations(args.initial_configurations) #(16)
    smbo.initialize(init_configs)
    smbo.fit_model()


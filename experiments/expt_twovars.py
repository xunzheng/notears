"""Experiments for linear Gaussian SEM with two variables."""
from notears import notears, utils
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os


def main():
    utils.set_random_seed(123)

    num_graph = 1000
    num_data_per_graph = 1

    n, d, s0, graph_type, sem_type = np.inf, 2, 1, 'ER', 'gauss'

    # equal variance
    w_ranges = ((-2.0, -0.5), (0.5, 2.0))
    noise_scale = [1., 1.]
    expt_name = 'equal_var'
    run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)

    # large a
    w_ranges = ((-2.0, -1.1), (1.1, 2.0))
    noise_scale = [1., 0.15]
    expt_name = 'large_a'
    run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)

    # small a
    w_ranges = ((-0.9, -0.5), (0.5, 0.9))
    noise_scale = [1, 0.15]
    expt_name = 'small_a'
    run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)


def run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name):
    os.mkdir(expt_name)
    os.chmod(expt_name, 0o777)
    perf = defaultdict(list)
    for ii in tqdm(range(num_graph)):
        B_true = utils.simulate_dag(d, s0, graph_type)
        W_true = utils.simulate_parameter(B_true, w_ranges=w_ranges)
        W_true_fn = os.path.join(expt_name, f'graph{ii:05}_W_true.csv')
        np.savetxt(W_true_fn, W_true, delimiter=',')
        for jj in range(num_data_per_graph):
            X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
            X_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_X.csv')
            np.savetxt(X_fn, X, delimiter=',')
            # notears
            W_notears = notears.notears_linear_l1(X, lambda1=0, loss_type='l2')
            assert utils.is_dag(W_notears)
            W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
            np.savetxt(W_notears_fn, W_notears, delimiter=',')
            # eval
            B_notears = (W_notears != 0)
            acc = utils.count_accuracy(B_true, B_notears)
            for metric in acc:
                perf[metric].append(acc[metric])
    # print stats
    print(expt_name)
    for metric in perf:
        print(metric, f'{np.mean(perf[metric]):.4f}', '+/-', f'{np.std(perf[metric]):.4f}')


if __name__ == '__main__':
    main()


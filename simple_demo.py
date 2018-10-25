"""Demo for the simple 50-line version of notears algorithm.

Steps:
1. Simulate a random graph with d nodes.
2. Simulate n samples from the SEM.
3. Run the simple notears algorithm.
4. Evaluate the predictive accuracy.

Note: this unregularized notears algorithm works in n >> d.
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
import glog as log
import networkx as nx

import utils


def notears_simple(X: np.ndarray,
                   max_iter: int = 100,
                   h_tol: float = 1e-8,
                   w_threshold: float = 0.3) -> np.ndarray:
    """Solve min_W ell(W; X) s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X: [n,d] sample matrix
        max_iter: max number of dual ascent steps
        h_tol: exit if |h(w)| <= h_tol
        w_threshold: fixed threshold for edge weights

    Returns:
        W: [d,d] solution
    """
    def _h(w):
        W = w.reshape([d, d])
        return np.trace(slin.expm(W * W)) - d

    def _func(w):
        W = w.reshape([d, d])
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
        h = _h(W)
        return loss + 0.5 * rho * h * h + alpha * h

    def _grad(w):
        W = w.reshape([d, d])
        loss_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W)
        E = slin.expm(W * W)
        obj_grad = loss_grad + (rho * (np.trace(E) - d) + alpha) * E.T * W * 2
        return obj_grad.flatten()

    n, d = X.shape
    w, w_new = np.zeros(d * d), np.zeros(d * d)
    rho, alpha, h, h_new = 1.0, 0.0, np.inf, np.inf
    bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
    for _ in range(max_iter):
        while rho < 1e+20:
            sol = sopt.minimize(_func, w, method='L-BFGS-B', jac=_grad, bounds=bnds)
            w_new = sol.x
            h_new = _h(w_new)
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol:
            break
    w[np.abs(w) < w_threshold] = 0
    return w.reshape([d, d])


if __name__ == '__main__':
    # configurations
    n, d = 1000, 10
    graph_type, degree, sem_type = 'erdos-renyi', 4, 'linear-gauss'
    log.info('Graph: %d node, avg degree %d, %s graph', d, degree, graph_type)
    log.info('Data: %d samples, %s SEM', n, sem_type)

    # graph
    log.info('Simulating graph ...')
    G = utils.simulate_random_dag(d, degree, graph_type)
    log.info('Simulating graph ... Done')

    # data
    log.info('Simulating data ...')
    X = utils.simulate_sem(G, n, sem_type)
    log.info('Simulating data ... Done')

    # solve optimization problem
    log.info('Solving equality constrained problem ...')
    W_est = notears_simple(X)
    G_est = nx.DiGraph(W_est)
    log.info('Solving equality constrained problem ... Done')

    # evaluate
    fdr, tpr, fpr, shd, nnz = utils.count_accuracy(G, G_est)
    log.info('Accuracy: fdr %f, tpr %f, fpr %f, shd %d, nnz %d',
             fdr, tpr, fpr, shd, nnz)

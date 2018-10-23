"""The 50 line version of the simple NOTEARS algorithm.

The full version is in ... TODO
"""

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt


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

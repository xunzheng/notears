#!/usr/bin/env python3
from notears import nonlinear, utils
import torch
import numpy as np
import argparse


def main(args):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    X = np.loadtxt(args.X_path, delimiter=',')
    n, d = X.shape
    model = nonlinear.NotearsMLP(dims=[d, args.hidden, 1], bias=True)
    W_est = nonlinear.notears_nonlinear(model, X, lambda1=args.lambda1, lambda2=args.lambda2)
    assert utils.is_dag(W_est)
    np.savetxt(args.W_path, W_est, delimiter=',')


def parse_args():
    parser = argparse.ArgumentParser(description='Run NOTEARS algorithm')
    parser.add_argument('X_path', type=str, help='n by p data matrix in csv format')
    parser.add_argument('--hidden', type=int, default=10, help='Number of hidden units')
    parser.add_argument('--lambda1', type=float, default=0.01, help='L1 regularization parameter')
    parser.add_argument('--lambda2', type=float, default=0.01, help='L2 regularization parameter')
    parser.add_argument('--W_path', type=str, default='W_est.csv', help='p by p weighted adjacency matrix of estimated DAG in csv format')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


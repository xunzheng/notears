#pragma once

// #define EIGEN_USE_MKL_ALL       // uncomment if available
// #define EIGEN_USE_BLAS          // uncomment if available
// #define EIGEN_USE_LAPACKE       // uncomment if available

#include "Eigen/Core"
#include "unsupported/Eigen/MatrixFunctions"

#include "problem.h"

using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


int F_eval(int n, int d, double *w_ptr, double *X_ptr, double lambda1, double *F_ptr) {
    Eigen::Map<RowMajorMatrixXd> X(X_ptr, n, d);
    Eigen::Map<RowMajorMatrixXd> W(w_ptr, d, d);
    double loss = 0.5 / n * (X * (Eigen::MatrixXd::Identity(d, d) - W)).squaredNorm();
    double l1norm = W.array().abs().sum();
    *F_ptr = loss + lambda1 * l1norm;
    return 0;
}

int h_eval(int d, double *w_ptr, double *h_ptr) {
    Eigen::Map<RowMajorMatrixXd> W(w_ptr, d, d);
    *h_ptr = W.cwiseAbs2().exp().trace() - d;
    return 0;
}

class NotearsProblem : public Problem {
public:
    int n, d;
    double *X_ptr;
    double rho, alpha;

    // Constructor
    NotearsProblem(int n, int d, double *X_ptr, double rho, double alpha) 
        : n(n), d(d), X_ptr(X_ptr), rho(rho), alpha(alpha) {}

    // Evaluate function at w, store in f_ptr
    int func(double *w_ptr, double *f_ptr) {
        Eigen::Map<RowMajorMatrixXd> X(X_ptr, n, d);
        Eigen::Map<RowMajorMatrixXd> W(w_ptr, d, d);
        double loss, h;
        loss = 0.5 / n * (X * (Eigen::MatrixXd::Identity(d, d) - W)).squaredNorm();
        h_eval(d, w_ptr, &h);
        *f_ptr = loss + 0.5 * rho * h * h + alpha * h;
        return 0;
    }
    
    // Evaluate gradient at w, store in g_ptr
    int grad(double *w_ptr, double *g_ptr) {
        Eigen::Map<RowMajorMatrixXd> X(X_ptr, n, d);
        Eigen::Map<RowMajorMatrixXd> W(w_ptr, d, d);
        Eigen::Map<RowMajorMatrixXd> G(g_ptr, d, d);
        G.setZero();
        G += - 1.0 / n * (X.transpose() * X * (Eigen::MatrixXd::Identity(d, d) - W));
        Eigen::MatrixXd E = W.cwiseAbs2().exp();
        G += (rho * (E.trace() - d) + alpha) * E.transpose().cwiseProduct(W) * 2;
        return 0;
    }
};
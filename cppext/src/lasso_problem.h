#pragma once

// #define EIGEN_USE_MKL_ALL       // uncomment if available
// #define EIGEN_USE_BLAS          // uncomment if available
// #define EIGEN_USE_LAPACKE       // uncomment if available

#include "Eigen/Core"

#include "problem.h"

using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;


// A lasso problem to test prox quasi newton.
class LassoProblem : public Problem {
public:
    int n, p;
    double *X_ptr, *y_ptr;

    LassoProblem(int n, int p, double *X_ptr, double *y_ptr)
        : n(n), p(p), X_ptr(X_ptr), y_ptr(y_ptr) {}

    int func(double *w_ptr, double *f_ptr) {
        Eigen::Map<RowMajorMatrixXd> X(X_ptr, n, p);
        Eigen::Map<Eigen::VectorXd> y(y_ptr, n);
        Eigen::Map<Eigen::VectorXd> w(w_ptr, p);
        *f_ptr = 0.5 / n * (y - X * w).squaredNorm();
        return 0;
    }

    int grad(double *w_ptr, double *g_ptr) {
        Eigen::Map<RowMajorMatrixXd> X(X_ptr, n, p);
        Eigen::Map<Eigen::VectorXd> y(y_ptr, n);
        Eigen::Map<Eigen::VectorXd> w(w_ptr, p);
        Eigen::Map<Eigen::VectorXd> g(g_ptr, p);
        g = - X.transpose() * (y - X * w) / n;
        return 0;
    }
};
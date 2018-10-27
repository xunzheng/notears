// #define EIGEN_USE_MKL_ALL       // uncomment if available
// #define EIGEN_USE_BLAS          // uncomment if available
// #define EIGEN_USE_LAPACKE       // uncomment if available

#include "Eigen/Core"
#include "Eigen/LU"

#include <iostream>
#include <iomanip>
#include <vector>
#include <deque>
#include <chrono>
#include <limits>
#include <random>
#include <algorithm>

#include "problem.h"


double soft_threshold(double x, double t) {
    if (x > t) {
        return x - t;
    } else if (x < - t) {
        return x + t;
    } else {
        return 0.0;
    }
}

class ProxQN {
public:
    int p; // size of w
    double *w_ptr; // optimization variable
    double lambda1; // parameter on l1 regularization
    int max_outer = 15000;
    int max_inner = 10;
    double ftol = 2.220446049250313e-09;
    double gtol = 1e-5;
    double c1 = 1e-3;
    double min_step = 1e-20;
    int m = 10;
    int damped = 1;
    int verbose = 0;

    ProxQN(int p, double *w_ptr, double lambda1) 
        : p(p), w_ptr(w_ptr), lambda1(lambda1) {}

    double l1norm(const std::vector<int> &active) {
        double l1 = 0.0;
        for (int j : active) {
            l1 += fabs(w_ptr[j]);
        }
        return l1;
    }

    int minimize(Problem *problem);
};

int ProxQN::minimize(Problem *problem) {
    Eigen::Map<Eigen::VectorXd> w(w_ptr, p);

    double gamma = 1.0;
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(p, 2 * m); // normal order
    Eigen::MatrixXd Qhat = Eigen::MatrixXd::Zero(p, 2 * m);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(p, m); // col mod order
    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(p, m);
    std::deque<std::deque<double>> SS, L; // [m,m], normal order
    std::deque<double> Ddiag; // [m,m]

    // active set
    std::vector<int> active;
    for (int j = 0; j < p; ++j) {
        active.push_back(j);
    }
    std::vector<int> active_prev, active_shuffled;
    double subgrad, normsg0, normsg, absg;  // subgrad and their l1 norm
    double M = 0.0;
    double Mout = 0.01 * lambda1; //the tolerance for a variable to be active.
    // After initialization, it is the infinity norm of the subgradients.
    int epoch_iter = -1; // iteration number in each epoch
    int memo_size, two_memo_size, m2;
    double epoch_eps = gtol;
    // if (epoch_eps < 0.0001) {
    //     epoch_eps = 0.0001;
    // }

    // check termination criteria
    double stop_criteria;
    double f_value = 0.;
    if (problem->func(w_ptr, &f_value) < 0) {
        return -1;
    }
    double obj = f_value + lambda1 * w.array().abs().sum();
    Eigen::VectorXd g = Eigen::VectorXd::Zero(p);
    double *g_ptr = g.data();
    if (problem->grad(w_ptr, g_ptr) < 0) {
        return -1;
    }
    double obj_prev = std::numeric_limits<double>::infinity();

    // solve subproblem by cd
    Eigen::VectorXd dvec = Eigen::VectorXd::Zero(p); // descent direction
    Eigen::VectorXd dhat = Eigen::VectorXd::Zero(2 * m); // Q.shape[1]
    Eigen::VectorXd Bdiag = Eigen::VectorXd::Zero(p);
    double a, b, c, z;
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine rng(seed);
    //std::default_random_engine rng(0);
    int num_inner;

    // line search along dvec
    double eta = 1.0;
    double eta_prev, l1norm_before, l1norm_after, delta, obj_new;
    Eigen::VectorXd w_new = Eigen::VectorXd::Zero(p);

    // update lbfgs variables
    int kk;
    double sy, ss, sbs, theta, SS_new, L_new;
    Eigen::VectorXd g_new = Eigen::VectorXd::Zero(p);
    double *g_new_ptr = g_new.data();
    Eigen::VectorXd y = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd s = Eigen::VectorXd::Zero(p);
    Eigen::VectorXd bs = Eigen::VectorXd::Zero(p);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * m, 2 * m); // inv of R

    if (verbose) {
        std::cout << std::setw(6) << "outer"
                  << std::setw(20) << "F(w)"
                  << std::setw(20) << "|w|"
                  << std::setw(20) << "active"
                  << std::setw(20) << "normsg"
                  << std::setw(20) << "stop_criteria" << std::endl;
    }
    int outer = 0;
    while (outer < max_outer) {
        // active set
        M = 0.0;
        ++epoch_iter;
        if (epoch_iter < m) {
            memo_size = epoch_iter + 1;
            two_memo_size = 2 * epoch_iter;
            dhat.head(two_memo_size).setZero();
        } else {
            memo_size = m;
            two_memo_size = 2 * m;
            dhat.setZero();
        }
        for (int j : active) {
            dvec[j] = 0.0;
        }
        active_prev = active;
        active.clear();
        normsg = 0.0;
        for (int j : active_prev) {
            if (w[j] < -1.0e-20) {
                subgrad = g[j] - lambda1;
                active.push_back(j);
                normsg += fabs(subgrad);
                M = fmax(M, fabs(subgrad));
            } else if (w[j] > 1.0e-20) {
                subgrad = g[j] + lambda1;
                active.push_back(j);
                normsg += fabs(subgrad);
                M = fmax(M, fabs(subgrad));
            } else {
                absg = fabs(g[j]) - lambda1;
                subgrad = fmax(absg, 0.0);
                if (absg > - Mout) {
                    active.push_back(j);
                    normsg += fabs(subgrad);
                    M = fmax(M, fabs(subgrad));
                }
            }
        }

        // check termination criteria
        if (outer == 0) {
            normsg0 = normsg;
        }
        stop_criteria = normsg / normsg0;
        if (verbose) {
            std::cout << std::setw(6) << outer
                      << std::setw(20) << obj
                      << std::setw(20) << l1norm(active)
                      << std::setw(20) << active.size()
                      << std::setw(20) << normsg
                      << std::setw(20) << stop_criteria << std::endl;
        }
        if (stop_criteria < epoch_eps || eta < min_step) {
            if ((int)active_prev.size() == p && stop_criteria < gtol) {
                if (verbose) {
                    std::cout << "reached gtol, terminate at outer " << outer << std::endl;
                }
                break;
            } else if ((int)active_prev.size() == p && fabs(obj_prev - obj) / fmax(fabs(obj_prev), fabs(obj)) <= ftol) {
                if (verbose) {
                    std::cout << "reached ftol, terminate at outer " << outer << std::endl;
                }
                break;
            } else if ((int)active_prev.size() == p && eta < min_step) {
                if (verbose) {
                    std::cout << "small step size, terminate at outer " << outer << std::endl;
                }
                break;
            } else {
                if (verbose) {
                    std::cout << "new epoch starts" << std::endl;
                }
                --outer;
                active.clear();
                for (int j = 0; j < p; ++j) {
                    active.push_back(j);
                }
                Mout = 0.01 * lambda1;
                SS.clear();
                L.clear();
                Ddiag.clear();
                if (problem->grad(w_ptr, g_ptr) < 0) {
                    return -1;
                }
                epoch_iter = -1;
                epoch_eps /= 10.0;
                if (epoch_eps < gtol) {
                    epoch_eps = gtol;
                }
                continue;
            }
        }
        Mout = M;
        obj_prev = obj;

        // solve subproblem by cd, get optimal dvec
        active_shuffled = active;
        num_inner = int(p / active.size());
        if (num_inner > max_inner) {
            num_inner = max_inner;
        }
        if (epoch_iter < m) {
            for (int j : active) {
                Bdiag[j] = gamma - Q.row(j).head(two_memo_size).dot(Qhat.row(j).head(two_memo_size));
            }
            for (int inner = 0; inner < num_inner; ++inner) {
                std::shuffle(active_shuffled.begin(), active_shuffled.end(), rng);
                for (int j : active_shuffled) {
                    a = Bdiag[j];
                    b = g[j] + gamma * dvec[j] - Q.row(j).head(two_memo_size).dot(dhat.head(two_memo_size));
                    c = w[j] + dvec[j];
                    z = - c + soft_threshold(c - b / a, lambda1 / a);
                    dvec[j] += z;
                    dhat.head(two_memo_size) += z * Qhat.row(j).head(two_memo_size);
                }
            }
        } else {
            for (int j : active) {
                Bdiag[j] = gamma - Q.row(j).dot(Qhat.row(j));
            }
            for (int inner = 0; inner < num_inner; ++inner) {
                std::shuffle(active_shuffled.begin(), active_shuffled.end(), rng);
                for (int j : active_shuffled) {
                    a = Bdiag[j];
                    b = g[j] + gamma * dvec[j] - Q.row(j).dot(dhat);
                    c = w[j] + dvec[j];
                    z = - c + soft_threshold(c - b / a, lambda1 / a);
                    dvec[j] += z;
                    dhat += z * Qhat.row(j);
                }
            }
        }

        // line search along dvec
        eta = 1.0;
        l1norm_before = l1norm(active);
        for (int j : active) {
            w[j] += dvec[j];
        }
        l1norm_after = l1norm(active);
        delta = lambda1 * (l1norm_after - l1norm_before);
        for (int j : active) {
            delta += g[j] * dvec[j];
        }
        while (true) {
            if (problem->func(w_ptr, &f_value) < 0) {
                return -1;
            }
            obj_new = f_value + lambda1 * l1norm_after;
            if (obj_new <= obj + c1 * eta * delta || eta < min_step) {
                obj = obj_new;
                break;
            }
            eta_prev = eta;
            eta *= 0.5;
            //if (eta < min_step) {
            //    PyErr_SetString(PyExc_ValueError, "step size too small, try larger gtol");
            //    return -1;
            //}
            for (int j : active) {
                w[j] += (eta - eta_prev) * dvec[j];
            }
            l1norm_after = l1norm(active);
        }

        // update lbfgs variables
        if (problem->grad(w_ptr, g_new_ptr) < 0) {
            return -1;
        }
        for (int j : active) {
            y[j] = g_new[j] - g[j];
            s[j] = eta * dvec[j];
            g[j] = g_new[j];
        }
        sy = 0.0;
        ss = 0.0;
        for (int j : active) {
            sy += s[j] * y[j];
            ss += s[j] * s[j];
        }
        if (damped > 0) {
            sbs = 0.0;
            if (epoch_iter < m) {
                for (int j : active) {
                    bs[j] = gamma * eta * dvec[j] - eta * Q.row(j).head(two_memo_size).dot(dhat.head(two_memo_size));
                    sbs += eta * dvec[j] * bs[j];
                }
            } else {
                for (int j : active) {
                    bs[j] = gamma * eta * dvec[j] - eta * Q.row(j).dot(dhat);
                    sbs += eta * dvec[j] * bs[j];
                }
            }
            if (sy >= 0.2 * sbs) {
                theta = 1.0;
            } else {
                theta = (0.8 * sbs) / (sbs - sy);
            }
            sy = 0.0;
            for (int j : active) {
                sy += s[j] * (theta * y[j] + (1.0 - theta) * bs[j]);
            }
        }
        gamma = sy / ss;
        if (sy < 0.0) {
            PyErr_SetString(PyExc_ValueError, "sy < 0");
            return -1;
        }
        if (epoch_iter < m) {
            SS.push_back(std::deque<double>());
            L.push_back(std::deque<double>());
            for (int k = 0; k < epoch_iter; ++k) {
                SS_new = 0.0;
                L_new = 0.0;
                for (int j : active) {
                    SS_new += s[j] * S(j,k);
                    L_new += s[j] * Y(j,k);
                }
                SS[k].push_back(SS_new);
                SS.back().push_back(SS_new);
                L[k].push_back(0.0);
                L.back().push_back(L_new);
            }
        } else {
            SS.pop_front();
            L.pop_front();
            for (int k = 0; k < m - 1; ++k) {
                SS[k].pop_front();
                L[k].pop_front();
                L[k].push_back(0.0);
            }
            SS.push_back(std::deque<double>());
            L.push_back(std::deque<double>());
            for (int k = 1; k < m; ++k) {
                kk = (k + epoch_iter) % m;
                SS_new = 0.0;
                L_new = 0.0;
                for (int j : active) {
                    SS_new += s[j] * S(j,kk);
                    L_new += s[j] * Y(j,kk);
                }
                SS[k-1].push_back(SS_new);
                SS.back().push_back(SS_new);
                L.back().push_back(L_new);
            }
        }
        SS.back().push_back(ss);
        L.back().push_back(0.0);
        if (epoch_iter < m) {
            for (int j : active) {
                S(j,epoch_iter) = s[j];
                Y(j,epoch_iter) = y[j];
            }
        } else {
            Ddiag.pop_front();
            kk = epoch_iter % m;
            for (int j : active) {
                S(j,kk) = s[j];
                Y(j,kk) = y[j];
            }
        }
        Ddiag.push_back(sy);
        if (epoch_iter < m) {
            for (int j : active) {
                for (int k = 0; k < memo_size; ++k) {
                    Q(j,k) = gamma * S(j,k);
                    Q(j,memo_size+k) = Y(j,k);
                }
            }
        } else {
            for (int j : active) {
                for (int k = 0; k < memo_size; ++k) {
                    kk = (epoch_iter + k + 1) % m;
                    Q(j,k) = gamma * S(j,kk);
                    Q(j,memo_size+k) = Y(j,kk);
                }
            }
        }
        if (epoch_iter < m) {
            m2 = 2 * memo_size;
            A.topLeftCorner(m2, m2).setZero();
            for (int k1 = 0; k1 < memo_size; ++k1) {
                for (int k2 = 0; k2 < memo_size; ++k2) {
                    A(k1,k2) = gamma * SS[k1][k2];
                    A(k1,memo_size+k2) = L[k1][k2];
                }
                A(memo_size+k1,memo_size+k1) = - Ddiag[k1];
            }
            A.block(memo_size, 0, memo_size, memo_size) = A.block(0, memo_size, memo_size, memo_size).transpose();
            Qhat.leftCols(m2) = A.topLeftCorner(m2, m2).lu().solve(Q.leftCols(m2).transpose()).transpose();
        } else {
            A.setZero();
            for (int k1 = 0; k1 < memo_size; ++k1) {
                for (int k2 = 0; k2 < memo_size; ++k2) {
                    A(k1,k2) = gamma * SS[k1][k2];
                    A(k1,memo_size+k2) = L[k1][k2];
                }
                A(memo_size+k1,memo_size+k1) = - Ddiag[k1];
            }
            A.bottomLeftCorner(memo_size, memo_size) = A.topRightCorner(memo_size, memo_size).transpose();
            Qhat = A.lu().solve(Q.transpose()).transpose();
        }
        ++outer;
    }

    return 0;
}
#pragma once


// A minimization problem for p-dimensional vector w.
class Problem {
public:
    // Evaluate function at w, store in f_ptr
    virtual int func(double *w_ptr, double *f_ptr) = 0;

    // Evaluate gradient at w, store in g_ptr
    virtual int grad(double *w_ptr, double *g_ptr) = 0;
};

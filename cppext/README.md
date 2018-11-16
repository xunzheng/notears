# cppext

This is a Python C++ extension that implements ProxQN and various 
related functions for NOTEARS.
The implementation of ProxQN was made possible by generous help of 
the authors of [ProxQN](http://users.ices.utexas.edu/~zhongkai/ProxQN/). 


## Contents

- `eigen/` - [Eigen 3.3.5](http://eigen.tuxfamily.org/index.php?title=Main_Page)
    as git submodule
- `setup.py` - standard python script for package installation
- `src/cppextmodule.cpp` - Python interface of C++ functions
- `src/proxqn.h` - ProxQN algorithm
- `src/problem.h` - class of problems that define zeroth and first order oracle 
- `src/notears_problem.h` - function and gradient of augmented lagrangian
- `src/lasso_problem.h` - function and gradient of lasso, for sanity check


## Eigen with BLAS/LAPACK

If you have BLAS/LAPACK installed, please uncomment corresponding lines
in [`setup.py`](setup.py), [`proxqn.h`](src/proxqn.h),
[`notears_problem.h`](src/notears_problem.h).  
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h"

#include "proxqn.h"
#include "notears_problem.h"
#include "lasso_problem.h"


static PyObject *F_func(PyObject *self, PyObject *args) {
    /* convert Python arguments */
    PyArrayObject *w_array;
    PyArrayObject *X_array;
    double lambda1;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &w_array,
                                         &PyArray_Type, &X_array,
                                         &lambda1)) {
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(w_array)) {
        PyErr_SetString(PyExc_ValueError, "require w_array to be in C-style order");
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(X_array)) {
        PyErr_SetString(PyExc_ValueError, "require X_array to be in C-style order");
        return NULL;
    }
    double *w_ptr = reinterpret_cast<double *>(PyArray_DATA(w_array));
    double *X_ptr = reinterpret_cast<double *>(PyArray_DATA(X_array));
    npy_intp n = PyArray_DIM(X_array, 0);
    npy_intp d = PyArray_DIM(X_array, 1);

    /* do function */
    double F_value;
    if (F_eval(n, d, w_ptr, X_ptr, lambda1, &F_value) < 0) {
        return NULL;
    }

    /* return something */
    return Py_BuildValue("d", F_value);
}

static PyObject *h_func(PyObject *self, PyObject *args) {
    /* convert Python arguments */
    PyArrayObject *w_array;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &w_array)) {
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(w_array)) {
        PyErr_SetString(PyExc_ValueError, "require w_array to be in C-style order");
        return NULL;
    }
    double *w_ptr = reinterpret_cast<double *>(PyArray_DATA(w_array));
    npy_intp ndim = PyArray_NDIM(w_array);
    npy_intp p = PyArray_DIM(w_array, 0);
    int d = 0;
    if (ndim == 1) {
        d = int(sqrt(p));
    } else if (ndim == 2) {
        d = p;
    } else {
        PyErr_SetString(PyExc_ValueError, "w has to be 1-dim or 2-dim");
        return NULL;
    }

    /* do function */
    double h_value;
    if (h_eval(d, w_ptr, &h_value) < 0) {
        return NULL;
    }

    /* return something */
    return Py_BuildValue("d", h_value);
}

static PyObject *minimize_subproblem(PyObject *self, PyObject *args) {
    /* convert Python arguments */
    PyArrayObject *w_array;
    PyArrayObject *X_array;
    double rho;
    double alpha;
    double lambda1;
    if (!PyArg_ParseTuple(args, "O!O!ddd", &PyArray_Type, &w_array, 
                                           &PyArray_Type, &X_array, 
                                           &rho, &alpha, &lambda1)) {
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(w_array)) {
        PyErr_SetString(PyExc_ValueError, "require w_array to be in C-style order");
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(X_array)) {
        PyErr_SetString(PyExc_ValueError, "require X_array to be in C-style order");
        return NULL;
    }
    double *w_ptr = reinterpret_cast<double *>(PyArray_DATA(w_array));
    double *X_ptr = reinterpret_cast<double *>(PyArray_DATA(X_array));
    npy_intp n = PyArray_DIM(X_array, 0);
    npy_intp d = PyArray_DIM(X_array, 1);
    
    /* do function */
    NotearsProblem notears_problem(n, d, X_ptr, rho, alpha);
    ProxQN proxqn(d * d, w_ptr, lambda1);
    if (proxqn.minimize(&notears_problem) < 0) {
        return NULL;
    }

    /* return something */
    PyArrayObject *w_copy_array = reinterpret_cast<PyArrayObject *>(PyArray_NewCopy(w_array, NPY_ANYORDER));
    return PyArray_Return(w_copy_array);
}

static PyObject *lasso_test(PyObject *self, PyObject *args) {
    /* convert Python arguments */
    PyArrayObject *w_array;
    PyArrayObject *X_array;
    PyArrayObject *y_array;
    double lambda1;
    if (!PyArg_ParseTuple(args, "O!O!O!d", &PyArray_Type, &w_array, 
                                           &PyArray_Type, &X_array, 
                                           &PyArray_Type, &y_array, 
                                           &lambda1)) {
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(w_array)) {
        PyErr_SetString(PyExc_ValueError, "require w_array to be in C-style order");
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(X_array)) {
        PyErr_SetString(PyExc_ValueError, "require X_array to be in C-style order");
        return NULL;
    }
    if (!PyArray_ISCARRAY_RO(y_array)) {
        PyErr_SetString(PyExc_ValueError, "require y_array to be in C-style order");
        return NULL;
    }
    double *w_ptr = reinterpret_cast<double *>(PyArray_DATA(w_array));
    double *X_ptr = reinterpret_cast<double *>(PyArray_DATA(X_array));
    double *y_ptr = reinterpret_cast<double *>(PyArray_DATA(y_array));
    npy_intp n = PyArray_DIM(X_array, 0);
    npy_intp p = PyArray_DIM(X_array, 1);
    
    /* do function */
    LassoProblem lasso_problem(n, p, X_ptr, y_ptr);
    ProxQN proxqn(p, w_ptr, lambda1);
    if (proxqn.minimize(&lasso_problem) < 0) {
        return NULL;
    }

    /* return something */
    PyArrayObject *w_copy_array = reinterpret_cast<PyArrayObject *>(PyArray_NewCopy(w_array, NPY_ANYORDER));
    return PyArray_Return(w_copy_array);
}

static PyMethodDef CppextMethods[] = {
    {"F_func", F_func, METH_VARARGS,
     "Evaluate F(w), w could be matrix or flattened vector"},
    {"h_func", h_func, METH_VARARGS,
     "Evaluate h(w), w could be matrix or flattened vector"},
    {"minimize_subproblem", minimize_subproblem, METH_VARARGS,
     "Minimize NOTEARS subproblem using ProxQN"},
    {"lasso_test", lasso_test, METH_VARARGS,
     "Solve lasso for sanity check"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyDoc_STRVAR(cppext_doc,
             "This is a C++ implementation of Proximal Quasi-Newton "
             "and some important functions.");

static struct PyModuleDef cppext_moduledef = {
    PyModuleDef_HEAD_INIT,
    "cppext",   /* name of module */
    cppext_doc, /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module,
                   or -1 if the module keeps state in global variables. */
    CppextMethods
};

PyMODINIT_FUNC PyInit_cppext(void) {
    PyObject *module = PyModule_Create(&cppext_moduledef);

    /* IMPORTANT: this must be called */
    import_array();

    return module;
}

int main(int argc, char *argv[]) {
    /* Convert to wchar */
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    PyMem_RawFree(program);
    return 0;
}
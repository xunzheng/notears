import numpy as np

from distutils.core import setup, Extension
from platform import system as get_os_name


# Configure the compiler based on the OS
if get_os_name().lower() == "darwin":
    os_compile_flags = ["-mmacosx-version-min=10.9"]
else:
    os_compile_flags = []

cppext_module = Extension(
    'cppext',
    sources=['src/cppextmodule.cpp'],
    depends=['src/proxqn.h',
             'src/problem.h', 
             'src/notears_problem.h',
             'src/lasso_problem.h'],
    include_dirs=[np.get_include(), 'eigen/'],
    extra_compile_args=['-O3',
                        # '-march=corei7-avx',  # uncomment if available
                        '-std=c++11',
                        '-Wno-parentheses'] + os_compile_flags,
    # uncomment if available
    # extra_link_args=['-framework',
    #                  'accelerate',
    #                  '/usr/local/opt/lapack/lib/liblapacke.dylib',
    #                  '-march=corei7-avx']
)

setup(name='cppext',
      version='1.0',
      description='C++ implementation of ProxQN and some important functions',
      ext_modules=[cppext_module])

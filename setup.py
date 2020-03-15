from setuptools import setup


setup(
    name='notears',
    version='3.0',
    description='Implementation of the NOTEARS algorithm',
    author='Xun Zheng',
    author_email='xzheng1@andrew.cmu.edu',
    url='https://github.com/xunzheng/notears',
    download_url='https://github.com/xunzheng/notears/archive/v3.0.zip',
    license='Apache License 2.0',
    keywords='notears causal discovery bayesian network structure learning',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    scripts=['bin/notears_linear',
             'bin/notears_nonlinear'],
    packages=['notears'],
    package_dir={'notears': 'notears'},
    install_requires=['numpy', 'scipy', 'python-igraph'],
)

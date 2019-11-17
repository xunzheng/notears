from setuptools import setup


setup(
    name='notears',
    version='2.0',
    description='Implementation of the NOTEARS algorithm',
    author='Xun Zheng',
    author_email='xzheng1@andrew.cmu.edu',
    url='https://github.com/xunzheng/notears',
    download_url='https://github.com/xunzheng/notears/archive/v2.0.zip',
    license='Apache License 2.0',
    keywords='notears causal discovery bayesian network structure learning',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    scripts=['bin/notears_linear_l1'],
    packages=['notears'],
    package_dir={'notears': 'src'},
    install_requires=['numpy', 'scipy', 'python-igraph'],
)

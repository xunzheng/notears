# DAGs with NO TEARS

This is an implementation of the following paper:

[DAGs with NO TEARS: Continuous Optimization for 
Structure Learning](https://arxiv.org/abs/1803.01422) 
([NIPS 2018](https://nips.cc/Conferences/2018/), Spotlight)

[Xun Zheng](https://www.cs.cmu.edu/~xunzheng/), 
[Bryon Aragam](https://www.cs.cmu.edu/~naragam/),
[Pradeep Ravikumar](https://www.cs.cmu.edu/~pradeepr/),
[Eric Xing](https://www.cs.cmu.edu/~epxing/).

If you find it useful, please consider citing:
```
@inproceedings{zheng2018dags,
    author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {NIPS},
    title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
    year = {2018}
}
```


## TL;DR

Check out [`simple_demo.py`](simple_demo.py) for the **50-line** NOTEARS algorithm.  


## Introduction

A directed graphical model (aka Bayesian network) with `d` nodes defines a 
distribution of random vector of size `d`. 
We are interested in the Bayesian Network Structure Learning (BNSL) problem: 
given `n` samples from such distribution, how to estimate the graph `G`? 

A major challenge of BNSL is enforcing the directed acyclic graph (DAG) 
constraint, which is **combinatorial**.
While existing approaches rely on local heuristics,
we introduce a fundamentally different strategy: we formulate it as a purely 
**continuous** optimization problem over real matrices that avoids this 
combinatorial constraint entirely. 
In other words, 

<img width="460" alt="characterization" src="https://user-images.githubusercontent.com/1810194/47379174-2eb1af00-d6c8-11e8-8dae-4626690127b9.png"/>

where `h` is a *smooth* function whose level set exactly characterizes the 
space of DAGs.


## Requirements

- Python 3.5+
- (optional) C++11 compiler


## Contents

- `simple_demo.py` - simple demo for the 50-line version of NOTEARS
- `utils.py` - graph simulation, data simulation, and accuracy evaluation
- `live_demo.ipynb` - live demo in jupyter notebook
- `notears.py` - code for the full version of NOTEARS
- `cppext` - C++ implementation of ProxQN used by the full version


## Download and run a quick demo

This is the simplest way to try out the code. 
Simply clone the repo, install packages, and run a quick demo.
```
git clone https://github.com/xunzheng/notears.git
cd notears/
pip install -r requirements.txt
python simple_demo.py
```
It runs the [50-line version](simple_demo.py) of NOTEARS 
without l1-regularization 
on a randomly generated 10-node Erdos-Renyi graph. 
Since the problem size is small, it will only take a few seconds.

You will get outputs like this:
```
I1026 02:19:54.995781 87863 simple_demo.py:77] Graph: 10 node, avg degree 4, erdos-renyi graph
I1026 02:19:54.995896 87863 simple_demo.py:78] Data: 1000 samples, linear-gauss SEM
I1026 02:19:54.995944 87863 simple_demo.py:81] Simulating graph ...
I1026 02:19:54.996556 87863 simple_demo.py:83] Simulating graph ... Done
I1026 02:19:54.996608 87863 simple_demo.py:86] Simulating data ...
I1026 02:19:54.997485 87863 simple_demo.py:88] Simulating data ... Done
I1026 02:19:54.997534 87863 simple_demo.py:91] Solving equality constrained problem ...
I1026 02:20:00.791475 87863 simple_demo.py:94] Solving equality constrained problem ... Done
I1026 02:20:00.791845 87863 simple_demo.py:99] Accuracy: fdr 0.000000, tpr 1.000000, fpr 0.000000, shd 0, nnz 17
```


## Install & run the full version

The Proximal Quasi-Newton algorithm is at the core of the full NOTEARS with 
l1-regularization. 
Hence for efficiency concerns it is implemented in a C++ module `cppext` 
using [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).

To install `cppext`, download Eigen submodule and compile the extension:
```
git submodule update --init --recursive
cd cppext/
python setup.py install
cd .. 
```

We are ready to run a live demo. Type
```
jupyter notebook
```
and click open [`live_demo.ipynb`](live_demo.ipynb) in
the browser.
Select *Kernel --> Restart & Run All*. 


(TODO: gif)


## Results

<img width="200" alt="W_true" src="https://user-images.githubusercontent.com/1810194/47596959-7c444b00-d958-11e8-8587-d355eaf3f7d1.png" />
<img width="600" alt="W_est_n1000" src="https://user-images.githubusercontent.com/1810194/47597035-1d330600-d959-11e8-9d59-d2ce3fac0f39.png" />
<img width="600" alt="W_est_n20" src"https://user-images.githubusercontent.com/1810194/47597063-608d7480-d959-11e8-8085-2c2a98d16704.png" />


## Custom implementations  

- Python: https://github.com/jmoss20/notears

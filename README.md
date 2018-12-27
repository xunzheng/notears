# DAGs with NO TEARS :no_entry_sign::droplet:

This is an implementation of the following paper:

[DAGs with NO TEARS: Continuous Optimization for 
Structure Learning](https://arxiv.org/abs/1803.01422) 
([NeurIPS 2018](https://nips.cc/Conferences/2018/), Spotlight)

[Xun Zheng](https://www.cs.cmu.edu/~xunzheng/), 
[Bryon Aragam](https://www.cs.cmu.edu/~naragam/),
[Pradeep Ravikumar](https://www.cs.cmu.edu/~pradeepr/),
[Eric Xing](https://www.cs.cmu.edu/~epxing/).

If you find it useful, please consider citing:
```
@inproceedings{zheng2018dags,
    author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {Advances in Neural Information Processing Systems},
    title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
    year = {2018}
}
```


## tl;dr Structure learning in <50 lines

Check out [`simple_demo.py`](simple_demo.py) for a complete, end-to-end implementation of the NOTEARS algorithm in fewer than **50 lines**.  


## Introduction

A directed acyclic graphical model (aka Bayesian network) with `d` nodes defines a 
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

- Simple NOTEARS (without l1 regularization)
    - `simple_demo.py` - the 50-line implementation of simple NOTEARS
    - `utils.py` - graph simulation, data simulation, and accuracy evaluation
- Full NOTEARS (with l1 regularization)
    - `cppext/` - C++ implementation of ProxQN
    - `notears.py` - the full NOTEARS with live progress monitoring
    - `live_demo.ipynb` - jupyter notebook for live demo


## Running a simple demo

The simplest way to try out NOTEARS is to run the toy demo:
```bash
$ git clone https://github.com/xunzheng/notears.git
$ cd notears/
$ pip install -r requirements.txt
$ python simple_demo.py
```
This runs the [50-line version](simple_demo.py) of NOTEARS 
without l1-regularization 
on a randomly generated 10-node Erdos-Renyi graph. 
Since the problem size is small, it will only take a few seconds.

You should see output like this:
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


## Running the full version

The Proximal Quasi-Newton algorithm is at the core of the full NOTEARS with 
l1-regularization. 
Hence for efficiency concerns it is implemented in a C++ module `cppext` 
using [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).

To install `cppext`, download Eigen submodule and compile the extension:
```bash
$ git submodule update --init --recursive
$ cd cppext/
$ python setup.py install
$ cd .. 
```

The code comes with a Jupyter notebook that runs a live demo. This allows you to monitor the progress as the algorithm runs. Type
```bash
$ jupyter notebook
```
and click open [`live_demo.ipynb`](live_demo.ipynb) in
the browser.
Select *Kernel --> Restart & Run All*. 


(TODO: gif)


## Examples: Erdos-Renyi graph

- Ground truth: `d = 20` nodes, `2d = 40` expected edges.

  <img width="193" alt="ER2_W_true" src="https://user-images.githubusercontent.com/1810194/47596959-7c444b00-d958-11e8-8587-d355eaf3f7d1.png" />

- Estimate with `n = 1000` samples: 
  `lambda = 0`, `lambda = 0.1`, and `FGS` (baseline).

  <img width="600" alt="ER2_W_est_n1000" src="https://user-images.githubusercontent.com/1810194/47597035-1d330600-d959-11e8-9d59-d2ce3fac0f39.png" />

  Both `lambda = 0` and `lambda = 0.1` are close to the ground truth graph 
  when `n` is large.
     
- Estimate with `n = 20` samples:
  `lambda = 0`, `lambda = 0.1`, and `FGS` (baseline).

  <img width="600" alt="ER2_W_est_n20" src="https://user-images.githubusercontent.com/1810194/47597063-608d7480-d959-11e8-8085-2c2a98d16704.png" />

  When `n` is small, `lambda = 0` perform worse while 
  `lambda = 0.1` remains accurate, showing the advantage of L1-regularization. 

## Examples: Scale-free graph

- Ground truth: `d = 20` nodes, `4d = 80` expected edges.

  <img width="193" alt="SF4_W_true" src="https://user-images.githubusercontent.com/1810194/47598929-a7876400-d971-11e8-903c-109d7f3754cb.png" />

  The degree distribution is significantly different from the Erdos-Renyi graph.
  One nice property of our method is that it is agnostic about the 
  graph structure.

- Estimate with `n = 1000` samples: 
  `lambda = 0`, `lambda = 0.1`, and `FGS` (baseline).

  <img width="600" alt="SF4_W_est_n1000" src="https://user-images.githubusercontent.com/1810194/47598936-c1c14200-d971-11e8-8572-a589c98de0b7.png" />

  The observation is similar to Erdos-Renyi graph: 
  both `lambda = 0` and `lambda = 0.1` accurately estimates the ground truth
  when `n` is large.

- Estimate with `n = 20` samples:
  `lambda = 0`, `lambda = 0.1`, and `FGS` (baseline).

  <img width="600" alt="SF4_W_est_n20" src="https://user-images.githubusercontent.com/1810194/47598941-dc93b680-d971-11e8-81db-72bd19866290.png" />

  Similarly, `lambda = 0` suffers from small `n` while 
  `lambda = 0.1` remains accurate, showing the advantage of L1-regularization. 


## Other implementations  

- Python: https://github.com/jmoss20/notears

# DAGs with NO TEARS

This is an implementation of the following paper:

[DAGs with NO TEARS: 
Continuous Optimization for Structure Learning](https://arxiv.org/abs/1803.01422), NIPS 2018. (Spotlight) 
\
Xun Zheng, Bryon Aragam, Pradeep Ravikumar, Eric Xing.

If you find it useful, please consider citing:
```
@inproceedings{zheng2018dags,
    author = {Zheng, Xun and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {NIPS},
    title = {{DAGs with NO TEARS: Continuous Optimization for Structure Learning}},
    year = {2018}
}
```


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
<img width="450" alt="characterization" src="https://user-images.githubusercontent.com/1810194/47379174-2eb1af00-d6c8-11e8-8dae-4626690127b9.png"/>
\
where `h` is a *smooth* function whose level set exactly characterizes the space of DAGs.


## Usage

1. Set up `python3`. We recommend working under virtual environment, e.g. the built-in `venv`: 
    ```
    python3 -m venv /path/to/notears_env
    source /path/to/notears_env/bin/activate
    pip install --upgrade pip setuptools wheel
    ```

2. Download & run. Clone the repo, install required packages, and run a simple demo:
    ```
    git clone https://github.com/xunzheng/notears.git
    cd notears/
    pip install -r requirements.txt
    python notears_simple_demo.py
    ```
    It runs the 50-line version of NOTEARS on a 10-node Erdos-Renyi graph. 

3. For more examples, run jupyter notebooks. 
    ```bash
    jupyter notebook 
    ```


## Results

Some heatmaps?


## Custom implementations  

- Python: https://github.com/jmoss20/notears

# PositivityConjectures
A RL approach to study the Stanley conjecture on chromatic symmetric polynomials

<img src="https://github.com/berczig/PositivityConjectures/blob/main/escher_stairs.jpg?raw=true" alt="drawing" width="300"/>

## The Math Problem:

Let $G$ be a finite graph, with $V(G)$ representing its vertices and $E(G)$ representing its edges.

### Definition
A **proper coloring** $c$ of $G$ is a function $c : V(G) \rightarrow \mathbb{N}$, such that no two adjacent vertices share the same color, i.e., for all $\{u, v\} \in E(G)$, we have $c(u) \neq c(v)$.

For a given coloring $c$, we can associate a monomial:
$$
x_c = \prod_{v \in V(G)} x_{c(v)},
$$
where $x_1, x_2, \dots$ are commuting variables. Let $\Pi(G)$ denote the set of all proper colorings of $G$, and let $\Lambda$ represent the ring of symmetric functions in the infinite set of variables $\{x_1, x_2, \dots\}$.

Stanley introduced the following polynomial:

### Definition
The **chromatic symmetric polynomial** $X_G \in \Lambda$ of a graph $G$ is defined as the sum of the monomials $x_c$ over all proper colorings $c \in \Pi(G)$:
$$
X_G = \sum_{c \in \Pi(G)} x_c.
$$

### Definition
The $m$-th **elementary symmetric polynomial** $e_m$ is defined as:
$$
e_m = \sum_{i_1 < i_2 < \dots < i_m} x_{i_1} x_{i_2} \dots x_{i_m},
$$
where $i_1, \dots, i_m \in \mathbb{N}$. Given a partition $\lambda = (\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_k)$, we define the elementary symmetric function $e_\lambda$ as $e_\lambda = \prod_{i=1}^k e_{\lambda_i}$. These functions form a basis for $\Lambda$, the ring of symmetric functions.

### Definition
A **unit interval graph** (UIG) is a graph whose vertices correspond to unit intervals $u_1, u_2, \dots, u_n$ on the real line, and two vertices $u_i$ and $u_j$ are connected by an edge if and only if the corresponding intervals intersect.

Stanley posed the following conjecture in [Stanley1995], which is simplified here from the original formulation involving incomparability graphs:

### Conjecture (Stanley-Stembridge, 1993)
The chromatic symmetric polynomial of a unit interval graph is $e$-positive, that is, if written in the elementary symmetric bases as 
$$ X^G = \sum_{\lambda} c_\lambda e_\lambda$$ 
then $c_\lambda \ge 0$ for all partition $\lambda$.

## The ML approach 

Unit interval graphs on n intervals are in 1-1 correspondence with sequences $(a_1 \le a_2 \le \ldots \le a_n)$ 
where $a_i \le i$ integers. 
Two main directions:
1) Supervised learning of the function $(a_1,\ldots, a_n) \mapsto c_\lambda$, in [SPC/Transformers] we have a small GPT model with promising results.
2) Learning Stanley coefficients as number of Escher-tuples satidsfying certasin condtitions. The condition graphs are found using RL. This approach based on recent results of Szenes-Rok, and lots of math intuition.  

## Features
- Generate all UIOs of length n using the area sequence encoding
- Calculate all $\lambda$-Eschers of a UIO, for $|\lambda| < 5$
- Calculate the e-basis coefficients $c_{\lambda}$ for any UIO, for $|\lambda| < 5$
- Train RL models to predict coefficients $c_{\lambda}$ by counting cores, for $|\lambda| < 5$

## Requirements
The only prerequisite for this project is python and git. The code has been tested on Python 3.11.2, but may also work for other versions.

If your python version is not working you can use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a new python 3.11.2 environment.

```
conda create --name SPCenv python=3.11.2
conda activate SPCenv
```
## Installation
To install open a command prompt and navigate to a folder where you want the project to be installed and run

```bash
git clone https://github.com/berczig/PositivityConjectures.git
cd PositivityConjectures
pip install -r requirements.txt
```
This will install the project and the required python modules. If you have issues with installing the requirements you can try to change the versions in [requirements.txt](requirements.txt).

Next install the project as a python package - navigate to the directory containing [pyproject.toml](pyproject.toml) (Make sure you have flit installed) and run
```
pip install -e .
```

## Usage
Run [SPC/UIO/main.py](main.py) to 
- generate the UIOs and the cores
- calculate the coefficients
- train a model on the data

You can change multiple parameters in main.py e.g. the partition or core generator.

Run [SPC/ResultViewer/result_Viewer.py](result_Viewer.py) to open a GUI to display the saved training data and saved models. The GUI lists your models in a grid, by clicking on them you will see more info about the training process and data used. The GUI will also make a table to summarize the results of all loaded models:

![alt text](https://github.com/berczig/PositivityConjectures/blob/main/result_viewer_preview.PNG?raw=true)

## Code Overview
Overview of the classes:
![alt text](https://github.com/berczig/PositivityConjectures/blob/main/classes.png?raw=true)

Overview Idea:
![alt text](https://github.com/berczig/PositivityConjectures/blob/main/overview.png?raw=true)

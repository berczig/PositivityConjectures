# PositivityConjectures
A RL approach to study the Stanley conjecture on chromatic symmetric polynomials

<img src="https://github.com/berczig/PositivityConjectures/blob/main/escher_stairs.jpg?raw=true" alt="drawing" width="300"/>

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

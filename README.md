# PositivityConjectures
A RL approach to study the Stanley conjecture on chromatic symmetric polynomials

## Features

## Requirements and Installation
Tested on Python 3.11.2

```bash
# Example for Python project
git clone https://github.com/berczig/PositivityConjectures.git
cd PositivityConjectures
pip install -r requirements.txt
```

To support sibling imports the project should be installed as a package. To install navigate to the directory containing [pyproject.toml](pyproject.toml) and run
```
pip install -e .
```
Make sure you have flit installed.

## Usage
Run [SPC/UIO/main.py](main.py). You can change multiple parameters in main.py e.g. the partition or core generator.

## Project Overview
Overview of the classes:
![alt text](https://github.com/berczig/PositivityConjectures/blob/main/classes.png?raw=true)

Overview Idea:
![alt text](https://github.com/berczig/PositivityConjectures/blob/main/overview.png?raw=true)



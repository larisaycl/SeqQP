# SequentialQP

This project was completed in collaboration with [VicHenFel](https://github.com/VicHenFel). 

## Overview
This repo contains the Python code for our final project of ME441 Engineering Optimization for Product Design and Manufacturing at Northwestern. For our project, we chose to implement Sequential Quadratic Programming in Python. We validated our SQP implementation by testing on benchmark optimization functions and compared results against SQP implementations in standard optimization packages such as Scipy and Isight.

The technical background of this project can be found in [this pdf (algorithm)](algorithm_sqp.pdf)

## Running the code
To run the code, do 

```
python3 run.py
```

from within the `src` directory

A summary of items in `src`:

* `sqp.py` - script containing a complete SQP pipeline
* `test_functions.py` - script containing benchmark functions for testing 
* `run.py` - main script to generate figures
* `figures/` - directory containing generated plots
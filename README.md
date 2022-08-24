# GNNGP: Graphical Neural Network as Gaussian Process

This repository contains pytorch implementation of GNNGP methods, as well as several test functions aiming for performance study.

# Usage

The `src/main.py` runs automatic experiments with following command line arguments:

## Requirements

`python >= 3.5.0`
`pytorch >= 1.9.0`

## Datasets

Available datasets in the repository include the Wikipedia Article Networks and the Planetoid dataset.

Upon first usage, an automatic download of the raw dataset may occur.

## Methods

## Metrics

## Logs

The parameter settings and runtime info will be logged. This include the following:

```
1. Hyperparameter settings - every hyperparameter used in the experiment.
2. Computation time - the time used for computation, in seconds.
3. Error metric - the error of the predicted result.
```

# References


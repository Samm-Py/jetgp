# oti_gp 
An Gaussian Process library with support for arbitrary-order derivative-enhanced training data.





## Dependencies
```oti_gp``` depends on the following libraries:

- ```pyoti```: A library to support Order Truncated Imaginary (OTI) numbers.
- ```sympy```.
- ```numpy```.
- ```scipy```.

## Installation instructions

Use the attached environment file to install all dependencies.

```bash
conda env create -f environment.yml 
```

## Local HTML documentation build

1. Activate the ``otigp`` conda environment

```bash
conda activate otigp
```

2. Change to ``docs`` directory and make the html documentation
```bash
cd docs
make html
```
# Reproducing the Report Code

This folder contains the active-learning scripts used for the transient fin
example in the report. The scripts require the local `jetgp` source tree and the
included patched/prebuilt `otilib-master` source tree. Run the code from Linux
or WSL.

## Expected Folder Layout

In a WSL environment with Python/Anaconda installed, run the following:

```bash
unzip Roberts_Final_Project_Code.zip
```

The code should be extracted with this layout:

```text
github_oti_gp/
├── jetgp/
│   ├── jetgp/
│   ├── active_learning/
│   │   ├── README_RUN.md
│   │   ├── environment_pyoti_2.yml
│   │   ├── example_3_hypad_fin_active_learning.py
│   │   └── ...
│   └── ...
└── otilib-master/
    └── ...
```

`jetgp` and `otilib-master` should remain sibling directories. The included
`otilib-master` has already been patched for JetGP and already contains the
compiled `pyoti` modules used for the report.

## Platform Notes

The tested environment is Linux/WSL with Python 3.9. The included compiled
`pyoti` extension modules have names like:

```text
pyoti/sparse.cpython-39-x86_64-linux-gnu.so
```

This means they are built for CPython 3.9 on 64-bit Linux. They should work on a
similar WSL/Linux setup. 

## Environment Setup

From `jetgp/active_learning`, create the conda environment:

```bash
conda env create -f environment_pyoti_2.yml
conda activate pyoti_2
```


## Register Local Libraries

From `jetgp/active_learning`, register the local source folders with the active
conda environment:

```bash
cd ..
conda develop .

cd ../otilib-master/build
conda develop .
```

If `conda develop` is not available, either install `conda-build` into the
environment or use `PYTHONPATH` for the current shell:

```bash
cd ..
export PYTHONPATH="$(pwd):$(pwd)/../otilib-master/build:${PYTHONPATH}"
```

The report scripts themselves were run from `jetgp/active_learning`.

## Run Examples 1 and 2

After activating the environment and registering the local libraries, run the
example scripts from `jetgp/active_learning`.

Example 1 runs the two-dimensional Branin-Hoo adaptive DOE case:

```bash
python example_1_branin_hoo.py
```

The script prints an iteration summary and writes figures to:

```text
example_1_figures/
```

Useful output figures include:

```text
initial_doe_mean_variance.png
initial_enrichment_directions.png
post_enrichment_mean_variance.png
iter_XX_mean_variance.png
iter_XX_directional_derivatives.png
iter_XX_eigen_spectrum.png
final_design.png
```

Example 2 runs the four-dimensional orthogonal-direction validation case:

```bash
python example_2_4d_orthogonal_direction_demo.py
```

The script prints the selected infill point, local derivative covariance
eigenvalues, selected eigen-directions, and optimizer validation results. It
writes figures to:

```text
example_2_figures/
```

The generated figures are:

```text
example_2_direction_components.png
example_2_optimizer_direction_components.png
example_2_optimizer_alignment.png
```

## Run the Transient Fin Example

From `jetgp/active_learning`:

```bash
python example_3_hypad_fin_active_learning.py \
  --case 2 --times 1 100 1000 \
  --n-init 2 --n-iter 10 --seed 1 \
  --rel-tol 0.01
```

The report used these settings. The script writes figures to:

```text
example_3_case_2_t1s_figures/
example_3_case_2_t100s_figures/
example_3_case_2_t1000s_figures/
```

The report specifically used:

```text
learning_curves_vs_cost.png
output_distributions.png
```

from each time-specific output folder.

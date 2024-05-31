# Experiments

This directory contains code for reproducing the experiments in the paper. Each subdirectory corresponds to a different set of experiments.
- `relational_games`: Visual relational reasoning with the "Relational Games" benchmark.
- `math`: Mathematical problem-solving.
- `tiny_stories`: Language modeling with the "Tiny Stories" benchmark.
- `vision`: Image recognition with the ImageNet dataset.

Each directory contains a `readme` which describes the experiment in more detail and contains instructions for reproducing the results in the paper.

You can replicate our python environment by using the `conda_environment.yml` file via:
```
conda env create -f conda_environment.yml
```

Detailed experimental logs for all experiments are publicly available for transparency and reproducibility. For each experimental run, this includes: metrics tracks over the course of training, version of code (git commit ID), the exact script that produced the run and its arguments, the hardware used for the run, etc. Links to the logs can be found in the readme associated to each experiment.
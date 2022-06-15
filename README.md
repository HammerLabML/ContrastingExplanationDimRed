# Diverse Contrasting Explanations of Dimensionality Reduction

This repository contains the implementation of the methods proposed in the paper ["Why Here and Not There?" -- Diverse Contrasting Explanations of Dimensionality Reduction](paper.pdf) by André Artelt, Alexander Schulz and Barbara Hammer.

The experiments as described in the paper are implemented in the folder [Implementation](Implementation/).

## Abstract

Dimensionality reduction is a popular preprocessing and a widely used tool in data mining. Transparency, which is usually achieved by means of explanations, is nowadays a widely accepted and crucial requirement of machine learning based systems like classifiers and recommender systems. However, transparency of dimensionality reduction and other data mining tools have not been considered much yet, still it is crucial to understand their behavior -- in particular practitioners might want to understand why a specific sample got mapped to a specific location.

In order to (locally) understand the behavior of a given dimensionality reduction method, we introduce the abstract concept of contrasting explanations for dimensionality reduction, and apply a realization of this concept to the specific application of explaining two dimensional data visualization.

## Details
### Implementation of experiments
The shell script `run_experiments.sh` runs all experiments.

### Other (important) stuff
#### Computation of counterfactual explanations

The implementation of the proposed algorithms for computing diverse counterfactual explanations can be found in the `Implementation` folder: `linear_dr.py`, `som_dr.py`, `ae_dr.py` and `tsne_dr.py`.

The baseline approach is implemented in `memory_counterfactual.py`.

#### Parametric t-SNE

We use the implementation (see `Implementation/parametric_tSNE`) from [https://github.com/jsilter/parametric_tsne](https://github.com/jsilter/parametric_tsne)

## Requirements

- Python3.6
- Packages as listed in `Implementation/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE](LICENSE)

## How to cite

You can cite the version on [arXiv](TODO)
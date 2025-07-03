# Equivariant Neural Field Networks for steady PDE surrogates on general geometries.

This repository contains the Pytorch implementation of enf2enf [Geometry aware inference of steady state PDEs using Equivariant Neural Fields representations](https://arxiv.org/abs/2504.18591), a neural operator approach to solve steady state PDEs on general geometries.
It is an attempt to translate in Pytorch the Jax implementation of enf2enf at https://github.com/giovannicatalani/enf2enf, and the original JAX implementation of the Equivariant Neural Fields architecture: https://github.com/david-knigge/enf-pde.


## Architecture Overview

Our architecture leverages equivariant neural fields to learn PDE solutions across different domains. The network processes geometric features and produces accurate field predictions while maintaining equivariance properties.

![Architecture Overview](figures/architecture_skectch_v4-1.png)

## Experiments
For the moment this repo implements the experiment on the Aifranns dataset shape encoding. More experiments will be added son. For the whole experiments refer to the original jax implementation.


### Airfrans Dataset

For airfoil simulations, we use the AirFRANS dataset to predict flow fields around airfoils at different angles of attack and flow conditions. 
![Pressure Distributions](figures/comparison_sample_81_-0.7650_92.7220-1.png)

To run experiments:
```bash
python airfrans_full.py
```

## Data
Data for Airfrans dataset can be found by downloading the airfrans package.
```bash
pip install airfrans
```
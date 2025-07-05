# Seminar topic: Deep Ensembles

A seminar project exploring Deep Ensembles and Bayesian Neural Networks for uncertainty estimation and robustness in deep learning.

## Overview

This repository contains code and experiments related to the seminar **"Probabilistic ML"**, focusing on the practical and theoretical aspects of combining Deep Ensembles with Bayesian Neural Networks (BNNs). The experiments are primarily conducted on image classification benchmarks such as MNIST, MNIST-Corrupted, and CIFAR-10.

## Motivation

Deep Ensembles and Bayesian Neural Networks are two state-of-the-art approaches for uncertainty estimation in neural networks. This project investigates their individual and combined performance, with a particular focus on:

- **Predictive accuracy**
- **Calibration**
- **Uncertainty estimation**
- **Robustness to out-of-distribution and corrupted data**

## Features

- Training and evaluation of single BNNs and Deep Ensembles of BNNs
- Experiments on clean and corrupted image datasets (e.g., MNIST, MNIST-Corrupted)
- Visualization of accuracy, negative log-likelihood (NLL), Brier score, predictive entropy, and expected calibration error (ECE)
- Out-of-distribution and robustness analysis

## Project Structure

```
Seminar Deep Ensembles/
├── Examples/
│   ├── Example_CIFAR_10.py
│   ├── Example_MNIST.py
│   ├── Example_MNIST_Corrupted.py
│   ├── Example_MNIST_OOD.py
│   └── Example_WINE.py
├── Model_Code/
│   ├── BNN_Model.py
│   ├── ConvolutionalBNN_Model.py
│   ├── ConvolutionalBNN_Model_Adversarial_Training.py
│   ├── Ensemble_helper.py
│   └── Save_and_Load_Models.py
├── plots/
│   ├── Saved_Plots/
│   │   ├── Plots_fair_comparison/
│   │   │   └── [Plots from experiments with equal total computational time for fair comparison]
│   │   └── [Standard result plots: accuracy, calibration, entropy, NLL, example images, etc.]
│   ├── Custom_plot_style.py
│   ├── Data_Plots.py
│   ├── Plots_CIFAR10.py
│   ├── Plots_Helper.py
│   ├── Plots_main.py
│   └── Plots_MNIST.py
├── Saved_Models/
│   └── [ensemble member subfolders with .h5 weights and .json metadata files]
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
```


## Getting Started

### Requirements

- Python 3.8+
- All dependencies are listed in `requirements.txt`

Install dependencies with:

```bash
pip install -r requirements.txt

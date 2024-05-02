# TurbulenceModelPINN: Advanced Turbulence Prediction with Physics-Informed Neural Networks

## Overview

TurbulenceModelPINN offers a sophisticated approach to predicting turbulence dynamics by leveraging the power of Physics-Informed Neural Networks (PINNs). This project is built on the PyTorch Lightning framework, facilitating streamlined model training, evaluation, and management. Designed to tackle the intricate challenge of turbulence prediction, it combines state-of-the-art machine learning techniques with the fundamental principles of fluid mechanics. The architecture encapsulates data preprocessing, training routines, performance evaluation, and extensive logging, providing a holistic solution for researchers and engineers in the field of computational fluid dynamics.

## Key Features

- **Efficient Data Handling:** Incorporates a tailored data module for optimal data manipulation, including loading, preprocessing, and batching operations.
- **Streamlined Model Training:** Harnesses PyTorch Lightning's advanced training capabilities to enhance model development efficiency and reproducibility.
- **Comprehensive Evaluation Framework:** Implements thorough evaluation methodologies on dedicated test datasets, facilitating accurate assessment of model performance against standard benchmarks.
- **In-depth Experiment Tracking:** Employs TensorBoard integration for detailed tracking of training processes, offering insightful visualisations of key metrics and model behavior over time.

## Folder Structure

The folder structure of the project is outlined below:

```bash
PINNs_repo/
├── data/                  # Directory for storing dataset files
├── models/
│   ├── NNModel.py         # Neural network model architecture
│   ├── TurbulenceDataModule.py  # LightningDataModule for handling turbulence data
│   └── TurbulenceDataset.py     # Custom dataset class for turbulence data
│   └── TurbulenceModelPINN.py   # Physics-Informed Neural Network model
├── results/
│   ├── raw_data/              # Directory for storing raw data files
│   ├── figures/           # Directory for storing visualisation outputs
│   └── visualisation.ipynb    # Jupyter notebook for visualising results
├── tb_logs/               # Directory for storing TensorBoard logs and model checkpoints
├── find_best_hyperparams.py  # Script for hyperparameter tuning
├── requirements.txt       # File containing required dependencies
├── run_inference.py       # Script for running inference on the trained model
├── run_test_inference.py  # Script for running inference on the test dataset
└── test_model.py          # Script for evaluating the model on the test dataset
└── train_model.py         # Script for training the model
```


## Getting Started

1. Create a virtual environment and activate it:

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

## Detailed Usage Guide

### Preparing Your Data

Prepare your dataset by ensuring it's correctly formatted and located within the `data/` directory. The project expects CSV files for training, validation, and testing phases.

### Model Training Procedure

To initiate the training process, execute:

```bash
python train_model.py
```

This will process your dataset, train the model according to predefined specifications, and generate logs of the training session in the `tb_logs/` directory. The weights of the trained model will be saved in the `checkpoints/` subdirectory of each version. For instance, the best weights of the model training Version 1 will be saved in `tb_logs/ChannelTurbulenceModelPINN/version_1/checkpoints/`.

Tailor the model architecture and training configurations to meet specific requirements by modifying the `train_model.py` script. Parameters such as input dimensions, the architecture of hidden layers, learning rates, and more can be adjusted to optimise performance for different turbulence characteristics.

#### Hyperparameter Tuning

To perform hyperparameter tuning, run the `find_best_hyperparams.py` script:

```bash
python find_best_hyperparams.py
```

This script will search for the best hyperparameters based on the specified search space and evaluation criteria. You can modify the hyperparameter search space and evaluation metrics to suit your specific needs.

#### Visualising Training Progress

Launch TensorBoard to visualise the training and evaluation metrics:

```bash
tensorboard --logdir=tb_logs/
```

Access the provided URL in your browser to explore the training logs and performance metrics.

### Model Evaluation

Upon completion of the training phase, evaluate the model's predictive accuracy on the test dataset:

```bash
python test_model.py
```

You'll have to modify the `test_model.py` script to load the desired model weights for evaluation. The script will output the model's performance metrics, including the Mean Squared Error (MSE) of each output variable.

You can also save the inferred results in a CSV file for further analysis and visualisation by using the `run_test_inference.py` script:

```bash 
python run_test_inference.py
```

### Performing Inference

Modify the `run_inference.py` script to load the desired model weights for inference and the features data. Execute the script to predict turbulence dynamics based on the input data:

```bash
python run_inference.py
```

This script will save the predictions in a CSV file for further analysis and visualisation.

### Visualising Results

A jupyter notebook `visualisation.ipynb` is provided in the `results/` directory for visualising the results of the model predictions. You can use this notebook to generate visualisations of the model's performance and compare the predictions with the ground truth data.

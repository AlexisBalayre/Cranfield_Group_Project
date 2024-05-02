import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt

# Importing modules for data handling and the neural network model
from model.TurbulenceDataModule import TurbulenceDataModule
from model.TurbulenceModelPINN import TurbulenceModelPINN

""" 
    This script is used to find the best hyperparameters for the model. 
    A grid search is performed over the hyperparameters to find the best combination.
    The results are saved to a CSV file for further analysis. 
"""
if __name__ == "__main__":
    # Fixed random seed for reproducibility of results
    torch.manual_seed(123)

    # Model initialisation with specified architecture parameters
    batch_sizes = [32, 64]  # Number of samples per batch
    train_dataset_path = "data/channel/train_dataset.csv"  # Training dataset
    val_dataset_path = "data/channel/val_dataset.csv"  # Validation dataset
    test_dataset_path = "data/channel/test_dataset.csv"  # Test dataset
    input_dim = 5  # Dimensionalityx of input features
    output_dim = 8  # Dimensionality of the model's output
    hidden_dims = [32, 64, 128, 256]  # Size of the model's hidden layers
    hidden_depths = [1, 2, 3, 4, 8]  # Number of hidden layers
    learning_rates = [1e-4, 5e-4, 1e-3]  # Initial learning rate
    max_epochs = 1000  # Maximum number of training epochs
    activation_functions = [
        "elu", "relu", "tanh", "sigmoid",
    ]  # Activation function for hidden layers
    loss_phys = [1, 10, 100]  # Weight for the momentum loss term (physical loss)
    loss_data = [1, 1e-1, 1e-2]

    # Create a dataframe to store the results
    results = pd.DataFrame(
        columns=[
            "batch_size",
            "hidden_dim",
            "hidden_depth",
            "learning_rate",
            "activation",
            "loss_phys",
            "loss_data",
            "mse_total",
            "rmse_total",
            "r2_total",
            "mse_U",
            "mse_dUdy",
            "mse_P",
            "mse_k",
            "mse_uu",
            "mse_vv",
            "mse_ww",
            "mse_uv",
            "r2_U",
            "r2_dUdy",
            "r2_P",
            "r2_k",
            "r2_uu",
            "r2_vv",
            "r2_ww",
            "r2_uv",
            "rmse_U",
            "rmse_dUdy",
            "rmse_P",
            "rmse_k",
            "rmse_uu",
            "rmse_vv",
            "rmse_ww",
            "rmse_uv",
        ]
    )

    for batch_size in batch_sizes:
        for hidden_dim in hidden_dims:
            for hidden_depth in hidden_depths:
                for learning_rate in learning_rates:
                    for activation in activation_functions:
                        for loss_p in loss_phys:
                            for loss_d in loss_data:
                                # Data Module
                                data_module = TurbulenceDataModule(
                                    train_dataset_path=train_dataset_path,
                                    val_dataset_path=val_dataset_path,
                                    test_dataset_path=test_dataset_path,
                                    batch_size=batch_size,  # Number of samples per batch
                                    num_workers=8,  # Number of subprocesses for data loading
                                )

                                # Number of time steps for cosine annealing
                                data_module.setup("fit")

                                # Data Module
                                data_module = TurbulenceDataModule(
                                    train_dataset_path=train_dataset_path,
                                    val_dataset_path=val_dataset_path,
                                    test_dataset_path=test_dataset_path,
                                    batch_size=batch_size,  # Number of samples per batch
                                    num_workers=8,  # Number of subprocesses for data loading
                                )

                                # Number of time steps for cosine annealing
                                data_module.setup("fit")  # Prepare data for the fitting process

                                # Model initialization with specified architecture parameters
                                model = TurbulenceModelPINN(
                                    batch_size=batch_size,  # Number of samples per batch
                                    max_steps=(
                                        max_epochs
                                        * len(data_module.train_dataset)
                                        // data_module.batch_size
                                    ),  # Maximum number of training steps
                                    lr=learning_rate,  # Learning rate
                                    input_dim=input_dim,  # Dimensionality of input features
                                    hidden_dim=hidden_dim,  # Size of the model's hidden layers
                                    output_dim=output_dim,  # Dimensionality of the model's output
                                    hidden_depth=hidden_depth,  # Number of hidden layers
                                    activation=activation,  # Activation function for hidden layers
                                    loss_phys_momentum_weight=loss_p,  # Weight for the momentum loss term (physical loss)
                                    loss_phys_k_weight=loss_p,  # Weight for the k loss term (physical loss)
                                    loss_bound_U_weight=loss_d,  # Weight for the U loss term (boundary loss)
                                    loss_bound_dUdy_weight=loss_d,  # Weight for the dUdy loss term (boundary loss)
                                    loss_bound_P_weight=loss_d,  # Weight for the P loss term (boundary loss)
                                    loss_bound_k_weight=loss_d,  # Weight for the k loss term (boundary loss)
                                    loss_bound_stress_weight=loss_d,  # Weight for the stress loss term (boundary loss)
                                )

                                # Logger setup for TensorBoard
                                logger = TensorBoardLogger(
                                    "tb_logs", name="ChannelTurbulenceModelPINNTest"
                                )

                                # Trainer initialization with configurations for training process
                                trainer = L.Trainer(
                                    max_epochs=max_epochs,  # Maximum number of epochs for training
                                    accelerator="cpu",  # Specifies the training will be on CPU
                                    devices="auto",  # Automatically selects the available devices
                                    logger=logger,  # Integrates the TensorBoard logger for tracking experiments
                                    deterministic=True,  # Ensures reproducibility of results
                                    precision=32,  # Use 32-bit floating point precision
                                )

                                # Training phase
                                trainer.fit(
                                    model,
                                    datamodule=data_module,
                                )  # Start training the model

                                # Compute the MSE over the test dataset
                                metrics = trainer.test(
                                    model, datamodule=data_module
                                )  # Test the model

                                # Convert the metrics to a pandas DataFrame
                                metrics = pd.DataFrame(metrics)

                                # Concatenate the version with the metrics
                                results = pd.concat(
                                    [
                                        results,
                                        pd.concat(
                                            [
                                                pd.DataFrame(
                                                    {
                                                        "batch_size": [batch_size],
                                                        "hidden_dim": [hidden_dim],
                                                        "hidden_depth": [hidden_depth],
                                                        "learning_rate": [learning_rate],
                                                        "activation": [activation],
                                                        "loss_phys": [loss_p],
                                                        "loss_data": [loss_d],
                                                    }
                                                ),
                                                metrics,
                                            ],
                                            axis=1,
                                        ),
                                    ]
                                )

                                # Save the results to a CSV file
                                results.to_csv("results/raw_data/hyperparams_metrics.csv", index=False)


    
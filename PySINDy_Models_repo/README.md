# Technical files for PySINDy models from the Cranfield Coop Group

This folder contains the technical files for implementing the different PySINDy models. It includes 7 Jupyter Notebooks files to be executed in this order:

## 1. get_data.ipynb

This file extracts DNS data from: https://turbulence.oden.utexas.edu/

## 2. DNS_Data_preparation.ipynb

This file is used to prepare the DNS data retrieved using the interpolation method for PySINDy and PINN models.

## 3. PySINDy_model.ipynb

This file is used to train, simulate and evaluate the PySINDy model.

## 4. PCA.ipynb

This file is used to apply PCA to specific columns of the dataset containing DNS data.

## 5. PySINDy_PCA_Model.ipynb

This file is used to train, simulate and evaluate the PySINDy model combined with PCA.

## 6. PySINDy_GA_Model.ipynb

This file is used to train, simulate and evaluate the PySINDy model optimized by the GA.

## 7. Comparison.ipynb

This file is used to compare different turbulence models.

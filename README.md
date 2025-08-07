# Neural Network for Distribution prediction

This is the repository for the Advanced Deep Learning Lab SS25 project: Predicting Runtime Distributions with TabPFN.

```
@proceedings{eggensperger-ijcai18,
  author = {K. Eggensperger and M. Lindauer and H. Hoos},
  title = {Neural Networks for Predicting Algorithm Runtime Distributions},
  booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI'18)},
  year      = {2018}
}
```

It includes scripts and notebooks for running the experiments shown in the paper.
This code has been written and tested with *Python 3.5*; all dependencies are listed in _requirements.txt_

# Data

The data used to train the networks can be found [here](http://www.ml4aad.org/wp-content/uploads/2018/04/DistNetData.zip).
After downloading, please put the content in `./data/`.

# How to train and evaluate models

1) Create predictions using `eval_model.py`

This script trains different models (DistNet, multi-output RFs, independent RFs) on different distribution families
(inverse Gaussian, Lognormal, Exponential) using crossvalidation. Running the script will train either a DistNet or
both forest-based models on one fold for a given distribution type and scenario. The predictions will be stored using pickle. 

Also see: ```python eval_model.py```

For example:
```
python eval_model.py --model lognormal_distfit.floc --scenario clasp_factoring --fold 0 --seed 100 --save ./TEST_100 --num_train_samples 100
python eval_model.py --model lognormal_nn.floc --scenario clasp_factoring --fold 0 --seed 100 --save ./TEST_100 --num_train_samples 100
```

**NOTE:** To perform the full crossvalidation and the reproduce the results from the paper you need to train each model on folds `[0, 1, ..., 9]` using seeds `[100, 200, ..., 1000]` 
for each distribution and number of training samples `[1, 2, 4, 8, 16, 32, 64, 100]`.

2) Analyse results using one of the two jupyter-notebooks provided in `/notebooks/`

**CreateTable_evalModel-MultiSeed**

Creates a table with average NLLHs for each scenario and model

**PlotSubsets_evalModel-MultiSeed**

Creates plots that show average NLLHs compared to the number of observations per instance used for training the model.

# Codebase Structure and Components

## Core Architecture

### Source Code (`src/`)
- **`distnet.py`**: Main DistNet implementation with Keras/TensorFlow backend
  - `FCNetDistribution`: Base class for distribution-specific neural networks
  - `ParamFCNetInvGaussFloc`: Inverse Gaussian distribution network with fixed location parameter
  - `ParamFCNetLognormalFloc`: Log-normal distribution network with fixed location parameter  
  - `ParamFCNetExponFloc`: Exponential distribution network with fixed location parameter
  - Custom loss functions implementing negative log-likelihood for each distribution family

- **`distnet_torch.py`**: PyTorch implementation of DistNet
  - `DistNet`: PyTorch neural network module for distribution parameter prediction
  - `DistNetModel`: Training and inference wrapper with early stopping and validation
  - Optimized for modern PyTorch workflows with gradient clipping and learning rate scheduling

- **`fcnet.py`**: Fully connected network base classes
  - `FCNetBase`: Configurable neural network architecture with hyperparameter spaces
  - Support for dropout, L2 regularization, batch normalization
  - Integration with ConfigSpace for automated hyperparameter optimization
  - Multiple optimizer and learning rate schedule options

- **`util.py`**: Utility functions
  - Custom EarlyStopping callback for continuous training across folds
  - Data normalization functions (zero mean, unit variance)

### Data Processing (`helper/`)
- **`load_data.py`**: Data loading and parsing utilities
  - Reading algorithm runtime results from CSV files
  - Feature extraction and instance management
  - Support for different data formats and scenarios

- **`preprocess.py`**: Data preprocessing pipeline
  - Timeout removal and instance filtering
  - Status-based filtering (removing CRASHED runs)
  - Constant instance detection and removal
  - Data quality assurance functions

- **`data_source_release.py`**: Data source management for experimental scenarios

### Experimental Scripts (`scripts/`)
- **`eval_model.py`**: Main evaluation script
  - Cross-validation framework for model comparison
  - Support for multiple model types: DistNet (neural networks), Random Forests, distribution fitting
  - Configurable training parameters (epochs, samples, folds)
  - Automated result storage with pickle serialization

- **`eval_lognormal_nn.py`**: Specialized evaluation for log-normal neural networks

### Analysis Notebooks (`notebooks/`)
- **`CreateTable_evalModel-MultiSeed.ipynb`**: Statistical analysis and result tables
- **`CreateTable_evalModel-MultiSeed_table4.ipynb`**: Reproduction of paper Table 4
- **`PlotSubsets_evalModel-MultiSeed.ipynb`**: Visualization of training subset effects
- **`Visualize.ipynb`**: General visualization and plotting utilities
- **`GeneratePictogram.ipynb`**: Pictogram generation for presentations
- **`Kolmogorov-Smirnov.ipynb`**: Statistical goodness-of-fit testing
- **`tabpfn_experiments.ipynb`**: Integration with TabPFN (Tabular Prior-data Fitted Networks)

### Automation Scripts (`shell_scripts/`)
- **`distnet.sh`**: Main training automation script
- **`figure4.sh`**: Automated generation of Figure 4 from the paper
- **`table3.sh`**: Automated generation of Table 3 results
- **`table4.sh`**: Automated generation of Table 4 results

## Model Types Supported

1. **DistNet Neural Networks**: Custom neural networks with distribution-specific loss functions
   - Inverse Gaussian, Log-normal, and Exponential distributions
   - Both Keras/TensorFlow and PyTorch implementations

2. **Baseline Models**: 
   - Multi-output Random Forests
   - Independent Random Forests  
   - Classical distribution fitting (scipy.stats)

3. **Advanced Methods**:
   - TabPFN integration for tabular data modeling
   - Mean prediction networks for comparison

## Key Features

- **Flexible Architecture**: Configurable network depth, width, and hyperparameters
- **Cross-validation Support**: 10-fold cross-validation with multiple random seeds
- **Multiple Backends**: Both TensorFlow/Keras and PyTorch implementations
- **Comprehensive Evaluation**: NLL-based model comparison with statistical testing
- **Automated Experiments**: Shell scripts for reproducing paper results
- **Rich Visualization**: Jupyter notebooks for analysis and plotting

## Dependencies

The project requires Python 3.5+ with scientific computing libraries:
- **Core**: numpy, scipy, scikit-learn
- **Deep Learning**: tensorflow, keras, torch (PyTorch)
- **Experiment Management**: configspace, tabulate
- **Visualization**: matplotlib, jupyter
- **Advanced Models**: tabpfn

# Further notes

On how to train DistNets and preprocess data, please have a look at the script `eval_model.py`. 
Also, please have a look at the other notebooks which provide further options to visualize and analyze runtime data.

# Discovery from Multisource Data

This repository contains code and data for discovering governing partial differential equations from multiple data sources. The method trains neural-network surrogates for several cases, builds candidate PDE libraries through a genetic search, and then refines the selected terms with coefficient-variation-based PIC selection.


## What This Project Does

The workflow is designed for PDE discovery when several datasets share the same governing structure but differ in initial conditions, parameters, or domains.

The main steps are:

1. Load multiple cases from a selected dataset.
2. Split each case into training and validation samples.
3. Train one neural network per case as a differentiable surrogate.
4. Use automatic differentiation to compute spatial and temporal derivatives.
5. Search over candidate PDE terms with a genetic algorithm.
6. Train multicase competitive PINN models to estimate shared PDE coefficients.
7. Rank candidate terms by coefficient variation and save the final discovered equation.

## Repository Layout

| Path | Description |
| --- | --- |
| `code/Discovery_from_multicase.py` | Main 1D multicase discovery script for Burgers, KdV, Allen-Cahn, Klein-Gordon, mixed, and wave-breaking datasets. |
| `code/Discovery_from_multicase_2D.py` | 2D multicase discovery script for pollutant transport and irregular Poisson data. |
| `code/Discovery_from_multicase_3D.py` | 3D multicase discovery script for heat-equation data. |
| `code/MTL_GA.py` | Core 1D neural-network, derivative, library, training, fitness, and genetic-search utilities. |
| `code/MTL_GA_2D.py` | 2D version of the core utilities. |
| `code/MTL_GA_3D.py` | 3D version of the core utilities. |
| `code/Burgers_example.ipynb` | Lightweight Burgers example notebook with optional pretraining, genetic search, and PIC selection cells. |
| `codePoisson_IC_example.ipynb` | Lightweight Poisson example notebook with optional pretraining, genetic search, and PIC selection cells. |
| `data/` | Datasets used by the experiments. See `data/README.md`. |
| `model_save/` | Generated neural-network checkpoints. |
| `result_save/` | Generated genetic-search caches, selected genomes, and final discovered equations. |

## Environments

The code is written for Python 3. Recommended Python version: 3.9 or newer.

Required packages:

- `numpy`
- `pandas`
- `scipy`
- `torch`
- `matplotlib`
- `tqdm`
- `jupyter`
- `ipykernel`

Install them with:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch installation instructions.

## Data

Datasets are stored in `data/`. The main examples use:

- `data/Burgers_equation`: seven 1D Burgers cases.
- `data/Poisson_2D_irregular`: seven irregular 2D Poisson cases.

Other included datasets cover KdV, Allen-Cahn, Klein-Gordon, pollutant transport, 3D heat, heterogeneous 2D flow, wave breaking, and a 2D wave equation. See `data/README.md` for details.

## Quick Start

### Run the Burgers notebook

```bash
jupyter notebook Upload_version/Burgers_example.ipynb
```

The notebook loads the Burgers data, shows the train/validation split, and provides optional cells for:

- pretraining the neural surrogates
- running genetic search
- running PIC selection

The expensive cells are disabled by default. Change the corresponding flags to `True` in the notebook when you want to run them.

### Run the Poisson notebook

```bash
jupyter notebook Upload_version/Poisson_IC_example.ipynb
```

This notebook follows the 2D irregular Poisson branch and uses functions loaded explicitly from `Upload_version/MTL_GA_2D.py`.

## Script-Based Reproduction

The scripts use configuration variables near the top of each file.

### 1D examples

Edit `code/Discovery_from_multicase.py`:

```python
prefix = 'Burgers_IC'
n_train = 100
n_valid = 100
noise_level = 0
mode = 'Discover'
seed = 42
```

Then run:

```bash
python Upload_version/Discovery_from_multicase.py
```

Supported `mode` values:

- `Discover`: pretrain models and run genetic search.
- `Valid`: inspect cached genome fitness scores.
- `PIC_select`: refine terms from the cached best genome and save the final selected equation.

### 2D examples

Edit `code/Discovery_from_multicase_2D.py`:

```python
prefix = 'Poisson_IC'
n_train = 10000
n_valid = 10000
noise_level = 0
mode = 'Discover'
seed = 42
```

Then run:

```bash
python code/Discovery_from_multicase_2D.py
```

### 3D examples

Edit `code/Discovery_from_multicase_3D.py`:

```python
prefix = 'Heat_3D'
n_train = 1000
n_valid = 1000
noise_level = 0
mode = 'Discover'
seed = 42
```

Then run:

```bash
python Upload_version/Discovery_from_multicase_3D.py
```

## Expected Outputs

Training and discovery produce two main output folders:

### `model_save/`

Contains trained neural-network checkpoints for each case. Typical contents:

- `checkpoint_best.pt`
- `checkpoint_last.pt`
- training logs

### `result_save/`

Contains genetic-search and PIC-selection outputs. For example:

```text
result_save/Burgers_IC_100_7/
  ga_config.json
  ga_best_per_gen.json
  fitness_cache.pkl
  final_discovered_equations.json
```

Important files:

- `ga_config.json`: genetic-search settings.
- `ga_best_per_gen.json`: best genome found at each generation.
- `fitness_cache.pkl`: cached genome fitness evaluations.
- `final_discovered_equations.json`: final PIC-selected equation, metrics, and coefficient-variation ranking.

During a successful run, the console prints:

- loaded case shapes
- training progress and validation loss
- evaluated genomes and fitness values
- best genome and best score
- final selected genome and equation terms

## Reproducing the Provided Example Runs

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open `code/Burgers_example.ipynb`.

3. Run the loading and splitting cells.

4. Set `RUN_PRETRAIN = True` and run the pretraining cell.

5. Set `RUN_GENETIC_SEARCH = True` and run the genetic-search cell.

6. Set `RUN_PIC_SELECT = True` and run the PIC-selection cell after `fitness_cache.pkl` is available.

The same sequence applies to `Upload_version/Poisson_IC_example.ipynb`.

## Notes

- The full discovery workflow can be computationally expensive, especially for 2D and 3D cases.
- CUDA is used automatically when available in most training functions.
- The notebooks create output directories before writing generated files, so they can be run from a clean checkout.
- Random sampling uses fixed seeds by default for reproducibility.

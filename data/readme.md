# Data Description

This folder contains the multi-source datasets used by the PDE discovery examples and scripts. Most datasets provide seven cases with different initial conditions or physical settings. Each case is loaded by the scripts.

## Dataset Summary

| Folder | Files | Format | Main variables | Used by |
| --- | ---: | --- | --- | --- |
| `Burgers_equation` | 7 | CSV | `U` matrix, shape `(256, 201)` for each case | `Discovery_from_multicase.py`, `Burgers_example.ipynb` |
| `KdV_equation` | 7 | MAT | `KDV_PARA_RK4_CFD6.uu`, `tt`, `x` | `Discovery_from_multicase.py` |
| `Allen_Cahn_equation` | 7 | MAT | `usol`, `t`, `x`, optional initial-condition metadata | `Discovery_from_multicase.py` |
| `KG_equation` | 7 | MAT | `usol`, `t`, `x`, optional initial-condition metadata | `Discovery_from_multicase.py` |
| `Poisson_2D_irregular` | 7 | MAT | `data`, columns `[x, y, t, u]` | `Discovery_from_multicase_2D.py`, `Poisson_IC_example.ipynb` |
| `Pollutant_transport` | 7 | MAT | `c_save`, `x`, `y`, `t_save` | `Discovery_from_multicase_2D.py` |
| `Heat_3D` | 7 | MAT | `u_save`, `x`, `y`, `z`, `t_save` | `Discovery_from_multicase_3D.py` |
| `Flow_2D_hetero_KLE_CN` | 7 | MAT | `h`, `u`, `K`, `x`, `y`, `t`, `xi`, `Ss` | heterogeneous 2D flow experiments |

## Dataset Notes

### `Burgers_equation`

Each file is named `Burgers_IC{i}.csv`, where `i = 1, ..., 7`. The CSV file contains a solution matrix `U`. The scripts construct:

- `x = linspace(-8, 8, Nx)`
- `t = linspace(0, 10, Nt)`
- `U.shape = (Nx, Nt)`

### `KdV_equation`

Each file is named `KDV_IC{i}.mat`. The solution is stored in a MATLAB struct:

- `KDV_PARA_RK4_CFD6.uu`: solution matrix, typically `(Nx, Nt)`
- `KDV_PARA_RK4_CFD6.tt`: time grid
- `KDV_PARA_RK4_CFD6.x`: spatial grid

### `Allen_Cahn_equation`

Each file is named `Allen_Cahn_IC{i}.mat`. The main fields are:

- `usol`: solution matrix
- `x`: spatial grid
- `t`: time grid
- `ic`, `ic_name`: optional initial-condition metadata

### `KG_equation`

Each file is named `KG_IC{i}.mat`. The main fields are:

- `usol`: solution matrix
- `x`: spatial grid
- `t`: time grid
- `ic`, `ic_name`: optional initial-condition metadata

The default left-hand side for this dataset is `utt`.

### `Poisson_2D_irregular`

Each file is named `Poisson_IC{i}.mat`. The field `data` is an irregular sample matrix:

- column 0: `x`
- column 1: `y`
- column 2: `t`
- column 3: `u`

The example notebook visualizes one time slice as a point cloud.

### `Pollutant_transport`

Each file is named `Pollutant_IC{i}.mat`. The main fields are:

- `c_save`: solution array with shape `(Nt, Nx, Ny)`
- `x`, `y`: spatial grids
- `t_save`: time grid

The scripts rename `c_save` to `U` internally.

### `Heat_3D`

Each file is named `Heat3D_IC{i}.mat`. The main fields are:

- `u_save`: solution array with shape `(Nt, Nx, Ny, Nz)`
- `x`, `y`, `z`: spatial grids
- `t_save`: time grid

### `Flow_2D_hetero_KLE_CN`

Each file is named `GW2D_case{i}.mat`. The dataset contains 2D heterogeneous flow fields:

- `h`: hydraulic head over space and time
- `u`: stored state/output field
- `K`: conductivity field
- `x`, `y`, `t`: grids
- `xi`, `Ss`: case parameters

The `figures` subfolder contains quick-look visualizations for each case.

## Loading Conventions

The scripts convert each dataset into one of the following internal formats:

- structured 1D cases: `{"U": U, "x": x, "t": t}`
- structured 2D cases: `{"U": U, "x": x, "y": y, "t": t}`
- structured 3D cases: `{"U": U, "x": x, "y": y, "z": z, "t": t}`
- irregular point clouds: matrix columns `[x, y, t, u]` or `[t, x, u]`

Train/validation samples are generated inside the scripts with deterministic random seeds.


# Run workflow:
# 1. Select a dataset by editing `prefix`, then load all available cases.
# 2. Optionally inject noise through `noise_level`.
# 3. Split every case into training and validation samples.
# 4. Use `mode` to choose the stage:
#    - `Discover`: run MCO-PDE to discover PDE from multi-source dataset and run genetic search.
#    - `Valid`: inspect cached genome fitness scores.
#    - `PIC_select`: refine terms from the cached best genome.

from MTL_GA import *
import os
import pandas as pd
import time
from scipy.io import loadmat
import random

# Add synthetic noise to each case in-place and return the updated dataset.
def add_noise(dataset, noise_level,noise_type='Gaussian', seed=42):
    np.random.seed(seed)
    for key, data in dataset.items():
        U = data["U"]
        if noise_type == 'Gaussian':
            noise_value = (noise_level / 100) * np.std(U) * np.random.randn(*U.shape)
            data["U"] = U + noise_value
        elif noise_type == 'Uniform':
            noise_value=np.random.uniform(-1, 1, size=U.shape)
            data["U"] = U * (1 + 0.01 * noise_level * noise_value)
    return dataset
# Convert MATLAB string-like objects to plain Python strings.
def _matlab_str_to_py(s):
        """Convert MATLAB saved string (char array / object) to Python str safely."""
        if isinstance(s, str):
            return s
        if isinstance(s, bytes):
            return s.decode("utf-8", errors="ignore")
        try:
            if hasattr(s, "shape") and len(s.shape) > 0:
                return "".join(np.ravel(s).astype(str))
        except Exception:
            pass
        return str(s)

# Main experiment configuration.
prefix = 'Burgers_IC'  # Dataset family to load.
n_train=100  # Number of training samples per case.
n_valid=100  # Number of validation samples per case.
noise_level=0  # Noise percentage; keep 0 for clean data.
mode='Discover'  # Workflow stage: Discover, Valid, PIC_select, or plot.
case_index=5  # Reserved for single-case workflows.
seed=42  # Random seed for reproducible sampling and noise.
dataset={}  # Raw case data before train/validation splitting.

if prefix == 'Burgers_IC':
    data_dir = "data/Burgers_equation"
    LHS='ut'
    for i in range(1,8):
        path = os.path.join(data_dir, f"{prefix}{i}.csv")
        if not os.path.exists(path):
            print(f"[warn] missing: {path}, skip.")
            continue
        U = np.array(pd.read_csv(path, header=None))  # Shape: (Nt, Nx).
        Nx, Nt = U.shape
        t = np.linspace(0,10, Nt)
        x = np.linspace(-8,8, Nx)
        dataset[prefix+f'_{i}'] ={'U':U, 't':t, 'x':x}

if prefix=='KdV_IC':
    LHS='ut'

    # Load one KdV MATLAB case and normalize array orientation.
    def load_kdv_mat(mat_path: str):
        """
        Read one KDV_IC*.mat file saved from MATLAB.
        Expected structure:
          KDV_PARA_RK4_CFD6.uu : (Nx, Nt)
          KDV_PARA_RK4_CFD6.tt : (Nt,)
          KDV_PARA_RK4_CFD6.x  : (Nx,)
        """
        mdict = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "KDV_PARA_RK4_CFD6" not in mdict:
            raise KeyError(f"Missing 'KDV_PARA_RK4_CFD6' in {mat_path}")

        s = mdict["KDV_PARA_RK4_CFD6"]

        U = np.asarray(s.uu, dtype=np.float64)
        t = np.asarray(s.tt, dtype=np.float64).reshape(-1)
        x = np.asarray(s.x, dtype=np.float64).reshape(-1)

        # --- ensure U matches x,t ---
        # expected: U.shape == (len(x), len(t))
        if U.shape == (t.size, x.size):
            U = U.T
        elif U.shape != (x.size, t.size):
            raise ValueError(
                f"Shape mismatch in {mat_path}: U{U.shape}, x{(x.size,)}, t{(t.size,)}"
            )

        return U, t, x

    data_dir = "data/KdV_equation"  # change to your folder

    dataset = {}

    for i in range(1,8):
        path = os.path.join(data_dir, f"{prefix}{i}.mat")
        if not os.path.exists(path):
            print(f"[warn] missing: {path}, skip.")
            continue

        try:
            U, t, x = load_kdv_mat(path)
        except Exception as e:
            print(f"[warn] failed to load {path}: {e}")
            continue

        dataset[f"{prefix}_{i}"] = {"U": U, "t": t, "x": x}

        print(f"[ok] loaded {path}: U={U.shape}, x={x.shape}, t={t.shape}")

if prefix=='Allen_Cahn_IC':
    LHS='ut'



    # Load one Allen-Cahn MATLAB case and normalize array orientation.
    def load_allen_cahn_mat(mat_path: str):
        """
        Read one Allen_Cahn_IC*.mat file saved from MATLAB.

        Expected variables in MAT:
          usol : (Nx, Nt)
          t    : (Nt,)
          x    : (Nx,)
          ic (optional)
          ic_name (optional)
        """
        mdict = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        # core fields
        if "usol" not in mdict or "t" not in mdict or "x" not in mdict:
            raise KeyError(f"Missing one of ['usol','t','x'] in {mat_path}")

        U = np.asarray(mdict["usol"], dtype=np.float64)
        t = np.asarray(mdict["t"], dtype=np.float64).reshape(-1)
        x = np.asarray(mdict["x"], dtype=np.float64).reshape(-1)

        # shape check / auto-fix
        # expected: U.shape == (len(x), len(t)) == (Nx, Nt)
        if U.shape == (t.size, x.size):
            U = U.T
        elif U.shape != (x.size, t.size):
            raise ValueError(
                f"Shape mismatch in {mat_path}: U{U.shape}, x{(x.size,)}, t{(t.size,)}"
            )

        meta = {}
        if "ic" in mdict:
            try:
                meta["ic"] = int(np.asarray(mdict["ic"]).squeeze())
            except Exception:
                meta["ic"] = mdict["ic"]

        if "ic_name" in mdict:
            meta["ic_name"] = _matlab_str_to_py(mdict["ic_name"])

        return U, t, x, meta


    # Main loading loop for all Allen-Cahn cases.
    data_dir = "data/Allen_Cahn_equation"  # Update this path for your local data folder.
    prefix = "Allen_Cahn_IC"  # Expected files: Allen_Cahn_IC1.mat ... Allen_Cahn_IC7.mat.

    dataset = {}

    for i in range(1, 8):
        path = os.path.join(data_dir, f"{prefix}{i}.mat")
        if not os.path.exists(path):
            print(f"[warn] missing: {path}, skip.")
            continue

        try:
            U, t, x, meta = load_allen_cahn_mat(path)
        except Exception as e:
            print(f"[warn] failed to load {path}: {e}")
            continue

        key = f"{prefix}_{i}"
        dataset[key] = {"U": U, "t": t, "x": x, **meta}

        print(f"[ok] loaded {path}: U={U.shape}, x={x.shape}, t={t.shape}, meta={meta}")

if prefix=='KG_IC':
    LHS='utt'

    # Load one Klein-Gordon MATLAB case and normalize array orientation.
    def load_kg_mat(mat_path: str):
        """
        Read one KG_IC*.mat file saved from MATLAB.

        Expected variables in MAT:
          usol : (Nx, Nt)
          t    : (Nt,)
          x    : (Nx,)
          ic (optional)
          ic_name (optional)
        """
        mdict = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

        if "usol" not in mdict or "t" not in mdict or "x" not in mdict:
            raise KeyError(f"Missing one of ['usol','t','x'] in {mat_path}")

        U = np.asarray(mdict["usol"], dtype=np.float64)
        t = np.asarray(mdict["t"], dtype=np.float64).reshape(-1)
        x = np.asarray(mdict["x"], dtype=np.float64).reshape(-1)

        # ensure expected shape: U == (Nx, Nt)
        if U.shape == (t.size, x.size):
            U = U.T
        elif U.shape != (x.size, t.size):
            raise ValueError(
                f"Shape mismatch in {mat_path}: U{U.shape}, x{(x.size,)}, t{(t.size,)}"
            )

        meta = {}
        if "ic" in mdict:
            try:
                meta["ic"] = int(np.asarray(mdict["ic"]).squeeze())
            except Exception:
                meta["ic"] = mdict["ic"]

        if "ic_name" in mdict:
            meta["ic_name"] = _matlab_str_to_py(mdict["ic_name"])

        return U, t, x, meta


    # Main loading loop for all Klein-Gordon cases.
    data_dir = "data/KG_equation"  # Update this path for your local data folder.
    prefix = "KG_IC"  # Expected files: KG_IC1.mat ... KG_IC7.mat.

    dataset = {}

    for i in range(1, 8):
        path = os.path.join(data_dir, f"{prefix}{i}.mat")
        if not os.path.exists(path):
            print(f"[warn] missing: {path}, skip.")
            continue

        try:
            U, t, x, meta = load_kg_mat(path)
        except Exception as e:
            print(f"[warn] failed to load {path}: {e}")
            continue

        key = f"{prefix}_{i}"
        dataset[key] = {"U": U, "t": t, "x": x, **meta}

        print(f"[ok] loaded {path}: U={U.shape}, x={x.shape}, t={t.shape}, meta={meta}")

    # At this point, dataset contains all KG cases that were found.

if noise_level>0:
    prefix=prefix+f"_noise{noise_level}"
    dataset=add_noise(dataset,noise_level=noise_level,seed=seed)



# Case data after random train/validation splitting.
split_dataset = {}
for key, data in dataset.items():
    if prefix=='Wave_breaking':
        X, Y, X_train, Y_train, X_valid, Y_valid = build_xy_split_wave_breaking(
            data, n_train=0.8, n_valid=0.2, seed=seed
        )
    else:
        X, Y, X_train, Y_train, X_valid, Y_valid = build_xy_split(
            data, n_train=n_train, n_valid=n_valid, seed=seed
        )
    split_dataset[key] = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_valid": X_valid,
        "Y_valid": Y_valid,
    }


# Estimate local PDE coefficients over x-windows for single-case CV ranking.
def _fit_coefficients_from_metadata_windows(
    model,
    data,
    library_fn,
    LHS,
    n_windows=5,
    device=None,
):
    """
    For a single case, split x into local windows and fit coefficients on
    metadata generated by the neural network over each x-window and all t.

    Returns
    -------
    beta_mat : np.ndarray
        Shape (n_terms, n_windows_kept)
    term_names : list[str]
    window_ranges : list[tuple[float, float]]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    if x.size < 2 or t.size < 2:
        raise ValueError("Single-case local-window CV requires at least 2 x points and 2 t points.")

    x_edges = np.linspace(float(np.min(x)), float(np.max(x)), n_windows + 1)
    beta_all = []
    term_names = None
    window_ranges = []

    model = model.to(device).eval()

    for i in range(n_windows):
        x_left = x_edges[i]
        x_right = x_edges[i + 1]
        if i == n_windows - 1:
            x_mask = (x >= x_left) & (x <= x_right)
        else:
            x_mask = (x >= x_left) & (x < x_right)

        x_local = x[x_mask]
        if x_local.size < 2:
            continue

        xx, tt = np.meshgrid(x_local, t, indexing="ij")
        Xc_np = np.column_stack([xx.reshape(-1), tt.reshape(-1)])
        Xc = torch.from_numpy(Xc_np).float().to(device)
        Xc.requires_grad_(True)

        u_c = model(Xc).reshape(-1, 1)
        Left_u_t, u_x, u_xx, u_xxx = compute_derivatives(u_c, Xc, LHS=LHS)

        lib_out = library_fn(u_c, u_x, u_xx, u_xxx)
        if isinstance(lib_out, tuple):
            Phi, lib_names = lib_out
            if term_names is None:
                term_names = lib_names
        else:
            Phi = lib_out

        sigma = torch.sqrt((Phi ** 2).mean(dim=0, keepdim=True)) + 1e-8
        Phi_s = Phi / sigma
        sol = torch.linalg.lstsq(Phi_s, Left_u_t)
        beta_s = sol.solution
        beta = (beta_s / sigma.T)

        beta_all.append(beta.detach().cpu().numpy().reshape(-1))
        window_ranges.append((float(x_left), float(x_right)))

    if not beta_all:
        raise ValueError("No valid local windows were available for CV computation.")

    beta_mat = np.stack(beta_all, axis=1)  # (n_terms, n_windows_kept)
    if term_names is None:
        term_names = [f"term_{i}" for i in range(beta_mat.shape[0])]

    return beta_mat, term_names, window_ranges


if mode=='Discover':
    start_time = time.time()
    models = [NN(Num_Hidden_Layers=5,
        Neurons_Per_Layer=50,
        Input_Dim=2,
        Output_Dim=1,
        Data_Type=torch.float32,
        Device='cuda',
        Activation_Function='Sin',
        Batch_Norm=False) for _ in split_dataset]

    save_dir = os.path.join("result_save", f"{prefix}_{n_train}_{len(split_dataset)}")
    os.makedirs(save_dir, exist_ok=True)

    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
    pretrain_cases(models, split_dataset, optimizers, n_train=n_train,n_valid=n_valid, device="cuda", log_dir=f"model_save/{prefix}_{n_train}",seed=seed)


    if "Allen_Cahn" in prefix:
        pop_size=100
        generations=20
    else:
        pop_size=50
        generations=10

    best_genome, best_score, best_info = genetic_search(
        models=models,
        cases=split_dataset,
        prefix=prefix,
        n_train=n_train,
        n_valid=n_valid,
        seed=seed,
        LHS=LHS,
        pop_size=pop_size,
        generations=generations,
        PINN_num_epoch=500,
    )
    print("Best genome:", best_genome)
    print("Best loss:", best_score)

    end_time = time.time()
    print('Total time:', end_time - start_time)


if mode=='Valid':
    # Load cached genome fitness results for the current experiment.
    def load_fitness_cache(n_train):
        cache_path = os.path.join("result_save", f"{prefix}_{n_train}_{len(dataset)}", "fitness_cache.pkl")
        if not os.path.exists(cache_path):
            print(f"no cache found: {cache_path}")
            return {}
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache

    # Load cached fitness scores, sort by score, and print the top candidates.
    fitness_cache = load_fitness_cache(n_train)
    if fitness_cache:
        # Cache format: key -> (score, info).
        sorted_items = sorted(fitness_cache.items(), key=lambda kv: kv[1][0])
        print("Top 5 genomes in cache:")
        for i, (key, (score, info)) in enumerate(sorted_items[:5], start=1):
            print(f"{i}. score={score:.6e}, key={key}")
            print(f"   info={info}")


if mode=='PIC_select':
    rng = np.random.default_rng(seed)

    models = [NN(Num_Hidden_Layers=5,
        Neurons_Per_Layer=50,
        Input_Dim=2,
        Output_Dim=1,
        Data_Type=torch.float32,
        Device='cuda',
        Activation_Function='Sin',
        Batch_Norm=False) for _ in split_dataset]

    cache_path = os.path.join("result_save", f"{prefix}_{n_train}_{len(dataset)}", "fitness_cache.pkl")
    if not os.path.exists(cache_path):
        raise OSError(f"no cache found: {cache_path}")

    with open(cache_path, "rb") as f:
        fitness_cache = pickle.load(f)

    sorted_items = sorted(fitness_cache.items(), key=lambda kv: kv[1][0])
    best_genome=sorted_items[0][0]
    print(f"Best genome: {best_genome}")


    library_fn, term_names = genome_to_library(best_genome)
    models,_ = train_multicase_competitive(
        models=models,
        cases=split_dataset,
        prefix=prefix,
        LHS=LHS,
        num_epochs=1000,
        n_train=n_train,
        n_valid=n_valid,
        seed=seed,
        n_colloc=1000,
        library_fn=library_fn,
        return_metrics=False)


    data_cache = []
    case_keys = list(split_dataset.keys())

    for key in case_keys:
        data = split_dataset[key]
        X_train, Y_train, X_valid, Y_valid = data['X_train'], data['Y_train'], data['X_valid'], data['Y_valid']
        x_min, x_max = float(X_train[:, 0].min()), float(X_train[:, 0].max())
        t_min, t_max = float(X_train[:, 1].min()), float(X_train[:, 1].max())
        data_cache.append((x_min, x_max, t_min, t_max))


    device = "cuda" if torch.cuda.is_available() else "cpu"
    case_keys = list(split_dataset.keys())

    if len(dataset) == 1:
        single_key = case_keys[0]
        beta_mat, term_names_cv, window_ranges = _fit_coefficients_from_metadata_windows(
            model=models[0],
            data=dataset[single_key],
            library_fn=library_fn,
            LHS=LHS,
            n_windows=5,
            device=device,
        )
        if term_names is None:
            term_names = term_names_cv

        beta_mean = np.mean(beta_mat, axis=1)
        beta_std = np.std(beta_mat, axis=1)
        cv = beta_std / (np.abs(beta_mean) + 1e-12)

        print("Single-case local-window CV (5 x-windows over full t):")
        for idx_win, (x_left, x_right) in enumerate(window_ranges, start=1):
            beta_str = ", ".join(f"{val:.4e}" for val in beta_mat[:, idx_win - 1])
            print(f"  window {idx_win}: x in [{x_left:.4g}, {x_right:.4g}] beta=[{beta_str}]")
    else:
        beta_all=[]
        for case_idx, key in enumerate(case_keys):
            model = models[case_idx].to(device).eval()
            x_min, x_max, t_min, t_max = data_cache[case_idx]
            x_c = rng.uniform(x_min, x_max, size=(1000, 1))
            t_c = rng.uniform(t_min, t_max, size=(1000, 1))
            Xc_np = np.hstack([x_c, t_c])

            Xc = torch.from_numpy(Xc_np).float().to(device)
            Xc.requires_grad_(True)

            u_c = model(Xc).reshape(-1, 1)

            Left_u_t, u_x, u_xx, u_xxx = compute_derivatives(u_c, Xc, LHS=LHS)

            lib_out = library_fn(u_c, u_x, u_xx, u_xxx)
            if isinstance(lib_out, tuple):
                Phi, lib_names = lib_out
                if term_names is None:
                    term_names = lib_names
            else:
                Phi = lib_out

            # scale columns by RMS (no centering)
            sigma = torch.sqrt((Phi ** 2).mean(dim=0, keepdim=True)) + 1e-8
            Phi_s = Phi / sigma
            sol = torch.linalg.lstsq(Phi_s, Left_u_t)  # QR/SVD-based
            beta_s = sol.solution
            beta = (beta_s / sigma.T)  # back to original scale
            beta_all.append(beta)

        # beta_all: list of (K,1) tensors
        beta_mat = torch.cat([b.detach().cpu() for b in beta_all], dim=1).numpy()  # (K, n_cases)

        beta_mean = np.mean(beta_mat, axis=1)
        beta_std = np.std(beta_mat, axis=1)
        cv = beta_std / (np.abs(beta_mean) + 1e-12)

    cv_sorted_idx = np.argsort(cv)  # small -> large

    print("CV:")
    for idx in cv_sorted_idx:
        print(f"{term_names[idx]}: cv={cv[idx]:.4e}")

    cv_list = []
    for idx in cv_sorted_idx:
        cv_list.append({
            "term": term_names[idx],
            "cv": float(cv[idx]),
        })


    term_names = ["*".join(t) for t in best_genome]
    sorted_term_names = [term_names[i] for i in cv_sorted_idx]

    # Build a lookup table from term name to CV value.
    term_cv = {term_names[i]: cv[i] for i in range(len(term_names))}


    results = []
    current_genome = []

    for step, term_name in enumerate(sorted_term_names, start=1):
        added = None
        for t in best_genome:
            if "*".join(t) == term_name:
                added = list(t)
                break
        if added is None:
            continue

        current_genome.append(added)

        library_fn, _ = genome_to_library(current_genome)
        metrics = train_multicase_competitive(
            models=models,
            cases=split_dataset,
            prefix=prefix,
            LHS=LHS,
            num_epochs=1000,
            n_train=n_train,
            n_valid=n_valid,
            seed=seed,
            n_colloc=1000,
            library_fn=library_fn,
            return_metrics=True
        )
        fitness_val = metrics["loss_data"] * metrics["loss_pde"]

        # Compute the mean CV for the current genome.
        current_terms = ["*".join(t) for t in current_genome]
        cv_mean = np.mean([term_cv[name] for name in current_terms])
        score = fitness_val * cv_mean

        results.append((current_genome.copy(), fitness_val, cv_mean, score))
        print(f"[step {step}] added={added}, genome={current_genome}")
        print(f"  fitness={fitness_val:.6e}, cv_mean={cv_mean:.6e}, score={score:.6e}\n")

    # Select the genome with the smallest score.
    best = min(results, key=lambda x: x[3])
    best_genome_final = best[0]
    print("\n=== All genomes and metrics ===")
    for g, fval, cvm, sc in results:
        print(f"genome={g}, fitness={fval:.6e}, cv_mean={cvm:.6e}, score={sc:.6e}")

    print(f"\n=== Final selected ===")
    print(f"genome={best_genome_final}, score={best[3]:.6e}, fitness={best[1]:.6e}, cv_mean={best[2]:.6e}")

    save_dir = os.path.join("result_save", f"{prefix}_{n_train}_{len(dataset)}")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "final_discovered_equations.json")

    all_entries = []
    for g, fval, cvm, sc in results:
        all_entries.append({
            "genome": g,
            "fitness": float(fval),
            "cv_mean": float(cvm),
            "score": float(sc),
        })

    final_selected = {
        "genome": best_genome_final,
        "fitness": float(best[1]),
        "cv_mean": float(best[2]),
        "score": float(best[3]),
    }

    payload = {
        "all_genomes_and_metrics": all_entries,
        "final_selected": final_selected,
    }

    library_fn, term_names = genome_to_library(best_genome_final)
    models,lines = train_multicase_competitive(
        models=models,
        cases=split_dataset,
        prefix=prefix,
        LHS=LHS,
        num_epochs=1000,
        n_train=n_train,
        n_valid=n_valid,
        seed=seed,
        n_colloc=1000,
        library_fn=library_fn,
        return_metrics=False
    )
    payload["final_equation_lines"] = lines
    payload["cv_list"] = cv_list

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved to {out_path}")

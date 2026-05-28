# Run workflow:
# 1. Select a dataset by editing `prefix`, then load all available cases.
# 2. Optionally inject noise through `noise_level`.
# 3. Split every case into training and validation samples.
# 4. Use `mode` to choose the stage:
#    - `Discover`: run MCO-PDE to discover PDE from multi-source dataset and run genetic search.
#    - `Valid`: inspect cached genome fitness scores.
#    - `PIC_select`: refine terms from the cached best genome.

from MTL_GA_2D import *
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
prefix = 'Pollutant_IC'  # Dataset family to load.
n_train=10000  # Number of training samples per case.
n_valid=10000  # Number of validation samples per case.
noise_level=0  # Noise percentage; keep 0 for clean data.
mode='Discover'  # Workflow stage: Discover, Valid, PINN, or PIC_select.
case_index=7  # Case index used by single-case PINN mode.
seed=42  # Random seed for reproducible sampling and noise.
dataset={}  # Raw case data before train/validation splitting.

if prefix=='Pollutant_IC':
    '''
    vx = 0.2;
    vy = 0.15;
    Dx = 5e-2;
    Dy = 5e-2;
    ut=0.05*uxx+0.05*uyy-0.4*ux-0.25*uy
    '''

    LHS='ut'
    # Load one pollutant-transport MATLAB case and return solution and grids.
    def load_pollutant_mat(mat_path: str, field_name: str = "c_save"):
        """
        Read MATLAB .mat saved by the MATLAB solver:

        Returns:
            u: np.ndarray  (Nt, Nx, Ny)  -- pollutant concentration (renamed as u)
            x: np.ndarray  (Nx,)
            y: np.ndarray  (Ny,)
            t: np.ndarray  (Nt,)

        Notes:
          - If you saved variable name is 'c', keep field_name='c'.
          - If you saved it as 'u', set field_name='u'.
        """
        md = loadmat(mat_path)

        def _vec(key):
            if key not in md:
                raise KeyError(f"Missing key '{key}' in {mat_path}. Keys={list(md.keys())}")
            return np.asarray(md[key]).squeeze().astype(np.float64)

        if field_name not in md:
            raise KeyError(f"Missing field '{field_name}' in {mat_path}. Keys={list(md.keys())}")

        u = np.asarray(md[field_name], dtype=np.float64)
        t = _vec("t_save")
        x = _vec("x")
        y = _vec("y")

        return u, x, y, t


    dataset = {}
    data_dir = "data/Pollutant_transport"  # Update this path for your local data folder.
    prefix = "Pollutant_IC"  # Expected files: Pollutant_IC1.mat ... Pollutant_IC7.mat.
    for i in range(1,8):
        path = os.path.join(data_dir, f"{prefix}{i}.mat")
        if not os.path.exists(path):
            print(f"[warn] missing: {path}, skip.")
            continue

        try:
            U, x, y, t = load_pollutant_mat(path)
        except Exception as e:
            print(f"[warn] failed to load {path}: {e}")
            continue

        key = f"{prefix}_{i}"

        # Store solution values and coordinate grids for this case.
        dataset[key] = {"U": U, "t": t, "x": x, "y": y}
        # Print the loaded array shapes for quick sanity checking.

        print(f"[ok] loaded {path}: U={U.shape}, x={x.shape}, t={t.shape}, y={y.shape}")

if prefix=='Poisson_IC':
    LHS='ut'

    dataset = {}
    prefix = "Poisson_IC"  # File-name prefix for Poisson cases.
    mat_dir = "Poisson_2D_irregular"  # Directory that contains Poisson .mat files.

    for i in range(1, 8):
        fp=f"data/{mat_dir}/{prefix}{i}.mat"
        mat = loadmat(fp)
        data = np.asarray(mat["data"])  # Shape: (N, 4).
        key = f"{prefix}_{i}"
        dataset[key] = data
        print(data.shape)

if noise_level>0:
    prefix=prefix+f"_noise{noise_level}"
    dataset=add_noise(dataset,noise_level=noise_level,seed=seed)



# Case data after random train/validation splitting.
split_dataset = {}
for key, data in dataset.items():
    if prefix=='Pollutant_IC':
        X, Y, X_train, Y_train, X_valid, Y_valid = build_xyz_split(
            data, n_train=n_train, n_valid=n_valid, seed=seed
        )
    elif prefix=='Poisson_IC':
        X, Y, X_train, Y_train, X_valid, Y_valid = build_matrix_split(
            data, n_train=n_train, n_valid=n_valid, seed=seed
        )
    split_dataset[key] = {
        "X":X,
        "Y":Y,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_valid": X_valid,
        "Y_valid": Y_valid,
    }
    print("X_shape:",X.shape)
    print("train_shape:",X_train.shape)



if mode=='Discover':
    start_time = time.time()
    models = [NN(Num_Hidden_Layers=5,
        Neurons_Per_Layer=50,
        Input_Dim=3,
        Output_Dim=1,
        Data_Type=torch.float32,
        Device='cuda',
        Activation_Function="Sin",
        Batch_Norm=False) for _ in split_dataset]

    save_dir = os.path.join("result_save", f"{prefix}_{n_train}_{len(split_dataset)}")
    os.makedirs(save_dir, exist_ok=True)

    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-3) for m in models]
    pretrain_cases(models, split_dataset, optimizers, n_train=n_train,num_epochs=50000, device="cuda", log_dir=f"model_save/{prefix}_{n_train}",seed=seed)


    pop_size=50
    generations=10

    best_genome, best_score, best_info = genetic_search_2D(
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
        Input_Dim=3,  # 2D: x,y,t
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
    best_genome = sorted_items[0][0]
    print(f"Best genome: {best_genome}")

    library_fn, term_names = genome_to_library_2d(best_genome)
    models,_ = train_multicase_competitive_2d(
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

    data_cache = []
    case_keys = list(split_dataset.keys())

    for key in case_keys:
        data = split_dataset[key]
        X_train = data['X_train']
        x_min, x_max = float(X_train[:, 0].min()), float(X_train[:, 0].max())
        y_min, y_max = float(X_train[:, 1].min()), float(X_train[:, 1].max())
        t_min, t_max = float(X_train[:, 2].min()), float(X_train[:, 2].max())
        data_cache.append((x_min, x_max, y_min, y_max, t_min, t_max))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_all=[]
    for case_idx, key in enumerate(case_keys):
        model = models[case_idx].to(device).eval()
        x_min, x_max, y_min, y_max, t_min, t_max = data_cache[case_idx]

        x_c = rng.uniform(x_min, x_max, size=(1000, 1))
        y_c = rng.uniform(y_min, y_max, size=(1000, 1))
        t_c = rng.uniform(t_min, t_max, size=(1000, 1))
        Xc_np = np.hstack([x_c, y_c, t_c])

        Xc = torch.from_numpy(Xc_np).float().to(device)
        Xc.requires_grad_(True)

        u_c = model(Xc).reshape(-1, 1)

        Left_u_t, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy = compute_derivatives_2d(u_c, Xc, LHS=LHS)

        lib_out = library_fn(u_c, u_x, u_xx, u_y, u_yy, u_xxx, u_yyy)
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
        beta_all.append(beta)

    beta_mat = torch.cat([b.detach().cpu() for b in beta_all], dim=1).numpy()
    beta_mean = np.mean(beta_mat, axis=1)
    beta_std = np.std(beta_mat, axis=1)
    cv = beta_std / (np.abs(beta_mean) + 1e-12)

    cv_sorted_idx = np.argsort(cv)
    print("CV:")
    for idx in cv_sorted_idx:
        print(f"{term_names[idx]}: cv={cv[idx]:.4e}")

    cv_list = [{"term": term_names[idx], "cv": float(cv[idx])} for idx in cv_sorted_idx]

    term_names = ["*".join(t) for t in best_genome]
    sorted_term_names = [term_names[i] for i in cv_sorted_idx]
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

        library_fn, _ = genome_to_library_2d(current_genome)
        metrics = train_multicase_competitive_2d(
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

        current_terms = ["*".join(t) for t in current_genome]
        cv_mean = np.mean([term_cv[name] for name in current_terms])
        score = fitness_val * cv_mean

        results.append((current_genome.copy(), fitness_val, cv_mean, score))
        print(f"[step {step}] added={added}, genome={current_genome}")
        print(f"  fitness={fitness_val:.6e}, cv_mean={cv_mean:.6e}, score={score:.6e}\n")

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

    library_fn, term_names = genome_to_library_2d(best_genome_final)
    models, lines = train_multicase_competitive_2d(
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




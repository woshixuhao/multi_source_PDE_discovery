import numpy as np
import copy
import torch
from torch import nn
import math
import copy
import os
from tqdm import tqdm
import json
import pickle
import matplotlib.pyplot as plt
# Trainable rational activation used by the neural networks.
class Rational(torch.nn.Module):
    def __init__(self,
                 Data_Type = torch.float32,
                 Device    = torch.device('cpu')):
        # This activation function is based on the following paper:
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational, self).__init__()

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU. These coefficients are listed
        # in appendix A of the paper.
        self.a = torch.nn.parameter.Parameter(
                        torch.tensor((1.1915, 1.5957, 0.5, .0218),
                                     dtype = Data_Type,
                                     device = Device))
        self.a.requires_grad_(True)

        self.b = torch.nn.parameter.Parameter(
                        torch.tensor((2.3830, 0.0, 1.0),
                                     dtype = Data_Type,
                                     device = Device))
        self.b.requires_grad_(True)

    def forward(self, X : torch.tensor):
        """ This function applies a rational function to each element of X.
        ------------------------------------------------------------------------
        Arguments:
        X: A tensor. We apply the rational function to every element of X.
        ------------------------------------------------------------------------
        Returns:
        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

        # Create aliases for self.a and self.b. This makes the code cleaner.
        a = self.a
        b = self.b

        # Evaluate the numerator and denominator. Because of how the * and +
        # operators work, this gets applied element-wise.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X))
        D_X = b[0] + X*(b[1] + b[2]*X)

        # Return R = N_X/D_X. This is also applied element-wise.
        return N_X/D_X

# Sine activation wrapper used by SIREN-style networks.
class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        x = torch.sin(x)
        return x

# Simple fully connected neural network used as an auxiliary architecture.
class ANN(nn.Module):
    def __init__(self,in_neuron,hidden_neuron,out_neuron):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(in_neuron,hidden_neuron)
        self.layer2 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer3 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer4 = nn.Linear(hidden_neuron, hidden_neuron)
        self.layer5 = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        x=self.layer1(x)
        x=torch.sin(x)
        x=self.layer2(x)
        x=torch.sin(x)
        x=self.layer3(x)
        x=torch.sin(x)
        x=self.layer4(x)
        x=torch.sin(x)
        x=self.layer5(x)
        return x

# Main fully connected network with configurable activation and batch normalization.
class NN(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers   : int          = 3,
                 Neurons_Per_Layer   : int          = 20,   # Neurons in each Hidden Layer
                 Input_Dim           : int          = 1,    # Dimension of the input
                 Output_Dim          : int          = 1,    # Dimension of the output
                 Data_Type           : torch.dtype  = torch.float32,
                 Device              : torch.device = torch.device('cpu'),
                 Activation_Function : str          = "Tanh",
                 Batch_Norm          : bool         = False):
        # For the code below to work, Num_Hidden_Layers, Neurons_Per_Layer,
        # Input_Dim, and Output_Dim must be positive integers.
        assert(Num_Hidden_Layers   > 0), "Num_Hidden_Layers must be positive. Got %du" % Num_Hidden_Layers;
        assert(Neurons_Per_Layer   > 0), "Neurons_Per_Layer must be positive. Got %u" % Neurons_Per_Layer;
        assert(Input_Dim           > 0), "Input_Dim must be positive. Got %u"  % Input_Dim;
        assert(Output_Dim          > 0), "Output_Dim must be positive. Got %u" % Output_Dim;

        super(NN, self).__init__()

        # Define object attributes.
        self.Input_Dim          : int  = Input_Dim
        self.Output_Dim         : int  = Output_Dim
        self.Num_Hidden_Layers  : int  = Num_Hidden_Layers
        self.Batch_Norm         : bool = Batch_Norm

        # Initialize the Layers. We hold all layers in a ModuleList.
        self.Layers = torch.nn.ModuleList()

        # Initialize Batch Normalization, if we're doing that.
        if(Batch_Norm == True):
            self.Norm_Layer = torch.nn.BatchNorm1d(
                                    num_features = Input_Dim,
                                    dtype        = Data_Type,
                                    device       = Device)

        # Append the first hidden layer. The domain of this layer is
        # R^{Input_Dim}. Thus, in_features = Input_Dim. Since this is a hidden
        # layer, its co-domain is R^{Neurons_Per_Layer}. Thus, out_features =
        # Neurons_Per_Layer.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Input_Dim,
                                out_features = Neurons_Per_Layer,
                                bias         = True ).to(dtype = Data_Type, device = Device))

        # Now append the rest of the hidden layers. Each maps from
        # R^{Neurons_Per_Layer} to itself. Thus, in_features = out_features =
        # Neurons_Per_Layer. We start at i = 1 because we already created the
        # 1st hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(torch.nn.Linear(
                                    in_features  = Neurons_Per_Layer,
                                    out_features = Neurons_Per_Layer,
                                    bias         = True ).to(dtype = Data_Type, device = Device))

        # Now, append the Output Layer, which has Neurons_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(torch.nn.Linear(
                                in_features  = Neurons_Per_Layer,
                                out_features = Output_Dim,
                                bias         = True ).to(dtype = Data_Type, device = Device))

        # Initialize the weight matrices, bias vectors in the network.
        if(Activation_Function == "Tanh" or Activation_Function == "Rational"):
            Gain : float = 0
            if  (Activation_Function == "Tanh"):
                Gain = 5./3.
            elif(Activation_Function == "Rational"):
                Gain = 1.41

            for i in range(self.Num_Hidden_Layers + 1):
                torch.nn.init.xavier_normal_(self.Layers[i].weight, gain = Gain)
                torch.nn.init.zeros_(self.Layers[i].bias)

        elif(Activation_Function == "Sin"):
            # The SIREN paper suggests initializing the elements of every weight
            # matrix (except for the first one) by sampling a uniform
            # distribution over [-c/root(n), c/root(n)], where c > root(6),
            # and n is the number of neurons in the layer. I use c = 3 > root(6).
            #
            # Further, for simplicity, I initialize each bias vector to be zero.
            a : float = 3./math.sqrt(Neurons_Per_Layer)
            for i in range(0, self.Num_Hidden_Layers + 1):
                torch.nn.init.uniform_( self.Layers[i].weight, -a, a)
                torch.nn.init.zeros_(   self.Layers[i].bias)

        # Finally, set the Network's activation functions.
        self.Activation_Functions = torch.nn.ModuleList()
        if  (Activation_Function == "Tanh"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Tanh())
        elif  (Activation_Function == "SiLU"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.SiLU())
        elif  (Activation_Function == "LeakyReLU"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.LeakyReLU())
        elif(Activation_Function == "Sin"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Sin())
        elif(Activation_Function == "Rational"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Rational(Data_Type = Data_Type, Device = Device))
        else:
            print("Unknown Activation Function. Got %s" % Activation_Function)
            print("Thrown by Neural_Network.__init__. Aborting.")
            exit();



    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ Forward method for the NN class. Note that the user should NOT call
        this function directly. Rather, they should call it through the __call__
        method (using the NN object like a function), which is part of the
        module class and calls forward.

        ------------------------------------------------------------------------
        Arguments:

        X: A batch of inputs. This should be a B by Input_Dim tensor, where B
        is the batch size. The ith row of X should hold the ith input.

        ------------------------------------------------------------------------
        Returns:

        If X is a B by Input_Dim tensor, then the output of this function is a
        B by Output_Dim tensor, whose ith row holds the value of the network
        applied to the ith row of X. """

        # If we are using batch normalization, then normalize the inputs.
        if(self.Batch_Norm == True):
            X = self.Norm_Layer(X);

        # Pass X through the hidden layers. Each has an activation function.
        for i in range(0, self.Num_Hidden_Layers):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer (with no activation function) and return.
        return self.Layers[self.Num_Hidden_Layers](X);

# Flatten a structured x-y-z-t solution and create train/validation samples.
def build_xyzt_split(data, n_train, n_valid, seed=42):
    x = data["x"]  # (Nx,)
    y = data["y"]  # (Ny,)
    z = data["z"]  # (Nz,)
    t = data["t"]  # (Nt,)
    U = data["U"]  # Shape: (Nt, Nx, Ny, Nz) or (Nx, Ny, Nz, Nt).

    # Ensure U has shape (Nx, Ny, Nz, Nt).
    if U.shape == (len(t), len(x), len(y), len(z)):
        U = np.transpose(U, (1, 2, 3, 0))  # (Nx, Ny, Nz, Nt)

    # Build the 4D coordinate grid.
    X_grid, Y_grid, Z_grid, T_grid = np.meshgrid(x, y, z, t, indexing="ij")  # (Nx,Ny,Nz,Nt)

    X = np.column_stack([
        X_grid.ravel(),
        Y_grid.ravel(),
        Z_grid.ravel(),
        T_grid.ravel()
    ])  # (N, 4)
    Y_out = U.ravel().reshape(-1, 1)  # (N, 1)

    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    if n_train + n_valid > N:
        raise ValueError("n_train + n_valid exceeds total samples")

    train_idx = idx[:n_train]
    valid_idx = idx[n_train:n_train + n_valid]

    X_train, Y_train = X[train_idx], Y_out[train_idx]
    X_valid, Y_valid = X[valid_idx], Y_out[valid_idx]

    return X, Y_out, X_train, Y_train, X_valid, Y_valid



BASE_GENES_2D = ("u", "ux", "uxx", "uxxx", "uy", "uyy", "uyyy")

# Compute 3D spatial and temporal derivatives with PyTorch autograd.
def compute_derivatives_3d(u, X, LHS="ut"):
    # X: (N,4) -> [x, y, z, t]
    grad_u = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_z = grad_u[:, 2:3]
    u_t = grad_u[:, 3:4]

    if LHS == "utt":
        u_tt = torch.autograd.grad(
            u_t, X, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0][:, 3:4]

    u_xx = torch.autograd.grad(
        u_x, X, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, X, grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    u_zz = torch.autograd.grad(
        u_z, X, grad_outputs=torch.ones_like(u_z),
        create_graph=True, retain_graph=True
    )[0][:, 2:3]

    u_xxx = torch.autograd.grad(
        u_xx, X, grad_outputs=torch.ones_like(u_xx),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    u_yyy = torch.autograd.grad(
        u_yy, X, grad_outputs=torch.ones_like(u_yy),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]

    u_zzz = torch.autograd.grad(
        u_zz, X, grad_outputs=torch.ones_like(u_zz),
        create_graph=True, retain_graph=True
    )[0][:, 2:3]

    if LHS == "ut":
        return u_t, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, u_z, u_zz, u_zzz
    else:
        return u_tt, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, u_z, u_zz, u_zzz


# Build an order-independent key for one multiplicative term.
def _canonical_module_key(module):
    # Multiplication is commutative here: ignore order but keep repeated factors.
    # For example, ["u", "ux"] and ["ux", "u"] share the same key.
    return tuple(sorted(module))

BASE_GENES_3D = ("u", "ux", "uxx", "uxxx", "uy", "uyy", "uyyy", "uz", "uzz", "uzzz")

# Randomly sample one 3D candidate PDE term under derivative constraints.
def sample_gene_module_3d(
    rng,
    base_genes=BASE_GENES_3D,
    min_len=1,
    max_len=3,
    allow_repeat=True,
    p=[1, 1, 0.5],
    p_drop_u=0.5,  # Probability of dropping u when a conflict occurs.
):
    if p is None:
        p = np.ones(max_len, dtype=float)
    p = np.asarray(p, dtype=float)
    p = p[:max_len]
    if p.sum() <= 0:
        raise ValueError("p must have positive sum")
    p = p / p.sum()

    lengths = np.arange(1, max_len + 1)
    k = int(rng.choice(lengths, p=p))
    k = max(k, min_len)

    # Choose the derivative direction: x, y, or z.
    axis = rng.choice(["x", "y", "z"])
    if axis == "x":
        axis_genes = ["ux", "uxx", "uxxx"]
        max_counts = {"ux": 2, "uxx": 1, "uxxx": 1}
        incompatible = ("uxxx", {"ux", "uxx"})
    elif axis == "y":
        axis_genes = ["uy", "uyy", "uyyy"]
        max_counts = {"uy": 2, "uyy": 1, "uyyy": 1}
        incompatible = ("uyyy", {"uy", "uyy"})
    else:
        axis_genes = ["uz", "uzz", "uzzz"]
        max_counts = {"uz": 2, "uzz": 1, "uzzz": 1}
        incompatible = ("uzzz", {"uz", "uzz"})

    gene_pool = ["u"] + axis_genes
    term = []
    counts = {g: 0 for g in gene_pool}

    # Generate under structural constraints before resolving u/derivative conflicts.
    while len(term) < k:
        g = rng.choice(gene_pool)

        # Do not combine third-order and lower-order derivatives in the same direction.
        if g == incompatible[0] and any(counts[h] > 0 for h in incompatible[1]):
            continue
        if g in incompatible[1] and counts[incompatible[0]] > 0:
            continue

        # Enforce maximum-count constraints.
        if g in max_counts and counts[g] >= max_counts[g]:
            continue

        term.append(g)
        counts[g] += 1

        # Guard against rare infinite sampling loops.
        if len(term) < k and all(
            (g not in max_counts or counts[g] >= max_counts[g]) for g in gene_pool
        ):
            break

    # Post-process conflicts between u and higher-order derivatives.
    if "u" in term:
        # Find which derivative orders are present for the selected direction.
        deriv_present = [g for g in axis_genes if g in term]

        if len(deriv_present) > 1:
            # Decide whether to drop u or derivative terms.
            if rng.random() < p_drop_u:
                # Drop u.
                term = [g for g in term if g != "u"]
            else:
                # Keep only one derivative order.
                keep = rng.choice(deriv_present)
                term = [g for g in term if (g == "u" or g == keep)]

    return term



# Randomly sample a full genome made of candidate PDE terms.
def sample_genome(rng, n_modules=None, min_modules=2, max_modules=5, **module_kwargs):
    if n_modules is None:
        n_modules = int(rng.integers(min_modules, max_modules + 1))

    raw = [sample_gene_module_3d(rng, **module_kwargs) for _ in range(n_modules)]

    seen = set()
    genome = []
    for mod in raw:
        key = _canonical_module_key(mod)
        if key not in seen:
            genome.append(mod)
            seen.add(key)
    return genome


# Convert a 3D genome into a callable PDE feature library.
def genome_to_library_3d(genome):
    term_names = ["*".join(g) for g in genome]

    def library_fn(
        u, u_x, u_xx, u_y, u_yy, u_z, u_zz,
        u_xxx=None, u_yyy=None, u_zzz=None
    ):
        gene_map = {
            "u": u,
            "ux": u_x,
            "uxx": u_xx,
            "uy": u_y,
            "uyy": u_yy,
            "uz": u_z,
            "uzz": u_zz,
        }
        if u_xxx is not None:
            gene_map["uxxx"] = u_xxx
        if u_yyy is not None:
            gene_map["uyyy"] = u_yyy
        if u_zzz is not None:
            gene_map["uzzz"] = u_zzz

        terms = []
        for module in genome:
            term = torch.ones_like(u)
            for g in module:
                term = term * gene_map[g]
            terms.append(term)

        if len(terms) == 0:
            Phi = torch.zeros((u.shape[0], 0), device=u.device, dtype=u.dtype)
        else:
            Phi = torch.cat(terms, dim=1)
        return Phi, term_names

    return library_fn, term_names



# Train a neural network on one case using data loss only.
def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    name: str = "baseline",
    num_epochs: int = 50000,
    log_interval: int = 1000,
    device: str = "cuda",
    log_dir: str = "model_save",
):
    """
    Train a model on provided train/valid splits.

    - X_train: (N, n_features) numpy
    - Y_train: (N,) or (N,1) numpy
    - X_valid: (M, n_features) numpy
    - Y_valid: (M,) or (M,1) numpy
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    def to_tensor(x_np, y_np):
        x_t = torch.from_numpy(x_np).float().to(device)
        y_np = y_np.reshape(-1, 1)
        y_t = torch.from_numpy(y_np).float().to(device)
        return x_t, y_t

    X_train_t, Y_train_t = to_tensor(X_train, Y_train)
    X_valid_t, Y_valid_t = to_tensor(X_valid, Y_valid)

    model = model.to(device)
    criterion = nn.MSELoss()

    save_dir = os.path.join(log_dir, f"model_{name}")
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# epoch\ttrain_loss\tvalid_loss\n")

    best_valid_loss = float("inf")
    best_epoch = -1

    print(f'\n\n Starting training {name}! \t\t')
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred_train = model(X_train_t).reshape(-1, 1)
        loss_train = criterion(pred_train, Y_train_t)
        loss_train.backward()
        optimizer.step()

        if (epoch % log_interval == 0) or (epoch == 1) or (epoch == num_epochs):
            model.eval()
            with torch.no_grad():
                pred_valid = model(X_valid_t).reshape(-1, 1)
                loss_valid = criterion(pred_valid, Y_valid_t)

            train_loss_val = loss_train.item()
            valid_loss_val = loss_valid.item()

            print(
                f"Epoch {epoch:6d} "
                f"train_loss = {train_loss_val:.6e}, "
                f"valid_loss = {valid_loss_val:.6e}"
            )

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch}\t{train_loss_val:.8e}\t"
                    f"{valid_loss_val:.8e}\n"
                )

            last_ckpt_path = os.path.join(save_dir, "checkpoint_last.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss_val,
                    "valid_loss": valid_loss_val,
                },
                last_ckpt_path,
            )

            if valid_loss_val < best_valid_loss:
                best_valid_loss = valid_loss_val
                best_epoch = epoch
                best_ckpt_path = os.path.join(save_dir, "checkpoint_best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss_val,
                        "valid_loss": valid_loss_val,
                    },
                    best_ckpt_path,
                )

    print(f"Training finished. Best valid loss = {best_valid_loss:.6e} at epoch {best_epoch}")
    print(f"Best model path: {os.path.join(save_dir, 'checkpoint_best.pt')}")

# Pretrain one neural model per case before PDE discovery.
def pretrain_cases(
    models,
    dataset,
    optimizers,
    n_train,
    num_epochs,
    device="cuda",
    log_interval=500,
    log_dir="model_save",
    seed=42,
):
    """
    Train each case in dataset once. Skip if checkpoint already exists.
    models/optimizers must align with dataset order.
    """
    if not isinstance(models, (list, tuple)):
        raise ValueError("models must be a list/tuple aligned with dataset order.")
    if not isinstance(optimizers, (list, tuple)):
        raise ValueError("optimizers must be a list/tuple aligned with dataset order.")
    if len(models) != len(dataset) or len(optimizers) != len(dataset):
        raise ValueError("models/optimizers length must match number of dataset cases.")

    for idx, (key, data) in enumerate(dataset.items()):
        name = f"{key}_{n_train}"
        save_dir = os.path.join(log_dir, f"model_{name}")
        ckpt_path = os.path.join(save_dir, "checkpoint_best.pt")
        if os.path.exists(ckpt_path):
            print(f"[skip] {name} already trained: {ckpt_path}")
            continue

        X_train, Y_train, X_valid, Y_valid =data['X_train'], data['Y_train'], data['X_valid'], data['Y_valid']
        model = models[idx]
        optimizer = optimizers[idx]

        train_model(
            model,
            optimizer,
            X_train,
            Y_train,
            X_valid,
            Y_valid,
            name=name,
            num_epochs=num_epochs,
            device=device,
            log_interval=log_interval,
            log_dir=log_dir,
        )


# Train one 3D PINN for a selected PDE library.
def train_pinn_single_case_3d(
    model,
    X_np, U_np,
    LHS,
    seed,
    num_epochs,
    alpha_pde,
    log_dir,
    name,
    device="cuda",
    log_interval=500,
    n_colloc=1000,
    x_range=None,
    y_range=None,
    z_range=None,
    t_range=None,
):
    """
    3D PDE loss on random collocation points sampled from (x,y,z,t) domain.
    Data loss still uses given (X_np, U_np).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # -------- load best model -------- #
    save_dir = os.path.join(log_dir, f"model_{name}")
    best_ckpt_path = os.path.join(save_dir, "checkpoint_best.pt")
    checkpoint = torch.load(best_ckpt_path, map_location=device)

    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X = torch.from_numpy(X_np).float().to(device)
    U = torch.from_numpy(U_np).float().to(device)

    if x_range is None:
        x_range = (float(X_np[:, 0].min()), float(X_np[:, 0].max()))
    if y_range is None:
        y_range = (float(X_np[:, 1].min()), float(X_np[:, 1].max()))
    if z_range is None:
        z_range = (float(X_np[:, 2].min()), float(X_np[:, 2].max()))
    if t_range is None:
        t_range = (float(X_np[:, 3].min()), float(X_np[:, 3].max()))

    rng = np.random.default_rng(seed)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        # ----- data loss -----
        X_data = X.detach().requires_grad_(True)
        u = model(X_data).reshape(-1, 1)
        loss_data = mse(u, U)

        # ----- PDE loss on collocation points -----
        x_c = rng.uniform(x_range[0], x_range[1], size=(n_colloc, 1))
        y_c = rng.uniform(y_range[0], y_range[1], size=(n_colloc, 1))
        z_c = rng.uniform(z_range[0], z_range[1], size=(n_colloc, 1))
        t_c = rng.uniform(t_range[0], t_range[1], size=(n_colloc, 1))
        X_c_np = np.hstack([x_c, y_c, z_c, t_c])

        X_c = torch.from_numpy(X_c_np).float().to(device)
        X_c.requires_grad_(True)

        u_c = model(X_c).reshape(-1, 1)

        Left_u_t, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, u_z, u_zz, u_zzz = compute_derivatives_3d(
            u_c, X_c, LHS=LHS
        )

        # 3D feature library.
        if "Heat" in name:
            Phi = torch.cat([u_xx, u_yy, u_zz], dim=1)

        beta = torch.linalg.lstsq(Phi, Left_u_t).solution
        r = Left_u_t - (Phi @ beta)

        loss_pde = mse(r, torch.zeros_like(r))
        loss = loss_data + alpha_pde * loss_pde
        loss.backward()
        optimizer.step()

        if epoch % log_interval == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"Epoch {epoch:6d} "
                f"loss={loss.item():.6e} "
                f"data={loss_data.item():.6e} "
                f"pde={loss_pde.item():.6e} "
                f"beta={beta.reshape(-1).cpu().data.numpy()} "
            )

    return model




# Train multicase 3D models while estimating shared PDE coefficients.
def train_multicase_competitive_3d(
    models,
    cases,  # dict: {case_key: {"X_train":..., "Y_train":..., "X_valid":..., "Y_valid":...}}
    prefix: str,
    LHS: str,
    num_epochs: int,
    n_train: int,
    n_valid: int,
    seed: int,
    n_colloc=1000,
    support_frac=0.7,
    kappa_weight=1e-2,
    alpha_start=0.0,
    alpha_end=1.0,
    tau_start=0.1,
    tau_end=5.0,
    rho=0.1,
    device="cuda",
    eps=1e-8,
    library_fn=None,
    return_metrics=False,
):
    mse = nn.MSELoss()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    params = []
    for m in models:
        m.to(device)
        params += list(m.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    case_keys = list(cases.keys())
    for idx, key in enumerate(case_keys):
        name = f"{key}_{n_train}"
        ckpt_path = os.path.join(f"model_save/{prefix}_{n_train}", f"model_{name}", "checkpoint_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            models[idx].load_state_dict(ckpt["model_state_dict"])
        else:
            raise FileNotFoundError(f"missing pretrained: {ckpt_path}")

    # cache train data and domains
    data_cache = []
    for key in case_keys:
        data = cases[key]
        X_train, Y_train = data["X_train"], data["Y_train"]
        X_t = torch.from_numpy(X_train).float().to(device)
        U_t = torch.from_numpy(Y_train).float().to(device)

        x_min, x_max = float(X_train[:, 0].min()), float(X_train[:, 0].max())
        y_min, y_max = float(X_train[:, 1].min()), float(X_train[:, 1].max())
        z_min, z_max = float(X_train[:, 2].min()), float(X_train[:, 2].max())
        t_min, t_max = float(X_train[:, 3].min()), float(X_train[:, 3].max())

        data_cache.append((key, X_t, U_t, (x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max)))

    def _ridge(Phi, y):
        sol = torch.linalg.lstsq(Phi, y)
        return sol.solution

    beta_ema = None
    term_names = None

    for epoch in tqdm(range(1, num_epochs + 1)):
        t_sched = min(1.0, epoch / max(1, num_epochs // 3))
        alpha = alpha_start + (alpha_end - alpha_start) * t_sched
        tau = tau_start + (tau_end - tau_start) * t_sched

        optimizer.zero_grad()
        loss_data_total = 0.0
        loss_pde_total = 0.0

        betas = []
        scores = []
        colloc_cache = []

        for model, (case_id, X_data, U_data, dom) in zip(models, data_cache):
            model.train()

            # data loss
            u_data = model(X_data).reshape(-1, 1)
            loss_data_total = loss_data_total + mse(u_data, U_data)

            # collocation points (x,y,z,t)
            x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max = dom
            x_c = rng.uniform(x_min, x_max, size=(n_colloc, 1))
            y_c = rng.uniform(y_min, y_max, size=(n_colloc, 1))
            z_c = rng.uniform(z_min, z_max, size=(n_colloc, 1))
            t_c = rng.uniform(t_min, t_max, size=(n_colloc, 1))
            Xc_np = np.hstack([x_c, y_c, z_c, t_c])

            Xc = torch.from_numpy(Xc_np).float().to(device)
            Xc.requires_grad_(True)

            u_c = model(Xc).reshape(-1, 1)

            # 3D derivatives
            Left_u_t, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, u_z, u_zz, u_zzz = compute_derivatives_3d(
                u_c, Xc, LHS=LHS
            )

            lib_out = library_fn(u_c, u_x, u_xx, u_y, u_yy, u_z, u_zz, u_xxx, u_yyy, u_zzz)
            if isinstance(lib_out, tuple):
                Phi, lib_names = lib_out
                if term_names is None:
                    term_names = lib_names
            else:
                Phi = lib_out

            sigma = torch.sqrt((Phi ** 2).mean(dim=0, keepdim=True)) + eps
            Phi_s = Phi / sigma

            n_sup = max(2, int(n_colloc * support_frac))
            perm = torch.randperm(n_colloc, device=device)
            sup_idx = perm[:n_sup]
            qry_idx = perm[n_sup:]

            Phi_sup = Phi_s[sup_idx]
            y_sup = Left_u_t[sup_idx]
            Phi_q = Phi_s[qry_idx]
            y_q = Left_u_t[qry_idx]

            beta_s = _ridge(Phi_sup, y_sup)
            beta = (beta_s / sigma.T)

            y_q_pred = Phi_q @ beta_s
            r_q = y_q - y_q_pred
            r_norm = (r_q.pow(2).mean()) / (y_q.pow(2).mean() + eps)

            sv = torch.linalg.svdvals(Phi_sup)
            kappa = sv.max() / (sv.min() + eps)

            score = r_norm + kappa_weight * torch.log(kappa + eps)

            betas.append(beta.detach())
            scores.append(score.detach())
            colloc_cache.append((Left_u_t, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, u_z, u_zz, u_zzz, u_c))

        scores_t = torch.stack(scores)
        weights = torch.softmax(-tau * scores_t, dim=0)

        beta_stack = torch.stack(betas, dim=0)
        beta_global = (weights[:, None, None] * beta_stack).sum(dim=0)

        if beta_ema is None:
            beta_ema = beta_global
        else:
            beta_ema = (1.0 - rho) * beta_ema + rho * beta_global

        beta_const = beta_ema.detach()

        for (Left_u_t, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy, u_z, u_zz, u_zzz, u_c) in colloc_cache:
            lib_out = library_fn(u_c, u_x, u_xx, u_y, u_yy, u_z, u_zz, u_xxx, u_yyy, u_zzz)
            Phi_raw = lib_out[0] if isinstance(lib_out, tuple) else lib_out
            r = Left_u_t - (Phi_raw @ beta_const)
            loss_pde_total = loss_pde_total + mse(r, torch.zeros_like(r))

        loss = loss_data_total + alpha * loss_pde_total
        loss.backward()
        optimizer.step()

        if epoch == num_epochs:
            beta_mean_np = beta_ema.detach().cpu().numpy().reshape(-1)
            beta_str = ", ".join([f"{v:.4e}" for v in beta_mean_np])
            print(
                f"data={loss_data_total.item():.6e} "
                f"pde={loss_pde_total.item():.6e} "
                f"fitness={loss_data_total.item()*loss_data_total.item():.6e} "
                f"beta=[{beta_str}]"
            )
            weights_np = weights.detach().cpu().numpy()
            beta_stack_np = beta_stack.detach().cpu().numpy()  # (S, K, 1)
            beta_std = beta_stack_np[:, :, 0].std(axis=0)  # (K,)
            for i, key in enumerate(case_keys):
                b = beta_stack[i].cpu().numpy().reshape(-1)
                b_str = ", ".join([f'{v:.4e}' for v in b])
                print(f"  {key}: w={weights_np[i]:.3f} beta=[{b_str}]")
            if term_names is not None:
                for name, val in zip(term_names, beta_mean_np):
                    print(f"  {name}: {val:.4e}")
            print("")



            lines = []

            for i, key in enumerate(case_keys):
                b = beta_stack[i].cpu().numpy().reshape(-1)
                b_str = ", ".join([f"{v:.4e}" for v in b])
                lines.append(f"{key}: w={weights_np[i]:.3f} beta=[{b_str}]")

            if term_names is not None:
                for name, val, std in zip(term_names, beta_mean_np, beta_std):
                    lines.append(f"{name}: {val:.4e}\u00b1{std:.4e}")

            lhs_str = LHS  # Example values: "ut" or "utt".
            terms_eq = []
            for name, coef in zip(term_names, beta_mean_np):
                terms_eq.append(f"{coef:.4e}*{name}")

            eq_str = lhs_str + " = " + " + ".join(terms_eq)
            lines.append(eq_str)

    if return_metrics:
        return {
            "loss_data": loss_data_total.item(),
            "loss_pde": loss_pde_total.item(),
            "loss_total": loss_data_total.item() + alpha * loss_pde_total.item(),
            "beta_ema": beta_ema.detach().cpu().numpy(),
            "term_names": term_names,
        }
    else:
        print({
            "loss_data": loss_data_total.item(),
            "loss_pde": loss_pde_total.item(),
            "loss_total": loss_data_total.item() + alpha * loss_pde_total.item(),
            "beta_ema": beta_ema.detach().cpu().numpy(),
            "term_names": term_names
        })
        return models,lines






# Evaluate one genome by training the corresponding PDE library.
def fitness(
    genome,
    models,
    cases,
    prefix,
    LHS,
    seed,
    num_epochs,
    n_train,
    n_valid,
    n_colloc=1000,
    device="cuda",

):
    library_fn, term_names = genome_to_library_3d(genome)
    result = train_multicase_competitive_3d(
        models=models,
        cases=cases,
        prefix=prefix,
        LHS=LHS,
        num_epochs=num_epochs,
        n_train=n_train,
        n_valid=n_valid,
        n_colloc=n_colloc,
        device=device,
        seed=seed,
        library_fn=library_fn,
        return_metrics=True,
    )
    total_loss = result["loss_data"]* result["loss_pde"]
    return total_loss, result

# Apply random term-level mutations to a genome.
def mutate_genome(rng, genome, p_mut=0.3, min_len=1, max_len=3):
    g = copy.deepcopy(genome)
    if rng.random() > p_mut:
        return g
    choice = rng.integers(0, 3)
    if choice == 0 and len(g) > 1:
        idx = int(rng.integers(0, len(g)))
        g.pop(idx)
    elif choice == 1:
        g.append(sample_gene_module_3d(rng, min_len=min_len, max_len=max_len))
    else:
        idx = int(rng.integers(0, len(g)))
        g[idx] = sample_gene_module_3d(rng, min_len=min_len, max_len=max_len)
    return g

# Combine terms from two parent genomes.
def crossover_genome(rng, g1, g2):
    if len(g1) == 0 or len(g2) == 0:
        return copy.deepcopy(g1 if len(g1) > 0 else g2)
    c1 = int(rng.integers(0, len(g1)))
    c2 = int(rng.integers(0, len(g2)))
    child = g1[:c1] + g2[c2:]
    return child if len(child) > 0 else copy.deepcopy(g1)

# Remove equivalent duplicate terms from a genome.
def dedupe_genome(genome):
    seen = set()
    out = []
    for mod in genome:
        key = tuple(sorted(mod))
        if key not in seen:
            out.append(mod)
            seen.add(key)
    return out

# Clean and constrain a genome after sampling, mutation, or crossover.
def sanitize_genome(genome, max_terms=5, seed=42):
    """
    Enforce genome constraints:
      1) If a term contains 'uxxx' and ('ux' or 'uxx'), randomly choose:
         - remove 'uxxx' with 50% prob
         - remove all 'ux' and 'uxx' with 50% prob
      2) Keep at most two 'ux' per term.
      3) Keep at most two 'uxx' per term.
      4) If genome has more than max_terms, randomly drop terms to max_terms.
    """
    rng = np.random.default_rng(seed)

    cleaned = []
    for term in genome:
        has_uxxx = "uxxx" in term
        has_ux = "ux" in term
        has_uxx = "uxx" in term

        remove_uxxx = False
        remove_ux_uxx = False
        if has_uxxx and (has_ux or has_uxx):
            if rng.random() < 0.5:
                remove_uxxx = True
            else:
                remove_ux_uxx = True

        ux_count = 0
        uxx_count = 0
        uxxx_seen = False
        new_term = []

        for g in term:
            if g == "uxxx":
                if remove_uxxx:
                    continue
                if not uxxx_seen:
                    new_term.append(g)
                    uxxx_seen = True
            elif g == "ux":
                if remove_ux_uxx:
                    continue
                if ux_count < 2:
                    new_term.append(g)
                    ux_count += 1
            elif g == "uxx":
                if remove_ux_uxx:
                    continue
                if uxx_count < 2:
                    new_term.append(g)
                    uxx_count += 1
            else:
                new_term.append(g)

        cleaned.append(new_term)

    cleaned = dedupe_genome(cleaned)

    if len(cleaned) > max_terms:
        keep_idx = rng.choice(len(cleaned), size=max_terms, replace=False)
        cleaned = [cleaned[i] for i in keep_idx]

    return cleaned


# Convert NumPy and tensor-like values into JSON-serializable objects.
def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.float16)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.int16)):
        return int(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


# Build a stable key for caching and comparing genomes.
def genome_key(genome):
    # Sort factors within each module and then sort modules for stable equivalence checks.
    mods = [tuple(sorted(m)) for m in genome]
    mods_sorted = tuple(sorted(mods))
    return mods_sorted

# Run the genetic search loop for 2D or 3D PDE discovery.
def genetic_search_2D(
    models,
    cases,
    prefix: str,
    n_train: int,
    n_valid: int,
    LHS:str,
    pop_size:int,
    PINN_num_epoch: int,
    generations: int,
    seed=42,
):
    rng = np.random.default_rng(seed)
    population = [sample_genome(rng) for _ in range(pop_size)]

    population=[sanitize_genome(p) for p in population]

    best = None
    best_score = np.inf
    best_info = None
    fitness_cache = {}
    save_dir = os.path.join("result_save", f"{prefix}_{n_train}_{len(cases)}")
    config_path = os.path.join(save_dir, "ga_config.json")


    exp_cfg = {
        "pop_size": pop_size,
        "generations": generations,
        "n_train": n_train,
        "n_valid": n_valid,
        "seed": seed,
        "PINN_num_epoch": PINN_num_epoch,
        "num of cases": len(cases),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exp_cfg, f, indent=2)



    for gen in range(generations):
        print(f"#################   Generation {gen}/{generations}   #####################")
        scored = []
        for idx,genome in enumerate(population):
            print(f'=================Evaluate {genome}, {idx}/{len(population)}===============')
            key = genome_key(genome)
            if key in fitness_cache:
                score, info = fitness_cache[key]
                print(f'fitness have been computed: {score}')
            else:
                score, info = fitness(genome, models, cases, prefix,LHS=LHS, seed=seed, num_epochs=PINN_num_epoch, n_train=n_train, n_valid=n_valid)
                #score, info = fitness_quick(genome, models, cases, prefix, n_train=n_train, n_valid=n_valid,seed=seed,penalty=1e-3)
                fitness_cache[key] = (score, info)

            scored.append((score, genome, info))
            if score < best_score:
                best_score, best, best_info = score, copy.deepcopy(genome), info

        scored.sort(key=lambda x: x[0])


        elite_k=int(pop_size/10)
        if elite_k==0:
            elite_k=1

        elites = [g for _, g, _ in scored[:elite_k]]

        # Keep elite genomes unchanged.
        new_pop = elites[:]

        # Generate candidates for the remaining population slots.
        gen_needed = pop_size - len(new_pop)
        generated = [sample_genome(rng) for _ in range(gen_needed)]

        # Crossover and mutate each generated candidate.
        for i in range(len(generated)):
            p1 = elites[int(rng.integers(0, len(elites)))]
            p2 = generated[int(rng.integers(0, len(generated)))]
            child = crossover_genome(rng, p1, p2)
            child = mutate_genome(rng, child)
            child = dedupe_genome(child)
            new_pop.append(child)

        population = new_pop
        population = [sanitize_genome(p) for p in population]

        print(f"[Gen {gen+1}] best_loss={best_score:.6e} best_genome={best}")

        if gen==0:
            if os.path.exists(os.path.join(save_dir, "ga_best_per_gen.json")):
                os.remove(os.path.join(save_dir, "ga_best_per_gen.json"))

        result_path = os.path.join(save_dir, "ga_best_per_gen.json")
        record = {
            "gen": gen + 1,
            "best_loss": float(best_score),
            "best_genome": best,
            "best_info":_jsonable(best_info)
        }
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(record)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


        cache_path = os.path.join(save_dir, "fitness_cache.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(fitness_cache, f)

    return best, best_score, best_info




# ---------------- Example usage ----------------
# best_genome, best_score, best_info = genetic_search(
#     models=models,
#     cases=cases,
#     pop_size=6,
#     generations=5,
#     elite_k=2,
#     seed=0,
# )
# print("Best genome:", best_genome)
# print("Best loss:", best_score)

from pathlib import Path
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "matnet": ROOT / "data" / "ATSP_data" / "Matnet_atsp250_128instances",
    "udc": ROOT / "data" / "ATSP_data" / "UDC_atsp250_128instances",
    "ctrl": ROOT / "data" / "ATSP_data" / "Ctrl_ATSP_data",
    "rrnco": ROOT
    / "data"
    / "ATSP_data"
    / "RRNCO_atsp250_64instances"
    / "atsp_n250_64_data",
}


def choose_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys())
    args = parser.parse_args()

    if args.dataset:
        return args.dataset, DATASETS[args.dataset]

    dataset_names = list(DATASETS.keys())
    print("Available datasets:")
    for idx, name in enumerate(dataset_names, start=1):
        print(f"{idx}: {name} ({DATASETS[name]})")

    while True:
        choice = input("Choose dataset: ").strip().lower()

        if choice in DATASETS:
            return choice, DATASETS[choice]

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(dataset_names):
                name = dataset_names[idx - 1]
                return name, DATASETS[name]

        print("Invalid dataset selection.")


def load_matrix_file(path: Path):
    if path.suffix == ".pt":
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, torch.Tensor):
            return loaded
        if isinstance(loaded, dict):
            for key in ("distance", "dist", "matrix", "cost_matrix"):
                if key in loaded:
                    return torch.as_tensor(loaded[key])
        raise TypeError(f"{path.name}: expected a tensor or a dictionary with a matrix.")

    if path.suffix == ".npz":
        with np.load(path) as loaded:
            key = "distance" if "distance" in loaded.files else loaded.files[0]
            return torch.from_numpy(loaded[key])

    raise ValueError(f"{path.name}: unsupported file type {path.suffix}")


def asymmetry_decomposition(D: torch.Tensor, eps: float = 1e-12):

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {tuple(D.shape)}")

    D = D.to(dtype=torch.float64)
    n = D.shape[0]

    K = 0.5 * (D - D.T)

    p = K.sum(dim=1) / n
    p = p - p.mean()

    G = p[:, None] - p[None, :]
    R = K - G

    k_energy = torch.sum(K * K).item()
    g_energy = torch.sum(G * G).item()
    r_energy = torch.sum(R * R).item()
    d_energy = torch.sum(D * D).item()

    if k_energy <= eps:
        rho = math.nan
        potential_share = math.nan
    else:
        rho = r_energy / k_energy
        potential_share = g_energy / k_energy

        if -1e-12 < rho < 0:
            rho = 0.0
        elif 1 < rho < 1 + 1e-12:
            rho = 1.0

        if -1e-12 < potential_share < 0:
            potential_share = 0.0
        elif 1 < potential_share < 1 + 1e-12:
            potential_share = 1.0

    if d_energy <= eps:
        rel_asymmetry = math.nan
        rel_circulation = math.nan
    else:
        rel_asymmetry = math.sqrt(k_energy / d_energy)
        rel_circulation = math.sqrt(r_energy / d_energy)

    decomposition_error = abs(k_energy - (g_energy + r_energy))
    tolerance = 1e-9 * max(1.0, k_energy)
    if decomposition_error > tolerance:
        print(
            "Warning: ||K||_F^2 != ||G||_F^2 + ||R||_F^2 within tolerance: "
            f"error={decomposition_error:.3e}"
        )

    return rho, potential_share, rel_asymmetry, rel_circulation, n


def iter_square_matrices(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")

    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"Expected [..., n, n], got shape {tuple(tensor.shape)}")

    n = tensor.shape[-1]
    if tensor.ndim == 2:
        yield tensor
    else:
        for matrix in tensor.reshape(-1, n, n):
            yield matrix


dataset_name, DATA_DIR = choose_dataset()

if not DATA_DIR.exists():
    print(f"Directory {DATA_DIR} does not exist.")
    raise SystemExit(1)

data_files = sorted(
    file
    for pattern in ("*.pt", "*.npz")
    for file in DATA_DIR.glob(pattern)
)
if not data_files:
    print(f"No .pt or .npz files found in {DATA_DIR}.")
    raise SystemExit(1)

print(f"Analyzing dataset: {dataset_name}")
print(f"Files: {len(data_files)}")

all_values = []
rhos = []
potential_shares = []
rel_asymmetries = []
rel_circulations = []
instance_sizes = []
num_symmetric = 0

for data_file in data_files:
    loaded = load_matrix_file(data_file)

    all_values.append(loaded.float().flatten())

    for D in iter_square_matrices(loaded):
        rho, potential_share, rel_asymmetry, rel_circulation, n = (
            asymmetry_decomposition(D)
        )

        if math.isnan(rho):
            num_symmetric += 1
        else:
            rhos.append(rho)
            potential_shares.append(potential_share)

        if not math.isnan(rel_asymmetry):
            rel_asymmetries.append(rel_asymmetry)
        if not math.isnan(rel_circulation):
            rel_circulations.append(rel_circulation)

        instance_sizes.append(n)

all_values = torch.cat(all_values)

plt.figure(figsize=(10, 6))
plt.hist(all_values.numpy(), bins=1000, density=True)
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Distribution of all values across all ATSP instances")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

if rel_asymmetries:
    plt.figure(figsize=(10, 6))
    plt.hist(rel_asymmetries, bins=50, density=True)
    plt.xlim(0.0, 1.0)
    plt.xlabel(r"$A(D) = \|K\|_F / \|D\|_F$")
    plt.ylabel("Density")
    plt.title("Distribution of relative asymmetry across ATSP instances")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No relative-asymmetry values to plot.")

num_instances = len(instance_sizes)
print(f"Number of matrices: {num_instances}")
print(f"Symmetric matrices (rho undefined): {num_symmetric}")

if rhos:
    rho_t = torch.tensor(rhos, dtype=torch.float64)
    print(f"rho mean:   {rho_t.mean().item():.6f}")
    print(f"rho median: {rho_t.median().item():.6f}")
    print(f"rho min:    {rho_t.min().item():.6f}")
    print(f"rho max:    {rho_t.max().item():.6f}")

if rel_asymmetries:
    a_t = torch.tensor(rel_asymmetries, dtype=torch.float64)
    print(f"relative asymmetry ||K||_F/||D||_F mean: {a_t.mean().item():.6f}")

if rel_circulations:
    c_t = torch.tensor(rel_circulations, dtype=torch.float64)
    print(f"relative circulation ||R||_F/||D||_F mean: {c_t.mean().item():.6f}")

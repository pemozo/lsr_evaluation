from pathlib import Path
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "matnet": ROOT / "data" / "ATSP_data" / "Matnet_atsp250_128instances",
    "udc": ROOT / "data" / "ATSP_data" / "UDC_atsp250_128instances",
    "ctrl": ROOT / "data" / "ATSP_data" / "Ctrl_ATSP_data1",
    "rrnco": (
        ROOT
        / "data"
        / "ATSP_data"
        / "RRNCO_atsp250_64instances"
        / "atsp_n250_64_data"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys())
    return parser.parse_args()


def choose_dataset(dataset_name=None):
    if dataset_name:
        return dataset_name, DATASETS[dataset_name]

    names = list(DATASETS)
    print("Available datasets:")
    for index, name in enumerate(names, start=1):
        print(f"{index}: {name} ({DATASETS[name]})")

    while True:
        choice = input("Choose dataset: ").strip().lower()
        if choice in DATASETS:
            return choice, DATASETS[choice]
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
            return name, DATASETS[name]
        print("Invalid dataset selection.")


def find_data_files(data_directory):
    if not data_directory.exists():
        raise FileNotFoundError(f"Directory {data_directory} does not exist.")

    files = sorted(
        path
        for pattern in ("*.pt", "*.npz")
        for path in data_directory.glob(pattern)
    )
    if not files:
        raise FileNotFoundError(f"No .pt or .npz files found in {data_directory}.")
    return files


def load_matrix_file(path):
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


def iter_square_matrices(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim < 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError(f"Expected [..., n, n], got shape {tuple(tensor.shape)}")

    size = tensor.shape[-1]
    if tensor.ndim == 2:
        yield tensor
    else:
        yield from tensor.reshape(-1, size, size)


def clamp_unit_roundoff(value):
    if -1e-12 < value < 0.0:
        return 0.0
    if 1.0 < value < 1.0 + 1e-12:
        return 1.0
    return value


def asymmetry_decomposition(matrix, eps=1e-12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {tuple(matrix.shape)}")

    matrix = matrix.to(dtype=torch.float64)
    size = matrix.shape[0]
    skew = 0.5 * (matrix - matrix.T)
    potential = skew.sum(dim=1) / size
    potential = potential - potential.mean()
    gradient = potential[:, None] - potential[None, :]
    residual = skew - gradient

    skew_energy = torch.sum(skew * skew).item()
    gradient_energy = torch.sum(gradient * gradient).item()
    residual_energy = torch.sum(residual * residual).item()
    matrix_energy = torch.sum(matrix * matrix).item()

    if skew_energy <= eps:
        rho = math.nan
        potential_share = math.nan
    else:
        rho = clamp_unit_roundoff(residual_energy / skew_energy)
        potential_share = clamp_unit_roundoff(gradient_energy / skew_energy)

    if matrix_energy <= eps:
        relative_asymmetry = math.nan
        relative_circulation = math.nan
    else:
        relative_asymmetry = math.sqrt(skew_energy / matrix_energy)
        relative_circulation = math.sqrt(residual_energy / matrix_energy)

    error = abs(skew_energy - gradient_energy - residual_energy)
    tolerance = 1e-9 * max(1.0, skew_energy)
    if error > tolerance:
        print(
            "Warning: ||K||_F^2 != ||G||_F^2 + ||R||_F^2 within tolerance: "
            f"error={error:.3e}"
        )

    return rho, potential_share, relative_asymmetry, relative_circulation


def compute_pair_asymmetry_metrics(matrix, eps=1e-12):
    if matrix.shape[0] < 2:
        return math.nan, math.nan

    matrix = matrix.to(dtype=torch.float64)
    size = matrix.shape[0]
    upper = torch.triu_indices(size, size, offset=1, device=matrix.device)
    forward = matrix[upper[0], upper[1]]
    reverse = matrix[upper[1], upper[0]]
    pair_asymmetry_share = torch.mean(
        (torch.abs(forward - reverse) > eps).to(torch.float64)
    ).item()

    centered_forward = forward - forward.mean()
    centered_reverse = reverse - reverse.mean()
    denominator = torch.linalg.vector_norm(
        centered_forward
    ) * torch.linalg.vector_norm(centered_reverse)
    if denominator.item() <= eps:
        reverse_correlation = math.nan
    else:
        reverse_correlation = (
            torch.dot(centered_forward, centered_reverse) / denominator
        ).item()

    return pair_asymmetry_share, reverse_correlation


def compute_triangle_inequality_metrics(matrix, eps=1e-12):
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square 2D matrix, got shape {tuple(matrix.shape)}")

    size = matrix.shape[0]
    total_triples = size * (size - 1) * (size - 2)
    if total_triples == 0:
        return 0, math.nan, math.nan, math.nan

    matrix = matrix.to(dtype=torch.float64)
    different_endpoints = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    violation_count = 0
    relative_strength_sum = 0.0
    relative_strength_count = 0
    relative_strength_max = math.nan
    zero_rhs_violations = 0

    for via in range(size):
        relevant = different_endpoints.clone()
        relevant[via, :] = False
        relevant[:, via] = False

        direct = matrix
        indirect = matrix[:, via, None] + matrix[via, None, :]
        violations = relevant & (direct > indirect + eps)
        current_count = int(violations.sum().item())
        if current_count == 0:
            continue

        violation_count += current_count
        direct_values = direct[violations]
        indirect_values = indirect[violations]
        finite_denominator = torch.abs(indirect_values) > eps
        zero_rhs_violations += current_count - int(finite_denominator.sum().item())

        if torch.any(finite_denominator):
            strengths = (
                direct_values[finite_denominator]
                - indirect_values[finite_denominator]
            ) / indirect_values[finite_denominator]
            relative_strength_sum += strengths.sum().item()
            relative_strength_count += int(strengths.numel())
            current_max = strengths.max().item()
            if math.isnan(relative_strength_max) or current_max > relative_strength_max:
                relative_strength_max = current_max

    violation_share = violation_count / total_triples
    if violation_count == 0:
        return violation_count, violation_share, math.nan, math.nan
    if zero_rhs_violations:
        return violation_count, violation_share, math.inf, math.inf

    relative_strength_mean = relative_strength_sum / relative_strength_count
    return violation_count, violation_share, relative_strength_mean, relative_strength_max


def analyze_files(data_files):
    metrics = {
        "rho": [],
        "potential_share": [],
        "relative_asymmetry": [],
        "relative_circulation": [],
        "pair_asymmetry_share": [],
        "reverse_correlation": [],
        "triangle_violation_count": [],
        "triangle_violation_share": [],
        "triangle_relative_violation_strength_mean": [],
        "triangle_relative_violation_strength_max": [],
    }
    all_values = []
    instance_count = 0
    symmetric_count = 0

    for data_file in data_files:
        loaded = load_matrix_file(data_file)
        for matrix in iter_square_matrices(loaded):
            matrix = matrix.to(dtype=torch.float64)
            all_values.append(matrix.flatten())
            instance_count += 1

            rho, potential_share, relative_asymmetry, relative_circulation = (
                asymmetry_decomposition(matrix)
            )
            if math.isnan(rho):
                symmetric_count += 1
            else:
                metrics["rho"].append(rho)
                metrics["potential_share"].append(potential_share)

            pair_asymmetry_share, reverse_correlation = (
                compute_pair_asymmetry_metrics(matrix)
            )
            (
                triangle_violation_count,
                triangle_violation_share,
                triangle_strength_mean,
                triangle_strength_max,
            ) = compute_triangle_inequality_metrics(matrix)

            values = {
                "relative_asymmetry": relative_asymmetry,
                "relative_circulation": relative_circulation,
                "pair_asymmetry_share": (
                    pair_asymmetry_share
                    if not math.isnan(relative_asymmetry)
                    else math.nan
                ),
                "reverse_correlation": reverse_correlation,
                "triangle_violation_count": triangle_violation_count,
                "triangle_violation_share": triangle_violation_share,
                "triangle_relative_violation_strength_mean": triangle_strength_mean,
                "triangle_relative_violation_strength_max": triangle_strength_max,
            }
            for name, value in values.items():
                if not math.isnan(value):
                    metrics[name].append(value)

    return torch.cat(all_values), metrics, instance_count, symmetric_count


def plot_distribution(values, xlabel, title, bins=50, xlim=None):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"No values to plot: {title}")
        return

    histogram_range = xlim
    if histogram_range is None:
        minimum = values.min()
        maximum = values.max()
        scale = max(abs(minimum), abs(maximum), 1.0)
        if maximum - minimum <= np.finfo(np.float64).eps * scale * bins:
            center = 0.5 * (minimum + maximum)
            padding = max(0.05 * abs(center), 0.5)
            histogram_range = (center - padding, center + padding)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, range=histogram_range, density=True)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_asymmetry_scatter(pair_shares, relative_asymmetries):
    if not pair_shares:
        print("No pairwise-asymmetry scatter values to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(pair_shares, relative_asymmetries, alpha=0.75)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel(r"$P_{asym} = N_{asym} / (n(n-1)/2)$")
    plt.ylabel(r"$A(D) = \|K\|_F / \|D\|_F$")
    plt.title("Pairwise asymmetry share vs. global Frobenius asymmetry")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def print_metric_summary(label, values, statistics=("mean",)):
    if not values:
        return

    tensor = torch.tensor(values, dtype=torch.float64)
    functions = {
        "mean": torch.mean,
        "median": torch.median,
        "min": torch.min,
        "max": torch.max,
    }
    for statistic in statistics:
        value = functions[statistic](tensor).item()
        print(f"{label} {statistic}: {value:.6f}")


def create_plots(all_values, metrics):
    plot_distribution(
        all_values.numpy(),
        "Value",
        "Distribution of all values across all ATSP instances",
        bins=1000,
    )
    plot_distribution(
        metrics["relative_asymmetry"],
        r"$A(D) = \|K\|_F / \|D\|_F$",
        "Distribution of relative asymmetry across ATSP instances",
        xlim=(0.0, 1.0),
    )
    plot_asymmetry_scatter(
        metrics["pair_asymmetry_share"], metrics["relative_asymmetry"]
    )
    plot_distribution(
        metrics["reverse_correlation"],
        r"$\rho_{reverse} = corr(c_{ij}, c_{ji})$",
        "Distribution of reverse cost correlation across ATSP instances",
        xlim=(-1.0, 1.0),
    )
    plot_distribution(
        metrics["triangle_violation_share"],
        r"$N_{viol} / (n(n-1)(n-2))$",
        "Distribution of triangle inequality violation rates",
        xlim=(0.0, 1.0),
    )
    plt.show()


def print_summary(metrics, instance_count, symmetric_count):
    print(f"Number of matrices: {instance_count}")
    print(f"Symmetric matrices (rho undefined): {symmetric_count}")
    print_metric_summary(
        "rho", metrics["rho"], statistics=("mean", "median", "min", "max")
    )
    print_metric_summary("potential share ||G||_F^2/||K||_F^2", metrics["potential_share"])
    print_metric_summary(
        "relative asymmetry ||K||_F/||D||_F", metrics["relative_asymmetry"]
    )
    print_metric_summary(
        "pairwise asymmetry share", metrics["pair_asymmetry_share"]
    )
    print_metric_summary("reverse correlation", metrics["reverse_correlation"])
    print_metric_summary(
        "relative circulation ||R||_F/||D||_F", metrics["relative_circulation"]
    )
    print_metric_summary(
        "triangle inequality violation count",
        metrics["triangle_violation_count"],
        statistics=("mean", "median", "min", "max"),
    )
    print_metric_summary(
        "triangle inequality violation share",
        metrics["triangle_violation_share"],
        statistics=("mean", "median", "min", "max"),
    )
    print_metric_summary(
        "relative triangle violation strength mean",
        metrics["triangle_relative_violation_strength_mean"],
        statistics=("mean", "median", "min", "max"),
    )
    print_metric_summary(
        "relative triangle violation strength max",
        metrics["triangle_relative_violation_strength_max"],
        statistics=("mean", "median", "min", "max"),
    )


def main():
    args = parse_args()
    dataset_name, data_directory = choose_dataset(args.dataset)

    try:
        data_files = find_data_files(data_directory)
    except FileNotFoundError as error:
        raise SystemExit(error) from error

    print(f"Analyzing dataset: {dataset_name}")
    print(f"Files: {len(data_files)}")
    all_values, metrics, instance_count, symmetric_count = analyze_files(data_files)
    print_summary(metrics, instance_count, symmetric_count)
    create_plots(all_values, metrics)


if __name__ == "__main__":
    main()

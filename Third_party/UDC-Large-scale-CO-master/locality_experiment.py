#!/usr/bin/env python3
"""Measure KNN locality inside segments of reproducible LKH-3 tours."""

import argparse
import ast
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import torch


TARGET_N = 250
DEFAULT_K_VALUES = (1, 2, 4, 8, 16)
DATASET_PATHS = {
    "ctrl_atsp_data": Path("data/ATSP_data/Ctrl_ATSP_data"),
    "matnet": Path("data/ATSP_data/Matnet_atsp250_128instances"),
    "rrnco": Path("data/ATSP_data/RRNCO_atsp250_64instances/atsp_n250_64_data"),
}
DATASET_TITLES = {
    "ctrl_atsp_data": "CTRL_ATSP_DATA",
    "matnet": "MATNET",
    "rrnco": "RRNCO",
}
MATRIX_KEYS = ("distance", "distance_matrix", "cost_matrix", "matrix", "data")


DETAIL_FIELDS = [
    "dataset",
    "instance_id",
    "N",
    "segment_size",
    "K",
    "symmetry_type",
    "lkh_seed",
    "lkh_objective",
    "lkh_scaled_objective",
    "offset",
    "locality_outgoing",
    "locality_incoming",
    "node_std_outgoing",
    "node_std_incoming",
    "random_baseline",
    "locality_lift_outgoing",
    "locality_lift_incoming",
]

INSTANCE_FIELDS = [
    "dataset",
    "instance_id",
    "N",
    "segment_size",
    "K",
    "symmetry_type",
    "lkh_seed",
    "lkh_objective",
    "lkh_scaled_objective",
    "lkh_runtime_seconds",
    "offset_count",
    "locality_outgoing",
    "locality_incoming",
    "node_std_outgoing",
    "node_std_incoming",
    "random_baseline",
    "locality_lift_outgoing",
    "locality_lift_incoming",
]

NODE_FIELDS = [
    "dataset",
    "instance_id",
    "N",
    "segment_size",
    "K",
    "node_id",
    "selected_source_index",
    "original_node_id",
    "symmetry_type",
    "lkh_seed",
    "locality_outgoing",
    "locality_incoming",
    "random_baseline",
    "locality_lift_outgoing",
    "locality_lift_incoming",
]

MANIFEST_FIELDS = [
    "dataset",
    "instance_id",
    "instance_index",
    "source_file",
    "source_dimension",
    "N",
    "node_selection",
    "node_selection_seed",
    "selected_source_indices",
    "selected_original_node_ids",
    "np_allclose_D_DT",
    "symmetry_type",
    "lkh_seed",
    "lkh_problem_type",
    "lkh_objective",
    "lkh_scaled_objective",
    "lkh_runtime_seconds",
]

SUMMARY_FIELDS = [
    "dataset",
    "K",
    "direction",
    "number_instances",
    "mean_locality",
    "std_over_instances",
    "std_over_nodes",
    "mean_within_instance_node_std",
    "median",
    "q25",
    "q75",
    "ci95_low",
    "ci95_high",
    "random_baseline",
    "mean_locality_lift",
]

COMPARISON_FIELDS = [
    "reference_dataset",
    "comparison_dataset",
    "K",
    "direction",
    "comparison_mode",
    "number_reference",
    "number_comparison",
    "reference_locality",
    "comparison_locality",
    "difference",
    "absolute_difference",
    "difference_ci95_low",
    "difference_ci95_high",
    "reference_lift",
    "comparison_lift",
    "difference_in_lift",
    "hedges_g",
]


@dataclass
class PreparedInstance:
    dataset: str
    instance_id: str
    instance_index: int
    source_file: Path
    source_dimension: int
    matrix: np.ndarray
    selected_source_indices: np.ndarray
    selected_original_node_ids: np.ndarray
    node_selection_seed: Optional[int]
    symmetry_type: str
    np_allclose_d_dt: bool


def parse_args(argv=None):
    script_dir = Path(__file__).resolve().parent
    default_project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Solve common 250-node matrices with LKH-3 and measure KNN locality "
            "inside offset-averaged tour segments."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_PATHS),
        default=list(DATASET_PATHS),
    )
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument("--ctrl-path", type=Path)
    parser.add_argument("--matnet-path", type=Path)
    parser.add_argument("--rrnco-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instances", type=int, default=64)
    parser.add_argument(
        "--segment-size",
        type=int,
        help="Override the subproblem size discovered from the UDC configuration.",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--incoming",
        choices=("auto", "always", "never"),
        default="auto",
        help="'auto' computes incoming locality for asymmetric matrices.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--max-trials",
        type=int,
        help="LKH MAX_TRIALS. Defaults to the repository convention 2*N.",
    )
    parser.add_argument(
        "--lkh-binary",
        default=os.environ.get("LKH_BINARY", "LKH"),
    )
    parser.add_argument(
        "--paired-comparisons",
        action="store_true",
        help=(
            "Use instance_id as a pairing key. Only enable this when matching IDs "
            "are known to describe genuinely paired instances."
        ),
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep temporary TSPLIB, parameter, and tour files after successful runs.",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.instances <= 64:
        parser.error("--instances must be between 1 and 64")
    if args.seed < 1:
        parser.error("--seed must be positive because LKH expects a positive seed")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.scale < 1 or args.runs < 1:
        parser.error("--scale and --runs must be at least 1")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be at least 1")
    args.k_values = list(dict.fromkeys(args.k_values))
    if not args.k_values or any(k < 1 or k >= TARGET_N for k in args.k_values):
        parser.error(f"every --k-values entry must be between 1 and {TARGET_N - 1}")
    return args


def discover_segment_size(project_root):
    config_dir = (
        project_root
        / "Third_party"
        / "UDC-Large-scale-CO-master"
        / "UDC"
        / "ATSP-AGNN-MatNet"
    )
    discoveries = []
    for path in sorted(config_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "env_params" for target in node.targets):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value == "sub_size":
                    try:
                        discoveries.append((path, int(ast.literal_eval(value))))
                    except (TypeError, ValueError, SyntaxError):
                        pass

    values = sorted({value for _, value in discoveries})
    if len(values) != 1:
        found = ", ".join(f"{path.name}={value}" for path, value in discoveries) or "none"
        raise ValueError(
            "The repository segment/subproblem size is not unambiguous "
            f"({found}). Supply --segment-size explicitly."
        )
    return values[0], discoveries


def resolve_segment_size(args):
    try:
        discovered, sources = discover_segment_size(args.project_root)
    except ValueError:
        if args.segment_size is None:
            raise
        discovered, sources = None, []
    segment_size = args.segment_size if args.segment_size is not None else discovered
    source = "--segment-size" if args.segment_size is not None else "repository env_params['sub_size']"
    if not 2 <= segment_size <= TARGET_N:
        raise ValueError(f"segment size must be between 2 and {TARGET_N}, got {segment_size}")
    if TARGET_N % segment_size != 0:
        raise ValueError(
            f"N={TARGET_N} is not divisible by segment size {segment_size} ({source}). "
            f"Unequal remainder segments are disabled; supply a divisor of {TARGET_N} "
            "via --segment-size."
        )
    return segment_size, discovered, sources, source


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def numpy_value(value, source):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy(), {}
    if isinstance(value, np.ndarray):
        return value, {}
    if isinstance(value, dict):
        for key in MATRIX_KEYS:
            if key in value:
                matrix = value[key]
                if torch.is_tensor(matrix):
                    matrix = matrix.detach().cpu().numpy()
                metadata = {item_key: item_value for item_key, item_value in value.items() if item_key != key}
                return np.asarray(matrix), metadata
    raise TypeError(f"Could not find a matrix tensor/array in {source}; got {type(value).__name__}")


def load_file(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            matrix_key = next((key for key in MATRIX_KEYS if key in archive.files), None)
            if matrix_key is None:
                raise KeyError(f"No supported matrix key {MATRIX_KEYS} in {path}")
            matrix = np.asarray(archive[matrix_key])
            metadata = {key: np.asarray(archive[key]) for key in archive.files if key != matrix_key}
            return matrix, metadata
    if path.suffix.lower() == ".pt":
        value = torch.load(str(path), map_location="cpu")
        return numpy_value(value, path)
    raise ValueError(f"Unsupported dataset file: {path}")


def scalar_metadata(metadata, key):
    if key not in metadata:
        return None
    value = metadata[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.item() if array.size == 1 else None


def canonical_instance_id(path, metadata, local_index, matrix_count):
    metadata_id = scalar_metadata(metadata, "instance_id")
    if metadata_id is not None:
        return str(metadata_id)
    if matrix_count > 1:
        return str(local_index)
    match = re.search(r"(\d+)$", path.stem)
    return str(int(match.group(1))) if match else path.stem


def iter_source_matrices(path, limit):
    path = path.resolve()
    if path.is_dir():
        files = sorted(
            (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".pt", ".npz"}),
            key=natural_key,
        )
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    emitted = 0
    for source in files:
        value, metadata = load_file(source)
        if value.ndim == 2:
            matrices = value[None, ...]
        elif value.ndim == 3:
            matrices = value
        else:
            raise ValueError(f"Expected a 2D or 3D matrix array in {source}, got {value.shape}")
        for local_index, matrix in enumerate(matrices):
            instance_id = canonical_instance_id(source, metadata, local_index, len(matrices))
            yield emitted, instance_id, source, np.asarray(matrix), metadata
            emitted += 1
            if emitted == limit:
                return
    raise ValueError(f"{path} provides only {emitted} instances; {limit} requested")


def validate_source_matrix(matrix, source):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix in {source}, got {matrix.shape}")
    if matrix.shape[0] < TARGET_N:
        raise ValueError(f"{source} has only {matrix.shape[0]} nodes; N={TARGET_N} is required")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Matrix in {source} contains NaN or infinite values")
    if (matrix < 0).any():
        raise ValueError(f"Matrix in {source} contains negative edge weights")
    return matrix


def use_full_matrix(matrix, metadata, source):
    source_n = matrix.shape[0]
    if source_n != TARGET_N:
        raise ValueError(
            f"{source} has shape {matrix.shape}, not ({TARGET_N}, {TARGET_N}). "
            "This experiment requires all 250 nodes and does not perform node sampling."
        )
    indices = np.arange(TARGET_N, dtype=np.int64)
    target_matrix = matrix.copy()
    original_ids = indices.copy()
    if "node_indices" in metadata:
        candidate = np.asarray(metadata["node_indices"]).reshape(-1)
        if candidate.size == source_n:
            original_ids = candidate[indices].astype(np.int64, copy=False)
    return target_matrix, indices, original_ids


def resolve_dataset_paths(args):
    overrides = {
        "ctrl_atsp_data": args.ctrl_path,
        "matnet": args.matnet_path,
        "rrnco": args.rrnco_path,
    }
    return {
        name: (overrides[name] if overrides[name] is not None else args.project_root / DATASET_PATHS[name]).resolve()
        for name in args.datasets
    }


def prepare_instances(args, dataset_paths):
    instances = []
    for dataset, path in dataset_paths.items():
        dataset_instances = []
        for index, instance_id, source, raw_matrix, metadata in iter_source_matrices(path, args.instances):
            raw_matrix = validate_source_matrix(raw_matrix, source)
            matrix, selected, original_ids = use_full_matrix(raw_matrix, metadata, source)
            allclose = bool(np.allclose(matrix, matrix.T))
            symmetry_type = "symmetric" if allclose else "asymmetric"
            dataset_instances.append(
                PreparedInstance(
                    dataset=dataset,
                    instance_id=instance_id,
                    instance_index=index,
                    source_file=source,
                    source_dimension=raw_matrix.shape[0],
                    matrix=matrix,
                    selected_source_indices=selected,
                    selected_original_node_ids=original_ids,
                    node_selection_seed=None,
                    symmetry_type=symmetry_type,
                    np_allclose_d_dt=allclose,
                )
            )
        symmetric = sum(item.np_allclose_d_dt for item in dataset_instances)
        treated = "symmetric" if symmetric == len(dataset_instances) else "asymmetric" if symmetric == 0 else "mixed"
        print(
            f"[{dataset}] np.allclose(D, D.T): symmetric={symmetric}, "
            f"asymmetric={len(dataset_instances) - symmetric}; dataset treated as {treated}",
            flush=True,
        )
        instances.extend(dataset_instances)
    return instances


def resolve_lkh_binary(value):
    binary = shutil.which(value)
    if binary is None:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            binary = str(candidate)
    if binary is None:
        raise FileNotFoundError(f"LKH executable '{value}' was not found")
    return str(Path(binary).resolve()) if Path(binary).exists() else binary


def scaled_matrix(matrix, scale, symmetric):
    lkh_matrix = (matrix + matrix.T) / 2.0 if symmetric else matrix
    scaled = np.rint(lkh_matrix * scale)
    if scaled.max() > np.iinfo(np.int32).max:
        raise OverflowError("A scaled edge weight exceeds the signed 32-bit integer range")
    scaled = scaled.astype(np.int64)
    np.fill_diagonal(scaled, 0)
    return scaled


def write_problem(path, name, matrix, symmetric):
    problem_type = "TSP" if symmetric else "ATSP"
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write(f"TYPE: {problem_type}\n")
        handle.write(f"DIMENSION: {matrix.shape[0]}\n")
        handle.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        handle.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            handle.write(" ".join(str(int(value)) for value in row))
            handle.write("\n")
        handle.write("EOF\n")
    return problem_type


def write_parameter_file(path, problem_name, tour_name, runs, max_trials, seed):
    path.write_text(
        "\n".join(
            [
                f"PROBLEM_FILE = {problem_name}",
                f"OUTPUT_TOUR_FILE = {tour_name}",
                f"RUNS = {runs}",
                f"MAX_TRIALS = {max_trials}",
                f"SEED = {seed}",
                "TRACE_LEVEL = 1",
                "",
            ]
        ),
        encoding="ascii",
    )


def read_tour(path, dimension):
    nodes = []
    in_tour = False
    reported_objective = None
    for raw_line in path.read_text(encoding="ascii", errors="replace").splitlines():
        objective_match = re.search(r"Length\s*=\s*(-?\d+)", raw_line, flags=re.IGNORECASE)
        if objective_match:
            reported_objective = int(objective_match.group(1))
        line = raw_line.strip()
        if line == "TOUR_SECTION":
            in_tour = True
            continue
        if not in_tour:
            continue
        for token in line.split():
            if token in {"-1", "EOF"}:
                in_tour = False
                break
            nodes.append(int(token))

    expected = set(range(1, dimension + 1))
    if len(nodes) != dimension or set(nodes) != expected:
        raise ValueError(f"Invalid LKH tour: expected every node 1..{dimension} exactly once")
    return np.asarray(nodes, dtype=np.int64) - 1, reported_objective


def tour_cost(matrix, tour):
    return matrix[tour, np.roll(tour, -1)].sum(dtype=np.float64)


def solve_instance(instance, args, lkh_binary, output_dir):
    lkh_seed = args.seed + instance.instance_index
    max_trials = args.max_trials or 2 * TARGET_N
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance.instance_id)
    instance_name = f"{instance.dataset}_{instance.instance_index:04d}_{safe_id}"
    work_root = output_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f"{instance_name}_", dir=work_root))
    problem_path = work_dir / f"{instance_name}.tsp"
    parameter_path = work_dir / f"{instance_name}.par"
    tour_path = work_dir / f"{instance_name}.tour"
    log_dir = output_dir / "lkh_logs" / instance.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{instance_name}.log"

    is_symmetric = instance.symmetry_type == "symmetric"
    integer_matrix = scaled_matrix(instance.matrix, args.scale, is_symmetric)
    problem_type = write_problem(problem_path, instance_name, integer_matrix, is_symmetric)
    write_parameter_file(
        parameter_path,
        problem_path.name,
        tour_path.name,
        args.runs,
        max_trials,
        lkh_seed,
    )

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [lkh_binary, parameter_path.name],
            cwd=work_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        runtime = time.perf_counter() - started
        log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(f"LKH exited with code {completed.returncode}; see {log_path}")
        if not tour_path.is_file():
            raise FileNotFoundError(f"LKH did not create a tour; see {log_path}")
        tour, reported_objective = read_tour(tour_path, TARGET_N)
        scaled_objective = int(round(tour_cost(integer_matrix, tour)))
        if reported_objective is not None and reported_objective != scaled_objective:
            raise ValueError(
                f"LKH reported objective {reported_objective}, but the parsed tour has "
                f"scaled cost {scaled_objective}"
            )
        objective = float(tour_cost(instance.matrix, tour))
    except Exception:
        print(f"[{instance.dataset}] retained failed work directory: {work_dir}", flush=True)
        raise

    if not args.keep_work:
        shutil.rmtree(work_dir)
    return {
        "tour": tour,
        "lkh_seed": lkh_seed,
        "lkh_objective": objective,
        "lkh_scaled_objective": scaled_objective,
        "lkh_runtime_seconds": runtime,
        "lkh_problem_type": problem_type,
    }


def knn_order(matrix):
    values = matrix.copy()
    np.fill_diagonal(values, np.inf)
    order = np.argsort(values, axis=1, kind="stable")
    if np.any(order[:, 0] == np.arange(matrix.shape[0])):
        raise AssertionError("The diagonal entered the KNN result")
    return order


def node_locality(order, segment_labels, k):
    neighbors = order[:, :k]
    return (segment_labels[neighbors] == segment_labels[:, None]).mean(axis=1)


def analyze_instance(instance, args, segment_size, lkh_binary, output_dir):
    solved = solve_instance(instance, args, lkh_binary, output_dir)
    tour = solved["tour"]
    outgoing_order = knn_order(instance.matrix)
    calculate_incoming = args.incoming == "always" or (
        args.incoming == "auto" and instance.symmetry_type == "asymmetric"
    )
    incoming_order = knn_order(instance.matrix.T) if calculate_incoming else outgoing_order
    baseline = (segment_size - 1) / (TARGET_N - 1)
    node_sums_out = {k: np.zeros(TARGET_N, dtype=np.float64) for k in args.k_values}
    node_sums_in = {k: np.zeros(TARGET_N, dtype=np.float64) for k in args.k_values}
    detail_rows = []

    for offset in range(segment_size):
        shifted = np.roll(tour, -offset)
        segments = shifted.reshape(-1, segment_size)
        labels = np.empty(TARGET_N, dtype=np.int64)
        for segment_index, nodes in enumerate(segments):
            labels[nodes] = segment_index
        for k in args.k_values:
            out_values = node_locality(outgoing_order, labels, k)
            in_values = node_locality(incoming_order, labels, k)
            incoming_disabled = args.incoming == "never" and instance.symmetry_type == "asymmetric"
            node_sums_out[k] += out_values
            if not incoming_disabled:
                node_sums_in[k] += in_values
            out_mean = float(out_values.mean())
            in_mean = math.nan if incoming_disabled else float(in_values.mean())
            detail_rows.append(
                {
                    "dataset": instance.dataset,
                    "instance_id": instance.instance_id,
                    "N": TARGET_N,
                    "segment_size": segment_size,
                    "K": k,
                    "symmetry_type": instance.symmetry_type,
                    "lkh_seed": solved["lkh_seed"],
                    "lkh_objective": solved["lkh_objective"],
                    "lkh_scaled_objective": solved["lkh_scaled_objective"],
                    "offset": offset,
                    "locality_outgoing": out_mean,
                    "locality_incoming": in_mean,
                    "node_std_outgoing": float(out_values.std(ddof=0)),
                    "node_std_incoming": math.nan if incoming_disabled else float(in_values.std(ddof=0)),
                    "random_baseline": baseline,
                    "locality_lift_outgoing": out_mean / baseline,
                    "locality_lift_incoming": in_mean / baseline if np.isfinite(in_mean) else math.nan,
                }
            )

    instance_rows = []
    node_rows = []
    for k in args.k_values:
        out_nodes = node_sums_out[k] / segment_size
        in_nodes = node_sums_in[k] / segment_size
        out_mean = float(out_nodes.mean())
        in_mean = float(in_nodes.mean())
        if args.incoming == "never" and instance.symmetry_type == "asymmetric":
            in_mean = math.nan
            in_node_std = math.nan
        else:
            in_node_std = float(in_nodes.std(ddof=0))
        instance_rows.append(
            {
                "dataset": instance.dataset,
                "instance_id": instance.instance_id,
                "N": TARGET_N,
                "segment_size": segment_size,
                "K": k,
                "symmetry_type": instance.symmetry_type,
                "lkh_seed": solved["lkh_seed"],
                "lkh_objective": solved["lkh_objective"],
                "lkh_scaled_objective": solved["lkh_scaled_objective"],
                "lkh_runtime_seconds": solved["lkh_runtime_seconds"],
                "offset_count": segment_size,
                "locality_outgoing": out_mean,
                "locality_incoming": in_mean,
                "node_std_outgoing": float(out_nodes.std(ddof=0)),
                "node_std_incoming": in_node_std,
                "random_baseline": baseline,
                "locality_lift_outgoing": out_mean / baseline,
                "locality_lift_incoming": in_mean / baseline if np.isfinite(in_mean) else math.nan,
            }
        )
        for node_id in range(TARGET_N):
            node_incoming = (
                math.nan
                if args.incoming == "never" and instance.symmetry_type == "asymmetric"
                else float(in_nodes[node_id])
            )
            node_rows.append(
                {
                    "dataset": instance.dataset,
                    "instance_id": instance.instance_id,
                    "N": TARGET_N,
                    "segment_size": segment_size,
                    "K": k,
                    "node_id": node_id,
                    "selected_source_index": int(instance.selected_source_indices[node_id]),
                    "original_node_id": int(instance.selected_original_node_ids[node_id]),
                    "symmetry_type": instance.symmetry_type,
                    "lkh_seed": solved["lkh_seed"],
                    "locality_outgoing": float(out_nodes[node_id]),
                    "locality_incoming": node_incoming,
                    "random_baseline": baseline,
                    "locality_lift_outgoing": float(out_nodes[node_id]) / baseline,
                    "locality_lift_incoming": (
                        node_incoming / baseline if np.isfinite(node_incoming) else math.nan
                    ),
                }
            )

    manifest_row = {
        "dataset": instance.dataset,
        "instance_id": instance.instance_id,
        "instance_index": instance.instance_index,
        "source_file": str(instance.source_file),
        "source_dimension": instance.source_dimension,
        "N": TARGET_N,
        "node_selection": "none (strict full matrix)",
        "node_selection_seed": instance.node_selection_seed,
        "selected_source_indices": json.dumps(instance.selected_source_indices.tolist()),
        "selected_original_node_ids": json.dumps(instance.selected_original_node_ids.tolist()),
        "np_allclose_D_DT": instance.np_allclose_d_dt,
        "symmetry_type": instance.symmetry_type,
        "lkh_seed": solved["lkh_seed"],
        "lkh_problem_type": solved["lkh_problem_type"],
        "lkh_objective": solved["lkh_objective"],
        "lkh_scaled_objective": solved["lkh_scaled_objective"],
        "lkh_runtime_seconds": solved["lkh_runtime_seconds"],
    }
    return detail_rows, instance_rows, node_rows, manifest_row


def stable_seed(base_seed, *parts):
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def bootstrap_mean_ci(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def aggregate(instance_rows, node_rows, args):
    rows = []
    for dataset in args.datasets:
        for k in args.k_values:
            selected = [row for row in instance_rows if row["dataset"] == dataset and row["K"] == k]
            for direction in ("outgoing", "incoming"):
                value_key = f"locality_{direction}"
                node_std_key = f"node_std_{direction}"
                values = np.asarray([row[value_key] for row in selected], dtype=np.float64)
                valid = np.isfinite(values)
                if not valid.any():
                    continue
                values = values[valid]
                node_stds = np.asarray([row[node_std_key] for row in selected], dtype=np.float64)[valid]
                all_node_values = np.asarray(
                    [
                        row[value_key]
                        for row in node_rows
                        if row["dataset"] == dataset
                        and row["K"] == k
                        and np.isfinite(row[value_key])
                    ],
                    dtype=np.float64,
                )
                baseline = float(selected[0]["random_baseline"])
                ci_low, ci_high = bootstrap_mean_ci(
                    values,
                    args.bootstrap_samples,
                    stable_seed(args.seed, "summary", dataset, k, direction),
                )
                mean = float(values.mean())
                rows.append(
                    {
                        "dataset": dataset,
                        "K": k,
                        "direction": direction,
                        "number_instances": len(values),
                        "mean_locality": mean,
                        "std_over_instances": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "std_over_nodes": float(all_node_values.std(ddof=0)),
                        "mean_within_instance_node_std": float(node_stds.mean()),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "random_baseline": baseline,
                        "mean_locality_lift": mean / baseline,
                    }
                )
    return rows


def hedges_g(reference, comparison):
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    if len(reference) < 2 or len(comparison) < 2:
        return math.nan
    pooled_n = len(reference) + len(comparison) - 2
    pooled_var = (
        (len(reference) - 1) * reference.var(ddof=1)
        + (len(comparison) - 1) * comparison.var(ddof=1)
    ) / pooled_n
    if pooled_var <= 0:
        return math.nan
    correction = 1.0 - 3.0 / (4.0 * (len(reference) + len(comparison)) - 9.0)
    return float(correction * (comparison.mean() - reference.mean()) / math.sqrt(pooled_var))


def independent_difference_ci(reference, comparison, samples, seed):
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ref_idx = rng.integers(0, len(reference), size=(samples, len(reference)))
    cmp_idx = rng.integers(0, len(comparison), size=(samples, len(comparison)))
    deltas = comparison[cmp_idx].mean(axis=1) - reference[ref_idx].mean(axis=1)
    return tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))


def paired_difference_ci(reference_map, comparison_map, samples, seed):
    keys = sorted(reference_map)
    differences = np.asarray([comparison_map[key] - reference_map[key] for key in keys])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def compare_datasets(instance_rows, args):
    rows = []
    reference_dataset = "ctrl_atsp_data"
    if reference_dataset not in args.datasets:
        return rows
    for comparison_dataset in ("matnet", "rrnco"):
        if comparison_dataset not in args.datasets:
            continue
        for k in args.k_values:
            for direction in ("outgoing", "incoming"):
                value_key = f"locality_{direction}"
                reference_rows = [
                    row for row in instance_rows
                    if row["dataset"] == reference_dataset and row["K"] == k and np.isfinite(row[value_key])
                ]
                comparison_rows = [
                    row for row in instance_rows
                    if row["dataset"] == comparison_dataset and row["K"] == k and np.isfinite(row[value_key])
                ]
                if not reference_rows or not comparison_rows:
                    continue
                reference = np.asarray([row[value_key] for row in reference_rows])
                comparison = np.asarray([row[value_key] for row in comparison_rows])
                mode = "independent"
                if args.paired_comparisons:
                    reference_map = {row["instance_id"]: row[value_key] for row in reference_rows}
                    comparison_map = {row["instance_id"]: row[value_key] for row in comparison_rows}
                    if set(reference_map) != set(comparison_map):
                        raise ValueError(
                            f"Cannot pair {comparison_dataset} with {reference_dataset}: instance_id sets differ"
                        )
                    mode = "paired_by_user_assertion"
                    ci_low, ci_high = paired_difference_ci(
                        reference_map,
                        comparison_map,
                        args.bootstrap_samples,
                        stable_seed(args.seed, "comparison", comparison_dataset, k, direction),
                    )
                else:
                    ci_low, ci_high = independent_difference_ci(
                        reference,
                        comparison,
                        args.bootstrap_samples,
                        stable_seed(args.seed, "comparison", comparison_dataset, k, direction),
                    )
                reference_mean = float(reference.mean())
                comparison_mean = float(comparison.mean())
                difference = comparison_mean - reference_mean
                baseline = float(reference_rows[0]["random_baseline"])
                reference_lift = reference_mean / baseline
                comparison_lift = comparison_mean / baseline
                rows.append(
                    {
                        "reference_dataset": reference_dataset,
                        "comparison_dataset": comparison_dataset,
                        "K": k,
                        "direction": direction,
                        "comparison_mode": mode,
                        "number_reference": len(reference),
                        "number_comparison": len(comparison),
                        "reference_locality": reference_mean,
                        "comparison_locality": comparison_mean,
                        "difference": difference,
                        "absolute_difference": abs(difference),
                        "difference_ci95_low": ci_low,
                        "difference_ci95_high": ci_high,
                        "reference_lift": reference_lift,
                        "comparison_lift": comparison_lift,
                        "difference_in_lift": comparison_lift - reference_lift,
                        "hedges_g": hedges_g(reference, comparison),
                    }
                )
    return rows


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_sort_key(row, dataset_order):
    instance_id = str(row["instance_id"])
    try:
        instance_key = (0, int(instance_id))
    except ValueError:
        instance_key = (1, instance_id)
    return (
        dataset_order.index(row["dataset"]),
        instance_key,
        row.get("K", 0),
        row.get("offset", 0),
    )


def format_float(value):
    return "nan" if value is None or not np.isfinite(value) else f"{value:.6f}"


def build_final_report(args, segment_size, discovered_segment, lkh_binary, manifests, summaries, comparisons, elapsed):
    baseline = (segment_size - 1) / (TARGET_N - 1)
    lines = [
        "=" * 60,
        "LOCALITY EXPERIMENT - FINAL RESULTS",
        "=" * 60,
        "",
        f"N                : {TARGET_N}",
        f"segment size     : {segment_size} (repository sub_size={discovered_segment})",
        f"K values         : {', '.join(str(k) for k in args.k_values)}",
        "node selection   : none (strict 250x250 full matrices)",
        f"LKH binary       : {lkh_binary}",
        f"LKH seed         : base {args.seed}; instance seed = base + instance_index",
        f"LKH RUNS         : {args.runs}",
        f"LKH MAX_TRIALS   : {args.max_trials or 2 * TARGET_N}",
        f"integer scale    : {args.scale}",
        f"bootstrap        : {args.bootstrap_samples} samples; seed base {args.seed}",
        f"workers          : {args.workers}",
        f"total runtime    : {elapsed:.2f} s",
        "number instances : " + ", ".join(
            f"{name}={sum(row['dataset'] == name for row in manifests)}" for name in args.datasets
        ),
        "",
        "RANDOM BASELINE",
        "-" * 15,
        f"(n-1)/(N-1) = ({segment_size}-1)/({TARGET_N}-1) = {baseline:.6f}",
        "",
    ]

    summary_lookup = {(row["dataset"], row["K"], row["direction"]): row for row in summaries}
    for dataset in args.datasets:
        title = DATASET_TITLES[dataset]
        dataset_manifests = [row for row in manifests if row["dataset"] == dataset]
        symmetry_types = sorted({row["symmetry_type"] for row in dataset_manifests})
        lines.extend([title, "-" * len(title)])
        lines.append(
            "symmetry: " + ", ".join(symmetry_types)
            + "; np.allclose(D,D.T) checked per instance"
        )
        is_all_symmetric = symmetry_types == ["symmetric"]
        for k in args.k_values:
            outgoing = summary_lookup[(dataset, k, "outgoing")]
            if is_all_symmetric:
                lines.append(
                    f"K={k:<2d} locality={outgoing['mean_locality']:.6f}  "
                    f"CI95=[{outgoing['ci95_low']:.6f}, {outgoing['ci95_high']:.6f}]  "
                    f"lift={outgoing['mean_locality_lift']:.6f}  "
                    f"sd_inst={outgoing['std_over_instances']:.6f}  "
                    f"sd_nodes={outgoing['std_over_nodes']:.6f}  "
                    f"mean_within_inst_sd_nodes={outgoing['mean_within_instance_node_std']:.6f}  "
                    f"median={outgoing['median']:.6f}  "
                    f"q25={outgoing['q25']:.6f}  q75={outgoing['q75']:.6f}"
                )
            else:
                incoming = summary_lookup.get((dataset, k, "incoming"))
                incoming_text = "incoming=disabled"
                if incoming is not None:
                    incoming_text = (
                        f"in={incoming['mean_locality']:.6f} "
                        f"CI95=[{incoming['ci95_low']:.6f}, {incoming['ci95_high']:.6f}] "
                        f"lift={incoming['mean_locality_lift']:.6f}"
                    )
                lines.append(
                    f"K={k:<2d} out={outgoing['mean_locality']:.6f} "
                    f"CI95=[{outgoing['ci95_low']:.6f}, {outgoing['ci95_high']:.6f}] "
                    f"lift={outgoing['mean_locality_lift']:.6f}  |  {incoming_text}"
                )
                lines.append(
                    f"     out sd_inst={outgoing['std_over_instances']:.6f} "
                    f"sd_nodes={outgoing['std_over_nodes']:.6f} "
                    f"mean_within_inst_sd_nodes={outgoing['mean_within_instance_node_std']:.6f} "
                    f"median={outgoing['median']:.6f} q25={outgoing['q25']:.6f} q75={outgoing['q75']:.6f}"
                )
                if incoming is not None:
                    lines.append(
                        f"     in  sd_inst={incoming['std_over_instances']:.6f} "
                        f"sd_nodes={incoming['std_over_nodes']:.6f} "
                        f"mean_within_inst_sd_nodes={incoming['mean_within_instance_node_std']:.6f} "
                        f"median={incoming['median']:.6f} q25={incoming['q25']:.6f} q75={incoming['q75']:.6f}"
                    )
        lines.append("")

    for comparison_dataset in ("matnet", "rrnco"):
        selected = [row for row in comparisons if row["comparison_dataset"] == comparison_dataset]
        if not selected:
            continue
        title = f"{DATASET_TITLES[comparison_dataset]} vs CTRL_ATSP_DATA"
        lines.extend([title, "-" * len(title)])
        lines.append(
            "comparison mode: " + selected[0]["comparison_mode"]
            + ("" if args.paired_comparisons else " (no provenance-backed pairing was assumed)")
        )
        for k in args.k_values:
            for direction in ("outgoing", "incoming"):
                row = next(
                    (item for item in selected if item["K"] == k and item["direction"] == direction),
                    None,
                )
                if row is None:
                    continue
                short_direction = "out" if direction == "outgoing" else "in "
                lines.append(
                    f"K={k:<2d} {short_direction} ctrl={row['reference_locality']:.6f} "
                    f"comparison={row['comparison_locality']:.6f} "
                    f"delta={row['difference']:+.6f} abs_delta={row['absolute_difference']:.6f} "
                    f"CI95=[{row['difference_ci95_low']:+.6f}, {row['difference_ci95_high']:+.6f}] "
                    f"lift_ctrl={row['reference_lift']:.6f} "
                    f"lift_comparison={row['comparison_lift']:.6f} "
                    f"delta_lift={row['difference_in_lift']:+.6f} "
                    f"Hedges_g={format_float(row['hedges_g'])}"
                )
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    segment_size, discovered_segment, segment_sources, segment_source = resolve_segment_size(args)
    dataset_paths = resolve_dataset_paths(args)
    lkh_binary = resolve_lkh_binary(args.lkh_binary)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(__file__).resolve().parent / "locality_results" / timestamp
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("LOCALITY EXPERIMENT CONFIGURATION", flush=True)
    print("=" * 60, flush=True)
    print(f"N={TARGET_N}", flush=True)
    print(
        f"segment_size={segment_size}; source={segment_source}; "
        f"repository_discovery={discovered_segment}",
        flush=True,
    )
    print(f"K_VALUES={args.k_values}", flush=True)
    print(f"datasets={dataset_paths}", flush=True)
    print(f"instances_per_dataset={args.instances}", flush=True)
    print("node_selection=none; strict 250x250 full matrices", flush=True)
    print(f"incoming={args.incoming}", flush=True)
    print(f"LKH binary={lkh_binary}", flush=True)
    print(
        f"LKH parameters: RUNS={args.runs}, MAX_TRIALS={args.max_trials or 2 * TARGET_N}, "
        f"SEED_BASE={args.seed}, SCALE={args.scale}, TRACE_LEVEL=1",
        flush=True,
    )
    print(
        f"bootstrap_samples={args.bootstrap_samples}, bootstrap_seed_base={args.seed}, "
        f"workers={args.workers}",
        flush=True,
    )
    print(f"output_dir={args.output_dir}", flush=True)
    print("=" * 60, flush=True)

    config = {
        "N": TARGET_N,
        "segment_size": segment_size,
        "segment_size_source": segment_source,
        "repository_segment_size": discovered_segment,
        "repository_segment_sources": [str(path) for path, _ in segment_sources],
        "k_values": args.k_values,
        "datasets": {key: str(value) for key, value in dataset_paths.items()},
        "instances_per_dataset": args.instances,
        "node_selection": "none (strict full matrix)",
        "incoming": args.incoming,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "workers": args.workers,
        "lkh_binary": lkh_binary,
        "lkh_runs": args.runs,
        "lkh_max_trials": args.max_trials or 2 * TARGET_N,
        "integer_scale": args.scale,
        "paired_comparisons": args.paired_comparisons,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    instances = prepare_instances(args, dataset_paths)
    detail_rows = []
    instance_rows = []
    node_rows = []
    manifests = []
    failures = []
    total = len(instances)
    completed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_instance,
                instance,
                args,
                segment_size,
                lkh_binary,
                args.output_dir,
            ): instance
            for instance in instances
        }
        for future in as_completed(futures):
            instance = futures[future]
            completed_count += 1
            try:
                details, scores, nodes, manifest = future.result()
                detail_rows.extend(details)
                instance_rows.extend(scores)
                node_rows.extend(nodes)
                manifests.append(manifest)
                print(
                    f"[{completed_count:03d}/{total:03d}] {instance.dataset} "
                    f"instance={instance.instance_id} type={instance.symmetry_type} "
                    f"cost={manifest['lkh_objective']:.8f} "
                    f"LKH_time={manifest['lkh_runtime_seconds']:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "dataset": instance.dataset,
                        "instance_id": instance.instance_id,
                        "source_file": str(instance.source_file),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[{completed_count:03d}/{total:03d}] FAILED {instance.dataset} "
                    f"instance={instance.instance_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    if failures:
        write_csv(
            args.output_dir / "failures.csv",
            ["dataset", "instance_id", "source_file", "error"],
            failures,
        )
        raise RuntimeError(
            f"{len(failures)} LKH/locality tasks failed. No aggregate result was produced; "
            f"see {args.output_dir / 'failures.csv'}"
        )

    manifests.sort(key=lambda row: result_sort_key(row, args.datasets))
    instance_rows.sort(key=lambda row: result_sort_key(row, args.datasets))
    node_rows.sort(key=lambda row: result_sort_key(row, args.datasets) + (row["node_id"],))
    detail_rows.sort(key=lambda row: result_sort_key(row, args.datasets))
    summaries = aggregate(instance_rows, node_rows, args)
    comparisons = compare_datasets(instance_rows, args)

    write_csv(args.output_dir / "manifest.csv", MANIFEST_FIELDS, manifests)
    write_csv(args.output_dir / "details.csv", DETAIL_FIELDS, detail_rows)
    write_csv(args.output_dir / "instance_scores.csv", INSTANCE_FIELDS, instance_rows)
    write_csv(args.output_dir / "node_scores.csv", NODE_FIELDS, node_rows)
    write_csv(args.output_dir / "summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(args.output_dir / "comparisons.csv", COMPARISON_FIELDS, comparisons)

    work_root = args.output_dir / "work"
    if work_root.is_dir() and not any(work_root.iterdir()):
        work_root.rmdir()
    elapsed = time.perf_counter() - started
    report = build_final_report(
        args,
        segment_size,
        discovered_segment,
        lkh_binary,
        manifests,
        summaries,
        comparisons,
        elapsed,
    )
    (args.output_dir / "final_report.txt").write_text(report + "\n", encoding="utf-8")
    print("", flush=True)
    print(report, flush=True)
    print(f"CSV and report files: {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None

#!/usr/bin/env python3
"""Evaluate the first N ATSP instances of the local datasets with LKH-3."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import numpy as np
import torch


DATASET_PATHS = {
    "udc": Path("data/ATSP_data/UDC_atsp250_128instances/ATSP_data250_n128.pt"),
    "matnet": Path("data/ATSP_data/Matnet_atsp250_128instances"),
    "ctrl": Path("data/ATSP_data/Ctrl_ATSP_data"),
    "rrnco": Path("data/ATSP_data/RRNCO_atsp250_64instances/atsp_n250_64_data"),
}

RESULT_FIELDS = [
    "dataset",
    "instance_index",
    "source_file",
    "dimension",
    "status",
    "tour_cost",
    "scaled_tour_cost",
    "runtime_seconds",
    "seed",
    "error",
]


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="Convert ATSP matrices to TSPLIB, solve them with LKH-3, and aggregate results."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASET_PATHS],
        default="all",
        help="Dataset to evaluate. 'all' evaluates all four datasets.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Override the path for a single selected dataset.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root,
        help="Experiment root containing data/ and Third_party/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "lkh3_results",
        help="Directory for generated instances, tours, logs, and summaries.",
    )
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--scale", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--max-trials",
        type=int,
        help="LKH trials per run. Defaults to 2 times the instance dimension.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--lkh-binary",
        default=os.environ.get("LKH_BINARY", "LKH"),
        help="Path or command name of the LKH-3 executable.",
    )
    args = parser.parse_args()

    if not 1 <= args.instances <= 64:
        parser.error("--instances must be between 1 and 64 for the common evaluation set")
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.max_trials is not None and args.max_trials < 1:
        parser.error("--max-trials must be at least 1")
    if args.data_path is not None and args.dataset == "all":
        parser.error("--data-path can only be used with one selected dataset")
    return args


def load_pt(path):
    value = torch.load(str(path), map_location="cpu")
    if not torch.is_tensor(value):
        raise TypeError(f"Expected a tensor in {path}, got {type(value).__name__}")
    return value.detach().cpu().numpy()


def load_npz(path):
    with np.load(path) as value:
        if "distance" not in value:
            raise KeyError(f"Missing 'distance' array in {path}")
        return np.asarray(value["distance"])


def load_datasets(path, limit):
    path = path.resolve()
    if path.is_dir():
        files = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".pt", ".npz"}
        )
        if len(files) < limit:
            raise ValueError(f"{path} contains only {len(files)} supported files; {limit} requested")
        for index, source in enumerate(files[:limit]):
            matrix = load_npz(source) if source.suffix.lower() == ".npz" else load_pt(source)
            yield index, source, validate_matrix(matrix, source)
        return

    if not path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    value = load_npz(path) if path.suffix.lower() == ".npz" else load_pt(path)
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D array in {path}, got shape {value.shape}")
    if value.shape[0] < limit:
        raise ValueError(f"{path} contains only {value.shape[0]} instances; {limit} requested")
    for index in range(limit):
        yield index, path, validate_matrix(value[index], path)


def validate_matrix(matrix, source):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix in {source}, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Matrix in {source} contains NaN or infinite values")
    if (matrix < 0).any():
        raise ValueError(f"Matrix in {source} contains negative edge weights")
    return matrix


def scale_matrix(matrix, scale):
    scaled = np.rint(matrix * scale).astype(np.int64)
    if scaled.max() > np.iinfo(np.int32).max:
        raise OverflowError("A scaled edge weight exceeds the signed 32-bit integer range")
    return scaled


def write_atsp(path, name, matrix):
    dimension = matrix.shape[0]
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"NAME: {name}\n")
        handle.write("TYPE: ATSP\n")
        handle.write(f"DIMENSION: {dimension}\n")
        handle.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        handle.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        handle.write("EDGE_WEIGHT_SECTION\n")
        for row in matrix:
            handle.write(" ".join(str(int(value)) for value in row))
            handle.write("\n")
        handle.write("EOF\n")


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
    in_tour = False
    nodes = []
    for raw_line in path.read_text(encoding="ascii", errors="replace").splitlines():
        line = raw_line.strip()
        if line == "TOUR_SECTION":
            in_tour = True
            continue
        if not in_tour:
            continue
        if line in {"-1", "EOF"}:
            break
        nodes.append(int(line))

    expected = set(range(1, dimension + 1))
    if len(nodes) != dimension or set(nodes) != expected:
        raise ValueError(f"Invalid LKH tour: expected each node 1..{dimension} exactly once")
    return np.asarray(nodes, dtype=np.int64) - 1


def tour_cost(matrix, tour):
    successors = np.roll(tour, -1)
    return matrix[tour, successors].sum(dtype=np.float64)


def write_results(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(name, data_path, args, lkh_binary):
    dataset_dir = args.output_dir.resolve() / name
    work_dir = dataset_dir / "work"
    log_dir = dataset_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = dataset_dir / "results.csv"
    rows = []

    print(f"[{name}] Loading first {args.instances} instances from {data_path}", flush=True)
    for index, source, matrix in load_datasets(data_path, args.instances):
        instance_name = f"{name}_{index:04d}"
        atsp_path = work_dir / f"{instance_name}.atsp"
        parameter_path = work_dir / f"{instance_name}.par"
        tour_path = work_dir / f"{instance_name}.tour"
        log_path = log_dir / f"{instance_name}.log"
        scaled = scale_matrix(matrix, args.scale)
        max_trials = args.max_trials or 2 * matrix.shape[0]
        instance_seed = args.seed + index
        write_atsp(atsp_path, instance_name, scaled)
        write_parameter_file(
            parameter_path,
            atsp_path.name,
            tour_path.name,
            args.runs,
            max_trials,
            instance_seed,
        )

        started = time.perf_counter()
        status = "ok"
        error = ""
        exact_cost = math.nan
        scaled_cost = math.nan
        try:
            completed = subprocess.run(
                [lkh_binary, parameter_path.name],
                cwd=work_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
            if completed.returncode != 0:
                raise RuntimeError(f"LKH exited with code {completed.returncode}; see {log_path}")
            if not tour_path.is_file():
                raise FileNotFoundError(f"LKH did not create {tour_path}")
            tour = read_tour(tour_path, matrix.shape[0])
            exact_cost = float(tour_cost(matrix, tour))
            scaled_cost = int(tour_cost(scaled, tour))
        except Exception as exc:
            status = "failed"
            error = str(exc)

        runtime = time.perf_counter() - started
        rows.append(
            {
                "dataset": name,
                "instance_index": index,
                "source_file": str(source),
                "dimension": matrix.shape[0],
                "status": status,
                "tour_cost": exact_cost,
                "scaled_tour_cost": scaled_cost,
                "runtime_seconds": runtime,
                "seed": instance_seed,
                "error": error,
            }
        )
        write_results(results_path, rows)
        print(
            f"[{name}] {index + 1:02d}/{args.instances}: {status}, "
            f"cost={exact_cost:.8f}, time={runtime:.2f}s",
            flush=True,
        )

    successful = [row for row in rows if row["status"] == "ok"]
    costs = [float(row["tour_cost"]) for row in successful]
    runtimes = [float(row["runtime_seconds"]) for row in successful]
    summary = {
        "dataset": name,
        "data_path": str(data_path.resolve()),
        "requested_instances": args.instances,
        "successful_instances": len(successful),
        "failed_instances": len(rows) - len(successful),
        "scale": args.scale,
        "runs": args.runs,
        "max_trials": args.max_trials or 2 * rows[0]["dimension"],
        "mean_tour_cost": statistics.fmean(costs) if costs else None,
        "population_std_tour_cost": statistics.pstdev(costs) if costs else None,
        "min_tour_cost": min(costs) if costs else None,
        "max_tour_cost": max(costs) if costs else None,
        "mean_runtime_seconds": statistics.fmean(runtimes) if runtimes else None,
        "total_runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
    }
    (dataset_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (dataset_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(f"[{name}] Summary: {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


def resolve_lkh_binary(value):
    binary = shutil.which(value)
    if binary is None:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            binary = str(candidate)
    if binary is None:
        raise FileNotFoundError(
            f"LKH executable '{value}' was not found. Rebuild the Apptainer image from Docker/dockerfile."
        )
    return binary


def main():
    args = parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    lkh_binary = resolve_lkh_binary(args.lkh_binary)
    names = list(DATASET_PATHS) if args.dataset == "all" else [args.dataset]
    summaries = []

    for name in names:
        if args.data_path is not None:
            data_path = args.data_path.expanduser().resolve()
        else:
            data_path = args.project_root / DATASET_PATHS[name]
        summaries.append(evaluate_dataset(name, data_path, args, lkh_binary))

    failed = sum(summary["failed_instances"] for summary in summaries)
    if failed:
        raise SystemExit(f"LKH evaluation finished with {failed} failed instances")


if __name__ == "__main__":
    main()

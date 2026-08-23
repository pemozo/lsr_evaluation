from pathlib import Path

import torch

from Ctrl_ATSProblemDef import get_random_problems


# All randomness (base matrices, pair placement, q values, and directions) is
# controlled by this parameter set.
problem_gen_params = {
    "int_min": 1,
    "int_max": 1_000_000,
    "scaler": 1_000_000,
    "lambda_asym": 0.30,
    "affected_fraction": 0.25,
    "q_mode": "constant",  # "constant" or "uniform"
    "q_value": 1.0,
    "q_min": 0.2,  # used only for q_mode="uniform"
    "q_max": 1.0,  # used only for q_mode="uniform"
    "placement_mode": "random",
    "direction_mode": "random",
    "seed": 42,
    "target_triangle_violation_rate": None,
    "triangle_violation_tolerance": 0.01,
    "triangle_violation_eps": 1e-7,
    "enforce_triangle_inequality": True,
}

num_instances = 64
node_cnt = 250

ROOT = Path(__file__).resolve().parents[2]
output_dir = ROOT / "data" / "ATSP_data" / "Ctrl_ATSP_data1"
output_dir.mkdir(parents=True, exist_ok=True)


# Generate one seeded batch. The generator consumes one continuous random
# stream, so rerunning the script is reproducible while instances within this
# batch are not accidentally reset to the same seed.
problems, diagnostics = get_random_problems(
    batch_size=num_instances,
    node_cnt=node_cnt,
    problem_gen_params=problem_gen_params,
    return_diagnostics=True,
)

for i, instance in enumerate(problems):
    output_path = output_dir / f"instance_{i:04d}.pt"
    torch.save(instance, output_path)
    print(f"[{i + 1}/{num_instances}] saved: {output_path}")

print(
    "Diagnostics: "
    "enforce_triangle_inequality="
    f"{diagnostics['enforce_triangle_inequality']}, "
    f"pairwise_asym_mean={diagnostics['pairwise_asym_mean'].mean().item():.6f}, "
    f"pairwise_asym_max={diagnostics['pairwise_asym_max'].max().item():.6f}, "
    f"asym_pair_count={diagnostics['asym_pair_count'].sum().item()}, "
    "asym_pair_fraction="
    f"{diagnostics['asym_pair_fraction'].mean().item():.6f}, "
    "triangle_violation_rate="
    f"{diagnostics['triangle_violation_rate'].mean().item():.6f}, "
    "triangle_violation_mean="
    f"{diagnostics['triangle_violation_mean'].mean().item():.6f}, "
    "triangle_violation_max="
    f"{diagnostics['triangle_violation_max'].max().item():.6f}, "
    "effective_lambda_asym="
    f"{diagnostics['effective_lambda_asym'].mean().item():.6f}"
)
if problem_gen_params["target_triangle_violation_rate"] is not None:
    print(
        "Triangle target: "
        f"requested={problem_gen_params['target_triangle_violation_rate']:.6f}, "
        "max_error="
        f"{diagnostics['triangle_violation_target_error'].max().item():.6f}, "
        "instances_within_tolerance="
        f"{diagnostics['triangle_violation_target_met'].sum().item()}/"
        f"{num_instances}"
    )
print("Finished.")

from pathlib import Path
import torch

from ATSProblemDef import get_random_problems


problem_gen_params = {
    "int_min": 0,
    "int_max": 1000000,
    "scaler": 1000000,
}

num_instances = 128
node_cnt = 250

ROOT = Path(__file__).resolve().parents[2]

output_dir = (
    ROOT
    / "data"
    / "ATSP_data"
    / "Matnet_atsp250_128instances"
)

output_dir.mkdir(parents=True, exist_ok=True)

for i in range(num_instances):

    problem = get_random_problems(
        batch_size=1,
        node_cnt=node_cnt,
        problem_gen_params=problem_gen_params
    )

    instance = problem[0]

    output_path = output_dir / f"instance_{i:04d}.pt"

    torch.save(instance, output_path)

    print(f"[{i+1}/{num_instances}] saved: {output_path}")

print("Finished.")

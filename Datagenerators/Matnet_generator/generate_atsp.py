import os
import torch

from ATSProblemDef import get_random_problems


problem_gen_params = {
    "int_min": 0,
    "int_max": 1000000,
    "scaler": 1000000,
}

num_instances = 64
node_cnt = 250

script_dir = os.path.dirname(os.path.abspath(__file__))

project_dir = os.path.dirname(script_dir)

output_dir = os.path.join(
    project_dir,
    "data",
    "ATSP_data",
    "Matnet_atsp250_64instances",
)

os.makedirs(output_dir, exist_ok=True)

for i in range(num_instances):

    problem = get_random_problems(
        batch_size=1,
        node_cnt=node_cnt,
        problem_gen_params=problem_gen_params
    )

    instance = problem[0]

    output_path = os.path.join(
        output_dir,
        f"instance_{i:04d}.pt"
    )

    torch.save(instance, output_path)

    print(f"[{i+1}/{num_instances}] saved: {output_path}")

print("Finished.")
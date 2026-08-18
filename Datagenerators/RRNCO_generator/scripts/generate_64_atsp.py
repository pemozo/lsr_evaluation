import json
import os
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GENERATOR_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = GENERATOR_DIR / "data" / "dataset"
OUTPUT_DIR = (
    ROOT
    / "data"
    / "ATSP_data"
    / "UDC_atsp250_64instances"
)

NUM_INSTANCES = 64
NUM_NODES = 250
SEED = 1234
SPLIT = "train"


rng = np.random.default_rng(SEED)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

split_file = os.path.join(
    DATA_DIR,
    "splited_cities_list.json"
)

with open(split_file, "r") as f:
    city_split = json.load(f)

cities = city_split[SPLIT]

print(f"Train cities available: {len(cities)}")

selected_cities = rng.choice(
    cities,
    size=NUM_INSTANCES,
    replace=False
)

for instance_id, city in enumerate(selected_cities):

    city_file = os.path.join(
        DATA_DIR,
        city,
        f"{city}_data.npz"
    )

    print(
        f"[{instance_id + 1:02d}/{NUM_INSTANCES}] "
        f"{city}"
    )

    with np.load(city_file, allow_pickle=True) as data:
        distance = np.asarray(
            data["distance"],
            dtype=np.float32
        )

    invalid = (
        ~np.isfinite(distance)
        | (distance > 1e5)
    )

    valid_indices = np.arange(
        distance.shape[0]
    )

    while invalid.any():

        invalid_per_node = (
            invalid.sum(axis=0)
            + invalid.sum(axis=1)
        )

        worst_node = np.argmax(
            invalid_per_node
        )

        distance = np.delete(
            distance,
            worst_node,
            axis=0
        )

        distance = np.delete(
            distance,
            worst_node,
            axis=1
        )

        valid_indices = np.delete(
            valid_indices,
            worst_node
        )

        invalid = (
            ~np.isfinite(distance)
            | (distance > 1e5)
        )

    if len(valid_indices) < NUM_NODES:
        raise RuntimeError(
            f"{city}: nur {len(valid_indices)} "
            f"gültige Knoten verfügbar"
        )

    local_indices = rng.choice(
        len(valid_indices),
        size=NUM_NODES,
        replace=False
    )

    original_indices = valid_indices[
        local_indices
    ]

    instance = distance[
        np.ix_(
            local_indices,
            local_indices
        )
    ].copy()

    # ATSP Diagonale
    np.fill_diagonal(
        instance,
        0.0
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"instance_{instance_id:03d}.npz"
    )

    np.savez_compressed(
        output_file,

        distance=instance.astype(np.float32),

        # Metadaten
        city=np.array(city),
        node_indices=original_indices.astype(np.int32),
        seed=np.array(SEED),
        instance_id=np.array(instance_id),
        num_nodes=np.array(NUM_NODES),
    )

    print(
        f"    saved: {output_file} "
        f"shape={instance.shape}"
    )


print()
print("Finished.")
print(
    f"{NUM_INSTANCES} instances written to:"
)
print(OUTPUT_DIR)

from pathlib import Path
import torch
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (ROOT / "data" / "ATSP_data"/ "Matnet_atsp250_64instances")

if DATA_DIR.exists():
    print(f"Directory {DATA_DIR} exists.")
else:
    print(f"Directory {DATA_DIR} does not exist.")
    exit(1)

pt_files = sorted(DATA_DIR.glob("*.pt"))

all_values = []

for pt_file in pt_files:
    instance = torch.load(pt_file, map_location="cpu")
    values = instance.float().flatten()
    all_values.append(values)

all_values = torch.cat(all_values)

plt.figure(figsize=(10, 6))

plt.hist(all_values.numpy(), bins=1000,density=True)

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Distribution of all values across all ATSP instances")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()

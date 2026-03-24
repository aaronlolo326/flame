import os

import numpy as np
import matplotlib.pyplot as plt

results_dir = "/storage/backup/hei/ttt/flame/results/"
runs = [
    "20260305_lact_stage1_bz1M_tied"
    # "20260317_e2e_stage1_bz1M"
]
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/losses_via_lm.png"

npy_paths = [os.path.join(results_dir, run_name, "loss", "loss_qwen3.npy") for run_name in runs]
# Plot one line chart that shows all data in npy_paths
print (npy_paths)

plt.figure(figsize=(25, 5))
for i, path in enumerate(npy_paths):
    if os.path.exists(path):
        data = np.load(path)
        # data = data[:,:100]
        # data can be [num_samples, num_tokens] or [num_tokens]
        if data.ndim == 1:
            avg_losses = data
            std_losses = np.zeros_like(data)
        else:
            avg_losses = data.mean(axis=0)
            std_losses = data.std(axis=0)

        x = np.arange(len(avg_losses))
        plt.plot(x, avg_losses, label=runs[i])
        plt.fill_between(
            x,
            avg_losses - std_losses,
            avg_losses + std_losses,
            alpha=0.2,
        )
    else:
        print(f"File not found: {path}")

plt.xlabel('Token Index')
plt.ylabel('Loss')
plt.title('Losses via LM')
plt.legend()
plt.savefig(output_path)


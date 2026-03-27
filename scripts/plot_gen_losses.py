import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline


def smooth_series(values, window_size):
    if window_size <= 1:
        return values

    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(values, kernel, mode="same")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--smooth-window",
    type=int,
    default=50,
    help="Moving-average window size for smoothing plotted curves.",
)
args = parser.parse_args()

results_dir = "/storage/backup/hei/ttt/flame/results/"
runs = [
    "20260305_lact_stage1_bz1M_tied",
    "20260307_lact_nolact-fa-swa_stage1",
    "20260309_lact_nolact-fa_stage1",
    # "20260320_lact_75lact25fa_stage1",
    "20260321_lact_fullrank_stage1",
    "Qwen__Qwen3-1.7B-Base"
    # "20260317_e2e_stage1_bz1M"
]


output_dir = "results"
suffix = "_256_20260326"
suffix = "_128"
suffix = "_256_topkNone"
prefill_len = 256
# suffix = ""
os.makedirs(output_dir, exist_ok=True)
output_path = f"{output_dir}/losses_via_lm{suffix}_smooth.png"

npy_paths = [os.path.join(results_dir, run_name, f"loss{suffix}", "loss_qwen3.npy") for run_name in runs]
# Plot one line chart that shows all data in npy_paths
print (npy_paths)

plt.figure(figsize=(15, 5))
for i, path in enumerate(npy_paths):
    if os.path.exists(path):
        data = np.load(path)
        # data = data[:,:512]
        # data can be [num_samples, num_tokens] or [num_tokens]
        if data.ndim == 1:
            avg_losses = data
            std_losses = np.zeros_like(data)
        else:
            avg_losses = data.mean(axis=0)
            std_losses = data.std(axis=0)
        print (avg_losses)


        x = np.arange(len(avg_losses))

        avg_losses = smooth_series(avg_losses, args.smooth_window)
        print (avg_losses)
        std_losses = smooth_series(std_losses, args.smooth_window)
        plt.plot(x, avg_losses, label=runs[i])
        # plt.fill_between(
        #     x,
        #     avg_losses - std_losses,
        #     avg_losses + std_losses,
        #     alpha=0.2,
        # )

        # # 100 represents number of points to make between T.min and T.max
        # xnew = np.linspace(x.min(), x.max(), 100)  
        # spl = make_interp_spline(x, avg_losses, k=3) 
        # ysmooth = spl(xnew)
        # plt.plot(xnew, ysmooth, label=runs[i])


    else:
        print(f"File not found: {path}")

plt.xlabel('Token Index')
plt.axvline(x=prefill_len-1, color='grey', linestyle='--', linewidth=2)
plt.ylabel('Loss')
# plt.ylim(2, 8)
plt.xlim(0, 8140)
plt.title('Losses via LM')
plt.legend()
plt.savefig(output_path)

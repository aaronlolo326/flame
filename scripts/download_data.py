from datasets import load_dataset
import huggingface_hub as hf # pip install huggingface_hub, if not installed

# ds = load_dataset("emozilla/Long-Data-Collections-Pretrain-Without-Books")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", num_proc=256)

# load fineweb-edu with parallel processing
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64)



# hf.snapshot_download(
#   repo_id="Efficient-Large-Model/llama3-dclm-filter-8k", 
# #   allow_patterns=["*.txt", "*.zarr.zip"], 
#   repo_type="dataset", 
#   local_dir="/storage/backup/hei/data/", 
#   local_dir_use_symlinks="auto"
# )

# hf.snapshot_download(
#   repo_id="mlfoundations/dclm-baseline-1.0", 
# #   allow_patterns=["*.txt", "*.zarr.zip"], 
#   repo_type="dataset", 
#   local_dir="/storage/backup/hei/data/", 
#   local_dir_use_symlinks="auto"
# )

# dataset = load_dataset("mlfoundations/dclm-baseline-1.0")

from datasets import load_dataset
load_dataset("/storage/backup/hei/data/fineweb100bt-qwen3-tokenized-packed-16384")
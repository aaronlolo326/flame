from datasets import load_dataset

# ds = load_dataset("emozilla/Long-Data-Collections-Pretrain-Without-Books")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT", num_proc=256)

# load fineweb-edu with parallel processing
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64)

from datasets import load_dataset
dataset = load_dataset("parquet", data_dir="/storage/backup/hei/data/fineweb100bt-qwen3-tokenized-packed-16384")
dataset.save_to_disk("/storage/backup/hei/data/fineweb100bt-qwen3-tokenized-packed-16384_arrow")

"""
Convert token datasets to an Arrow dataset by decoding with one tokenizer
and re-encoding with another, then chunking and packing.

Supported input formats:
- zarr token arrays
- MDS datasets with `input_ids` samples

Usage:
    python decode_encode_toarrow.py \
        --data_dirs /path/to/data_a /path/to/data_b \
        --data_format zarr|mds|auto \
        --tokenizer_decode gpt2 \
        --tokenizer_encode meta-llama/Llama-2-7b-hf \
        --out_dir /path/to/output \
        --chunk_strategy eos|by_seq_len|by_indices \
        --seq_len 1024 \
        --pack_strategy by_context_len|none \
        --context_len 2048 \
        --num_workers 8 \
        --batch_size 10000
"""

import argparse
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Iterator, List, Optional, Sequence

import numpy as np
import zarr
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets
from streaming import StreamingDataset
from transformers import AutoTokenizer


Batch = dict


# ── helpers ──────────────────────────────────────────────────────────────────

def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    return tok


def get_eos_id(tokenizer) -> int:
    eid = tokenizer.eos_token_id
    if eid is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path!r} has no eos_token_id")
    return eid

def get_special_tokens_id(tokenizer) -> set:
    special_ids = set(tokenizer.all_special_ids)
    return special_ids


def normalize_data_dirs(data_dirs: Optional[Sequence[str]], zarr_path: Optional[str]) -> List[str]:
    if data_dirs:
        return [os.path.abspath(path) for path in data_dirs]
    if zarr_path:
        return [os.path.abspath(zarr_path)]
    raise ValueError("Provide --data_dirs or the legacy --zarr_path")


def infer_data_format(path: str) -> str:
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "index.json")):
            return "mds"
        if path.endswith(".zarr") or os.path.exists(os.path.join(path, ".zgroup")) or os.path.exists(os.path.join(path, ".zarray")):
            return "zarr"
    raise ValueError(f"Could not infer data format for {path!r}; please set --data_format explicitly")


def resolve_data_format(data_format: str, path: str) -> str:
    return infer_data_format(path) if data_format == "auto" else data_format


def open_zarr_array(path: str, array_key: Optional[str]):
    store = zarr.open(path, mode="r")
    if array_key:
        data = store[array_key]
    elif isinstance(store, zarr.Array):
        data = store
    else:
        keys = list(store.keys())
        print(f"Zarr keys found in {path}: {keys}")
        data = store[keys[0]]
        print(f"Using key: {keys[0]!r}")
    return data


def iter_zarr_batches(path: str, batch_size: int, array_key: Optional[str]) -> Iterator[Batch]:
    data = open_zarr_array(path, array_key)
    total_tokens = data.shape[0]
    for i in range(0, total_tokens, batch_size):
        yield {
            "token_ids": np.asarray(data[i : i + batch_size], dtype=np.int32),
            "indices": None,
        }


def iter_mds_batches(path: str, batch_size: int) -> Iterator[Batch]:
    dataset = StreamingDataset(
        local=path,
        remote=None,
        shuffle=False,
        batch_size=1,
    )

    buffered: List[np.ndarray] = []
    buffered_indices: List[np.ndarray] = []
    buffered_tokens = 0
    for sample in dataset:
        if "input_ids" not in sample:
            raise KeyError(f"MDS sample from {path!r} is missing 'input_ids'")
        sample_ids = np.asarray(sample["input_ids"], dtype=np.int32).reshape(-1)
        if sample_ids.size == 0:
            continue

        sample_indices = None
        if "indices" in sample and sample["indices"] is not None:
            raw_indices = np.asarray(sample["indices"], dtype=np.int64).reshape(-1, 2)
            if raw_indices.size:
                valid_rows = raw_indices[:, 1] > raw_indices[:, 0]
                sample_indices = raw_indices[valid_rows]

        buffered.append(sample_ids)
        if sample_indices is not None and sample_indices.size:
            adjusted = sample_indices + buffered_tokens
            buffered_indices.append(adjusted.astype(np.int64))
        buffered_tokens += sample_ids.size
        if buffered_tokens >= batch_size:
            yield {
                "token_ids": np.concatenate(buffered),
                "indices": np.concatenate(buffered_indices) if buffered_indices else None,
            }
            buffered = []
            buffered_indices = []
            buffered_tokens = 0

    if buffered:
        yield {
            "token_ids": np.concatenate(buffered),
            "indices": np.concatenate(buffered_indices) if buffered_indices else None,
        }


def get_total_tokens(data_dirs: Sequence[str], data_format: str, array_key: Optional[str]) -> Optional[int]:
    if data_format == "mds":
        return None
    return sum(int(open_zarr_array(path, array_key).shape[0]) for path in data_dirs)


def iter_input_batches(
    data_dirs: Sequence[str],
    data_format: str,
    batch_size: int,
    array_key: Optional[str],
) -> Iterator[np.ndarray]:
    for path in data_dirs:
        path_format = resolve_data_format(data_format, path)
        print(f"Opening {path_format} dataset from: {path}")
        if path_format == "zarr":
            yield from iter_zarr_batches(path, batch_size, array_key)
        elif path_format == "mds":
            yield from iter_mds_batches(path, batch_size)
        else:
            raise ValueError(f"Unsupported data_format: {path_format!r}")


# ── chunking ─────────────────────────────────────────────────────────────────

def chunk_by_eos(token_ids: np.ndarray, eos_id: int) -> List[np.ndarray]:
    """Split 1-D token array at every EOS token (inclusive)."""
    positions = np.where(token_ids == eos_id)[0]
    chunks = []
    prev = 0
    for pos in positions:
        chunks.append(token_ids[prev : pos + 1])
        prev = pos + 1
    if prev < len(token_ids):
        chunks.append(token_ids[prev:])
    return [c for c in chunks if len(c) > 0]


def chunk_by_seq_len(token_ids: np.ndarray, seq_len: int) -> List[np.ndarray]:
    """Split 1-D token array into fixed-length windows (last chunk may be shorter)."""
    return [token_ids[i : i + seq_len] for i in range(0, len(token_ids), seq_len)]


def chunk_by_indices(token_ids: np.ndarray, indices: np.ndarray) -> List[np.ndarray]:
    """Split a 1-D token array using [start, end) index pairs."""
    if (
        indices.ndim != 2
        or indices.shape[1] != 2
        or indices.shape[0] == 0
        or indices[0, 0] != 0
        or indices[-1, 1] != token_ids.shape[0]
        or np.any(indices[:, 1] <= indices[:, 0])
        or np.any(indices[:-1, 1] != indices[1:, 0])
    ):
        # raise ValueError("indices must be contiguous, non-empty [start, end) pairs covering token_ids exactly")
        print ("Indices with non-contiguous or empty [start, end)", f"{len(token_ids)=}, {indices=}")
        print ("Indices with non-contiguous or empty [start, end)", f"{len(token_ids)=}, {indices=}", file=open('debug.log', 'a'))
        for i,j in indices:
            print(i, j, file=open('debug.log', 'a'))
        print("-----------", file=open('debug.log', 'a'))
        return None
    chunks = np.split(token_ids, indices[:-1, 1])
    return chunks


# ── packing ──────────────────────────────────────────────────────────────────

def pack_by_context_len(chunks: List[np.ndarray], context_len: int, pad_id: int = 0) -> List[np.ndarray]:
    """
    Greedily pack chunks into fixed-length sequences.
    Sequences that are too long are truncated.
    """
    packed = []
    buf = []
    buf_len = 0
    for chunk in chunks:
        if buf_len + len(chunk) <= context_len:
            buf.append(chunk)
            buf_len += len(chunk)
        else:
            if buf:
                seq = np.concatenate(buf)
                # pad if needed
                if len(seq) < context_len:
                    seq = np.pad(seq, (0, context_len - len(seq)), constant_values=pad_id)
                packed.append(seq[:context_len])
            # start new buffer with current chunk (truncate if chunk itself too long)
            buf = [chunk[:context_len]]
            buf_len = len(buf[0])
    if buf:
        seq = np.concatenate(buf)
        if len(seq) < context_len:
            seq = np.pad(seq, (0, context_len - len(seq)), constant_values=pad_id)
        packed.append(seq[:context_len])
    return packed


# ── worker ───────────────────────────────────────────────────────────────────

def process_batch(
    batch: Batch,
    dec_tok: str,
    enc_tok: str,
    chunk_strategy: str,
    seq_len: Optional[int],
    pack_strategy: str,
    context_len: Optional[int],
    check: bool=False
):
    """Process a batch of raw token ids plus optional chunk boundary metadata."""
    # Load tokenizers inside worker to avoid pickling issues
    batch_token_ids = batch["token_ids"]
    batch_indices = batch.get("indices")

    dec_eos = get_eos_id(dec_tok)
    enc_eos = get_eos_id(enc_tok)
    enc_pad = enc_tok.pad_token_id if enc_tok.pad_token_id is not None else 0


    # 2. Split into sentences/documents using the decode tokenizer's EOS as delimiter
    # We work on the raw ids for chunking.
    if chunk_strategy == "eos":
        chunks_ids = chunk_by_eos(batch_token_ids, dec_eos)
    elif chunk_strategy == "by_indices":
        if batch_indices is None:
            raise ValueError("chunk_strategy='by_indices' requires MDS samples with an 'indices' field")
        chunks_ids = chunk_by_indices(batch_token_ids, batch_indices)
        # if chunks_ids is not None:
        #     return []
        if check:
            print ("chunk boundaries (by_indices):\n", batch_indices)
    elif chunk_strategy == "by_seq_len":
        assert seq_len, "seq_len required for by_seq_len chunk strategy"
        # Decode full text once, re-encode, then split
        text = dec_tok.decode(batch_token_ids.tolist(), skip_special_tokens=True)
        full_ids = np.array(enc_tok.encode(text, add_special_tokens=False), dtype=np.int32)
        re_encoded = chunk_by_seq_len(full_ids, seq_len)
        chunks_ids = None
    else:
        raise ValueError(f"Unknown chunk_strategy: {chunk_strategy!r}")
    
    if check:
        print (f"{len(chunks_ids)=}")

    if chunks_ids is None:
        processed = []

    else:
        if chunk_strategy in {"eos", "by_indices"}:
            # Re-encode each chunk independently
            re_encoded = []
            for i, chunk in enumerate(chunks_ids):
                # chunk = chunk[~np.isin(chunk, list(dec_special_ids))]  # remove special tokens before decoding
                chunk_text = dec_tok.decode(chunk.tolist(), skip_special_tokens=True)
                if check:
                    print ("decode----------------------------\n", chunk_text[:200], "...", chunk_text[-200:])
                    print (f"{len(chunk_text)=}\n")
                ids = enc_tok.encode(chunk_text, add_special_tokens=False)
                if i != len(chunks_ids) - 1:  # if not the last chunk, ensure it ends with EOS
                    ids.append(enc_eos)  # add EOS back to the end of each chunk for the encode tokenizer
                if check:
                    text = enc_tok.decode(ids, skip_special_tokens=False)
                    print ("encode-decode----------------------------\n", text[:200], "...", text[-200:])
                    print (f"{len(text)=}\n")
                    print (f"-----------------------------------------------------------------------------------------------------------------------\n")

                re_encoded.append(np.array(ids, dtype=np.int32))

        # 3. Pack
        if pack_strategy == "by_context_len":
            assert context_len, "context_len required for by_context_len pack strategy"
            sequences = pack_by_context_len(re_encoded, context_len, pad_id=enc_pad)
        elif pack_strategy == "none":
            sequences = re_encoded
        else:
            raise ValueError(f"Unknown pack_strategy: {pack_strategy!r}")

        processed = [s.tolist() for s in sequences]

    return processed

def write_shard(sequences: List[List[int]], out_dir: str, shard_idx: int, 
                use_fixed: bool, context_len: Optional[int], write_arrow: bool=False):
    """Write a single shard to parquet."""
    if use_fixed:
        arr_type = pa.list_(pa.int32(), context_len)
        pa_array = pa.array(sequences, type=arr_type)
    else:
        pa_array = pa.array(sequences, type=pa.list_(pa.int32()))
    
    table = pa.table({"input_ids": pa_array})
    base = os.path.join(out_dir, f"shard_{shard_idx:05d}")

    pq.write_table(table, f"{base}.parquet", compression="zstd", row_group_size=1024)

    # optionally write arrow IPC
    if write_arrow:
        with pa.OSFile(f"{base}.arrow", "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    return base, len(table)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", default=None, help="One or more input directories")
    parser.add_argument("--zarr_path", default=None, help="Legacy single zarr path; prefer --data_dirs")
    parser.add_argument("--data_format", default="auto", choices=["auto", "zarr", "mds"])
    parser.add_argument("--tokenizer_decode", required=True)
    parser.add_argument("--tokenizer_encode", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk_strategy", required=True, choices=["eos", "by_seq_len", "by_indices"])
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pack_strategy", required=True, choices=["by_context_len", "none"])
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50_000, help="Tokens per worker batch")
    parser.add_argument("--num_workers", type=int, default=max(1, cpu_count() - 29))
    parser.add_argument("--array_key", type=str, default=None, help="Key inside zarr store (leave blank to auto-detect)")
    parser.add_argument("--shard_size", type=int, default=100_000, help="Number of sequences per output shard")
    parser.add_argument("--write_arrow", action="store_true", help="Also write Arrow IPC shards alongside Parquet (faster loads, larger disk)")
    parser.add_argument("--check", action="store_true", help="Check before proceeding")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data_dirs = normalize_data_dirs(args.data_dirs, args.zarr_path)

    # Validate format detection up front so we fail before multiprocessing starts.
    for path in data_dirs:
        resolve_data_format(args.data_format, path)

    dec_tok = load_tokenizer(args.tokenizer_decode)
    enc_tok = load_tokenizer(args.tokenizer_encode)
    if args.check:
        # ── sample preview ───────────────────────────────────────────────────────
        print("\n── Sample preview (first batch) ──")
        try:
            sample_batch = next(iter_input_batches(data_dirs, args.data_format, args.batch_size, args.array_key))
        except StopIteration:
            raise ValueError("No input tokens found in the provided datasets")
        sample_seqs = process_batch(
            sample_batch,
            dec_tok,
            enc_tok,
            args.chunk_strategy,
            args.seq_len,
            args.pack_strategy,
            args.context_len,
            check=args.check
        )
        if sample_seqs:
            print(f"Sequence lengths: min={min(len(s) for s in sample_seqs)}, "
                f"max={max(len(s) for s in sample_seqs)}, "
                f"mean={sum(len(s) for s in sample_seqs)/len(sample_seqs):.1f}")

        confirm = input("\nDoes this look correct? Proceed with full dataset? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    # ── process in parallel ──────────────────────────────────────────────────
    from queue import Queue
    from threading import Thread
    def prefetch_batches(batch_iter: Iterator[Batch], prefetch: int = 4):
        """
        Reads batches in a background thread and yields numpy arrays.
        """
        q = Queue(maxsize=prefetch)

        def reader():
            for batch in batch_iter:
                q.put(batch)
            q.put(None)  # sentinel

        t = Thread(target=reader, daemon=True)
        t.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

        t.join()
    batches = prefetch_batches(
        iter_input_batches(data_dirs, args.data_format, args.batch_size, args.array_key),
        prefetch=4,
    )

    total_tokens = get_total_tokens(data_dirs, args.data_format, args.array_key)
    if total_tokens is not None:
        num_batches = (total_tokens + args.batch_size - 1) // args.batch_size
        print(f"\nTotal tokens in dataset(s): {total_tokens:,}")
        print(f"Processing {num_batches} batches with {args.num_workers} workers...")
    else:
        num_batches = None
        print(f"\nProcessing MDS input from {len(data_dirs)} director{'y' if len(data_dirs) == 1 else 'ies'} with {args.num_workers} workers...")

    worker_fn = partial(
        process_batch,
        dec_tok=dec_tok,
        enc_tok=enc_tok,
        chunk_strategy=args.chunk_strategy,
        seq_len=args.seq_len,
        pack_strategy=args.pack_strategy,
        context_len=args.context_len,
    )

    all_sequences: List[List[int]] = []
    shard_datasets = []

    from tqdm import tqdm
    with Pool(args.num_workers) as pool:
        for i, result in enumerate(tqdm(pool.imap_unordered(worker_fn, batches, chunksize=4), total=num_batches, desc="Processing batches")):
            all_sequences.extend(result)

            while len(all_sequences) >= args.shard_size:
                to_write, all_sequences = all_sequences[:args.shard_size], all_sequences[args.shard_size:]
                shard_datasets.append(Dataset.from_dict({"input_ids": to_write}))
                print(f"  Buffered shard {len(shard_datasets)-1} ({len(to_write):,} seqs)")

    if all_sequences:
        shard_datasets.append(Dataset.from_dict({"input_ids": all_sequences}))

    from datasets import concatenate_datasets
    dataset = concatenate_datasets(shard_datasets)
    dataset.save_to_disk(args.out_dir)

    print(f"\nDataset saved → {args.out_dir}")
    print(f"Total sequences: {len(dataset):,}")
    print(f"Load with: datasets.load_from_disk('{args.out_dir}')")
    
    # from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset, load_from_disk
    # d = load_from_disk(args.out_dir)
    # breakpoint()

if __name__ == "__main__":
    main()

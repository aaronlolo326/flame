"""
Convert zarr token arrays to Arrow dataset by decoding with one tokenizer
and re-encoding with another, then chunking and packing.

Usage:
    python zarr_to_arrow.py \
        --zarr_path /path/to/data.zarr \
        --tokenizer_decode gpt2 \
        --tokenizer_encode meta-llama/Llama-2-7b-hf \
        --out_dir /path/to/output \
        --chunk_strategy eos|by_seq_len \
        --seq_len 1024 \
        --pack_strategy by_context_len|none \
        --context_len 2048 \
        --num_workers 8 \
        --batch_size 10000
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Optional
import json

import numpy as np
import zarr
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


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
    batch_token_ids: np.ndarray,
    dec_tok: str,
    enc_tok: str,
    chunk_strategy: str,
    seq_len: Optional[int],
    pack_strategy: str,
    context_len: Optional[int],
    check: bool=False
):
    """Process a batch of raw token ids (1-D array from zarr)."""
    # Load tokenizers inside worker to avoid pickling issues

    dec_eos = get_eos_id(dec_tok)
    enc_eos = get_eos_id(enc_tok)
    dec_special_ids = get_special_tokens_id(dec_tok)
    enc_pad = enc_tok.pad_token_id if enc_tok.pad_token_id is not None else 0


    # 2. Split into sentences/documents using the decode tokenizer's EOS as delimiter
    # We work on the raw ids for chunking.
    if chunk_strategy == "eos":
        chunks_ids = chunk_by_eos(batch_token_ids, dec_eos)
        # Re-encode each chunk independently
        re_encoded = []
        for i, chunk in enumerate(chunks_ids):
            # chunk = chunk[~np.isin(chunk, list(dec_special_ids))]  # remove special tokens before decoding
            chunk_text = dec_tok.decode(chunk.tolist(), skip_special_tokens=True)
            if check:
                print ("decode:", chunk_text[:200], "...", chunk_text[-200:])
            ids = enc_tok.encode(chunk_text, add_special_tokens=False)
            if i != len(chunks_ids) - 1:  # if not the last chunk, ensure it ends with EOS
                ids.append(enc_eos)  # add EOS back to the end of each chunk for the encode tokenizer
            if check:
                text = enc_tok.decode(ids, skip_special_tokens=False)
                print ("decode2:", text[:200], "...", text[-200:])
                print ("--------------------------------------------------------------------------")

            re_encoded.append(np.array(ids, dtype=np.int32))
    elif chunk_strategy == "by_seq_len":
        assert seq_len, "seq_len required for by_seq_len chunk strategy"
        # Decode full text once, re-encode, then split
        text = dec_tok.batch_decode(batch_token_ids.tolist(), skip_special_tokens=False)
        full_ids = np.array(enc_tok.encode(text, add_special_tokens=False), dtype=np.int32)
        re_encoded = chunk_by_seq_len(full_ids, seq_len)
    else:
        raise ValueError(f"Unknown chunk_strategy: {chunk_strategy!r}")

    # 3. Pack
    if pack_strategy == "by_context_len":
        assert context_len, "context_len required for by_context_len pack strategy"
        sequences = pack_by_context_len(re_encoded, context_len, pad_id=enc_pad)
    elif pack_strategy == "none":
        sequences = re_encoded
    else:
        raise ValueError(f"Unknown pack_strategy: {pack_strategy!r}")

    return [s.tolist() for s in sequences]

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
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--tokenizer_decode", required=True)
    parser.add_argument("--tokenizer_encode", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk_strategy", required=True, choices=["eos", "by_seq_len"])
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

    # ── open zarr ────────────────────────────────────────────────────────────
    print(f"Opening zarr dataset from: {args.zarr_path}")
    store = zarr.open(args.zarr_path, mode="r")

    if args.array_key:
        data = store[args.array_key]
    elif isinstance(store, zarr.Array):
        print("isinstance(store, zarr.Array)")
        data = store
    else:
        # Auto-detect first array
        keys = list(store.keys())
        print(f"Zarr keys found: {keys}")
        data = store[keys[0]]
        print(f"Using key: {keys[0]!r}")

    dec_tok = load_tokenizer(args.tokenizer_decode)
    enc_tok = load_tokenizer(args.tokenizer_encode)
    if args.check:
        # ── sample preview ───────────────────────────────────────────────────────
        print("\n── Sample preview (first batch) ──")
        sample_batch = data[: min(args.batch_size, data.shape[0])]
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

    # flat = data[:].ravel().astype(np.int32)
    total_tokens = store.shape[0]
    print (f"\nTotal tokens in dataset: {total_tokens:,}")

    # ── process in parallel ──────────────────────────────────────────────────
    # batches = [data[i : i + args.batch_size] for i in range(0, total_tokens, args.batch_size)]
    # print(f"\nProcessing {len(batches)} batches with {args.num_workers} workers...")

    # def iter_batches(data, batch_size, total_tokens):
    #     for i in range(0, total_tokens, batch_size):
    #         yield data[i : i + batch_size]
    # batches = iter_batches(data, args.batch_size, total_tokens)

    from queue import Queue
    from threading import Thread
    def prefetch_batches(data, batch_size, total_tokens, prefetch=4):
        """
        Reads zarr slices in a background thread and yields numpy arrays.
        prefetch controls how many batches are read ahead into the queue.
        """
        q = Queue(maxsize=prefetch)

        def reader():
            for i in range(0, total_tokens, batch_size):
                q.put(data[i : i + batch_size])  # blocks if queue is full
            q.put(None)  # sentinel

        t = Thread(target=reader, daemon=True)
        t.start()

        while True:
            batch = q.get()
            if batch is None:
                break
            yield batch

        t.join()
    batches = prefetch_batches(data, args.batch_size, total_tokens, prefetch=4)

    num_batches = (total_tokens + args.batch_size - 1) // args.batch_size
    print(f"\nProcessing {num_batches} batches with {args.num_workers} workers...")

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

    # After the pool loop, instead of write_shard calls:
    from datasets import Dataset

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
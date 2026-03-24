import torch
import triton
import triton.testing
import math

# ── Shared benchmark configurations ──
BENCH_DOC_LENS = {
    "varied_4doc":  [4096, 3072, 2048, 1024],
    "varied_23doc": [1, 4213, 1906, 295, 537, 1580, 213, 475, 743, 659,
                     1414, 230, 1520, 116, 745, 327, 181, 193, 187, 96,
                     110, 303, 340],
    "uniform_2doc": [8192, 8192],
    "single_doc":   [16384],
}


def make_cu_seqlens(doc_lens, device="cuda"):
    """Build cu_seqlens [G+1] from a list of doc lengths."""
    return torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(doc_lens), 0).tolist()),
        dtype=torch.int32, device=device,
    )


def get_chunk_info(cu_seqlens, chunk_size, chunk_idx):
    """Compute per-doc chunk ranges and gather indices for chunk testing.

    Args:
        cu_seqlens: [G+1] int32 tensor, full doc boundaries
        chunk_size: int, tokens per chunk
        chunk_idx: int, which chunk

    Returns:
        chunk_cu_seqlens: [G+1] int32 tensor for gathered chunk tokens
        indices: 1D int64 tensor of token positions in the packed buffer
        max_chunks: max number of chunks across all docs
    """
    G = cu_seqlens.shape[0] - 1
    device = cu_seqlens.device
    doc_starts = cu_seqlens[:-1].long()
    doc_ends = cu_seqlens[1:].long()

    chunk_starts = torch.clamp(doc_starts + chunk_idx * chunk_size, max=doc_ends)
    chunk_ends = torch.clamp(doc_starts + (chunk_idx + 1) * chunk_size, max=doc_ends)
    chunk_lens = chunk_ends - chunk_starts  # 0 for inactive docs

    chunk_cu_seqlens = torch.zeros(G + 1, dtype=torch.int32, device=device)
    chunk_cu_seqlens[1:] = chunk_lens.cumsum(0).int()

    # Build flat index tensor
    indices = torch.cat([
        torch.arange(s.item(), e.item(), device=device)
        for s, e in zip(chunk_starts, chunk_ends)
    ]) if chunk_lens.sum() > 0 else torch.empty(0, dtype=torch.long, device=device)

    doc_lens = doc_ends - doc_starts
    max_chunks = max(math.ceil(dl.item() / chunk_size) for dl in doc_lens)

    return chunk_cu_seqlens, indices, max_chunks


def _as_tuple(output):
    if isinstance(output, tuple):
        return output
    return (output,)


def _align_for_compare(triton_tensor, ref_tensor):
    if triton_tensor.device != ref_tensor.device:
        ref_tensor = ref_tensor.to(triton_tensor.device)
    if triton_tensor.dtype != ref_tensor.dtype:
        triton_tensor = triton_tensor.float()
        ref_tensor = ref_tensor.float()
    return triton_tensor, ref_tensor


def test_correctness(
    triton_fn,
    ref_fn,
    triton_args,
    ref_args,
    output_transform=None,
    atol=1e-4,
    rtol=1e-4,
    debug=False,
):
    """Test kernel correctness by comparing triton and reference implementations.

    Args:
        triton_fn: Triton kernel function
        ref_fn: Reference implementation function
        triton_args: Arguments for triton function
        ref_args: Arguments for reference function
        output_transform: Optional function to transform triton outputs before comparison
        atol: Absolute tolerance for torch.allclose
        rtol: Relative tolerance for torch.allclose
        debug: If True, print detailed sample comparisons

    Returns:
        bool: True if all outputs match within tolerance
    """
    triton_out = triton_fn(*triton_args)
    ref_out = ref_fn(*ref_args)

    # Apply transform if needed (e.g., permute)
    if output_transform:
        triton_out = output_transform(triton_out)

    triton_out = _as_tuple(triton_out)
    ref_out = _as_tuple(ref_out)

    if len(triton_out) != len(ref_out):
        print(f"✗ FAIL: output count mismatch (triton={len(triton_out)}, ref={len(ref_out)})")
        return False

    # Compare each output
    matches = []
    diffs = []
    mean_diffs = []
    for t, r in zip(triton_out, ref_out):
        t, r = _align_for_compare(t, r)
        matches.append(torch.allclose(t, r, atol=atol, rtol=rtol))
        diffs.append((t - r).abs().max().item())
        mean_diffs.append((t - r).abs().mean().item())

    # Print results
    for i, (diff, mean_diff, match, t, r) in enumerate(
        zip(diffs, mean_diffs, matches, triton_out, ref_out)
    ):
        print(f"Output {i}: mean_diff={mean_diff:.2e}, max_diff={diff:.2e}, match={match}")

        if debug:
            # Print elements with biggest errors
            t_flat = t.flatten()
            r_flat = r.flatten()
            abs_diff = (t_flat - r_flat).abs()
            n_samples = min(10, len(t_flat))

            # Get indices of largest errors
            top_indices = torch.topk(abs_diff, n_samples).indices

            print(f"  {'Idx':<8} {'Triton':<20} {'Ref':<20} {'Diff':<15}")
            print(f"  {'-'*63}")

            for j in top_indices:
                idx = j.item()
                tv_val, rv_val = t_flat[idx].item(), r_flat[idx].item()
                diff_val = abs_diff[idx].item()
                print(f"  {idx:<8} {tv_val:<20.6f} {rv_val:<20.6f} {diff_val:<15.2e}")

    passed = all(matches)
    print(f"{'✓ PASS' if passed else '✗ FAIL'} (atol={atol}, rtol={rtol})")
    return passed


def benchmark(
    triton_fn, ref_fn, triton_args, ref_args,
    warmup_iters=100, enable_grad=False,
    warmup_ms=5000, rep_ms=2000,
    verbose=True,
):
    """Benchmark speed and memory of triton vs reference implementation.

    Args:
        triton_fn: Triton kernel function
        ref_fn: Reference implementation function
        triton_args: Arguments for triton function
        ref_args: Arguments for reference function
        warmup_iters: Number of warmup iterations
        enable_grad: If True, enable gradients (for backward pass benchmarking)
        warmup_ms: Warmup time for do_bench (ms)
        rep_ms: Measurement time for do_bench (ms)

    Returns:
        dict: Benchmark results with keys: time_triton, time_ref, speedup, mem_triton, mem_ref
    """
    from contextlib import nullcontext

    ctx = nullcontext() if enable_grad else torch.no_grad()
    quantiles = [0.5, 0.2, 0.8]

    with ctx:
        # Warmup + benchmark Triton (dedicated warmup right before measurement)
        for _ in range(warmup_iters):
            triton_fn(*triton_args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        time_triton, _, _ = triton.testing.do_bench(
            lambda: triton_fn(*triton_args),
            warmup=warmup_ms, rep=rep_ms, quantiles=quantiles,
        )
        mem_triton = torch.cuda.max_memory_allocated() / 1e9

        # Warmup + benchmark Reference (dedicated warmup right before measurement)
        for _ in range(warmup_iters):
            ref_fn(*ref_args)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        time_ref, _, _ = triton.testing.do_bench(
            lambda: ref_fn(*ref_args),
            warmup=warmup_ms, rep=rep_ms, quantiles=quantiles,
        )
        mem_ref = torch.cuda.max_memory_allocated() / 1e9

        speedup = time_ref / time_triton

        if verbose:
            print(f"\nSpeed (ms): Triton {time_triton:.6f} | Ref {time_ref:.6f} | Speedup {speedup:.2f}x")
            print(f"Memory (GB): Triton {mem_triton:.6f} | Ref {mem_ref:.6f}")

        return {
            'time_triton': time_triton,
            'time_ref': time_ref,
            'speedup': speedup,
            'mem_triton': mem_triton,
            'mem_ref': mem_ref
        }


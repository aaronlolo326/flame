import torch


def compute_varlen_args(cu_seqlens, chunk_size=0, chunk_idx=0):
    """Precompute eff_lens, bos_arr, max_sl from cu_seqlens + chunk params.
    Call once per chunk iteration, pass results to all kernels."""
    doc_starts = cu_seqlens[:-1].long()
    doc_ends = cu_seqlens[1:].long()
    if chunk_size > 0:
        bos_arr = torch.clamp(doc_starts + chunk_idx * chunk_size, max=doc_ends)
        eos_arr = torch.clamp(doc_starts + (chunk_idx + 1) * chunk_size, max=doc_ends)
        eff_lens = (eos_arr - bos_arr).int()
        max_sl = chunk_size
    else:
        bos_arr = doc_starts
        eff_lens = (doc_ends - doc_starts).int()
        max_sl = eff_lens.max().item()
    return eff_lens, bos_arr, max_sl

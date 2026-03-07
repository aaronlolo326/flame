python -m benchmarks.throughput.export_whole_model_configs \
  --base-config configs/gated_deltanet_1B.json \
  --seq-len 16384 \
  --sliding-window 2163848 \
  --num-attn-heads 16 \
  --num-lact-heads 4 \
  --output-dir configs

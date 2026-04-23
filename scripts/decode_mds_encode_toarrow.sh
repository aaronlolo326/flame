DECODE_TOKENIZER_PATH="meta-llama/Llama-3.2-1B"
TOKENIZER_PATH="Qwen/Qwen3-4B-Base"

data_dir="/storage/backup/hei/data"
mds_subsets=(
    "thestackv1_concat_by_repo-524288"
    "thestackv1_concat_by_repo-65536"
    "book-524288"
    "book-65536"
    "fineweb-edu"
    "fineweb-2023-50"
    "stackexchange"
    "dolmawiki"
    "tuluv2"
    "arxiv"
    "openwebmath"
    "textbooks"
)

mds_dirs=""
for subset in "${mds_subsets[@]}"; do
    mds_dirs="$mds_dirs $data_dir/$subset"
done

python scripts/decode_encode_toarrow.py \
    --data_dirs $mds_dirs \
    --data_format mds \
    --tokenizer_decode ${DECODE_TOKENIZER_PATH} \
    --tokenizer_encode ${TOKENIZER_PATH} \
    --out_dir /storage/backup/hei/data/qwen3-prolong512Kto128K_train \
    --chunk_strategy by_indices \
    --pack_strategy by_context_len \
    --context_len 131072 \
    --num_workers 200 \
    --batch_size 1000000 \
    --write_arrow \
    --check

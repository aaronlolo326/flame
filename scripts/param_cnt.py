import logging
from flame.config_manager import JobConfig
from flame.tools.utils import get_nparams_and_flops
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from torchtitan.tools.logging import logger, init_logger

def main(job_config: JobConfig):

    import json
    with open(job_config.model.config, "r") as f:
        model_json = json.load(f)
        model_type = model_json.get("model_type")
        layer_types = model_json.get("layer_types")
    from custom_models import MODEL_TYPE_TO_PARENT_DIR
    parent_dir = MODEL_TYPE_TO_PARENT_DIR[model_type]
    import importlib
    importlib.import_module(f"custom_models.{parent_dir}")

    print(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
    # calculate model size and flops per token
    num_distinct_params, nparams_embedding, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )
    print(f"Number of distinct model parameters = {num_distinct_params:,}")
    if model_config.tie_word_embeddings:
        print("Model ties input and output word embeddings, so")
        print(f"Number of total (non-distinct) parameters = {num_distinct_params + nparams_embedding:,}")
    else:
        print(f"Model does not tie input and output word embeddings, so the number of unique model parameters is the above.")
    print(f"Number of embedding parameters = {nparams_embedding:,}")
    print(f"Number of flops per token = {num_flops_per_token:,}")

if __name__ == '__main__':

    init_logger()
    # Configure logger format to include filename and line number
    formatter = logging.Formatter(
        "[titan] %(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    config = JobConfig()
    config.parse_args()
    main(config)

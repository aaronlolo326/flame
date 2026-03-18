# MODEL_TYPE_TO_PARENT_DIR = {
#     "lact_swiglu": "lact_model",
#     "qwen3_gdn": "qwen3_gdn"
# }

import warnings

from .lact_model import *

try:
    from .qwen3_ import *
except Exception as exc:
    warnings.warn(f"Skipping qwen3_ custom model registration: {exc}")

# from .qwen3_gdn import *

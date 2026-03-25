# MODEL_TYPE_TO_PARENT_DIR = {
#     "lact_swiglu": "lact_model",
#     "qwen3_gdn": "qwen3_gdn"
# }

# from .lact_model import *
# from .qwen3_gdn import *
from . import lact_model
from . import qwen3_
from . import lact_model_mlp
from . import e2e_legacy
from . import ttt_e2e_lact_backbone
from . import ttt_e2e_multi_level
# import warnings

from . import hybrid_qwen3_lact_model

# try:
#     from . import *
# except Exception as exc:
#     warnings.warn(f"Skipping qwen3_ custom model registration: {exc}")

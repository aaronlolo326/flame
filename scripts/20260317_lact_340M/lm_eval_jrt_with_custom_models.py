#!/usr/bin/env python3
import sys


PREFIX_LM_EVAL_DIR = "/work/yufei/projects/prefix-linear-attention/lm-eval-harness"

if PREFIX_LM_EVAL_DIR not in sys.path:
    sys.path.insert(0, PREFIX_LM_EVAL_DIR)

import custom_models  # noqa: F401
from lm_eval.__main__ import cli_evaluate


if __name__ == "__main__":
    cli_evaluate()

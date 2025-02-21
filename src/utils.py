import logging as py_logging

from datasets.utils import logging as ds_logging
from tokenizers import Tokenizer
from transformers.utils import logging as tr_logging


def setup_logging():
    py_logging.basicConfig(level=py_logging.INFO)
    tr_logging.set_verbosity_info()
    tr_logging.disable_progress_bar()
    ds_logging.set_verbosity_info()
    ds_logging.disable_progress_bar()


def add_required_tokens(tokenizer: Tokenizer) -> Tokenizer:
    required_token_types = ["bos_token", "eos_token", "cls_token", "sep_token", "pad_token"]
    default_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
    }

    # Ensure each required token type exists
    tokens_to_add = {}
    for token_type in required_token_types:
        current_token = getattr(tokenizer, token_type, None)
        default_token = default_tokens[token_type]

        if current_token is None:
            current_token = default_token

        # Check if the token string is in the vocabulary
        if current_token not in tokenizer.vocab:
            tokens_to_add[token_type] = current_token

    # Add missing tokens to the tokenizer
    if tokens_to_add:
        tokenizer.add_special_tokens(tokens_to_add)

    # Check for bos/cls conflict
    if tokenizer.bos_token_id == tokenizer.cls_token_id:
        new_token = tokenizer.cls_token + "_1"
        tokenizer.add_special_tokens({"cls_token": new_token})

    # Check for eos/sep conflict
    if tokenizer.eos_token_id == tokenizer.sep_token_id:
        new_token = tokenizer.sep_token + "_1"
        tokenizer.add_special_tokens({"sep_token": new_token})

    return tokenizer

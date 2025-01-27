import logging
import logging as py_logging

import numpy as np
import torch
from datasets.utils import logging as ds_logging
from numpy import ndarray
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer
from transformers.utils import logging as tr_logging


def setup_logging():
    py_logging.basicConfig(level=py_logging.INFO)
    tr_logging.set_verbosity_info()
    tr_logging.disable_progress_bar()
    ds_logging.set_verbosity_info()
    ds_logging.disable_progress_bar()


def make_classifier_prompt(
    input_seq: ndarray,
    positives: list[ndarray],
    tokenizer: PreTrainedTokenizer,
    rng,
    negatives: list[ndarray] = [],
    train_flag: bool = False,
) -> tuple[ndarray, ndarray]:
    classes_list = positives + negatives

    label_mask = np.concatenate((np.full(len(positives), -1, dtype=int), np.full(len(negatives), -2, dtype=int)))
    if train_flag:
        permutation = rng.permutation(len(classes_list))
        classes_list = [classes_list[i] for i in permutation]
        label_mask = label_mask[permutation]

    result_prompt = np.concatenate([np.append(class_name, tokenizer.cls_token_id) for class_name in classes_list])
    extended_label_mask = np.concatenate(
        [
            np.append(np.full(len(class_name), i, dtype=int), mask)
            for i, (class_name, mask) in enumerate(zip(classes_list, label_mask))
        ]
    )

    result_prompt = np.concatenate(
        [
            [tokenizer.bos_token_id],
            result_prompt,
            [tokenizer.sep_token_id],
            input_seq,
            [tokenizer.eos_token_id],
        ]
    )

    extended_label_mask = np.concatenate(
        [
            np.array([-4]),
            extended_label_mask,
            np.array([-4]),
            np.full(len(input_seq), -3),
            np.array([-4]),
        ]
    )

    return result_prompt, extended_label_mask


def prepare_batch(result_prompts: list[ndarray], classes_masks: list[ndarray], tokenizer) -> tuple[torch.Tensor, ...]:
    batch_size = len(result_prompts)
    max_len = max(len(res_prompt) for res_prompt in result_prompts)
    if tokenizer.model_max_length is not None:
        model_max_length = tokenizer.model_max_length
        max_len = min(max_len, model_max_length)
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    classes_mask = torch.full((batch_size, max_len), -4, dtype=torch.long)

    for i in range(len(result_prompts)):
        result_prompt = result_prompts[i]
        label_mask = classes_masks[i]
        c_len = len(result_prompt)
        if model_max_length is not None:
            c_len = min(c_len, model_max_length)
            result_prompt = result_prompt[:c_len]
            label_mask = label_mask[:c_len]
            if result_prompt[-1] != tokenizer.eos_token_id:
                result_prompt[-1] = tokenizer.eos_token_id
                # -4 stands for pad tokens and bos/eos/sep token positions
                label_mask[-1] = -4
        input_ids[i, :c_len] = torch.tensor(result_prompt)
        attention_mask[i, :c_len] = 1
        classes_mask[i, :c_len] = torch.tensor(label_mask)

    return input_ids, attention_mask, classes_mask


def add_required_tokens(tokenizer: Tokenizer) -> Tokenizer:
    required_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
    }

    tokenizer.add_special_tokens(required_tokens)
    return tokenizer


def prepare_inference_batch(input_texts, tokenizer, classes):
    rng = np.random.default_rng()
    batch_size = len(input_texts)
    tokenized_texts = []
    for text in input_texts:
        tokenized_text = tokenizer(text, add_special_tokens=False).input_ids
        tokenized_texts.append(tokenized_text)
    tokenized_classes = [tokenizer(sample_class, add_special_tokens=False) for sample_class in classes]

    result_prompts = []
    label_masks = []

    for tokenized_text in tokenized_texts:
        result_prompt, label_mask = make_classifier_prompt(tokenized_text, tokenized_classes, tokenizer, rng)

        result_prompts.append(result_prompt)
        label_masks.append(label_mask)

    return prepare_batch(result_prompts, label_masks)

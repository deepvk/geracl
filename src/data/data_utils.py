import numpy as np
import torch
from numpy import ndarray


def make_classifier_prompt(
    input_seq: ndarray,
    special_token_ids: dict[int],
    classes_list: list[ndarray],
    positive_labels: list[list[int]] = None,
) -> tuple[ndarray, ndarray]:
    if positive_labels:
        label_mask = [-2] * len(classes_list)
        for i in positive_labels:
            label_mask[i] = -1
        label_mask = np.array(label_mask, dtype=int)
    else:
        label_mask = np.full(len(classes_list), -1, dtype=int)

    result_prompt = np.concatenate(
        [np.append(class_name, special_token_ids["cls_token"]) for class_name in classes_list]
    )
    extended_label_mask = np.concatenate(
        [
            np.append(np.full(len(class_name), i, dtype=int), mask)
            for i, (class_name, mask) in enumerate(zip(classes_list, label_mask))
        ]
    )

    result_prompt = np.concatenate(
        [
            [special_token_ids["bos_token"]],
            result_prompt,
            [special_token_ids["sep_token"]],
            input_seq,
            [special_token_ids["eos_token"]],
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


def prepare_batch(
    result_prompts: list[ndarray],
    classes_masks: list[ndarray],
    pad_token_id: int,
    eos_token_id: int,
    model_max_length: int = None,
) -> tuple[torch.Tensor, ...]:
    batch_size = len(result_prompts)
    max_len = max(len(res_prompt) for res_prompt in result_prompts)
    max_len = min(max_len, 1100)
    if model_max_length is not None:
        model_max_length = model_max_length
        max_len = min(max_len, model_max_length)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    classes_mask = torch.full((batch_size, max_len), -4, dtype=torch.long)

    for i in range(len(result_prompts)):
        result_prompt = result_prompts[i]
        label_mask = classes_masks[i]
        c_len = len(result_prompt)
        c_len = min(c_len, 1100)
        if model_max_length is not None:
            c_len = min(c_len, model_max_length)
            result_prompt = result_prompt[:c_len]
            label_mask = label_mask[:c_len]
            if result_prompt[-1] != eos_token_id:
                result_prompt[-1] = eos_token_id
                # -4 stands for pad tokens and bos/eos/sep token positions
                label_mask[-1] = -4
        input_ids[i, :c_len] = torch.tensor(result_prompt)
        attention_mask[i, :c_len] = 1
        classes_mask[i, :c_len] = torch.tensor(label_mask)

    return input_ids, attention_mask, classes_mask


def prepare_inference_batch(input_texts, classes, tokenizer):
    tokenized_texts = tokenizer(input_texts, add_special_tokens=False).input_ids
    tokenized_classes = [tokenizer(sample_classes, add_special_tokens=False).input_ids for sample_classes in classes]

    result_prompts = []
    label_masks = []
    classes_count = [len(sample_classes) for sample_classes in tokenized_classes]
    special_token_ids = {
        "bos_token": tokenizer.bos_token_id,
        "cls_token": tokenizer.cls_token_id,
        "sep_token": tokenizer.sep_token_id,
        "eos_token": tokenizer.eos_token_id,
    }

    for tokenized_text, tokenized_sample_classes in tokenized_texts:
        result_prompt, label_mask = make_classifier_prompt(tokenized_text, special_token_ids, tokenized_sample_classes)

        result_prompts.append(result_prompt)
        label_masks.append(label_mask)

    input_ids, attention_mask, classes_mask = prepare_batch(
        result_prompts,
        label_masks,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.model_max_length if tokenizer.model_max_length else None,
    )
    return input_ids, attention_mask, classes_mask, torch.tensor(classes_count)

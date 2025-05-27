import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from geracl.data.batch_creation import prepare_inference_batch
from geracl.model.hf_wrapper import GeraclHF
from geracl.utils import add_required_tokens


def print_macro_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average="macro")
    recall = recall_score(true_labels, predicted_labels, average="macro")
    f1 = f1_score(true_labels, predicted_labels, average="macro")

    print(f"accuracy={accuracy:.2f}\nprecision = {precision:.2f}\nRecall = {recall:.2f}\nF1 = {f1:.2f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


DATASETS = [
    # name,           split,  label_col,  class_source,                          batch, n_classes
    (
        "ai-forever/kinopoisk-sentiment-classification",
        "validation",
        "label",
        lambda dataset: ["негативный", "нейтральный", "позитивный"],
        20,
        3,
    ),
    (
        "ai-forever/headline-classification",
        "validation",
        "label",
        lambda dataset: ["спорт", "происшествия", "политика", "наука", "культура", "экономика"],
        100,
        6,
    ),
    (
        "ai-forever/ru-scibench-grnti-classification",
        "test",
        "label_text",
        lambda dataset: list(sorted(set(dataset["label_text"]))),
        21,
        28,
    ),
    (
        "ai-forever/ru-scibench-oecd-classification",
        "test",
        "label_text",
        lambda dataset: list(sorted(set(dataset["label_text"]))),
        29,
        29,
    ),
    (
        "ai-forever/inappropriateness-classification",
        "test",
        "label",
        lambda dataset: ["приличный", "неприличный"],
        50,
        2,
    ),
]


def permute_labels(rng, class_list):
    perm = rng.permutation(len(class_list))
    return [class_list[i] for i in perm], perm


def compute_real_preds(logits, n_classes):
    """
    logits: (batch*n_classes,) 1-D tensor that the current script appends
    """
    return torch.argmax(logits.view(-1, n_classes), dim=1).tolist()


@torch.inference_mode()
def evaluate_one_dataset(
    model, tokenizer, name, split, label_col, class_fn, batch_size, n_classes, input_prompt, device="cuda"
):
    dataset = load_dataset(name)[split]
    class_set = class_fn(dataset)

    rng = np.random.default_rng()
    input_classes, target_labels = [], []

    for gold in dataset[label_col]:
        if isinstance(gold, float):
            gold = int(gold)
        permuted, perm = permute_labels(rng, class_set)
        input_classes.append(permuted)
        # gold may be int idx or text; handle both
        gold_idx = gold if isinstance(gold, int) else class_set.index(gold)
        target_labels.append(perm.tolist().index(gold_idx))

    preds, texts = [], dataset["text"]
    for i in tqdm(range(0, len(texts), batch_size), desc=name):
        if i + batch_size < len(texts):
            batch = prepare_inference_batch(
                texts[i : i + batch_size], input_classes[i : i + batch_size], tokenizer, input_prompt=input_prompt
            )
        else:
            batch = prepare_inference_batch(texts[i:], input_classes[i:], tokenizer, input_prompt=input_prompt)
        input_ids, attention_mask, classes_mask, classes_count = [x.to(device) for x in batch]
        logits = model(input_ids, attention_mask, classes_mask, classes_count)
        preds.extend(compute_real_preds(logits.cpu(), n_classes))

    # de-permute back to original label space
    real_targets, real_preds = [], []
    for classes, gold_result, pred_result in zip(input_classes, target_labels, preds):
        perm = [class_set.index(c) for c in classes]
        real_targets.append(perm[gold_result])
        real_preds.append(perm[pred_result])

    return real_targets, real_preds


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a model and dump metrics.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model on HuggingFace.")
    parser.add_argument(
        "--metrics_path",
        type=str,
        required=False,
        default="metrics.json",
        help="Output path where evaluation metrics json file will be written. Default: 'metrics.json'.",
    )
    parser.add_argument(
        "--input_prompt", type=str, required=False, default=None, help="Input prompt to the model. Default: None."
    )
    return parser.parse_args()


def run_mteb_evaluation(model, tokenizer, metrics_path, input_prompt):
    out = dict()
    for dataset_name, split, label_col, class_fn, bs, classes_num in DATASETS:
        targets, preds = evaluate_one_dataset(
            model, tokenizer, dataset_name, split, label_col, class_fn, bs, classes_num, input_prompt
        )
        out[dataset_name] = print_macro_metrics(targets, preds)
    json.dump(out, open(metrics_path, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model_path
    metrics_path = args.metrics_path
    input_prompt = args.input_prompt

    model = GeraclHF.from_pretrained(model_path).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer = add_required_tokens(tokenizer)
    metrics = dict()
    metrics["model_path"] = model_path

    run_mteb_evaluation(model, tokenizer, metrics_path, input_prompt)

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

    # toxic_dataset = load_dataset('parquet', data_files={'train': '/data/toxic_dataset/train.parquet', 'test': '/data/toxic_dataset/test.parquet'})
    # features = ['toxic', 'insults', 'violence', 'suicide', 'harassment']
    # num_labels = 5

    # # Dictionary to hold binary classification datasets for each label
    # binary_test_datasets = {}

    # # Create a separate dataset for each label (0 to 4)
    # for label in range(num_labels):
    #     # Use a lambda function to create the binary label.
    #     # The default parameter `label=label` ensures that the lambda "captures" the current label.
    #     binary_dataset = toxic_dataset['test'].map(lambda x, label=label: {
    #         "text": x["text"],
    #         "binary_label": 1 if label in x["labels"] else 0
    #     })
    #     binary_test_datasets[f"binary_{features[label]}"] = binary_dataset

    # ru_features = ['токсичность', 'оскорбление', 'насилие', 'суицид', 'домогательство']
    # for name, ds in binary_test_datasets.items():
    #     # Remove the 'labels' column if it exists
    #     if 'labels' in ds.column_names:
    #         binary_test_datasets[name] = ds.remove_columns("labels")

    # answers = dict()
    # for j in range(5):
    #     dataset = binary_test_datasets[f"binary_{features[j]}"]
    #     input_classes_set = ['нейтральный', ru_features[j]]
    #     #input_classes_set = ['нейтральный', f"{ru_features[j]} {toxic_class_descriptions[ru_features[j]]}"]
    #     input_classes = [input_classes_set for _ in range(len(dataset))]

    #     input_texts = dataset['text']
    #     batch = prepare_inference_batch(input_texts, input_classes, tokenizer, input_prompt="classification: ")
    #     input_ids, attention_mask, classes_mask, classes_count = batch

    #     predictions = []
    #     for i in tqdm(range(len(input_ids) // 50)):
    #         input_ids_batch = input_ids[(i * 50): ((i + 1) * 50)].to('cuda')
    #         attention_mask_batch = attention_mask[(i * 50): ((i + 1) * 50)].to('cuda')
    #         classes_mask_batch = classes_mask[(i * 50): ((i + 1) * 50)].to('cuda')
    #         classes_count_batch = classes_count[(i * 50): ((i + 1) * 50)].to('cuda')
    #         #for i in range(10):
    #         #    print(tokenizer.decode(input_ids_batch[i]).strip('<pad>'))
    #         with torch.no_grad():
    #             y_hat = model(input_ids_batch, attention_mask_batch, classes_mask_batch, classes_count_batch)
    #             predictions.append(y_hat)

    #     real_predictions = []
    #     for i in range(len(predictions)):
    #         for k in range(len(predictions[i]) // 2):
    #             pred = torch.argmax(predictions[i][k*2: (k * 2) + 2])
    #             #print(predictions[i][k*6: (k * 6) + 6])
    #             real_predictions.append(pred.tolist())
    #     answers[features[j]] = real_predictions

    #     print(ru_features[j])
    #     #metrics[ru_features[j]] = print_binary_metrics(dataset['binary_label'], real_predictions)
    #     print('-----------')

    # y_true_tensor = torch.zeros(len(toxic_dataset['test']), 5, dtype=torch.long)
    # y_true_labels = toxic_dataset['test']['labels']
    # for i, labels in enumerate(y_true_labels):
    #     y_true_tensor[i, labels] = 1

    # y_pred = np.column_stack((answers[features[0]], answers[features[1]], answers[features[2]], answers[features[3]], answers[features[4]]))
    # y_true = y_true_tensor.cpu()
    # exact_match_accuracy = accuracy_score(y_true, y_pred)
    # print("Exact Match Accuracy:", exact_match_accuracy)

    # # F1 Score (micro averaged across all labels)
    # f1_micro = f1_score(y_true, y_pred, average='micro')
    # print("Micro F1 Score:", f1_micro)

    # # Alternatively, macro averaged F1 Score
    # f1_macro = f1_score(y_true, y_pred, average='macro')
    # print("Macro F1 Score:", f1_macro)

    # # Get a detailed report
    # report = classification_report(y_true, y_pred)
    # print("Classification Report:\n", report)

    # # Get multilabel confusion matrices (one for each label)
    # mcm = multilabel_confusion_matrix(y_true, y_pred)
    # print("Multilabel Confusion Matrices:\n", mcm)

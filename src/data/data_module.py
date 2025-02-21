from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from loguru import logger
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.data_processing import (
    generate_classes,
    generate_classes_llm_negatives,
    generate_classes_task_creation,
    shuffle_classes,
)
from src.data.data_utils import make_classifier_prompt, prepare_batch
from src.data.dataset import ZeroShotClassificationDataset
from src.utils import add_required_tokens


class ZeroShotClassificationDataModule(LightningDataModule):
    """Lightning data module for data handling.
    Provides dataloader for all splits, e.g. `train_dataloader` method.
    Public methods and attributes allow to retrieve information about texts and their classes.

    And `tokenizers` to tokenize data, e.g. `deepvk/USER-base`
    """

    def __init__(
        self,
        batch_size: int = 16,
        val_batch_size: int = 16,
        num_workers: int = 20,
        tokenizer_name: str = "deepvk/USER-base",
        config: str = "task_creation_negatives",
        model_max_length: int = None,
    ):
        """Data module constructor.

        :param batch_size: train batch size;
        :param val_batch_size: validation and test batch size;
        :param num_workers: Number of workers for data loaders;
        :param tokenizer_name: name of the tokenizer, "deepvk/USER-base" by default.
        """
        super().__init__()
        if config not in {"multiclass", "multilabel", "llm_negatives", "task_creation_negatives"}:
            raise ValueError("Invalid DataModule config parameter.")
        self._config = config
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = num_workers
        self._tokenizer_name = tokenizer_name
        logger.info(f"Downloading and opening '{self._tokenizer_name}' tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        self._tokenizer = add_required_tokens(self._tokenizer)
        if model_max_length is not None:
            self._tokenizer.model_max_length = model_max_length
        self._special_token_ids = {
            "bos_token": self._tokenizer.bos_token_id,
            "cls_token": self._tokenizer.cls_token_id,
            "sep_token": self._tokenizer.sep_token_id,
            "eos_token": self._tokenizer.eos_token_id,
        }

    def setup(self, stage: Optional[str] = None):
        logger.info("Downloading and opening 'deepvk/synthetic_classes' dataset")
        data = load_dataset("deepvk/synthetic-classes", self._config)
        train_data = load_dataset("deepvk/synthetic-classes", "task_creation_negatives_train", split="train")

        self._datasets = {}
        self._labels = {}

        self._datasets["train"] = train_data

        for split in ["validation", "test"]:
            logger.info(f"Initializing {split} dataset")
            self._datasets[split] = ZeroShotClassificationDataset(data[split], self._tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self._datasets["train"],
            batch_size=self._batch_size,
            collate_fn=self._task_creation_train_collate_fn,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._datasets["validation"],
            batch_size=self._val_batch_size,
            collate_fn=self._val_test_collate_fn,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._datasets["test"],
            batch_size=self._val_batch_size,
            collate_fn=self._val_test_collate_fn,
            num_workers=self._num_workers,
        )

    def _task_creation_train_collate_fn(self, samples) -> Tuple[torch.Tensor, ...]:
        new_classes = generate_classes_task_creation(samples)

        positive_classes = [[sample_classes[0]] for sample_classes in new_classes]
        new_classes, positive_labels = shuffle_classes(new_classes, positive_classes)
        new_samples = [(samples[i]["text"], new_classes[i], positive_labels[i]) for i in range(len(samples))]
        texts = [sample_text for (sample_text, _, _) in new_samples]
        tokenized_texts = self._tokenizer(texts, add_special_tokens=False).input_ids
        tokenized_classes = [
            self._tokenizer(sample_classes, add_special_tokens=False).input_ids
            for (_, sample_classes, _) in new_samples
        ]

        new_samples = [(tokenized_texts[i], tokenized_classes[i], positive_labels[i]) for i in range(len(samples))]

        return self._val_test_collate_fn(new_samples)

    def _llm_negatives_train_collate_fn(self, samples) -> Tuple[torch.Tensor, ...]:
        positive_classes = [sample["classes"] for sample in samples]
        llm_negatives = [self._train_negatives["negatives"].loc[sample["idx"]] for sample in samples]
        new_classes = generate_classes_llm_negatives(llm_negatives, positive_classes)

        new_classes, positive_labels = shuffle_classes(new_classes, positive_classes)
        new_samples = [(samples[i]["text"], new_classes[i], positive_labels[i]) for i in range(len(samples))]

        texts = [sample_text for (sample_text, _, _) in new_samples]
        tokenized_texts = self._tokenizer(texts, add_special_tokens=False).input_ids
        tokenized_classes = [
            self._tokenizer(sample_classes, add_special_tokens=False).input_ids
            for (_, sample_classes, _) in new_samples
        ]

        new_samples = [(tokenized_texts[i], tokenized_classes[i], positive_labels[i]) for i in range(len(samples))]

        return self._val_test_collate_fn(new_samples)

    def _train_collate_fn(self, samples: list[tuple[ndarray, List[ndarray]]]) -> Tuple[torch.Tensor, ...]:
        classes = [sample["classes"] for sample in samples]
        new_classes, positive_labels = generate_classes(classes, self._config)
        new_samples = [(samples[i]["text"], new_classes[i], positive_labels[i]) for i in range(len(samples))]

        texts = [sample_text for (sample_text, _, _) in new_samples]
        tokenized_texts = self._tokenizer(texts, add_special_tokens=False).input_ids
        tokenized_classes = [
            self._tokenizer(sample_classes, add_special_tokens=False).input_ids
            for (_, sample_classes, _) in new_samples
        ]

        new_samples = [(tokenized_texts[i], tokenized_classes[i], positive_labels[i]) for i in range(len(samples))]

        return self._val_test_collate_fn(new_samples)

    def _val_test_collate_fn(self, samples: list[tuple[ndarray, list[ndarray], list[int]]]) -> tuple[torch.Tensor, ...]:
        result_prompts = []
        label_masks = []
        classes_count = [len(sample_classes) for (_, sample_classes, _) in samples]
        positive_labels = [sample_positive_labels for (_, _, sample_positive_labels) in samples]

        for input_seq, sample_classes, sample_positive_labels in samples:
            result_prompt, label_mask = make_classifier_prompt(
                input_seq, self._special_token_ids, sample_classes, sample_positive_labels
            )

            result_prompts.append(result_prompt)
            label_masks.append(label_mask)

        input_ids, attention_mask, classes_mask = prepare_batch(
            result_prompts,
            label_masks,
            self._tokenizer.pad_token_id,
            self._tokenizer.eos_token_id,
            self._tokenizer.model_max_length if self._tokenizer.model_max_length else None,
        )
        return input_ids, attention_mask, classes_mask, torch.tensor(classes_count), positive_labels

    @property
    def tokenizer(self) -> Tokenizer:
        """Return current tokenizer instance."""
        return self._tokenizer

import json
from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from geracl.data.batch_creation import make_classifier_prompt, prepare_batch
from geracl.data.data_processing import choose_synthetic_classes, generate_classes, shuffle_classes
from geracl.data.dataset import ZeroShotClassificationDataset
from geracl.utils import add_required_tokens


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
        config: str = "synthetic_positives_multiclass",
        model_max_length: int = None,
        include_scenarios: bool = False,
        input_prompt: str = None,
    ):
        """Data module constructor.

        :param batch_size: train batch size;
        :param val_batch_size: validation and test batch size;
        :param num_workers: Number of workers for data loaders;
        :param tokenizer_name: name of the tokenizer, "deepvk/USER-base" by default;
        :param config: config of the CLAZER HuggingFace dataset;
        :param model_max_length: Maximum sequence length of the embedder model;
        :param include_scenarios: whether to include classification scenarios in the prompt or not.
          Works only when config = "synthetic_classes";
        :param input_prompt: Input prompt to the embedder model, e.g. "classification: ".
        """
        super().__init__()
        if config not in {
            "synthetic_positives_multiclass",
            "synthetic_positives_multilabel",
            "synthetic_classes",
            "ru_mteb_classes",
            "ru_mteb_extended_classes",
            "real_world_extended_expanded",
        }:
            raise ValueError("Invalid DataModule config parameter.")
        if include_scenarios and config != "synthetic_classes":
            raise ValueError(
                "Invalid DataModule include_scenarios parameter. It can be true only with the 'synthetic_classes' dataset config."
            )
        self._config = config
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._num_workers = num_workers
        self._include_scenarios = include_scenarios
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
        self._val_test_collate_fn = partial(
            self._generic_val_test_collate_fn, include_scenarios=include_scenarios, input_prompt=input_prompt
        )

        if self._config == "synthetic_classes":
            self._train_collate_fn = self._synthetic_classes_train_collate_fn
        elif self._config in {"synthetic_positives_multiclass", "synthetic_positives_multilabel"}:
            self._train_collate_fn = self._synthetic_positives_train_collate_fn
        elif self._config == "real_world_extended_expanded":
            self._train_collate_fn = self._extended_expanded_train_collate_fn
        else:
            self._train_collate_fn = self._val_test_collate_fn

    def setup(self, stage: Optional[str] = None):
        logger.info("Downloading and opening 'deepvk/synthetic-classes' dataset")

        data = load_dataset("deepvk/synthetic-classes", self._config)

        if self._config == "synthetic_classes":
            train_data = load_dataset("deepvk/synthetic-classes", "synthetic_classes_train", split="train")
        elif self._config in {"synthetic_positives_multiclass", "synthetic_positives_multilabel"}:
            train_data = load_dataset("deepvk/synthetic-classes", "synthetic_positives", split="train")

        self._datasets = {}
        self._labels = {}
        if self._config not in {
            "synthetic_classes",
            "synthetic_positives_multiclass",
            "synthetic_positives_multilabel",
        }:
            splits = ["train", "validation", "test"]
        else:
            splits = ["validation", "test"]
            logger.info("Initializing train dataset")
            self._datasets["train"] = train_data

        for split in splits:
            logger.info(f"Initializing {split} dataset")
            self._datasets[split] = ZeroShotClassificationDataset(data[split], self._tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self._datasets["train"],
            batch_size=self._batch_size,
            collate_fn=self._train_collate_fn,
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

    def _synthetic_classes_train_collate_fn(self, samples) -> Tuple[torch.Tensor, ...]:
        if self._include_scenarios:
            classes, scenarios = choose_synthetic_classes(samples, include_scenarios=self._include_scenarios)
            for i in range(len(scenarios)):
                scenarios[i] = scenarios[i] + ": "
        else:
            classes = choose_synthetic_classes(samples, include_scenarios=self._include_scenarios)

        positive_classes = [[sample_classes[0]] for sample_classes in classes]
        new_classes, positive_labels = shuffle_classes(classes, positive_classes)
        new_samples = [(samples[i]["text"], new_classes[i], positive_labels[i]) for i in range(len(samples))]
        texts = [sample_text for (sample_text, _, _) in new_samples]
        tokenized_texts = self._tokenizer(texts, add_special_tokens=False).input_ids
        tokenized_classes = [
            self._tokenizer(sample_classes, add_special_tokens=False).input_ids
            for (_, sample_classes, _) in new_samples
        ]
        if self._include_scenarios:
            tokenized_scenarios = self._tokenizer(scenarios, add_special_tokens=False).input_ids
            new_samples = [
                (tokenized_texts[i], tokenized_classes[i], positive_labels[i], tokenized_scenarios[i])
                for i in range(len(samples))
            ]
        else:
            new_samples = [(tokenized_texts[i], tokenized_classes[i], positive_labels[i]) for i in range(len(samples))]

        return self._val_test_collate_fn(new_samples)

    def _synthetic_positives_train_collate_fn(
        self, samples: list[tuple[ndarray, List[ndarray]]]
    ) -> Tuple[torch.Tensor, ...]:
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

    def _generic_val_test_collate_fn(
        self, samples: list[tuple[ndarray, list[ndarray], list[int]]], input_prompt=None, include_scenarios=None
    ) -> tuple[torch.Tensor, ...]:
        result_prompts = []
        label_masks = []
        if len(input_prompt) != 0:
            tokenized_prompt = self._tokenizer(input_prompt, add_special_tokens=False).input_ids
        else:
            tokenized_prompt = input_prompt
        if len(samples[0]) == 3:
            classes_count = [len(sample_classes) for (_, sample_classes, _) in samples]
            positive_labels = [sample_positive_labels for (_, _, sample_positive_labels) in samples]
            for input_seq, sample_classes, sample_positive_labels in samples:
                result_prompt, label_mask = make_classifier_prompt(
                    input_seq,
                    self._special_token_ids,
                    sample_classes,
                    starting_prompt=tokenized_prompt,
                    positive_labels=sample_positive_labels,
                )

                result_prompts.append(result_prompt)
                label_masks.append(label_mask)
        else:
            classes_count = [len(sample_classes) for (_, sample_classes, _, _) in samples]
            positive_labels = [sample_positive_labels for (_, _, sample_positive_labels, _) in samples]
            for input_seq, sample_classes, sample_positive_labels, sample_scenario in samples:
                if not include_scenarios:
                    sample_scenario = np.array([], dtype=int)
                result_prompt, label_mask = make_classifier_prompt(
                    input_seq,
                    self._special_token_ids,
                    sample_classes,
                    starting_prompt=tokenized_prompt,
                    scenario=sample_scenario,
                    positive_labels=sample_positive_labels,
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

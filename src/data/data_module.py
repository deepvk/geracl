import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from numpy import ndarray
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import ZeroShotClassificationDataset
from src.utils import add_required_tokens, make_classifier_prompt, prepare_batch

_logger = logging.getLogger(__name__)


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
        tokenizer_name: str = "deepvk/USER-base",
    ):
        """Data module constructor.

        :param batch_size: train batch size;
        :param val_batch_size: validation and test batch size;
        :param tokenizer_name: name of the tokenizer, "deepvk/USER-base" by default.
        """
        super().__init__()
        self._batch_size = batch_size
        self._val_batch_size = val_batch_size
        self._tokenizer_name = tokenizer_name
        _logger.info(f"Downloading and opening '{self._tokenizer_name}' tokenizer")
        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        self._tokenizer = add_required_tokens(self._tokenizer)

    def setup(self, stage: Optional[str] = None):
        _logger.info(f"Downloading and opening 'llama_annotations_ru_c4' dataset")
        # data = load_dataset('MikhailVyrodov/llama_annotations_ru_c4')
        train_data = load_dataset("json", data_files="/data/c4_annotations_dataset/first_train.jsonl")
        val_data = load_dataset("json", data_files="/data/c4_annotations_dataset/first_val.jsonl")
        test_data = load_dataset("json", data_files="/data/c4_annotations_dataset/first_test.jsonl")
        data = {
            "train": train_data["train"],
            "validation": val_data["train"],
            "test": test_data["train"],
        }
        self._rng = np.random.default_rng()

        self._datasets = {}
        for split in ["train", "validation", "test"]:
            _logger.info(f"Initializing {split} dataset")
            self._datasets[split] = ZeroShotClassificationDataset(data[split], self._tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self._datasets["train"],
            batch_size=self._batch_size,
            collate_fn=self._collate_fn,
            num_workers=20,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._datasets["validation"],
            batch_size=self._val_batch_size,
            collate_fn=self._collate_fn,
            num_workers=20,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self._datasets["test"],
            batch_size=self._val_batch_size,
            collate_fn=self._collate_fn,
            num_workers=20,
        )

    def _collate_fn(self, samples: list[tuple[ndarray, List[ndarray]]]) -> Tuple[torch.Tensor, ...]:
        result_prompts = []
        label_masks = []

        for input_seq, sample_classes in samples:
            positives = [sample_classes[0]]
            negatives = sample_classes[1:]
            result_prompt, label_mask = make_classifier_prompt(
                input_seq,
                positives,
                self._tokenizer,
                self._rng,
                negatives,
                train_flag=True,
            )

            result_prompts.append(result_prompt)
            label_masks.append(label_mask)

        return prepare_batch(result_prompts, label_masks, self._tokenizer)

    @property
    def tokenizer(self) -> Tokenizer:
        """Return current tokenizer instance."""
        return self._tokenizer

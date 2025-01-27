import logging

from datasets import Dataset
from numpy import ndarray
from tokenizers import Tokenizer
from torch.utils.data import Dataset

_logger = logging.getLogger(__name__)


SAMPLE = tuple[ndarray, list[ndarray]]


class ZeroShotClassificationDataset(Dataset):
    """Dataset for zero-shot classification task.
    It uses texts with one positive class and 2-4 negative classes.
    """

    def __init__(self, data: dict[str, list[str]], tokenizer: Tokenizer):
        """Dataset constructor.

        :param data: Dictionary with input texts and list of their positive and negative classes.
                     Positive class is always first in the list.
        :param tokenizer: Tokenizer that used to tokenize text.
        """
        _logger.info(f"Initializing dataset")

        self._dataset = data
        self._tokenizer = tokenizer

    def tokenize(self, sample):
        text = sample["text"]
        classes = sample["classes"]

        encoded_text = self._tokenizer(text, add_special_tokens=False).input_ids
        encoded_classes = [
            self._tokenizer(sample_class, add_special_tokens=False).input_ids for sample_class in classes
        ]

        return encoded_text, encoded_classes

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        return self.tokenize(sample)

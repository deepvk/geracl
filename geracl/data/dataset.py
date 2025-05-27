from datasets import Dataset
from loguru import logger
from numpy import ndarray
from tokenizers import Tokenizer
from torch.utils.data import Dataset

SAMPLE = tuple[ndarray, list[ndarray]]


class ZeroShotClassificationDataset(Dataset):
    """Dataset for zero-shot classification task.
    It uses either texts with one positive class and 2-4 negative classes
    or texts with 2-3 positive classes and 2-3 negative classes.
    """

    def __init__(self, data: dict[str, list[str], list[int]], tokenizer: Tokenizer):
        """Dataset constructor.

        :param data: Dictionary with input texts and list of their positive and negative classes.
                     Positive class is always first in the list.
        :param tokenizer: Tokenizer that used to tokenize text.
        """
        logger.info("Initializing dataset")

        self._dataset = data
        self._tokenizer = tokenizer

    def tokenize(self, sample):
        text = sample["text"]
        classes = sample["classes"]
        label = sample["labels"]
        if isinstance(label, int):
            label = [label]
        scenario = None
        if "scenarios" in sample:
            scenario = sample["scenarios"]

        encoded_text = self._tokenizer(text, add_special_tokens=False).input_ids
        encoded_classes = self._tokenizer(classes, add_special_tokens=False).input_ids
        if scenario is not None:
            scenario = scenario + ": "
            encoded_scenario = self._tokenizer(scenario, add_special_tokens=False).input_ids
            return encoded_text, encoded_classes, label, encoded_scenario
        else:
            return encoded_text, encoded_classes, label

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        return self.tokenize(sample)

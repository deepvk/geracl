import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from geracl.data.batch_creation import prepare_inference_batch


class ZeroShotClassificationPipeline:
    def __init__(self, model, tokenizer, device="cuda", progress_bar=True, input_prompt=None):
        self.model = model
        if isinstance(tokenizer, str):
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self._tokenizer = tokenizer
        self.progress_bar = progress_bar

        if not isinstance(device, torch.device):
            if torch.cuda.is_available() and "cuda" in device:
                self.device = torch.device(device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        if self.model.device != self.device:
            self.model.to(self.device)

        self._rng = np.random.default_rng()
        self.input_prompt = input_prompt

    @torch.no_grad()
    def get_similarities(self, texts, labels, same_labels=True, batch_size=100):
        if isinstance(texts, str):
            texts = [texts]

        results = []
        if same_labels:
            labels = [labels for _ in range(batch_size)]

        iterable = range(0, len(texts), batch_size)
        if self.progress_bar:
            iterable = tqdm(iterable)

        for idx in iterable:
            batch_texts = texts[idx : idx + batch_size]

            if same_labels:
                tokenized_inputs = prepare_inference_batch(
                    batch_texts, labels, self._tokenizer, input_prompt=self.input_prompt
                )
            else:
                tokenized_inputs = prepare_inference_batch(
                    batch_texts, labels[idx : idx + batch_size], self._tokenizer, input_prompt=self.input_prompt
                )
            input_ids, attention_mask, classes_mask, classes_count = [x.to(self.device) for x in tokenized_inputs]
            similarities = self.model(input_ids, attention_mask, classes_mask, classes_count)
            results.append(similarities)

        return results

    @torch.no_grad()
    def __call__(self, texts, labels, batch_size=100):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(labels[0], str):
            same_labels = True
        else:
            same_labels = False

        similarities = self.get_similarities(texts, labels, same_labels, batch_size)

        real_predictions = []
        for i in range(len(similarities)):
            if same_labels:
                real_predictions.extend(torch.argmax(similarities[i].view(-1, len(labels)), dim=1).tolist())
            else:
                label_idx = 0
                for k in range(batch_size):
                    if label_idx + len(labels[(i * batch_size) + k]) > len(similarities[i]):
                        break
                    pred = torch.argmax(similarities[i][label_idx : label_idx + len(labels[(i * batch_size) + k])])
                    label_idx += len(labels[(i * batch_size) + k])
                    real_predictions.append(pred.tolist())

        return real_predictions

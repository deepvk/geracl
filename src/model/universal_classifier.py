import logging
from itertools import chain
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from tokenizers import Tokenizer
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from transformers import AutoModel

_logger = logging.getLogger(__name__)


class ZeroShotClassifier(LightningModule):
    """Lightning module that encapsulate all routines for zero-shot text classification.

    Maybe used as regular Torch module on inference: forward pass returns predicted classes.
    Also support training via lightning Trainer, see: https://pytorch-lightning.readthedocs.io/en/stable/.

    Use HuggingFace models as backbone to embed tokens, e.g. "USER-base":
    https://huggingface.co/deepvk/USER-base
    Reports binary cross-entropy loss, binary F1-score and binary AUROC during training.
    """

    def __init__(
        self,
        embedder_name: str = "deepvk/USER-base",
        *,
        unfreeze_embedder: bool = False,
        ffn_dim: int = 2048,
        ffn_dropout: float = 0.1,
        device: str = "cuda",
        tokenizer: Tokenizer,
    ):
        """
        :param embedder_name: name of pretrained HuggingFace model to embed tokens.
        :param unfreeze_embedder: if `True` ten train top classifier and backbone module.
        """
        super().__init__()

        self._device = device
        self._token_embedder = AutoModel.from_pretrained(embedder_name).to(self._device)
        self._token_embedder.resize_token_embeddings(len(tokenizer))
        self._mlp = nn.Sequential(
            nn.Linear(self._token_embedder.config.hidden_size, ffn_dim),
            nn.Dropout(ffn_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, self._token_embedder.config.hidden_size),
        ).to(self._device)

        self._validation_step_outputs = {"predictions": [], "target": [], "loss": []}
        self._test_step_outputs = {"predictions": [], "target": [], "loss": []}
        self._training_step_outputs = {"loss": []}
        self._tokenizer = tokenizer

        if not unfreeze_embedder:
            _logger.info(f"Freezing embedding model: {self._token_embedder.__class__.__name__}")
            for param in self._token_embedder.parameters():
                param.requires_grad = False

        self._auroc_metric = MetricCollection({f"{split}_auroc": BinaryAUROC() for split in ["train", "val", "test"]})

        self._f1_metric = MetricCollection(
            {f"{split}_f1": BinaryF1Score(threshold=0.1) for split in ["train", "val", "test"]}
        )

    def forward(self, input_ids: Tensor, attention_mask: Tensor, classes_mask: Tensor) -> Tensor:  # type: ignore
        """Forward pass of zero-shot classification model.
        Could be used during inference to classify text.

        :param input_ids: [batch size; seq len] -- batch with pretokenized texts.
        :param attention_mask: [batch size; seq len] -- attention mask with 0 for padding tokens.
        :return: [batch size; seq len] -- .
        """
        bs, seq_len = attention_mask.shape

        mlp_outputs, text_embeddings, classes_count, _ = self._get_embeddings(
            input_ids, attention_mask, classes_mask, train_flag=False
        )

        cur_pos = 0
        predictions = torch.zeros((bs))
        for i in range(bs):
            classes_embeddings = mlp_outputs[cur_pos : (cur_pos + classes_count[i])]
            cur_pos += classes_count[i]
            similarities = torch.matmul(classes_embeddings, text_embeddings[i].unsqueeze(-1))
            predictions[i] = similarities.argmax()

        return predictions

    def configure_optimizers(self, optimizer_cls=torch.optim.AdamW, scheduler_cls=None):
        """
        :param optimizer_cls: PyTorch optimizer class, e.g. `torch.optim.AdamW`.
        :param scheduler_cls: PyTorch scheduler class, e.g. `torch.optim.lr_scheduler.LambdaLR`.
                                If `None`, then constant lr.
        """
        parameters = chain(self._token_embedder.parameters(), self._mlp.parameters())
        optimizer = optimizer_cls(parameters)
        if scheduler_cls is None:
            return optimizer
        lr_scheduler = scheduler_cls(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def _get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        classes_mask: torch.Tensor,
        train_flag: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        bs, seq_len = attention_mask.shape

        embeddings = self._token_embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # -3 stands for input text tokens in classes_mask
        mask = classes_mask == -3
        masked_values = embeddings * mask.unsqueeze(-1)
        num_filtered = mask.sum(dim=1)
        text_embeddings = masked_values.sum(dim=1) / num_filtered.unsqueeze(-1)

        mlp_input = []
        classes_count = []
        if train_flag:
            positive_classes = []
        for i in range(bs):
            sample_classes_mask = classes_mask[i]
            emb = embeddings[i]

            positive_mask = sample_classes_mask >= 0
            positive_sample_classes_mask = sample_classes_mask[positive_mask]
            positive_embeddings = emb[positive_mask]

            unique_labels = positive_sample_classes_mask.unique()
            classes_count.append(len(unique_labels))
            positive_classes.append([])

            sample_class_embeddings = []
            for label in unique_labels:
                positions = torch.nonzero(sample_classes_mask == label, as_tuple=True)[0]
                start_idx = positions.min().item()
                end_idx = positions.max().item()

                class_mean = emb[start_idx : end_idx + 2].mean(dim=0)
                sample_class_embeddings.append(class_mean)

                if train_flag:
                    if sample_classes_mask[end_idx + 1] == -1:
                        positive_classes[-1].append(label)

            sample_class_embeddings = torch.stack(sample_class_embeddings, dim=0)
            mlp_input += sample_class_embeddings

        mlp_input = torch.stack(mlp_input)
        # [all_classes_in_batch, embedding_dim]
        mlp_outputs = self._mlp(mlp_input.to(self._device))

        if train_flag:
            return (
                mlp_outputs,
                text_embeddings,
                classes_count,
                positive_classes,
            )
        else:
            return mlp_outputs, text_embeddings, classes_count, None

    def shared_step(self, batch: Tuple[Tensor, ...], split: str) -> STEP_OUTPUT:
        """Shared step of them that used during training and evaluation.
        Make forward pass of the model, calculate loss and metric and log them.

        :param batch: Tuple of
            > input_ids [batch size; seq len] – input tokens ids padded to the same length;
            > attention_mask [batch size; seq len] – mask with padding description, 0 means PAD token;
            > classes_mask [batch size; seq len] - labels of each token.
        :param split: name of current split, one of `train`, `val`, or `test`.
        :return: loss on the current batch.
        """
        input_ids, attention_mask, classes_mask = batch
        bs, seq_len = attention_mask.shape

        (
            mlp_outputs,
            text_embeddings,
            classes_count,
            positive_classes,
        ) = self._get_embeddings(input_ids, attention_mask, classes_mask)

        # [all_classes_in_batch, embedding_dim]
        wide_text_embeddings = torch.zeros((mlp_outputs.shape[0], text_embeddings.shape[1])).to(self._device)
        idx = 0
        for i, sample_class_count in enumerate(classes_count):
            wide_text_embeddings[idx : (idx + sample_class_count), :] = text_embeddings[i]
            idx += sample_class_count

        similarities = torch.sum(mlp_outputs * wide_text_embeddings, dim=-1)

        target = torch.zeros(similarities.shape[0]).to(self._device)
        idx = 0
        for i, sample_class_count in enumerate(classes_count):
            for positive_class in positive_classes[i]:
                target[idx + positive_class] = 1
            idx += sample_class_count

        criterion = nn.BCEWithLogitsLoss()
        batch_loss = criterion(similarities, target)

        with torch.no_grad():
            if split == "train":
                self._training_step_outputs["loss"].append(batch_loss.item())
            elif split == "val":
                self._validation_step_outputs["loss"].append(batch_loss.item())
            else:
                self._test_step_outputs["loss"].append(batch_loss.item())

            probs = torch.sigmoid(similarities)
            batch_auroc = self._get_auroc_metric(split)(probs, target)
            batch_f1 = self._get_f1_metric(split)(probs, target)

        self.log_dict(
            {
                f"{split}/step_loss": batch_loss.item(),
                f"{split}/step_auroc": batch_auroc,
                f"{split}/step_f1": batch_f1,
            }
        )
        return batch_loss

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "train")

    def on_train_epoch_end(self):
        epoch_auroc = self._get_auroc_metric("train").compute()
        self._get_auroc_metric("train").reset()

        epoch_f1 = self._get_f1_metric("train").compute()
        self._get_f1_metric("train").reset()

        epoch_loss = torch.tensor(
            self._training_step_outputs["loss"],
            dtype=torch.float32,
            device=self._device,
        ).mean()
        self.log_dict(
            {
                "train/epoch_loss": epoch_loss,
                "train/epoch_auroc": epoch_auroc,
                "train/epoch_f1": epoch_f1,
            }
        )
        self._training_step_outputs.clear()
        self._training_step_outputs = {"loss": []}

    def on_validation_epoch_end(self):
        epoch_auroc = self._get_auroc_metric("val").compute()
        self._get_auroc_metric("val").reset()

        epoch_f1 = self._get_f1_metric("val").compute()
        self._get_f1_metric("val").reset()

        epoch_loss = torch.tensor(
            self._validation_step_outputs["loss"],
            dtype=torch.float32,
            device=self._device,
        ).mean()
        self.log_dict(
            {
                "val/epoch_loss": epoch_loss,
                "val/epoch_auroc": epoch_auroc,
                "val/epoch_f1": epoch_f1,
            }
        )
        self._validation_step_outputs.clear()
        self._validation_step_outputs = {"predictions": [], "target": [], "loss": []}

    def on_test_epoch_end(self):
        epoch_auroc = self._get_auroc_metric("test").compute()
        self._get_auroc_metric("test").reset()

        epoch_f1 = self._get_f1_metric("test").compute()
        self._get_f1_metric("test").reset()

        epoch_loss = torch.tensor(self._test_step_outputs["loss"], dtype=torch.float32, device=self._device).mean()
        self.log_dict(
            {
                "test/epoch_loss": epoch_loss,
                "test/epoch_auroc": epoch_auroc,
                "test/epoch_f1": epoch_f1,
            }
        )
        self._test_step_outputs.clear()
        self._test_step_outputs = {"predictions": [], "target": [], "loss": []}

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "test")

    def _get_auroc_metric(self, split: str) -> BinaryAUROC:
        return self._auroc_metric[f"{split}_auroc"]

    def _get_f1_metric(self, split: str) -> BinaryF1Score:
        return self._f1_metric[f"{split}_f1"]

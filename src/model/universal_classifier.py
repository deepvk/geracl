from itertools import chain
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from transformers import AutoModel, PreTrainedTokenizer


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
        tokenizer: PreTrainedTokenizer,
        optimizer_args: dict = None,
        scheduler_args: dict = None,
    ):
        """
        :param embedder_name: name of pretrained HuggingFace model to embed tokens.
        :param unfreeze_embedder: if `True` ten train top classifier and backbone module.
        :param ffn_dim: hidden dimension of mlp layer.
        :param ffn_dropout: dropout of mlp layer.
        :param device: name of device to train the model on.
        :param tokenizer: Tokenizer object of backbone model.
        :param optimizer_args: Dict with arguments to initalize optimizer.
        :param scheduler_args: Dict with arguments to initalize scheduler.
        """
        super().__init__()
        self.save_hyperparameters()

        self._device = device
        self._token_embedder = AutoModel.from_pretrained(embedder_name).to(self._device)
        self._token_embedder.resize_token_embeddings(len(tokenizer))
        self._mlp = nn.Sequential(
            nn.Linear(self._token_embedder.config.hidden_size, ffn_dim),
            nn.Dropout(ffn_dropout),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, self._token_embedder.config.hidden_size),
        ).to(self._device)

        self._step_outputs = {f"{split}_loss": [] for split in ["val", "test"]}
        self._tokenizer = tokenizer

        if not unfreeze_embedder:
            logger.info(f"Freezing embedding model: {self._token_embedder.__class__.__name__}")
            for param in self._token_embedder.parameters():
                param.requires_grad = False

        self._auroc_metric = MetricCollection({f"{split}_auroc": BinaryAUROC() for split in ["train", "val", "test"]})

        self._f1_metric = MetricCollection(
            {f"{split}_f1": BinaryF1Score(threshold=0.1) for split in ["train", "val", "test"]}
        )

        self._optimizer_args = optimizer_args if optimizer_args is not None else None
        self._scheduler_args = scheduler_args if scheduler_args is not None else None

    def configure_optimizers(self):
        """
        :param optimizer_cls: PyTorch optimizer class, e.g. `torch.optim.AdamW`.
        :param scheduler_cls: PyTorch scheduler class, e.g. `torch.optim.lr_scheduler.LambdaLR`.
                                If `None`, then constant lr.
        """
        parameters = chain(self._token_embedder.parameters(), self._mlp.parameters())

        if self._optimizer_args:
            module_name, class_name = self._optimizer_args["class_path"].rsplit(".", 1)
            optimizer_cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
            optimizer = optimizer_cls(parameters, **self._optimizer_args["init_params"])
        else:
            optimizer = torch.optim.AdamW(parameters)

        if self._scheduler_args is None:
            return optimizer

        module_name, class_name = self._scheduler_args["class_path"].rsplit(".", 1)
        scheduler_cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        lr_scheduler = scheduler_cls(optimizer, **self._scheduler_args["init_params"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def _get_text_embeddings(
        self,
        embeddings: torch.Tensor,
        classes_mask: torch.Tensor,
    ) -> torch.Tensor:
        # -3 stands for input text tokens in classes_mask
        mask = classes_mask == -3
        masked_values = embeddings * mask.unsqueeze(-1)
        num_filtered = mask.sum(dim=1)
        text_embeddings = masked_values.sum(dim=1) / num_filtered.unsqueeze(-1)  # [batch size; embed dim]
        return text_embeddings

    def _get_classes_embeddings(
        self, embeddings: torch.Tensor, classes_mask: torch.Tensor, classes_count: torch.Tensor
    ) -> torch.Tensor:
        bs, _ = classes_mask.shape

        mlp_input = []

        for i in range(bs):
            sample_classes_mask = classes_mask[i]
            emb = embeddings[i]

            sample_class_embeddings = []
            for label in range(classes_count[i]):
                positions = torch.nonzero(sample_classes_mask == label, as_tuple=True)[0]
                start_idx = positions.min().item()
                end_idx = positions.max().item()

                # +2 because we should include token on end_idx position and the [CLS] token after it
                class_mean = emb[start_idx : end_idx + 2].mean(dim=0)
                sample_class_embeddings.append(class_mean)

            sample_class_embeddings = torch.stack(
                sample_class_embeddings, dim=0
            )  # [classes count in sample; embed dim]
            mlp_input += sample_class_embeddings

        mlp_input = torch.stack(mlp_input)  # [all classes in batch; embed dim]
        classes_embeddings = self._mlp(mlp_input.to(self._device))  # [all classes in batch; embed dim]

        return classes_embeddings

    def forward(self, input_ids: Tensor, attention_mask: Tensor, classes_mask: Tensor, classes_count: Tensor) -> Tensor:  # type: ignore
        """Forward pass of zero-shot classification model.
        Could be used during inference to classify text.

        :param input_ids: [batch size; seq len] -- batch with pretokenized texts.
        :param attention_mask: [batch size; seq len] -- attention mask with 0 for padding tokens.
        :param classes_mask: [batch size; seq len] - labels of each token.
        :return: [all classes in batch] -- logits of classifier for each class in batch.
        """
        # [batch size; seq len; embed dim]
        embeddings = self._token_embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_embeddings = self._get_text_embeddings(embeddings, classes_mask)  # [batch size; embed dim]
        classes_embeddings = self._get_classes_embeddings(
            embeddings, classes_mask, classes_count
        )  # [all classes in batch; embed dim]

        # [all classes in batch; embed dim]
        wide_text_embeddings = torch.zeros((classes_embeddings.shape[0], text_embeddings.shape[1])).to(self._device)
        idx = 0
        for i, sample_class_count in enumerate(classes_count):
            wide_text_embeddings[idx : (idx + sample_class_count), :] = text_embeddings[i]
            idx += sample_class_count

        similarities = torch.sum(classes_embeddings * wide_text_embeddings, dim=-1)  # [all classes in batch]

        return similarities

    def shared_step(self, batch: Tuple[Tensor, ...], split: str) -> STEP_OUTPUT:
        """Shared step of them that used during training and evaluation.
        Make forward pass of the model, calculate loss and metric and log them.

        :param batch: Tuple of
            > input_ids [batch size; seq len] – input tokens ids padded to the same length;
            > attention_mask [batch size; seq len] – mask with padding description, 0 means PAD token;
            > classes_mask [batch size; seq len] - labels of each token;
            > classes_count [batch_size] - classes count for each sample.
            > positive_classes [batch_size] - indices of positive classes for each sample.
        :param split: name of current split, one of `train`, `val`, or `test`.
        :return: loss on the current batch.
        """

        input_ids, attention_mask, classes_mask, classes_count, positive_classes = batch

        forward_batch = input_ids, attention_mask, classes_mask, classes_count

        similarities = self.forward(*forward_batch)  # [all classes in batch]

        target = torch.zeros(similarities.shape[0]).to(self._device)  # [all classes in batch]
        idx = 0
        for i, sample_class_count in enumerate(classes_count):
            for positive_class in positive_classes[i]:
                target[idx + positive_class] = 1
            idx += sample_class_count

        batch_loss = binary_cross_entropy_with_logits(similarities, target)

        with torch.no_grad():
            if split != "train":
                self._step_outputs[f"{split}_loss"].append(batch_loss.item())

            probs = torch.sigmoid(similarities)
            batch_auroc = self._auroc_metric[f"{split}_auroc"](probs, target)
            batch_f1 = self._f1_metric[f"{split}_f1"](probs, target)

        if split == "train":
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

    def _report_metrics(self, split: str, loss: list = None):
        epoch_auroc = self._auroc_metric[f"{split}_auroc"].compute()
        self._auroc_metric[f"{split}_auroc"].reset()

        epoch_f1 = self._f1_metric[f"{split}_f1"].compute()
        self._f1_metric[f"{split}_f1"].reset()

        self.log_dict(
            {
                f"{split}/epoch_auroc": epoch_auroc,
                f"{split}/epoch_f1": epoch_f1,
            }
        )

        if loss:
            epoch_loss = torch.tensor(
                loss,
                dtype=torch.float32,
                device=self._device,
            ).mean()

            self.log(f"{split}/epoch_loss", epoch_loss)

    def on_train_epoch_end(self):
        self._report_metrics("train")

    def on_validation_epoch_end(self):
        self._report_metrics("val", self._step_outputs["val_loss"])

        self._step_outputs["val_loss"].clear()

    def on_test_epoch_end(self):
        self._report_metrics("test", self._step_outputs["test_loss"])

        self._step_outputs["test_loss"].clear()

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "test")

from functools import partial
from itertools import chain
from typing import Tuple

import torch
import torch.optim
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC, BinaryF1Score

from geracl.model.geracl_core import GeraclCore
from geracl.utils import cosine_lambda, focal_loss_with_logits, linear_lambda


class Geracl(LightningModule):
    """Lightning module that encapsulate all routines for zero-shot text classification.

    Maybe used as regular Torch module on inference: forward pass returns predicted classes.
    Also support training via lightning Trainer, see: https://pytorch-lightning.readthedocs.io/en/stable/.

    Use HuggingFace models as backbone to embed tokens, e.g. "USER-base":
    https://huggingface.co/deepvk/USER-base
    Reports accuracy, binary F1-score, binary AUROC and either BCE loss or focal loss during training.
    """

    def __init__(
        self,
        embedder_name: str = "deepvk/USER-base",
        *,
        unfreeze_embedder: bool = False,
        ffn_dim: int = 2048,
        ffn_classes_dropout: float = 0.1,
        ffn_text_dropout: float = 0.1,
        device: str = "cuda",
        tokenizer_len: int,
        pooling_type: str = "mean",
        loss_args: dict = None,
        optimizer_args: dict = None,
        scheduler_args: dict = None,
    ):
        """
        :param embedder_name: name of pretrained HuggingFace model to embed tokens.
        :param unfreeze_embedder: if `True` then train top mlp layers along with backbone module.
        :param ffn_dim: hidden dimension of mlp layers.
        :param ffn_classes_dropout: dropout of the mlp layer used for transforming input classes embeddings.
        :param ffn_text_dropout: dropout of the mlp layer used for transforming input text embedding.
        :param device: name of device to train the model on.
        :param tokenizer_len: sumber of tokens in tokenizer.
        :param pooling_type: sentence embedding's pooling type (either mean or first).
        :param loss_args: dict with arguments to choose appropriate loss function.
        :param optimizer_args: dict with arguments to initalize optimizer.
        :param scheduler_args: dict with arguments to initalize scheduler.
        """
        super().__init__()
        self.save_hyperparameters()

        self._optimizer_args = optimizer_args if optimizer_args is not None else None
        self._scheduler_args = scheduler_args if scheduler_args is not None else None
        self._loss_args = loss_args if loss_args is not None else None

        if self._loss_args["loss_type"] not in {"bce", "focal"}:
            raise ValueError("Invalid loss type config parameter.")

        self._device = device
        self._classification_core = GeraclCore(
            embedder_name=embedder_name,
            ffn_dim=ffn_dim,
            ffn_classes_dropout=ffn_classes_dropout,
            ffn_text_dropout=ffn_text_dropout,
            device=device,
            tokenizer_len=tokenizer_len,
            pooling_type=pooling_type,
            loss_args=loss_args,
        )

        if not unfreeze_embedder:
            logger.info(f"Freezing embedding model: {self._classification_core._token_embedder.__class__.__name__}")
            for param in self._classification_core._token_embedder.parameters():
                param.requires_grad = False

        self._step_outputs = {
            f"{split}_{metric}": []
            for split in ["val", "test", "train"]
            for metric in ["loss", "predictions", "target"]
        }

        self._auroc_metric = MetricCollection({f"{split}_auroc": BinaryAUROC() for split in ["train", "val", "test"]})

        self._f1_metric = MetricCollection(
            {f"{split}_binary_f1": BinaryF1Score(threshold=0.1) for split in ["train", "val", "test"]}
        )

    def configure_optimizers(self):
        parameters = chain(
            self._classification_core._token_embedder.parameters(),
            self._classification_core._mlp_classes.parameters(),
            self._classification_core._mlp_text.parameters(),
        )

        if self._optimizer_args:
            module_name, class_name = self._optimizer_args["class_path"].rsplit(".", 1)
            optimizer_cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
            if "init_params" in self._optimizer_args:
                optimizer = optimizer_cls(parameters, **self._optimizer_args["init_params"])
            else:
                optimizer = optimizer_cls(parameters)
        else:
            optimizer = torch.optim.AdamW(parameters)

        if self._scheduler_args is None:
            return optimizer

        if self._scheduler_args["scheduler"] == "linear":
            linear_lambda_with_total_steps = partial(
                linear_lambda,
                total_steps=self._scheduler_args["total_steps"],
                warmup_steps=self._scheduler_args["warmup_steps"],
            )
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda_with_total_steps)
        else:
            cosine_lambda_with_total_steps = partial(
                cosine_lambda,
                total_steps=self._scheduler_args["total_steps"],
                warmup_steps=self._scheduler_args["warmup_steps"],
            )
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lambda_with_total_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def forward(self, input_ids: Tensor, attention_mask: Tensor, classes_mask: Tensor, classes_count: Tensor) -> Tensor:  # type: ignore
        return self._classification_core.forward(input_ids, attention_mask, classes_mask, classes_count)

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
        bs = len(input_ids)

        forward_batch = input_ids, attention_mask, classes_mask, classes_count

        similarities = self.forward(*forward_batch)

        target = torch.zeros(similarities.shape[0]).to(self._device)
        idx = 0
        for i, sample_class_count in enumerate(classes_count):
            for positive_class in positive_classes[i]:
                target[idx + positive_class] = 1
            idx = idx + sample_class_count

        if self._loss_args["loss_type"] == "bce":
            batch_loss = binary_cross_entropy_with_logits(similarities, target)
        elif self._loss_args["loss_type"] == "focal":
            batch_loss = focal_loss_with_logits(
                similarities,
                target,
                alpha=self._loss_args["init_params"]["alpha"],
                gamma=self._loss_args["init_params"]["gamma"],
                label_smoothing=self._loss_args["init_params"]["label_smoothing"],
                ignore_index=self._loss_args["init_params"]["ignore_index"],
                reduction="mean",
            )

        with torch.no_grad():
            if split != "train":
                self._step_outputs[f"{split}_loss"].append(batch_loss.item())

                predicted_classes = []
                idx = 0
                for i in range(bs):
                    predicted_class_idx = similarities[idx : (idx + classes_count[i])].argmax()
                    predicted_classes.append(predicted_class_idx)
                    idx = idx + classes_count[i]

                self._step_outputs[f"{split}_predictions"] = self._step_outputs[f"{split}_predictions"] + [
                    pred_class.to("cpu") for pred_class in predicted_classes
                ]
                self._step_outputs[f"{split}_target"] = self._step_outputs[f"{split}_target"] + positive_classes

                probs = torch.sigmoid(similarities)
                batch_auroc = self._auroc_metric[f"{split}_auroc"](probs, target)
                batch_f1 = self._f1_metric[f"{split}_binary_f1"](probs, target)

        if split == "train":
            self.log_dict(
                {
                    f"{split}/step_loss": batch_loss.item(),
                }
            )
        return batch_loss

    def _report_metrics(self, split: str, loss: list = None):
        if split != "train":
            epoch_auroc = self._auroc_metric[f"{split}_auroc"].compute()
            self._auroc_metric[f"{split}_auroc"].reset()

            epoch_binary_f1 = self._f1_metric[f"{split}_binary_f1"].compute()
            self._f1_metric[f"{split}_binary_f1"].reset()

            y_true = self._step_outputs[f"{split}_target"]
            y_pred = self._step_outputs[f"{split}_predictions"]
            y_true_multiclass = [true_classes[0] for true_classes in y_true]
            accuracy = accuracy_score(y_true_multiclass, y_pred)
            self.log(f"{split}/epoch_accuracy", accuracy)

            self.log_dict(
                {
                    f"{split}/epoch_auroc": epoch_auroc,
                    f"{split}/epoch_binary_f1": epoch_binary_f1,
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
        self._step_outputs["train_predictions"].clear()
        self._step_outputs["train_target"].clear()

    def on_validation_epoch_end(self):
        self._report_metrics("val", self._step_outputs["val_loss"])

        self._step_outputs["val_loss"].clear()
        self._step_outputs["val_predictions"].clear()
        self._step_outputs["val_target"].clear()

    def on_test_epoch_end(self):
        self._report_metrics("test", self._step_outputs["test_loss"])

        self._step_outputs["test_loss"].clear()
        self._step_outputs["test_predictions"].clear()
        self._step_outputs["test_target"].clear()

    def validation_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "test")

    def training_step(self, batch: Tuple[Tensor, ...], batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self.shared_step(batch, "train")

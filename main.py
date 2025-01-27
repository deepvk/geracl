import logging
from argparse import ArgumentParser

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.data_module import ZeroShotClassificationDataModule
from src.model.universal_classifier import ZeroShotClassifier
from src.utils import setup_logging


def configure_arg_parser() -> ArgumentParser:
    argument_parser = ArgumentParser(description="CLI for zero shot classification model.")
    argument_parser.add_argument("--device", required=True, help="Device to train on.", type=str)
    argument_parser.add_argument(
        "--unfreeze-embedder",
        action="store_true",
        help="Freeze embedder parameters before training.",
    )
    argument_parser.add_argument("--n-steps", required=True, help="Number of training steps.", type=int)
    argument_parser.add_argument(
        "--eval-steps",
        required=True,
        help="Perform validation every --eval-steps steps.",
        type=int,
    )
    argument_parser.add_argument("--log-steps", help="How often to add logging rows.", default=50, type=int)

    return argument_parser


def train(
    n_steps: int = 10000,
    accelerator: str = "gpu",
    eval_steps: int = 2000,
    gradient_clip: float = 0.0,
    log_steps: int = 50,
    seed: int = 7,
    wandb_project_name: str = "universal_classifier",
    unfreeze_embedder: bool = False,
):
    """Main function to run model training.

    :param n_steps: number of training steps.
    :param accelerator: name of accelerator, e.g. "gpu".
    :param eval_steps: period of evaluation.
    :param gradient_clip: gradient clipping value, 0.0 means no gradient clipping.
    :param log_steps: period of logging step.
    :param seed: random seed.
    :param wandb_project_name: name of W&B project to log progress.
    """
    seed_everything(seed)
    setup_logging()

    wandb_logger = WandbLogger(project=wandb_project_name)
    checkpoint_callback = ModelCheckpoint(
        "/data/checkpoints/",
        filename="step_{step}",
        every_n_train_steps=eval_steps,
        save_top_k=-1,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    lr_logger = LearningRateMonitor("step")

    data_module = ZeroShotClassificationDataModule()
    tokenizer = data_module._tokenizer
    if accelerator == "gpu":
        model = ZeroShotClassifier(unfreeze_embedder=unfreeze_embedder, device="cuda", tokenizer=tokenizer)
    else:
        model = ZeroShotClassifier(unfreeze_embedder=unfreeze_embedder, device=accelerator, tokenizer=tokenizer)

    trainer = Trainer(
        accelerator=accelerator,
        callbacks=[lr_logger, checkpoint_callback],
        gradient_clip_val=gradient_clip,
        log_every_n_steps=log_steps,
        logger=wandb_logger,
        max_steps=n_steps,
        val_check_interval=eval_steps,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    _arg_parser = configure_arg_parser()
    _args = _arg_parser.parse_args()

    train(
        n_steps=_args.n_steps,
        eval_steps=_args.eval_steps,
        log_steps=_args.log_steps,
        accelerator=_args.device,
        unfreeze_embedder=_args.unfreeze_embedder,
    )

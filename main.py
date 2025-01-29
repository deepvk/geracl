from argparse import ArgumentParser
from collections.abc import Mapping

import wandb
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.data.data_module import ZeroShotClassificationDataModule
from src.model.universal_classifier import ZeroShotClassifier
from src.utils import setup_logging


def merge_configs(default, override):
    """Recursively merge default config with user override."""
    for key, value in override.items():
        if isinstance(value, Mapping) and key in default:
            default[key] = merge_configs(default[key], value)
        else:
            default[key] = value
    return default


def load_config_with_defaults(file_path, default_config):
    """Load user config and merge with defaults."""
    with open(file_path, "r") as file:
        user_config = yaml.safe_load(file)
    return merge_configs(default_config, user_config)


def configure_arg_parser() -> ArgumentParser:
    argument_parser = ArgumentParser(description="CLI for zero shot classification model.")
    argument_parser.add_argument(
        "--config-path", help="Yaml file with training parameters.", default="default", type=str
    )
    return argument_parser


if __name__ == "__main__":
    seed = 7
    _arg_parser = configure_arg_parser()
    _args = _arg_parser.parse_args()

    with open("src/configs/default.yaml", "r") as file:
        default_config = yaml.safe_load(file)

    if _args.config_path == "default":
        config = default_config
    else:
        config = load_config_with_defaults(_args.config_path, default_config)

    seed_everything(seed)
    setup_logging()

    wandb_logger = WandbLogger(project=config["other"]["wandb_project"])
    checkpoint_callback = ModelCheckpoint(
        config["other"]["checkpoints_dir"],
        filename="step_{step}",
        every_n_train_steps=config["trainer"]["val_check_interval"],
        save_top_k=-1,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    lr_logger = LearningRateMonitor("step")

    data_module = ZeroShotClassificationDataModule()
    data_module = ZeroShotClassificationDataModule(**config["data_module"])
    model = ZeroShotClassifier(**config["model"], tokenizer=data_module.tokenizer)

    trainer = Trainer(
        **config["trainer"],
        callbacks=[lr_logger, checkpoint_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)

    wandb.finish()

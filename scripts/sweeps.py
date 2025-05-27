import wandb
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from geracl.data.data_module import ZeroShotClassificationDataModule
from geracl.model.geracl import Geracl
from geracl.utils import setup_logging
from main import load_config_with_defaults

sweep_configuration = {
    "method": "grid",
    "name": "my_sweep",
    "metric": {
        "name": "val/epoch_accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "max_lr": {"values": [5e-6, 1e-6, 1e-5, 5e-5]},
        "weight_decay": {"values": [0, 0.01, 0.1]},
    },
}


def train_sweep():
    with open("src/configs/default.yaml", "r") as file:
        default_config = yaml.safe_load(file)

    config = load_config_with_defaults("src/configs/custom.yaml", default_config)

    seed = 7
    run = wandb.init(dir=config["other"]["wandb_dir"])

    # Get hyperparameters from sweep
    sweep_config = wandb.config

    config["model"]["optimizer_args"]["init_params"] = dict()
    config["model"]["optimizer_args"]["init_params"]["lr"] = sweep_config.max_lr
    assert "scheduler_args" in config["model"]
    config["model"]["scheduler_args"]["total_steps"] = config["builder"]["val_check_interval"] * config.max_epochs

    seed_everything(seed)
    setup_logging()

    wandb_logger = WandbLogger(project=config["other"]["wandb_project"], save_dir=config["other"]["wandb_dir"])
    checkpoint_callback = ModelCheckpoint(
        config["other"]["checkpoints_dir"],
        filename="step_{step}",
        save_top_k=-1,
        auto_insert_metric_name=False,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(monitor="val/epoch_accuracy", patience=3, mode="max")
    lr_logger = LearningRateMonitor("step")

    data_module = ZeroShotClassificationDataModule()
    data_module = ZeroShotClassificationDataModule(**config["data_module"])
    model = Geracl(**config["model"], tokenizer_len=len(data_module.tokenizer))

    trainer = Trainer(
        **config["trainer"],
        callbacks=[lr_logger, checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_configuration, project="universal_classifier")
    wandb.agent(sweep_id, function=train_sweep)

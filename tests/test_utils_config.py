import shutil
import time
import tomllib
from pathlib import Path

import pytest
import torch

from ml_tools.utils.utils import Config


@pytest.fixture
def config():
    if Path("tests/output/logs").exists():
        shutil.rmtree("tests/output/logs")

    config = Config.generate("tests/data/test.config.toml", silent=False, extra_config={})

    yield config


def test_config_generate(config):
    with open("tests/data/test.config.toml", mode="rb") as f:
        config_from_toml = tomllib.load(f)

    assert config.train_settings.batch_size == config_from_toml["train_settings"]["batch_size"]
    assert config.train_settings.early_stop_patience == config_from_toml["train_settings"]["early_stop_patience"]
    assert config.train_settings.epochs == config_from_toml["train_settings"]["epochs"]
    assert config.train_settings.exp_name == config_from_toml["train_settings"]["exp_name"]
    assert config.train_settings.k_folds == config_from_toml["train_settings"]["k_folds"]
    assert config.train_settings.logging_per_batch == config_from_toml["train_settings"]["logging_per_batch"]
    assert config.train_settings.lr == config_from_toml["train_settings"]["lr"]
    assert config.train_settings.lr_decay == config_from_toml["train_settings"]["lr_decay"]
    assert config.train_settings.test_size == config_from_toml["train_settings"]["test_size"]
    assert config.train_settings.valid_size == config_from_toml["train_settings"]["valid_size"]
    assert config.train_settings.device == torch.device("gpu" if torch.cuda.is_available() else "cpu")
    assert (
        config.train_settings.lr_finder_settings.initial_value
        == config_from_toml["train_settings"]["lr_finder_settings"]["initial_value"]
    )
    assert (
        config.train_settings.lr_finder_settings.final_value
        == config_from_toml["train_settings"]["lr_finder_settings"]["final_value"]
    )
    assert (
        config.train_settings.lr_finder_settings.beta
        == config_from_toml["train_settings"]["lr_finder_settings"]["beta"]
    )

    assert config.log_settings.backup == config_from_toml["log_settings"]["backup"]
    assert config.log_settings.backup_dir.exists()
    assert config.log_settings.backup_dir.name == config_from_toml["log_settings"]["backup_dir_name"]
    assert config.log_settings.log_dir.exists()
    assert config.log_settings.log_filename == config_from_toml["log_settings"]["log_filename"]
    assert config.log_settings.mlflow_dir.exists()

    config.print("test log")
    config.logger.info("test log 2")

    config.init_mlflow()
    config.print(config.mlflow_writer.exp_id)
    config.mlflow_writer.log_metric("accuracy", "1.0")
    config.mlflow_writer.log_text("test text", "test_text")
    test_model = torch.nn.Linear(1, 1)
    config.mlflow_writer.log_pytorch_model(test_model, "test_model")
    config.save_pytorch_model(test_model, "test_model")
    config.close_mlflow()

    config.add_logger("logger_2")
    config.ex_logger.logger_2.info("test")


def test_config_generate_without_toml(config):
    config = Config.generate()
    assert isinstance(config, Config)

"""
This module contains utility functions and classes for machine learning tasks.

The module includes various utility functions and classes that can be used for common machine learning tasks. It provides functionalities for handling data, logging, tracking experiments with MLflow, and more.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import string
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from glob import glob
from logging import Logger
from os import PathLike
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Callable, Optional

import cpuinfo
import mlflow
import nltk
import numpy as np
import psutil
import torch
from attrdict import AttrDict
from colorama import Fore, Style
from IPython import get_ipython
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME, MLFLOW_SOURCE_NAME, MLFLOW_USER
from pyunpack import Archive
from torchinfo import summary

if sys.version_info.minor < 11:
    import toml
else:
    import tomllib

from ml_tools.utils.logger import get_logger, kill_logger


class NvidiaSmiAttributes(Enum):
    index = "index"
    uuid = "uuid"
    name = "name"
    timestamp = "timestamp"
    memory_total = "memory.total"
    memory_free = "memory.free"
    memory_used = "memory.used"
    utilization_gpu = "utilization.gpu"
    utilization_memory = "utilization.memory"


class CpuRamAttributes(Enum):
    cpu_usage = "cpu_usage"
    ram_usage_total = "ram_usage_total"
    ram_usage_available = "ram_usage_available"
    ram_usage_percent = "ram_usage_percent"
    ram_usage_used = "ram_usage_used"
    ram_usage_free = "ram_usage_free"


if not (Path(nltk.downloader.Downloader().download_dir) / "tokenizers" / "punkt").exists():
    nltk.download("punkt", quiet=True)

if not (Path(nltk.downloader.Downloader().download_dir) / "taggers" / "averaged_perceptron_tagger").exists():
    nltk.download("averaged_perceptron_tagger", quiet=True)


def now() -> datetime:
    """
    Get the current datetime in the JST (Japan Standard Time) timezone.

    Returns:
        datetime: The current datetime in JST.
    """
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)


def is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMInteractiveShell":
            return True  # Jupyter notebook qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal ipython
        elif "google.colab" in sys.modules:
            return True  # Google Colab
        else:
            return False
    except NameError:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Phase(Enum):
    DEV = "dev"
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    SUBMISSION = "submission"


def get_enum_from_value(enum_class, value: Any):
    """
    Retrieves the enum item from the given enum class based on the provided value.

    Args:
        enum_class (Enum): The enum class to retrieve the item from.
        value (Any): The value to match against the enum item's value.

    Returns:
        Enum: The enum item that matches the provided value.

    Raises:
        KeyError: If no enum item is found with the provided value.
    """
    value_map = {item.value: item for item in enum_class}
    return value_map[value]


def filepath_to_uri(path: Path) -> str:
    """
    Convert a file path to a URI.

    Args:
        path (Path): The file path to convert.

    Returns:
        str: The URI representation of the file path.
    """
    return urllib.parse.urljoin("file:", urllib.request.pathname2url(str(path.resolve().absolute())))


class MlflowWriter:
    """
    A utility class for interacting with MLflow to log experiments, parameters, metrics, artifacts, and models.
    """

    def __init__(self, exp_name: str, tracking_uri: str, logger: Optional[Logger] = None):
        """
        Initializes an instance of MlflowWriter.

        Args:
            exp_name (str): The name of the MLflow experiment.
            tracking_uri (str): The URI of the MLflow tracking server.
            logger (Optional[Logger], optional): The logger to use for logging messages. Defaults to None.
        """
        self.__exp_name = exp_name
        self.__tracking_uri = filepath_to_uri(Path(tracking_uri))
        self.__logger = logger
        self.__print = lambda x: print(x) if self.__logger is None else lambda x: self.__logger.info(x)
        self.__initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """
        Returns:
            bool: True if the MlflowWriter is initialized, False otherwise.
        """
        return self.__initialized

    def __initialize__(self):
        """
        Initializes the MLflow client and creates a new experiment if it doesn't exist.
        """
        self.client = MlflowClient(tracking_uri=self.__tracking_uri)
        self.run = None
        self.run_id = None
        try:
            self.exp_id = self.client.create_experiment(self.__exp_name)
        except Exception as e:
            self.__print(e)
            self.exp_id = self.client.get_experiment_by_name(self.__exp_name).experiment_id

        self.experiment = self.client.get_experiment(self.exp_id)

        self.__print("New experiment started")
        self.__print(f"Name: {self.experiment.name}")
        self.__print(f"Experiment ID: {self.experiment.experiment_id}")
        self.__print(f"Artifact Location: {self.experiment.artifact_location}")

        mlflow.set_tracking_uri(self.__tracking_uri)
        mlflow.set_experiment(self.__exp_name)

    def initialize(self, tags=None):
        """
        Initializes the MlflowWriter by creating a new run and setting it as the active run.

        Args:
            tags (Optional[dict], optional): Tags to associate with the run. Defaults to None.
        """
        self.__initialize__()
        self.create_new_run(tags=tags)
        self.__initialized = True

    def create_new_run(self, tags=None):
        """
        Creates a new run within the MLflow experiment.

        Args:
            tags (Optional[dict], optional): Tags to associate with the run. Defaults to None.
        """
        self.run = self.client.create_run(self.exp_id, tags=tags)
        assert self.run is not None

        self.run_id = self.run.info.run_id
        self.__print(f"New run started: {self.run.info.run_name}")

        mlflow.tracking.fluent._active_run_stack.append(self.run)

    def terminate(self):
        """
        Terminates the current run.
        """
        self.client.set_terminated(self.run_id, RunStatus.to_string(RunStatus.FINISHED))
        self.__initialized = False

    def log_param(self, key: str, value: Any):
        """
        Logs a parameter for the current run.

        Args:
            key (str): The name of the parameter.
            value (Any): The value of the parameter.
        """
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key: str, value: Any, step=None):
        """
        Logs a metric for the current run.

        Args:
            key (str): The name of the metric.
            value (Any): The value of the metric.
            step (Optional[int], optional): The step number. Defaults to None.
        """
        self.client.log_metric(self.run_id, key, value, step=step)

    def log_artifact(self, local_path: str):
        """
        Logs an artifact for the current run.

        Args:
            local_path (str): The local path of the artifact.
        """
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str = ""):
        """
        Logs a dictionary as an artifact for the current run.

        Args:
            dictionary (dict[str, Any]): The dictionary to log.
            artifact_file (str, optional): The name of the artifact file. Defaults to "".
        """
        self.client.log_dict(self.run_id, dictionary, artifact_file)

    def log_text(self, text: str, artifact_file: str):
        """
        Logs a text string as an artifact for the current run.

        Args:
            text (str): The text string to log.
            artifact_file (str): The name of the artifact file.
        """
        self.client.log_text(self.run_id, text, artifact_file)

    def log_pytorch_model(self, model, artifact_path: str):
        """
        Logs a PyTorch model as an artifact for the current run.

        Args:
            model: The PyTorch model to log.
            artifact_path (str): The path to save the model artifact.
        """
        mlflow.pytorch.log_model(pytorch_model=model, artifact_path=artifact_path)


@dataclass
class LrFinderSettings(object):
    """
    A class representing the settings for a learning rate finder.

    Attributes:
        initial_value (float): The initial learning rate value.
        final_value (float): The final learning rate value.
        beta (float): The beta value used for smoothing the loss curve.
    """

    initial_value: float = 1e-8
    final_value: float = 10
    beta: float = 0.98

    @classmethod
    def from_dict(cls, config: dict):
        """
        Creates an instance of LrFinderSettings from a dictionary.

        Args:
            config (dict): A dictionary containing the configuration values.

        Returns:
            LrFinderSettings: An instance of LrFinderSettings with the specified configuration values.
        """
        settings = cls()
        if "initial_value" in config:
            settings.initial_value = float(config["initial_value"])
            settings.final_value = float(config["final_value"])
            settings.beta = float(config["beta"])
        return settings


@dataclass
class TrainSettings(object):
    """
    A class representing the settings for training a model.

    Attributes:
        device (torch.device): The device to use for training. Defaults to torch.device("cpu").
        exp_name (str): The name of the experiment. Defaults to "default_exp".
        k_folds (int): The number of folds for cross-validation. Defaults to 3.
        epochs (int): The number of training epochs. Defaults to 250.
        batch_size (int): The batch size for training. Defaults to 32.
        valid_size (float): The proportion of data to use for validation. Defaults to 0.1.
        test_size (float): The proportion of data to use for testing. Defaults to 0.1.
        lr (float): The learning rate for training. Defaults to 0.001.
        lr_decay (float): The learning rate decay factor. Defaults to 0.9.
        early_stop_patience (int): The number of epochs to wait for early stopping. Defaults to 10.
        logging_per_batch (int): The number of batches to wait before logging. Defaults to 5.
        lr_finder_settings (LrFinderSettings): The settings for learning rate finder. Defaults to LrFinderSettings().
        ex_args (dict): Extra arguments for training settings. Defaults to an empty dictionary.
    """

    device: torch.device = torch.device("cpu")
    exp_name: str = "default_exp"
    k_folds: int = 3
    epochs: int = 250
    batch_size: int = 32
    valid_size: float = 0.1
    test_size: float = 0.1
    lr: float = 0.001
    lr_decay: float = 0.9
    early_stop_patience: int = 10
    logging_per_batch: int = 5
    lr_finder_settings: LrFinderSettings = field(default_factory=LrFinderSettings)
    ex_args: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict) -> TrainSettings:
        """
        Creates a TrainSettings object from a dictionary.

        Args:
            config (dict): The dictionary containing the configuration settings.

        Returns:
            TrainSettings: The TrainSettings object created from the dictionary.
        """
        settings = cls()
        if "device" in config:
            settings.device = torch.device(config["device"])
        elif torch.cuda.is_available():
            settings.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            settings.device = torch.device("mps")
        else:
            settings.device = torch.device("cpu")

        if "exp_name" in config:
            settings.exp_name = str(config["exp_name"])
        if "k_fold" in config:
            settings.k_folds = int(config["k_folds"])
        if "epochs" in config:
            settings.epochs = int(config["epochs"])
        if "batch_size" in config:
            settings.batch_size = int(config["batch_size"])
        if "valid_size" in config:
            settings.valid_size = float(config["valid_size"])
        if "test_size" in config:
            settings.test_size = float(config["test_size"])
        if "lr" in config:
            settings.lr = float(config["lr"])
        if "lr_decay" in config:
            settings.lr_decay = float(config["lr_decay"])
        if "early_stop_patience" in config:
            settings.early_stop_patience = int(config["early_stop_patience"])
        if "logging_per_batch" in config:
            settings.logging_per_batch = int(config["logging_per_batch"])
        if "lr_finder_settings" in config:
            settings.lr_finder_settings = LrFinderSettings.from_dict(config["lr_finder_settings"])

        default_fields = list(settings.__dataclass_fields__.keys())
        for key, value in config.items():
            if key not in default_fields:
                settings.ex_args[key] = value

        return settings


@dataclass
class LogSettings(object):
    """
    A class representing the settings for logging in a machine learning system.

    Attributes:
        log_dir (Path): The directory where log files will be stored.
        log_filename (str): The name of the log file.
        log_file (Path): The full path to the log file.
        data_dir (Path): The directory where data files will be stored.
        backup_dir (Path): The directory where backup files will be stored.
        local_cache_dir (Path): The directory where local cache files will be stored.
        global_cache_dir (Path): The directory where global cache files will be stored.
        output_dir (Path): The directory where output files will be stored.
        weights_dir (Path): The directory where weight files will be stored.
        mlflow_dir (Path): The directory where MLflow files will be stored.
        backup (bool): Flag indicating whether to create backup files.

    Methods:
        from_dict(cls, config: dict, exp_name: str = "exp", timestamp: datetime = datetime.now()) -> LogSettings:
            Creates a LogSettings object from a dictionary of configuration options.

    """

    log_dir: Path = Path("logs")
    log_filename: str = "system.log"
    log_file: Path = Path("")
    data_dir: Path = Path("logs/data")
    backup_dir: Path = Path("logs/backup")
    local_cache_dir: Path = Path("logs/cache")
    global_cache_dir: Path = Path("logs/cache")
    output_dir: Path = Path("logs/outputs")
    weights_dir: Path = Path("logs/weights")
    mlflow_dir: Path = Path("logs/mlflow")
    backup: bool = False

    @classmethod
    def from_dict(cls, config: dict, exp_name: str = "exp", timestamp: datetime = datetime.now()) -> LogSettings:
        """
        Creates a LogSettings object from a dictionary of configuration options.

        Args:
            config (dict): A dictionary containing the configuration options.
            exp_name (str): The name of the experiment. Defaults to "exp".
            timestamp (datetime): The timestamp for the log directory. Defaults to the current datetime.

        Returns:
            LogSettings: A LogSettings object with the specified configuration options.

        """
        settings = cls()
        if "log_dir" in config:
            settings.log_dir = Path(config["log_dir"])
        if "log_filename" in config:
            settings.log_filename = str(config["log_filename"])
        if "backup" in config:
            if isinstance(config["backup"], str):
                settings.backup = config["backup"].lower() == "true"
            elif isinstance(config["backup"], bool):
                settings.backup = config["backup"]
            else:
                settings.backup = False

        settings.log_dir = settings.log_dir / f"{exp_name}_{timestamp.strftime('%Y%m%d%H%M%S')}"
        settings.log_file = settings.log_dir / settings.log_filename

        if "backup_dir_name" in config:
            settings.backup_dir = settings.log_dir / config["backup_dir_name"]
        if "data_dir_name" in config:
            settings.data_dir = settings.log_dir / config["data_dir_name"]
        if "output_dir_name" in config:
            settings.output_dir = settings.log_dir / config["output_dir_name"]
        if "weights_dir_name" in config:
            settings.weights_dir = settings.log_dir / config["weights_dir_name"]
        if "mlflow_dir_name" in config:
            settings.mlflow_dir = settings.log_dir / config["mlflow_dir_name"]

        if "local_cache_dir_name" in config:
            settings.local_cache_dir = settings.log_dir / config["local_cache_dir_name"]
        if "global_cache_dir_name" in config:
            settings.global_cache_dir = Path(config["global_cache_dir_name"])

        return settings


@dataclass
class Config(object):
    """
    Configuration class for managing settings and options.

    Attributes:
        logger (Logger): The logger instance.
        mlflow_writer (MlflowWriter): The MLflow writer instance.
        config_path (Path): The path to the configuration file.
        timestamp (datetime): The timestamp of the configuration.
        train_settings (TrainSettings): The settings for training.
        log_settings (LogSettings): The settings for logging.
        ex_logger (AttrDict): The extra logger instance.

    Methods:
        get_hash: Generates a random hash.
        now: Returns the current timestamp.
        generate: Generates a configuration object.
        __mkdirs: Creates necessary directories.
        print: Prints the given message.
        init_mlflow: Initializes MLflow.
        close_mlflow: Terminates MLflow.
        describe_cpu: Describes CPU information.
        describe_gpu: Describes NVIDIA GPU information.
        describe_m1_silicon: Describes M1 Silicon GPU information.
        describe_model: Describes the given model.
        get_gpu_usage: Retrieves GPU usage information.
        get_cpu_ram_usage: Retrieves CPU and RAM usage information.
    """

    logger: Logger
    mlflow_writer: MlflowWriter
    config_path: Path = Path("")
    timestamp: datetime = field(default_factory=datetime.now)
    train_settings: TrainSettings = field(default_factory=TrainSettings)
    log_settings: LogSettings = field(default_factory=LogSettings)
    ex_logger: AttrDict = field(default_factory=lambda: AttrDict({}))

    __TEXT = "config: {KEY_1:15s} - {KEY_2:20s}: {VALUE}"

    @classmethod
    def get_hash(cls, size: int = 12) -> str:
        """
        Generate a random hash string of the specified size.

        Parameters:
            size (int): The length of the hash string to generate. Default is 12.

        Returns:
            str: The randomly generated hash string.
        """
        chars = string.ascii_lowercase + string.digits
        return "".join(random.SystemRandom().choice(chars) for _ in range(size))

    @classmethod
    def now(cls) -> datetime:
        """
        Get the current datetime in JST timezone.

        Returns:
            datetime: The current datetime in JST timezone.
        """
        JST = timezone(timedelta(hours=9))
        return datetime.now(JST)

    @classmethod
    def generate(cls, config_path: str = "", silent: bool = False, extra_config: dict[str, Any] = {}) -> Config:
        """
        Generate a Config object based on the provided configuration file path and additional configuration.

        Args:
            cls (class): The class to instantiate the Config object.
            config_path (str, optional): The path to the configuration file. Defaults to "".
            silent (bool, optional): Whether to suppress log output. Defaults to False.
            extra_config (dict[str, Any], optional): Additional configuration to override the values in the file. Defaults to {}.

        Returns:
            Config: The generated Config object.

        Raises:
            FileNotFoundError: If the specified configuration file is not found.
        """
        settings = cls(logger=Logger(""), mlflow_writer=MlflowWriter("", ""), config_path=Path(config_path))

        if config_path:
            if sys.version_info.minor < 11:
                config: dict = toml.load(config_path)
            else:
                with open(config_path, mode="rb") as f:
                    config = tomllib.load(f)
            if len(extra_config) > 0:
                for s in ["train_settings", "log_settings"]:
                    if s in extra_config:
                        config[s].update(extra_config[s])
        else:
            config = {
                "train_settings": {
                    "exp_name": "sample_exp",
                },
                "log_settings": {
                    "backup": "False",
                    "log_dir": "logs",
                    "log_filename": "system.log",
                    "backup_dir_name": "backup",
                    "data_dir_name": "data",
                    "output_dir_name": "outputs",
                    "mlflow_dir_name": "mlflow",
                    "weights_dir_name": "weights",
                    "local_cache_dir_name": "cache",
                    "global_cache_dir_name": "logs/global_cache",
                },
            }

        # set attributes
        settings.timestamp = settings.now()
        if "yaml_path" in config:
            settings.config_path = Path(config["yaml_path"])
        if "train_settings" in config:
            settings.train_settings = TrainSettings.from_dict(config["train_settings"])
        if "log_settings" in config:
            settings.log_settings = LogSettings.from_dict(
                config["log_settings"], exp_name=settings.train_settings.exp_name, timestamp=settings.timestamp
            )

        # set logger
        if hasattr(settings, "logger") and isinstance(settings.logger, Logger):
            kill_logger(settings.logger)
        settings.logger = get_logger(name="config", logfile=str(settings.log_settings.log_file), silent=silent)

        # show config
        settings.logger.info("====== show config =========")
        settings.logger.info(settings.__TEXT.format(KEY_1="root", KEY_2="config_path", VALUE=settings.config_path))
        settings.logger.info(settings.__TEXT.format(KEY_1="root", KEY_2="timestamp", VALUE=settings.timestamp))
        for key in settings.log_settings.__dataclass_fields__.keys():
            settings.logger.info(
                settings.__TEXT.format(KEY_1="log_settings", KEY_2=key, VALUE=getattr(settings.log_settings, key))
            )
        for key in settings.train_settings.__dataclass_fields__.keys():
            settings.logger.info(
                settings.__TEXT.format(KEY_1="train_settings", KEY_2=key, VALUE=getattr(settings.train_settings, key))
            )
        settings.logger.info("============================")

        # CPU info
        settings.describe_cpu(print_fn=settings.print)

        # NVIDIA GPU info
        if torch.cuda.is_available():
            settings.describe_gpu(print_fn=settings.print)

        # mkdir
        settings.__mkdirs()

        return settings

    def __mkdirs(self):
        """
        Create necessary directories for logging and data storage.

        This method creates the following directories:
        - log_dir: Directory for log files
        - backup_dir: Directory for backup files
        - data_dir: Directory for data files
        - weights_dir: Directory for weight files
        - output_dir: Directory for output files
        - mlflow_dir: Directory for MLflow files
        - local_cache_dir: Directory for local cache files
        - global_cache_dir: Directory for global cache files
        """
        self.log_settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.weights_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.mlflow_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_settings.global_cache_dir.mkdir(parents=True, exist_ok=True)

    def print(self, x):
        """
        Prints the given value.

        If a logger is available, the value is logged as an info message.
        Otherwise, it is printed to the console.

        Args:
            x: The value to be printed.
        """
        if self.logger is None:
            print(x)
        else:
            self.logger.info(x)

    def init_mlflow(self):
        """
        Initializes the MLflow writer.

        This method initializes the MLflow writer by creating an instance of the MlflowWriter class
        and calling its `initialize` method.
        """
        self.mlflow_writer = MlflowWriter(
            exp_name=self.train_settings.exp_name,
            tracking_uri=str(self.log_settings.mlflow_dir.resolve().absolute()),
            logger=self.logger,
        )
        self.mlflow_writer.initialize()

    def close_mlflow(self):
        """
        Closes the MLflow writer.

        This method terminates the MLflow writer, ensuring that all pending data is flushed and resources are released.
        """
        self.mlflow_writer.terminate()

    @classmethod
    def describe_cpu(cls, print_fn: Callable = print):
        """
        Prints information about the CPU.

        Args:
            print_fn (Callable, optional): The function used to print the CPU information. Defaults to print.
        """
        print_fn("====== cpu info ============")
        for key, value in cpuinfo.get_cpu_info().items():
            print_fn(f"CPU INFO: {key:20s}: {value}")
        print_fn("============================")

    @classmethod
    def describe_gpu(cls, nvidia_smi_path="nvidia-smi", no_units=True, print_fn: Callable = print):
        """
        Retrieves and prints information about the available GPUs using the NVIDIA System Management Interface (nvidia-smi).

        Args:
            nvidia_smi_path (str, optional): The path to the nvidia-smi executable. Defaults to "nvidia-smi".
            no_units (bool, optional): Whether to exclude units from the output. Defaults to True.
            print_fn (Callable, optional): The function used for printing the output. Defaults to print.

        Raises:
            CalledProcessError: If an error occurs while executing the nvidia-smi command.
        """
        try:
            keys = [item.value for item in NvidiaSmiAttributes]
            nu_opt = "" if not no_units else ",nounits"
            cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
            output = subprocess.check_output(cmd, shell=True)
            raw_lines = [line.strip() for line in output.decode().split("\n") if line.strip() != ""]
            lines = [{k: v for k, v in zip(keys, line.split(", "))} for line in raw_lines]

            print_fn("====== show GPU information =========")
            for line in lines:
                for k, v in line.items():
                    print_fn(f"{k:25s}: {v}")
            print_fn("=====================================")
        except CalledProcessError:
            print_fn("====== show GPU information =========")
            print_fn("  No GPU was found.")
            print_fn("=====================================")

    @classmethod
    def describe_m1_silicon(cls, print_fn: Callable = print):
        """
        Prints information about the availability of the Mac-M1 GPU.

        Args:
            print_fn (Callable): The function used to print the output. Defaults to `print`.
        """
        print_fn("====== show GPU information =========")
        if torch.backends.mps.is_available():
            print_fn("  Mac-M1 GPU is available.")
        else:
            print_fn("  Mac-M1 GPU is NOT available.")
        print_fn("=====================================")

    @classmethod
    def describe_model(
        cls,
        model: torch.nn.Module,
        input_size: Optional[tuple[int]] = None,
        input_data=None,
        print_fn: Callable = print,
    ):
        """
        Generates a summary of the given model, including input/output sizes, number of parameters,
        kernel sizes, and multiply-add operations.

        Args:
            model (torch.nn.Module): The model to describe.
            input_size (Optional[tuple[int]]): The size of the input data. Defaults to None.
            input_data: The input data to use for the summary. Defaults to None.
            print_fn (Callable): The function used to print the summary. Defaults to print.
        """
        if input_data is None:
            summary_str = summary(
                model,
                input_size=input_size,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )
        else:
            summary_str = summary(
                model,
                input_data=input_data,
                col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )

        for line in summary_str.__str__().split("\n"):
            print_fn(line)

    @classmethod
    def get_gpu_usage(
        cls, nvidia_smi_path="nvidia-smi", no_units=True, logger: Optional[Logger] = None
    ) -> list[dict[NvidiaSmiAttributes, str]]:
        """
        Retrieves GPU usage information using the NVIDIA System Management Interface (nvidia-smi).

        Args:
            nvidia_smi_path (str, optional): Path to the nvidia-smi executable. Defaults to "nvidia-smi".
            no_units (bool, optional): Whether to exclude units from the output. Defaults to True.
            logger (Optional[Logger], optional): Logger object for logging information. Defaults to None.

        Returns:
            list[dict[NvidiaSmiAttributes, str]]: A list of dictionaries containing GPU usage information.
                Each dictionary represents the usage information for a single GPU and contains key-value pairs
                where the keys are attributes defined in the NvidiaSmiAttributes enum and the values are the
                corresponding attribute values.

        Raises:
            CalledProcessError: If an error occurs while executing the nvidia-smi command.
        """
        try:
            keys = [item.value for item in NvidiaSmiAttributes]
            nu_opt = "" if not no_units else ",nounits"
            cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
            if logger:
                logger.info(f"{cmd}")
            output = subprocess.check_output(cmd, shell=True)
            raw_lines = [line.strip() for line in output.decode().split("\n") if line.strip() != ""]
            info = [
                {get_enum_from_value(NvidiaSmiAttributes, k).value: v for k, v in zip(keys, line.split(", "))}
                for line in raw_lines
            ]
            return info

        except CalledProcessError:
            if logger:
                logger.warning("====== show GPU information =========")
                logger.warning("  No GPU was found.")
                logger.warning("=====================================")
            return []

    @classmethod
    def get_cpu_ram_usage(cls, logger: Optional[Logger] = None) -> dict[CpuRamAttributes, str]:
        """
        Get the CPU and RAM usage information.

        Args:
            logger (Optional[Logger]): An optional logger object to log any warnings.

        Returns:
            dict[CpuRamAttributes, str]: A dictionary containing the CPU and RAM usage information.
                - CpuRamAttributes.cpu_usage: A list of CPU usage percentages for each CPU core.
                - CpuRamAttributes.ram_usage_total: The total RAM available in bytes.
                - CpuRamAttributes.ram_usage_available: The available RAM in bytes.
                - CpuRamAttributes.ram_usage_percent: The percentage of RAM usage.
                - CpuRamAttributes.ram_usage_used: The used RAM in bytes.
                - CpuRamAttributes.ram_usage_free: The free RAM in bytes.

        """
        try:
            cpu_usage = psutil.cpu_percent(percpu=True)
            ram_usage = psutil.virtual_memory()
            return {
                CpuRamAttributes.cpu_usage: cpu_usage,
                CpuRamAttributes.ram_usage_total: ram_usage.total,
                CpuRamAttributes.ram_usage_available: ram_usage.available,
                CpuRamAttributes.ram_usage_percent: ram_usage.percent,
                CpuRamAttributes.ram_usage_used: ram_usage.used,
                CpuRamAttributes.ram_usage_free: ram_usage.free,
            }
        except Exception:
            if logger:
                logger.warning("Failed to get CPU RAM usage.")
            return {}

    def backup_logs(self):
        """Copy the log directory to the backup directory.

        This method deletes the existing backup directory and creates a new one.
        Then, it copies the contents of the log directory to the backup directory.
        """
        backup_dir = Path(self.log_settings.backup_dir)
        if backup_dir.exists():
            shutil.rmtree(str(backup_dir))
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.log_settings.log_dir, self.log_settings.backup_dir)

    def add_logger(self, name: str, silent: bool = False):
        """
        Adds a logger to the `ex_logger` dictionary.

        Parameters:
            name (str): The name of the logger.
            silent (bool, optional): If True, the logger will not print log messages to the console.
                Defaults to False.
        """
        self.ex_logger[name] = get_logger(name=name, logfile=str(self.log_settings.log_file), silent=silent)

    def fix_seed(self, seed=42):
        """
        Fix the random seed for reproducibility.

        Args:
            seed (int): The seed value to set for random number generators.
        """
        self.print(self.__TEXT.format(KEY_1="root", KEY_2="seed", VALUE=seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def save_pytorch_model(self, model, model_name: str):
        """
        Saves a PyTorch model using MLflow.

        Args:
            model: The PyTorch model to be saved.
            model_name (str): The name of the model.

        Raises:
            AssertionError: If the MLflow writer is not initialized.
        """
        assert self.mlflow_writer.is_initialized
        self.mlflow_writer.log_pytorch_model(model, str(self.log_settings.log_dir / model_name))


def timedelta2HMS(total_sec: int) -> str:
    h = total_sec // 3600
    m = total_sec % 3600 // 60
    s = total_sec % 60
    return f"{h:2d}h {m:2d}m {s:2d}s"


def __show_progress__(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = "=" * bar_num
    if bar_num != max_bar:
        progress_element += ">"
    bar_fill = " "
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        Fore.LIGHTCYAN_EX,
        f"[{bar}] {percentage:.2f}% ( {total_size_kb:.0f}KB )\r",
        end="",
    )


def download(url: str, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    print(Fore.LIGHTGREEN_EX, "download from:", end="")
    print(Fore.WHITE, url)
    urllib.request.urlretrieve(url, filepath, __show_progress__)
    print("")  # 改行
    print(Style.RESET_ALL, end="")


def un7zip(src_path: str, dst_path: str):
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    Archive(src_path).extractall(dst_path)
    for dirname, _, filenames in os.walk(dst_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def isint(s: str) -> bool:
    """Check the argument string is integer or not.

    Args:
        s (str): string value.

    Returns:
        bool: If the given string is integer or not.
    """
    try:
        int(s, 10)
    except ValueError:
        return False
    else:
        return True


def isfloat(s: str) -> bool:
    """Check the argument string is float or not.

    Args:
        s (str): string value.

    Returns:
        bool: If the given string is float or not.
    """
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def to_int(s: str, default: int = 10) -> int:
    """Convert the argument string to integer.

    Args:
        s (str): string value.

    Returns:
        int: integer value.
    """
    try:
        return int(s)
    except:
        return default


class JsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that extends the functionality of the default JSONEncoder class.
    """

    def default(self, obj):
        """
        Override the default method to handle custom object serialization.

        Args:
            obj: The object to be serialized.

        Returns:
            The serialized representation of the object.

        Raises:
            TypeError: If the object cannot be serialized.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__iter__"):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.strftime("%Y%m%d %H:%M:%S.%f")
        elif isinstance(obj, date):
            return datetime(obj.year, obj.month, obj.day, 0, 0, 0).strftime("%Y%m%d %H:%M:%S.%f")
        else:
            return super().default(obj)

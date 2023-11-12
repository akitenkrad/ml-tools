from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from nlp_tools.datasets.base import BaseDataset
from nlp_tools.utils.utils import Config


class BaseTrainer(ABC):
    def __init__(
        self,
        config: Config,
        name: str,
        model: nn.Module,
        dataset: BaseDataset,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
    ):
        self.config = config
        self.name = name
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    @abstractmethod
    def find_lr(self, init_value=1e-8, final_value=10.0, beta=0.98):
        pass

    @abstractmethod
    def fit(self, x):
        pass

    def iterdesc(
        self,
        fold: Optional[int] = None,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        batch_total: Optional[int] = None,
        loss: Optional[float] = None,
        lr: Optional[float] = None,
    ):
        """get description for iter

        Args:
            fold (Optional[int]): fold index
            epoch (Optional[int]): epoch index
            batch (Optional[int]): batch index
            batch_total (Optional[int]): total batch number
            loss (Optional[float]): loss
            lr (Optional[float]): learning rate
        """
        desc = "["
        if fold is not None:
            desc += f" Fold {fold:02d}"
        if epoch is not None:
            if len(desc) > 1:
                desc += " |"
            desc += f" Epoch {epoch:03d}"
        if batch is not None:
            if len(desc) > 1:
                desc += " |"
                desc += f" Batch {batch:05d}"
            if batch_total is not None:
                desc += f"/{batch_total:05d} ({(batch/batch_total) * 100.0: 5.2f}%)"
        desc += "]"

        if loss is not None:
            desc += f" Loss:{loss: 6.3f}"
        if lr is not None:
            if len(desc) > 1:
                desc += " |"
                desc += f" LR:{lr: 9.7f}"

        return desc

    def save_model(self, name: str):
        """save model in config.weights.log_weights_dir

        Args:
            name (str): name of the model to save
        """
        save_dir = Path(self.config.weights.log_weights_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(save_dir / name))

import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.base import BaseDataset
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml_tools.trainer.base import BaseTrainer
from ml_tools.utils.utils import Config, is_notebook
from ml_tools.utils.watchers import LossWatcher

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Trainer(BaseTrainer):
    def __init__(
        self,
        config: Config,
        name: str,
        model: nn.Module,
        dataset: BaseDataset,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
    ):
        super().__init__(config, name, model, dataset, optimizer, lr_scheduler)

    def loss_function(self, out, target):
        return torch.sqrt(torch.mean((out - target) ** 2))

    def fit(self):
        # configuration
        self.config.add_logger("train", silent=False)
        self.model.train().to(self.config.train.device)
        global_step = 0
        epochs = self.config.train.epochs

        # initialize learning rate
        self.optimizer.param_groups[0]["lr"] = self.config.train.lr

        tensorboard_dir = self.config.log.log_dir / "tensorboard"
        with SummaryWriter(str(tensorboard_dir)) as tb_writer:
            kfold = KFold(n_splits=self.config.train.k_folds, shuffle=True)
            for fold_index, (train_indices, valid_indices) in enumerate(kfold.split(self.dataset)):
                # prepare dataloader
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                valid_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                train_dl = DataLoader(self.dataset, batch_size=self.config.train.batch_size, sampler=train_subsampler)
                valid_dl = DataLoader(self.dataset, batch_size=self.config.train.batch_size, sampler=valid_subsampler)

                # Epoch loop
                for epoch_index in range(epochs):
                    loss_watcher = LossWatcher("train_loss")

                    # Batch loop
                    for batch_index, (x, y) in enumerate(train_dl):
                        x = x.to(self.config.train.device)
                        y = y.to(self.config.train.device)

                        out = self.model(x)

                        self.optimizer.zero_grad()
                        loss = self.loss_function(out, x)
                        loss.backward()
                        self.optimizer.step()

                        # put logs
                        loss_watcher.put(loss.item())

                        # global step for summary writer
                        global_step += 1
                        tb_writer.add_scalar("train loss", loss.item(), global_step)

                        if batch_index > 0 and batch_index % self.config.train.logging_per_batch == 0:
                            log = self.iterdesc(
                                fold=fold_index,
                                epoch=epoch_index,
                                batch=batch_index,
                                batch_total=len(train_dl),
                                loss=loss.item(),
                            )
                            self.config.log.train.info(log)

                    # validation loop
                    for valid_index, (x, y) in enumerate(valid_dl):
                        x = x.to(self.config.train.device)
                        y = y.to(self.config.train.device)
                        out = self.model(x)
                        loss = self.loss_function(out, x)
                        loss_watcher.put(loss.item())

                    tb_writer.add_scalar("valid loss", loss_watcher.mean, global_step)
                    self.config.log.train.info(
                        self.iterdesc(fold=fold_index, epoch=epoch_index, batch=valid_index, loss=loss_watcher.mean)
                    )

                    # step scheduler
                    self.lr_scheduler.step()

                    # logging
                    last_lr = self.lr_scheduler.get_last_lr()[0]
                    log = self.iterdesc(fold=fold_index, epoch=epoch_index, loss=loss_watcher.mean, lr=last_lr)
                    self.config.log.train.info(log)

                    # save best model
                    if loss_watcher.is_best:
                        self.save_model(f"{self.name}_best.pt")

                    # save model regularly
                    if epoch_index % 5 == 0:
                        self.save_model(f"{self.name}_f{fold_index}_e{epoch_index}.pt")
                # end of Epoch
                self.save_model(f"{self.name}_last_at_f{fold_index}.pt")
            # end of k-fold
            self.save_model(f"{self.name}_last.pt")

    def find_lr(self, init_value=1e-8, final_value=10.0, beta=0.98):
        self.model.train().to(self.config.train.device)
        self.config.add_logger("lr_finder", silent=True)
        self.dataset.to_train()
        dl = DataLoader(self.dataset, batch_size=self.config.train.batch_size)

        num = len(dl) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]["lr"] = lr

        avg_loss, best_loss = 0.0, 0.0
        losses, log_lrs = [], []
        stop_counter = 10

        with tqdm(enumerate(dl), total=len(dl), desc=f"[B:{0:05d}] lr:{lr:.8f} best_loss:{-1:.3f}") as it:
            for idx, (x, y) in it:
                # process model and calculate loss
                x = x.to(self.config.train.device)
                y = y.to(self.config.train.device)
                out = self.model(x)
                loss = self.loss_function(out, x)

                # compute the smoothed loss
                avg_loss = beta * avg_loss + (1 - beta) * loss.item()
                smoothed_loss = avg_loss / (1 - beta ** (idx + 1))

                # stop if the loss is exploding
                if idx > 0 and smoothed_loss > 1e3 * (best_loss + 1e-10):
                    stop_counter -= 1

                    if stop_counter <= 0:
                        break

                # record the best loss
                if smoothed_loss < best_loss or idx == 0:
                    best_loss = smoothed_loss

                # store the values
                losses.append(smoothed_loss)
                log_lrs.append(lr)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update progress bar
                desc = "[B:{:05d}] lr:{:.10f} best_loss:{:.6f} loss:{:.6f}".format(idx + 1, lr, best_loss, loss.item())
                it.set_description(desc)
                self.config.log.lr_finder.info(desc)

                # update learning rate
                lr *= mult
                self.optimizer.param_groups[0]["lr"] = lr

            # save figure
            save_path = self.config.log.log_dir / "lr_finder" / f"{self.name}_lr_loss_curve.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=log_lrs[10:-5],
                    y=losses[10:-5],
                )
            )
            fig.update_yaxes(type="log")
            fig.update_xaxes(type="log")
            fig.update_layout(xaxis_title="Learning Rate", yaxis_title="Loss")
            fig.write_image(str(save_path.expanduser().absolute()), engine="kaleido")
            fig.show()

import torch
import torch.nn as nn
import torch.optim as optim
from datasets.base import BaseDataset
from ml_tool.utils.utils import Config, is_notebook
from ml_tool.utils.watchers import LossWatcher
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Trainer(object):
    def __init__(
        self,
        config: Config,
        discriminator: nn.Module,
        generator: nn.Module,
        dataset: BaseDataset,
        generator_optimizer: optim.Optimizer,
        discriminator_optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
    ):
        self.config = config
        self.discriminator = discriminator
        self.generator = generator
        self.dataset = dataset
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.lr_scheduler = lr_scheduler

    def __fit_generator(
        self,
        fold_index: int,
        epoch_index: int,
        total: int,
        batch_size: int,
        global_step: int,
        loss_watcher: LossWatcher,
    ):
        self.generator.train()
        self.discriminator.eval()
        loss_function = nn.BCELoss()
        with tqdm(range(total), desc=f"[Epoch {epoch_index:3d} | Generator | Batch {0:3d}]", leave=False) as batch_it:
            for batch_index, _ in enumerate(batch_it):
                x = torch.rand(batch_size, 100, dtype=torch.float32).to(self.config.train.device)
                y = torch.ones(batch_size, dtype=torch.float32).to(self.config.train.device)
                generated_images = self.generator(x)

                with torch.no_grad():
                    out = self.discriminator(generated_images)
                    probs = torch.sigmoid(out)

                loss = loss_function(probs, y)

                self.generator_optimizer.zero_grad()
                loss.backward()
                self.generator_optimizer.step()

                # put logs
                tb_writer.add_scalar("generator train loss", loss.item(), global_step)
                loss_watcher.put(loss.item())
                batch_it.set_description(
                    f"[Epoch {epoch_index:3d} | Generator |  Batch {batch_index:3d}] Loss: {loss.item():.3f}"
                )

                # global step for summary writer
                global_step += 1

                if batch_index > 0 and batch_index % self.config.train.logging_per_batch == 0:
                    log = (
                        f"[Fold {fold_index:02d}"
                        + f" | Epoch {epoch_index:03d}"
                        + " | Generator"
                        + f" | Batch {batch_index:05d}/{total:05d} ({(batch_index/total) * 100.0:.2f}%)"
                        + f"] Loss:{loss.item():.3f}"
                    )
                    self.config.log.train.info(log)
        return global_step

    def __fit_discriminator(
        self,
        fold_index: int,
        epoch_index: int,
        dl: DataLoader,
        batch_size: int,
        global_step: int,
        loss_watcher: LossWatcher,
    ):
        self.generator.eval()
        self.discriminator.train()
        loss_function = nn.BCELoss()
        with tqdm(dl, desc=f"[Epoch {epoch_index:3d} | Discriminator | Batch {0:3d}]", leave=False) as batch_it:
            for batch_index, x_images in enumerate(batch_it):
                # Generated batch
                x = torch.rand(batch_size, 100, dtype=torch.float32).to(self.config.train.device)
                y = torch.zeros(batch_size, dtype=torch.float32).to(self.config.train.device)

                with torch.no_grad():
                    generated_images = self.generator(x)

                out = self.discriminator(generated_images)
                probs = torch.sigmoid(out)

                loss = loss_function(probs, y)
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

                # put logs
                tb_writer.add_scalar("discriminator train loss - generated", loss.item(), global_step)
                loss_watcher.put(loss.item())
                batch_it.set_description(
                    f"[Epoch {epoch_index:3d} | Generator |  Batch {batch_index:3d}] Loss: {loss.item():.3f}"
                )

                # Dataset Batch
                y = torch.ones(batch_size, dtype=torch.float32).to(self.config.train.device)
                out = self.discriminator(x_images)
                probs = torch.sigmoid(out)

                loss = loss_function(probs, y)
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                self.discriminator_optimizer.step()

                # global step for summary writer
                global_step += 1

                if batch_index > 0 and batch_index % self.config.train.logging_per_batch == 0:
                    log = (
                        f"[Fold {fold_index:02d}"
                        + f" | Epoch {epoch_index:03d}"
                        + " | Generator"
                        + f" | Batch {batch_index:05d}/{len(dl):05d} ({(batch_index/len(dl)) * 100.0:.2f}%)"
                        + f"] Loss:{loss.item():.3f}"
                    )
                    self.config.log.train.info(log)

        return global_step

    def fit(self):
        # configuration
        self.discriminator.train().to(self.config.train.device)
        self.generator.train().to(self.config.train.device)
        self.dataset.to_train()
        global_step = 0
        k_folds = self.config.train.k_folds
        epochs = self.config.train.epochs
        batch_size = self.config.train.batch_size
        tensorboard_dir = self.config.log.log_dir / "tensorboard" / f'exp_{self.config.now().strftime("%Y%m%d-%H%M%S")}'

        kfold = KFold(n_splits=k_folds, shuffle=True)
        with tqdm(enumerate(kfold.split(self.dataset)), total=k_folds, desc=f"[Fold {0:2d}]") as fold_it:
            for fold_index, train_indices in fold_it:
                fold_it.set_description(f"[Fold {fold_index:2d}]")

                # prepare dataloader
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
                train_dl = DataLoader(
                    self.dataset,
                    batch_size=batch_size,
                    sampler=train_subsampler,
                    collate_fn=self.dataset.collate_fn,
                )

                # initialize learning rate
                self.optimizer.param_groups[0]["lr"] = self.config.train.lr

                # Epoch roop
                with tqdm(
                    range(epochs), total=epochs, desc=f"[Fold {fold_index:2d} | Epoch   0]", leave=False
                ) as epoch_it:
                    for epoch_index in epoch_it:
                        generator_loss_watcher = LossWatcher("generator_loss")
                        discriminator_loss_watcher = LossWatcher("discriminator_loss")

                        global_step = self.__fit_generator(
                            fold_index,
                            epoch_index,
                            len(train_dl),
                            batch_size,
                            global_step,
                            generator_loss_watcher,
                            tb_writer,
                        )

                        global_step = self.__fit_discriminator(
                            fold_index,
                            epoch_index,
                            train_dl,
                            batch_size,
                            global_step,
                            generator_loss_watcher,
                            tb_writer,
                        )

                        # step scheduler
                        self.lr_scheduler.step()

                        # update iteration description
                        desc = (
                            f"[Fold {fold_index:2d} | Epoch {epoch_index:3d}]"
                            + f" Generator Loss: {generator_loss_watcher.mean:.5f}"
                            + f" | Discriminator Loss:{discriminator_loss_watcher.mean:.5f}"
                        )
                        epoch_it.set_description(desc)

                        # logging
                        last_lr = self.lr_scheduler.get_last_lr()[0]
                        log = (
                            f"[Fold {fold_index:2d} | Epoch {epoch_index:3d}]"
                            + f" Generator Loss: {generator_loss_watcher.mean:.5f}"
                            + f" | Discriminator Loss:{discriminator_loss_watcher.mean:.5f}"
                            + f" | LR: {last_lr:.7f}"
                        )
                        self.config.log.train.info(log)

                        # save best model
                        if generator_loss_watcher.is_best or discriminator_loss_watcher.is_best:
                            self.save_model(f"{self.name}_best.pt")

                        # save model regularly
                        if epoch_index % 5 == 0:
                            self.save_model(f"{self.name}_f{fold_index}e{epoch_index}.pt")

                # end of Epoch
                self.save_model(f"{self.name}_last_f{fold_index}.pt")

        # end of k-fold
        self.config.backup_logs()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from datetime import datetime
import os
import shutil
import numpy as np
import copy


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield tuple(t.to(self.device) for t in b)

    def __len__(self):
        """Return the number of batches"""
        return len(self.dataloader)


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 batch_size=16,
                 val_batch_size=32,
                 val_subset_size=1000,
                 lr=0.001,
                 weight_decay=0.0,
                 snapshot_interval=1000,
                 snapshot_path='snapshots',
                 val_interval=1000,
                 gradient_clipping=None):

        self.model = model
        self.device = next(model.parameters()).device

        # Store training parameters
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_subset_size = val_subset_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.snapshot_interval = snapshot_interval
        self.snapshot_path = snapshot_path
        self.val_interval = val_interval
        self.gradient_clipping = gradient_clipping

        # Initialize training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.start_time = None
        self.loss_history = []  # Add this line to initialize loss_history
        
        # create data loaders with generator for reproducibility
        # Create data loaders with generator for reproducibility
        generator = torch.Generator().manual_seed(42)  # Change torch.generator() to torch.Generator()
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        print(f"\nsplitting dataset:")
        print(f"  total size: {len(dataset)}")
        print(f"  train size: {train_size}")
        print(f"  val size: {val_size}")

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator
        )

        self.val_dataloader = dataloader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=false,
            num_workers=0
        )

        print(f"\ndataloaders created:")
        print(f"  training batches: {len(self.dataloader)}")
        print(f"  validation batches: {len(self.val_dataloader)}")

        # initialize optimizer and scheduler
        self.optimizer = optim.adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.reducelronplateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=true
        )

        # ensure model is in the right format for mps
        if self.device.type == 'mps':
            print("\nconfiguring model for mps device...")
            self.model = self.model.to(dtype=torch.float32)
            # force cpu for operations that might be unstable on mps
            self.use_cpu_validation = true
        else:
            self.use_cpu_validation = false

    def train(self, epochs=10, resume_from=None):
        """Train the model."""
        self.start_time = datetime.now()
        print(f"\nStarting training at {self.start_time}")
        print(f"Training for {epochs} epochs")
        print(f"training parameters:")
        print(f"  batch size: {self.batch_size}")
        print(f"  learning rate: {self.lr:.2e}")
        print(f"  validation interval: {self.val_interval} steps")
        print(f"  snapshot interval: {self.snapshot_interval} steps")
        print(f"  gradient clipping: {self.gradient_clipping}")
        print(f"  device: {self.device}\n")

        for epoch in range(self.current_epoch, epochs):
            self.model.train()

            for batch_idx, (x, target) in enumerate(self.dataloader):
                # move to device and ensure correct dtype
                x = x.to(self.device, dtype=torch.float32)
                target = target.to(self.device, dtype=torch.long)

                # forward pass
                output = self.model(x)
                target = target.view(-1)
                loss = f.cross_entropy(output, target)

                self.loss_history.append(loss.item())

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping
                    )

                self.optimizer.step()

                # progress reporting
                if batch_idx % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    elapsed = datetime.now() - self.start_time
                    progress = batch_idx / len(self.dataloader) * 100
                    print(f"\repoch {epoch:>2}/{epochs-1} "
                          f"[{progress:>3.0f}%] "
                          f"- loss = {loss.item():.6f} "
                          f"(lr = {current_lr:.2e}) "
                          f"- {elapsed}", end='')

                # validation
                if self.current_step % self.val_interval == 0:
                    val_loss = self.validate()
                    self.scheduler.step(val_loss)
                    self.model.train()

                # save checkpoint
                if self.current_step % self.snapshot_interval == 0:
                    self.save_checkpoint(epoch)

                self.current_step += 1

            self.current_epoch = epoch + 1
            print(f"\ncompleted epoch {epoch}")

    def validate(self):
        """validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        print(f"\nvalidation ({len(self.val_dataloader)} batches):")
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.val_dataloader):
                try:
                    # move to device and ensure correct dtype
                    x = x.to(self.device, dtype=torch.float32)
                    target = target.to(self.device, dtype=torch.long)

                    # debug info for first batch
                    if batch_idx == 0:
                        print(f"\nfirst batch info:")
                        print(f"x: shape={x.shape}, device={x.device}, dtype={x.dtype}")
                        print(f"target: shape={target.shape}, device={target.device}, dtype={target.dtype}")
                        print(f"model: device={next(self.model.parameters()).device}\n")

                    # forward pass
                    output = self.model(x)
                    target = target.view(-1)
                    loss = f.cross_entropy(output, target)

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 5 == 0:
                        print(f"\rbatch {batch_idx:>4}/{len(self.val_dataloader)} "
                              f"[{batch_idx/len(self.val_dataloader)*100:>3.0f}%] "
                              f"- current loss = {loss.item():.7f}", end='')

                except exception as e:
                    print(f"\nerror in validation batch {batch_idx}:")
                    print(f"x shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
                    print(f"target shape: {target.shape}, device: {target.device}, dtype: {target.dtype}")
                    print(f"model device: {next(self.model.parameters()).device}")
                    raise e

        avg_loss = total_loss / num_batches
        print(f"\nvalidation loss: {avg_loss:.6f}")

        # save if best
        if avg_loss < self.best_val_loss:
            print("new best validation loss!")
            self.best_val_loss = avg_loss
            self.save_checkpoint(self.current_epoch, is_best=true)

        return avg_loss


    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """Save a checkpoint of the model."""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pt'

        path = os.path.join(self.snapshot_path, filename)

        # Ensure directory exists
        os.makedirs(self.snapshot_path, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.loss_history[-1] if len(self.loss_history) > 0 else None,  # Check if loss_history has items
            'best_val_loss': self.best_val_loss,
            'current_step': self.current_step
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # If this is the best model, save a copy
        if is_best:
            best_path = os.path.join(self.snapshot_path, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_step = checkpoint.get('current_step', 0)
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

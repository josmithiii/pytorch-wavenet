import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import copy
import time
import datetime  # Proper import for datetime
import os
import shutil


def print_last_loss(opt):
    """Print the last loss value."""
    print(f"Loss: {opt.loss_history[-1]:.6f}")


def print_last_validation_result(opt):
    """Print the last validation result."""
    print(f"Validation Loss: {opt.validation_results[-1]:.6f}")


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
        
        # Create data loaders with generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        print(f"\nSplitting dataset:")
        print(f"  Total size: {len(dataset)}")
        print(f"  Train size: {train_size}")
        print(f"  Val size: {val_size}")

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

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"\nDataloaders created:")
        print(f"  Training batches: {len(self.dataloader)}")
        print(f"  Validation batches: {len(self.val_dataloader)}")

        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )

        # Ensure model is in the right format for MPS
        if self.device.type == 'mps':
            print("\nConfiguring model for MPS device...")
            self.model = self.model.to(dtype=torch.float32)
            # Force CPU for operations that might be unstable on MPS
            self.use_cpu_validation = True
        else:
            self.use_cpu_validation = False

    def train(self, epochs=10, resume_from=None):
        """Train the model for the specified number of epochs."""
        # Initialize training state
        self.start_time = time.time()
        print(f"\nStarting training at {datetime.datetime.now()}")
        print(f"Training for {epochs} epochs")
        print(f"Training parameters:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr:.2e}")
        print(f"  Validation interval: {self.val_interval} steps")
        print(f"  Snapshot interval: {self.snapshot_interval} steps")
        print(f"  Gradient clipping: {self.gradient_clipping}")
        print(f"  Device: {self.device}")
        print()

        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resuming from checkpoint: {resume_from}")
            print(f"Current epoch: {self.current_epoch}")
            print(f"Current step: {self.current_step}")
            print()

        # Track shape mismatch warnings
        shape_warning_count = 0
        max_shape_warnings = 3  # Only show first 3 warnings
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            self.model.train()

            for batch_idx, (x, target) in enumerate(self.dataloader):
                # Move to device and ensure correct dtype
                x = x.to(self.device, dtype=torch.float32)
                target = target.to(self.device, dtype=torch.long)

                # Forward pass
                output = self.model(x)
                
                # Reshape target to match output
                target = target.reshape(-1)
                
                # Check if shapes match, if not, adjust output
                if output.size(0) != target.size(0):
                    if shape_warning_count < max_shape_warnings:
                        print(f"Warning: Output shape {output.size()} doesn't match target shape {target.shape}")
                        # Truncate output to match target size
                        if output.size(0) > target.size(0):
                            output = output[:target.size(0)]
                        # Or pad target to match output size
                        else:
                            # This is a fallback, but it's better to fix the model's forward method
                            target = target[:output.size(0)]
                        print(f"Adjusted shapes - Output: {output.size()}, Target: {target.size()}")
                    elif shape_warning_count == max_shape_warnings:
                        print("Suppressing further shape mismatch warnings...")
                    shape_warning_count += 1
                    
                    # Always adjust the shapes even if we don't print warnings
                    if output.size(0) > target.size(0):
                        output = output[:target.size(0)]
                    else:
                        target = target[:output.size(0)]
                
                loss = F.cross_entropy(output, target)

                self.loss_history.append(loss.item())

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clipping
                    )

                self.optimizer.step()

                # Progress reporting
                if batch_idx % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - self.start_time
                    progress = batch_idx / len(self.dataloader) * 100
                    print(f"\rEpoch {epoch:>2}/{epochs-1} "
                          f"[{progress:>3.0f}%] "
                          f"- Loss = {loss.item():.6f} "
                          f"(lr = {current_lr:.2e}) "
                          f"- {elapsed}", end='')

                # Validation
                if self.current_step % self.val_interval == 0:
                    val_loss = self.validate()
                    self.scheduler.step(val_loss)
                    self.model.train()

                # Save checkpoint
                if self.current_step % self.snapshot_interval == 0:
                    self.save_checkpoint(epoch)

                self.current_step += 1

            self.current_epoch = epoch + 1
            print(f"\nCompleted epoch {epoch}")

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Track shape mismatch warnings
        shape_warning_count = 0
        max_shape_warnings = 2  # Only show first 2 warnings

        print(f"\nValidation ({len(self.val_dataloader)} batches):")
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.val_dataloader):
                try:
                    # Move to device and ensure correct dtype
                    x = x.to(self.device, dtype=torch.float32)
                    target = target.to(self.device, dtype=torch.long)

                    # Debug info for first batch
                    if batch_idx == 0:
                        print(f"\nFirst batch info:")
                        print(f"x: shape={x.shape}, device={x.device}, dtype={x.dtype}")
                        print(f"target: shape={target.shape}, device={target.device}, dtype={target.dtype}")
                        print(f"model: device={next(self.model.parameters()).device}\n")

                    # Forward pass
                    output = self.model(x)
                    target = target.reshape(-1)
                    
                    # Check if shapes match, if not, adjust output
                    if output.size(0) != target.size(0):
                        if shape_warning_count < max_shape_warnings:
                            print(f"Warning: Validation output shape {output.size()} doesn't match target shape {target.shape}")
                            # Truncate output to match target size
                            if output.size(0) > target.size(0):
                                output = output[:target.size(0)]
                            # Or pad target to match output size (less ideal)
                            else:
                                # This is a fallback, but it's better to fix the model's forward method
                                target = target[:output.size(0)]
                            print(f"Adjusted validation shapes - Output: {output.size()}, Target: {target.size()}")
                        elif shape_warning_count == max_shape_warnings:
                            print("Suppressing further validation shape mismatch warnings...")
                        shape_warning_count += 1
                        
                        # Always adjust the shapes even if we don't print warnings
                        if output.size(0) > target.size(0):
                            output = output[:target.size(0)]
                        else:
                            target = target[:output.size(0)]
                    
                    loss = F.cross_entropy(output, target)

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 5 == 0:
                        print(f"\rBatch {batch_idx:>4}/{len(self.val_dataloader)} "
                              f"[{batch_idx/len(self.val_dataloader)*100:>3.0f}%] "
                              f"- Current Loss = {loss.item():.7f}", end='')

                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}:")
                    print(f"x shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
                    print(f"target shape: {target.shape}, device: {target.device}, dtype: {target.dtype}")
                    print(f"model device: {next(self.model.parameters()).device}")
                    raise e

        avg_loss = total_loss / num_batches
        print(f"\nValidation Loss: {avg_loss:.6f}")
        
        # Print total shape warnings if any were suppressed
        if shape_warning_count > max_shape_warnings:
            print(f"Total validation shape mismatch warnings: {shape_warning_count}")
        
        # Save if best
        if avg_loss < self.best_val_loss:
            print("New best validation loss!")
            self.best_val_loss = avg_loss
            self.save_checkpoint(self.current_epoch, is_best=True)
            
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
            'loss': self.loss_history[-1] if len(self.loss_history) > 0 else None,
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

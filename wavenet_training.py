import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from wavenet_modules import *
import os
from utils import debug_print
from tqdm import tqdm


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 lr=0.001,
                 weight_decay=0.0,
                 snapshot_path='snapshots',
                 snapshot_name='snapshot',
                 snapshot_interval=1000,
                 log_path='logs',
                 gradient_clipping=None):
        self.model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.gradient_clipping = gradient_clipping

        # Set device and dtype
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float32

        # Move model to device
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                  lr=self.lr,
                                  weight_decay=self.weight_decay)

        os.makedirs(self.snapshot_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        self.logger = Logger(log_path=log_path)

        self.current_step = 0
        self.loss_history = []

    def train(self, epochs=10):
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            
            # Create progress bar with minimal format
            pbar = tqdm(total=len(self.dataloader), 
                       desc=f"Epoch {epoch}",
                       unit='batch',
                       leave=True,
                       ncols=60,
                       bar_format='{percentage:3.0f}% | {n_fmt}/{total_fmt} [{rate_fmt}]')
            
            for (x, target) in iter(self.dataloader):
                debug_print(f"Batch shapes before transfer - x: {x.shape}, target: {target.shape}")
                debug_print(f"Batch devices before transfer - x: {x.device}, target: {target.device}")
                
                # Move tensors to device and ensure correct dtype
                x = x.to(device=self.device, dtype=self.dtype)
                target = target.to(device=self.device)
                
                debug_print(f"Batch devices after transfer - x: {x.device}, target: {target.device}")
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(x)
                debug_print(f"Output shape: {output.shape}, Target shape: {target.shape}")
                
                # Reshape output and target for loss calculation
                if output.dim() == 2:
                    target = target.view(-1)
                else:
                    output = output.permute(0, 2, 1).contiguous()
                    output = output.view(-1, output.size(-1))
                    target = target.view(-1)
                
                # Calculate loss
                loss = self.criterion(output, target)
                loss_value = loss.item()
                
                # Backward pass and optimize
                loss.backward()
                if self.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.gradient_clipping)
                self.optimizer.step()
                
                # Update progress bar without loss display
                pbar.update(1)
                
                # Log progress (less frequently)
                if self.current_step % 100 == 0:
                    # Log to file silently
                    self.logger.log_training(loss_value, self.current_step, print_to_console=False)
                    # Print loss to console once
                    print(f" - Loss = {loss_value:.6f}")
                
                self.loss_history.append(loss_value)
                
                # Save snapshot if needed
                if self.current_step > 0 and self.current_step % self.snapshot_interval == 0:
                    self.save_snapshot()
                
                self.current_step += 1
            
            pbar.close()
            print()  # Add newline after epoch

    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        for (x, target) in iter(self.dataloader):
            x = Variable(x.type(self.dtype))
            target = Variable(target.view(-1).type(self.dtype))

            output = self.model(x)
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            total_loss += loss.item()

            predictions = torch.max(output, 1)[1].view(-1)
            correct_pred = torch.eq(target, predictions)
            accurate_classifications += torch.sum(correct_pred).item()
        # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
        # print("average loss: ", total_loss / len(self.dataloader))
        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = accurate_classifications / (len(self.dataset)*self.dataset.target_length)
        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_accuracy


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    samples = np.stack(samples, axis=0)
    return samples

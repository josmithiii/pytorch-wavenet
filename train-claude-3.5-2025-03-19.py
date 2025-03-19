import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
from tqdm import tqdm

# Updated generate_audio function
def generate_audio(model, length=16000, temperature=1.0):
    """
    Generate audio samples using the trained model.
    Args:
        model: Trained WaveNet model
        length: Number of samples to generate
        temperature: Controls randomness (higher = more random, lower = more deterministic)
    """
    device = next(model.parameters()).device
    model.eval()  # Set to evaluation mode
    
    # Start with zeros and convert to one-hot encoding
    current_sample = torch.zeros(1, 256, model.receptive_field).to(device)
    generated_samples = []

    print(f"\nGenerating {length} samples...")

    with torch.no_grad():
        # We need to generate samples in chunks since the model outputs multiple samples at once
        for i in tqdm(range(0, length, model.output_length)):
            # Get model prediction
            output = model(current_sample)
            
            # Reshape output to [batch_size, output_length, num_classes]
            output = output.reshape(1, model.output_length, 256)
            
            # Apply temperature
            if temperature != 1:
                output = output / temperature

            # Process each output step
            for j in range(min(model.output_length, length - i)):
                # Get probabilities for current timestep
                probabilities = F.softmax(output[0, j], dim=0)
                
                # Sample from the output distribution
                next_sample_idx = torch.multinomial(probabilities, 1).item()
                
                # Append to generated samples
                generated_samples.append(next_sample_idx)
                
                # Create one-hot encoding for the new sample
                if j < model.output_length - 1 or i + model.output_length < length:
                    next_sample_onehot = torch.zeros(1, 256, 1).to(device)
                    next_sample_onehot[0, next_sample_idx, 0] = 1
                    
                    # Shift input window and add new sample
                    current_sample = torch.roll(current_sample, -1, dims=2)
                    current_sample[:, :, -1] = next_sample_onehot.squeeze(-1)

    # Convert to numpy array
    samples = np.array(generated_samples, dtype=np.int16)

    # Scale back to audio range
    samples = samples - 128

    return samples

def generate_and_log_samples(model, writer, step, temperature=1.0):
    """
    Generate audio samples and log them to TensorBoard.
    Args:
        model: The trained WaveNet model
        writer: TensorBoard SummaryWriter
        step: Current training step
        temperature: Controls randomness in generation
    """
    print(f"Generating audio samples at step {step}...")
    
    # Generate samples with different temperatures
    samples_t1 = generate_audio(model, length=8000, temperature=1.0)
    samples_t07 = generate_audio(model, length=8000, temperature=0.7)
    
    # Log audio to TensorBoard
    writer.add_audio(f'generated_audio/temp_1.0', samples_t1, step, sample_rate=16000)
    writer.add_audio(f'generated_audio/temp_0.7', samples_t07, step, sample_rate=16000)
    
    print(f"Audio samples generated and logged at step {step}")

# Memory dataset for better performance
class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)
        self.train = dataset.train if hasattr(dataset, 'train') else True

        # Print sample item shape during initialization
        sample_item = self.dataset[0]
        print("\nDataset sample info:")
        print(f"Sample item type: {type(sample_item)}")
        if isinstance(sample_item, tuple):
            print(f"Sample x shape: {sample_item[0].shape}")
            print(f"Sample target shape: {sample_item[1].shape}")
        else:
            print(f"Sample shape: {sample_item.shape}")

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return self.length

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def main():
    # Set up device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize model with parameters matching the checkpoint
    model = WaveNetModel(layers=6, blocks=4, dilation_channels=8,
                        residual_channels=8, skip_channels=16,
                        classes=256, output_length=8,
                        kernel_size=3, dropout_rate=0.2)
    model = model.to(device)
    
    print(f"Model receptive field: {model.receptive_field}")
    print(f"Model parameter count: {model.parameter_count()}")

    # Create dataset
    data = WavenetDataset(
        dataset_file='train_samples/bach_chaconne/dataset.npz',
        item_length=model.receptive_field + model.output_length - 1,
        target_length=model.output_length,
        file_location='train_samples/bach_chaconne',
        test_stride=20
    )
    
    # Load dataset file
    print("Loading dataset file:", data.dataset_file)
    with np.load(data.dataset_file) as dataset:
        print("Available keys in dataset:", dataset.files)
        data.data = dataset['arr_0']
    
    print(f"Dataset has {len(data)} items")
    print(f"Dataset shape: {data.data.shape}")

    # Create memory dataset
    memory_dataset = MemoryDataset(data)

    # Create trainer with improved parameters
    trainer = WavenetTrainer(
        model=model,
        dataset=memory_dataset,
        batch_size=32,  # Increased from 16
        val_batch_size=32,
        val_subset_size=1000,  # Increased from 500
        lr=0.0005,  # Reduced initial learning rate
        weight_decay=0.01,  # Added L2 regularization
        gradient_clipping=1,
        snapshot_interval=500,  # More frequent snapshots
        snapshot_path='snapshots',
        val_interval=500  # More frequent validation
    )

    # Set up TensorBoard logging
    log_dir = f'runs/wavenet_{time.strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"View training progress with: tensorboard --logdir=runs")
    
    # Generate initial samples
    try:
        generate_and_log_samples(model, writer, 0)
    except Exception as e:
        print(f"Error generating initial samples: {e}")
    
    # Training loop with TensorBoard logging
    try:
        print('\nStarting training...')
        start_time = time.time()
        
        # Custom training loop for better TensorBoard integration
        for epoch in range(8):  # 8 epochs
            print(f"\nEpoch {epoch}/7")
            epoch_loss = 0.0
            
            # Training phase
            model.train()
            for i, (x, target) in enumerate(trainer.dataloader):
                step = epoch * len(trainer.dataloader) + i
                
                # Move data to device
                x = x.to(device)
                target = target.to(device)
                
                # Forward pass
                output = model(x)
                loss = F.cross_entropy(output, target.reshape(-1))
                
                # Backward pass
                trainer.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if trainer.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.gradient_clipping)
                
                # Update weights
                trainer.optimizer.step()
                
                # Log training metrics
                if i % 100 == 0:
                    print(f"Batch {i}/{len(trainer.dataloader)} [{epoch * 100 + i * 100 // (len(trainer.dataloader) * 8):3d}%] - Loss: {loss.item():.7f}")
                    writer.add_scalar('training/loss', loss.item(), step)
                    writer.add_scalar('training/learning_rate', trainer.optimizer.param_groups[0]['lr'], step)
                
                # Validation
                if step % trainer.val_interval == 0:
                    val_loss = trainer.validate()
                    print(f"Validation Loss: {val_loss:.7f}")
                    writer.add_scalar('validation/loss', val_loss, step)
                    
                    # Update learning rate based on validation loss
                    if trainer.scheduler is not None:
                        trainer.scheduler.step(val_loss)
                    
                    # Save best model
                    if trainer.best_val_loss is None or val_loss < trainer.best_val_loss:
                        trainer.best_val_loss = val_loss
                        trainer.save_checkpoint(epoch, is_best=True)
                        print(f"New best validation loss!")
                
                # Save checkpoint
                if step % trainer.snapshot_interval == 0:
                    trainer.save_checkpoint(epoch)
                
                # Generate audio samples
                if step % 2000 == 0 and step > 0:
                    try:
                        generate_and_log_samples(model, writer, step)
                    except Exception as e:
                        print(f"Error generating samples at step {step}: {e}")
                
                # Log model parameters
                if step % 5000 == 0:
                    for name, param in model.named_parameters():
                        writer.add_histogram(f"parameters/{name}", param.data, step)
                        if param.grad is not None:
                            writer.add_histogram(f"gradients/{name}", param.grad.data, step)
            
            # End of epoch
            print(f"Epoch {epoch} completed")
            
        print(f"\nTraining completed in {time.time() - start_time:.2f} seconds")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint(epoch)
    except Exception as e:
        print(f"\nError during training:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise
    finally:
        writer.close()
        print(f"TensorBoard logs saved to: {log_dir}")
        print(f"View training results with: tensorboard --logdir=runs")

if __name__ == "__main__":
    main() 
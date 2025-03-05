# Created from demo.ipynb and debugged using Claude 3.5 Sonnet

import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from scipy.io import wavfile
import torch
import multiprocessing
import numpy as np
from utils import set_debug
import argparse
from tqdm import tqdm
import scipy.io.wavfile
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)

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
        item = self.dataset[idx]
        # Print shape occasionally for debugging
        if idx % 1000 == 0:
            if isinstance(item, tuple):
                print(f"\nBatch {idx} shapes:")
                print(f"x: {item[0].shape}")
                print(f"target: {item[1].shape}")
        return item

    def __len__(self):
        return self.length

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        print(f"DeviceDataLoader initialized with device: {device}")

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for x, target in self.dl:
            debug_print(f"Batch shapes before transfer - x: {x.shape}, target: {target.shape}")
            debug_print(f"Batch devices before transfer - x: {x.device}, target: {target.device}")

            x = x.to(self.device)
            target = target.to(self.device)

            debug_print(f"Batch devices after transfer - x: {x.device}, target: {target.device}")
            yield x, target

def generate_audio(model, device, length, temperature=1.0):
    """
    Generate audio samples using a trained WaveNet model.
    
    Args:
        model: The WaveNet model
        device: The device to use for generation
        length: Number of samples to generate
        temperature: Temperature for sampling (higher = more random)
        
    Returns:
        Generated audio samples as a numpy array
    """
    # Set model to evaluation mode
    model.eval()
    
    # Print generation parameters
    print(f"Generating {length} audio samples with temperature {temperature}")
    
    # Initialize with zeros
    current_sample = torch.zeros(1, 1, model.receptive_field, device=device)
    generated_samples = []
    
    # Generate audio samples one by one
    with torch.no_grad():
        for i in tqdm(range(length)):
            # Forward pass
            output = model(current_sample)
            
            # Get the last output (corresponds to the next sample)
            if output.size(0) > 1:
                # If output has multiple samples, take the last one
                output = output[-1:, :]
                
            # Apply temperature and sample
            output = output.div(temperature).exp()
            output = output / torch.sum(output)
            
            # Sample from the distribution
            dist = torch.distributions.Categorical(output)
            sample = dist.sample()
            
            # Save the generated sample
            generated_samples.append(sample.item())
            
            # Update the current sample
            current_sample = torch.roll(current_sample, -1, dims=2)
            current_sample[0, 0, -1] = sample.float() / 255.0
    
    # Convert to numpy array
    samples = np.array(generated_samples)
    
    return samples

def generate_and_log_samples(model, step, temperature=1.0):
    """
    Generate audio samples and prepare them for logging.
    Args:
        model: The trained WaveNet model
        step: Current training step (for logging)
        temperature: Controls randomness in generation
    Returns:
        Dictionary containing the generated audio data
    """
    samples = generate_audio(model, model.device, length=16000, temperature=temperature)
    
    data = {
        'temperature': temperature,
        'samples': samples,
        'step': step,
        'sample_rate': 16000
    }
    
    return data

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='WaveNet training and generation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='Mode: train or generate')
    parser.add_argument('--output', type=str, default='generated_audio.wav',
                        help='Output file name for generated audio')
    parser.add_argument('--length', type=int, default=16000,
                        help='Length of generated audio in samples')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for sampling (higher = more random)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--start-fresh', action='store_true',
                        help='Start training from scratch')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--model-dir', type=str, default='snapshots',
                        help='Directory to save model checkpoints')
    
    # Model architecture parameters
    parser.add_argument('--layers', type=int, default=10,
                        help='Number of layers in each block')
    parser.add_argument('--blocks', type=int, default=4,
                        help='Number of blocks in the model')
    parser.add_argument('--dilation-channels', type=int, default=32,
                        help='Number of channels in the dilated convolutions')
    parser.add_argument('--residual-channels', type=int, default=32,
                        help='Number of channels in the residual connections')
    parser.add_argument('--skip-channels', type=int, default=256,
                        help='Number of channels in the skip connections')
    parser.add_argument('--output-length', type=int, default=16,
                        help='Output length of the model')
    parser.add_argument('--dropout-rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use from dataset (for quick testing)')
    parser.add_argument('--test-stride', type=int, default=20,
                        help='Stride for dataset subsampling (higher values = smaller dataset)')
    parser.add_argument('--max-batches', type=int, default=None, 
                        help='Maximum number of batches per epoch (for quick testing)')
    
    args = parser.parse_args()
    
    # Set debug printing on/off
    set_debug(False)  # Change to True when you need debug output

    # Check MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    def get_device():
        """
        Get the best available device for training.
        
        Returns:
            torch.device: The device to use for training
        """
        # Check for CUDA first
        if torch.cuda.is_available():
            print("CUDA available: True")
            return torch.device('cuda')
        else:
            print("CUDA available: False")
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps'):
                print(f"MPS available: {torch.backends.mps.is_available()}")
                print(f"MPS built: {torch.backends.mps.is_built()}")
                if torch.backends.mps.is_available():
                    # MPS has reshape issues, use CPU instead
                    print("MPS available but using CPU to avoid reshape issues")
                    pass
            
            # Fall back to CPU
            return torch.device('cpu')

    device = get_device()
    print(f"Using device: {device}")

    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create model with consistent parameters
    model = WaveNetModel(
        layers=args.layers,
        blocks=args.blocks,
        dilation_channels=args.dilation_channels,
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels,
        output_length=args.output_length,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    print(f"Created WaveNet model with parameters:")
    print(f"  - Layers: {args.layers}")
    print(f"  - Blocks: {args.blocks}")
    print(f"  - Dilation channels: {args.dilation_channels}")
    print(f"  - Residual channels: {args.residual_channels}")
    print(f"  - Skip channels: {args.skip_channels}")
    print(f"  - Output length: {args.output_length}")
    print(f"  - Dropout rate: {args.dropout_rate}")
    print(f"Total parameters: {model.parameter_count():,}")

    # Create dataset and dataloader
    data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                         item_length=model.receptive_field + model.output_length - 1,
                         target_length=model.output_length,
                         file_location='train_samples/bach_chaconne',
                         test_stride=args.test_stride)

    print("Loading dataset file:", data.dataset_file)
    with np.load(data.dataset_file) as dataset:
        print("Available keys in dataset:", dataset.files)
        data.data = dataset['arr_0']

    print('the dataset has ' + str(len(data)) + ' items')

    print(f"Dataset type: {type(data.data)}")
    print(f"Dataset shape: {data.data.shape}")
    print(f"Dataset dtype: {data.data.dtype}")

    # Create memory dataset (with optional sample limiting)
    if args.max_samples is not None and args.max_samples < len(data):
        print(f"Limiting dataset to {args.max_samples} samples (out of {len(data)} available)")
        indices = list(range(min(args.max_samples, len(data))))
        limited_data = [data[i] for i in indices]
        memory_dataset = MemoryDataset(limited_data)
        print(f"Created limited dataset with {len(memory_dataset)} samples")
    else:
        memory_dataset = MemoryDataset(data)
        print(f"Using full dataset with {len(memory_dataset)} samples")

    # Create trainer with memory_dataset
    trainer = WavenetTrainer(
        model=model,
        dataset=memory_dataset,
        batch_size=args.batch_size,
        val_batch_size=32,
        val_subset_size=1000,  # Increased from 500
        lr=args.learning_rate,
        weight_decay=0.01,  # Added L2 regularization
        gradient_clipping=1,
        snapshot_interval=500,  # More frequent snapshots
        snapshot_path=args.model_dir,
        val_interval=500,  # More frequent validation
        max_batches_per_epoch=args.max_batches
    )

    if args.mode == 'train':
        # Set up TensorBoard logging
        log_dir = os.path.join(args.model_dir, time.strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logs will be saved to: {log_dir}")
        print("To view training progress, run:")
        print(f"tensorboard --logdir={args.model_dir}")
        
        # Set up TensorboardLogger with the writer
        logger = TensorboardLogger(
            log_interval=200,
            validation_interval=200,
            generate_interval=500,
            generate_function=generate_and_log_samples,
            log_dir=log_dir
        )
        trainer.logger = logger

        # Start training with additional error handling
        print('\nStarting training...')
        tic = time.time()
        try:
            trainer.train(epochs=args.epochs, resume_from=args.checkpoint)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            trainer.save_checkpoint(trainer.current_epoch)
            print("Checkpoint saved. You can resume training using --resume")
        except Exception as e:
            print(f"\nError during training:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Model device: {next(model.parameters()).device}")
            raise
        finally:
            toc = time.time()
            print(f"Training took {toc - tic} seconds.")
            print("\nTraining complete!")
            print("To analyze training results, run: tensorboard --logdir=runs")
            print("Then open http://localhost:6006 in your browser")
            print("\nTips:")
            print("- Use --start-fresh to train a new model from scratch")
            print("- Use --resume to continue training from a checkpoint")
            print("- Use --mode generate to generate audio with a trained model")
            writer.close()
            
    else:
        # Load best model for generation if available
        best_model_path = os.path.join(args.model_dir, 'checkpoint_epoch_0.pt')
        if os.path.exists(best_model_path):
            print(f"Loading model from: {best_model_path}")
            # Create a new model with the same architecture as the saved model
            checkpoint = torch.load(best_model_path, map_location=device)
            # Recreate the model with the same parameters
            model = WaveNetModel(layers=args.layers,
                                blocks=args.blocks,
                                dilation_channels=args.dilation_channels,
                                residual_channels=args.residual_channels,
                                skip_channels=args.skip_channels,
                                output_length=args.output_length,
                                dropout_rate=args.dropout_rate)
            model = model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from: {best_model_path}")
        else:
            print(f"Could not find model at: {best_model_path}")

        samples = generate_audio(model, device, args.length, args.temperature)

        # Convert from uint8 (0-255) to int16 for WAV file
        # Shift to center around 0 and scale to int16 range
        samples_int16 = (samples.astype(np.int16) - 128) * 256

        # Save as WAV file
        scipy.io.wavfile.write(args.output, 16000, samples_int16)
        print(f"\nGenerated audio saved to: {args.output}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

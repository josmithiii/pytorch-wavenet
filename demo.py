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

def generate_audio(model, device, length=16000, temperature=1.0):
    """
    Generate audio samples using the trained model.
    Args:
        model: Trained WaveNet model
        device: Computation device (mps/cuda/cpu)
        length: Number of samples to generate
        temperature: Controls randomness (higher = more random, lower = more deterministic)
    """
    model.eval()  # Set to evaluation mode

    # Start with zeros
    current_sample = torch.zeros(1, 1, model.receptive_field).to(device)
    generated_samples = []

    print(f"\nGenerating {length} samples...")

    with torch.no_grad():
        for i in tqdm(range(length)):
            # Get model prediction
            output = model(current_sample)

            # Apply temperature
            if temperature != 1:
                output = output / temperature

            # Sample from the output distribution
            probabilities = F.softmax(output[:, :, -1], dim=1)
            next_sample = torch.multinomial(probabilities, 1)

            # Append to generated samples
            generated_samples.append(next_sample.item())

            # Shift input window and add new sample
            current_sample = torch.roll(current_sample, -1, dims=2)
            current_sample[0, 0, -1] = next_sample

    # Convert to numpy array
    samples = np.array(generated_samples, dtype=np.int16)

    # Scale back to audio range
    samples = samples - 2**15

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
    # Set debug printing on/off
    set_debug(False)  # Change to True when you need debug output

    # Check MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device = get_device()
    print(f"Using device: {device}")

    # Set up model
    model = WaveNetModel(layers=6,
                        blocks=4,
                        dilation_channels=16,
                        residual_channels=16,
                        skip_channels=32,
                        output_length=8,
                        bias=False)
    model = load_latest_model_from('snapshots', use_cuda=False)

    # Ensure model is on correct device
    model = model.to(device)
    print(f"Model device after transfer: {next(model.parameters()).device}")
    print(f"Start conv weight device: {model.start_conv.weight.device}")

    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())

    # Create dataset and dataloader (rest of the setup remains the same)
    data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                         item_length=model.receptive_field + model.output_length - 1,
                         target_length=model.output_length,
                         file_location='train_samples/bach_chaconne',
                         test_stride=20)

    print("Loading dataset file:", data.dataset_file)
    with np.load(data.dataset_file) as dataset:
        print("Available keys in dataset:", dataset.files)
        data.data = dataset['arr_0']

    print('the dataset has ' + str(len(data)) + ' items')

    print(f"Dataset type: {type(data.data)}")
    print(f"Dataset shape: {data.data.shape}")
    print(f"Dataset dtype: {data.data.dtype}")

    # Create memory dataset
    memory_dataset = MemoryDataset(data)

    # Create trainer with memory_dataset instead of dataset
    trainer = WavenetTrainer(
        model=model,
        dataset=memory_dataset,
        batch_size=16, # was 8
        val_batch_size=32,
        val_subset_size=500,
        lr=0.001,
        snapshot_interval=1000,
        val_interval=1000,
        gradient_clipping=1,
        num_workers=4
    )

    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'generate'], default='train')
    parser.add_argument('--output', default='generated_audio.wav')
    parser.add_argument('--length', type=int, default=16000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--resume', help='path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--tensorboard-dir', default='runs', help='TensorBoard log directory')
    args = parser.parse_args()

    if args.mode == 'train':
        # Set up TensorBoard logging
        log_dir = os.path.join(args.tensorboard_dir, time.strftime('%Y%m%d-%H%M%S'))
        writer = SummaryWriter(log_dir)
        print(f"\nTensorBoard logs will be saved to: {log_dir}")
        print("To view training progress, run:")
        print(f"tensorboard --logdir={args.tensorboard_dir}")
        
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
            trainer.train(epochs=args.epochs, resume_from=args.resume)
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
            print('Training took {} seconds.'.format(toc - tic))
            writer.close()
            
        print("\nTraining complete!")
        print(f"To analyze training results, run: tensorboard --logdir={args.tensorboard_dir}")
        print("Then open http://localhost:6006 in your browser")

    else:
        # Load best model for generation if available
        best_model_path = os.path.join('snapshots', 'best_model.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from: {best_model_path}")

        samples = generate_audio(model, device, args.length, args.temperature)

        # Save as WAV file
        scipy.io.wavfile.write(args.output, 16000, samples)
        print(f"\nGenerated audio saved to: {args.output}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

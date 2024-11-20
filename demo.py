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

class MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.length = len(dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

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

    # Create trainer
    trainer = WavenetTrainer(model=model,
                            dataset=memory_dataset,
                            lr=0.001,
                            weight_decay=0.0,
                            gradient_clipping=None,
                            snapshot_path='snapshots',
                            snapshot_name='saber_model',
                            snapshot_interval=100000)

    # Create dataloader with debug wrapper
    base_dataloader = torch.utils.data.DataLoader(
        dataset=memory_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Set to 0 for debugging
        persistent_workers=False
    )
    trainer.dataloader = DeviceDataLoader(base_dataloader, device)

    print("dataloader length: ", len(trainer.dataloader))
    print("dataset length:", len(memory_dataset))

    # Start training with additional error handling
    print('\nStarting training...')
    tic = time.time()
    try:
        trainer.train(epochs=20)
    except Exception as e:
        print(f"\nError during training:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Model device: {next(model.parameters()).device}")
        raise
    toc = time.time()
    print('Training took {} seconds.'.format(toc - tic))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

import time
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from scipy.io import wavfile
import torch
import multiprocessing
import numpy as np

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
            yield x.to(self.device), target.to(self.device)

def main():
    # Force CPU
    device = "cpu"
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
    model = model.to(device)

    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())

    # Create dataset
    data = WavenetDataset(dataset_file='train_samples/bach_chaconne/dataset.npz',
                         item_length=model.receptive_field + model.output_length - 1,
                         target_length=model.output_length,
                         file_location='train_samples/bach_chaconne',
                         test_stride=20)

    # Load data into memory
    print("Loading dataset file:", data.dataset_file)
    with np.load(data.dataset_file) as dataset:
        print("Available keys in dataset:", dataset.files)
        data.data = dataset['arr_0']

    print('the dataset has ' + str(len(data)) + ' items')

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

    # Create dataloader
    base_dataloader = torch.utils.data.DataLoader(
        dataset=memory_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        persistent_workers=True
    )
    trainer.dataloader = DeviceDataLoader(base_dataloader, device)

    print("dataloader length: ", len(trainer.dataloader))
    print("dataset length:", len(memory_dataset))

    # Start training
    print('\nStarting training...')
    tic = time.time()
    trainer.train(epochs=20)
    toc = time.time()
    print('Training took {} seconds.'.format(toc - tic))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

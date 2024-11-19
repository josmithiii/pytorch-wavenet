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
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.length = len(dataset)
        self.device = device
    
    def __getitem__(self, idx):
        x, target = self.dataset[idx]
        return (x.to(self.device), target.to(self.device))
    
    def __len__(self):
        return self.length

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for b in self.dl:
            yield tuple(t.to(self.device) for t in b)

def main():
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device = get_device()
    print(f"Using device: {device}")

    # Set up model and move to device
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

    # Create memory dataset with device
    memory_dataset = MemoryDataset(data, device)

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
        num_workers=0,  # Set back to 0 temporarily for debugging
        persistent_workers=False
    )
    trainer.dataloader = DeviceDataLoader(base_dataloader, device)

    print("dataloader length: ", len(trainer.dataloader))
    print("dataset length:", len(memory_dataset))

    # Start training
    print('start training...')
    tic = time.time()
    trainer.train(batch_size=8,
                 epochs=20)
    toc = time.time()
    print('Training took {} seconds.'.format(toc - tic))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main() 
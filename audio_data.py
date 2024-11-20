import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
from utils import debug_print


class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 target_length,
                 file_location=None,
                 test_stride=None):

        self.dataset_file = dataset_file
        self.item_length = item_length
        self.target_length = target_length
        # Load data with explicit type conversion
        with np.load(self.dataset_file) as dataset:
            raw_data = dataset['arr_0']
            print(f"Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
            # Convert to the correct type and ensure it's contiguous
            self.data = np.ascontiguousarray(raw_data, dtype=np.int64)
        print(f"Converted data shape: {self.data.shape}, dtype: {self.data.dtype}")
        self.file_location = file_location
        self._test_stride = test_stride

    def __getitem__(self, idx):
        if self._test_stride is not None:
            idx = idx * self._test_stride

        start_pos = idx
        end_pos = idx + self.item_length
        target_pos = idx + (self.item_length - self.target_length)
        target_end_pos = target_pos + self.target_length

        try:
            # Get samples
            sample = self.data[start_pos:end_pos]
            target_sample = self.data[target_pos:target_end_pos]

            # One-hot encode the input (256 channels for 8-bit audio)
            sample_one_hot = one_hot_encode(sample, channels=256)

            # Convert to torch tensors
            example = torch.tensor(sample_one_hot, dtype=torch.float32)
            target = torch.tensor(target_sample, dtype=torch.long)

            # Reshape example to [channels, time] format
            example = example.transpose(0, 1)  # [time, channels] -> [channels, time]

            # Ensure target matches the expected output size
            target = target.view(-1)  # Flatten target

            debug_print(f"Dataset item shapes - example: {example.shape}, target: {target.shape}")
            return example, target
        except Exception as e:
            print(f"Error in __getitem__: {str(e)}")
            print(f"idx: {idx}, start_pos: {start_pos}, end_pos: {end_pos}")
            raise

    def __len__(self):
        if self._test_stride is not None:
            return (len(self.data) - self.item_length) // self._test_stride
        return len(self.data) - self.item_length

    def get_example_file(self, idx):
        if self.file_location is None:
            return None
        files = os.listdir(self.file_location)
        wav_files = [f for f in files if f.endswith('.wav')]
        if idx >= len(wav_files):
            return None
        return os.path.join(self.file_location, wav_files[idx])


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def list_all_audio_files(location):
    audio_files = []
    for dirpath, dirnames, filenames in os.walk(location):
        for filename in [f for f in filenames if f.endswith((".mp3", ".wav", ".aif", "aiff"))]:
            audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print("found no audio files in " + location)
    return audio_files


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def one_hot_encode(data, channels=256):
    """Convert data to one-hot encoded format."""
    data = np.asarray(data, dtype=np.int64)
    one_hot = np.zeros((data.size, channels), dtype=np.float32)
    one_hot[np.arange(data.size), data.ravel()] = 1
    return one_hot


def one_hot_decode(data):
    """Convert one-hot encoded data back to indices."""
    return np.argmax(data, axis=1)

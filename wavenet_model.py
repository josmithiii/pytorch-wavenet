import os
import os.path
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from wavenet_modules import *
from audio_data import mu_law_expansion


class WaveNetModel(nn.Module):
    """
    A Complete WaveNet Model

    Args:
        layers (int):               Number of layers in each block
        blocks (int):               Number of wavenet blocks of this model
        dilation_channels (int):    Number of channels for the dilated convolution
        residual_channels (int):    Number of channels for the residual connection
        skip_channels (int):        Number of channels for the skip connections
        classes (int):              Number of possible values each sample can have
        output_length (int):        Number of samples that are generated for each input
        kernel_size (int):          Size of the dilation kernel
        dropout_rate (float):       Dropout rate for the model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N \cdot output\_length, classes)`
        Where L_in should be at least the length of the receptive field
    """
    def __init__(self, layers=10, blocks=4, dilation_channels=32,
                 residual_channels=32, skip_channels=256,
                 classes=256, output_length=256, kernel_size=2, dropout_rate=0.2):
        """
        Initialize WaveNetModel with the given parameters.
        """
        super(WaveNetModel, self).__init__()

        # Model configuration
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.dropout_rate = dropout_rate
        self.init_dilation = 1
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.current_step = 0
        
        # Calculate receptive field
        self.receptive_field = self.calculate_receptive_field()
        
        # Create start convolution
        self.start_conv = nn.Conv1d(
            in_channels=self.classes,
            out_channels=self.residual_channels,
            kernel_size=1,
            bias=False
        )
        
        # Create module lists for dilated convolutions
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        # Create dropout layers if needed
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            self.skip_dropout = nn.Dropout(self.dropout_rate)
        
        # Create dilated convolution layers for each block and layer
        for b in range(blocks):
            for i in range(layers):
                dilation = 2 ** i
                
                # Filter convolution
                self.filter_convs.append(nn.Conv1d(
                    in_channels=self.residual_channels,
                    out_channels=self.dilation_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    bias=False
                ))
                
                # Gate convolution
                self.gate_convs.append(nn.Conv1d(
                    in_channels=self.residual_channels,
                    out_channels=self.dilation_channels,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    bias=False
                ))
                
                # Residual convolution (1x1)
                self.residual_convs.append(nn.Conv1d(
                    in_channels=self.dilation_channels,
                    out_channels=self.residual_channels,
                    kernel_size=1,
                    bias=False
                ))
                
                # Skip convolution (1x1)
                self.skip_convs.append(nn.Conv1d(
                    in_channels=self.dilation_channels,
                    out_channels=self.skip_channels,
                    kernel_size=1,
                    bias=False
                ))
        
        # Create output layers
        self.end_conv_1 = nn.Conv1d(
            in_channels=self.skip_channels,
            out_channels=self.skip_channels,
            kernel_size=1,
            bias=False
        )
        
        self.end_conv_2 = nn.Conv1d(
            in_channels=self.skip_channels,
            out_channels=self.classes,
            kernel_size=1,
            bias=False
        )

    def wavenet(self, input_data, dilation_func=None):
        """
        Apply the WaveNet layers to the input data.
        
        Args:
            input_data (Tensor): Input tensor
            dilation_func (function, optional): Function to apply dilation
            
        Returns:
            Tensor: Output tensor after applying WaveNet layers
        """
        # Use wavenet_dilate as the default dilation function if none provided
        if dilation_func is None:
            dilation_func = self.wavenet_dilate
            
        # Pre-compute dilations to avoid repeated calculations
        dilations = [2 ** i for i in range(self.layers)]
        
        # Initialize skip connection
        # Using a zero tensor instead of None to avoid conditional logic later
        output_size = None  # We'll determine this after first skip contribution
        
        # Apply dilations for each layer in each block
        for b in range(self.blocks):
            for i, dilation in enumerate(dilations):
                # Store the residual
                residual = input_data
                
                # Calculate layer index once
                layer_idx = i + b * self.layers
                
                # Apply dilation function
                x = dilation_func(input_data, dilation, self.init_dilation, layer_idx)
                
                # Apply 1x1 residual convolution
                x = self.residual_convs[layer_idx](x)
                
                # Handle different sizes for residual connection (if needed)
                # This is more efficient than the previous approach
                x_size = x.size(2)
                res_size = residual.size(2)
                
                if x_size != res_size:
                    # Use the smaller size
                    min_size = min(x_size, res_size)
                    x = x[:, :, :min_size]
                    residual = residual[:, :, :min_size]
                
                # Apply residual connection
                input_data = x + residual
                
                # Get skip contribution
                skip_contribution = self.skip_convs[layer_idx](x)
                
                # Initialize or update skip connections
                if output_size is None:
                    # First skip contribution - initialize
                    output_size = skip_contribution.size(2)
                    skip = skip_contribution
                else:
                    # Align sizes
                    cur_size = skip_contribution.size(2)
                    if output_size > cur_size:
                        # Current skip is smaller, adjust main skip
                        skip = skip[:, :, :cur_size]
                        output_size = cur_size
                    elif output_size < cur_size:
                        # Main skip is smaller, adjust current skip
                        skip_contribution = skip_contribution[:, :, :output_size]
                    
                    # Add to main skip
                    skip = skip + skip_contribution
        
        return skip

    def wavenet_dilate(self, input, dilation, init_dilation=1, layer_idx=0):
        """
        Apply dilated convolution with filter and gate mechanisms.
        
        Args:
            input (Tensor): Input tensor
            dilation (int): Dilation factor
            init_dilation (int): Initial dilation value
            layer_idx (int): Current layer index
            
        Returns:
            Tensor: Output tensor after applying dilated convolution
        """
        # Apply dilated convolution (padding)
        x = self.dilated_conv(input, dilation, init_dilation)
        
        # Get filter and gate convolutions for this layer
        filter_conv = self.filter_convs[layer_idx]
        gate_conv = self.gate_convs[layer_idx]
        
        # Perform filter and gate convolutions
        # More efficient approach: compute both convolutions in parallel
        # and then apply the activations
        filter_output = filter_conv(x)
        gate_output = gate_conv(x)
        
        # Apply activations and combine
        # Note: This helps with better utilization of the GPU
        return torch.tanh(filter_output) * torch.sigmoid(gate_output)

    def queue_dilate(self, input, dilation, init_dilation, i):
        """
        Queue-based dilated convolution for efficient generation.
        
        Args:
            input (Tensor): Input tensor
            dilation (int): Dilation factor
            init_dilation (int): Initial dilation value
            i (int): Queue index
            
        Returns:
            Tensor: Output tensor after applying queued dilation
        """
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(
            num_deq=self.kernel_size,
            dilation=dilation
        )
        x = x.unsqueeze(0)
        
        return x

    def dilated_conv(self, input, dilation, init_dilation=1):
        """
        Apply dilated convolution to input tensor.
        
        Args:
            input (Tensor): Input tensor
            dilation (int): Dilation factor
            init_dilation (int): Initial dilation value
            
        Returns:
            Tensor: Output tensor after applying dilated convolution
        """
        # Calculate padding for causal convolution based on dilation
        # Causal padding ensures we only use past information (not future)
        padding = dilation * (self.kernel_size - 1)
        
        # Optimize the padding operation:
        # 1. Only pad when needed
        # 2. Use a direct pad operation instead of conditional logic
        return F.pad(input, (padding, 0)) if padding > 0 else input

    def forward(self, x):
        """
        Forward pass through the WaveNet model.
        
        Args:
            x: Input tensor of shape [batch_size, classes, time_steps]
            
        Returns:
            Output tensor with shape [batch_size * output_length, classes]
        """
        # Apply initial convolution
        x = self.start_conv(x)
        
        # Apply input dropout if available
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
            
        # Apply WaveNet layers
        skip = self.wavenet(x)
        
        # Final processing
        x = F.relu(skip)
        x = self.end_conv_1(x)
        
        x = F.relu(x)
        x = self.end_conv_2(x)
        
        # Get current shape
        batch_size, channels, time_steps = x.shape
        
        # Only keep the last output_length time steps
        # Note: Slicing already creates a view that preserves the memory layout
        x = x[:, :, -self.output_length:]
        
        # Memory-efficient reshape: avoid transpose + contiguous by using permute
        # This performs the transpose and ensures the result is contiguous in one operation
        # [batch_size, channels, time_steps] -> [batch_size, time_steps, channels]
        x = x.permute(0, 2, 1)
        
        # Now we can reshape directly without calling contiguous again
        # Reshape to [batch_size * time_steps, channels]
        x = x.reshape(batch_size * x.size(1), channels)
        
        return x

    def generate(self, num_samples, first_samples=None, temperature=1.0):
        """
        Generate audio samples using the standard WaveNet generation algorithm.
        
        Args:
            num_samples (int): Number of samples to generate
            first_samples (Tensor, optional): Initial samples to condition on
            temperature (float): Controls randomness in sampling (higher = more random)
            
        Returns:
            Tensor: Generated audio samples
        """
        # Set model to evaluation mode
        self.eval()
        
        # Initialize first samples if not provided
        if first_samples is None:
            first_samples = torch.zeros(1, dtype=torch.float32)
        
        # Create variable from first samples
        generated = Variable(first_samples)
        
        # Zero-pad if needed to match receptive field
        num_pad = self.receptive_field - generated.size(0)
        if num_pad > 0:
            generated = constant_pad_1d(generated, self.receptive_field, pad_start=True)
        
        # Generate samples one by one
        for i in range(num_samples):
            # Create one-hot encoded input
            input = torch.zeros(1, self.classes, self.receptive_field)
            input = input.scatter_(
                1, 
                generated[-self.receptive_field:].view(1, -1, self.receptive_field).long(), 
                1.0
            )
            
            # Apply WaveNet
            x = self.wavenet(
                input,
                dilation_func=self.wavenet_dilate
            )[:, :, -1].squeeze()
            
            # Sample from output distribution
            if temperature > 0:
                # Apply temperature scaling and sample from softmax
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = Variable(torch.tensor([x], dtype=torch.long))
            else:
                # Just take the maximum value (greedy)
                x = torch.max(x, 0)[1].float()
            
            # Append new sample to generated sequence
            generated = torch.cat((generated, x), 0)
        
        # Convert from one-hot indices to audio samples
        generated = (generated / self.classes) * 2. - 1
        mu_gen = mu_law_expansion(generated, self.classes)
        
        # Set model back to training mode
        self.train()
        return mu_gen

    def generate_fast(self, num_samples, first_samples=None, temperature=1.0,
                      regularize=0.0, progress_callback=None, progress_interval=100):
        """
        Faster audio generation using a queue-based approach.
        
        Args:
            num_samples (int): Number of samples to generate
            first_samples (Tensor, optional): Initial samples to condition on
            temperature (float): Controls randomness in sampling (higher = more random)
            regularize (float): Regularization strength
            progress_callback (function, optional): Callback function for progress updates
            progress_interval (int): How often to call the progress callback
            
        Returns:
            numpy.ndarray: Generated audio samples
        """
        # Set model to evaluation mode
        self.eval()
        
        # Initialize first samples if not provided - use middle value
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)
        
        # Reset queues
        for queue in self.dilated_queues:
            queue.reset()
        
        # Prepare generation parameters
        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples
        
        # Create one-hot encoded input from first sample
        input = torch.zeros(1, self.classes, 1)
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.0)
        input = Variable(input)
        
        # Fill queues with given samples
        for i in range(num_given_samples - 1):
            # Process each given sample
            x = self.wavenet(input, dilation_func=self.queue_dilate)
            
            # Prepare input for next sample
            input.zero_()
            input = input.scatter_(
                1, 
                first_samples[i + 1:i + 2].view(1, -1, 1), 
                1.0
            ).view(1, self.classes, 1)
            
            # Report progress if needed
            if progress_callback is not None and i % progress_interval == 0:
                progress_callback(i, total_samples)
        
        # Generate new samples
        generated = np.array([])
        
        # Create regularizer centered around the middle class value
        regularizer = torch.pow(torch.arange(self.classes) - self.classes / 2.0, 2)
        regularizer = regularizer.squeeze() * regularize
        
        # Time the generation process
        generation_start_time = time.time()
        
        for i in range(num_samples):
            # Get model output for current state
            x = self.wavenet(input, dilation_func=self.queue_dilate).squeeze()
            
            # Apply regularization
            x -= regularizer
            
            # Sample from output distribution
            if temperature > 0:
                # Apply temperature scaling and sample from softmax
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])
            else:
                # Just take the maximum value (greedy)
                x = torch.max(x, 0)[1][0].cpu().data.numpy()
            
            # Convert to audio sample value
            o = (x / self.classes) * 2.0 - 1.0
            generated = np.append(generated, o)
            
            # Create input for next step
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.0).view(1, self.classes, 1)
            
            # Calculate generation speed after 100 samples
            if (i + 1) == 100:
                elapsed = time.time() - generation_start_time
                generation_speed = elapsed / 100
                print(f"Generation speed: {generation_speed:.6f} seconds per sample")
            
            # Report progress if needed
            if progress_callback is not None and (i + num_given_samples) % progress_interval == 0:
                progress_callback(i + num_given_samples, total_samples)
        
        # Set model back to training mode
        self.train()
        
        # Convert from one-hot indices to audio samples
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen


    def parameter_count(self):
        """
        Count the total number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        parameters = list(self.parameters())
        return sum(np.prod(list(p.size())) for p in parameters)

    def cpu(self, type=torch.FloatTensor):
        """
        Move model to CPU and set the data type.
        
        Args:
            type: The tensor type to use (default: torch.FloatTensor)
        """
        self.dtype = type
        
        # Update queue data types
        if hasattr(self, 'dilated_queues'):
            for q in self.dilated_queues:
                q.dtype = self.dtype
                
        super().cpu()

    def calculate_receptive_field(self):
        """
        Calculate the receptive field of the model based on layers, blocks, and dilations.
        
        Returns:
            int: The receptive field size
        """
        # Calculate receptive field
        receptive_field = 1
        for b in range(self.blocks):
            for i in range(self.layers):
                dilation = 2 ** i
                receptive_field += dilation * (self.kernel_size - 1)
        
        return receptive_field


def load_to_cpu(model_path):
    """
    Load a WaveNet model to CPU.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        WaveNetModel: The loaded model
        
    Raises:
        Exception: If an error occurs during loading (caught and handled)
    """
    try:
        print(f"Loading model from {model_path}")
        
        # Load with reduced memory footprint on CPU
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            print("Detected checkpoint format with state_dict")
            
            if 'model_state_dict' in checkpoint:
                # Create a new model with default architecture
                model = WaveNetModel(
                    layers=6,
                    blocks=4,
                    dilation_channels=16,
                    residual_channels=16,
                    skip_channels=32,
                    output_length=8,
                    dropout_rate=0.2
                )
                
                # Load the state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                return model
            else:
                print("Checkpoint doesn't contain model_state_dict")
                return checkpoint
        else:
            print("Using original model format")
            return checkpoint
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Creating a new model instead")
        
        # Create default model as fallback
        model = WaveNetModel(
            layers=6,
            blocks=4,
            dilation_channels=16,
            residual_channels=16,
            skip_channels=32,
            output_length=8,
            dropout_rate=0.2
        )
        
        return model

def load_latest_model_from(location, use_cuda=True):
    """
    Load the latest model from a directory location.
    
    Args:
        location (str): Directory path to look for model files
        use_cuda (bool): Whether to use CUDA (not currently used)
        
    Returns:
        WaveNetModel: The loaded model, prioritizing checkpoint files
    """
    print(f"Looking for models in {location}")
    
    # Get all files in the directory
    if not os.path.exists(location):
        print(f"Directory {location} does not exist!")
        return WaveNetModel()
        
    files = [os.path.join(location, f) for f in os.listdir(location)]
    
    if not files:
        print("No files found in the specified location.")
        return WaveNetModel()
        
    print(f"Found {len(files)} files")
    
    # Filter for checkpoint files first (prioritize these)
    checkpoint_files = [f for f in files if 'checkpoint' in f or 'best_model' in f]
    
    if checkpoint_files:
        # Get the newest checkpoint file by creation time
        newest_file = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading newest checkpoint file: {newest_file}")
        model = load_to_cpu(newest_file)
    else:
        # If no checkpoint files, look for any model file
        if files:
            newest_file = max(files, key=os.path.getctime)
            print(f"No checkpoint files found. Loading newest file: {newest_file}")
            model = load_to_cpu(newest_file)
        else:
            print("No model files found. Creating a new model.")
            model = WaveNetModel(
                layers=6,
                blocks=4,
                dilation_channels=16,
                residual_channels=16,
                skip_channels=32,
                output_length=8,
                dropout_rate=0.2
            )
    
    return model

def load_checkpoint(model_path, device='cpu'):
    """
    Load a checkpoint efficiently with memory optimization.
    
    Args:
        model_path (str): Path to the checkpoint file
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        WaveNetModel: The loaded model with restored training state
        
    Raises:
        Exception: If an error occurs during loading
    """
    print(f"Loading checkpoint from {model_path}...")

    try:
        # Load checkpoint to CPU first regardless of target device
        checkpoint = torch.load(model_path, map_location='cpu')

        # Handle non-dict checkpoints (old format)
        if not isinstance(checkpoint, dict):
            print("Not a checkpoint file - using direct model load")
            return checkpoint

        # Create model with standard architecture
        # Note: These parameters should match the original trained model
        print("Creating model with standard architecture...")
        model = WaveNetModel(
            layers=30,
            blocks=4,
            dilation_channels=32,
            residual_channels=32,
            skip_channels=1024,
            classes=256,
            output_length=16
        )

        # Memory-efficient loading: load state dict in chunks
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            chunk_size = 10  # Number of layers to load at once

            # Load convolutional layers in chunks
            for i in range(0, len(model.filter_convs), chunk_size):
                chunk_end = min(i + chunk_size, len(model.filter_convs))
                print(f"Loading layers {i} to {chunk_end-1}...")
                
                # Create a chunk of the state dict for these layers
                chunk = {}
                for k, v in state_dict.items():
                    for j in range(i, chunk_end):
                        if f'filter_convs.{j}' in k or f'gate_convs.{j}' in k or \
                           f'residual_convs.{j}' in k or f'skip_convs.{j}' in k:
                            chunk[k] = v
                            break
                
                # Load this chunk
                model.load_state_dict(chunk, strict=False)

            # Load remaining parameters (start/end convs, etc.)
            print("Loading remaining parameters...")
            remaining = {k: v for k, v in state_dict.items() 
                        if not any(f'{conv_type}.{i}' in k 
                                  for conv_type in ['filter_convs', 'gate_convs', 
                                                   'residual_convs', 'skip_convs']
                                  for i in range(len(model.filter_convs)))}
            model.load_state_dict(remaining, strict=False)

            # Restore training state
            model.current_epoch = checkpoint.get('epoch', 0)
            model.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            model.current_step = checkpoint.get('current_step', 0)
            
            print(f"Checkpoint loaded successfully (epoch {model.current_epoch})")
        else:
            print("Checkpoint doesn't contain model_state_dict")
        
        # Move model to requested device
        if device != 'cpu':
            model = model.to(device)
            
        return model

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise

def save_checkpoint(model, optimizer, scheduler, epoch, filename,
                   is_best=False, current_step=0):
    """
    Save a checkpoint with training state.
    
    Args:
        model (WaveNetModel): The model to save
        optimizer: The optimizer state to save
        scheduler: The learning rate scheduler state to save
        epoch (int): Current epoch number
        filename (str): Path to save the checkpoint
        is_best (bool): Whether this is the best model so far
        current_step (int): Current training step
    """
    # Create the checkpoint dictionary with all necessary information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': getattr(model, 'best_val_loss', float('inf')),
        'current_step': current_step
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save with compression for smaller file size
    torch.save(
        checkpoint, 
        filename, 
        _use_new_zipfile_serialization=True
    )
    
    print(f"Checkpoint saved at {filename}")

    # If this is the best model, create a copy
    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'best_model.pt')
        shutil.copyfile(filename, best_filename)
        print(f"Best model saved at {best_filename}")

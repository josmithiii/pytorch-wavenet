import os
import os.path
import time
from wavenet_modules import *
from audio_data import *


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model
        dropout_rate (Float):       Dropout rate for the model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """
    def __init__(self, layers=10, blocks=4, dilation_channels=32,
                 residual_channels=32, skip_channels=256,
                 classes=256, output_length=256, kernel_size=2, dropout_rate=0.2):
        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.output_length = output_length
        self.dropout_rate = dropout_rate
        
        # Add init_dilation attribute
        self.init_dilation = 1
        
        # Calculate receptive field and initialize dilations
        self.receptive_field = self.calculate_receptive_field()
        
        # Create start convolution
        print("Creating start convolution...")
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                   out_channels=self.residual_channels,
                                   kernel_size=1,
                                   bias=False)
        
        # Create dilated convolution layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        # Create dropout layers only if dropout_rate > 0
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
            self.skip_dropout = nn.Dropout(self.dropout_rate)
        
        # Create dilated convolution layers for each block and layer
        for b in range(blocks):
            for i in range(layers):
                dilation = 2 ** i
                print(f"Creating dilated convolution {i}...")
                
                # Filter convolution
                self.filter_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                  out_channels=self.dilation_channels,
                                                  kernel_size=self.kernel_size,
                                                  dilation=dilation,
                                                  bias=False))
                
                # Gate convolution
                self.gate_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                out_channels=self.dilation_channels,
                                                kernel_size=self.kernel_size,
                                                dilation=dilation,
                                                bias=False))
                
                # Residual convolution (1x1)
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                    out_channels=self.residual_channels,
                                                    kernel_size=1,
                                                    bias=False))
                
                # Skip convolution (1x1)
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                out_channels=self.skip_channels,
                                                kernel_size=1,
                                                bias=False))
        
        # Create output layers
        self.end_conv_1 = nn.Conv1d(in_channels=self.skip_channels,
                                   out_channels=self.skip_channels,
                                   kernel_size=1,
                                   bias=False)
        self.end_conv_2 = nn.Conv1d(in_channels=self.skip_channels,
                                   out_channels=self.classes,
                                   kernel_size=1,
                                   bias=False)
        
        print("WaveNetModel initialization complete")

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
            
        # Initialize skip connection as None
        skip = None
        
        # Apply dilations for each layer in each block
        for b in range(self.blocks):
            for i in range(self.layers):
                # Get the current dilation
                current_dilation = 2 ** i
                
                # Store the residual
                residual = input_data
                
                # Apply dilation function with correct layer index
                layer_idx = i + b * self.layers
                x = dilation_func(input_data, current_dilation, self.init_dilation, layer_idx)
                
                # Apply 1x1 residual convolution
                residual_conv = self.residual_convs[layer_idx]
                x = residual_conv(x)
                
                # Add residual connection with size check
                if x.size(2) != residual.size(2):
                    # Adjust time dimension if needed
                    if x.size(2) < residual.size(2):
                        residual = residual[:, :, :x.size(2)]
                    else:
                        x = x[:, :, :residual.size(2)]
                
                input_data = x + residual
                
                # Apply 1x1 skip convolution and add to skip connections
                skip_conv = self.skip_convs[layer_idx]
                skip_contribution = skip_conv(x)
                
                # Add skip connection
                if skip is None:  # First skip connection
                    skip = skip_contribution
                else:
                    # Ensure dimensions match before adding
                    if skip.size(2) != skip_contribution.size(2):
                        # Adjust time dimension if needed
                        if skip.size(2) > skip_contribution.size(2):
                            skip = skip[:, :, :skip_contribution.size(2)]
                        else:
                            skip_contribution = skip_contribution[:, :, :skip.size(2)]
                    
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
        # Apply dilated convolution
        x = self.dilated_conv(input, dilation, init_dilation)
        
        # Get filter and gate convolutions for this layer
        filter_conv = self.filter_convs[layer_idx]
        gate_conv = self.gate_convs[layer_idx]
        
        # Apply filter (tanh) and gate (sigmoid) convolutions
        filter_output = torch.tanh(filter_conv(x))
        gate_output = torch.sigmoid(gate_conv(x))
        
        # Combine filter and gate outputs
        return filter_output * gate_output

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
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
        # Calculate padding based on dilation
        padding = (dilation * (self.kernel_size - 1)) // 2
        
        # Apply causal convolution padding
        if padding > 0:
            # Add padding to the left (causal)
            padded = F.pad(input, (padding, 0))
        else:
            padded = input
        
        # Return the padded input (the actual convolution is applied in wavenet_dilate)
        return padded

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
        
        # For shape debugging 
        # print(f"Output shape before reshape: {x.shape}")
        
        # Ensure tensor is contiguous and only keep the last output_length time steps
        x = x[:, :, -self.output_length:].contiguous()
        
        # Update time_steps after truncation
        batch_size, channels, time_steps = x.shape
        
        # Transpose to [batch_size, time_steps, channels]
        x = x.transpose(1, 2).contiguous()
        
        # Reshape to [batch_size * time_steps, channels]
        x = x.reshape(batch_size * time_steps, channels)
        
        # For shape debugging
        # print(f"Final output shape: {x.shape}, expected target shape: {batch_size * self.output_length}")
        
        return x

    def generate(self,
                 num_samples,
                 first_samples=None,
                 temperature=1.):
        self.eval()
        if first_samples is None:
            first_samples = self.dtype(1).zero_()
        generated = Variable(first_samples, volatile=True)

        num_pad = self.receptive_field - generated.size(0)
        if num_pad > 0:
            generated = constant_pad_1d(generated, self.scope, pad_start=True)
            print("pad zero")

        for i in range(num_samples):
            input = Variable(torch.FloatTensor(1, self.classes, self.receptive_field).zero_())
            input = input.scatter_(1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.)

            x = self.wavenet(input,
                             dilation_func=self.wavenet_dilate)[:, :, -1].squeeze()

            if temperature > 0:
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = Variable(torch.LongTensor([x]))#np.array([x])
            else:
                x = torch.max(x, 0)[1].float()

            generated = torch.cat((generated, x), 0)

        generated = (generated / self.classes) * 2. - 1
        mu_gen = mu_law_expansion(generated, self.classes)

        self.train()
        return mu_gen

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)

        # reset queues
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input.zero_()
            input = input.scatter_(1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        regularizer = torch.pow(Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            x -= regularizer

            if temperature > 0:
                # sample from softmax distribution
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.classes) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1), 1.).view(1, self.classes, 1)

            if (i+1) == 100:
                toc = time.time()
                print("one generating step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen


    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
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
    """Load model to CPU."""
    try:
        print(f"Loading model from {model_path}")
        # Load with reduced memory footprint
        checkpoint = torch.load(model_path, map_location='cpu')

        if isinstance(checkpoint, dict):
            print("Detected checkpoint format with state_dict")
            if 'model_state_dict' in checkpoint:
                # Create a new model with the same architecture
                model = WaveNetModel(layers=6,
                                    blocks=4,
                                    dilation_channels=16,
                                    residual_channels=16,
                                    skip_channels=32,
                                    output_length=8,
                                    dropout_rate=0.2)
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
        model = WaveNetModel(layers=6,
                            blocks=4,
                            dilation_channels=16,
                            residual_channels=16,
                            skip_channels=32,
                            output_length=8,
                            dropout_rate=0.2)
        return model

def load_latest_model_from(location, use_cuda=True):
    """Load the latest model from a location."""
    print(f"Looking for models in {location}")
    files = [location + "/" + f for f in os.listdir(location)]
    print(f"Found files: {files}")

    # Filter for checkpoint files first
    checkpoint_files = [f for f in files if 'checkpoint' in f or 'best_model' in f]
    
    if checkpoint_files:
        # Get the newest checkpoint file
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
            model = WaveNetModel(layers=6,
                                blocks=4,
                                dilation_channels=16,
                                residual_channels=16,
                                skip_channels=32,
                                output_length=8,
                                dropout_rate=0.2)

    return model

def load_checkpoint(model_path, device='cpu'):
    """Load a checkpoint efficiently."""
    print(f"Loading checkpoint from {model_path}...")

    try:
        # Load metadata only first
        checkpoint = torch.load(model_path, map_location='cpu')

        if not isinstance(checkpoint, dict):
            print("Not a checkpoint file - using direct model load")
            return checkpoint

        print("Creating model with matching architecture...")
        model = WaveNetModel(
            layers=30,
            blocks=4,
            dilation_channels=32,
            residual_channels=32,
            skip_channels=1024,
            classes=256,
            output_length=16
        )

        # Load state dict in chunks
        state_dict = checkpoint['model_state_dict']
        chunk_size = 10  # Number of layers to load at once

        for i in range(0, len(model.filter_convs), chunk_size):
            chunk = {k: v for k, v in state_dict.items()
                    if f'filter_convs.{i}' in k
                    or f'gate_convs.{i}' in k
                    or f'residual_convs.{i}' in k
                    or f'skip_convs.{i}' in k}
            model.load_state_dict(chunk, strict=False)
            print(f"Loaded layers {i} to {min(i+chunk_size, len(model.filter_convs))}")

        # Load remaining parameters
        remaining = {k: v for k, v in state_dict.items()
                    if not any(f'convs.{i}' in k
                    for i in range(len(model.filter_convs)))}
        model.load_state_dict(remaining, strict=False)
        print("Loaded remaining parameters")

        # Store training state
        model.current_epoch = checkpoint.get('epoch', 0)
        model.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        model.current_step = checkpoint.get('current_step', 0)

        print(f"Checkpoint loaded successfully (epoch {model.current_epoch})")
        return model

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise

def save_checkpoint(model, optimizer, scheduler, epoch, filename,
                   is_best=False, current_step=0):
    """Save a checkpoint efficiently."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': model.best_val_loss,
        'current_step': current_step
    }

    # Save with compression
    torch.save(checkpoint, filename, _use_new_zipfile_serialization=True)

    if is_best:
        best_filename = os.path.join(os.path.dirname(filename), 'best_model.pt')
        shutil.copyfile(filename, best_filename)

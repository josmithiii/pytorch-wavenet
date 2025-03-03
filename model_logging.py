import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import threading
import time

class Logger:
    def __init__(self, log_path='logs'):
        os.makedirs(log_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_path, f'training_{timestamp}.log')
        print(f'Logging to: {self.log_file}')

    def log_training(self, loss, step, print_to_console=True):
        """Log training progress to file and optionally to console."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_str = f"Loss = {loss:.6f}"
        
        # Always write to file with timestamp
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {log_str}\n")
        
        # Only print loss to console if requested
        if print_to_console:
            print(log_str)

    def log_validation(self, loss, step):
        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Validation Step {step}: Loss = {loss:.6f}"
        print(message)  # Print to console
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

class TensorboardLogger:
    def __init__(self, log_interval=200, validation_interval=200, generate_interval=500, 
                 generate_function=None, log_dir="logs"):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.generate_interval = generate_interval
        self.generate_function = generate_function
        self.last_generation_step = 0
        self.is_running = True
        
        # Start generation thread if function provided
        if generate_function is not None:
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()

    def _generation_loop(self):
        while self.is_running:
            if hasattr(self, 'current_step') and \
               self.current_step - self.last_generation_step >= self.generate_interval:
                self.generate_function(self.current_step)
                self.last_generation_step = self.current_step
            time.sleep(1)  # Check every second

    def log(self, loss, step, print_to_console=True):
        self.current_step = step
        if step % self.log_interval == 0:
            self.writer.add_scalar('Loss/train', loss, step)
            if print_to_console:
                print(f"Step {step}: Loss = {loss:.6f}")

    def log_validation(self, loss, step):
        if step % self.validation_interval == 0:
            self.writer.add_scalar('Loss/validation', loss, step)
            print(f"Validation Step {step}: Loss = {loss:.6f}")

    def audio_summary(self, tag, audio_data, step, sr=16000):
        """Log audio data to tensorboard"""
        self.writer.add_audio(tag, audio_data, step, sample_rate=sr)

    def close(self):
        self.is_running = False
        if hasattr(self, 'generation_thread'):
            self.generation_thread.join()
        self.writer.close()

    def __del__(self):
        self.close()

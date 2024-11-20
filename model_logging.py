import os
from datetime import datetime

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

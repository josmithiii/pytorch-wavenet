import os
from datetime import datetime

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(log_path, exist_ok=True)
        self.log_file = os.path.join(log_path, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        print(f"Logging to: {self.log_file}")

    def log_training(self, loss, step):
        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Step {step}: Loss = {loss:.6f}"
        print(message)  # Print to console
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def log_validation(self, loss, step):
        message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Validation Step {step}: Loss = {loss:.6f}"
        print(message)  # Print to console
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

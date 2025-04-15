import os

import numpy as np
import random
import pickle


class RBUF:
    def __init__(self, max_len=10000):
        self.data = []
        self.training_set = []
        self.validation_set = []
        self.max_len = max_len
        self.training_max = int(max_len * 0.8)
        self.validation_max = max_len - self.training_max

    def init_from_file(self, file_path):
        try:
            with open(file_path, "rb") as f:
                print("File size:", os.path.getsize(file_path))
                data = pickle.load(f)
                self.data = data[-self.max_len:] if len(data) > self.max_len else data
                split = int(0.8 * len(self.data))
                self.training_set = self.data[:split]
                self.validation_set = self.data[split:]
        except FileNotFoundError:
            print("No file found")

    def get_training_set(self, batch_size):
        return random.sample(self.training_set, min(batch_size, len(self.training_set)))

    def get_validation_set(self, batch_size):
        return random.sample(self.validation_set, min(batch_size, len(self.validation_set)))

    def add_data_point(self, data_point):
        # Add to full data buffer
        if len(self.data) >= self.max_len:
            self.data.pop(0)
        self.data.append(data_point)

        # Add to training or validation buffer
        if random.random() < 0.8:
            if len(self.training_set) >= self.training_max:
                self.training_set.pop(0)
            self.training_set.append(data_point)
        else:
            if len(self.validation_set) >= self.validation_max:
                self.validation_set.pop(0)
            self.validation_set.append(data_point)

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.data, f)

    def print_rbuf(self):
        print("All data:", self.data)
        print("Training:", self.training_set)
        print("Validation:", self.validation_set)

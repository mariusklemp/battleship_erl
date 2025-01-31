import numpy as np
import random
import pickle


class RBUF:
    def __init__(self, max_len=5000):
        self.data = []
        self.training_set = []
        self.validation_set = []
        self.max_len = max_len

    def get_batch(self, batch_size):
        # Return a weighted random sample from self.data
        weights = np.linspace(0, 1, len(self.data))
        return random.choices(self.data, weights=weights, k=batch_size)

    def get_training_set(self, batch_size):
        weights = np.linspace(0, 1, len(self.training_set))
        return random.choices(self.training_set, weights=weights, k=batch_size)

    def get_validation_set(self):
        return self.validation_set

    def add_data_point(self, data_point):
        if len(self.data) > self.max_len:
            self.data.pop(0)
        self.data.append(data_point)

    def add_training_data(self, data_point):
        self.training_set.append(data_point)

    def add_validation_data(self, data_point):
        self.validation_set.append(data_point)

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.data, f)

    def load_from_file(self, file_path):
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if len(data) > self.max_len:
                    self.data = data[-self.max_len :]
                else:
                    self.data = data
                self.training_set = self.data[: int(0.8 * len(self.data))]
                self.validation_set = self.data[int(0.8 * len(self.data)) :]
        except FileNotFoundError:
            print("No file found")

    def print_rbuf(self):
        print(self.data)

import os
import random
import pickle


class RBUF:
    def __init__(self, max_len=10000, val_fraction=0.2):
        self.data = []
        self.training_set = []
        self.validation_set = []
        self.max_len = max_len
        self.val_fraction = val_fraction
        self.training_max = int(max_len * (1 - val_fraction))

    def init_from_file(self, file_path: str):
        """
        Load saved data from disk, then carve out a static validation set once
        (randomly shuffled), and put the rest into training. After this point,
        validation_set never changes.
        """
        try:
            size = os.path.getsize(file_path)
            print(f"Loading {file_path} ({size} bytes)…")
            with open(file_path, "rb") as f:
                loaded = pickle.load(f)
        except FileNotFoundError:
            print("No file found at", file_path)
            return

        # Keep only the most recent `max_len` points
        if len(loaded) > self.max_len:
            self.data = loaded[-self.max_len :]
        else:
            self.data = loaded[:]

        # Shuffle once so train/val are drawn from the same distribution
        random.shuffle(self.data)

        # Static split
        split_idx = int(len(self.data) * (1 - self.val_fraction))
        self.training_set   = self.data[:split_idx]
        self.validation_set = self.data[split_idx:]

        print(
            f"Initialized buffer: "
            f"{len(self.training_set)} train, {len(self.validation_set)} val samples."
        )

    def add_data_point(self, data_point):
        """
        Always add new points to data and training_set; validation_set remains frozen.
        """
        # append to full buffer
        if len(self.data) >= self.max_len:
            self.data.pop(0)
        self.data.append(data_point)

        # add to training buffer (drop oldest if full)
        if len(self.training_set) >= self.training_max:
            self.training_set.pop(0)
        self.training_set.append(data_point)

    def get_training_set(self, batch_size):
        # random sample from current training buffer
        return random.sample(self.training_set, min(batch_size, len(self.training_set)))

    def get_validation_set(self):
        """
        Always return the entire static validation set.
        """
        return list(self.validation_set)

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.data, f)

    def print_rbuf(self):
        print("Total stored:", len(self.data))
        print("Training set size:", len(self.training_set))
        print("Validation set size:", len(self.validation_set))

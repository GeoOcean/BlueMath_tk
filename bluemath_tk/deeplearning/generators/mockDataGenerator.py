import numpy as np
import keras.utils


class MockDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        num_images: int,
        input_frames: int = 1,
        output_frames: int = 1,
        batch_size: int = 8,
        input_height: int = 256,
        input_width: int = 256,
        output_height: int = 256,
        output_width: int = 256,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.batch_size = batch_size
        self.num_images = num_images

    @property
    def num_batches(self):
        return int(np.floor(self.num_images / self.batch_size))

    def __len__(self) -> int:
        """Returns the total number of batches."""
        return self.num_batches

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates one batch of random data"""
        # Generate random input and output data
        inputs = np.random.rand(
            self.batch_size, self.input_height, self.input_width, self.input_frames
        )
        outputs = np.random.rand(
            self.batch_size, self.output_height, self.output_width, self.output_frames
        )

        return inputs, outputs

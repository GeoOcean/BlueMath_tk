import numpy as np
import xarray as xr
import keras.utils


class DataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        msl_path,
        tp_path,
        num_images,
        sequential=False,
        batch_size=1,
    ):
        # create memory-mapped files for high_res and low_res datasets

        # inputs

        self.msl = xr.open_dataarray(msl_path).values[:, :64, :64]

        # outputs

        self.tp = xr.open_dataarray(tp_path).values[:, :64, :64]

        # set boolean for sequential or random dataset
        self.sequential = sequential
        # counter for keeping track of seuquential generator
        self.counter = 0
        self.num_images = num_images
        # set sequence len
        # flag for diffusion/unet
        self.batch_size = batch_size
        self.num_samples = self.msl.shape[0]

    @property
    def num_batches(self):
        return int(np.floor(self.num_images / self.batch_size))

    def __len__(self):
        return self.num_batches

    def min_max_normalize(self, arr, min, max):
        normalized = (arr - min) / (max - min)
        result = np.where(np.isnan(arr), np.nan, normalized)
        return result

    # must be called to restart the sequential
    def counter_reset(self):
        self.counter = 0

    def __getitem__(self, idx):

        # prepare the resulting array
        inputs = np.zeros((self.batch_size, 64, 64, 1))
        outputs = np.zeros((self.batch_size, 64, 64, 1))

        # random path
        if self.sequential == False:
            # compose the batch one element at the time
            for i in range(self.batch_size):
                # get a random number in range
                random = np.random.randint(0, self.num_samples - 1)

                # inputs
                inputs[i, :, :, 0] = self.min_max_normalize(
                    self.msl[random], 95680, 104256
                )

                # outputs
                outputs[i, :, :, 0] = self.min_max_normalize(
                    self.tp[random], 0.0, 0.02197266
                )

        # sequential path
        if self.sequential == True:
            # compose the batch one element at the time
            for i in range(self.batch_size):
                # inputs
                inputs[i, :, :, 0] = self.min_max_normalize(
                    self.msl[self.counter], 95680, 104256
                )

                # outputs
                outputs[i, :, :, 0] = self.min_max_normalize(
                    self.tp[self.counter], 0.0, 0.02197266
                )

                self.counter = self.counter + 1
        return inputs, outputs

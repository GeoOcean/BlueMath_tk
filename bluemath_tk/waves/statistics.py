import numpy as np


def upcrossing(series):
    """
    Performs fast fourier transformation FFt by extracting
    the wave series mean and interpolating to find the cross-zero

    time   - time series
    watlev - surface elevation series

    return - Wave heigths and periods
    """

    # remove mean
    series[:, 1] -= np.mean(series[:, 1])

    # find 0's and calculate upcrossings
    zeros = np.where(np.diff(np.sign(series[:, 1])) > 0)[0]

    # calculate amplitudes (Hi) and periods (Ti)
    Ti = np.diff(zeros)
    Hi = [
        np.max(series[:, 1][zeros[i] : zeros[i + 1]])
        - np.min(series[:, 1][zeros[i] : zeros[i + 1]])
        for i in range(len(zeros) - 1)
    ]

    return Hi, Ti

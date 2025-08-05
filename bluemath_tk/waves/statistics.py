from typing import List, Tuple

import numpy as np


def upcrossing(series: np.ndarray) -> Tuple[List[float], np.ndarray]:
    """
    Calculate wave heights and periods using the zero-crossing method.

    This function performs wave analysis by detecting zero-crossings in a surface
    elevation time series. It removes the mean from the series and identifies
    upcrossings (where the signal crosses zero from negative to positive).
    For each wave cycle between upcrossings, it calculates the wave height
    (difference between maximum and minimum) and wave period (time between crossings).

    Parameters
    ----------
    series : np.ndarray
        A 2D array with shape (n_samples, 2) where:
        - series[:, 0] contains the time values
        - series[:, 1] contains the surface elevation values

    Returns
    -------
    Tuple[List[float], np.ndarray]
        A tuple containing:
        - Hi : List of wave heights (amplitudes) for each wave cycle
        - Ti : Array of wave periods (time intervals between zero crossings)

    Notes
    -----
    - The function removes the mean from the surface elevation series before analysis
    - Wave heights are calculated as the difference between maximum and minimum
      values within each wave cycle
    - Wave periods are the time intervals between consecutive zero upcrossings
    - This method is commonly used in oceanography for wave analysis

    Examples
    --------
    >>> import numpy as np
    >>> time = np.linspace(0, 10, 1000)
    >>> elevation = np.sin(2 * np.pi * 0.5 * time) + 0.1 * np.random.randn(1000)
    >>> series = np.column_stack([time, elevation])
    >>> heights, periods = upcrossing(series)
    """

    # Remove mean from surface elevation series
    series[:, 1] -= np.mean(series[:, 1])

    # Find zero crossings (where signal changes from negative to positive)
    zeros = np.where(np.diff(np.sign(series[:, 1])) > 0)[0]

    # Calculate wave periods (time intervals between zero crossings)
    Ti = np.diff(zeros)

    # Calculate wave heights (amplitudes) for each wave cycle
    Hi = [
        np.max(series[:, 1][zeros[i] : zeros[i + 1]])
        - np.min(series[:, 1][zeros[i] : zeros[i + 1]])
        for i in range(len(zeros) - 1)
    ]

    return Hi, Ti

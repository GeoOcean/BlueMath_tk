from typing import Tuple

import numpy as np
from scipy import signal


def spectral_analysis(
    water_level: np.ndarray, delta_time: float
) -> Tuple[float, float, float, float]:
    """
    Performs spectral analysis of water level variable to separate wave components.

    This function uses Welch's method to estimate the power spectral density of the
    water level time series and then separates the energy into different frequency
    bands: incident waves (IC), infragravity waves (IG), and very low frequency
    waves (VLF). The significant wave heights are calculated for each component.

    Parameters
    ----------
    water_level : np.ndarray
        Water level time series in meters. Should be a 1D array of measurements.
    delta_time : float
        Time interval between consecutive samples in seconds.

    Returns
    -------
    Tuple[float, float, float, float]
        A tuple containing the significant wave heights in meters:
        - Hsi: Total significant wave height
        - HIC: Incident wave significant height (f > 0.04 Hz)
        - HIG: Infragravity wave significant height (0.004 Hz < f < 0.04 Hz)
        - HVLF: Very low frequency wave significant height (0.001 Hz < f < 0.004 Hz)

    Notes
    -----
    The function uses the following frequency bands for wave component separation:
    - Incident waves (IC): f > 0.04 Hz
    - Infragravity waves (IG): 0.004 Hz < f < 0.04 Hz
    - Very low frequency waves (VLF): 0.001 Hz < f < 0.004 Hz

    The significant wave height is calculated as H_s = 4 * sqrt(m0), where m0 is
    the zeroth moment of the spectrum.

    Examples
    --------
    >>> import numpy as np
    >>> water_level = np.random.randn(1000)  # 1000 samples
    >>> delta_time = 1.0  # 1 second intervals
    >>> Hsi, HIC, HIG, HVLF = spectral_analysis(water_level, delta_time)
    """

    # Estimate power spectral density using Welch's method
    f, E = signal.welch(
        water_level,
        fs=1 / delta_time,
        nfft=512,
        scaling="density",
    )
    E = np.reshape(E, -1)

    # Slice frequency to divide energy into components:

    # Incident waves IC (f > 0.04 Hz)
    fIC = f[np.where(f > 0.04)]
    EIC = E[np.where(f > 0.04)]
    mIC = np.trapz(EIC, x=fIC)

    # Infragravity waves IG (0.004 Hz < f < 0.04 Hz)
    fIG = f[(np.where(f > 0.004) and np.where(f < 0.04))]
    EIG = E[(np.where(f > 0.004) and np.where(f < 0.04))]
    mIG = np.trapz(EIG, x=fIG)

    # Very low frequency waves VLF (0.001 Hz < f < 0.004 Hz)
    fVLF = f[(np.where(f > 0.001) and np.where(f < 0.004))]
    EVLF = E[(np.where(f > 0.001) and np.where(f < 0.004))]
    mVLF = np.trapz(EVLF, x=fVLF)

    # Total spectral moment
    m0 = np.trapz(E, x=f)

    # Calculate significant wave heights (H_s = 4 * sqrt(m0))
    Hsi = 4 * np.sqrt(m0)
    HIC = 4 * np.sqrt(mIC)
    HIG = 4 * np.sqrt(mIG)
    HVLF = 4 * np.sqrt(mVLF)

    return Hsi, HIC, HIG, HVLF

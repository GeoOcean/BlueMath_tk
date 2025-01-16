import numpy as np
from scipy import signal


def spectral_analysis(water_level, delta_time):
    """
    Performs spectral analisys of water level variable

    water_level - water level series (m)
    delta_time - time delta for sample (s)
    """

    # estimate power specrtal density
    f, E = signal.welch(
        water_level,
        fs=1 / delta_time,
        nfft=512,
        scaling="density",
    )
    E = np.reshape(E, -1)

    # Slice frequency to divide energy into components:

    # Incident waves IC
    fIC = f[np.where(f > 0.04)]
    EIC = E[np.where(f > 0.04)]
    mIC = np.trapz(EIC, x=fIC)

    # Infragravity waves IG
    fIG = f[(np.where(f > 0.004) and np.where(f < 0.04))]
    EIG = E[(np.where(f > 0.004) and np.where(f < 0.04))]
    mIG = np.trapz(EIG, x=fIG)

    # Very low frequency waves VLF
    fVLF = f[(np.where(f > 0.001) and np.where(f < 0.004))]
    EVLF = E[(np.where(f > 0.001) and np.where(f < 0.004))]
    mVLF = np.trapz(EVLF, x=fVLF)

    m0 = np.trapz(E, x=f)

    Hsi = 4 * np.sqrt(m0)
    HIC = 4 * np.sqrt(mIC)
    HIG = 4 * np.sqrt(mIG)
    HVLF = 4 * np.sqrt(mVLF)

    return Hsi, HIC, HIG, HVLF

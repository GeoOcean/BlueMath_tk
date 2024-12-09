import numpy as np


def energy_spectrum(hs, tp, gamma, duration):
    """
    spec: Dataset with vars for the whole partitions
    S(f,dir) = S(f) * D(dir)
    S(f) ----- D(dir)
    Meshgrid de x freqs - dir - z energy
    """

    # Defining frequency series - tend length
    freqs = np.linspace(0.02, 1, duration)

    S = []
    fp = 1 / tp

    for f in freqs:
        if f <= fp:
            sigma = 0.07
        if f > fp:
            sigma = 0.09

        Beta = (0.06238 / (0.23 + 0.0336 * gamma - 0.185 * (1.9 + gamma) ** -1)) * (
            1.094 - 0.01915 * np.log(gamma)
        )
        Sf = (
            Beta
            * (hs**2)
            * (tp**-4)
            * (f**-5)
            * np.exp(-1.25 * (tp * f) ** -4)
            * gamma ** (np.exp((-((tp * f - 1) ** 2)) / (2 * sigma**2)))
        )
        S.append(Sf)

    return S


def series_Jonswap(waves):
    """
    Generate surface elevation from PSD df = 1/tendc

    waves - dictionary
              T       - Period (s)
              H       - Height (m)
              gamma   - Jonswap spectrum  peak parammeter
              warmup  - spin up time (s)
              deltat  - delta time (s)
              tendc   - simulation period (s)

    returns 2D numpy array with series time and elevation
    """

    # waves properties
    hs = waves["H"]
    tp = waves["T"]
    gamma = waves["gamma"]
    warmup = waves["warmup"]
    deltat = waves["deltat"]
    tendc = waves["tendc"]

    # series duration
    duration = int(tendc + warmup)
    time = np.arange(0, duration, deltat)

    # series frequency
    # why? wee user manual
    freqs = np.linspace(0.02, 1, duration)
    delta_f = freqs[1] - freqs[0]

    # calculate energy spectrum
    S = energy_spectrum(hs, tp, gamma, duration)

    # series elevation
    teta = np.zeros((len(time)))

    # calculate aij
    for f in range(len(freqs)):
        ai = np.sqrt(S[f] * 2 * delta_f)
        eps = np.random.rand() * (2 * np.pi)

        # calculate elevation
        teta = teta + ai * np.cos(2 * np.pi * freqs[f] * time + eps)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series


def series_TMA(waves, depth):
    """
    Generate surface elevation from PSD df = 1/tendc

    waves - dictionary
              T       - Period (s)
              H       - Height (m)
              gamma   - Jonswap spectrum  peak parammeter
              warmup  - spin up time (s)
              deltat  - delta time (s)
              tendc   - simulation period (s)

    returns 2D numpy array with series time and elevation
    """

    # waves properties
    hs = waves["H"]
    tp = waves["T"]
    gamma = waves["gamma"]
    warmup = waves["warmup"]
    deltat = waves["deltat"]
    tendc = waves["tendc"]
    g = 9.801

    # series duration
    duration = int(tendc + warmup)
    time = np.arange(0, duration, deltat)

    # series frequency
    # why? wee user manual
    freqs = np.linspace(0.02, 1, duration)
    delta_f = freqs[1] - freqs[0]

    # calculate JONSWAP energy spectrum
    S = energy_spectrum(hs, tp, gamma, duration)
    m0_Jonswap = np.trapz(S, x=freqs)

    # Transform JONSWAP into TMA by (omega, h)
    S_TMA = []
    for pf, f in enumerate(freqs):
        """
        omega = 2 * np.pi * f
        
        thv = omega * np.sqrt(depth / g)
        if thv < 1: 
            fi = 0.5 * (thv)**2
        elif (thv >= 1) and (thv < 2):
            fi = 1 - 0.5 * (2 - thv)**2
        else:
            fi = 1
        """
        # S_TMA.append(S[pf] * fi)

        # As in Holthuijsen
        L = waves_dispersion(1 / f, depth)[0]
        k = 2 * np.pi / L
        n = 0.5 * (1 + ((2 * k * depth) / (np.sinh(2 * k * depth))))
        fi = (1 / (2 * n)) * np.tanh(k * depth)

        S_TMA.append(S[pf] * fi)

    m0_TMA = np.trapz(S_TMA, x=freqs)

    S_TMA_mod = [i * (m0_Jonswap / m0_TMA) for i in S_TMA]

    # series elevation
    teta = np.zeros((len(time)))

    # calculate aij
    for f in range(len(freqs)):
        ai = np.sqrt(S_TMA_mod[f] * 2 * delta_f)
        eps = np.random.rand() * (2 * np.pi)

        # calculate elevation
        teta = teta + ai * np.cos(2 * np.pi * freqs[f] * time + eps)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series


def series_TMA_bimodal(waves, depth):
    """
    Generate surface elevation from PSD df = 1/tendc

    waves - dictionary
              T       - Period (s)
              H       - Height (m)
              gamma   - Jonswap spectrum  peak parammeter
              warmup  - spin up time (s)
              deltat  - delta time (s)
              tendc   - simulation period (s)

    returns 2D numpy array with series time and elevation
    """

    # waves properties
    hs1 = waves["Hs1"]
    tp1 = waves["Tp1"]
    gamma1 = waves["gamma1"]
    hs2 = waves["Hs2"]
    tp2 = waves["Tp2"]
    gamma2 = waves["gamma2"]
    warmup = waves["warmup"]
    deltat = waves["deltat"]
    tendc = waves["tendc"]

    # series duration
    # TODO: puede que haya algun problema en esta suma
    duration = int(tendc + warmup)
    time = np.arange(0, duration, deltat)

    # series frequency
    freqs = np.linspace(0.02, 1, duration)
    delta_f = freqs[1] - freqs[0]

    # calculate energy spectrum for each wave system
    S1 = energy_spectrum(hs1, tp1, gamma1, duration)
    S2 = energy_spectrum(hs2, tp2, gamma2, duration)

    m01_Jonswap = np.trapz(S1, x=freqs)
    m02_Jonswap = np.trapz(S2, x=freqs)

    # Transform JONSWAP into TMA by (omega, h)
    S_TMA_1, S_TMA_2 = [], []
    for pf, f in enumerate(freqs):
        # As in Holthuijsen
        L = waves_dispersion(1 / f, depth)[0]
        k = 2 * np.pi / L
        n = 0.5 * (1 + ((2 * k * depth) / (np.sinh(2 * k * depth))))
        fi = (1 / (2 * n)) * np.tanh(k * depth)

        S_TMA_1.append(S1[pf] * fi)
        S_TMA_2.append(S2[pf] * fi)

    m0_TMA_1 = np.trapz(S_TMA_1, x=freqs)
    m0_TMA_2 = np.trapz(S_TMA_2, x=freqs)

    S_TMA_mod_1 = [i * (m01_Jonswap / m0_TMA_1) for i in S_TMA_1]
    S_TMA_mod_2 = [i * (m02_Jonswap / m0_TMA_2) for i in S_TMA_2]

    # bimodal wave spectra
    S = np.sum([S_TMA_mod_1, S_TMA_mod_2], axis=0)

    # series elevation
    teta = np.zeros((len(time)))

    # calculate aij
    for f in range(len(freqs)):
        ai = np.sqrt(S[f] * 2 * delta_f)
        eps = np.random.rand() * (2 * np.pi)

        # calculate elevation
        teta = teta + ai * np.cos(2 * np.pi * freqs[f] * time + eps)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series


def series_Jonswap_bimodal(waves):
    """
    Generate surface elevation from PSD df = 1/tendc

    waves - dictionary
              T       - Period (s)
              H       - Height (m)
              gamma   - Jonswap spectrum  peak parammeter
              warmup  - spin up time (s)
              deltat  - delta time (s)
              tendc   - simulation period (s)

    returns 2D numpy array with series time and elevation
    """

    # waves properties
    hs1 = waves["Hs1"]
    tp1 = waves["Tp1"]
    gamma1 = waves["gamma1"]
    hs2 = waves["Hs2"]
    tp2 = waves["Tp2"]
    gamma2 = waves["gamma2"]
    warmup = waves["warmup"]
    deltat = waves["deltat"]
    tendc = waves["tendc"]

    # series duration
    # TODO: puede que haya algun problema en esta suma
    duration = int(tendc + warmup)
    time = np.arange(0, duration, deltat)

    # series frequency
    freqs = np.linspace(0.02, 1, duration)
    delta_f = freqs[1] - freqs[0]

    # calculate energy spectrum for each wave system
    S1 = energy_spectrum(hs1, tp1, gamma1, duration)
    S2 = energy_spectrum(hs2, tp2, gamma2, duration)
    # bimodal wave spectra
    S = np.sum([S1, S2], axis=0)

    # series elevation
    teta = np.zeros((len(time)))

    # calculate aij
    for f in range(len(freqs)):
        ai = np.sqrt(S[f] * 2 * delta_f)
        eps = np.random.rand() * (2 * np.pi)

        # calculate elevation
        teta = teta + ai * np.cos(2 * np.pi * freqs[f] * time + eps)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series


def series_regular_monochromatic(waves):
    """
    Generates monochromatic regular waves series

    waves - dictionary
              T      - Period (s)
              H      - Height (m)
              WL     - Water level (m)
              warmup - spin up time (s)
              deltat - delta time (s)
              tendc  - simulation period (s)

    returns 2D numpy array with series time and elevation
    """

    # waves properties
    T = waves["T"]
    H = waves["H"]
    WL = waves["WL"]
    warmup = waves["warmup"]
    deltat = waves["deltat"]
    tendc = waves["tendc"]

    # series duration
    duration = tendc + int(warmup)
    time = np.arange(0, duration, deltat)

    # series elevation
    teta = (H / 2) * np.cos((2 * np.pi / T) * time)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series


def series_regular_bichromatic(waves):
    """
    Generates bichromatic regular waves series

    waves - dictionary
              T1     - Period component 1 (s)
              T2     - Period component 2 (s)
              H      - Height (m)
              WL     - Water level (m)
              warmup - spin up time (s)
              deltat - delta time (s)
              tendc  - simulation period (s)
    """

    # waves properties
    T1 = waves["T1"]
    T2 = waves["T2"]
    H = waves["H"]
    WL = waves["WL"]
    warmup = waves["warmup"]
    deltat = waves["deltat"]
    tendc = waves["tendc"]

    # series duration
    duration = tendc + int(warmup)
    time = np.arange(0, duration, deltat)

    # series elevation
    teta = (H / 2) * np.cos((2 * np.pi / T1) * time) + (H / 2) * np.cos(
        (2 * np.pi / T2) * time
    )

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series


def waves_dispersion(T, h):
    "Solve the wave dispersion relation"

    L1 = 1
    L2 = ((9.81 * T**2) / 2 * np.pi) * np.tanh(h * 2 * np.pi / L1)
    umbral = 1

    while umbral > 0.1:
        L2 = ((9.81 * T**2) / (2 * np.pi)) * np.tanh((h * 2 * np.pi) / L1)
        umbral = np.abs(L2 - L1)
        L1 = L2

    L = L2
    k = (2 * np.pi) / L
    c = np.sqrt(9.8 * np.tanh(k * h) / k)

    return (L, k, c)

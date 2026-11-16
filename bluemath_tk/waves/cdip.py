"""
CDIP wave buoys: hourly bulk parameters from the THREDDS netCDF records.

CDIP (Coastal Data Information Program, Scripps) serves every station as one
archive file (``{station}p1_historic.nc``, all past deployments) and one
real-time file (``{station}p1_rt.nc``). Each record (30 min) holds the band
energy density ``E(f)`` and the first directional moments ``a1(f)``,
``b1(f)``. :func:`cdip_hourly` averages them per hour (records with a good
primary flag only) and computes, from the averaged spectrum:

- ``hs`` = 4 sqrt(m0), ``tp`` = 1 / f of the peak band, ``tm01`` = m0 / m1;
- ``dir``: mean direction (coming from, nautical) of the energy-weighted
  ``a1``, ``b1``;
- ``spr``: directional spread sqrt(2 (1 - r1)) (Kuik et al. 1988, deg), the
  WAVEWATCH III definition, with r1 = |(a1, b1)|.

:func:`cdip_hourly_spectra` gives the hourly 2D spectrum ``efth`` on the same
hours: ``E(f)`` spread over direction by the Maximum Entropy Method of Lygre &
Krogstad (1986) from ``a1, b1, a2, b2`` (:func:`mem_spreading`, the method
CDIP uses for its own directional spectra).

The buoy position of each hour comes from the GPS fixes, so a station moved
between deployments can be split by position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

#: CDIP THREDDS OPeNDAP root.
CDIP_THREDDS = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip"


def cdip_url(station: str | int, realtime: bool = False) -> str:
    """OPeNDAP URL of the archive (default) or real-time file of a station."""

    name = f"{int(station):03d}p1"
    return f"{CDIP_THREDDS}/realtime/{name}_rt.nc" if realtime else f"{CDIP_THREDDS}/archive/{name}/{name}_historic.nc"


def bulk_from_moments(
    energy: np.ndarray, a1: np.ndarray, b1: np.ndarray, freq: np.ndarray, bandwidth: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Bulk ``hs``, ``tp``, ``tm01``, ``dir``, ``spr`` from band spectra on ``(time, freq)``.

    Parameters
    ----------
    energy : np.ndarray
        Band energy density (m²/Hz).
    a1, b1 : np.ndarray
        First directional Fourier moments per band (coming-from convention).
    freq, bandwidth : np.ndarray
        Band centre frequency and width (Hz).

    Returns
    -------
    dict of np.ndarray
        ``hs`` (m), ``tp`` (s), ``tm01`` (s, m0 / m1), ``dir`` (deg, nautical,
        coming from), ``spr`` (deg).
    """

    band = energy * bandwidth
    m0 = np.nansum(band, axis=1)
    m1 = np.nansum(band * freq, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a1m = np.nansum(band * a1, axis=1) / m0
        b1m = np.nansum(band * b1, axis=1) / m0
        r1 = np.clip(np.hypot(a1m, b1m), 0.0, 1.0)
        peak = np.nanargmax(np.where(np.isfinite(energy), energy, -np.inf), axis=1)
        out = {
            "hs": 4.0 * np.sqrt(m0),
            "tp": 1.0 / freq[peak],
            "tm01": m0 / m1,
            "dir": np.degrees(np.arctan2(b1m, a1m)) % 360.0,
            "spr": np.degrees(np.sqrt(2.0 * (1.0 - r1))),
        }
    empty = ~(m0 > 0)
    for values in out.values():
        values[empty] = np.nan
    return out


def mem_spreading(
    a1: np.ndarray, b1: np.ndarray, a2: np.ndarray, b2: np.ndarray, dirs: np.ndarray
) -> np.ndarray:
    """
    Directional spreading D(θ) from the first four Fourier moments (MEM).

    Maximum Entropy Method of Lygre & Krogstad (1986), normalised so that
    the sum of ``D * Δθ`` over ``dirs`` is 1 (Δθ = the uniform spacing of
    ``dirs``). It reproduces the four moments exactly. Where ``r2`` is well
    below ``r1²`` (e.g. any cos-2s spreading) it splits one peak into two
    on either side of the mean direction (a known MEM artefact): read
    narrow directional detail of MEM spectra with care.

    Parameters
    ----------
    a1, b1, a2, b2 : np.ndarray
        Moments of the same shape (e.g. ``(time, freq)``), in the convention
        of ``dirs`` (``a1 = <cos θ>``, ``b1 = <sin θ>``).
    dirs : np.ndarray
        Uniformly spaced directions over 360° (deg; nautical, coming from
        for CDIP).

    Returns
    -------
    np.ndarray
        ``D`` on ``a1.shape + (len(dirs),)``, in 1/deg; NaN where a moment
        is NaN.
    """

    c1 = (a1 + 1j * b1)[..., None]
    c2 = (a2 + 1j * b2)[..., None]
    r1sq = np.minimum(np.abs(c1) ** 2, 1.0 - 1e-6)
    phi1 = (c1 - c2 * np.conj(c1)) / (1.0 - r1sq)
    phi2 = c2 - c1 * phi1
    theta = np.radians(np.asarray(dirs, dtype=float))
    denom = np.abs(1.0 - phi1 * np.exp(-1j * theta) - phi2 * np.exp(-2j * theta)) ** 2
    spread = np.real(1.0 - phi1 * np.conj(c1) - phi2 * np.conj(c2)) / denom
    step = 360.0 / len(dirs)
    with np.errstate(invalid="ignore", divide="ignore"):
        return spread / (np.sum(spread, axis=-1, keepdims=True) * step)


def _hourly_moments(ds: xr.Dataset, moments: tuple[str, ...]) -> tuple[pd.DatetimeIndex, np.ndarray, dict, np.ndarray]:
    """Hourly mean ``E(f)`` and energy-weighted moments of the good records, and the record count."""

    good = ds["waveFlagPrimary"].values == 1
    time = pd.DatetimeIndex(ds["waveTime"].values[good]).floor("h")
    energy = ds["waveEnergyDensity"].values[good]
    frame = {"E": energy} | {m: energy * ds[f"wave{m.upper()}Value"].values[good] for m in moments}
    hourly = {k: pd.DataFrame(v, index=time).groupby(level=0).mean() for k, v in frame.items()}
    hours = hourly["E"].index
    e = hourly["E"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = {m: hourly[m].values / e for m in moments}
    counts = pd.Series(1, index=time).groupby(level=0).sum().reindex(hours).values
    return hours, e, mom, counts


def cdip_hourly(ds: xr.Dataset) -> xr.Dataset:
    """
    Hourly bulk parameters and position of a CDIP station file.

    Parameters
    ----------
    ds : xr.Dataset
        A CDIP station file (:func:`cdip_url`), opened with xarray.

    Returns
    -------
    xr.Dataset
        ``hs``, ``tp``, ``tm01``, ``dir``, ``spr`` and ``latitude``, ``longitude``
        (GPS, nearest fix within 6 h) on hourly ``time``; ``n_records`` is
        the number of good records averaged.
    """

    hours, e, mom, counts = _hourly_moments(ds, ("a1", "b1"))
    with np.errstate(invalid="ignore", divide="ignore"):
        bulk = bulk_from_moments(
            e, mom["a1"], mom["b1"], ds["waveFrequency"].values, ds["waveBandwidth"].values,
        )

    gps = pd.DataFrame(
        {"latitude": ds["gpsLatitude"].values, "longitude": ds["gpsLongitude"].values},
        index=pd.DatetimeIndex(ds["gpsTime"].values),
    ).sort_index()
    gps = gps[~gps.index.duplicated()].dropna()
    position = gps.reindex(hours, method="nearest", tolerance=pd.Timedelta("6h"))

    out = xr.Dataset(
        {name: ("time", values) for name, values in bulk.items()}
        | {
            "latitude": ("time", position["latitude"].values),
            "longitude": ("time", position["longitude"].values),
            "n_records": ("time", counts),
        },
        coords={"time": hours.values},
    )
    out["hs"].attrs = {"long_name": "Significant wave height (4 sqrt m0)", "units": "m"}
    out["tp"].attrs = {"long_name": "Peak wave period", "units": "s"}
    out["tm01"].attrs = {"long_name": "Mean wave period m0/m1", "units": "s"}
    out["dir"].attrs = {"long_name": "Mean wave direction (coming from, a1/b1)", "units": "degree"}
    out["spr"].attrs = {"long_name": "Directional spread sqrt(2(1-r1))", "units": "degree"}
    out.attrs = {"source": "CDIP (Scripps), THREDDS", "station": str(ds.attrs.get("id", ""))}
    return out


def cdip_hourly_spectra(ds: xr.Dataset, dirs: np.ndarray) -> xr.Dataset:
    """
    Hourly 2D spectra of a CDIP station file, on the hours of :func:`cdip_hourly`.

    Parameters
    ----------
    ds : xr.Dataset
        A CDIP station file (:func:`cdip_url`), opened with xarray.
    dirs : np.ndarray
        Uniformly spaced directions over 360° (deg, nautical, coming from).

    Returns
    -------
    xr.Dataset
        ``efth`` on ``(time, freq, dir)`` in m²/Hz/deg: hourly ``E(f)`` x
        :func:`mem_spreading` of the hourly energy-weighted ``a1, b1, a2, b2``.
    """

    hours, e, mom, _ = _hourly_moments(ds, ("a1", "b1", "a2", "b2"))
    efth = e[..., None] * mem_spreading(mom["a1"], mom["b1"], mom["a2"], mom["b2"], dirs)
    out = xr.Dataset(
        {"efth": (("time", "freq", "dir"), efth.astype("float32"))},
        coords={"time": hours.values, "freq": ds["waveFrequency"].values, "dir": np.asarray(dirs, dtype=float)},
    )
    out["efth"].attrs = {"long_name": "Directional energy density (MEM)", "units": "m2 Hz-1 degree-1"}
    out.attrs = {"source": "CDIP (Scripps), THREDDS", "station": str(ds.attrs.get("id", ""))}
    return out

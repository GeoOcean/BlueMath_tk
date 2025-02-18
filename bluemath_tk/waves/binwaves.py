from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from ..core.plotting.base_plotting import DefaultStaticPlotting


def transform_CAWCR_WS(cawcr_dataset: xr.Dataset) -> xr.Dataset:
    """
    Transform the wave spectra from CAWCR format to binwaves format.

    Parameters
    ----------
    cawcr_dataset : xr.Dataset
        The wave spectra dataset in CAWCR format.

    Returns
    -------
    xr.Dataset
        The wave spectra dataset in binwaves format.
    """

    ds = cawcr_dataset.rename({"frequency": "freq", "direction": "dir"})
    ds["efth"] = ds["efth"] * np.pi / 180.0
    ds["dir"] = ds["dir"] - 180.0
    ds["dir"] = np.where(ds["dir"] < 0, ds["dir"] + 360, ds["dir"])

    return ds


def process_kp_coefficients(
    swan_ds: xr.Dataset,
    spectrum_freq: np.ndarray,
    spectrum_dir: np.ndarray,
    latitude: float,
    longitude: float,
):
    """
    This function processes the propagation coefficients for all the grid points
    within the SWAN simulation output.
    It takes a long time to run but it only needs to be done once per location.

    Parameters
    ----------
    swan_ds : xr.Dataset
        The SWAN processed dataset.
    spectrum_freq : np.ndarray
        The frequency array.
    spectrum_dir : np.ndarray
        The direction array.
    latitude : float
        The latitude of the point of interest.
    longitude : float
        The longitude of the point of interest.

    Returns
    -------
    xr.Dataset
        The propagation coefficients dataset.
    """

    kp_matrix = np.full(
        [
            len(swan_ds.case_num),
            len(spectrum_freq),
            len(spectrum_dir),
        ],
        0.0,
    )

    swan_point = swan_ds.sel(
        Xp=longitude,
        Yp=latitude,
        method="nearest",
    )
    # TODO: Check if this is the correct way to handle NaN values
    if any(np.isnan(swan_point["TPsmoo"].values)):
        raise ValueError("NaN values found for variable TPsmoo_part")

    # Tp mask
    swan_point_cut = swan_point[["Hsig", "TPsmoo", "Dir"]].where(
        swan_point["TPsmoo"] > 0,
        drop=True,
    )

    # get k,f,d
    kfd = xr.Dataset(
        {
            "k": swan_point_cut.Hsig,
            "f": 1 / swan_point_cut.TPsmoo,
            "d": swan_point_cut.Dir,
        }
    )

    # fill kp
    for case_num in kfd.case_num.values:
        kfd_c = kfd.sel(case_num=case_num)

        # get k,f,d and clean nans
        k = kfd_c.k.values
        f = kfd_c.f.values
        d = kfd_c.d.values

        k = k[~np.isnan(f)]
        d = d[~np.isnan(f)]
        f = f[~np.isnan(f)]

        # set case kp at point
        for c in range(len(f)):
            i = np.argmin(np.abs(spectrum_freq - f[c]))
            j = np.argmin(np.abs(spectrum_dir - d[c]))
            kp_matrix[case_num, i, j] = k[c]

    return xr.Dataset(
        {
            "kp": (["case", "freq", "dir"], kp_matrix),
            "swan_freqs": (["case"], 1.0 / swan_point_cut.TPsmoo),
            "swan_dirs": (["case"], swan_point_cut.Dir),
        },
        coords={
            "case": swan_ds.case_num,
            "freq": spectrum_freq,
            "dir": spectrum_dir,
        },
    )


def reconstruc_spectra(
    spectra_ds: xr.Dataset,
    kp_coeffs: xr.Dataset,
):
    """
    Reconstruct the wave spectra using the kp coefficients.

    Parameters
    ----------
    spectra_ds : xr.Dataset
        The offshore wave spectra dataset.
    kp_coeffs : xr.Dataset
        The nearshore kp coefficients dataset.

    Returns
    -------
    xr.Dataset
        The nearshore reconstructed wave spectra dataset.
    """

    EFTH = np.full(
        np.shape(spectra_ds.efth.values),
        0,
    )

    for case in range(len(kp_coeffs.case)):
        freq_, dir_ = (
            kp_coeffs.isel(case=case).swan_freqs.values,
            kp_coeffs.isel(case=case).swan_dirs.values,
        )
        efth_case = spectra_ds.sel(freq=freq_, dir=dir_, method="nearest")
        kp_case = kp_coeffs.sortby("dir").isel(case=case)

        EFTH = EFTH + (efth_case.efth * kp_case.kp**2).values

    # ns_sp = off_sp.drop(("Wspeed", "Wdir", "Depth")).copy()

    return xr.Dataset(
        {
            "efth": (["time", "freq", "dir"], EFTH),
        },
        coords={
            "time": spectra_ds.time,
            "freq": spectra_ds.freq,
            "dir": spectra_ds.dir,
        },
    )


def plot_selected_subset_parameters(
    selected_subset: pd.DataFrame,
    color: str = "blue",
    **kwargs,
) -> Tuple[plt.figure, plt.axes]:
    """
    Plot the selected subset parameters.

    Parameters
    ----------
    selected_subset : pd.DataFrame
        The selected subset parameters.
    color : str, optional
        The color to use in the plot. Default is "blue".
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the scatter plot function.

    Returns
    -------
    plt.figure
        The figure object containing the plot.
    plt.axes
        Array of axes objects for the subplots.
    """

    # Create figure and axes
    default_static_plot = DefaultStaticPlotting()
    fig, axes = default_static_plot.get_subplots(
        nrows=len(selected_subset) - 1,
        ncols=len(selected_subset) - 1,
        sharex=False,
        sharey=False,
    )

    for c1, v1 in enumerate(list(selected_subset.columns)[1:]):
        for c2, v2 in enumerate(list(selected_subset.columns)[:-1]):
            default_static_plot.plot_scatter(
                ax=axes[c2, c1],
                x=selected_subset[v1],
                y=selected_subset[v2],
                c=color,
                alpha=0.6,
                **kwargs,
            )
            if c1 == c2:
                axes[c2, c1].set_xlabel(list(selected_subset.columns)[c1 + 1])
                axes[c2, c1].set_ylabel(list(selected_subset.columns)[c2])
            elif c1 > c2:
                axes[c2, c1].xaxis.set_ticklabels([])
                axes[c2, c1].yaxis.set_ticklabels([])
            else:
                fig.delaxes(axes[c2, c1])

    return fig, axes


def plot_grid_cases(spectra: xr.Dataset, cases_id: np.ndarray, figsize: tuple = (8, 8)):
    """
    Function to plot the cases with different colors.

    Parameters
    ----------
    spectra : xr.Dataset
        The wave spectra dataset.
    cases_id : np.ndarray
        The cases ids for the color.
    figsize : tuple, optional
        The figure size. Default is (8, 8).

    Returns
    -------
    plt.Figure
        The figure.
    """

    # generate figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    # prepare data
    x = np.append(
        np.deg2rad(spectra.dir.values - 7.5), np.deg2rad(spectra.dir.values - 7.5)[0]
    )
    y = np.append(0, spectra.freq.values)
    z = cases_id

    # custom colormap
    cmn = np.vstack(
        (
            cm.get_cmap("plasma", 124)(np.linspace(0, 0.9, 70)),
            cm.get_cmap("magma_r", 124)(np.linspace(0.1, 0.4, 80)),
            cm.get_cmap("rainbow_r", 124)(np.linspace(0.1, 0.8, 80)),
            cm.get_cmap("Blues_r", 124)(np.linspace(0.4, 0.8, 40)),
            cm.get_cmap("cubehelix_r", 124)(np.linspace(0.1, 0.8, 80)),
        )
    )
    cmn = ListedColormap(cmn, name="cmn")

    # plot cases id
    p1 = ax.pcolormesh(
        x,
        y,
        z,
        vmin=0,
        vmax=np.nanmax(cases_id),
        edgecolor="grey",
        linewidth=0.005,
        cmap=cmn,
        shading="flat",
    )

    # customize axes
    ax.set_theta_zero_location("N", offset=0)
    ax.set_theta_direction(-1)
    ax.tick_params(
        axis="both",
        colors="black",
        labelsize=14,
        pad=10,
    )

    # add colorbar
    plt.colorbar(p1, pad=0.1, shrink=0.7).set_label("Case ID", fontsize=16)

    return fig

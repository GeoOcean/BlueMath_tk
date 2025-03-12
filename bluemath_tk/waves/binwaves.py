import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from wavespectra.input.swan import read_swan

from ..core.dask import setup_dask_client
from ..core.plotting.base_plotting import DefaultStaticPlotting


def generate_swan_cases(
    frequencies_array: np.ndarray,
    directions_array: np.ndarray,
) -> xr.Dataset:
    """
    Generate the SWAN cases monocromatic wave parameters.

    Parameters
    ----------
    directions_array : np.ndarray
        The directions array.
    frequencies_array : np.ndarray
        The frequencies array.

    Returns
    -------
    xr.Dataset
        The SWAN monocromatic cases Dataset with coordinates freq and dir.
    """

    # Wave parameters
    gamma = 50  # waves gamma
    spr = 2  # waves directional spread

    # Initialize data arrays for each variable
    hs = np.zeros((len(directions_array), len(frequencies_array)))
    tp = np.zeros((len(directions_array), len(frequencies_array)))
    gamma_arr = np.full((len(directions_array), len(frequencies_array)), gamma)
    spr_arr = np.full((len(directions_array), len(frequencies_array)), spr)

    # Fill hs and tp arrays
    for i, freq in enumerate(frequencies_array):
        period = 1 / freq
        hs_val = 1.0 if period > 5 else 0.1
        hs[:, i] = hs_val
        tp[:, i] = np.round(period, 4)

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars={
            "hs": (["dir", "freq"], hs),
            "tp": (["dir", "freq"], tp),
            "spr": (["dir", "freq"], spr_arr),
            "gamma": (["dir", "freq"], gamma_arr),
        },
        coords={
            "dir": directions_array,
            "freq": frequencies_array,
        },
    )

    # To get DataFrame if needed:
    # df = ds.to_dataframe().reset_index()

    return ds


def process_kp_coefficients(
    list_of_input_spectra: List[str],
    list_of_output_spectra: List[str],
) -> xr.Dataset:
    """
    Process the kp coefficients from the output and input spectra.

    Parameters
    ----------
    list_of_input_spectra : List[str]
        The list of input spectra files.
    list_of_output_spectra : List[str]
        The list of output spectra files.

    Returns
    -------
    xr.Dataset
        The kp coefficients Dataset in frequency and direction.
    """

    output_kp_list = []

    for i, (input_spec_file, output_spec_file) in enumerate(
        zip(list_of_input_spectra, list_of_output_spectra)
    ):
        try:
            input_spec = read_swan(input_spec_file).squeeze().efth
            output_spec = (
                read_swan(output_spec_file)
                .efth.squeeze()
                .drop_vars("time")
                .expand_dims({"case_num": [i]})
            )
            kp = output_spec / input_spec.sum(dim=["freq", "dir"])
            output_kp_list.append(kp)
        except Exception as e:
            print(f"Error processing {input_spec_file} and {output_spec_file}")
            print(e)

    return (
        xr.concat(output_kp_list, dim="case_num")
        .fillna(0.0)
        .sortby("freq")
        .sortby("dir")
    )


def reconstruc_spectra(
    offshore_spectra: xr.Dataset,
    kp_coeffs: xr.Dataset,
    num_workers: int = None,
    memory_limit: float = 0.5,
    chunk_sizes: dict = {"time": 24},
):
    """
    Reconstruct the onshore spectra using offshore spectra and kp coefficients.

    Parameters
    ----------
    offshore_spectra : xr.Dataset
        The offshore spectra dataset.
    kp_coeffs : xr.Dataset
        The kp coefficients dataset.
    num_workers : int, optional
        The number of workers to use. Default is None.
    memory_limit : float, optional
        The memory limit to use. Default is 0.5.
    chunk_sizes : dict, optional
        The chunk sizes to use. Default is {"time": 24}.

    Returns
    -------
    xr.Dataset
        The reconstructed onshore spectra dataset.
    """

    # Setup Dask client
    if num_workers is None:
        num_workers = os.environ.get("BLUEMATH_NUM_WORKERS", 2)
    client = setup_dask_client(n_workers=num_workers, memory_limit=memory_limit)

    try:
        # Process with controlled chunks
        offshore_spectra_chunked = offshore_spectra.chunk(
            {"time": chunk_sizes.get("time", 24)}
        )
        kp_coeffs_chunked = kp_coeffs.chunk({"site": 1})
        with ProgressBar():
            onshore_spectra = (
                (offshore_spectra_chunked * kp_coeffs_chunked).sum(dim="case_num")
            ).compute()
        return onshore_spectra

    finally:
        client.close()


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


def plot_selected_cases_grid(
    frequencies: np.ndarray,
    directions: np.ndarray,
    figsize: Tuple[int, int] = (8, 8),
    **kwargs,
):
    """
    Plot the selected subset parameters.

    Parameters
    ----------
    frequencies : np.ndarray
        The frequencies array.
    directions : np.ndarray
        The directions array.
    figsize : tuple, optional
        The figure size. Default is (8, 8).
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the pcolormesh function.
    """

    # generate figure and axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="polar")

    # prepare data
    x = np.append(np.deg2rad(directions - 7.5), np.deg2rad(directions - 7.5)[0])
    y = np.append(0, frequencies)
    z = (
        np.array(range(len(frequencies) * len(directions)))
        .reshape(len(directions), len(frequencies))
        .T
    )

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
        vmax=np.nanmax(z),
        edgecolor="grey",
        linewidth=0.005,
        cmap=cmn,
        shading="flat",
        **kwargs,
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

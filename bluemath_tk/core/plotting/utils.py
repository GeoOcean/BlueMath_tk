from typing import List, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.ticker import MaxNLocator


def get_list_of_colors_for_colormap(
    cmap: Union[str, Colormap], num_colors: int
) -> list:
    """
    Get a list of colors from a colormap.

    Parameters
    ----------
    cmap : Union[str, Colormap]
        The colormap to use.
    num_colors : int
        The number of colors to generate.

    Returns
    -------
    list
        A list of colors generated from the colormap.
    """

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    return [cmap(i) for i in range(0, 256, 256 // num_colors)]


def create_cmap_from_colors(
    color_list: List[str], name: str = "custom"
) -> colors.LinearSegmentedColormap:
    """
    Create a colormap from a list of hex colors.

    Parameters
    ----------
    color_list : List[str]
        List of hex color codes (e.g., ["#ff0000", "#00ff00"])
    name : str, optional
        Name for the colormap. Default is "custom".

    Returns
    -------
    colors.LinearSegmentedColormap
        A colormap created from the provided colors.
    """

    rgb_colors = [colors.hex2color(color) for color in color_list]

    return colors.LinearSegmentedColormap.from_list(name, rgb_colors, N=256)


def _as_colormap(cmap: Union[str, List[str], Colormap], name: str = "cmap") -> Colormap:
    """
    Convert a colormap name or a list of hex colors into a Colormap.
    """

    if isinstance(cmap, str):
        return plt.get_cmap(cmap)
    elif isinstance(cmap, list):
        return colors.LinearSegmentedColormap.from_list(name, cmap)

    return cmap


def _bounds_with_anchor(
    value_range: Tuple[float, float],
    num: int,
    anchor: Tuple[float, float] = None,
) -> np.ndarray:
    """
    Build `num` boundaries spanning `value_range`, optionally pinning a colormap
    fraction to a given data value.

    Parameters
    ----------
    value_range : Tuple[float, float]
        Data range covered by the colormap.
    num : int
        Number of boundaries to generate.
    anchor : Tuple[float, float], optional
        (colormap_fraction, data_value) pair. The colour sitting at
        `colormap_fraction` of the colormap is placed at `data_value`, and each
        side of it is stretched linearly. If None, boundaries are evenly spaced.

    Returns
    -------
    np.ndarray
        Monotonically increasing boundaries of length `num`.
    """

    vmin, vmax = float(value_range[0]), float(value_range[1])
    if anchor is None:
        return np.linspace(vmin, vmax, num)

    fraction, value = float(anchor[0]), float(anchor[1])
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Anchor fraction must be in (0, 1), got {fraction}.")
    if not vmin < value < vmax:
        raise ValueError(
            f"Anchor value {value} must lie strictly within the data range ({vmin}, {vmax})."
        )

    num_low = int(round(fraction * (num - 1))) + 1

    return np.concatenate(
        [
            np.linspace(vmin, value, num_low),
            np.linspace(value, vmax, num - num_low + 1)[1:],
        ]
    )


def single_colormap(
    cmap: Union[str, List[str], Colormap],
    value_range: Tuple[float, float],
    name: str = "cmap",
    cmap_range: Tuple[float, float] = (0.0, 1.0),
    anchor: Tuple[float, float] = None,
    num_colors: int = 256,
) -> Tuple[ListedColormap, BoundaryNorm]:
    """
    Build a single colormap and its matching norm over a data range.

    Same interface as `join_colormaps`, for the cases where only one of the two
    colormaps is actually needed (e.g. bathymetry without any emerged land).

    Parameters
    ----------
    cmap : Union[str, List[str], Colormap]
        Input colormap (name, list of hex codes, or Colormap object).
    value_range : Tuple[float, float]
        Value range in the data domain covered by the colormap.
    name : str
        Name of the output colormap.
    cmap_range : Tuple[float, float]
        Portion of the colormap to use (from 0 to 1).
    anchor : Tuple[float, float]
        Optional (colormap_fraction, data_value) pair pinning a colour to a data
        value, stretching both sides of it linearly.
    num_colors : int
        Number of discrete colors to sample.

    Returns
    -------
    ListedColormap
        Colormap object.
    BoundaryNorm
        Normalization for mapping data to colors.
    """

    cmap = _as_colormap(cmap, name=name)
    newcolors = cmap(np.linspace(cmap_range[0], cmap_range[1], num_colors))
    bounds = _bounds_with_anchor(value_range, num_colors + 1, anchor)
    norm = BoundaryNorm(boundaries=bounds, ncolors=num_colors)

    return colors.ListedColormap(newcolors, name=name), norm


def nice_ticks(
    value_range: Tuple[float, float],
    num_ticks: int = 7,
    include: float = None,
    ends: bool = True,
) -> np.ndarray:
    """
    Compute round tick values covering a data range.

    Parameters
    ----------
    value_range : Tuple[float, float]
        Data range to cover.
    num_ticks : int
        Approximate number of ticks. Default is 7.
    include : float
        Optional value to force into the ticks (e.g. the shoreline at 0), if it
        falls inside the range.
    ends : bool
        Whether to always tick both ends of the range. Round ticks landing too
        close to an end are dropped in their favour. Default is True.

    Returns
    -------
    np.ndarray
        Sorted tick values, all within the data range.
    """

    vmin, vmax = float(value_range[0]), float(value_range[1])
    ticks = MaxNLocator(nbins=num_ticks, steps=[1, 2, 2.5, 5, 10]).tick_values(
        vmin, vmax
    )
    step = ticks[1] - ticks[0] if len(ticks) > 1 else (vmax - vmin)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

    if ends:
        # Leave room for the end labels, then place them at the exact data limits
        ticks = ticks[(ticks > vmin + 0.3 * step) & (ticks < vmax - 0.3 * step)]
    if include is not None and vmin <= include <= vmax:
        ticks = np.union1d(ticks, [float(include)])
    if ends:
        ticks = np.union1d(ticks, [vmin, vmax])

    return ticks


def format_ticks(ticks: Union[List[float], np.ndarray], max_decimals: int = 3) -> List[str]:
    """
    Format tick values as short labels, using the fewest decimals that still
    keeps every label distinct.

    Useful for ticks placed at exact data limits, so that e.g. a shallowest
    value of -0.3 m reads as "0" rather than "-0.3".

    Parameters
    ----------
    ticks : Union[List[float], np.ndarray]
        Tick values to label.
    max_decimals : int
        Maximum number of decimals to fall back on. Default is 3.

    Returns
    -------
    List[str]
        One label per tick.
    """

    values = np.asarray(ticks, dtype=float)

    for decimals in range(max_decimals + 1):
        rounded = np.round(values, decimals)
        rounded[rounded == 0.0] = 0.0  # Avoid "-0" labels
        labels = [f"{value:.{decimals}f}" for value in rounded]
        if len(set(labels)) == len(labels):
            break

    return labels


def join_colormaps(
    cmap1: Union[str, List[str], Colormap],
    cmap2: Union[str, List[str], Colormap],
    name: str = "joined_cmap",
    range1: Tuple[float, float] = (0.0, 1.0),
    range2: Tuple[float, float] = (0.0, 1.0),
    value_range1: Tuple[float, float] = None,
    value_range2: Tuple[float, float] = None,
    anchor1: Tuple[float, float] = None,
    anchor2: Tuple[float, float] = None,
) -> Tuple[ListedColormap, BoundaryNorm]:
    """
    Join two colormaps into one, with value ranges specified for each.

    Parameters
    ----------
    cmap1, cmap2 : Union[str, List[str], Colormap]
        Input colormaps (name, list of hex codes, or Colormap object).
    name : str
        Name of the output colormap.
    range1, range2 : Tuple[float, float]
        Portion of each colormap to use (from 0 to 1).
    value_range1, value_range2 : Tuple[float, float]
        Value ranges in the data domain corresponding to each colormap.
    anchor1, anchor2 : Tuple[float, float]
        Optional (colormap_fraction, data_value) pairs pinning a colour of each
        colormap to a data value, stretching both sides of it linearly.

    Returns
    -------
    ListedColormap
        Merged colormap object.
    BoundaryNorm
        Normalization for mapping data to colors.
    """

    # Convert each input to a Colormap if needed
    cmap1 = _as_colormap(cmap1, name="cmap1")
    cmap2 = _as_colormap(cmap2, name="cmap2")

    # Get colors from each colormap
    colors1 = cmap1(np.linspace(range1[0], range1[1], 128))
    colors2 = cmap2(np.linspace(range2[0], range2[1], 128))
    newcolors = np.vstack((colors1, colors2))

    # Create corresponding boundaries in data space
    if value_range1 is not None and value_range2 is not None:
        # Values are cast to float, as scalar xarray objects break np.linspace wrapping
        bounds1 = _bounds_with_anchor(value_range1, 129, anchor1)
        bounds2 = _bounds_with_anchor(value_range2, 129, anchor2)
        all_bounds = np.sort(np.concatenate([bounds1[:-1], bounds2]))

        norm = BoundaryNorm(boundaries=all_bounds, ncolors=len(newcolors))

        return colors.ListedColormap(newcolors, name=name), norm
    else:
        return colors.ListedColormap(newcolors, name=name)


if __name__ == "__main__":
    # Join two named colormaps using only middle 80% of each
    cmap = join_colormaps("viridis", "plasma", range1=(0.1, 0.9), range2=(0.1, 0.9))

    # Join a named colormap with a list of colors
    cmap = join_colormaps("viridis", ["#ff0000", "#00ff00", "#0000ff"])

    # Join two lists of colors
    cmap = join_colormaps(["#ff0000", "#00ff00"], ["#0000ff", "#ffff00"])

    # Join with custom name and ranges
    cmap = join_colormaps(
        "viridis", "plasma", name="my_custom_cmap", range1=(0.0, 0.5), range2=(0.5, 1.0)
    )

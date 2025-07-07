from typing import List, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap


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


def join_colormaps(
    cmap1: Union[str, List[str], Colormap],
    cmap2: Union[str, List[str], Colormap],
    name: str = "joined_cmap",
    range1: Tuple[float, float] = (0.0, 1.0),
    range2: Tuple[float, float] = (0.0, 1.0),
    value_range1: Tuple[float, float] = None,
    value_range2: Tuple[float, float] = None,
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

    Returns
    -------
    ListedColormap
        Merged colormap object.
    BoundaryNorm
        Normalization for mapping data to colors.
    """

    # Convert cmap1 to a Colormap if needed
    if isinstance(cmap1, str):
        cmap1 = plt.get_cmap(cmap1)
    elif isinstance(cmap1, list):
        cmap1 = colors.LinearSegmentedColormap.from_list("cmap1", cmap1)

    if isinstance(cmap2, str):
        cmap2 = plt.get_cmap(cmap2)
    elif isinstance(cmap2, list):
        cmap2 = colors.LinearSegmentedColormap.from_list("cmap2", cmap2)

    # Get colors from each colormap
    colors1 = cmap1(np.linspace(range1[0], range1[1], 128))
    colors2 = cmap2(np.linspace(range2[0], range2[1], 128))
    newcolors = np.vstack((colors1, colors2))

    # Create corresponding boundaries in data space
    if value_range1 is not None and value_range2 is not None:
        bounds1 = np.linspace(value_range1[0], value_range1[1], 129)
        bounds2 = np.linspace(value_range2[0], value_range2[1], 129)
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

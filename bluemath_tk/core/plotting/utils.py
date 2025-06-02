from typing import List, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap


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
) -> colors.ListedColormap:
    """
    Join two colormaps into one. Each input can be either a colormap name, a list of colors,
    or an existing colormap. Optionally crop each colormap to a specific range.

    Parameters
    ----------
    cmap1 : Union[str, List[str], Colormap]
        First colormap (name, color list, or colormap object).
    cmap2 : Union[str, List[str], Colormap]
        Second colormap (name, color list, or colormap object).
    name : str, optional
        Name for the resulting colormap. Default is "joined_cmap".
    range1 : Tuple[float, float], optional
        Range of colors to use from first colormap (start, end) between 0 and 1.
        Default is (0.0, 1.0) using the full range.
    range2 : Tuple[float, float], optional
        Range of colors to use from second colormap (start, end) between 0 and 1.
        Default is (0.0, 1.0) using the full range.

    Returns
    -------
    colors.ListedColormap
        A new colormap that combines both input colormaps.

    Examples
    --------
    >>> # Join two colormaps using only middle 80% of each
    >>> cmap = join_colormaps("viridis", "plasma", range1=(0.1, 0.9), range2=(0.1, 0.9))
    >>> # Join colormaps with different ranges
    >>> cmap = join_colormaps("viridis", "plasma", range1=(0.0, 0.5), range2=(0.5, 1.0))
    """

    # Convert inputs to colormaps if they aren't already
    if isinstance(cmap1, str):
        if cmap1.startswith("#"):
            cmap1 = create_cmap_from_colors([cmap1])
        else:
            cmap1 = plt.get_cmap(cmap1)
    elif isinstance(cmap1, list):
        cmap1 = create_cmap_from_colors(cmap1)

    if isinstance(cmap2, str):
        if cmap2.startswith("#"):
            cmap2 = create_cmap_from_colors([cmap2])
        else:
            cmap2 = plt.get_cmap(cmap2)
    elif isinstance(cmap2, list):
        cmap2 = create_cmap_from_colors(cmap2)

    # Create the joined colormap with specified ranges
    colors1 = cmap1(np.linspace(range1[0], range1[1], 128))
    colors2 = cmap2(np.linspace(range2[0], range2[1], 128))
    newcolors = np.vstack((colors1, colors2))

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

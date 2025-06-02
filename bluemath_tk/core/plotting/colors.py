from typing import Any, Dict

import cmocean
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

default_colors = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]

hex_colors_water = [
    "#4a84b5",
    "#5493c8",
    "#5fa9d1",
    "#74c3dc",
    "#8ed7e8",
    "#a0e2ef",
    "#b7f1eb",
    "#c8ebd8",
    "#d7e8c3",
    "#e2e5a5",
    "#f4cda0",
    "#f1e2c6",
]

hex_colors_land = [
    "#cfe2bd",
    "#aece91",
    "#7eb14f",
    "#76ac44",
    "#6ea739",
    "#66a22e",
    "#518134",
    "#518134",
    "#3E731E",
]


scatter_defaults = {
    "figsize": (9, 8),
    "marker": ".",
    "color_data": "#00CC96",
    "color_subset": "#AB63FA",
    "alpha_data": 0.5,
    "alpha_subset": 0.7,
    "size_data": 10,
    "size_centroid": 70,
    "fontsize": 12,
}


def get_config_variables() -> Dict[str, Dict[str, Any]]:
    """
    Get configuration variables for different meteorological parameters.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary containing configuration for different variables:
        - geo500hpa: Geopotential height at 500hPa
        - mslp: Mean Sea Level Pressure
        - mslp_grad: MSLP gradient
        - sst: Sea Surface Temperature
        - other: Default configuration
    """

    config_variables = {}

    config_variables["geo500hpa"] = {
        "cmap": "Blues",
        "limits": (45000, 60000),
        "label": "Geo500Hpa [m]",
    }

    config_variables["mslp"] = {
        "cmap": "RdBu_r",
        "limits": (1014 - 20, 1014 + 20),
        "label": "MSLP [mbar]",
    }

    config_variables["mslp_grad"] = {
        "cmap": "BuGn",
        "limits": (0, 100),
        "label": "MSLP - grad [mbar]",
    }

    config_variables["sst"] = {
        "cmap": "hot_r",
        "limits": (0, 30),
        "label": "SST - [°C]",
    }

    config_variables["other"] = {
        "cmap": "rainbow",
        "limits": (None, None),
        "label": " ",
    }

    return config_variables


def colors_awt() -> np.ndarray:
    """
    Get colors for Annual Weather Types (6 categories).

    Returns
    -------
    np.ndarray
        Array of RGB colors for 6 AWT categories
    """

    l_colors_dwt = [
        (155 / 255.0, 0, 0),
        (1, 0, 0),
        (255 / 255.0, 216 / 255.0, 181 / 255.0),
        (164 / 255.0, 226 / 255.0, 231 / 255.0),
        (0 / 255.0, 190 / 255.0, 255 / 255.0),
        (51 / 255.0, 0 / 255.0, 207 / 255.0),
    ]

    return np.array(l_colors_dwt)


def colors_mjo() -> np.ndarray:
    """
    Get colors for MJO 25 categories.

    Returns
    -------
    np.ndarray
        Array of RGB colors for 25 MJO categories.
    """

    l_named_colors = [
        "lightskyblue",
        "deepskyblue",
        "royalblue",
        "mediumblue",
        "darkblue",
        "darkblue",
        "darkturquoise",
        "turquoise",
        "maroon",
        "saddlebrown",
        "chocolate",
        "gold",
        "orange",
        "orangered",
        "red",
        "firebrick",
        "Purple",
        "darkorchid",
        "mediumorchid",
        "magenta",
        "mediumslateblue",
        "blueviolet",
        "darkslateblue",
        "indigo",
        "darkgray",
    ]

    np_colors_rgb = np.array([mcolors.to_rgb(c) for c in l_named_colors])

    return np_colors_rgb


def colors_dwt(num_clusters: int) -> np.ndarray:
    """
    Get colors for Daily Weather Types.

    Parameters
    ----------
    num_clusters : int
        Number of clusters to get colors for.

    Returns
    -------
    np.ndarray
        Array of RGB colors for the specified number of clusters.
    """

    l_colors_dwt = [
        (1.0000, 0.1344, 0.0021),
        (1.0000, 0.2669, 0.0022),
        (1.0000, 0.5317, 0.0024),
        (1.0000, 0.6641, 0.0025),
        (1.0000, 0.9287, 0.0028),
        (0.9430, 1.0000, 0.0029),
        (0.6785, 1.0000, 0.0031),
        (0.5463, 1.0000, 0.0032),
        (0.2821, 1.0000, 0.0035),
        (0.1500, 1.0000, 0.0036),
        (0.0038, 1.0000, 0.1217),
        (0.0039, 1.0000, 0.2539),
        (0.0039, 1.0000, 0.4901),
        (0.0039, 1.0000, 0.6082),
        (0.0039, 1.0000, 0.8444),
        (0.0039, 1.0000, 0.9625),
        (0.0039, 0.8052, 1.0000),
        (0.0039, 0.6872, 1.0000),
        (0.0040, 0.4510, 1.0000),
        (0.0040, 0.3329, 1.0000),
        (0.0040, 0.0967, 1.0000),
        (0.1474, 0.0040, 1.0000),
        (0.2655, 0.0040, 1.0000),
        (0.5017, 0.0040, 1.0000),
        (0.6198, 0.0040, 1.0000),
        (0.7965, 0.0040, 1.0000),
        (0.8848, 0.0040, 1.0000),
        (1.0000, 0.0040, 0.9424),
        (1.0000, 0.0040, 0.8541),
        (1.0000, 0.0040, 0.6774),
        (1.0000, 0.0040, 0.5890),
        (1.0000, 0.0040, 0.4124),
        (1.0000, 0.0040, 0.3240),
        (1.0000, 0.0040, 0.1473),
        (0.9190, 0.1564, 0.2476),
        (0.7529, 0.3782, 0.4051),
        (0.6699, 0.4477, 0.4584),
        (0.5200, 0.5200, 0.5200),
        (0.4595, 0.4595, 0.4595),
        (0.4100, 0.4100, 0.4100),
        (0.3706, 0.3706, 0.3706),
        (0.2000, 0.2000, 0.2000),
        (0, 0, 0),
    ]

    np_colors_base = np.array(l_colors_dwt)
    np_colors_rgb = np_colors_base[:num_clusters]

    return np_colors_rgb


def colors_fams_3() -> np.ndarray:
    """
    Get colors for 3 wave families.

    Returns
    -------
    np.ndarray
        Array of RGB colors for 3 wave families.
    """

    l_named_colors = [
        "gold",
        "darkgreen",
        "royalblue",
    ]

    np_colors_rgb = np.array([mcolors.to_rgb(c) for c in l_named_colors])

    return np_colors_rgb


def colors_interp(num_clusters: int) -> np.ndarray:
    """
    Generate interpolated colors from Spectral colormap.

    Parameters
    ----------
    num_clusters : int
        Number of clusters to generate colors for.

    Returns
    -------
    np.ndarray
        Array of RGB colors interpolated from Spectral colormap.
    """

    scm = cm.get_cmap("Spectral", num_clusters)
    mnorm = mcolors.Normalize(vmin=0, vmax=num_clusters)
    l_colors = [scm(mnorm(i)) for i in range(num_clusters)]

    return np.array(l_colors)[:, :-1]


def get_cluster_colors(num_clusters: int) -> np.ndarray:
    """
    Get appropriate colors for clustering based on number of clusters.

    Parameters
    ----------
    num_clusters : int
        Number of clusters to get colors for.

    Returns
    -------
    np.ndarray
        Array of RGB colors for the specified number of clusters.
    """

    if num_clusters == 6:
        np_colors_rgb = colors_awt()  # Annual Weather Types
    elif num_clusters == 25:
        np_colors_rgb = colors_mjo()  # MJO Categories
    elif num_clusters in [36, 42]:
        np_colors_rgb = colors_dwt(num_clusters)  # Daily Weather Types
    else:
        np_colors_rgb = colors_interp(num_clusters)  # interpolate

    return np_colors_rgb


def GetFamsColors(num_fams: int) -> np.ndarray:
    """
    Get colors for wave families.

    Parameters
    ----------
    num_fams : int
        Number of wave families.

    Returns
    -------
    np.ndarray
        Array of RGB colors for the specified number of wave families.
    """

    if num_fams == 3:
        np_colors_rgb = colors_fams_3()  # chosen colors
    else:
        np_colors_rgb = colors_interp(num_fams)  # interpolate

    return np_colors_rgb


def colormap_bathy(topat: float, topag: float) -> ListedColormap:
    """
    Create custom colormap for bathymetry plot.

    Parameters
    ----------
    topat : float
        Maximum topography value.
    topag : float
        Minimum bathymetry value.

    Returns
    -------
    ListedColormap
        Custom colormap combining YlGnBu_r and turbid colormaps.
    """

    colors2 = "YlGnBu_r"
    colors1 = cmocean.cm.turbid

    bottom = plt.get_cmap(colors2, -topag * 100)
    top = plt.get_cmap(colors1, topat * 100)

    newcolors = np.vstack(
        (
            bottom(np.linspace(0, 0.8, -topag * 100)),
            top(np.linspace(0.1, 1, topat * 100)),
        )
    )

    return ListedColormap(newcolors)


def colormap_spectra() -> ListedColormap:
    """
    Create custom colormap for spectra plots combining RdBu and rainbow colormaps.

    Returns
    -------
    ListedColormap
        Custom colormap combining RdBu and rainbow colormaps.
    """

    top = cm.get_cmap("RdBu", 128)
    bottom = cm.get_cmap("rainbow", 128)
    newcolors = np.vstack(
        (top(np.linspace(0.5, 0.8, 50)), bottom(np.linspace(0.2, 1, 128)))
    )

    return ListedColormap(newcolors, name="newcmp")

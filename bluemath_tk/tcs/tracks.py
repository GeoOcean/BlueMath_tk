import math
import os.path as op
from datetime import timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numpy import polyfit

from ..core.geo import geo_distance, shoot


def is_within_rectangle(lon, lat, min_lon, max_lon, min_lat, max_lat):
    """
    Check if a given (lon, lat) coordinate is inside a rectangular region.

    Parameters:
    lon (float): Longitude of the point to check.
    lat (float): Latitude of the point to check.
    min_lon (float): Minimum longitude of the rectangle.
    max_lon (float): Maximum longitude of the rectangle.
    min_lat (float): Minimum latitude of the rectangle.
    max_lat (float): Maximum latitude of the rectangle.

    Returns:
    bool: True if the point is inside the rectangle, False otherwise.
    """
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def is_within_circle(lon_c, lat_c, radius, coords):
    """
    Check if any point in a given list of coordinates is within a circular region.

    Parameters:
    lon_c (float): Longitude of the circle center.
    lat_c (float): Latitude of the circle center.
    radius (float): Radius of the circle.
    coords (list of tuples): List of (lon, lat) coordinates to check.

    Returns:
    bool: True if at least one coordinate is inside the circle, False otherwise.
    """
    for lon, lat in coords:
        distancia = math.sqrt((lon - lon_c) ** 2 + (lat - lat_c) ** 2)
        if distancia <= radius:
            return True
    return False


def get_category_color(press, color):
    """
    Categorizes a pressure value into storm intensity levels and assigns a corresponding color.

    Parameters:
    press (float): Atmospheric pressure value.
    color (list): List of colors corresponding to each category.

    Returns:
    tuple: (category, color) where:
        - category (int): Storm intensity category (0 to 5).
        - color (str): Assigned color based on the category.
    """

    # Determine storm category based on pressure
    if press >= 1000:
        cat = 0
    elif press >= 979:
        cat = 1
    elif press >= 964:
        cat = 2
    elif press >= 949:
        cat = 3
    elif press >= 920:
        cat = 4
    else:  # press < 920
        cat = 5

    # Assign color based on pressure category
    return (cat, color[cat] if press > 0 else "grey")


def create_cartopy_map(ax=None, grid=True):
    """
    Create a Cartopy map with high-resolution coastlines and optional gridlines.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot, optional): Existing axis to plot on. Defaults to None.
    grid (bool, optional): Whether to display latitude/longitude gridlines. Defaults to True.

    Returns:
    tuple: (fig, ax) where fig is the figure and ax is the axis object.
    """

    # Create figure and axis if none is provided
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    else:
        fig = ax.figure  # Retrieve figure from provided axis

    # Add high-resolution coastlines and map features
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "coastline", "10m", edgecolor="grey", fc="none"
        ),
        linewidth=0.4,
    )
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, fc="silver", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, fc="lightgrey", alpha=0.01)

    ax.set_aspect("equal")

    # Add gridlines if requested
    if grid:
        gl = ax.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            color="gainsboro",
        )
        gl.top_labels = False
        gl.right_labels = False

    return fig, ax


def get_category_color(cat):
    colors = {1: "#2b44d6", 2: "#70a848", 3: "#e6a40c", 4: "#e14d0b", 5: "#9200d9"}
    return colors.get(cat, "grey")


def fix_extent(ax, extent):
    """Fix the extent of the plot in latitude only."""
    mlat = np.mean(extent[2:])
    xtrm_data = np.array([[extent[0], mlat], [extent[1], mlat]])
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    xtrm = proj_to_data.transform(xtrm_data)
    ax.set_ylim(xtrm[:, 1].min(), xtrm[:, 1].max())


def axplot_cartopy_robin(ax):
    """Set up a Robinson projection map with Cartopy."""
    ax.gridlines(
        draw_labels=False,
        dms=True,
        x_inline=False,
        y_inline=False,
        color="gainsboro",
        zorder=1,
    )

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "coastline", "10m", edgecolor="grey", fc="none"
        ),
        linewidth=0.4,
    )
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND, fc="silver", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, fc="lightgrey", alpha=0.01)

    fix_extent(ax, (-180, 180, -60, 58))
    return ax


def axplot_shy_swath_cartopy(
    fig,
    ax,
    xds_bulk,
    storm,
    path_data_shytcwaves,
    path_databases,
    vmin=0,
    vmax=10,
    cmap="hot",
    label="Hs (m)",
    colorbar=True,
):
    """Plot the storm swath on a Cartopy map."""
    df_storm = historic_track_preprocessing(
        storm, path_data_shytcwaves, path_databases, center="WMO"
    )

    xdlon, xdlat, xdmaxhs = (
        xds_bulk.lon.values,
        xds_bulk.lat.values,
        xds_bulk.hswath.values,
    )
    lon_size, lat_size = np.unique(xdlon).size, np.unique(xdlat).size
    XX, YY, ZZ = map(
        lambda arr: np.reshape(arr, (lat_size, lon_size)), [xdlon, xdlat, xdmaxhs]
    )

    im = ax.pcolor(
        XX,
        YY,
        ZZ,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=label)

    ax.plot(
        storm.lon.values,
        storm.lat.values,
        "-",
        color="k",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )

    for lo1, la1, lo2, la2, cat in zip(
        df_storm.longitude[:-1],
        df_storm.latitude[:-1],
        df_storm.longitude[1:],
        df_storm.latitude[1:],
        df_storm.category[:-1],
    ):
        ax.plot(
            [lo1, lo2],
            [la1, la2],
            "-",
            color=get_category_color(cat),
            linewidth=3,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

    return im


def axplot_st_cartopy(ax, storm, path_data_shytcwaves, path_databases):
    """Plot storm tracks on a Cartopy map."""
    df_storm = historic_track_preprocessing(
        storm, path_data_shytcwaves, path_databases, center="WMO"
    )

    for lo1, la1, lo2, la2, cat in zip(
        df_storm.longitude[:-1],
        df_storm.latitude[:-1],
        df_storm.longitude[1:],
        df_storm.latitude[1:],
        df_storm.category[:-1],
    ):
        color = get_category_color(cat)
        ax.plot(
            [lo1, lo2],
            [la1, la2],
            "-",
            color=color,
            linewidth=3,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(lo2, la2, ".", color=color, linewidth=1, transform=ccrs.PlateCarree())


# dictionary -- center names (IBTrACS)
d_vns_idcenter = {
    "USA": "usa",
    "TOKYO": "tokyo",
    "CMA": "cma",
    "HKO": "hko",
    "NEWDELHI": "newdelhi",
    "REUNION": "reunion",
    "BOM": "bom",
    "WELLINGTON": "wellington",
    "NADI": "nadi",
    "WMO": "wmo",
    #        'FORECAST':     'forecast',
}

# dictionary -- basin codes (IBTrACS centers with official data)
d_vns_basinscenter = {
    "USA": ["NA", "SA", "EP", "WP", "SP", "NI", "SI"],
    "TOKYO": ["EP", "WP"],
    "CMA": ["EP", "WP"],
    "HKO": ["EP", "WP"],
    "NEWDELHI": ["WP", "NI"],
    "REUNION": ["SI"],
    "BOM": ["SP", "SI"],
    "WELLINGTON": ["SP", "SI"],
    "NADI": ["SP"],
    "WMO": ["NA", "SA", "EP", "WP", "SP", "NI", "SI"],
    "FORECAST": ["AL", "LS", "EP", "CP", "WP", "IO", "SH"],
    # north atlantic [100-0W], south atlantic [],
    # east/central/west NP, [100-180E, 180-140W, 140-80W],
    # north indian [40-100E], southern hemisphere [20E-120W]
}

# dictionary -- conversion factors (1-min avg to X-min avg)
#########
# IBTrACS database is used to extract historical storm track data.
# However, specialized centers (RSMCs) employ different procedures to compute
# the maximum sustained winds. Therefore a conversion factor is applied to
# obtain the estimated 1-min average maximum sustained winds (input variable
# for "make_vortex_files").
# Harper et al. (2010) recommends 0.93 to convert from 1-min to 10-min (at sea
# conditions) instead of the traditional 0.88 factor.
#########
d_vns_windfac = {
    "USA": 1,  # 1-min, (RSMC Miami, RSMC Honolulu, NHC, JTWC, CPHC)
    "TOKYO": 0.93,  # 10-min, (RSMC Tokyo (JMA))
    "CMA": 0.99,  # 2-min, (Chinese Meteorological Administration
    # Shanghai Typhoon Institute)
    "HKO": 0.93,  # 10-min, (Hong Kong Observatory (operated by JMA)
    "NEWDELHI": 0.99,  # 3-min, (RSMC New Delhi)
    "REUNION": 0.93,  # 10-min, (RSMC La Reunion (MeteoFrance))
    "BOM": 0.93,  # 10-min, (Australian Bureau of Meteorology)
    "WELLINGTON": 0.93,  # 10-min, (New Zealand MetService)
    "NADI": 0.93,  # 10-min, (RSMC Nadi, Fiji Meteorological Service)
    "WMO": 1,  # mix of official centers (ASSUMPTION)
}

# dictionary -- color codes
d_vns_colorcenter = {
    "USA": "r",
    "TOKYO": "lightcoral",
    "CMA": "lime",
    "HKO": "gold",
    "NEWDELHI": "m",
    "REUNION": "magenta",
    "BOM": "b",
    "WELLINGTON": "slategrey",
    "NADI": "c",
    "WMO": "k",
}


def get_dictionary_center(center="WMO"):
    "Returns dictionary of the IBTrACS center"

    # check center name
    if not center in d_vns_idcenter.keys():
        print("Wrong center entry!")
        print(
            "Enter a valid center: USA, TOKYO, CMA, HKO, NEWDELHI, REUNION, \
              BOM, WELLINGTON, NADI, WMO"
        )

    if center in d_vns_idcenter.keys():
        # get center name
        source = d_vns_idcenter[center]
        dict_pressure = source + "_pres"
        dict_maxwinds = source + "_wind"
        dict_basin = "basin"
        dict_dist2land = "dist2land"

        # lon, lat
        if source == "wmo":
            dict_lon, dict_lat = "lon", "lat"
        else:
            dict_lon, dict_lat = source + "_lon", source + "_lat"

        # radii RMW is provided by centers: USA, REUNION, BOM
        if source == "usa" or source == "reunion" or source == "bom":
            dict_radii = source + "_rmw"
        else:
            dict_radii = None

            #        if source=='forecast':
            #            dict_lon, dict_lat = 'longitude','latitude'
            #            dict_pressure = 'pressure'
            #            dict_maxwinds = 'maxwinds'
            #            dict_radii = 'rmw'
            #            dict_basin = 'Basin'

            # variables dictionary
        d_vns = {
            "source": center,
            "basin": dict_basin,
            "time": "time",
            "longitude": dict_lon,
            "latitude": dict_lat,
            "pressure": dict_pressure,
            "maxwinds": dict_maxwinds,
            "rmw": dict_radii,
            "dist2land": dict_dist2land,
        }

    return d_vns


# filenames -- PREPROCESSING IBTrACS
file_ibtracs = "IBTrACS.ALL.v04r00.nc"
file_coeff_pmin_wmax = "ibtracs_coef_pmin_wmax.nc"


###############################################################################
# PREPROCESSING PLOT SHOWING AVAILABLE DATA
###############################################################################

import matplotlib.pyplot as plt


def check_storm_data(xds):
    """
    xds:   (xarray.Dataset) historic track dataset (storm dimension)

    returns:  plot of storm available data provided by all ibtracs centers
    """

    for center in d_vns_idcenter.keys():
        # dictionary for IBTrACS center
        d_vns = get_dictionary_center(center=center)

        # get var time
        ytime = xds[d_vns["time"]].values[:]

        # remove NaTs (time)
        st_lon = xds[d_vns["longitude"]].values[~np.isnat(ytime)]
        st_lat = xds[d_vns["latitude"]].values[~np.isnat(ytime)]
        st_prs = xds[d_vns["pressure"]].values[~np.isnat(ytime)]
        st_win = xds[d_vns["maxwinds"]].values[~np.isnat(ytime)]
        if d_vns["rmw"]:
            st_rmw = xds[d_vns["rmw"]].values[~np.isnat(ytime)]
        st_tim = ytime[~np.isnat(ytime)]

        # longitude convention [0-360º]
        st_lon[st_lon < 0] += 360

        # winds are converted to 1-min avg
        st_win = st_win / d_vns_windfac[center]

        # plot center data (full NaN variables ommitted)
        color = d_vns_colorcenter[center]  # color code

        plt.subplot(141)  # storm coordinates
        plt.plot(st_lon, st_lat, "-", c=color, label=center)
        plt.legend()
        if not np.isnan(st_prs).all():
            plt.subplot(142)  # storm minimum central pressure
            plt.plot(st_tim, st_prs, ".-", c=color, label=center)
            plt.legend()

        if not np.isnan(st_win).all():
            plt.subplot(143)  # storm maximum winds
            plt.plot(st_tim, st_win, ".-", c=color, label=center)
            plt.legend()

        if d_vns["rmw"] and not np.isnan(st_rmw).all():
            #        if d_vns['rmw']:
            #            if not np.isnan(st_rmw).all():
            plt.subplot(144)  # storm radii of maximum winds
            plt.plot(st_tim, st_rmw, ".-", c=color, label=center)
            plt.legend()

    # plot attributes
    plt.subplot(141)
    plt.title("Storm coordinates", fontweight="bold")
    plt.xlabel("Longitude (º)"), plt.ylabel("Latitude (º)")
    plt.subplot(142)
    plt.title("Minimum central pressure [mbar]", fontweight="bold")
    plt.subplot(143)
    plt.title("Maximum sustained winds [kt]", fontweight="bold")
    plt.subplot(144)
    plt.title("Radii of max winds [nmile]", fontweight="bold")
    for i in [2, 3, 4]:
        plt.subplot(1, 4, i)
        plt.xticks(rotation=60)

    plt.gcf().set_size_inches(20, 4)

    return plt


###############################################################################
# PREPROCESSING IBTrACS -- statistical relationship Pmin-Wmax
###############################################################################


def Extract_basin_storms(xds, id_basin):
    """
    Selects storms with genesis at the "id_basin" (NA,SA,WP,EP,SP,NI,SI)

    returns:  xds_basin (xarray.Dataset)
    """

    # select genesis basin (np.bytes_)
    origin_basin = xds.sel(date_time=xds.date_time[0]).basin.values
    origin_basin = np.asarray([c.decode("UTF-8") for c in origin_basin])

    # select storms within given basin
    storm_pos = np.where(origin_basin == id_basin)[0]

    # extract storms for a given genesis basin
    xds_basin = xds.sel(storm=storm_pos)

    return xds_basin


def ibtracs_fit_pmin_wmax(path_databases):
    """
    Generate third order polynomial fitting coefficients for statistical
    relationship between minimum pressure (Pmin) and maximum wind speed (Wmax)

    Note: maximum wind speeds are converted to 1-min average (all RSMC centers)

    returns:  xds (xarray.Dataset)
    """

    # get ibtracs data
    path_ibtracs = op.join(path_databases, "ibtracs", file_ibtracs)
    xds_ibtracs = xr.open_dataset(path_ibtracs)

    # all basins, all centers
    basin_all = ["NA", "SA", "WP", "EP", "SP", "NI", "SI"]
    center_all = [
        "USA",
        "TOKYO",
        "CMA",
        "HKO",
        "NEWDELHI",
        "REUNION",
        "BOM",
        "WELLINGTON",
        "NADI",
        "WMO",
    ]

    # set storms id as coordinate
    xds_ibtracs["stormid"] = (("storm"), xds_ibtracs.storm.values)
    xds_ibtracs.set_coords("stormid")

    # polynomial order
    N = 3
    coef_fit = np.nan * np.zeros((len(center_all), len(basin_all), N + 1))
    pres_data = np.nan * np.zeros((len(center_all), len(basin_all), 200000))
    wind_data = np.nan * np.zeros((len(center_all), len(basin_all), 200000))

    # loop for all centers
    for ic, center in enumerate(center_all):
        d_vns_center = get_dictionary_center(center=center)
        dict_land = d_vns_center["dist2land"]
        dict_pres = d_vns_center["pressure"]  # [mbar]
        dict_wind = d_vns_center["maxwinds"]  # [kt]
        basin_ids = d_vns_basinscenter[center]  # basin

        # loop for all basins
        for basin in basin_ids:
            # select storms genesis at basin X
            xds_basin = Extract_basin_storms(xds_ibtracs, basin)

            # extract and reshape basin data: pressure, wind, landbasin
            newshape = xds_basin.storm.size * xds_basin.date_time.size

            landbasin = xds_basin[dict_land].values.reshape(newshape)
            Pbasin = xds_basin[dict_pres].values.reshape(newshape)
            Wbasin = xds_basin[dict_wind].values.reshape(newshape)

            # winds are converted to 1-min avg [kt]
            Wbasin /= d_vns_windfac[center]

            PWbasin_s = np.column_stack((Pbasin, Wbasin, landbasin))

            # index for removing NaNs (including landmask)
            ix_nonan = ~np.isnan(PWbasin_s).any(axis=1)
            PWbasin_s = PWbasin_s[ix_nonan]

            # Fitting Polynomial Regression to the dataset
            X, y = PWbasin_s[:, 0], PWbasin_s[:, 1]
            u = polyfit(X, y, deg=N)

            # store coefficients
            ibasin = np.where(basin == np.array(basin_all))[0][0]
            coef_fit[ic, ibasin, :] = u
            pres_data[ic, ibasin, : PWbasin_s.shape[0]] = PWbasin_s[:, 0].T
            wind_data[ic, ibasin, : PWbasin_s.shape[0]] = PWbasin_s[:, 1].T

    # dataset
    xds = xr.Dataset(
        {
            "coef": (("center", "basin", "polynomial"), coef_fit),
            "pres": (("center", "basin", "data"), pres_data),
            "wind": (("center", "basin", "data"), wind_data),
        },
        {
            "center": np.asarray(center_all),
            "basin": np.asarray(basin_all),
        },
    )
    xds.coef.attrs["name"] = "Fitting polynomial coefficients (Pmin, Wmax)"
    xds.coef.attrs["units"] = "Pressure (mbar), Wind speed (kt, 1-min avg)"

    return xds


def check_ibtracs_coefs(path_data_shytcwaves, path_databases):
    """
    Check if the coefficient file exists, otherwise the file is generated

    returns:  xds_coef (xarray.Dataset)
    """

    # ibtracs fitting coefficients path
    path_coef = op.join(path_data_shytcwaves, file_coeff_pmin_wmax)

    if op.isfile(path_coef):
        xds_coef = xr.open_dataset(path_coef)
    else:
        print("Calculating Pmin-Wmax fitting coefficients...")
        xds_coef = ibtracs_fit_pmin_wmax(path_databases)
        print("Done \n")

    return xds_coef


###############################################################################
# STORM PROCESSING
# Historical storms extracted from IBTrACS database are selected by center and
# basin, preprocessed to assign statistical estimates of maximum winds and
# radii of maximum winds, when those variables are not provided. Track
# coordinates are interpolated at each swan computational timestep.
###############################################################################


def get_category(ycpres):
    "Returns storm category according to central pressure"

    categ = []
    for i in range(len(ycpres)):
        if (ycpres[i] == 0) or (np.isnan(ycpres[i])):
            categ.append(6)
        elif ycpres[i] < 920:
            categ.append(5)
        elif ycpres[i] < 944:
            categ.append(4)
        elif ycpres[i] < 964:
            categ.append(3)
        elif ycpres[i] < 979:
            categ.append(2)
        elif ycpres[i] < 1000:
            categ.append(1)
        elif ycpres[i] >= 1000:
            categ.append(0)

    return categ


def wind2rmw(wmax, vmean, latitude):
    """
    wmax    - maximum sustained winds [kt]
    vmean   - mean translational speed [kt]

    returns:  radii of maximum winds RMW (Knaff et al. 2016)  [nautical mile]
    """

    # constants
    pifac = np.arccos(-1) / 180  # pi/180
    beta = 0.9  # conversion factor of wind speed

    # Substract the translational storm speed from the observed maximum
    # wind speed to avoid distortion in the Holland curve fit.
    # The translational speed will be added back later
    vkt = wmax - vmean  # [kt]

    # Convert wind speed from 10m altitude to wind speed at the top of
    # the atmospheric boundary layer
    vgrad = vkt / beta  # [kt]

    # Knaff et al. (2016) - Radius of maximum wind (RMW)
    rm = (
        218.3784
        - 1.2014 * vgrad
        + np.power(vgrad / 10.9844, 2)
        - np.power(vgrad / 35.3052, 3)
        - 145.509 * np.cos(latitude * pifac)
    )  # [nmile]

    return rm


def get_vmean(lat1, lon1, lat2, lon2, deltat):
    """
    lat1,lon1,lat2,lon2  - segment endpoints
    deltat               - timestep [hours]

    returns:  forward direction and mean translation speed between coordinates
    """

    RE = 6378.135  # earth radius [km]
    arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)  # great circle

    r = arcl_h * np.pi / 180.0 * RE  # distance between coordinates [km]
    vmean = r / deltat  # translation speed [km/h]

    vu = vmean * np.sin((gamma_h + 180) * np.pi / 180)
    vy = vmean * np.cos((gamma_h + 180) * np.pi / 180)

    return gamma_h, vmean, vu, vy  # [º], [km/h]


def wind2pres(xds_coef, st_wind, st_center, st_basin):
    """
    xds_coef    - (xarray.Dataset) fitting coefficients for Pmin-Wmax
    st_wind     - storm maxwinds [kt, 1-min avg]
    st_center   - storm center (RSMC, IBTrACS)
    st_basin    - storm basin (NA,SA,WP,EP,SP,NI,SI)

    returns:  statistical maximum wind speed [kt, 1-min avg]
    """

    pmin_pred = []
    for iwind in st_wind:
        # select Pmin-Wmax coefficients
        coeff = xds_coef.sel(center=st_center, basin=st_basin).coef.values[:]

        # aisle equation
        coeff[-1] -= iwind

        # pressure root
        pmin_pred.append(
            np.real(np.roots(coeff))[np.where(np.imag(np.roots(coeff)) == 0)[0][0]]
        )

    return np.array(pmin_pred)  # [mbar]


def resample_storm_6h(storm):
    """
    Storms may provide data every 3h (more often than 6h) or at random times.
    In thoseee cases it is necessary to resample data every 6h for subsequent
    functions
    """
    storm = storm.isel(date_time=~np.isnat(storm.time.values))
    storm = storm.isel(date_time=~np.isnan(storm.wmo_pres.values))
    storm = storm.reset_coords(["lon", "lat"])
    ibasin = storm.basin.values[0]
    storm = (
        storm.swap_dims({"date_time": "time"}).resample(time="6H").interpolate("linear")
    )
    storm["basin"] = (("time"), np.array([ibasin] * storm.time.size))
    return storm


def historic_track_preprocessing(
    xds,
    path_data_shytcwaves,
    path_databases,
    center="WMO",
    forecast_on=False,
    database_on=False,
    st_param=False,
):
    """
    xds         - (xarray.Dataset) historic storm track dataset (storm dim)
    center      - ibtracs center code
    forecast_on - True when track is forecasted (not IBTrACS)
    database_on - True when data is kept at 0,6,12,18 hours
    st_param    - True when data is kept as original

    Historic track is preprocessed: remove NaT data, remove NaN data,
    longitude convention [0º-360º], time format, calculate vmean,
    get storm category, convert maxwinds to 1-min average

    * Returns historical data only

    returns:  df (pandas.Dataframe)
    """

    # dictionary for IBTrACS center
    d_vns = get_dictionary_center(center=center)

    print(center)

    # get names of variables
    nm_tim = d_vns["time"]
    nm_bas = d_vns["basin"]
    nm_lon = d_vns["longitude"]
    nm_lat = d_vns["latitude"]
    nm_prs = d_vns["pressure"]
    nm_win = d_vns["maxwinds"]
    nm_rmw = d_vns["rmw"]

    # get var time
    ytime = xds[nm_tim].values  # dates format: datetime64

    # remove NaTs (time)
    ydist = xds["dist2land"].values[~np.isnat(ytime)]  # distance to land [km]
    ybasin = xds[nm_bas].values[~np.isnat(ytime)]  # basin
    ylat_tc = xds[nm_lat].values[~np.isnat(ytime)]  # latitude
    ylon_tc = xds[nm_lon].values[~np.isnat(ytime)]  # longitude
    ycpres = xds[nm_prs].values[~np.isnat(ytime)]  # pressure [mbar]
    ywind = xds[nm_win].values[~np.isnat(ytime)]  # wind speed [kt]
    if nm_rmw:
        yradii = xds[nm_rmw].values[~np.isnat(ytime)]  # rmw [nmile]
    ytime = ytime[~np.isnat(ytime)]

    # remove common NaNs (pressure & wind)
    posnonan_p_w = np.unique(
        np.concatenate((np.argwhere(~np.isnan(ycpres)), np.argwhere(~np.isnan(ywind))))
    )
    ytime = ytime[posnonan_p_w]
    ydist = ydist[posnonan_p_w]
    ybasin = ybasin[posnonan_p_w]
    ylat_tc = ylat_tc[posnonan_p_w]
    ylon_tc = ylon_tc[posnonan_p_w]
    ycpres = ycpres[posnonan_p_w]
    ywind = ywind[posnonan_p_w]
    if nm_rmw:
        yradii = yradii[posnonan_p_w]
    if not nm_rmw:
        yradii = np.full(ycpres.size, np.nan)  # [np.nan] * ycpres.size

    ###########################################################################
    # forecast input may lack pressure
    if forecast_on:
        # convert basin ids (IBTrACS equivalent)
        dict_basin_forecast = {
            "AL": "NA",
            "LS": "SA",
            "EP": "EP",
            "CP": "WP",
            "WP": "WP",
            "SH": "SP",
            "IO": "NI",
        }
        ybasin = np.array([dict_basin_forecast[ybas] for ybas in ybasin])
        ybasin[(ylat_tc < 0) & (ylon_tc < 135)] = "SI"

        if np.isnan(ycpres).any():
            # ibtracs fitting coefficients path
            xds_coef = check_ibtracs_coefs(path_data_shytcwaves, path_databases)

            # fill pressure gaps
            ycpres[np.isnan(ycpres)] = wind2pres(
                xds_coef, ywind[np.isnan(ycpres)], center, ybasin[0]
            )

        ybasin = np.array([np.bytes_(ybas) for ybas in ybasin])
    ###########################################################################

    # remove NaNs (pressure)
    ytime = ytime[~np.isnan(ycpres)]
    st_dist2land = ydist[~np.isnan(ycpres)]
    ybasin = ybasin[~np.isnan(ycpres)]
    st_lat = ylat_tc[~np.isnan(ycpres)]
    st_lon = ylon_tc[~np.isnan(ycpres)]
    st_pres = ycpres[~np.isnan(ycpres)]
    st_wind = ywind[~np.isnan(ycpres)]
    if nm_rmw:
        st_rmw = yradii[~np.isnan(ycpres)]
    if not nm_rmw:
        st_rmw = np.full(st_pres.size, np.nan)  # [np.nan] * st_pres.size

    # conflicting times
    dft = pd.DataFrame({"time": ytime})
    dft["time"] = pd.to_datetime(dft["time"], format="%Y-%m-%d %H:%M:%S").dt.round("1h")
    hr = dft["time"].dt.hour.values

    # duplicate times
    pos_out = np.where(np.diff(hr) == 0)[0]

    # keep 0,3,6,9,12,15,18 hours (remove in between data)
    # TODO: option to interpolate to fixed target hours
    if database_on:
        pos_in = np.where((hr == 0) | (hr == 6) | (hr == 12) | (hr == 18))[0]
    else:
        pos_in = np.where(
            (hr == 0)
            | (hr == 6)
            | (hr == 12)
            | (hr == 18)
            | (hr == 3)
            | (hr == 9)
            | (hr == 15)
            | (hr == 21)
        )[0]
    # for parameterized tracks hours can be random
    if st_param:
        pos_in = np.where(hr > 0)

    # remove duplicates
    pos = np.setdiff1d(pos_in, pos_out)  # pos_in[pos_in != pos_out].ravel()
    ytime = ytime[pos]
    st_dist2land = st_dist2land[pos]
    ybasin = ybasin[pos]
    st_lat = st_lat[pos]
    st_lon = st_lon[pos]
    st_pres = st_pres[pos]
    st_wind = st_wind[pos]
    st_rmw = st_rmw[pos]

    # only when storm data available
    if st_pres.size > 0:
        # longitude convention: [0º,360º]
        st_lon[st_lon < 0] = st_lon[st_lon < 0] + 360

        # round dates to hour ---> half hour
        round_to = 3600 / 2
        st_time = []
        for i in range(len(ytime)):
            dt = ytime[i].astype("datetime64[s]").tolist()
            seconds = (dt - dt.min).seconds
            rounding = (seconds + round_to / 2) // round_to * round_to
            out = dt + timedelta(0, rounding - seconds, -dt.microsecond)
            st_time.append(out)
        st_time = np.asarray(st_time)

        # storm coordinates timestep [hours]
        ts = st_time[1:] - st_time[:-1]
        ts = [ts[i].total_seconds() / 3600 for i in range(ts.size)]
        ts.append(np.nan)

        # calculate Vmean
        st_vmean, st_move = [], []
        for i in range(0, len(st_time) - 1):
            # consecutive storm coordinates
            lon1, lon2 = st_lon[i], st_lon[i + 1]
            lat1, lat2 = st_lat[i], st_lat[i + 1]

            # translation speed
            gamma_h, vel_mean, _, _ = get_vmean(lat1, lon1, lat2, lon2, ts[i])
            st_vmean.append(vel_mean / 1.852)  # translation speed [km/h to kt]
            st_move.append(gamma_h)  # forward direction [º]
        st_vmean.append(np.nan)
        st_move.append(np.nan)

        # mean value
        vmean = np.nanmean(st_vmean)  # [kt]

        # storm category
        categ = get_category(st_pres)

        # convert (X)-min to 1-min avg winds (depends on center)
        st_wind = st_wind / d_vns_windfac[center]

        # get basin string
        st_basin = [str(c.decode("UTF-8")) for c in ybasin]

        # store storm variables
        df = pd.DataFrame(
            index=st_time,
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

        df["center"] = center
        df["basin"] = st_basin
        if "dist2land" in xds.keys():
            df["dist2land"] = st_dist2land
        else:
            df["dist2land"] = np.nan
        df["longitude"] = st_lon
        df["latitude"] = st_lat
        df["move"] = st_move
        df["mean_velocity"] = st_vmean
        df["pressure"] = st_pres
        df["maxwinds"] = st_wind
        df["rmw"] = st_rmw
        df["category"] = categ
        df["timestep"] = ts
        df["storm_vmean"] = vmean

        df.attrs = {
            "dist2land": "km",
            "velocity": "kt",
            "pressure": "mbar",
            "maxwinds": "kt, 1-min average sustained winds (converted from X-min)",
            "rmw": "nmile",
            "timestep": "hour",
        }

    else:
        df = pd.DataFrame(
            index=[],
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

    return df


def historic_track_interpolation(
    df,
    dt_comp,
    path_data_shytcwaves,
    path_databases,
    y0=None,
    x0=None,
    great_circle=True,
    wind_estimate_on=False,
    fit=False,
    interpolation=True,
    mode="first",
    radi_estimate_on=True,
):
    """
    df                    - (pandas.Dataframe) storm variables
    x0, y0                - target coordinates (longitude, latitude)
    lat0,lon0,lat1,lon1   - bound limits for numerical domain
    dt_comp               - computation time step [minutes]

    great_circle          - True for distances over great circles
    wind_estimate_on      - True for empirical estimate instead of historical
    fit                   - True for empirical vmax at wind=0 (start of storm)
    interpolation         - True for interpolating storm historic variables
                            False for constant storm segments, in that case:
    mode ('first','mean') - chooses value for constant segments
    radi_estimate_on      - by default NaN values are estimated (Knapp et al.)
                            False for interpolation if some values are given

    Calculates storm track variables at interpolated track points (swan
    computational timestep "dt_comp") from ibtracs historical data and from
    empirical estimates that fill historical data gaps:
    * maximum winds:        fitting coefficients for Pmin-Wmax (center,basin)
    * radii maximum winds:  Knaff et al. 2016 approximation

    returns:  st (pandas.Dataframe) with interpolated storm coordinate variables
    """

    RE = 6378.135  # earth radius [km]

    # historic storm variables
    st_time = df.index.values[:]  # datetime format
    st_center = df["center"].values[0]
    st_basin = df["basin"].values[:]
    st_dist2land = df["dist2land"].values[:]  # [km]
    st_lon = df["longitude"].values[:]
    st_lat = df["latitude"].values[:]
    st_vmean = df["mean_velocity"].values[:]  # [kt]
    st_pres = df["pressure"].values[:]  # [mbar]
    st_wind = df["maxwinds"].values[:]  # [kt, 1-min avg]
    st_rmw = df["rmw"].values[:]  # [nmile]
    st_step = df["timestep"].values[:]  # [hour]
    wind_fill = np.full(st_pres.size, False)
    rmw_fill = np.full(st_pres.size, False)

    # select Pmin-Wmax polynomial fitting coefficients (IBTrACS center,basin)
    xds_coef = check_ibtracs_coefs(path_data_shytcwaves, path_databases)
    coefs = xds_coef.sel(center=st_center, basin=st_basin).coef.values[:]

    p1, p2, p3, p4 = coefs[:, 0], coefs[:, 1], coefs[:, 2], coefs[:, 3]
    wind_estimate = (
        p1 * np.power(st_pres, 3)
        + p2 * np.power(st_pres, 2)
        + p3 * np.power(st_pres, 1)
        + p4
    )

    # maxwinds gaps filled with Pmin-Wmax coefficients
    posnan = np.argwhere(np.isnan(st_wind))  # np.where(st_wind==np.nan)[0]
    poszero = np.where(st_wind == 0)[0]
    if np.isnan(st_wind).all() or wind_estimate_on:  # wind NOT provided
        wind_fill = np.full(st_wind.size, True)
        st_wind = wind_estimate
    ###########################################################################
    elif np.isnan(st_wind).any():  # wind with some NaNs
        if fit and posnan.size > 0:  # data filled (for wind=0)
            wind_fill[posnan] = True
            st_wind[posnan] = wind_estimate[posnan]
    elif fit and poszero.size > 0:  # wind provided
        wind_fill[poszero] = True
        st_wind[poszero] = wind_estimate[poszero]
    ###########################################################################

    #    else:                                             # wind provided
    #        pos = np.where(st_wind==0)[0]
    #        if fit and pos.size>0:    # data filled (for wind=0)
    #            wind_fill[pos] = True
    #            st_wind[pos]   = wind_estimate[pos]

    # radii of maximum winds gaps filled with Knaff et al. (2016) estimate
    rmw_estimate = wind2rmw(st_wind, st_vmean, st_lat)
    if radi_estimate_on:
        if np.isnan(st_rmw).all():  # all
            rmw_fill = np.full(rmw_estimate.size, True)
            st_rmw = rmw_estimate
        elif np.isnan(st_rmw).any():  # some
            rmw_fill[np.isnan(st_rmw)] = True
            st_rmw[np.isnan(st_rmw)] = rmw_estimate[np.isnan(st_rmw)]
    else:
        if np.isnan(st_rmw).any():
            st_rmw = df.rmw.interpolate(method="time").values

    # storm variables (original values or mean values)
    if mode == "mean":
        st_pres = np.mean((st_pres, np.append(st_pres[1:], np.nan)), axis=0)
        st_wind = np.mean((st_wind, np.append(st_wind[1:], np.nan)), axis=0)
        st_rmw = np.mean((st_rmw, np.append(st_rmw[1:], np.nan)), axis=0)

    # number of time steps between consecutive interpolated storm coordinates
    # to match SWAN computational timestep
    ts_h = st_step  # hours
    nts = np.asarray(st_step) * 60 / dt_comp  # number of intervals

    # initialize interpolated variables
    bas, dist, lon, lat, move, vmean, vu, vy = [], [], [], [], [], [], [], []
    p0, vmax, rmw, vmaxfill, rmwfill = [], [], [], [], []
    time_input = np.empty((0,), dtype="datetime64[ns]")

    for i in range(0, len(st_time) - 1):
        # time array for SWAN input
        date_ini = st_time[i]
        time_input0 = pd.date_range(
            date_ini, periods=int(nts[i]), freq="{0}MIN".format(dt_comp)
        )
        time_input = np.append(np.array(time_input), np.array(time_input0))

        # consecutive storm coordinates
        lon1, lon2 = st_lon[i], st_lon[i + 1]
        lat1, lat2 = st_lat[i], st_lat[i + 1]

        # translation speed
        arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)
        r = arcl_h * np.pi / 180.0 * RE  # distance between coordinates [km]
        dx = r / nts[i]  # distance during time step [km]
        tx = ts_h[i] / nts[i]  # time period during time step [h]
        vx = float(dx) / tx / 3.6  # translation speed [km/h to m/s]
        vx = vx / 0.5144  # translation speed [m/s to kt]

        # interpolate at timesteps
        for j in range(int(nts[i])):
            bas.append(st_basin[i])
            dist.append(st_dist2land[i])

            # append interpolated lon, lat
            if not great_circle:
                glon = (
                    lon1 - (dx * 180 / (RE * np.pi)) * np.sin(gamma_h * np.pi / 180) * j
                )
                glat = (
                    lat1 - (dx * 180 / (RE * np.pi)) * np.cos(gamma_h * np.pi / 180) * j
                )
            else:
                glon, glat, baz = shoot(lon1, lat1, gamma_h + 180, float(dx) * j)
            lon.append(glon)
            lat.append(glat)

            # append storm track parameters
            move.append(gamma_h)
            vmean.append(vx)
            vu.append(vx * np.sin((gamma_h + 180) * np.pi / 180))
            vy.append(vx * np.cos((gamma_h + 180) * np.pi / 180))

            # append pmin, wind, rmw (interpolated/constant)
            if interpolation:
                p0.append(st_pres[i] + j * (st_pres[i + 1] - st_pres[i]) / nts[i])
                vmax.append(st_wind[i] + j * (st_wind[i + 1] - st_wind[i]) / nts[i])
                rmw.append(st_rmw[i] + j * (st_rmw[i + 1] - st_rmw[i]) / nts[i])
                vmaxfill.append(wind_fill[i] or wind_fill[i + 1])
                rmwfill.append(rmw_fill[i] or rmw_fill[i + 1])
            else:
                p0.append(st_pres[i])
                vmax.append(st_wind[i])
                rmw.append(st_rmw[i])
                vmaxfill.append(wind_fill[i])
                rmwfill.append(rmw_fill[i])

    # longitude convention [0º,360º]
    lon = np.array(lon)
    lon[lon < 0] = lon[lon < 0] + 360

    # method attribute
    n_hour = dt_comp / 60
    if interpolation:
        method = "interpolation ({0}h)".format(n_hour)
    else:
        if mode == "mean":
            method = "segment ({0}h, mean value)".format(n_hour)
        elif mode == "first":
            method = "segment ({0}h, node value)".format(n_hour)

    # storm track (pd.DataFrame)
    st = pd.DataFrame(
        index=time_input,
        columns=[
            "center",
            "basin",
            "dist2land",
            "lon",
            "lat",
            "move",
            "vf",
            "vfx",
            "vfy",
            "pn",
            "p0",
            "vmax",
            "rmw",
            "vmaxfill",
            "rmwfill",
        ],
    )

    st["center"] = st_center
    st["basin"] = bas
    st["dist2land"] = dist  # distance to nearest land [km]
    print(len(lon))
    st["lon"] = lon  # longitude coordinate
    st["lat"] = lat  # latitude coordinate
    st["move"] = move  # gamma, forward direction
    st["vf"] = vmean  # translational speed [kt]
    st["vfx"] = vu  # x-component
    st["vfy"] = vy  # y-component
    st["pn"] = 1013  # average pressure at the surface [mbar]
    st["p0"] = p0  # minimum central pressure [mbar]
    st["vmax"] = vmax  # maximum winds [kt, 1-min avg]
    st["rmw"] = rmw  # radii of maximum winds [nmile]
    st["vmaxfill"] = vmaxfill  # True if maxwinds filled with Pmin-Vmax
    st["rmwfill"] = rmwfill  # True if radii filled with Knaff 2016

    # add metadata
    st.attrs = {
        "method": method,
        "center": st_center,
        "dist": "km",
        "vf": "kt",
        "p0": "mbar",
        "vmax": "kt, 1-min avg",
        "rmw": "nmile",
        "x0": x0,
        "y0": y0,
        "R": 4,
    }

    return st, time_input


def track_triming(st, lat00, lon00, lat01, lon01):
    """
    st       - (pandas.Dataframe) storm track variables
    lon,lat  - geographical limits of target domain

    all storm coordinates outside the limits of target domain are discarded

    returns:  st_trim (pandas.Dataframe)
    """

    lo, la = st.lon.values[:], st.lat.values[:]

    # select track coordinates within the target domain area
    st_trim = st.iloc[(lo <= lon01) & (lo >= lon00) & (la <= lat01) & (la >= lat00)]

    # get min/max times within the area
    tmin = np.min(st_trim.index.values)
    tmax = np.max(st_trim.index.values)

    # select all the time window
    st_trim = st.iloc[(st.index >= tmin) & (st.index <= tmax)]

    # add metadata
    st_trim.attrs = st.attrs

    return st_trim


def track_triming_circle(st, plon, plat, radii):
    """
    st        - (pandas.Dataframe) storm track variables
    plon,plat - circle center
    radii     - circle domain

    all storm coordinates outside the limits of target domain are discarded

    returns:  st_trim (pandas.Dataframe)
    """

    lo, la = st.lon.values[:], st.lat.values[:]

    # stack storm longitude, latitude
    lonlat_s = np.column_stack((lo, la))

    # calculate geodesic distance (degree)
    geo_dist = []
    for lon_ps, lat_ps in lonlat_s:
        geo_dist.append(GeoDistance(lat_ps, lon_ps, plat, plon))
    geo_dist = np.asarray(geo_dist)

    # find storm inside circle and calculate parameters
    if (geo_dist < radii).any() & (geo_dist.size > 1):
        # storm inside circle
        ix_in = np.where(geo_dist < radii)[0][:]
        # select track coordinates within the target domain area
        st_trim = st.iloc[ix_in]

    else:
        st_trim = st.iloc[st.lat.values[:] > 90]

    # get min/max times within the area
    tmin = np.min(st_trim.index.values)
    tmax = np.max(st_trim.index.values)

    # select all the time window
    st_trim = st.iloc[(st.index >= tmin) & (st.index <= tmax)]

    # add metadata
    st_trim.attrs = st.attrs

    return st_trim


def stopmotion_trim_circle(df_seg, plon, plat, radii):
    """
    df_seg    - (pandas.Dataframe) stopmotion segments
    plon,plat - circle center
    radii     - circle domain

    all storm coordinates outside the limits of target domain are discarded

    returns:  st_trim (pandas.Dataframe)
    """

    lo0, la0 = df_seg.lon_i.values[:], df_seg.lat_i.values[:]
    lo1, la1 = df_seg.lon_t.values[:], df_seg.lat_t.values[:]

    # stack storm longitude, latitude
    lonlat_s0 = np.column_stack((lo0, la0))
    lonlat_s1 = np.column_stack((lo1, la1))

    # calculate geodesic distance (degree)
    geo_dist0, geo_dist1 = [], []
    for lon_ps, lat_ps in lonlat_s0:
        geo_dist0.append(GeoDistance(lat_ps, lon_ps, plat, plon))
    geo_dist0 = np.asarray(geo_dist0)
    for lon_ps, lat_ps in lonlat_s1:
        geo_dist1.append(GeoDistance(lat_ps, lon_ps, plat, plon))
    geo_dist1 = np.asarray(geo_dist1)

    # get minimum distance of the two target endpoints
    geo_dist = np.min((geo_dist0, geo_dist1), axis=0)

    # find storm inside circle and calculate parameters
    if (geo_dist < radii).any() & (geo_dist.size > 1):
        # storm inside circle
        ix_in = np.where(geo_dist < radii)[0][:]
        # select track coordinates within the target domain area
        st_trim = df_seg.iloc[ix_in]

        # get min/max times within the area
        tmin = np.min(st_trim.index.values)
        tmax = np.max(st_trim.index.values)

        # select all the time window
        st_trim = df_seg.iloc[(df_seg.index >= tmin) & (df_seg.index <= tmax)]

        # add metadata
        st_trim.attrs = df_seg.attrs

    else:  # empty dataframe
        st_trim = pd.DataFrame(columns=df_seg.columns)

    return st_trim


def track_extent(st, time_input, dt_comp, time_extent=48):
    """
    Function to concatenate an additional simulation period to evaluate
    propagation of the wave directional spectra (under no wind forcing)

    st           - (pandas.Dataframe) storm track variables
    time_input   - (numpy.ndarray) time coordinates
    dt_comp      - (float) swan timestep [minutes]
    time_extent  - (float) time to extent dataframe with NaNs [hours]

    all storm coordinates outside the limits of target domain are discarded

    returns:  st_new (pandas.Dataframe)
    """

    # total simulation period (hours): storm + propagation
    time_storm = (time_input[-1] - time_input[0]).astype(
        "timedelta64[h]"
    ) / np.timedelta64(1, "h")
    T = time_storm + time_extent

    # original track period
    size_ini = st.index.size
    date_ini = st.index[0]

    # total simulation period
    size_new = int(T * 60 / dt_comp)
    time_input_new = pd.date_range(
        date_ini, periods=size_new, freq="{0}MIN".format(dt_comp)
    )
    st_new = pd.DataFrame(index=time_input_new, columns=list(st.keys()))

    # combine
    st_new.values[:size_ini, :] = st.values
    if st.attrs["x0"]:
        st_new.x0 = st.attrs["x0"]
    if st.attrs["y0"]:
        st_new.y0 = st.attrs["y0"]

    # [OPTIONAL] override SWAN storm case computational delta time
    st_new.attrs["override_dtcomp"] = "{0} MIN".format(dt_comp)

    # generate wave event (empty)
    we = pd.DataFrame(
        index=time_input_new, columns=["hs", "t02", "dir", "spr", "U10", "V10"]
    )
    we["level"] = 0
    we["tide"] = 0

    return st_new, we


###############################################################################
# HYTCWAVES
# PARAMETERIZED storm tracks
# Given the storm parameters (HyTCWaves methodology) the track coordinates are
# calculated at each swan computational timestep
###############################################################################


def entrance_coords(delta, gamma, x0, y0, R, lon0, lon1, lat0, lat1):
    """
    delta, gamma               - storm track parameters
    x0, y0                     - site coordinates (longitude, latitude)
    R                          - radius (º)
    lon0, lon1, lat0, lat1     - computational coordinates (outer grid)

    returns:  (x1,y1) initial storm coordinates within computational domain
    """

    # enter point in the radius
    xc = x0 + R * np.sin(delta * np.pi / 180)
    yc = y0 + R * np.cos(delta * np.pi / 180)

    # calculate angles that determine the storm boundary entrance  [degrees]
    ang_1 = np.arctan((lon1 - xc) / (lat1 - yc)) * 180 / np.pi  # upper right corner
    ang_2 = np.arctan((lon1 - xc) / (lat0 - yc)) * 180 / np.pi + 180  # lower right
    ang_3 = np.arctan((lon0 - xc) / (lat0 - yc)) * 180 / np.pi + 180  # lower left
    ang_4 = np.arctan((lon0 - xc) / (lat1 - yc)) * 180 / np.pi + 360  # upper left

    if (gamma > ang_1) & (gamma < ang_2):
        x1 = lon1
        d = (x1 - xc) / np.sin(gamma * np.pi / 180)
        y1 = yc + d * np.cos(gamma * np.pi / 180)

    elif (gamma > ang_2) & (gamma < ang_3):
        y1 = lat0
        d = (y1 - yc) / np.cos(gamma * np.pi / 180)
        x1 = xc + d * np.sin(gamma * np.pi / 180)

    elif (gamma > ang_3) & (gamma < ang_4):
        x1 = lon0
        d = (x1 - xc) / np.sin(gamma * np.pi / 180)
        y1 = yc + d * np.cos(gamma * np.pi / 180)

    elif (gamma > ang_4) | (gamma < ang_1):
        y1 = lat1
        d = (y1 - yc) / np.cos(gamma * np.pi / 180)
        x1 = xc + d * np.sin(gamma * np.pi / 180)

    return x1, y1


def track_site_parameters(
    step,
    pmin,
    vmean,
    delta,
    gamma,
    x0,
    y0,
    lon0,
    lon1,
    lat0,
    lat1,
    R,
    date_ini,
    center="WMO",
    basin="SP",
):
    """
    step                       - computational time step (minutes)
    pmin, vmean, delta, gamma  - storm track parameters   (NOTE: vmean in [kt])
    x0, y0                     - site coordinates (longitude, latitude)
    lon0, lon1, lat0, lat1     - enter point in computational grid
    R                          - radius (º)
    date_ini                   - initial date 'yyyy-mm-dd HH:SS'
    great_circle               - default option

    Calculates storm track variables from storm parameters within the study area

    returns:  st (pandas.Dataframe)
    """

    # storm entrance coordinates at the domain boundary
    x1, y1 = entrance_coords(delta, gamma, x0, y0, R, lon0, lon1, lat0, lat1)

    # calculate computational timestep storm coordinates
    # Note: velocity input in shoot function must be km/h
    lon, lat = [x1], [y1]
    i = 1
    glon, glat, baz = shoot(x1, y1, gamma + 180, vmean * 1.852 * i * step / 60)
    if glon < 0:
        glon += 360
    while (glon < lon1) & (glon > lon0) & (glat < lat1) & (glat > lat0):
        lon.append(glon)
        lat.append(glat)
        i += 1
        glon, glat, baz = shoot(x1, y1, gamma + 180, vmean * 1.852 * i * step / 60)
        if glon < 0:
            glon += 360
    frec = len(lon)

    # select Pmin-Vmax polynomial fitting coefficients (IBTrACS center,basin)
    xds_coef = check_ibtracs_coefs(path_data_shytcwaves, path_databases)
    p1, p2, p3, p4 = xds_coef.sel(center=center, basin=basin).coef.values[:]
    wind_estimate = (
        p1 * np.power(pmin, 3) + p2 * np.power(pmin, 2) + p3 * np.power(pmin, 1) + p4
    )

    # radii of maximum winds is filled with Knaff (2016) estimate
    rmw_estimate = wind2rmw(
        np.full(lat.size, wind_estimate), np.full(lat.size, vmean), lat
    )

    # velocity components
    vfx = vmean * np.sin((gamma + 180) * np.pi / 180)  # [kt]
    vfy = vmean * np.cos((gamma + 180) * np.pi / 180)  # [kt]

    # time array for SWAN input
    time_input = pd.date_range(date_ini, periods=frec, freq="{0}min".format(step))

    # storm track (pd.DataFrame)
    st = pd.DataFrame(
        index=time_input,
        columns=["lon", "lat", "move", "vf", "vfx", "vfy", "pn", "p0", "vmax", "rmw"],
    )

    st["lon"] = lon
    st["lat"] = lat
    st["move"] = gamma  # gamma, forward direction
    st["vf"] = vmean  # translation speed [kt]
    st["vfx"] = vfx  # x-component
    st["vfy"] = vfy  # y-component
    st["pn"] = 1013  # average pressure at the surface [mbar]
    st["p0"] = pmin  # minimum central pressure [mbar]
    st["vmax"] = wind_estimate  # maximum winds [kt, 1-min avg]
    st["rmw"] = rmw_estimate  # radii of maximum winds [nmile]

    # add metadata
    st.attrs = {
        "vf": "kt",
        "p0": "mbar",
        "vmax": "kt, 1-min avg",
        "rmw": "nmile",
        "x0": x0,
        "y0": y0,
        "R": 4,
    }

    return st


###############################################################################
# synthetic tracks


def nakajo_track_preprocessing(xds, center="WMO"):
    """
    xds         - (xarray.Dataset) historic storm track dataset (storm dim)

    Synthetic track is preprocessed: remove NaT data, remove NaN data,
    longitude convention [0º-360º], time format, calculate vmean,
    get storm category (winds, rmw are not provided)

    returns:  df (pandas.Dataframe)
    """

    # get names of variables
    nm_tim = "time"
    nm_lon = "ylon_TC"
    nm_lat = "ylat_TC"
    nm_prs = "yCPRES"

    # get var time
    ytime = xds[nm_tim].values  # dates format: datetime64

    # remove NaTs (time)
    ylat_tc = xds[nm_lat].values[~np.isnat(ytime)]  # latitude
    ylon_tc = xds[nm_lon].values[~np.isnat(ytime)]  # longitude
    ycpres = xds[nm_prs].values[~np.isnat(ytime)]  # pressure [mbar]
    ytime = ytime[~np.isnat(ytime)]

    # remove common NaNs (pressure)
    posnonan_p_w = np.unique(np.argwhere(~np.isnan(ycpres)))
    ytime = ytime[posnonan_p_w]
    st_lat = ylat_tc[posnonan_p_w]
    st_lon = ylon_tc[posnonan_p_w]
    st_pres = ycpres[posnonan_p_w]

    ###########################################################################
    # longitude convention: [0º,360º]
    st_lon[st_lon < 0] = st_lon[st_lon < 0] + 360

    # get basin
    ybasin = np.empty(ytime.size, dtype="S10")
    for i in range(ytime.size):
        ilo, ila = st_lon[i], st_lat[i]
        if (ila < 0) & (ilo < 135) & (ilo > 20):
            ybasin[i] = "SI"
        elif (ila < 0) & (ilo > 135) & (ilo < 290):
            ybasin[i] = "SP"
        elif (ila < 0) & (ilo < 20) & (ilo > 290):
            ybasin[i] = "SA"
        elif (ila > 0) & (ilo > 20) & (ilo < 100):
            ybasin[i] = "NI"
        elif (ila > 0) & (ilo > 100) & (ilo < 180):
            ybasin[i] = "WP"
        elif (ila > 0) & (ilo > 180) & (ilo < 260):
            ybasin[i] = "EP"
        elif (ila > 0) & (ila < 15) & (ilo > 260) & (ilo < 275):
            ybasin[i] = "EP"
        else:
            ybasin[i] = "NA"

    ###########################################################################

    # only when storm data available
    if st_pres.size > 0:
        # round dates to hour
        round_to = 3600
        st_time = []
        for i in range(len(ytime)):
            dt = ytime[i].astype("datetime64[s]").tolist()
            seconds = (dt - dt.min).seconds
            rounding = (seconds + round_to / 2) // round_to * round_to
            out = dt + timedelta(0, rounding - seconds, -dt.microsecond)
            st_time.append(out)
        st_time = np.asarray(st_time)

        # storm coordinates timestep [hours]
        ts = st_time[1:] - st_time[:-1]
        ts = [ts[i].total_seconds() / 3600 for i in range(ts.size)]
        ts.append(np.nan)

        # calculate Vmean
        st_vmean, st_move = [], []
        for i in range(0, len(st_time) - 1):
            # consecutive storm coordinates
            lon1, lon2 = st_lon[i], st_lon[i + 1]
            lat1, lat2 = st_lat[i], st_lat[i + 1]

            # translation speed
            gamma_h, vel_mean, _, _ = get_vmean(lat1, lon1, lat2, lon2, ts[i])
            st_vmean.append(vel_mean / 1.852)  # translation speed [km/h to kt]
            st_move.append(gamma_h)  # forward direction [º]
        st_vmean.append(np.nan)
        st_move.append(np.nan)

        # mean value
        vmean = np.nanmean(st_vmean)  # [kt]

        # storm category
        categ = get_category(st_pres)

        # get basin string
        st_basin = [str(c.decode("UTF-8")) for c in ybasin]

        # store storm variables
        df = pd.DataFrame(
            index=st_time,
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

        df["center"] = center  # hypothesis winds 1-min
        df["basin"] = st_basin
        df["dist2land"] = 100
        df["longitude"] = st_lon
        df["latitude"] = st_lat
        df["move"] = st_move
        df["mean_velocity"] = st_vmean
        df["pressure"] = st_pres
        df["maxwinds"] = np.nan
        df["rmw"] = np.nan
        df["category"] = categ
        df["timestep"] = ts
        df["storm_vmean"] = vmean

        df.attrs = {
            "dist2land": "km",
            "velocity": "kt",
            "pressure": "mbar",
            "maxwinds": "kt, 1-min average sustained winds (converted from X-min)",
            "rmw": "nmile",
            "timestep": "hour",
        }

    else:
        df = pd.DataFrame(
            index=[],
            columns=[
                "center",
                "basin",
                "dist2land",
                "longitude",
                "latitude",
                "move",
                "mean_velocity",
                "pressure",
                "maxwinds",
                "rmw",
                "category",
                "timestep",
                "storm_vmean",
            ],
        )

    return df

import datetime
import os.path as op
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from ..core.constants import EARTH_RADIUS
from ..core.geo import geodesic_azimuth, geodesic_distance_azimuth, shoot
from ..datamining.mda import find_nearest_indices
from ..waves.superpoint import superpoint_calculation
from .tracks import (
    PATHS,
    get_vmean,
    historic_track_interpolation,
    historic_track_preprocessing,
    track_triming,
)

###############################################################################
# STOPMOTION functions
# functions to preprocess and interpolate coordinates at swan computational
# timesteps for any storm track according to the SHyTCWaves methodology units:
# 6-hour target segment preceded by 24-hour warmup segments

# storm2stopmotion --> parameterization of a storm track into segments
# stopmotion_interpolation --> generate stop-motion events
###############################################################################


def storm2stopmotion(df_storm: pd.DataFrame) -> pd.DataFrame:
    """
    Generate stopmotion segments from storm track.

    Parameters
    ----------
    df_storm : pd.DataFrame
        Storm track DataFrame containing:
        - lon : Longitude coordinates
        - lat : Latitude coordinates
        - p0 : Minimum central pressure (mbar)
        - vmax : Maximum sustained winds (kt)
        - rmw : Radius of maximum winds (nmile)
        - vmaxfill : Boolean indicating if winds were filled with estimates
        - rmwfill : Boolean indicating if RMW was filled with estimates

    Returns
    -------
    pd.DataFrame
        Stopmotion segments with columns:
        - vseg : Mean translational speed (kt)
        - dvseg : Speed variation (kt)
        - pseg : Segment pressure (mbar)
        - dpseg : Pressure variation (mbar)
        - wseg : Segment maximum winds (kt)
        - dwseg : Wind variation (kt)
        - rseg : Segment RMW (nmile)
        - drseg : RMW variation (nmile)
        - aseg : Azimuth from geographic North (degrees)
        - daseg : Azimuth variation (degrees)
        - lseg : Latitude (degrees)
        - laseg : Absolute latitude (degrees)
        - dlaseg : Latitude variation (degrees)
        - lon_w : Warmup origin longitude
        - lat_w : Warmup origin latitude
        - lon_i : Target origin longitude
        - lat_i : Target origin latitude
        - lon_t : Target endpoint longitude
        - lat_t : Target endpoint latitude

    Notes
    -----
    The function generates stopmotion segments methodology from 6-hour storm track segments:

    A. Warmup segment (24h):
        - 4 segments to define start/end coordinates
        - Defines {Vmean, relative angle}
        - Last 4th segment defines mean {Pmin, Wmax, Rmw}
        - Endpoint defines {lat}

    B. Target segment (6h):
        Defines variations {dP, dV, dW, dR, dAng}

    The absolute value of latitude is stored (start of target segment).
    Relative angle is referenced to geographic north (southern hemisphere
    is multiplied by -1).
    """

    # remove NaNs
    df_ = df_storm.dropna()
    df_["time"] = df_.index.values

    # constant segments variables
    lon = df_["lon"].values[:]
    lat = df_["lat"].values[:]
    pres = df_["p0"].values[:]  # [mbar]
    wind = df_["vmax"].values[:]  # [kt]
    rmw = df_["rmw"].values[:]  # [nmile]

    # generate stopmotion segments: 24h warmup + 6h target segments

    # timestep [hours]
    dt = np.diff(df_.index) / np.timedelta64(1, "h")

    # warmup 4-segments (24h) variables
    lo0 = np.full(df_.shape[0], np.nan)  # warmup x-coordinate
    la0 = np.full(df_.shape[0], np.nan)  # warmup y-coordinate
    vseg = np.full(df_.shape[0], np.nan)  # mean translational speed
    vxseg = np.full(df_.shape[0], np.nan)  # (dirx)
    vyseg = np.full(df_.shape[0], np.nan)  # (diry)
    aseg = np.full(df_.shape[0], np.nan)  # azimuth, geographic North

    # warmup last-segment (6h) variables
    pseg = np.full(df_.shape[0], np.nan)  # segment pressure
    wseg = np.full(df_.shape[0], np.nan)  # segment maxwinds
    rseg = np.full(df_.shape[0], np.nan)  # segment radii rmw
    lseg = np.full(df_.shape[0], np.nan)  # latitude (north hemisphere)
    laseg = np.full(df_.shape[0], np.nan)  # (absolute)

    # target segment (6h) variables
    lo1 = np.full(df_.shape[0], np.nan)  # target origin x-coordinate
    la1 = np.full(df_.shape[0], np.nan)  # idem y-coordinate
    dv = np.full(df_.shape[0], np.nan)  # translational speed variation
    dvx = np.full(df_.shape[0], np.nan)  # (dirx)
    dvy = np.full(df_.shape[0], np.nan)  # (diry)
    da = np.full(df_.shape[0], np.nan)  # azimuth variation
    dp = np.full(df_.shape[0], np.nan)  # pressure variation
    dw = np.full(df_.shape[0], np.nan)  # maxwinds variation
    dr = np.full(df_.shape[0], np.nan)  # radii rmw variation
    dl = np.full(df_.shape[0], np.nan)  # latitude variation
    dla = np.full(df_.shape[0], np.nan)  # (absolute)
    lo2 = np.full(df_.shape[0], np.nan)  # target endpoint x-coordinate
    la2 = np.full(df_.shape[0], np.nan)  # idem y-coordinate

    # loop
    for i in np.arange(1, dt.size):
        # get stopmotion endpoints coordinates (24h+6h)
        if i < 4:  # < four preceding segments
            # number of "missing" preceding segments to last 24h
            n_missing = 4 - i

            # last available preceding segment
            lon1, lon2 = lon[1], lon[0]
            lat1, lat2 = lat[1], lat[0]

            # distance of last available preceding segment
            arcl_h, gamma_h = geodesic_distance_azimuth(lat2, lon2, lat1, lon1)
            r = arcl_h * np.pi / 180.0 * EARTH_RADIUS  # distance [km]

            # shoot backwards to calculate (lo0,la0) of 24h preceding warmup
            dist = r * n_missing
            glon, glat, baz = shoot(lon2, lat2, gamma_h + 180, dist)

            # endpoint coordinates (-24h, 0h, 6h)
            lon_0, lon_i, lon_i1 = glon, lon[i], lon[i + 1]
            lat_0, lat_i, lat_i1 = glat, lat[i], lat[i + 1]

        if i >= 4:  # >= four preceding segments
            # endpoint coordinates (-24h, 0h, 6h)
            lon_0, lon_i, lon_i1 = lon[i - 4], lon[i], lon[i + 1]
            lat_0, lat_i, lat_i1 = lat[i - 4], lat[i], lat[i + 1]

        # segment endpoints
        lo0[i], lo1[i], lo2[i] = lon_0, lon_i, lon_i1
        la0[i], la1[i], la2[i] = lat_0, lat_i, lat_i1

        # warmup 4-segments (24h) variables
        _, vseg[i], vxseg[i], vyseg[i] = get_vmean(lat_0, lon_0, lat_i, lon_i, 24)
        aseg[i] = geodesic_azimuth(lat_0, lon_0, lat_i, lon_i)
        #        aseg[i] = calculate_azimut(lon_0, lat_0, lon_i, lat_i)

        # warmup last-segment (6h) variables
        pseg[i] = pres[i - 1]
        wseg[i] = wind[i - 1]
        rseg[i] = rmw[i - 1]
        lseg[i] = lat_i
        laseg[i] = np.abs(lat_i)

        # target segment (6h) variables
        _, v, vx, vy = get_vmean(lat_i, lon_i, lat_i1, lon_i1, dt[i : i + 1].sum())
        dv[i] = v - vseg[i]  # [km/h]
        dvx[i] = vx - vxseg[i]
        dvy[i] = vy - vyseg[i]
        dp[i] = pres[i] - pres[i - 1]  # [mbar]
        dw[i] = wind[i] - wind[i - 1]  # [kt]
        dr[i] = rmw[i] - rmw[i - 1]  # [nmile]
        dl[i] = lat_i1 - lat_i  # [º]
        dla[i] = np.abs(dl[i])

        # angle variation
        ang1 = aseg[i]
        ang2 = geodesic_azimuth(lat_i, lon_i, lat_i1, lon_i1)
        #        ang2 = calculate_azimut(lon_i, lat_i, lon_i1, lat_i1)
        dt_ang = ang2 - ang1  # [º]
        sign = np.sign(lseg[i])  # hemisphere: north (+), south (-)

        if (ang2 > ang1) & (dt_ang < 180):
            da[i] = sign * (dt_ang)
        elif (ang2 > ang1) & (dt_ang > 180):
            da[i] = sign * (dt_ang - 360)
        elif (ang2 < ang1) & (dt_ang > -180):
            da[i] = sign * (dt_ang)
        elif (ang2 < ang1) & (dt_ang < -180):
            da[i] = sign * (dt_ang + 360)

    # add to dataframe
    df_["vseg"] = vseg / 1.852  # [kt]
    #    df_['vxseg'] = vxseg / 1.852
    #    df_['vyseg'] = vyseg / 1.852
    df_["dvseg"] = dv / 1.852
    #    df_['dvxseg'] = dvx / 1.852
    #    df_['dvyseg'] = dvy / 1.852
    df_["pseg"] = pseg  # [mbar]
    df_["dpseg"] = dp
    df_["wseg"] = wseg  # [kt, 1-min avg]
    df_["dwseg"] = dw
    df_["rseg"] = rseg  # [nmile]
    df_["drseg"] = dr
    df_["aseg"] = aseg  # [º]
    df_["daseg"] = da
    df_["lseg"] = lseg  # [º]
    df_["laseg"] = laseg
    df_["dlaseg"] = dla
    df_["lon_w"] = lo0  # warmup origin
    df_["lat_w"] = la0
    df_["lon_i"] = lo1  # target origin
    df_["lat_i"] = la1
    df_["lon_t"] = lo2  # target endpoint
    df_["lat_t"] = la2

    return df_


def stopmotion_interpolation(
    df_seg: pd.DataFrame,
    st: pd.DataFrame = None,
    t_warm: int = 24,
    t_seg: int = 6,
    t_prop: int = 42,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Generate SWAN cases in cartesian coordinates from stopmotion parameterized segments.

    Parameters
    ----------
    df_seg : pd.DataFrame
        Stopmotion segments containing:
        - vseg : Mean translational speed (kt)
        - pseg : Segment pressure (mbar)
        - wseg : Maximum winds (kt)
        - rseg : RMW (nmile)
        - laseg : Absolute latitude (degrees)
        - dvseg : Speed variation (kt)
        - dpseg : Pressure variation (mbar)
        - dwseg : Wind variation (kt)
        - drseg : RMW variation (nmile)
        - daseg : Azimuth variation (degrees)

    st : pd.DataFrame, optional
        Real storm track data. If None, MDA segments are used (unrelated to historic tracks).
    t_warm : int, optional
        Warmup period in hours. Default is 24.
    t_seg : int, optional
        Target period in hours. Default is 6.
    t_prop : int, optional
        Propagation period in hours. Default is 42.

    Returns
    -------
    Tuple[List[pd.DataFrame], List[pd.DataFrame]]
        Two lists containing:

        1. List of storm track DataFrames with columns:
           - x, y : Cartesian coordinates (m)
           - lon, lat : Geographic coordinates (degrees)
           - vf : Translation speed (kt)
           - vfx, vfy : Velocity components (kt)
           - pn : Surface pressure (1013 mbar)
           - p0 : Minimum central pressure (mbar)
           - vmax : Maximum winds (kt)
           - rmw : RMW (nmile)
           - latitude : Latitude with hemisphere sign

        2. List of empty wave event DataFrames with columns:
           - hs, t02, dir, spr : Wave parameters
           - U10, V10 : Wind components
           - level, tide : Water level parameters

    Notes
    -----
    The function generates SWAN cases in cartesian coordinates following SHyTCWaves configuration:

    A. Warmup period (24h):
       Over the negative x-axis ending at (x,y)=(0,0)

    B. Target period (6h):
       Starting at (x,y)=(0,0)

    C. Propagation period (42h):
       No track coordinates (no wind forcing)
    """

    # sign hemisphere (+north, -south)
    if isinstance(st, pd.DataFrame):  # historic track
        sign = np.sign(st["lat"][0])
        method = st.attrs["method"]
        center = st.attrs["center"]
    else:  # mda cases
        sign = 1
        method = "mda"
        center = "mda"

    # remove NaNs
    df = df_seg.dropna()

    # number of stopmotion events
    N = df.shape[0]

    # list of SWAN cases (paramterized events)
    st_list, we_list = [], []
    for i in range(N):
        seg_i = df.iloc[i]

        # stopmotion unit parameters
        vseg = seg_i["vseg"] * 1.852  # [km/h]
        dvseg = seg_i["dvseg"] * 1.852
        pseg = seg_i["pseg"]  # [mbar]
        dpseg = seg_i["dpseg"]
        wseg = seg_i["wseg"]  # [kt, 1-min avg]
        dwseg = seg_i["dwseg"]
        rseg = seg_i["rseg"]  # [nmile]
        drseg = seg_i["drseg"]
        daseg = seg_i["daseg"]  # [º]
        laseg = seg_i["laseg"]  # [º]

        # vmean criteria for SWAN computational timestep [minutes]
        seg_vmean = vseg + dvseg
        if (vseg > 20) or (seg_vmean > 20):
            dt_comp = 10
        else:
            dt_comp = 20

        # time array for SWAN input
        ts = t_warm + t_seg + t_prop  # [h] simulation duration
        ts = np.asarray(ts) * 60 / dt_comp  # [] intervals of computation

        ts_warmup = int(t_warm * 60 / dt_comp)
        ts_segment = int(t_seg * 60 / dt_comp)

        # random initial date
        date_ini = pd.Timestamp(1999, 12, 31, 0)
        time_input = pd.date_range(
            date_ini, periods=int(ts), freq="{0}MIN".format(dt_comp)
        )
        time_input = np.array(time_input)

        # vortex input variables
        x = np.full(int(ts), np.nan)  # [m]
        y = np.full(int(ts), np.nan)  # [m]
        vmean = np.full(int(ts), np.nan)  # [km/h]
        ut = np.full(int(ts), np.nan)
        vt = np.full(int(ts), np.nan)
        p0 = np.full(int(ts), np.nan)  # [mbar]
        vmax = np.full(int(ts), np.nan)  # [kt, 1-min avg]
        rmw = np.full(int(ts), np.nan)  # [nmile]
        lat = np.full(int(ts), np.nan)  # [º]

        # (A) preceding 24h segment: over negative x-axis ending at (x,y)=(0,0)

        for j in np.arange(0, ts_warmup):
            if j == 0:
                x[j] = -vseg * 24 * 10**3
            else:
                x[j] = x[j - 1] + vseg * (dt_comp / 60) * 10**3
            y[j] = 0
            vmean[j] = vseg
            ut[j] = vseg
            vt[j] = 0
            p0[j] = pseg
            vmax[j] = wseg
            rmw[j] = rseg
            lat[j] = laseg

        # (B) target 6h segment: starting at (x,y)=(0,0)

        for j in np.arange(ts_warmup, ts_warmup + ts_segment):
            vel = vseg + dvseg  # [km/h]
            velx = vel * np.sin((daseg * sign + 90) * np.pi / 180)
            vely = vel * np.cos((daseg * sign + 90) * np.pi / 180)

            x[j] = x[j - 1] + velx * (dt_comp / 60) * 10**3
            y[j] = y[j - 1] + vely * (dt_comp / 60) * 10**3
            vmean[j] = vel
            ut[j] = velx
            vt[j] = vely
            p0[j] = pseg + dpseg
            vmax[j] = wseg + dwseg
            rmw[j] = rseg + drseg
            lat[j] = laseg

        # (C) propagation 42h segment: remaining values of data arrays

        # store dataframe
        st_seg = pd.DataFrame(
            index=time_input,
            columns=["x", "y", "vf", "vfx", "vfy", "pn", "p0", "vmax", "rmw", "lat"],
        )

        st_seg["x"] = x  # [m]
        st_seg["y"] = y
        st_seg["lon"] = x  # (idem for plots)
        st_seg["lat"] = y
        st_seg["vf"] = vmean / 1.852  # [kt]
        st_seg["vfx"] = ut / 1.852
        st_seg["vfy"] = vt / 1.852
        st_seg["pn"] = 1013  # [mbar]
        st_seg["p0"] = p0
        st_seg["vmax"] = vmax  # [kt]
        st_seg["rmw"] = rmw  # [nmile]
        st_seg["latitude"] = lat * sign  # [º]

        # add metadata
        st_seg.attrs = {
            "method": method,
            "center": center,
            "override_dtcomp": "{0} MIN".format(dt_comp),
            "x0": 0,
            "y0": 0,
            "p0": "mbar",
            "vf": "kt",
            "vmax": "kt, 1-min avg",
            "rmw": "nmile",
        }

        # append to stopmotion event list
        st_list.append(st_seg)

        # generate wave event (empty)
        we = pd.DataFrame(
            index=time_input, columns=["hs", "t02", "dir", "spr", "U10", "V10"]
        )
        we["level"] = 0
        we["tide"] = 0
        we_list.append(we)

    return st_list, we_list


###############################################################################
# STOPMOTION ensemble
# functions that collect 6h-segments from library cases to obtain the hybrid
# storm track (analogue or closest to the real track); those segments are
# rotated and assigned time-geographical coordinates. The ensemble and the
# envelope is calculate at each control point
###############################################################################


def find_analogue(
    df_library: pd.DataFrame, df_case: pd.DataFrame, ix_weights: List[float]
) -> np.ndarray:
    """
    Find the minimum distance in a 10-dimensional normalized space.

    Parameters
    ----------
    df_library : pd.DataFrame
        Library parameters DataFrame containing the 10 SHyTCWaves parameters:
        {pseg, vseg, wseg, rseg, dp, dv, dw, dr, dA, lat}
    df_case : pd.DataFrame
        Target case parameters DataFrame with same structure as df_library.
    ix_weights : List[float]
        Weight factors for each parameter dimension.

    Returns
    -------
    np.ndarray
        Indices of nearest points in the library for each case point.

    Notes
    -----
    The function finds the minimum distance in a normalized space corresponding
    to the SHyTCWaves 10 parameters:
    - pseg : Segment pressure
    - vseg : Mean translational speed
    - wseg : Maximum winds
    - rseg : RMW
    - dp : Pressure variation
    - dv : Speed variation
    - dw : Wind variation
    - dr : RMW variation
    - dA : Azimuth variation
    - lat : Latitude
    """

    # remove NaNs from storm segments
    df_case = df_case.dropna()
    data_case = df_case[
        [
            "daseg",
            "dpseg",
            "pseg",
            "dwseg",
            "wseg",
            "dvseg",
            "vseg",
            "drseg",
            "rseg",
            "laseg",
        ]
    ].values

    # library segments parameter
    data_lib = df_library[
        [
            "daseg",
            "dpseg",
            "pseg",
            "dwseg",
            "wseg",
            "dvseg",
            "vseg",
            "drseg",
            "rseg",
            "laseg",
        ]
    ].values
    ix_directional = [0]

    # get indices of nearest n-dimensional point
    ix_near = find_nearest_indices(
        query_points=data_case,
        reference_points=data_lib,
        directional_indices=ix_directional,
        weights=ix_weights,
    )

    return ix_near


def analogue_endpoints(df_seg: pd.DataFrame, df_analogue: pd.DataFrame) -> pd.DataFrame:
    """
    Add segment endpoint coordinates by looking up the real target segment.

    Parameters
    ----------
    df_seg : pd.DataFrame
        Parameterized historical storm track DataFrame containing:
        - lon, lat : Target origin coordinates
        - aseg : Warmup azimuth
        - daseg : Target azimuth variation
    df_analogue : pd.DataFrame
        Analogue segments from library containing:
        - vseg : Mean translational speed (kt)
        - dvseg : Speed variation (kt)
        - daseg : Target azimuth variation

    Returns
    -------
    pd.DataFrame
        Updated analogue segments DataFrame with added columns:
        - aseg : Warmup azimuth
        - lon_w, lat_w : Warmup origin coordinates
        - lon_i, lat_i : Target origin coordinates (from df_seg)
        - lon_t, lat_t : Target endpoint coordinates

    Notes
    -----
    The function:
    1. Calculates warmup origin by shooting backwards 24h from target origin
    2. Uses target origin from historical track
    3. Calculates target endpoint by shooting forwards 6h from target origin
    4. Accounts for hemisphere when calculating angles
    5. Converts longitudes to [0-360°] convention
    """

    # remove NaNs
    df_seg = df_seg[~df_seg.isna().any(axis=1)]
    # df_analogue = df_analogue[~df_analogue.isna().any(axis=1)]

    # get hemisphere
    # relative angles are multiplied by "sign" to account for hemisphere
    sign = np.sign(df_seg.lat.values[0])

    # get historic variables
    lon0 = df_seg.lon.values  # target origin coords
    lat0 = df_seg.lat.values
    aseg = df_seg.aseg.values  # warmup azimuth
    daseg = df_seg.daseg.values * sign  # target azimuth variation

    # get analogue variables
    vseg1_an = df_analogue.vseg.values  # warmup velocity [kt]
    dvseg_an = df_analogue.dvseg.values
    vseg2_an = vseg1_an + dvseg_an  # target velocity
    daseg_an = df_analogue.daseg.values * sign  # target azimuth variation

    # new variables
    az1_an = np.full(lon0.shape, np.nan)
    glon1_an = np.full(lon0.shape, np.nan)
    glon2_an = np.full(lon0.shape, np.nan)
    glat1_an = np.full(lon0.shape, np.nan)
    glat2_an = np.full(lon0.shape, np.nan)

    for i in range(df_seg.shape[0]):
        # azimuth angles for ensemble
        az2 = aseg[i] + daseg[i]  # target azimuth (historic-fixed)
        az1 = az2 - daseg_an[i]  # warmup azimuth (stopmotion analogue)

        # shoot backwards to warmup origin
        dist1 = vseg1_an[i] * 1.852 * 24  # [km]
        glon1, glat1, baz = shoot(lon0[i], lat0[i], az1 + 180, dist1)

        # shoot forwards to target endpoint
        dist2 = vseg2_an[i] * 1.852 * 6  # [km]
        glon2, glat2, baz = shoot(lon0[i], lat0[i], az2 + 180 - 180, dist2)

        # store
        az1_an[i] = az1
        glon1_an[i] = glon1
        glon2_an[i] = glon2
        glat1_an[i] = glat1
        glat2_an[i] = glat2

    # longitude convention
    glon1_an[glon1_an < 0] += 360
    glon2_an[glon2_an < 0] += 360

    # add to dataframe
    df_analogue["aseg"] = az1_an
    df_analogue["lon_w"] = glon1_an  # warmup origin
    df_analogue["lat_w"] = glat1_an
    df_analogue["lon_i"] = df_seg.lon.values
    df_analogue["lat_i"] = df_seg.lat.values
    df_analogue["lon_t"] = glon2_an
    df_analogue["lat_t"] = glat2_an

    return df_analogue


def stopmotion_st_bmu(
    df_analogue: pd.DataFrame,
    df_seg: pd.DataFrame,
    st: pd.DataFrame,
    cp_lon_ls: List[float],
    cp_lat_ls: List[float],
    max_dist: float = 60,
    list_out: bool = False,
    tqdm_out: bool = False,
    text_out: bool = True,
    mode: str = "",
) -> xr.Dataset:
    """
    Extract bulk wave parameters from library analogue cases.

    Parameters
    ----------
    df_analogue : pd.DataFrame
        Analogue prerun segments from library.
    df_seg : pd.DataFrame
        Storm 6h-segments parameters.
    st : pd.DataFrame
        Storm track interpolated every 6h.
    cp_lon_ls : List[float]
        Control point longitude coordinates.
    cp_lat_ls : List[float]
        Control point latitude coordinates.
    max_dist : float, optional
        Maximum distance (km) to extract closest node. Default is 60.
    list_out : bool, optional
        Whether to return list of datasets instead of merged dataset. Default is False.
    tqdm_out : bool, optional
        Whether to show progress bar. Default is False.
    text_out : bool, optional
        Whether to show text output. Default is True.
    mode : str, optional
        High or low resolution library indices. Default is "".

    Returns
    -------
    xr.Dataset
        Wave directional spectra with dimensions:
        - case : Storm segments
        - point : Control points
        - time : Time steps

        Contains variables:
        - hs : Significant wave height
        - tp : Peak period
        - lon, lat : Control point coordinates
        - ix_near : Nearest point indices
        - pos_nonan : Valid point mask
        - bmu : Best matching unit indices
        - hsbmu : Maximum Hs per point and time
        - tpbmu : Tp at maximum Hs
        - hswath : Maximum Hs per point
        - tswath : Tp at maximum Hs per point

    Notes
    -----
    The function:
    1. Accesses library analogue cases for a given storm track
    2. Calculates distance and angle from target segment origin to control points
    3. Extracts wave parameters at closest nodes for each analogue segment
    4. Computes bulk parameter envelope and swath
    5. Handles both high and low resolution libraries
    """

    # remove NaN
    df_seg = df_seg[~df_seg.isna().any(axis=1)]
    df_analogue = df_analogue[~df_analogue.isna().any(axis=1)]

    # assign time
    df_seg["time"] = df_seg.index.values

    # get hemisphere
    sign = np.sign(df_seg.lat.values[0])

    xds_list = []

    if tqdm_out:
        array = tqdm(range(df_seg.shape[0]))
    else:
        array = range(df_seg.shape[0])

    for iseg in array:  # each segment
        # get storm segment 'i'
        df_icase = df_seg.iloc[iseg]
        df_ianalogue = df_analogue.iloc[iseg]
        iseg_analogue = df_ianalogue.name  # analogue id
        aseg = df_ianalogue.aseg  # analogue warmseg azimuth (N)

        # ---------------------------------------------------------------------
        # get storm coordinates at "seg_time"
        seg_time = np.datetime64(df_icase.time)  # timestamp to datetime64
        ind_time_st = np.where(st.index.values == seg_time)[0][0]

        # storm coordinates (real storm eye)
        st_lon = st.iloc[ind_time_st].lon  # target origin longitude
        st_lat = st.iloc[ind_time_st].lat  # target origin latitude
        st_time = st.index.values[ind_time_st]  # time coordinates

        # ---------------------------------------------------------------------
        # get cp coordinates in relative system (radii, angle)
        cp_dist_ls, cp_ang_ls = get_cp_radii_angle(
            st_lat, st_lon, cp_lat_ls, cp_lon_ls, sign, aseg
        )

        # get SWAN output mask indices (rad, ang)
        mask_rad, mask_ang = get_mask_radii_angle(iseg_analogue, mode=mode)

        # find closest index
        ix_near = find_nearest(cp_dist_ls, cp_ang_ls, mask_rad, mask_ang)
        pos_nonan = np.abs(mask_rad[ix_near] - cp_dist_ls) <= max_dist

        # ---------------------------------------------------------------------
        # load library Hs reconstructed
        xds_rec = xr.open_dataset(PATHS["SHYTCWAVES_BULK"])
        xds_rec["time"] = pd.date_range("2000-01-01", periods=48, freq="1H")

        # extract HS,TP at closest points (case,point,time)
        hs_arr = np.full((ix_near.size, xds_rec.time.values.size), np.nan)
        hs_arr[pos_nonan, :] = (
            xds_rec.hs.isel(case=iseg_analogue, point=ix_near[pos_nonan]).load().values
        )

        tp_arr = np.full((ix_near.size, xds_rec.time.values.size), np.nan)
        tp_arr[pos_nonan, :] = (
            xds_rec.tp.isel(case=iseg_analogue, point=ix_near[pos_nonan]).load().values
        )

        # time array
        hour_intervals = xds_rec.time.size
        time = [st_time + np.timedelta64(1, "h") * i for i in range(hour_intervals)]
        time_array = np.array(time)

        # store dataset
        xds = xr.Dataset(
            {
                "hs": (("case", "point", "time"), np.expand_dims(hs_arr, axis=0)),
                "tp": (("case", "point", "time"), np.expand_dims(tp_arr, axis=0)),
                "lon": (("point"), cp_lon_ls),
                "lat": (("point"), cp_lat_ls),
                "ix_near": (("case", "point"), np.expand_dims(ix_near, axis=0)),
                "pos_nonan": (("case", "point"), np.expand_dims(pos_nonan, axis=0)),
            },
            coords={
                "case": [iseg],
                "time": time_array,
            },
        )
        xds_list.append(xds)

    if list_out:
        xds_out = xds_list
    else:
        # merge
        xds_out = xr.merge(xds_list)
        if text_out:
            print("Merging bulk envelope...", datetime.datetime.now())

        # add envelope variables
        xds_bmu = xds_out.copy()
        hsval = xds_bmu.hs.values[:]
        hsval[np.isnan(hsval)] = 0  # remove nan
        bmu = np.argmax(hsval, axis=0).astype(float)  # bmu indices
        hsmax = np.sort(hsval, axis=0)[-1, :, :]  # max over 'case'

        # bmu, hs
        bmu[hsmax == 0] = np.nan  # restitute nans
        hsmax[hsmax == 0] = np.nan

        xds_out["bmu"] = (("point", "time"), bmu)
        xds_out["hsbmu"] = (("point", "time"), hsmax)

        # tp
        tpmax = np.full(xds_out.hsbmu.shape, np.nan)
        nanmask = ~np.isnan(bmu)
        mesht, meshp = np.meshgrid(
            np.arange(0, xds_out.time.size), np.arange(0, xds_out.point.size)
        )

        tpmax[nanmask] = xds_out.tp.values[
            bmu.ravel()[nanmask.ravel()].astype("int64"),
            meshp.ravel()[nanmask.ravel()],
            mesht.ravel()[nanmask.ravel()],
        ]
        xds_out["tpbmu"] = (("point", "time"), tpmax)

        # add swath variables
        hh = hsmax.copy()
        hh[np.isnan(hh)] = 0
        posw = np.argmax(hh, axis=1)

        xds_out["hswath"] = (("point"), hsmax[np.arange(0, xds_out.point.size), posw])
        xds_out["tswath"] = (("point"), tpmax[np.arange(0, xds_out.point.size), posw])

    return xds_out


def stopmotion_st_spectra(
    df_analogue: pd.DataFrame,
    df_seg: pd.DataFrame,
    st: pd.DataFrame,
    cp_lon_ls: List[float],
    cp_lat_ls: List[float],
    cp_names: List[str] = [],
    max_dist: float = 60,
    list_out: bool = False,
    tqdm_out: bool = False,
    text_out: bool = True,
    mode: str = "",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Function to access the library analogue cases for a given storm track,
    calculate distance and angle from the target segment origin to the control
    point (relative coordinate system), and extract the directional wave
    spectra at the closest node (for every analogue segment)

    Parameters
    ----------
    df_analogue : pd.DataFrame
        Analogue prerun segments from library.
    df_seg : pd.DataFrame
        Storm 6h-segments parameters.
    st : pd.DataFrame
        Storm track interpolated every 6h.
    cp_lon_ls : List[float]
        Control point geographical coordinates.
    cp_lat_ls : List[float]
        Control point geographical coordinates.
    cp_names : List[str], optional
        Control point names. Default is [].
    max_dist : float, optional
        Maximum distance [km] to extract closest node. Default is 60.
    list_out : bool, optional
        Whether to list output. Default is False.
    tqdm_out : bool, optional
        Whether to use tqdm. Default is False.
    text_out : bool, optional
        Whether to print text. Default is True.
    mode : str, optional
        Mode. Default is "".

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        - xds_spec : Wave directional spectra (dim 'case')
        - xds_bmu : BMU indices

    Notes
    -----
    The function:
    1. Removes NaN values from df_seg and df_analogue
    2. Assigns time to df_seg
    3. Gets hemisphere sign
    4. Gets bmu (wavespectra reconstructed)
    5. Opens seg_sim dataset
    """

    # remove NaN
    df_seg = df_seg[~df_seg.isna().any(axis=1)]
    df_analogue = df_analogue[~df_analogue.isna().any(axis=1)]

    # assign time
    df_seg["time"] = df_seg.index.values

    # get hemisphere
    sign = np.sign(df_seg.lat.values[0])

    # get bmu (wavespectra reconstructed)
    # it provides 'time,bmu,ix_near,pos_nonan' (point,time)
    xds_bmu = stopmotion_st_bmu(
        df_analogue=df_analogue,
        df_seg=df_seg,
        st=st,
        cp_lon_ls=cp_lon_ls,
        cp_lat_ls=cp_lat_ls,
        max_dist=max_dist,
        list_out=list_out,
        tqdm_out=tqdm_out,
        text_out=text_out,
        mode=mode,
    )

    # spectral energy
    seg_sim = xr.open_dataset(
        op.join(PATHS["SHYTCWAVES_SPECTRA"], "0000/spec_outpts_main.nc")
    )
    efth_arr = np.full(
        (
            seg_sim.frequency.size,
            seg_sim.direction.size,
            xds_bmu.point.size,
            xds_bmu.time.size,
        ),
        np.nan,
    )  # 38,72,cp,t

    if tqdm_out:
        array = tqdm(range(xds_bmu.case.size))
    else:
        array = range(xds_bmu.case.size)
    for iseg in array:
        # get storm segment 'i'
        df_icase = df_seg.iloc[iseg]
        df_ianalogue = df_analogue.iloc[iseg]
        iseg_analogue = df_ianalogue.name  # analogue id

        # ---------------------------------------------------------------------
        # get storm coordinates at "seg_time"
        seg_time = np.datetime64(df_icase.time)  # timestamp to datetime64
        st_time = st.index.values[st.index.values == seg_time][0]

        # ---------------------------------------------------------------------
        # get analogue segment from library
        filename = "spec_outpts_main.nc"
        p_analogue = op.join(
            PATHS["SHYTCWAVES_SPECTRA"], f"{iseg_analogue:04d}", filename
        )
        # load file
        seg_sim = xr.open_dataset(p_analogue)  # freq,dir,2088,48

        # time array
        hour_intervals = seg_sim.time.size
        time = [st_time + np.timedelta64(1, "h") * i for i in range(hour_intervals)]
        time_array = np.array(time)

        # get intersect time iseg vs xds_bmu
        _, ix_time_st, ix_time_shy = np.intersect1d(
            time_array, xds_bmu.time.values, return_indices=True
        )

        # find all closest grid points
        shy_inear = xds_bmu.ix_near.values[iseg, :].astype("int64")  # case,point

        # find bmu indices for iseg
        in_pt, in_t = np.where(xds_bmu.bmu.values == iseg)

        # get indices of casei
        in_t_ = in_t - ix_time_shy[0]

        # reorder spectral directions
        base = 5
        if mode == "_lowres":
            base = 10  # depends on the library dirs delta
        efth_case = seg_sim.isel(point=shy_inear)  # .isel(point=in_pt, time=in_t_)
        if sign < 0:
            efth_case["direction"] = 360 - seg_sim.direction.values
            new_dirs = np.round(
                efth_case.direction.values + base * round(df_icase.aseg / base) + 90, 1
            )
        else:
            new_dirs = np.round(
                efth_case.direction.values + base * round(df_icase.aseg / base) - 90, 1
            )
        new_dirs = np.mod(new_dirs, 360)
        new_dirs[new_dirs > 270] -= 360
        efth_case["direction"] = new_dirs
        efth_case = efth_case.sel(direction=seg_sim.direction.values)

        # insert spectral values for bmu=iseg
        efth_arr[:, :, in_pt, in_t] = efth_case.efth.values[:, :, in_pt, in_t_]

    if text_out:
        print("Inserting envelope spectra...", datetime.datetime.now())

    # store dataset
    xds_spec = xr.Dataset(
        {
            "efth": (("freq", "dir", "point", "time"), efth_arr),
            "lon": (("point"), np.array(cp_lon_ls)),
            "lat": (("point"), np.array(cp_lat_ls)),
            "station": (("point"), np.array(cp_names)),
        },
        coords={
            "freq": seg_sim.frequency.values,
            "dir": seg_sim.direction.values,
            "point": xds_bmu.point.values,
            "time": xds_bmu.time.values,
        },
    )

    return xds_spec, xds_bmu


def get_cp_radii_angle(
    st_lat: float,
    st_lon: float,
    cp_lat_ls: List[float],
    cp_lon_ls: List[float],
    sign: int,
    aseg: float,
) -> Tuple[List[float], List[float]]:
    """
    Extract control point distances and angles in the relative coordinate system.

    Parameters
    ----------
    st_lat : float
        Storm center latitude.
    st_lon : float
        Storm center longitude.
    cp_lat_ls : List[float]
        Control point latitudes.
    cp_lon_ls : List[float]
        Control point longitudes.
    sign : int
        Hemisphere indicator: 1 for north, -1 for south.
    aseg : float
        Azimuth of the analogue warm segment (from geographic north).

    Returns
    -------
    Tuple[List[float], List[float]]
        Two lists containing:
        - cp_dist_ls : Distances from storm center to control points (km)
        - cp_ang_ls : Angles from storm center to control points (degrees)

    Notes
    -----
    The function:
    1. Calculates great circle distances between storm center and control points
    2. Converts distances from degrees to kilometers
    3. Calculates angles relative to geographic north
    4. Transforms angles to relative coordinate system
    5. Adjusts angles for southern hemisphere
    """

    cp_dist_ls, cp_ang_ls = [], []
    for i in range(len(cp_lat_ls)):
        cp_lat, cp_lon = cp_lat_ls[i], cp_lon_ls[i]

        # get point polar reference
        # azimut is refered to geographical north (absolute system)
        arcl_h, ang_abs = geodesic_distance_azimuth(st_lat, st_lon, cp_lat, cp_lon)
        cp_dist_ls.append(arcl_h * np.pi / 180.0 * EARTH_RADIUS)  # [km]

        # change of coordinate system (absolute to relative)
        ang_rel = ang_abs - (aseg - 90)
        if ang_rel < 0:
            ang_rel = np.mod(ang_rel, 360)

        # south hemisphere effect
        if sign == -1:
            if (ang_rel >= 0) and (ang_rel <= 180):
                ang_rel = 180 - ang_rel
            elif (ang_rel >= 180) and (ang_rel <= 360):
                ang_rel = 360 - (ang_rel - 180)

        cp_ang_ls.append(ang_rel)

    return cp_dist_ls, cp_ang_ls  # [km], [º]


def get_mask_radii_angle(icase: int, mode: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract radii and angle indices for output points.

    Parameters
    ----------
    icase : int
        Analogue case ID.
    mode : str, optional
        Option to select SHyTCWaves library resolution ('', '_lowres'). Default is "".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays containing:
        - rad : Radial distances from storm center (km)
        - ang : Angles from storm center (degrees)

    Notes
    -----
    The function:
    1. Loads output indices associated with distances/angles to target origin
    2. Determines grid size (small/medium/large) based on case ID
    3. Extracts appropriate radii and angle arrays for the grid size
    4. For low resolution mode, uses half the number of radial points
    """

    # load output indices associated with distances/angles to target origin
    if mode == "_lowres":
        xds_mask_ind = xr.open_dataset(PATHS["SHYTCWAVES_MDA_MASK_INDICES_LOWRES"])
    else:
        xds_mask_ind = xr.open_dataset(PATHS["SHYTCWAVES_MDA_MASK_INDICES"])

    # load MDA indices (grid sizes)
    xds_ind_mda = xr.open_dataset(PATHS["SHYTCWAVES_MDA_INDICES"])

    # get grid code
    pos_small = np.where(icase == xds_ind_mda.indices_small)[0]
    pos_medium = np.where(icase == xds_ind_mda.indices_medium)[0]
    pos_large = np.where(icase == xds_ind_mda.indices_large)[0]

    if len(pos_small) == 1:
        rad = xds_mask_ind.radii_sma.values[:] / 1000
        ang = xds_mask_ind.angle_sma.values[:]

    elif len(pos_medium) == 1:
        rad = xds_mask_ind.radii_med.values[:] / 1000
        ang = xds_mask_ind.angle_med.values[:]

    elif len(pos_large) == 1:
        rad = xds_mask_ind.radii_lar.values[:] / 1000
        ang = xds_mask_ind.angle_lar.values[:]

    return rad, ang  # [km], [º]


def find_nearest(
    cp_rad_ls: List[float],
    cp_ang_ls: List[float],
    mask_rad: np.ndarray,
    mask_ang: np.ndarray,
) -> np.ndarray:
    """
    Find nearest points in normalized space of radii and angles.

    Parameters
    ----------
    cp_rad_ls : List[float]
        Control point radial distances.
    cp_ang_ls : List[float]
        Control point angles.
    mask_rad : np.ndarray
        SWAN output point radial distances.
    mask_ang : np.ndarray
        SWAN output point angles.

    Returns
    -------
    np.ndarray
        Indices of nearest points in SWAN output grid for each control point.

    Notes
    -----
    The function:
    1. Creates dataframes from control points and SWAN grid points
    2. Treats radial distances as scalar values
    3. Treats angles as directional values (circular)
    4. Uses nearest neighbor search in normalized space
    """

    # create dataframes
    df_cp = pd.DataFrame({"radii": cp_rad_ls, "angle": cp_ang_ls})

    df_mask = pd.DataFrame({"radii": mask_rad, "angle": mask_ang})

    # indices
    ix_directional = [1]

    # get indices of nearest n-dimensional point
    ix_near = find_nearest_indices(
        query_points=df_cp.values,
        reference_points=df_mask.values,
        directional_indices=ix_directional,
    )

    return ix_near


###############################################################################
# SHyTCWaves APPLICATION

# Functions that calculate the ensemble/reconstruction for a storm track
# from either historical or forecast/predicted tracks
###############################################################################


def get_coef_calibration() -> np.ndarray:
    """
    Get calibration coefficients for SHyTCWaves model pressure bias correction.

    Returns
    -------
    np.ndarray
        Linear fit coefficients for pressure bias correction.

    Notes
    -----
    The function:
    1. Uses Saffir-Simpson category center pressures as reference points
    2. Applies calibrated pressure deltas based on validation against satellite data
    3. Performs linear fit to get correction coefficients
    """

    p = [1015, 990, 972, 954, 932, 880]  # Saffir-Simpson center categories
    dp = [-17, -15, -12.5, -7, +2.5, +10]  # calibrated "dP" for shytcwaves

    coef = np.polyfit(p, dp, 1)  # order 1 fitting

    return coef


##########################################
# SHYTCWAVES - historical track


def historic2shytcwaves_cluster(
    path_save: str,
    tc_name: str,
    storm: xr.Dataset,
    center: str,
    lon: np.ndarray,
    lat: np.ndarray,
    dict_site: Optional[dict] = None,
    calibration: bool = True,
    mode: str = "",
    database_on: bool = False,
    st_param: bool = False,
    extract_bulk: bool = True,
    max_segments: int = 300,
) -> None:
    """
    Process historical storm track data using SHyTCWaves methodology.

    Parameters
    ----------
    path_save : str
        Base path to store results, without the file name.
    tc_name : str
        Storm name.
    storm : xr.Dataset
        Storm track dataset with standard IBTrACS variables:

        Required:
        - longitude, latitude : Storm coordinates
        - pressure : Central pressure (mbar)
        - maxwinds : Maximum sustained winds (kt)

        Optional:
        - rmw : Radius of maximum winds (nmile)
        - dist2land : Distance to land
        - basin : Basin identifier

    center : str
        IBTrACS center code.
    lon : np.ndarray
        Longitude coordinates for swath calculation.
    lat : np.ndarray
        Latitude coordinates for swath calculation.
    dict_site : dict, optional
        Site data for superpoint building. Default is None.
        Must contain:
        - lonpts : Longitude coordinates
        - latpts : Latitude coordinates
        - namepts : Site names
        - site : Site identifier
        - sectors : Sectors
        - deg_superposition : Superposition degree

    calibration : bool, optional
        Whether to apply SHyTCWaves calibration. Default is True.
    mode : str, optional
        High or low resolution library indices. Default is "".
    database_on : bool, optional
        Whether to keep data only at 0,6,12,18 hours. Default is False.
    st_param : bool, optional
        Whether to keep data as original. Default is False.
    extract_bulk : bool, optional
        Whether to extract bulk wave parameters. Default is True.
    max_segments : int, optional
        Maximum number of segments to process. Default is 300.

    Notes
    -----
    The function processes historical storm tracks in several steps:
    1. Performs stopmotion segmentation at 6h intervals
    2. Optionally applies SHyTCWaves calibration to track parameters
    3. Trims track to target domain
    4. Finds analogue segments from library
    5. Extracts bulk wave parameters and/or spectral data
    6. Saves results to specified directory
    """

    # stopmotion segmentation, 6h interval
    df = historic_track_preprocessing(
        xds=storm,
        center=center,
        forecast_on=False,
        database_on=database_on,
        st_param=st_param,
    )
    dt_int_minutes = 6 * 60  # [minutes] constant segments

    # optional: shytcwaves calibration of track parameters
    if calibration:
        coef = get_coef_calibration()  # linear fitting
        df["pressure"] = df["pressure"].values * (1 + coef[0]) + coef[1]
        df["maxwinds"] = np.nan

        st, _ = historic_track_interpolation(
            df,
            dt_int_minutes,
            interpolation=False,
            mode="mean",
            fit=True,
            radi_estimate_on=True,
        )
    else:
        st, _ = historic_track_interpolation(
            df, dt_int_minutes, interpolation=False, mode="mean"
        )

    # skip when only NaN or 0
    lons, lats = st.lon.values, st.lat.values
    if (np.unique(lons[~np.isnan(lons)]).all() == 0) & (
        np.unique(lats[~np.isnan(lats)]).all() == 0
    ):
        print("No track coordinates")

    else:
        st_trim = track_triming(st, lat[0], lon[0], lat[-1], lon[-1])

        # store tracks for shytcwaves
        st.to_pickle(op.join(path_save, f"{tc_name}_track.pkl"))
        st_trim.to_pickle(op.join(path_save, f"{tc_name}_track_trim.pkl"))

        # parameterized segts (24h warmup + 6htarget)
        df_seg = storm2stopmotion(st_trim)

        if df_seg.shape[0] > 2:
            print(f"st: {st.shape[0]}, df_seg: {df_seg.shape[0]}")
            st_list, we_list = stopmotion_interpolation(df_seg, st=st_trim)

            # analogue segments from library
            df_mda = xr.open_dataset(PATHS["SHYTCWAVES_MDA"]).to_dataframe()
            ix_weights = [1] * 10  # equal weights
            ix = find_analogue(df_mda, df_seg, ix_weights)

            df_analogue = df_mda.iloc[ix]
            df_analogue = analogue_endpoints(df_seg, df_analogue)

            # extract bulk envelope (to plot swaths)
            if extract_bulk:
                mesh_lo, mesh_la = np.meshgrid(lon, lat)
                print(
                    f"Number of segments: {len(st_list)}, number of swath nodes: {mesh_lo.size}"
                )

                if len(st_list) < max_segments:
                    xds_shy_bulk = stopmotion_st_bmu(
                        df_analogue,
                        df_seg,
                        st_trim,
                        list(np.ravel(mesh_lo)),
                        list(np.ravel(mesh_la)),
                        max_dist=60,
                        mode=mode,
                    )
                    # store
                    xds_shy_bulk.to_netcdf(
                        op.join(path_save, f"{tc_name}_xds_shy_bulk.nc")
                    )

            # extract spectra envelope
            if isinstance(dict_site, dict):
                xds_shy_spec, _ = stopmotion_st_spectra(
                    df_analogue,
                    df_seg,
                    st_trim,
                    cp_lon_ls=dict_site["lonpts"],
                    cp_lat_ls=dict_site["latpts"],
                    cp_names=dict_site["namepts"],
                    mode=mode,
                )
                # store
                xds_shy_spec.to_netcdf(
                    op.join(
                        path_save,
                        f"{tc_name}_xds_shy_spec_{dict_site['site']}.nc",
                    )
                )

                # build superpoint
                xds_shy_sp = superpoint_calculation(
                    xds_shy_spec.efth,
                    "point",
                    dict_site["sectors"],
                )
                # store
                xds_shy_sp.to_netcdf(
                    op.join(
                        path_save,
                        f"{tc_name}_xds_shy_sp_{dict_site['site']}.nc",
                    )
                )

    print("Files stored.")

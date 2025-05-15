import datetime
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

from ..core.geo import GeoAzimuth, gc_distance, shoot
from .mda import nearest_indexes, nearest_indexes_weighted
from .tracks import (
    Extract_basin_storms,
    centers_config_params,
    get_category,
    get_vmean,
    historic_track_interpolation,
    historic_track_preprocessing,
    nakajo_track_preprocessing,
    stopmotion_trim_circle,
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


def storm2stopmotion(df_storm):
    """
    df_storm    - (pandas.DataFrame)  Variables from storm preprocessing:
                  (lon,lat,p0,vmax,rmw, vmaxfill,rmwfill)
                  * _fill inform of historical gaps filled with estimates
                  * p0,vmax,rmw: mean values for 6h forward segment

    Generation of stopmotion units methodology from storm track segments of 6h:

    A.  Warmup segment (24h):
            4 segments to define start/end coordinates, {Vmean, relative angle}
            last 4th segment, mean {Pmin, Wmax, Rmw}, endpoint {lat}

    B.  Target segment (6h): {dP,dV,dW,dR,dAng}

    * Absolute value of latitude is stored (start of target segment)
    * Relative angle is referenced to the geographic north (southern
      hemisphere is multiplied with a factor of -1)

    returns:  df_ (pandas.Dataframe)
    """

    # no need to remove NaTs,NaNs -> historic_track_preprocessing
    # no need to remove wind/rmw -> historic_track_interpolation filled gaps

    # remove NaN
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
            arcl_h, gamma_h = gc_distance(lat2, lon2, lat1, lon1)
            RE = 6378.135  # earth radius [km]
            r = arcl_h * np.pi / 180.0 * RE  # distance [km]

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
        aseg[i] = GeoAzimuth(lat_0, lon_0, lat_i, lon_i)
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
        ang2 = GeoAzimuth(lat_i, lon_i, lat_i1, lon_i1)
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


def stopmotion_interpolation(df_seg, st=None, t_warm=24, t_seg=6, t_prop=42):
    """
    df_seg      - (pandas.DataFrame)  Stopmotion parameterized units:
                  (vseg,pseg,wseg,rseg,laseg,dvseg,dpseg,dwseg,drseg,daseg)
    st          - (pandas.DataFrame)  real storm
                  "None" for MDA segments (unrelated to historic tracks)
    t_warm      - warmup period [hour]
    t_seg       - target period [hour]
    t_prop      - propagation period [hour]

    Generation of SWAN cases, cartesian coordinates (SHyTCWaves configuration)
    A.  Warmup period (24h): over the negative x-axis ending at (x,y)=(0,0)
    B.  Target period (6h): starting at (x,y)=(0,0)
    C.  Propagation period (42h): no track coordinates (no wind forcing)

    returns:  st_list (pandas.DataFrame)
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


def find_analogue(df_library, df_case, ix_weights):
    """
    Finds the minimum distance in a n-dimensional normalized space,
    corresponding to the shytcwaves 10 parameters:
        {pseg, vseg, wseg, rseg, dp, dv, dw, dr, dA, lat}

    df_library  - (pandas.Dataframe) library parameters
    df_case     - (pandas.Dataframe) target case parameters
    ix_weights  - (array) columns indices weigth factors
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
    ix_scalar = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    ix_directional = [0]

    # get indices of nearest n-dimensional point
    ix_near = nearest_indexes_weighted(
        data_case, data_lib, ix_scalar, ix_directional, ix_weights
    )

    return ix_near


def analogue_endpoints(df_seg, df_analogue):
    """
    Adds segments endpoint coordinates looking up at the real target segment:
        * origin of warmup segment (shoot backwards)
        * origin of target segment (same as parameterized track)
        * end of target segment (shoot forwards)

    df_seg          - (pandas.DataFrame) parameterized historical storm
    df_analogue     - (pandas.DataFrame) analogue segments from library
    """

    # remove NaNs
    df_seg = df_seg[~df_seg.isna().any(axis=1)]

    # get hemishpere
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
    path_library,
    df_analogue,
    df_seg,
    path_mda,
    st,
    cp_lon_ls,
    cp_lat_ls,
    max_dist=60,
    list_out=False,
    tqdm_out=False,
    text_out=True,
    mode="",
):
    """
    Function to access the library analogue cases for a given storm track,
    calculate distance and angle from the target segment origin to the control
    point (relative coordinate system), and extract the directional wave
    spectra at the closest node (for every analogue segment)

    path_library    - (path) library of prerun segments
    path_mda        - (path) mda sample indices (grid sizes)

    df_analogue     - (pandas.DataFrame) analogue prerun segments from library
    df_seg          - (pandas.DataFrame) storm 6h-segments parameters
    st              - (pandas.DataFrame) storm track interpolated every 6h
    cp_lon/lat_ls   - (list) control point geographical coordinates
    max_dist        - (float) maximum distance [km] to extract closest node
    mode            - (str) high or low resolution library indices

    returns:    (xarray.Dataset) wave directional spectra (dim 'case')
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
        mask_rad, mask_ang = get_mask_radii_angle(path_mda, iseg_analogue, mode=mode)

        # find closest index
        ix_near = find_nearest(cp_dist_ls, cp_ang_ls, mask_rad, mask_ang)
        pos_nonan = np.abs(mask_rad[ix_near] - cp_dist_ls) <= max_dist

        # ---------------------------------------------------------------------
        # load library Hs reconstructed
        xds_rec = xr.open_dataset(
            op.join(path_library, "library_shytcwaves_bulk_params.nc")
        )

        # extract HS,TP at closest points (case,point,time)
        hs_arr = np.full((ix_near.size, xds_rec.time.values.size), np.nan)
        hs_arr[pos_nonan, :] = xds_rec.hs.values[iseg_analogue, ix_near[pos_nonan], :]

        tp_arr = np.full((ix_near.size, xds_rec.time.values.size), np.nan)
        tp_arr[pos_nonan, :] = xds_rec.tp.values[iseg_analogue, ix_near[pos_nonan], :]

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


def get_cp_radii_angle(st_lat, st_lon, cp_lat_ls, cp_lon_ls, sign, aseg):
    """
    Extracts the distance and angle of the control point in the relative
    coordinate system (analogue segment)

    st_lat, st_lon  - (float) storm coordinates
    cp_lat, cp_lon  - (list) control point coordinates
    sign            - (1) north hemisphere / (-1) south hemisphere
    aseg            - azimuth of the analogue warm segment (geographic north)
    """

    # earth radius [km]
    RE = 6378.135

    cp_dist_ls, cp_ang_ls = [], []
    for i in range(len(cp_lat_ls)):
        cp_lat, cp_lon = cp_lat_ls[i], cp_lon_ls[i]

        # get point polar reference
        # azimut is refered to geographical north (absolute system)
        arcl_h, ang_abs = gc_distance(st_lat, st_lon, cp_lat, cp_lon)
        cp_dist_ls.append(arcl_h * np.pi / 180.0 * RE)  # [km]

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


def get_mask_radii_angle(path_mda, icase, mode=""):
    """
    Extracts the indices for radii/angle at the output points

    path_mda    - directory of library mda
    icase       - analogue case id
    mode        - option to select shytcwaves library (high or low resolution)
    """

    # load output indices associated with distances/angles to target origin
    name = "mda_mask_indices" + mode + ".nc"
    p_mask = op.join(path_mda, name)
    if not op.isfile(p_mask):
        xds_mask_ind = get_mask_indices(path_mda, mode=mode)
    else:
        xds_mask_ind = xr.open_dataset(p_mask)

    # load MDA indices (grid sizes)
    #    xds_ind_mda = xr.open_dataset(op.join(path_mda, 'shytcwaves_mda_indices.nc'))
    xds_ind_mda = xr.open_dataset(op.join(path_mda, "shytcwaves_mda_indices_clean.nc"))

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


def get_mask_indices(path_mda, save=False, mode=""):
    """
    Creates file with the swan cases (grids small/medium/large)
    polar coordinates (radii and angle)

    path_mda    - directory of shytcwaves library files
    save        - option to save in same directory
    mode        - option to select shytcwaves library (high or low resolution)
    """

    # load library indices file
    #    xds_mda = xr.open_dataset(op.join(path_mda, 'shytcwaves_mda_indices.nc'))  # before cleaning wrong cases
    # some MDA cases were removed afterwards (wrong swan simulations)
    xds_mda = xr.open_dataset(op.join(path_mda, "shytcwaves_mda_indices_clean.nc"))

    # get one case for each grid size
    case_sma = xds_mda.indices_small.values[0]
    case_med = xds_mda.indices_medium.values[0]
    case_lar = xds_mda.indices_large.values[0]

    rr, aa = [], []
    # get analogue case
    for ic, case_id in enumerate([case_sma, case_med, case_lar]):
        # get polar coords (radii/distance, angle)
        rc, thetac = find_analogue_grid_coords(path_mda, case_id)

        if mode == "_lowres":  # new library of cases with half rings
            rc = rc[:, ::2]
            thetac = thetac[:, ::2]

        # to point dimension
        rr.append(np.reshape(rc, -1))  # [m]
        aa.append(np.reshape(thetac, -1))  # [º]

    # store all grid sizes mask indices
    xd = xr.Dataset(
        {
            "radii_sma": (("point_sma",), rr[0]),
            "angle_sma": (("point_sma",), aa[0]),
            "radii_med": (("point_med",), rr[1]),
            "angle_med": (("point_med",), aa[1]),
            "radii_lar": (("point_lar",), rr[2]),
            "angle_lar": (("point_lar",), aa[2]),
        }
    )

    # store dataset
    if save:
        xd.to_netcdf(op.join(path_mda, "mda_mask_indices" + mode + ".nc"))

    return xd


def find_analogue_grid_coords(path_mda, icase):
    """
    Finds the analogue segment case grid size (small/medium/large) and
    calculates the corresponding grid of polar coordinates
    """

    # load MDA indices (grid sizes)
    #    xds_ind_mda = xr.open_dataset(op.join(p_mda, 'shytcwaves_mda_indices.nc'))  # before cleaning wrong cases
    xds_ind_mda = xr.open_dataset(op.join(path_mda, "shytcwaves_mda_indices_clean.nc"))

    # get grid code
    pos_small = np.where(icase == xds_ind_mda.indices_small)[0]
    pos_medium = np.where(icase == xds_ind_mda.indices_medium)[0]
    pos_large = np.where(icase == xds_ind_mda.indices_large)[0]

    # number of rings
    if len(pos_small) == 1:
        num, rings = 0, 29
    elif len(pos_medium) == 1:
        num, rings = 1, 33
    elif len(pos_large) == 1:
        num, rings = 2, 33

    # get sizes
    xsize_1 = xds_ind_mda.x_size_left.values[num] * 10**3
    xsize_2 = xds_ind_mda.x_size_right.values[num] * 10**3
    ysize_1 = xds_ind_mda.y_size_left.values[num] * 10**3
    ysize_2 = xds_ind_mda.y_size_right.values[num] * 10**3

    #  (cartesian convention)
    # computational grid extent
    res = 15 * 10**3  # km to m   # TODO: resolution variable

    # create domain centered at (x,y)=(0,0)
    lon = np.arange(-xsize_1, xsize_2, res)
    lat = np.arange(-ysize_1, ysize_2, res)

    # custom output coordinates
    _, _, _, _, rc, thetac = generate_polar_coords(
        lon, lat, res, rings=rings, radii_ini=5000, angle_inc=5
    )

    return rc, thetac


def find_nearest(cp_rad_ls, cp_ang_ls, mask_rad, mask_ang):
    """
    Finds the minimum distance in a n-dimensional normalized space.

    cp_rad_ls, cp_ang_ls    - (list) control point
    mask_rad, mask_and      - (array) SWAN output point dimension
    """

    # create dataframes
    df_cp = pd.DataFrame({"radii": cp_rad_ls, "angle": cp_ang_ls})

    df_mask = pd.DataFrame({"radii": mask_rad, "angle": mask_ang})

    # indices
    ix_scalar = [0]
    ix_directional = [1]

    # get indices of nearest n-dimensional point
    ix_near = nearest_indexes(df_cp.values, df_mask.values, ix_scalar, ix_directional)

    return ix_near


###############################################################################
# SHyTCWaves APPLICATION

# Functions that calculate the ensemble/reconstruction for a storm track
# from either historical or forecast/predicted tracks
###############################################################################


def get_coef_calibration():
    """
    SHyTCWaves model was validated with satellite data, with a bias correction
    performed according to intensity categories
    """

    p = [1015, 990, 972, 954, 932, 880]  # Saffir-Simpson center categories
    dp = [-17, -15, -12.5, -7, +2.5, +10]  # calibrated "dP" for shytcwaves

    coef = np.polyfit(p, dp, 1)  # order 1 fitting

    return coef


##########################################
# SHYTCWAVES - historical track


def historic2shytcwaves_cluster(
    p_save,
    path_mda,
    p_library,
    tc_name,
    storm,
    center,
    lon,
    lat,
    dict_site=None,  # dict_site={},
    calibration=True,
    mode="",
    database_on=False,
    st_param=False,
    extract_bulk=True,
    max_segments=300,
):
    """
    Function that for a storm track and a target domain, provides the shytcwaves
    estimated induced spectral wave energy over time

    p_save     - (path) store for results
    tc_name    - (str) name
    iprediction- (float) predicted number
    storm      - (xarray.Dataset) with standard variables extracted from IBTrACS
                 compulsory: 'longitude,latitude', pressure[mbar], maxwinds[kt]
                 optional: rmw[nmile], dist2land[], basin[str]
    center     - (str) IBTrACS center track data (many are available)
    lon,lat    - (array) longitude/latitude nodes for calculating swath
    dict_site  - (dict) site data for superpoint building
    """

    # A: stopmotion segmentation, 6h interval
    df = historic_track_preprocessing(
        storm,
        center=center,
        database_on=database_on,
        forecast_on=False,
        st_param=st_param,
    )
    dt_int_seg = 6 * 60  # [minutes] constant segments

    # optional: shytcwaves calibration of track parameters
    if calibration:
        coef = get_coef_calibration()  # lineal fitting
        df["pressure"] = df["pressure"].values * (1 + coef[0]) + coef[1]
        df["maxwinds"] = np.nan

        st, _ = historic_track_interpolation(
            df,
            dt_int_seg,
            interpolation=False,
            mode="mean",
            fit=True,
            radi_estimate_on=True,
        )
    else:
        st, _ = historic_track_interpolation(
            df, dt_int_seg, interpolation=False, mode="mean"
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
        st.to_pickle(op.join(p_save, "{0}_track.pkl".format(tc_name)))
        st_trim.to_pickle(op.join(p_save, "{0}_track_trim.pkl".format(tc_name)))

        # parameterized segts (24h warmup + 6htarget)
        df_seg = storm2stopmotion(st_trim)

        if df_seg.shape[0] > 2:
            print("st:", st.shape[0], "df_seg:", df_seg.shape[0])
            st_list, we_list = stopmotion_interpolation(df_seg, st=st_trim)

            #######################################################################
            # B: analogue segments from library
            df_mda = pd.read_pickle(op.join(path_mda, "shytcwaves_mda_clean.pkl"))
            ix_weights = [1] * 10  # equal weights
            ix = find_analogue(df_mda, df_seg, ix_weights)

            df_analogue = df_mda.iloc[ix]
            df_analogue = analogue_endpoints(df_seg, df_analogue)

            st_list_analogue, we_list_analogue = stopmotion_interpolation(
                df_analogue, st=st_trim
            )

            #######################################################################
            # C: extract bulk envelope (to plot swaths)

            if extract_bulk:
                mesh_lo, mesh_la = np.meshgrid(lon, lat)
                print(
                    "Number of segments: {0}, number of swath nodes: {1}".format(
                        len(st_list), mesh_lo.size
                    )
                )

                if len(st_list) < max_segments:
                    xds_shy_bulk = stopmotion_st_bmu(
                        p_library,
                        df_analogue,
                        df_seg,
                        path_mda,
                        st_trim,
                        list(np.ravel(mesh_lo)),
                        list(np.ravel(mesh_la)),
                        max_dist=60,
                        mode=mode,
                    )
                    # store
                    xds_shy_bulk.to_netcdf(
                        op.join(p_save, "{0}_xds_shy_bulk.nc".format(tc_name))
                    )

            #######################################################################
            # D: extract spectra envelope
            if type(dict_site) == dict:
                xds_shy_spec, _ = stopmotion_st_spectra(
                    p_library,
                    df_analogue,
                    df_seg,
                    path_mda,
                    st_trim,
                    dict_site["lonpts"],
                    dict_site["latpts"],
                    cp_names=dict_site["namepts"],
                    max_dist=60,
                    list_out=False,
                    mode=mode,
                )
                # store
                xds_shy_spec.to_netcdf(
                    op.join(
                        p_save,
                        "{0}_xds_shy_spec_{1}.nc".format(tc_name, dict_site["site"]),
                    )
                )

                ###################################################################
    #                # E: build superpoint
    #                stations = list(np.arange(0, xds_shy_spec.point.size))
    #                xds_shy_sp = SuperPoint_Superposition(xds_shy_spec, stations,
    #                                                      dict_site['sectors'],
    #                                                      dict_site['deg_superposition'])
    #                # store
    #                xds_shy_sp.to_netcdf(op.join(p_save,
    #                    '{0}_xds_shy_sp_{2}.nc'.format(
    #                            tc_name, dict_site['site'])))

    print("Files stored.\n")

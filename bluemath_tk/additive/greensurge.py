import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path

from ..core.operations import get_degrees_from_uv


def create_triangle_mask(
    lon_grid: np.ndarray, lat_grid: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """
    Create a mask for a triangle defined by its vertices.

    Parameters
    ----------
    lon_grid : np.ndarray
        The longitude grid.
    lat_grid : np.ndarray
        The latitude grid.
    triangle : np.ndarray
        The triangle vertices.

    Returns
    -------
    np.ndarray
        The mask for the triangle.
    """

    triangle_path = Path(triangle)
    # if lon_grid.ndim == 1:
    #     lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
    inside_mask = triangle_path.contains_points(points)
    mask = inside_mask.reshape(lon_grid.shape)

    return mask


def create_triangle_mask_from_points(
    lon: np.ndarray, lat: np.ndarray, triangle: np.ndarray
) -> np.ndarray:
    """
    Create a mask indicating which scattered points are inside a triangle.

    Parameters
    ----------
    lon : np.ndarray
        1D array of longitudes of the points.
    lat : np.ndarray
        1D array of latitudes of the points.
    triangle : np.ndarray
        (3, 2) array containing the triangle vertices as (lon, lat) pairs.

    Returns
    -------
    np.ndarray
        1D boolean array of same length as lon/lat indicating points inside the triangle.
    """

    points = np.column_stack((lon, lat))  # Shape (N, 2)
    triangle_path = Path(triangle)
    mask = triangle_path.contains_points(points)

    return mask


def GS_LinearWindDragCoef(Wspeed, CD_Wl_abc, Wl_abc):
    Wla = Wl_abc[0]
    Wlb = Wl_abc[1]
    Wlc = Wl_abc[2]
    CDa = CD_Wl_abc[0]
    CDb = CD_Wl_abc[1]
    CDc = CD_Wl_abc[2]

    # coefs lines y=ax+b
    if not Wla == Wlb:
        a_CDline_ab = (CDa - CDb) / (Wla - Wlb)
        b_CDline_ab = CDb - a_CDline_ab * Wlb
    else:
        a_CDline_ab = 0
        b_CDline_ab = CDa
    if not Wlb == Wlc:
        a_CDline_bc = (CDb - CDc) / (Wlb - Wlc)
        b_CDline_bc = CDc - a_CDline_bc * Wlc
    else:
        a_CDline_bc = 0
        b_CDline_bc = CDb
    a_CDline_cinf = 0
    b_CDline_cinf = CDc

    if Wspeed <= Wlb:
        CD = a_CDline_ab * Wspeed + b_CDline_ab
    elif Wspeed > Wlb and Wspeed <= Wlc:
        CD = a_CDline_bc * Wspeed + b_CDline_bc
    else:
        CD = a_CDline_cinf * Wspeed + b_CDline_cinf

    return CD.values


def GS_windsetup_reconstruction_tri(
    p_GFD_libdir, ds_GFD_info, ds_wind_partition, d_guarda, tini, tend
):
    Wdir = ds_GFD_info.Wdir
    AWdir = ds_GFD_info.Wdir[1] - ds_GFD_info.Wdir[0]
    NTT = p_GFD_libdir.tes.values
    ND = len(ds_GFD_info.Wdir)
    Wspeed = ds_GFD_info.Wspeed.values
    DT_GS = ds_GFD_info.simul_hours
    CD_Wl_abc = ds_GFD_info.CD_Wl_abc
    Wl_abc = ds_GFD_info.Wl_abc
    dt = ds_GFD_info.simul_dt_hours.values
    time_hours = np.arange(tini, tend, np.timedelta64(int(60 * dt.item()), "m"))
    Dir_tes = ds_wind_partition.Dir_tes.sel(time=time_hours)
    Wspeed_tes = ds_wind_partition.Wspeed_tes.sel(time=time_hours)
    CD_base = GS_LinearWindDragCoef(Wspeed, CD_Wl_abc, Wl_abc)
    Ntime = len(time_hours)
    cont = 0
    discret_dir = Wdir.values
    A = 0

    for t in range(Ntime):
        for NT in NTT.astype(int):
            if Dir_tes[NT, t] > ((ND - 1) * AWdir):
                discret_dir = np.where(discret_dir == 0, 360, discret_dir)
            else:
                discret_dir = Wdir.values
            dif_dir = np.abs(discret_dir - Dir_tes[NT, t].values)
            pos_Dir = dif_dir.argmin(axis=0)  # from 0 to 23
            # maps
            WL_case = p_GFD_libdir["mesh2d_s1"].sel(tes=NT).sel(dir=pos_Dir).values
            WL_case = np.nan_to_num(WL_case, nan=0)
            if A == 0:
                WL_GS_WindSetUp = np.zeros((Ntime, WL_case.shape[1]))
                WL_case_scale_inter = np.zeros((WL_case.shape))
                A = 1
            # re-scale and drag coefficient
            Wspeed_tes_case = Wspeed_tes[NT, t].values
            CD_tes_case = GS_LinearWindDragCoef(Wspeed_tes_case, CD_Wl_abc, Wl_abc)
            WL_case_scale_inter += (
                WL_case * (Wspeed_tes_case**2 / Wspeed**2) * (CD_tes_case / CD_base)
            )

        DT_GS_aux = np.min((DT_GS, Ntime))
        if (Ntime - t) >= (DT_GS_aux):
            WL_GS_WindSetUp[t : (t + DT_GS_aux - d_guarda), :] = (
                WL_GS_WindSetUp[t : (t + DT_GS_aux - d_guarda), :]
                + WL_case_scale_inter[d_guarda:DT_GS_aux, :]
            )
            WL_case_scale_inter = np.zeros((WL_case.shape))
        else:
            cont += 1
            WL_GS_WindSetUp[t : (t + DT_GS_aux - cont - d_guarda), :] = (
                WL_GS_WindSetUp[t : (t + DT_GS_aux - cont - d_guarda), :]
                + WL_case_scale_inter[d_guarda : DT_GS_aux - cont, :]
            )
            WL_case_scale_inter = np.zeros((WL_case.shape))

    ds_WL_GS_WindSetUp = xr.Dataset(
        {
            "WL": (["time", "nface"], WL_GS_WindSetUp),
            "lon": (["nface"], p_GFD_libdir.mesh2d_face_x.values),
            "lat": (["nface"], p_GFD_libdir.mesh2d_face_y.values),
        },
        coords={"time": time_hours, "nface": p_GFD_libdir.mesh2d_nFaces.values},
    )
    return ds_WL_GS_WindSetUp


def axplot_var_map_tri(
    ax,
    XX,
    YY,
    TT,
    vv,
    vmin=None,
    vmax=None,
    cmap=plt.get_cmap("seismic"),
):
    "plot 2D map with variable data"

    # cplot v lims
    if vmin == None:
        vmin = np.nanmin(vv)
    if vmax == None:
        vmax = np.nanmax(vv)

    # plot variable 2D map
    pm = ax.tripcolor(
        XX,
        YY,
        TT,
        facecolors=vv,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        transform=ccrs.PlateCarree(),
    )

    # return pcolormesh
    return pm


def plot_GS_vs_dynamic_windsetup_tri(
    ds_WL_GS_WindSetUp,
    ds_WL_dynamic_WindSetUp,
    ds_GFD_calc_info,
    t,
    vmin,
    vmax,
    swath=False,
    figsize=[15, 12],
    loadcost=None,
):
    X = ds_GFD_calc_info.mesh2d_node_x.values
    Y = ds_GFD_calc_info.mesh2d_node_y.values
    triangles = ds_GFD_calc_info.node_triangle[:, 0:3].values
    XD = ds_WL_dynamic_WindSetUp.mesh2d_node_x.values
    YD = ds_WL_dynamic_WindSetUp.mesh2d_node_y.values
    TriangleD = ds_WL_dynamic_WindSetUp.mesh2d_face_nodes.values[:, 0:3] - 1

    if swath:
        if vmin == None:
            vmin = 0
            xds_GS = np.nanmax((ds_WL_GS_WindSetUp["WL"].values), axis=0)
            xds_DY = np.nanmax((ds_WL_dynamic_WindSetUp["mesh2d_s1"].values), axis=0)
            if vmax == None:
                vmax = float(np.nanmax(xds_GS))
            ccmap1 = "CMRmap_r"
            ccmap2 = "CMRmap_r"
        if vmax == None:
            vmax = 0
            xds_GS = np.nanmin((ds_WL_GS_WindSetUp["WL"].values), axis=0)
            xds_DY = np.nanmin((ds_WL_dynamic_WindSetUp["mesh2d_s1"].values), axis=0)
            if vmin == None:
                vmin = float(np.nanmin(xds_GS))
            ccmap1 = "CMRmap"
            ccmap2 = "CMRmap"

    else:
        xds_GS = ds_WL_GS_WindSetUp["WL"].sel(time=t)
        xds_DY = ds_WL_dynamic_WindSetUp["mesh2d_s1"].sel(time=t)
        ccmap1 = "bwr"
        ccmap2 = "bwr"
        # maximum and minimum wind values
        if vmax == None:
            vmax = max(
                np.abs(float(np.nanmax(xds_GS))), np.abs(float(np.nanmin(xds_DY)))
            )
        if vmin == None:
            vmin = vmax * -1

    fig, (axs) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    WL_units = "m"

    pm = axplot_var_map_tri(
        axs[0], XD, YD, TriangleD, xds_DY, vmin=vmin, vmax=vmax, cmap=ccmap1
    )
    cbar1 = fig.colorbar(pm, ax=axs[0], orientation="horizontal", pad=0.1)
    cbar1.ax.set_xlabel(
        "{0} ({1})".format("WL", WL_units),
        rotation=0,
        va="bottom",
        fontweight="bold",
        labelpad=15,
    )

    pm = axplot_var_map_tri(
        axs[1], X, Y, triangles, xds_GS, vmin=vmin, vmax=vmax, cmap=ccmap2
    )
    cbar2 = fig.colorbar(pm, ax=axs[1], orientation="horizontal", pad=0.1)
    cbar2.ax.set_xlabel(
        "{0} ({1})".format("WL", WL_units),
        rotation=0,
        va="bottom",
        fontweight="bold",
        labelpad=15,
    )

    if loadcost:
        axs[0].coastlines()
        axs[1].coastlines()

    if swath:
        axs[0].set_title("SWATH Dynamic Wind SetUp", fontsize=16, fontweight="bold")
        axs[1].set_title("SWATH GreenSurge Wind SetUp", fontsize=16, fontweight="bold")
    else:
        date_0 = ds_WL_GS_WindSetUp["time"].sel(time=t).values
        fmt = "%d-%b-%Y %H:%M%p"
        t_str = pd.to_datetime(str(date_0)).strftime(fmt)
        ttl_t = "Time: {0}".format(t_str)

        axs[0].set_title(
            "Dynamic Wind SetUp, \n{0}".format(ttl_t), fontsize=16, fontweight="bold"
        )
        axs[1].set_title(
            "GreenSurge Wind SetUp, \n{0}".format(ttl_t), fontsize=16, fontweight="bold"
        )

    axs[0].set_extent(
        [
            np.nanmin(ds_WL_GS_WindSetUp.lon.values),
            np.nanmax(ds_WL_GS_WindSetUp.lon.values),
            np.nanmin(ds_WL_GS_WindSetUp.lat.values),
            np.nanmax(ds_WL_GS_WindSetUp.lat.values),
        ]
    )
    axs[1].set_extent(
        [
            np.nanmin(ds_WL_GS_WindSetUp.lon.values),
            np.nanmax(ds_WL_GS_WindSetUp.lon.values),
            np.nanmin(ds_WL_GS_WindSetUp.lat.values),
            np.nanmax(ds_WL_GS_WindSetUp.lat.values),
        ]
    )


def wind_layer_from_era5(ds_Wind, ds_GFD_info):
    times = ds_Wind.valid_time.values
    era_lon = ds_Wind.longitude.values
    era_lat = ds_Wind.latitude.values
    in_lon, in_lat = np.meshgrid(era_lon, era_lat)
    in_u = ds_Wind.u10.values
    in_v = ds_Wind.v10.values
    in_P = ds_Wind.msl.values

    c_lon = ds_GFD_info.lon_grid.values
    c_lat = ds_GFD_info.lat_grid.values
    cg_lon, cg_lat = np.meshgrid(c_lon, c_lat)

    hld_u = np.zeros((*cg_lon.shape, len(times)))
    hld_v = np.zeros((*cg_lon.shape, len(times)))
    hld_p = np.zeros((*cg_lon.shape, len(times)))

    for i in range(len(times)):
        hld_u[:, :, i] = griddata(
            (in_lon.flatten(), in_lat.flatten()),
            in_u[i].flatten(),
            (cg_lon, cg_lat),
            method="linear",
        )
        hld_v[:, :, i] = griddata(
            (in_lat.flatten(), in_lon.flatten()),
            in_v[i].flatten(),
            (cg_lat, cg_lon),
            method="linear",
        )
        hld_p[:, :, i] = griddata(
            (in_lat.flatten(), in_lon.flatten()),
            in_P[i].flatten(),
            (cg_lat, cg_lon),
            method="linear",
        )

    hld_W = np.sqrt(hld_u**2 + hld_v**2)
    hld_D = get_degrees_from_uv(-hld_u, -hld_v)

    # generate vortex dataset
    xds_vortex = xr.Dataset(
        {
            "W": (("lat", "lon", "time"), hld_W, {"units": "m/s"}),
            "u": (("lat", "lon", "time"), hld_u, {"units": "m/s"}),
            "v": (("lat", "lon", "time"), hld_v, {"units": "m/s"}),
            "p": (("lat", "lon", "time"), hld_p, {"units": "mbar"}),
            "Dir": (("lat", "lon", "time"), hld_D, {"units": "º"}),
        },
        coords={
            "lat": c_lat,
            "lon": c_lon,
            "time": times,
        },
    )
    xds_vortex.attrs["xlabel"] = "lat"
    xds_vortex.attrs["ylabel"] = "lon"
    return xds_vortex


def GS_wind_partition_tri(ds_GFD_info, xds_vortex):
    # Code to split (partition) original TC-induced wind fields (regardless of their origin: from Holland vortex model or from forecasting systems)
    # taking into account spatial (i.e. cells) and temporal (i.e. length of sustained winds) resolutions defined in GFD

    # WARNING: only valid for discretization based on correlative square bings -->
    # TO DO: generalize (any kind of discretization)

    NTT = ds_GFD_info.teselas.values
    NumT = int(ds_GFD_info.NT)
    M = len(ds_GFD_info.M)
    N = len(ds_GFD_info.N)
    lon_grid = ds_GFD_info.lon_grid
    lat_grid = ds_GFD_info.lat_grid

    node_triangle = ds_GFD_info.node_triangle

    lon_teselas = ds_GFD_info.lon_node.isel(Node=node_triangle).values
    lat_teselas = ds_GFD_info.lat_node.isel(Node=node_triangle).values

    if np.abs(np.mean(lon_grid) - np.mean(lon_teselas)) > 180:
        lon_teselas = lon_teselas + 360

    # TC_info
    Ntime = len(xds_vortex.time)
    time = xds_vortex.time.values

    # storage
    U_tes = np.zeros((NumT, Ntime))
    V_tes = np.zeros((NumT, Ntime))
    Dir_tes = np.zeros((NumT, Ntime))
    Wspeed_tes = np.zeros((NumT, Ntime))

    for i in range(Ntime):
        W_grid = xds_vortex.W.values[:, :, i]
        Dir_grid = (270 - xds_vortex.Dir.values[:, :, i]) * np.pi / 180

        u_sel_t = W_grid * np.cos(Dir_grid)
        v_sel_t = W_grid * np.sin(Dir_grid)

        for NT in (NTT - 1).astype(int):
            X0, X1, X2 = lon_teselas[NT, :]
            Y0, Y1, Y2 = lat_teselas[NT, :]

            triangle = [(X0, Y0), (X1, Y1), (X2, Y2)]

            mask = create_triangle_mask(lon_grid, lat_grid, triangle)

            u_sel = u_sel_t[mask]
            v_sel = v_sel_t[mask]
            Dir = Dir_grid[mask]

            u_mean = np.nanmean(u_sel)
            v_mean = np.nanmean(v_sel)

            U_tes[NT, i] = u_mean
            V_tes[NT, i] = v_mean

            Dir_aux_2 = (circmean(Dir)) * 180 / (np.pi)
            Dir_aux = np.arctan2(v_mean, u_mean) * 180 / (np.pi)

            if np.abs(np.mean(Dir_aux_2 % 360) - np.mean(Dir_aux % 360)) > 5:
                print(
                    f"Issue arg(|W|) ¡= teta, diff {np.mean(Dir_aux_2 % 360) - np.mean(Dir_aux % 360)}"
                )

            Dir_tes[NT, i] = get_degrees_from_uv(-u_mean, -v_mean)

            Wspeed_tes[NT, i] = np.sqrt(u_mean**2 + v_mean**2)

    ds_wind_partition = xr.Dataset(
        {
            "U_tes": (["Ntes", "time"], U_tes),
            "V_tes": (["Ntes", "time"], V_tes),
            "Dir_tes": (["Ntes", "time"], Dir_tes),
            "Wspeed_tes": (["Ntes", "time"], Wspeed_tes),
            "lon_teselas": (("Ntes", "NN"), lon_teselas),
            "lat_teselas": (("Ntes", "NN"), lat_teselas),
            "lon_node": (("Node"), ds_GFD_info.lon_node.values),
            "lat_node": (("Node"), ds_GFD_info.lat_node.values),
            "node_triangle": (("Ntes", "NN"), node_triangle.values),
        },
        coords={
            "Ntes": (("Ntes"), NTT),
            "time": (("time"), time),
            "NN": (("NN"), [1, 2, 3]),
            "Node": (("Node"), ds_GFD_info.Node.values),
        },
    )

    return ds_wind_partition

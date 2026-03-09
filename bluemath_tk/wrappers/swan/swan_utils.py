import os
import os.path as op
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from scipy.io import loadmat, whosmat
import pandas as pd
import warnings

from ...core.operations import nautical_to_mathematical
from ...additive.additive import (
    create_triangle_mask_from_points,
    read_adcirc_grd,
)

sbatch_file_greenwaves = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

source /nfs/home/geocean/faugeree/miniforge3/etc/profile.d/conda.sh
conda activate work

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchSwan.sh --case-dir $case_dir

output_file="${case_dir}/output.mat"
output_file_raw="${case_dir}/output.raw"

python3 - <<EOF
import os, xarray as xr, numpy as np, struct
from bluemath_tk.wrappers.swan.swan_utils import mat_to_xr_dataset

output_file = r"${output_file}"
output_file_raw = r"${output_file_raw}"

ds = mat_to_xr_dataset(output_file, variables=["Hsig"])
data = ds["Hsig"].values.astype(np.float32)
shape = list(data.shape)
shape += [0] * (4 - len(shape))
header = struct.pack("4i", *shape) + bytes(256 - 16)

with open(output_file_raw, "wb") as f:
    f.write(header)
    f.write(data.tobytes())

# if ${SLURM_ARRAY_TASK_ID} != 1:
#     os.remove(output_file)
EOF
"""

def generate_forcing_file_GreenWaves(
    case_context: dict,
    case_dir: str,
    ds_GFD_info: xr.Dataset,
):
    """
    Generate the wind forcing files for a case in netCDF format (optimized version).
    """
    triangle_index = case_context.get("tesela")
    direction_index = case_context.get("direction")
    wind_direction = ds_GFD_info.wind_directions.values[direction_index]
    wind_speed = case_context.get("wind_magnitude")

    ref_date = pd.to_datetime(ds_GFD_info.reference_date.values.item())
    new_date = ref_date + pd.Timedelta(hours=int(case_context.get("simul_time")))
    d_date = pd.Timedelta(seconds=int(case_context.get("time_step_comp")))

    time = np.arange(ref_date, new_date + d_date, step = d_date, dtype = "datetime64[s]")

    connectivity = ds_GFD_info.triangle_forcing_connectivity
    triangle_longitude = ds_GFD_info.node_forcing_longitude.isel(
        node_forcing_index=connectivity
    ).values
    triangle_latitude = ds_GFD_info.node_forcing_latitude.isel(
        node_forcing_index=connectivity
    ).values

    connectivity_compo = ds_GFD_info.triangle_computation_connectivity.values
    node_lon = ds_GFD_info.node_computation_longitude.values.ravel()
    node_lat = ds_GFD_info.node_computation_latitude.values.ravel()

    # Selection triangle vertices
    x0, x1, x2 = triangle_longitude[triangle_index, :]
    y0, y1, y2 = triangle_latitude[triangle_index, :]
    triangle_vertices = np.array([(x0, y0), (x1, y1), (x2, y2)], dtype=float)

    # Compute centroid with NumPy (avoids shapely dependency)
    centroid = triangle_vertices.mean(axis=0)
    scale_factor = 1.001
    verts_buffered = centroid + (triangle_vertices - centroid) * scale_factor

    # Triangle mask (uses matplotlib Path)
    triangle_mask = create_triangle_mask_from_points(
        node_lon,
        node_lat,
        verts_buffered,
    )

    # Wind computation
    angle_rad = nautical_to_mathematical(wind_direction) * np.pi / 180
    wind_u = -np.cos(angle_rad) * wind_speed
    wind_v = -np.sin(angle_rad) * wind_speed

    # Initialize and assign wind arrays
    n_points = len(node_lon)
    n_time = len(time)
    windx = np.zeros((n_time, n_points))
    windy = np.zeros((n_time, n_points))



    windx[:, triangle_mask] = wind_u
    windy[:, triangle_mask] = wind_v

    with open(op.join(case_dir, case_context.get("forcing_file", "wind_forcing.wind")), "w") as f:
        for t in range(n_time):
            time_str = np.datetime_as_string(time[t], unit="s")
            time_str = time_str.replace("T", ".").replace("-", "").replace(":", "")
            f.write(f"{time_str}\n")
            f.write("wind x component\n")
            np.savetxt(f, windx[t, :].reshape(1, -1), fmt="%.3f")
            f.write("wind y component\n")
            np.savetxt(f, windy[t, :].reshape(1, -1), fmt="%.3f")


def mat_to_xr_dataset(path, variables=None, time_range=None):
    """
    Convert a MATLAB file to xarray.Dataset, loading only requested data.
    
    Parameters
    ----------
    path : str
        Path to .mat file
    variables : list of str, optional
        Variable names to load (e.g., ['Hsig', 'Windv_x']). 
        If None, loads all variables.
    time_range : tuple of (start, end), optional
        Only load timesteps within this range.
        e.g., ('2000-01-01', '2000-01-02')
    
    Returns
    -------
    xr.Dataset
    """
    mat_info = whosmat(path)
    all_keys = [name for name, shape, dtype in mat_info]

    vars_with_keys = {}
    
    for key in all_keys:
        if key.startswith(("Xp", "Yp", "__")):
            continue
        parts = key.rsplit("_", 2)
        if len(parts) >= 3:
            varname = "_".join(parts[:-2])
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            try:
                timestamp = pd.to_datetime(timestamp_str, format="%Y%m%d_%H%M%S")
                vars_with_keys.setdefault(varname, []).append((key, timestamp))
            except ValueError:
                continue

    if variables is not None:
        vars_with_keys = {k: v for k, v in vars_with_keys.items() if k in variables}
    
    if not vars_with_keys:
        raise ValueError(f"No matching variables found. Available: {list(vars_with_keys.keys())}")

    if time_range is not None:
        t_start, t_end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        for varname in vars_with_keys:
            vars_with_keys[varname] = [
                (k, t) for k, t in vars_with_keys[varname] 
                if t_start <= t <= t_end
            ]

    keys_to_load = {"Xp", "Yp"}
    for varname, key_times in vars_with_keys.items():
        keys_to_load.update(k for k, t in key_times)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate variable name")
        mat = loadmat(path, variable_names=list(keys_to_load))
    
    lon = mat["Xp"].ravel().astype(np.float32)
    lat = mat["Yp"].ravel().astype(np.float32)

    ds_vars = {}

    for varname, key_times in vars_with_keys.items():
        key_times.sort(key=lambda x: x[1])  # Sort by timestamp
        
        n_times = len(key_times)
        n_points = lon.size
        data_array = np.empty((n_times, n_points), dtype=np.float32)
        
        for i, (key, _) in enumerate(key_times):
            data_array[i] = mat[key].ravel()
        
        time_index = pd.DatetimeIndex([t for _, t in key_times])
        
        unique_times, time_indices = np.unique(time_index, return_index=True)
        data_array = data_array[time_indices]
        time_index = pd.DatetimeIndex(unique_times)
        
        ds_vars[varname] = (("time", "points"), data_array)
    
    return xr.Dataset(
        data_vars=ds_vars,
        coords={
            "time": time_index,
            "lon": ("points", lon),
            "lat": ("points", lat),
        },
    )


def vortex2SWAN(
    mesh: str,
    ds_vortex: xr.Dataset,
    output_path: str,
    fill_value: float = 0,
) -> xr.Dataset:
    """
    Convert the vortex dataset to a Delft3D FM compatible netCDF forcing file.

    Parameters
    ----------
    mesh : str
        The mesh path.
    ds_vortex : xarray.Dataset
        The vortex dataset containing wind speed and pressure data.
    path_output : str
        The output path where the netCDF file will be saved.
    ds_name : str
        The name of the output netCDF file, default is "forcing_Tonga_vortex.nc".
    forcing_ext : str
        The extension for the forcing file, default is "GreenSurge_GFDcase_wind.ext".
    fill_value : float
        The fill value to use for missing data, default is 0.
        
    Returns
    -------
    xarray.Dataset
        A dataset containing the interpolated wind speed and pressure data,
        ready for use in Delft3D FM.
    """

    Nodes_calc, _, _ = read_adcirc_grd(mesh)

    longitude = Nodes_calc[:, 1]
    latitude = Nodes_calc[:, 2]

    dt = (ds_vortex.time[1] - ds_vortex.time[0]).values
    time = np.arange(ds_vortex.time.size) * dt
    n_time = len(time)

    lat_interp = xr.DataArray(latitude, dims="node")
    lon_interp = xr.DataArray(longitude, dims="node")
    angle = np.deg2rad((270 - ds_vortex.Dir.values) % 360)
    W = ds_vortex.W.values

    windx_data = (W * np.cos(angle)).astype(np.float32)
    windy_data = (W * np.sin(angle)).astype(np.float32)
    pressure_data = ds_vortex.p.values.astype(np.float32)

    if windx_data.ndim == 3:
        windx_data = np.transpose(windx_data, (2, 0, 1))
        windy_data = np.transpose(windy_data, (2, 0, 1))
        pressure_data = np.transpose(pressure_data, (2, 0, 1))

    ds_vortex_interp = xr.Dataset(
        {
            "windx": (
                ("time", "latitude", "longitude"),
                np.nan_to_num(windx_data, nan=fill_value),
            ),
            "windy": (
                ("time", "latitude", "longitude"),
                np.nan_to_num(windy_data, nan=fill_value),
            ),
            "airpressure": (
                ("time", "latitude", "longitude"),
                np.nan_to_num(pressure_data, nan=fill_value),
            ),
        },
        coords={
            "time": time,
            "latitude": ds_vortex.lat.values,
            "longitude": ds_vortex.lon.values,
        },
    )
    forcing_dataset = ds_vortex_interp.interp(latitude=lat_interp, longitude=lon_interp)

    windx = forcing_dataset.windx.values
    windy = forcing_dataset.windy.values

    with open(output_path, "w") as f:
        for t in range(n_time):
            time_str = np.datetime_as_string(time[t], unit="s")
            time_str = time_str.replace("T", ".").replace("-", "").replace(":", "")
            f.write(f"{time_str}\n")
            f.write("wind x component\n")
            np.savetxt(f, windx[t, :].reshape(1, -1), fmt="%.3f")
            f.write("wind y component\n")
            np.savetxt(f, windy[t, :].reshape(1, -1), fmt="%.3f")
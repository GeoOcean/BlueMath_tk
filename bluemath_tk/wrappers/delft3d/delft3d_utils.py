import os
import os.path as op
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from ...core.operations import nautical_to_mathematical
from ...additive.additive import (
    create_triangle_mask_from_points,
    generate_structured_points_vectorized,
    point_to_segment_distance_vectorized,
)

sbatch_file_greensurge = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00

source /nfs/home/geocean/faugeree/miniforge3/etc/profile.d/conda.sh
conda activate work

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
launchDelft3dcomp.sh --case-dir $case_dir

output_file="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map.nc"
output_file_raw="${case_dir}/dflowfmoutput/GreenSurge_GFDcase_map.raw"

python3 - <<EOF
import os, xarray as xr, numpy as np, struct

output_file = r"${output_file}"
output_file_raw = r"${output_file_raw}"

ds = xr.open_dataset(output_file)
data = ds["mesh2d_s1"].values.astype(np.float32)
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

def generate_grid_forcing_file_netCDF_D3DFM(
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

    # Tolerance based on edge size
    edge_lengths = np.linalg.norm(
        np.roll(triangle_vertices, -1, axis=0) - triangle_vertices, axis=1
    )
    tol = np.mean(edge_lengths) * 0.001

    # Vectorized distance to 3 edges (~100x faster than shapely loop)
    dist_edge_01 = point_to_segment_distance_vectorized(
        node_lon, node_lat, x0, y0, x1, y1
    )
    dist_edge_12 = point_to_segment_distance_vectorized(
        node_lon, node_lat, x1, y1, x2, y2
    )
    dist_edge_20 = point_to_segment_distance_vectorized(
        node_lon, node_lat, x2, y2, x0, y0
    )
    dist_to_boundary = np.minimum(
        np.minimum(dist_edge_01, dist_edge_12), dist_edge_20
    )

    # Points on the boundary
    mask_on_edge = dist_to_boundary < tol

    # Triangles with at least one node on the boundary
    mask_tri_to_refine = np.any(mask_on_edge[connectivity_compo], axis=1)

    # Vectorized version of generate_structured_points
    lon_structured, lat_structured = generate_structured_points_vectorized(
        connectivity_compo[mask_tri_to_refine],
        node_lon,
        node_lat,
    )

    # Concatenate points
    longitude_points_computation = np.concatenate(
        [node_lon, lon_structured.ravel()]
    )
    latitude_points_computation = np.concatenate([node_lat, lat_structured.ravel()])

    # Triangle mask (uses matplotlib Path)
    triangle_mask = create_triangle_mask_from_points(
        longitude_points_computation,
        latitude_points_computation,
        verts_buffered,
    )

    # Wind computation
    angle_rad = nautical_to_mathematical(wind_direction) * np.pi / 180
    wind_u = -np.cos(angle_rad) * wind_speed
    wind_v = -np.sin(angle_rad) * wind_speed

    # Initialize and assign wind arrays
    n_points = len(longitude_points_computation)
    windx = np.zeros((4, n_points))
    windy = np.zeros((4, n_points))
    windx[:2, triangle_mask] = wind_u
    windy[:2, triangle_mask] = wind_v

    # Build forcing dataset
    ds_forcing = xr.Dataset(
        {
            "time": ds_GFD_info["time_forcing_index"],
            "node": xr.DataArray(np.arange(n_points), dims=["node"]),
            "longitude": xr.DataArray(
                longitude_points_computation,
                dims=["node"],
                attrs={
                    "description": "Longitude of each mesh node of the computational grid",
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "latitude": xr.DataArray(
                latitude_points_computation,
                dims=["node"],
                attrs={
                    "description": "Latitude of each mesh node of the computational grid",
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "windx": xr.DataArray(
                windx,
                dims=["time", "node"],
                attrs={
                    "coordinates": "time node",
                    "long_name": "Wind speed in x direction",
                    "standard_name": "windx",
                    "units": "m s-1",
                },
            ),
            "windy": xr.DataArray(
                windy,
                dims=["time", "node"],
                attrs={
                    "coordinates": "time node",
                    "long_name": "Wind speed in y direction",
                    "standard_name": "windy",
                    "units": "m s-1",
                },
            ),
        }
    )

    ds_forcing.to_netcdf(op.join(case_dir, "forcing.nc"))

def generate_grid_forcing_file_D3DFM(
    case_context: dict,
    case_dir: str,
    ds_GFD_info: xr.Dataset,
):
    """
    Generate the wind files for a case.

    Parameters
    ----------
    case_context : dict
        The case context.
    case_dir : str
        The case directory.
    ds_GFD_info : xr.Dataset
        The dataset with the GFD information.
    """
    dir_steps = case_context.get("dir_steps")
    real_dirs = np.linspace(0, 360, dir_steps + 1)[:-1]
    i_tes = case_context.get("tesela")
    i_dir = case_context.get("direction")
    real_dir = real_dirs[i_dir]
    dt_forz = case_context.get("dt_forz")
    wind_magnitude = case_context.get("wind_magnitude")
    simul_time = case_context.get("simul_time")

    node_triangle = ds_GFD_info.triangle_forcing_connectivity.isel(
        element_forcing_index=i_tes
    )
    lon_teselas = ds_GFD_info.node_forcing_longitude.isel(
        node_forcing_index=node_triangle
    ).values
    lat_teselas = ds_GFD_info.node_forcing_latitude.isel(
        node_forcing_index=node_triangle
    ).values

    lon_grid = ds_GFD_info.lon_grid.values
    lat_grid = ds_GFD_info.lat_grid.values

    x_llcenter = lon_grid[0]
    y_llcenter = lat_grid[0]

    n_cols = len(lon_grid)
    n_rows = len(lat_grid)

    dx = (lon_grid[-1] - lon_grid[0]) / n_cols
    dy = (lat_grid[-1] - lat_grid[0]) / n_rows
    X0, X1, X2 = lon_teselas
    Y0, Y1, Y2 = lat_teselas

    triangle = [(X0, Y0), (X1, Y1), (X2, Y2)]
    mask = create_triangle_mask_from_points(lon_grid, lat_grid, triangle).astype(
        int
    )
    mask_int = np.flip(mask, axis=0)  # Flip to match grid orientation

    u = -np.cos(nautical_to_mathematical(real_dir) * np.pi / 180) * wind_magnitude
    v = -np.sin(nautical_to_mathematical(real_dir) * np.pi / 180) * wind_magnitude
    u_mat = mask_int * u
    v_mat = mask_int * v

    file_name_u = op.join(case_dir, "GFD_wind_file.amu")
    file_name_v = op.join(case_dir, "GFD_wind_file.amv")

    with open(file_name_u, "w+") as fu, open(file_name_v, "w+") as fv:
        fu.write(
            "### START OF HEADER\n"
            + "### This file is created by Deltares\n"
            + "### Additional commments\n"
            + "FileVersion = 1.03\n"
            + "filetype = meteo_on_equidistant_grid\n"
            + "NODATA_value = -9999.0\n"
            + f"n_cols = {n_cols}\n"
            + f"n_rows = {n_rows}\n"
            + "grid_unit = degree\n"
            + f"x_llcenter = {x_llcenter}\n"
            + f"y_llcenter = {y_llcenter}\n"
            + f"dx = {dx}\n"
            + f"dy = {dy}\n"
            + "n_quantity = 1\n"
            + "quantity1 = x_wind\n"
            + "unit1 = m s-1\n"
            + "### END OF HEADER\n"
        )
        fv.write(
            "### START OF HEADER\n"
            + "### This file is created by Deltares\n"
            + "### Additional commments\n"
            + "FileVersion = 1.03\n"
            + "filetype = meteo_on_equidistant_grid\n"
            + "NODATA_value = -9999.0\n"
            + f"n_cols = {n_cols}\n"
            + f"n_rows = {n_rows}\n"
            + "grid_unit = degree\n"
            + f"x_llcenter = {x_llcenter}\n"
            + f"y_llcenter = {y_llcenter}\n"
            + f"dx = {dx}\n"
            + f"dy = {dy}\n"
            + "n_quantity = 1\n"
            + "quantity1 = y_wind\n"
            + "unit1 = m s-1\n"
            + "### END OF HEADER\n"
        )
        for time in range(4):
            if time == 0:
                time_real = time
            elif time == 1:
                time_real = dt_forz
            elif time == 2:
                time_real = dt_forz + 0.01
            elif time == 3:
                time_real = simul_time
            fu.write(f"TIME = {time_real} hours since 2022-01-01 00:00:00 +00:00\n")
            fv.write(f"TIME = {time_real} hours since 2022-01-01 00:00:00 +00:00\n")
            if time in [0, 1]:
                fu.write(format_matrix(u_mat) + "\n")
                fv.write(format_matrix(v_mat) + "\n")
            else:
                fu.write(format_zeros(u_mat.shape) + "\n")
                fv.write(format_zeros(v_mat.shape) + "\n")


def format_matrix(mat):
    return "\n".join(
        " ".join(f"{x:.1f}" if abs(x) > 0.01 else "0" for x in line) for line in mat
    )

def format_zeros(mat_shape):
    return "\n".join("0 " * mat_shape[1] for _ in range(mat_shape[0]))
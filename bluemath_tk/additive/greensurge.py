from functools import lru_cache
import struct
import numpy as np
import xarray as xr
from tqdm import tqdm



def GS_LinearWindDragCoef(
    Wspeed: np.ndarray, CD_Wl_abc: np.ndarray, Wl_abc: np.ndarray
) -> np.ndarray:
    """
    Calculate the linear drag coefficient based on wind speed and specified thresholds.

    Parameters
    ----------
    Wspeed : np.ndarray
        Wind speed values (1D array).
    CD_Wl_abc : np.ndarray
        Coefficients for the drag coefficient calculation, should be a 1D array of length 3.
    Wl_abc : np.ndarray
        Wind speed thresholds for the drag coefficient calculation, should be a 1D array of length 3.

    Returns
    -------
    np.ndarray
        Calculated drag coefficient values based on the input wind speed.
    """

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

    return CD


def actualize_grid_info(
    path_ds_origin: str,
    ds_GFD_calc_info: xr.Dataset,
) -> None:
    """
    Actualizes the grid information in the GFD calculation info dataset
    by adding the node coordinates and triangle connectivity from the original dataset.
    Parameters
    ----------
    path_ds_origin : str
        Path to the original dataset containing the mesh2d node coordinates.
    ds_GFD_calc_info : xr.Dataset
        The dataset containing the GFD calculation information to be updated.
    Returns
    -------
    ds_GFD_calc_info : xr.Dataset
        The updated dataset with the node coordinates and triangle connectivity.
    """

    ds_ori = xr.open_dataset(path_ds_origin)

    ds_GFD_calc_info["node_computation_longitude"] = (
        ("node_cumputation_index",),
        ds_ori.mesh2d_node_x.values,
    )
    ds_GFD_calc_info["node_computation_latitude"] = (
        ("node_cumputation_index",),
        ds_ori.mesh2d_node_y.values,
    )
    ds_GFD_calc_info["triangle_computation_connectivity"] = (
        ("element_computation_index", "triangle_forcing_nodes"),
        (ds_ori.mesh2d_face_nodes.values - 1).astype("int32"),
    )

    return ds_GFD_calc_info

def GS_windsetup_reconstruction_nc(
    greensurge_dataset,
    ds_gfd_metadata: xr.Dataset,
    wind_direction_input: xr.Dataset,
    velocity_thresholds: np.ndarray = np.array([0, 100, 100]),
    drag_coefficients: np.ndarray = np.array([0.00063, 0.00723, 0.00723]),
) -> xr.Dataset:
    """
    Reconstructs the GreenSurge wind setup using the provided wind direction input and metadata.

    Parameters
    ----------
    greensurge_dataset : xr.Dataset
        xarray Dataset containing the GreenSurge mesh and forcing data.
    ds_gfd_metadata: xr.Dataset
        xarray Dataset containing metadata for the GFD mesh.
    wind_direction_input: xr.Dataset
        xarray Dataset containing wind direction and speed data.
    velocity_thresholds : np.ndarray
        Array of velocity thresholds for drag coefficient calculation.
    drag_coefficients : np.ndarray
        Array of drag coefficients corresponding to the velocity thresholds.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing the reconstructed wind setup.
    """

    velocity_thresholds = np.asarray(velocity_thresholds)
    drag_coefficients = np.asarray(drag_coefficients)

    direction_bins = ds_gfd_metadata.wind_directions.values
    forcing_cell_indices = ds_gfd_metadata.element_forcing_index.values
    wind_speed_reference = ds_gfd_metadata.wind_speed.values.item()
    base_drag_coeff = GS_LinearWindDragCoef(
        wind_speed_reference, drag_coefficients, velocity_thresholds
    )
    time_step_hours = ds_gfd_metadata.time_step_hours.values

    time_start = wind_direction_input.time.values.min()
    time_end = wind_direction_input.time.values.max()
    duration_in_steps = (
        int((ds_gfd_metadata.simulation_duration_hours.values) / time_step_hours) + 1
    )
    output_time_vector = np.arange(
        time_start, time_end, np.timedelta64(int(60 * time_step_hours.item()), "m")
    )
    num_output_times = len(output_time_vector)

    direction_data = wind_direction_input.Dir.values
    wind_speed_data = wind_direction_input.W.values

    ds_ex = xr.open_dataset(f"{greensurge_dataset}/GF_T_0_D_0/dflowfmoutput/GreenSurge_GFDcase_map.nc")

    n_faces = ds_ex["mesh2d_s1"].shape

    wind_setup_output = np.zeros((num_output_times, n_faces[1]))
    water_level_accumulator = np.zeros(n_faces)

    for time_index in tqdm(range(num_output_times), desc="Processing time steps"):
        water_level_accumulator[:] = 0
        for cell_index in forcing_cell_indices.astype(int):
            current_dir = direction_data[cell_index, time_index] % 360
            adjusted_bins = np.where(direction_bins == 0, 360, direction_bins)
            closest_direction_index = np.abs(adjusted_bins - current_dir).argmin()

            water_level_case = xr.open_dataset(
                f"{greensurge_dataset}/GF_T_{cell_index}_D_{closest_direction_index}/dflowfmoutput/GreenSurge_GFDcase_map.nc"
            )["mesh2d_s1"].values
            water_level_case = np.nan_to_num(water_level_case, nan=0)

            wind_speed_value = wind_speed_data[cell_index, time_index]
            drag_coeff_value = GS_LinearWindDragCoef(
                wind_speed_value, drag_coefficients, velocity_thresholds
            )

            scaling_factor = (wind_speed_value**2 / wind_speed_reference**2) * (
                drag_coeff_value / base_drag_coeff
            )
            water_level_accumulator += water_level_case * scaling_factor

        step_window = min(duration_in_steps, num_output_times - time_index)
        if (num_output_times - time_index) > step_window:
            wind_setup_output[time_index : time_index + step_window] += (
                water_level_accumulator
            )
        else:
            shift_counter = step_window - (num_output_times - time_index)
            wind_setup_output[
                time_index : time_index + step_window - shift_counter
            ] += water_level_accumulator[: step_window - shift_counter]

    ds_wind_setup = xr.Dataset(
        {"WL": (["time", "nface"], wind_setup_output)},
        coords={
            "time": output_time_vector,
            "nface": np.arange(wind_setup_output.shape[1]),
        },
    )
    ds_wind_setup.attrs["description"] = "Wind setup from GreenSurge methodology"

    return ds_wind_setup

@lru_cache(maxsize=256)
def read_raw_with_header(raw_path: str) -> np.ndarray:
    """
    Read a .raw file with a 256-byte header and return a numpy float32 array.

    Parameters
    ----------
    raw_path : str
        Path to the .raw file.

    Returns
    -------
    np.ndarray
        The data array reshaped according to the header dimensions.
    """
    with open(raw_path, "rb") as f:
        header_bytes = f.read(256)
        dims = list(struct.unpack("4i", header_bytes[:16]))
        dims = [d for d in dims if d > 0]
        if len(dims) == 0:
            raise ValueError(f"{raw_path}: invalid header, no dimension > 0 found")
        data = np.fromfile(f, dtype=np.float32)
    expected_size = np.prod(dims)
    if data.size != expected_size:
        raise ValueError(
            f"{raw_path}: size mismatch (data={data.size}, expected={expected_size}, shape={dims})"
        )
    return np.reshape(data, dims)


def greensurge_wind_setup_reconstruction_raw(
    greensurge_dataset: str,
    ds_GFD_info_update: xr.Dataset,
    xds_vortex_interp: xr.Dataset,
) -> xr.Dataset:
    """
    Compute the GreenSurge wind contribution and return an xarray Dataset with the results.

    Parameters
    ----------
    greensurge_dataset : str
        Path to the GreenSurge dataset directory.
    ds_GFD_info_update : xr.Dataset
        Updated GreenSurge information dataset.
    xds_vortex_interp : xr.Dataset
        Interpolated vortex dataset.

    Returns
    -------
    xr.Dataset
        Dataset containing the GreenSurge wind setup contribution.
    """

    ds_gfd_metadata = ds_GFD_info_update
    wind_direction_input = xds_vortex_interp
    velocity_thresholds = np.array([0, 100, 100])
    drag_coefficients = np.array([0.00063, 0.00723, 0.00723])

    direction_bins = ds_gfd_metadata.wind_directions.values
    forcing_cell_indices = ds_gfd_metadata.element_forcing_index.values
    wind_speed_reference = ds_gfd_metadata.wind_speed.values.item()
    base_drag_coeff = GS_LinearWindDragCoef(
        wind_speed_reference, drag_coefficients, velocity_thresholds
    )
    time_step_hours = ds_gfd_metadata.time_step_hours.values

    time_start = wind_direction_input.time.values.min()
    time_end = wind_direction_input.time.values.max()
    duration_in_steps = (
        int((ds_gfd_metadata.simulation_duration_hours.values) / time_step_hours) + 1
    )
    output_time_vector = np.arange(
        time_start, time_end, np.timedelta64(int(time_step_hours * 60), "m")
    )
    num_output_times = len(output_time_vector)

    direction_data = wind_direction_input.Dir.values
    wind_speed_data = wind_direction_input.W.values

    sample_path = (
        f"{greensurge_dataset}/GF_T_0_D_0/dflowfmoutput/GreenSurge_GFDcase_map.raw"
    )
    sample_data = read_raw_with_header(sample_path)
    n_faces = sample_data.shape[-1]
    wind_setup_output = np.zeros((num_output_times, n_faces), dtype=np.float32)
    water_level_accumulator = np.zeros(sample_data.shape, dtype=np.float32)

    for time_index in tqdm(range(num_output_times), desc="Processing time steps"):
        water_level_accumulator[:] = 0
        for cell_index in forcing_cell_indices.astype(int):
            current_dir = direction_data[cell_index, time_index] % 360
            adjusted_bins = np.where(direction_bins == 0, 360, direction_bins)
            closest_direction_index = np.abs(adjusted_bins - current_dir).argmin()

            raw_path = f"{greensurge_dataset}/GF_T_{cell_index}_D_{closest_direction_index}/dflowfmoutput/GreenSurge_GFDcase_map.raw"
            water_level_case = read_raw_with_header(raw_path)

            water_level_case = np.nan_to_num(water_level_case, nan=0)

            wind_speed_value = wind_speed_data[cell_index, time_index]
            drag_coeff_value = GS_LinearWindDragCoef(
                wind_speed_value, drag_coefficients, velocity_thresholds
            )

            scaling_factor = (wind_speed_value**2 / wind_speed_reference**2) * (
                drag_coeff_value / base_drag_coeff
            )
            water_level_accumulator += water_level_case * scaling_factor

        step_window = min(duration_in_steps, num_output_times - time_index)
        if (num_output_times - time_index) > step_window:
            wind_setup_output[time_index : time_index + step_window] += (
                water_level_accumulator
            )
        else:
            shift_counter = step_window - (num_output_times - time_index)
            wind_setup_output[
                time_index : time_index + step_window - shift_counter
            ] += water_level_accumulator[: step_window - shift_counter]

    ds_wind_setup = xr.Dataset(
        {"WL": (["time", "nface"], wind_setup_output)},
        coords={
            "time": output_time_vector,
            "nface": np.arange(wind_setup_output.shape[1]),
        },
    )
    return ds_wind_setup
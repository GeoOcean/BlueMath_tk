import xarray as xr

from bluemath_tk.topo_bathy.swan_grid import generate_grid_parameters
from bluemath_tk.waves.binwaves import process_kp_coefficients, transform_CAWCR_WS
from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper

# Usage example
if __name__ == "__main__":
    # Load GEBCO bathymetry
    bathy_data = xr.open_dataset(
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/bati_tarawa_500m_LONLAT.nc"
    )
    # Generate grid parameters
    grid_parameters = generate_grid_parameters(bathy_data)
    # Define the input parameters
    templates_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swan/templates/kapi_biwaves/"
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/kapi/"
    # Load swan model parameters
    model_parameters = (
        xr.open_dataset("/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/subset.nc")
        .to_dataframe()
        .iloc[:10]
        .to_dict(orient="list")
    )
    # Create an instance of the SWAN model wrapper
    swan_wrapper = SwanModelWrapper(
        templates_dir=templates_dir,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_wrapper.build_cases(mode="one_by_one")
    # List available launchers
    print(swan_wrapper.list_available_launchers())
    # Run the model
    # swan_wrapper.run_cases(launcher="docker", parallel=True)
    # Post-process the output files
    postprocessed_ds = swan_wrapper.postprocess_cases()
    print(postprocessed_ds)
    # Load spectra example
    spectra = xr.open_dataset(
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/Waves_Cantabria_356.08_43.82.nc"
    )
    spectra_transformed = transform_CAWCR_WS(spectra)
    # Extract binwaves kp coeffs
    kp_coeffs = process_kp_coefficients(
        swan_ds=postprocessed_ds,
        spectrum_freq=spectra_transformed.freq.values,
        spectrum_dir=spectra_transformed.dir.values,
        latitude=43.3,
        longitude=173.0,
    )
    print(kp_coeffs)
    # Reconstruct spectra with kp coeffs

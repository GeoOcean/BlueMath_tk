import xarray as xr
from bluemath_tk.topo_bathy.swan_grid import generate_grid_parameters
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
        .iloc[:100]
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
    swan_wrapper.run_cases(launcher="docker", parallel=True)
    swan_wrapper.run_cases_bulk("sbatch javi_slurm.sh")

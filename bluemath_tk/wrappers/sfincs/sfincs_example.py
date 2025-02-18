import os.path as op
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
from hydromt_sfincs import SfincsModel, utils

from bluemath_tk.wrappers.sfincs.sfincs_wrapper import SfincsModelWrapper


class SfincsPabloModelWrapper(SfincsModelWrapper):
    """
    Wrapper for the SFINCS model (Pablo version 10/02/2025).
    """

    p_data = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/SFINCS/data"

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
        catalogs_list: List[str],
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        catalogs_list : List[str]
            The list of catalogs.
        """

        # Instantiate the model
        sf_model = SfincsModel(data_libs=catalogs_list, root=case_dir, mode="w+")
        sf_model.setup_grid(
            x0=408002.5,
            y0=8467002.5,
            dx=5.0,
            dy=5.0,
            nmax=1659,
            mmax=3299,
            rotation=0,
            epsg=32702,
        )
        datasets_dep = [{"elevtn": op.join(self.p_data, "apia_dem_v2.tif")}]
        sf_model.setup_dep(datasets_dep=datasets_dep)
        sf_model.setup_mask_active(zmin=-4, reset_mask=True)
        sf_model.setup_mask_bounds(btype="waterlevel", zmax=-1.99, reset_bounds=True)
        sf_model.setup_mask_bounds(btype="outflow", zmin=1, reset_bounds=False)
        sf_model.setup_config(
            **{
                "tref": "20000101 000000",
                "tstart": "20000101 000000",
                "tstop": "20000101 022315",
                "dtout": "100",
            }
        )
        #### Load location of focings csv
        location_points = pd.read_csv(op.join(self.p_data, "forcing_points.csv"))
        # start_points = pd.read_csv(op.join(self.p_data, "start_points.csv"))
        # end_points = pd.read_csv(op.join(self.p_data, "end_points.csv"))
        #### Load forcing levels csv
        water_levels = pd.read_csv(
            op.join(
                self.p_data,
                f"water_levels_sfincs_apia_rp_{case_context.get('waterlevel')}.csv",
            ),
            index_col=0,
        )
        water_levels.columns = range(1, len(water_levels.columns) + 1)
        #### Loc points
        x = location_points.x.values
        y = location_points.y.values

        # add to Geopandas dataframe as needed by HydroMT
        pnts = gpd.points_from_xy(x, y)
        index = np.arange(1, len(x) + 1, 1)  # NOTE that the index should start at one
        bnd = gpd.GeoDataFrame(index=index, geometry=pnts, crs=sf_model.crs)
        # add time series
        time = pd.date_range(
            start=utils.parse_datetime(sf_model.config["tstart"]),
            end=utils.parse_datetime(sf_model.config["tstop"]),
            periods=len(water_levels),
        )
        water_levels.index = time
        # water_levels[2] = water_levels[1]
        sf_model.setup_waterlevel_forcing(timeseries=water_levels, locations=bnd[0::49])
        sf_model.write()  # write all

    def build_cases(
        self,
        catalogs_list: List[str],
        mode: str = "one_by_one",
    ) -> None:
        """
        Build the input files for all cases.

        Parameters
        ----------
        catalogs_list : List[str]
            The list of catalogs.
        mode : str, optional
            The mode to build the cases. Default is "one_by_one".

        Raises
        ------
        ValueError
            If the cases were not properly built
        """

        super().build_cases(mode=mode)
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            self.build_case(
                case_context=case_context,
                case_dir=case_dir,
                catalogs_list=catalogs_list,
            )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/sfincs/templates/"
    )
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/sfincs/pablo/"
    # Load swan model parameters
    model_parameters = {"waterlevel": [5.0, 20.0, 50.0, 100.0]}
    # Create an instance of the SWAN model wrapper
    sfincs_wrapper = SfincsPabloModelWrapper(
        templates_dir=templates_dir,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    sfincs_wrapper.build_cases(
        mode="one_by_one",
        catalogs_list=[
            "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/SFINCS/catalogues/manning_cat.yml",
            "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_data/SFINCS/catalogues/topo_cat.yml",
        ],
    )
    # List available launchers
    print(sfincs_wrapper.list_available_launchers())
    # Run the model
    sfincs_wrapper.run_cases(launcher="docker", parallel=True)

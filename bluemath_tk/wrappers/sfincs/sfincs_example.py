import os.path as op

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt_sfincs import SfincsModel, utils

from bluemath_tk.wrappers.sfincs.sfincs_wrapper import SfincsModelWrapper


class SfincsPabloModelWrapper(SfincsModelWrapper):
    """
    Wrapper for the SFINCS model (Pablo version 26/02/2025).
    """

    p_data = "/path/to/data"
    p_hybeat = "/path/to/hybeat"
    cluster_centroids = pd.read_csv(op.join(p_data, "tc_representative.csv"))
    simulation_xr = xr.open_dataset(op.join(p_data, "representative_events.nc"))
    location_points = pd.read_csv(op.join(p_data, "forcing_points.csv"))
    datasets_dep = [{"elevtn": op.join(p_data, "apia_5m_soften.tif")}]
    catalogs_list = [
        op.join(p_data, "catalogues/manning_cat.yml"),
        op.join(p_data, "catalogues/topo_cat.yml"),
    ]

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        # Load TC position from cluster centroids
        TC_pos = self.cluster_centroids.TC_pos.values[int(case_context.get("case_num"))]

        # Instantiate the SFINCS model
        sf_model = SfincsModel(data_libs=self.catalogs_list, root=case_dir, mode="w+")

        # Set different parameters for the model
        sf_model.setup_grid(
            x0=408002.5,
            y0=8467002.5,
            dx=5.0,
            dy=5.0,
            nmax=1523,
            mmax=2494,
            rotation=0,
            epsg=32702,
        )
        sf_model.setup_dep(datasets_dep=self.datasets_dep)
        sf_model.setup_mask_active(zmin=-2, zmax=10, reset_mask=True)
        sf_model.setup_mask_bounds(btype="waterlevel", zmax=-1.99, reset_bounds=True)
        sf_model.setup_mask_bounds(btype="outflow", zmin=1, reset_bounds=False)
        sf_model.setup_config(
            **{
                "tref": "20000101 000000",
                "tstart": "20000101 000000",
                "tstop": "20000101 060000",
                "dtout": "100",
            }
        )

        # Load wave data for the selected TC
        wave_setup = pd.read_csv(
            op.join(self.p_hybeat, f"tc_{TC_pos}/HyBeat_level_tc_{TC_pos}.csv"),
            index_col=0,
        )
        wave_ig = pd.read_csv(
            op.join(self.p_hybeat, f"tc_{TC_pos}/HyBeat_ig_tc_{TC_pos}.csv"),
            index_col=0,
        )
        swl = pd.read_csv(
            op.join(self.p_hybeat, f"tc_{TC_pos}/HyBeat_swl_tc_{TC_pos}.csv"),
            index_col=0,
        )

        # Create custom waterlevel forcing depending on swl values
        if np.sum(swl.iloc[9:16]["SWL"].values) > 0:
            wave_setup = wave_setup.iloc[9:16]
            wave_ig = wave_ig.iloc[9:16]
            swl = swl.iloc[9:16]
            t_10s = np.arange(0, 6 * 3600, 10)
            transects = np.arange(60, 110, 1)
            waterlevels = pd.DataFrame()
            hours = np.arange(0, 7 * 3600, 3600)
            for i in transects:
                setup = np.array(wave_setup[f"{i}.0"].values)
                ig = np.array(wave_ig[f"{i}.0"].values)
                wave_setup_interp = np.interp(t_10s, hours, setup)
                swl_interp = np.interp(t_10s, hours, swl["SWL"].values)
                ig_amplitude_interp = np.interp(t_10s, hours, ig)
                ig_interp = ig_amplitude_interp * np.sin(2 * np.pi * t_10s / 200)
                _result = np.column_stack(
                    (t_10s, wave_setup_interp, swl_interp, ig_interp)
                )
                twl = wave_setup_interp + swl_interp + ig_interp
                waterlevels[f"{i}.0"] = twl

            waterlevels.columns = range(1, len(waterlevels.columns) + 1)
            # Loc points
            x = self.location_points.x.values
            y = self.location_points.y.values

            # Add to Geopandas dataframe as needed by HydroMT
            pnts = gpd.points_from_xy(x, y)
            index = np.arange(
                1, len(x) + 1, 1
            )  # NOTE that the index should start at one
            bnd = gpd.GeoDataFrame(index=index, geometry=pnts, crs=sf_model.crs)

            # Add time series
            time = pd.date_range(
                start=utils.parse_datetime(sf_model.config["tstart"]),
                end=utils.parse_datetime(sf_model.config["tstop"]),
                periods=len(waterlevels),
            )
            waterlevels.index = time
            sf_model.setup_waterlevel_forcing(timeseries=waterlevels, locations=bnd)
            sf_model.write()  # write all

        else:
            swl_pos = np.nonzero(swl["SWL"].values)[0]
            sf_model.setup_config(
                **{
                    "tref": "20000101 000000",
                    "tstart": "20000101 000000",
                    "tstop": f"20000101 {len(swl_pos):02}0000",
                    "dtout": "100",
                }
            )

            wave_setup = wave_setup.iloc[swl_pos]
            wave_ig = wave_ig.iloc[swl_pos]
            swl = swl.iloc[swl_pos]
            t_10s = np.arange(0, len(swl_pos) * 3600, 10)
            transects = np.arange(60, 110, 1)
            waterlevels = pd.DataFrame()
            hours = np.arange(0, (len(swl_pos)) * 3600, 3600)

            for i in transects:
                setup = np.array(wave_setup[f"{i}.0"].values)
                ig = np.array(wave_ig[f"{i}.0"].values)
                wave_setup_interp = np.interp(t_10s, hours, setup)
                swl_interp = np.interp(t_10s, hours, swl["SWL"].values)
                ig_amplitude_interp = np.interp(t_10s, hours, ig)
                ig_interp = ig_amplitude_interp * np.sin(2 * np.pi * t_10s / 200)
                _result = np.column_stack(
                    (t_10s, wave_setup_interp, swl_interp, ig_interp)
                )
                twl = wave_setup_interp + swl_interp + ig_interp
                waterlevels[f"{i}.0"] = twl

            waterlevels.columns = range(1, len(waterlevels.columns) + 1)
            # Loc points
            x = self.location_points.x.values
            y = self.location_points.y.values

            # Add to Geopandas dataframe as needed by HydroMT
            pnts = gpd.points_from_xy(x, y)
            index = np.arange(
                1, len(x) + 1, 1
            )  # NOTE that the index should start at one
            bnd = gpd.GeoDataFrame(index=index, geometry=pnts, crs=sf_model.crs)

            # Add time series
            time = pd.date_range(
                start=utils.parse_datetime(sf_model.config["tstart"]),
                end=utils.parse_datetime(sf_model.config["tstop"]),
                periods=len(waterlevels),
            )
            waterlevels.index = time
            sf_model.setup_waterlevel_forcing(timeseries=waterlevels, locations=bnd)
            sf_model.write()  # write all

    def build_cases(
        self,
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
    sfincs_wrapper.build_cases()
    # List available launchers
    print(sfincs_wrapper.list_available_launchers())
    # Run the model
    sfincs_wrapper.run_cases(launcher="docker", parallel=True)

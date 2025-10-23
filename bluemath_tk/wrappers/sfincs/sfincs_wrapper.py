import os
import os.path as op
from typing import List

import numpy as np
import pandas as pd
from hydromt_sfincs import SfincsModel

from .._base_wrappers import BaseModelWrapper


class SfincsModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SFINCS model.

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {}

    available_launchers = {
        "docker": "docker run --rm -v .:/case_dir -w /case_dir deltares/sfincs-cpu",
        "cluster": "launchSfincs.sh",
    }

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SFINCS model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )

    def setup_dem(self, sf: SfincsModel, case_context: dict) -> List[dict]:
        """
        Setup the DEM for the SFINCS model.
        """

        ds = sf.data_catalog.get_rasterdataset(
            data_like=case_context.get("path_to_dem_tif"),
            variables=[case_context.get("dem_tif_var_name")],
            geom=sf.region,
            meta={"version": "1"},
        )

        datasets_dep = [{"da": ds}]  # "zmin": 0.001

        sf.setup_dep(datasets_dep=datasets_dep)

        return datasets_dep

    def setup_friction(self, sf: SfincsModel, case_context: dict) -> List[dict]:
        """
        Setup the friction for the SFINCS model.
        """

        dataset_rgh = sf.data_catalog.get_rasterdataset(
            data_like=case_context.get("path_to_rgh_tif"),
        )

        datasets_rgh = [{"manning": dataset_rgh}]

        sf.setup_manning_roughness(
            datasets_rgh=datasets_rgh,
            rgh_lev_land=0,  # the minimum elevation of the land
        )

        return datasets_rgh

    def setup_infiltration(self, sf: SfincsModel, case_context: dict) -> List[dict]:
        """
        Setup the infiltration for the SFINCS model.
        """

        p_infiltration = case_context.get("path_to_inf_tif")
        ant_moisture = "avg"

        dataset_inf = sf.data_catalog.get_rasterdataset(p_infiltration)

        dataset_inf.name = "cn_{0}".format(ant_moisture)

        sf.setup_cn_infiltration(
            dataset_inf.compute(), antecedent_moisture="{0}".format(ant_moisture)
        )

        return dataset_inf

    def setup_outflow(self, sf: SfincsModel, case_context: dict) -> None:
        """
        Setup the outflow for the SFINCS model.
        """

        gdf = sf.data_catalog.get_geodataframe(
            data_like=case_context.get("path_to_outflow_shp")
        )

        sf.setup_mask_bounds(btype="outflow", include_mask=gdf, reset_bounds=True)

    def setup_waterlevel_mask(self, sf: SfincsModel, case_context: dict) -> None:
        """
        Setup the waterlevel mask for the SFINCS model.
        """

        gdf = sf.data_catalog.get_geodataframe(
            data_like=case_context.get("path_to_waterlevel_shp")
        )

        sf.setup_mask_bounds(btype="waterlevel", include_mask=gdf, reset_bounds=True)

    def set_ctimes(self, case_context: dict) -> str:
        """
        Determine the start time (TSTART) end time (TSTOP) for the simulation based on
        precipitation and water level forcing, add 1 hour,
        and return it formatted as 'YYYYMMDD HHMMSS'.
        """

        precip_forcing = case_context.get("precipitation_forcing")
        waterlevel_forcing = case_context.get("waterlevel_forcing")

        tstart_times, tstop_times = [], []

        if precip_forcing is not None:
            tstart_precip = precip_forcing["time"].values[0]
            tstop_precip = precip_forcing["time"].values[-1]

            tstart_times.append(tstart_precip)
            tstop_times.append(tstop_precip)

        if waterlevel_forcing is not None:
            tstart_waterlevel = waterlevel_forcing.index.values[0]
            tstop_waterlevel = waterlevel_forcing.index.values[-1]

            tstart_times.append(tstart_waterlevel)
            tstop_times.append(tstop_waterlevel)

        if not tstart_times:
            raise ValueError("No forcing data found to determine TSTART.")

        if not tstop_times:
            raise ValueError("No forcing data found to determine TSTOP.")

        # Get the latest time and add one hour
        tstop_max = max(tstop_times)
        tstop_plus_1h = tstop_max + np.timedelta64(1, "h")

        tstart_min = min(tstart_times)

        # Convert to pandas.Timestamp (works with numpy.datetime64) then format
        formatted_tstop = pd.to_datetime(tstop_plus_1h).strftime("%Y%m%d %H%M%S")
        formatted_tstart = pd.to_datetime(tstart_min).strftime("%Y%m%d %H%M%S")

        return formatted_tstart, formatted_tstop

    def build_template_case(self) -> None:
        """
        Build a base SFINCS model case used as a template for multiple simulations.
        This function sets up all the static components of the model that are
        common to all cases, such as grid definition, DEM, friction,
        infiltration, subgrid configuration, and boundary masks.
        """

        sf = SfincsModel(root=self.templates_dir, mode="w+")

        sf.setup_grid(
            x0=self.fixed_parameters["x0"],
            y0=self.fixed_parameters["y0"],
            dx=self.fixed_parameters["dx"],
            dy=self.fixed_parameters["dy"],
            nmax=self.fixed_parameters["nmax"],
            mmax=self.fixed_parameters["mmax"],
            rotation=self.fixed_parameters["rotation"],
            epsg=self.fixed_parameters["epsg"],
        )

        datasets_dep = self.setup_dem(sf=sf, case_context=self.fixed_parameters)

        sf.setup_mask_active(
            mask=self.fixed_parameters.get("path_to_mask"),
        )

        datasets_rgh = self.setup_friction(sf, case_context=self.fixed_parameters)

        _dataset_inf = self.setup_infiltration(sf, case_context=self.fixed_parameters)

        sf.setup_subgrid(
            datasets_dep=datasets_dep,
            datasets_rgh=datasets_rgh,
            nr_subgrid_pixels=self.fixed_parameters.get("nr_subgrid_pixels"),
            write_dep_tif=True,
            write_man_tif=False,
        )

        self.setup_outflow(sf=sf, case_context=self.fixed_parameters)

        self.setup_waterlevel_mask(sf=sf, case_context=self.fixed_parameters)

        _ = sf.plot_basemap(bmap="sat", zoomlevel=12)

        sf.write()

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """
        Build the base SFINCS model. This includes setting up the grid,
        depth, friction, mask, outflow and waterlevel mask. It also
        applies the precipitation and waterlevel forcing if specified.
        """

        sf = SfincsModel(root=case_dir, mode="w+")

        sf.setup_grid(
            x0=case_context["x0"],
            y0=case_context["y0"],
            dx=case_context["dx"],
            dy=case_context["dy"],
            nmax=case_context["nmax"],
            mmax=case_context["mmax"],
            rotation=case_context["rotation"],
            epsg=case_context["epsg"],
        )
        tstart, tstop = self.set_ctimes(case_context=case_context)

        sf.config["tstop"] = tstop
        sf.config["tstart"] = tstart
        sf.config["dtout"] = 60
        sf.config["storemeteo"] = 1

        if case_context.get("quickly_waterlevel_forcing") is not None:
            """
            NOTE - There is not a specific Python function yet, 
            but one could call the setup_waterlevel_forcing function 
            twice with saving the files in between and changing their names
            """

            sf.setup_waterlevel_forcing(
                timeseries=case_context.get("quickly_waterlevel_forcing"),
                locations=case_context.get("gdf_boundary_points"),
            )
            sf.write_forcing()
            os.rename(op.join(case_dir, "sfincs.bzs"), op.join(case_dir, "sfincs.bzi"))

        if case_context.get("slowly_waterlevel_forcing") is not None:
            sf.setup_waterlevel_forcing(
                timeseries=case_context.get("slowly_waterlevel_forcing"),
                locations=case_context.get("gdf_boundary_points"),
            )
            sf.write_forcing()

        if case_context.get("precipitation_forcing") is not None:
            sf.setup_precip_forcing_from_grid(
                precip=case_context.get("precipitation_forcing"), aggregate=False
            )

        if case_context.get("gdf_crs") is not None:
            sf.setup_observation_lines(
                locations=case_context.get("gdf_crs"), merge=False
            )

        if case_context.get("gdf_obs") is not None:
            sf.setup_observation_points(locations=case_context.get("gdf_obs"))

        # if case_context.get("precipitation_forcing") is not None and case_context.get("waterlevel_forcing") is not None:
        #    self.setup_rivers(sf)
        #    sf.setup_river_inflow(
        #        rivers=case_context.get("precipitation_forcing"), keep_rivers_geom=True
        #    )

        # sf.write_forcing()

        sf.write()

from typing import List

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
        Initialize the Delft3d model wrapper.
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


class SfincsAlbaModelWrapper(SfincsModelWrapper):
    """
    Wrapper for the SFINCS model.
    This class is a subclass of the SfincsModelWrapper class and is
    specifically designed for the Alba model.
    """

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

        datasets_dep = self.setup_dem(sf=sf, case_context=case_context)

        sf.setup_mask_active(
            mask=case_context.get("path_to_mask"),
        )

        # datasets_rgh = self.setup_friction(sf)

        # dataset_inf = self.setup_infiltration(sf)

        # sf.setup_subgrid(
        #     datasets_dep=datasets_dep,
        #     datasets_rgh=datasets_rgh,
        #     nr_subgrid_pixels=case_context.get("nr_subgrid_pixels"),
        #     write_dep_tif=True,
        #     write_man_tif=False,
        # )

        self.setup_outflow(sf=sf, case_context=case_context)

        self.setup_waterlevel_mask(sf=sf, case_context=case_context)

        sf.setup_waterlevel_forcing(
            timeseries=case_context.get("waterlevel_forcing"),
            locations=case_context.get("gdf_boundary_points"),
        )

        # sf.setup_precip_forcing_from_grid(
        #     precip=case_context.get("dataset_precipitation"), aggregate=False
        # )

        if case_context.get("gdf_crs") is not None:
            sf.setup_observation_lines(
                locations=case_context.get("gdf_crs"), merge=False
            )

        if case_context.get("gdf_obs") is not None:
            sf.setup_observation_points(locations=case_context.get("gdf_obs"))

        if case_context.get("precip") and case_context.get("waterlev"):
            # self.setup_rivers(sf)
            sf.setup_river_inflow(
                rivers=case_context.get("precip"), keep_rivers_geom=True
            )

        sf.write_forcing()

        # self.config_update(sf)

        sf.write()

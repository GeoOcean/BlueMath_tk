import os

import numpy as np

from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.datamining.mda import MDA
from bluemath_tk.waves.series import series_TMA, waves_dispersion
from bluemath_tk.wrappers.swash.swash_wrapper import SwashModelWrapper


def convert_seconds_to_hour_minutes_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return f"{str(int(hour)).zfill(2)}{str(int(minutes)).zfill(2)}{str(int(seconds)).zfill(2)}.000"


class VeggySwashModelWrapper(SwashModelWrapper):
    """
    Wrapper for the SWASH model with vegetation.
    """

    gamma: int = 2
    deltat: int = 1

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

        # Build the input waves
        waves_dict = {
            "H": case_context["Hs"],
            "T": np.sqrt(
                (case_context["Hs"] * 2 * np.pi)
                / (self.gravity * case_context["Hs_L0"])
            ),
            "gamma": self.gamma,
            "warmup": self.warmup,
            "deltat": self.deltat,
            "tendc": self.tendc,
        }
        waves = series_TMA(waves=waves_dict, depth=self.depth_array[0])
        # Save the waves to a file
        self.write_array_in_file(
            array=waves, filename=os.path.join(case_dir, "waves.bnd")
        )

        # Calculate computational parameters
        tendc = convert_seconds_to_hour_minutes_seconds(self.tendc + self.warmup)
        L1, _k1, _c1 = waves_dispersion(T=waves_dict["T"], h=1.0)
        _L, _k, c = waves_dispersion(T=waves_dict["T"], h=self.depth_array[0])
        # comp_dx = L1 / np.abs(self.depth[0]) # MAL: Hecho por JAvi y Valva
        dx = L1 / self.n_nodes_per_wavelength
        deltc = 0.5 * dx / (np.sqrt(self.gravity * self.depth_array[0]) + np.abs(c))
        mxc = int(self.xlenc / dx)

        # Update the case context
        case_context["xlenc"] = self.xlenc
        case_context["mxc"] = mxc
        case_context["mxinp"] = self.mxinp
        case_context["dxinp"] = self.dxinp
        case_context["deltc"] = deltc
        case_context["tendc"] = tendc


class ChySwashModelWrapper(SwashModelWrapper):
    """
    Wrapper for the SWASH model with vegetation.
    """

    gamma: int = 2
    deltat: int = 1
    default_Cf = 0.002

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

        # Build the input waves
        waves_dict = {
            "H": case_context["Hs"],
            "T": np.sqrt(
                (case_context["Hs"] * 2 * np.pi)
                / (self.gravity * case_context["Hs_L0"])
            ),
            "gamma": self.gamma,
            "warmup": self.warmup,
            "deltat": self.deltat,
            "tendc": self.tendc,
        }
        waves = series_TMA(waves=waves_dict, depth=self.depth_array[0])
        # Save the waves to a file
        self.write_array_in_file(
            array=waves, filename=os.path.join(case_dir, "waves.bnd")
        )

        # Build the input friction file
        friction = np.ones((len(self.depth_array))) * self.default_Cf
        # friction[int(self.Cf_init):int(self.Cf_end)] = case_context["Cf"]
        np.savetxt(os.path.join(case_dir, "friction.txt"), friction, fmt="%.6f")

        # Calculate computational parameters
        tendc = convert_seconds_to_hour_minutes_seconds(self.tendc + self.warmup)
        L1, _k1, _c1 = waves_dispersion(T=waves_dict["T"], h=1.0)
        _L, _k, c = waves_dispersion(T=waves_dict["T"], h=self.depth_array[0])
        # comp_dx = L1 / np.abs(self.depth[0]) # MAL: Hecho por JAvi y Valva
        dx = L1 / self.n_nodes_per_wavelength
        deltc = 0.5 * dx / (np.sqrt(self.gravity * self.depth_array[0]) + np.abs(c))
        mxc = int(self.xlenc / dx)

        # Update the case context
        case_context["xlenc"] = self.xlenc
        case_context["mxc"] = mxc
        case_context["mxinp"] = self.mxinp
        case_context["dxinp"] = self.dxinp
        case_context["deltc"] = deltc
        case_context["tendc"] = tendc


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swash/templates/"
    )
    # Get 5 cases using LHS and MDA
    lhs = LHS(num_dimensions=3)
    lhs_data = lhs.generate(
        dimensions_names=["Hs", "Hs_L0", "vegetation_height"],
        lower_bounds=[0.5, 0.0, 0.0],
        upper_bounds=[3.0, 0.05, 1.5],
        num_samples=500,
    )
    mda = MDA(num_centers=5)
    mda.logger.setLevel("DEBUG")
    mda.fit(data=lhs_data)
    model_parameters = mda.centroids.to_dict(orient="list")
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swash/"
    # Create an instance of the SWASH model wrapper
    swash_wrapper = VeggySwashModelWrapper(
        templates_dir=templates_dir,
        model_parameters=model_parameters,
        output_dir=output_dir,
        depth_array=np.loadtxt(
            "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swash/templates/depth.bot"
        ),
        dxinp=1.0,
        n_nodes_per_wavelength=60,
        tendc=7200,
        warmup=7200 * 0.15,
    )
    # Build the input files
    swash_wrapper.build_cases(mode="one_by_one")
    # List available launchers
    print(swash_wrapper.list_available_launchers())
    # Run the model using docker_serial launcher
    # swash_wrapper.run_cases(launcher="docker_serial", num_workers=5)
    # Post-process the output files
    # postprocessed_data = swash_wrapper.postprocess_cases(num_workers=1)
    # print(postprocessed_data)

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper
from bluemath_tk.topo_bathy.profiles import linear
from bluemath_tk.waves.series import series_TMA
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.datamining.mda import MDA


class SwashModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWASH model.
    https://swash.sourceforge.io/online_doc/swashuse/swashuse.html#input-and-output-files
    """

    default_parameters = {
        "vegetation_height": float,
    }

    def __init__(
        self,
        templates_dir: str,
        templates_name: dict,
        model_parameters: dict,
        output_dir: str,
    ):
        super().__init__(
            templates_dir=templates_dir,
            templates_name=templates_name,
            model_parameters=model_parameters,
            output_dir=output_dir,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(self.__class__.__name__)
        self._swash_exec: str = None

    @property
    def swash_exec(self) -> str:
        return self._swash_exec

    def set_swash_exec(self, swash_exec: str) -> None:
        self._swash_exec = swash_exec

    @staticmethod
    def _read_tabfile(file_path: str) -> pd.DataFrame:
        """
        Read a tab file and return a pandas DataFrame.
        This function is used to read the output files of SWASH.

        Parameters
        ----------
        file_path : str
            The file path.

        Returns
        -------
        pd.DataFrame
            The pandas DataFrame.
        """

        f = open(file_path, "r")
        lines = f.readlines()
        # read head colums (variables names)
        names = lines[4].split()
        names = names[1:]  # Eliminate '%'
        # read data rows
        values = pd.Series(lines[7:]).str.split(expand=True).values.astype(float)
        df = pd.DataFrame(values, columns=names)
        f.close()

        return df

    def _convert_output_tabs_to_nc(
        self, case_id: int, output_path: str, run_path: str
    ) -> xr.Dataset:
        """
        Convert tab files to a netCDF file.

        Parameters
        ----------
        output_path : str
            The output path.
        run_path : str + MDA
            The run path.

        Returns
        -------
        xr.Dataset
            The xarray Dataset.
        """

        df_output = self._read_tabfile(file_path=output_path)
        df_output.set_index(
            ["Xp", "Yp", "Tsec"], inplace=True
        )  # set index to Xp, Yp and Tsec
        ds_ouput = df_output.to_xarray()

        df_run = self._read_tabfile(file_path=run_path)
        df_run.set_index(["Tsec"], inplace=True)
        ds_run = df_run.to_xarray()

        # merge output files to one xarray.Dataset
        ds = xr.merge([ds_ouput, ds_run], compat="no_conflicts")

        # assign correct coordinate case_id
        ds.coords["case_id"] = case_id

        return ds

    def run_model(self, case_dir: str, log_file: str = "swash_exec.log") -> None:
        """
        Run the SWASH model for the specified case.

        Parameters
        ----------
        case_dir : str
            The case directory.
        log_file : str, optional
            The log file name. Default is "swash_exec.log".

        Raises
        ------
        ValueError
            If the SWASH executable was not set.
        """

        if not self.swash_exec:
            raise ValueError("The SWASH executable was not set.")
        # check if windows OS
        is_win = sys.platform.startswith("win")
        if is_win:
            cmd = "cd {0} && {1} input".format(case_dir, self.swan_exec)
        else:
            cmd = "cd {0} && {1} -input input.sws".format(case_dir, self.swan_exec)
        # redirect output
        cmd += f" 2>&1 > {log_file}"
        # execute command
        self._exec_bash_commands(str_cmd=cmd)


class VeggySwashModelWrapper(SwashModelWrapper):
    """
    Wrapper for the SWASH model with vegetation.
    """

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
        depth: np.ndarray = None,
        plants: np.ndarray = None,
    ) -> None:
        if depth is not None:
            # Save the depth to a file
            self.write_array_in_file(
                array=depth, filename=os.path.join(case_dir, "depth.txt")
            )
        if plants is not None:
            # Save the plants to a file
            self.write_array_in_file(
                array=plants, filename=os.path.join(case_dir, "plants.txt")
            )
        # Build the input waves
        waves_dict = {
            "H": case_context["Hs"],
            "T": np.sqrt(
                (case_context["Hs"] * 2 * np.pi) / (9.806 * case_context["Hs_L0"])
            ),
            "gamma": 2,
            "warmup": 180,
            "deltat": 1,
            "tendc": 1800,
        }
        waves = series_TMA(waves=waves_dict, depth=depth[0])
        # Save the waves to a file
        self.write_array_in_file(
            array=waves, filename=os.path.join(case_dir, "waves.bnd")
        )

    def build_cases(
        self,
        mode: str = "all_combinations",
        depth: np.ndarray = None,
        plants: np.ndarray = None,
    ) -> None:
        super().build_cases(mode=mode)
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            self.build_case(
                case_context=case_context,
                case_dir=case_dir,
                depth=depth,
                plants=plants,
            )


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swash/templates/"
    )
    templates_name = ["input.sws"]
    # Get 5 cases using LHS and MDA
    lhs = LHS(num_dimensions=4)
    lhs_data = lhs.generate(
        dimensions_names=["Hs", "Hs_L0", "vegetation_height"],
        lower_bounds=[0.5, 0.0, 0.0],
        upper_bounds=[3.0, 0.05, 1.5],
        num_samples=500,
    )
    mda = MDA(num_centers=5)
    mda.fit(data=lhs_data)
    model_parameters = mda.centroids.to_dict(orient="list")
    output_dir = "C:/Users/UsuarioUC/Documents/BlueMath_tk/tests_data/swash"
    # Create the depth
    """
    dx:      bathymetry mesh resolution at x axes (m)
    h0:      offshore depth (m)
    bCrest:  beach heigh (m)
    m:       profile slope
    Wfore:   flume length before slope toe (m)
    """
    linear_depth = linear(dx=100, h0=10, bCrest=2, m=1, Wfore=10)
    # Create the plants
    plants = np.zeros(linear_depth.size)
    plants[linear_depth < 1] = 1.0
    # Create an instance of the SWASH model wrapper
    swan_model = VeggySwashModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="all_combinations", depth=linear_depth)

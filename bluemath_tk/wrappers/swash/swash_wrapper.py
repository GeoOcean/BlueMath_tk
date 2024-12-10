import sys
import pandas as pd
import xarray as xr
from .._base_wrappers import BaseModelWrapper


class SwashModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWASH model.
    https://swash.sourceforge.io/online_doc/swashuse/swashuse.html#input-and-output-files

    Attributes
    ----------
    swash_exec : str
        The SWASH executable path.
    default_parameters : dict
        The default parameters type for the model.

    Methods
    -------
    set_swash_exec(swash_exec: str) -> None
        Set the SWASH executable path.
    _read_tabfile(file_path: str) -> pd.DataFrame
        Read a tab file and return a pandas DataFrame.
    _convert_output_tabs_to_nc(case_id: int, output_path: str, run_path: str) -> xr.Dataset
        Convert output tabs files to a netCDF file.
    run_model(case_dir: str, log_file: str = "swash_exec.log") -> None
        Run the SWASH model for the specified case.
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
    ) -> None:
        """
        Initialize the SWASH model wrapper.

        Parameters
        ----------
        templates_dir : str
            The directory where the templates are stored.
        templates_name : list
            The names of the templates.
        model_parameters : dict
            The parameters to be used in the templates.
        output_dir : str
            The directory where the output files will be saved.
        """

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
            cmd = "cd {0} && {1} input".format(case_dir, self.swash_exec)
        else:
            cmd = "cd {0} && {1} -input input.sws".format(case_dir, self.swash_exec)
        # redirect output
        cmd += f" 2>&1 > {log_file}"
        # execute command
        self._exec_bash_commands(str_cmd=cmd)

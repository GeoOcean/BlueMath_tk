import os
import sys
import numpy as np
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html
    """

    default_parameters = {
        "hs": float,
        "tp": float,
        "dir": float,
        "spr": float,
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
        self._swan_exec: str = None

    @property
    def swan_exec(self) -> str:
        return self._swan_exec

    def set_swan_exec(self, swan_exec: str) -> None:
        self._swan_exec = swan_exec

    def build_case_hyswan(self, case_context: dict, case_dir: str) -> None:
        if case_context.get("wind"):
            wind = np.random.rand(10, 10)
            self.write_array_in_file(
                array=wind, filename=os.path.join(case_dir, "wind_file.dat")
            )

    def build_cases(
        self, mode: str = "all_combinations", swan_type: str = "HySwan"
    ) -> None:
        super().build_cases(mode=mode)
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            if swan_type == "HySwan":
                self.build_case_hyswan(case_context=case_context, case_dir=case_dir)

    def run_model(self, case_dir: str, log_file: str = "swan_exec.log") -> None:
        """
        Run the SWAN model for the specified case.

        Parameters
        ----------
        case_dir : str
            The case directory.
        log_file : str, optional
            The log file name. Default is "swan_exec.log".

        Raises
        ------
        ValueError
            If the SWAN executable was not set.
        """

        if not self.swan_exec:
            raise ValueError("The SWAN executable was not set.")
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

    def run_cases(self) -> None:
        return super().run_cases()


class MySwanModelWrapper(SwanModelWrapper):
    def build_cases(
        self,
        mode: str = "all_combinations",
    ) -> None:
        # Call the base class method to retain the original functionality
        super().build_cases(mode=mode)


# Usage example
if __name__ == "__main__":
    # Define the input parameters
    templates_dir = (
        "/home/tausiaj/GitHub-GeoOcean/BlueMath/bluemath_tk/wrappers/swan/templates/"
    )
    templates_name = ["struc_input.swn"]
    hs = np.linspace(0, 6, 5)
    tp = np.linspace(0, 20, 5)
    dir = np.linspace(0, 360, 5)
    spr = np.linspace(0, 50, 5)
    wind = np.random.choice([False, True], size=5)
    model_parameters = {
        "hs": hs,
        "tp": tp,
        "dir": dir,
        "spr": spr,
        "wind": wind,
    }
    output_dir = "/home/tausiaj/GitHub-GeoOcean/BlueMath/test_cases/swan/"
    # Instantiate the model
    swan_model = SwanModelWrapper(
        templates_dir=templates_dir,
        templates_name=templates_name,
        model_parameters=model_parameters,
        output_dir=output_dir,
    )
    # Build the input files
    swan_model.build_cases(mode="one_by_one", swan_type="HySwan")

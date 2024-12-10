import sys
from .._base_wrappers import BaseModelWrapper


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html

    Attributes
    ----------
    swan_exec : str
        The SWAN executable path.
    default_parameters : dict
        The default parameters type for the model.

    Methods
    -------
    set_swan_exec(swan_exec: str) -> None
        Set the SWAN executable path.
    run_model(case_dir: str, log_file: str = "swan_exec.log") -> None
        Run the SWAN model for the specified case.
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
        """
        Initialize the SWAN model wrapper.

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
        self._swan_exec: str = None

    @property
    def swan_exec(self) -> str:
        return self._swan_exec

    def set_swan_exec(self, swan_exec: str) -> None:
        self._swan_exec = swan_exec

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

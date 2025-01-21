import os
import copy
import itertools
from typing import List, Union
import subprocess
import numpy as np
import xarray as xr
from jinja2 import Environment, FileSystemLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core.models import BlueMathModel
from ._utils_wrappers import write_array_in_file, copy_files


class BaseModelWrapper(BlueMathModel):
    """
    Base class for numerical models wrappers.

    Attributes
    ----------
    templates_dir : str
        The directory where the templates are stored.
    model_parameters : dict
        The parameters to be used in the templates.
    output_dir : str
        The directory where the output files will be saved.
    env : Environment
        The Jinja2 environment.
    templates_name : List[str]
        The names of the templates.
    cases_dirs : List[str]
        The list with cases directories.
    cases_context : List[dict]
        The list with cases context.

    Methods
    -------
    _check_parameters_type -> None
        Check if the parameters have the correct type.
    _exec_bash_commands -> None
        Execute bash commands.
    list_available_launchers -> dict
        List the available launchers.
    set_cases_dirs_from_output_dir -> None
        Set the cases directories from the output directory.
    write_array_in_file -> None
        Write an array in a file.
    copy_files -> None
        Copy file(s) from source to destination.
    render_file_from_template -> None
        Render a file from a template.
    create_cases_context_one_by_one -> List[dict]
        Create an array of dictionaries with the combinations of values from the
        input dictionary, one by one.
    create_cases_context_all_combinations -> List[dict]
        Create an array of dictionaries with each possible combination of values
        from the input dictionary.
    build_cases -> None
        Create the cases folders and render the input files.
    run_case -> None
        Run the case based on the launcher specified.
    run_cases -> None
        Run the cases based on the launcher specified.
        Parallel execution is optional.
        Cases to run can be specified.
    run_cases_bulk -> None
        Run the cases based on the launcher specified.
    postprocess_case -> None
        Postprocess the model output for a specific case.
    join_postprocessed_files -> xr.Dataset
        Join the postprocessed files.
    postprocess_cases -> Union[xr.Dataset, List[xr.Dataset]]
        Postprocess the model output.
    """

    def __init__(
        self,
        templates_dir: str,
        model_parameters: dict,
        output_dir: str,
        templates_name: List[str] = "all",
        default_parameters: dict = None,
    ) -> None:
        """
        Initialize the BaseModelWrapper.
        """

        super().__init__()
        if default_parameters is not None:
            self._check_parameters_type(
                default_parameters=default_parameters, model_parameters=model_parameters
            )
        self.templates_dir = templates_dir
        self.model_parameters = model_parameters
        self.output_dir = output_dir
        self._env = Environment(loader=FileSystemLoader(self.templates_dir))
        if templates_name == "all":
            self.logger.warning(
                f"Templates name is 'all', so all templates in {self.templates_dir} will be used."
            )
            self.templates_name = self.env.list_templates()
            self.logger.info(f"Templates names: {self.templates_name}")
        else:
            self.templates_name = templates_name
        self.cases_dirs: List[str] = []
        self.cases_context: List[dict] = []

    @property
    def env(self) -> Environment:
        return self._env

    def _check_parameters_type(
        self, default_parameters: dict, model_parameters: dict
    ) -> None:
        """
        Check if the parameters have the correct type.
        This function is called in the __init__ method of the BaseModelWrapper,
        but default_parameters are defined in the child classes.
        This way, child classes can define default types for parameters.

        Parameters
        ----------
        default_parameters : dict
            The default parameters type for the model.
        model_parameters : dict
            The parameters to be used in the templates.

        Raises
        ------
        ValueError
            If a parameter has the wrong type.
        """

        for model_param, param_value in model_parameters.items():
            if model_param not in default_parameters:
                self.logger.warning(
                    f"Parameter {model_param} is not in the default_parameters"
                )
            else:
                if isinstance(param_value, (list, np.ndarray)) and all(
                    isinstance(item, default_parameters[model_param])
                    for item in param_value
                ):
                    self.logger.info(
                        f"Parameter {model_param} has the correct type: {default_parameters[model_param]}"
                    )
                else:
                    raise ValueError(
                        f"Parameter {model_param} has the wrong type: {default_parameters[model_param]}"
                    )

    @staticmethod
    def _exec_bash_commands(
        str_cmd: str, out_file: str = None, err_file: str = None
    ) -> None:
        """
        Execute bash commands.

        Parameters
        ----------
        str_cmd : str
            The bash command.
        out_file : str, optional
            The name of the output file. If None, the output will be printed in the terminal.
            Default is None.
        err_file : str, optional
            The name of the error file. If None, the error will be printed in the terminal.
            Default is None.
        """

        _stdout = None
        _stderr = None

        if out_file:
            _stdout = open(out_file, "w")
        if err_file:
            _stderr = open(err_file, "w")

        s = subprocess.Popen(str_cmd, shell=True, stdout=_stdout, stderr=_stderr)
        s.wait()

        if out_file:
            _stdout.flush()
            _stdout.close()
        if err_file:
            _stderr.flush()
            _stderr.close()

    def list_available_launchers(self) -> dict:
        """
        List the available launchers.

        Returns
        -------
        dict
            A list with the available launchers.
        """

        if hasattr(self, "available_launchers"):
            return self.available_launchers
        else:
            raise AttributeError("The attribute available_launchers is not defined.")

    def set_cases_dirs_from_output_dir(self) -> None:
        """
        Set the cases directories from the output directory.
        """

        if self.cases_dirs:
            self.logger.warning("Cases directories already set... resetting.")

        self.cases_dirs = sorted(
            [
                os.path.join(self.output_dir, case_dir)
                for case_dir in os.listdir(self.output_dir)
            ]
        )

        self.logger.info(f"Cases directories set from {self.output_dir}.")

    def write_array_in_file(self, array: np.ndarray, filename: str) -> None:
        """
        Write an array in a file.

        Parameters
        ----------
        array : np.ndarray
            The array to be written. Can be 1D or 2D.
        filename : str
            The name of the file.
        """

        write_array_in_file(array=array, filename=filename)

    def copy_files(self, src: str, dst: str) -> None:
        """
        Copy file(s) from source to destination.

        Parameters
        ----------
        src : str
            The source file.
        dst : str
            The destination file.
        """

        copy_files(src=src, dst=dst)

    def render_file_from_template(
        self, template_name: str, context: dict, output_filename: str = None
    ) -> None:
        """
        Render a file from a template.

        Parameters
        ----------
        template_name : str
            The name of the template file.
        context : dict
            The context to be used in the template.
        output_filename : str, optional
            The name of the output file. If None, it will be saved in the output
            directory with the same name as the template.
            Default is None.
        """

        template = self.env.get_template(name=template_name)
        rendered_content = template.render(context)
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, template_name)
        with open(output_filename, "w") as f:
            f.write(rendered_content)

    def create_cases_context_one_by_one(self) -> List[dict]:
        """
        Create an array of dictionaries with the combinations of values from the
        input dictionary, one by one.

        Returns
        -------
        List[dict]
            A list of dictionaries, each representing a unique combination of
            parameter values.
        """

        num_cases = len(next(iter(self.model_parameters.values())))
        array_of_contexts = []
        for param, values in self.model_parameters.items():
            if len(values) != num_cases:
                raise ValueError(
                    f"All parameters must have the same number of values in one_by_one mode, check {param}"
                )

        for case_num in range(num_cases):
            case_context = {
                param: values[case_num]
                for param, values in self.model_parameters.items()
            }
            array_of_contexts.append(case_context)

        return array_of_contexts

    def create_cases_context_all_combinations(self) -> List[dict]:
        """
        Create an array of dictionaries with each possible combination of values
        from the input dictionary.

        Returns
        -------
        List[dict]
            A list of dictionaries, each representing a unique combination of
            parameter values.
        """

        keys = self.model_parameters.keys()
        values = self.model_parameters.values()
        combinations = itertools.product(*values)

        array_of_contexts = [
            dict(zip(keys, combination)) for combination in combinations
        ]

        return array_of_contexts

    def build_cases(self, mode: str = "one_by_one") -> None:
        """
        Create the cases folders and render the input files.

        Parameters
        ----------
        mode : str, optional
            The mode to create the cases. Can be "all_combinations" or "one_by_one".
            Default is "one_by_one".
        """

        if mode == "all_combinations":
            self.cases_context = self.create_cases_context_all_combinations()
        elif mode == "one_by_one":
            self.cases_context = self.create_cases_context_one_by_one()
        else:
            raise ValueError(f"Invalid mode to create cases: {mode}")
        for case_num, case_context in enumerate(self.cases_context):
            case_context["case_num"] = case_num
            case_dir = os.path.join(self.output_dir, f"{case_num:04}")
            self.cases_dirs.append(case_dir)
            os.makedirs(case_dir, exist_ok=True)
            for template_name in self.templates_name:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=os.path.join(case_dir, template_name),
                )
        self.logger.info(
            f"{len(self.cases_dirs)} cases created in {mode} mode and saved in {self.output_dir}"
        )

    def run_case(
        self,
        case_dir: str,
        launcher: str,
        ouput_log_file: str = "wrapper_out.log",
        error_log_file: str = "wrapper_error.log",
    ) -> None:
        """
        Run the case based on the launcher specified.

        Parameters
        ----------
        case_dir : str
            The case directory.
        launcher : str
            The launcher to run the case.
        ouput_log_file : str, optional
            The name of the output log file. Default is "wrapper_out.log".
        error_log_file : str, optional
            The name of the error log file. Default is "wrapper_error.log".
        """

        # Get launcher command from the available launchers
        launcher = self.list_available_launchers().get(launcher, launcher)

        # Run the case in the case directory
        self.logger.info(f"Running case in {case_dir} with launcher={launcher}.")
        os.chdir(case_dir)
        self._exec_bash_commands(
            str_cmd=launcher,
            out_file=ouput_log_file,
            err_file=error_log_file,
        )

    def run_cases(
        self,
        launcher: str,
        parallel: bool = False,
        cases_to_run: List[int] = None,
    ) -> None:
        """
        Run the cases based on the launcher specified.
        Parallel execution is optional.
        Cases to run can be specified.

        Parameters
        ----------
        launcher : str
            The launcher to run the cases.
        parallel : bool, optional
            If True, the cases will be run in parallel. Default is False.
        cases_to_run : List[int], optional
            The list with the cases to run. Default is None.
        """

        # Get launcher command from the available launchers
        launcher = self.list_available_launchers().get(launcher, launcher)

        if cases_to_run is not None:
            self.logger.warning(
                f"Cases to run was specified, so just {cases_to_run} will be run."
            )
            cases_dir_to_run = [self.cases_dirs[case] for case in cases_to_run]
        else:
            cases_dir_to_run = copy.deepcopy(self.cases_dirs)

        if parallel:
            num_threads = self.get_num_processors_available()
            self.logger.debug(
                f"Running cases in parallel with launcher={launcher}. Number of threads: {num_threads}."
            )
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                future_to_case = {
                    executor.submit(self.run_case, case_dir, launcher): case_dir
                    for case_dir in cases_dir_to_run
                }
                for future in as_completed(future_to_case):
                    case_dir = future_to_case[future]
                    try:
                        future.result()
                    except Exception as exc:
                        self.logger.error(
                            f"Job for {case_dir} generated an exception: {exc}."
                        )
        else:
            self.logger.info(f"Running cases sequentially with launcher={launcher}.")
            for case_dir in cases_dir_to_run:
                try:
                    self.run_case(
                        case_dir=case_dir,
                        launcher=launcher,
                    )
                except Exception as exc:
                    self.logger.error(
                        f"Job for {case_dir} generated an exception: {exc}."
                    )

        if launcher == "docker" or "docker" in launcher:
            # TODO: ALWAYS remove ALL stopped containers after running all cases
            remove_stopped_containers_cmd = 'docker ps -a --filter "ancestor=tausiaj/swash-image:latest" -q | xargs docker rm'
            self._exec_bash_commands(str_cmd=remove_stopped_containers_cmd)

        self.logger.info("All cases ran successfully.")

    def run_cases_bulk(
        self,
        launcher: str,
    ) -> None:
        """
        Run the cases based on the launcher specified.

        Parameters
        ----------
        launcher : str
            The launcher to run the cases.
        """

        self.logger.info(f"Running cases with launcher={launcher}.")
        self._exec_bash_commands(str_cmd=launcher)

    def postprocess_case(self, case_num: int, case_dir: str) -> None:
        """
        Postprocess the model output.

        Parameters
        ----------
        case_num : int
            The case number.
        case_dir : str
            The case directory.
        """

        raise NotImplementedError("The method postprocess_case must be implemented.")

    def join_postprocessed_files(
        self, postprocessed_files: List[xr.Dataset]
    ) -> xr.Dataset:
        """
        Join the postprocessed files.

        Parameters
        ----------
        postprocessed_files : List[xr.Dataset]
            The list with the postprocessed files.
        """

        raise NotImplementedError(
            "The method join_postprocessed_files must be implemented."
        )

    def postprocess_cases(
        self, cases_to_postprocess: List[int] = None
    ) -> Union[xr.Dataset, List[xr.Dataset]]:
        """
        Postprocess the model output.

        Parameters
        ----------
        cases_to_postprocess : List[int], optional
            The list with the cases to postprocess. Default is None.

        Returns
        -------
        xr.Dataset or List[xr.Dataset]
            The postprocessed file or the list with the postprocessed files.
        """

        # TODO: Check if this option is necessary
        if os.path.exists(os.path.join(self.output_dir, "output_postprocessed.nc")):
            self.logger.warning(
                "Output postprocessed file already exists. Skipping postprocessing."
            )
            return xr.open_dataset(
                os.path.join(self.output_dir, "output_postprocessed.nc")
            )

        if not self.cases_dirs:
            self.logger.warning(
                "Cases directories are not set and will be searched from the output directory."
            )
            self.set_cases_dirs_from_output_dir()

        if cases_to_postprocess is not None:
            self.logger.warning(
                f"Cases to postprocess was specified, so just {cases_to_postprocess} will be postprocessed."
            )
            cases_dir_to_postprocess = [
                self.cases_dirs[case] for case in cases_to_postprocess
            ]
        else:
            cases_to_postprocess = list(range(len(self.cases_dirs)))
            cases_dir_to_postprocess = copy.deepcopy(self.cases_dirs)

        postprocessed_files = []
        for case_num, case_dir in zip(cases_to_postprocess, cases_dir_to_postprocess):
            self.logger.info(f"Postprocessing case {case_num} in {case_dir}.")
            postprocessed_file = self.postprocess_case(
                case_num=case_num, case_dir=case_dir
            )
            postprocessed_files.append(postprocessed_file)

        try:
            return self.join_postprocessed_files(
                postprocessed_files=postprocessed_files
            )
        except NotImplementedError as exc:
            self.logger.error(f"Error joining postprocessed files: {exc}")
            return postprocessed_files

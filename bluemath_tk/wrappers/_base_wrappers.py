import copy
import itertools
import os
import os.path as op
import subprocess
import threading
from abc import ABC
from queue import Queue
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr
from jinja2 import Environment, FileSystemLoader

from ..core.models import BlueMathModel
from ._utils_wrappers import copy_files, write_array_in_file

sbatch_file_example = """
#!/bin/bash
#SBATCH --job-name=your_job_name  # Job name
#SBATCH --partition=geocean       # Standard output and error log
#SBATCH --mem=4gb                 # Memory per node in GB (see also --mem-per-cpu)

case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
yourLauncher.sh --case-dir $case_dir > $case_dir/wrapper_out.log 2> $case_dir/wrapper_error.log
"""


class BaseModelWrapper(BlueMathModel, ABC):
    """
    Base class for numerical models wrappers.
    This is an abstract base class that cannot be instantiated directly.

    Attributes
    ----------
    templates_dir : str
        The directory where the templates are searched.
    metamodel_parameters : dict
        The parameters to be used for all cases.
    fixed_parameters : dict
        The fixed parameters for the cases.
    output_dir : str
        The directory where the output cases are saved.
    templates_name : List[str]
        The names of the templates to use.
    cases_context : List[dict]
        The list with cases context.
    cases_dirs : List[str]
        The list with cases directories.
    sbatch_file_example : str
        The example sbatch file.
    """

    sbatch_file_example = sbatch_file_example

    def __new__(cls, *args, **kwargs):
        if cls is BaseModelWrapper:
            raise TypeError(
                "BaseModelWrapper is an abstract base class and cannot be instantiated directly. "
                "For basic testing, you can use DummyModelWrapper from this same file."
            )
        return super().__new__(cls)

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: List[str] = "all",
        default_parameters: dict = None,
    ) -> None:
        """
        Initialize the BaseModelWrapper.

        Parameters
        ----------
        templates_dir : str
            The directory where the templates are searched.
            Both binary and text files are supported, for the case where the user
            needs to have a fixed binary file in all cases directories.
        metamodel_parameters : dict
            The parameters to be used for the different cases.
        fixed_parameters : dict
            The fixed parameters for the cases.
        output_dir : str
            The directory where the output cases are saved.
        templates_name : List[str], optional
            The names of the templates to use. Default is "all".
        default_parameters : dict, optional
            The default parameters for the cases. Default is None.

        Warnings
        --------
        All fixed_parameters and metamodel_parameters must be strings.
        """

        super().__init__()

        if default_parameters is not None:
            fixed_parameters = self._check_parameters_type(
                default_parameters=default_parameters,
                metamodel_parameters=metamodel_parameters,
                fixed_parameters=fixed_parameters,
            )

        self.templates_dir = templates_dir
        self.metamodel_parameters = metamodel_parameters
        self.fixed_parameters = fixed_parameters
        self.output_dir = output_dir
        self._env = Environment(loader=FileSystemLoader(self.templates_dir))
        if templates_name == "all":
            self.logger.info(
                f"Templates name is 'all', so all templates in {self.templates_dir} will be used."
            )
            self.templates_name = self.env.list_templates()
            self.logger.info(f"Templates names: {self.templates_name}")
        else:
            self.templates_name = templates_name
        self.cases_context: List[dict] = None
        self.cases_dirs: List[str] = None
        self.thread: threading.Thread = None
        self.status_queue: Queue = None

    @property
    def env(self) -> Environment:
        return self._env

    def _check_parameters_type(
        self,
        default_parameters: dict,
        metamodel_parameters: dict,
        fixed_parameters: dict,
    ) -> None:
        """
        Check if the parameters have the correct type.
        This functions checks if the parameters in the metamodel_parameters have the
        correct type according to the default_parameters.
        Then, it updates the fixed_parameters with the default_parameters values.

        Raises
        ------
        ValueError
            If a parameter has the wrong type.
        """

        for metamodel_param, param_value in metamodel_parameters.items():
            if metamodel_param not in default_parameters:
                self.logger.warning(
                    f"Parameter {metamodel_param} is not in the default_parameters"
                )
            else:
                if isinstance(param_value, (list, np.ndarray)) and all(
                    isinstance(item, default_parameters[metamodel_param]["type"])
                    for item in param_value
                ):
                    self.logger.info(
                        f"Parameter {metamodel_param} has the correct type: {default_parameters[metamodel_param]}"
                    )
                else:
                    raise ValueError(
                        f"Parameter {metamodel_param} has the wrong type: {default_parameters[metamodel_param]}"
                    )
        for default_param, param_info in default_parameters.items():
            if (
                default_param not in fixed_parameters
                and param_info.get("value") is not None
            ):
                fixed_parameters[default_param] = param_info.get("value")

        return fixed_parameters

    def _exec_bash_commands(
        self, str_cmd: str, out_file: str = None, err_file: str = None, cwd: str = None
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
        cwd : str, optional
            The current working directory. Default is None.
        """

        _stdout = None
        _stderr = None

        if out_file:
            _stdout = open(out_file, "w")
        if err_file:
            _stderr = open(err_file, "w")

        try:
            _s = subprocess.run(
                str_cmd,
                shell=True,
                stdout=_stdout,
                stderr=_stderr,
                cwd=cwd,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running command: {str_cmd}")
            self.logger.error(f"Error: {e}")

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

    def list_default_parameters(self) -> pd.DataFrame:
        """
        List the default parameters.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the default parameters.
        """

        if hasattr(self, "default_parameters"):
            return pd.DataFrame(self.default_parameters)
        else:
            raise AttributeError(
                "The attribute default_parameters is not defined in the child class."
            )

    def set_cases_dirs_from_output_dir(self) -> None:
        """
        Set the cases directories from the output directory.
        """

        raise NotImplementedError(
            "This method is deprecated. Use load_cases() method instead."
        )

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
            output_filename = op.join(self.output_dir, template_name)
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

        num_cases = len(next(iter(self.metamodel_parameters.values())))
        array_of_contexts = []
        for param, values in self.metamodel_parameters.items():
            if len(values) != num_cases:
                raise ValueError(
                    f"All parameters must have the same number of values in one_by_one mode, check {param}"
                )

        for case_num in range(num_cases):
            case_context = {
                param: values[case_num]
                for param, values in self.metamodel_parameters.items()
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

        keys = self.metamodel_parameters.keys()
        values = self.metamodel_parameters.values()
        combinations = itertools.product(*values)

        array_of_contexts = [
            dict(zip(keys, combination)) for combination in combinations
        ]

        return array_of_contexts

    def load_cases(
        self,
        mode: str = "one_by_one",
        cases_name_format: callable = lambda ctx: f"{ctx.get('case_num'):04}",
    ) -> None:
        """
        Create the cases context and directories.

        Parameters
        ----------
        mode : str, optional
            The mode to create the cases. Can be "all_combinations" or "one_by_one".
            Default is "one_by_one".
        cases_name_format : callable, optional
            The function to format the case name. Default is a lambda function
            that formats the case number with leading zeros.
        """

        if mode == "all_combinations":
            self.cases_context = self.create_cases_context_all_combinations()
        elif mode == "one_by_one":
            self.cases_context = self.create_cases_context_one_by_one()
        else:
            raise ValueError(f"Invalid mode to create cases: {mode}")

        # Set cases_dirs to empty list
        self.cases_dirs = []

        for case_num, case_context in enumerate(self.cases_context):
            case_context["case_num"] = case_num
            case_dir = op.join(self.output_dir, cases_name_format(case_context))
            self.cases_dirs.append(case_dir)
            os.makedirs(case_dir, exist_ok=True)
            case_context.update(self.fixed_parameters)

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        pass

    def build_case_and_render_files(
        self,
        case_context: str,
        case_dir: str,
    ) -> None:
        """
        Build the input files and context for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        self.build_case(
            case_context=case_context,
            case_dir=case_dir,
        )
        for template_name in self.templates_name:
            try:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=op.join(case_dir, template_name),
                )
            except UnicodeDecodeError as _ude:
                self.copy_files(
                    src=op.join(self.templates_dir, template_name),
                    dst=op.join(case_dir, template_name),
                )

    def build_cases(
        self,
        mode: str = "one_by_one",
        cases_name_format: callable = lambda ctx: f"{ctx.get('case_num'):04}",
        cases_to_build: List[int] = None,
        num_workers: int = None,
    ) -> None:
        """
        Create the cases folders and render the input files.

        Parameters
        ----------
        mode : str, optional
            The mode to create the cases. Can be "all_combinations" or "one_by_one".
        cases_name_format : callable, optional
            The function to format the case name. Default is a lambda function
            that formats the case number with leading zeros.
            Default is "one_by_one".
        cases_to_build : List[int], optional
            The list with the cases to build. Default is None.
        num_workers : int, optional
            The number of parallel workers. Default is None.

        Raises
        ------
        ValueError
            If the mode is not valid.
        """

        self.load_cases(
            mode=mode,
            cases_name_format=cases_name_format,
        )
        self.logger.debug(
            f"Cases context and directory created with {len(self.cases_context)} cases."
        )

        if num_workers is None:
            num_workers = self.num_workers

        if cases_to_build is None:
            cases_to_build = list(range(len(self.cases_context)))
        else:
            self.logger.warning(
                f"cases_to_build was specified, so just {cases_to_build} will be built."
            )

        cases_context_to_build = [self.cases_context[case] for case in cases_to_build]
        cases_dir_to_build = [self.cases_dirs[case] for case in cases_to_build]

        if num_workers > 1:
            self.logger.debug(
                f"Building cases in parallel. Number of workers: {num_workers}."
            )
            _results = self.parallel_execute(
                func=self.build_case_and_render_files,
                items=zip(cases_context_to_build, cases_dir_to_build),
                num_workers=num_workers,
            )
        else:
            self.logger.debug("Building cases sequentially.")
            for case_context, case_dir in zip(
                cases_context_to_build, cases_dir_to_build
            ):
                self.build_case_and_render_files(
                    case_context=case_context,
                    case_dir=case_dir,
                )

        self.logger.info(
            f"{len(self.cases_dirs)} cases created in {mode} mode and saved in {self.output_dir}"
        )

        # Save an example sbatch file in the output directory
        with open(f"{self.output_dir}/sbatch_example.sh", "w") as file:
            file.write(self.sbatch_file_example)
        self.logger.info(f"SBATCH example file generated in {self.output_dir}")

    def run_case(
        self,
        case_dir: str,
        launcher: str,
        output_log_file: str = "wrapper_out.log",
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
        output_log_file : str, optional
            The name of the output log file. Default is "wrapper_out.log".
        error_log_file : str, optional
            The name of the error log file. Default is "wrapper_error.log".
        """

        # Get launcher command from the available launchers
        launcher = self.list_available_launchers().get(launcher, launcher)

        # Run the case in the case directory
        self.logger.info(f"Running case in {case_dir} with launcher={launcher}.")
        output_log_file = op.join(case_dir, output_log_file)
        error_log_file = op.join(case_dir, error_log_file)
        self._exec_bash_commands(
            str_cmd=launcher,
            out_file=output_log_file,
            err_file=error_log_file,
            cwd=case_dir,
        )

    def run_cases(
        self,
        launcher: str,
        cases_to_run: List[int] = None,
        num_workers: int = None,
    ) -> None:
        """
        Run the cases based on the launcher specified.
        Cases to run can be specified.
        Parallel execution is optional by modifying the num_workers parameter.

        Parameters
        ----------
        launcher : str
            The launcher to run the cases.
        cases_to_run : List[int], optional
            The list with the cases to run. Default is None.
        num_workers : int, optional
            The number of parallel workers. Default is None.
        """

        if self.cases_context is None or self.cases_dirs is None:
            raise ValueError(
                "Cases context or cases directories are not set. Please run load_cases() first."
            )

        if num_workers is None:
            num_workers = self.num_workers

        # Get launcher command from the available launchers
        launcher = self.list_available_launchers().get(launcher, launcher)

        if cases_to_run is not None:
            self.logger.warning(
                f"Cases to run was specified, so just {cases_to_run} will be run."
            )
            cases_dir_to_run = [self.cases_dirs[case] for case in cases_to_run]
        else:
            cases_dir_to_run = copy.deepcopy(self.cases_dirs)

        if num_workers > 1:
            self.logger.debug(
                f"Running cases in parallel with launcher={launcher}. Number of workers: {num_workers}."
            )
            _results = self.parallel_execute(
                func=self.run_case,
                items=cases_dir_to_run,
                num_workers=num_workers,
                launcher=launcher,
            )
        else:
            self.logger.debug(f"Running cases sequentially with launcher={launcher}.")
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

        self.logger.info("All cases executed.")

    def _run_cases_with_status(
        self,
        launcher: str,
        cases_to_run: List[int],
        num_workers: int,
        status_queue: Queue,
    ) -> None:
        """
        Run the cases and update the status queue.

        Parameters
        ----------
        launcher : str
            The launcher to run the cases.
        cases_to_run : List[int]
            The list with the cases to run.
        num_workers : int
            The number of parallel workers.
        status_queue : Queue
            The queue to update the status.
        """

        try:
            self.run_cases(launcher, cases_to_run, num_workers)
            status_queue.put("Completed")
        except Exception as e:
            status_queue.put(f"Error: {e}")

    def run_cases_in_background(
        self,
        launcher: str,
        cases_to_run: List[int] = None,
        num_workers: int = None,
    ) -> None:
        """
        Run the cases in the background based on the launcher specified.
        Cases to run can be specified.
        Parallel execution is optional by modifying the num_workers parameter.

        Parameters
        ----------
        launcher : str
            The launcher to run the cases.
        cases_to_run : List[int], optional
            The list with the cases to run. Default is None.
        num_workers : int, optional
            The number of parallel workers. Default is None.
        """

        if num_workers is None:
            num_workers = self.num_workers

        self.status_queue = Queue()
        self.thread = threading.Thread(
            target=self._run_cases_with_status,
            args=(launcher, cases_to_run, num_workers, self.status_queue),
        )
        self.thread.start()

    def get_thread_status(self) -> str:
        """
        Get the status of the background thread.

        Returns
        -------
        str
            The status of the background thread.
        """

        if self.thread is None:
            return "Not started"
        elif self.thread.is_alive():
            return "Running"
        else:
            return self.status_queue.get()

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
        self._exec_bash_commands(str_cmd=launcher, cwd=self.output_dir)

    def monitor_cases(
        self, cases_status: dict, value_counts: str
    ) -> Union[pd.DataFrame, dict]:
        """
        Return the status of the cases.
        This method is used to monitor the cases and log relevant information.
        It is called in the child class to monitor the cases.

        Parameters
        ----------
        cases_status : dict
            The dictionary with the cases status.
            Each key is the base case directory name and the value is the status of the case.
            This status can be any string.
        value_counts : str, optional
            The value counts to be returned.
            If "simple", it returns a dictionary with the number of cases in each status.
            If "percentage", it returns a DataFrame with the percentage of cases in each status.
            If "cases", it returns a dictionary with the cases in each status.
            Default is None.

        Returns
        -------
        Union[pd.DataFrame, dict]
            The cases status as a pandas DataFrame or a dictionary with aggregated info.
        """

        full_monitorization_df = pd.DataFrame(
            cases_status.items(), columns=["Case", "Status"]
        )
        if value_counts:
            value_counts_df = full_monitorization_df.set_index("Case").value_counts()
            if value_counts == "simple":
                return value_counts_df
            elif value_counts == "percentage":
                return value_counts_df / len(full_monitorization_df) * 100
            value_counts_unique_values = [
                run_type[0] for run_type in value_counts_df.index.values
            ]
            value_counts_dict = {
                run_type: list(
                    full_monitorization_df.where(
                        full_monitorization_df["Status"] == run_type
                    )
                    .dropna()["Case"]
                    .values
                )
                for run_type in value_counts_unique_values
            }
            return value_counts_dict
        else:
            return full_monitorization_df

    def postprocess_case(self, **kwargs) -> None:
        """
        Postprocess the model output.
        """

        raise NotImplementedError("The method postprocess_case must be implemented.")

    def join_postprocessed_files(self, **kwargs) -> xr.Dataset:
        """
        Join the postprocessed files.
        """

        raise NotImplementedError(
            "The method join_postprocessed_files must be implemented."
        )

    def postprocess_cases(
        self,
        cases_to_postprocess: List[int] = None,
        write_output_nc: bool = False,
        clean_after: bool = False,
        **kwargs,
    ) -> Union[xr.Dataset, List[xr.Dataset]]:
        """
        Postprocess the model output.
        All extra keyword arguments will be passed to the postprocess_case method.

        Parameters
        ----------
        cases_to_postprocess : List[int], optional
            The list with the cases to postprocess. Default is None.
        write_output_nc : bool, optional
            Write the output postprocessed file. Default is False.
        clean_after : bool, optional
            Clean the cases directories after postprocessing. Default is False.
        **kwargs
            Additional keyword arguments to be passed to the postprocess_case method.

        Returns
        -------
        xr.Dataset or List[xr.Dataset]
            The postprocessed file or the list with the postprocessed files.
        """

        if self.cases_context is None or self.cases_dirs is None:
            raise ValueError(
                "Cases context or cases directories are not set. Please run load_cases() first."
            )

        output_postprocessed_file_path = op.join(
            self.output_dir, "output_postprocessed.nc"
        )
        if op.exists(output_postprocessed_file_path):
            self.logger.warning(
                "Output postprocessed file already exists. Skipping postprocessing."
            )
            return xr.open_dataset(output_postprocessed_file_path)

        if cases_to_postprocess is not None:
            self.logger.warning(
                f"Cases to postprocess was specified, so just {cases_to_postprocess} will be postprocessed."
            )
            self.logger.warning(
                "Remember you can just use postprocess_case method to postprocess a single case."
            )
            cases_dir_to_postprocess = [
                self.cases_dirs[case] for case in cases_to_postprocess
            ]
        else:
            cases_to_postprocess = list(range(len(self.cases_dirs)))
            cases_dir_to_postprocess = copy.deepcopy(self.cases_dirs)

        postprocessed_files = []
        for case_num, case_dir in zip(cases_to_postprocess, cases_dir_to_postprocess):
            try:
                postprocessed_file = self.postprocess_case(
                    case_num=case_num, case_dir=case_dir, **kwargs
                )
                postprocessed_files.append(postprocessed_file)
            except Exception as e:
                self.logger.error(
                    f"Output not postprocessed for case {case_num}. Error: {e}."
                )

        try:
            output_postprocessed = self.join_postprocessed_files(
                postprocessed_files=postprocessed_files
            )
            if write_output_nc:
                self.logger.info(
                    f"Writing output postprocessed file to {output_postprocessed_file_path}."
                )
                output_postprocessed.to_netcdf(output_postprocessed_file_path)
            if clean_after:
                self.logger.warning("Cleaning up all cases dirs.")
                for case_dir in self.cases_dirs:
                    os.rmdir(case_dir)
                self.logger.info("Clean up completed.")
            return output_postprocessed

        except NotImplementedError as exc:
            self.logger.error(f"Error joining postprocessed files: {exc}")
            return postprocessed_files


class DummyModelWrapper(BaseModelWrapper):
    """
    Dummy model wrapper to test the BaseModelWrapper class.
    """

    def build_case(self, case_context: dict, case_dir: str) -> None:
        pass

    def postprocess_case(self, **kwargs) -> None:
        pass

    def join_postprocessed_files(self, **kwargs) -> xr.Dataset:
        return xr.Dataset()

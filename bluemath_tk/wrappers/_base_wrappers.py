import os
import itertools
from abc import abstractmethod
from typing import List
import subprocess
import numpy as np
from jinja2 import Environment, FileSystemLoader
from ..core.models import BlueMathModel


class BaseModelWrapper(BlueMathModel):
    """
    Base class for numerical models wrappers.

    Attributes
    ----------
    templates_dir : str
        The directory where the templates are stored.
    templates_name : List[str]
        The names of the templates.
    model_parameters : dict
        The parameters to be used in the templates.
    output_dir : str
        The directory where the output files will be saved.
    env : Environment
        The Jinja2 environment.
    cases_dirs : List[str]
        The list with cases directories.
    cases_context : List[dict]
        The list with cases context.

    Methods
    -------
    _check_parameters_type(default_parameters, model_parameters)
        Check if the parameters have the correct type.
    _exec_bash_commands(str_cmd, out_file=None, err_file=None)
        Execute bash commands.
    create_cases_context_one_by_one()
        Create an array of dictionaries with the combinations of values from the
        input dictionary, one by one.
    create_cases_context_all_combinations()
        Create an array of dictionaries with each possible combination of values
        from the input dictionary.
    render_file_from_template(template_name, context, output_filename=None)
        Render a file from a template.
    write_array_in_file(array, filename)
        Write an array in a file.
    copy_files(src, dst)
        Copy file(s) from source to destination.
    build_cases(mode="all_combinations")
        Create the cases folders and render the input files.
    run_cases()
        Run the cases.
    run_model(case_dir)
        Run the model for a specific case (abstract method).
    """

    def __init__(
        self,
        templates_dir: str,
        templates_name: List[str],
        model_parameters: dict,
        output_dir: str,
        default_parameters: dict = None,
    ) -> None:
        """
        Initialize the BaseModelWrapper.

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
        default_parameters : dict, optional
            The default parameters type for the model. If None, the parameters will
            not be checked.
            Default is None.
        """

        super().__init__()
        if default_parameters is not None:
            self._check_parameters_type(
                default_parameters=default_parameters, model_parameters=model_parameters
            )
        self.templates_dir = templates_dir
        self.templates_name = templates_name
        self.model_parameters = model_parameters
        self.output_dir = output_dir
        self._env = Environment(loader=FileSystemLoader(self.templates_dir))
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

    def create_cases_context_one_by_one(self) -> List[dict]:
        """
        Create an array of dictionaries with the combinations of values from the
        input dictionary, one by one.

        Returns
        -------
        array_of_contexts : list
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
        array_of_contexts : list
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

        with open(filename, "w") as f:
            if array.ndim == 1:
                for item in array:
                    f.write(f"{item}\n")
            elif array.ndim == 2:
                for row in array:
                    f.write(" ".join(map(str, row)) + "\n")
            else:
                raise ValueError("Only 1D and 2D arrays are supported")

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

        if os.path.isdir(src):
            os.makedirs(dst, exist_ok=True)
            for file in os.listdir(src):
                with open(file, "r") as f:
                    content = f.read()
                with open(os.path.join(dst, file), "w") as f:
                    f.write(content)
        else:
            with open(src, "r") as f:
                content = f.read()
            with open(dst, "w") as f:
                f.write(content)

    def build_cases(self, mode: str = "all_combinations") -> None:
        """
        Create the cases folders and render the input files.

        Parameters
        ----------
        mode : str, optional
            The mode to create the cases. Can be "all_combinations" or "one_by_one".
            Default is "all_combinations".
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

    def run_cases(self) -> None:
        """
        Run the cases.
        """

        if self.cases_dirs:
            for case_dir in self.cases_dirs:
                self.logger.info(f"Running case in {case_dir}")
                self.run_model(case_dir=case_dir)
            self.logger.info("All cases ran successfully.")
        else:
            raise ValueError("No cases to run.")

    @abstractmethod
    def run_model(self, case_dir: str) -> None:
        """
        Run the model.

        Parameters
        ----------
        case_dir : str
            The directory of the case.
        """

        pass

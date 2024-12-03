import os
import itertools
from typing import List
from jinja2 import Environment, FileSystemLoader
from ..core.models import BlueMathModel


class BaseModelWrapper(BlueMathModel):
    """
    Base class for model wrappers.

    Attributes
    ----------
    templates_dir : str
        The directory where the templates are stored.
    templates_name : list
        The names of the templates.
    model_parameters : dict
        The parameters to be used in the templates.
    output_dir : str
        The directory where the output files will be saved.
    env : Environment
        The Jinja2 environment.

    Methods
    -------
    create_cases_context_one_by_one()
        Create an array of dictionaries with the combinations of values from the
        input dictionary, one by one.
    create_cases_context_all_combinations()
        Create an array of dictionaries with each possible combination of values
        from the input dictionary.
    render_file_from_template(template_name, context, output_filename=None)
        Render a file from a template.
    build_cases()
        Build the cases.
    run_cases()
        Run the cases.
    """

    def __init__(
        self,
        templates_dir: str,
        templates_name: List[str],
        model_parameters: dict,
        output_dir: str,
    ):
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
        """

        super().__init__()
        self.templates_dir = templates_dir
        self.templates_name = templates_name
        self.model_parameters = model_parameters
        self.output_dir = output_dir
        self.env = Environment(loader=FileSystemLoader(self.templates_dir))

    def create_cases_context_one_by_one(self):
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

    def create_cases_context_all_combinations(self):
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
    ):
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

    def write_array_in_file(self, array, filename):
        """
        Write an array in a file.

        Parameters
        ----------
        array : np.array
            The array to be written.
        filename : str
            The name of the file.
        """

        with open(filename, "w") as f:
            for item in array:
                f.write(f"{item}\n")

import os
from typing import List
from jinja2 import Environment, FileSystemLoader
from abc import abstractmethod
from ..core.models import BlueMathModel


class BaseModelWrapper(BlueMathModel):
    """
    Base class for model wrappers.
    """

    @abstractmethod
    def __init__(
        self,
        templates_dir: str,
        templates_name: List[str],
        model_parameters: dict,
        output_dir: str,
    ):
        super().__init__()
        self.templates_dir = templates_dir
        self.templates_name = templates_name
        self.model_parameters = model_parameters
        self.output_dir = output_dir
        self.env = Environment(loader=FileSystemLoader(self.templates_dir))

    def build_cases_one_by_one(self):
        num_cases = len(next(iter(self.model_parameters.values())))
        for param, values in self.model_parameters.items():
            if len(values) != num_cases:
                raise ValueError(
                    f"All parameters must have the same number of values in one_by_one mode, check {param}"
                )
        for case_num in range(num_cases):
            case_folder = os.path.join(self.output_dir, f"{case_num:04}")
            os.makedirs(case_folder, exist_ok=True)
            case_context = {
                param: values[case_num]
                for param, values in self.model_parameters.items()
            }
            for template_name in self.templates_name:
                self.render_file_from_template(
                    template_name=template_name,
                    context=case_context,
                    output_filename=os.path.join(case_folder, template_name),
                )

    def build_cases(self, mode: str = "one_by_one"):
        if mode == "one_by_one":
            self.build_cases_one_by_one()

    def render_file_from_template(
        self, template_name: str, context: dict, output_filename: str = None
    ):
        template = self.env.get_template(name=template_name)
        rendered_content = template.render(context)
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, template_name)
        with open(output_filename, "w") as f:
            f.write(rendered_content)

    @abstractmethod
    def run_model(self, input_file):
        pass

import numpy as np
from bluemath_tk.wrappers.schism.schism_wrapper import SchismModelWrapper
import os

class SchismNBModelWrapper(SchismModelWrapper):
    """
    Wrapper for the Schism model.
    """

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
        os.mkdir(os.path.join(case_dir, 'outputs'), exist_ok=True)

    def build_cases(
        self,
        mode: str = "one_by_one",
    ) -> None:
        """
        Build the input files for all cases.

        Parameters
        ----------
        mode : str, optional
            The mode to build the cases. Default is "one_by_one".

        Raises
        ------
        ValueError
            If the cases were not properly built
        """

        super().build_cases(mode=mode)
        if not self.cases_context or not self.cases_dirs:
            raise ValueError("Cases were not properly built.")
        for case_context, case_dir in zip(self.cases_context, self.cases_dirs):
            self.build_case(
                case_context=case_context,
                case_dir=case_dir,
            )

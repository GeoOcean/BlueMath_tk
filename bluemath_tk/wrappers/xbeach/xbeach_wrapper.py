from .._base_wrappers import BaseModelWrapper
import math

class XBeachModelWrapper(BaseModelWrapper):
    """
    Wrapper for the XBeach model.
    https://xbeach.readthedocs.io/en/latest/

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {
        "comptime": {
            "type": int,
            "value": 3600,
            "description": "The computational time.",
        },
        "wbctype": {
            "type": str,
            "value": "off",
            "description": "The time step for the simulation.",
        },
    }

    available_launchers = {
        "geoocean-cluster": "launchXbeach.sh",
    }

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the XBeach model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )

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

        if self.fixed_parameters['wbctype'] == 'jonstable':
            with open(f"{case_dir}/jonswap.txt", "w") as f:
                for i in range(math.ceil(self.fixed_parameters['comptime'] / 3600)):
                    f.write(f"{case_context['Hs']} {case_context['Tp']} {case_context['Dir']} 3.300000 30.000000 3600.000000 1.000000 \n")
        


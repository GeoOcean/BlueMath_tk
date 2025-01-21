from .._base_wrappers import BaseModelWrapper


class SwanModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SWAN model.
    https://swanmodel.sourceforge.io/online_doc/swanuse/swanuse.html

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {
        "hs": float,
        "tp": float,
        "dir": float,
        "spr": float,
    }

    available_launchers = {
        "bash": "swanrun -input input.sws",
        "docker": "docker run --rm -v .:/case_dir -w /case_dir tausiaj/swan-geoocean:41.51 swanrun -input input.sws",
    }

    def __init__(
        self,
        templates_dir: str,
        model_parameters: dict,
        output_dir: str,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWASH model wrapper.
        """

        super().__init__(
            templates_dir=templates_dir,
            model_parameters=model_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )

import os.path as op

from bluemath_tk.wrappers.delft3d.delft3d_wrapper import Delft3dModelWrapper


class Delft3DfmModelWrapper(Delft3dModelWrapper):
    """
    Wrapper for the Delft3D model with flow mode.
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

        self.copy_files(
            src=case_context.get("grid_nc_file"),
            dst=op.join(case_dir, op.basename(case_context.get("grid_nc_file"))),
        )

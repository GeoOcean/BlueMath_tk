"""
Wrappers for the SnapWave nearshore wave model.

SnapWave (https://github.com/danoroelvink/snapwave) propagates offshore wave
conditions, given at a set of boundary nodes, over an unstructured mesh, and
writes bulk parameters at observation points (``his_file``) and optionally
over the mesh (``map_file``).

Two flavours share the same case setup:

- :class:`SnapWaveMetaModelWrapper`: stationary cases sampled from a design
  (e.g. LHS), with unit Hs on one active boundary node. Postprocessed on a
  ``case_num`` dimension, for building a metamodel.
- :class:`SnapWaveDynamicModelWrapper`: one case per time step of a boundary
  forcing time series (see
  :func:`~bluemath_tk.wrappers.snapwave.snapwave_utils.boundary_time_series`).
  Postprocessed on ``time``.

Templates are not shipped with the toolkit: each project keeps its own (see
BlueMath ``climate_services/hindcast/HyWaves_Netherlands/templates``). The
context they receive is documented on each wrapper class.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from .._base_wrappers import BaseModelWrapper
from .snapwave_utils import (
    POINT_VARS,
    read_his_file,
    write_points_to_txt,
    write_polygon_vertices_to_txt,
)

#: Shell lines that put SnapWave on the PATH on the GeoOcean cluster.
GEOOCEAN_SNAPWAVE_MODULE = (
    "module use /nfs/software/geocean/modulefiles\nmodule load snapwave"
)

_NUMBER = (int, float, np.integer, np.floating)
_SAVE_MAP = {
    "type": (bool, np.bool_),
    "value": None,
    "description": "Also write the mesh output (map_file) for this case.",
}

SLURM_ARRAY_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --output={logs_dir}/%A_%a.out
#SBATCH --error={logs_dir}/%A_%a.err
{extra_sbatch}
{setup}

# SnapWave has no thread-count option of its own: it inherits libgomp's
# default, which uses every online CPU unless told otherwise.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

case_dir=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {case_dirs_file})
cd "$case_dir"
{snapwave_cmd} > wrapper_out.log 2> wrapper_error.log
rm -f snapwave.upw
"""


class SnapWaveModelWrapper(BaseModelWrapper):
    """
    Shared SnapWave case setup: boundary nodes, enclosure and observation points.

    Each case folder gets ``boundary.txt``, ``enclosure.txt`` and the
    observation-points file next to the rendered templates. The template
    context gets ``n_nodes`` (number of boundary nodes), and ``map_file`` /
    ``map_dirspr`` from the optional boolean ``save_map`` case parameter.

    Attributes
    ----------
    available_launchers : dict
        ``"default"`` runs ``snapwave`` from the PATH; ``"geoocean-cluster"``
        loads the cluster module first. For SLURM job arrays use
        :meth:`write_slurm_array` / :meth:`run_cases_slurm`.
    """

    default_parameters: dict = {}

    available_launchers = {
        "default": "snapwave",
        "geoocean-cluster": f"{GEOOCEAN_SNAPWAVE_MODULE}\nsnapwave",
    }

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        boundary_nodes: np.ndarray,
        enclosure_polygon: Polygon | MultiPolygon,
        output_points: np.ndarray | pd.DataFrame,
        templates_name: list[str] | str = "all",
        debug: bool = True,
        obs_filename: str = "output_sites.txt",
        his_filename: str = "output_sites.nc",
        point_dim: str = "stations",
        point_coords: dict[str, str] | None = None,
        point_variables: tuple[str, ...] | None = tuple(POINT_VARS.values()),
        store_parameters: tuple[str, ...] = (),
    ) -> None:
        """
        Initialise the SnapWave wrapper.

        Parameters
        ----------
        templates_dir : str
            Folder with the ``snapwave.inp`` and forcing (``hs.txt``, ...)
            templates.
        metamodel_parameters : dict
            Per-case parameters (lists of equal length).
        fixed_parameters : dict
            Parameters shared by every case (e.g. ``gridfile``).
        output_dir : str
            Folder where case folders are created.
        boundary_nodes : np.ndarray
            ``(n_nodes, 2)`` boundary node coordinates, in the order of the
            per-node forcing columns.
        enclosure_polygon : Polygon or MultiPolygon
            Computational enclosure (see
            :func:`~bluemath_tk.wrappers.snapwave.snapwave_utils.build_enclosure_polygon`).
        output_points : np.ndarray or pd.DataFrame
            Observation points: ``(n, 2)`` array, or a DataFrame with ``lon``
            and ``lat`` columns.
        templates_name : list of str or "all", optional
            Templates to render. Default is "all".
        debug : bool, optional
            DEBUG-level logging. Default is True.
        obs_filename : str, optional
            Observation-points file name (``obsfile`` in ``snapwave.inp``).
        his_filename : str, optional
            History output name (``his_file`` in ``snapwave.inp``).
        point_dim : str, optional
            Name of the observation-points dimension in postprocessed output.
            Default is "stations" (SnapWave's own name).
        point_coords : dict, optional
            ``{coord_name: column}`` of *output_points* (a DataFrame) attached
            as coordinates on *point_dim* when postprocessing, e.g.
            ``{"site_depth": "depth"}``. When given, SnapWave's native station
            coordinates (``station_x``, ``station_y``, ...) are dropped.
        point_variables : tuple of str, optional
            Output variables kept when postprocessing (short names, see
            :data:`~bluemath_tk.wrappers.snapwave.snapwave_utils.POINT_VARS`).
            None keeps everything. Default is ``("hs", "tp", "dir", "spr")``.
        store_parameters : tuple of str, optional
            Scalar case parameters stored in each postprocessed case as
            ``f"{name}_forcing"`` variables along the case dimension (e.g.
            ``("tp", "dir", "spr", "wl")`` for metamodel training data).
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

        self.boundary_nodes = np.asarray(boundary_nodes, dtype=float).reshape(-1, 2)
        self.enclosure_polygon = enclosure_polygon
        self.point_metadata = (
            output_points.reset_index(drop=True)
            if isinstance(output_points, pd.DataFrame)
            else None
        )
        if point_coords and self.point_metadata is None:
            raise ValueError("point_coords needs output_points as a DataFrame")
        if isinstance(output_points, pd.DataFrame):
            output_points = output_points[["lon", "lat"]].to_numpy(float)
        self.output_points = np.asarray(output_points, dtype=float).reshape(-1, 2)
        self.obs_filename = obs_filename
        self.his_filename = his_filename
        self.point_dim = point_dim
        self.point_coords = dict(point_coords or {})
        self.point_variables = point_variables
        self.store_parameters = tuple(store_parameters)

    @property
    def n_nodes(self) -> int:
        """Number of boundary nodes."""

        return len(self.boundary_nodes)

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """
        Write boundary, enclosure and observation files into a case folder.

        Parameters
        ----------
        case_context : dict
            Case parameters; ``n_nodes``, ``map_file`` and ``map_dirspr`` are
            added for the templates.
        case_dir : str
            Case folder.
        """

        case_context["n_nodes"] = self.n_nodes
        write_points_to_txt(self.boundary_nodes, os.path.join(case_dir, "boundary.txt"))
        write_polygon_vertices_to_txt(
            self.enclosure_polygon, os.path.join(case_dir, "enclosure.txt")
        )
        write_points_to_txt(
            self.output_points, os.path.join(case_dir, self.obs_filename)
        )
        save_map = bool(case_context.get("save_map", False))
        case_context["map_file"] = "output_map.nc" if save_map else ""
        case_context["map_dirspr"] = int(save_map)

    def run_case(
        self,
        case_num: int,
        case_dir: str,
        case_context: dict,
        launcher: str,
        output_log_file: str = "wrapper_out.log",
        error_log_file: str = "wrapper_error.log",
    ) -> None:
        """
        Run one case and delete SnapWave's ``snapwave.upw`` scratch file.

        Parameters
        ----------
        case_num, case_dir, case_context, launcher
            See :meth:`BaseModelWrapper.run_case`.
        output_log_file, error_log_file : str, optional
            Log file names inside *case_dir*.
        """

        super().run_case(
            case_num=case_num,
            case_dir=case_dir,
            case_context=case_context,
            launcher=launcher,
            output_log_file=output_log_file,
            error_log_file=error_log_file,
        )
        upw = os.path.join(case_dir, "snapwave.upw")
        if os.path.exists(upw):
            os.remove(upw)

    def monitor_cases(self, value_counts: str | None = None):
        """
        Case status from the presence of the history output file.

        Parameters
        ----------
        value_counts : str, optional
            Passed to :meth:`BaseModelWrapper.monitor_cases`.

        Returns
        -------
        pd.DataFrame or dict
            See :meth:`BaseModelWrapper.monitor_cases`.
        """

        cases_status = {
            os.path.basename(case_dir): (
                "FINISHED"
                if os.path.exists(os.path.join(case_dir, self.his_filename))
                else "NOT STARTED"
            )
            for case_dir in self.cases_dirs
        }

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def write_slurm_array(
        self,
        filename: str = "snapwave_array.sh",
        partition: str = "geocean",
        cpus_per_task: int = 2,
        mem: str = "4gb",
        job_name: str = "snapwave",
        setup: str = GEOOCEAN_SNAPWAVE_MODULE,
        snapwave_cmd: str = "snapwave",
        logs_dir: str = "slurm_logs",
        extra_sbatch: list[str] | None = None,
    ) -> str:
        """
        Write a SLURM job-array script (one task per case) to ``output_dir``.

        Also writes ``case_dirs.txt`` (see :meth:`cases_dir_to_txt`), which
        the script reads by ``SLURM_ARRAY_TASK_ID``. Submit it with
        :meth:`run_cases_slurm`, or by hand with
        ``sbatch --array=1-<n_cases> snapwave_array.sh`` from ``output_dir``.

        Parameters
        ----------
        filename : str, optional
            Script name inside ``output_dir``. Default is "snapwave_array.sh".
        partition : str, optional
            SLURM partition. Default is "geocean".
        cpus_per_task : int, optional
            Cores per case; also sets ``OMP_NUM_THREADS``. Default is 2.
        mem : str, optional
            Memory per case. Default is "4gb".
        job_name : str, optional
            SLURM job name. Default is "snapwave".
        setup : str, optional
            Shell lines run before SnapWave (e.g. ``module load``). Default
            loads the GeoOcean cluster module; pass "" if ``snapwave`` is
            already on the PATH.
        snapwave_cmd : str, optional
            SnapWave command or binary path. Default is "snapwave".
        logs_dir : str, optional
            SLURM log folder, relative to ``output_dir``. Default is
            "slurm_logs".
        extra_sbatch : list of str, optional
            Additional ``#SBATCH`` option lines, e.g. ``["--time=00:30:00"]``.

        Returns
        -------
        str
            Path of the written script.
        """

        case_dirs_file = self.cases_dir_to_txt()
        os.makedirs(os.path.join(self.output_dir, logs_dir), exist_ok=True)
        extra = "\n".join(f"#SBATCH {line}" for line in (extra_sbatch or []))
        script = SLURM_ARRAY_TEMPLATE.format(
            job_name=job_name,
            partition=partition,
            cpus_per_task=cpus_per_task,
            mem=mem,
            logs_dir=logs_dir,
            extra_sbatch=extra,
            setup=setup,
            case_dirs_file=case_dirs_file,
            snapwave_cmd=snapwave_cmd,
        )
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(script)
        self.logger.info(f"SLURM array script for {len(self.cases_dirs)} cases: {path}")

        return path

    def run_cases_slurm(
        self,
        cases_to_run: list[int] | None = None,
        max_parallel: int | None = None,
        **script_kwargs,
    ) -> None:
        """
        Write the SLURM array script and submit it with ``sbatch``.

        Parameters
        ----------
        cases_to_run : list of int, optional
            0-based case indices to submit. Default is every case.
        max_parallel : int, optional
            Maximum simultaneously running tasks (``%`` array throttle).
        **script_kwargs
            Passed to :meth:`write_slurm_array`.
        """

        script = self.write_slurm_array(**script_kwargs)
        if cases_to_run is None:
            array = f"1-{len(self.cases_dirs)}"
        else:
            array = ",".join(str(int(i) + 1) for i in cases_to_run)
        if max_parallel is not None:
            array += f"%{int(max_parallel)}"
        self.run_cases_bulk(launcher=f"sbatch --array={array} {script}")

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Read one case's history output (``hs``, ``tp``, ``dir``, ``spr``).

        Parameters
        ----------
        case_num : int
            Case index.
        case_dir : str
            Case folder.
        case_context : dict
            Case parameters.

        Returns
        -------
        xr.Dataset
            History output on ``(time, point_dim)``.
        """

        ds = read_his_file(case_dir, self.his_filename)

        return self._format_output(ds, case_context, row_dim="time")

    def _format_output(
        self, ds: xr.Dataset, case_context: dict, row_dim: str
    ) -> xr.Dataset:
        """
        Apply the point/variable options to one case's history output.

        Parameters
        ----------
        ds : xr.Dataset
            Output of :func:`read_his_file` on ``(row_dim, stations)``.
        case_context : dict
            Case parameters (for :attr:`store_parameters`).
        row_dim : str
            Case dimension (``time`` or ``case_num``).

        Returns
        -------
        xr.Dataset
            Formatted output.
        """

        if self.point_variables is not None:
            ds = ds[[v for v in self.point_variables if v in ds.data_vars]]
        if self.point_coords:
            n_points = ds.sizes["stations"]
            if n_points != len(self.point_metadata):
                raise ValueError(
                    f"{n_points} stations in output but {len(self.point_metadata)} "
                    "output_points rows"
                )
            ds = ds.drop_vars([c for c in ds.coords if c != row_dim])
        if self.point_dim != "stations":
            ds = ds.rename_dims({"stations": self.point_dim})
        if self.point_coords:
            # assigned after renaming, so a coordinate named like the dimension
            # becomes its index (``.sel(sites=...)`` works)
            ds = ds.assign_coords(
                {
                    name: (self.point_dim, self.point_metadata[col].to_numpy())
                    for name, col in self.point_coords.items()
                }
            )
        for name in self.store_parameters:
            if name in case_context:
                ds[f"{name}_forcing"] = ((row_dim,), [float(case_context[name])])

        return ds

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
    ) -> xr.Dataset:
        """
        Concatenate postprocessed cases along ``time``.

        Parameters
        ----------
        postprocessed_files : list of xr.Dataset
            One dataset per case.

        Returns
        -------
        xr.Dataset
            All cases.
        """

        return xr.concat(postprocessed_files, dim="time")


class SnapWaveMetaModelWrapper(SnapWaveModelWrapper):
    """
    Stationary SnapWave cases for a metamodel (e.g. LHS over Tp, Dir, Spr, WL).

    Metamodel templates are expected to apply unit Hs at the
    ``active_node`` (1-based) case parameter and zero elsewhere, with the
    case's ``tp``, ``dir``, ``spr`` and ``wl`` at every node. Each case is
    postprocessed to a single ``case_num`` row.
    """

    default_parameters = {
        "tp": {"type": _NUMBER, "value": None, "description": "Peak period (s)."},
        "dir": {
            "type": _NUMBER,
            "value": None,
            "description": "Mean wave direction (nautical, coming from, deg).",
        },
        "spr": {
            "type": _NUMBER,
            "value": None,
            "description": "Directional spread (deg).",
        },
        "wl": {"type": _NUMBER, "value": None, "description": "Water level (m)."},
        "active_node": {
            "type": (int, np.integer),
            "value": None,
            "description": "1-based boundary node with unit Hs.",
        },
        "save_map": _SAVE_MAP,
    }

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Read one stationary case, indexed by ``case_num`` instead of ``time``.

        Parameters
        ----------
        case_num : int
            Case index.
        case_dir : str
            Case folder.
        case_context : dict
            Case parameters.

        Returns
        -------
        xr.Dataset
            Output on ``(case_num, stations)``.
        """

        ds = read_his_file(case_dir, self.his_filename)
        ds = ds.isel(time=[-1]).rename({"time": "case_num"})
        ds = ds.assign_coords(case_num=[case_num])
        ds = self._format_output(ds, case_context, row_dim="case_num")
        if self.split_by is not None:
            ds = ds.assign_coords(
                {self.split_by: ("case_num", [case_context[self.split_by]])}
            )

        return ds

    def __init__(
        self,
        *args,
        split_by: str | None = None,
        split_filename: str = "output_sites_{}.nc",
        **kwargs,
    ) -> None:
        """
        Initialise the metamodel wrapper.

        Parameters
        ----------
        *args, **kwargs
            See :class:`SnapWaveModelWrapper`.
        split_by : str, optional
            Case parameter (e.g. ``"active_node"``) by which the joined output is
            split into one NetCDF per value in ``output_dir``. Default is None
            (no split).
        split_filename : str, optional
            File name pattern for the split files, formatted with the value.
            Default is "output_sites_{}.nc".
        """

        self.split_by = split_by
        self.split_filename = split_filename
        super().__init__(*args, **kwargs)

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
    ) -> xr.Dataset | dict:
        """
        Concatenate postprocessed cases along ``case_num`` (optionally split).

        Parameters
        ----------
        postprocessed_files : list of xr.Dataset
            One dataset per case.

        Returns
        -------
        xr.Dataset or dict
            All cases; with ``split_by``, ``{value: written NetCDF path}``.
        """

        joined = xr.concat(postprocessed_files, dim="case_num")
        if self.split_by is None:
            return joined

        written = {}
        for value, part in joined.groupby(self.split_by):
            path = os.path.join(self.output_dir, self.split_filename.format(value))
            part.drop_vars(self.split_by).to_netcdf(path)
            written[value] = path
        self.logger.info(f"Wrote {len(written)} files split by {self.split_by}.")

        return written


class SnapWaveDynamicModelWrapper(SnapWaveModelWrapper):
    """
    One SnapWave case per time step of a boundary forcing time series.

    Build ``metamodel_parameters`` with
    :func:`~bluemath_tk.wrappers.snapwave.snapwave_utils.boundary_time_series`;
    dynamic templates are expected to read ``tref`` and the
    per-node ``hs_nodes``, ``tp_nodes``, ``dir_nodes``, ``spr_nodes`` (and
    optional ``wl_nodes``) lists.
    """

    default_parameters = {
        "tref": {
            "type": str,
            "value": None,
            "description": "Case time, 'YYYYmmdd HHMMSS'.",
        },
        **{
            f"{var}_nodes": {
                "type": list,
                "value": None,
                "description": f"{var} at each boundary node.",
            }
            for var in ("hs", "tp", "dir", "spr", "wl")
        },
        "save_map": _SAVE_MAP,
    }

"""
HyWaves metamodel for one goal: MDA -> per-variable PCA -> stacked exact GP.

Training data are stationary SnapWave runs from one goal (unit offshore Hs,
sampled Tp/Dir/Spr/WL), on ``(case_num, sites)``. The metamodel maps offshore
forcing to nearshore fields:

1. :func:`detect_removed_sites` / :func:`drop_sites` - optional site filter.
2. :func:`fit_mda` - MDA centroids of the forcing are the training cases (the
   rest is held out).
3. :func:`fit_pcas` - one PCA per predicted variable over the site fields
   (``dir`` as ``dir_u``/``dir_v``).
4. :func:`fit_gp` - an exact GP from the forcing to all stacked PCs.
5. :func:`predict_fields` - GP -> PCA inverse -> fields, with physical floors
   (:data:`MIN_HS`, :data:`MIN_DIRECTIONAL_SPREAD_DEG`).

Offshore hindcasts enter through :func:`goal_forcing` (bulk or one spectral
partition) and :func:`valid_timesteps`; predicted ``hs`` is a transfer
coefficient, scaled by the offshore Hs before summing goals (see
:mod:`.reconstruction`).

``vars_to_predict`` maps each output variable to ``"raw"`` (passed through
from the forcing, e.g. ``tp``) or ``{"vars_to_stack": [...], "pca_variance": f}``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ...core.operations import get_degrees_from_uv, in_nautical_sector
from ...datamining.mda import MDA
from ...datamining.pca import PCA
from ...interpolation.gps import ExactGPInterpolation

logger = logging.getLogger(__name__)

#: Bulk (whole-spectrum) forcing; spectral partitions are ``0, 1, 2, ...``.
BULK_PARTITION = -1

# Directional spreading is GP-predicted and can extrapolate to near-zero or
# negative values for rare forcing. wavespectra's Cartwright spreading uses
# s = 2 / deg2rad(dspr)**2 - 1: as dspr -> 0 the spreading underflows to zero
# on the whole direction grid and its normalisation divides by zero, turning
# the reconstructed spectrum (and any average over it) into inf/NaN.
MIN_DIRECTIONAL_SPREAD_DEG = 1.0

# Predicted ``hs`` is a per-goal transfer coefficient (scaled by offshore Hs),
# so it cannot be negative. The PCA inverse can still return small negative
# values at calm, sheltered sites (down to about -0.18 at Netherlands 5 m
# sites). Flooring keeps them as near-zero estimates and stops them adding
# spurious energy when goals are summed.
MIN_HS = 0.0

#: Default PCA NaN handling: fraction of NaN cases above which a site is
#: dropped, and the fill value for the remaining NaNs.
DEFAULT_PCA_PREPROCESSING: dict[str, dict[str, float]] = {
    "nan_threshold_to_drop": {"hs": 0.99, "dir_u": 0.99, "dir_v": 0.99, "spr": 0.99},
    "value_to_replace_nans": {"hs": 0.0, "dir_u": 0.0, "dir_v": 0.0, "spr": 0.0},
}

PcaVariance = float | dict[str, float] | None


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def detect_removed_sites(
    cases: xr.Dataset, site_filter: Mapping[str, float]
) -> list[str]:
    """
    Sites whose training fields break a ``{var}_max`` / ``{var}_min`` rule.

    Parameters
    ----------
    cases : xr.Dataset
        SnapWave cases on ``(case_num, sites)``.
    site_filter : mapping
        e.g. ``{"hs_max": 1.5}`` drops sites where ``hs`` exceeds 1.5 in any
        case; ``{"tp_min": 0}`` drops sites where ``tp`` is below 0 in any case.

    Returns
    -------
    list[str]
        Sorted labels of the sites to drop.
    """

    bad: set[str] = set()
    for key, threshold in site_filter.items():
        if key.endswith("_max"):
            var, op = key[:-4], "max"
        elif key.endswith("_min"):
            var, op = key[:-4], "min"
        else:
            logger.warning("Ignoring site_filter key %r (expected {var}_max/_min)", key)
            continue
        if var not in cases.data_vars:
            logger.warning("site_filter %r: variable %r not in cases", key, var)
            continue
        if op == "max":
            agg = cases[var].max("case_num")
            offenders = agg.where(agg > float(threshold), drop=True)
        else:
            agg = cases[var].min("case_num")
            offenders = agg.where(agg < float(threshold), drop=True)
        bad.update(str(s) for s in offenders.sites.values)

    return sorted(bad)


def drop_sites(cases: xr.Dataset, site_ids: Iterable[str]) -> xr.Dataset:
    """
    Remove sites by label.

    Parameters
    ----------
    cases : xr.Dataset
        Dataset with a ``sites`` coordinate.
    site_ids : iterable of str
        Labels to remove.

    Returns
    -------
    xr.Dataset
        The subset.
    """

    remove = {str(s) for s in site_ids}
    if not remove:
        return cases

    return cases.sel(sites=[s for s in cases.sites.values if str(s) not in remove])


def add_direction_components(cases: xr.Dataset) -> xr.Dataset:
    """
    Add ``dir_u``/``dir_v`` (unit-vector components of ``dir``) for PCA.

    Parameters
    ----------
    cases : xr.Dataset
        Dataset with a ``dir`` variable in degrees.

    Returns
    -------
    xr.Dataset
        The dataset with ``dir_u`` and ``dir_v``.
    """

    from ...core.operations import get_uv_components

    u, v = get_uv_components(cases["dir"])

    return cases.assign(dir_u=u, dir_v=v)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_mda(
    forcing: pd.DataFrame,
    num_centers: int,
    directional_variables: list[str],
) -> tuple[MDA, np.ndarray, np.ndarray]:
    """
    MDA selection of training cases; the rest are held out for validation.

    Parameters
    ----------
    forcing : pd.DataFrame
        Forcing of every case (rows = cases, columns = MDA inputs).
    num_centers : int
        Number of MDA centroids (largest training size you will fit).
    directional_variables : list of str
        Directional columns (degrees) of *forcing*.

    Returns
    -------
    tuple[MDA, np.ndarray, np.ndarray]
        Fitted MDA, training case indices (in MDA order, so the first ``n``
        give a size-``n`` design) and the sorted held-out indices.
    """

    mda = MDA(num_centers=num_centers)
    mda.fit(forcing, directional_variables=directional_variables)
    train_idx = np.asarray(mda.centroid_real_indices, dtype=int)
    test_idx = np.asarray(
        sorted(set(range(len(forcing))) - set(train_idx.tolist())), dtype=int
    )

    return mda, train_idx, test_idx


def pca_targets(
    vars_to_predict: Mapping[str, Any], pca_variance: PcaVariance = None
) -> list[tuple[str, float, list[str]]]:
    """
    PCA targets from a ``vars_to_predict`` specification.

    Parameters
    ----------
    vars_to_predict : mapping
        Variable -> ``"raw"`` or ``{"vars_to_stack", "pca_variance"}``.
    pca_variance : float or dict, optional
        Override of the explained variance (all variables, or per variable).

    Returns
    -------
    list[tuple[str, float, list[str]]]
        ``(name, explained_variance, vars_to_stack)`` for every non-raw variable.
    """

    out = []
    for name, spec in vars_to_predict.items():
        if spec == "raw":
            continue
        if isinstance(pca_variance, dict) and name in pca_variance:
            variance = float(pca_variance[name])
        elif isinstance(pca_variance, (int, float)):
            variance = float(pca_variance)
        else:
            variance = float(spec["pca_variance"])
        out.append((name, variance, list(spec["vars_to_stack"])))

    return out


def _pca_kept_sites(pca: PCA) -> dict[str, list[str]]:
    """Stacked variable -> site labels kept by the PCA NaN threshold."""

    sites = np.asarray(pca.coords_values.get("sites", []))
    if sites.size == 0:
        return {}

    return {
        var: [str(sites[i]) for i in np.asarray(positions, dtype=int)]
        for var, positions in getattr(pca, "not_nan_positions", {}).items()
    }


def fit_pcas(
    cases: xr.Dataset,
    targets: list[tuple[str, float, list[str]]],
    value_to_replace_nans: Mapping[str, float] | None = None,
    nan_threshold_to_drop: Mapping[str, float] | None = None,
) -> tuple[dict[str, PCA], dict[str, Any]]:
    """
    One PCA per target over the site fields of the training cases.

    Parameters
    ----------
    cases : xr.Dataset
        Training cases on ``(case_num, sites)`` (with ``dir_u``/``dir_v`` if
        ``dir`` is a target, see :func:`add_direction_components`).
    targets : list
        From :func:`pca_targets`.
    value_to_replace_nans, nan_threshold_to_drop : mapping, optional
        PCA NaN handling per stacked variable. Default
        :data:`DEFAULT_PCA_PREPROCESSING`.

    Returns
    -------
    tuple[dict[str, PCA], dict]
        Fitted PCAs by target, and a summary (sites kept per target and per
        stacked variable, number dropped).
    """

    replace = {
        **DEFAULT_PCA_PREPROCESSING["value_to_replace_nans"],
        **(value_to_replace_nans or {}),
    }
    drop = {
        **DEFAULT_PCA_PREPROCESSING["nan_threshold_to_drop"],
        **(nan_threshold_to_drop or {}),
    }
    n_sites = int(cases.sizes["sites"])
    pcas: dict[str, PCA] = {}
    summary: dict[str, Any] = {
        "pca_sites_kept": {},
        "pca_sites_kept_by_stack": {},
        "pca_sites_dropped": {},
    }
    for name, variance, stack in targets:
        pca = PCA(n_components=variance)
        pca.fit_transform(
            data=cases,
            vars_to_stack=stack,
            coords_to_stack=["sites"],
            pca_dim_for_rows="case_num",
            value_to_replace_nans=replace,
            nan_threshold_to_drop=drop,
        )
        kept_by_stack = _pca_kept_sites(pca)
        sets = [set(kept_by_stack.get(v, [])) for v in stack]
        kept = (
            sorted(set.intersection(*sets))
            if sets and sets[0]
            else kept_by_stack.get(stack[0], [])
        )
        summary["pca_sites_kept"][name] = kept
        summary["pca_sites_kept_by_stack"].update(
            {v: kept_by_stack[v] for v in stack if v in kept_by_stack}
        )
        summary["pca_sites_dropped"][name] = n_sites - len(kept)
        pcas[name] = pca

    return pcas, summary


def stack_pcs(pcas: Mapping[str, PCA]) -> pd.DataFrame:
    """
    Concatenate the PCs of every target into the GP target matrix.

    Parameters
    ----------
    pcas : mapping
        Fitted PCAs by target name.

    Returns
    -------
    pd.DataFrame
        Columns ``{target}__{pc}``, one row per training case.
    """

    frames = []
    for name, pca in pcas.items():
        df = pca.pcs_df.copy()
        df.columns = [f"{name}__{c}" for c in df.columns]
        frames.append(df)

    return pd.concat(frames, axis=1)


def split_pcs(pcs: np.ndarray, pcas: Mapping[str, PCA]) -> dict[str, np.ndarray]:
    """
    Split a stacked GP prediction back into per-target PC blocks.

    Parameters
    ----------
    pcs : np.ndarray
        Columns in :func:`stack_pcs` order.
    pcas : mapping
        Fitted PCAs by target name.

    Returns
    -------
    dict[str, np.ndarray]
        PC block per target.
    """

    blocks, offset = {}, 0
    for name, pca in pcas.items():
        k = pca.pcs_df.shape[1]
        blocks[name] = pcs[:, offset : offset + k]
        offset += k

    return blocks


def fit_gp(
    centroids: pd.DataFrame,
    pcas: Mapping[str, PCA],
    directional_variables: list[str],
    epochs: int = 1000,
) -> ExactGPInterpolation:
    """
    Exact GP from forcing (MDA centroids) to all stacked PCs.

    Parameters
    ----------
    centroids : pd.DataFrame
        Forcing of the training cases, in the same order as the PCA rows.
    pcas : mapping
        Fitted PCAs by target name (see :func:`fit_pcas`).
    directional_variables : list of str
        Directional columns of *centroids*.
    epochs : int, optional
        GP optimisation iterations. Default is 1000.

    Returns
    -------
    ExactGPInterpolation
        The fitted GP.
    """

    gp = ExactGPInterpolation(epochs=epochs)
    gp.fit(
        subset_data=centroids,
        subset_directional_variables=directional_variables,
        target_data=stack_pcs(pcas),
    )

    return gp


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_fields(
    gp: ExactGPInterpolation,
    pcas: Mapping[str, PCA],
    forcing: pd.DataFrame,
    vars_to_predict: Mapping[str, Any],
    sites: list[str],
) -> xr.Dataset:
    """
    Nearshore fields for each forcing row: GP -> PCA inverse.

    ``dir`` is rebuilt from ``dir_u``/``dir_v``; ``raw`` variables are copied
    from the forcing to every site; ``hs`` and ``spr`` are floored at
    :data:`MIN_HS` and :data:`MIN_DIRECTIONAL_SPREAD_DEG`.

    Parameters
    ----------
    gp : ExactGPInterpolation
        Fitted GP (see :func:`fit_gp`).
    pcas : mapping
        Fitted PCAs by target name.
    forcing : pd.DataFrame
        Forcing rows (columns as at fit time); the index labels the rows.
    vars_to_predict : mapping
        Same specification as at fit time.
    sites : list of str
        Sites to return (must be kept by every PCA).

    Returns
    -------
    xr.Dataset
        Predicted variables on ``(case_num, sites)``.
    """

    if forcing.index.name != "case_num":
        forcing = forcing.rename_axis("case_num")
    rows = forcing.index.values
    blocks = split_pcs(gp.predict(dataset=forcing).values, pcas)

    out: dict[str, xr.DataArray] = {}
    for name, spec in vars_to_predict.items():
        if spec == "raw":
            v = forcing[name].to_numpy()
            out[name] = xr.DataArray(
                np.broadcast_to(v[:, None], (len(v), len(sites))),
                dims=("case_num", "sites"),
                coords={"case_num": rows, "sites": sites},
            )
            continue
        fields = pcas[name].inverse_transform(
            xr.Dataset(
                {"PCs": (("case_num", "n_component"), blocks[name])},
                coords={"case_num": rows},
            )
        )
        if name == "dir":
            arr = xr.DataArray(
                get_degrees_from_uv(fields["dir_u"].values, fields["dir_v"].values),
                dims=("case_num", "sites"),
                coords={
                    "case_num": rows,
                    "sites": fields["dir_u"].coords["sites"].values,
                },
            )
        else:
            arr = fields[name]
            if "time" in arr.dims:
                arr = arr.rename(time="case_num")
        if name == "spr":
            arr = arr.clip(min=MIN_DIRECTIONAL_SPREAD_DEG)
        elif name == "hs":
            arr = arr.clip(min=MIN_HS)
        out[name] = arr.sel(sites=sites)

    return xr.Dataset(out)


# ---------------------------------------------------------------------------
# Offshore forcing
# ---------------------------------------------------------------------------


def goal_forcing(
    offshore_goal: xr.Dataset, partition: int = BULK_PARTITION, wl: float = 0.0
) -> tuple[pd.DataFrame, xr.DataArray]:
    """
    Metamodel forcing and offshore Hs of one goal, bulk or one partition.

    Parameters
    ----------
    offshore_goal : xr.Dataset
        Offshore time series at the goal: bulk ``hs``, ``tp``, ``dir``, ``spr``
        and partitions ``phs{i}``, ``ptp{i}``, ``pdir{i}``, ``pspr{i}``.
    partition : int, optional
        :data:`BULK_PARTITION` (default) or a partition number.
    wl : float, optional
        Constant water level added as the ``wl`` column. Default is 0.

    Returns
    -------
    tuple[pd.DataFrame, xr.DataArray]
        Forcing (``tp``, ``dir``, ``spr``, ``wl``) indexed by time, and the
        offshore Hs that scales the predicted transfer coefficient.
    """

    if "seapoint" in offshore_goal.coords:
        offshore_goal = offshore_goal.drop_vars("seapoint")
    if partition == BULK_PARTITION:
        part = offshore_goal[["hs", "tp", "dir", "spr"]]
    else:
        i = int(partition)
        part = offshore_goal[[f"phs{i}", f"ptp{i}", f"pdir{i}", f"pspr{i}"]].rename(
            {f"phs{i}": "hs", f"ptp{i}": "tp", f"pdir{i}": "dir", f"pspr{i}": "spr"}
        )
    df = part[["tp", "dir", "spr"]].to_dataframe()
    df["wl"] = wl

    return df, part["hs"]


def valid_timesteps(
    forcing: pd.DataFrame,
    hs_offshore: xr.DataArray,
    partition: int,
    required_columns: Iterable[str] = ("tp", "dir", "spr"),
    sector: tuple[float, float] | None = None,
) -> pd.Series:
    """
    Time steps the metamodel can reconstruct for one goal and partition.

    Parameters
    ----------
    forcing : pd.DataFrame
        Output of :func:`goal_forcing`.
    hs_offshore : xr.DataArray
        Offshore Hs of the same partition.
    partition : int
        For spectral partitions, steps without energy (``hs <= 0``) are dropped.
    required_columns : iterable of str, optional
        Forcing columns that must be finite.
    sector : tuple[float, float], optional
        Nautical ``(left, right)`` sector of the goal; directions outside it
        are dropped (that goal's metamodel was not trained there).

    Returns
    -------
    pd.Series
        Boolean mask indexed like *forcing*.
    """

    valid = forcing[list(required_columns)].notna().all(axis=1)
    if sector is not None:
        valid &= pd.Series(
            in_nautical_sector(forcing["dir"].to_numpy(), *sector), index=forcing.index
        )
    if partition == BULK_PARTITION:
        return valid

    row_dim = forcing.index.name or "time"
    hs = (
        hs_offshore
        if hs_offshore.dims[0] == row_dim
        else hs_offshore.rename({hs_offshore.dims[0]: row_dim})
    )
    hs_pos = hs.reindex({row_dim: forcing.index}).fillna(0).values > 0

    return valid & hs_pos

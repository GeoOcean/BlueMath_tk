#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import os
import os.path as op
import time
import pickle
from itertools import permutations
import glob
import shutil

# pip
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import ndtri  # norm inv
from scipy.stats import genextreme, gumbel_l, spearmanr, norm, weibull_min
from statsmodels.distributions.empirical_distribution import ECDF
from numpy.random import choice, multivariate_normal, randint, rand

# fix tqdm for notebook
from tqdm import tqdm as tqdm_base


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, "_instances"):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


# tk
from .statistical import Empirical_ICDF
from .extremes import FitGEV_KMA_Frechet, Smooth_GEV_Shape, ACOV
from .io.aux_nc import StoreBugXdset

from .database import clean_files
from .plotting.extremes import (
    Plot_GEVParams,
    Plot_ChromosomesProbs,
    Plot_SigmaCorrelation,
)


class Climate_Emulator(object):
    """
    KMA - DWTs Climate Emulator

    Fit Waves extremes model (GEV, Gumbel, Weibull distributions) and DWTs series
    Simulate (probabilistic) a new dataset of Waves from a custom DWTs series

    This custom emulator uses Waves data divided by families (sea, swell_1,
    swell_2, ...), each family is defined by 3 variables: Hs, Tp, Dir


    Some configuration parameters:

    A chromosomes methodology for statistical fit is available, it will handle
    on/off probabilities for each different waves family combination.

    Aditional variables, outside of waves families, can be given to the emulator.

    A variable has to be selected as the PROXY to calculate storms max. (default: AWL)

    User can manually set a variable for an empircal / weibull / gev distribution.
    """

    def __init__(self, p_base):
        # max. Total Water level for each storm data
        self.KMA_MS = None
        self.WVS_MS = None

        # Waves TCs WTs
        self.WVS_TCs = None

        # extremes model params
        self.GEV_Par = None  # GEV fitting parameters
        self.GEV_Par_S = None  # GEV simulation sampled parameters
        self.sigma = None  # Pearson sigma correlation

        # chromosomes
        self.do_chrom = True  # True for chromosomes combinations methodology
        self.chrom = None  # chromosomes and chromosomes probs

        # waves families and variables related parameters
        self.fams = []  # waves families to use in emulator
        self.vars_GEV = []  # variables handled with GEV fit
        self.vars_EMP = []  # varibles handled with Empirical fit
        self.vars_WBL = []  # varibles handled with Weibull fit
        self.extra_variables = []  # extra variables to use in emulator (no waves families)

        # to override a particular variable simulation distribution with a Empirical
        self.sim_icdf_empirical_override = []  # full variable name: "family_var_WT", example: ['swell_2_Hs_23', 'sea_Hs_15', ...]

        # simulated waves filter
        self.sim_waves_filter = {
            "hs": (0, 8),
            "tp": (2, 25),
            "ws": (0, 0.06),
        }

        # paths
        self.p_base = p_base
        self.p_config = op.join(p_base, "config.pk")
        self.p_WVS_MS = op.join(p_base, "WVS_MaxStorm.nc")
        self.p_KMA_MS = op.join(p_base, "KMA_MaxStorm.nc")
        self.p_WVS_TCs = op.join(p_base, "WVS_TCs.nc")
        self.p_chrom = op.join(p_base, "chromosomes.nc")
        self.p_GEV_Par = op.join(p_base, "GEV_Parameters.nc")
        self.p_GEV_Sigma = op.join(p_base, "GEV_SigmaCorrelation.nc")

        self.p_report_fit = op.join(p_base, "report_fit")
        self.p_report_sim = op.join(p_base, "report_sim")

        # output simulation default storage paths (folders)
        self.p_sims = op.join(self.p_base, "Simulations")  # folder
        self.p_sim_wvs_notcs = op.join(self.p_sims, "WAVES_noTCs")
        self.p_sim_wvs_tcs = op.join(self.p_sims, "WAVES_TCs")
        self.p_sim_tcs = op.join(self.p_sims, "TCs")

    def ConfigVariables(self, config):
        """
        Set waves families names
        Set optional extra variables names
        Set variables distribution: GEV, Empirical, Weibull
        Activate / Deactivate chromosomes methodology (if False, all-on combo)

        config = {
            'waves_families': ['sea', 'swell_1', ...],
            'extra_variables': ['wind', 'slp', ...]
            'distribution': [
                ('sea_Tp', 'Empirical'),
                ('wind', 'Weibull'),
                ...
            },
            'do_chromosomes': True,
        }
        """

        # get data from config dict
        fams = config["waves_families"]  # waves families to use

        # variables lists for each distribution
        l_GEV_vars = []  # GEV
        l_EMP_vars = []  # Empirical
        l_WBL_vars = []  # Weibull

        # Default Hs, Tp, Dir distributions
        GEV_vn = ["Hs", "Tp"]
        EMP_vn = ["Dir"]

        # mount family_variable lists
        for f in fams:
            for v in GEV_vn:
                l_GEV_vars.append("{0}_{1}".format(f, v))
            for v in EMP_vn:
                l_EMP_vars.append("{0}_{1}".format(f, v))

        # now add extra variables to GEV list
        if "extra_variables" in config.keys():
            for vn in config["extra_variables"]:
                l_GEV_vars.append("{0}".format(vn))

            # update extra variables parameter
            self.extra_variables = config["extra_variables"]

        # set custom distribution choices
        if "distribution" in config.keys():
            for vn, vd in config["distribution"]:
                # clean variable default distribution
                if vn in l_GEV_vars:
                    l_GEV_vars.pop(l_GEV_vars.index(vn))
                if vn in l_EMP_vars:
                    l_EMP_vars.pop(l_EMP_vars.index(vn))

                # now asign variable new distribution
                if vd == "GEV":
                    l_GEV_vars.append(vn)
                elif vd == "Empirical":
                    l_EMP_vars.append(vn)
                elif vd == "Weibull":
                    l_WBL_vars.append(vn)

        # chromosomes combination methodology option
        if "do_chromosomes" in config.keys():
            self.do_chrom = config["do_chromosomes"]

        # store configuration: families, variables distribution lists
        self.fams = fams
        self.vars_GEV = l_GEV_vars
        self.vars_EMP = l_EMP_vars
        self.vars_WBL = l_WBL_vars

        # log
        print("Waves Families: {0}".format(self.fams))
        print("Extra Variables: {0}".format(self.extra_variables))
        print("GEV distribution: {0}".format(self.vars_GEV))
        print("Empirical distribution: {0}".format(self.vars_EMP))
        print("Weibull distribution: {0}".format(self.vars_WBL))
        print("Do chromosomes combinations: {0}".format(self.do_chrom))

    def FitExtremes(self, KMA, WVS, config, proxy="AWL"):
        """
        GEV extremes fitting.
        Input data (waves vars series and bmus) shares time dimension

        KMA        - xarray.Dataset, vars: bmus (time,), cenEOFs(n_clusters, n_features)
        WVS        - xarray.Dataset: (time,), Hs, Tp, Dir, TC_category
                                     (time,), fam_V, {fam: sea, swell_1, ... V: Hs, Tp, Dir}
                                     (time,), extra_var1, extra_var2, ...
        config     - configuration dictionary: view self.ConfigVariables()
        proxy      - variable used to get DWTs max. storms (default AWL).
                     proxy variable has to be inside WVS xarray.Dataset
        """

        # configure waves fams variables parameters from config dict
        self.ConfigVariables(config)
        print("Max. Storms PROXY: {0}".format(proxy))

        # store TCs WTs waves
        WVS_TCs = WVS.copy()  # WVS.where(~np.isnan(WVS.TC_category), drop=True)

        # TODO select waves without TCs ?
        # WVS = WVS.where(np.isnan(WVS.TC_category), drop=True)

        # get start and end dates for each storm
        lt_storm_dates = self.Calc_StormsDates(KMA)

        # calculate max. PROXY variable for each storm
        ms_PROXY = self.Calc_StormsMaxProxy(WVS[proxy], lt_storm_dates)

        # select waves at storms max.
        ms_WVS = WVS.sel(time=ms_PROXY.time)
        ms_WVS[proxy] = ms_PROXY

        # reindex KMA, then select KMA data at storms max.
        KMA_rs = KMA.reindex(time=WVS.time, method="pad")
        ms_KMA = KMA_rs.sel(time=ms_PROXY.time)

        # GEV: Fit each wave family to a GEV distribution (KMA bmus)
        GEV_Par = self.Calc_GEVParams(ms_KMA, ms_WVS)

        # chromosomes combinations methodology
        if self.do_chrom:
            # calculate chromosomes and probabilities
            chromosomes = self.Calc_Chromosomes(ms_KMA, ms_WVS)

            # Calculate sigma spearman for each KMA - fams chromosome
            d_sigma = self.Calc_SigmaCorrelation(ms_KMA, ms_WVS, GEV_Par)

        else:
            # only one chromosome combination: all - on
            chromosomes = self.AllOn_Chromosomes(ms_KMA)

            # Calculate sigma spearman for each KMA (all chromosomes on)
            d_sigma = self.Calc_SigmaCorrelation_AllOn_Chromosomes(
                ms_KMA, ms_WVS, GEV_Par
            )

        # store data
        self.WVS_MS = ms_WVS
        self.WVS_TCs = WVS_TCs
        self.KMA_MS = ms_KMA
        self.GEV_Par = GEV_Par
        self.chrom = chromosomes
        self.sigma = d_sigma
        self.Save()

    def Save(self):
        "Saves fitted climate emulator data"

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        clean_files(
            [
                self.p_WVS_MS,
                self.p_WVS_TCs,
                self.p_KMA_MS,
                self.p_chrom,
                self.p_GEV_Par,
                self.p_GEV_Sigma,
                self.p_config,
            ]
        )

        # store .nc files
        self.WVS_MS.to_netcdf(self.p_WVS_MS)
        self.KMA_MS.to_netcdf(self.p_KMA_MS)
        self.WVS_TCs.to_netcdf(self.p_WVS_TCs)
        self.chrom.to_netcdf(self.p_chrom)
        self.GEV_Par.to_netcdf(self.p_GEV_Par)

        # store pickle
        pickle.dump(self.sigma, open(self.p_GEV_Sigma, "wb"))

        # store config
        pickle.dump(
            (
                self.fams,
                self.vars_GEV,
                self.vars_EMP,
                self.vars_WBL,
                self.extra_variables,
            ),
            open(self.p_config, "wb"),
        )

    def SaveSim(self, WVS_sim, TCs_sim, WVS_upd, n_sim):
        "Store waves and TCs simulation"

        # storage data and folders
        d_fs = [WVS_sim, TCs_sim, WVS_upd]
        p_fs = [self.p_sim_wvs_notcs, self.p_sim_tcs, self.p_sim_wvs_tcs]

        # store each simulation at different nc file
        for d, p in zip(d_fs, p_fs):
            if not op.isdir(p):
                os.makedirs(p)

            nm = "{0:08d}.nc".format(n_sim)  # sim code
            StoreBugXdset(d, op.join(p, nm))

    def Load(self):
        "Loads fitted climate emulator data"

        # store .nc files
        self.WVS_MS = xr.open_dataset(self.p_WVS_MS)
        self.KMA_MS = xr.open_dataset(self.p_KMA_MS)
        self.WVS_TCs = xr.open_dataset(self.p_WVS_TCs)
        self.chrom = xr.open_dataset(self.p_chrom)
        self.GEV_Par = xr.open_dataset(self.p_GEV_Par)

        # store pickle
        self.sigma = pickle.load(open(self.p_GEV_Sigma, "rb"))

        # load config
        self.fams, self.vars_GEV, self.vars_EMP, self.vars_WBL, self.extra_variables = (
            pickle.load(open(self.p_config, "rb"))
        )

    def LoadSim(self, n_sim=0):
        "Load waves and TCs simulations"

        WVS_sim, TCs_sim, WVS_upd = None, None, None

        nm = "{0:08d}.nc".format(n_sim)  # sim code
        p_WVS_sim = op.join(self.p_sim_wvs_notcs, nm)
        p_TCS_sim = op.join(self.p_sim_tcs, nm)
        p_WVS_upd = op.join(self.p_sim_wvs_tcs, nm)

        if op.isfile(p_WVS_sim):
            WVS_sim = xr.open_dataset(p_WVS_sim)

        if op.isfile(p_TCS_sim):
            TCs_sim = xr.open_dataset(p_TCS_sim)

        if op.isfile(p_WVS_upd):
            WVS_upd = xr.open_dataset(p_WVS_upd)

        return WVS_sim, TCs_sim, WVS_upd

    def LoadSim_All(self, n_sim_ce=0, TCs=True):
        """
        Load all waves and TCs (1 DWT -> 1 output) simulations.

        Each max. storm simulation has a different time dimension,
        returns a pandas.DataFrame with added columns 'time' and 'n_sim'

        output will merge WVS_upd and TCs_sim data variables

        a unique n_sim_ce (inner climate emulator simulation) has to be chosen.

        TCs - True / False. Load WVS (TCs updated) + TCs / WVS (without TCs) simulations
        """

        # count available waves simulations
        n_sims_DWTs = len(glob.glob(op.join(self.p_sim_wvs_tcs, "*.nc")))

        # iterate over simulations
        l_sims = []
        for n in range(n_sims_DWTs):
            if TCs:
                # Load simulated Waves (TCs updated) and TCs
                _, TCs_sim, WVS_upd = self.LoadSim(n_sim=n)

                pd_sim = xr.merge(
                    [WVS_upd.isel(n_sim=n_sim_ce), TCs_sim.isel(n_sim=n_sim_ce)]
                ).to_dataframe()

            else:
                # Load simulated Waves (without TCs)
                WVS_sim, _, _ = self.LoadSim(n_sim=n)

                pd_sim = WVS_sim.isel(n_sim=n_sim_ce).to_dataframe()

            # add columns
            pd_sim["n_sim"] = n  # store simulation index with data
            pd_sim["time"] = pd_sim.index  # store dates as a variable

            l_sims.append(pd_sim)

        # join all waves simulations
        all_sims = pd.concat(l_sims, ignore_index=True)

        return all_sims

    def Set_Simulation_Folder(self, p_sim, copy_WAVES_noTCs=False):
        """
        Modifies climate emulator default simulation path

        p_sim - new simulation path
        copy_WAVES_noTCs - copies simulated waves (no TCs) from original path
        """

        # store base path
        p_base = self.p_sims

        # update simulation and files paths
        self.p_sims = p_sim
        self.p_sim_wvs_notcs = op.join(self.p_sims, "WAVES_noTCs")
        self.p_sim_wvs_tcs = op.join(self.p_sims, "WAVES_TCs")
        self.p_sim_tcs = op.join(self.p_sims, "TCs")

        # optional copy files
        if copy_WAVES_noTCs:
            if op.isdir(self.p_sim_wvs_notcs):
                shutil.rmtree(self.p_sim_wvs_notcs)
            shutil.copytree(op.join(p_base, "WAVES_noTCs"), self.p_sim_wvs_notcs)

    def Calc_StormsDates(self, xds_KMA):
        "Returns list of tuples with each storm start and end times"

        # locate dates where KMA WT changes (bmus series)
        bmus_diff = np.diff(xds_KMA.bmus.values)
        ix_ch = np.where((bmus_diff != 0))[0] + 1
        ix_ch = np.insert(ix_ch, 0, 0)
        ds_ch = xds_KMA.time.values[ix_ch]  # dates where WT changes

        # list of tuples with (date start, date end) for each storm (WT window)
        dates_tup_WT = [
            (ds_ch[c], ds_ch[c + 1] - np.timedelta64(1, "D"))
            for c in range(len(ds_ch) - 1)
        ]
        dates_tup_WT.append(
            (dates_tup_WT[-1][1] + np.timedelta64(1, "D"), xds_KMA.time.values[-1])
        )

        return dates_tup_WT

    def Calc_StormsMaxProxy(self, wvs_PROXY, lt_storm_dates):
        "Returns xarray.Dataset with max. PROXY variable value and time"

        # find max PROXY inside each storm
        ms_PROXY = []
        ms_times = []
        for d1, d2 in lt_storm_dates:
            # get TWL inside WT window
            wt_PROXY = wvs_PROXY.sel(time=slice(d1, d2 + np.timedelta64(23, "h")))[:]

            # get window maximum TWL date
            wt_PROXY_max = wt_PROXY.where(
                wt_PROXY == wt_PROXY.max(), drop=True
            ).squeeze()

            # append data
            ms_PROXY.append(wt_PROXY_max.values)
            ms_times.append(wt_PROXY_max.time.values)

        return wvs_PROXY.sel(time=ms_times)

    def Calc_GEVParams(self, xds_KMA_MS, xds_WVS_MS):
        """
        Fits each WT (KMA.bmus) waves families data to a GEV distribtion
        Requires KMA and WVS families at storms max. TWL

        Returns xarray.Dataset with GEV shape, location and scale parameters
        """

        vars_gev = self.vars_GEV
        bmus = xds_KMA_MS.bmus.values[:]
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)

        xds_GEV_Par = xr.Dataset(
            coords={
                "n_cluster": np.arange(n_clusters) + 1,
                "parameter": ["shape", "location", "scale"],
            }
        )

        # Fit each wave family var to GEV distribution (using KMA bmus)
        for vn in vars_gev:
            gp_pars = FitGEV_KMA_Frechet(bmus, n_clusters, xds_WVS_MS[vn].values[:])
            xds_GEV_Par[vn] = (
                (
                    "n_cluster",
                    "parameter",
                ),
                gp_pars,
            )

        return xds_GEV_Par

    def Calc_Chromosomes(self, xds_KMA_MS, xds_WVS_MS):
        """
        Calculate chromosomes and probabilities from KMA.bmus data

        Returns xarray.Dataset vars: chrom, chrom_probs. dims: WT, wave_family
        """

        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        fams_chrom = self.fams
        l_vc = ["{0}_Hs".format(x) for x in fams_chrom]

        # get chromosomes matrix
        np_vc = np.column_stack([xds_WVS_MS[vn].values for vn in l_vc])
        chrom = ChromMatrix(np_vc)

        # calculate chromosomes probabilities
        probs = np.zeros((n_clusters, chrom.shape[0]))
        for i in range(n_clusters):
            c = i + 1
            pos = np.where((bmus == c))[0]

            # get variables chromosomes at cluster
            var_c = np_vc[pos, :]
            var_c[~np.isnan(var_c)] = 1
            var_c[np.isnan(var_c)] = 0

            # count chromosomes
            ucs, ccs = np.unique(var_c, return_counts=True, axis=0)
            tcs = var_c.shape[0]

            # get probs of each chromosome
            for uc, cc in zip(ucs, ccs):
                # skip all empty chromosomes
                if ~uc.any():
                    continue

                pc = np.where(np.all(uc == chrom, axis=1))[0][0]
                probs[i, pc] = cc / tcs

        # chromosomes dataset
        return xr.Dataset(
            {
                "chrom": (
                    (
                        "n",
                        "wave_family",
                    ),
                    chrom,
                ),
                "probs": (
                    (
                        "WT",
                        "n",
                    ),
                    probs,
                ),
            },
            coords={
                "WT": np.arange(n_clusters) + 1,
                "wave_family": fams_chrom,
            },
        )

    def AllOn_Chromosomes(self, xds_KMA_MS):
        """
        Generate Fake chromosomes and probabilities from KMA.bmus data in order
        to use only all-on chromosome combination

        Returns xarray.Dataset vars: chrom, chrom_probs. dims: WT, wave_family
        """

        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        fams_chrom = self.fams

        # fake chromosomes and probabilities
        chrom = np.ones((1, len(fams_chrom)))
        probs = np.ones((n_clusters, 1))

        # chromosomes dataset
        return xr.Dataset(
            {
                "chrom": (
                    (
                        "n",
                        "wave_family",
                    ),
                    chrom,
                ),
                "probs": (
                    (
                        "WT",
                        "n",
                    ),
                    probs,
                ),
            },
            coords={
                "WT": np.arange(n_clusters) + 1,
                "wave_family": fams_chrom,
            },
        )

    def Calc_SigmaCorrelation(self, xds_KMA_MS, xds_WVS_MS, xds_GEV_Par):
        "Calculate Sigma Pearson correlation for each WT-chromosome combo"

        bmus = xds_KMA_MS.bmus.values[:]
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        wvs_fams = self.fams
        vars_extra = self.extra_variables
        vars_GEV = self.vars_GEV

        # smooth GEV shape parameter
        d_shape = {}
        for vn in vars_GEV:
            sh_GEV = xds_GEV_Par.sel(parameter="shape")[vn].values[:]
            d_shape[vn] = Smooth_GEV_Shape(cenEOFs, sh_GEV)

        # Get sigma correlation for each KMA cluster
        d_sigma = {}  # nested dict [WT][crom]
        for iwt in range(n_clusters):
            c = iwt + 1
            pos = np.where((bmus == c))[0]
            d_sigma[c] = {}

            # current cluster waves
            xds_K_wvs = xds_WVS_MS.isel(time=pos)

            # get chromosomes from waves (0/1)
            var_c = np.column_stack(
                [xds_K_wvs["{0}_Hs".format(x)].values[:] for x in wvs_fams]
            )
            var_c[~np.isnan(var_c)] = 1
            var_c[np.isnan(var_c)] = 0
            chrom = ChromMatrix(var_c)

            # get sigma for each chromosome
            for ucix, uc in enumerate(chrom):
                wt_crom = 1  # data / no data

                # find data position for this chromosome
                p_c = np.where((var_c == uc).all(axis=1))[0]

                # if not enought data, get all chromosomes with shared 1s
                if len(p_c) < 20:
                    p1s = np.where(uc == 1)[0]
                    p_c = np.where((var_c[:, p1s] == uc[p1s]).all(axis=1))[0]

                    wt_crom = 0  # data / no data

                # select waves chrom data
                xds_chr_wvs = xds_K_wvs.isel(time=p_c)

                # solve normal inverse GEV/EMP/WBL CDF for each active chromosome
                to_corr = np.empty((0, len(p_c)))  # append for spearman correlation
                for i_c in np.where(uc == 1)[0]:
                    # get wave family chromosome variables
                    fam_n = wvs_fams[i_c]
                    vn_Hs = "{0}_Hs".format(fam_n)
                    vn_Tp = "{0}_Tp".format(fam_n)
                    vn_Dir = "{0}_Dir".format(fam_n)

                    vv_Hs = xds_chr_wvs[vn_Hs].values[:]
                    vv_Tp = xds_chr_wvs[vn_Tp].values[:]
                    vv_Dir = xds_chr_wvs[vn_Dir].values[:]

                    # Hs
                    norm_Hs = self.CDF_Distribution(
                        vn_Hs, vv_Hs, xds_GEV_Par, d_shape, iwt
                    )

                    # Tp
                    norm_Tp = self.CDF_Distribution(
                        vn_Tp, vv_Tp, xds_GEV_Par, d_shape, iwt
                    )

                    # Dir
                    norm_Dir = self.CDF_Distribution(
                        vn_Dir, vv_Dir, xds_GEV_Par, d_shape, iwt
                    )

                    # normal inverse CDF
                    u_cdf = np.column_stack([norm_Hs, norm_Tp, norm_Dir])
                    u_cdf[u_cdf >= 1.0] = 0.999999
                    inv_n = ndtri(u_cdf)

                    # concatenate data for correlation
                    to_corr = np.concatenate((to_corr, inv_n.T), axis=0)

                # concatenate extra variables for correlation
                for vn in vars_extra:
                    vv = xds_chr_wvs[vn].values[:]

                    norm_vn = self.CDF_Distribution(vn, vv, xds_GEV_Par, d_shape, iwt)
                    norm_vn[norm_vn >= 1.0] = 0.999999

                    inv_n = ndtri(norm_vn)
                    to_corr = np.concatenate((to_corr, inv_n[:, None].T), axis=0)

                # sigma: spearman correlation
                corr, pval = spearmanr(to_corr, axis=1)

                # store data at dict
                d_sigma[c][ucix] = {"corr": corr, "data": len(p_c), "wt_crom": wt_crom}

        return d_sigma

    def Calc_SigmaCorrelation_AllOn_Chromosomes(
        self, xds_KMA_MS, xds_WVS_MS, xds_GEV_Par
    ):
        "Calculate Sigma Pearson correlation for each WT, all on chrom combo"

        bmus = xds_KMA_MS.bmus.values[:]
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        wvs_fams = self.fams
        vars_extra = self.extra_variables
        vars_GEV = self.vars_GEV

        # smooth GEV shape parameter
        d_shape = {}
        for vn in vars_GEV:
            sh_GEV = xds_GEV_Par.sel(parameter="shape")[vn].values[:]
            d_shape[vn] = Smooth_GEV_Shape(cenEOFs, sh_GEV)

        # Get sigma correlation for each KMA cluster
        d_sigma = {}  # nested dict [WT][crom]
        for iwt in range(n_clusters):
            c = iwt + 1
            pos = np.where((bmus == c))[0]
            d_sigma[c] = {}

            # current cluster waves
            xds_K_wvs = xds_WVS_MS.isel(time=pos)

            # append data for spearman correlation
            to_corr = np.empty((0, len(xds_K_wvs.time)))

            # solve normal inverse GEV/EMP/WBL CDF for each waves family
            for fam_n in wvs_fams:
                # get wave family variables
                vn_Hs = "{0}_Hs".format(fam_n)
                vn_Tp = "{0}_Tp".format(fam_n)
                vn_Dir = "{0}_Dir".format(fam_n)

                vv_Hs = xds_K_wvs[vn_Hs].values[:]
                vv_Tp = xds_K_wvs[vn_Tp].values[:]
                vv_Dir = xds_K_wvs[vn_Dir].values[:]

                # fix fams nan: Hs 0, Tp mean, dir mean
                p_nans = np.where(np.isnan(vv_Hs))[0]
                vv_Hs[p_nans] = 0
                vv_Tp[p_nans] = np.nanmean(vv_Tp)
                vv_Dir[p_nans] = np.nanmean(vv_Dir)

                # Hs
                norm_Hs = self.CDF_Distribution(vn_Hs, vv_Hs, xds_GEV_Par, d_shape, iwt)

                # Tp
                norm_Tp = self.CDF_Distribution(vn_Tp, vv_Tp, xds_GEV_Par, d_shape, iwt)

                # Dir
                norm_Dir = self.CDF_Distribution(
                    vn_Dir, vv_Dir, xds_GEV_Par, d_shape, iwt
                )

                # normal inverse CDF
                u_cdf = np.column_stack([norm_Hs, norm_Tp, norm_Dir])
                u_cdf[u_cdf >= 1.0] = 0.999999
                inv_n = ndtri(u_cdf)

                # concatenate data for correlation
                to_corr = np.concatenate((to_corr, inv_n.T), axis=0)

            # concatenate extra variables for correlation
            for vn in vars_extra:
                vv = xds_K_wvs[vn].values[:]

                norm_vn = self.CDF_Distribution(vn, vv, xds_GEV_Par, d_shape, iwt)
                norm_vn[norm_vn >= 1.0] = 0.999999

                inv_n = ndtri(norm_vn)
                to_corr = np.concatenate((to_corr, inv_n[:, None].T), axis=0)

            # sigma: spearman correlation
            corr, pval = spearmanr(to_corr, axis=1)

            # store data at dict (keep cromosomes structure)
            d_sigma[c][0] = {"corr": corr, "data": len(xds_K_wvs.time), "wt_crom": 1}

        return d_sigma

    def GEV_Parameters_Sampling(self, n_sims):
        """
        Sample new GEV/GUMBELL parameters using GEV/GUMBELL asymptotic variances

        num_sims  - number of GEV parameters to sample
        """

        xds_GEV_Par = self.GEV_Par
        vars_gev = self.vars_GEV
        xds_KMA_MS = self.KMA_MS
        xds_WVS_MS = self.WVS_MS

        # get KMA data
        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        cenEOFs = xds_KMA_MS.cenEOFs.values[:]

        # dataset for storing parameters
        xds_par_samp = xr.Dataset(
            {},
            coords={
                "parameter": ["shape", "location", "scale"],
                "n_cluster": np.arange(n_clusters) + 1,
                "simulation": range(n_sims),
            },
        )

        # simulate variables
        for vn in vars_gev:
            # GEV/GUMBELL parameters
            pars_GEV = xds_GEV_Par[vn]
            sha = pars_GEV.sel(parameter="shape").values[:]
            sca = pars_GEV.sel(parameter="scale").values[:]
            loc = pars_GEV.sel(parameter="location").values[:]

            # location parameter Extremal Index (Gev)
            index = np.ones(sha.shape)
            mu_b = loc - (sca / sha) * (1 - np.power(index, sha))
            psi_b = sca * np.power(index, sha)

            # location parameter Extremal Index (Gumbell)
            sha_gbl = 0.0000000001
            pos_gbl = np.where(sha == sha_gbl)[0]
            cls_gbl = pos_gbl + 1  # Gumbell Weather Types

            # update mu_b
            mu_b[pos_gbl] = loc[pos_gbl] + sca[pos_gbl] * np.log(index[pos_gbl])

            # output holder
            out_ps = np.ndarray((n_clusters, n_sims, 3)) * np.nan

            # sample Gumbel or GEV parameters for each WT
            for i in range(n_clusters):
                c = i + 1  # WT ID

                # get var values at cluster and remove nans
                p_bmus = np.where((bmus == c))[0]
                var_wvs = xds_WVS_MS[vn].isel(time=p_bmus).values[:]
                var_wvs = var_wvs[~np.isnan(var_wvs)]

                # Gumbel WTs: parameters sampling
                if c in cls_gbl:
                    # GUMBELL Loglikelihood function acov
                    theta = (loc[i], sca[i])
                    acov = ACOV(gumbel_l.nnlf, theta, var_wvs)

                    # GUMBELL params used for multivar. normal random generation
                    theta_gen = np.array([mu_b[i], sca[i]])
                    theta_gbl = multivariate_normal(theta_gen, acov, n_sims)

                    # mount "GEV" params for simulation
                    theta_sim = np.ones((n_sims, 3)) * sha_gbl
                    theta_sim[:, 1:] = theta_gbl

                # GEV WTs: parameters sampling
                else:
                    # GEV Loglikelihood function acov
                    theta = (sha[i], loc[i], sca[i])
                    acov = ACOV(genextreme.nnlf, theta, var_wvs)

                    # GEV params used for multivar. normal random generation
                    theta_gen = np.array([sha[i], mu_b[i], psi_b[i]])
                    theta_sim = multivariate_normal(theta_gen, acov, n_sims)

                # store sampled GEV/GUMBELL params
                out_ps[i, :, :] = theta_sim[:, :]

            # smooth shape parameter
            for j in range(n_sims):
                shape_wts = out_ps[:, j, 0]
                out_ps[:, j, 0] = Smooth_GEV_Shape(cenEOFs, shape_wts)

            # append output to dataset
            xds_par_samp[vn] = (("n_cluster", "simulation", "parameter"), out_ps)

        return xds_par_samp

    def Simulate_Waves(
        self, xds_DWT, n_sims=1, filters={"hs": False, "tp": False, "ws": False}
    ):
        """
        Climate Emulator DWTs waves simulation

        xds_DWT          - xarray.Dataset, vars: evbmus_sims (time,)
        n_sims           - number of simulations to compute
        filters          - filter simulated waves by hs, tp, and/or wave setpness
        """

        # TODO can optimize?

        # max. storm waves and KMA
        xds_KMA_MS = self.KMA_MS
        xds_WVS_MS = self.WVS_MS
        xds_WVS_TCs = self.WVS_TCs
        xds_chrom = self.chrom
        xds_GEV_Par = self.GEV_Par
        sigma = self.sigma

        # vars needed
        dwt_bmus_sim = xds_DWT.evbmus_sims.values[:]
        dwt_time_sim = xds_DWT.time.values[:]
        bmus = xds_KMA_MS.bmus.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)
        chrom = xds_chrom.chrom.values[:]
        chrom_probs = xds_chrom.probs.values[:]

        # iterate n_sims
        ls_wvs_sim = []
        for i_sim in range(n_sims):
            # get number of simulations
            idw, iuc = np.unique(dwt_bmus_sim, return_counts=True)
            num_gev_sims = np.max(iuc)

            # Sample GEV/GUMBELL parameters
            xds_GEV_Par_Sampled = self.GEV_Parameters_Sampling(num_gev_sims)

            # generate waves
            wvs_sim = self.GenerateWaves(
                bmus,
                n_clusters,
                chrom,
                chrom_probs,
                sigma,
                xds_WVS_MS,
                xds_WVS_TCs,
                xds_GEV_Par_Sampled,
                dwt_bmus_sim,
                dwt_time_sim,
                filters=filters,
            )
            ls_wvs_sim.append(wvs_sim)

        # concatenate simulations
        WVS_sims = xr.concat(ls_wvs_sim, "n_sim")

        return WVS_sims

    def Simulate_TCs(
        self,
        xds_DWT,
        WVS_sims,
        xds_TCs_params,
        xds_TCs_simulation,
        prob_change_TCs,
        MU_WT,
        TAU_WT,
        extra_vars_update=[],
    ):
        """
        Climate Emulator DWTs TCs simulation

        xds_DWT             - xarray.Dataset, vars: evbmus_sims (time,)
        WVS_sim             - xarray.Dataset, output from Simulate_Waves()

        xds_TCs_params      - xr.Dataset. vars(storm): pressure_min
        xds_TCs_simulation  - xr.Dataset. vars(storm): mu, hs, ss, tp, dir
        prob_change_TCs     - cumulative probabilities of TC category change
        MU_WT, TAU_WT       - intradaily hidrographs for each WT

        extra_vars_update - list(string), optional extra variables to update
        with value from "xds_TCs_simulation"
        """

        # max. storm waves and KMA
        xds_KMA_MS = self.KMA_MS

        # vars needed
        dwt_bmus_sim = xds_DWT.evbmus_sims.values[:]
        dwt_time_sim = xds_DWT.time.values[:]
        n_clusters = len(xds_KMA_MS.n_clusters)

        # iterate waves simulations
        ls_tcs_sim = []
        ls_wvs_upd = []
        for i_sim in WVS_sims.n_sim:
            wvs_s = WVS_sims.sel(n_sim=i_sim)

            # generate TCs
            tcs_sim, wvs_upd_sim = self.GenerateTCs(
                n_clusters,
                dwt_bmus_sim,
                dwt_time_sim,
                xds_TCs_params,
                xds_TCs_simulation,
                prob_change_TCs,
                MU_WT,
                TAU_WT,
                wvs_s,
                extra_vars_update=extra_vars_update,
            )
            ls_tcs_sim.append(tcs_sim)
            ls_wvs_upd.append(wvs_upd_sim)

        # concatenate simulations
        TCs_sim = xr.concat(ls_tcs_sim, "n_sim")
        WVS_upd = xr.concat(ls_wvs_upd, "n_sim")

        return TCs_sim, WVS_upd

    def CDF_Distribution(self, vn, vv, xds_GEV_Par, d_shape, i_wt):
        """
        Switch function: GEV / Empirical / Weibull

        Check variable distribution and calculates CDF

        vn - var name
        vv - var value
        i_wt - Weather Type index
        xds_GEV_Par , d_shape: GEV data used in sigma correlation
        """

        # get GEV / EMPIRICAL / WEIBULL variables list
        vars_GEV = self.vars_GEV
        vars_EMP = self.vars_EMP
        vars_WBL = self.vars_WBL

        # switch variable name
        if vn in vars_GEV:
            # gev CDF
            sha_g = d_shape[vn][i_wt]
            loc_g = xds_GEV_Par.sel(parameter="location")[vn].values[i_wt]
            sca_g = xds_GEV_Par.sel(parameter="scale")[vn].values[i_wt]
            norm_VV = genextreme.cdf(vv, -1 * sha_g, loc_g, sca_g)

        elif vn in vars_EMP:
            # empirical CDF
            ecdf = ECDF(vv)
            norm_VV = ecdf(vv)

        elif vn in vars_WBL:
            # Weibull CDF
            norm_VV = weibull_min.cdf(vv, *weibull_min.fit(vv))

        return norm_VV

    def ICDF_Distribution(self, vn, vv, pb, xds_GEV_Par, i_wt):
        """
        Switch function: GEV / Empirical / Weibull

        Check variable distribution and calculates ICDF

        vn - var name
        vv - var value
        pb - var simulation probs
        i_wt - Weather Type index
        xds_GEV_Par: GEV parameters
        """

        # optional empirical var_wt override
        fv = "{0}_{1}".format(vn, i_wt + 1)
        if fv in self.sim_icdf_empirical_override:
            ppf_VV = Empirical_ICDF(vv, pb)
            return ppf_VV

        # get GEV / EMPIRICAL / WEIBULL variables list
        vars_GEV = self.vars_GEV
        vars_EMP = self.vars_EMP
        vars_WBL = self.vars_WBL

        # switch variable name
        if vn in vars_GEV:
            # gev ICDF
            sha_g = xds_GEV_Par.sel(parameter="shape")[vn].values[i_wt]
            loc_g = xds_GEV_Par.sel(parameter="location")[vn].values[i_wt]
            sca_g = xds_GEV_Par.sel(parameter="scale")[vn].values[i_wt]
            ppf_VV = genextreme.ppf(pb, -1 * sha_g, loc_g, sca_g)

        elif vn in vars_EMP:
            # empirical ICDF
            ppf_VV = Empirical_ICDF(vv, pb)

        elif vn in vars_WBL:
            # Weibull ICDF
            ppf_VV = weibull_min.ppf(pb, *weibull_min.fit(vv))

        return ppf_VV

    def GenerateWaves(
        self,
        bmus,
        n_clusters,
        chrom,
        chrom_probs,
        sigma,
        xds_WVS_MS,
        xds_WVS_TCs,
        xds_GEV_Par_Sampled,
        DWT,
        DWT_time,
        filters={"hs": False, "tp": False, "ws": False},
    ):
        """
        Climate Emulator DWTs waves simulation

        bmus                 - KMA max. storms bmus series
        n_clusters           - KMA number of clusters
        chrom, chrom_probs   - chromosomes and probabilities
        sigma                - pearson correlation for each WT
        xds_GEV_Par_Sampled  - GEV/GUMBELL parameters sampled for simulation
        DWT                  - np.array with DWT bmus sim series (dims: time,)
        filters              - filter simulated waves by hs, tp, and/or wave setpness

        Returns xarray.Dataset with simulated storm data
            vars:
                *fam*_*vn* (fam: sea, swell_1, swell_2, vn: Hs, Tp, Dir),
                DWT_sim
            dims: storm
        """

        # waves parameters filters
        hs_min, hs_max = self.sim_waves_filter["hs"]
        tp_min, tp_max = self.sim_waves_filter["tp"]
        ws_min, ws_max = self.sim_waves_filter["ws"]

        # waves families - variables (sorted for simulation output)
        wvs_fams = self.fams
        wvs_fams_vars = [
            ("{0}_{1}".format(f, vn)) for f in wvs_fams for vn in ["Hs", "Tp", "Dir"]
        ]

        # extra variables (optional)
        vars_extra = self.extra_variables

        # simulate one value for each storm
        dwt_df = np.diff(DWT)
        dwt_df[-1] = 1  # ensure last day storm
        ix_ch = np.where((dwt_df != 0))[0] + 1
        ix_ch = np.insert(ix_ch, 0, 0)  # get first day storm
        DWT_sim = DWT[ix_ch]
        DWT_time_sim = DWT_time[ix_ch]

        # new progress bar
        pbar = tqdm(total=len(DWT_sim), desc="C.E: Sim. Waves")

        # Simulate
        srl = len(wvs_fams) * 3 + len(vars_extra)  # simulation row length
        sims_out = np.zeros((len(DWT_sim), srl))
        c = 0
        while c < len(DWT_sim):
            WT = int(DWT_sim[c])
            iwt = WT - 1

            # KMA Weather Types waves generation
            if WT <= n_clusters:
                # get random chromosome (weigthed choice)
                pr = chrom_probs[iwt] / np.sum(chrom_probs[iwt])
                ci = choice(range(chrom.shape[0]), 1, p=pr)
                crm = chrom[ci].astype(int).squeeze()

                # get sigma correlation for this WT - crm combination
                corr = sigma[WT][int(ci)]["corr"]
                mvn_m = np.zeros(corr.shape[0])
                sims = multivariate_normal(mvn_m, corr)
                prob_sim = norm.cdf(sims, 0, 1)

                # solve normal inverse CDF for each active chromosome
                ipbs = 0  # prob_sim aux. index
                sim_row = np.zeros(srl)
                for i_c in np.where(np.atleast_1d(crm) == 1)[0]:
                    # random sampled GEV
                    rd = np.random.randint(0, len(xds_GEV_Par_Sampled.simulation))
                    xds_GEV_Par = xds_GEV_Par_Sampled.isel(simulation=rd)

                    # get wave family chromosome variables
                    fam_n = wvs_fams[i_c]
                    vn_Hs = "{0}_Hs".format(fam_n)
                    vn_Tp = "{0}_Tp".format(fam_n)
                    vn_Dir = "{0}_Dir".format(fam_n)

                    # use WAVES that are not from TCs
                    vv_Hs = xds_WVS_MS[vn_Hs].values[:]
                    vv_Tp = xds_WVS_MS[vn_Tp].values[:]
                    vv_Dir = xds_WVS_MS[vn_Dir].values[:]

                    pb_Hs = prob_sim[ipbs + 0]
                    pb_Tp = prob_sim[ipbs + 1]
                    pb_Dir = prob_sim[ipbs + 2]
                    ipbs += 3

                    # Hs
                    ppf_Hs = self.ICDF_Distribution(
                        vn_Hs, vv_Hs, pb_Hs, xds_GEV_Par, iwt
                    )

                    # Tp
                    ppf_Tp = self.ICDF_Distribution(
                        vn_Tp, vv_Tp, pb_Tp, xds_GEV_Par, iwt
                    )

                    # Dir
                    ppf_Dir = self.ICDF_Distribution(
                        vn_Dir, vv_Dir, pb_Dir, xds_GEV_Par, iwt
                    )

                    # store simulation data
                    is0, is1 = (
                        wvs_fams.index(fam_n) * 3,
                        (wvs_fams.index(fam_n) + 1) * 3,
                    )
                    sim_row[is0:is1] = [ppf_Hs, ppf_Tp, ppf_Dir]

                # solve normal inverse CDF for each extra variable
                ipbs = len(wvs_fams) * 3
                for vn in vars_extra:
                    # random sampled GEV
                    rd = np.random.randint(0, len(xds_GEV_Par_Sampled.simulation))
                    xds_GEV_Par = xds_GEV_Par_Sampled.isel(simulation=rd)

                    vv = xds_WVS_MS[vn].values[:]
                    pb_vv = prob_sim[ipbs]
                    ppf_vv = self.ICDF_Distribution(vn, vv, pb_vv, xds_GEV_Par, iwt)

                    # store simulation data
                    sim_row[ipbs] = ppf_vv
                    ipbs += 1

            # TCs Weather Types waves generation
            else:
                # Get TC-WT waves fams data
                ixtc = np.where(xds_WVS_TCs.TC_category == WT - n_clusters - 1)[0]
                tws = xds_WVS_TCs.isel(time=ixtc)

                # select random state
                ri = randint(len(tws.time))

                # generate sim_row with sorted waves families variables
                sim_row = np.stack(
                    [tws[vn].values[ri] for vn in wvs_fams_vars + vars_extra]
                )

            # Filters

            # all 0 chromosomes
            # if all(c == 0 for c in crm):
            #    continue

            # nan / negative values
            if np.isnan(sim_row).any() or len(np.where(sim_row < 0)[0]) != 0:
                continue

            # custom "bad data" filter
            dir_s = sim_row[2 : len(wvs_fams) * 3 : 3][crm == 1]
            if any(v > 360.0 for v in dir_s):
                continue

            # wave hs
            if filters["hs"]:
                hs_s = sim_row[0 : len(wvs_fams) * 3 : 3][crm == 1]
                if any(v <= hs_min for v in hs_s) or any(v >= hs_max for v in hs_s):
                    continue

            # wave tp
            if filters["tp"]:
                tp_s = sim_row[1 : len(wvs_fams) * 3 : 3][crm == 1]
                if any(v <= tp_min for v in tp_s) or any(v >= tp_max for v in tp_s):
                    continue

            # wave stepness
            if filters["ws"]:
                hs_s = sim_row[0 : len(wvs_fams) * 3 : 3][crm == 1]
                tp_s = sim_row[1 : len(wvs_fams) * 3 : 3][crm == 1]
                ws_s = hs_s / (1.56 * tp_s**2)
                if any(v <= ws_min for v in ws_s) or any(v >= ws_max for v in ws_s):
                    continue

            # store simulation
            sim_row[sim_row == 0] = np.nan  # nan data at crom 0
            sims_out[c] = sim_row
            c += 1

            # progress bar
            pbar.update(1)

        pbar.close()

        # dataset for storing output
        xds_wvs_sim = xr.Dataset(
            {
                "DWT": (("time",), DWT_sim),
            },
            coords={"time": DWT_time_sim},
        )
        for c, vn in enumerate(wvs_fams_vars + vars_extra):
            xds_wvs_sim[vn] = (("time",), sims_out[:, c])

        return xds_wvs_sim

    def GenerateTCs(
        self,
        n_clusters,
        DWT,
        DWT_time,
        TCs_params,
        TCs_simulation,
        prob_TCs,
        MU_WT,
        TAU_WT,
        xds_wvs_sim,
        extra_vars_update=[],
    ):
        """
        Climate Emulator DWTs TCs simulation

        n_clusters      - KMA number of clusters
        DWT             - np.array with DWT bmus sim series (dims: time,)

        TCs_params      - xr.Dataset. vars(storm): pressure_min
        TCs_simulation  - xr.Dataset. vars(storm): mu, hs, ss, tp, dir
        prob_TCs        - cumulative probabilities of TC category change
        MU_WT, TAU_WT   - intradaily hidrographs for each WT
        xds_wvs_sim     - xr.Dataset, waves simulated without TCs (for updating)

        extra_vars_update - list(string), optional extra variables to update with value from "TCs_simulation"

        returns xarray.Datasets with updated Waves and simulated TCs data
            vars waves:
                *fam*_*vn* (fam: sea, swell_1, swell_2 ..., vn: Hs, Tp, Dir),
            vars TCS:
                mu, tau, ss
            dims: storm
        """

        # wave family to modify
        mod_fam = "sea"  # TODO input parameter

        # waves families - variables (sorted for simulation output)
        wvs_fams = self.fams
        wvs_fams_vars = [
            ("{0}_{1}".format(f, vn)) for f in wvs_fams for vn in ["Hs", "Tp", "Dir"]
        ]

        # simulate one value for each storm
        dwt_df = np.diff(DWT)
        dwt_df[-1] = 1  # ensure last day storm
        ix_ch = np.where((dwt_df != 0))[0] + 1
        ix_ch = np.insert(ix_ch, 0, 0)  # get first day storm
        DWT_sim = DWT[ix_ch]
        DWT_time_sim = DWT_time[ix_ch]

        # get simulated waves for updating
        sim_wvs = np.column_stack([xds_wvs_sim[vn].values[:] for vn in wvs_fams_vars])

        # get simulated extra variables for updating (optional)
        if extra_vars_update:
            sim_extra = np.column_stack(
                [xds_wvs_sim[vn].values[:] for vn in extra_vars_update]
            )

        # new progress bar
        pbar = tqdm(total=len(DWT_sim), desc="C.E: Sim. TCs  ")

        # Simulate TCs (mu, ss, tau)
        sims_out = np.zeros((len(DWT_sim), 3))
        c = 0
        while c < len(DWT_sim):
            WT = int(DWT_sim[c])
            iwt = WT - 1

            do_upd_wvs = False  # to record when to update simulated waves

            # KMA Weather Types tcs generation
            if WT <= n_clusters:
                # get random MU,TAU from current WT
                # TODO: random excluyente?
                ri = randint(len(MU_WT[iwt]))
                mu_s = MU_WT[iwt][ri]
                tau_s = TAU_WT[iwt][ri]
                ss_s = 0

            # TCs Weather Types waves generation
            else:
                # get probability of category change for this WT
                prob_t = np.append(prob_TCs[:, iwt - n_clusters], 1)

                # generate random r2 category
                ri = rand()
                si = np.where(prob_t >= ri)[0][0]

                if si == len(prob_t) - 1:
                    # TC does not enter. random mu_s, 0.5 tau_s, 0 ss_s
                    all_MUs = np.concatenate(MU_WT)
                    ri = randint(len(all_MUs))
                    # TODO: check mu 0s, set nans (?)

                    mu_s = all_MUs[ri]
                    tau_s = 0.5
                    ss_s = 0

                else:
                    # locate TCs with category "si"
                    s_pmin = TCs_params.pressure_min.values[:]
                    p1, p2 = {
                        0: (1000, np.nanmax(s_pmin) + 1),
                        1: (979, 1000),
                        2: (964, 979),
                        3: (944, 964),
                        4: (920, 944),
                        5: (np.nanmin(s_pmin) - 1, 920),
                    }[si]

                    psi = np.where((s_pmin > p1) & (s_pmin <= p2))[0]

                    if psi.any():
                        # get a random TC from psi indexes
                        ri = choice(psi)

                        mu_s = TCs_simulation.mu.values[ri]
                        ss_s = TCs_simulation.ss.values[ri]
                        tau_s = 0.5

                        # Get waves family data from simulated TCs (numerical+rbf)
                        mod_fam_Hs = TCs_simulation.hs.values[ri]
                        mod_fam_Tp = TCs_simulation.tp.values[ri]
                        mod_fam_Dir = TCs_simulation.dir.values[ri]

                        # locate index of wave family to modify
                        ixu = wvs_fams.index(mod_fam) * 3

                        # replace waves: only sea family
                        upd_wvs = sim_wvs[c, :] * 0
                        upd_wvs[ixu : ixu + 3] = [mod_fam_Hs, mod_fam_Tp, mod_fam_Dir]
                        do_upd_wvs = True

                        # replace extra variables (optional)
                        if extra_vars_update:
                            upd_extra = sim_extra[c, :] * 0
                            for ve_c, ve in enumerate(extra_vars_update):
                                upd_extra[ve_c] = TCs_simulation[ve].values[ri]

                    else:
                        # TODO: no deberia caer aqui
                        mu_s = 0
                        ss_s = 0
                        tau_s = 0

            sim_row = np.array([mu_s, tau_s, ss_s])

            # no nans or values < 0 stored
            if ~np.isnan(sim_row).any() and len(np.where(sim_row < 0)[0]) == 0:
                # store TCs sim
                sims_out[c] = sim_row

                if do_upd_wvs:
                    # update waves: only sea from TCs
                    sim_wvs[c, :] = upd_wvs

                    # update_extra_variables (optional)
                    if extra_vars_update:
                        sim_extra[c, :] = upd_extra

                c += 1

                # progress bar
                pbar.update(1)

        pbar.close()

        # update waves simulation
        xds_WVS_sim_updated = xr.Dataset(
            {
                "DWT": (("time",), DWT_sim),
            },
            coords={"time": DWT_time_sim},
        )
        for c, vn in enumerate(wvs_fams_vars):
            xds_WVS_sim_updated[vn] = (("time",), sim_wvs[:, c])

        # add extra variables to waves simulation update (optional)
        if extra_vars_update:
            for c, vn in enumerate(extra_vars_update):
                xds_WVS_sim_updated[vn] = (("time",), sim_extra[:, c])

        # generated TCs
        xds_TCs_sim = xr.Dataset(
            {
                "mu": (("time",), sims_out[:, 0]),
                "tau": (("time",), sims_out[:, 1]),
                "ss": (("time",), sims_out[:, 2]),
                "DWT": (("time",), DWT_sim),
            },
            coords={"time": DWT_time_sim},
        )

        return xds_TCs_sim, xds_WVS_sim_updated

    def Report_Fit(self, vns_GEV=["Hs"], show=True, plot_chrom=True, plot_sigma=True):
        """
        Report for extremes model fitting

        - GEV parameters for vns_GEV
        - chromosomes probabilities
        - sigma correlation
        """

        f_out = []

        # GEV variables reports
        for vn in vns_GEV:
            fs = self.Report_Gev(vn=vn, show=show)
            f_out = f_out + fs

        # Chromosomes and Sigma reports
        chrom = self.chrom
        d_sigma = self.sigma

        # Plot cromosomes probabilities
        if plot_chrom:
            f = Plot_ChromosomesProbs(chrom, show=show)
            f_out.append(f)

        # Plot sigma correlation triangle
        if plot_sigma:
            f = Plot_SigmaCorrelation(chrom, d_sigma, show=show)
            f_out.append(f)

        return f_out

    def Report_Gev(self, vn="Hs", show=True):
        "Plot vn variable GEV parameters for each WT. variables: Hs, Tp"

        # GEV parameters
        GEV_Par = self.GEV_Par.copy(deep="True")

        # locate fam_vn variables
        vars_gev_params = [x for x in self.vars_GEV if vn in x]

        f_out = []
        for gvn in vars_gev_params:
            f = Plot_GEVParams(GEV_Par[gvn], show=show)
            f_out.append(f)

        return f_out

    def Report_Sim_QQplot(self, WVS_sim, vn, n_sim=0, show=True):
        """
        QQplot for variable vn
        """

        # TODO
        # f_out = Plot_QQplot(kma_bmus, d_sigma, show=show)

        return None


def ChromMatrix(vs):
    "Return chromosome matrix for np.array vs (n x nvars)"

    n_cols = vs.shape[1]
    chrom = np.empty((0, n_cols), int)
    b = np.zeros(n_cols)
    for c in range(n_cols):
        b[c] = 1
        for r in set(permutations(b.tolist())):
            chrom = np.row_stack([chrom, np.array(r)])

    return chrom

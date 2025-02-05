#!/usr/bin/env python
# -*- coding: utf-8 -*-

# common
import pickle
import time
import sys
import os
import os.path as op
from collections import OrderedDict
from datetime import datetime, date, timedelta

# pip
import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.discrete.discrete_model as sm
import scipy.stats as stat
import xarray as xr

# fix tqdm for notebook
from tqdm import tqdm as tqdm_base


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, "_instances"):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


# fix library
from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

# tk
from .io.aux_nc import StoreBugXdset
from .util.time_operations import npdt64todatetime as npdt2dt
from .kma import Persistences
from .plotting.alr import Plot_PValues, Plot_Params, Plot_Terms
from .plotting.wts import (
    Plot_Compare_PerpYear,
    Plot_Compare_Transitions,
    Plot_Compare_Persistences,
)
from .plotting.alr import Plot_Compare_Covariate
from .plotting.alr import Plot_Log_Sim


class ALR_WRP(object):
    "AutoRegressive Logistic Model Wrapper"

    def __init__(self, p_base):
        # data needed for ALR fitting
        self.xds_bmus_fit = None
        self.cluster_size = None

        # ALR terms
        self.d_terms_settings = {}
        self.terms_fit = {}
        self.terms_fit_names = []

        # temporal data storage
        self.mk_order = 0
        self.cov_names = []

        # ALR model core
        self.model = None

        # config (only tested with statsmodels library)
        self.model_library = "statsmodels"  # sklearn / statsmodels

        # paths
        self.p_base = p_base

        # alr model and terms_fit
        self.p_save_model = op.join(p_base, "model.sav")
        self.p_save_terms_fit = op.join(p_base, "terms.sav")

        # store fit and simulated bmus
        self.p_save_fit_xds = op.join(p_base, "xds_input.nc")
        self.p_save_sim_xds = op.join(p_base, "xds_output.nc")

        # export folders for figures
        self.p_report_fit = op.join(p_base, "report_fit")
        self.p_report_sim = op.join(p_base, "report_sim")

        # log sim
        self.p_log_sim_xds = op.join(p_base, "xds_log_sim.nc")

    def SetFitData(self, cluster_size, xds_bmus_fit, d_terms_settings):
        """
        Sets data needed for ALR fitting

        cluster_size - number of clusters in classification
        xds_bmus_fit - xarray.Dataset vars: bmus, dims: time
        d_terms_settings - terms settings. See "SetFittingTerms"
        """

        self.cluster_size = cluster_size
        self.xds_bmus_fit = xds_bmus_fit
        self.SetFittingTerms(d_terms_settings)

        # save bmus series used for fitting
        self.SaveBmus_Fit()

    def SetFittingTerms(self, d_terms_settings):
        "Set terms settings that will be used for fitting"

        # default settings used for ALR terms
        default_settings = {
            "mk_order": 0,
            "constant": False,
            "long_term": False,
            "seasonality": (False, []),
            "covariates": (False, []),
            "covariates_seasonality": (False, []),
        }

        # join user and default input
        for k in default_settings.keys():
            if k not in d_terms_settings:
                d_terms_settings[k] = default_settings[k]

        # generate ALR terms
        bmus_fit = self.xds_bmus_fit.bmus.values
        time_fit = self.xds_bmus_fit.time.values
        cluster_size = self.cluster_size

        self.terms_fit, self.terms_fit_names = self.GenerateALRTerms(
            d_terms_settings, bmus_fit, time_fit, cluster_size, time2yfrac=True
        )

        # store data
        self.mk_order = d_terms_settings["mk_order"]
        self.d_terms_settings = d_terms_settings

    def GenerateALRTerms(
        self, d_terms_settings, bmus, time, cluster_size, time2yfrac=False
    ):
        "Generate ALR terms from user terms settings"

        # terms stored at OrderedDict
        terms = OrderedDict()
        terms_names = []

        # time options (time has to bee yearly fraction)
        if time2yfrac:
            time_yfrac = self.GetFracYears(time)
        else:
            time_yfrac = time

        # constant term
        if d_terms_settings["constant"]:
            terms["constant"] = np.ones((bmus.size, 1))
            terms_names.append("intercept")

        # time term (use custom time array with year decimals)
        if d_terms_settings["long_term"]:
            terms["long_term"] = np.ones((bmus.size, 1))
            terms["long_term"][:, 0] = time_yfrac
            terms_names.append("long_term")

        # seasonality term
        if d_terms_settings["seasonality"][0]:
            phases = d_terms_settings["seasonality"][1]
            temp_seas = np.zeros((len(time_yfrac), 2 * len(phases)))
            c = 0
            for a in phases:
                temp_seas[:, c] = np.cos(a * np.pi * time_yfrac)
                temp_seas[:, c + 1] = np.sin(a * np.pi * time_yfrac)
                terms_names.append("ss_cos_{0}".format(a))
                terms_names.append("ss_sin_{0}".format(a))
                c += 2
            terms["seasonality"] = temp_seas

        # Covariates term
        if d_terms_settings["covariates"][0]:
            # covariates dataset (vars: cov_values, dims: time, cov_names)
            xds_cov = d_terms_settings["covariates"][1]
            cov_names = xds_cov.cov_names.values
            self.cov_names = cov_names  # storage

            # normalize covars
            if not "cov_norm" in xds_cov.keys():
                cov_values = xds_cov.cov_values.values
                cov_norm = (cov_values - cov_values.mean(axis=0)) / cov_values.std(
                    axis=0
                )
            else:
                # simulation covars are previously normalized
                cov_norm = xds_cov.cov_norm.values

            # generate covar terms
            for i in range(cov_norm.shape[1]):
                cn = cov_names[i]
                terms[cn] = np.transpose(np.asmatrix(cov_norm[:, i]))
                terms_names.append(cn)

                # Covariates seasonality
                if d_terms_settings["covariates_seasonality"][0]:
                    cov_season = d_terms_settings["covariates_seasonality"][1]

                    if cov_season[i]:
                        terms["{0}_cos".format(cn)] = np.multiply(
                            terms[cn].T, np.cos(2 * np.pi * time_yfrac)
                        ).T
                        terms["{0}_sin".format(cn)] = np.multiply(
                            terms[cn].T, np.sin(2 * np.pi * time_yfrac)
                        ).T
                        terms_names.append("{0}_cos".format(cn))
                        terms_names.append("{0}_sin".format(cn))

        # markov term
        if d_terms_settings["mk_order"] > 0:
            # dummi for markov chain
            def dummi(csize):
                D = np.ones((csize - 1, csize)) * -1
                for i in range(csize - 1):
                    D[i, csize - 1 - i] = csize - i - 1
                    D[i, csize - 1 + 1 - i :] = 0
                return D

            def dummi_norm(csize):
                D = np.zeros((csize, csize - 1))
                for i in range(csize - 1):
                    D[i, i] = float((csize - i - 1)) / (csize - i)
                    D[i + 1 :, i] = -1.0 / (csize - i)

                return np.transpose(np.flipud(D))

            def helmert_ints(csize, reverse=False):
                D = np.zeros((csize, csize - 1))
                for i in range(csize - 1):
                    D[i, i] = csize - i - 1
                    D[i + 1 :, i] = -1.0

                if reverse:
                    return np.fliplr(np.flipud(D))
                else:
                    return D

            def helmert_norm(csize, reverse=False):
                D = np.zeros((csize, csize - 1))
                for i in range(csize - 1):
                    D[i, i] = float((csize - i - 1)) / (csize - i)
                    D[i + 1 :, i] = -1.0 / (csize - i)

                if reverse:
                    return np.fliplr(np.flipud(D))
                else:
                    return D

            #  helmert
            dum = helmert_norm(cluster_size, reverse=True)

            # solve markov order N
            mk_order = d_terms_settings["mk_order"]
            for i in range(mk_order):
                Z = np.zeros((bmus.size, cluster_size - 1))
                for indz in range(bmus.size - i - 1):
                    Z[indz + i + 1, :] = np.squeeze(dum[bmus[indz] - 1, :])

                terms["markov_{0}".format(i + 1)] = Z

                for ics in range(cluster_size - 1):
                    terms_names.append("mk{0}_{1}".format(i + 1, ics + 1))

        return terms, terms_names

    def GetFracYears(self, time):
        "Returns time in custom year decimal format"

        # fix np.datetime64
        if not "year" in dir(time[0]):
            time_0 = pd.to_datetime(time[0])
            time_1 = pd.to_datetime(time[-1])
            time_d = pd.to_datetime(time[1])
        else:
            time_0 = time[0]
            time_1 = time[-1]
            time_d = time[1]

        # resolution year
        if time_d.year - time_0.year == 1:
            return range(time_1.year - time_0.year + 1)

        # resolution day: get start/end data
        y0 = time_0.year
        m0 = time_0.month
        d0 = time_0.day
        y1 = time_1.year
        m1 = time_1.month
        d1 = time_1.day

        # start "year cicle" at 01/01
        d_y0 = date(y0, 1, 1)

        # time array
        d_0 = date(y0, m0, d0)
        d_1 = date(y1, m1, d1)

        # year_decimal from year start to d1
        delta_y0 = d_1 - d_y0
        y_fraq_y0 = np.array(range(delta_y0.days + 1)) / 365.25

        # cut year_decimal from d_0
        i0 = (d_0 - d_y0).days
        y_fraq = y_fraq_y0[i0:]

        return y_fraq

    def FitModel(self, max_iter=1000):
        "Fits ARL model using sklearn"

        # get fitting data
        X = np.concatenate(list(self.terms_fit.values()), axis=1)
        y = self.xds_bmus_fit.bmus.values

        # fit model
        print("\nFitting autoregressive logistic model ...")
        start_time = time.time()

        if self.model_library == "statsmodels":
            # mount data with pandas
            X = pd.DataFrame(X, columns=self.terms_fit_names)
            y = pd.DataFrame(y, columns=["bmus"])

            # TODO: CAPTURAR LA EVOLUCION DE L (maximun-likelihood)
            self.model = sm.MNLogit(y, X).fit(
                method="lbfgs",
                maxiter=max_iter,
                retall=True,
                full_output=True,
                disp=True,
                warn_convergence=True,
                missing="raise",
            )

        elif self.model_library == "sklearn":
            # use sklearn logistig regression
            self.model = linear_model.LogisticRegression(
                penalty="l2", C=1e5, fit_intercept=False, solver="lbfgs"
            )
            self.model.fit(X, y)

        else:
            print("wrong config: {0} not in model_library".format(self.model_library))
            sys.exit()

        elapsed_time = time.time() - start_time
        print("Optimization done in {0:.2f} seconds\n".format(elapsed_time))

        # save fitted model
        self.SaveModel()

    def SaveModel(self):
        "Saves fitted model (and fitting terms) for future use"

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        # save model
        pickle.dump(self.model, open(self.p_save_model, "wb"))

        # save terms fit
        pickle.dump(
            [self.d_terms_settings, self.terms_fit, self.terms_fit_names],
            open(self.p_save_terms_fit, "wb"),
        )

    def LoadModel(self):
        "Load fitted model (and fitting terms)"

        # load model
        self.model = pickle.load(open(self.p_save_model, "rb"))

        # load terms fit
        self.d_terms_settings, self.terms_fit, self.terms_fit_names = pickle.load(
            open(self.p_save_terms_fit, "rb")
        )

        # load aux data
        if self.d_terms_settings["covariates"][0]:
            cov_names = self.d_terms_settings["covariates"][1].cov_names.values
            self.cov_names = cov_names

        self.mk_order = self.d_terms_settings["mk_order"]

    def SaveBmus_Fit(self):
        "Saves bmus - fit for future use"

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        self.xds_bmus_fit.attrs["cluster_size"] = self.cluster_size
        self.xds_bmus_fit.to_netcdf(self.p_save_fit_xds, "w")

    def LoadBmus_Fit(self):
        "Load bmus - fit"

        self.xds_bmus_fit = xr.open_dataset(self.p_save_fit_xds)
        self.cluster_size = self.xds_bmus_fit.attrs["cluster_size"]

        return self.xds_bmus_fit

    def SaveBmus_Sim(self):
        "Saves bmus - sim for future use"

        if not op.isdir(self.p_base):
            os.makedirs(self.p_base)

        self.xds_bmus_sim.to_netcdf(self.p_save_sim_xds, "w")

    def LoadBmus_Sim(self):
        "Load bmus - sim"

        self.xds_bmus_sim = xr.open_dataset(self.p_save_sim_xds)

        return self.xds_bmus_sim

    def Report_Fit(self, terms_fit=False, summary=False, show=True):
        "Report containing model fitting info"

        # load model
        self.LoadModel()

        # get data
        try:
            pval_df = self.model.pvalues.transpose()
            params_df = self.model.params.transpose()
            name_terms = pval_df.columns.tolist()
        except:
            # TODO: no converge?
            print("warning - statsmodels MNLogit could not provide p-values")
            return

        # output figs
        l_figs = []

        # plot p-values
        f = Plot_PValues(pval_df.values, name_terms, show=show)
        l_figs.append(f)

        # plot parameters
        f = Plot_Params(params_df.values, name_terms, show=show)
        l_figs.append(f)

        # plot terms used for fitting
        if terms_fit:
            f = self.Report_Terms_Fit(p_rep_trms, show=show)
            l_figs.append(f)

        # write summary
        if summary:
            summ = self.model.summary()
            print(summ.as_text())

    def Report_Terms_Fit(self, show=True):
        "Plot terms used for model fitting"

        # load bmus fit
        self.LoadBmus_Fit()

        # get data for plotting
        term_mx = np.concatenate(list(self.terms_fit.values()), axis=1)
        term_ds = [npdt2dt(t) for t in self.xds_bmus_fit.time.values]
        term_ns = self.terms_fit_names

        # Plot terms
        f = Plot_Terms(term_mx, term_ds, term_ns, show=show)
        return f

    def Simulate(
        self,
        num_sims,
        time_sim,
        xds_covars_sim=None,
        log_sim=False,
        overfit_filter=False,
        of_probs=0.98,
        of_pers=5,
    ):
        """
        Launch ARL model simulations

        num_sims           - number of simulations to compute
        time_sim           - time array to solve

        xds_covars_sim     - xr.Dataset (time,), cov_values
            Covariates used at simulation, compatible with "n_sim" dimension
            ("n_sim" dimension (optional) will be iterated with each simulation)

        log_sim            - Store a .nc file with all simulation detailed information.

        filters for exceptional ALR overfit probabilities situation and patch:

        overfit_filter     - overfit filter activation
        of_probs           - overfit filter probabilities activation
        of_pers            - overfit filter persistences activation
        """

        class SimLog(object):
            """
            simulatiom log - records and stores detail info for each time step and n_sim
            """

            def __init__(
                self, time_yfrac, mk_order, num_sims, cluster_size, terms_fit_names
            ):
                # initialize variables to record
                self.terms = np.nan * np.zeros(
                    (
                        len(time_yfrac) - mk_order,
                        num_sims,
                        mk_order + 1,
                        len(terms_fit_names),
                    )
                )
                self.probs = np.nan * np.zeros(
                    (len(time_yfrac) - mk_order, num_sims, mk_order + 1, cluster_size)
                )
                self.ptrns = np.nan * np.zeros(
                    (len(time_yfrac) - mk_order, num_sims, cluster_size)
                )
                self.nrnd = np.nan * np.zeros((len(time_yfrac) - mk_order, num_sims))
                self.evbmu_sims = np.nan * np.zeros(
                    (len(time_yfrac) - mk_order, num_sims)
                )

                # overfit filter variables (filter states and filtered bmus)
                self.of_state = np.zeros(
                    (len(time_yfrac) - mk_order, num_sims), dtype=bool
                )
                self.of_bmus = np.nan * np.zeros((len(time_yfrac) - mk_order, num_sims))

            def Add(self, ix_t, ix_s, terms, prob, probTrans, nrnd, of_state, of_bmus):
                # add iteration to log
                self.terms[ix_t, ix_s, :, :] = terms
                self.probs[ix_t, ix_s, :, :] = prob
                self.ptrns[ix_t, ix_s, :] = probTrans
                self.nrnd[ix_t, ix_s] = nrnd
                self.evbmu_sims[ix_t, ix_s] = np.where(probTrans > nrnd)[0][0] + 1

                # add overfit filter data to log
                self.of_state[ix_t, ix_s] = of_state
                self.of_bmus[ix_t, ix_s] = of_bmus

            def Save(self, p_save, terms_names):
                # use xarray to store netcdf
                xds_log = xr.Dataset(
                    {
                        "alr_terms": (
                            (
                                "time",
                                "n_sim",
                                "mk",
                                "terms",
                            ),
                            self.terms,
                        ),
                        "probs": (("time", "n_sim", "mk", "n_clusters"), self.probs),
                        "probTrans": (("time", "n_sim", "n_clusters"), self.ptrns),
                        "nrnd": (("time", "n_sim"), self.nrnd),
                        "evbmus_sims": (("time", "n_sim"), self.evbmu_sims.astype(int)),
                        "overfit_filter_state": (("time", "n_sim"), self.of_state),
                        "evbmus_sims_filtered": (
                            ("time", "n_sim"),
                            self.of_bmus.astype(int),
                        ),
                    },
                    coords={
                        "time": time_sim[mk_order:],
                        "terms": terms_names,
                    },
                )

                StoreBugXdset(xds_log, p_save)
                print("simulation data log stored at {0}\n".format(p_save))

        class OverfitFilter(object):
            """
            overfit filter for alr outlayer outputs.
            """

            def __init__(self, probs_lim, pers_lim):
                self.active = False
                self.probs_lim = probs_lim
                self.pers_lim = pers_lim
                self.log = ""

            def CheckStatus(self, n_sim, prob, bmus):
                "check current iteration filter status"

                # active filter
                if self.active:
                    # continuation condition
                    self.active = np.nanmax(prob[-1, :]) >= self.probs_lim

                    # log when deactivated
                    if self.active == False:
                        self.log += (
                            "sim. {0:02d} - {1} - deactivated (max prob {2})\n".format(
                                n_sim, time_sim[i], np.nanmax(prob[-1, :])
                            )
                        )

                # inactive filter
                else:
                    # re-activation condition
                    self.active = np.nanmax(prob[-1, :]) >= self.probs_lim and np.all(
                        evbmus[-1 * self.pers_lim :] == new_bmus
                    )

                    # log when activated
                    if self.active:
                        self.log += (
                            "sim. {0:02d} - {1} - activated (max prob {2})\n".format(
                                n_sim, time_sim[i], np.nanmax(prob[-1, :])
                            )
                        )

            def PrintLog(self):
                "Print filter log"

                if self.log != "":
                    print("overfit filter log")
                    print(self.log)

        # switch library probabilities predictor function
        if self.model_library == "statsmodels":
            pred_prob_fun = self.model.predict
        elif self.model_library == "sklearn":
            pred_prob_fun = self.model.predict_proba
        else:
            print("wrong config: {0} not in model_library".format(self.model_library))
            sys.exit()

        # get needed data
        evbmus_values = self.xds_bmus_fit.bmus.values
        time_fit = self.xds_bmus_fit.time.values
        mk_order = self.mk_order

        # times at datetime
        if isinstance(time_sim[0], np.datetime64):
            time_sim = [npdt2dt(t) for t in time_sim]

        # print some info
        tf0 = str(time_fit[0])[:10]
        tf1 = str(time_fit[-1])[:10]
        ts0 = str(time_sim[0])[:10]
        ts1 = str(time_sim[-1])[:10]
        print("ALR model fit   : {0} --- {1}".format(tf0, tf1))
        print("ALR model sim   : {0} --- {1}".format(ts0, ts1))

        # generate time yearly fractional array
        time_yfrac = self.GetFracYears(time_sim)

        # use a d_terms_settigs copy
        d_terms_settings_sim = self.d_terms_settings.copy()

        # initialize optional simulation log
        if log_sim:
            SL = SimLog(
                time_yfrac, mk_order, num_sims, self.cluster_size, self.terms_fit_names
            )

        # initialize ALR overfit filter
        ofilt = OverfitFilter(of_probs, of_pers)

        # initialize ALR simulated bmus array, and overfit filter register array
        evbmus_sims = np.zeros((len(time_yfrac), num_sims))
        ofbmus_sims = np.zeros((len(time_yfrac), num_sims), dtype=bool)

        # start simulations
        print("\nLaunching {0} simulations...\n".format(num_sims))
        for n in range(num_sims):
            # preload some data (simulation covariates)
            cvtxt = ""
            if xds_covars_sim != None:
                # check if n_sim dimension in xds_covars_sim
                if "n_sim" in xds_covars_sim.dims:
                    sim_covars_T = xds_covars_sim.isel(n_sim=n).cov_values.values
                    cvtxt = " (Covs. {0:03d})".format(n + 1)
                else:
                    sim_covars_T = xds_covars_sim.cov_values.values

                sim_covars_T_mean = sim_covars_T.mean(axis=0)
                sim_covars_T_std = sim_covars_T.std(axis=0)

            # progress bar
            pbar = tqdm(
                total=len(time_yfrac) - mk_order,
                file=sys.stdout,
                desc="Sim. Num. {0:03d}{1}".format(n + 1, cvtxt),
            )

            evbmus = evbmus_values[1 : mk_order + 1]
            for i in range(len(time_yfrac) - mk_order):
                # handle simulation covars
                if d_terms_settings_sim["covariates"][0]:
                    # normalize step covars
                    sim_covars_evbmus = sim_covars_T[i : i + mk_order + 1]
                    sim_cov_norm = (
                        sim_covars_evbmus - sim_covars_T_mean
                    ) / sim_covars_T_std

                    # mount step xr.dataset for sim covariates
                    xds_cov_sim_step = xr.Dataset(
                        {
                            "cov_norm": (("time", "cov_names"), sim_cov_norm),
                        },
                        coords={"cov_names": self.cov_names},
                    )

                    d_terms_settings_sim["covariates"] = (True, xds_cov_sim_step)

                # generate time step ALR terms
                terms_i, terms_names = self.GenerateALRTerms(
                    d_terms_settings_sim,
                    np.append(evbmus[i : i + mk_order], 0),
                    time_yfrac[i : i + mk_order + 1],
                    self.cluster_size,
                    time2yfrac=False,
                )

                # Event sequence simulation  (sklearn)
                X = np.concatenate(list(terms_i.values()), axis=1)
                prob = pred_prob_fun(X)  # statsmodels // sklearn functions
                probTrans = np.cumsum(prob[-1, :])

                # generate random cluster with ALR probs
                nrnd = np.random.rand()
                new_bmus = np.where(probTrans > nrnd)[0][0] + 1

                # overfit filter status swich (if active)
                if overfit_filter:
                    ofilt.CheckStatus(n, prob, np.append(evbmus, new_bmus))

                # override overfit bmus if filter active
                if ofilt.active:
                    # criteria: random bmus from that date of the year at  historical
                    ix_of = np.random.choice(
                        np.where(
                            self.xds_bmus_fit["time.dayofyear"]
                            == time_sim[i].timetuple().tm_yday
                        )[0]
                    )
                    new_bmus = self.xds_bmus_fit.bmus.values[ix_of]

                # append_bmus
                evbmus = np.append(evbmus, new_bmus)

                # store overfit filter status
                ofbmus_sims[i + mk_order, n] = ofilt.active

                # optional detail log
                if log_sim:
                    SL.Add(i, n, X, prob, probTrans, nrnd, ofilt.active, new_bmus)

                # update progress bar
                pbar.update(1)

            evbmus_sims[:, n] = evbmus

            # close progress bar
            pbar.close()
        print()  # white line after all progress bars

        # return ALR simulation data in a xr.Dataset
        xds_out = xr.Dataset(
            {
                "evbmus_sims": (("time", "n_sim"), evbmus_sims.astype(int)),
                "ofbmus_sims": (("time", "n_sim"), ofbmus_sims),
            },
            coords={
                "time": time_sim,
            },
        )

        # save output
        StoreBugXdset(xds_out, self.p_save_sim_xds)

        # save log file
        if log_sim:
            SL.Save(self.p_log_sim_xds, terms_names)

        # overfit filter log
        ofilt.PrintLog()

        return xds_out

    def Report_Sim(
        self,
        py_month_ini=1,
        persistences_hists=False,
        persistences_table=False,
        show=True,
    ):
        """
        Report that Compare fitting to simulated bmus

        py_month_ini  - start month for PerpetualYear bmus comparison
        """
        # TODO: add arg n_sim = None (for plotting only one sim output)

        # load fit and sim bmus
        xds_ALR_fit = self.LoadBmus_Fit()
        xds_ALR_sim = self.LoadBmus_Sim()

        # report folder and files
        p_save = self.p_report_sim

        # get data
        cluster_size = self.cluster_size
        bmus_values_sim = xds_ALR_sim.evbmus_sims.values[:]
        bmus_dates_sim = xds_ALR_sim.time.values[:]
        bmus_values_hist = np.reshape(xds_ALR_fit.bmus.values, [-1, 1])
        bmus_dates_hist = xds_ALR_fit.time.values[:]
        num_sims = bmus_values_sim.shape[1]

        # calculate bmus persistences
        pers_hist = Persistences(bmus_values_hist.flatten())
        lsp = [Persistences(bs) for bs in bmus_values_sim.T.astype(int)]
        pers_sim = {k: np.concatenate([x[k] for x in lsp]) for k in lsp[0].keys()}

        # fix datetime 64 dates
        if isinstance(bmus_dates_sim[0], np.datetime64):
            bmus_dates_sim = [npdt2dt(t) for t in bmus_dates_sim]
        if isinstance(bmus_dates_hist[0], np.datetime64):
            bmus_dates_hist = [npdt2dt(t) for t in bmus_dates_hist]

        # output figs
        l_figs = []

        # Plot Perpetual Year (daily) - bmus wt
        fig_PP = Plot_Compare_PerpYear(
            cluster_size,
            bmus_values_sim,
            bmus_dates_sim,
            bmus_values_hist,
            bmus_dates_hist,
            n_sim=num_sims,
            month_ini=py_month_ini,
            show=show,
        )
        l_figs.append(fig_PP)

        # Plot WTs Transition (probability change / scatter Fit vs. ACCUMULATED Sim)
        sttl = "Cluster Probabilities Transitions: All Simulations"
        fig_CT = Plot_Compare_Transitions(
            cluster_size,
            bmus_values_hist,
            bmus_values_sim,
            sttl=sttl,
            show=show,
        )
        l_figs.append(fig_CT)

        # Plot Persistences comparison Fit vs Sim
        if persistences_hists:
            fig_PS = Plot_Compare_Persistences(
                cluster_size,
                pers_hist,
                pers_sim,
                show=show,
            )
            l_figs.append(fig_PS)

        # persistences set table
        if persistences_table:
            print("Persistences by WT (set)")
            for c in range(cluster_size):
                wt = c + 1
                p_h = pers_hist[wt]
                p_s = pers_sim[wt]

                print("WT: {0}".format(wt))
                print("  hist : {0}".format((sorted(set(p_h)))))
                print("  sim. : {0}".format(sorted(set(p_s))))

        # TODO export handling (if show=False)
        # p_save = self.p_report_sim
        # p_rep_PY = None
        # p_rep_VL = None
        # if export:
        #    if not op.isdir(p_save): os.mkdir(p_save)
        #    p_rep_PY = op.join(p_save, 'PerpetualYear.png')
        #    p_rep_VL = op.join(p_save, 'Transitions.png')

        return l_figs

        # TODO: Plot Perpetual Year (monthly)
        # TODO: load covariates if needed for ploting
        # TODO: activate this
        if self.d_terms_settings["covariates"][0]:
            # TODO eliminar tiempos duplicados
            time_hist_covars = bmus_dates_hist
            time_sim_covars = bmus_dates_sim

            # covars fit
            xds_cov_fit = self.d_terms_settings["covariates"][1]
            cov_names = xds_cov_fit.cov_names.values
            cov_fit_values = xds_cov_fit.cov_values.values

            # covars sim
            cov_sim_values = xds_cov_sim.cov_values.values

            for ic, cn in enumerate(cov_names):
                # get covariate data
                cf_val = cov_fit_values[:, ic]
                cs_val = cov_sim_values[:, ic]

                # plot covariate - bmus wt
                p_rep_cn = op.join(p_save, "{0}_comp.png".format(cn))
                Plot_Compare_Covariate(
                    cluster_size,
                    bmus_values_sim,
                    bmus_dates_sim,
                    bmus_values_hist,
                    bmus_dates_hist,
                    cs_val,
                    time_sim_covars,
                    cf_val,
                    time_hist_covars,
                    cn,
                    n_sim=num_sims,
                    p_export=p_rep_cn,
                )

    def Report_Sim_Log(self, n_sim=0, t_slice=None, show=True):
        """
        Interactive plot for simulation log

        n_sim  - simulation log to plot
        """

        # load fit and sim bmus
        xds_log = xr.open_dataset(self.p_log_sim_xds, decode_times=True)

        # get simulation
        log_sim = xds_log.isel(n_sim=n_sim)

        if t_slice != None:
            log_sim = log_sim.sel(time=t_slice)

        # plot interactive report
        Plot_Log_Sim(log_sim)

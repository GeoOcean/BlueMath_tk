import os

import numpy as np
from scipy.interpolate import interp1d

from .._base_wrappers import BaseModelWrapper

sbatch_file_example = """#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks (MPI processes)
#SBATCH --partition=geocean     # Standard output and error log
#SBATCH --nodes=1               # Number of nodes to use
#SBATCH --mem=4gb               # Memory per node in GB (see also --mem-per-cpu)
#SBATCH --time=24:00:00


case_dir=$(ls | awk "NR == $SLURM_ARRAY_TASK_ID")
cd $case_dir
launchSchism.sh
"""


class SchismModelWrapper(BaseModelWrapper):
    """
    Wrapper for the Schism model.

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    """

    default_parameters = {}

    available_launchers = {"geoocean-cluster": "launchSchism.sh"}

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
        Initialize the Schism model wrapper.
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

        self.sbatch_file_example = sbatch_file_example

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

        os.makedirs(os.path.join(case_dir, "outputs"), exist_ok=True)


class SchismHyCoFfEE(SchismModelWrapper):
    """
    Schism wrapper: HyCoFfEE : A Hybrid Compound Flooding metamodel for Estuarine Environment.
    """

    default_parameters = {
        # .param file
        "indvel": {
            "type": float,
            "value": 1,
            "description": "Method for computing velocity at nodes: 0= linear shape function; 1 = averaging method",
        },
        "ihorcon": {
            "type": float,
            "value": 0,
            "description": "Horizontal viscosity option: 0= no viscosity is used; 1= Lapacian",
        },
        "ishapiro": {
            "type": float,
            "value": 0,
            "description": "on/off flag for Shapiro filter: 0= off; 1= on",
        },
        "inter_mom": {
            "type": float,
            "value": 0,
            "description": " Interpolation is used for velocity at foot of char: 0=linear; 1=Kriging (define kr_co ); -1= spatial Kriging (define krvel.gr3)",
        },
        # CORE
        "ipre": {
            "type": float,
            "value": 0,
            "description": "Pre-process flag: 0= normal run; 1= only for single CPU!",
        },
        "ibc": {
            "type": float,
            "value": 1,
            "description": "Barotropic flag",
        },
        "ibtp": {
            "type": float,
            "value": 0,
            "description": "Baroclinic flag",
        },
        # TRACERS
        "ntracer_gen": {
            "type": float,
            "value": 2,
            "description": "# of tracers",
        },
        "ntracer_age": {
            "type": float,
            "value": 4,
            "description": "Age calculation, must be 2*ntracer_gen",
        },
        # SEDIMENTS
        "sed_class": {
            "type": float,
            "value": 5,
            "description": "SED3D",
        },
        # ECO
        "eco_class": {
            "type": float,
            "value": 27,
            "description": "EcoSim. Must be bewteen [25,60]",
        },
        # OPT
        "ieos_type": {
            "type": float,
            "value": 0,
            "description": "UNESCO 1980 (nonlinear); =1: linear function of T ONLY, i.e. ",
        },
        "ieos_pres": {
            "type": float,
            "value": 0,
            "description": "Used only if ieos_type=0. 0: without pressure effects ",
        },
        "eos_a": {
            "type": float,
            "value": -0.1,
            "description": "Needed if ieos_type=1; should be <=0",
        },
        "eos_b": {
            "type": float,
            "value": 1001,
            "description": "Needed if ieos_type=1",
        },
        # WWM
        "msc2": {
            "type": float,
            "value": 24,
            "description": "Grid for WWM. Same as msc in .nml ... for consitency check between SCHISM and WWM",
        },
        "mdc2": {
            "type": float,
            "value": 30,
            "description": "Grid for WWM. Same as mdc in .nml ... for consitency check between SCHISM and WWM",
        },
        # SCHOUT
        "output_WL": {
            "type": float,
            "value": 1,
            "description": "Elev. [m] = 0: off; 1: on",
        },
        "output_wind_velocity": {
            "type": float,
            "value": 1,
            "description": "Wind velocity vector [m/s] = 0: off; 1: on",
        },
        "output_water_temperature": {
            "type": float,
            "value": 0,
            "description": "Water temperature [C] = 0: off; 1: on",
        },
        "output_depth_averaged_vel": {
            "type": float,
            "value": 1,
            "description": "Depth-averaged vel vector [m/s] = 0: off; 1: on",
        },
        "output_water_salinity": {
            "type": float,
            "value": 0,
            "description": "Water salinity [PSU] = 0: off; 1: on",
        },
    }

    def generate_time_series(
        self,
        case_context: dict,
        case_dir: str,
    ):
        """
        Generate the required th files for the case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        print(case_context["case_num"])

        # Calculate the start and end times in seconds
        qp = case_context.get("Qp")
        qb = case_context.get("Qb")
        tp = case_context.get("Tp")
        tau_ss_d = case_context.get("Tau_ss_d")
        tau = case_context.get("Tau")
        cm = case_context.get("CM")
        sb = case_context.get("Sb")
        sp = case_context.get("Sp")
        hidro = case_context.get("hidro")

        # Puede ser que tau(ss-d) sea negativo pero Tp sea tan pequeño que las 24 horas del surge empiecen antes que el hidrograma de caudal
        time0 = tp + abs(tau_ss_d) - 24

        # hidrograma SCS
        hidro_Qp = hidro.Q * (qp - qb) + qb
        if (
            tau_ss_d <= 0 and time0 >= 0 and tp != 0
        ):  # el hidrograma marca el inicio del tiempo
            hidro_hours = hidro.t.values * tp
        else:
            if tau_ss_d <= 0 and time0 < 0 and tp != 0:
                hidro_hours = (
                    hidro.t.values * tp + abs(time0)
                )  # el storm surge marca el inicio y hay que sumar el Tau con respecto al surge
            else:
                if tau_ss_d > 0:
                    t0 = 24 - (tp - tau_ss_d)
                    hidro_hours = (
                        hidro.t.values * tp + t0
                    )  # el storm surge marca el inicio y hay que sumar el Tau con respecto al surge
                else:  # el desfase es cero
                    if (
                        tp == 0
                    ):  # el tiempo de concentración también, con lo que se supone que el caudal es constante e igual al base
                        hidro_hours = np.arange(0, 49, 1)
                        hidro_Qp = np.ones(len(hidro_hours)) * qp

        hidro_seg = hidro_hours * 3600

        # duración teniendo en cuenta los desfases entre picos de caudal y storm surge
        # en el caso de Tau(ss-d) < 0 y Tp suficientemente grande para que el hidrograma de caudal marque el inicio
        timef = tp + abs(tau_ss_d) + 24

        # AT
        # Creamos un array con los valores de tiempo (48 horas)
        int_min = case_context.get("dt")

        if (
            timef > np.max(hidro_hours) and tau_ss_d < 0 and time0 > 0
        ):  # añadido nuevo el time0>0
            t = np.arange(0, timef * 3600, int_min)
            # interpolamos caudal a esa resolución temporal
            f = interp1d(hidro_seg, hidro_Qp, kind="linear")
            # f = interp1d(hidro_seg, hidro_Qp, kind="linear", fill_value="extrapolate", bounds_error=False) # CAMBIADO!!!!

            t2 = np.arange(0, np.max(hidro_seg), int_min)
            hidro_Qp_5_1 = f(t2)
            t3 = np.max(t) - np.max(t2)
            hidro_Qp_5_2 = np.zeros(int(t3 / int_min)) + hidro_Qp.values[-1]
            hidro_Qp_5 = np.hstack((hidro_Qp_5_1, hidro_Qp_5_2))
        else:
            if (
                hidro_hours[-1] < 48 and tau_ss_d < 0 and time0 > 0
            ):  # añadido nuevo el time0>0
                t = np.arange(0, 48 * 3600, int_min)
                # interpolamos caudal a esa resolución temporal
                f = interp1d(hidro_seg, hidro_Qp, kind="linear")
                t2 = np.arange(0, np.max(hidro_seg), int_min)
                hidro_Qp_5_1 = f(t2)
                t3 = np.max(t) - np.max(t2)
                hidro_Qp_5_2 = np.zeros(int(t3 / int_min)) + hidro_Qp.values[-1]
                hidro_Qp_5 = np.hstack((hidro_Qp_5_1, hidro_Qp_5_2))
            else:
                if tau_ss_d <= 0 and time0 < 0 and tp != 0 and hidro_hours[-1] > 48:
                    t = np.arange(0, np.max(hidro_seg), int_min)
                    hidro_Qp_5_1 = np.full(int(abs(time0 * 3600 / int_min)), qb)
                    f = interp1d(hidro_seg, hidro_Qp, kind="linear")
                    t2 = np.arange(np.min(hidro_seg), np.max(hidro_seg), int_min)
                    hidro_Qp_5_2 = f(t2)
                    hidro_Qp_5 = np.hstack((hidro_Qp_5_1, hidro_Qp_5_2))
                else:
                    if tau_ss_d <= 0 and time0 < 0 and tp != 0 and hidro_hours[-1] < 48:
                        t = np.arange(0, 48 * 3600, int_min)
                        hidro_Qp_5_1 = np.full(int(abs(time0 * 3600 / int_min)), qb)
                        f = interp1d(hidro_seg, hidro_Qp, kind="linear")
                        t2 = np.arange(np.min(hidro_seg), np.max(hidro_seg), int_min)
                        hidro_Qp_5_2 = f(t2)
                        time_left = 48 - hidro_hours[-1]
                        hidro_Qp_5_3 = np.full(int(abs(time_left * 3600 / int_min)), qb)
                        hidro_Qp_5 = np.hstack(
                            (hidro_Qp_5_1, hidro_Qp_5_2, hidro_Qp_5_3)
                        )
                    else:
                        if hidro_hours[-1] < 48:
                            t0 = 24 - (tp - tau_ss_d)
                            # timef2 = t0 + np.max()
                            t = np.arange(0, 48 * 3600, int_min)
                            hidro_Qp_5_0 = np.zeros(int(t0 * 3600 / int_min)) + qb
                            # interpolamos caudal a esa resolución temporal
                            f = interp1d(hidro_seg, hidro_Qp, kind="linear")
                            t2 = np.arange(
                                np.min(hidro_seg), np.max(hidro_seg), int_min
                            )
                            hidro_Qp_5_1 = f(t2)
                            t3 = np.max(t) - np.max(t2)
                            hidro_Qp_5_2 = np.zeros(int(t3 / int_min)) + qb
                            hidro_Qp_5 = np.hstack(
                                (hidro_Qp_5_0, hidro_Qp_5_1, hidro_Qp_5_2)
                            )
                        else:
                            t = np.arange(0, np.max(hidro_seg), int_min)
                            # interpolamos caudal a esa resolución temporal
                            f = interp1d(hidro_seg, hidro_Qp, kind="linear")
                            if tau_ss_d > 0:
                                t0 = 24 - (tp - tau_ss_d)
                                # hidro_Qp_5_1 = np.full(int(df_v.iloc[n]['Tau_ss_d']*3600/5),qb)
                                hidro_Qp_5_1 = np.full(int(t0 * 3600 / int_min), qb)
                                t2 = np.arange(
                                    np.min(hidro_seg), np.max(hidro_seg), int_min
                                )
                                hidro_Qp_5_2 = f(t2)
                                hidro_Qp_5 = np.hstack((hidro_Qp_5_1, hidro_Qp_5_2))
                            else:
                                hidro_Qp_5 = f(t)  # el hidrograma abarca todo el tiempo

        # Tau (desfase de la pleamar con respecto al pico de caudal Qp - en horas)
        # tau = tau

        # AT
        at = np.sin(np.pi / 2 + np.pi * tau / 6 + 2 * np.pi * t / (12 * 3600)) * cm / 2

        # Storm surge
        time0 = tp + abs(tau_ss_d) - 24

        if tau_ss_d < 0 and (time0 > 0):  # el hidrograma marca el inicio del tiempo
            # desde el inicio del hidrograma hasta que empieza el pico, el valor del surge es el base
            # time0=df_v.iloc[n]['Tp'] + abs(df_v.iloc[n]['Tau_ss_d']) - 24
            ss0 = np.full(int(time0 * 3600 / int_min), sb)
            # 24 horas hasta el pico
            ss1 = np.linspace(sb, sp, int(24 * 3600 / int_min))
            # 24 horas después del pico
            ss2 = np.linspace(sp, sb, int(24 * 3600 / int_min))
            # lo que quede hasta el final del hidrograma
            timef2 = tp + abs(tau_ss_d) + 24
            time_final = np.max(t) - (timef2 * 3600 - int_min)
            ss3 = np.full(int(time_final / int_min), sb)
            ss = np.hstack((ss0, ss1, ss2, ss3))

        else:
            # storm surge marca el inicio del tiempo, surge centrado en las primeras 48 horas
            ss1 = np.linspace(sb, sp, int(24 * 3600 / int_min))
            ss2 = np.linspace(sp, sb, int(24 * 3600 / int_min))
            timet = time0 + hidro_hours[-1]

            if timet < 48:
                ss = np.hstack((ss1, ss2))
            else:
                # a partir de las 48 horas, el surge tiene el valor del base
                time_final = np.max(t) - (48 * 3600 - int_min)
                ss3 = np.full(int(time_final / int_min), sb)
                ss = np.hstack((ss1, ss2, ss3))

        t_inicial = np.arange(0, 4 * 24 * 3600, int_min)
        ss_inicial = np.zeros(len(t_inicial)) + sb
        at_inicial = (
            np.sin(np.pi / 2 + np.pi * tau / 6 + 2 * np.pi * t_inicial / (12 * 3600))
            * cm
            / 2
        )
        q_inicial = np.zeros(len(t_inicial)) - qb

        t_series = np.hstack((t_inicial, t_inicial[-1] + t))
        at_series = np.hstack((at_inicial, at))

        if len(ss) < len(at):
            ss4 = np.full(int(len(at) - len(ss)), ss[-1])
            ss = np.hstack((ss, ss4))
            ss_series = np.hstack((ss_inicial, ss))
        else:
            ss_series = np.hstack((ss_inicial, ss))

        if len(hidro_Qp_5) < len(at):
            qq4 = np.full(int(len(at) - len(hidro_Qp_5)), -hidro_Qp_5[-1])
            hidro_Qp_5 = np.hstack((hidro_Qp_5, qq4))
            qq_series = np.hstack((q_inicial, -hidro_Qp_5))
        else:
            qq_series = np.hstack((q_inicial, -hidro_Qp_5))

        AT_SS_series = ss_series + at_series

        return t_series, at_series, ss_series, AT_SS_series, qq_series

    def rndays_calculation(
        self,
        case_context: dict,
        case_dir: str,
        t_series: np.ndarray,
    ):
        """
        Calculate the rndays parameter (time of simulation in days).

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        t_series : np.ndarray
            The time series array.
        """

        # Calculate the rndays parameter
        t_rndays = t_series[-1] / (3600 * 24)

        int_min = case_context.get("dt")  # Define the time step in seconds for .th file

        dt_nspool = case_context.get("dt") * case_context.get("nspool")

        # Generate time vector for .th file
        times_sim = np.arange(
            0,
            t_series[-1] + (case_context.get("ihfskip") / dt_nspool / 4.0) * int_min,
            int_min,
        )

        return t_rndays, times_sim

    def generate_wind_th(
        self,
        case_context: dict,
        case_dir: str,
        times: np.ndarray,
    ):
        """
        Generate the wind files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        ####### PARAMETERS OBTAINED FROM metamodel_parameters #######
        wind_magnitude = case_context.get("wind_magnitude")
        wind_direction_era5 = case_context.get("wind_direction")

        wind_direction_SCHISM = (90 - wind_direction_era5) % 360
        #############################################################

        u_wind = (
            -np.ones(len(times))
            * wind_magnitude
            * np.cos(np.radians(wind_direction_SCHISM))
        )
        v_wind = (
            -np.ones(len(times))
            * wind_magnitude
            * np.sin(np.radians(wind_direction_SCHISM))
        )

        # Define the output file path inside the directory
        output_path = os.path.join(case_dir, "wind.th")

        # Create the output file
        with open(output_path, "w+") as file:
            # Write each line as time, u_wind, and v_wind separated by spaces
            for time, u, v in zip(times, u_wind, v_wind):
                file.write(f"{time} {u} {v}\n")

    def generate_at_th(
        self,
        case_context: dict,
        case_dir: str,
        at_series: np.ndarray,
        times: np.ndarray,
    ):
        """
        Generate the at.th file for a case (not required by SCHISM, but nice to plot).

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        # Define the output file path inside the directory
        output_path = os.path.join(case_dir, "at.th")

        # Create the output file
        with open(output_path, "w+") as file:
            # Write each line as time and at_series separated by spaces
            for time, at_serie in zip(times, at_series):
                file.write(f"{time} {at_serie}\n")

    def generate_ss_th(
        self,
        case_context: dict,
        case_dir: str,
        ss_series: np.ndarray,
        times: np.ndarray,
    ):
        """
        Generate the ss.th file for a case (not required by SCHISM, but nice to plot).

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        # Define the output file path inside the directory
        output_path = os.path.join(case_dir, "ss.th")

        # Create the output file
        with open(output_path, "w+") as file:
            # Write each line as time and ss_series separated by spaces
            for time, ss_serie in zip(times, ss_series):
                file.write(f"{time} {ss_serie}\n")

    def generate_elev_th(
        self,
        case_context: dict,
        case_dir: str,
        AT_SS_series: np.ndarray,
        times: np.ndarray,
    ):
        """
        Generate the elev.th file for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        # Define the output file path inside the directory
        output_path = os.path.join(case_dir, "elev.th")

        # Create the output file
        with open(output_path, "w+") as file:
            # Write each line as time and AT_SS_series separated by spaces
            for time, ss_serie in zip(times, AT_SS_series):
                file.write(f"{time} {ss_serie}\n")

    def generate_flux_th(
        self,
        case_context: dict,
        case_dir: str,
        qq_series: np.ndarray,
        times: np.ndarray,
    ):
        """
        Generate the elev.th file for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        # Define the output file path inside the directory
        output_path = os.path.join(case_dir, "flux.th")

        # Create the output file
        with open(output_path, "w+") as file:
            # Write each line as time and AT_SS_series separated by spaces
            for time, qq_serie in zip(times, qq_series):
                file.write(f"{time} {qq_serie}\n")

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ):
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        super().build_case(case_context=case_context, case_dir=case_dir)

        # Calculate time series from the cases
        t_series, at_series, ss_series, AT_SS_series, qq_series = (
            self.generate_time_series(
                case_context=case_context,
                case_dir=case_dir,
            )
        )

        # Calculate the rndays parameter and time vector for the th files
        case_context["rnday"], times_sim = self.rndays_calculation(
            case_context=case_context,
            case_dir=case_dir,
            t_series=t_series,
        )

        # Generate the wind.th file for each case
        self.generate_wind_th(
            case_context=case_context,
            case_dir=case_dir,
            times=times_sim,
        )

        # Generate the at.th file for each case
        self.generate_at_th(
            case_context=case_context,
            case_dir=case_dir,
            at_series=at_series,
            times=times_sim,
        )

        # Generate the ss.th file for each case
        self.generate_ss_th(
            case_context=case_context,
            case_dir=case_dir,
            ss_series=ss_series,
            times=times_sim,
        )

        # Generate the ss.th file for each case
        self.generate_elev_th(
            case_context=case_context,
            case_dir=case_dir,
            AT_SS_series=AT_SS_series,
            times=times_sim,
        )

        # Generate the flux.th file for each case
        self.generate_flux_th(
            case_context=case_context,
            case_dir=case_dir,
            qq_series=qq_series,
            times=times_sim,
        )


class SchismOnlyWindModelWrapper(SchismModelWrapper):
    """
    Schism wrapper: only wind case.
    """

    default_parameters = {
        # .param file
        "indvel": {
            "type": float,
            "value": 1,
            "description": "Method for computing velocity at nodes: 0= linear shape function; 1 = averaging method",
        },
        "ihorcon": {
            "type": float,
            "value": 0,
            "description": "Horizontal viscosity option: 0= no viscosity is used; 1= Lapacian",
        },
        "ishapiro": {
            "type": float,
            "value": 0,
            "description": "on/off flag for Shapiro filter: 0= off; 1= on",
        },
        "inter_mom": {
            "type": float,
            "value": 0,
            "description": " Interpolation is used for velocity at foot of char: 0=linear; 1=Kriging (define kr_co ); -1= spatial Kriging (define krvel.gr3)",
        },
        # CORE
        "ipre": {
            "type": float,
            "value": 0,
            "description": "Pre-process flag: 0= normal run; 1= only for single CPU!",
        },
        "ibc": {
            "type": float,
            "value": 0,
            "description": "Barotropic flag",
        },
        "ibtp": {
            "type": float,
            "value": 0,
            "description": "Baroclinic flag",
        },
        # TRACERS
        "ntracer_gen": {
            "type": float,
            "value": 2,
            "description": "# of tracers",
        },
        "ntracer_age": {
            "type": float,
            "value": 4,
            "description": "Age calculation, must be 2*ntracer_gen",
        },
        # SEDIMENTS
        "sed_class": {
            "type": float,
            "value": 5,
            "description": "SED3D",
        },
        # ECO
        "eco_class": {
            "type": float,
            "value": 27,
            "description": "EcoSim. Must be bewteen [25,60]",
        },
        # OPT
        "ieos_type": {
            "type": float,
            "value": 0,
            "description": "UNESCO 1980 (nonlinear); =1: linear function of T ONLY, i.e. ",
        },
        "ieos_pres": {
            "type": float,
            "value": 0,
            "description": "Used only if ieos_type=0. 0: without pressure effects ",
        },
        "eos_a": {
            "type": float,
            "value": -0.1,
            "description": "Needed if ieos_type=1; should be <=0",
        },
        "eos_b": {
            "type": float,
            "value": 1001,
            "description": "Needed if ieos_type=1",
        },
        # WWM
        "msc2": {
            "type": float,
            "value": 24,
            "description": "Grid for WWM. Same as msc in .nml ... for consitency check between SCHISM and WWM",
        },
        "mdc2": {
            "type": float,
            "value": 30,
            "description": "Grid for WWM. Same as mdc in .nml ... for consitency check between SCHISM and WWM",
        },
        # SCHOUT
        "output_WL": {
            "type": float,
            "value": 1,
            "description": "Elev. [m] = 0: off; 1: on",
        },
        "output_wind_velocity": {
            "type": float,
            "value": 1,
            "description": "Wind velocity vector [m/s] = 0: off; 1: on",
        },
        "output_water_temperature": {
            "type": float,
            "value": 0,
            "description": "Water temperature [C] = 0: off; 1: on",
        },
        "output_depth_averaged_vel": {
            "type": float,
            "value": 1,
            "description": "Depth-averaged vel vector [m/s] = 0: off; 1: on",
        },
        "output_water_salinity": {
            "type": float,
            "value": 0,
            "description": "Water salinity [PSU] = 0: off; 1: on",
        },
    }

    def generate_windth(
        self,
        case_context: dict,
        case_dir: str,
    ):
        """
        Generate the wind files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        ####### PARAMETERS OBTAINED FROM metamodel_parameters #######
        wind_magnitude = case_context.get("wind_magnitude")
        wind_direction = case_context.get("wind_direction")
        #############################################################

        total_seconds = (
            case_context.get("rndays") * 24 * 3600
        )  # Define times in seconds for .th file
        times = np.arange(
            0, total_seconds + case_context.get("dt"), case_context.get("dt")
        )  # Generate time vector
        u_wind = -[wind_magnitude * np.cos(np.radians(wind_direction))] * len(
            times
        )  # Calculate u_wind
        v_wind = -[wind_magnitude * np.sin(np.radians(wind_direction))] * len(
            times
        )  # Calculate v_wind

        # Define the output file path inside the directory
        output_path = os.path.join(case_dir, "wind.th")

        # Create the output file
        with open(output_path, "w+") as file:
            # Write each line as time, u_wind, and v_wind separated by spaces
            for time, u, v in zip(times, u_wind, v_wind):
                file.write(f"{time} {u} {v}\n")

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ):
        """
        Build the input files for a case.

        Parameters
        ----------
        case_context : dict
            The case context.
        case_dir : str
            The case directory.
        """

        super().build_case(case_context=case_context, case_dir=case_dir)

        case_context["start_year"] = case_context.get("tini").year
        case_context["start_month"] = case_context.get("tini").month
        case_context["start_day"] = case_context.get("tini").day
        case_context["start_hour"] = case_context.get("tini").hour
        case_context["rndays"] = (
            case_context.get("tend") - case_context.get("tini")
        ).total_seconds() / 86400

        # Generate wind file for each case
        self.generate_windth(
            case_context=case_context,
            case_dir=case_dir,
        )

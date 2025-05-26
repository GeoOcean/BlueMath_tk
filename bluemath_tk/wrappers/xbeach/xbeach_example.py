import os.path as op
import sys

import numpy as np
import pandas as pd
import xarray as xr
from wavespectra.construct.direction import cartwright
from wavespectra.construct.frequency import jonswap

from bluemath_tk.wrappers.xbeach.xbeach_wrapper import XBeachModelWrapper

example_directions = [
    7.5,
    22.5,
    37.5,
    52.5,
    67.5,
    82.5,
    97.5,
    112.5,
    127.5,
    142.5,
    157.5,
    172.5,
    187.5,
    202.5,
    217.5,
    232.5,
    247.5,
    262.5,
    277.5,
    292.5,
    307.5,
    322.5,
    337.5,
    352.5,
]
example_frequencies = [
    0.035,
    0.0385,
    0.04235,
    0.046585,
    0.0512435,
    0.05636785,
    0.06200463,
    0.0682051,
    0.07502561,
    0.08252817,
    0.090781,
    0.0998591,
    0.10984501,
    0.12082952,
    0.13291247,
    0.14620373,
    0.1608241,
    0.17690653,
    0.19459718,
    0.21405691,
    0.2354626,
    0.25900885,
    0.28490975,
    0.31340075,
    0.3447408,
    0.37921488,
    0.4171364,
    0.45885003,
    0.50473505,
]


def create_dep(filename, dep):
    with open(filename, "w") as f:
        for row in dep:
            for j in range(0, len(row), 12):
                sub_row = row[j : j + 12]
                if len(sub_row) < 12:
                    sub_row = np.append(sub_row, -999)
                for val in sub_row:
                    if val >= 0:
                        f.write(f"   {val:.7E}")
                    else:
                        f.write(f"  {val:.7E}")
                f.write("\n")


class ManuXBeachModelWrapper(XBeachModelWrapper):
    """
    Wrapper for the XBeach model.
    """

    locations = {
        "x": [
            465859.59375,
            466361.03125,
            466962.75,
            467564.46875,
        ],
        "y": [
            4810529.5,
            4809724.5,
            4808919.5,
            4808114.5,
        ],
    }

    @staticmethod
    def create_vardens(ds):
        """
        Create the vardens file from the dataset.
        """

        t = ""

        # Frequencies
        t += "{0} \n".format(len(ds.freq))
        for freq in ds.freq.values:
            t += "{0}\n".format(freq)

        # Directions
        t += "{0} \n".format(len(ds.dir))
        for dirt in sorted(ds.dir.values):
            t += "{0}\n".format(dirt)

        # Sea_surface_wave_directional_variance_spectral_density
        for _pf, freq in enumerate(ds.freq.values):
            for _pd, dirt in enumerate(sorted(ds.dir.values)):
                var = ds.sel(freq=freq, dir=dirt).efth.values
                if np.isnan(var):
                    var = 0.0
                t += "{0}\t".format(var)
            t += "\n"

        return t

    @staticmethod
    def write_file(path, text):
        with open(path, "w") as f:
            f.write(text)

    def build_case(
        self,
        case_context: dict,
        case_dir: str,
    ):
        """
        Build the XBeach case.
        """

        case_context["case_number_string"] = f"{case_context.get('case_index'):03d}"

        # /////////////////////// BATHY \\\\\\\\\\\\\\\\\\\\\\\\\
        pcs = (
            case_context.get("subset_mda")
            .isel(case=case_context.get("case_index"))[["PC1", "PC2", "PC3", "PC4"]]
            .to_array()
            .values
        )  # PCs
        anomalies = pcs @ case_context.get("eofs")  # Anomalies
        Z = np.reshape(
            anomalies + case_context.get("mean_bat").values.flatten(),
            (case_context.get("dems").x.shape[0], case_context.get("dems").x.shape[1]),
        )
        create_dep(
            op.join(
                case_dir,
                f"bathy_{case_context.get('case_index'):03d}.dep",
            ),
            Z,
        )

        # /////////////////////// SPECTRA \\\\\\\\\\\\\\\\\\\\\\\\\\
        ef = jonswap(
            freq=example_frequencies,
            fp=1
            / case_context.get("subset_mda")
            .isel(case=case_context.get("case_index"))
            .Tp.values,
            gamma=case_context.get("subset_mda")
            .isel(case=case_context.get("case_index"))
            .gamma.values,
            hs=case_context.get("subset_mda")
            .isel(case=case_context.get("case_index"))
            .Hs.values,
        )
        gth = cartwright(
            dir=example_directions,
            dm=case_context.get("subset_mda")
            .isel(case=case_context.get("case_index"))
            .Dir.values,
            dspr=case_context.get("subset_mda")
            .isel(case=case_context.get("case_index"))
            .SPR.values,
        )
        efth = ef * gth  # Offshore spectra
        off_sp = xr.Dataset(
            {"efth": (["time", "freq", "dir"], efth.expand_dims("time", axis=0).data)},
            coords={
                "dir": example_directions,
                "time": pd.to_datetime(["1996-09-23 11:00"]),
                "freq": example_frequencies,
            },
        )
        # BinWaves
        sp = off_sp.sortby(["freq", "dir"])

        for p, (x, y) in enumerate(zip(self.locations["x"], self.locations["y"])):
            kp = xr.open_dataset(
                op.join("inputs", "kp_grid", "kp_lon_{0}_lat_{1}.nc".format(x, y))
            )

            EFTH = np.full(np.shape(sp.efth.values), 0)

            for case in range(len(kp.case)):
                sys.stdout.write(
                    "Reconstructing Point {0}. Bin {1} \r".format(p + 1, case + 1)
                )
                sys.stdout.flush()

                freq_, dir_ = (
                    case_context.get("subset_mda").isel(case=case).freq.values,
                    case_context.get("subset_mda").isel(case=case).Dir.values,
                )
                efth_case = sp.sel(freq=freq_, dir=dir_, method="nearest")
                kp_case = kp.sortby("dir").isel(case=case)

                EFTH = EFTH + (efth_case.efth * kp_case.kp**2).values

            near_sp = sp.copy()
            near_sp["efth"] = (
                ("time", "freq", "dir"),
                EFTH,
            )  # Nearshore spectra (reconstructed)

            # Save vardens
            vardens = near_sp.copy()
            vardens["dir"] = (270 - (vardens["dir"])) % 360
            spec = self.create_vardens(vardens.isel(time=0))
            self.write_file(
                op.join(case_dir, "vardens{0}.txt").format(p + 1),
                spec,
            )

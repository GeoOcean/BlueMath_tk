import numpy as np
import xarray as xr


def transform_CAWCR_WS(ds):
    ds = ds.rename({"frequency": "freq", "direction": "dir"})
    ds["efth"] = ds["efth"] * np.pi / 180.0
    ds["dir"] = ds["dir"] - 180.0
    ds["dir"] = np.where(ds["dir"] < 0, ds["dir"] + 360, ds["dir"])
    return ds


def process_kp_coefficients(
    swan_ds: xr.Dataset,
    spectrum_freq: np.ndarray,
    spectrum_dir: np.ndarray,
    latitude: float,
    longitude: float,
):
    """
    This function processes the propagation coefficients for all the grid points within the
    SWAN simulation output (out_sim).
    It takes a long time to run but it only needs to be done once per location

    p_store_kp - location to save all the kp coefficients. One file per location
    out_sim    - output from SWAN simulations. Dimensions: cases x lat x lon.
                 variables: Hs_part, Tp_part, Dir_part
    spectrum     - spectra dataset (freq and dir coordinates)
    override   - bool, True for re-calculate already solved points
    rotated_mesh - if True looks for closer location in Xp and Yp
    """

    kp_matrix = np.full(
        [
            len(swan_ds.case_num),
            len(spectrum_freq),
            len(spectrum_dir),
        ],
        0.0,
    )

    swan_point = swan_ds.sel(
        Xp=longitude,
        Yp=latitude,
        method="nearest",
    )
    # TODO: Check if this is the correct way to handle NaN values
    if any(np.isnan(swan_point["TPsmoo"].values)):
        raise ValueError("NaN values found for variable TPsmoo_part")

    # Tp mask
    swan_point_cut = swan_point[["Hsig", "TPsmoo", "Dir"]].where(
        swan_point["TPsmoo"] > 0,
        drop=True,
    )

    # get k,f,d
    kfd = xr.Dataset(
        {
            "k": swan_point_cut.Hsig,
            "f": 1 / swan_point_cut.TPsmoo,
            "d": swan_point_cut.Dir,
        }
    )

    # fill kp
    for case_num in kfd.case_num.values:
        kfd_c = kfd.sel(case_num=case_num)

        # get k,f,d and clean nans
        k = kfd_c.k.values
        f = kfd_c.f.values
        d = kfd_c.d.values

        k = k[~np.isnan(f)]
        d = d[~np.isnan(f)]
        f = f[~np.isnan(f)]

        # set case kp at point
        for c in range(len(f)):
            i = np.argmin(np.abs(spectrum_freq - f[c]))
            j = np.argmin(np.abs(spectrum_dir - d[c]))
            kp_matrix[case_num, i, j] = k[c]

    # prepare output dataset
    return xr.Dataset(
        {
            "kp": (["case", "freq", "dir"], kp_matrix),
            "swan_freqs": (["case"], 1.0 / swan_point_cut.TPsmoo),
            "swan_dirs": (["case"], swan_point_cut.Dir),
        },
        coords={
            "case": swan_ds.case_num,
            "freq": spectrum_freq,
            "dir": spectrum_dir,
        },
    )


def reconstruc_spectra(
    spectra_ds: xr.Dataset,
    kp_coeffs: xr.Dataset,
):
    EFTH = np.full(
        np.shape(spectra_ds.efth.values),
        0,
    )

    for case in range(len(kp_coeffs.case)):
        freq_, dir_ = (
            kp_coeffs.isel(case=case).swan_freqs.values,
            kp_coeffs.isel(case=case).swan_dirs.values,
        )
        efth_case = spectra_ds.sel(freq=freq_, dir=dir_, method="nearest")
        kp_case = kp_coeffs.sortby("dir").isel(case=case)

        EFTH = EFTH + (efth_case.efth * kp_case.kp**2).values

    # ns_sp = off_sp.drop(("Wspeed", "Wdir", "Depth")).copy()

    return xr.Dataset(
        {
            "efth": (["time", "freq", "dir"], EFTH),
        },
        coords={
            "time": spectra_ds.time,
            "freq": spectra_ds.freq,
            "dir": spectra_ds.dir,
        },
    )

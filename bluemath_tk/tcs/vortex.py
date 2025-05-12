import numpy as np
import xarray as xr
from great_circle_calculator.great_circle_calculator import distance_between_points


def get_xy_labels(coords_mode):
    "returns labels for x,y axis, function of swan coordinates mode"

    if coords_mode == "SPHERICAL":
        lab_x, lab_x_long = "lon", "Longitude (º)"
        lab_y, lab_y_long = "lat", "Latitude (º)"

    elif coords_mode == "CARTESIAN":
        lab_x, lab_x_long = "X", "X (m)"
        lab_y, lab_y_long = "Y", "Y (m)"

    return lab_x, lab_x_long, lab_y, lab_y_long


def vortex_model(storm_track, swan_mesh, coords_mode="SPHERICAL"):
    """
    storm_track - (pandas.DataFrame)
    - obligatory fields:            vfx, vfy, p0, pn, index, vmax, rmw
    - for SPHERICAL coordinates:    lon, lat
    - for CARTESIAN coordiantes:    x, y, latitude

    swan_mesh   -  computational grid (mxc, myc, xpc, ypc, xlenc, ylenc)

    coords_mode - 'SHPERICAL' / 'CARTESIAN' swan project coordinates mode

    The Dynamic Holland Model is used to generate wind vortex fields from
    storm track coordinate parameters.
    Wind model code (from ADCIRC, transcribed by Antonio Espejo) and later
    modified by Sara O. van Vloten to include TCs at southern hemisphere

    returns:  xds_vortex (xarray.Dataset)
    """

    # parameters
    beta = 0.9  # conversion factor of wind speed
    rho_air = 1.15  # air density
    w = 2 * np.pi / 86184.2  # Earth's rotation velocity [rad/s]
    pifac = np.arccos(-1) / 180  # pi/180
    one2ten = 0.93  # conversion of 1-min winds to 10-min avg
    # note: Harper (2010) obtained a higher factor than the traditional 0.8928

    # input variables
    storm_vfx = storm_track.vfx.values[:]  # [kt] translational speed
    storm_vfy = storm_track.vfy.values[:]
    storm_p0 = storm_track.p0.values[:]  # [mbar] minimum central pressure
    storm_pn = storm_track.pn.values[:]
    times = storm_track.index[:]
    storm_vmax = storm_track.vmax.values[:]  # [kt] max wind speed (1-min avg)
    storm_rmw = storm_track.rmw.values[:]  # [nmile] radii of maxwinds

    # coordinate system dependant variables
    if coords_mode == "SPHERICAL":
        storm_x = storm_track.lon.values[:]
        storm_y = storm_track.lat.values[:]
        storm_lat = storm_track.lat.values[:]

    if coords_mode == "CARTESIAN":
        storm_x = storm_track.x.values[:]
        storm_y = storm_track.y.values[:]
        storm_lat = storm_track.latitude.values[:]

    # correction when track is in south hemisphere for vortex generation
    south_hemisphere = any(i < 0 for i in storm_lat)

    # swan mesh: computational grid (for generating vortex wind)
    mxc = swan_mesh.cg["mxc"]
    myc = swan_mesh.cg["myc"]
    xpc = swan_mesh.cg["xpc"]
    ypc = swan_mesh.cg["ypc"]
    xpc_xlenc = swan_mesh.cg["xpc"] + swan_mesh.cg["xlenc"]
    ypc_ylenc = swan_mesh.cg["ypc"] + swan_mesh.cg["ylenc"]

    # prepare meshgrid
    cg_lon = np.linspace(xpc, xpc_xlenc, mxc)
    cg_lat = np.linspace(ypc, ypc_ylenc, myc)
    mg_lon, mg_lat = np.meshgrid(cg_lon, cg_lat)

    # wind output holder
    hld_W = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))
    hld_D = np.zeros((len(cg_lat), len(cg_lon), len(storm_p0)))

    # each time needs 2D (mesh) wind files (U,V)
    for c, (lo, la, la_orig, p0, pn, ut, vt, vmax, rmw) in enumerate(
        zip(
            storm_x,
            storm_y,
            storm_lat,
            storm_p0,
            storm_pn,
            storm_vfx,
            storm_vfy,
            storm_vmax,
            storm_rmw,
        )
    ):
        # generate vortex field when storm is given
        if (
            all(np.isnan(i) for i in (lo, la, la_orig, p0, pn, ut, vt, vmax, rmw))
            == False
        ):
            # get distance and angle between points
            r = distance_between_points((mg_lon, mg_lat), (lo, la))

            # angle correction for southern hemisphere
            if south_hemisphere:
                thet = np.arctan2((mg_lat - la) * pifac, -(mg_lon - lo) * pifac)
            else:
                thet = np.arctan2((mg_lat - la) * pifac, (mg_lon - lo) * pifac)

            # wind model (adcirc)
            CPD = (pn - p0) * 100  # central pressure deficit [Pa]
            if CPD < 100:
                CPD = 100  # limit central pressure deficit

            f = 2 * w * np.sin(abs(la_orig) * np.pi / 180)  # Coriolis

            # Substract the translational storm speed from the observed maximum
            # wind speed to avoid distortion in the Holland curve fit.
            # The translational speed will be added back later
            vkt = vmax - np.sqrt(np.power(ut, 2) + np.power(vt, 2))  # [kt]

            # Convert wind speed from 10m altitude to wind speed at the top of
            # the atmospheric boundary layer
            vgrad = vkt / beta  # [kt]
            vm = vgrad * 0.52  # [m/s]

            # radii of maximum winds (historical and/or estimated, Knaff 2016)
            rm = rmw  # [nmile]
            rm *= 1.852 * 1000  # [nmile to m]
            rn = rm / r  # []

            # Holland B parameter with upper and lower limits
            B = rho_air * np.exp(1) * np.power(vm, 2) / CPD
            if B > 2.5:
                B = 2.5
            elif B < 1:
                B = 1

            # Wind velocity at each node and time step   [m/s]
            vg = (
                np.sqrt(
                    np.power(rn, B) * np.exp(1 - np.power(rn, B)) * np.power(vm, 2)
                    + np.power(r, 2) * np.power(f, 2) / 4
                )
                - r * f / 2
            )

            # Determine translation speed that should be added to final storm
            # wind speed. This is tapered to zero as the storm wind tapers to
            # zero toward the eye of the storm and at far long distances
            vtae = (abs(vg) / vgrad) * ut  # [m/s]
            vtan = (abs(vg) / vgrad) * vt

            # find the velocity components and convert from wind at the top
            # of the atmospheric boundary layer to wind at 10m elevation
            hemisphere_sign = 1 if south_hemisphere else -1
            ve = hemisphere_sign * vg * beta * np.sin(thet)  # [m/s]
            vn = vg * beta * np.cos(thet)

            # convert from 1-minute averaged winds to 10-minute averaged winds
            ve = ve * one2ten  # [m/s]
            vn = vn * one2ten

            # add the storm translation speed
            vfe = ve + vtae  # [m/s]
            vfn = vn + vtan

            # wind module
            W = np.sqrt(np.power(vfe, 2) + np.power(vfn, 2))  # [m/s]

            # surface pressure field
            pr = p0 + (pn - p0) * np.exp(-np.power(rn, B))  # [mbar]
            py, px = np.gradient(pr)
            ang = np.arctan2(py, px) + np.sign(la_orig) * np.pi / 2.0

            # hold wind data (m/s)
            hld_W[:, :, c] = W
            hld_D[:, :, c] = 270 - np.rad2deg(ang)  # direction (º clock. north)

        else:
            # hold null wind data when storm coordinates are not provided
            hld_W[:, :, c] = 0
            hld_D[:, :, c] = 0  # direction (º clock. north)

    # spatial axis labels
    lab_x, lab_x_long, lab_y, lab_y_long = get_xy_labels(coords_mode)

    # generate vortex dataset
    xds_vortex = xr.Dataset(
        {
            "W": ((lab_y, lab_x, "time"), hld_W, {"units": "m/s"}),
            "Dir": ((lab_y, lab_x, "time"), hld_D, {"units": "º"}),
        },
        coords={
            lab_y: cg_lat,
            lab_x: cg_lon,
            "time": times,
        },
    )
    xds_vortex.attrs["xlabel"] = lab_x_long
    xds_vortex.attrs["ylabel"] = lab_y_long

    return xds_vortex

import copernicusmarine

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
    dataset_version="202411",
    variables=[
        "VHM0",
        "VHM0_SW1",
        "VHM0_SW2",
        "VHM0_WW",
        "VMDR",
        "VMDR_SW1",
        "VMDR_SW2",
        "VMDR_WW",
        "VPED",
        "VSDX",
        "VSDY",
        "VTM01_SW1",
        "VTM01_SW2",
        "VTM01_WW",
        "VTM02",
        "VTM10",
        "VTPK",
    ],
    minimum_longitude=-10.43452696741375,
    maximum_longitude=-0.5556814090161573,
    minimum_latitude=42.03998421470398,
    maximum_latitude=46.133428506857676,
    start_datetime="1980-01-01T00:00:00",
    end_datetime="2023-04-30T21:00:00",
    coordinates_selection_method="strict-inside",
    disable_progress_bar=False,
)

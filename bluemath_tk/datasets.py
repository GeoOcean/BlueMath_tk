'''

- Project: Bluemath{toolkit}
- File: data_examples.py
- Description: Examples Datasets
- Author: GeoOcean Research Group, Universidad de Cantabria
- Created Date: 26 January 2024
- License: MIT
- Repository: https://gitlab.com/geoocean/bluemath/toolkit/

'''

import os

import numpy as np
import xarray as xr
import pandas as pd

p_data = '/lustre/geocean/DATA/hidronas1/volume3/Laura/DATA/'

class DATA:

    def get_2d_dataset():
        np.random.seed(42)
        # Definir longitudes y latitudes
        coord1 = np.linspace(-100, 100, 20)
        coord2 = np.linspace(-100, 100, 20)
        coord3 = np.arange(1, 50)  # Por ejemplo, 10 pasos de tiempo
        
        # Crear una cuadrícula 3D con las coordenadas lon, lat y tiempo
        coord1, coord2, coord3 = np.meshgrid(coord1, coord2, coord3, indexing='ij')
        
        # Crear una variable X que dependa de lon, lat y tiempo (función sinusoidal en el tiempo)
        X = (
            np.sin(np.radians(coord1)) * np.cos(np.radians(coord2)) * np.sin(coord3) +
            np.sin(2 * np.radians(coord1)) * np.cos(2 * np.radians(coord2)) * np.sin(2 * coord3) +
            np.sin(3 * np.radians(coord1)) * np.cos(3 * np.radians(coord2)) * np.sin(3 * coord3)
        )
        
        Y = -np.sin(X)
        
        
        # Crear un dataset con xarray
        ds = xr.Dataset(
            {
                'X': (['coord1', 'coord2', 'coord3'], X),
                'Y': (['coord1', 'coord2', 'coord3'], Y),
            },
            coords={'coord1': coord1[:, 0, 0], 'coord2': coord2[0, :, 0], 'coord3': coord3[0, 0, :]}
        )
        
        return ds


    def get_1d_dataframe():
        # Set random seed for reproducibility
        np.random.seed(42)
        
        #Example with random Data
        df = pd.DataFrame(
            {
                'x1':np.random.rand(1000)*7, 
                'x2':np.random.rand(1000)*20, 
                'x3':np.random.rand(1000)*50,
            }
        )

        return df

    def get_bulk_waves(p_data = p_data):

        ds_waves = xr.open_dataset(os.path.join(p_data, 'examples_bluemath', 'example_bulk_waves.nc'))
        
        return ds_waves


    def get_part_waves(p_data = p_data):

        ds_waves = xr.open_dataset(os.path.join(p_data, 'examples_bluemath', 'example_part_waves.nc'))
        
        return ds_waves

        



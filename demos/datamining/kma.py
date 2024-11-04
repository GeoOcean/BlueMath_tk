from bluemath_tk.datamining.kma import KMA
import numpy as np
import pandas as pd

# For MDA class usage, a pandas dataframe is required. Each column will represent a different variable

# Example with random Data
np.random.seed(42)
df = pd.DataFrame({
    'Hs': np.random.rand(1000)*7,
    'Tp': np.random.rand(1000)*20,
    'Dir': np.random.rand(1000)*360
})


kma_ob = KMA(data=df, ix_directional=['Dir'])
kma_ob.run(10)
kma_ob.scatter_data()
kma_ob.scatter_data(norm=True, plot_centroids=True)
kma_ob.scatter_bmus()
kma_ob.scatter_bmus(plot_centroids=True)
kma_ob.scatter_bmus(norm=True,plot_centroids=True)



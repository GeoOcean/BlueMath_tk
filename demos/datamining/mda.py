
from bluemath_tk.datamining.mda import MDA
import numpy as np
import pandas as pd

## Data

#For MDA class usage, a pandas dataframe is required. Each column will represent a different variable

#Example with random Data

df = pd.DataFrame({
    'Hs': np.random.rand(1000)*7,
    'Tp': np.random.rand(1000)*20,
    'Dir': np.random.rand(1000)*360
})


mda_ob = MDA(data=df, ix_directional=['Dir'])
mda_ob.run(10)
centroid=mda_ob.nearest_centroid(df.values[2])
mda_ob.scatter_data()
mda_ob.scatter_data(plot_centroids=True)
mda_ob.scatter_data(norm=True, plot_centroids=True)


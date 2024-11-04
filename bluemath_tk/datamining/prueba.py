import pandas as pd
import numpy as np
from bluemath_tk.datamining.mda import MDA

df = pd.DataFrame({
    'Hs': np.random.rand(1000)*7,
    'Tp': np.random.rand(1000)*20,
    'Dir': np.random.rand(1000)*360
})
mda_ob = MDA(data=df, ix_directional=['Dir'])
mda_ob.run(10)
mda_ob.scatter_data()
import bluemath_tk.core.data 
import pandas as pd
import numpy as np
data = pd.DataFrame({
    'Hs': np.random.rand(1000)*7,
     'Tp': np.random.rand(1000)*20,
     'Dir': np.random.rand(1000)*720
})
scale_factor = {'Hs': [0.015, 7], 'Tp': [0, 20]}
ix_directional = ['Dir']
data_norm, scale_factor = bluemath_tk.core.data.normalize(data, ix_directional)

print(data_norm)
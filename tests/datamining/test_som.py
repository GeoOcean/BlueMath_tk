import unittest
import numpy as np
import pandas as pd
from bluemath_tk.datamining.som import SOM


class TestSOM(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "Hs": np.random.rand(1000) * 7,
                "Tp": np.random.rand(1000) * 20,
                "Dir": np.random.rand(1000) * 360,
            }
        )
        self.som = SOM(som_shape=(3, 3), num_dimensions=4)

    def test_fit(self):
        self.som.fit(data=self.df, directional_variables=["Dir"])
        self.assertIsInstance(self.som.centroids, pd.DataFrame)
        self.assertEqual(self.som.centroids.shape[0], 9)

    def test_predict(self):
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        )
        self.som.fit(data=self.df, directional_variables=["Dir"])
        nearest_centroids, nearest_centroid_df = self.som.predict(data=data_sample)
        self.assertIsInstance(nearest_centroids, np.ndarray)
        self.assertEqual(len(nearest_centroids), 15)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 15)

    def test_fit_predict(self):
        nearest_centroids, nearest_centroid_df = self.som.fit_predict(
            data=self.df, directional_variables=["Dir"]
        )
        self.assertIsInstance(nearest_centroids, np.ndarray)
        self.assertEqual(len(nearest_centroids), 1000)
        self.assertIsInstance(nearest_centroid_df, pd.DataFrame)
        self.assertEqual(nearest_centroid_df.shape[0], 1000)

    def test_activation_response(self):
        self.som.fit(data=self.df, directional_variables=["Dir"])
        act_resp = self.som.activation_response()
        self.assertIsInstance(act_resp, np.ndarray)
        self.assertEqual(act_resp.shape, (3, 3))

    def test_get_centroids_probs_for_labels(self):
        data_sample = pd.DataFrame(
            {
                "Hs": np.random.rand(15) * 7,
                "Tp": np.random.rand(15) * 20,
                "Dir": np.random.rand(15) * 360,
            }
        )
        self.som.fit(data=self.df, directional_variables=["Dir"])
        centroids_probs = self.som.get_centroids_probs_for_labels(
            data=data_sample,
            labels=np.random.randint(0, 3, 15),
        )
        self.assertIsInstance(centroids_probs, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()

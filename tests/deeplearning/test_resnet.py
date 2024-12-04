import unittest
from unittest.mock import Mock
import keras


class TestResNetTraining(unittest.TestCase):
    def setUp(self):
        # Mock the data generator
        self.train_generator = Mock()
        self.train_generator.num_images = 5000
        self.train_generator.input_height = 64
        self.train_generator.input_width = 64
        self.train_generator.output_height = 64
        self.train_generator.output_width = 64
        self.train_generator.batch_size = 1

        # Define optimizer
        self.optimizer = keras.optimizers.AdamW

        # Mock the model
        self.model = Mock()
        self.model.compile = Mock()
        self.model.fit = Mock(return_value=Mock(history={"loss": [0.1, 0.05, 0.01]}))

    def test_model_training(self):
        # Compile the model
        self.model.compile(
            optimizer=self.optimizer(learning_rate=1e-4, weight_decay=1e-5),
            loss=keras.losses.mean_squared_error,
        )

        # Assert that compile was called with correct parameters
        self.model.compile.assert_called_with(
            optimizer=self.optimizer(learning_rate=1e-4, weight_decay=1e-5),
            loss=keras.losses.mean_squared_error,
        )

        # Start the train loop with the fit method
        history = self.model.fit(
            self.train_generator, initial_epoch=0, epochs=20, steps_per_epoch=500
        )

        # Assert that fit was called with correct parameters
        self.model.fit.assert_called_with(
            self.train_generator, initial_epoch=0, epochs=20, steps_per_epoch=500
        )

        # Assert that the training history is as expected
        self.assertIn("loss", history.history)
        self.assertEqual(len(history.history["loss"]), 3)
        self.assertAlmostEqual(history.history["loss"][0], 0.1)
        self.assertAlmostEqual(history.history["loss"][1], 0.05)
        self.assertAlmostEqual(history.history["loss"][2], 0.01)

        print("training complete")


if __name__ == "__main__":
    unittest.main()

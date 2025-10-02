import os
import os.path as op
import shutil
import tempfile
import unittest

import xarray as xr

from bluemath_tk.wrappers._base_wrappers import DummyModelWrapper


class TestBaseModelWrapper(unittest.TestCase):
    """Test the BaseModelWrapper class."""

    def setUp(self):
        """Set up the test environment."""

        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = op.join(self.test_dir, "templates")
        self.output_dir = op.join(self.test_dir, "output")
        os.makedirs(self.templates_dir)
        os.makedirs(self.output_dir)

        # Create a simple template file
        self.template_content = """
            parameter: {{ parameter }}
            value: {{ value }}
            fixed_parameter: {{ fixed_parameter }}
        """
        with open(op.join(self.templates_dir, "test.template"), "w") as f:
            f.write(self.template_content)

        # Test parameters
        self.metamodel_parameters = {
            "parameter": ["A", "B", "C"],
            "value": [1, 2, 3],
        }
        self.fixed_parameters = {
            "fixed_parameter": "fixed_value",
        }

        # Create wrapper instance
        self.wrapper = DummyModelWrapper(
            templates_dir=self.templates_dir,
            metamodel_parameters=self.metamodel_parameters,
            fixed_parameters=self.fixed_parameters,
            output_dir=self.output_dir,
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_load_cases_one_by_one(self):
        """Test loading cases in one_by_one mode"""

        self.wrapper.load_cases(mode="one_by_one")
        self.assertEqual(len(self.wrapper.cases_context), 3)
        self.assertEqual(len(self.wrapper.cases_dirs), 3)

        # Check case contexts
        for i, context in enumerate(self.wrapper.cases_context):
            self.assertEqual(
                context["parameter"], self.metamodel_parameters["parameter"][i]
            )
            self.assertEqual(context["value"], self.metamodel_parameters["value"][i])
            self.assertEqual(
                context["fixed_parameter"], self.fixed_parameters["fixed_parameter"]
            )

    def test_load_cases_all_combinations(self):
        """Test loading cases in all_combinations mode"""

        self.wrapper.load_cases(mode="all_combinations")
        self.assertEqual(len(self.wrapper.cases_context), 9)  # 3x3 combinations
        self.assertEqual(len(self.wrapper.cases_dirs), 9)

    def test_build_cases(self):
        """Test building cases"""

        self.wrapper.load_cases(mode="one_by_one")
        self.wrapper.build_cases()

        # Check that case directories were created
        for case_dir in self.wrapper.cases_dirs:
            self.assertTrue(op.exists(case_dir))
            self.assertTrue(op.exists(op.join(case_dir, "test.template")))

    def test_postprocess_cases(self):
        """Test postprocessing cases"""

        self.wrapper.load_cases(mode="one_by_one")
        self.wrapper.build_cases()

        # Postprocess should return an empty dataset for DummyModelWrapper
        result = self.wrapper.postprocess_cases()
        self.assertIsInstance(result, xr.Dataset)
        self.assertEqual(len(result.data_vars), 0)


if __name__ == "__main__":
    unittest.main()

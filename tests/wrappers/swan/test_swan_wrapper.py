import os
import os.path as op
import shutil
import subprocess
import tempfile
import unittest
from unittest import skipIf

import numpy as np

from bluemath_tk.wrappers.swan.swan_wrapper import SwanModelWrapper


def is_docker_available():
    """Check if Docker is available and running"""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class TestSwanModelWrapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = op.join(self.test_dir, "templates")
        self.output_dir = op.join(self.test_dir, "output")
        os.makedirs(self.templates_dir)
        os.makedirs(self.output_dir)

        # Test parameters
        self.metamodel_parameters = {
            "Hs": [1.0],  # Wave heights
            "Tp": [5.0],  # Peak periods
            "Dir": [0.0],  # Wave directions
            "Spr": [10.0],  # Directional spread
        }
        self.fixed_parameters = {}

        # Create a simple depth file
        self.depth_array = np.ones(100) * 10.0  # 10m depth everywhere
        np.savetxt(op.join(self.templates_dir, "depth.dat"), self.depth_array)

        # Create wrapper instance
        self.wrapper = SwanModelWrapper(
            templates_dir=self.templates_dir,
            metamodel_parameters=self.metamodel_parameters,
            fixed_parameters=self.fixed_parameters,
            output_dir=self.output_dir,
            depth_array=self.depth_array,
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_default_parameters(self):
        """Test that default parameters are correctly set"""

        self.assertIsNotNone(self.wrapper.default_parameters)

    def test_available_launchers(self):
        """Test that launchers are correctly defined"""

        self.assertIsNotNone(self.wrapper.available_launchers)
        self.assertIn("serial", self.wrapper.available_launchers)
        self.assertIn("docker_serial", self.wrapper.available_launchers)

    def test_list_available_output_variables(self):
        """Test listing available output variables"""

        output_vars = self.wrapper.list_available_output_variables()
        self.assertIsInstance(output_vars, list)
        self.assertGreater(len(output_vars), 0)
        self.assertIn("Hsig", output_vars)
        self.assertIn("Tm02", output_vars)
        self.assertIn("Dir", output_vars)

    def test_build_cases(self):
        """Test building multiple cases"""

        self.wrapper.build_cases(mode="one_by_one")

        # Check that case directories were created
        for case_dir in self.wrapper.cases_dirs:
            self.assertTrue(op.exists(case_dir))
            self.assertTrue(op.exists(op.join(case_dir, "depth.dat")))


class TestSwanModelWrapperIntegration(unittest.TestCase):
    """Integration tests for SwanModelWrapper that require Docker"""

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = op.join(self.test_dir, "templates")
        self.output_dir = op.join(self.test_dir, "output")
        os.makedirs(self.templates_dir)
        os.makedirs(self.output_dir)

        # Test parameters
        self.metamodel_parameters = {
            "Hs": [1.0],  # Wave heights
            "Tp": [5.0],  # Peak periods
            "Dir": [0.0],  # Wave directions
            "Spr": [10.0],  # Directional spread
        }
        self.fixed_parameters = {}

        # Create a simple depth file
        self.depth_array = np.ones(100) * 10.0  # 10m depth everywhere
        np.savetxt(op.join(self.templates_dir, "depth.dat"), self.depth_array)

        # Create wrapper instance
        self.wrapper = SwanModelWrapper(
            templates_dir=self.templates_dir,
            metamodel_parameters=self.metamodel_parameters,
            fixed_parameters=self.fixed_parameters,
            output_dir=self.output_dir,
            depth_array=self.depth_array,
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    @skipIf(not is_docker_available(), "Docker is not available")
    def test_run_cases_docker(self):
        """Test running multiple cases with Docker"""
        # Build cases
        self.wrapper.build_cases(mode="one_by_one")

        # Run cases with Docker
        self.wrapper.run_cases(launcher="docker_serial")

        # Check that output files were created
        for case_dir in self.wrapper.cases_dirs:
            # Check for wrapper log file
            log_file = op.join(case_dir, "wrapper_out.log")
            self.assertTrue(op.exists(log_file))

            # Read and output last 10 lines of log
            with open(log_file, "r") as f:
                log_lines = f.readlines()
                last_lines = log_lines[-10:] if len(log_lines) >= 10 else log_lines
                print("\nLast 10 lines of wrapper_out.log:")
                print("".join(last_lines))


if __name__ == "__main__":
    unittest.main()

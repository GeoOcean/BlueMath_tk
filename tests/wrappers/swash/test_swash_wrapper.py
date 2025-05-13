import os
import os.path as op
import shutil
import subprocess
import tempfile
import unittest
from unittest import skipIf

import numpy as np

from bluemath_tk.wrappers.swash.swash_wrapper import (
    ChySwashModelWrapper,
    HySwashVeggyModelWrapper,
    SwashModelWrapper,
)


def is_docker_available():
    """Check if Docker is available and running"""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class TestSwashModelWrapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = op.join(self.test_dir, "templates")
        self.output_dir = op.join(self.test_dir, "output")
        os.makedirs(self.templates_dir)
        os.makedirs(self.output_dir)

        # Test parameters
        self.metamodel_parameters = {
            "Hs": [1.0, 2.0, 3.0],  # Wave heights
            "Hs_L0": [0.01, 0.02, 0.03],  # Wave heights at deep water
        }
        self.fixed_parameters = {}

        # Create a simple depth file
        self.depth_array = np.ones(100) * 10.0  # 10m depth everywhere
        np.savetxt(op.join(self.templates_dir, "depth.bot"), self.depth_array)

        # Create wrapper instance
        self.wrapper = SwashModelWrapper(
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

    def test_list_available_postprocess_vars(self):
        """Test listing available postprocess variables"""

        output_vars = self.wrapper.list_available_postprocess_vars()
        self.assertIsInstance(output_vars, list)
        self.assertGreater(len(output_vars), 0)
        self.assertIn("Ru2", output_vars)
        self.assertIn("Runlev", output_vars)
        self.assertIn("Msetup", output_vars)
        self.assertIn("Hrms", output_vars)
        self.assertIn("Hfreqs", output_vars)

    def test_build_cases(self):
        """Test building multiple cases"""

        self.wrapper.build_cases(mode="one_by_one")

        # Check that case directories were created
        for case_dir in self.wrapper.cases_dirs:
            self.assertTrue(op.exists(case_dir))
            self.assertTrue(op.exists(op.join(case_dir, "depth.bot")))


class TestHySwashVeggyModelWrapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = op.join(self.test_dir, "templates")
        self.output_dir = op.join(self.test_dir, "output")
        os.makedirs(self.templates_dir)
        os.makedirs(self.output_dir)

        # Test parameters
        self.metamodel_parameters = {
            "Hs": [1.0, 2.0, 3.0],  # Wave heights
            "Hs_L0": [0.01, 0.02, 0.03],  # Wave heights at deep water
            "WL": [0.0, 0.5, 1.0],  # Water levels
            "vegetation_height": [0.5, 1.0, 1.5],  # Vegetation heights
            "plants_density": [100, 500, 1000],  # Plant densities
        }
        self.fixed_parameters = {
            "dxinp": 1.0,  # Input spacing
            "Plants_ini": 50,  # Vegetation start cell
            "Plants_fin": 80,  # Vegetation end cell
            "comptime": 180,  # Computational time
            "warmup": 0,  # Warmup time
            "n_nodes_per_wavelength": 60,  # Number of nodes per wavelength
        }

        # Create a simple depth file
        self.depth_array = np.ones(100) * 10.0  # 10m depth everywhere
        np.savetxt(op.join(self.templates_dir, "depth.bot"), self.depth_array)

        # Create wrapper instance
        self.wrapper = HySwashVeggyModelWrapper(
            templates_dir=self.templates_dir,
            metamodel_parameters=self.metamodel_parameters,
            fixed_parameters=self.fixed_parameters,
            output_dir=self.output_dir,
            depth_array=self.depth_array,
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_plants_file_creation(self):
        """Test that plants file is created with correct values"""

        self.wrapper.build_cases(mode="one_by_one")
        plants = np.loadtxt(op.join(self.wrapper.cases_dirs[0], "plants.txt"))
        self.assertEqual(len(plants), len(self.depth_array))
        self.assertTrue(
            np.all(
                plants[
                    self.fixed_parameters["Plants_ini"] : self.fixed_parameters[
                        "Plants_fin"
                    ]
                ]
                == self.metamodel_parameters["plants_density"][0]
            )
        )


class TestChySwashModelWrapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.templates_dir = op.join(self.test_dir, "templates")
        self.output_dir = op.join(self.test_dir, "output")
        os.makedirs(self.templates_dir)
        os.makedirs(self.output_dir)

        # Test parameters
        self.metamodel_parameters = {
            "Hs": [1.0, 2.0, 3.0],  # Wave heights
            "Hs_L0": [0.01, 0.02, 0.03],  # Wave heights at deep water
            "WL": [0.0, 0.5, 1.0],  # Water levels
            "Cf": [0.001, 0.002, 0.003],  # Friction coefficients
        }
        self.fixed_parameters = {
            "dxinp": 1.0,  # Input spacing
            "default_Cf": 0.002,  # Friction manning coefficient (m^-1/3 s)
            "Cf_ini": 5,  # Friction start cell
            "Cf_fin": 8,  # Friction end cell
            "comptime": 180,  # Computational time
            "warmup": 0,  # Warmup time
            "n_nodes_per_wavelength": 60,  # Number of nodes per wavelength
        }

        # Create a simple depth file
        self.depth_array = np.ones(100) * 10.0  # 10m depth everywhere
        np.savetxt(op.join(self.templates_dir, "depth.bot"), self.depth_array)

        # Create wrapper instance
        self.wrapper = ChySwashModelWrapper(
            templates_dir=self.templates_dir,
            metamodel_parameters=self.metamodel_parameters,
            fixed_parameters=self.fixed_parameters,
            output_dir=self.output_dir,
            depth_array=self.depth_array,
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_friction_file_creation(self):
        """Test that friction file is created with correct values"""

        self.wrapper.build_cases(mode="one_by_one")
        friction = np.loadtxt(op.join(self.wrapper.cases_dirs[0], "friction.txt"))
        self.assertEqual(len(friction), len(self.depth_array))
        self.assertTrue(
            np.all(
                friction[
                    self.fixed_parameters["Cf_ini"] : self.fixed_parameters["Cf_fin"]
                ]
                == self.metamodel_parameters["Cf"][0]
            )
        )


class TestSwashModelWrapperIntegration(unittest.TestCase):
    """Integration tests for SwashModelWrapper that require Docker"""

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
            "Hs_L0": [0.01],  # Wave heights at deep water
        }
        self.fixed_parameters = {}

        # Create a simple depth file
        self.depth_array = np.ones(100) * 10.0  # 10m depth everywhere
        np.savetxt(op.join(self.templates_dir, "depth.bot"), self.depth_array)

        # Create wrapper instance
        self.wrapper = SwashModelWrapper(
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

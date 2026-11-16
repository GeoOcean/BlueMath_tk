import unittest

import numpy as np

from bluemath_tk.wrappers.snapwave.snapwave_wrapper import _array_spec, slurm_array_blocks


class TestSlurmBlocks(unittest.TestCase):
    def test_blocks_below_limit(self):
        blocks = slurm_array_blocks(np.arange(1, 24001), 9999)
        self.assertEqual([o for o, _ in blocks], [0, 9999, 19998])
        for offset, lines in blocks:
            ids = lines - offset
            self.assertGreaterEqual(ids.min(), 1)
            self.assertLessEqual(ids.max(), 9999)
        self.assertEqual(sum(len(b) for _, b in blocks), 24000)

    def test_subset_and_spec(self):
        blocks = slurm_array_blocks(np.array([3, 4, 5, 9, 10001, 10002]), 9999)
        self.assertEqual(_array_spec(blocks[0][1] - blocks[0][0]), "3-5,9")
        self.assertEqual(_array_spec(blocks[1][1] - blocks[1][0]), "2-3")


if __name__ == "__main__":
    unittest.main()

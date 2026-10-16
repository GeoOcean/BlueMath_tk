import os
import unittest

from bluemath_tk.core.resources import GIB, Machine, child_threads, limit_threads, machine, run_processes


def _threads_seen(x: int) -> tuple[int, str | None]:
    return x * 2, os.environ.get("OMP_NUM_THREADS")


class TestResources(unittest.TestCase):
    def test_machine_and_usable(self):
        host = machine()
        self.assertGreaterEqual(host.cores, 1)
        self.assertGreater(host.memory_total, 0)
        cores, memory = Machine(48, 400 * GIB, 512 * GIB).usable(0.75)
        self.assertEqual((cores, memory), (36, 300 * GIB))
        self.assertEqual(Machine(1, GIB, GIB).usable()[0], 1)

    def test_child_threads_restores(self):
        before = os.environ.get("OMP_NUM_THREADS")
        with child_threads(3):
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "3")
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), before)
        with limit_threads(1):
            pass

    def test_run_processes_keeps_order(self):
        jobs = [(i,) for i in range(5)]
        here = run_processes(_threads_seen, jobs, workers=1)
        self.assertEqual([r[0] for r in here], [0, 2, 4, 6, 8])
        pooled = run_processes(_threads_seen, jobs, workers=2, threads=1)
        self.assertEqual([r[0] for r in pooled], [0, 2, 4, 6, 8])
        self.assertTrue(all(r[1] == "1" for r in pooled))


if __name__ == "__main__":
    unittest.main()

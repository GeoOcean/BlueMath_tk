import unittest

from bluemath_tk.core.resources import GIB, Machine
from bluemath_tk.waves.hywaves import resources

BIG = Machine(48, 480 * GIB, 512 * GIB)
LAPTOP = Machine(6, 12 * GIB, 16 * GIB)


class TestPlans(unittest.TestCase):
    def test_big_machine_takes_a_year_with_every_site(self):
        res = resources.plan_reconstruction(38, 24, 10, host=BIG)
        self.assertEqual((res.device, res.threads, res.workers, res.months, res.sites),
                         ("cpu", 36, 36, 12, None))
        self.assertEqual(res.memory, 360 * GIB)

    def test_fewer_months_then_fewer_sites(self):
        res = resources.plan_reconstruction(5000, 12, 40, save_spectra=True, host=LAPTOP)
        self.assertEqual(res.months, 1)
        self.assertIsNotNone(res.sites)
        self.assertEqual(resources.plan_reconstruction(38, 24, 40, host=LAPTOP).months, 12)
        need = resources.call_memory(res.months, res.sites, 40, workers=res.workers, save_spectra=True)
        self.assertLessEqual(need["total"], res.memory)

    def test_workers_capped_by_memory(self):
        self.assertEqual(resources.plan_reconstruction(3, 1, 10, host=BIG).workers, 36)
        tiny = Machine(48, int(4 * GIB / 0.75), GIB * 64)
        res = resources.plan_reconstruction(1, 1, 10, host=tiny)
        self.assertEqual(res.workers, 3)  # 3 x (0.75 + 0.5) GiB in 4 GiB
        with self.assertRaises(MemoryError):
            resources.plan_reconstruction(1, 1, 10, host=Machine(4, GIB // 4, GIB))

    def test_sum_budget_and_hours(self):
        self.assertEqual(resources.sum_budget(100 * GIB, 0, 4), 4 * resources.SUM_BYTES_PER_WORKER)
        self.assertEqual(resources.sum_budget(GIB, 2 * GIB, 4), 0)
        self.assertLess(resources.hours(100, 1, 10, 36), resources.hours(100, 1, 10, 1))

    def test_gpu_months_and_fits(self):
        gpu = Machine(16, 100 * GIB, 128 * GIB, "A100", 40 * GIB)
        self.assertEqual(resources.plan_reconstruction(38, 24, 10, host=gpu).months, 4)
        self.assertEqual(resources.plan_fit(60, GIB, host=gpu), resources.FitResources("cuda", 1, 12))
        self.assertEqual(resources.plan_fit(60, 2 * GIB, host=BIG).workers, 36)
        self.assertEqual(resources.plan_fit(60, 100 * GIB, host=BIG).workers, 3)
        self.assertEqual(resources.plan_fit(5, GIB, host=BIG).workers, 5)


if __name__ == "__main__":
    unittest.main()

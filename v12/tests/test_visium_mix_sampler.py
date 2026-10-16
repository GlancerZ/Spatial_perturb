import unittest
import numpy as np
from spatial_pretrain.data.visium_mix_sampler import FixedDomainBatchSampler


class FixedDomainWindows(unittest.TestCase):
    def build(self,world=1,rank=0,visium=16):
        return FixedDomainBatchSampler(np.arange(100),
            {str(i):np.arange(100+i*20,120+i*20) for i in range(5)},
            updates=12,visium_centers=visium,world=world,rank=rank,seed=42)

    def test_same_global_windows_across_topologies(self):
        reference=self.build()
        for world in (1,2,4):
            ranks=[self.build(world,r) for r in range(world)]
            for step in range(12):
                windows=[list(iter(s))[step] for s in ranks]
                np.testing.assert_array_equal(np.concatenate(windows),reference.global_window(step))
                self.assertEqual(sum(i>=100 for i in np.concatenate(windows)),16)

    def test_resume_cursor_reproduces_remaining_windows(self):
        full=list(iter(self.build(2,1)))
        resumed=self.build(2,1)
        resumed.seek(0,5)
        self.assertEqual(list(iter(resumed)),full[5:])
        self.assertEqual(len(resumed),7)

    def test_balanced_physical_sample_exposure(self):
        windows=np.concatenate([self.build().global_window(i) for i in range(5)])
        self.assertEqual([int(((windows>=100+i*20)&(windows<120+i*20)).sum()) for i in range(5)], [16]*5)

    def test_control_uses_only_xenium(self):
        sampler=self.build(visium=0)
        self.assertTrue(all(np.asarray(window).max()<100 for window in sampler))

    def test_reject_overlapping_sources(self):
        with self.assertRaises(ValueError):
            FixedDomainBatchSampler(np.arange(10),{"a":np.arange(5,15)},updates=2)


if __name__=="__main__":
    unittest.main()

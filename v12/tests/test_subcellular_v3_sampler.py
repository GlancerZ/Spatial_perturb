import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('trainer_v3', ROOT / 'scripts/train_subcellular_v3.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
GlobalBatchSampler = module.GlobalBatchSampler



class GlobalBatchTests(unittest.TestCase):
    def test_world_size_migration_preserves_global_windows_and_seek(self):
        serial = GlobalBatchSampler(137, 64, 16, seed=9)
        whole = list(serial)
        ranks = [list(GlobalBatchSampler(137, 64, 16, 4, rank, 9)) for rank in range(4)]
        combined = [batch for update in zip(*ranks) for batch in update]
        assert whole == combined
        serial.seek(0, 1)
        assert list(serial) == whole[4:]
        for rank in range(4):
            resumed = GlobalBatchSampler(137, 64, 16, 4, rank, 9)
            resumed.seek(0, 1)
            assert list(resumed) == ranks[rank][1:]
        values = [index for batch in whole for index in batch]
        assert len(set(values[:137])) == 137
        assert len(values) == 192
        assert values[137:] == values[:55]


    def test_small_dataset_padding_and_epoch_change(self):
        sampler = GlobalBatchSampler(3, 64, 16)
        first = list(sampler)
        assert len(first) == 4
        assert set(sum(first, [])) == {0, 1, 2}
        sampler.seek(1, 0)
        assert len(list(sampler)) == 4
        sampler.seek(1, 1)
        assert list(sampler) == []

if __name__ == "__main__":
    unittest.main()

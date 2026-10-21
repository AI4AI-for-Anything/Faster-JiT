"""CPU checks for cache admission, corruption detection and PIL/tensor flip equivalence.

Run from the repository root: python -m unittest discover -s tests
The full 8-rank input/epoch/resume checks belong to the external frozen bench.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

from util import crop_cache
from util.crop import center_crop_arr


class CropCacheTest(unittest.TestCase):
    def test_verified_cache_rng_and_corruption(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='jit_crop_test_') as tmp:
            root = Path(tmp)
            train = root / 'data' / 'train' / 'class_a'
            train.mkdir(parents=True)
            rng = np.random.default_rng(42)
            for i, (h, w) in enumerate([(15, 21), (33, 40), (128, 155), (77, 51)]):
                Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(train / f'{i}.jpg')
            cache = str(root / 'cache')
            # This tiny fixture need not reserve the production cache's 200 GB.
            with patch.object(crop_cache.os, 'statvfs') as stat:
                stat.return_value.f_bavail = 1_000_000_000_000
                stat.return_value.f_frsize = 1
                crop_cache.build(str(root / 'data'), cache, 16, 2)
            open_cache = lambda: crop_cache.open_cached_train_set(
                str(train.parent), cache, 16, transforms.RandomHorizontalFlip())
            self.assertIsNone(open_cache(), 'unverified caches must not be used')
            self.assertEqual(crop_cache.verify(str(root / 'data'), cache, 16, 2, False, 2), 0)
            self.assertIsNone(open_cache(), 'sample checks cannot authorize a full cache')
            self.assertEqual(crop_cache.verify(str(root / 'data'), cache, 16, 0, True, 2), 0)
            cached = open_cache()
            self.assertIsNotNone(cached)
            ref = datasets.ImageFolder(str(train.parent), transform=transforms.Compose([
                transforms.Lambda(lambda im: center_crop_arr(im, 16)),
                transforms.RandomHorizontalFlip(), transforms.PILToTensor()]))
            for seed in range(8):
                torch.manual_seed(seed)
                a = torch.utils.data.default_collate([ref[i] for i in range(len(ref))])
                state = torch.get_rng_state()
                torch.manual_seed(seed)
                b = torch.utils.data.default_collate([cached[i] for i in range(len(cached))])
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                for x, y in zip(a, b):
                    self.assertTrue(torch.equal(x, y))
                    self.assertEqual((x.dtype, x.shape, x.stride()), (y.dtype, y.shape, y.stride()))
            data_f, _, meta_f = crop_cache.cache_files(cache, 16)
            with open(data_f, 'r+b') as f:
                byte = f.read(1)
                f.seek(0)
                f.write(bytes([byte[0] ^ 1]))
            self.assertIsNone(open_cache(), 'a modified cache invalidates its receipt')
            # Matching cache checksums alone cannot authorize incorrect pixels.
            meta = json.loads(Path(meta_f).read_text())
            meta['chunk_sha256'] = [hashlib.sha256(Path(data_f).read_bytes()).hexdigest()]
            Path(meta_f).write_text(json.dumps(meta))
            self.assertEqual(crop_cache.verify(str(root / 'data'), cache, 16, 0, True, 2), 1)
            self.assertFalse(Path(meta_f + '.verified.json').exists())


if __name__ == '__main__':
    unittest.main()

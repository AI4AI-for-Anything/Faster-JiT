"""Precomputed center-crop cache for the JiT training input pipeline (switch JIT_CROP_CACHE=1).

The upstream training transform is

    Compose([Lambda(center_crop_arr(img, img_size)), RandomHorizontalFlip(), PILToTensor()])

applied to ImageFolder samples. Its first stage has no randomness: ImageFolder's loader (PIL decode + RGB), the BOX
halvings, the BICUBIC resize and the center crop of util/crop.py. This module runs exactly that code once per training
image and stores the uint8 HWC crops in one flat file. Training then reads a crop from the file and applies the
unchanged RandomHorizontalFlip, so every DataLoader worker draws the same random numbers in the same order as the
upstream pipeline and the batches handed to the model are bitwise identical. Sampler, DataLoader, worker count and
seeding are not touched.

Build (CPU only; about 252 GB for ImageNet-1k train at 256x256):
    python -m util.crop_cache build  --data_path $IMAGENET_PATH --cache_path $CROP_CACHE_PATH [--img_size 256]
Verify (--full recomputes EVERY crop from the JPEGs and checks every cache checksum):
    python -m util.crop_cache verify --data_path $IMAGENET_PATH --cache_path $CROP_CACHE_PATH [--samples 2000] [--full]
Use in training:
    export JIT_CROP_CACHE=1 CROP_CACHE_PATH=/path/to/cache     (scripts/configs/faster.env sets JIT_CROP_CACHE=1)
Training requires a successful --full verification receipt. Keep the source dataset and cache immutable after
verification; rebuild and reverify if either changes. If the cache is missing or does not match the dataset,
the code prints a line containing "fallback" and trains with
the upstream ImageFolder pipeline.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

FORMAT = 'faster-jit-center-crop-cache-v1'
CHUNK = 2048  # images per build task and per checksum


def crop_source_sha256():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crop.py'), 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def cache_files(cache_path, img_size):
    base = os.path.join(cache_path, f'train_{img_size}')
    return base + '.u8', base + '.labels.npy', base + '.meta.json'


def list_train_set(train_root):
    """The upstream sample list: torchvision ImageFolder (sorted classes, sorted files) and its loader."""
    import torchvision.datasets as datasets
    return datasets.ImageFolder(train_root)


def samples_sha256(samples, train_root):
    h = hashlib.sha256()
    for path, label in samples:
        h.update(f'{os.path.relpath(path, train_root)}\t{label}\n'.encode())
    return h.hexdigest()


def reference_crop(loader, path, img_size):
    """Deterministic part of the upstream transform, run with the upstream code: uint8 array [H, W, 3]."""
    from util.crop import center_crop_arr
    return np.asarray(center_crop_arr(loader(path), img_size))


def pillow_version():
    import PIL
    return PIL.__version__


def file_signature(path):
    st = os.stat(path)
    return dict(size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def verification_identity(data_f, labels_f, meta_f):
    return dict(data=file_signature(data_f), labels=file_signature(labels_f),
                meta_sha256=hashlib.sha256(Path(meta_f).read_bytes()).hexdigest())


# ---------------------------------------------------------------------------------------------------------- build
_B = {}  # per-process build state (fork-inherited sample list, lazily opened output map)


def _build_chunk(c):
    samples, loader, out, n, s = _B['samples'], _B['loader'], _B['out'], _B['n'], _B['img_size']
    mm = _B.get('mm')
    if mm is None:
        mm = _B['mm'] = np.memmap(out, dtype=np.uint8, mode='r+', shape=(n, s, s, 3))
    lo, hi = c * CHUNK, min(n, (c + 1) * CHUNK)
    for i in range(lo, hi):
        a = reference_crop(loader, samples[i][0], s)
        if a.shape != (s, s, 3) or a.dtype != np.uint8:
            raise RuntimeError(f'unexpected crop {a.shape} {a.dtype} for {samples[i][0]}')
        mm[i] = a
    return c, hashlib.sha256(mm[lo:hi]).hexdigest()


def build(data_path, cache_path, img_size, workers):
    import torch
    torch.set_num_threads(1)
    train_root = os.path.join(data_path, 'train')
    t0 = time.time()
    ds = list_train_set(train_root)
    n = len(ds.samples)
    data_f, labels_f, meta_f = cache_files(cache_path, img_size)
    os.makedirs(cache_path, exist_ok=True)
    need = n * img_size * img_size * 3
    free = os.statvfs(cache_path).f_bavail * os.statvfs(cache_path).f_frsize
    print(f'{n} images, {need / 2**30:.1f} GiB to write, {free / 2**30:.1f} GiB free in {cache_path}', flush=True)
    if any(os.path.exists(p) for p in (data_f, labels_f, meta_f, data_f + '.partial')):
        raise FileExistsError('Cache files already exist; use a new empty cache directory')
    if free - need < 200_000_000_000:
        raise RuntimeError('Cache build must leave at least 200 GB free on local disk')
    tmp = data_f + '.partial'
    with open(tmp, 'wb') as f:
        f.truncate(need)
    _B.clear()
    _B.update(samples=ds.samples, loader=ds.loader, out=tmp, n=n, img_size=img_size)
    nchunks = (n + CHUNK - 1) // CHUNK
    shas = [None] * nchunks
    done = 0
    with mp.get_context('fork').Pool(workers) as pool:
        for c, h in pool.imap_unordered(_build_chunk, range(nchunks)):
            shas[c] = h
            done += 1
            if done % 50 == 0 or done == nchunks:
                el = time.time() - t0
                print(f'{done}/{nchunks} chunks, {el:.0f} s, {min(n, done * CHUNK) / el:.0f} img/s', flush=True)
    fd = os.open(tmp, os.O_RDWR)
    os.fsync(fd)  # all crops on disk before the cache can look complete
    os.close(fd)
    os.replace(tmp, data_f)
    np.save(labels_f, np.asarray(ds.targets, dtype=np.int64))
    meta = dict(format=FORMAT, img_size=img_size, num_samples=n, shape=[n, img_size, img_size, 3], dtype='uint8',
                layout='HWC, ImageFolder sample order', chunk=CHUNK, chunk_sha256=shas,
                samples_sha256=samples_sha256(ds.samples, train_root), classes=len(ds.classes),
                crop_source_sha256=crop_source_sha256(), pillow=pillow_version(), numpy=np.__version__,
                built=time.strftime('%Y-%m-%d %H:%M:%S'), build_seconds=round(time.time() - t0, 1), workers=workers)
    with open(meta_f + '.tmp', 'w') as f:
        json.dump(meta, f)
    os.replace(meta_f + '.tmp', meta_f)
    print(f'cache complete: {data_f} ({need / 2**30:.1f} GiB) in {meta["build_seconds"]} s', flush=True)


# ---------------------------------------------------------------------------------------------------------- checks
def check_meta(meta, ds, train_root, img_size, data_f):
    """Return None if the cache matches this dataset and code, else the reason."""
    n = len(ds.samples)
    if meta.get('format') != FORMAT:
        return f"unknown cache format {meta.get('format')}"
    if meta['img_size'] != img_size or meta['num_samples'] != n:
        return f"cache has {meta['num_samples']} images at {meta['img_size']}px, dataset has {n} at {img_size}px"
    if not os.path.exists(data_f) or os.path.getsize(data_f) != n * img_size * img_size * 3:
        return 'cache data file missing or of the wrong size'
    if meta['samples_sha256'] != samples_sha256(ds.samples, train_root):
        return 'cache was built from a different image list'
    if meta['crop_source_sha256'] != crop_source_sha256():
        return 'util/crop.py changed since the cache was built'
    if meta['pillow'] != pillow_version():
        return f"cache built with Pillow {meta['pillow']}, running {pillow_version()} (resize output may differ)"
    if meta['numpy'] != np.__version__:
        return 'NumPy version changed since the cache was built'
    if meta.get('chunk') != CHUNK or len(meta.get('chunk_sha256', [])) != (n + CHUNK - 1) // CHUNK:
        return 'invalid cache chunk manifest'
    return None


def _verify_chunk(c):
    mm, n = _B['mm'], _B['n']
    lo, hi = c * CHUNK, min(n, (c + 1) * CHUNK)
    bad = 0
    for i in range(lo, hi):
        expected = reference_crop(_B['loader'], _B['samples'][i][0], _B['img_size'])
        bad += not np.array_equal(mm[i], expected)
    return c, hashlib.sha256(mm[lo:hi]).hexdigest(), bad


def verify(data_path, cache_path, img_size, samples, full, workers, seed=0):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    train_root = os.path.join(data_path, 'train')
    ds = list_train_set(train_root)
    data_f, labels_f, meta_f = cache_files(cache_path, img_size)
    receipt_f = meta_f + '.verified.json'
    if full and os.path.exists(receipt_f):
        os.remove(receipt_f)
    meta = json.loads(Path(meta_f).read_text())
    why = check_meta(meta, ds, train_root, img_size, data_f)
    if why:
        print('FAIL:', why)
        return 1
    labels = np.load(labels_f)
    if not np.array_equal(labels, np.asarray(ds.targets, dtype=np.int64)):
        print('FAIL: labels differ')
        return 1
    n = len(ds.samples)
    identity = verification_identity(data_f, labels_f, meta_f)
    mm = np.memmap(data_f, dtype=np.uint8, mode='r', shape=(n, img_size, img_size, 3))
    idx = np.random.default_rng(seed).choice(n, size=0 if full else min(samples, n), replace=False)
    bad = [int(i) for i in idx if not np.array_equal(mm[i], reference_crop(ds.loader, ds.samples[i][0], img_size))]
    if not full:
        print(f'{len(idx)} random images recomputed from the JPEGs with util/crop.py: {len(idx) - len(bad)} identical')
    if bad:
        print('FAIL: differing indices', bad[:20])
        return 1
    if full:
        _B.clear()
        _B.update(mm=mm, n=n, samples=ds.samples, loader=ds.loader, img_size=img_size)
        nchunks = (n + CHUNK - 1) // CHUNK
        got, bad_count = {}, 0
        with mp.get_context('fork').Pool(workers) as pool:
            for c, h, bad in pool.imap_unordered(_verify_chunk, range(nchunks)):
                got[c] = h
                bad_count += bad
                if len(got) % 50 == 0 or len(got) == nchunks:
                    print(f'{len(got)}/{nchunks} chunks recomputed, {time.time() - t0:.0f} s, '
                          f'{bad_count} differing images', flush=True)
        wrong = [c for c in range(nchunks) if got[c] != meta['chunk_sha256'][c]]
        print(f'{nchunks} chunk checksums: {nchunks - len(wrong)} match')
        if wrong or bad_count:
            print(f'FAIL: {len(wrong)} differing checksums, {bad_count} differing reference crops')
            return 1
        if identity != verification_identity(data_f, labels_f, meta_f):
            print('FAIL: cache changed during verification')
            return 1
        receipt = dict(identity=identity, reference_crops_compared=n, seconds=round(time.time() - t0, 1),
                       verified=time.strftime('%Y-%m-%d %H:%M:%S'))
        with open(receipt_f + '.tmp', 'w') as f:
            json.dump(receipt, f, indent=2)
        os.replace(receipt_f + '.tmp', receipt_f)
        print(f'{n} / {n} reference crops bitwise identical; verification receipt written')
    print('PASS')
    return 0


# ---------------------------------------------------------------------------------------------------------- training
class CenterCropCacheDataset:
    """Drop-in for ImageFolder(train, transform=Compose([center crop, RandomHorizontalFlip(), PILToTensor()])):
    item i = (transform(uint8 tensor [3, S, S] of the cached crop), label), same order and labels as ImageFolder."""

    def __init__(self, data_f, labels, img_size, transform, meta):
        self.data_f, self.labels, self.img_size, self.transform, self.meta = data_f, labels, img_size, transform, meta
        self.n = len(labels)
        self._mm, self._pid = None, None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch
        if self._pid != os.getpid():  # open the map lazily in each DataLoader worker
            s = self.img_size
            self._mm, self._pid = np.memmap(self.data_f, dtype=np.uint8, mode='r', shape=(self.n, s, s, 3)), os.getpid()
        # same tensor as PILToTensor(Image) of the crop: CHW view of an HWC uint8 array
        img = torch.from_numpy(np.array(self._mm[i])).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])

    def __repr__(self):
        return (f'CenterCropCacheDataset\n    Number of datapoints: {self.n}\n    Cache: {self.data_f}\n'
                f'    Transform (after the cached center crop): {self.transform}')


def open_cached_train_set(train_root, cache_path, img_size, transform, spot_checks=4):
    """The cached training set, or None after printing a 'fallback' line (caller uses the upstream pipeline)."""
    def fallback(why):
        print(f'JIT_CROP_CACHE fallback: {why}; using the upstream ImageFolder decode/resize pipeline', flush=True)
        return None
    if not cache_path:
        return fallback('CROP_CACHE_PATH is not set')
    data_f, labels_f, meta_f = cache_files(cache_path, img_size)
    if not os.path.exists(meta_f):
        return fallback(f'no complete cache at {meta_f}')
    ds = list_train_set(train_root)
    try:
        meta = json.loads(Path(meta_f).read_text())
        receipt = json.loads(Path(meta_f + '.verified.json').read_text())
        if (receipt.get('identity') != verification_identity(data_f, labels_f, meta_f)
                or receipt.get('reference_crops_compared') != len(ds.samples)):
            return fallback('cache changed or full verification is incomplete; run verify --full')
    except (OSError, ValueError):
        return fallback('missing or unreadable cache verification; run verify --full')
    why = check_meta(meta, ds, train_root, img_size, data_f)
    if why:
        return fallback(why)
    labels = np.load(labels_f)
    if not np.array_equal(labels, np.asarray(ds.targets, dtype=np.int64)):
        return fallback('cached labels differ from ImageFolder')
    mm = np.memmap(data_f, dtype=np.uint8, mode='r', shape=(len(labels), img_size, img_size, 3))
    for i in np.linspace(0, len(labels) - 1, spot_checks).astype(np.int64):  # fixed indices, no RNG use
        if not np.array_equal(mm[i], reference_crop(ds.loader, ds.samples[i][0], img_size)):
            return fallback(f'cached image {int(i)} differs from its JPEG')
    del mm
    print(f'JIT_CROP_CACHE enabled: {len(labels)} cached {img_size}px center crops from {data_f} '
          f'(built {meta["built"]}), RandomHorizontalFlip applied per sample as upstream', flush=True)
    return CenterCropCacheDataset(data_f, labels, img_size, transform, meta)


def main():
    p = argparse.ArgumentParser(description='Build or verify the JiT center-crop cache.')
    p.add_argument('mode', choices=['build', 'verify'])
    p.add_argument('--data_path', required=True, help='ImageNet root that contains train/')
    p.add_argument('--cache_path', required=True)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    p.add_argument('--samples', type=int, default=2000, help='verify: random images recomputed from the JPEGs')
    p.add_argument('--full', action='store_true', help='verify: recompute every reference crop and check all hashes; required before training')
    a = p.parse_args()
    if a.img_size < 1 or a.workers < 1 or a.samples < 0:
        p.error('img_size and workers must be positive; samples must be nonnegative')
    if a.mode == 'build':
        build(a.data_path, a.cache_path, a.img_size, a.workers)
        return 0
    return verify(a.data_path, a.cache_path, a.img_size, a.samples, a.full, a.workers)


if __name__ == '__main__':
    sys.exit(main())

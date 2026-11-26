import random

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from imgcnn import raster
from imgcnn.hdfs import HDFS


def dataset_meta(cfg, fs):
    path = f"{cfg.hdfs.root}/meta/dataset.json"
    if fs.status(path) is None:
        raise SystemExit(f"{path} is missing in HDFS, run the build command first")
    return fs.read_json(path)


def shard_files(cfg, fs, split):
    folder = f"{cfg.hdfs.root}/shards/{split}"
    if fs.status(folder) is None:
        return []
    return [path for path, _ in fs.files(folder) if "/part-" in path]


class ShardStream(IterableDataset):
    def __init__(self, cfg, files, size, seed, rank=0, world=1):
        self.url = cfg.hdfs.url
        self.user = cfg.hdfs.user
        self.files = files[rank::world]
        self.size = size
        self.seed = seed
        self.batch = cfg.train.batch_size
        self.buffer = cfg.train.buffer
        self.channels = cfg.train.channels
        self.recognized_only = cfg.train.recognized_only

    def samples(self, files, rng):
        fs = HDFS(self.url, self.user)
        keep = self.channels * self.size * self.size
        while True:
            order = list(files)
            rng.shuffle(order)
            seen = 0
            for path in order:
                for line in fs.lines(path):
                    label, _, _, recognized, image = line.split(b"\t")
                    if self.recognized_only and recognized != b"1":
                        continue
                    seen += 1
                    yield raster.decode(image)[:keep], int(label)
            if seen == 0:
                return

    def __iter__(self):
        info = get_worker_info()
        files = self.files if info is None else self.files[info.id :: info.num_workers]
        if not files:
            return
        rng = random.Random(f"{self.seed}:{0 if info is None else info.id}")
        pool, images, labels = [], [], []
        for item in self.samples(files, rng):
            if len(pool) < self.buffer:
                pool.append(item)
                continue
            slot = rng.randrange(self.buffer)
            pool[slot], item = item, pool[slot]
            images.append(item[0])
            labels.append(item[1])
            if len(labels) == self.batch:
                pixels = torch.frombuffer(bytearray().join(images), dtype=torch.uint8)
                yield pixels.view(-1, self.channels, self.size, self.size), torch.tensor(labels)
                images, labels = [], []


def train_batches(cfg, files, size, seed, rank=0, world=1):
    stream = ShardStream(cfg, files, size, seed, rank, world)
    workers = min(cfg.train.workers, len(stream.files))
    if workers == 0:
        return iter(stream)
    return iter(DataLoader(stream, batch_size=None, num_workers=workers, prefetch_factor=4))


def load_split(cfg, fs, split, size, channels, limit=0):
    keep = channels * size * size
    chunks, labels, countries, recognized, keys = [], [], [], [], []
    for path in shard_files(cfg, fs, split):
        images = []
        for line in fs.lines(path):
            label, key_id, country, flag, image = line.split(b"\t")
            images.append(raster.decode(image)[:keep])
            labels.append(int(label))
            countries.append(country.decode())
            recognized.append(flag == b"1")
            keys.append(key_id.decode())
            if limit and len(labels) >= limit:
                break
        if images:
            pixels = torch.frombuffer(bytearray().join(images), dtype=torch.uint8)
            chunks.append(pixels.view(-1, channels, size, size))
        if limit and len(labels) >= limit:
            break
    if not chunks:
        raise SystemExit(f"no {split} shards in HDFS, run the build command first")
    return {
        "images": torch.cat(chunks),
        "labels": torch.tensor(labels),
        "countries": countries,
        "recognized": torch.tensor(recognized),
        "keys": keys,
    }

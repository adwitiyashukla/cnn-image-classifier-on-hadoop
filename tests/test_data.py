import random
from collections import Counter

import pytest
import torch

from conftest import FakeFS, load_small, make_drawing
from imgcnn import data, raster


def make_shards(split="train", files=4, per_file=50):
    rng = random.Random(0)
    shards = {}
    for index in range(files):
        lines = []
        for row in range(per_file):
            image = raster.encode(raster.render(make_drawing(rng)["drawing"], 16, 1.0))
            label = (index * per_file + row) % 3
            flag = int(row % 5 != 0)
            lines.append(f"{label}\tkey{index}_{row}\tUS\t{flag}\t{image}".encode())
        shards[f"/imgcnn/shards/{split}/part-{index:05d}"] = lines
        shards[f"/imgcnn/shards/{split}/_SUCCESS"] = []
    return shards


@pytest.fixture
def fake(monkeypatch):
    shards = make_shards()
    fs = FakeFS(shards)
    monkeypatch.setattr(data, "HDFS", lambda url, user: fs)
    return fs


def take(stream, count):
    iterator = iter(stream)
    return [next(iterator) for _ in range(count)]


def test_stream_yields_full_batches(fake, cfg):
    files = data.shard_files(cfg, fake, "train")
    assert len(files) == 4
    images, labels = take(data.ShardStream(cfg, files, 16, "seed"), 1)[0]
    assert images.shape == (16, 3, 16, 16) and images.dtype == torch.uint8
    assert labels.shape == (16,) and labels.dtype == torch.int64


def test_stream_covers_every_sample_before_repeating(fake, cfg):
    files = data.shard_files(cfg, fake, "train")
    seen = Counter()
    for images, _ in take(data.ShardStream(cfg, files, 16, "seed"), 25):
        for image in images:
            seen[image.numpy().tobytes()] += 1
    assert len(seen) == 200
    assert max(seen.values()) <= 3


def test_stream_order_depends_on_the_seed(fake, cfg):
    files = data.shard_files(cfg, fake, "train")
    first = take(data.ShardStream(cfg, files, 16, "a"), 2)
    again = take(data.ShardStream(cfg, files, 16, "a"), 2)
    other = take(data.ShardStream(cfg, files, 16, "b"), 2)
    assert all(torch.equal(x[0], y[0]) for x, y in zip(first, again))
    assert not all(torch.equal(x[0], y[0]) for x, y in zip(first, other))


def test_ranks_read_different_shards(fake, cfg):
    files = data.shard_files(cfg, fake, "train")
    one = data.ShardStream(cfg, files, 16, "s", rank=0, world=2)
    two = data.ShardStream(cfg, files, 16, "s", rank=1, world=2)
    assert not set(one.files) & set(two.files)
    assert sorted(one.files + two.files) == files


def test_ink_only_and_recognized_only(fake):
    cfg, _ = load_small("train.channels=1", "train.recognized_only=true")
    files = data.shard_files(cfg, fake, "train")
    stream = data.ShardStream(cfg, files, 16, "s")
    images, _ = take(stream, 1)[0]
    assert images.shape == (16, 1, 16, 16)
    distinct = {image for _, (image, _) in zip(range(500), stream.samples(files, random.Random(0)))}
    assert len(distinct) == 160


def test_empty_file_list_ends_the_stream(fake, cfg):
    assert list(data.ShardStream(cfg, [], 16, "s")) == []


def test_load_split_reads_everything(cfg):
    fs = FakeFS(make_shards("test", files=2, per_file=30))
    split = data.load_split(cfg, fs, "test", 16, 3)
    assert split["images"].shape == (60, 3, 16, 16)
    assert split["labels"].tolist() == [i % 3 for i in range(60)]
    assert split["keys"][0] == "key0_0" and split["countries"][0] == "US"
    assert split["recognized"].sum().item() == 48
    assert data.load_split(cfg, fs, "test", 16, 1, limit=45)["images"].shape == (45, 1, 16, 16)


def test_missing_split_is_reported(cfg):
    empty = FakeFS({})
    assert data.shard_files(cfg, empty, "val") == []
    with pytest.raises(SystemExit, match="build command"):
        data.load_split(cfg, empty, "val", 16, 3)
    with pytest.raises(SystemExit, match="build command"):
        data.dataset_meta(cfg, empty)

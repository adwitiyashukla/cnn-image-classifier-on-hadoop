import json
import random
from pathlib import Path

import pytest
import torch

from imgcnn import config, models, raster
from imgcnn.export import Exported

ROOT = Path(__file__).resolve().parents[1]
SMALL = [
    "resnet.stem=8",
    "resnet.widths=[8, 16]",
    "resnet.depths=[1, 1]",
    "mobilenet.stem=8",
    "mobilenet.expand=[1, 2]",
    "mobilenet.channels=[8, 16]",
    "mobilenet.repeats=[1, 1]",
    "mobilenet.strides=[1, 2]",
    "mobilenet.head=32",
    "convnext.widths=[8, 16]",
    "convnext.depths=[1, 1]",
    "train.batch_size=16",
    "train.threads=1",
    "train.workers=0",
    "train.buffer=32",
    "hdfs.user=tester",
]


def load_small(*extra):
    return config.load(ROOT / "config.toml", [*SMALL, *extra])


@pytest.fixture
def cfg():
    return load_small()[0]


def make_drawing(rng, word="cat", key_id=None, recognized=True, country="US"):
    strokes = []
    for _ in range(rng.randint(1, 5)):
        points = rng.randint(2, 12)
        strokes.append([[rng.randint(0, 255) for _ in range(points)], [rng.randint(0, 255) for _ in range(points)]])
    return {
        "word": word,
        "countrycode": country,
        "timestamp": "2017-03-02 23:25:10.07453 UTC",
        "recognized": recognized,
        "key_id": key_id or str(rng.randrange(10**15, 10**16)),
        "drawing": strokes,
    }


@pytest.fixture
def drawings():
    rng = random.Random(7)
    words = ["cat", "dog", "line", "tree"]
    items = [
        make_drawing(rng, words[i % 4], recognized=i % 5 != 0, country=("US", "GB", "IN")[i % 3]) for i in range(400)
    ]
    return items


def as_lines(items):
    return [json.dumps(item) + "\n" for item in items]


class FakeFS:
    def __init__(self, shards, meta=None):
        self.shards = shards
        self.meta = meta

    def files(self, path):
        prefix = path.rstrip("/") + "/"
        return sorted((name, len(lines)) for name, lines in self.shards.items() if name.startswith(prefix))

    def lines(self, path):
        yield from self.shards[path]

    def status(self, path):
        if path.endswith("dataset.json"):
            return {"length": 1} if self.meta else None
        prefix = path.rstrip("/") + "/"
        return {"type": "DIRECTORY"} if any(name.startswith(prefix) for name in self.shards) else None

    def read_json(self, path):
        return self.meta


SHAPES = {
    "bar": lambda rng: [[[0, 200 + rng.randint(0, 55)], [100, 100 + rng.randint(0, 6)]]],
    "pole": lambda rng: [[[100, 100 + rng.randint(0, 6)], [0, 200 + rng.randint(0, 55)]]],
    "box": lambda rng: [[[0, 250, 250, 0, 0], [0, 0, 200 + rng.randint(0, 50), 200 + rng.randint(0, 50), 0]]],
}


def shape_line(rng, label, name, key, size):
    image = raster.encode(raster.render(SHAPES[name](rng), size, 1.0))
    flag = int(rng.random() < 0.9)
    return f"{label}\t{key}\t{rng.choice(['US', 'GB', 'IN'])}\t{flag}\t{image}".encode()


def make_dataset(size=16, files=2, per_split=None):
    rng = random.Random(3)
    names = sorted(SHAPES)
    per_split = per_split or {"train": 240, "val": 60, "test": 90}
    shards, counts, recognized = {}, {}, {}
    for split, total in per_split.items():
        rows = [shape_line(rng, i % 3, names[i % 3], f"{split}{i}", size) for i in range(total)]
        rng.shuffle(rows)
        for index in range(files):
            shards[f"/imgcnn/shards/{split}/part-{index:05d}"] = rows[index::files]
        shards[f"/imgcnn/shards/{split}/_SUCCESS"] = []
        counts[split] = total
        recognized[split] = sum(row.split(b"\t")[3] == b"1" for row in rows)
    meta = {"classes": names, "size": size, "width": 1.0, "seed": 1, "counts": counts, "recognized": recognized}
    return FakeFS(shards, meta)


def export_tiny(folder, names, size=32):
    cfg, _ = load_small()
    model = models.build(cfg, len(names), "resnet").eval()
    torch.onnx.export(
        Exported(model).eval(),
        (torch.rand(2, 3, size, size),),
        str(folder / "model.onnx"),
        input_names=["image"],
        output_names=["logits", "maps"],
        dynamic_shapes={"image": {0: torch.export.Dim("batch")}},
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    (folder / "model.json").write_text(json.dumps({"classes": names, "size": size, "channels": 3, "width": 2.0}))
    return model

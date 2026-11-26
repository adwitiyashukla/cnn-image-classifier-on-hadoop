import base64
import hashlib
import json

import pytest

from conftest import load_small
from imgcnn import ingest

BODY = b"".join(b'{"word":"cat","n":%d}\n' % i for i in range(400))


class Download:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for start in range(0, len(self.body), 1000):
            yield self.body[start : start + 1000]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Store:
    files = {}

    def __init__(self, url=None, user=None):
        pass

    def status(self, path):
        return {"length": len(self.files[path])} if path in self.files else None

    def create(self, path, data):
        self.files[path] = data if isinstance(data, bytes) else b"".join(data)

    def delete(self, path):
        del self.files[path]

    def rename(self, source, target):
        self.files[target] = self.files.pop(source)

    def read_json(self, path):
        return json.loads(self.files[path])

    def write_json(self, path, value):
        self.files[path] = json.dumps(value).encode()

    def mkdirs(self, path):
        pass


@pytest.fixture
def store(monkeypatch):
    Store.files = {}
    monkeypatch.setattr(ingest, "HDFS", Store)
    monkeypatch.setattr(ingest.requests, "get", lambda url, stream=True, timeout=60: Download(BODY))
    return Store


def item(body=BODY, name="The Eiffel Tower"):
    return {"name": name, "size": len(body), "md5": base64.b64encode(hashlib.md5(body).digest()).decode()}


def test_whole_file_is_copied_and_checked(store):
    cfg, _ = load_small("source.max_mb=0")
    record = ingest.copy_one(cfg, item())
    assert store.files["/imgcnn/raw/The_Eiffel_Tower.ndjson"] == BODY
    assert record["drawings"] == 400 and record["bytes"] == len(BODY) and record["md5_checked"]
    assert not any(path.startswith("/imgcnn/incoming/") for path in store.files)


def test_second_run_skips_finished_files(store):
    cfg, _ = load_small("source.max_mb=0")
    ingest.copy_one(cfg, item())
    assert ingest.copy_one(cfg, item())["skipped"]


def test_checksum_mismatch_leaves_nothing_behind(store):
    cfg, _ = load_small("source.max_mb=0")
    with pytest.raises(OSError, match="md5 mismatch"):
        ingest.copy_one(cfg, {**item(), "md5": "AAAA"})
    assert store.files == {}


def test_size_cap_cuts_on_a_line_border(store, monkeypatch):
    cfg, _ = load_small("source.max_mb=1")
    big = BODY * 200
    monkeypatch.setattr(ingest.requests, "get", lambda url, stream=True, timeout=60: Download(big))
    record = ingest.copy_one(cfg, item(big))
    saved = store.files["/imgcnn/raw/The_Eiffel_Tower.ndjson"]
    assert saved.endswith(b"}\n") and big.startswith(saved)
    assert (1 << 20) - 40 < len(saved) <= 1 << 20
    assert record["drawings"] == saved.count(b"\n") and not record["md5_checked"]


def test_missing_final_newline_is_added(store, monkeypatch):
    cfg, _ = load_small("source.max_mb=0")
    cut = BODY[:-1]
    monkeypatch.setattr(ingest.requests, "get", lambda url, stream=True, timeout=60: Download(cut))
    record = ingest.copy_one(cfg, item(cut))
    assert store.files["/imgcnn/raw/The_Eiffel_Tower.ndjson"] == BODY
    assert record["drawings"] == 400


def test_manifest_can_come_from_a_file(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {"name": "full/simplified/zebra.ndjson", "size": "5", "md5Hash": "x"},
                    {"name": "full/simplified/The Eiffel Tower.ndjson", "size": "7", "md5Hash": "y"},
                    {"name": "full/simplified/cat.ndjson", "size": "9", "md5Hash": "z"},
                    {"name": "full/simplified/ant.ndjson", "size": "3", "md5Hash": "w"},
                ]
            }
        )
    )
    cfg, _ = load_small(f"source.manifest={path.as_posix()}", "source.classes=3")
    assert ingest.load_manifest(cfg) == [
        {"name": "The Eiffel Tower", "size": 7, "md5": "y"},
        {"name": "ant", "size": 3, "md5": "w"},
        {"name": "cat", "size": 9, "md5": "z"},
    ]

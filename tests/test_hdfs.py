import json

import pytest

from imgcnn import hdfs, jobs
from imgcnn.config import namespace


class Reply:
    def __init__(self, status=200, body=None, chunks=()):
        self.status_code = status
        self.body = body or {}
        self.chunks = chunks
        self.text = json.dumps(self.body)

    def json(self):
        return self.body

    def iter_content(self, size):
        yield from self.chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Wire:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def request(self, method, url, params=None):
        self.calls.append((method, url, params))
        return self.replies.pop(0)

    def get(self, url, params=None, stream=False):
        self.calls.append(("GET", url, params))
        return self.replies.pop(0)

    def put(self, url, data=None, headers=None):
        self.calls.append(("PUT", url, b"".join(data) if not isinstance(data, bytes) else data))
        return self.replies.pop(0)


def client(*replies):
    fs = hdfs.HDFS("http://namenode:9870/", "adi")
    fs.http = Wire(replies)
    return fs


def test_lines_are_rebuilt_across_chunk_borders():
    fs = client(Reply(chunks=[b"alpha\nbe", b"ta\n", b"gam", b"ma\ndelta"]))
    assert list(fs.lines("/x")) == [b"alpha", b"beta", b"gamma", b"delta"]


def test_trailing_newline_gives_no_empty_line():
    fs = client(Reply(chunks=[b"one\ntwo\n"]))
    assert list(fs.lines("/x")) == [b"one", b"two"]


def test_paths_are_quoted_and_the_user_is_sent():
    fs = client(Reply(body={"FileStatus": {"length": 3}}))
    assert fs.status("/raw/The Eiffel Tower.ndjson") == {"length": 3}
    method, url, params = fs.http.calls[0]
    assert url == "http://namenode:9870/webhdfs/v1/raw/The%20Eiffel%20Tower.ndjson"
    assert params == {"op": "GETFILESTATUS", "user.name": "adi"}


def test_missing_path_has_no_status():
    assert client(Reply(status=404)).status("/nothing") is None


def test_errors_carry_the_operation():
    with pytest.raises(OSError, match="MKDIRS /a: 403"):
        client(Reply(status=403)).mkdirs("/a")


def test_create_follows_the_datanode_location():
    fs = client(Reply(body={"Location": "http://datanode:9864/webhdfs/v1/f?op=CREATE"}), Reply(status=201))
    fs.create("/f", iter([b"ab", b"cd"]))
    assert fs.http.calls[1] == ("PUT", "http://datanode:9864/webhdfs/v1/f?op=CREATE", b"abcd")


def test_files_walks_folders():
    listing = {
        "FileStatuses": {
            "FileStatus": [
                {"pathSuffix": "train", "type": "DIRECTORY", "length": 0},
                {"pathSuffix": "_SUCCESS", "type": "FILE", "length": 0},
            ]
        }
    }
    inner = {"FileStatuses": {"FileStatus": [{"pathSuffix": "part-00000", "type": "FILE", "length": 7}]}}
    fs = client(Reply(body=listing), Reply(body=inner))
    assert fs.files("/shards") == [("/shards/_SUCCESS", 0), ("/shards/train/part-00000", 7)]


LOG = """2026-11-11 19:18:40,300 INFO mapreduce.Job: Running job: job_1794441950443_0005
2026-11-11 19:18:40,301 INFO mapreduce.Job: Counters: 3
\tFile System Counters
\t\tHDFS: Number of bytes read=1085203346
\tMap-Reduce Framework
\t\tMap input records=2432747
\t\tShuffled Maps =20
\timgcnn
\t\tdrawings=2432747
\t\tselected_train=344705
2026-11-11 19:18:40,352 INFO streaming.StreamJob: Output directory: /imgcnn/shards
"""


def test_counters_are_read_from_the_job_log():
    counters = jobs.parse_counters(LOG)
    assert counters["imgcnn"] == {"drawings": 2432747, "selected_train": 344705}
    assert counters["Map-Reduce Framework"]["Map input records"] == 2432747
    assert counters["Map-Reduce Framework"]["Shuffled Maps"] == 20
    assert counters["File System Counters"]["HDFS: Number of bytes read"] == 1085203346
    assert jobs.parse_counters("no counters here") == {}


def test_quotas_become_hash_thresholds():
    cfg = namespace({"dataset": {"train_per_class": 600, "val_per_class": 100, "test_per_class": 200}})
    profile = {"classes": {"cat": {"drawings": 2010}, "yoga": {"drawings": 500}, "gone": {"drawings": 4}}}
    audit = {"per_class": {"cat": 10, "gone": 4}}
    limits, short = jobs.thresholds(cfg, profile, audit)
    assert limits["cat"] == pytest.approx([0.1, 0.15, 0.45])
    assert limits["yoga"] == pytest.approx([200 / 900, 300 / 900, 1.0])
    assert short == ["yoga"]
    assert "gone" not in limits

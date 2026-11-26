import random
from collections import Counter

import numpy as np

from conftest import as_lines, export_tiny, make_drawing
from imgcnn import mr, raster


def run(job, role, lines, params=None, drop=frozenset()):
    out = []
    mr.counters.clear()
    mr.JOBS[(job, role)](lines, out.append, params or {}, drop)
    return out, dict(mr.counters)


def shuffle(*outputs):
    return sorted(line for output in outputs for line in output)


def params_for(words, limits, size=16):
    return {"seed": 5, "size": size, "width": 1.0, "classes": sorted(words), "thresholds": {w: limits for w in words}}


def test_profile_counts_match_a_direct_count(drawings):
    lines = as_lines(drawings)
    first, _ = run("profile", "map", lines[:150])
    second, counters = run("profile", "map", lines[150:])
    reduced, _ = run("profile", "reduce", shuffle(first, second))
    got = {tuple(row[:3]): tuple(map(int, row[3:])) for row in (line.rstrip("\n").split("\t") for line in reduced)}
    expected = Counter()
    strokes = Counter()
    points = Counter()
    for item in drawings:
        key = ("class", item["word"], str(int(item["recognized"])))
        expected[key] += 1
        strokes[key] += len(item["drawing"])
        points[key] += sum(len(xs) for xs, _ in item["drawing"])
        expected[("country", item["countrycode"], str(int(item["recognized"])))] += 1
    assert len(reduced) == len(expected)
    for key, count in expected.items():
        assert got[key][0] == count
        if key[0] == "class":
            assert got[key][1:] == (strokes[key], points[key])
    assert counters["drawings"] == 250


def test_malformed_lines_are_counted_and_skipped(drawings):
    lines = as_lines(drawings[:5]) + ["{broken\n", "\n", '{"word": "cat"}\n', "[1, 2]\n"]
    out, counters = run("audit", "map", lines)
    assert len(out) == 5
    assert counters == {"drawings": 5, "malformed": 3}


def audit(items):
    mapped, _ = run("audit", "map", as_lines(items))
    reduced, counters = run("audit", "reduce", shuffle(mapped))
    rows = [line.rstrip("\n").split("\t") for line in reduced]
    drops = {row[1]: row[2] for row in rows if row[0] == "drop"}
    return drops, [row for row in rows if row[0] == "group"], counters


def test_audit_keeps_one_copy_of_a_repeated_drawing():
    rng = random.Random(1)
    shape = make_drawing(rng)["drawing"]
    items = [make_drawing(rng) for _ in range(20)]
    for key_id in ("30", "10", "20"):
        items.append({**make_drawing(rng, key_id=key_id), "drawing": shape})
    drops, groups, counters = audit(items)
    assert drops == {"30": "duplicate", "20": "duplicate"}
    assert groups == []
    assert counters["duplicate_groups"] == 1 and counters["duplicate_dropped"] == 2


def test_audit_majority_class_keeps_the_shape():
    rng = random.Random(2)
    shape = make_drawing(rng)["drawing"]
    items = [{**make_drawing(rng, "line", key_id=str(i)), "drawing": shape} for i in range(1, 5)]
    items.append({**make_drawing(rng, "bathtub", key_id="99"), "drawing": shape})
    drops, groups, counters = audit(items)
    assert drops == {"2": "duplicate", "3": "duplicate", "4": "duplicate", "99": "conflict"}
    assert groups[0][2:] == ["5", "line:4,bathtub:1"]
    assert counters["conflict_groups"] == 1


def test_audit_drops_every_copy_without_a_majority():
    rng = random.Random(3)
    shape = make_drawing(rng)["drawing"]
    items = [{**make_drawing(rng, word, key_id=str(i)), "drawing": shape} for i, word in enumerate(["sun", "moon"])]
    drops, _, _ = audit(items)
    assert drops == {"0": "conflict", "1": "conflict"}


def test_split_is_stable_and_follows_the_thresholds():
    params = params_for(["cat"], [0.1, 0.3, 0.8])
    picks = [mr.split_of(str(i), "cat", params) for i in range(20000)]
    assert picks == [mr.split_of(str(i), "cat", params) for i in range(20000)]
    share = Counter(picks)
    assert abs(share["test"] / 20000 - 0.1) < 0.01
    assert abs(share["val"] / 20000 - 0.2) < 0.01
    assert abs(share["train"] / 20000 - 0.5) < 0.015
    assert abs(share[None] / 20000 - 0.2) < 0.01
    assert mr.split_of("1", "unknown word", params) is None


def test_growing_the_quota_keeps_earlier_picks():
    small = params_for(["cat"], [0.05, 0.1, 0.3])
    large = params_for(["cat"], [0.05, 0.1, 0.6])
    for i in range(3000):
        before = mr.split_of(str(i), "cat", small)
        if before is not None:
            assert mr.split_of(str(i), "cat", large) == before


def test_build_writes_decodable_records(drawings):
    words = {item["word"] for item in drawings}
    params = params_for(words, [0.2, 0.4, 1.0])
    blocked = {drawings[0]["key_id"], drawings[1]["key_id"]}
    mapped, counters = run("build", "map", as_lines(drawings), params, blocked)
    assert counters["dropped"] == 2
    assert len(mapped) == len(drawings) - 2
    assert counters["selected_train"] + counters["selected_val"] + counters["selected_test"] == len(mapped)
    kept = [item for item in drawings if item["key_id"] not in blocked]
    recognized = sum(counters.get(f"recognized_{split}", 0) for split in ("train", "val", "test"))
    assert recognized == sum(item["recognized"] for item in kept)
    reduced, _ = run("build", "reduce", shuffle(mapped))
    by_key = {item["key_id"]: item for item in drawings}
    seen = set()
    for line in reduced:
        split, label, key_id, country, recognized, image = line.rstrip("\n").split("\t")
        item = by_key[key_id]
        assert split == mr.split_of(key_id, item["word"], params)
        assert params["classes"][int(label)] == item["word"]
        assert country == item["countrycode"] and recognized == str(int(item["recognized"]))
        assert raster.decode(image) == raster.render(item["drawing"], 16, 1.0)
        seen.add(key_id)
    assert seen == set(by_key) - blocked


def test_build_order_key_shuffles_classes(drawings):
    params = params_for({item["word"] for item in drawings}, [0.0, 0.0, 1.0])
    mapped, _ = run("build", "map", as_lines(drawings), params)
    labels = [int(line.split("\t")[2]) for line in sorted(mapped)]
    runs = sum(a != b for a, b in zip(labels, labels[1:]))
    assert runs > len(labels) / 2
    assert np.unique(labels).size == 4


def test_score_reduce_adds_up_and_ranks_suspects():
    params = {"classes": ["cat", "dog", "tree"], "threshold": 0.8, "top_k": 2}
    rows = [
        ("cat", "k1", "US", 1, 0.90, 0, 0.90, 1, 2),
        ("cat", "k2", "US", 1, 0.05, 1, 0.93, 2, 0),
        ("cat", "k3", "GB", 0, 0.02, 1, 0.97, 2, 0),
        ("cat", "k4", "GB", 0, 0.30, 2, 0.50, 0, 1),
        ("cat", "k5", "IN", 1, 0.01, 2, 0.85, 1, 0),
        ("dog", "k6", "US", 1, 0.70, 1, 0.70, 0, 2),
    ]
    lines = ["\t".join(map(str, row)) + "\n" for row in rows]
    reduced, _ = run("score", "reduce", lines, params)
    out = [line.rstrip("\n").split("\t") for line in reduced]
    cat = next(row for row in out if row[:2] == ["class", "cat"])
    assert cat[2:6] == ["2", "0", "2", "1"]
    assert cat[7:11] == ["3", "1", "3", "2"]
    suspects = [row for row in out if row[0] == "suspect"]
    assert [row[2] for row in suspects] == ["k3", "k2"]
    assert suspects[0][3] == "dog"
    dog = next(row for row in out if row[:2] == ["class", "dog"])
    assert dog[7:11] == ["1", "1", "1", "0"]


def test_unit_hash_is_uniform_and_salted():
    values = [mr.unit(str(i), "a") for i in range(5000)]
    assert 0 <= min(values) and max(values) < 1
    assert abs(sum(values) / 5000 - 0.5) < 0.02
    assert mr.unit("1", "a") != mr.unit("1", "b")


def test_score_map_runs_the_model_on_unused_drawings(drawings, tmp_path, monkeypatch):
    words = sorted({item["word"] for item in drawings})
    export_tiny(tmp_path, words, size=16)
    monkeypatch.chdir(tmp_path)
    params = {**params_for(words, [0.1, 0.2, 0.5]), "channels": 3, "fraction": 1.0, "threshold": 0.5, "top_k": 3}
    params.update(batch=32, threads=1)
    blocked = {drawings[0]["key_id"]}
    mapped, counters = run("score", "map", as_lines(drawings), params, blocked)
    kept = [item for item in drawings if item["key_id"] not in blocked]
    unused = [item for item in kept if mr.split_of(item["key_id"], item["word"], params) is None]
    assert len(mapped) == len(unused) == counters["scored"]
    assert counters["skipped_dataset"] == len(kept) - len(unused)
    for line in mapped:
        word, key_id, _, _, p_true, guess, p_guess, second, third = line.rstrip("\n").split("\t")
        assert word in words and 0 <= float(p_true) <= float(p_guess) + 1e-4 <= 1.001
        assert len({guess, second, third}) == 3
    reduced, _ = run("score", "reduce", shuffle(mapped), params)
    rows = [line.split("\t") for line in reduced if line.startswith("class\t")]
    assert sum(int(row[2]) + int(row[7]) for row in rows) == len(unused)


def test_score_map_samples_a_fraction(drawings, tmp_path, monkeypatch):
    words = sorted({item["word"] for item in drawings})
    export_tiny(tmp_path, words, size=16)
    monkeypatch.chdir(tmp_path)
    params = {**params_for(words, [0.0, 0.0, 0.0]), "channels": 3, "fraction": 0.25, "threshold": 0.5, "top_k": 3}
    params.update(batch=7, threads=1)
    mapped, counters = run("score", "map", as_lines(drawings), params)
    assert 60 < len(mapped) < 140
    assert counters["scored"] == len(mapped)

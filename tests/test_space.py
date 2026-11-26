import json
import random
import shutil
import subprocess

import numpy as np
import pytest

from conftest import ROOT, export_tiny, make_drawing
from imgcnn import export, raster


def run_js(jobs):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = ROOT / "tests" / "doodle.mjs"
    done = subprocess.run([node, str(script)], input=json.dumps(jobs), capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_javascript_rasterizer_matches_python():
    rng = random.Random(3)
    drawings = [make_drawing(rng)["drawing"] for _ in range(200)]
    drawings += [[[[5], [9]]], [[[0, 255], [0, 0]], [[], []]], [[[1.5, 80.25, 300.0], [2.0, 2.0, 900.5]]]]
    drawings.append([[[i, i + 20], [0, 255]] for i in range(0, 240, 20)])
    jobs = [["render", strokes, size, width] for strokes in drawings for size, width in ((32, 2.0), (48, 1.5))]
    for job, pixels in zip(jobs, run_js(jobs)):
        wanted = raster.render(*job[1:])
        plane = job[2] * job[2]
        assert bytes(pixels[: 2 * plane]) == wanted[: 2 * plane]
        assert max(abs(a - b) for a, b in zip(pixels[2 * plane :], wanted[2 * plane :])) <= 1


def test_simplify_drops_points_on_a_straight_line():
    line = [[x, 2 * x] for x in range(0, 100, 5)]
    corner = [[0, 0], [50, 1], [100, 0], [100, 50], [100, 100]]
    straight, bent, twin = run_js([["simplify", line], ["simplify", corner], ["simplify", [[3, 3], [3, 3]]]])
    assert straight == [line[0], line[-1]]
    assert bent == [[0, 0], [100, 0], [100, 100]]
    assert twin == [[3, 3], [3, 3]]


def test_normalize_scales_to_the_quick_draw_range():
    base = [[[10.0, 60.0, 110.0], [500.0, 500.0, 550.0]]]
    moved = [[[x * 3 + 400 for x in base[0][0]], [y * 3 + 90 for y in base[0][1]]]]
    first, second = run_js([["normalize", base], ["normalize", moved]])
    xs, ys = first[0]
    assert min(xs) == 0 and max(xs) == 255
    assert min(ys) == 0 and max(ys) in (127, 128)
    assert all(abs(a - b) <= 1 for axis in zip(first[0], second[0]) for a, b in zip(*axis))


@pytest.mark.parametrize("strokes", [[], [[[], []]], [[[1.0, 2.0], [1.0]]], [[[1.0, 1e300], [1.0, 2.0]]]])
def test_bad_drawings_are_rejected(strokes):
    assert run_js([["normalize", strokes]]) == [None]


def test_read_ranks_classes_and_scales_the_heat_map():
    meta = {"classes": ["cat", "dog", "tree"], "size": 2, "channels": 3}
    maps = {"dims": [1, 3, 2, 2], "data": [9, 9, 9, 9, -1, 0, 2, 4, 5, 5, 5, 5]}
    output = {"logits": {"data": [0.5, 2.5, 1.5]}, "maps": maps}
    result = run_js([["read", output, meta, [0, 255, 7, 8]]])[0]
    assert [guess["label"] for guess in result["guesses"]] == ["dog", "tree", "cat"]
    assert sum(guess["p"] for guess in result["guesses"]) == pytest.approx(1)
    assert result["heat"] == [[0, 0], [0.5, 1]]
    assert result["image"] == [[0, 255], [7, 8]]


def test_tensor_scales_pixels_like_the_training_code():
    values = run_js([["tensor", list(range(256)), {"size": 16, "channels": 1}]])[0]
    assert values == [float(np.float32(i) / np.float32(255)) for i in range(256)]


def test_space_folder_has_the_page_and_the_model(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    export_tiny(run_dir, ["cat", "dog", "tree"])
    out = tmp_path / "bundle"
    export.bundle(run_dir, out)
    assert (out / "README.md").read_text(encoding="utf-8").startswith("---\n")
    assert "./doodle.js" in (out / "index.html").read_text() and (out / "doodle.js").exists()
    assert (out / "model" / "model.onnx").stat().st_size > 0
    assert json.loads((out / "model" / "model.json").read_text())["classes"] == ["cat", "dog", "tree"]


def test_space_folder_needs_an_exported_model(tmp_path):
    with pytest.raises(SystemExit, match="export command"):
        export.bundle(tmp_path, tmp_path / "bundle")

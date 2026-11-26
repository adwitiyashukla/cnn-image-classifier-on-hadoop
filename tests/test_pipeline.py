import json

import pytest
import torch

from conftest import load_small, make_dataset
from imgcnn import data, evaluate, export, report, train

FAST = (
    "train.epochs=3",
    "train.lr=0.01",
    "train.log_every=5",
    "train.checkpoint_every=5",
    "eval.batch_size=32",
    "eval.bootstrap=20",
    "eval.robustness_samples=30",
    "eval.min_country=1",
    "eval.gradcam=4",
    "export.latency_runs=20",
    "export.calibration_samples=32",
)


@pytest.fixture
def cluster(monkeypatch, tmp_path):
    fs = make_dataset()
    monkeypatch.setattr(data, "HDFS", lambda url, user: fs)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    runs = tmp_path / "runs"

    def settings(*extra):
        return load_small(*FAST, f"runs.dir={runs.as_posix()}", *extra)

    return fs, runs, settings


def test_train_evaluate_export_and_report(cluster):
    fs, runs, settings = cluster
    cfg, raw = settings()
    run_dir = runs / "resnet"
    train.train(cfg, raw, fs, run_dir)
    history = json.loads((run_dir / "metrics.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert history[-1]["val_top1"] > 0.9
    assert json.loads((run_dir / "config.json").read_text()) == raw

    result = evaluate.evaluate(cfg, fs, run_dir)
    assert result["samples"] == 90
    assert result["top1"] > 0.9
    assert len(result["deletion"]) == 4
    json.dumps(result, allow_nan=False)
    for name in ("eval.json", "gradcam.png", "reliability.png"):
        assert (run_dir / name).stat().st_size > 0

    exported = export.export(cfg, fs, run_dir)
    assert exported["onnx_max_abs_difference"] < 1e-3
    assert exported["exported_macs"] > exported["macs"]
    assert exported["top1"]["onnx"] == pytest.approx(result["top1"])
    assert exported["top1"]["int8"] > 0.8
    assert json.loads((run_dir / "model.json").read_text())["width"] == 1.0

    report.report(cfg, runs, "resnet")
    summary = json.loads((runs / "report" / "summary.json").read_text())
    assert summary["runs"][0]["run"] == "resnet"
    assert summary["runs"][0]["top1"] == pytest.approx(result["top1"])
    for name in ("curves.png", "tradeoff.png", "robustness.png", "countries.png"):
        assert (runs / "report" / name).stat().st_size > 0


def test_refused_resume_keeps_the_saved_config(cluster):
    fs, runs, settings = cluster
    cfg, raw = settings("train.epochs=1")
    train.train(cfg, raw, fs, runs / "resnet")
    other, other_raw = settings("train.epochs=1", "train.lr=0.02")
    with pytest.raises(SystemExit, match="different config"):
        train.train(other, other_raw, fs, runs / "resnet")
    assert json.loads((runs / "resnet" / "config.json").read_text())["train"]["lr"] == 0.01


def test_resume_key_ignores_runtime_settings(cluster):
    _, _, settings = cluster
    _, raw = settings()
    _, other = settings("train.threads=2", "train.workers=1", "train.log_every=7", "train.checkpoint_every=9")
    assert train.resume_key(raw, 1, 100) == train.resume_key(other, 1, 100)
    assert "threads" in raw["train"]
    _, changed = settings("train.lr=0.02")
    assert train.resume_key(raw, 1, 100) != train.resume_key(changed, 1, 100)
    assert train.resume_key(raw, 1, 100) != train.resume_key(raw, 2, 100)


def test_finished_run_is_left_alone(cluster, capsys):
    fs, runs, settings = cluster
    cfg, raw = settings("train.epochs=1")
    train.train(cfg, raw, fs, runs / "resnet")
    stamp = (runs / "resnet" / "model.pt").stat().st_mtime_ns
    capsys.readouterr()
    train.train(cfg, raw, fs, runs / "resnet")
    assert "already trained" in capsys.readouterr().out
    assert (runs / "resnet" / "model.pt").stat().st_mtime_ns == stamp


def test_interrupted_run_resumes_with_its_running_sums(cluster, monkeypatch, capsys):
    fs, runs, settings = cluster
    cfg, raw = settings("train.epochs=1")
    real = data.train_batches

    def faulty(*args, **options):
        for index, item in enumerate(real(*args, **options)):
            if index == 12:
                raise RuntimeError("power cut")
            yield item

    monkeypatch.setattr(data, "train_batches", faulty)
    with pytest.raises(RuntimeError, match="power cut"):
        train.train(cfg, raw, fs, runs / "resnet")
    saved = torch.load(runs / "resnet" / "last.pt", weights_only=False)
    assert saved["state"]["step"] == 10
    assert saved["progress"]["seen"] == 160
    assert not (runs / "resnet" / "model.pt").exists()
    monkeypatch.setattr(data, "train_batches", real)
    cfg, raw = settings("train.epochs=1", "train.checkpoint_every=7", "train.log_every=3")
    train.train(cfg, raw, fs, runs / "resnet")
    assert "resuming from step 10 of 15" in capsys.readouterr().out
    history = json.loads((runs / "resnet" / "metrics.json").read_text())
    assert len(history) == 1 and history[0]["step"] == 15
    assert history[0]["seconds"] >= saved["progress"]["seconds"]


def test_more_processes_than_shards_is_reported(cluster, monkeypatch):
    fs, runs, settings = cluster
    cfg, raw = settings()
    monkeypatch.setattr(train, "distributed", lambda device: (0, 5))
    with pytest.raises(SystemExit, match="cannot feed 5 processes"):
        train.train(cfg, raw, fs, runs / "resnet")


def test_missing_shards_are_reported(cluster):
    fs, runs, settings = cluster
    cfg, raw = settings()
    fs.shards = {name: rows for name, rows in fs.shards.items() if "/train/" not in name}
    with pytest.raises(SystemExit, match="build command"):
        train.train(cfg, raw, fs, runs / "resnet")


def test_recognized_only_counts_recognized_drawings(cluster, capsys):
    fs, runs, settings = cluster
    cfg, raw = settings("train.epochs=1", "train.recognized_only=true")
    train.train(cfg, raw, fs, runs / "resnet")
    assert f"{fs.meta['recognized']['train']:,} train images" in capsys.readouterr().out


def test_corruptions_keep_shape_and_range():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(6, 3, 16, 16, generator=generator)
    for kind, levels in evaluate.LEVELS.items():
        for level in levels:
            out = evaluate.corrupt(x, kind, level if kind != "erase" else level // 2, generator)
            assert out.shape == x.shape, kind
            assert out.min() >= -1e-6 and out.max() <= 1 + 1e-6, kind
            assert not torch.equal(out, x), kind


def test_deletion_test_removes_the_chosen_cells(cfg):
    from imgcnn import models

    model = models.build(cfg, 5, "resnet").eval()
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(8, 3, 32, 32, generator=generator)
    result = evaluate.deletion_test(model, x, torch.arange(8) % 5, 4, generator)
    cells = result["grid"] ** 2
    assert result["cells"] == 4
    assert result["ink_removed_cam"] == pytest.approx(4 / cells, abs=0.05)
    assert result["ink_removed_random"] == pytest.approx(4 / cells, abs=0.05)
    assert 0 <= result["p_cam_deleted"] <= 1

import json
import math
import random

import pytest
import torch
from torch.nn import functional as F

from conftest import load_small
from imgcnn import models, train

CPU = torch.device("cpu")


def batch(seed, n=16, classes=5):
    generator = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8, generator=generator)
    return images, torch.randint(0, classes, (n,), generator=generator)


def test_soft_cross_entropy_matches_torch():
    logits = torch.randn(8, 5)
    labels = torch.randint(0, 5, (8,))
    plain = train.soft_cross_entropy(logits, train.smooth(labels, 5, 0.0))
    assert torch.allclose(plain, F.cross_entropy(logits, labels), atol=1e-6)
    smoothed = train.soft_cross_entropy(logits, train.smooth(labels, 5, 0.1))
    assert torch.allclose(smoothed, F.cross_entropy(logits, labels, label_smoothing=0.1), atol=1e-6)


def test_learning_rate_warms_up_then_follows_a_cosine(cfg):
    rates = [train.learning_rate(cfg.train, step, 100, 10) for step in range(100)]
    assert math.isclose(rates[0], cfg.train.lr / 10)
    assert math.isclose(rates[9], cfg.train.lr)
    assert max(rates) == rates[9] == rates[10]
    assert all(a >= b for a, b in zip(rates[10:], rates[11:]))
    assert math.isclose(rates[55], cfg.train.lr / 2, rel_tol=1e-6)
    assert rates[99] < cfg.train.lr * 0.001


def test_augment_with_zero_ranges_is_the_identity():
    cfg, _ = load_small("train.rotate=0", "train.scale=0", "train.shift=0", "train.flip=false")
    x = torch.rand(4, 3, 32, 32)
    assert torch.allclose(train.augment(x, cfg.train, torch.Generator().manual_seed(0)), x, atol=1e-5)


def test_flip_mirrors_about_half_of_the_batch():
    cfg, _ = load_small("train.rotate=0", "train.scale=0", "train.shift=0", "train.flip=true")
    x = torch.rand(64, 1, 32, 32)
    out = train.augment(x, cfg.train, torch.Generator().manual_seed(0))
    same = torch.isclose(out, x, atol=1e-5).flatten(1).all(1)
    mirrored = torch.isclose(out, x.flip(3), atol=1e-5).flatten(1).all(1)
    assert (same | mirrored).all()
    assert 16 < mirrored.sum() < 48


def test_augment_keeps_the_drawing_inside_the_frame(cfg):
    x = torch.zeros(32, 3, 32, 32)
    x[:, :, 8:24, 8:24] = 1
    out = train.augment(x, cfg.train, torch.Generator().manual_seed(3))
    assert out.shape == x.shape
    assert (out.sum((1, 2, 3)) > 0.7 * x.sum((1, 2, 3))).all()


def test_mixup_blends_images_and_labels():
    cfg, _ = load_small("train.mixup=0.4")
    x = torch.rand(8, 3, 32, 32)
    targets = train.smooth(torch.arange(8) % 5, 5, 0.0)
    mixed, soft = train.mix(x, targets, cfg.train, torch.Generator().manual_seed(0), random.Random(0))
    assert torch.allclose(soft.sum(1), torch.ones(8))
    other = torch.randperm(8, generator=torch.Generator().manual_seed(0))
    lam = random.Random(0).betavariate(0.4, 0.4)
    assert torch.allclose(mixed, lam * x + (1 - lam) * x[other], atol=1e-6)
    assert torch.allclose(soft, lam * targets + (1 - lam) * targets[other], atol=1e-6)


def test_cutmix_label_weight_equals_the_pasted_area():
    cfg, _ = load_small("train.cutmix=1.0")
    x = torch.zeros(2, 1, 32, 32)
    x[1] = 1
    targets = train.smooth(torch.tensor([0, 1]), 2, 0.0)
    for seed in range(20):
        mixed, soft = train.mix(x, targets, cfg.train, torch.Generator().manual_seed(seed), random.Random(seed))
        for row in range(2):
            foreign = (mixed[row] != x[row]).float().mean().item()
            assert math.isclose(soft[row, 1 - row].item(), foreign, abs_tol=1e-6)


def test_mix_is_off_by_default(cfg):
    x = torch.rand(4, 3, 32, 32)
    targets = train.smooth(torch.arange(4), 5, 0.1)
    mixed, soft = train.mix(x, targets, cfg.train, torch.Generator(), random.Random(0))
    assert mixed is x and soft is targets


def test_ema_follows_the_weights(cfg):
    model = models.build(cfg, 5, "resnet")
    ema = train.EMA(model, 0.9)
    start = ema.shadow.fc.weight.clone()
    with torch.no_grad():
        model.fc.weight.add_(1.0)
        model.norm.running_mean.add_(2.0)
    ema.update(model)
    decay = min(0.9, 2 / 11)
    assert torch.allclose(ema.shadow.fc.weight, start + (1 - decay), atol=1e-6)
    assert torch.allclose(ema.shadow.norm.running_mean, torch.full_like(model.norm.running_mean, 2 * (1 - decay)))
    assert not any(p.requires_grad for p in ema.shadow.parameters())


def patterned(n=16, classes=5):
    labels = torch.arange(n) % classes
    images = torch.randint(0, 40, (n, 3, 32, 32), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    for row, label in enumerate(labels.tolist()):
        images[row, :, 4 + 5 * label : 10 + 5 * label, 4 + 5 * label : 10 + 5 * label] = 255
    return images, labels


@pytest.mark.parametrize("name", list(models.FAMILIES))
def test_a_few_steps_fit_one_batch(name):
    cfg, _ = load_small(
        f"train.model={name}",
        "train.augment=false",
        "train.lr=0.01",
        "train.warmup_epochs=0",
        "train.label_smoothing=0",
    )
    trainer = train.Trainer(cfg, 5, 300, 300, CPU)
    images, labels = patterned()
    losses = [trainer.train_step(images, labels)[0] for _ in range(60)]
    assert losses[-1] < 0.5 * losses[0]
    assert trainer.step == 60


def test_resume_continues_exactly_where_training_stopped():
    cfg, _ = load_small("train.mixup=0.2")
    batches = [batch(seed) for seed in range(8)]
    straight = train.Trainer(cfg, 5, 8, 4, CPU)
    for images, labels in batches:
        straight.train_step(images, labels)
    first = train.Trainer(cfg, 5, 8, 4, CPU)
    for images, labels in batches[:4]:
        first.train_step(images, labels)
    second = train.Trainer(cfg, 5, 8, 4, CPU)
    second.load(first.state())
    assert second.step == 4
    for images, labels in batches[4:]:
        second.train_step(images, labels)
    for name, value in straight.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name]), name
    for name, value in straight.ema.shadow.state_dict().items():
        assert torch.equal(value, second.ema.shadow.state_dict()[name]), name


def test_saved_model_loads_back(tmp_path):
    cfg, raw = load_small()
    trainer = train.Trainer(cfg, 5, 4, 4, CPU)
    images, labels = batch(1)
    trainer.train_step(images, labels)
    names = ["a", "b", "c", "d", "e"]
    train.save(
        {
            "state": trainer.best().state_dict(),
            "classes": names,
            "size": 32,
            "channels": 3,
            "model": "resnet",
            "config": raw,
        },
        tmp_path / "model.pt",
    )
    model, saved = train.load_model(tmp_path)
    assert saved["classes"] == names
    assert torch.allclose(
        train.predict(model, images, CPU, 8), train.predict(trainer.best(), images, CPU, 16), atol=1e-5
    )


def test_missing_model_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="train command"):
        train.load_model(tmp_path)


def test_bench_times_every_family_and_thread_count(tmp_path, capsys):
    cfg, _ = load_small(
        f"runs.dir={tmp_path.as_posix()}", "bench.warmup_seconds=0", "bench.seconds=0.01", "bench.rounds=2"
    )
    train.bench(cfg)
    result = json.loads((tmp_path / "bench.json").read_text())
    assert [row["model"] for row in result["models"]] == ["resnet", "mobilenet", "convnext"]
    assert all(row["images_per_second"] > 0 and row["params"] > 0 for row in result["models"])
    assert result["channels_first_images_per_second"] > 0
    assert result["thread_sweep"][0]["threads"] == 1
    assert result["rounds"] == 2 and result["threads"] == 1
    assert torch.get_num_threads() == 1
    assert "thread sweep for resnet" in capsys.readouterr().out

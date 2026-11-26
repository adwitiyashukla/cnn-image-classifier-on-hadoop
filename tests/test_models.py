import pytest
import torch
from torch import nn

from conftest import ROOT, load_small
from imgcnn import config, models
from imgcnn.evaluate import grad_cam
from imgcnn.train import make_optimizer

FAMILIES = list(models.FAMILIES)


@pytest.mark.parametrize("name", FAMILIES)
def test_forward_shape(cfg, name):
    model = models.build(cfg, 7, name)
    assert model(torch.rand(3, 3, 32, 32)).shape == (3, 7)
    assert models.build(cfg, 7, name, channels=1)(torch.rand(2, 1, 32, 32)).shape == (2, 7)


@pytest.mark.parametrize("name", FAMILIES)
def test_default_models_end_on_a_four_by_four_map(name):
    cfg, _ = config.load(ROOT / "config.toml")
    model = models.build(cfg, 345, name).eval()
    with torch.no_grad():
        assert model.features(torch.rand(1, 3, 32, 32)).shape[-2:] == (4, 4)


@pytest.mark.parametrize("name", FAMILIES)
def test_class_maps_average_to_the_logits(cfg, name):
    model = models.build(cfg, 7, name).eval()
    x = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)
        maps = model.maps(x)
    assert torch.allclose(maps.mean((2, 3)) + model.fc.bias, logits, atol=1e-5)


@pytest.mark.parametrize("name", FAMILIES)
def test_grad_cam_matches_the_class_map(cfg, name):
    model = models.build(cfg, 7, name).eval()
    nn.init.normal_(model.fc.weight, std=0.5)
    x = torch.rand(4, 3, 32, 32)
    target = torch.tensor([0, 3, 6, 2])
    cam, _ = grad_cam(model, x, target)
    with torch.no_grad():
        maps = model.maps(x)[torch.arange(4), target]
    cells = maps.shape[-1] * maps.shape[-2]
    assert torch.allclose(cam * cells, maps.clamp_min(0), atol=1e-5)
    pieces, logits = grad_cam(model, x, target, batch=3)
    assert torch.allclose(pieces, cam, atol=1e-6)
    with torch.no_grad():
        assert torch.allclose(logits, model(x), atol=1e-5)


def test_complexity_matches_a_hand_count():
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 4, 1, stride=2), nn.Flatten(), nn.Linear(64, 5))
    stats = models.complexity(model, 3, 8)
    assert stats["macs"] == 8 * 8 * 8 * 27 + 4 * 4 * 4 * 8 + 64 * 5
    assert stats["params"] == 224 + 36 + 325


def test_depthwise_convolution_is_counted_per_group():
    model = nn.Sequential(nn.Conv2d(6, 6, 3, padding=1, groups=6), nn.Flatten(), nn.Linear(6 * 16, 2))
    assert models.complexity(model, 6, 4)["macs"] == 6 * 16 * 9 + 96 * 2


def test_complexity_leaves_training_mode_alone(cfg):
    model = models.build(cfg, 7, "resnet").train()
    models.complexity(model, 3, 32)
    assert model.training


def test_drop_path_only_acts_in_training():
    layer = models.DropPath(0.5)
    x = torch.ones(2000, 1, 2, 2)
    assert torch.equal(layer.eval()(x), x)
    out = layer.train()(x)
    assert set(out.unique().tolist()) == {0.0, 2.0}
    assert abs(out.mean().item() - 1) < 0.1


def test_layer_norm_matches_torch():
    x = torch.randn(2, 6, 5, 5)
    ours = models.LayerNorm2d(6)
    theirs = nn.LayerNorm(6, eps=1e-6)
    assert torch.allclose(ours(x), theirs(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2), atol=1e-5)


def test_weight_decay_skips_norms_and_biases():
    cfg, _ = load_small("train.weight_decay=0.1")
    model = models.build(cfg, 7, "convnext")
    decayed, plain = make_optimizer(cfg.train, model).param_groups
    assert decayed["weight_decay"] == 0.1 and plain["weight_decay"] == 0
    assert all(p.ndim > 1 for p in decayed["params"])
    assert all(p.ndim <= 1 for p in plain["params"])
    assert len(decayed["params"]) + len(plain["params"]) == len(list(model.parameters()))

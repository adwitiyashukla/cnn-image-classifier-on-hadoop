import pytest

from conftest import ROOT, load_small
from imgcnn import config


def test_default_config_loads():
    cfg, raw = config.load(ROOT / "config.toml")
    assert cfg.dataset.image_size == 32
    assert cfg.train.model in ("resnet", "mobilenet", "convnext")
    assert raw["hdfs"]["user"]
    assert set(raw) == set(config.SCHEMA)


def test_ci_config_inherits_from_the_default():
    cfg, _ = config.load(ROOT / "config.ci.toml")
    default, _ = config.load(ROOT / "config.toml")
    assert cfg.source.classes > 0 and cfg.source.max_mb > 0
    assert cfg.train.lr == default.train.lr


def test_overrides_are_typed():
    cfg, _ = load_small("train.epochs=3", "train.model=convnext", "train.augment=false", "train.lr=0.01")
    assert cfg.train.epochs == 3 and cfg.train.model == "convnext"
    assert cfg.train.augment is False and cfg.train.lr == 0.01
    assert cfg.resnet.widths == [8, 16]


def test_integers_are_accepted_for_float_fields():
    cfg, _ = load_small("train.lr=1", "dataset.line_width=2")
    assert cfg.train.lr == 1.0 and isinstance(cfg.dataset.line_width, float)


@pytest.mark.parametrize(
    "override, message",
    [
        ("train.epochs=0", "train.epochs"),
        ("train.model=vgg", "train.model"),
        ("train.channels=2", "train.channels"),
        ("train.lr=fast", "train.lr"),
        ("train.augment=1", "train.augment"),
        ("train.epochs=true", "train.epochs"),
        ("dataset.image_size=30", "dataset.image_size"),
        ("resnet.widths=[8]", "resnet"),
        ("resnet.depths=[0, 1]", "resnet.depths"),
        ("train.speed=3", "unknown keys"),
        ("extra.key=1", "unknown sections"),
        ("train.epochs", "section.key=value"),
        ("score.fraction=0", "score.fraction"),
        ("train.log_every=0", "train.log_every"),
        ("train.checkpoint_every=0", "train.checkpoint_every"),
        ("eval.bootstrap=0", "eval.bootstrap"),
        ("eval.gradcam=0", "eval.gradcam"),
        ("export.latency_runs=1", "export.latency_runs"),
        ("source.retries=-1", "source.retries"),
        ("source.classes=2", "source.classes"),
        ("train.momentum=0", "train.momentum"),
        ("bench.seconds=0", "bench.seconds"),
        ("bench.rounds=0", "bench.rounds"),
        ('hdfs.root="/imgcnn/"', "hdfs.root"),
    ],
)
def test_bad_values_are_rejected(override, message):
    with pytest.raises(config.ConfigError, match=message):
        load_small(override)


def test_mixup_and_cutmix_cannot_be_combined():
    with pytest.raises(config.ConfigError, match="mixup or cutmix"):
        load_small("train.mixup=0.2", "train.cutmix=1.0")


def test_missing_key_is_reported(tmp_path):
    lines = (ROOT / "config.toml").read_text().splitlines()
    path = tmp_path / "config.toml"
    path.write_text("\n".join(line for line in lines if not line.startswith("epochs = ")))
    with pytest.raises(config.ConfigError, match=r"missing keys in \[train\]"):
        config.load(path)


def test_base_file_is_merged(tmp_path):
    path = tmp_path / "child.toml"
    path.write_text(f'base = "{(ROOT / "config.toml").as_posix()}"\n\n[train]\nepochs = 2\n')
    cfg, _ = config.load(path)
    assert cfg.train.epochs == 2
    assert cfg.dataset.image_size == 32

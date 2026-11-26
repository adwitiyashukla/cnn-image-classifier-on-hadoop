import getpass
import tomllib
from pathlib import Path
from types import SimpleNamespace

SCHEMA = {
    "hdfs": {"url": str, "user": str, "root": str},
    "hadoop": {"home": str, "python": str, "split_mb": int, "reducers": int, "shards": int, "score_map_mb": int},
    "source": {"url": str, "manifest": str, "classes": int, "max_mb": int, "workers": int, "retries": int},
    "dataset": {
        "seed": int,
        "image_size": int,
        "line_width": float,
        "train_per_class": int,
        "val_per_class": int,
        "test_per_class": int,
    },
    "train": {
        "model": str,
        "channels": int,
        "epochs": int,
        "batch_size": int,
        "optimizer": str,
        "lr": float,
        "weight_decay": float,
        "momentum": float,
        "warmup_epochs": float,
        "label_smoothing": float,
        "mixup": float,
        "cutmix": float,
        "augment": bool,
        "rotate": float,
        "scale": float,
        "shift": float,
        "flip": bool,
        "ema": float,
        "clip_grad": float,
        "fraction": float,
        "recognized_only": bool,
        "val_samples": int,
        "threads": int,
        "workers": int,
        "buffer": int,
        "log_every": int,
        "checkpoint_every": int,
        "seed": int,
        "device": str,
    },
    "resnet": {"stem": int, "widths": list, "depths": list},
    "mobilenet": {"stem": int, "expand": list, "channels": list, "repeats": list, "strides": list, "head": int},
    "convnext": {"widths": list, "depths": list, "drop_path": float},
    "eval": {"batch_size": int, "bootstrap": int, "robustness_samples": int, "min_country": int, "gradcam": int},
    "export": {"latency_runs": int, "calibration_samples": int},
    "bench": {"warmup_seconds": float, "seconds": float, "rounds": int},
    "score": {"fraction": float, "threshold": float, "top_k": int, "batch": int, "threads": int},
    "scratch": {
        "classes": int,
        "train_per_class": int,
        "test_per_class": int,
        "epochs": int,
        "lr": float,
        "batch_size": int,
    },
    "runs": {"dir": str},
}

CHECKS = [
    ("dataset.image_size", lambda v: v in (16, 24, 32, 48, 64), "must be 16, 24, 32, 48 or 64"),
    ("dataset.line_width", lambda v: 0.5 <= v <= 6, "must be between 0.5 and 6"),
    ("dataset.train_per_class", lambda v: v > 0, "must be positive"),
    ("dataset.val_per_class", lambda v: v > 0, "must be positive"),
    ("dataset.test_per_class", lambda v: v > 0, "must be positive"),
    ("hdfs.root", lambda v: len(v) > 1 and v.startswith("/") and not v.endswith("/"), "must look like /imgcnn"),
    ("hadoop.split_mb", lambda v: v >= 1, "must be at least 1"),
    ("hadoop.score_map_mb", lambda v: v >= 256, "must be at least 256"),
    ("hadoop.reducers", lambda v: v >= 1, "must be at least 1"),
    ("hadoop.shards", lambda v: v >= 1, "must be at least 1"),
    ("source.classes", lambda v: v == 0 or v >= 3, "must be 0 (all) or at least 3"),
    ("source.retries", lambda v: v >= 0, "must not be negative"),
    ("source.max_mb", lambda v: v >= 0, "must be 0 (whole file) or a positive size"),
    ("source.workers", lambda v: 1 <= v <= 16, "must be between 1 and 16"),
    ("train.model", lambda v: v in ("resnet", "mobilenet", "convnext"), "must be resnet, mobilenet or convnext"),
    ("train.channels", lambda v: v in (1, 3), "must be 1 (ink only) or 3 (ink, order, time)"),
    ("train.optimizer", lambda v: v in ("adamw", "sgd"), "must be adamw or sgd"),
    ("train.device", lambda v: v in ("auto", "cpu", "cuda"), "must be auto, cpu or cuda"),
    ("train.epochs", lambda v: v >= 1, "must be at least 1"),
    ("train.batch_size", lambda v: v >= 2, "must be at least 2"),
    ("train.lr", lambda v: 0 < v < 10, "must be in (0, 10)"),
    ("train.weight_decay", lambda v: v >= 0, "must not be negative"),
    ("train.warmup_epochs", lambda v: v >= 0, "must not be negative"),
    ("train.label_smoothing", lambda v: 0 <= v < 1, "must be in [0, 1)"),
    ("train.mixup", lambda v: v >= 0, "must not be negative"),
    ("train.cutmix", lambda v: v >= 0, "must not be negative"),
    ("train.ema", lambda v: v == 0 or 0.9 <= v < 1, "must be 0 (off) or in [0.9, 1)"),
    ("train.clip_grad", lambda v: v >= 0, "must not be negative"),
    ("train.fraction", lambda v: 0 < v <= 1, "must be in (0, 1]"),
    ("train.val_samples", lambda v: v >= 0, "must not be negative"),
    ("train.threads", lambda v: v >= 0, "must be 0 (auto) or positive"),
    ("train.workers", lambda v: v >= 0, "must not be negative"),
    ("train.buffer", lambda v: v >= 1, "must be at least 1"),
    ("train.momentum", lambda v: 0 < v < 1, "must be in (0, 1)"),
    ("train.log_every", lambda v: v >= 1, "must be at least 1"),
    ("train.checkpoint_every", lambda v: v >= 1, "must be at least 1"),
    ("eval.batch_size", lambda v: v >= 1, "must be at least 1"),
    ("eval.bootstrap", lambda v: v >= 1, "must be at least 1"),
    ("eval.robustness_samples", lambda v: v >= 1, "must be at least 1"),
    ("eval.min_country", lambda v: v >= 1, "must be at least 1"),
    ("eval.gradcam", lambda v: v >= 1, "must be at least 1"),
    ("export.latency_runs", lambda v: v >= 20, "must be at least 20"),
    ("export.calibration_samples", lambda v: v >= 1, "must be at least 1"),
    ("bench.warmup_seconds", lambda v: v >= 0, "must not be negative"),
    ("bench.seconds", lambda v: v > 0, "must be positive"),
    ("bench.rounds", lambda v: v >= 1, "must be at least 1"),
    ("convnext.drop_path", lambda v: 0 <= v < 1, "must be in [0, 1)"),
    ("score.fraction", lambda v: 0 < v <= 1, "must be in (0, 1]"),
    ("score.threshold", lambda v: 0 < v < 1, "must be in (0, 1)"),
    ("score.batch", lambda v: v >= 1, "must be at least 1"),
    ("score.top_k", lambda v: v >= 1, "must be at least 1"),
    ("score.threads", lambda v: v >= 1, "must be at least 1"),
    ("scratch.classes", lambda v: 2 <= v <= 345, "must be between 2 and 345"),
    ("scratch.train_per_class", lambda v: v >= 1, "must be at least 1"),
    ("scratch.test_per_class", lambda v: v >= 1, "must be at least 1"),
    ("scratch.epochs", lambda v: v >= 1, "must be at least 1"),
    ("scratch.batch_size", lambda v: v >= 1, "must be at least 1"),
    ("scratch.lr", lambda v: v > 0, "must be positive"),
]

LIST_LENGTHS = [
    ("resnet", ("widths", "depths")),
    ("mobilenet", ("expand", "channels", "repeats", "strides")),
    ("convnext", ("widths", "depths")),
]


class ConfigError(ValueError):
    pass


def parse_value(raw):
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def read(path):
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    base = data.pop("base", None)
    if base is None:
        return data
    merged = read(path.parent / base)
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] must be a table")
        merged.setdefault(section, {}).update(values)
    return merged


def check_type(name, value, kind):
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind is int and isinstance(value, bool):
        raise ConfigError(f"{name} must be an integer")
    if kind is list:
        if not isinstance(value, list) or not value or not all(isinstance(v, int) and v > 0 for v in value):
            raise ConfigError(f"{name} must be a non-empty list of positive integers")
        return value
    if not isinstance(value, kind):
        raise ConfigError(f"{name} must be {kind.__name__}, got {type(value).__name__}")
    return value


def load(path, overrides=()):
    data = read(path)
    for item in overrides:
        key, sep, raw = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot:
            raise ConfigError(f"override {item!r} must look like section.key=value")
        data.setdefault(section, {})[name] = parse_value(raw.strip())
    unknown = set(data) - set(SCHEMA)
    if unknown:
        raise ConfigError(f"unknown sections: {sorted(unknown)}")
    clean = {}
    for section, fields in SCHEMA.items():
        values = data.get(section, {})
        extra = set(values) - set(fields)
        if extra:
            raise ConfigError(f"unknown keys in [{section}]: {sorted(extra)}")
        missing = set(fields) - set(values)
        if missing:
            raise ConfigError(f"missing keys in [{section}]: {sorted(missing)}")
        clean[section] = {k: check_type(f"{section}.{k}", values[k], kind) for k, kind in fields.items()}
    for name, rule, message in CHECKS:
        section, key = name.split(".")
        if not rule(clean[section][key]):
            raise ConfigError(f"{name} {message}, got {clean[section][key]!r}")
    for section, keys in LIST_LENGTHS:
        lengths = {len(clean[section][k]) for k in keys}
        if len(lengths) != 1:
            raise ConfigError(f"[{section}] lists {keys} must have the same length")
    if clean["train"]["mixup"] > 0 and clean["train"]["cutmix"] > 0:
        raise ConfigError("use mixup or cutmix, not both")
    if not clean["hdfs"]["user"]:
        clean["hdfs"]["user"] = getpass.getuser()
    return namespace(clean), clean


def namespace(clean):
    return SimpleNamespace(**{section: SimpleNamespace(**values) for section, values in clean.items()})

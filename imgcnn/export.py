import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

from imgcnn import data, models
from imgcnn.train import load_model

REPO = Path(__file__).resolve().parent.parent
SPACE_FILES = ("README.md", "index.html", "doodle.js")


class Exported(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        maps = self.model.maps(image)
        return maps.mean((2, 3)) + self.model.fc.bias, maps


class Calibration(CalibrationDataReader):
    def __init__(self, images, batch=64):
        self.batches = iter([{"image": images[i : i + batch]} for i in range(0, len(images), batch)])

    def get_next(self):
        return next(self.batches, None)


def session(path, threads=0):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def to_float(images):
    return images.numpy().astype(np.float32) / 255.0


def accuracy(run, images, labels, batch=512):
    hits = 0
    for start in range(0, len(images), batch):
        logits = run(to_float(images[start : start + batch]))
        hits += int((logits.argmax(1) == labels[start : start + batch].numpy()).sum())
    return hits / len(images)


def latency(run, sample, runs):
    for _ in range(10):
        run(sample)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        run(sample)
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return {"median_ms": times[len(times) // 2], "p95_ms": times[int(len(times) * 0.95) - 1]}


def throughput(run, batch, seconds=3.0):
    run(batch)
    started, done = time.perf_counter(), 0
    while time.perf_counter() - started < seconds:
        run(batch)
        done += len(batch)
    return done / (time.perf_counter() - started)


def export(cfg, fs, run_dir):
    run_dir = Path(run_dir)
    model, saved = load_model(run_dir)
    classes, size, channels = saved["classes"], saved["size"], saved["channels"]
    wrapper = Exported(model).eval()
    val = data.load_split(cfg, fs, "val", size, channels, cfg.export.calibration_samples)
    test = data.load_split(cfg, fs, "test", size, channels)
    onnx_path, int8_path = run_dir / "model.onnx", run_dir / "model.int8.onnx"
    example = torch.from_numpy(to_float(val["images"][:8]))
    logging.getLogger("torch.onnx").setLevel(logging.ERROR)
    torch.onnx.export(
        wrapper,
        (example,),
        str(onnx_path),
        input_names=["image"],
        output_names=["logits", "maps"],
        dynamic_shapes={"image": {0: torch.export.Dim("batch")}},
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    prepared = run_dir / "model.prepared.onnx"
    quant_pre_process(str(onnx_path), str(prepared))
    quantize_static(
        str(prepared),
        str(int8_path),
        Calibration(to_float(val["images"])),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    prepared.unlink()
    meta = {"classes": classes, "size": size, "channels": channels, "width": saved["width"]}
    (run_dir / "model.json").write_text(json.dumps(meta))
    stats = models.complexity(model, channels, size)
    with torch.no_grad():
        cells = wrapper(example[:1])[1][0, 0].numel()
    stats["exported_macs"] = stats["macs"] + (cells - 1) * model.fc.in_features * model.fc.out_features

    def torch_run(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()

    def onnx_run(path, threads=0):
        active = session(path, threads)
        return lambda x: active.run(["logits"], {"image": x})[0]

    probe = to_float(test["images"][:256])
    difference = float(np.abs(onnx_run(onnx_path)(probe) - torch_run(probe)).max())
    runners = {"torch": torch_run, "onnx": onnx_run(onnx_path), "int8": onnx_run(int8_path)}
    single = {"onnx": onnx_run(onnx_path, 1), "int8": onnx_run(int8_path, 1)}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    one = {
        name: latency(run, probe[:1], cfg.export.latency_runs) for name, run in {"torch": torch_run, **single}.items()
    }
    torch.set_num_threads(threads)
    report = {
        **stats,
        "onnx_max_abs_difference": difference,
        "bytes": {
            "torch": (run_dir / "model.pt").stat().st_size,
            "onnx": onnx_path.stat().st_size,
            "int8": int8_path.stat().st_size,
        },
        "top1": {name: accuracy(run, test["images"], test["labels"]) for name, run in runners.items()},
        "latency_one_thread": one,
        "images_per_second": {name: throughput(run, probe) for name, run in runners.items()},
    }
    (run_dir / "export.json").write_text(json.dumps(report, indent=1))
    print(
        f"{saved['model']}: {report['params'] / 1e6:.2f}M params, {report['macs'] / 1e6:.1f}M MACs "
        f"({report['exported_macs'] / 1e6:.1f}M with the class maps in the exported graph), "
        f"onnx output differs from torch by at most {difference:.2e}"
    )
    for name in runners:
        print(
            f"{name:6s} {report['bytes'][name] / 1e6:6.2f} MB  top-1 {report['top1'][name]:.4f}  "
            f"{one[name]['median_ms']:6.2f} ms per drawing on one thread  "
            f"{report['images_per_second'][name]:8.0f} img/s batched"
        )
    return report


def bundle(run_dir, out):
    run_dir, out = Path(run_dir), Path(out)
    weights = ("model.onnx", "model.json")
    for name in weights:
        if not (run_dir / name).exists():
            raise SystemExit(f"{run_dir / name} is missing, run the export command first")
    (out / "model").mkdir(parents=True, exist_ok=True)
    for name in SPACE_FILES:
        shutil.copyfile(REPO / "space" / name, out / name)
    for name in weights:
        shutil.copyfile(run_dir / name, out / "model" / name)
    size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    print(f"space folder with the model from {run_dir} is in {out}, {size / 1e6:.1f} MB")

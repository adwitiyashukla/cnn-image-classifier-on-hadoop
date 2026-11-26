import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from imgcnn import data, raster

WIDTHS = (16, 32, 64)
MOMENTUM = 0.9


class Conv:
    def __init__(self, cin, cout, kernel, rng, dtype):
        scale = np.sqrt(2 / (cin * kernel * kernel))
        self.params = {"w": rng.normal(0, scale, (cout, cin, kernel, kernel)).astype(dtype), "b": np.zeros(cout, dtype)}
        self.grads = {}
        self.kernel = kernel

    def forward(self, x, train):
        k, pad = self.kernel, self.kernel // 2
        n, c, h, w = x.shape
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(2, 3))
        self.columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, c * k * k)
        self.shape = x.shape
        weight = self.params["w"].reshape(len(self.params["w"]), -1)
        out = self.columns @ weight.T + self.params["b"]
        return out.reshape(n, h, w, -1).transpose(0, 3, 1, 2)

    def backward(self, grad):
        k, pad = self.kernel, self.kernel // 2
        n, c, h, w = self.shape
        flat = grad.transpose(0, 2, 3, 1).reshape(n * h * w, -1)
        self.grads["w"] = (flat.T @ self.columns).reshape(self.params["w"].shape)
        self.grads["b"] = flat.sum(0)
        weight = self.params["w"].reshape(len(self.params["w"]), -1)
        columns = (flat @ weight).reshape(n, h, w, c, k, k)
        padded = np.zeros((n, c, h + 2 * pad, w + 2 * pad), grad.dtype)
        for i in range(k):
            for j in range(k):
                padded[:, :, i : i + h, j : j + w] += columns[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return padded[:, :, pad : pad + h, pad : pad + w]


class BatchNorm:
    def __init__(self, channels, dtype, momentum=0.1, eps=1e-5):
        self.params = {"w": np.ones(channels, dtype), "b": np.zeros(channels, dtype)}
        self.grads = {}
        self.mean = np.zeros(channels, dtype)
        self.var = np.ones(channels, dtype)
        self.momentum, self.eps = momentum, eps

    def forward(self, x, train):
        if train:
            mean, var = x.mean((0, 2, 3)), x.var((0, 2, 3))
            count = x.size / x.shape[1]
            self.mean = (1 - self.momentum) * self.mean + self.momentum * mean
            self.var = (1 - self.momentum) * self.var + self.momentum * var * count / (count - 1)
        else:
            mean, var = self.mean, self.var
        self.std = np.sqrt(var + self.eps)[None, :, None, None]
        self.normed = (x - mean[None, :, None, None]) / self.std
        return self.params["w"][None, :, None, None] * self.normed + self.params["b"][None, :, None, None]

    def backward(self, grad):
        self.grads["w"] = (grad * self.normed).sum((0, 2, 3))
        self.grads["b"] = grad.sum((0, 2, 3))
        scaled = grad * self.params["w"][None, :, None, None]
        centered = scaled - scaled.mean((0, 2, 3), keepdims=True)
        return (centered - self.normed * (scaled * self.normed).mean((0, 2, 3), keepdims=True)) / self.std


class ReLU:
    params, grads = {}, {}

    def forward(self, x, train):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class MaxPool:
    params, grads = {}, {}

    def forward(self, x, train):
        n, c, h, w = x.shape
        windows = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
        self.index = windows.argmax(-1)[..., None]
        self.shape = x.shape
        return np.take_along_axis(windows, self.index, -1)[..., 0]

    def backward(self, grad):
        n, c, h, w = self.shape
        windows = np.zeros((n, c, h // 2, w // 2, 4), grad.dtype)
        np.put_along_axis(windows, self.index, grad[..., None], -1)
        return windows.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class Head:
    def __init__(self, cin, classes, rng, dtype):
        self.params = {"w": rng.normal(0, 0.01, (classes, cin)).astype(dtype), "b": np.zeros(classes, dtype)}
        self.grads = {}

    def forward(self, x, train):
        self.shape = x.shape
        self.pooled = x.mean((2, 3))
        return self.pooled @ self.params["w"].T + self.params["b"]

    def backward(self, grad):
        n, c, h, w = self.shape
        self.grads["w"] = grad.T @ self.pooled
        self.grads["b"] = grad.sum(0)
        return np.broadcast_to((grad @ self.params["w"])[:, :, None, None] / (h * w), self.shape)


def cross_entropy(logits, labels):
    shifted = logits - logits.max(1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    rows = np.arange(len(labels))
    grad = np.exp(log_probs)
    grad[rows, labels] -= 1
    return -log_probs[rows, labels].mean(), grad / len(labels)


class Network:
    def __init__(self, classes, widths=WIDTHS, seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.layers, cin = [], 1
        for width in widths:
            self.layers += [Conv(cin, width, 3, rng, dtype), BatchNorm(width, dtype), ReLU(), MaxPool()]
            cin = width
        self.layers.append(Head(cin, classes, rng, dtype))
        self.velocity = {}

    def forward(self, x, train=True):
        for layer in self.layers:
            x = layer.forward(x, train)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, x, labels, rate):
        loss, grad = cross_entropy(self.forward(x), labels)
        self.backward(grad)
        for index, layer in enumerate(self.layers):
            for name, value in layer.params.items():
                velocity = MOMENTUM * self.velocity.get((index, name), 0) + layer.grads[name]
                self.velocity[(index, name)] = velocity
                value -= rate * velocity
        return loss

    def accuracy(self, x, labels, batch=500):
        hits = 0
        for start in range(0, len(x), batch):
            hits += (
                self.forward(x[start : start + batch], train=False).argmax(1) == labels[start : start + batch]
            ).sum()
        return hits / len(x)


class TorchHead(nn.Module):
    def __init__(self, cin, classes):
        super().__init__()
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


def twin(network):
    layers, pairs = [], []
    for layer in network.layers:
        if isinstance(layer, Conv):
            cout, cin, k, _ = layer.params["w"].shape
            module = nn.Conv2d(cin, cout, k, padding=k // 2)
            pairs += [(module.weight, layer, "w"), (module.bias, layer, "b")]
        elif isinstance(layer, BatchNorm):
            module = nn.BatchNorm2d(len(layer.mean), eps=layer.eps, momentum=layer.momentum)
            pairs += [(module.weight, layer, "w"), (module.bias, layer, "b")]
        elif isinstance(layer, ReLU):
            module = nn.ReLU()
        elif isinstance(layer, MaxPool):
            module = nn.MaxPool2d(2)
        else:
            classes, cin = layer.params["w"].shape
            module = TorchHead(cin, classes)
            pairs += [(module.fc.weight, layer, "w"), (module.fc.bias, layer, "b")]
        layers.append(module)
    model = nn.Sequential(*layers).to(torch.from_numpy(network.layers[0].params["w"]).dtype)
    with torch.no_grad():
        for tensor, layer, name in pairs:
            tensor.copy_(torch.from_numpy(layer.params[name]))
    return model, pairs


def gradient_gap(network, model, pairs, x, labels):
    loss, grad = cross_entropy(network.forward(x), labels)
    network.backward(grad)
    model.zero_grad()
    reference = nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(labels))
    reference.backward()
    ours = np.concatenate([layer.grads[name].ravel() for _, layer, name in pairs])
    theirs = np.concatenate([tensor.grad.numpy().ravel() for tensor, _, _ in pairs])
    gap = float(np.linalg.norm(ours - theirs) / np.linalg.norm(theirs))
    return gap, abs(float(loss) - reference.item())


def collect(cfg, fs, split, size, classes, per_class):
    pixels = size * size
    images, labels, filled = [], [], [0] * classes
    for path in data.shard_files(cfg, fs, split):
        for line in fs.lines(path):
            label, _, _, _, image = line.split(b"\t")
            label = int(label)
            if label >= classes or filled[label] >= per_class:
                continue
            filled[label] += 1
            images.append(np.frombuffer(raster.decode(image)[:pixels], np.uint8))
            labels.append(label)
            if min(filled) >= per_class:
                break
        if min(filled) >= per_class:
            break
    if not images:
        raise SystemExit(f"no {split} drawings for the first {classes} classes, run the build command first")
    x = np.stack(images).reshape(-1, 1, size, size).astype(np.float32) / 255
    return x, np.array(labels)


def scratch(cfg, fs, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = data.dataset_meta(cfg, fs)
    size, s = meta["size"], cfg.scratch
    names = meta["classes"][: s.classes]
    x_train, y_train = collect(cfg, fs, "train", size, len(names), s.train_per_class)
    x_test, y_test = collect(cfg, fs, "test", size, len(names), s.test_per_class)
    print(f"{len(names)} classes, {len(x_train):,} train and {len(x_test):,} test drawings, ink channel only")
    if len(x_train) < s.batch_size:
        raise SystemExit(f"{len(x_train)} train drawings cannot fill one batch of {s.batch_size}")

    exact = Network(len(names), seed=cfg.train.seed, dtype=np.float64)
    model, pairs = twin(exact)
    gap, loss_gap = gradient_gap(
        exact, model, pairs, x_train[: s.batch_size].astype(np.float64), y_train[: s.batch_size]
    )
    print(
        f"numpy backward pass against torch autograd in float64: relative gradient error {gap:.2e}, "
        f"loss differs by {loss_gap:.2e}"
    )

    network = Network(len(names), seed=cfg.train.seed)
    model, pairs = twin(network)
    optimizer = torch.optim.SGD(model.parameters(), lr=s.lr, momentum=MOMENTUM)
    order = np.random.default_rng(cfg.train.seed)
    curves = {"numpy": [], "torch": []}
    seconds = {"numpy": 0.0, "torch": 0.0}
    steps = 0
    for epoch in range(s.epochs):
        index = order.permutation(len(x_train))
        totals = {"numpy": 0.0, "torch": 0.0}
        batches = 0
        for start in range(0, len(index) - s.batch_size + 1, s.batch_size):
            rows = index[start : start + s.batch_size]
            x, y = x_train[rows], y_train[rows]
            began = time.perf_counter()
            totals["numpy"] += float(network.step(x, y, s.lr))
            seconds["numpy"] += time.perf_counter() - began
            began = time.perf_counter()
            model.train()
            loss = nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            seconds["torch"] += time.perf_counter() - began
            totals["torch"] += loss.item()
            batches += 1
        steps += batches
        for name in curves:
            curves[name].append(totals[name] / batches)
        print(
            f"epoch {epoch + 1}/{s.epochs}: numpy loss {curves['numpy'][-1]:.4f}, torch loss {curves['torch'][-1]:.4f}",
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        torch_top1 = (model(torch.from_numpy(x_test)).argmax(1).numpy() == y_test).mean()
    report = {
        "classes": names,
        "train": len(x_train),
        "test": len(x_test),
        "gradient_error": gap,
        "numpy": {
            "top1": float(network.accuracy(x_test, y_test)),
            "loss": curves["numpy"],
            "ms_per_step": 1000 * seconds["numpy"] / steps,
        },
        "torch": {"top1": float(torch_top1), "loss": curves["torch"], "ms_per_step": 1000 * seconds["torch"] / steps},
    }
    (run_dir / "scratch.json").write_text(json.dumps(report, indent=1))
    print(
        f"test top-1: numpy {report['numpy']['top1']:.4f}, torch {report['torch']['top1']:.4f} from the same "
        f"weights and batches; {report['numpy']['ms_per_step']:.0f} ms per step in numpy, "
        f"{report['torch']['ms_per_step']:.0f} ms in torch"
    )
    return report

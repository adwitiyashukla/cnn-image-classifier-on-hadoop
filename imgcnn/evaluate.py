import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from imgcnn import data, metrics, plots
from imgcnn.train import LAYOUT, load_model, pick_device, predict

LEVELS = {
    "noise": (0.05, 0.1, 0.2),
    "blur": (0.5, 1.0, 1.5),
    "fade": (0.7, 0.5, 0.3),
    "thick": (0.25, 0.5, 1.0),
    "dropout": (0.1, 0.25, 0.4),
    "erase": (8, 12, 16),
    "rotate": (15, 30, 45),
    "shift": (3, 5, 7),
    "shrink": (0.8, 0.65, 0.5),
}


def warp(x, angle=0.0, zoom=1.0, dx=None, dy=None):
    n = x.shape[0]
    cos, sin = math.cos(math.radians(angle)) / zoom, math.sin(math.radians(angle)) / zoom
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    if dx is not None:
        theta[:, 0, 2], theta[:, 1, 2] = dx, dy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def corrupt(x, kind, level, generator):
    n, channels, size, _ = x.shape
    if kind == "noise":
        return (x + level * torch.randn(x.shape, generator=generator)).clamp(0, 1)
    if kind == "blur":
        axis = torch.arange(-2, 3).float()
        kernel = torch.exp(-(axis**2) / (2 * level**2))
        kernel = torch.outer(kernel, kernel)
        kernel = (kernel / kernel.sum()).expand(channels, 1, 5, 5)
        return F.conv2d(x, kernel, padding=2, groups=channels)
    if kind == "fade":
        return x * level
    if kind == "thick":
        return torch.lerp(x, F.max_pool2d(x, 3, 1, 1), level)
    if kind == "dropout":
        return x * (torch.rand(n, 1, size, size, generator=generator) >= level)
    if kind == "erase":
        x = x.clone()
        tops = torch.randint(0, size - level + 1, (n,), generator=generator).tolist()
        lefts = torch.randint(0, size - level + 1, (n,), generator=generator).tolist()
        for i, (top, left) in enumerate(zip(tops, lefts)):
            x[i, :, top : top + level, left : left + level] = 0
        return x
    if kind == "rotate":
        return warp(x, angle=level)
    if kind == "shift":
        angle = torch.rand(n, generator=generator) * 2 * math.pi
        step = 2 * level / size
        return warp(x, dx=torch.cos(angle) * step, dy=torch.sin(angle) * step)
    return warp(x, zoom=level)


@torch.no_grad()
def predict_float(model, x, device, batch):
    outputs = []
    for start in range(0, len(x), batch):
        outputs.append(model(x[start : start + batch].to(device, memory_format=LAYOUT)).float().cpu())
    return torch.cat(outputs)


def grad_cam(model, x, target, batch=256):
    maps, outputs = [], []
    for start in range(0, len(x), batch):
        with torch.no_grad():
            features = model.features(x[start : start + batch])
        features.requires_grad_(True)
        logits = model.fc(features.mean((2, 3)))
        score = logits.gather(1, target[start : start + batch, None]).sum()
        gradient = torch.autograd.grad(score, features)[0]
        weights = gradient.mean((2, 3), keepdim=True)
        maps.append(F.relu((weights * features).sum(1)).detach())
        outputs.append(logits.detach())
    return torch.cat(maps), torch.cat(outputs)


def deletion_test(model, x, labels, cells, generator, batch=256):
    cam, logits = grad_cam(model, x, labels, batch)
    n, grid = cam.shape[0], cam.shape[-1]
    cell = x.shape[-1] // grid
    flat = cam.flatten(1)
    inked = F.max_pool2d(x[:, :1], cell).flatten(1) > 0
    chosen = flat.topk(cells, 1).indices
    noise = torch.rand(flat.shape, generator=generator).masked_fill(~inked, -1)
    random = noise.topk(cells, 1).indices

    def delete(index):
        mask = torch.ones(n, grid * grid)
        mask.scatter_(1, index, 0.0)
        mask = mask.view(n, 1, grid, grid).repeat_interleave(cell, 2).repeat_interleave(cell, 3)
        return x * mask

    def confidence(images):
        out = predict_float(model, images, x.device, batch)
        return out.softmax(1).gather(1, labels[:, None]).mean().item()

    by_cam, by_random = delete(chosen), delete(random)
    ink = x[:, 0].sum().item()
    return {
        "cells": cells,
        "grid": grid,
        "p_original": logits.softmax(1).gather(1, labels[:, None]).mean().item(),
        "p_cam_deleted": confidence(by_cam),
        "p_random_deleted": confidence(by_random),
        "ink_removed_cam": 1 - by_cam[:, 0].sum().item() / ink,
        "ink_removed_random": 1 - by_random[:, 0].sum().item() / ink,
    }


def share(hits):
    return hits.float().mean().item() if len(hits) else None


def evaluate(cfg, fs, run_dir):
    run_dir = Path(run_dir)
    device = pick_device(cfg)
    model, saved = load_model(run_dir, device)
    classes, size, channels = saved["classes"], saved["size"], saved["channels"]
    count = len(classes)
    batch = cfg.eval.batch_size
    test = data.load_split(cfg, fs, "test", size, channels)
    val = data.load_split(cfg, fs, "val", size, channels)
    labels = test["labels"]
    logits = predict(model, test["images"], device, batch)
    rank = metrics.ranks(logits, labels)
    correct = rank == 0
    precision = torch.where(rank < 3, 1.0 / (rank + 1), torch.zeros(len(rank)))
    report = {"samples": len(labels), **metrics.summarize(logits, labels)}
    report["interval"] = {
        "top1": metrics.bootstrap(correct, cfg.eval.bootstrap),
        "top3": metrics.bootstrap(rank < 3, cfg.eval.bootstrap),
        "map3": metrics.bootstrap(precision, cfg.eval.bootstrap),
    }

    predicted = logits.argmax(1)
    confusion = torch.bincount(labels * count + predicted, minlength=count * count).view(count, count)
    recall = (confusion.diag().float() / confusion.sum(1).clamp_min(1)).tolist()
    order = sorted((i for i in range(count) if confusion[i].sum() > 0), key=lambda i: recall[i])
    report["classes"] = {
        "worst": [{"class": classes[i], "top1": recall[i]} for i in order[:10]],
        "best": [{"class": classes[i], "top1": recall[i]} for i in order[::-1][:10]],
        "below_half": sum(recall[i] < 0.5 for i in order),
        "per_class": {classes[i]: recall[i] for i in order},
    }
    swaps = torch.triu(confusion + confusion.T, diagonal=1)
    pairs = []
    for index in swaps.flatten().topk(min(15, count * count)).indices.tolist():
        a, b = divmod(index, count)
        if swaps[a, b] == 0:
            continue
        pairs.append({"a": classes[a], "b": classes[b], "a_as_b": int(confusion[a, b]), "b_as_a": int(confusion[b, a])})
    report["confused_pairs"] = pairs

    by_country = defaultdict(lambda: [0, 0])
    for country, hit in zip(test["countries"], correct.tolist()):
        by_country[country][0] += 1
        by_country[country][1] += hit
    countries = [
        {"country": c, "n": n, "top1": hit / n} for c, (n, hit) in by_country.items() if n >= cfg.eval.min_country
    ]
    countries.sort(key=lambda row: -row["top1"])
    report["countries"] = countries
    flag = test["recognized"]
    report["recognized"] = {
        "recognized": {"n": int(flag.sum()), "top1": share(correct[flag])},
        "unrecognized": {"n": int((~flag).sum()), "top1": share(correct[~flag])},
    }

    val_logits = predict(model, val["images"], device, batch)
    temperature = metrics.fit_temperature(val_logits, val["labels"])
    ece_before, rows_before = metrics.calibration(logits, labels)
    ece_after, rows_after = metrics.calibration(logits / temperature, labels)
    report["calibration"] = {
        "temperature": temperature,
        "ece": ece_before,
        "ece_scaled": ece_after,
        "nll": F.cross_entropy(logits, labels).item(),
        "nll_scaled": F.cross_entropy(logits / temperature, labels).item(),
        "bins": rows_before,
        "bins_scaled": rows_after,
    }

    generator = torch.Generator().manual_seed(cfg.train.seed)
    subset = min(len(labels), cfg.eval.robustness_samples)
    clean = test["images"][:subset].float() / 255
    clean_top1 = correct[:subset].float().mean().item()
    robustness = {}
    for kind, levels in LEVELS.items():
        scores = []
        for level in levels:
            out = predict_float(model, corrupt(clean, kind, level, generator), device, batch)
            scores.append((out.argmax(1) == labels[:subset]).float().mean().item())
        robustness[kind] = {"levels": list(levels), "top1": scores}
    mean = sum(sum(r["top1"]) for r in robustness.values()) / sum(len(r["top1"]) for r in robustness.values())
    report["robustness"] = {"samples": subset, "clean_top1": clean_top1, "mean_top1": mean, "corruptions": robustness}

    probe = min(len(labels), 2048)
    cpu_model = model.cpu()
    sample = test["images"][:probe].float() / 255
    report["deletion"] = [
        deletion_test(cpu_model, sample, labels[:probe], cells, generator, batch) for cells in (1, 2, 3, 4)
    ]
    shown = test["images"][: cfg.eval.gradcam].float() / 255
    with torch.no_grad():
        guess = cpu_model(shown).argmax(1)
    cam, cam_logits = grad_cam(cpu_model, shown, guess)
    heat = F.interpolate(cam[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
    heat = heat / heat.flatten(1).max(1).values.clamp_min(1e-9)[:, None, None]
    captions = []
    for row, (index, p) in enumerate(zip(guess.tolist(), cam_logits.softmax(1).max(1).values.tolist())):
        truth = classes[labels[row]]
        captions.append(f"{classes[index]} {p:.2f}" + ("" if classes[index] == truth else f"\n(is {truth})"))
    plots.gradcam(shown[:, 0].numpy(), heat.numpy(), captions, run_dir / "gradcam.png")
    plots.reliability(rows_before, rows_after, ece_before, ece_after, run_dir / "reliability.png")
    if pairs:
        plots.confused_pairs(pairs, run_dir / "confused_pairs.png")
    (run_dir / "eval.json").write_text(json.dumps(report, indent=1))

    low, high = report["interval"]["top1"]
    deletion = report["deletion"][1]
    print(f"test set: {len(labels):,} drawings, {count} classes")
    print(
        f"top-1 {report['top1']:.4f} (95% CI {low:.4f} to {high:.4f}), top-3 {report['top3']:.4f}, "
        f"MAP@3 {report['map3']:.4f}"
    )
    groups = {
        name: "none in the test set" if group["top1"] is None else f"{group['top1']:.4f}"
        for name, group in report["recognized"].items()
    }
    print(f"recognized drawings {groups['recognized']}, unrecognized {groups['unrecognized']}")
    print(f"ECE {ece_before:.4f} raw, {ece_after:.4f} after temperature {temperature:.3f}")
    print(f"corrupted top-1 {mean:.4f} on average against {clean_top1:.4f} clean")
    print(
        f"Grad-CAM deletion: p(true) {deletion['p_original']:.3f}, {deletion['p_cam_deleted']:.3f} after removing the "
        f"{deletion['cells']} hottest cells, {deletion['p_random_deleted']:.3f} after removing {deletion['cells']} "
        f"random inked cells"
    )
    if pairs:
        print(f"most confused pair: {pairs[0]['a']} and {pairs[0]['b']}")
    return report

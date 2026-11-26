import math

import matplotlib.pyplot as plt
import numpy as np

from imgcnn import raster

plt.switch_backend("Agg")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
RULE = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
STEPS = ("#86b6ef", "#2a78d6", "#104281")


def canvas(rows=1, columns=1, size=(7, 4), frame=True, **options):
    figure, axes = plt.subplots(rows, columns, figsize=size, facecolor=SURFACE, **options)
    for axis in np.atleast_1d(axes).flat:
        axis.set_facecolor(SURFACE)
        if not frame:
            axis.axis("off")
            continue
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(RULE)
        axis.tick_params(colors=MUTED, labelsize=8, length=0)
        axis.grid(axis="y", color=RULE, linewidth=0.8)
        axis.set_axisbelow(True)
    return figure, axes


def label(axis, x=None, y=None, title=None):
    if x:
        axis.set_xlabel(x, color=MUTED, fontsize=9)
    if y:
        axis.set_ylabel(y, color=MUTED, fontsize=9)
    if title:
        axis.set_title(title, color=INK, fontsize=10, loc="left")


def reference(axis, value, text):
    axis.axhline(value, color=MUTED, linewidth=1)
    axis.text(1.01, value, text, transform=axis.get_yaxis_transform(), fontsize=8, color=INK, va="center")


def finish(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def gradcam(ink, heat, captions, path, columns=8):
    rows = math.ceil(len(ink) / columns)
    figure, axes = canvas(rows, columns, (columns * 1.5, rows * 1.8), frame=False)
    for axis, drawing, warm, caption in zip(np.atleast_1d(axes).flat, ink, heat, captions):
        wash = np.zeros((*warm.shape, 4))
        wash[..., 0], wash[..., 1], wash[..., 2], wash[..., 3] = 0.92, 0.41, 0.2, 0.7 * warm**2
        axis.imshow(1 - drawing, cmap="gray", vmin=0, vmax=1)
        axis.imshow(wash)
        axis.set_title(caption, fontsize=7, color=INK)
    finish(figure, path)


def reliability(before, after, ece_before, ece_after, path):
    figure, axes = canvas(1, 2, (8, 3.8), sharey=True)
    panels = (
        (axes[0], before, f"raw, ECE {ece_before:.3f}"),
        (axes[1], after, f"temperature scaled, ECE {ece_after:.3f}"),
    )
    for axis, rows, title in panels:
        axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1)
        middles = [(row["low"] + row["high"]) / 2 for row in rows]
        axis.bar(middles, [row["accuracy"] for row in rows], width=0.055, color=BLUE)
        label(axis, x="confidence", title=title)
    label(axes[0], y="accuracy")
    finish(figure, path)


def confused_pairs(pairs, path):
    names = [f"{pair['a']} / {pair['b']}" for pair in pairs][::-1]
    figure, axis = canvas(size=(7, 0.3 * len(pairs) + 1))
    axis.barh(names, [pair["a_as_b"] + pair["b_as_a"] for pair in pairs][::-1], height=0.7, color=BLUE)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=RULE, linewidth=0.8)
    label(axis, x="test drawings swapped between the two classes")
    finish(figure, path)


def curves(history, path):
    epochs = [row["epoch"] for row in history]
    figure, axis = canvas(size=(6.5, 3.8))
    series = [("val_top1", "weights", BLUE)]
    if "ema_val_top1" in history[-1]:
        series.append(("ema_val_top1", "EMA weights", ORANGE))
    for key, name, color in series:
        values = [row[key] for row in history]
        axis.plot(epochs, values, color=color, linewidth=2, marker="o", markersize=4, label=name)
        axis.annotate(
            f"{values[-1]:.3f}",
            (epochs[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="center",
        )
    axis.set_xticks(epochs)
    if len(series) > 1:
        axis.legend(fontsize=8, frameon=False, labelcolor=MUTED, loc="lower right")
    label(axis, x="epoch", y="validation top-1")
    finish(figure, path)


def tradeoff(rows, path):
    rows = [row for row in rows if row["macs"] and row["top1"] and row["onnx_ms"]]
    if not rows:
        return
    figure, axes = canvas(1, 2, (9, 3.8), sharey=True)
    panels = (
        (axes[0], "macs", 1e6, "multiply-adds per drawing (millions)"),
        (axes[1], "onnx_ms", 1, "ONNX latency on one thread (ms)"),
    )
    for axis, key, scale, title in panels:
        for row in rows:
            x = row[key] / scale
            axis.scatter(x, row["top1"], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            axis.annotate(
                row["run"], (x, row["top1"]), fontsize=8, color=INK, xytext=(6, 5), textcoords="offset points"
            )
        axis.margins(x=0.25, y=0.2)
        label(axis, x=title)
    label(axes[0], y="test top-1")
    finish(figure, path)


def robustness(result, path):
    data = result["robustness"]
    kinds = list(data["corruptions"])
    figure, axis = canvas(size=(8, 3.8))
    width = 0.26
    for level, color in enumerate(STEPS):
        scores = [data["corruptions"][kind]["top1"][level] for kind in kinds]
        axis.bar(
            np.arange(len(kinds)) + (level - 1) * (width + 0.02),
            scores,
            width,
            color=color,
            label=f"severity {level + 1}",
        )
    reference(axis, data["clean_top1"], f"clean {data['clean_top1']:.3f}")
    axis.set_xticks(np.arange(len(kinds)), kinds)
    axis.legend(fontsize=8, frameon=False, labelcolor=MUTED, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    label(axis, y="test top-1")
    finish(figure, path)


def countries(result, path):
    rows = result["countries"]
    if not rows:
        return
    scores = np.array([row["top1"] for row in rows])
    spread = 1.96 * np.sqrt(scores * (1 - scores) / np.array([row["n"] for row in rows]))
    figure, axis = canvas(size=(9, 3.8))
    axis.errorbar(
        range(len(rows)), scores, yerr=spread, fmt="o", color=BLUE, ecolor=STEPS[0], markersize=4, linewidth=1
    )
    reference(axis, result["top1"], f"all {result['top1']:.3f}")
    axis.set_xticks(range(len(rows)), [row["country"] for row in rows], rotation=90, fontsize=7)
    label(axis, y="test top-1 with 95% interval")
    finish(figure, path)


def suspects(score, size, width, path, columns=6):
    shown = [item for item in score["top_suspects"] if item.get("drawing")]
    if not shown:
        return
    rows = math.ceil(len(shown) / columns)
    figure, axes = canvas(rows, columns, (columns * 1.6, rows * 1.9), frame=False)
    for axis, item in zip(np.atleast_1d(axes).flat, shown):
        side = 3 * size
        image = np.frombuffer(raster.render(item["drawing"], side, 3 * width), np.uint8)[: side * side]
        axis.imshow(255 - image.reshape(side, side), cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"filed as {item['word']}\nmodel says {item['predicted']} {item['p']:.2f}", fontsize=7, color=INK
        )
    finish(figure, path)

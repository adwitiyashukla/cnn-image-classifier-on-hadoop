import json
from pathlib import Path

from imgcnn import plots


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def collect(runs_dir):
    rows = []
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        config, history = read(folder / "config.json"), read(folder / "metrics.json")
        if not config or not history:
            continue
        t, last = config["train"], history[-1]
        result = read(folder / "eval.json") or {}
        exported = read(folder / "export.json") or {}
        rows.append(
            {
                "run": folder.name,
                "model": t["model"],
                "channels": t["channels"],
                "epochs": len(history),
                "fraction": t["fraction"],
                "augment": t["augment"],
                "mix": "mixup" if t["mixup"] > 0 else "cutmix" if t["cutmix"] > 0 else "none",
                "smoothing": t["label_smoothing"],
                "recognized_only": t["recognized_only"],
                "val_top1": last.get("ema_val_top1", last["val_top1"]),
                "val_top1_raw": last["val_top1"],
                "top1": result.get("top1"),
                "top3": result.get("top3"),
                "map3": result.get("map3"),
                "ece": result.get("calibration", {}).get("ece"),
                "ece_scaled": result.get("calibration", {}).get("ece_scaled"),
                "corrupted_top1": result.get("robustness", {}).get("mean_top1"),
                "minutes": sum(row["seconds"] for row in history) / 60,
                "images_per_second": sum(row["images_per_second"] for row in history) / len(history),
                "params": exported.get("params"),
                "macs": exported.get("exported_macs"),
                "int8_top1": exported.get("top1", {}).get("int8"),
                "onnx_ms": exported.get("latency_one_thread", {}).get("onnx", {}).get("median_ms"),
                "int8_ms": exported.get("latency_one_thread", {}).get("int8", {}).get("median_ms"),
                "history": history,
            }
        )
    return rows


def cell(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def report(cfg, runs_dir, main):
    runs_dir = Path(runs_dir)
    out = runs_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    pipeline = {
        name: read(runs_dir / "pipeline" / f"{name}.json") for name in ("ingest", "profile", "audit", "dataset")
    }
    if pipeline["ingest"]:
        i = pipeline["ingest"]
        print(
            f"ingest   {i['classes']} files, {i['drawings']:,} drawings, {i['bytes'] / 1e9:.2f} GB, {i['seconds']:.0f}s"
        )
    if pipeline["profile"]:
        p = pipeline["profile"]
        print(
            f"profile  {p['drawings']:,} drawings, {len(p['classes'])} classes, {len(p['countries'])} countries, "
            f"{p['recognized'] / max(p['drawings'], 1):.1%} recognized, {p['run']['seconds']:.0f}s"
        )
    if pipeline["audit"]:
        a = pipeline["audit"]
        groups = a["duplicate_groups"] + a["conflict_groups"]
        print(f"audit    {a['dropped']:,} dropped in {groups:,} groups, {a['run']['seconds']:.0f}s")
    if pipeline["dataset"]:
        d = pipeline["dataset"]
        print(
            f"build    train {d['counts']['train']:,}, val {d['counts']['val']:,}, test {d['counts']['test']:,}, "
            f"{d['run']['seconds']:.0f}s"
        )
    rows = collect(runs_dir)
    columns = (
        "run",
        "model",
        "channels",
        "epochs",
        "fraction",
        "val_top1",
        "top1",
        "top3",
        "map3",
        "ece_scaled",
        "corrupted_top1",
        "int8_top1",
        "onnx_ms",
        "minutes",
    )
    widths = [max(len(c), *(len(cell(r[c])) for r in rows)) if rows else len(c) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in rows:
        print("  ".join(cell(row[c], 1 if c == "minutes" else 4).ljust(w) for c, w in zip(columns, widths)))
    if rows:
        plots.tradeoff(rows, out / "tradeoff.png")
    history = read(runs_dir / main / "metrics.json")
    if history:
        plots.curves(history, out / "curves.png")
    result = read(runs_dir / main / "eval.json")
    if result:
        plots.robustness(result, out / "robustness.png")
        plots.countries(result, out / "countries.png")
    score = read(runs_dir / main / "score.json")
    if score:
        plots.suspects(score, cfg.dataset.image_size, cfg.dataset.line_width, out / "suspects.png")
        rec, unrec = score["by_flag"]["recognized"], score["by_flag"]["unrecognized"]
        print(
            f"score    {score['scored']:,} unused drawings, class balanced top-1 {score['class_balanced_top1']:.4f}, "
            f"confident disagreements "
            f"{rec['suspect_rate']:.2%} of recognized and {unrec['suspect_rate']:.2%} of unrecognized drawings"
        )
    summary = {"pipeline": pipeline, "runs": [{k: v for k, v in row.items() if k != "history"} for row in rows]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"figures and summary.json are in {out}")

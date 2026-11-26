import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
SPLITS = ("train", "val", "test")


def parse_counters(text):
    counters, group = {}, None
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if "Counters:" in line]
    if not starts:
        return counters
    for line in lines[starts[-1] + 1 :]:
        if line.startswith("\t\t"):
            name, _, value = line.strip().rpartition("=")
            if value.lstrip("-").isdigit():
                counters.setdefault(group, {})[name.strip()] = int(value)
        elif line.startswith("\t"):
            group = line.strip()
        else:
            break
    return counters


class Hadoop:
    def __init__(self, cfg, fs):
        home = cfg.hadoop.home or os.environ.get("HADOOP_HOME", "")
        if not home:
            raise SystemExit("set HADOOP_HOME or hadoop.home in the config")
        self.home = Path(home)
        self.bin = str(self.home / "bin" / "hadoop")
        jars = sorted((self.home / "share" / "hadoop" / "tools" / "lib").glob("hadoop-streaming-*.jar"))
        if not jars:
            raise SystemExit(f"no hadoop-streaming jar under {self.home}")
        self.jar = str(jars[-1])
        self.libjar = REPO / "hadoop" / "build" / "imgcnn-hadoop.jar"
        self.cfg, self.fs = cfg, fs
        self.logs = Path(cfg.runs.dir) / "hadoop"
        self.logs.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        name,
        inputs,
        output,
        mapper,
        reducer=None,
        reducers=0,
        files=(),
        settings=None,
        key_fields=1,
        split_output=False,
    ):
        if split_output and not self.libjar.exists():
            raise SystemExit(f"{self.libjar} is missing, run hadoop/setup.sh install first")
        if self.fs.status(output):
            self.fs.delete(output)
        conf = {
            "mapreduce.job.name": f"imgcnn-{name}",
            "mapreduce.job.reduces": reducers,
            "mapreduce.input.fileinputformat.split.maxsize": self.cfg.hadoop.split_mb << 20,
            "stream.map.input.ignoreKey": "true",
            "stream.num.map.output.key.fields": key_fields,
            **(settings or {}),
        }
        log = self.logs / f"{name}.log"
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "imgcnn.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                for module in ("__init__.py", "raster.py", "mr.py"):
                    archive.write(PACKAGE / module, f"imgcnn/{module}")
            cmd = [self.bin, "jar", self.jar]
            for key, value in conf.items():
                cmd += ["-D", f"{key}={value}"]
            if split_output:
                cmd += ["-libjars", str(self.libjar)]
            cmd += ["-files", ",".join([str(bundle), *map(str, files)])]
            for path in inputs:
                cmd += ["-input", path]
            cmd += ["-output", output, "-inputformat", "org.apache.hadoop.mapred.lib.CombineTextInputFormat"]
            cmd += ["-mapper", mapper, "-cmdenv", "PYTHONPATH=imgcnn.zip"]
            if reducer:
                cmd += ["-reducer", reducer]
            if split_output:
                cmd += ["-outputformat", "imgcnn.SplitOutputFormat"]
            env = dict(os.environ)
            if split_output:
                env["HADOOP_CLASSPATH"] = str(self.libjar)
            print(f"{name}: running on YARN, log in {log}", flush=True)
            started = time.time()
            with open(log, "w") as handle:
                code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env).returncode
            seconds = round(time.time() - started, 1)
        text = log.read_text(errors="replace")
        if code != 0:
            raise SystemExit(f"hadoop job {name} failed with exit code {code}, see {log}")
        job = next(
            (w for line in text.splitlines() if "Running job:" in line for w in line.split() if w.startswith("job_")),
            "",
        )
        return {"job": job, "seconds": seconds, "counters": parse_counters(text)}


def meta(cfg, name):
    return f"{cfg.hdfs.root}/meta/{name}"


def save_local(cfg, name, value):
    folder = Path(cfg.runs.dir) / "pipeline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(value, indent=1, sort_keys=True))


def rows(fs, path):
    for file, _ in fs.files(path):
        if "/part-" in file:
            for line in fs.lines(file):
                if line:
                    yield line.decode().split("\t")


def counter(run, name):
    return run["counters"].get("imgcnn", {}).get(name, 0)


def command(cfg, job, role, python=None):
    return f"{python or cfg.hadoop.python} -m imgcnn.mr {job} {role}"


def profile(cfg, fs, hadoop):
    root = cfg.hdfs.root
    output = f"{root}/jobs/profile"
    run = hadoop.run(
        "profile",
        [f"{root}/raw"],
        output,
        mapper=command(cfg, "profile", "map"),
        reducer=command(cfg, "profile", "reduce"),
        reducers=cfg.hadoop.reducers,
        key_fields=3,
    )
    classes = defaultdict(lambda: {"drawings": 0, "recognized": 0, "strokes": 0, "points": 0})
    countries = defaultdict(lambda: {"drawings": 0, "recognized": 0})
    for kind, name, recognized, n, strokes, points in rows(fs, output):
        target = classes[name] if kind == "class" else countries[name]
        target["drawings"] += int(n)
        if recognized == "1":
            target["recognized"] += int(n)
        if kind == "class":
            target["strokes"] += int(strokes)
            target["points"] += int(points)
    total = sum(c["drawings"] for c in classes.values())
    recognized = sum(c["recognized"] for c in classes.values())
    summary = {
        "drawings": total,
        "recognized": recognized,
        "malformed": counter(run, "malformed"),
        "classes": dict(sorted(classes.items())),
        "countries": dict(sorted(countries.items(), key=lambda kv: -kv[1]["drawings"])),
        "run": run,
    }
    fs.write_json(meta(cfg, "profile.json"), summary)
    save_local(cfg, "profile.json", summary)
    print(
        f"profile: {total:,} drawings in {len(classes)} classes from {len(countries)} countries, "
        f"{recognized / max(total, 1):.1%} recognized, {summary['malformed']} malformed lines, {run['seconds']}s"
    )
    return summary


def audit(cfg, fs, hadoop):
    root = cfg.hdfs.root
    output = f"{root}/jobs/audit"
    run = hadoop.run(
        "audit",
        [f"{root}/raw"],
        output,
        mapper=command(cfg, "audit", "map"),
        reducer=command(cfg, "audit", "reduce"),
        reducers=cfg.hadoop.reducers,
    )
    dropped, reasons, groups, ids = Counter(), Counter(), [], []
    for row in rows(fs, output):
        if row[0] == "drop":
            ids.append(row[1])
            reasons[row[2]] += 1
            dropped[row[3]] += 1
        elif row[0] == "group":
            groups.append({"size": int(row[2]), "labels": row[3]})
    groups.sort(key=lambda g: (-g["size"], g["labels"]))
    fs.create(meta(cfg, "drop.txt"), "".join(f"{i}\n" for i in sorted(ids)).encode())
    summary = {
        "dropped": len(ids),
        "reasons": dict(reasons),
        "duplicate_groups": counter(run, "duplicate_groups"),
        "conflict_groups": counter(run, "conflict_groups"),
        "per_class": dict(sorted(dropped.items())),
        "largest_groups": groups[:40],
        "run": run,
    }
    fs.write_json(meta(cfg, "audit.json"), summary)
    save_local(cfg, "audit.json", summary)
    print(
        f"audit: {len(ids):,} drawings dropped ({reasons['duplicate']:,} exact copies inside a class, "
        f"{reasons['conflict']:,} copies of a shape that sits under another class), {run['seconds']}s"
    )
    return summary


def thresholds(cfg, profile_summary, audit_summary):
    d = cfg.dataset
    limits, short = {}, []
    for word, stats in profile_summary["classes"].items():
        n = stats["drawings"] - audit_summary["per_class"].get(word, 0)
        if n <= 0:
            continue
        wanted = d.test_per_class + d.val_per_class + d.train_per_class
        if wanted > n:
            short.append(word)
        share = min(1.0, n / wanted) / n
        t_test = d.test_per_class * share
        t_val = t_test + d.val_per_class * share
        limits[word] = [t_test, t_val, min(1.0, wanted * share)]
    return limits, short


def build(cfg, fs, hadoop):
    root = cfg.hdfs.root
    profile_summary = fs.read_json(meta(cfg, "profile.json"))
    audit_summary = fs.read_json(meta(cfg, "audit.json"))
    limits, short = thresholds(cfg, profile_summary, audit_summary)
    if short:
        print(f"build: {len(short)} classes have fewer drawings than the quota, their splits shrink in proportion")
    classes = sorted(limits)
    params = {
        "seed": cfg.dataset.seed,
        "size": cfg.dataset.image_size,
        "width": cfg.dataset.line_width,
        "classes": classes,
        "thresholds": limits,
    }
    with tempfile.TemporaryDirectory() as tmp:
        params_path = Path(tmp) / "params.json"
        params_path.write_text(json.dumps(params))
        drop_path = Path(tmp) / "drop.txt"
        drop_path.write_bytes(fs.read_bytes(meta(cfg, "drop.txt")))
        run = hadoop.run(
            "build",
            [f"{root}/raw"],
            f"{root}/shards",
            mapper=command(cfg, "build", "map"),
            reducer=command(cfg, "build", "reduce"),
            reducers=cfg.hadoop.shards,
            files=[params_path, drop_path],
            split_output=True,
        )
    counts, recognized, sizes = {}, {}, {}
    for split in SPLITS:
        folder = f"{root}/shards/{split}"
        files = fs.files(folder) if fs.status(folder) else []
        counts[split] = counter(run, f"selected_{split}")
        recognized[split] = counter(run, f"recognized_{split}")
        sizes[split] = {"files": len(files), "bytes": sum(size for _, size in files)}
    empty = [split for split in SPLITS if counts[split] == 0 or sizes[split]["files"] == 0]
    if empty:
        raise SystemExit(f"build left these splits empty: {', '.join(empty)}, see {hadoop.logs / 'build.log'}")
    dataset = {**params, "counts": counts, "recognized": recognized, "shards": sizes, "run": run}
    fs.write_json(meta(cfg, "dataset.json"), dataset)
    save_local(cfg, "dataset.json", {k: v for k, v in dataset.items() if k != "thresholds"})
    print(
        f"build: train {counts['train']:,}, val {counts['val']:,}, test {counts['test']:,} images "
        f"({sum(s['bytes'] for s in sizes.values()) / 1e9:.2f} GB in {sum(s['files'] for s in sizes.values())} files), "
        f"{run['seconds']}s"
    )
    return dataset


def fetch_drawings(fs, root, wanted):
    found = {}
    by_class = defaultdict(set)
    for word, key_id in wanted:
        by_class[word].add(key_id)
    for word, ids in by_class.items():
        for line in fs.lines(f"{root}/raw/{word.replace(' ', '_')}.ndjson"):
            if any(key_id.encode() in line for key_id in ids):
                item = json.loads(line)
                if item["key_id"] in ids:
                    found[item["key_id"]] = item["drawing"]
            if len(ids & found.keys()) == len(ids):
                break
    return found


def score(cfg, fs, hadoop, run_dir, channels, examples=24):
    root = cfg.hdfs.root
    model = Path(run_dir) / "model.onnx"
    if not model.exists():
        raise SystemExit(f"{model} is missing, run the export command first")
    dataset = fs.read_json(meta(cfg, "dataset.json"))
    params = {
        "seed": dataset["seed"],
        "size": dataset["size"],
        "width": dataset["width"],
        "classes": dataset["classes"],
        "thresholds": dataset["thresholds"],
        "channels": channels,
        "fraction": cfg.score.fraction,
        "threshold": cfg.score.threshold,
        "top_k": cfg.score.top_k,
        "batch": cfg.score.batch,
        "threads": cfg.score.threads,
    }
    output = f"{root}/jobs/score"
    with tempfile.TemporaryDirectory() as tmp:
        params_path = Path(tmp) / "params.json"
        params_path.write_text(json.dumps(params))
        drop_path = Path(tmp) / "drop.txt"
        drop_path.write_bytes(fs.read_bytes(meta(cfg, "drop.txt")))
        run = hadoop.run(
            "score",
            [f"{root}/raw"],
            output,
            mapper=command(cfg, "score", "map", sys.executable),
            reducer=command(cfg, "score", "reduce", sys.executable),
            reducers=cfg.hadoop.reducers,
            files=[params_path, drop_path, f"{model.resolve()}#model.onnx"],
            settings={"mapreduce.map.memory.mb": cfg.hadoop.score_map_mb},
        )
    per_class, suspects = {}, []
    totals = {flag: [0, 0, 0, 0] for flag in ("unrecognized", "recognized")}
    for row in rows(fs, output):
        if row[0] == "class":
            word, cells = row[1], row[2:]
            entry = {}
            for i, flag in enumerate(("unrecognized", "recognized")):
                n, hit, top3, suspect, _ = cells[5 * i : 5 * i + 5]
                entry[flag] = {"n": int(n), "top1": int(hit), "top3": int(top3), "suspect": int(suspect)}
                for j, value in enumerate((n, hit, top3, suspect)):
                    totals[flag][j] += int(value)
            per_class[word] = entry
        elif row[0] == "suspect":
            suspects.append(
                {
                    "word": row[1],
                    "key_id": row[2],
                    "predicted": row[3],
                    "p": float(row[4]),
                    "recognized": row[5] == "1",
                    "country": row[6],
                }
            )
    suspects.sort(key=lambda s: -s["p"])
    shown = suspects[:examples]
    strokes = fetch_drawings(fs, root, [(s["word"], s["key_id"]) for s in shown])
    for s in shown:
        s["drawing"] = strokes.get(s["key_id"])
    scored = sum(t[0] for t in totals.values())
    recalls = []
    for entry in per_class.values():
        count = entry["recognized"]["n"] + entry["unrecognized"]["n"]
        if count:
            recalls.append((entry["recognized"]["top1"] + entry["unrecognized"]["top1"]) / count)
    summary = {
        "scored": scored,
        "fraction": cfg.score.fraction,
        "threshold": cfg.score.threshold,
        "top1": sum(t[1] for t in totals.values()) / max(scored, 1),
        "class_balanced_top1": sum(recalls) / max(len(recalls), 1),
        "top3": sum(t[2] for t in totals.values()) / max(scored, 1),
        "suspect_rate": sum(t[3] for t in totals.values()) / max(scored, 1),
        "by_flag": {
            flag: {
                "n": t[0],
                "top1": t[1] / max(t[0], 1),
                "top3": t[2] / max(t[0], 1),
                "suspect_rate": t[3] / max(t[0], 1),
            }
            for flag, t in totals.items()
        },
        "per_class": per_class,
        "top_suspects": shown,
        "run": run,
    }
    (Path(run_dir) / "score.json").write_text(json.dumps(summary, indent=1))
    rec, unrec = summary["by_flag"]["recognized"], summary["by_flag"]["unrecognized"]
    print(
        f"score: {scored:,} unused drawings scored on YARN in {run['seconds']}s, top-1 {summary['top1']:.2%} "
        f"({summary['class_balanced_top1']:.2%} class balanced), top-3 {summary['top3']:.2%}; confident "
        f"disagreements {rec['suspect_rate']:.2%} on recognized vs {unrec['suspect_rate']:.2%} on unrecognized drawings"
    )
    return summary

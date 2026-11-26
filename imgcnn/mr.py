import hashlib
import heapq
import io
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import groupby

from imgcnn import raster

counters = Counter()


def unit(text, salt):
    digest = hashlib.blake2b(f"{salt}:{text}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 18446744073709551616.0


def split_of(key_id, word, params):
    limits = params["thresholds"].get(word)
    if limits is None:
        return None
    u = unit(key_id, params["seed"])
    if u < limits[0]:
        return "test"
    if u < limits[1]:
        return "val"
    if u < limits[2]:
        return "train"
    return None


def drawings(lines):
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            counters["malformed"] += 1
            continue
        if not isinstance(item, dict) or not {"word", "key_id", "drawing"} <= item.keys():
            counters["malformed"] += 1
            continue
        counters["drawings"] += 1
        yield item


def first_field(line):
    return line.split("\t", 1)[0]


def profile_map(lines, out, params, drop):
    stats = defaultdict(lambda: [0, 0, 0])
    for item in drawings(lines):
        recognized = 1 if item.get("recognized") else 0
        strokes = item["drawing"]
        row = stats[("class", item["word"], recognized)]
        row[0] += 1
        row[1] += len(strokes)
        row[2] += sum(len(xs) for xs, _ in strokes)
        stats[("country", item.get("countrycode") or "ZZ", recognized)][0] += 1
    for (kind, name, recognized), (n, strokes, points) in sorted(stats.items()):
        out(f"{kind}\t{name}\t{recognized}\t{n}\t{strokes}\t{points}\n")


def profile_reduce(lines, out, params, drop):
    rows = (line.rstrip("\n").split("\t") for line in lines)
    for key, group in groupby(rows, key=lambda r: tuple(r[:3])):
        total = [0, 0, 0]
        for row in group:
            for i in range(3):
                total[i] += int(row[3 + i])
        out("\t".join(key) + f"\t{total[0]}\t{total[1]}\t{total[2]}\n")


def audit_map(lines, out, params, drop):
    for item in drawings(lines):
        shape = json.dumps(item["drawing"], separators=(",", ":")).encode()
        digest = hashlib.blake2b(shape, digest_size=8).hexdigest()
        out(f"{digest}\t{item['word']}\t{item['key_id']}\n")


def audit_reduce(lines, out, params, drop):
    for digest, group in groupby(lines, key=first_field):
        members = [line.rstrip("\n").split("\t")[1:3] for line in group]
        if len(members) < 2:
            continue
        words = Counter(word for word, _ in members)
        major, count = words.most_common(1)[0]
        keep = None
        if 2 * count > len(members):
            keep = min(key_id for word, key_id in members if word == major)
        for word, key_id in members:
            if key_id == keep:
                continue
            reason = "duplicate" if keep is not None and word == major else "conflict"
            out(f"drop\t{key_id}\t{reason}\t{word}\n")
            counters[f"{reason}_dropped"] += 1
        counters["duplicate_groups" if len(words) == 1 else "conflict_groups"] += 1
        if len(members) >= 5 or len(words) > 1:
            labels = ",".join(f"{word}:{n}" for word, n in sorted(words.items(), key=lambda kv: (-kv[1], kv[0])))
            out(f"group\t{digest}\t{len(members)}\t{labels}\n")


def build_map(lines, out, params, drop):
    labels = {word: i for i, word in enumerate(params["classes"])}
    size, width, seed = params["size"], params["width"], params["seed"]
    for item in drawings(lines):
        key_id, word = item["key_id"], item["word"]
        if key_id in drop:
            counters["dropped"] += 1
            continue
        split = split_of(key_id, word, params)
        if split is None:
            continue
        image = raster.encode(raster.render(item["drawing"], size, width))
        order = hashlib.blake2b(f"order:{seed}:{key_id}".encode(), digest_size=8).hexdigest()
        country = item.get("countrycode") or "ZZ"
        recognized = 1 if item.get("recognized") else 0
        out(f"{order}\t{split}\t{labels[word]}\t{key_id}\t{country}\t{recognized}\t{image}\n")
        counters[f"selected_{split}"] += 1
        counters[f"recognized_{split}"] += recognized


def build_reduce(lines, out, params, drop):
    for line in lines:
        out(line.split("\t", 1)[1])


def score_map(lines, out, params, drop):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = params["threads"]
    options.inter_op_num_threads = 1
    session = ort.InferenceSession("model.onnx", options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    size, width, channels = params["size"], params["width"], params["channels"]
    salt = f"score:{params['seed']}"
    labels = {word: i for i, word in enumerate(params["classes"])}
    images, meta = [], []

    def flush():
        batch = np.frombuffer(b"".join(images), dtype=np.uint8).reshape(-1, 3, size, size)
        x = batch[:, :channels].astype(np.float32) / 255.0
        logits = session.run(None, {name: x})[0].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        top = np.argsort(-probs, axis=1)[:, :3]
        for (word, key_id, country, recognized), p, t in zip(meta, probs, top):
            true = labels[word]
            out(f"{word}\t{key_id}\t{country}\t{recognized}\t{p[true]:.4f}\t{t[0]}\t{p[t[0]]:.4f}\t{t[1]}\t{t[2]}\n")
        counters["scored"] += len(meta)
        images.clear()
        meta.clear()

    for item in drawings(lines):
        key_id, word = item["key_id"], item["word"]
        if key_id in drop or word not in labels:
            continue
        if split_of(key_id, word, params) is not None:
            counters["skipped_dataset"] += 1
            continue
        if unit(key_id, salt) >= params["fraction"]:
            continue
        images.append(raster.render(item["drawing"], size, width))
        meta.append((word, key_id, item.get("countrycode") or "ZZ", 1 if item.get("recognized") else 0))
        if len(images) == params["batch"]:
            flush()
    if images:
        flush()


def score_reduce(lines, out, params, drop):
    classes = params["classes"]
    threshold, top_k = params["threshold"], params["top_k"]
    for word, group in groupby(lines, key=first_field):
        stats = {0: [0, 0, 0, 0, 0.0], 1: [0, 0, 0, 0, 0.0]}
        heap = []
        for line in group:
            _, key_id, country, recognized, p_true, pred, p_pred, second, third = line.rstrip("\n").split("\t")
            row = stats[int(recognized)]
            guess = classes[int(pred)]
            p_pred = float(p_pred)
            hit = guess == word
            in_top3 = hit or classes[int(second)] == word or classes[int(third)] == word
            suspect = not hit and p_pred >= threshold
            row[0] += 1
            row[1] += hit
            row[2] += in_top3
            row[3] += suspect
            row[4] += float(p_true)
            if suspect:
                entry = (p_pred, key_id, guess, recognized, country)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                else:
                    heapq.heappushpop(heap, entry)
        cells = []
        for flag in (0, 1):
            n, hit, top3, suspect, p_sum = stats[flag]
            cells += [str(n), str(hit), str(top3), str(suspect), f"{p_sum:.4f}"]
        out(f"class\t{word}\t" + "\t".join(cells) + "\n")
        for p_pred, key_id, guess, recognized, country in sorted(heap, reverse=True):
            out(f"suspect\t{word}\t{key_id}\t{guess}\t{p_pred:.4f}\t{recognized}\t{country}\n")


JOBS = {
    ("profile", "map"): profile_map,
    ("profile", "reduce"): profile_reduce,
    ("audit", "map"): audit_map,
    ("audit", "reduce"): audit_reduce,
    ("build", "map"): build_map,
    ("build", "reduce"): build_reduce,
    ("score", "map"): score_map,
    ("score", "reduce"): score_reduce,
}


def load_side_files():
    params = {}
    if os.path.exists("params.json"):
        with open("params.json") as f:
            params = json.load(f)
    drop = set()
    if os.path.exists("drop.txt"):
        with open("drop.txt") as f:
            drop = {line.strip() for line in f if line.strip()}
    return params, drop


def main(argv):
    job = JOBS[(argv[1], argv[2])]
    params, drop = load_side_files()
    source = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    sink = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    job(source, sink.write, params, drop)
    sink.flush()
    for name, value in sorted(counters.items()):
        sys.stderr.write(f"reporter:counter:imgcnn,{name},{value}\n")
    sys.stderr.flush()


if __name__ == "__main__":
    main(sys.argv)

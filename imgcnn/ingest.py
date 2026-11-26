import base64
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

from imgcnn.hdfs import HDFS


def hdfs_name(name):
    return name.replace(" ", "_")


def load_manifest(cfg):
    source = cfg.source.manifest
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        data = response.json()
    else:
        data = json.loads(Path(source).read_text())
    items = []
    for item in data["items"]:
        name = item["name"].rsplit("/", 1)[-1].removesuffix(".ndjson")
        items.append({"name": name, "size": int(item["size"]), "md5": item.get("md5Hash", "")})
    items.sort(key=lambda i: i["name"])
    return items[: cfg.source.classes] if cfg.source.classes else items


def copy_one(cfg, item):
    fs = HDFS(cfg.hdfs.url, cfg.hdfs.user)
    root = cfg.hdfs.root
    target = f"{root}/raw/{hdfs_name(item['name'])}.ndjson"
    marker = f"{root}/meta/ingest/{hdfs_name(item['name'])}.json"
    cap = cfg.source.max_mb << 20
    if fs.status(marker) and fs.status(target):
        done = fs.read_json(marker)
        if done["cap"] == cap:
            return {**done, "skipped": True}
    url = f"{cfg.source.url.rstrip('/')}/{quote(item['name'])}.ndjson"
    started = time.time()
    digest = hashlib.md5()
    stats = {"bytes": 0, "drawings": 0}
    temporary = f"{root}/incoming/{hdfs_name(item['name'])}.ndjson"
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        def body():
            taken, carry = 0, b""
            for chunk in response.iter_content(1 << 20):
                digest.update(chunk)
                if cap:
                    chunk = chunk[: cap - taken]
                    taken += len(chunk)
                data = carry + chunk
                cut = data.rfind(b"\n") + 1
                carry = data[cut:]
                if cut:
                    stats["bytes"] += cut
                    stats["drawings"] += data.count(b"\n", 0, cut)
                    yield data[:cut]
                if cap and taken >= cap:
                    return
            if carry:
                stats["bytes"] += len(carry) + 1
                stats["drawings"] += 1
                yield carry + b"\n"

        fs.create(temporary, body())
    checked = not cap and bool(item["md5"])
    if checked and base64.b64encode(digest.digest()).decode() != item["md5"]:
        fs.delete(temporary)
        raise OSError(f"md5 mismatch for {item['name']}")
    if fs.status(target):
        fs.delete(target)
    fs.rename(temporary, target)
    record = {
        "name": item["name"],
        "cap": cap,
        "bytes": stats["bytes"],
        "drawings": stats["drawings"],
        "md5_checked": checked,
        "seconds": round(time.time() - started, 1),
    }
    fs.write_json(marker, record)
    return record


def copy_with_retries(cfg, item):
    for attempt in range(cfg.source.retries + 1):
        try:
            return copy_one(cfg, item)
        except (requests.RequestException, OSError) as error:
            if attempt == cfg.source.retries:
                raise
            print(f"retrying {item['name']} after: {error}", flush=True)
            time.sleep(2**attempt)


def ingest(cfg):
    items = load_manifest(cfg)
    fs = HDFS(cfg.hdfs.url, cfg.hdfs.user)
    root = cfg.hdfs.root
    for folder in ("raw", "incoming", "meta/ingest"):
        fs.mkdirs(f"{root}/{folder}")
    started = time.time()
    results = []
    with ThreadPoolExecutor(cfg.source.workers) as pool:
        futures = [pool.submit(copy_with_retries, cfg, item) for item in items]
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            note = "already in HDFS" if r.get("skipped") else f"{r['seconds']}s"
            print(
                f"[{i}/{len(items)}] {r['name']}: {r['bytes'] / 1e6:.1f} MB, {r['drawings']:,} drawings, {note}",
                flush=True,
            )
    seconds = round(time.time() - started, 1)
    copied = [r for r in results if not r.get("skipped")]
    summary = {
        "classes": len(results),
        "bytes": sum(r["bytes"] for r in results),
        "drawings": sum(r["drawings"] for r in results),
        "md5_checked": sum(r["md5_checked"] for r in results),
        "seconds": seconds,
        "copied_now": len(copied),
        "mb_per_second": round(sum(r["bytes"] for r in copied) / 1e6 / max(seconds, 1e-9), 1),
    }
    fs.write_json(f"{root}/meta/ingest.json", summary)
    folder = Path(cfg.runs.dir) / "pipeline"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ingest.json").write_text(json.dumps(summary, indent=1))
    print(
        f"ingest: {summary['classes']} classes, {summary['drawings']:,} drawings, {summary['bytes'] / 1e9:.2f} GB "
        f"in HDFS, {summary['md5_checked']} files md5 checked, {seconds}s"
    )
    return summary

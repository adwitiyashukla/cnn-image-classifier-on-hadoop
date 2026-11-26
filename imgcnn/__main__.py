import argparse
import json
import sys
from pathlib import Path

from imgcnn import config as configuration

COMMANDS = (
    "ingest",
    "profile",
    "audit",
    "build",
    "bench",
    "train",
    "evaluate",
    "export",
    "space",
    "score",
    "scratch",
    "report",
)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m imgcnn")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--name", default="")
    args = parser.parse_args(argv)
    try:
        cfg, raw = configuration.load(args.config, args.set)
    except (configuration.ConfigError, FileNotFoundError) as error:
        raise SystemExit(f"config error: {error}")
    run_dir = Path(cfg.runs.dir) / (args.name or cfg.train.model)

    if args.command == "ingest":
        from imgcnn.ingest import ingest

        ingest(cfg)
        return

    from imgcnn.hdfs import HDFS

    fs = HDFS(cfg.hdfs.url, cfg.hdfs.user)

    if args.command in ("profile", "audit", "build"):
        from imgcnn import jobs

        getattr(jobs, args.command)(cfg, fs, jobs.Hadoop(cfg, fs))
    elif args.command == "score":
        from imgcnn import jobs

        saved = json.loads((run_dir / "config.json").read_text())
        jobs.score(cfg, fs, jobs.Hadoop(cfg, fs), run_dir, saved["train"]["channels"])
    elif args.command == "bench":
        from imgcnn.train import bench

        bench(cfg)
    elif args.command == "train":
        from imgcnn.train import train

        train(cfg, raw, fs, run_dir)
    elif args.command == "evaluate":
        from imgcnn.evaluate import evaluate

        evaluate(cfg, fs, run_dir)
    elif args.command == "export":
        from imgcnn.export import export

        export(cfg, fs, run_dir)
    elif args.command == "space":
        from imgcnn.export import bundle

        bundle(run_dir, Path(cfg.runs.dir) / "space")
    elif args.command == "scratch":
        from imgcnn.scratch import scratch

        scratch(cfg, fs, Path(cfg.runs.dir) / "scratch")
    elif args.command == "report":
        from imgcnn.report import report

        report(cfg, Path(cfg.runs.dir), args.name or cfg.train.model)


if __name__ == "__main__":
    main(sys.argv[1:])

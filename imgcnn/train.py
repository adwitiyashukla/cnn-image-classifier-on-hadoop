import copy
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from imgcnn import config as configuration
from imgcnn import data, metrics, models

RESUME_SECTIONS = ("dataset", "train", "resnet", "mobilenet", "convnext")
RUNTIME_KEYS = ("threads", "workers", "log_every", "checkpoint_every")
LAYOUT = torch.channels_last


def pick_device(cfg):
    if cfg.train.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.train.device)


def distributed(device):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world


def learning_rate(cfg, step, total, warmup):
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(cfg, model):
    decay = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": plain, "weight_decay": 0.0}]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)


def smooth(labels, classes, smoothing):
    targets = torch.full((len(labels), classes), smoothing / classes, device=labels.device)
    targets.scatter_(1, labels[:, None], 1 - smoothing + smoothing / classes)
    return targets


def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, 1)).sum(1).mean()


def augment(x, cfg, generator):
    n = x.shape[0]
    angle = (torch.rand(n, generator=generator) * 2 - 1) * math.radians(cfg.rotate)
    zoom = 1 + (torch.rand(n, generator=generator) * 2 - 1) * cfg.scale
    shift = (torch.rand(n, 2, generator=generator) * 2 - 1) * 2 * cfg.shift
    side = torch.ones(n)
    if cfg.flip:
        side = torch.where(torch.rand(n, generator=generator) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack(
        [torch.stack([cos * side, -sin, shift[:, 0]], 1), torch.stack([sin * side, cos, shift[:, 1]], 1)], 1
    )
    grid = F.affine_grid(theta.to(x.device), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def mix(x, targets, cfg, generator, rng):
    alpha = cfg.mixup or cfg.cutmix
    if alpha <= 0:
        return x, targets
    lam = rng.betavariate(alpha, alpha)
    other = torch.randperm(x.shape[0], generator=generator).to(x.device)
    if cfg.mixup > 0:
        x = lam * x + (1 - lam) * x[other]
    else:
        size = x.shape[-1]
        side = int(round(size * math.sqrt(1 - lam)))
        top, left = rng.randint(0, size - side), rng.randint(0, size - side)
        x = x.clone()
        x[:, :, top : top + side, left : left + side] = x[other][:, :, top : top + side, left : left + side]
        lam = 1 - side * side / (size * size)
    return x, lam * targets + (1 - lam) * targets[other]


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        for name, value in self.shadow.state_dict().items():
            if value.dtype.is_floating_point:
                value.lerp_(source[name], 1 - decay)
            else:
                value.copy_(source[name])


class Trainer:
    def __init__(self, cfg, classes, total_steps, steps_per_epoch, device, rank=0, world=1, name=None, layout=LAYOUT):
        self.cfg = cfg.train
        self.layout = layout
        self.device = device
        self.rank = rank
        self.classes = classes
        self.total_steps = total_steps
        self.warmup = int(self.cfg.warmup_epochs * steps_per_epoch)
        torch.manual_seed(self.cfg.seed)
        self.model = models.build(cfg, classes, name).to(device, memory_format=layout)
        self.ema = EMA(self.model, self.cfg.ema) if self.cfg.ema > 0 and rank == 0 else None
        self.net = self.model
        if world > 1:
            self.net = torch.nn.parallel.DistributedDataParallel(self.model)
        self.optimizer = make_optimizer(self.cfg, self.model)
        self.amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp)
        self.generator = torch.Generator().manual_seed(self.cfg.seed * 7919 + rank)
        self.rng = random.Random(self.cfg.seed * 7919 + rank)
        self.step = 0

    def train_step(self, images, labels):
        cfg = self.cfg
        rate = learning_rate(cfg, self.step, self.total_steps, self.warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        x = images.to(self.device).float().div_(255)
        labels = labels.to(self.device)
        if cfg.augment:
            x = augment(x, cfg, self.generator)
        targets = smooth(labels, self.classes, cfg.label_smoothing)
        x, targets = mix(x, targets, cfg, self.generator, self.rng)
        self.model.train()
        with torch.autocast(self.device.type, enabled=self.amp):
            logits = self.net(x.contiguous(memory_format=self.layout))
        loss = soft_cross_entropy(logits.float(), targets)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if cfg.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema:
            self.ema.update(self.model)
        self.step += 1
        return loss.item(), (logits.argmax(1) == labels).sum().item()

    def best(self):
        return self.ema.shadow if self.ema else self.model

    def state(self):
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.shadow.state_dict() if self.ema else None,
            "ema_updates": self.ema.updates if self.ema else 0,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "generator": self.generator.get_state(),
            "rng": self.rng.getstate(),
            "step": self.step,
        }

    def load(self, state):
        self.model.load_state_dict(state["model"])
        if self.ema and state["ema"] is not None:
            self.ema.shadow.load_state_dict(state["ema"])
            self.ema.updates = state["ema_updates"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        if self.rank == 0:
            self.generator.set_state(state["generator"])
            self.rng.setstate(state["rng"])
        else:
            seed = self.cfg.seed * 7919 + self.rank + 104729 * state["step"]
            self.generator.manual_seed(seed)
            self.rng.seed(seed)
        self.step = state["step"]


@torch.no_grad()
def predict(model, images, device, batch):
    model.eval()
    outputs = []
    for start in range(0, len(images), batch):
        x = images[start : start + batch].to(device, memory_format=LAYOUT).float().div_(255)
        outputs.append(model(x).float().cpu())
    return torch.cat(outputs)


def save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def resume_key(raw, world, total):
    config = {section: dict(raw[section]) for section in RESUME_SECTIONS}
    for name in RUNTIME_KEYS:
        del config["train"][name]
    return {"config": config, "world": world, "steps": total}


def load_model(run_dir, device=torch.device("cpu")):
    path = Path(run_dir) / "model.pt"
    if not path.exists():
        raise SystemExit(f"{path} is missing, run the train command first")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = configuration.namespace(saved["config"])
    model = models.build(cfg, len(saved["classes"])).to(device, memory_format=LAYOUT)
    model.load_state_dict(saved["state"])
    return model.eval(), saved


def train(cfg, raw, fs, run_dir):
    device = pick_device(cfg)
    rank, world = distributed(device)
    if device.type == "cuda" and world > 1:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
    chief = rank == 0
    threads = cfg.train.threads or (torch.get_num_threads() if world == 1 else os.cpu_count() or 1)
    torch.set_num_threads(max(1, threads // world))
    meta = data.dataset_meta(cfg, fs)
    classes, size, channels = meta["classes"], meta["size"], cfg.train.channels
    files = data.shard_files(cfg, fs, "train")
    if not files:
        raise SystemExit("no train shards in HDFS, run the build command first")
    if len(files) < world:
        raise SystemExit(f"{len(files)} train shards cannot feed {world} processes, use fewer processes")
    used = min(len(files), max(world, math.ceil(len(files) * cfg.train.fraction)))
    available = meta["recognized"]["train"] if cfg.train.recognized_only else meta["counts"]["train"]
    samples = available * used // len(files)
    files = files[:used]
    batch = cfg.train.batch_size
    steps_per_epoch = max(1, samples // (batch * world))
    total = steps_per_epoch * cfg.train.epochs
    trainer = Trainer(cfg, len(classes), total, steps_per_epoch, device, rank, world)
    run_dir = Path(run_dir)
    last = run_dir / "last.pt"
    key = resume_key(raw, world, total)
    history = []
    progress = {"loss_sum": 0.0, "hits": 0, "seen": 0, "seconds": 0.0}
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["key"] != key:
            raise SystemExit(f"{last} was made with a different config, use --name with a new name or delete it")
        trainer.load(saved["state"])
        history, progress = saved["history"], saved["progress"]
        if trainer.step >= total and (run_dir / "model.pt").exists():
            if chief:
                print(f"{run_dir} is already trained, nothing to do")
            return
        if chief:
            print(f"resuming from step {trainer.step} of {total}", flush=True)
    if chief:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(raw, indent=1))
        stats = models.complexity(trainer.model, channels, size)
        val = data.load_split(cfg, fs, "val", size, channels, cfg.train.val_samples)
        print(
            f"{cfg.train.model}: {stats['params'] / 1e6:.2f}M params, {stats['macs'] / 1e6:.1f}M MACs, "
            f"{samples:,} train images in {used} shards, {len(val['labels']):,} val images, "
            f"{steps_per_epoch} steps per epoch, {world} process(es) x {torch.get_num_threads()} threads on {device}",
            flush=True,
        )

    def checkpoint():
        running = {"loss_sum": loss_sum, "hits": hits, "seen": seen, "seconds": time.time() - began}
        save({"state": trainer.state(), "history": history, "progress": running, "key": key}, last)

    batches = data.train_batches(cfg, files, size, f"{cfg.train.seed}:{trainer.step}", rank, world)
    loss_sum, hits, seen = progress["loss_sum"], progress["hits"], progress["seen"]
    began = time.time() - progress["seconds"]
    window, window_seen = time.time(), 0
    for step in range(trainer.step, total):
        images, labels = next(batches)
        loss, correct = trainer.train_step(images, labels)
        loss_sum += loss * len(labels)
        hits += correct
        seen += len(labels)
        window_seen += len(labels)
        if not chief:
            continue
        number = step + 1
        epoch = step // steps_per_epoch + 1
        if number % cfg.train.log_every == 0:
            now = time.time()
            rate = window_seen * world / (now - window)
            left = (total - number) * batch * world / rate
            print(
                f"epoch {epoch}/{cfg.train.epochs} step {number}/{total} loss {loss_sum / seen:.3f} "
                f"top1 {hits / seen:.3f} {rate:.0f} img/s, about {left / 60:.0f} min left",
                flush=True,
            )
            window, window_seen = now, 0
        if number % steps_per_epoch == 0:
            row = {
                "epoch": epoch,
                "step": number,
                "train_loss": loss_sum / seen,
                "train_top1": hits / seen,
                "images_per_second": seen * world / (time.time() - began),
                "seconds": time.time() - began,
            }
            scores = metrics.summarize(
                predict(trainer.model, val["images"], device, cfg.eval.batch_size), val["labels"]
            )
            row.update({f"val_{k}": v for k, v in scores.items()})
            if trainer.ema:
                scores = metrics.summarize(
                    predict(trainer.best(), val["images"], device, cfg.eval.batch_size), val["labels"]
                )
                row.update({f"ema_val_{k}": v for k, v in scores.items()})
            history.append(row)
            shown = row.get("ema_val_top1", row["val_top1"])
            print(
                f"epoch {epoch} done: train loss {row['train_loss']:.3f}, val top-1 {row['val_top1']:.4f}, "
                f"ema val top-1 {shown:.4f}, {row['images_per_second']:.0f} img/s",
                flush=True,
            )
            (run_dir / "metrics.json").write_text(json.dumps(history, indent=1))
            began = window = time.time()
            loss_sum, hits, seen, window_seen = 0.0, 0, 0, 0
            checkpoint()
        elif number % cfg.train.checkpoint_every == 0:
            checkpoint()
    del batches
    if world > 1:
        torch.distributed.destroy_process_group()
    if chief:
        save(
            {
                "state": trainer.best().state_dict(),
                "classes": classes,
                "size": size,
                "width": meta["width"],
                "channels": channels,
                "model": cfg.train.model,
                "config": raw,
            },
            run_dir / "model.pt",
        )
        spent = sum(row["seconds"] for row in history)
        final = history[-1] if history else {}
        print(
            f"saved {run_dir / 'model.pt'}: val top-1 {final.get('ema_val_top1', final.get('val_top1', 0)):.4f}, "
            f"training time {spent / 60:.1f} min",
            flush=True,
        )


def bench(cfg):
    device = pick_device(cfg)
    size, channels, batch = cfg.dataset.image_size, cfg.train.channels, cfg.train.batch_size
    classes = cfg.source.classes or 345
    epoch_images = classes * cfg.dataset.train_per_class
    images = torch.randint(0, 256, (batch, channels, size, size), dtype=torch.uint8)
    labels = torch.randint(0, classes, (batch,))
    default = cfg.train.threads or torch.get_num_threads()
    main = cfg.train.model

    def rate(trainer, threads, seconds):
        torch.set_num_threads(threads)
        trainer.train_step(images, labels)
        started, steps = time.time(), 0
        while steps == 0 or time.time() - started < seconds:
            trainer.train_step(images, labels)
            steps += 1
        return steps * batch / (time.time() - started)

    families = {name: Trainer(cfg, classes, 1000, 100, device, name=name) for name in models.FAMILIES}
    plain = Trainer(cfg, classes, 1000, 100, device, name=main, layout=torch.contiguous_format)
    cases = [(("model", name), trainer, default) for name, trainer in families.items()]
    cases.append((("layout", "channels first"), plain, default))
    if device.type == "cpu":
        cores = os.cpu_count() or 1
        counts = sorted({t for t in (1, 2, 4, 6, 8, 10, 12, 16, cores) if t <= cores})
        cases += [(("threads", count), families[main], count) for count in counts]
    b = cfg.bench
    print(
        f"training throughput at batch {batch}, {size}x{size}x{channels}, {default} threads, {device}: "
        f"{b.warmup_seconds:.0f}s warm-up, then the median of {b.rounds} rounds of {b.seconds:.0f}s per case",
        flush=True,
    )
    rate(families[main], default, b.warmup_seconds)
    samples = {key: [] for key, _, _ in cases}
    for _ in range(b.rounds):
        for key, trainer, threads in cases:
            samples[key].append(rate(trainer, threads, b.seconds))
    torch.set_num_threads(default)
    speed = {key: sorted(values)[len(values) // 2] for key, values in samples.items()}
    rows = []
    for name, trainer in families.items():
        stats = models.complexity(trainer.model, channels, size)
        minutes = epoch_images / speed["model", name] / 60
        rows.append({"model": name, **stats, "images_per_second": speed["model", name], "epoch_minutes": minutes})
        print(
            f"{name:10s} {stats['params'] / 1e6:5.2f}M params {stats['macs'] / 1e6:6.1f}M MACs "
            f"{speed['model', name]:7.0f} img/s {minutes:6.1f} min per epoch"
        )
    first, last = speed["layout", "channels first"], speed["model", main]
    print(f"{main} with channels first memory layout {first:.0f} img/s, channels last {last:.0f} img/s")
    sweep = [{"threads": key[1], "images_per_second": value} for key, value in speed.items() if key[0] == "threads"]
    if sweep:
        print(f"thread sweep for {main}")
    for row in sweep:
        print(f"{row['threads']:3d} threads {row['images_per_second']:7.0f} img/s")
    folder = Path(cfg.runs.dir)
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "batch": batch,
        "threads": default,
        "rounds": b.rounds,
        "seconds": b.seconds,
        "models": rows,
        "channels_first_images_per_second": first,
        "thread_sweep": sweep,
    }
    (folder / "bench.json").write_text(json.dumps(result, indent=1))

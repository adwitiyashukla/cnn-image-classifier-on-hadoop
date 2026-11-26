import os
import socket

import torch
import torch.multiprocessing as mp

from conftest import load_small
from imgcnn import train

STEPS = 5
CLASSES = 6
SHARED = ("convnext.drop_path=0", "train.augment=false", "train.ema=0", "train.threads=1")


def batches():
    generator = torch.Generator().manual_seed(11)
    for _ in range(STEPS):
        images = torch.randint(0, 256, (16, 3, 32, 32), dtype=torch.uint8, generator=generator)
        yield images, torch.randint(0, CLASSES, (16,), generator=generator)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, world, port, path, model):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    cfg, _ = load_small(f"train.model={model}", *SHARED)
    device = torch.device("cpu")
    assert train.distributed(device) == (rank, world)
    trainer = train.Trainer(cfg, CLASSES, STEPS, STEPS, device, rank, world)
    for images, labels in batches():
        trainer.train_step(images[rank::world], labels[rank::world])
    if rank == 0:
        torch.save(trainer.model.state_dict(), path)
    torch.distributed.destroy_process_group()


def weight_gap(model, path):
    port = free_port()
    context = mp.get_context("spawn")
    workers = [context.Process(target=worker, args=(rank, 2, port, path, model)) for rank in range(2)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(180)
        assert process.exitcode == 0
    cfg, _ = load_small(f"train.model={model}", *SHARED)
    single = train.Trainer(cfg, CLASSES, STEPS, STEPS, torch.device("cpu"))
    for images, labels in batches():
        single.train_step(images, labels)
    shared = torch.load(path)
    return max((value - shared[name]).abs().max().item() for name, value in single.model.named_parameters())


def test_two_processes_match_one_process(tmp_path):
    assert weight_gap("convnext", tmp_path / "ddp.pt") < 1e-5


def test_batch_norm_statistics_break_the_match(tmp_path):
    assert weight_gap("resnet", tmp_path / "ddp.pt") > 1e-4

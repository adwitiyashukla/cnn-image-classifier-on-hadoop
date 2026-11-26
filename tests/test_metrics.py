import math

import torch

from imgcnn import metrics


def test_ranks_and_summary_on_a_small_case():
    logits = torch.tensor([[5.0, 1.0, 0.0, 0.0], [1.0, 5.0, 3.0, 0.0], [4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    labels = torch.tensor([0, 2, 2, 3])
    assert metrics.ranks(logits, labels).tolist() == [0, 1, 2, 3]
    scores = metrics.summarize(logits, labels)
    assert scores["top1"] == 0.25
    assert scores["top3"] == 0.75
    assert math.isclose(scores["map3"], (1 + 1 / 2 + 1 / 3) / 4, rel_tol=1e-6)
    assert math.isclose(scores["loss"], torch.nn.functional.cross_entropy(logits, labels).item(), rel_tol=1e-6)


def test_ties_and_broken_rows_count_as_misses():
    flat = torch.zeros(3, 5)
    labels = torch.tensor([0, 1, 2])
    assert metrics.summarize(flat, labels)["top1"] == 0
    assert metrics.summarize(flat, labels)["top3"] == 0
    broken = torch.full((3, 5), float("nan"))
    scores = metrics.summarize(broken, labels)
    assert scores["top1"] == 0 and scores["top3"] == 0 and scores["map3"] == 0
    mixed = torch.tensor([[2.0, 1.0, 0.0], [float("inf"), 0.0, 0.0]])
    assert metrics.ranks(mixed, torch.tensor([0, 0])).tolist() == [0, 3]


def test_calibration_error_of_an_overconfident_model():
    logits = torch.tensor([[50.0, 0.0]] * 10)
    labels = torch.tensor([0] * 6 + [1] * 4)
    ece, rows = metrics.calibration(logits, labels)
    assert math.isclose(ece, 0.4, abs_tol=1e-6)
    assert len(rows) == 1 and rows[0]["count"] == 10


def test_calibration_error_is_small_for_sampled_labels():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(20000, 5, generator=generator) * 2
    labels = torch.multinomial(logits.softmax(1), 1, generator=generator)[:, 0]
    ece, _ = metrics.calibration(logits, labels)
    assert ece < 0.02


def test_temperature_scaling_recovers_the_scale():
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(20000, 10, generator=generator) * 2
    labels = torch.multinomial(logits.softmax(1), 1, generator=generator)[:, 0]
    temperature = metrics.fit_temperature(logits * 3, labels)
    assert abs(temperature - 3) < 0.15
    before, _ = metrics.calibration(logits * 3, labels)
    after, _ = metrics.calibration(logits * 3 / temperature, labels)
    assert after < before / 3


def test_bootstrap_interval_brackets_the_mean():
    generator = torch.Generator().manual_seed(2)
    values = torch.rand(4000, generator=generator) < 0.7
    low, high = metrics.bootstrap(values, 400)
    mean = values.float().mean().item()
    assert low < mean < high
    expected = 2 * 1.96 * math.sqrt(mean * (1 - mean) / 4000)
    assert abs((high - low) - expected) < 0.3 * expected

import numpy as np
import torch
from torch import nn

from imgcnn import scratch


def data(seed=0, n=6, size=16, classes=5):
    rng = np.random.default_rng(seed)
    return rng.random((n, 1, size, size)), rng.integers(0, classes, n)


def test_backward_pass_matches_autograd():
    x, labels = data()
    network = scratch.Network(5, widths=(4, 8), seed=1, dtype=np.float64)
    model, pairs = scratch.twin(network)
    gap, loss_gap = scratch.gradient_gap(network, model, pairs, x, labels)
    assert gap < 1e-9
    assert loss_gap < 1e-12


def test_backward_pass_matches_finite_differences():
    x, labels = data(seed=3)
    network = scratch.Network(5, widths=(4, 8), seed=2, dtype=np.float64)
    loss, grad = scratch.cross_entropy(network.forward(x), labels)
    network.backward(grad)
    rng = np.random.default_rng(0)
    step = 1e-6
    for layer in network.layers:
        for name, value in layer.params.items():
            analytic = layer.grads[name].copy()
            for _ in range(3):
                index = tuple(rng.integers(0, s) for s in value.shape)
                keep = value[index]
                value[index] = keep + step
                up, _ = scratch.cross_entropy(network.forward(x), labels)
                value[index] = keep - step
                down, _ = scratch.cross_entropy(network.forward(x), labels)
                value[index] = keep
                numeric = (up - down) / (2 * step)
                assert abs(numeric - analytic[index]) < 1e-6 + 1e-4 * abs(numeric)


def test_training_tracks_torch_step_for_step():
    network = scratch.Network(5, widths=(4, 8), seed=4, dtype=np.float64)
    model, _ = scratch.twin(network)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=scratch.MOMENTUM)
    for step in range(5):
        x, labels = data(seed=10 + step, n=8)
        ours = network.step(x, labels, 0.05)
        loss = nn.functional.cross_entropy(model(torch.from_numpy(x)), torch.from_numpy(labels))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert abs(ours - loss.item()) < 1e-9
    x, _ = data(seed=99)
    model.eval()
    with torch.no_grad():
        reference = model(torch.from_numpy(x)).numpy()
    assert np.abs(network.forward(x, train=False) - reference).max() < 1e-8


def test_convolution_matches_torch_on_its_own():
    rng = np.random.default_rng(5)
    layer = scratch.Conv(3, 4, 3, rng, np.float64)
    x = rng.random((2, 3, 7, 9))
    reference = nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(layer.params["w"]), padding=1)
    assert np.abs(layer.forward(x, True) - reference.numpy()).max() < 1e-12


def test_max_pool_sends_the_gradient_to_the_first_maximum():
    pool = scratch.MaxPool()
    x = np.zeros((1, 1, 4, 4))
    x[0, 0, 2, 3] = 5
    out = pool.forward(x, True)
    assert out.tolist() == [[[[0, 0], [0, 5]]]]
    back = pool.backward(np.ones((1, 1, 2, 2)))
    expected = np.zeros((4, 4))
    expected[0, 0] = expected[0, 2] = expected[2, 0] = expected[2, 3] = 1
    assert (back[0, 0] == expected).all()


def test_cross_entropy_gradient_sums_to_zero():
    rng = np.random.default_rng(6)
    loss, grad = scratch.cross_entropy(rng.normal(size=(4, 5)) * 30, np.array([0, 1, 2, 3]))
    assert np.isfinite(loss)
    assert np.abs(grad.sum(1)).max() < 1e-12

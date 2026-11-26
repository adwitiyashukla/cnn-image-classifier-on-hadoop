import torch
from torch.nn import functional as F


def ranks(logits, labels):
    true = logits.gather(1, labels[:, None])
    rank = (logits >= true).sum(1) - 1
    broken = ~torch.isfinite(logits).all(1)
    return torch.where(broken, logits.shape[1], rank)


def summarize(logits, labels):
    rank = ranks(logits, labels)
    precision = torch.where(rank < 3, 1.0 / (rank + 1), torch.zeros(len(rank)))
    return {
        "top1": (rank == 0).float().mean().item(),
        "top3": (rank < 3).float().mean().item(),
        "map3": precision.mean().item(),
        "loss": F.cross_entropy(logits, labels).item(),
    }


def calibration(logits, labels, bins=15):
    probs = logits.softmax(1)
    confidence, predicted = probs.max(1)
    correct = (predicted == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for low, high in zip(edges[:-1].tolist(), edges[1:].tolist()):
        inside = (confidence > low) & (confidence <= high)
        count = int(inside.sum())
        if count == 0:
            continue
        accuracy = correct[inside].mean().item()
        mean_confidence = confidence[inside].mean().item()
        ece += abs(accuracy - mean_confidence) * count / len(labels)
        rows.append({"low": low, "high": high, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    return ece, rows


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_t.exp().item()


def bootstrap(values, resamples, seed=0, chunk=20):
    generator = torch.Generator().manual_seed(seed)
    values = values.float()
    means = []
    for start in range(0, resamples, chunk):
        rows = min(chunk, resamples - start)
        index = torch.randint(len(values), (rows, len(values)), generator=generator)
        means.append(values[index].mean(1))
    means = torch.cat(means)
    return [means.quantile(0.025).item(), means.quantile(0.975).item()]

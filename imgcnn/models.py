import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DropPath(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if not self.training or self.rate == 0:
            return x
        keep = 1 - self.rate
        mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep)
        return x * mask / keep


class Classifier(nn.Module):
    def maps(self, x):
        return F.conv2d(self.features(x), self.fc.weight[:, :, None, None])

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))


class PreActBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        skip = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + skip


class ResNet(Classifier):
    def __init__(self, cfg, channels, classes):
        super().__init__()
        self.stem = nn.Conv2d(channels, cfg.stem, 3, 1, 1, bias=False)
        blocks, cin = [], cfg.stem
        for stage, (width, depth) in enumerate(zip(cfg.widths, cfg.depths)):
            for i in range(depth):
                blocks.append(PreActBlock(cin, width, 2 if i == 0 and stage > 0 else 1))
                cin = width
        self.blocks = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, classes)

    def features(self, x):
        return F.relu(self.norm(self.blocks(self.stem(x))))


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.squeeze = nn.Conv2d(channels, hidden, 1)
        self.excite = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        gate = x.mean((2, 3), keepdim=True)
        gate = torch.sigmoid(self.excite(F.relu(self.squeeze(gate))))
        return x * gate


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hidden = cin * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6()]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(),
            SqueezeExcite(hidden),
            nn.Conv2d(hidden, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.body = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        out = self.body(x)
        return x + out if self.residual else out


class MobileNet(Classifier):
    def __init__(self, cfg, channels, classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, cfg.stem, 3, 1, 1, bias=False), nn.BatchNorm2d(cfg.stem), nn.ReLU6()
        )
        blocks, cin = [], cfg.stem
        for expand, width, repeat, stride in zip(cfg.expand, cfg.channels, cfg.repeats, cfg.strides):
            for i in range(repeat):
                blocks.append(InvertedResidual(cin, width, stride if i == 0 else 1, expand))
                cin = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(cin, cfg.head, 1, bias=False), nn.BatchNorm2d(cfg.head), nn.ReLU6())
        self.fc = nn.Linear(cfg.head, classes)

    def features(self, x):
        return self.head(self.blocks(self.stem(x)))


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.expand = nn.Conv2d(dim, 4 * dim, 1)
        self.project = nn.Conv2d(4 * dim, dim, 1)
        self.scale = nn.Parameter(torch.full((dim,), 1e-2))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        out = self.project(F.gelu(self.expand(self.norm(self.depthwise(x)))))
        return x + self.drop_path(out * self.scale[:, None, None])


class ConvNeXt(Classifier):
    def __init__(self, cfg, channels, classes):
        super().__init__()
        total = sum(cfg.depths)
        rates = [cfg.drop_path * i / max(total - 1, 1) for i in range(total)]
        layers = [nn.Conv2d(channels, cfg.widths[0], 2, 2), LayerNorm2d(cfg.widths[0])]
        cin, index = cfg.widths[0], 0
        for width, depth in zip(cfg.widths, cfg.depths):
            if width != cin:
                layers += [LayerNorm2d(cin), nn.Conv2d(cin, width, 2, 2)]
                cin = width
            for _ in range(depth):
                layers.append(ConvNeXtBlock(cin, rates[index]))
                index += 1
        self.blocks = nn.Sequential(*layers)
        self.norm = LayerNorm2d(cin)
        self.fc = nn.Linear(cin, classes)

    def features(self, x):
        return self.norm(self.blocks(x))


FAMILIES = {"resnet": ResNet, "mobilenet": MobileNet, "convnext": ConvNeXt}


def build(cfg, classes, name=None, channels=None):
    name = name or cfg.train.model
    model = FAMILIES[name](getattr(cfg, name), channels or cfg.train.channels, classes)
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name == "convnext":
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def complexity(model, channels, size):
    macs = 0

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
            macs += output[0].numel() * kernel
        else:
            macs += module.in_features * module.out_features

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, channels, size, size, device=next(model.parameters()).device))
    model.train(training)
    for hook in hooks:
        hook.remove()
    return {"params": sum(p.numel() for p in model.parameters()), "macs": macs}

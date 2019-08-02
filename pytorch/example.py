import torchvision
import torch
from torch import nn
from torch.nn import functional as F
import collections

import os
import numpy as np

import matplotlib.pyplot as plt

from layer_rotation_control import SGD
from layer_rotation_monitoring import get_weights, compute_cosine_distances
from scipy.spatial.distance import cosine


batch_size = 32
num_classes = 10
epochs = 100
verbose = 0


class Lambda(nn.Module):
    def __init__(self, lambd):
        self.lambd = lambd
        super().__init__()

    def forward(self, x):
        return self.lambd(x)


model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=0),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),

    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=0),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d((2,2)),

    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(),
    nn.Conv2d(256, 256, 3, padding=0),
    nn.BatchNorm2d(256),
    nn.ReLU(),

    Lambda(lambda x: F.avg_pool2d(x, x.shape[2:4])),
    Lambda(lambda x: x.view(x.shape[0], 256)),
    nn.Linear(256, num_classes),
    nn.Softmax(dim=-1)
)
init = get_weights(model)

training_dataset = torchvision.datasets.CIFAR10(
    train=True,
    root="./cifar",
    download=True,
    transform=torchvision.transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(training_dataset,
    batch_size=batch_size,
    shuffle=True,)

testing_dataset = torchvision.datasets.CIFAR10(
    train=False,
    root="./cifar",
    download=True,
    transform=torchvision.transforms.ToTensor())
testing_loader = torch.utils.data.DataLoader(testing_dataset,
    batch_size=batch_size,
    shuffle=True,)

lr = 3**-3
optimizer = SGD([{"params": list(p for n, p in model.named_parameters()
                                 if n.endswith(".bias")),
                  "layca": False
                 },
                 {"params": list(p for n, p in model.named_parameters()
                                 if n.endswith(".weight")),
                  "layca": True
                 },
                ], lr=lr)

# learning rate schedule: divide learning rate by 5 at epochs 70 and 90
def schedule(epoch):
    new_lr = lr
    if epoch > 70:
        new_lr *= 0.2
    if epoch > 90:
        new_lr *= 0.2
    return new_lr

model.to("cuda")


class MetricsHandler:
    def __init__(self):
        self._metrics = {}
        self._aggregators = {}
        self._bs = []
        self._results = collections.defaultdict(list)

    def add(self, name, metric, agg):
        assert name not in self._aggregators
        self._metrics[name] = metric
        self._aggregators[name] = agg
        return self

    def agg(self, name, agg):
        assert name not in self._aggregators
        self._aggregators[name] = agg
        return self

    def record(self, name, value):
        self._results[name].append(value)

    def compute_on_batch(self, y, t, batch_size):
        loss = None
        self._bs.append(batch_size)
        for name, metric in self._metrics.items():
            value = metric(y, t)
            if name == "loss":
                loss = value
            value = value.detach().cpu().item()
            self._results[name].append(value)
        return loss

    def summarize(self, prefix=""):
        summary = {prefix+name: self._aggregators[name](np.array(self._results[name]), np.array(self._bs))
                   for name, lst in self._results.items()
                   }
        self._results.clear()
        self._bs.clear()
        return summary


def accuracy(y, t):
    return torch.sum(torch.argmax(y, dim=1) == t).type(torch.float32)


metrics_handler = (MetricsHandler()
    .add("loss", F.cross_entropy, lambda means,bs: np.sum(means*bs)/np.sum(bs))
    .add("acc", accuracy, lambda sums,bs: np.sum(sums)/np.sum(bs))
)

metrics = collections.defaultdict(list)

from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    optimizer.state["lr"] = schedule(epoch)
    model.train()
    for batch in training_loader:
        x, t = batch
        x = x.to("cuda") / 255.0
        t = t.to("cuda")
        optimizer.zero_grad()
        y = model(x)
        loss = metrics_handler.compute_on_batch(y, t, x.shape[0])
        loss.backward()
        optimizer.step()

    for k, v in metrics_handler.summarize("trn.").items():
        metrics[k].append(v)

    model.eval()
    for batch in testing_loader:
        x, t = batch
        x = x.to("cuda") / 255.0
        t = t.to("cuda")
        y = model(x)
        loss = metrics_handler.compute_on_batch(y, t, x.shape[0])

    for k, v in metrics_handler.summarize("val.").items():
        metrics[k].append(v)

    print(f"Acc: {metrics['trn.acc'][-1]}, {metrics['val.acc'][-1]} Loss: {metrics['trn.loss'][-1]}, {metrics['val.loss'][-1]}")

    for k, v in compute_cosine_distances(model, init).items():
        metrics["rot."+k].append(v)


for k, v in metrics.items():
    if k.startswith("rot."):
        plt.plot(v)
plt.show()

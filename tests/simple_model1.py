"""Simple tests of NN's"""
import torch


x = torch.randn(50, 2)
print(x)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


print(log_softmax(x))

print(x.unsqueeze(-1))

from torch import nn, softmax
import torch
x = torch.randn(2, 3)
print('x:', x)

softmax = torch.nn.Softmax(dim=1)
y = softmax(x)
print('y:', y)
z=y.argmax(dim=1)
print('z:', z)
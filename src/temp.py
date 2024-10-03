from tqdm import tqdm
import time
import torch


a = torch.randn(3, 1)
b = torch.randn(3, 3)
print(a)
print(b)
print(a*b)
print(torch.mul(a, b))
print(a.mul(b))

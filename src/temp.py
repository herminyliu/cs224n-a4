from tqdm import tqdm
import time
import torch


import torch

# 创建两个复数张量
a_real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
a_imag = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
a = torch.view_as_complex(torch.stack((a_real, a_imag), dim=-1))
print(a)
b_real = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
b_imag = torch.tensor([[13.0, 14.0], [15.0, 16.0]])
b = torch.view_as_complex(torch.stack((b_real, b_imag), dim=-1))
print(b)
# 逐元素复数相乘
result = a * b
print(result)


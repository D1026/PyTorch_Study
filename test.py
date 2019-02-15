import torch
import numpy as np

# print('--- test ---')
#
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
#
# y = x + 2
# print(y)
#
# a = torch.randn(2, 2)
# a = ((a * 3) / (a - 1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a * a).sum()
# print(b.grad_fn)

# a = torch.randn(2, 2, 3)
# print(a.sum())
# print('---------')
# print(a.shape)
# a = a[:, None, :, :]
# print(a.shape)
# print(a.size())
# print(*a.size())
# print(torch.zeros(*a.size()))

# a = torch.randn(3, 9, 1, 4, 5)
# # b = torch.randn(1, 9, 8, 1, 4)
# # p = a @ b
# # print(p.size())

x = np.arange(24).reshape(2, 3, 4)
y = np.arange(8).reshape(2, 1, 4)
print('x：', x)
print('y：', y)
a = torch.from_numpy(x)
b = torch.from_numpy(y)
print(a.shape)
print(a.sum(dim=0, keepdim=True).shape)
print(a.sum(dim=0, keepdim=False).shape)
print(a*b)


a = torch.tensor([1, 2, 3])
b = torch.tensor([4])
print(a+b)

torch.tensor.sum()

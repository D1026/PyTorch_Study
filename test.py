import torch
import numpy as np
a = torch.Tensor([
    [
        [1, 2],
        [3, 4]
    ],
    [
         [5, 6],
         [7, 8]
     ]           ])
print(a.shape)
c = a + torch.Tensor([[0, 1],
                  [0, 1]])
print(c)
# import tensorflow as tf
# tf.zeros_like()
# tf.reduce_sum()
# a = np.arange(10)
# print(a)
# print(type(a.tolist()))
# import tensorflow as tf
#
# a = tf.constant([[[1, 2, 3], [4, 5, 6]],
#                 [[1, 2, 3], [4, 5, 6]]])
# b = tf.concat(a, axis=1)
# with tf.Session() as se:
#     se.run(b)
#     print(b.eval())

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

# x = np.arange(8).reshape(2, 1, 4)
# y = np.arange(12).reshape(3, 4)
# print('x：', x)
# print('y：', y)
#
# a = torch.from_numpy(x)
# b = torch.from_numpy(y)
#
# c = a*b
# print(c)
# print(c.shape)
#
# print(a.shape[0])
# a = a + b
# print(a)
#
# print(a.shape)
# a = a.unsqueeze(-1)
# print(a.shape)
# a = torch.reshape(a, (2, 3, 4))
# print(a.shape)
# c = [*a.size()]
# print(c)
# print(a.sum(dim=0, keepdim=True).shape)
# print(a.sum(dim=0, keepdim=False).shape)
# print(a*b)


# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4])
# print(a+b)
#
# torch.tensor.sum()

# a = tf.fill([2, 3], 0.0)
# print(a)
# b = tf.constant(1.0)
# c = a + b
#
# with tf.Session() as se:
#     se.run(c)
#     print(a.eval())
#     print(c.eval())

# a = torch.tensor([[1, 2],
#                   [3, 4]])
# print(a.transpose(0, 1))

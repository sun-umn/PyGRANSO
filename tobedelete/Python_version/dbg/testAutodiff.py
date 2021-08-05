import torch

x = torch.tensor([[1.,2.,3.],[1.,2.,3.]], requires_grad=True)

y = abs(x)

print(y)

y.sum().backward(retain_graph=True)
print(x.grad)

z = abs(x)

z.sum().backward(retain_graph=True)
print(x.grad)

pass
import torch

x = torch.tensor(-10.5, requires_grad=True)
print(x.data)
print(x.grad)
print(x.grad_fn)

y = abs(x)

print(y)
print(y.grad)
print(y.grad_fn)
y.backward()
print(x.grad)

pass
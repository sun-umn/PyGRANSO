import torch

a = torch.tensor([0., -.5], requires_grad=True)
# b = torch.tensor([6., 4.], requires_grad=True)

Q = abs(a)

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print( a.grad)
# print(-2*b == b.grad)

C = Q.detach().numpy()

print(C)
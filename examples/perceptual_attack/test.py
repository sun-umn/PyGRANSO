import torch
import torch.nn.functional as F
import math

# input = torch.randn(3, 5, requires_grad=True)
# target = torch.randint(5, (3,), dtype=torch.int64)

input = torch.tensor([ [0., 1.], [0.5, 0.5] ])
target = torch.tensor([1,1])

loss = F.cross_entropy(input, target)
print(loss)

print((math.log(math.e/(1+math.e)) + math.log(0.5))/2)
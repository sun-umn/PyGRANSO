import sys
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
import private.bfgsHessianInverseLimitedMem as lbfgsHI
import torch

device = torch.device('cpu' )

H = torch.eye(2).to(device=device, dtype=torch.double)
scaleH0 = True
limited_mem_fixed_scaling = True
limited_mem_size = 1
limited_mem_warm_start = None

bfgs_obj =  lbfgsHI.bfgsHessianInverseLimitedMem(H,scaleH0,limited_mem_fixed_scaling,limited_mem_size,limited_mem_warm_start,device)

s = torch.tensor([[0.1077], [-0.1743]]).to(device=device, dtype=torch.double)
y = torch.tensor([[6.2551], [-7.5789]]).to(device=device, dtype=torch.double)
sty = 1.9952
damped = False
skipped = bfgs_obj.update(s,y,sty,damped)

s = torch.tensor([[-0.0273], [0.0317]]).to(device=device, dtype=torch.double)
y = torch.tensor([[-6.6478], [5.5789]]).to(device=device, dtype=torch.double)
sty = 0.3584
damped = False
skipped = bfgs_obj.update(s,y,sty,damped)

s = torch.tensor([[0.0159], [-0.0142]]).to(device=device, dtype=torch.double)
y = torch.tensor([[6.5762], [-5.5789]]).to(device=device, dtype=torch.double)
sty = 0.1837
damped = False
skipped = bfgs_obj.update(s,y,sty,damped)


q = torch.tensor([[8.7341], [-8.0]]).to(device=device, dtype=torch.double)
r = bfgs_obj.applyH(q)
print(r)
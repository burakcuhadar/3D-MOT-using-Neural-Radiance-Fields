import torch
from pytorch3d.transforms import random_rotation

import theseus as th
from theseus import LieGroupTensor
from theseus.geometry.lie_group import LieGroup

with torch.no_grad():
    x = torch.randn(1000, 3, requires_grad=False).cuda()
    rotation = random_rotation().cuda().requires_grad_(False)
    print("gt rot\n", rotation)
    trans = torch.tensor([1.0, 2.0, 3.0]).cuda().requires_grad_(False)
    y = torch.einsum("ij,nj->ni", rotation, x) + trans
    #y = torch.einsum("ij,nj->ni", rotation, x)
    y = y.detach()

my_se3 = LieGroupTensor.from_identity(1, LieGroup.SE3).cuda()

optimizer = torch.optim.Adam([my_se3], lr=0.0001)


for i in range(100000):
    optimizer.zero_grad()
    
    #y_hat = my_se3.Act(x)

    loss = torch.nn.functional.l1_loss(y_hat, y)
    

    
    loss.backward()
    optimizer.step()
    if i % 10000 == 0:
        print(loss.item())

print("final loss:", loss.item())

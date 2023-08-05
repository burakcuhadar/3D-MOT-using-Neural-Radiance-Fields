import torch

from pytorch3d.transforms import random_rotation
from lietorch import SO3


x = torch.randn(1000, 3, requires_grad=False).cuda()
rotation = random_rotation().cuda().requires_grad_(False)
print(rotation)
# y = x @ rotation
y = torch.matmul(rotation, x.T).T
y = y.detach()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.phi = torch.nn.Parameter(torch.zeros((3,), requires_grad=True).cuda())

    def forward(self, x):
        so3 = SO3(self.phi)[None]
        return so3.act(x)


model = Model().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for i in range(1000):
    y_hat = model(x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())

print(SO3(model.phi).matrix())

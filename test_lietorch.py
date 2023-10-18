import torch

import numpy as np
from pytorch3d.transforms import random_rotation
from lietorch import SO3, SE3
from scipy.spatial.transform import Rotation


with torch.no_grad():
    x = torch.randn(1000, 3, requires_grad=False)
    #x = torch.cat((x, torch.ones(1000,1, device="cuda")), dim=-1)
    rotation = random_rotation().requires_grad_(False)
    #rotation = SO3.exp(torch.randn(3).cuda()).matrix()
    print(rotation)
    #y = torch.cat((x, torch.ones(1000,1, device="cuda")), dim=-1) @ rotation
    #y = torch.matmul(rotation, x.T).T
    y = torch.einsum("ij,nj->ni", rotation, x)
    y += torch.tensor([1.0, 2.0, 3.0])
    y = y.detach()



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #self.phi = torch.nn.Parameter(torch.zeros((3,), requires_grad=True).cuda())
        #self.pose = torch.nn.Parameter(torch.zeros((7,), requires_grad=True).cuda())
        self.pose =  torch.nn.Parameter(SE3.Identity(1).log())
        print("pose", self.pose.shape)
        #self.se3 = SE3(self.pose)
        

    def forward(self, x):
        #so3 = SO3(self.phi)[None]
        #return so3.act(x)
        #return torch.matmul(SO3.exp(self.phi).matrix(), x.T).T
        #
        #x_homog = torch.cat((x, torch.ones(x.shape[0],1)), dim=-1)
        #return torch.einsum("ij,nj->ni", SE3.exp(self.pose).matrix(), x_homog)[:,:3]

        return SE3.exp(self.pose).act(x)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

for i in range(60000):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    #loss = torch.nn.functional.huber_loss(y_hat, y, delta=1.0)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(loss.item())

print(loss.item())
print(model.pose)
print(SE3.exp(model.pose).matrix())
print(SE3.exp(model.pose).translation())
print(SE3.exp(model.pose).vec())
print(SE3.exp(model.pose).log())
print(SE3.exp(model.pose).data)
print(SE3.exp(model.pose).manifold_dim)
print(SE3.exp(model.pose).tangent_shape)
print(SE3.exp(model.pose).matrix())
#print('rotation', Rotation.from_quat(model.pose[0,3:]).as_matrix())
#print(SE3.InitFromVec(torch.from_numpy(Rotation.from_quat(model.pose.detach().numpy()))).matrix())

"""
pose_mat = data['poses']  # [n, 4, 4]
quat = Rotation.from_matrix(pose_mat[:, :3, :3]).as_quat()
trans = pose_mat[:, :3, 3]
pose_data = np.concatenate((trans, quat), axis=-1)
T = SE3.InitFromVec(torch.tensor(pose_data))
error = (T.matrix() - torch.tensor(pose_mat)).sum()
print(error)
"""
quat = Rotation.from_matrix(rotation).as_quat()
print("quat", quat)
pose_data = np.concatenate((torch.tensor([1.0, 2.0, 3.0]), quat), axis=-1)
T = SE3.InitFromVec(torch.tensor(pose_data))
print(T.matrix())

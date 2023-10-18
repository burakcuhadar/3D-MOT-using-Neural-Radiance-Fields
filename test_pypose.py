import torch
from pytorch3d.transforms import random_rotation
import pypose as pp
from pypose.optim import LM
from pypose.optim.strategy import Constant
from pypose.optim.scheduler import StopOnPlateau
from utils.rigid import exp_se3, to_homogenous, skew
from models.nerf import MLP
from scipy.spatial.transform import Rotation


class SE3Field(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(1, 6)
        self.encoder = MLP(
            input_dim=6,
            output_dim=6,
            net_depth=4,
            skip_layer=2,
            net_width=64,
        )

    def forward(self, pts):
        out = self.encoder(self.embedding(torch.zeros(1, dtype=torch.long, device=pts.device)))
        w = out[0, :3]
        v = out[0, 3:]
        theta = torch.linalg.norm(w, dim=-1)
        w = w / theta
        v = v / theta
        screw_axis = torch.cat([w, v], dim=-1)
        transform = exp_se3(screw_axis, theta)
        warped_points = torch.einsum("ij,nj->ni", transform, to_homogenous(pts))[:,:3]
        
        return warped_points
        

class MySE3(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, pose, input):
        ctx.save_for_backward(input)
        matrix = exp_se3(pose, torch.linalg.norm(pose[:3], dim=-1))
        transformed = torch.einsum("ij,nj->ni", matrix, to_homogenous(input))[:,:3]
        return transformed
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        input_mean = input.mean(dim=0)
        input_skew = skew(input_mean)
        
        jacobian = torch.eye(3, 6, device=grad_output.device)
        jacobian[:, 3:] = -input_skew

        pose_grad = torch.einsum("ni,ij->nj", grad_output, jacobian)

        return pose_grad, None

with torch.no_grad():
    x = torch.randn(1000, 3, requires_grad=False)
    rotation = random_rotation().requires_grad_(False)
    print("gt rot\n", rotation)
    trans = torch.tensor([1.0, 2.0, 3.0]).requires_grad_(False)
    y = torch.einsum("ij,nj->ni", rotation, x) + trans
    #y = torch.einsum("ij,nj->ni", rotation, x)
    y = y.detach()

with torch.no_grad():
    rot_init = torch.from_numpy(Rotation.from_matrix(rotation).as_quat()).float()
    trans_init = trans.clone().float()
    se3_init = torch.cat((trans_init, rot_init), dim=-1)

my_se3 = pp.Parameter(pp.LieTensor(se3_init, ltype=pp.SE3_type))
#my_se3 = pp.Parameter(pp.identity_SE3(1))
#my_se3 = torch.nn.Parameter(pp.identity_SE3(1).cuda())

#my_se3 = pp.Parameter(pp.identity_SO3(1).cuda())
#my_se3 = torch.nn.Parameter(pp.identity_SE3(1).cuda())
#S = torch.nn.Parameter(torch.tensor([1.,0.,0.,0.,0.,0.]).cuda())
#theta = torch.nn.Parameter(torch.zeros(1).cuda())
#field = SE3Field().cuda()
#pose = torch.nn.Parameter(torch.zeros(6).cuda())

"""
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pose = pp.Parameter(pp.identity_SE3(1))
    def forward(self, inp):
        #y_hat = self.pose @ inp
        y_hat = self.pose.Act(inp)
        return torch.nn.functional.mse_loss(y_hat, y, reduction="none")
        #return torch.nn.functional.l1_loss(y_hat, y)
        #return torch.nn.functional.huber_loss(y_hat, y)

model = MyModel().cuda()

strategy = Constant(damping=1e-6)
optimizer = LM(model, strategy=strategy)
scheduler = StopOnPlateau(optimizer, steps=10000, patience=3, decreasing=1e-3, verbose=False)        

#scheduler.optimize(input=x)

for i in range(100):
    loss = optimizer.step(input=x)
    scheduler.step(loss)
    if i % 10 == 0:
        print(loss.item())

print("after opt", model.pose.matrix())
"""


optimizer = torch.optim.Adam([my_se3], lr=0.0001)
#optimizer = torch.optim.NAdam([my_se3], lr=0.0001, betas=(0.9, 0.95))
#optimizer = torch.optim.RAdam([my_se3], lr=0.0005)
#optimizer = torch.optim.SGD([my_se3], lr=0.0001)
#optimizer = torch.optim.Adam([S, theta], lr=0.0001)
#optimizer = torch.optim.Adam(field.parameters(), lr=0.0005)
#optimizer = torch.optim.Adam([pose], lr=0.0001)

for i in range(100000):
    optimizer.zero_grad()
    #y_hat = my_se3 @ x
    x_homog = torch.cat((x, torch.ones(x.shape[0],1)), dim=-1)
    #y_hat = torch.einsum("ij,nj->ni", pp.SE3(my_se3).matrix()[0,...], x_homog)[:,:3]
    y_hat = my_se3.Act(x)

    loss = torch.nn.functional.mse_loss(y_hat, y)
    #loss = torch.nn.functional.huber_loss(y_hat, y, delta=1.0)
    #loss = torch.nn.functional.l1_loss(y_hat, y)
    
    '''
    mat = exp_se3(S, theta)
    y_hat = torch.einsum("ij,nj->ni", mat, to_homogenous(x))[:,:3]
    loss = torch.nn.functional.mse_loss(y_hat, y)
    '''

    '''
    y_hat = field(x)
    loss = torch.nn.functional.l1_loss(y_hat, y)
    '''
    
    '''
    y_hat = MySE3.apply(pose, x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    '''

    
    loss.backward()
    optimizer.step()
    if i % 10000 == 0:
        print(loss.item())

print("final loss:", loss.item())
#print("after opt\n", exp_se3(S, theta))
#print("after opt\n", field(torch.eye(3, device=x.device)))
#print("after opt\n", pose)
print(my_se3.matrix())
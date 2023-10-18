import torch
from scipy.spatial.transform import Rotation


"""
ckpt_path = "/home/stud/cuhadar/storage/user/star_trans_only_lightning/3D-MOT-using-Neural-Radiance-Fields/sbatch/ckpts/online/sfhbh1da/epoch=58.ckpt"
ckpt = torch.load(ckpt_path)
print(ckpt["state_dict"]["poses"].cpu().numpy())
"""

from lietorch import SO3, SE3
import pypose as pp
from numpy import pi

"""
pose = torch.tensor([1,0, 2.0, 3.0, pi/2, pi/3, pi/4])
print("pose", pose)
quat = SE3.exp(pose).vec()
print("quat", quat)
#back_to_pose = SO3.exp(quat).log()
#print("back_to_pose", back_to_pose)
"""


pose = torch.tensor([pi/2, pi/3, pi/4])
print("pose", pose)
quat = Rotation.from_rotvec(pose).as_quat()
print("quat", quat)
back_to_pose = Rotation.from_quat(quat).as_rotvec()
print("back_to_pose", back_to_pose)


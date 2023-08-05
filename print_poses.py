import torch

ckpt_path = "/home/stud/cuhadar/storage/user/star_trans_only_lightning/3D-MOT-using-Neural-Radiance-Fields/sbatch/ckpts/online/sfhbh1da/epoch=58.ckpt"
ckpt = torch.load(ckpt_path)
print(ckpt["state_dict"]["poses"].cpu().numpy())

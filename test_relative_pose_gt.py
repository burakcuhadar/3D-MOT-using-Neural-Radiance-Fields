from datasets import dataset_dict
from utils.io import *

parser = config_parser()
args = parser.parse_args()
args.use_viewdirs = True
args.N_importance = 16
args.scale_factor = 0.2
args.datadir = "/usr/stud/cuhadar/storage/user/3D-MOT-using-Neural-Radiance-Fields/data/carla_dynamic"


ds = dataset_dict['carla_star'](args, split='train_online', num_frames=15)

print(ds.gt_relative_poses)


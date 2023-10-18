from utils.metrics import get_pose_metrics
from utils.dataset import rotation_metric
from scipy.spatial.transform import Rotation as R
import numpy as np

rot1 = R.from_euler('zyx', [0, 45, 30], degrees=True).as_rotvec()[None, ...]
rot2 = R.from_euler('zyx', [90, 45, 30], degrees=True).as_rotvec()[None, ...]
print(rotation_metric(rot1, rot2))



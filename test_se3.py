from datasets.carla_star_online import StarOnlineDataset
from utils.io import config_parser
from lietorch import SE3, SO3
from scipy.spatial.transform import Rotation
import numpy as np
import torch


def log_scipy(pose_mat):
    quat = Rotation.from_matrix(pose_mat[:, :3, :3]).as_rotvec()
    trans = pose_mat[:, :3, 3]
    pose_data = np.concatenate((trans, quat), axis=-1).astype(np.float32)
    return pose_data


def se3_exp_map(poses):
    matrices = np.tile(np.eye(4, dtype=np.float32), (16, 1, 1))
    matrices[:, :3, :3] = (
        SO3.exp(torch.tensor(poses[:, 3:])).matrix().numpy()[:, :3, :3]
    )
    matrices[:, :3, 3] = poses[:, :3]
    return matrices


parser = config_parser()
args = parser.parse_args()

ds = StarOnlineDataset(args, "train", 5)

gt_relative_poses_matrices = ds.gt_relative_poses_matrices.numpy()
print("matrices")
print(gt_relative_poses_matrices)


# print("poses with lietorch")
# print(SE3.log(gt_relative_poses_matrices))

print("poses with scipy")
poses_scipy = log_scipy(gt_relative_poses_matrices)
print(poses_scipy)
# matrices_scipy = SE3.exp(torch.tensor(poses_scipy)).matrix().numpy()
matrices_scipy = se3_exp_map(poses_scipy)
print("scipy matrices", matrices_scipy)

print("scipy matrices diff", np.abs(gt_relative_poses_matrices - matrices_scipy).sum())


points = torch.randn(100, 3, dtype=torch.float32)
points_homog = torch.zeros(100, 4, dtype=torch.float32)
points_homog[:, :3] = points
gt_points_transformed = torch.einsum(
    "ij,nj->ni", torch.from_numpy(gt_relative_poses_matrices[0, ...]), points_homog
)
# using SO3.matrix
mat_points_transformed = torch.einsum(
    "ij,nj->ni", torch.from_numpy(matrices_scipy[0, ...]), points_homog
)
print("diff SO3.mat", (gt_points_transformed - mat_points_transformed))

# using SO3.act
act_points_transformed = SO3.exp(torch.from_numpy(poses_scipy[0, 3:][None, ...])).act(
    points
) + torch.from_numpy(poses_scipy[0, :3][None, ...])
print(
    "diff SO3.act",
    (gt_points_transformed[:, :3] - act_points_transformed).abs(),
)

"""
print("diff")
# print(np.abs(gt_relative_poses_matrices - matrices_scipy))
large_diff = np.abs(gt_relative_poses_matrices - matrices_scipy) > 1e-7
# print(large_diff)
if np.any(large_diff):
    print("large diff detected!")
"""

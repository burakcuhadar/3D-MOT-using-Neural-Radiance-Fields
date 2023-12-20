import numpy as np
import os
import re

import torch
import pypose as pp
from lietorch import SE3
from pytorch3d.transforms import se3_log_map as se3_log_map_p3d
from scipy.spatial.transform import Rotation


def load_intrinsics(args):
    intrinsics = np.load(
        os.path.join(args.datadir, "intrinsics.npy"), allow_pickle=True
    ).item()
    H = intrinsics["h"]
    W = intrinsics["w"]
    fov = intrinsics["fov"]
    focal = W / (2 * np.tan(fov * np.pi / 360))

    return H, W, focal


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]

def from_ue4_to_nerf_pts(pts):
    change_ue4_to_nerf = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
    return np.einsum('ij,nj->ni', change_ue4_to_nerf, pts)

def from_ue4_to_nerf(pose):
    change_ue4_to_nerf = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)

    # inverse of the above
    change_nerf_to_ue4 = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

    new_pose = np.eye(pose.shape[0], pose.shape[1])

    # Rotation
    new_pose[:3, :3] = change_ue4_to_nerf @ pose[:3, :3] @ change_nerf_to_ue4
    # Translation
    new_pose[:3, -1] = change_ue4_to_nerf @ pose[:3, -1]

    return new_pose


def invert_transformation(t):
    if len(t.shape) == 2:
        t_inv = np.eye(4, dtype=np.float32)
        t_inv[:3, :3] = t[:3, :3].T
        t_inv[:3, -1] = -t[:3, :3].T @ t[:3, -1]
    elif len(t.shape) == 3:
        t_inv = np.eye(4, dtype=np.float32)[None, ...].repeat(t.shape[0], axis=0)
        t_inv[:, :3, :3] = t[:, :3, :3].transpose(0, 2, 1)
        #t_inv[:, :3, 3] = -t_inv[:, :3, :3] @ t[:, :3, 3]
        t_inv[:, :3, 3] = -np.einsum('ijk,ik->ij', t_inv[:, :3, :3], t[:, :3, 3])
    return t_inv

@torch.no_grad()
def se3_log_map(matrices):
    """ old, remove
    matrices_p3d = np.eye(4, dtype=np.float32)[None, ...].repeat(matrices.shape[0], axis=0)
    matrices_p3d[:, :3, :3] = matrices[:, :3, :3]   
    matrices_p3d[:, 3, :3] = matrices[:, :3, 3]
    se3_log = se3_log_map_p3d(torch.from_numpy(matrices_p3d)).numpy()
    """

    """ old, remove
    rot = Rotation.from_matrix(matrices[:, :3, :3]).as_rotvec()
    trans = matrices[:, :3, 3]
    return np.concatenate((trans, rot), axis=-1).astype(np.float32)
    """
    
    """
    quat = Rotation.from_matrix(matrices[:, :3, :3]).as_quat()
    trans = matrices[:, :3, 3]
    pose_data = np.concatenate((trans, quat), axis=-1)
    T = SE3.InitFromVec(torch.tensor(pose_data))
    se3_log = T.log().numpy()
    """

    se3_log = pp.mat2SE3(matrices).tensor().numpy()

    return se3_log

def to_quaternion(pose):
    if pose.shape[-1] == 3:
        return Rotation.from_rotvec(pose).as_quat()
    elif pose.shape[-1] == 6:
        rot = Rotation.from_rotvec(pose[:, 3:]).as_quat()
        trans = pose[:, :3]
        return np.concatenate([trans, rot], axis=-1)
    else:
        raise ValueError("pose must be either 3 or 6 dimensional")

def to_rotvec(pose):
    if pose.shape[-1] == 4:
        return Rotation.from_quat(pose).as_rotvec()
    elif pose.shape[-1] == 7:
        rot = Rotation.from_quat(pose[:, 3:]).as_rotvec()
        trans = pose[:, :3]
        return np.concatenate([trans, rot], axis=-1)
    else:
        raise ValueError("pose must be either 4 or 7 dimensional")
    
def to_euler(rot):
    if len(rot.shape) >=2 and rot.shape[-1] == 3 and rot.shape[-2] == 3:
        return Rotation.from_matrix(rot).as_euler("xyz")
    # from axis-angle to euler
    if rot.shape[-1] == 3:
        return Rotation.from_rotvec(rot).as_euler("xyz")
    # from quaternion to euler
    elif rot.shape[-1] == 4:
        return Rotation.from_quat(rot).as_euler("xyz")
    else:
        raise ValueError("rot must be either 3 or 4 dimensional")

def to_matrix(rot):
    if len(rot.shape) >=2 and rot.shape[-1] == 3 and rot.shape[-2] == 3:
        return rot
    if rot.shape[-1] == 3:
        return Rotation.from_rotvec(rot).as_matrix()
    elif rot.shape[-1] == 4:
        return Rotation.from_quat(rot).as_matrix()
    else:
        raise ValueError("rot must be either 3 or 4 dimensional")

# Using Deviation from the Identity Matrix(see https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf)
def rotation_metric(rot1, rot2):
    rot1 = to_matrix(rot1) # N, 3, 3
    rot2 = to_matrix(rot2) # N, 3, 3

    return np.linalg.norm(np.eye(3) - rot1 @ rot2.transpose(0, 2, 1), axis=(1, 2))

    
    

def pose_translational(t):
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]], dtype=np.float32
    )


# Translation in UE4
trans_t = lambda t: np.array(
    [[1, 0, 0, t], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)
trans_y = lambda t: np.array(
    [[1, 0, 0, 0], [0, 1, 0, t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)
trans_z = lambda z: np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]], dtype=np.float32
)
# Rotation around z-axis in UE4
rot_theta = lambda th: np.array(
    [
        [np.cos(th), np.sin(th), 0, 0],
        [-np.sin(th), np.cos(th), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)
# Rotation around y-axis in UE4
rot_phi = lambda phi: np.array(
    [
        [np.cos(phi), 0, -np.sin(phi), 0],
        [0, 1, 0, 0],
        [np.sin(phi), 0, np.cos(phi), 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def pose_spherical(theta, radius):
    c2w = trans_z(6.0)
    c2w = rot_phi(-25.0 / 180.0 * np.pi) @ c2w
    c2w = rot_theta(-np.pi) @ c2w
    c2w = trans_t(radius) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w

    c2w = from_ue4_to_nerf(c2w)
    return c2w

def pose_rotational(deg):
    pose = trans_t(-25.)
    pose = rot_theta(deg / 180.0 * np.pi) @ pose
    

    pose = from_ue4_to_nerf(pose).astype(np.float32)
    return pose

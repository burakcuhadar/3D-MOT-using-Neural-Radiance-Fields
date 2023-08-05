import numpy as np
import os
import re
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
    t_inv = np.eye(4, dtype=np.float32)
    t_inv[:3, :3] = t[:3, :3].T
    t_inv[:3, -1] = -t[:3, :3].T @ t[:3, -1]
    return t_inv


def se3_log_map(matrices):
    rot = Rotation.from_matrix(matrices[:, :3, :3]).as_rotvec()
    trans = matrices[:, :3, 3]
    return np.concatenate((trans, rot), axis=-1).astype(np.float32)


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

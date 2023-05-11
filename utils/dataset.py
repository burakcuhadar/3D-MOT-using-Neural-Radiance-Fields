import numpy as np
import os
import re


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

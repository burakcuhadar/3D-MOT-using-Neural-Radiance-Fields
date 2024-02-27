# Taken from: https://github.com/kwea123/nerf_pl/

import torch
import numpy as np
import mcubes
import trimesh
import os
import cv2
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import open3d as o3d

# from plyfile import PlyData, PlyElement #TODO pip install plyfile


### Tune these parameters until the whole object lies tightly in range with little noise ###
N = 256  # controls the resolution, set this number small here because we're only finding
# good ranges here, not yet for mesh reconstruction; we can set this number high
# when it comes to final reconstruction.
xmin, xmax = -0.8, 0.8  # left/right range
ymin, ymax = -0.8, 0.8  # forward/backward range
zmin, zmax = -0.8, 0.8  # up/down range
## Attention! the ranges MUST have the same length!
sigma_threshold = 50.0  # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
occ_threshold = 0.2
############################################################################################


@torch.no_grad()
def f(models, rays, N_samples, N_importance, chunk, white_back):
    result = render_rays(
        models,
        embeddings,
        rays[i : i + chunk],
        N_samples,
        False,
        0,
        0,
        N_importance,
        chunk,
        white_back,
        test_time=True,
    )

    return result


def extract_color_mesh(nerf_fine, dataset, mesh_name):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    # sigma is independent of direction, so any value here will produce the same result

    # predict sigma (occupancy) for each grid location
    print("Predicting occupancy ...")
    with torch.no_grad():
        sigma, rgb = nerf_fine(xyz_[:, None, :], dir_)

    sigma = sigma[:, 0].cpu().numpy()
    sigma = np.maximum(sigma, 0).reshape(N, N, N)

    # perform marching cube algorithm to retrieve vertices and triangle mesh
    print("Extracting mesh ...")
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)

    ##### Until mesh extraction here, it is the same as the original repo. ######

    vertices_ = (vertices / N).astype(np.float32)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    face = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = triangles

    PlyData(
        [
            PlyElement.describe(vertices_[:, 0], "vertex"),
            PlyElement.describe(face, "face"),
        ]
    ).write(f"{mesh_name}.ply")

    # remove noise in the mesh by keeping only the biggest cluster
    print("Removing noise ...")
    mesh = o3d.io.read_triangle_mesh(f"{mesh_name}.ply")
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(
        f"Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces."
    )

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics)
    W, H = dataset.W, dataset.H
    K = np.array(
        [[dataset.focal, 0, W / 2], [0, dataset.focal, H / 2], [0, 0, 1]]
    ).astype(np.float32)

    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1)  # (N, 4)

    ## buffers to store the final averaged color
    non_occluded_sum = np.zeros((N_vertices, 1))
    v_color_sum = np.zeros((N_vertices, 3))

    # Step 2. project the vertices onto each training image to infer the color
    print("Fusing colors ...")
    for idx in tqdm(range(len(dataset.imgs))):
        ## read image of this pose
        image = dataset.imgs[idx, 0]

        ## read the camera to world relative pose
        P_c2w = np.concatenate(
            [dataset.poses[idx], np.array([0, 0, 0, 1]).reshape(1, 4)], 0
        )
        P_w2c = np.linalg.inv(P_c2w)[:3]  # (3, 4)
        ## project vertices from world coordinate to camera coordinate
        vertices_cam = P_w2c @ vertices_homo.T  # (3, N) in "right up back"
        vertices_cam[1:] *= -1  # (3, N) in "right down forward"
        ## project vertices from camera coordinate to pixel coordinate
        vertices_image = (K @ vertices_cam).T  # (N, 3)
        depth = (
            vertices_image[:, -1:] + 1e-5
        )  # the depth of the vertices, used as far plane
        vertices_image = vertices_image[:, :2] / depth
        vertices_image = vertices_image.astype(np.float32)
        vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W - 1)
        vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H - 1)

        ## compute the color on these projected pixel coordinates
        ## using bilinear interpolation.
        ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
        ## so we split the input into chunks.
        colors = []
        remap_chunk = int(3e4)
        for i in range(0, N_vertices, remap_chunk):
            colors += [
                cv2.remap(
                    image,
                    vertices_image[i : i + remap_chunk, 0],
                    vertices_image[i : i + remap_chunk, 1],
                    interpolation=cv2.INTER_LINEAR,
                )[:, 0]
            ]
        colors = np.vstack(colors)  # (N_vertices, 3)

        ## predict occlusion of each vertex
        ## we leverage the concept of NeRF by constructing rays coming out from the camera
        ## and hitting each vertex; by computing the accumulated opacity along this path,
        ## we can know if the vertex is occluded or not.
        ## for vertices that appear to be occluded from every input view, we make the
        ## assumption that its color is the same as its neighbors that are facing our side.
        ## (think of a surface with one side facing us: we assume the other side has the same color)

        ## ray's origin is camera origin
        rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
        ## ray's direction is the vector pointing from camera origin to the vertices
        rays_d = torch.FloatTensor(vertices_) - rays_o  # (N_vertices, 3)
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
        ## the far plane is the depth of the vertices, since what we want is the accumulated
        ## opacity along the path from camera origin to the vertices
        far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
        results = f(
            [nerf_fine],
            embeddings,
            torch.cat([rays_o, rays_d, near, far], 1).cuda(),
            args.N_samples,
            0,
            args.chunk,
            dataset.white_back,
        )
        opacity = (
            results["opacity_coarse"].cpu().numpy()[:, np.newaxis]
        )  # (N_vertices, 1)
        opacity = np.nan_to_num(opacity, 1)

        non_occluded = (
            np.ones_like(non_occluded_sum) * 0.1 / depth
        )  # weight by inverse depth
        # near=more confident in color
        non_occluded += opacity < occ_threshold

        v_color_sum += colors * non_occluded
        non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    v_colors = v_color_sum / non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr + v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]

    face = np.empty(len(triangles), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = triangles

    PlyData(
        [PlyElement.describe(vertex_all, "vertex"), PlyElement.describe(face, "face")]
    ).write(f"{mesh_name}.ply")

    print("Done!")


def extract_mesh(nerf_fine, mesh_name):
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()

    with torch.no_grad():
        sigma, rgb = nerf_fine(xyz_[:, None, :], dir_)

    sigma = sigma[:, 0].cpu().numpy()
    sigma = np.maximum(sigma, 0)
    sigma = sigma.reshape(N, N, N)

    # The below lines are for visualization, COMMENT OUT once you find the best range and increase N!
    vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
    mcubes.export_mesh(vertices, triangles, f"{mesh_name}.dae")

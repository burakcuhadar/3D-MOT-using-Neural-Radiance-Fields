import torch


def compute_depth_loss(depth, gt_depth, near, far):
    # Dont consider the points that are out of our bounding box, depth > far, depth < near
    mask = torch.logical_and(gt_depth < far, gt_depth > near)

    depth_loss = torch.mean(((depth[mask] - gt_depth[mask]) / gt_depth[mask]) ** 2)
    return depth_loss


def compute_sigma_loss(
    weights,
    z_vals,
    dists,
    depths,
    near,
    far,
    err=1,
):
    weights_ = torch.where(weights <= 0, torch.finfo(torch.float32).eps, weights)

    # Dont consider the points that are out of our bounding box, depth > far, depth < near
    mask = torch.logical_and(depths < far, depths > near)

    """
    print("near", near)
    print("far", far)

    print("depths min", torch.min(depths))
    print("depths max", torch.max(depths))
    print("z_vals min", torch.min(z_vals))
    print("z_vals max", torch.max(z_vals))

    print("depths min", torch.min(depths[mask]))
    print("depths max", torch.max(depths[mask]))
    print("z_vals min", torch.min(z_vals[mask]))
    print("z_vals max", torch.max(z_vals[mask]))

    print("z_vals - depths min", torch.min(z_vals - depths[:, None]))
    print("z_vals - depths max", torch.max(z_vals - depths[:, None]))
    print("z_vals - depths min masked", torch.min(z_vals[mask] - depths[mask, None]))
    print("z_vals - depths max masked", torch.max(z_vals[mask] - depths[mask, None]))

    print("dists min", torch.min(dists))
    print("dists max", torch.max(dists))
    print("dists min mask", torch.min(dists[mask]))
    print("dists max mask", torch.max(dists[mask]))

    print("weights min", torch.min(weights))
    print("weights max", torch.max(weights))
    print("weights min mask", torch.min(weights[mask]))
    print("weights max mask", torch.max(weights[mask]))
    """

    loss = (
        -torch.log(weights_[mask])
        * torch.exp(
            -((z_vals[mask] - depths[mask, None]) ** 2) / (2 * err)
        )
        * dists[mask]
    )

    loss = torch.sum(loss, dim=1).mean()

    return loss


# used for debugging in callbacks/check_batch_grad.py
def compute_sigma_loss_per_ray(
    weights,
    z_vals,
    dists,
    depths,
    err=1,
):
    weights_ = torch.where(weights <= 0, torch.finfo(torch.float32).eps, weights)
    loss = (
        -torch.log(weights_)
        * torch.exp(-((z_vals - depths[:, None]) ** 2) / (2 * err))
        * dists
    )
    loss = torch.sum(loss, dim=1)

    return loss


"""
def compute_depth_loss(depth_map, z_vals, weights, target_depth, target_valid_depth):
    pred_mean = depth_map[target_valid_depth]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = ((z_vals[target_valid_depth] - pred_mean.unsqueeze(-1)).pow(2) * weights[target_valid_depth]).sum(-1) + 1e-5
    target_mean = target_depth[..., 0][target_valid_depth]
    target_std = target_depth[..., 1][target_valid_depth]
    apply_depth_loss = is_not_in_expected_distribution(pred_mean, pred_var, target_mean, target_std)
    pred_mean = pred_mean[apply_depth_loss]
    if pred_mean.shape[0] == 0:
        return torch.zeros((1,), device=depth_map.device, requires_grad=True)
    pred_var = pred_var[apply_depth_loss]
    target_mean = target_mean[apply_depth_loss]
    target_std = target_std[apply_depth_loss]
    f = nn.GaussianNLLLoss(eps=0.001)
    return float(pred_mean.shape[0]) / float(target_valid_depth.shape[0]) * f(pred_mean, target_mean, pred_var)


"""

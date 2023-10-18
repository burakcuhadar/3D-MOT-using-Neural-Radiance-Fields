# adapted from https://github.com/google/nerfies/blob/main/nerfies/rigid_body.py

import torch

def rp_to_se3(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Rotation and translation to homogeneous transform.

    Args:
        R: (3, 3) An orthonormal rotation matrix.
        p: (3,) A 3-vector representing an offset.

    Returns:
        X: (4, 4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    p = p.reshape((3, 1))
    return torch.cat([torch.cat([R, p], dim=1), torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=R.device)], dim=0)

def to_homogenous(v):
  return torch.cat([v, torch.ones_like(v[..., :1])], dim=-1)



def skew(w: torch.Tensor) -> torch.Tensor:
    """Build a skew matrix ("cross product matrix") for vector w.

    Modern Robotics Eqn 3.30.

    Args:
        w: (3,) A 3-vector

    Returns:
        W: (3, 3) A skew matrix such that W @ v == w x v
    """
    return torch.tensor([[0.0, -w[2], w[1]],
                        [w[2], 0.0, -w[0]],
                        [-w[1], w[0], 0.0]], device=w.device)


def exp_so3(w: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.

    Args:
        w: (3,) An axis of rotation.
        theta: An angle of rotation.

    Returns:
        R: (3, 3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    W = skew(w)
    return torch.eye(3, device=w.device) + torch.sin(theta) * W + (1.0 - torch.cos(theta)) * W @ W

def exp_se3(S: torch.Tensor, theta: float) -> torch.Tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.

    Modern Robotics Eqn 3.88.

    Args:
        S: (6,) A screw axis of motion.
        theta: Magnitude of motion.

    Returns:
        a_X_b: (4, 4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    w, v = torch.split(S, 3)
    W = skew(w)
    R = exp_so3(w, theta)
    p = (theta * torch.eye(3, device=S.device) + (1.0 - torch.cos(theta)) * W +
        (theta - torch.sin(theta)) * W @ W) @ v
    return rp_to_se3(R, p)


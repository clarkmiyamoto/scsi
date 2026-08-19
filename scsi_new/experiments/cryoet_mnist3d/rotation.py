import torch

########################################################
# 3D pose branch: the 6D continuous rotation representation (Zhou et al., "On the Continuity
# of Rotation Representations in Neural Networks", CVPR 2019 — Gram-Schmidt orthogonalization,
# "GSO"). A rotation is represented as a RAW, unconstrained (B,6) vector; the actual (B,3,3)
# rotation matrix is only materialized where one is actually needed (rotating a volume,
# computing a geodesic error). Deliberately NOT quaternions: representing pose in this flat
# ambient space means the 3D pose branch needs no geodesic interpolant and no manifold-aware
# ODE step at all — it reuses `interpolant()` above verbatim (same function as the image
# branch, just with (B,1) time broadcast instead of (B,1,1,1,1)) and integrates via plain
# Euler, exactly like the image branch. See mnist_cryoem/CLAUDE.md for the full rationale.
########################################################

def gram_schmidt_to_matrix(v6: torch.Tensor) -> torch.Tensor:
    """
    (B, 6) -> (B, 3, 3) orthonormal rotation matrix.

    v6 = [a1(3), a2(3)], raw/unconstrained. b1 = normalize(a1); b2 = normalize(a2 - (b1.a2)*b1)
    (Gram-Schmidt: strip the a1-component out of a2, then normalize); b3 = cross(b1, b2)
    (right-handed by construction, so det(R) = +1 — a proper rotation, never a reflection).
    Columns [b1, b2, b3] form R. Well-defined almost everywhere: degenerate only if a1 = 0 or
    a2 is parallel to a1, a measure-zero set a randomly-initialized or trained network won't hit.
    """
    a1, a2 = v6[..., :3], v6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_gram_schmidt(R: torch.Tensor) -> torch.Tensor:
    """
    (B, 3, 3) -> (B, 6): the first two columns of R, flattened. The canonical representative —
    since R's own columns are already orthonormal, gram_schmidt_to_matrix(matrix_to_gram_schmidt(R))
    recovers R exactly (up to floating point).
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rotation_geodesic_angle(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """
    (B, 3, 3), (B, 3, 3) -> (B,) geodesic angle (radians) between two rotation matrices, via
    the trace formula: angle = arccos(clamp((trace(R_a^T R_b) - 1) / 2, -1, 1)). SO(3) analogue
    of wrap_to_pi-based circular error — used for the 3D pool-vs-truth diagnostic.
    """
    R_rel = torch.matmul(R_a.transpose(-1, -2), R_b)
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7))


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues' rotation formula: rotation by `angle` about a fixed unit `axis`.

    Args:
        axis:  (3,) unit vector (broadcasts across the batch — ONE fixed physical axis)
        angle: (B,) radians
    Returns:
        (B, 3, 3)
    """
    axis = axis / axis.norm()
    zero = torch.zeros((), device=angle.device, dtype=angle.dtype)
    ax, ay, az = axis[0], axis[1], axis[2]
    K = torch.stack([
        torch.stack([zero, -az, ay]),
        torch.stack([az, zero, -ax]),
        torch.stack([-ay, ax, zero]),
    ])  # (3, 3), skew-symmetric cross-product matrix of `axis`
    I3 = torch.eye(3, device=angle.device, dtype=angle.dtype)
    sin_a = torch.sin(angle)[:, None, None]
    cos_a = torch.cos(angle)[:, None, None]
    K = K.unsqueeze(0)  # (1, 3, 3), broadcasts against the batch below
    return I3.unsqueeze(0) + sin_a * K + (1.0 - cos_a) * torch.matmul(K, K)
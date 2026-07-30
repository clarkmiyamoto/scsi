"""
SO(3)/Gram-Schmidt correctness checks for the 3D/CryoET generalization.

Run as a script (asserts + summary, non-zero exit on failure):

    uv run python test_so3_math.py

The `test_*` functions are also pytest-compatible. This is the validation gate described in
mnist_cryoem/CLAUDE.md — run it to green BEFORE wiring up pretrain_3d.py or main_3d.py. It is
narrower than a quaternion-based test suite would need (the 6D representation needs no
slerp/frame-convention checks — see si.py's module docstring), but the tilt-composition-order
and clipping checks remain essential: a bug in either produces no shape error and no obvious
symptom until reconstructions are qualitatively wrong.
"""
import torch

from si import (
    gram_schmidt_to_matrix, matrix_to_gram_schmidt, interpolant,
)
from corruption import sample_uniform_rotation_so3, sample_tilt_series_rotations_so3, project_2d
from data import load_mnist_volumes_3d

torch.manual_seed(0)


def test_gram_schmidt_validity():
    """Random (B,6) Gaussian inputs -> gram_schmidt_to_matrix output is always a proper
    rotation: R^T R = I (orthonormal) and det(R) = +1 (not a reflection)."""
    v6 = torch.randn(64, 6)
    R = gram_schmidt_to_matrix(v6)
    I = torch.eye(3).expand(64, 3, 3)
    orth_err = (torch.matmul(R.transpose(-1, -2), R) - I).abs().max().item()
    det_err = (torch.det(R) - 1.0).abs().max().item()
    assert orth_err < 1e-5, f"R^T R != I, max err {orth_err:.2e}"
    assert det_err < 1e-4, f"det(R) != 1, max err {det_err:.2e}"
    return max(orth_err, det_err)


def test_gram_schmidt_round_trip():
    """For a genuine rotation matrix R (Haar-uniform), gram_schmidt_to_matrix(
    matrix_to_gram_schmidt(R)) recovers R exactly — its own columns are already orthonormal."""
    R = sample_uniform_rotation_so3(64)
    R_rt = gram_schmidt_to_matrix(matrix_to_gram_schmidt(R))
    err = (R_rt - R).abs().max().item()
    assert err < 1e-4, f"round-trip mismatch, max err {err:.2e}"
    return err


def test_tilt_composition_order():
    """
    Pulling the FIXED lab-frame tilt_axis back through R_total^{-1} (into "just-mounted,
    pre-tilt" coordinates) must give the SAME vector for every tilt within one acquisition.

    Why R^{-1}, not R: with the correct order R_total(angle) = R_tilt(angle) @ R_mount,
    R_total(angle)^{-1} @ tilt_axis = R_mount^T @ (R_tilt(-angle) @ tilt_axis) = R_mount^T @
    tilt_axis (R_tilt rotates ABOUT tilt_axis by construction, so it fixes tilt_axis itself) —
    independent of `angle`. With the WRONG order (mount outer, tilt inner), this quantity
    traces a circle as `angle` varies instead of staying fixed — so this check does discriminate
    the two orders, unlike checking R_total @ tilt_axis directly (which is NOT invariant even
    for the correct order, since tilt_axis lives in the lab frame, not the mounted frame).
    """
    tilt_axis = (0.0, 1.0, 0.0)
    R = sample_tilt_series_rotations_so3(
        n_acquisitions=8, n_tilts=6, tilt_increment=0.1, tilt_axis=tilt_axis,
    )  # (8, 6, 3, 3)
    axis = torch.tensor(tilt_axis)
    # R^{-1} @ axis == R^T @ axis (R is orthogonal) -- einsum "atji,j->ati" transposes R in place.
    mapped = torch.einsum("atji,j->ati", R, axis)  # (8, 6, 3)
    ref = mapped[:, :1, :]  # first tilt of each acquisition
    err = (mapped - ref).abs().max().item()
    assert err < 1e-4, f"tilt axis not shared across one acquisition's tilts, max err {err:.2e}"
    return err


def test_mass_invariance_no_clipping():
    """Rotating a volume must not change its total projected mass — if it does, the digit's
    support is clipping against the cube's boundary under some rotations (bad margin defaults),
    or the black-fill/rotation convention is wrong."""
    from corruption import rotate_3d

    x = load_mnist_volumes_3d(n_images_per_class=4, digit_classes=[0, 3, 7])
    mass0 = project_2d(x).sum(dim=(1, 2, 3))  # (N,)

    max_rel_err = 0.0
    for _ in range(20):
        R = sample_uniform_rotation_so3(x.size(0))
        x_rot = rotate_3d(x, R)
        mass_rot = project_2d(x_rot).sum(dim=(1, 2, 3))
        rel_err = ((mass_rot - mass0).abs() / mass0.abs().clamp_min(1e-3)).max().item()
        max_rel_err = max(max_rel_err, rel_err)
    assert max_rel_err < 0.05, (
        f"projected mass not rotation-invariant, max relative err {max_rel_err:.3f} "
        f"(check load_mnist_volumes_3d's inplane_size/depth_extent margins)"
    )
    return max_rel_err


def test_pose_interpolant_integrator_roundtrip():
    """For style='linear', si.interpolant's pose_dot_t is CONSTANT in t (alpha_dot/beta_dot are
    constants), so integrating the pose branch's plain-Euler update
    (pose = pose + pose_dot_t * dt) from pose_z with that constant velocity must land on
    pose_hat exactly at t=1, regardless of step count — this is the same interpolant/integrator
    pairing used everywhere else in this codebase (image branch included), just checked here
    for the newly-added pose usage."""
    B = 16
    pose_z = torch.randn(B, 6)
    pose_hat = torch.randn(B, 6)
    t0 = torch.zeros(B, 1)
    _, pose_dot_t = interpolant(pose_z, pose_hat, t0, style="linear")

    pose = pose_z.clone()
    n_steps = 37
    dt = 1.0 / n_steps
    for _ in range(n_steps):
        pose = pose + pose_dot_t * dt

    err = (pose - pose_hat).abs().max().item()
    assert err < 1e-4, f"pose integrator round-trip mismatch, max err {err:.2e}"
    return err


def main() -> int:
    checks = [
        ("Gram-Schmidt validity (orthonormal, det=+1)", test_gram_schmidt_validity),
        ("Gram-Schmidt round-trip", test_gram_schmidt_round_trip),
        ("Tilt composition order (single shared axis)", test_tilt_composition_order),
        ("Mass invariance / no clipping", test_mass_invariance_no_clipping),
        ("Pose interpolant/integrator round-trip", test_pose_interpolant_integrator_roundtrip),
    ]
    failures = 0
    for name, fn in checks:
        try:
            extra = fn()
            note = f"  (max err {extra:.2e})" if isinstance(extra, float) else ""
            print(f"  PASS  {name}{note}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"[test_so3_math] {len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

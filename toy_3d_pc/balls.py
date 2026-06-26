"""Turn a point cloud into solid geometry: place a ball at every point.

Builds a triangle mesh that is the union of one small **icosphere** per point and
writes it as a Wavefront ``.obj`` (renders as a solid surface in the W&B 3D viewer
or any mesh tool); can also rasterize a quick local PNG preview. This is purely a
visualization aid for SCSI samples -- distinct from the ``ball`` *forward-model*
splat in :mod:`corruption`, which renders the analytic projection of a solid ball.
"""
from __future__ import annotations

import numpy as np


def icosphere(subdivisions: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Unit sphere as a subdivided icosahedron -> (verts (V,3), faces (F,3))."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    def normalize(v: list[float]) -> list[float]:
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        return [v[0] / n, v[1] / n, v[2] / n]

    verts = [normalize(v) for v in verts]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in cache:
            return cache[key]
        vi, vj = verts[i], verts[j]
        verts.append(normalize([(vi[k] + vj[k]) / 2 for k in range(3)]))
        cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(subdivisions):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def point_cloud_to_balls(
    points: np.ndarray, radius: float = 0.05, subdivisions: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Union mesh: one icosphere of ``radius`` centered at each input point."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    sv, sf = icosphere(subdivisions)
    n, v = points.shape[0], sv.shape[0]
    verts = (radius * sv[None] + points[:, None, :]).reshape(-1, 3)
    faces = (sf[None] + (np.arange(n) * v)[:, None, None]).reshape(-1, 3)
    return verts.astype(np.float32), faces.astype(np.int64)


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle mesh to a Wavefront .obj (faces are 1-indexed in OBJ)."""
    with open(path, "w") as fh:
        np.savetxt(fh, verts, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, faces + 1, fmt="f %d %d %d")


def save_balls_obj(
    points: np.ndarray, path: str, radius: float = 0.05, subdivisions: int = 1
) -> tuple[int, int]:
    """Build the ball mesh for ``points`` and write it to ``path``. Returns (V, F)."""
    verts, faces = point_cloud_to_balls(points, radius, subdivisions)
    write_obj(path, verts, faces)
    return verts.shape[0], faces.shape[0]


def _shade(tris: np.ndarray, base=(0.30, 0.61, 0.91)) -> np.ndarray:
    """Lambert shading per triangle for a nicer PNG. tris: (F,3,3) -> (F,4) RGBA."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    light = np.array([0.4, 0.5, 0.9])
    light /= np.linalg.norm(light)
    intensity = np.clip(n @ light, 0.0, 1.0) * 0.65 + 0.35
    rgb = np.clip(np.asarray(base)[None] * intensity[:, None], 0, 1)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)


def render_balls_png(
    clouds: list[np.ndarray],
    path: str,
    radius: float = 0.05,
    subdivisions: int = 1,
    max_balls: int = 200,
    lim: float = 1.6,
) -> None:
    """Render a grid of ball meshes to a PNG (matplotlib, headless)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    m = len(clouds)
    cols = min(m, 4)
    rows = (m + cols - 1) // cols
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    for i, pts in enumerate(clouds):
        pts = np.asarray(pts).reshape(-1, 3)
        if pts.shape[0] > max_balls:  # subsample for the preview only
            idx = np.random.default_rng(0).choice(pts.shape[0], max_balls, replace=False)
            pts = pts[idx]
        verts, faces = point_cloud_to_balls(pts, radius, subdivisions)
        tris = verts[faces]
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        coll = Poly3DCollection(tris, facecolors=_shade(tris), edgecolor="none")
        ax.add_collection3d(coll)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"sample {i}")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[balls] wrote {path}")

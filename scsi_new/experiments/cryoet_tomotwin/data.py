"""
TomoTwin-100 ground-truth volumes as a SCSI clean-signal pool, plus the CryoET tilt-series
forward channel -- the "real biomolecule" counterpart of cryoet_mnist3d/data.py (extruded EMNIST
digits). Same skeleton: a Config dataclass, load_*_volumes -> (N, 1, V, V, V) in [-1, 1],
build_observations -> TensorDataset of tilt series, build_viz_pool, and a __main__ that opens an
interactive 3D window (marching-cubes isosurfaces + a strip of parallel-beam projections).

WHAT TOMOTWIN-100 IS
    The compositional-heterogeneity half of CryoBench (Jeon et al., 2024; arXiv:2408.05526),
    Zenodo record 12528292. 100 distinct biomolecular complexes (the TomoTwin reference set of
    Rice et al., 2023 -- PDB entries such as 1bxn / RuBisCO, 6z80, ...), each rendered from its
    atomic model with ChimeraX `molmap` and downsampled to a 128^3 box at 4.5 A/voxel
    (cella = 576 A; the header is authoritative and is read per file). Densities are
    non-negative with an exact-zero background and only ~0.2% occupied voxels -- very sparse,
    which is why every volume DEFLATE-compresses to ~30-85 KB inside the archive.

    Archive layout (Tomotwin-100.zip, 6.2 GB):
        Tomotwin-100/vols/128_org/NNN_pdbid.mrc     100 GT volumes, MRC mode 2, 128^3   <- used here
        Tomotwin-100/images/snr0.01/*_particles.mrcs 100 stacks x 1000 imgs, 128^2
        Tomotwin-100/combined_poses.pkl              (rots (1e5,3,3), trans (1e5,2))
        Tomotwin-100/combined_ctfs.pkl               per-particle CTF params
        Tomotwin-100/gt_latents.pkl                  (1e5,) int64 conformation label (1000/class)
        Tomotwin-100/init_mask/{init,mask,backproj_snr001}.mrc

    Only vols/128_org/*.mrc is needed for the SCSI clean-signal pool, and it is ~7 MB total
    once compressed, so by default this module range-requests just those members straight out
    of the remote zip (no 6.2 GB download). `download_tomotwin_full()` / `--full` fetches the
    whole archive when the real particle stacks / poses are actually wanted (a supervised
    baseline); this file never uses them.

FORWARD CHANNEL
    corruption_channel / _rotate_3d / _project_2d below are copied from
    cryoet_mnist3d/{corruption.py, rotation.py} verbatim in behaviour -- inlined so this stays a
    single self-contained file. When cryoet_tomotwin/ grows the full 6-file skeleton, split them
    back out into corruption.py + rotation.py and delete the copies here.

NORMALISATION
    Raw density is [0, dmax] with background 0. We map background 0 -> -1 and a shared
    `density_scale` -> +1 (clamped), matching the repo-wide [-1, 1] / background = -1 convention
    (rotate_3d's +1 shift, the SI interpolant, the E-step re-corruption all assume it). The
    default scale is the max density over the loaded pool (printed for inspection). A per-volume
    p99.9 -- what cryoet_mnist3d uses -- does NOT transfer here: at 0.2% occupancy the 99.9th
    percentile over all voxels sits near the *median* occupied voxel and would clip half the
    density into saturation.
"""

from __future__ import annotations

import io
import json
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset

# --------------------------------------------------------------------------------------------- #
# Constants                                                                                     #
# --------------------------------------------------------------------------------------------- #

ZENODO_URL = "https://zenodo.org/records/12528292/files/Tomotwin-100.zip?download=1"
ZIP_ROOT = "Tomotwin-100"
VOLS_PREFIX = f"{ZIP_ROOT}/vols/128_org/"
NATIVE_VOL_SIZE = 128
N_COMPLEXES = 100

# Data lives next to this file (experiments/cryoet_tomotwin/data/), so `uv run data.py` behaves
# the same from any cwd. `data/` is gitignored repo-wide.
DATA_ROOT = Path(__file__).resolve().parent / "data"

_MRC_DTYPE = {0: "<i1", 1: "<i2", 2: "<f4", 6: "<u2", 12: "<f2"}
_MRC_DTYPE_BE = {0: ">i1", 1: ">i2", 2: ">f4", 6: ">u2", 12: ">f2"}


@dataclass
class Config_Dataset_Tomotwin:
    # Dataset ---------------------------------------------------------------------------------- #
    n_volumes: int | None = None            # first N of the 100 complexes (dataset order); None -> all
    pdb_ids: list[str] | None = None        # e.g. ["1bxn", "6z80"]; overrides n_volumes, keeps dataset order
    vol_size: int = NATIVE_VOL_SIZE         # trilinear-downsampled if < 128 (never upsampled)
    density_scale: float | None = None      # +1 maps here; None -> max density over the loaded pool

    # Corruption channel (tilt series: mount-rotate -> tilt -> parallel project -> AWGN) ------- #
    num_tilts: int = 16
    tilt_increment_deg: float = 7.5         # matches cryoet_mnist3d default
    noise_std: float = 0.3                  # NOT comparable to cryoet_mnist3d: a ray here sums 128
                                            # sparse ~1.5-peak voxels, not 32 dense [-1,1] ones.
    tilt_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Misc ----------------------------------------------------------------------------------- #
    seed: int = 42
    data_root: Path = DATA_ROOT


# --------------------------------------------------------------------------------------------- #
# Remote-zip partial extraction (range requests -- avoids the 6.2 GB download)                  #
# --------------------------------------------------------------------------------------------- #

def _http(session, method: str, url: str, *, headers: dict | None = None, stream: bool = False,
          retries: int = 8, timeout: int = 120):
    """`session.get`/`session.head` with exponential backoff over Zenodo's frequent 5xx and
    connection timeouts. Returns the Response (raised-for-status)."""
    import requests

    last = None
    for i in range(retries):
        try:
            r = session.request(method, url, headers=headers, allow_redirects=True,
                                timeout=timeout, stream=stream)
            r.raise_for_status()
            return r
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            last = e
            wait = 3 + 3 * i
            print(f"  [zenodo] {type(e).__name__}, retry {i + 1}/{retries} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Zenodo request failed after {retries} tries: {last}")


def _range(session, url: str, start: int, end: int) -> bytes:
    """Inclusive byte range [start, end]."""
    r = _http(session, "GET", url, headers={"Range": f"bytes={start}-{end}"})
    return r.content


def _zip_index_path(data_root: Path) -> Path:
    return data_root / ZIP_ROOT / ".zip_index.json"


def _load_zip_index(session, url: str, data_root: Path) -> dict[str, list]:
    """
    {member_name: [method, comp_size, uncomp_size, local_header_offset]} for every entry in the
    remote archive's central directory. Parsed once (EOCD -> ZIP64 EOCD -> central directory,
    three range requests) then cached to data/Tomotwin-100/.zip_index.json, so repeat runs -- and
    fully-offline runs once the .mrc files exist -- touch the network zero times.
    """
    cache = _zip_index_path(data_root)
    if cache.exists():
        return json.loads(cache.read_text())

    size = int(_http(session, "HEAD", url, timeout=60).headers["Content-Length"])
    tail = _range(session, url, size - 65536, size - 1)

    eocd = tail.rfind(b"PK\x05\x06")
    if eocd == -1:
        raise RuntimeError("no End-Of-Central-Directory record in archive tail")
    total, cd_size, cd_off = struct.unpack("<HII", tail[eocd + 10:eocd + 20])

    if cd_off == 0xFFFFFFFF or total == 0xFFFF:  # ZIP64
        loc = tail.rfind(b"PK\x06\x07")
        z64_off = struct.unpack("<Q", tail[loc + 8:loc + 16])[0]
        z = _range(session, url, z64_off, z64_off + 55)
        total, cd_size, cd_off = struct.unpack("<QQQ", z[32:56])

    cd = io.BytesIO(_range(session, url, cd_off, cd_off + cd_size - 1))
    index: dict[str, list] = {}
    while cd.read(4) == b"PK\x01\x02":
        (_, _, _, method, _, _, _, csize, usize,
         nlen, elen, clen, _, _, _, lho) = struct.unpack("<HHHHHHIIIHHHHHII", cd.read(42))
        name = cd.read(nlen).decode("utf-8", "replace")
        extra = cd.read(elen)
        cd.read(clen)
        if 0xFFFFFFFF in (usize, csize, lho):  # patch from the ZIP64 extra field
            ei = io.BytesIO(extra)
            while ei.tell() < len(extra):
                tag, sz = struct.unpack("<HH", ei.read(4))
                blk = ei.read(sz)
                if tag == 0x0001:
                    vals = iter(struct.unpack("<" + "Q" * (len(blk) // 8), blk))
                    if usize == 0xFFFFFFFF:
                        usize = next(vals)
                    if csize == 0xFFFFFFFF:
                        csize = next(vals)
                    if lho == 0xFFFFFFFF:
                        lho = next(vals)
        index[name] = [method, csize, usize, lho]

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index))
    return index


def _extract_member(session, url: str, index: dict[str, list], name: str) -> bytes:
    """Raw (decompressed) bytes of one archive member, via a single range request for its
    local header + compressed payload."""
    method, csize, _usize, lho = index[name]
    lh = _range(session, url, lho, lho + 29)
    if lh[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {name!r}")
    nlen, elen = struct.unpack("<HH", lh[26:30])
    start = lho + 30 + nlen + elen
    comp = _range(session, url, start, start + csize - 1)
    if method == 0:
        return comp
    if method == 8:
        return zlib.decompress(comp, -15)
    raise RuntimeError(f"unsupported compression method {method} for {name!r}")


def _volume_members(index: dict[str, list]) -> list[str]:
    """vols/128_org/*.mrc member names in dataset order (NNN_pdbid.mrc -> sorted by NNN)."""
    return sorted(n for n in index
                  if n.startswith(VOLS_PREFIX) and n.endswith(".mrc") and n != VOLS_PREFIX)


def _select_members(members: list[str], config: Config_Dataset_Tomotwin) -> list[str]:
    if config.pdb_ids is not None:
        want = {p.lower() for p in config.pdb_ids}
        picked = [m for m in members if Path(m).stem.split("_", 1)[1].lower() in want]
        missing = want - {Path(m).stem.split("_", 1)[1].lower() for m in picked}
        if missing:
            raise ValueError(f"pdb_ids not in TomoTwin-100: {sorted(missing)}")
        return picked
    if config.n_volumes is not None:
        return members[:config.n_volumes]
    return members


def download_tomotwin_volumes(config: Config_Dataset_Tomotwin) -> list[Path]:
    """
    Ensure the selected ground-truth .mrc volumes exist under
    data/Tomotwin-100/vols/128_org/, range-extracting any that are missing straight out of the
    remote archive (~30-85 KB compressed each; ~7 MB for all 100). Returns their local paths in
    dataset order. No-ops entirely -- no network -- once the files are present.
    """
    data_root = Path(config.data_root)
    (data_root / VOLS_PREFIX).mkdir(parents=True, exist_ok=True)

    # The cached .zip_index.json is the authoritative member list, so the selection is resolved
    # the same way whether or not the .mrc files exist yet. Fully offline when the cache is
    # present AND every selected file is already on disk.
    cache = _zip_index_path(data_root)
    index = json.loads(cache.read_text()) if cache.exists() else None
    if index is not None:
        picked = _select_members(_volume_members(index), config)
        paths = [data_root / m for m in picked]
        if all(p.exists() for p in paths):
            return paths

    import requests

    with requests.Session() as session:
        if index is None:
            index = _load_zip_index(session, ZENODO_URL, data_root)
            picked = _select_members(_volume_members(index), config)
            paths = [data_root / m for m in picked]
        if not picked:
            raise RuntimeError("no volumes selected -- check n_volumes / pdb_ids")
        todo = [(m, p) for m, p in zip(picked, paths) if not p.exists()]
        for i, (name, dst) in enumerate(todo):
            print(f"  [{i + 1:>3}/{len(todo)}] {name}", flush=True)
            dst.write_bytes(_extract_member(session, ZENODO_URL, index, name))
    return paths


def download_tomotwin_full(config: Config_Dataset_Tomotwin, *, keep_zip: bool = True) -> Path:
    """
    Stream the ENTIRE 6.2 GB archive to data/Tomotwin-100.zip and unpack it (particle stacks,
    poses, CTFs, latents). Only needed for a pose-supervised baseline -- nothing in this module
    consumes images/ or the .pkl files. Returns the extracted data/Tomotwin-100/ directory.
    """
    import requests

    data_root = Path(config.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{ZIP_ROOT}.zip"
    out_dir = data_root / ZIP_ROOT

    if not out_dir.exists() or not any(out_dir.rglob("*.mrcs")):
        with requests.Session() as session:
            r = _http(session, "GET", ZENODO_URL, stream=True, timeout=120)
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            with open(zip_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  downloading {done / 1e9:5.2f} / {total / 1e9:.2f} GB", end="",
                              flush=True)
            print()
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_root)
        if not keep_zip:
            zip_path.unlink()
    return out_dir


# --------------------------------------------------------------------------------------------- #
# MRC reader (mode 0/1/2/6/12, little- or big-endian, skips the extended header)                #
# --------------------------------------------------------------------------------------------- #

def _read_mrc(path: Path) -> tuple[np.ndarray, float]:
    """
    Returns (volume (nz, ny, nx) float32, voxel_size_angstrom). MRC2014: 1024-byte header,
    int32 nx/ny/nz/mode at bytes 0-16, int32 nsymbt (extended-header length) at 92-96,
    float32 cella (A) at 40-52, mx/my/mz at 28-40.
    """
    raw = path.read_bytes()
    hdr = raw[:1024]
    nx, ny, nz, mode = struct.unpack("<iiii", hdr[0:16])
    dtypes = _MRC_DTYPE
    if mode not in dtypes:  # try the other byte order before giving up
        nx, ny, nz, mode = struct.unpack(">iiii", hdr[0:16])
        dtypes = _MRC_DTYPE_BE
        if mode not in dtypes:
            raise ValueError(f"{path.name}: unsupported MRC mode {mode}")
    nsymbt = struct.unpack(dtypes[2][0] + "i", hdr[92:96])[0]
    mx, my, mz = struct.unpack(dtypes[2][0] + "iii", hdr[28:40])
    xlen, ylen, zlen = struct.unpack(dtypes[2][0] + "fff", hdr[40:52])
    apix = float(xlen / mx) if mx else 0.0

    off = 1024 + nsymbt
    arr = np.frombuffer(raw[off:off + nx * ny * nz * np.dtype(dtypes[mode]).itemsize],
                        dtype=dtypes[mode]).reshape(nz, ny, nx)
    return np.array(arr, dtype=np.float32), apix  # np.array(copy) -> writable, torch-safe


# --------------------------------------------------------------------------------------------- #
# Volume pool                                                                                   #
# --------------------------------------------------------------------------------------------- #

def list_tomotwin_ids(config: Config_Dataset_Tomotwin) -> list[str]:
    """`NNN_pdbid` stems for the selected volumes, in the same order as load_tomotwin_volumes."""
    return [p.stem for p in download_tomotwin_volumes(config)]


def load_tomotwin_volumes(config: Config_Dataset_Tomotwin) -> torch.Tensor:
    """
    Selected GT volumes as (N, 1, V, V, V) float32 in [-1, 1], background -1. Downloads the .mrc
    files if needed, trilinear-downsamples to config.vol_size when it is below the native 128,
    then maps raw density [0, density_scale] -> [-1, 1] with a shared scale (config.density_scale
    or the pool max).
    """
    paths = download_tomotwin_volumes(config)

    raw = []
    for p in paths:
        vol, _apix = _read_mrc(p)
        raw.append(torch.from_numpy(vol))
    x = torch.stack(raw).unsqueeze(1)  # (N, 1, 128, 128, 128), raw density >= 0

    V = config.vol_size
    if V < x.shape[-1]:
        x = F.interpolate(x, size=(V, V, V), mode="trilinear", align_corners=False)
    elif V > x.shape[-1]:
        raise ValueError(f"vol_size {V} > native {x.shape[-1]}; upsampling not supported")

    scale = config.density_scale if config.density_scale is not None else float(x.max())
    x = (x / max(scale, 1e-8)).clamp(0.0, 1.0) * 2.0 - 1.0
    return x.clamp(-1.0, 1.0).contiguous()


# --------------------------------------------------------------------------------------------- #
# CryoET forward channel                                                                        #
# Behavioural copy of cryoet_mnist3d/{rotation.py, corruption.py} -- see module docstring.      #
# --------------------------------------------------------------------------------------------- #

def _sample_uniform_rotation_so3(B: int, device=None) -> torch.Tensor:
    """Haar-uniform (B, 3, 3): unit quaternion from a normalised 4D Gaussian -> rotation matrix."""
    q = torch.randn(B, 4, device=device)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotate by each `angle` (rad) about one fixed `axis`, given in grid-sample
    (W, H, D) order. Returns (N, 3, 3)."""
    axis = axis / axis.norm().clamp_min(1e-8)
    ax, ay, az = axis.unbind()
    zero = torch.zeros((), device=angle.device, dtype=angle.dtype)
    K = torch.stack([
        torch.stack([zero, -az, ay]),
        torch.stack([az, zero, -ax]),
        torch.stack([-ay, ax, zero]),
    ])
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    s = torch.sin(angle)[:, None, None]
    c = torch.cos(angle)[:, None, None]
    return eye + s * K + (1 - c) * (K @ K)


def _sample_tilt_series_rotations_so3(n_acq: int, n_tilts: int, tilt_increment: float,
                                      tilt_axis=(0.0, 1.0, 0.0), device=None) -> torch.Tensor:
    """R_total = R_tilt(angle_i) @ R_mount: one Haar mount rotation per acquisition (inner),
    then n_tilts steps of `tilt_increment` rad about the fixed lab-frame `tilt_axis` (outer),
    from an independent uniform start offset. Returns (n_acq, n_tilts, 3, 3)."""
    R_mount = _sample_uniform_rotation_so3(n_acq, device)
    start = torch.rand(n_acq, device=device) * 2 * torch.pi
    steps = torch.arange(n_tilts, device=device, dtype=start.dtype) * tilt_increment
    angles = (start[:, None] + steps[None, :]).reshape(-1)
    axis = torch.tensor(tilt_axis, device=device, dtype=start.dtype)
    R_tilt = _axis_angle_to_matrix(axis, angles).reshape(n_acq, n_tilts, 3, 3)
    return R_tilt @ R_mount[:, None, :, :]


def _rotate_3d(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate each (1, D, H, W) volume by its own R, background filled with true -1 (shift to the
    x+1 frame where zero-padding IS the background, sample, shift back). affine_grid/grid_sample
    index the grid as (W, H, D), so R acts in that order."""
    zeros = torch.zeros(R.size(0), 3, 1, device=x.device, dtype=R.dtype)
    theta = torch.cat([R, zeros], dim=2)
    grid = F.affine_grid(theta, x.shape, align_corners=True)
    x_rot = F.grid_sample(x + 1.0, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
    return x_rot - 1.0


def _project_2d(x: torch.Tensor) -> torch.Tensor:
    """Parallel-beam projection: integrate along depth (axis -3). (B, 1, D, H, W) -> (B, 1, H, W).

    Summed in the [-1, 1] frame like cryoet_mnist3d/corruption.py, so every projection carries a
    ~-V DC pedestal from the background (-1 per voxel). A backprojection warm start subtracts it
    (cryoet_mnist3d/pseudoinverse.py does exactly this with a -vol_size pedestal); the __main__
    preview just lets imshow autoscale past it.
    """
    return x.sum(dim=-3)


def _channel_batch(vol_size: int, num_tilts: int, budget_bytes: int = 256 << 20) -> int:
    """Largest B keeping the B*T rotated-volume intermediate under `budget_bytes` (grid_sample
    materialises all B*T volumes at once). ~2 at 128^3/T=16, ~128 at 32^3."""
    per = num_tilts * vol_size ** 3 * 4
    return max(1, budget_bytes // max(per, 1))


def _apply_channel(x: torch.Tensor, rotations: torch.Tensor, noise_std: float,
                   chunk: int) -> torch.Tensor:
    """rotate -> project -> AWGN for a (B, 1, D, H, W) pool against a (B, T, 3, 3) tilt series,
    looped over the batch in `chunk`-sized pieces. Returns (B, T, 1, H, W)."""
    out = []
    for i in range(0, x.size(0), chunk):
        xs, R = x[i:i + chunk], rotations[i:i + chunk]
        B, _, D, H, W = xs.shape
        T = R.size(1)
        xs_rep = xs.unsqueeze(1).expand(-1, T, -1, -1, -1, -1).reshape(B * T, 1, D, H, W)
        y = _project_2d(_rotate_3d(xs_rep, R.reshape(B * T, 3, 3)))       # (B*T, 1, H, W)
        y = y + noise_std * torch.randn_like(y)
        out.append(y.reshape(B, T, 1, H, W))
    return torch.cat(out, dim=0)


def corruption_channel(x: torch.Tensor, num_tilts: int = 16, tilt_increment_deg: float = 7.5,
                       noise_std: float = 0.3, tilt_axis=(0.0, 1.0, 0.0),
                       rotations: torch.Tensor | None = None) -> torch.Tensor:
    """Black-box forward model: draw a fresh random tilt series (Haar mount + evenly spaced tilts
    about a fixed axis) unless `rotations` (B, T, 3, 3) is given, then rotate -> project -> AWGN.
    (B, 1, D, H, W) in [-1, 1] -> (B, T, 1, H, W)."""
    if rotations is None:
        inc = tilt_increment_deg * torch.pi / 180.0
        rotations = _sample_tilt_series_rotations_so3(x.size(0), num_tilts, inc,
                                                     tilt_axis=tilt_axis, device=x.device)
    chunk = _channel_batch(x.shape[-1], rotations.size(1))
    return _apply_channel(x, rotations, noise_std, chunk)


# --------------------------------------------------------------------------------------------- #
# SCSI dataset assembly                                                                         #
# --------------------------------------------------------------------------------------------- #

def build_observations(config: Config_Dataset_Tomotwin) -> Dataset:
    """Load the volume pool and apply the forward model once per volume, chunked.
    Returns a TensorDataset of one (N, num_tilts, 1, V, V) observation tensor."""
    vol_gt = load_tomotwin_volumes(config)
    inc = config.tilt_increment_deg * torch.pi / 180.0
    chunk = _channel_batch(config.vol_size, config.num_tilts)

    obs = []
    for i in range(0, vol_gt.size(0), chunk):
        R = _sample_tilt_series_rotations_so3(min(chunk, vol_gt.size(0) - i), config.num_tilts,
                                             inc, tilt_axis=config.tilt_axis)
        obs.append(_apply_channel(vol_gt[i:i + chunk], R, config.noise_std, chunk))
    return TensorDataset(torch.cat(obs, dim=0))


def build_viz_pool(config: Config_Dataset_Tomotwin, n_pool: int, viz_seed: int) -> dict:
    """Small diagnostic pool for wandb viz: n_pool volumes at deterministic positions in the
    SAME pool build_observations() draws from, with x0 / rotations / y seeded independently by
    viz_seed. Runs on CPU with the RNG state saved/restored. `rotations` is (B, T, 3, 3) -- the
    3D analogue of the 2D pool's scalar theta, named to match corruption_channel's kwarg."""
    vol_gt = load_tomotwin_volumes(config)
    idx = torch.linspace(0, vol_gt.size(0) - 1, min(n_pool, vol_gt.size(0))).round().long()
    x_gt = vol_gt[idx]

    rng_state = torch.get_rng_state()
    torch.manual_seed(viz_seed)
    V = config.vol_size
    x0 = torch.randn(x_gt.size(0), 1, V, V, V)
    inc = config.tilt_increment_deg * torch.pi / 180.0
    rotations = _sample_tilt_series_rotations_so3(x_gt.size(0), config.num_tilts, inc,
                                                 tilt_axis=config.tilt_axis)
    y = _apply_channel(x_gt, rotations, config.noise_std,
                       _channel_batch(V, config.num_tilts))
    torch.set_rng_state(rng_state)

    return {"x_gt": x_gt, "x0": x0, "rotations": rotations, "y": y,
            "ids": [list_tomotwin_ids(config)[i] for i in idx.tolist()]}


# --------------------------------------------------------------------------------------------- #
# __main__ : interactive 3D preview                                                             #
# --------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    # One rotatable matplotlib figure: top row = marching-cubes isosurface of each selected GT
    # volume (all 3D panels share a view -- drag any to orbit all, scroll to zoom); below it, a
    # clean parallel-beam projection and then `--n_proj` noisy tilt-series projections straight
    # from corruption_channel. `--save PATH` also writes a static PNG (headless-safe).
    import argparse

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  -- registers the "3d" projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes

    parser = argparse.ArgumentParser(description="Preview TomoTwin-100 GT volumes + projections")
    parser.add_argument("--pdb_ids", type=str, nargs="+", default=None,
                        help="e.g. 1bxn 6z80 ; overrides --n_volumes")
    parser.add_argument("--n_volumes", type=int, default=4, help="first N complexes to preview")
    parser.add_argument("--vol_size", type=int, default=NATIVE_VOL_SIZE,
                        help="trilinear-downsample below the native 128 for a faster preview")
    parser.add_argument("--num_tilts", type=int, default=16)
    parser.add_argument("--tilt_increment_deg", type=float, default=7.5)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--n_proj", type=int, default=3, help="tilt projections shown per volume")
    parser.add_argument("--iso_frac", type=float, default=0.15,
                        help="isosurface level as a fraction of the pool's max density "
                             "(pre-normalisation); ignored if --level is given")
    parser.add_argument("--level", type=float, default=None,
                        help="explicit isosurface level in [-1, 1] (background -1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true",
                        help="also download+extract the entire 6.2 GB archive (images/poses/ctf)")
    parser.add_argument("--save", type=str, default=None, help="also write a static PNG here")
    parser.add_argument("--no_show", action="store_true", help="skip the interactive window")
    args = parser.parse_args()

    config = Config_Dataset_Tomotwin(
        n_volumes=None if args.pdb_ids else args.n_volumes, pdb_ids=args.pdb_ids,
        vol_size=args.vol_size, num_tilts=args.num_tilts,
        tilt_increment_deg=args.tilt_increment_deg, noise_std=args.noise_std, seed=args.seed,
    )

    if args.full:
        print("Fetching the full archive (6.2 GB) ...")
        print(f"  -> {download_tomotwin_full(config)}")

    # Everything below flows through the shipped build_viz_pool path (the one a wandb run uses).
    n_sel = len(args.pdb_ids) if args.pdb_ids else args.n_volumes
    pool = build_viz_pool(config, n_pool=n_sel, viz_seed=config.seed)
    vol_gt, y, ids = pool["x_gt"], pool["y"], pool["ids"]   # (N,1,V,V,V) in [-1,1]; (N,T,1,V,V)
    N, _, V, _, _ = vol_gt.shape

    clean = _project_2d(vol_gt)                       # (N, 1, V, V), no rotation / no noise
    sig_std = clean.std().item()
    # Per-volume normalised peak: (norm_max + 1) / 2 is that complex's density as a fraction of
    # the SHARED pool scale. A fixed --iso_frac sits at a different fraction of each complex's own
    # peak, so columns with a low peak here get a thin/auto-levelled isosurface (see below).
    peaks = [f"{i}:{float((vol_gt[c, 0].max() + 1) / 2):.2f}" for c, i in enumerate(ids)]
    print(f"\nTomoTwin-100  |  {N} volume(s)  ids={ids}")
    print(f"  volumes {tuple(vol_gt.shape)}  normalised min/mean/max "
          f"{vol_gt.min():+.2f}/{vol_gt.mean():+.2f}/{vol_gt.max():+.2f}")
    print(f"  per-volume peak / pool scale: {peaks}")
    print(f"  tilt series {tuple(y.shape)}  value range {y.min():+.1f}..{y.max():+.1f}   "
          f"clean-proj std {sig_std:.2f}   noise_std {config.noise_std}   "
          f"~SNR {sig_std / max(config.noise_std, 1e-8):.1f}")
    print(f"  matplotlib backend: {plt.get_backend()}")

    level = args.level if args.level is not None else (2.0 * args.iso_frac - 1.0)
    tilt_idx = torch.linspace(0, config.num_tilts - 1, args.n_proj).round().long().tolist()

    n_rows = 1 + 1 + args.n_proj                      # isosurface + clean proj + n_proj tilts
    row_labels = ["GT isosurface", "projection\n(clean, no tilt)"] + \
                 [f"tilt idx {k}\n(step {config.tilt_increment_deg:g}deg)" for k in tilt_idx]

    def _draw_isosurface(ax, vol: np.ndarray, lvl: float, color: str) -> None:
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1), zoom=2.4)
        vmin, vmax = float(vol.min()), float(vol.max())
        used = lvl
        if not (vmin < lvl < vmax):                   # shared level misses this complex's range
            fg = vol[vol > vmin + 0.02 * (vmax - vmin)]
            if fg.size == 0:
                ax.text2D(0.5, 0.5, "empty", ha="center", va="center",
                          transform=ax.transAxes, fontsize=8)
                return
            used = float(np.percentile(fg, 70.0))     # per-volume fallback so the panel isn't blank
            ax.text2D(0.5, 0.0, f"auto lvl {used:+.2f}", ha="center", va="bottom",
                      transform=ax.transAxes, fontsize=7, color="0.45")
        verts, faces, _, _ = marching_cubes(vol, level=used)
        ax.add_collection3d(Poly3DCollection(verts[faces], alpha=0.6, linewidths=0.0,
                                             facecolor=color))
        ax.set_xlim(0, vol.shape[0]); ax.set_ylim(0, vol.shape[1]); ax.set_zlim(0, vol.shape[2])

    proj_vmin, proj_vmax = float(y.min()), float(y.max())
    clean_vmin, clean_vmax = float(clean.min()), float(clean.max())
    fig = plt.figure(figsize=(3.0 * N, 2.6 * n_rows))
    grid = [[None] * N for _ in range(n_rows)]        # grid[row][col]
    axes3d = []
    for col in range(N):
        ax_iso = fig.add_subplot(n_rows, N, col + 1, projection="3d")
        _draw_isosurface(ax_iso, vol_gt[col, 0].numpy(), level, "#4c72b0")
        ax_iso.set_title(f"{ids[col]}", fontsize=10)
        grid[0][col] = ax_iso
        axes3d.append(ax_iso)

        ax_clean = fig.add_subplot(n_rows, N, N + col + 1)
        ax_clean.imshow(clean[col, 0].numpy(), cmap="gray", vmin=clean_vmin, vmax=clean_vmax)
        ax_clean.set_xticks([]); ax_clean.set_yticks([])
        grid[1][col] = ax_clean

        for r, k in enumerate(tilt_idx):
            ax = fig.add_subplot(n_rows, N, (2 + r) * N + col + 1)
            ax.imshow(y[col, k, 0].numpy(), cmap="gray", vmin=proj_vmin, vmax=proj_vmax)
            ax.set_xticks([]); ax.set_yticks([])
            grid[2 + r][col] = ax

    for row in range(1, n_rows):                      # row labels down the left-most column
        grid[row][0].set_ylabel(row_labels[row], fontsize=9)
    axes3d[0].text2D(-0.08, 0.5, row_labels[0], rotation="vertical", va="center", ha="center",
                     transform=axes3d[0].transAxes, fontsize=9)
    for ax in axes3d[1:]:
        ax.shareview(axes3d[0])
    axes3d[0].view_init(elev=18, azim=-60)

    fig.suptitle(f"TomoTwin-100  |  V={V}  T={config.num_tilts}  "
                 f"tilt_step={config.tilt_increment_deg}deg  noise_std={config.noise_std}  "
                 f"isosurface@{level:+.2f}", fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.02, top=0.92, wspace=0.05, hspace=0.12)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"Saved -> {out}")
    if not args.no_show:
        if plt.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template"):
            print(f"backend {plt.get_backend()!r} is non-interactive; re-run with --save PATH.")
        else:
            print("Interactive: drag any 3D panel to orbit all, scroll to zoom, close to exit.")
            plt.show()

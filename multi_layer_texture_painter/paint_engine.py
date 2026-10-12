"""Pure-numpy painting maths for MultiLayer Texture Painter Pro.

Nothing in this module imports ``bpy`` so it can be unit-tested outside Blender.
All pixel arrays are float32 with shape (height, width, 4) in the same layout as
``bpy.types.Image.pixels`` (row 0 = bottom of the image).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional, only used for the "seamless clone" healing method
    import cv2  # type: ignore
except Exception:  # noqa: BLE001 - any import problem simply disables OpenCV
    cv2 = None

BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive end)

HAS_CV2 = cv2 is not None


# --------------------------------------------------------------------------- #
# Change detection
# --------------------------------------------------------------------------- #

def changed_bbox(old: np.ndarray, new: np.ndarray) -> Optional[BBox]:
    """Bounding box of all pixels that differ between two images, or None."""
    changed = np.any(old != new, axis=2)
    rows = np.flatnonzero(changed.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(changed.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #

BLEND_FUNCS = {
    "MIX": lambda d, s: s + 0 * d,
    "MULTIPLY": lambda d, s: d * s,
    "ADD": lambda d, s: d + s,
    "SUBTRACT": lambda d, s: d - s,
    "SCREEN": lambda d, s: 1.0 - (1.0 - d) * (1.0 - s),
    "OVERLAY": lambda d, s: np.where(d < 0.5, 2.0 * d * s, 1.0 - 2.0 * (1.0 - d) * (1.0 - s)),
    "DARKEN": lambda d, s: np.minimum(d, s),
    "LIGHTEN": lambda d, s: np.maximum(d, s),
    "DIFFERENCE": lambda d, s: np.abs(d - s),
}


def blend_apply(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray, mode: str = "MIX") -> np.ndarray:
    """Blend ``src`` over ``dst`` weighted by per-pixel ``alpha`` (h, w).

    ``src`` may be a single RGBA colour of shape (4,) or an array broadcastable to ``dst``.
    The RGB channels use the selected blend mode, alpha uses the "over" operator.
    Returns a new array.
    """
    d = dst[..., :3]
    s = np.asarray(src, dtype=np.float32)[..., :3]
    blended = BLEND_FUNCS.get(mode, BLEND_FUNCS["MIX"])(d, s)
    a = alpha[..., None]
    out = np.empty_like(dst)
    out[..., :3] = d + (blended - d) * a
    src_a = np.asarray(src, dtype=np.float32)[..., 3]
    out[..., 3] = dst[..., 3] + (1.0 - dst[..., 3]) * alpha * src_a
    np.clip(out, 0.0, 1.0, out=out)
    return out


def transfer_delta(dst: np.ndarray, old: np.ndarray, new: np.ndarray, opacity: float) -> np.ndarray:
    """Add the change ``new - old`` (scaled by opacity) onto ``dst``."""
    return np.clip(dst + (new - old) * opacity, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Recovering "what the brush did" from before/after pixels
# --------------------------------------------------------------------------- #

def fit_coverage(
    old: np.ndarray, new: np.ndarray, target: np.ndarray, changed: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Per-pixel brush coverage ``a`` such that ``new ~= old + a * (target - old)``.

    Least squares over the four channels, clamped to [0, 1]. ``target`` is the colour
    (or source image patch) that was painted. Returns (coverage, squared residual over
    the changed pixels); the residual lets callers compare competing hypotheses.
    """
    t = target - old
    d = new - old
    num = (d * t).sum(axis=-1)
    den = (t * t).sum(axis=-1)
    a = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)
    np.clip(a, 0.0, 1.0, out=a)
    a *= changed
    recon = old + a[..., None] * t
    err = float(((recon - new) ** 2).sum(axis=-1)[changed].sum())
    return a.astype(np.float32), err


def pick_color(
    old: np.ndarray, new: np.ndarray, changed: np.ndarray, candidates: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose the candidate colour that best explains the stroke; returns (colour, coverage)."""
    best = None
    for cand in candidates:
        a, err = fit_coverage(old, new, cand, changed)
        if best is None or err < best[0]:
            best = (err, cand, a)
    assert best is not None
    return best[1], best[2]


def sample_shifted(
    src: np.ndarray, bbox: BBox, dst_size: Tuple[int, int], off_u: float, off_v: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour sample of ``src`` for the pixels of ``bbox`` in a (w, h) image.

    A destination texel at UV ``p`` reads the source at ``p - offset`` (UV space), which
    makes the sampling independent of resolution differences. Returns (patch, valid)
    where ``valid`` marks texels whose source position lies inside the source image.
    """
    x0, y0, x1, y1 = bbox
    dw, dh = dst_size
    sh, sw = src.shape[:2]
    ix = np.floor(((np.arange(x0, x1) + 0.5) / dw - off_u) * sw).astype(np.int64)
    iy = np.floor(((np.arange(y0, y1) + 0.5) / dh - off_v) * sh).astype(np.int64)
    vx = (ix >= 0) & (ix < sw)
    vy = (iy >= 0) & (iy < sh)
    ix = np.clip(ix, 0, sw - 1)
    iy = np.clip(iy, 0, sh - 1)
    return src[iy[:, None], ix[None, :]], vy[:, None] & vx[None, :]


def estimate_source_shift(
    src: np.ndarray, new: np.ndarray, mask: np.ndarray, bbox: BBox, min_score: float = 0.6
) -> Optional[Tuple[float, float]]:
    """Find where a cloned stroke came from, as a UV offset usable with ``sample_shifted``.

    Used when the clone tool gives no UV offset (3D viewport): the painted pixels ``new``
    inside ``bbox`` are matched against ``src`` by normalised cross-correlation (FFT) of
    their luminance, restricted to the changed pixels ``mask``. Only same-size images are
    supported. Returns (off_u, off_v) or None when no confident match exists (flat or
    ambiguous texture).
    """
    x0, y0, x1, y1 = bbox
    h, w = src.shape[:2]
    bh, bw = y1 - y0, x1 - x0
    if bh > h or bw > w or not mask.any() or mask.sum() < 16:
        return None
    s = src[..., :3].mean(axis=-1).astype(np.float64)
    s -= s.mean()
    t = new[..., :3].mean(axis=-1).astype(np.float64)
    t = (t - t[mask].mean()) * mask
    t_norm = math.sqrt(float((t * t).sum()))
    if t_norm < 1e-6:
        return None
    # corr[y, x] = sum_ij t[i, j] * s[y + i, x + j]  (top-left of the match)
    corr = np.fft.irfft2(np.fft.rfft2(s) * np.conj(np.fft.rfft2(t, s=(h, w))), s=(h, w))
    corr = corr[: h - bh + 1, : w - bw + 1]
    ii = np.pad((s * s).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    energy = ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw]
    score = corr / (np.sqrt(np.maximum(energy, 1e-12)) * t_norm)
    py, px = np.unravel_index(int(np.argmax(score)), score.shape)
    if score[py, px] < min_score:
        return None
    # source texel = destination texel + (px - x0): offset in UV is the negative of that
    return float(-(px - x0) / w), float(-(py - y0) / h)


# --------------------------------------------------------------------------- #
# Fill patterns
# --------------------------------------------------------------------------- #

def build_pattern(
    h: int, w: int, kind: str, c1: np.ndarray, c2: np.ndarray, size: int = 64
) -> np.ndarray:
    """RGBA pattern: SOLID, CHECKER, GRADIENT_H or GRADIENT_V."""
    c1 = np.asarray(c1, dtype=np.float32)
    c2 = np.asarray(c2, dtype=np.float32)
    if kind == "SOLID":
        return np.broadcast_to(c1, (h, w, 4)).copy()
    if kind == "CHECKER":
        yy, xx = np.mgrid[0:h, 0:w]
        t = (((xx // max(size, 1)) + (yy // max(size, 1))) % 2).astype(np.float32)
    elif kind == "GRADIENT_H":
        t = np.broadcast_to(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, w))
    else:  # GRADIENT_V
        t = np.broadcast_to(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (h, w))
    t = t[..., None]
    return (c1 * (1.0 - t) + c2 * t).astype(np.float32)


# --------------------------------------------------------------------------- #
# Healing brush
# --------------------------------------------------------------------------- #

def falloff_weights(dist: np.ndarray, kind: str) -> np.ndarray:
    """Brush falloff curve; ``dist`` is the distance to the centre in units of the radius."""
    t = np.clip(dist, 0.0, 1.0)
    if kind == "SMOOTH":
        w = 1.0 - (3.0 * t * t - 2.0 * t * t * t)
    elif kind == "SPHERE":
        w = np.sqrt(1.0 - t * t)
    elif kind == "ROOT":
        w = np.sqrt(1.0 - t)
    elif kind == "SHARP":
        w = (1.0 - t) ** 2
    elif kind == "LINEAR":
        w = 1.0 - t
    else:  # CONSTANT
        w = np.ones_like(t)
    return (w * (dist < 1.0)).astype(np.float32)


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur of an (h, w, c) array using cumulative sums (edge-clamped)."""
    if radius < 1:
        return a
    out = a.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * 3
        pad[axis] = (radius + 1, radius)
        padded = np.pad(out, pad, mode="edge")
        cs = np.cumsum(padded, axis=axis)
        n = 2 * radius + 1
        hi = np.take(cs, np.arange(n, cs.shape[axis]), axis=axis)
        lo = np.take(cs, np.arange(0, cs.shape[axis] - n), axis=axis)
        out = (hi - lo) / n
    return out.astype(np.float32)


def _seamless_clone(s3: np.ndarray, d3: np.ndarray, weight: np.ndarray) -> Optional[np.ndarray]:
    """OpenCV Poisson blend of a source patch into the destination patch (8-bit)."""
    if cv2 is None:
        return None
    mask = (weight > 0.05).astype(np.uint8) * 255
    mask[:2, :] = mask[-2:, :] = 0
    mask[:, :2] = mask[:, -2:] = 0
    ys, xs = np.nonzero(mask)
    if ys.size < 16:
        return None
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    center = (int(bx0 + (bx1 - bx0) // 2), int(by0 + (by1 - by0) // 2))
    src8 = np.ascontiguousarray(np.clip(s3, 0, 1) * 255 + 0.5, dtype=np.uint8)
    dst8 = np.ascontiguousarray(np.clip(d3, 0, 1) * 255 + 0.5, dtype=np.uint8)
    try:
        res = cv2.seamlessClone(src8, dst8, mask, center, cv2.NORMAL_CLONE)
    except Exception:  # noqa: BLE001 - cv2.error or shape problems: use the numpy fallback
        return None
    return res.astype(np.float32) / 255.0


def boundary_blend(s3: np.ndarray, d3: np.ndarray, iterations: int = 40) -> np.ndarray:
    """Poisson-style heal: source texture inside, destination colours on the patch border.

    The difference ``d - s`` is known on the border of the patch and is spread smoothly over
    the interior (bilinear start + Jacobi relaxation of the Laplace equation). Adding it back to
    the source keeps all source detail while the result joins the destination without a seam,
    and anything in the destination interior (a blemish) is replaced rather than preserved.
    """
    diff = (d3 - s3).astype(np.float32)
    h, w = diff.shape[:2]
    if h < 3 or w < 3:
        return s3 + diff
    top, bot = diff[0:1, :], diff[-1:, :]
    left, right = diff[:, 0:1], diff[:, -1:]
    ty = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    tx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    corners = (
        (1 - ty) * (1 - tx) * diff[0, 0] + (1 - ty) * tx * diff[0, -1]
        + ty * (1 - tx) * diff[-1, 0] + ty * tx * diff[-1, -1]
    )
    field = (1 - ty) * top + ty * bot + (1 - tx) * left + tx * right - corners
    field[0, :], field[-1, :], field[:, 0], field[:, -1] = diff[0, :], diff[-1, :], diff[:, 0], diff[:, -1]
    for _ in range(iterations):
        field[1:-1, 1:-1] = 0.25 * (
            field[:-2, 1:-1] + field[2:, 1:-1] + field[1:-1, :-2] + field[1:-1, 2:]
        )
    return s3 + field


def heal_patch(
    src: np.ndarray, dst: np.ndarray, weight: np.ndarray, albedo_match: bool = True, method: str = "AUTO"
) -> np.ndarray:
    """Heal one patch. ``weight`` already contains falloff * strength (h, w).

    1. The source is adapted to the destination:
         POISSON    boundary_blend: source detail, destination colours at the patch border
         MEAN       corrected = S + (mean(D) - mean(S))              (mask weighted colour shift)
         FREQUENCY  corrected = (S - lowpass(S)) + lowpass(D)        (source texture, destination
                    low frequencies)
         AUTO       OpenCV seamlessClone if cv2 is importable, else POISSON
    2. The corrected source is feathered into the destination with ``weight``.
    """
    s3, d3 = src[..., :3], dst[..., :3]
    corrected: Optional[np.ndarray] = None
    if not albedo_match:
        corrected = s3
    else:
        if method in {"AUTO", "CV2"}:
            corrected = _seamless_clone(s3, d3, weight)
        if corrected is None and method == "MEAN":
            wsum = float(weight.sum())
            if wsum < 1e-6:
                return dst
            w3 = weight[..., None]
            mean_s = (s3 * w3).sum(axis=(0, 1)) / wsum
            mean_d = (d3 * w3).sum(axis=(0, 1)) / wsum
            corrected = s3 + (mean_d - mean_s)
        if corrected is None and method == "FREQUENCY":
            r = max(1, min(src.shape[0], src.shape[1]) // 8)
            low_s = box_blur(box_blur(s3, r), r)  # two passes ~ triangle/gaussian kernel
            low_d = box_blur(box_blur(d3, r), r)
            corrected = (s3 - low_s) + low_d
        if corrected is None:
            corrected = boundary_blend(s3, d3)
    w3 = weight[..., None]
    out = dst.copy()
    out[..., :3] = d3 * (1.0 - w3) + corrected * w3
    out[..., 3] = dst[..., 3] * (1.0 - weight) + src[..., 3] * weight
    np.clip(out, 0.0, 1.0, out=out)
    return out


def heal_dab(
    dst: np.ndarray,
    src: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    radius: float,
    strength: float,
    falloff: str = "SMOOTH",
    albedo_match: bool = True,
    method: str = "AUTO",
    flip_x: bool = False,
    flip_y: bool = False,
) -> Optional[BBox]:
    """Apply one healing dab in place on ``dst``; returns the modified bbox (or None).

    (cx, cy) is the destination centre and (sx, sy) the source centre, both in texels.
    ``src`` may be the same array as ``dst``; the source patch is copied before writing.
    With ``flip_x`` / ``flip_y`` the source patch is mirrored around the source centre,
    so a destination texel at ``cx + d`` reads the source at ``sx - d`` instead of ``sx + d``.
    """
    radius = max(float(radius), 1.0)
    icx, icy = int(round(cx)), int(round(cy))
    isx, isy = int(round(sx)), int(round(sy))
    r = int(math.ceil(radius))
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    # Clip the patch so both the destination and the (offset / mirrored) source stay in bounds.
    if flip_x:
        mx = icx + isx  # destination x reads source mx - x
        x0, x1 = max(icx - r, 0, mx - sw + 1), min(icx + r + 1, dw, mx + 1)
    else:
        dx = isx - icx
        x0, x1 = max(icx - r, 0, -dx), min(icx + r + 1, dw, sw - dx)
    if flip_y:
        my = icy + isy
        y0, y1 = max(icy - r, 0, my - sh + 1), min(icy + r + 1, dh, my + 1)
    else:
        dy = isy - icy
        y0, y1 = max(icy - r, 0, -dy), min(icy + r + 1, dh, sh - dy)
    if x1 <= x0 or y1 <= y0:
        return None
    xs = slice(mx - x1 + 1, mx - x0 + 1) if flip_x else slice(x0 + dx, x1 + dx)
    ys = slice(my - y1 + 1, my - y0 + 1) if flip_y else slice(y0 + dy, y1 + dy)
    patch_src = src[ys, xs]
    if flip_x:
        patch_src = patch_src[:, ::-1]
    if flip_y:
        patch_src = patch_src[::-1, :]
    patch_src = patch_src.copy()
    patch_dst = dst[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    weight = falloff_weights(np.hypot(xx - icx, yy - icy) / radius, falloff) * float(strength)
    if not weight.any():
        return None
    patch_dst[...] = heal_patch(patch_src, patch_dst.copy(), weight, albedo_match, method)
    return x0, y0, x1, y1


# --------------------------------------------------------------------------- #
# Undo history (tile based so a stroke only stores what it touched)
# --------------------------------------------------------------------------- #

TILE = 128


@dataclass
class Patch:
    name: str
    x0: int
    y0: int
    x1: int
    y1: int
    before: np.ndarray
    after: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(self.before.nbytes + self.after.nbytes)


class UndoRecorder:
    """Collects "before" tiles lazily; call ``touch`` *before* modifying a region."""

    def __init__(self) -> None:
        self._before: Dict[Tuple[str, int, int], np.ndarray] = {}

    def __bool__(self) -> bool:
        return bool(self._before)

    def touch(self, name: str, arr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
        h, w = arr.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return
        for ty in range(y0 // TILE, (y1 - 1) // TILE + 1):
            for tx in range(x0 // TILE, (x1 - 1) // TILE + 1):
                key = (name, tx, ty)
                if key not in self._before:
                    self._before[key] = arr[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE].copy()

    def finish(self, arrays: Dict[str, np.ndarray]) -> List[Patch]:
        """Snapshot the "after" state of every touched tile and return the patches."""
        patches = []
        for (name, tx, ty), before in self._before.items():
            if name not in arrays:
                continue
            y0, x0 = ty * TILE, tx * TILE
            y1, x1 = y0 + before.shape[0], x0 + before.shape[1]
            patches.append(Patch(name, x0, y0, x1, y1, before, arrays[name][y0:y1, x0:x1].copy()))
        return patches


class History:
    """Linear undo/redo stack of patch groups with a memory cap."""

    def __init__(self, max_groups: int = 50, max_bytes: int = 768 * 1024 * 1024) -> None:
        self.max_groups = max_groups
        self.max_bytes = max_bytes
        self._undo: List[List[Patch]] = []
        self._redo: List[List[Patch]] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, group: List[Patch]) -> None:
        if not group:
            return
        self._undo.append(group)
        self._redo.clear()
        total = sum(p.nbytes for g in self._undo for p in g)
        while self._undo and (len(self._undo) > self.max_groups or total > self.max_bytes):
            total -= sum(p.nbytes for p in self._undo.pop(0))

    def pop_undo(self) -> Optional[List[Patch]]:
        if not self._undo:
            return None
        group = self._undo.pop()
        self._redo.append(group)
        return group

    def pop_redo(self) -> Optional[List[Patch]]:
        if not self._redo:
            return None
        group = self._redo.pop()
        self._undo.append(group)
        return group

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

"""Image, brush and UV helpers for MultiLayer Texture Painter Pro."""
from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import bpy
import numpy as np
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.interpolate import poly_3d_calc


# --------------------------------------------------------------------------- #
# Image pixel access
# --------------------------------------------------------------------------- #

def image_is_valid(image: Optional[bpy.types.Image]) -> bool:
    """True if the image exists and has a usable pixel buffer."""
    if image is None:
        return False
    w, h = image.size
    return w > 0 and h > 0 and image.has_data


def read_pixels(image: bpy.types.Image, out: Optional[np.ndarray] = None) -> np.ndarray:
    """Read the image into a float32 (h, w, 4) array using foreach_get.

    If ``out`` has the right element count it is filled in place (no allocation).
    """
    w, h = image.size
    count = w * h * 4
    if out is None or out.size != count:
        out = np.empty((h, w, 4), dtype=np.float32)
    image.pixels.foreach_get(out.reshape(-1))
    return out


def write_pixels(image: bpy.types.Image, arr: np.ndarray, final: bool = False) -> None:
    """Write a float32 (h, w, 4) array back with foreach_set and refresh the image.

    ``final`` additionally frees the GPU copy so the next draw re-uploads it.
    """
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    image.pixels.foreach_set(flat)
    image.update()
    if final:
        refresh_gpu_texture(image)


def refresh_gpu_texture(image: bpy.types.Image) -> None:
    """Force the 3D viewport to re-upload the image.

    image.update() refreshes the Image Editor display only; materials in the
    viewport keep drawing the cached GPU texture until it is freed.
    """
    try:
        image.gl_free()
    except RuntimeError:
        pass


def tag_redraw_all() -> None:
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in {"IMAGE_EDITOR", "VIEW_3D"}:
                area.tag_redraw()


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0.0), 1 / 2.4) - 0.055).astype(np.float32)


def gamma_color_for_image(color, image: Optional[bpy.types.Image]) -> np.ndarray:
    """Convert a UI (gamma-encoded) RGB/RGBA colour to the space ``image.pixels`` uses.

    Byte images expose their raw (sRGB encoded) values, float images are linear.
    """
    c = np.array(list(color)[:4], dtype=np.float32)
    if c.size == 3:
        c = np.append(c, 1.0).astype(np.float32)
    if image is not None and image.is_float and image.colorspace_settings.name.lower().startswith("srgb"):
        c[:3] = srgb_to_linear(c[:3])
    return c


# --------------------------------------------------------------------------- #
# Brush / paint context
# --------------------------------------------------------------------------- #

def get_brush(context: bpy.types.Context) -> Optional[bpy.types.Brush]:
    ip = context.scene.tool_settings.image_paint
    return ip.brush if ip else None


def brush_tool(brush: Optional[bpy.types.Brush]) -> str:
    """Image paint tool identifier (DRAW, CLONE, FILL, ...)."""
    if brush is None:
        return "DRAW"
    for attr in ("image_brush_type", "image_tool", "image_paint_tool"):  # renamed across versions
        value = getattr(brush, attr, None)
        if value:
            return value
    return "DRAW"


def _clone_owner(context: bpy.types.Context):
    """Clone settings moved from Brush (<= 4.2) to ImagePaint (newer); use whichever has them."""
    ip = context.scene.tool_settings.image_paint
    if ip is not None and hasattr(ip, "clone_image"):
        return ip
    return get_brush(context)


def get_clone_image(context: bpy.types.Context) -> Optional[bpy.types.Image]:
    return getattr(_clone_owner(context), "clone_image", None)


def set_clone_image(context: bpy.types.Context, image: Optional[bpy.types.Image]) -> None:
    owner = _clone_owner(context)
    if owner is not None and hasattr(owner, "clone_image"):
        owner.clone_image = image


def get_clone_offset(context: bpy.types.Context) -> Tuple[float, float]:
    off = getattr(_clone_owner(context), "clone_offset", None)
    return (float(off[0]), float(off[1])) if off is not None else (0.0, 0.0)


def get_ups(context: bpy.types.Context):
    """Unified paint settings (ToolSettings in 4.x, the Paint struct in 5.x)."""
    ts = context.scene.tool_settings
    ups = getattr(ts, "unified_paint_settings", None)
    return ups if ups is not None else ts.image_paint.unified_paint_settings


def get_brush_size(context: bpy.types.Context) -> float:
    brush = get_brush(context)
    ups = get_ups(context)
    if ups.use_unified_size or brush is None:
        return float(ups.size)
    return float(brush.size)


def get_brush_color(context: bpy.types.Context) -> Tuple[float, float, float]:
    brush = get_brush(context)
    ups = get_ups(context)
    if ups.use_unified_color or brush is None:
        return tuple(ups.color)[:3]
    return tuple(brush.color)[:3]


def get_active_paint_image(context: bpy.types.Context) -> Optional[bpy.types.Image]:
    """The image Blender's native brush currently paints on."""
    space = context.space_data
    if space is not None and space.type == "IMAGE_EDITOR" and space.image is not None:
        return space.image
    ip = context.scene.tool_settings.image_paint
    if ip.mode == "IMAGE" and ip.canvas is not None:
        return ip.canvas
    ob = context.active_object
    if ob is not None and ob.active_material is not None:
        mat = ob.active_material
        slots = mat.texture_paint_images
        idx = mat.paint_active_slot
        if 0 <= idx < len(slots) and slots[idx] is not None:
            return slots[idx]
    return ip.canvas


# --------------------------------------------------------------------------- #
# Node helpers
# --------------------------------------------------------------------------- #

def find_uv_map_name(node: bpy.types.Node) -> str:
    """Follow the Vector input back to a UV Map node and return its UV map name."""
    cur = node
    for _ in range(8):
        sock = cur.inputs.get("Vector")
        if sock is None or not sock.is_linked:
            return ""
        src = sock.links[0].from_node
        if src.type == "UVMAP":
            return src.uv_map
        cur = src
    return ""


def collect_selected_image_nodes(
    context: bpy.types.Context, scope: str
) -> Iterator[Tuple[bpy.types.Material, bpy.types.Node]]:
    """Yield (material, node) for selected Image Texture nodes with an image.

    scope 'EDITOR': the node tree shown in the active Shader Editor.
    scope 'OBJECT': every material of the active object (selection is stored per node,
    so nodes selected in several materials can be collected in one go).
    """
    def from_tree(mat, tree):
        for node in tree.nodes:
            if node.select and node.type == "TEX_IMAGE" and node.image is not None:
                yield mat, node

    if scope == "EDITOR":
        space = context.space_data
        if space is None or space.type != "NODE_EDITOR" or space.edit_tree is None:
            return
        mat = space.id if isinstance(space.id, bpy.types.Material) else None
        if mat is None and context.object is not None:
            mat = context.object.active_material
        yield from from_tree(mat, space.edit_tree)
    else:
        ob = context.active_object
        if ob is None:
            return
        for slot in ob.material_slots:
            mat = slot.material
            if mat is not None and mat.use_nodes and mat.node_tree is not None:
                yield from from_tree(mat, mat.node_tree)


# --------------------------------------------------------------------------- #
# Screen -> UV
# --------------------------------------------------------------------------- #

def image_editor_uv(region: bpy.types.Region, x: float, y: float) -> Tuple[float, float]:
    u, v = region.view2d.region_to_view(x, y)
    return float(u), float(v)


def image_editor_screen_px_per_texel(region: bpy.types.Region, image: bpy.types.Image) -> float:
    """How many screen pixels one image texel covers at the current zoom."""
    p0 = region.view2d.view_to_region(0.0, 0.0, clip=False)
    p1 = region.view2d.view_to_region(1.0, 0.0, clip=False)
    return max(abs(p1[0] - p0[0]) / max(image.size[0], 1), 1e-6)


class MeshUVPicker:
    """Ray-casts the evaluated mesh from the viewport and returns the hit UV."""

    def __init__(self, context: bpy.types.Context, obj: bpy.types.Object, uv_name: str = ""):
        self.obj = obj
        self.depsgraph = context.evaluated_depsgraph_get()
        self.eval_obj = obj.evaluated_get(self.depsgraph)
        self.mesh = self.eval_obj.to_mesh()
        tris = self.mesh.loop_triangles
        n = len(tris)
        self.tri_poly = np.empty(n, dtype=np.int32)
        tris.foreach_get("polygon_index", self.tri_poly)
        self.tri_loops = np.empty(n * 3, dtype=np.int32)
        tris.foreach_get("loops", self.tri_loops)
        self.tri_loops = self.tri_loops.reshape(n, 3)
        self.tri_verts = np.empty(n * 3, dtype=np.int32)
        tris.foreach_get("vertices", self.tri_verts)
        self.tri_verts = self.tri_verts.reshape(n, 3)
        uv_layer = self.mesh.uv_layers.get(uv_name) if uv_name else None
        uv_layer = uv_layer or self.mesh.uv_layers.active
        self.has_uv = uv_layer is not None
        if self.has_uv:
            uvs = np.empty(len(uv_layer.data) * 2, dtype=np.float32)
            uv_layer.data.foreach_get("uv", uvs)
            self.uv = uvs.reshape(-1, 2)
        self.order = np.argsort(self.tri_poly, kind="stable")
        self.sorted_poly = self.tri_poly[self.order]
        self.inv_world = obj.matrix_world.inverted()

    def free(self) -> None:
        try:
            self.eval_obj.to_mesh_clear()
        except RuntimeError:
            pass

    def pick(self, region, rv3d, coord: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if not self.has_uv:
            return None
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        o = self.inv_world @ origin
        d = (self.inv_world.to_3x3() @ direction).normalized()
        hit, loc, _normal, poly = self.obj.ray_cast(o, d, depsgraph=self.depsgraph)
        if not hit:
            return None
        lo = np.searchsorted(self.sorted_poly, poly, side="left")
        hi = np.searchsorted(self.sorted_poly, poly, side="right")
        best = None
        best_score = -1e9
        verts = self.mesh.vertices
        for ti in self.order[lo:hi]:
            vi = self.tri_verts[ti]
            tri = [verts[int(i)].co for i in vi]
            w = poly_3d_calc(tri, loc)
            score = min(w)
            if score > best_score:
                best_score, best = score, (ti, w)
        if best is None:
            return None
        ti, w = best
        uv = sum((self.uv[int(l)] * wi for l, wi in zip(self.tri_loops[ti], w)), np.zeros(2, dtype=np.float32))
        return float(uv[0]), float(uv[1])

    def texel_radius(self, region, rv3d, coord, screen_radius: float, size: Tuple[int, int]) -> Optional[float]:
        """Texel radius that corresponds to ``screen_radius`` pixels at ``coord``.

        Probes a few pixels in four directions and keeps the smallest estimate, so a
        probe that lands across a UV seam cannot inflate the radius.
        """
        a = self.pick(region, rv3d, coord)
        if a is None:
            return None
        step = min(4.0, max(float(screen_radius), 0.1))
        best = None
        for dx, dy in ((step, 0.0), (0.0, step), (-step, 0.0), (0.0, -step)):
            b = self.pick(region, rv3d, (coord[0] + dx, coord[1] + dy))
            if b is None:
                continue
            du = (b[0] - a[0]) * size[0]
            dv = (b[1] - a[1]) * size[1]
            r = math.hypot(du, dv) / step * screen_radius
            if r > 0.5 and (best is None or r < best):
                best = r
        return best

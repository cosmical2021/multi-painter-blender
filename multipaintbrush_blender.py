bl_info = {
    "name": "Multi-Node Texture Paint & Clone Pro",
    "author": "3D Pipeline Tools",
    "version": (1, 5, 1),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar (N) > Tool > Brush Settings > Multi-Node Paint Pro",
    "description": "Multi-node painting, Self-Source Cloning, Healing, OpenCV Inpaint, Path Clone with Source Preview, Undo/Redo & Brush Texture/Alpha.",
    "category": "Paint",
}

import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =====================================================================
# 1. GLOBAALI TILA & 3D VIEWPORT OVERLAY
# =====================================================================

class PaintProRuntime:
    source_path_3d = []
    source_path_uv = []
    path_arc_lengths = []
    total_path_length = 0.0
    
    single_clone_uv = Vector((0.5, 0.5))
    single_clone_3d = None
    
    hover_3d = None
    hover_normal = Vector((0.0, 0.0, 1.0))
    hover_world_radius = 0.1
    hover_uv_to_world = 1.0
    hover_angle = 0.0
    hover_strength = 1.0
    current_source_uv = Vector((0.5, 0.5))
    
    draw_handle = None
    is_modal_active = False

RUNTIME = PaintProRuntime()


def draw_viewport_overlay():
    context = bpy.context
    if not hasattr(context, "scene") or not context.scene:
        return
    settings = context.scene.mnpp_settings
    if not settings.show_overlay:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    # 1. Kloonipolku
    if len(RUNTIME.source_path_3d) > 1:
        gpu.state.line_width_set(3.0)
        pts = list(RUNTIME.source_path_3d)
        if settings.close_path:
            pts.append(pts[0])
            shader.uniform_float("color", (0.1, 0.9, 0.3, 0.9))
        else:
            shader.uniform_float("color", (1.0, 0.5, 0.0, 0.9))
            
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": pts})
        shader.bind()
        batch.draw(shader)

    # 2. Kloonilähde
    if RUNTIME.single_clone_3d is not None:
        gpu.state.point_size_set(10.0)
        shader.uniform_float("color", (0.0, 0.8, 1.0, 1.0))
        batch = batch_for_shader(shader, 'POINTS', {"pos": [RUNTIME.single_clone_3d]})
        shader.bind()
        batch.draw(shader)

    # 3. Path Clone -lähdealueen esikatselu (Violetti rengas)
    if RUNTIME.is_modal_active and settings.tool_mode == 'PATH_CLONE' and settings.show_path_source_preview:
        if RUNTIME.hover_3d is not None:
            center = RUNTIME.hover_3d
            n = RUNTIME.hover_normal.normalized()
            rad = max(0.0005, RUNTIME.hover_world_radius)
            up = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.99 else Vector((1.0, 0.0, 0.0))
            t1 = n.cross(up).normalized()
            t2 = n.cross(t1).normalized()
            offset = n * (rad * 0.02 + 0.0008)
            
            preview_pts = [
                center + offset + (t1 * math.cos((2.0 * math.pi * i) / 12.0) +
                                   t2 * math.sin((2.0 * math.pi * i) / 12.0)) * (rad * 0.6)
                for i in range(13)
            ]
            gpu.state.line_width_set(1.5)
            shader.uniform_float("color", (0.9, 0.1, 0.9, 0.8))
            batch_prev = batch_for_shader(shader, 'LINE_STRIP', {"pos": preview_pts})
            shader.bind()
            batch_prev.draw(shader)

    # 4. 3D-sivellinrengas
    if RUNTIME.is_modal_active and RUNTIME.hover_3d is not None:
        center = RUNTIME.hover_3d
        n = RUNTIME.hover_normal.normalized()
        rad = max(0.0005, RUNTIME.hover_world_radius)
        up = Vector((0.0, 0.0, 1.0)) if abs(n.z) < 0.99 else Vector((1.0, 0.0, 0.0))
        t1 = n.cross(up).normalized()
        t2 = n.cross(t1).normalized()
        offset = n * (rad * 0.02 + 0.0005)
        
        segments = 48
        circle_pts = [
            center + offset + (t1 * math.cos((2.0 * math.pi * i) / segments) +
                               t2 * math.sin((2.0 * math.pi * i) / segments)) * rad
            for i in range(segments + 1)
        ]

        mode_colors = {
            'PAINT': (1.0, 1.0, 1.0, 0.9),
            'CLONE': (0.0, 0.85, 1.0, 0.9),
            'PATH_CLONE': (0.2, 1.0, 0.4, 0.9),
            'HEAL': (1.0, 0.8, 0.1, 0.9),
            'INPAINT_MASK': (1.0, 0.15, 0.15, 0.95),
        }
        col = mode_colors.get(settings.tool_mode, (1.0, 1.0, 1.0, 0.9))

        gpu.state.line_width_set(2.0)
        shader.uniform_float("color", col)
        batch_ring = batch_for_shader(shader, 'LINE_STRIP', {"pos": circle_pts})
        shader.bind()
        batch_ring.draw(shader)

        if settings.use_rake:
            ang = RUNTIME.hover_angle
            dir_vec = (t1 * math.cos(ang) + t2 * math.sin(ang)) * rad
            line_pts = [center + offset, center + offset + dir_vec]
            gpu.state.line_width_set(2.0)
            batch_rake = batch_for_shader(shader, 'LINES', {"pos": line_pts})
            batch_rake.draw(shader)

    gpu.state.blend_set('NONE')


# =====================================================================
# 2. SIVELLINASETUKSET, BRUSH TEXTURE & UV-RAYCAST
# =====================================================================

def get_active_brush_params(context):
    ts = context.tool_settings
    ip = ts.image_paint
    brush = ip.brush if ip else None
    ups = getattr(ip, "unified_paint_settings", None) or getattr(ts, "unified_paint_settings", None)

    radius_px = 35
    strength = 1.0
    color = (1.0, 1.0, 1.0)

    if ups and getattr(ups, "use_unified_size", False):
        radius_px = ups.size
    elif brush and hasattr(brush, "size"):
        radius_px = brush.size

    if ups and getattr(ups, "use_unified_strength", False):
        strength = ups.strength
    elif brush and hasattr(brush, "strength"):
        strength = brush.strength

    if ups and getattr(ups, "use_unified_color", False):
        color = ups.color
    elif brush and hasattr(brush, "color"):
        color = brush.color

    return float(max(1, radius_px)), float(strength), color


def get_brush_texture_pixel_array(brush):
    """KORJATTU: Tarkistaa turvallisesti brush.texture ja brush.texture.image ominaisuudet."""
    if not brush or not hasattr(brush, "texture") or not brush.texture:
        return None
    tex = brush.texture
    if not hasattr(tex, "image") or not tex.image:
        return None
    
    img = tex.image
    iw, ih = img.size
    raw = np.empty(iw * ih * 4, dtype=np.float32)
    img.pixels.foreach_get(raw)
    return raw.reshape((ih, iw, 4))


def set_active_brush_size(context, new_size):
    val = int(max(1, min(2000, round(new_size))))
    ts = context.tool_settings
    ip = ts.image_paint
    brush = ip.brush if ip else None
    ups = getattr(ip, "unified_paint_settings", None) or getattr(ts, "unified_paint_settings", None)
    if ups and hasattr(ups, "size"): ups.size = val
    if brush and hasattr(brush, "size"): brush.size = val


def set_active_brush_strength(context, new_strength):
    val = float(max(0.01, min(1.0, new_strength)))
    ts = context.tool_settings
    ip = ts.image_paint
    brush = ip.brush if ip else None
    ups = getattr(ip, "unified_paint_settings", None) or getattr(ts, "unified_paint_settings", None)
    if ups and hasattr(ups, "strength"): ups.strength = val
    if brush and hasattr(brush, "strength"): brush.strength = val


def get_view3d_window_region(context, event):
    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            for reg in area.regions:
                if reg.type == 'WINDOW':
                    if (reg.x <= event.mouse_x <= reg.x + reg.width and
                        reg.y <= event.mouse_y <= reg.y + reg.height):
                        return area, reg, area.spaces.active.region_3d

    if context.area and context.area.type == 'VIEW_3D':
        for reg in context.area.regions:
            if reg.type == 'WINDOW':
                return context.area, reg, context.area.spaces.active.region_3d

    return None, None, None


def get_selected_image_nodes(context):
    obj = context.active_object
    if not obj or not obj.active_material or not obj.active_material.use_nodes:
        return []
    
    nodes = []
    for n in obj.active_material.node_tree.nodes:
        if n.type == 'TEX_IMAGE' and n.select and n.image:
            nodes.append(n)
    return nodes


class SurfaceUVCaster:
    def __init__(self, context, obj):
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        
        self.matrix_world = eval_obj.matrix_world.copy()
        self.matrix_inv = self.matrix_world.inverted()
        self.normal_matrix = self.matrix_inv.transposed().to_3x3()
        
        mesh = eval_obj.to_mesh()
        self.bm = bmesh.new()
        self.bm.from_mesh(mesh)
        eval_obj.to_mesh_clear()
        
        bmesh.ops.triangulate(self.bm, faces=self.bm.faces)
        self.bm.faces.ensure_lookup_table()
        
        self.bvh = BVHTree.FromBMesh(self.bm)
        self.uv_layer = self.bm.loops.layers.uv.active

    def raycast_screen_full(self, context, event):
        if not self.uv_layer:
            return None, None, None, 1.0
            
        area, win_region, rv3d = get_view3d_window_region(context, event)
        if not win_region or not rv3d:
            return None, None, None, 1.0
            
        coord = (event.mouse_x - win_region.x, event.mouse_y - win_region.y)
        if coord[0] < 0 or coord[1] < 0 or coord[0] > win_region.width or coord[1] > win_region.height:
            return None, None, None, 1.0
        
        origin_world = view3d_utils.region_2d_to_origin_3d(win_region, rv3d, coord)
        vec_world = view3d_utils.region_2d_to_vector_3d(win_region, rv3d, coord)
        
        origin_local = self.matrix_inv @ origin_world
        vec_local = (self.matrix_inv.to_3x3() @ vec_world).normalized()
        
        loc, norm, face_idx, dist = self.bvh.ray_cast(origin_local, vec_local)
        if loc is None or face_idx is None:
            return None, None, None, 1.0
            
        face = self.bm.faces[face_idx]
        v0, v1, v2 = [l.vert.co for l in face.loops]
        uv0, uv1, uv2 = [l[self.uv_layer].uv.copy() for l in face.loops]
        
        bary = self._barycentric(loc, v0, v1, v2)
        uv = uv0 * bary[0] + uv1 * bary[1] + uv2 * bary[2]
        
        uv_wrapped = Vector((uv.x % 1.0, uv.y % 1.0))
        world_loc = self.matrix_world @ loc
        world_norm = (self.normal_matrix @ norm).normalized()

        wv0, wv1, wv2 = self.matrix_world @ v0, self.matrix_world @ v1, self.matrix_world @ v2
        area_3d = 0.5 * (wv1 - wv0).cross(wv2 - wv0).length
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0
        area_uv = 0.5 * abs(duv1.x * duv2.y - duv1.y * duv2.x)
        uv_to_world = math.sqrt(area_3d / area_uv) if area_uv > 1e-8 else 1.0

        return world_loc, uv_wrapped, world_norm, uv_to_world

    def raycast_screen(self, context, event):
        world_loc, uv_wrapped, _, _ = self.raycast_screen_full(context, event)
        return world_loc, uv_wrapped

    @staticmethod
    def _barycentric(p, a, b, c):
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = v0.dot(v0)
        d01 = v0.dot(v1)
        d11 = v1.dot(v1)
        d20 = v2.dot(v0)
        d21 = v2.dot(v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-8:
            return (1.0, 0.0, 0.0)
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return (u, v, w)

    def free(self):
        if self.bm.is_valid:
            self.bm.free()


class ImageNodeBuffer:
    def __init__(self, node, max_undo_steps=10):
        self.node = node
        self.image = node.image
        self.w, self.h = self.image.size
        
        raw = np.empty(self.w * self.h * 4, dtype=np.float32)
        self.image.pixels.foreach_get(raw)
        self.pixels = raw.reshape((self.h, self.w, 4))
        
        self.source_snapshot = self.pixels.copy()
        self.clean_before_inpaint = self.pixels.copy()
        self.inpaint_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        
        self.max_undo = max_undo_steps
        self.undo_stack = [self.pixels.copy()]
        self.redo_stack = []

    def push_undo(self):
        self.undo_stack.append(self.pixels.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.pixels = self.undo_stack[-1].copy()
            self.commit()
            return True
        return False

    def redo(self):
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.pixels = state.copy()
            self.commit()
            return True
        return False

    def commit(self):
        self.image.pixels.foreach_set(self.pixels.ravel())
        self.image.update()
        self.image.update_tag()


# =====================================================================
# 3. SIVELLINMOOTTORI JA BRUSH TEXTURE / LUMINANCE -TUKI
# =====================================================================

def sample_brush_patch(buf, center_uv, radius_px, angle_rad):
    h, w, _ = buf.shape
    cx = center_uv.x * w
    cy = center_uv.y * h
    
    r = int(max(1, radius_px))
    y_grid, x_grid = np.mgrid[-r:r+1, -r:r+1]
    
    if abs(angle_rad) > 1e-4:
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rx = x_grid * cos_a - y_grid * sin_a
        ry = x_grid * sin_a + y_grid * cos_a
    else:
        rx, ry = x_grid, y_grid
        
    sample_x = np.round(cx + rx).astype(np.int32) % w
    sample_y = np.round(cy + ry).astype(np.int32) % h
    
    return buf[sample_y, sample_x], sample_x, sample_y, x_grid, y_grid


def apply_brush_dab(node_buf, target_uv, source_uv, radius_px, strength, angle_rad, mode, brush_color, brush_tex_arr, tex_mode):
    r = int(max(1, radius_px))
    _, tx, ty, xg, yg = sample_brush_patch(node_buf.pixels, target_uv, r, 0.0)
    
    dist = np.sqrt(xg**2 + yg**2) / float(r)
    inside = dist <= 1.0
    falloff = np.where(inside, (1.0 - dist) ** 1.5, 0.0)
    
    tex_mask = 1.0
    if brush_tex_arr is not None:
        th, tw, _ = brush_tex_arr.shape
        nx = (xg + r) / float(2 * r)
        ny = (yg + r) / float(2 * r)
        tex_x = np.clip((nx * tw).astype(np.int32), 0, tw - 1)
        tex_y = np.clip((ny * th).astype(np.int32), 0, th - 1)
        sampled_tex = brush_tex_arr[tex_y, tex_x]
        
        if tex_mode == 'LUMINANCE':
            tex_mask = 0.2126 * sampled_tex[..., 0] + 0.7152 * sampled_tex[..., 1] + 0.0722 * sampled_tex[..., 2]
        elif tex_mode == 'ALPHA':
            tex_mask = sampled_tex[..., 3]
        elif tex_mode == 'RGB':
            tex_mask = sampled_tex[..., :3].mean(axis=-1)
            
        tex_mask = np.where(inside, tex_mask, 0.0)

    alpha_mask = (falloff * strength * tex_mask)[..., np.newaxis]
    dst_patch = node_buf.pixels[ty, tx]

    if mode == 'PAINT':
        color_arr = np.array([brush_color[0], brush_color[1], brush_color[2], 1.0], dtype=np.float32)
        node_buf.pixels[ty, tx] = dst_patch * (1.0 - alpha_mask) + color_arr * alpha_mask

    elif mode in {'CLONE', 'PATH_CLONE'}:
        src_patch, _, _, _, _ = sample_brush_patch(node_buf.source_snapshot, source_uv, r, angle_rad)
        node_buf.pixels[ty, tx] = dst_patch * (1.0 - alpha_mask) + src_patch * alpha_mask

    elif mode == 'HEAL':
        src_patch, _, _, _, _ = sample_brush_patch(node_buf.source_snapshot, source_uv, r, angle_rad)
        src_low = np.mean(src_patch[inside], axis=0) if np.any(inside) else np.mean(src_patch, axis=(0, 1))
        dst_low = np.mean(dst_patch[inside], axis=0) if np.any(inside) else np.mean(dst_patch, axis=(0, 1))
        
        src_high_freq = src_patch - src_low
        healed_patch = np.clip(dst_low + src_high_freq, 0.0, 1.0)
        healed_patch[..., 3] = dst_patch[..., 3]
        node_buf.pixels[ty, tx] = dst_patch * (1.0 - alpha_mask) + healed_patch * alpha_mask

    elif mode == 'INPAINT_MASK':
        node_buf.inpaint_mask[ty[inside], tx[inside]] = 255
        red_stroke = np.array([1.0, 0.05, 0.05, 1.0], dtype=np.float32)
        stroke_alpha = (inside.astype(np.float32) * 0.85)[..., np.newaxis]
        node_buf.pixels[ty, tx] = dst_patch * (1.0 - stroke_alpha) + red_stroke * stroke_alpha


def evaluate_path_clone_uv(distance_travelled, close_path):
    uvs = RUNTIME.source_path_uv
    lengths = RUNTIME.path_arc_lengths
    total_len = RUNTIME.total_path_length
    
    if len(uvs) < 2 or total_len < 1e-6:
        return uvs[0] if uvs else Vector((0.5, 0.5)), 0.0

    if close_path:
        d = distance_travelled % total_len
    else:
        cycle = 2.0 * total_len
        m = distance_travelled % cycle
        d = m if m <= total_len else cycle - m

    for i in range(len(lengths) - 1):
        if lengths[i] <= d <= lengths[i+1]:
            seg_len = lengths[i+1] - lengths[i]
            t = (d - lengths[i]) / seg_len if seg_len > 1e-6 else 0.0
            uv_a = uvs[i]
            uv_b = uvs[(i + 1) % len(uvs)]
            interp_uv = uv_a.lerp(uv_b, t)
            tangent = uv_b - uv_a
            path_angle = math.atan2(tangent.y, tangent.x)
            return interp_uv, path_angle

    return uvs[-1], 0.0


def run_opencv_inpaint_on_buffers(buffers, settings):
    algo = cv2.INPAINT_TELEA if settings.inpaint_algo == 'TELEA' else cv2.INPAINT_NS
    radius = settings.inpaint_radius

    for b in buffers:
        if not np.any(b.inpaint_mask):
            continue
        clean_rgb = np.clip(b.clean_before_inpaint[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
        inpainted_rgb = cv2.inpaint(clean_rgb, b.inpaint_mask, radius, algo)
        
        b.pixels[:, :, :3] = inpainted_rgb.astype(np.float32) / 255.0
        b.clean_before_inpaint = b.pixels.copy()
        b.source_snapshot = b.pixels.copy()
        b.inpaint_mask.fill(0)
        b.commit()


# =====================================================================
# 4. OPERAATTORIT
# =====================================================================

class PAINT_OT_multi_node_modal(bpy.types.Operator):
    bl_idname = "paint.mnpp_modal_stroke"
    bl_label = "Activate Multi-Node Brush"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Valitse ensin Mesh-objekti!")
            return {'CANCELLED'}

        nodes = get_selected_image_nodes(context)
        if not nodes:
            self.report({'ERROR'}, "Valitse vähintään yksi Image Texture -node Shader Editorista!")
            return {'CANCELLED'}

        self.settings = context.scene.mnpp_settings
        if self.settings.tool_mode == 'PATH_CLONE' and len(RUNTIME.source_path_uv) < 2:
            self.report({'ERROR'}, "Piirrä ensin lähdepolku '1. Draw Source Path' -painikkeella!")
            return {'CANCELLED'}

        if self.settings.tool_mode == 'INPAINT_MASK' and not HAS_CV2:
            self.report({'ERROR'}, "OpenCV (cv2) ei ole asennettuna Blenderin Pythoniin!")
            return {'CANCELLED'}

        self.caster = SurfaceUVCaster(context, obj)
        if not self.caster.uv_layer:
            self.caster.free()
            self.report({'ERROR'}, "Objektilla ei ole aktiivista UV-karttaa!")
            return {'CANCELLED'}

        self.buffers = [ImageNodeBuffer(n) for n in nodes]
        self.is_drawing = False
        self.last_uv = None
        self.current_angle = 0.0
        self.stroke_distance = 0.0
        self.initial_target_uv = None

        self.resizing_mode = None
        self.resize_start_x = 0
        self.resize_init_val = 35.0

        if RUNTIME.draw_handle is None:
            RUNTIME.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_viewport_overlay, (), 'WINDOW', 'POST_VIEW'
            )

        RUNTIME.is_modal_active = True
        context.window.cursor_modal_set('PAINT_BRUSH')
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, f"Multi-Node Brush aktiivinen | Undo: Ctrl+Z | Redo: Ctrl+Shift+Z")
        return {'RUNNING_MODAL'}

    def update_hover_ring(self, context, event):
        radius_px, strength, _ = get_active_brush_params(context)
        RUNTIME.hover_strength = strength

        if self.resizing_mode is not None:
            if RUNTIME.hover_3d is not None:
                RUNTIME.hover_world_radius = (radius_px / 1024.0) * RUNTIME.hover_uv_to_world
            return

        w_loc, uv, w_norm, uv_to_world = self.caster.raycast_screen_full(context, event)
        if w_loc is not None:
            RUNTIME.hover_3d = w_loc
            RUNTIME.hover_normal = w_norm
            RUNTIME.hover_uv_to_world = uv_to_world
            RUNTIME.hover_world_radius = (radius_px / 1024.0) * uv_to_world
            RUNTIME.current_source_uv = uv
        else:
            RUNTIME.hover_3d = None

    def modal(self, context, event):
        area, win_reg, rv3d = get_view3d_window_region(context, event)
        if area:
            area.tag_redraw()

        # Undo / Redo (Ctrl + Z & Ctrl + Shift + Z / Ctrl + Y)
        if event.ctrl and event.value == 'PRESS':
            if event.type == 'Z':
                if event.shift:
                    success = any(b.redo() for b in self.buffers)
                    if success: self.report({'INFO'}, "Redo: Tila palautettu.")
                else:
                    success = any(b.undo() for b in self.buffers)
                    if success: self.report({'INFO'}, "Undo: Viimeisin veto kumottu.")
                return {'RUNNING_MODAL'}
            elif event.type == 'Y':
                success = any(b.redo() for b in self.buffers)
                if success: self.report({'INFO'}, "Redo: Tila palautettu.")
                return {'RUNNING_MODAL'}

        if self.resizing_mode is not None:
            if event.type == 'MOUSEMOVE':
                dx = event.mouse_x - self.resize_start_x
                if self.resizing_mode == 'SIZE':
                    new_sz = max(1.0, self.resize_init_val + dx * 0.5)
                    set_active_brush_size(context, new_sz)
                    if area: area.header_text_set(f"Brush Size: {int(new_sz)} px")
                elif self.resizing_mode == 'STRENGTH':
                    new_str = max(0.01, min(1.0, self.resize_init_val + dx * 0.003))
                    set_active_brush_strength(context, new_str)
                    if area: area.header_text_set(f"Brush Strength: {new_str:.2f}")
                self.update_hover_ring(context, event)
                return {'RUNNING_MODAL'}

            elif event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
                self.resizing_mode = None
                if area: area.header_text_set(None)
                return {'RUNNING_MODAL'}

            elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
                if self.resizing_mode == 'SIZE': set_active_brush_size(context, self.resize_init_val)
                else: set_active_brush_strength(context, self.resize_init_val)
                self.resizing_mode = None
                if area: area.header_text_set(None)
                self.update_hover_ring(context, event)
                return {'RUNNING_MODAL'}
            return {'RUNNING_MODAL'}

        if not self.is_drawing and event.value == 'PRESS':
            if event.type == 'F':
                radius_px, strength, _ = get_active_brush_params(context)
                self.resize_start_x = event.mouse_x
                if event.shift:
                    self.resizing_mode = 'STRENGTH'
                    self.resize_init_val = strength
                else:
                    self.resizing_mode = 'SIZE'
                    self.resize_init_val = radius_px
                return {'RUNNING_MODAL'}

            if event.type in {'LEFT_BRACKET', 'MINUS', 'NUMPAD_MINUS', 'PAGE_DOWN'}:
                radius_px, _, _ = get_active_brush_params(context)
                set_active_brush_size(context, radius_px - max(1.0, radius_px * 0.12))
                self.update_hover_ring(context, event)
                return {'RUNNING_MODAL'}

            if event.type in {'RIGHT_BRACKET', 'PLUS', 'EQUAL', 'NUMPAD_PLUS', 'PAGE_UP'}:
                radius_px, _, _ = get_active_brush_params(context)
                set_active_brush_size(context, radius_px + max(1.0, radius_px * 0.12))
                self.update_hover_ring(context, event)
                return {'RUNNING_MODAL'}

        if event.type == 'ESC' or (event.type == 'RIGHTMOUSE' and event.value == 'PRESS' and not self.is_drawing):
            self.caster.free()
            RUNTIME.is_modal_active = False
            RUNTIME.hover_3d = None
            if area: area.header_text_set(None)
            context.window.cursor_modal_restore()
            self.report({'INFO'}, "Multi-Node Brush suljettu.")
            return {'FINISHED'}

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} or event.alt:
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            self.update_hover_ring(context, event)

        if win_reg is None or not (win_reg.x <= event.mouse_x <= win_reg.x + win_reg.width and
                                   win_reg.y <= event.mouse_y <= win_reg.y + win_reg.height):
            if not self.is_drawing:
                return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and event.ctrl:
            world_pt, uv = self.caster.raycast_screen(context, event)
            if uv is not None:
                RUNTIME.single_clone_3d = world_pt
                RUNTIME.single_clone_uv = uv
                self.report({'INFO'}, f"Kloonilähde asetettu UV: ({uv.x:.3f}, {uv.y:.3f})")
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            world_pt, uv = self.caster.raycast_screen(context, event)
            if uv is None:
                return {'RUNNING_MODAL'}

            self.is_drawing = True
            self.last_uv = uv
            self.initial_target_uv = uv
            self.stroke_distance = 0.0
            
            for b in self.buffers:
                b.push_undo()
                b.source_snapshot = b.pixels.copy()
                if self.settings.tool_mode == 'INPAINT_MASK':
                    b.clean_before_inpaint = b.pixels.copy()
                    b.inpaint_mask.fill(0)
                
            self.stamp_along_segment(context, uv, uv)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.is_drawing:
                self.is_drawing = False
                self.last_uv = None
                if self.settings.tool_mode == 'INPAINT_MASK' and self.settings.auto_inpaint:
                    run_opencv_inpaint_on_buffers(self.buffers, self.settings)
                    if area: area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE' and self.is_drawing:
            _, target_uv = self.caster.raycast_screen(context, event)
            if target_uv is not None and self.last_uv is not None:
                self.stamp_along_segment(context, self.last_uv, target_uv)
                self.last_uv = target_uv
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def stamp_along_segment(self, context, uv_start, uv_end):
        radius_px, strength, color = get_active_brush_params(context)
        ts = context.tool_settings
        brush = ts.image_paint.brush if ts.image_paint else None
        brush_tex_arr = get_brush_texture_pixel_array(brush)

        delta = uv_end - uv_start
        dist = delta.length
        if dist > 0.4:
            uv_start = uv_end
            dist = 0.0

        if dist > 1e-5 and self.settings.use_rake:
            self.current_angle = math.atan2(delta.y, delta.x)
            RUNTIME.hover_angle = self.current_angle

        ref_w = max(1, self.buffers[0].w)
        step_uv = max(0.0015, (radius_px / float(ref_w)) * 0.22)
        steps = max(1, int(dist / step_uv))
        mode = self.settings.tool_mode

        for i in range(1, steps + 1):
            t = i / float(steps) if dist > 1e-6 else 1.0
            cur_uv = uv_start.lerp(uv_end, t)
            if dist > 1e-6:
                self.stroke_distance += dist / float(steps)

            angle_rad = self.current_angle
            source_uv = cur_uv

            if mode in {'CLONE', 'HEAL'}:
                if self.settings.aligned_clone and self.initial_target_uv is not None:
                    offset = cur_uv - self.initial_target_uv
                    source_uv = Vector(((RUNTIME.single_clone_uv.x + offset.x) % 1.0,
                                        (RUNTIME.single_clone_uv.y + offset.y) % 1.0))
                else:
                    source_uv = RUNTIME.single_clone_uv

            elif mode == 'PATH_CLONE':
                source_uv, path_angle = evaluate_path_clone_uv(
                    self.stroke_distance * self.settings.path_speed,
                    self.settings.close_path
                )
                if self.settings.use_rake:
                    angle_rad = self.current_angle - path_angle

            for b in self.buffers:
                scaled_radius = max(1.0, radius_px * (b.w / 1024.0))
                apply_brush_dab(b, cur_uv, source_uv, scaled_radius, strength, angle_rad, mode, color, brush_tex_arr, self.settings.brush_tex_channel)

        for b in self.buffers:
            b.commit()


class PAINT_OT_draw_clone_path(bpy.types.Operator):
    bl_idname = "paint.mnpp_draw_path"
    bl_label = "Draw Source Path on Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        if not context.active_object or context.active_object.type != 'MESH':
            self.report({'ERROR'}, "Valitse ensin Mesh-objekti!")
            return {'CANCELLED'}

        self.caster = SurfaceUVCaster(context, context.active_object)
        if not self.caster.uv_layer:
            self.caster.free()
            self.report({'ERROR'}, "Objektilla ei ole UV-karttaa!")
            return {'CANCELLED'}

        RUNTIME.source_path_3d.clear()
        RUNTIME.source_path_uv.clear()
        RUNTIME.path_arc_lengths.clear()
        RUNTIME.total_path_length = 0.0
        self.drawing = False

        if RUNTIME.draw_handle is None:
            RUNTIME.draw_handle = bpy.types.SpaceView3D.draw_handler_add(
                draw_viewport_overlay, (), 'WINDOW', 'POST_VIEW'
            )

        context.window.cursor_modal_set('CROSSHAIR')
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Piirrä lähdepolku 3D-pintaan pitämällä LMB pohjassa.")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        area, win_reg, rv3d = get_view3d_window_region(context, event)
        if area: area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'RET', 'ESC'}:
            self._finalize_path(context)
            self.caster.free()
            context.window.cursor_modal_restore()
            return {'FINISHED'}

        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} or event.alt:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if win_reg and (win_reg.x <= event.mouse_x <= win_reg.x + win_reg.width and
                            win_reg.y <= event.mouse_y <= win_reg.y + win_reg.height):
                self.drawing = True
                self._add_point(context, event)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE' and self.drawing:
            self.drawing = False
            self._finalize_path(context)
            self.caster.free()
            context.window.cursor_modal_restore()
            return {'FINISHED'}

        if event.type == 'MOUSEMOVE' and self.drawing:
            self._add_point(context, event)
            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _add_point(self, context, event):
        pt3d, uv = self.caster.raycast_screen(context, event)
        if pt3d is not None and uv is not None:
            if not RUNTIME.source_path_3d or (pt3d - RUNTIME.source_path_3d[-1]).length > 0.002:
                RUNTIME.source_path_3d.append(pt3d)
                RUNTIME.source_path_uv.append(uv)

    def _finalize_path(self, context):
        uvs = RUNTIME.source_path_uv
        if len(uvs) < 2:
            self.report({'WARNING'}, "Polku jäi liian lyhyeksi.")
            return
            
        RUNTIME.path_arc_lengths = [0.0]
        total = 0.0
        for i in range(1, len(uvs)):
            total += (uvs[i] - uvs[i-1]).length
            RUNTIME.path_arc_lengths.append(total)
            
        if context.scene.mnpp_settings.close_path:
            total += (uvs[0] - uvs[-1]).length
            RUNTIME.path_arc_lengths.append(total)
            
        RUNTIME.total_path_length = total
        context.scene.mnpp_settings.tool_mode = 'PATH_CLONE'
        self.report({'INFO'}, f"Kloonipolku tallennettu ({len(uvs)} pistettä).")


# =====================================================================
# 5. ASETUKSET & KÄYTTÖLIITTYMÄ
# =====================================================================

class MNPP_Settings(bpy.types.PropertyGroup):
    tool_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('PAINT', "Multi-Paint (Normal Brush)", "Maalaa kaikille valituille nodeille yhtäaikaa"),
            ('CLONE', "Multi-Clone (Self-Source)", "Jokainen node kloonaa itseään omasta lähteestään"),
            ('PATH_CLONE', "Path Clone (Loop / Ping-Pong)", "Kloonaa piirrettyä polkua pitkin kaikille nodeille"),
            ('HEAL', "Multi-Healing Brush", "Saumaton taajuuseroteltu korjaussivellin kaikille nodeille"),
            ('INPAINT_MASK', "OpenCV Inpaint Brush", "Maalaa punainen viiva 3D-pintaan -> OpenCV poistaa kohdan"),
        ],
        default='PAINT'
    )
    use_rake: bpy.props.BoolProperty(
        name="Rake (Cursor Angle)",
        description="Kääntää sivellintä ja kloonauslähdettä dynaamisesti kursorin liikesuunnan mukaan",
        default=True
    )
    aligned_clone: bpy.props.BoolProperty(
        name="Aligned Clone",
        description="Lähde liikkuu suhteessa siveltimen aloituspisteeseen",
        default=True
    )
    close_path: bpy.props.BoolProperty(
        name="Close Path (Continuous Loop)",
        description="Suljettu = jatkuva loop ympäri. Avoin = ping-pong päästä päähän",
        default=False
    )
    path_speed: bpy.props.FloatProperty(
        name="Path Speed",
        description="Kuinka nopeasti lähde etenee polkua pitkin verrattuna siveltimen liikkeeseen",
        default=1.0, min=0.05, max=10.0
    )
    inpaint_radius: bpy.props.IntProperty(
        name="Inpaint Radius",
        description="OpenCV Inpaint -naapuruston säde pikseleinä",
        default=5, min=1, max=50
    )
    inpaint_algo: bpy.props.EnumProperty(
        name="Algorithm",
        items=[
            ('TELEA', "Fast Marching (Telea)", "Nopea ja terävä reunan jatke"),
            ('NS', "Navier-Stokes (Fluid)", "Virtausmekaniikkaan perustuva jatkuvuus"),
        ],
        default='TELEA'
    )
    auto_inpaint: bpy.props.BoolProperty(
        name="Process Immediately on Release",
        description="Suorittaa OpenCV Inpaintin heti kun päästät punaisen viivan maalauksesta irti",
        default=True
    )
    show_overlay: bpy.props.BoolProperty(name="Show 3D Ring & Path Overlay", default=True)
    show_path_source_preview: bpy.props.BoolProperty(
        name="Show Path Source Preview",
        description="Näytä Path Clone -tilassa esikatselurengas siitä alueesta, jota kopioidaan",
        default=True
    )
    brush_tex_channel: bpy.props.EnumProperty(
        name="Brush Texture Channel",
        description="Mitä Brush Texture -kanavaa käytetään siveltimen maskina",
        items=[
            ('NONE', "Disabled", "Älä käytä sivellintekstuuria"),
            ('LUMINANCE', "Luminance", "Käytä tekstuurin luminanssia/kirkkautta"),
            ('ALPHA', "Alpha", "Käytä tekstuurin alpha-kanavaa"),
            ('RGB', "RGB Average", "Käytä värien keskiarvoa"),
        ],
        default='NONE'
    )


class VIEW3D_PT_multi_node_paint_pro(bpy.types.Panel):
    bl_label = "Multi-Node Paint & Clone Pro"
    bl_idname = "VIEW3D_PT_multi_node_paint_pro"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'
    bl_parent_id = "VIEW3D_PT_tools_brush_settings"

    def draw(self, context):
        layout = self.layout
        s = context.scene.mnpp_settings
        nodes = get_selected_image_nodes(context)

        box = layout.box()
        row = box.row()
        row.label(text=f"Selected Image Nodes: {len(nodes)}", icon='NODE_TEXTURE')
        if not nodes:
            box.label(text="Select Image Nodes in Shader Editor!", icon='ERROR')
        else:
            col = box.column(align=True)
            for idx, n in enumerate(nodes):
                col.label(text=f"{idx+1}. {n.name} ({n.image.name})", icon='IMAGE_DATA')

        layout.separator()
        layout.prop(s, "tool_mode", text="")
        
        row = layout.row(align=True)
        row.prop(s, "use_rake", icon='CON_ROTLIKE', toggle=True)
        row.prop(s, "show_overlay", text="3D Ring", icon='OVERLAY', toggle=True)

        if s.tool_mode in {'CLONE', 'HEAL'}:
            cbox = layout.box()
            cbox.label(text="Clone / Heal Settings:", icon='BRUSH_DATA')
            cbox.prop(s, "aligned_clone")
            cbox.label(text="Ctrl + LMB on mesh sets source", icon='INFO')

        elif s.tool_mode == 'PATH_CLONE':
            pbox = layout.box()
            pbox.label(text="Path Clone Settings:", icon='CURVE_PATH')
            pbox.operator("paint.mnpp_draw_path", text="1. Draw Source Path", icon='GREASEPENCIL')
            pbox.prop(s, "close_path", icon='IPO_CIRC' if s.close_path else 'IPO_BACK')
            pbox.prop(s, "path_speed")
            pbox.prop(s, "show_path_source_preview", icon='IMAGE_ZDEPTH')
            mode_txt = "Mode: Continuous Loop" if s.close_path else "Mode: Open Ping-Pong"
            pbox.label(text=mode_txt, icon='FILE_REFRESH')

        elif s.tool_mode == 'INPAINT_MASK':
            ibox = layout.box()
            ibox.label(text="OpenCV Multi-Inpaint:", icon='MOD_MASK')
            if not HAS_CV2: ibox.label(text="cv2 missing! Install opencv-python", icon='ERROR')
            ibox.prop(s, "inpaint_algo")
            ibox.prop(s, "inpaint_radius")
            ibox.prop(s, "auto_inpaint")
            ibox.label(text="Paint red stroke -> release LMB to heal", icon='INFO')

        tbox = layout.box()
        tbox.label(text="Brush Texture Mask:", icon='TEXTURE')
        tbox.prop(s, "brush_tex_channel", text="")

        kbox = layout.box()
        kcol = kbox.column(align=True)
        kcol.label(text="Active Shortcuts:", icon='KEYINGSET')
        kcol.label(text="• Undo / Redo: Ctrl+Z / Ctrl+Shift+Z")
        kcol.label(text="• Size: F (drag) or [ / ]")
        kcol.label(text="• Strength: Shift + F (drag)")

        layout.separator()
        col = layout.column(align=True)
        col.scale_y = 1.6
        btn_txt = "BRUSH ACTIVE (ESC / RMB to Stop)" if RUNTIME.is_modal_active else "Activate Multi-Node Brush"
        btn_icon = 'REC' if RUNTIME.is_modal_active else 'BRUSH_DATA'
        col.operator("paint.mnpp_modal_stroke", text=btn_txt, icon=btn_icon, depress=RUNTIME.is_modal_active)


# =====================================================================
# REKISTERÖINTI
# =====================================================================

classes = (
    MNPP_Settings,
    PAINT_OT_multi_node_modal,
    PAINT_OT_draw_clone_path,
    VIEW3D_PT_multi_node_paint_pro,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mnpp_settings = bpy.props.PointerProperty(type=MNPP_Settings)

def unregister():
    if RUNTIME.draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(RUNTIME.draw_handle, 'WINDOW')
        RUNTIME.draw_handle = None
    del bpy.types.Scene.mnpp_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
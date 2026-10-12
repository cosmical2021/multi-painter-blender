"""All operators: sync, fill, clone source, history, multi-paint (modal) and healing brush (modal)."""
import time
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty
from bpy.types import Operator

from . import paint_engine as pe
from . import utils


# --------------------------------------------------------------------------- #
# Shared runtime state (module level so it resets when the add-on reloads)
# --------------------------------------------------------------------------- #

class _Runtime:
    running: bool = False      # multi-paint modal is active
    stop: bool = False         # request the multi-paint modal to end
    paused: bool = False       # multi-paint idles while the healing brush owns the image
    healing: bool = False      # healing/clone modal is active
    heal_clone: bool = False   # the active modal runs in clone mode (Multi Clone Brush)
    heal_op = None             # the running MLTP_OT_healing_brush instance
    heal_stop: bool = False    # request the healing modal to end
    generation: int = 0        # bumped whenever pixels are changed outside the multi-paint modal
    history: pe.History = pe.History()


RT = _Runtime()


@persistent
def _on_load_post(_dummy) -> None:
    RT.running = RT.stop = RT.paused = RT.healing = RT.heal_stop = RT.heal_clone = False
    RT.heal_op = None
    RT.history.clear()


def layer_opacity(settings, slot) -> float:
    return float(slot.opacity) if settings.use_opacity else 1.0


def enabled_targets(scene, exclude=None, size: Optional[Tuple[int, int]] = None):
    """(slot, image) for every enabled, valid layer, optionally filtered by image size."""
    out = []
    for slot in scene.mltp_layers:
        img = slot.image
        if not slot.is_active or not utils.image_is_valid(img) or img is exclude:
            continue
        if size is not None and tuple(img.size) != tuple(size):
            continue
        out.append((slot, img))
    return out


def apply_group(group: List[pe.Patch], undo: bool, arrays: Dict[str, np.ndarray]) -> None:
    """Restore (undo) or reapply (redo) a patch group on the images and the given array cache."""
    touched = set()
    for p in (reversed(group) if undo else group):
        img = bpy.data.images.get(p.name)
        if not utils.image_is_valid(img):
            continue
        arr = arrays.get(p.name)
        if arr is None:
            arr = arrays[p.name] = utils.read_pixels(img)
        if arr.shape[0] < p.y1 or arr.shape[1] < p.x1:
            continue
        arr[p.y0:p.y1, p.x0:p.x1] = p.before if undo else p.after
        touched.add(p.name)
    for name in touched:
        utils.write_pixels(bpy.data.images[name], arrays[name], final=True)
    utils.tag_redraw_all()


# --------------------------------------------------------------------------- #
# Layer list management
# --------------------------------------------------------------------------- #

class MLTP_OT_sync_selected_nodes(Operator):
    bl_idname = "mltp.sync_selected_nodes"
    bl_label = "Sync Selected Nodes"
    bl_description = "Collect the images of the selected Image Texture nodes into the layer list"
    bl_options = {"REGISTER"}

    scope: EnumProperty(
        name="Scope",
        items=[
            ("EDITOR", "Shader Editor", "Selected nodes in the node tree shown in the Shader Editor"),
            ("OBJECT", "Active Object", "Selected Image Texture nodes in every material of the active object"),
        ],
        default="EDITOR",
    )
    append: BoolProperty(name="Append", description="Keep the existing layers", default=False)

    def execute(self, context):
        scene = context.scene
        s = scene.mltp_settings
        scope = self.scope
        if scope == "EDITOR" and not (context.space_data and context.space_data.type == "NODE_EDITOR"):
            scope = "OBJECT"  # invoked from a 3D view / image editor panel
        found = list(utils.collect_selected_image_nodes(context, scope))
        if not found:
            self.report({"WARNING"}, "No selected Image Texture nodes with an image found")
            return {"CANCELLED"}

        layers = scene.mltp_layers
        if not self.append:
            layers.clear()
        existing = {(sl.material.name if sl.material else "", sl.node_name) for sl in layers}

        ref = None
        paint_img = utils.get_active_paint_image(context)
        if layers and utils.image_is_valid(layers[0].image):
            ref = tuple(layers[0].image.size)
        elif utils.image_is_valid(paint_img):
            ref = tuple(paint_img.size)

        added = skipped = resized = 0
        for mat, node in found:
            key = (mat.name if mat else "", node.name)
            img = node.image
            if key in existing:
                continue
            if not utils.image_is_valid(img):
                skipped += 1
                continue
            if ref is None:
                ref = tuple(img.size)
            if tuple(img.size) != ref:
                if s.match_dimensions:
                    img.scale(*ref)
                    resized += 1
                else:
                    skipped += 1
                    continue
            slot = layers.add()
            slot.name = img.name
            slot.image = img
            slot.material = mat
            slot.node_name = node.name
            slot.uv_map = utils.find_uv_map_name(node)
            existing.add(key)
            added += 1

        if layers:
            scene.mltp_active_index = len(layers) - 1
        RT.generation += 1
        msg = f"Synced {added} layer(s)"
        if resized:
            msg += f", resized {resized}"
        if skipped:
            msg += f", skipped {skipped} (size mismatch or no pixel data)"
        self.report({"INFO"} if added else {"WARNING"}, msg)
        return {"FINISHED"}


class MLTP_OT_clear_layers(Operator):
    bl_idname = "mltp.clear_layers"
    bl_label = "Clear Layers"
    bl_description = "Remove all layers from the list (images are not deleted)"

    def execute(self, context):
        context.scene.mltp_layers.clear()
        context.scene.mltp_active_index = 0
        RT.generation += 1
        return {"FINISHED"}


class MLTP_OT_remove_layer(Operator):
    bl_idname = "mltp.remove_layer"
    bl_label = "Remove Layer"
    bl_description = "Remove the active layer from the list"

    @classmethod
    def poll(cls, context):
        return len(context.scene.mltp_layers) > 0

    def execute(self, context):
        scene = context.scene
        scene.mltp_layers.remove(scene.mltp_active_index)
        scene.mltp_active_index = max(0, min(scene.mltp_active_index, len(scene.mltp_layers) - 1))
        RT.generation += 1
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Fill
# --------------------------------------------------------------------------- #

class MLTP_OT_multi_fill(Operator):
    bl_idname = "mltp.multi_fill"
    bl_label = "Fill All Layers"
    bl_description = "Fill every enabled layer with the colour / pattern, using each layer's opacity and blend mode"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.mltp_layers) > 0 or utils.get_active_paint_image(context) is not None

    def execute(self, context):
        scene = context.scene
        s = scene.mltp_settings
        primary = utils.get_active_paint_image(context)
        targets = [(sl, img) for sl, img in enabled_targets(scene)]
        if s.fill_include_primary and utils.image_is_valid(primary) and all(img is not primary for _, img in targets):
            targets.append((None, primary))
        if not targets:
            self.report({"WARNING"}, "Nothing to fill")
            return {"CANCELLED"}

        base = list(utils.get_brush_color(context)) if s.fill_use_brush_color else list(s.fill_color)[:3]
        rec = pe.UndoRecorder()
        arrays: Dict[str, np.ndarray] = {}
        for slot, img in targets:
            arr = utils.read_pixels(img)
            h, w = arr.shape[:2]
            c1 = utils.gamma_color_for_image(base, img)
            c2 = utils.gamma_color_for_image(list(s.fill_color2)[:3], img)
            pattern = pe.build_pattern(h, w, s.fill_pattern, c1, c2, s.fill_checker_size)
            op = s.fill_opacity * (layer_opacity(s, slot) if slot else 1.0)
            mode = slot.blend_mode if slot else "MIX"
            rec.touch(img.name, arr, 0, 0, w, h)
            arr = pe.blend_apply(arr, pattern, np.full((h, w), op, dtype=np.float32), mode)
            arrays[img.name] = arr
            utils.write_pixels(img, arr, final=True)
        RT.history.push(rec.finish(arrays))
        RT.generation += 1
        utils.tag_redraw_all()
        self.report({"INFO"}, f"Filled {len(targets)} image(s)")
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Clone source
# --------------------------------------------------------------------------- #

class MLTP_OT_set_clone_source(Operator):
    bl_idname = "mltp.set_clone_source"
    bl_label = "Set Clone Source"
    bl_description = "Set the clone source image of all layers at once"
    bl_options = {"REGISTER", "UNDO"}

    mode: EnumProperty(
        name="Source",
        items=[
            ("SELF", "Each Layer Itself", "Every layer clones from its own image"),
            ("BRUSH", "Brush Clone Image", "Every layer clones from the brush's clone image"),
            ("ACTIVE", "Active Layer's Source", "Every layer clones from the active layer's clone image "
             "(or its image if none is set)"),
        ],
        default="SELF",
    )

    @classmethod
    def poll(cls, context):
        return len(context.scene.mltp_layers) > 0

    def execute(self, context):
        scene = context.scene
        layers = scene.mltp_layers
        shared = None
        if self.mode == "BRUSH":
            shared = utils.get_clone_image(context)
            if shared is None:
                self.report({"WARNING"}, "No clone image is set in the paint settings")
                return {"CANCELLED"}
        elif self.mode == "ACTIVE":
            act = layers[scene.mltp_active_index]
            shared = act.clone_image or act.image
        for slot in layers:
            slot.clone_image = slot.image if self.mode == "SELF" else shared
        if shared is not None:
            utils.set_clone_image(context, shared)
        RT.generation += 1
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Undo / redo of mirrored edits
# --------------------------------------------------------------------------- #

class MLTP_OT_history(Operator):
    bl_idname = "mltp.history"
    bl_label = "Multi-Layer Undo/Redo"
    bl_description = "Undo or redo the last mirrored edit on the layer images"

    action: EnumProperty(items=[("UNDO", "Undo", ""), ("REDO", "Redo", "")], default="UNDO")

    def execute(self, context):
        undo = self.action == "UNDO"
        group = RT.history.pop_undo() if undo else RT.history.pop_redo()
        if group is None:
            self.report({"INFO"}, "Nothing to " + self.action.lower())
            return {"CANCELLED"}
        apply_group(group, undo, {})
        RT.generation += 1
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Multi-layer painting (modal, pass-through)
# --------------------------------------------------------------------------- #

class MLTP_OT_multi_paint(Operator):
    """Mirror Blender's native strokes (draw / clone / fill / soften / smear) onto all layers.

    The native brush keeps painting the active image. A timer compares that image with
    its previous state; the changed region is turned back into "what the brush did"
    (colour + per-pixel coverage, or the clone source) and replayed on every other
    enabled layer with its own opacity and blend mode. Only the changed bounding box is
    processed, everything is numpy.
    """

    bl_idname = "mltp.multi_paint"
    bl_label = "Start Multi-Paint"
    bl_description = "Mirror native paint strokes onto all enabled layers"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.mltp_layers) > 0 and not RT.running

    # -- lifecycle ----------------------------------------------------------
    def invoke(self, context, event):
        wm = context.window_manager
        s = context.scene.mltp_settings
        self._timer = wm.event_timer_add(s.update_interval, window=context.window)
        self._from_view3d = context.area is not None and context.area.type == "VIEW_3D"
        self._skip_until = 0.0
        self._need_reset = True
        self._generation = RT.generation
        self._sign = 0  # clone offset direction: 0 = undecided, +1 / -1 once detected
        self._shift = None  # clone source shift recovered from the pixels (3D clone), per stroke
        self._recorder: Optional[pe.UndoRecorder] = None
        self._last_change = 0.0
        self.primary = None
        self.prev: Optional[np.ndarray] = None
        self.buf: Optional[np.ndarray] = None
        self.size = (0, 0)
        self.cache: Dict[str, np.ndarray] = {}
        self.src_cache: Dict[str, np.ndarray] = {}
        RT.running, RT.stop = True, False
        wm.modal_handler_add(self)
        self.report({"INFO"}, "Multi-Paint running: paint normally, strokes are mirrored to the layers")
        return {"RUNNING_MODAL"}

    def _end(self, context):
        self._finish_group(context)
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        RT.running = RT.stop = False

    def cancel(self, context):
        self._end(context)

    def modal(self, context, event):
        if RT.stop or not RT.running or (self._from_view3d and context.mode != "PAINT_TEXTURE"):
            self._end(context)
            return {"FINISHED"}
        if event.type == "TIMER":
            if not RT.paused:
                self._tick(context)
        elif event.value == "PRESS" and event.type in {"Z", "Y"} and (event.ctrl or event.oskey):
            # Blender undoes the native stroke on the active image; undo our mirrored edits too.
            redo = event.type == "Y" or event.shift
            self._finish_group(context)
            group = RT.history.pop_redo() if redo else RT.history.pop_undo()
            if group:
                apply_group(group, not redo, self.cache)
            self._skip_until = time.time() + 0.6
            self._need_reset = True
        return {"PASS_THROUGH"}

    # -- caches ---------------------------------------------------------------
    def _reset_caches(self, context) -> None:
        scene = context.scene
        self.primary = utils.get_active_paint_image(context)
        self.cache.clear()
        self.src_cache.clear()
        self.prev = self.buf = None
        self._generation = RT.generation
        if not utils.image_is_valid(self.primary):
            return
        self.size = tuple(self.primary.size)
        self.prev = utils.read_pixels(self.primary)
        self.buf = np.empty_like(self.prev)
        for _slot, img in enabled_targets(scene, exclude=self.primary, size=self.size):
            self.cache[img.name] = utils.read_pixels(img)

    def _source_array(self, img: bpy.types.Image) -> Optional[np.ndarray]:
        if img is self.primary:
            return self.prev
        if img.name in self.cache:
            return self.cache[img.name]
        if img.name not in self.src_cache and utils.image_is_valid(img):
            self.src_cache[img.name] = utils.read_pixels(img)
        return self.src_cache.get(img.name)

    def _finish_group(self, context) -> None:
        if self._recorder is None:
            return
        group = self._recorder.finish(self.cache)
        self._recorder = None
        self._shift = None
        RT.history.push(group)
        if group and context.scene.mltp_settings.push_undo_step:
            try:
                bpy.ops.ed.undo_push(message="MultiLayer Paint")
            except RuntimeError:
                pass

    # -- per timer tick ---------------------------------------------------------
    def _tick(self, context) -> None:
        now = time.time()
        if now < self._skip_until:
            return
        s = context.scene.mltp_settings
        primary = utils.get_active_paint_image(context)
        if (
            self._need_reset
            or primary is not self.primary
            or self._generation != RT.generation
            or (primary is not None and utils.image_is_valid(primary) and tuple(primary.size) != self.size)
        ):
            self._finish_group(context)
            self._reset_caches(context)
            self._need_reset = False
            return
        if self.prev is None:
            return

        cur = utils.read_pixels(primary, self.buf)
        bbox = pe.changed_bbox(self.prev, cur)
        if bbox is None:
            if self._recorder is not None and now - self._last_change > s.commit_delay:
                self._finish_group(context)
            return
        self._last_change = now
        try:
            self._propagate(context, s, cur, bbox)
        finally:
            self.prev, self.buf = cur, self.prev

    def _propagate(self, context, s, cur: np.ndarray, bbox: pe.BBox) -> None:
        scene = context.scene
        x0, y0, x1, y1 = bbox
        old = self.prev[y0:y1, x0:x1]
        new = cur[y0:y1, x0:x1]
        changed = np.any(old != new, axis=2)
        brush = utils.get_brush(context)
        tool = utils.brush_tool(brush)

        targets = [
            (slot, slot.image) for slot in scene.mltp_layers
            if slot.is_active and slot.image is not None and slot.image.name in self.cache
            and slot.image is not self.primary
        ]
        if not targets:
            return

        mode = "DELTA"  # soften / smear / erase: transfer the raw change
        if tool in {"DRAW", "FILL"} and (brush is None or brush.blend == "MIX"):
            mode = "DRAW"
        elif tool == "CLONE" and brush is not None:
            mode = "CLONE"

        coverage = color = None
        off = (0.0, 0.0)
        sign = 1
        csm, shared = s.clone_source_mode, None
        if mode == "DRAW":
            base = np.array(utils.get_brush_color(context), dtype=np.float32)
            rgba = np.append(base, 1.0).astype(np.float32)
            cands = [
                rgba,
                np.append(utils.srgb_to_linear(base), 1.0).astype(np.float32),
                np.append(utils.linear_to_srgb(base), 1.0).astype(np.float32),
            ]
            color, coverage = pe.pick_color(old, new, changed, cands)
        elif mode == "CLONE":
            csm = s.clone_source_mode
            shared = utils.get_clone_image(context)
            src_arr = self._source_array(shared or self.primary)
            off = utils.get_clone_offset(context)
            if abs(off[0]) + abs(off[1]) >= 1e-7:
                # The sign convention of clone_offset is detected from the primary's own stroke:
                # the hypothesis that explains the pixels with the lower residual wins.
                results = []
                for sg in ((self._sign,) if self._sign else (1, -1)):
                    patch, valid = pe.sample_shifted(src_arr, bbox, self.size, sg * off[0], sg * off[1])
                    a, err = pe.fit_coverage(old, new, patch, changed & valid)
                    results.append((err, sg, a))
                results.sort(key=lambda r: r[0])
                _err, sign, coverage = results[0]
                if not self._sign and len(results) == 2 and results[0][0] < 0.5 * results[1][0]:
                    self._sign = sign
            else:
                # 3D clone reports no UV offset. Recover where the stroke came from by matching the
                # painted pixels against the source, so every layer can sample *its own* texture there.
                off = self._shift if csm != "SHARED" else None
                if off is None and csm != "SHARED" and src_arr is not None \
                        and src_arr.shape[:2] == (self.size[1], self.size[0]):
                    off = pe.estimate_source_shift(src_arr, new, changed, bbox)
                    self._shift = off
                if off is None:
                    if csm == "SHARED":
                        mode = "COPY"  # no usable offset: copy the primary's cloned result
                        off = (0.0, 0.0)
                    else:
                        # OWN / PER_LAYER promise layer-specific pixels; copying the primary's
                        # colours would be wrong, so leave the layers untouched instead.
                        if not getattr(self, "_clone_warned", False):
                            self._clone_warned = True
                            report = getattr(self, "report", None)
                            if report is not None:
                                report({"WARNING"},
                                       "Clone position not recoverable from this stroke; layers left "
                                       "unchanged. Use the Multi Clone Brush for exact per-layer cloning")
                        return
                else:
                    patch, valid = pe.sample_shifted(src_arr, bbox, self.size, *off)
                    coverage, _err = pe.fit_coverage(old, new, patch, changed & valid)

        if self._recorder is None:
            self._recorder = pe.UndoRecorder()
        for slot, img in targets:
            arr = self.cache[img.name]
            region = arr[y0:y1, x0:x1]
            op = layer_opacity(s, slot)
            if mode == "DRAW":
                col = color
                if img.is_float != self.primary.is_float:  # byte (sRGB) <-> float (linear) layers
                    col = color.copy()
                    col[:3] = utils.srgb_to_linear(col[:3]) if img.is_float else utils.linear_to_srgb(col[:3])
                result = pe.blend_apply(region, col, coverage * op, slot.blend_mode)
            elif mode == "CLONE":
                if csm == "OWN":
                    src_img = img
                elif csm == "PER_LAYER":
                    src_img = slot.clone_image or img
                else:
                    src_img = shared or img
                src_arr = self._source_array(src_img)
                if src_arr is None:
                    continue
                patch, valid = pe.sample_shifted(src_arr, bbox, self.size, sign * off[0], sign * off[1])
                result = pe.blend_apply(region, patch, coverage * valid * op, slot.blend_mode)
            elif mode == "COPY":
                result = pe.blend_apply(region, new, changed.astype(np.float32) * op, slot.blend_mode)
            else:
                result = pe.transfer_delta(region, old, new, op)
            self._recorder.touch(img.name, arr, x0, y0, x1, y1)
            arr[y0:y1, x0:x1] = result
            utils.write_pixels(img, arr)
        utils.tag_redraw_all()


class MLTP_OT_stop_multi_paint(Operator):
    bl_idname = "mltp.stop_multi_paint"
    bl_label = "Stop Multi-Paint"
    bl_description = "Stop mirroring strokes"

    @classmethod
    def poll(cls, context):
        return RT.running

    def execute(self, context):
        RT.stop = True
        return {"FINISHED"}


# --------------------------------------------------------------------------- #
# Healing brush (modal, replaces the native stroke)
# --------------------------------------------------------------------------- #

class MLTP_OT_stop_healing(Operator):
    bl_idname = "mltp.stop_healing"
    bl_label = "Stop Healing Brush"
    bl_description = "Leave healing mode"

    def execute(self, context):
        RT.heal_stop = True
        RT.healing = False
        return {"FINISHED"}


class MLTP_OT_healing_brush(Operator):
    """Photoshop-style healing brush that works on the paint image and all layers at once.

    Ctrl+LMB (or Alt+LMB) picks the source point. LMB drags heal from the source, keeping a
    fixed offset (aligned). Ctrl+Z / Ctrl+Shift+Z undo / redo. Esc or RMB leaves.
    Works in the Image Editor and in the 3D viewport (Texture Paint mode, UV based).
    """

    bl_idname = "mltp.healing_brush"
    bl_label = "Healing Brush"
    bl_description = "Heal with a clone source, matching colour/luminance (Ctrl+Click sets the source)"
    bl_options = {"REGISTER"}

    clone_mode: BoolProperty(
        name="Clone Mode",
        description="Plain clone instead of healing: copy each layer's own pixels from the "
        "source position without adapting colour or luminance",
        default=False,
        options={"SKIP_SAVE"},
    )

    @staticmethod
    def find_view3d_area(context) -> Optional[bpy.types.Area]:
        """Largest 3D viewport in the current window.

        Lets the button also work from the Properties editor's Tool tab, where
        Blender mirrors the viewport's sidebar panels.
        """
        areas = [a for a in context.screen.areas if a.type == "VIEW_3D"]
        return max(areas, key=lambda a: a.width * a.height) if areas else None

    @staticmethod
    def block_reason(context) -> str:
        """Why the brush cannot start here ("" when it can)."""
        area = context.area
        if area is None:
            return "No editor"
        if area.type == "IMAGE_EDITOR":
            if context.space_data.image is None:
                return "Open an image in this Image Editor (or use this panel in the 3D viewport)"
            return ""
        if area.type != "VIEW_3D" and MLTP_OT_healing_brush.find_view3d_area(context) is None:
            return "Use it in the Image Editor or the 3D viewport"
        if context.mode != "PAINT_TEXTURE":
            return "Switch the 3D viewport to Texture Paint mode"
        ob = context.active_object
        if ob is None or ob.type != "MESH":
            return "Select a mesh object"
        return ""

    @classmethod
    def poll(cls, context):
        reason = cls.block_reason(context)
        if reason:
            cls.poll_message_set(reason)
        return not reason

    # -- lifecycle ----------------------------------------------------------
    def invoke(self, context, event):
        scene = context.scene
        s = scene.mltp_settings
        cur, was_active = RT.heal_op, RT.healing
        if cur is not None:  # end any live (or lingering) brush now; its modal exits on the next event
            cur._end(context)
        RT.heal_op = None
        RT.healing = False
        RT.heal_stop = False
        if was_active and (cur is None or bool(cur.clone_mode) == bool(self.clone_mode)):
            return {"CANCELLED"}  # same button pressed again: plain toggle off
        # other mode was active (or nothing was): start this mode
        primary = utils.get_active_paint_image(context)
        if not utils.image_is_valid(primary) and len(scene.mltp_layers):
            primary = scene.mltp_layers[scene.mltp_active_index].image
        if not utils.image_is_valid(primary):
            self.report({"ERROR"}, "No paint image with pixel data")
            return {"CANCELLED"}

        self.primary = primary
        self.size = tuple(primary.size)

        # The brush is not bound to one editor: every mouse event is resolved to the
        # Image Editor or 3D viewport under the cursor. The picker enables the 3D path.
        self.picker: Optional[utils.MeshUVPicker] = None
        ob = context.active_object
        if context.mode == "PAINT_TEXTURE" and ob is not None and ob.type == "MESH":
            uv_name = next((sl.uv_map for sl in scene.mltp_layers if sl.image is primary), "")
            picker = utils.MeshUVPicker(context, ob, uv_name)
            if picker.has_uv:
                self.picker = picker
            else:
                picker.free()
        if self.picker is None and context.area.type == "VIEW_3D":
            self.report({"ERROR"}, "The active object has no UV map")
            return {"CANCELLED"}

        images = [primary]
        if s.sample_mode == "ALL":
            images += [img for _sl, img in enabled_targets(scene, exclude=primary, size=self.size)]
        self.images = {img.name: img for img in images}
        self.arrays = {img.name: utils.read_pixels(img) for img in images}

        self.src_uv: Optional[Tuple[float, float]] = None
        self.offset_px: Optional[Tuple[float, float]] = None
        self.anchor_px: Tuple[float, float] = (0.0, 0.0)  # dest texel of the first (aligning) click
        self.stroking = False
        self.last_px = (0.0, 0.0)
        self.last_mouse = (0.0, 0.0)
        self.uv_area_type = ""
        self.texels_per_px = 1.0
        self.remain = 0.0
        self.radius = 16.0
        self.dirty = False
        self.last_flush = 0.0
        self.recorder: Optional[pe.UndoRecorder] = None

        RT.healing = True
        RT.heal_clone = bool(self.clone_mode)
        RT.heal_op = self
        self._dead = False
        RT.paused = True
        tool_name = "Cloning" if self.clone_mode else "Healing"
        context.window.cursor_modal_set("PAINT_CROSS")
        context.workspace.status_text_set(
            f"{tool_name}: Ctrl+LMB set source  |  LMB drag  |  Ctrl+Z undo  |  Esc / RMB exit"
        )
        self.report({"INFO"}, f"{tool_name} active: Ctrl+Click to set the source, then drag. Esc to exit")
        utils.tag_redraw_all()
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _end(self, context):
        if getattr(self, "_dead", False):  # already ended (e.g. by a brush switch)
            return
        self._dead = True
        try:
            self._end_stroke()
            self._flush(force=True, final=True)
        finally:  # whatever happens, never leave the brush flagged active / the UI locked
            if self.picker is not None:
                self.picker.free()
            context.window.cursor_modal_restore()
            context.workspace.status_text_set(None)
            if RT.heal_op is self or RT.heal_op is None:  # don't reset state a newer brush owns
                RT.heal_op = None
                RT.healing = False
                RT.paused = False
            RT.generation += 1
            utils.tag_redraw_all()

    def cancel(self, context):
        self._end(context)

    # -- helpers --------------------------------------------------------------
    def _locate(self, context, event):
        """(area, region) of the supported editor under the mouse, or (None, None)."""
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                if self.picker is None:
                    continue
            elif area.type == "IMAGE_EDITOR":
                img = area.spaces.active.image
                if img is None or tuple(img.size) != self.size:
                    continue
            else:
                continue
            for r in area.regions:
                if r.type == "WINDOW" and r.x <= event.mouse_x < r.x + r.width \
                        and r.y <= event.mouse_y < r.y + r.height:
                    return area, r
        return None, None

    def _inside(self, context, event) -> bool:
        return self._locate(context, event)[1] is not None

    def _uv(self, context, event) -> Optional[Tuple[float, float]]:
        area, region = self._locate(context, event)
        if region is None:
            return None
        self.uv_area_type = area.type
        coord = (event.mouse_x - region.x, event.mouse_y - region.y)
        if area.type == "VIEW_3D":
            return self.picker.pick(region, region.data, coord)
        return utils.image_editor_uv(region, *coord)

    def _update_radius(self, context, event) -> None:
        area, region = self._locate(context, event)
        if region is None:
            return
        screen_radius = max(utils.get_brush_size(context) * 0.5, 1.0)
        coord = (event.mouse_x - region.x, event.mouse_y - region.y)
        if area.type == "VIEW_3D":
            r = self.picker.texel_radius(region, region.data, coord, screen_radius, self.size)
            if r is not None:
                self.radius = r
        else:
            self.radius = screen_radius / utils.image_editor_screen_px_per_texel(region, self.primary)
        self.texels_per_px = self.radius / screen_radius

    def _flush(self, force: bool = False, final: bool = False) -> None:
        if not self.dirty:
            if final:  # pixels were already written on mouse release; still invalidate the GPU copy
                for img in self.images.values():
                    utils.refresh_gpu_texture(img)
                utils.tag_redraw_all()
            return
        now = time.time()
        if not force and now - self.last_flush < 0.15:
            return
        # The 3D viewport shows a cached GPU texture, so every flush must free it
        # or the heal stays invisible on the mesh until something re-uploads it.
        for name, img in self.images.items():
            utils.write_pixels(img, self.arrays[name], final=True)
        utils.tag_redraw_all()
        self.dirty = False
        self.last_flush = now
        if not final and not bpy.app.background:
            # gl_free only refreshes engines that re-fetch the GPU texture each draw
            # (EEVEE); a Cycles rendered viewport restarts only on a depsgraph image
            # update, and the depsgraph never runs while stroke events keep coming.
            try:
                for img in self.images.values():
                    img.update_tag()
                bpy.context.view_layer.update()
            except Exception:  # noqa: BLE001 - display refresh must never kill the stroke
                pass
            # Dabs can outpace the event queue, and Blender only redraws once the queue is
            # empty — without this swap a whole stroke stays invisible until mouse release.
            try:
                bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
            except RuntimeError:
                pass

    def _end_stroke(self) -> None:
        if not self.stroking:
            return
        self.stroking = False
        if self.recorder:
            RT.history.push(self.recorder.finish(self.arrays))
        self.recorder = None

    def _dab(self, context, px: float, py: float, pressure: float) -> None:
        s = context.scene.mltp_settings
        brush = utils.get_brush(context)
        strength = s.healing_strength
        radius = self.radius
        if brush is not None:
            if brush.use_pressure_strength:
                strength *= pressure
            if brush.use_pressure_size:
                radius *= max(pressure, 0.05)
        ox, oy = self.offset_px
        ax, ay = self.anchor_px
        # Mirrored projection around the source point: the destination texel at
        # anchor + d samples the source at src_point - d instead of src_point + d.
        mirror_x, mirror_y = s.clone_mirror_x, s.clone_mirror_y
        sx = (2.0 * ax + ox - px) if mirror_x else (px + ox)
        sy = (2.0 * ay + oy - py) if mirror_y else (py + oy)
        r = int(np.ceil(radius)) + 2
        albedo_match = False if self.clone_mode else s.albedo_match
        for name, arr in self.arrays.items():
            self.recorder.touch(name, arr, int(px) - r, int(py) - r, int(px) + r + 1, int(py) + r + 1)
            box = pe.heal_dab(
                arr, arr, px, py, sx, sy, radius, strength,
                s.healing_falloff, albedo_match, s.healing_method,
                flip_x=mirror_x, flip_y=mirror_y,
            )
            if box is not None:
                self.dirty = True

    def _stroke_to(self, context, uv, pressure: float) -> None:
        w, h = self.size
        cur = (uv[0] * w, uv[1] * h)
        dx, dy = cur[0] - self.last_px[0], cur[1] - self.last_px[1]
        length = float(np.hypot(dx, dy))
        if length <= 1e-6:
            return
        step = max(1.0, 2.0 * self.radius * context.scene.mltp_settings.healing_spacing / 100.0)
        t = step - self.remain
        while t <= length:
            f = t / length
            self._dab(context, self.last_px[0] + dx * f, self.last_px[1] + dy * f, pressure)
            t += step
        self.remain = length - (t - step)
        self.last_px = cur

    # -- events ---------------------------------------------------------------
    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as exc:  # never leave the brush stuck "active"
            self.report({"ERROR"}, f"Healing brush stopped: {exc}")
            self._end(context)
            return {"CANCELLED"}

    def _modal(self, context, event):
        if getattr(self, "_dead", False):  # ended by a brush switch: just release the handler
            return {"FINISHED"}
        if RT.heal_stop:
            RT.heal_stop = False
            self._end(context)
            return {"FINISHED"}
        if event.type == "ESC" or (event.type == "RIGHTMOUSE" and event.value == "PRESS" and self._inside(context, event)):
            self._end(context)
            return {"FINISHED"}

        if event.value == "PRESS" and event.type in {"Z", "Y"} and (event.ctrl or event.oskey) and not self.stroking:
            redo = event.type == "Y" or event.shift
            group = RT.history.pop_redo() if redo else RT.history.pop_undo()
            if group:
                apply_group(group, not redo, self.arrays)
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and self._inside(context, event):
                uv = self._uv(context, event)
                if uv is None:
                    return {"RUNNING_MODAL"}
                if event.ctrl or event.alt:
                    self.src_uv, self.offset_px = uv, None
                    self.report({"INFO"}, "Source point set")
                    return {"RUNNING_MODAL"}
                if self.src_uv is None:
                    self.report({"WARNING"}, "Ctrl+Click first to set the source point")
                    return {"RUNNING_MODAL"}
                w, h = self.size
                if self.offset_px is None:  # aligned: first stroke fixes the source/destination offset
                    self.offset_px = ((self.src_uv[0] - uv[0]) * w, (self.src_uv[1] - uv[1]) * h)
                    self.anchor_px = (uv[0] * w, uv[1] * h)
                self._update_radius(context, event)
                self.stroking = True
                self.recorder = pe.UndoRecorder()
                self.last_px = (uv[0] * w, uv[1] * h)
                self.last_mouse = (event.mouse_x, event.mouse_y)
                self.remain = 0.0
                self._dab(context, *self.last_px, event.pressure)
                self._flush(force=True)
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE" and self.stroking:
                self._flush(force=True)
                self._end_stroke()
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if event.type in {"MOUSEMOVE", "INBETWEEN_MOUSEMOVE"} and self.stroking:
            uv = self._uv(context, event)
            if uv is not None:
                self._update_radius(context, event)
                w, h = self.size
                cur = (uv[0] * w, uv[1] * h)
                screen_d = float(np.hypot(event.mouse_x - self.last_mouse[0],
                                          event.mouse_y - self.last_mouse[1]))
                texel_d = float(np.hypot(cur[0] - self.last_px[0], cur[1] - self.last_px[1]))
                limit = max(4.0 * self.texels_per_px * max(screen_d, 1.0), 3.0 * self.radius)
                if self.uv_area_type == "VIEW_3D" and texel_d > limit:
                    # The cursor crossed a UV seam onto another island: continue the
                    # stroke there instead of smearing dabs across the texture between.
                    self.last_px, self.remain = cur, 0.0
                    self._dab(context, *cur, event.pressure)
                else:
                    self._stroke_to(context, uv, event.pressure)
                self._flush()
            self.last_mouse = (event.mouse_x, event.mouse_y)
            return {"RUNNING_MODAL"}

        return {"PASS_THROUGH"}


classes = (
    MLTP_OT_sync_selected_nodes,
    MLTP_OT_clear_layers,
    MLTP_OT_remove_layer,
    MLTP_OT_multi_fill,
    MLTP_OT_set_clone_source,
    MLTP_OT_history,
    MLTP_OT_multi_paint,
    MLTP_OT_stop_multi_paint,
    MLTP_OT_healing_brush,
    MLTP_OT_stop_healing,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)
    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)


def unregister() -> None:
    RT.stop = True
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

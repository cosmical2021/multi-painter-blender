"""Panels and the layer UIList."""
import bpy
from bpy.types import Panel, UIList

from . import utils
from .operators import RT


class MLTP_UL_layers(UIList):
    bl_idname = "MLTP_UL_layers"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        paint_img = utils.get_active_paint_image(context)
        is_paint_target = item.image is not None and item.image is paint_img
        row = layout.row(align=True)
        row.prop(item, "is_active", text="", emboss=False, icon="HIDE_OFF" if item.is_active else "HIDE_ON")
        name = row.row(align=True)
        name.alert = is_paint_target  # highlight the image the native brush paints on
        name.label(
            text=item.image.name if item.image else "<missing image>",
            icon="BRUSH_DATA" if is_paint_target else "IMAGE_DATA",
        )
        row.prop(item, "opacity", text="", slider=True)
        row.prop(item, "blend_mode", text="")


def draw_layer_ui(layout: bpy.types.UILayout, context: bpy.types.Context, node_editor: bool = False) -> None:
    scene = context.scene
    s = scene.mltp_settings
    scope = "EDITOR" if node_editor else "OBJECT"

    row = layout.row(align=True)
    op = row.operator("mltp.sync_selected_nodes", text="Sync Selected Nodes", icon="FILE_REFRESH")
    op.scope, op.append = scope, False
    op = row.operator("mltp.sync_selected_nodes", text="", icon="ADD")
    op.scope, op.append = scope, True
    row.operator("mltp.clear_layers", text="", icon="TRASH")

    row = layout.row()
    row.template_list("MLTP_UL_layers", "", scene, "mltp_layers", scene, "mltp_active_index", rows=4)
    col = row.column(align=True)
    col.operator("mltp.remove_layer", text="", icon="REMOVE")

    layers = scene.mltp_layers
    if 0 <= scene.mltp_active_index < len(layers):
        act = layers[scene.mltp_active_index]
        box = layout.box()
        box.prop(act, "clone_image", text="Clone Source")
        if act.uv_map:
            box.label(text=f"UV Map: {act.uv_map}", icon="GROUP_UVS")

    if node_editor:
        return

    col = layout.column(align=True)
    if RT.running:
        col.operator("mltp.stop_multi_paint", text="Stop Multi-Paint", icon="PAUSE", depress=True)
    else:
        col.operator("mltp.multi_paint", text="Start Multi-Paint", icon="PLAY")
    row = col.row(align=True)
    row.operator("mltp.history", text="Undo", icon="LOOP_BACK").action = "UNDO"
    row.operator("mltp.history", text="Redo", icon="LOOP_FORWARDS").action = "REDO"

    box = layout.box()
    box.label(text="Options")
    box.prop(s, "match_dimensions")
    box.prop(s, "use_opacity")
    box.prop(s, "update_interval")
    box.prop(s, "commit_delay")
    box.prop(s, "push_undo_step")


class _SubPanelMixin:
    bl_category = "Tool"
    bl_options = {"DEFAULT_CLOSED"}


def draw_clone(layout, context):
    s = context.scene.mltp_settings
    # Dedicated brush: exact per-layer clone (each layer samples its own pixels at the
    # source offset), unlike the mirroring below which must guess the native brush's source.
    if RT.healing and RT.heal_clone:
        layout.operator("mltp.stop_healing", text="Clone Brush ACTIVE - click to stop", icon="PAUSE", depress=True)
    else:
        from .operators import MLTP_OT_healing_brush
        op = layout.operator("mltp.healing_brush", text="Multi Clone Brush", icon="BRUSH_DATA")
        op.clone_mode = True
        reason = MLTP_OT_healing_brush.block_reason(context)
        if reason:
            box = layout.box()
            box.alert = True
            box.label(text=reason, icon="ERROR")
    col = layout.column(align=True)
    col.label(text="Ctrl+Click: source, drag: clone", icon="INFO")
    col.label(text="Every layer clones from its own image.")
    col.prop(s, "healing_strength")
    col.prop(s, "healing_falloff")
    row = layout.row(align=True)
    row.prop(s, "clone_mirror_x", text="Mirror Horizontal", toggle=True, icon="MOD_MIRROR")
    row.prop(s, "clone_mirror_y", text="Mirror Vertical", toggle=True, icon="MOD_MIRROR")

    layout.separator()
    layout.label(text="Native Clone Brush Mirroring")
    layout.prop(s, "clone_source_mode", text="")
    col = layout.column(align=True)
    for mode, label in (("SELF", "Each Layer Itself"), ("BRUSH", "Brush Clone Image"), ("ACTIVE", "Active Layer's Source")):
        col.operator("mltp.set_clone_source", text=label).mode = mode
    layout.label(text="Use the normal clone brush while Multi-Paint runs.", icon="INFO")
    layout.label(text="Own Image: node 1 clones from node 1, node 2 from node 2.")


def draw_fill(layout, context):
    s = context.scene.mltp_settings
    layout.prop(s, "fill_pattern")
    layout.prop(s, "fill_use_brush_color")
    if not s.fill_use_brush_color:
        layout.prop(s, "fill_color")
    if s.fill_pattern != "SOLID":
        layout.prop(s, "fill_color2")
    if s.fill_pattern == "CHECKER":
        layout.prop(s, "fill_checker_size")
    layout.prop(s, "fill_opacity")
    layout.prop(s, "fill_include_primary")
    layout.operator("mltp.multi_fill", icon="BRUSHES_ALL")


def draw_healing(layout, context):
    s = context.scene.mltp_settings
    col = layout.column(align=True)
    col.label(text="1. Press the button below", icon="INFO")
    col.label(text="2. Ctrl+Click: pick the source point")
    col.label(text="3. Click and drag over the blemish")
    col.label(text="Esc / right click leaves, Ctrl+Z undoes")
    if RT.healing and not RT.heal_clone:
        layout.operator("mltp.stop_healing", text="Healing ACTIVE - click to stop", icon="PAUSE", depress=True)
        layout.label(text="Ctrl+Click in the view to set the source", icon="EYEDROPPER")
    else:
        from .operators import MLTP_OT_healing_brush
        reason = MLTP_OT_healing_brush.block_reason(context)
        layout.operator("mltp.healing_brush", icon="BRUSH_DATA").clone_mode = False
        if reason:
            box = layout.box()
            box.alert = True
            box.label(text=reason, icon="ERROR")
    col = layout.column(align=True)
    col.prop(s, "healing_strength")
    col.prop(s, "healing_falloff")
    col.prop(s, "healing_spacing")
    row = layout.row(align=True)
    row.prop(s, "clone_mirror_x", text="Mirror Horizontal", toggle=True, icon="MOD_MIRROR")
    row.prop(s, "clone_mirror_y", text="Mirror Vertical", toggle=True, icon="MOD_MIRROR")
    layout.prop(s, "healing_method")
    layout.prop(s, "sample_mode")
    layout.prop(s, "albedo_match")
    from . import paint_engine
    if not paint_engine.HAS_CV2:
        layout.label(text="cv2 not found: using numpy blending", icon="INFO")


def _make_panels(space_type: str, prefix: str, poll):
    """Build the main panel and its sub-panels for one editor type."""
    main_id = f"MLTP_PT_main_{prefix}"

    class Main(Panel):
        bl_idname = main_id
        bl_label = "MultiLayer Paint"
        bl_space_type = space_type
        bl_region_type = "UI"
        bl_category = "Tool"

        @classmethod
        def poll(cls, context):
            return poll(context)

        def draw(self, context):
            draw_layer_ui(self.layout, context)

    def sub(name, label, fn):
        class Sub(_SubPanelMixin, Panel):
            bl_idname = f"MLTP_PT_{name}_{prefix}"
            bl_label = label
            bl_space_type = space_type
            bl_region_type = "UI"
            bl_parent_id = main_id

            @classmethod
            def poll(cls, context):
                return poll(context)

            def draw(self, context):
                fn(self.layout, context)

        Sub.__name__ = Sub.bl_idname
        return Sub

    Main.__name__ = main_id
    return [Main, sub("clone", "Multi Clone", draw_clone), sub("fill", "Multi Fill", draw_fill),
            sub("heal", "Healing Brush", draw_healing)]


class MLTP_PT_nodes(Panel):
    bl_idname = "MLTP_PT_nodes"
    bl_label = "MultiLayer Paint"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Tool"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space is not None and space.tree_type == "ShaderNodeTree"

    def draw(self, context):
        draw_layer_ui(self.layout, context, node_editor=True)


classes = (
    MLTP_UL_layers,
    *_make_panels("IMAGE_EDITOR", "image", lambda c: True),
    *_make_panels("VIEW_3D", "view3d", lambda c: c.mode == "PAINT_TEXTURE"),
    MLTP_PT_nodes,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister() -> None:
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

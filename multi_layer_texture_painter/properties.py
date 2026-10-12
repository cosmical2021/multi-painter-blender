"""Property groups and scene properties."""
import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup

BLEND_ITEMS = [
    ("MIX", "Mix", "Replace with the painted colour"),
    ("MULTIPLY", "Multiply", "Darken by multiplying"),
    ("ADD", "Add", "Add the painted colour"),
    ("SUBTRACT", "Subtract", "Subtract the painted colour"),
    ("SCREEN", "Screen", "Lighten (inverse multiply)"),
    ("OVERLAY", "Overlay", "Multiply or screen depending on the layer colour"),
    ("DARKEN", "Darken", "Keep the darker value"),
    ("LIGHTEN", "Lighten", "Keep the lighter value"),
    ("DIFFERENCE", "Difference", "Absolute difference"),
]


class MLTP_LayerSlot(PropertyGroup):
    """One synced Image Texture node."""

    image: PointerProperty(type=bpy.types.Image, name="Image")
    material: PointerProperty(type=bpy.types.Material, name="Material")
    # Pointers to nodes are not possible in Blender, so the node is stored by name.
    node_name: bpy.props.StringProperty(name="Node")
    uv_map: bpy.props.StringProperty(name="UV Map", description="UV map found through the node's Vector input")
    clone_image: PointerProperty(
        type=bpy.types.Image,
        name="Clone Source",
        description="Image this layer clones from (empty: the brush clone image, then the layer itself)",
    )
    opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    blend_mode: EnumProperty(name="Blend", items=BLEND_ITEMS, default="MIX")
    is_active: BoolProperty(name="Enabled", description="Mirror strokes onto this layer", default=True)


class MLTP_Settings(PropertyGroup):
    match_dimensions: BoolProperty(
        name="Match Dimensions",
        description="Resize synced images that differ in size from the reference image "
        "(otherwise they are skipped). This modifies those images",
        default=True,
    )
    use_opacity: BoolProperty(name="Use Opacity", description="Apply each layer's opacity", default=True)
    update_interval: FloatProperty(
        name="Update Interval",
        description="Seconds between mirror updates while painting (raise for 4K+ textures)",
        default=0.1, min=0.02, max=2.0, subtype="TIME", unit="TIME",
    )
    commit_delay: FloatProperty(
        name="Stroke Idle Time",
        description="Seconds without changes after which a stroke counts as finished (undo grouping)",
        default=0.5, min=0.1, max=3.0, subtype="TIME", unit="TIME",
    )
    clone_source_mode: EnumProperty(
        name="Clone Source",
        items=[
            ("OWN", "Own Image", "Every layer clones from its own image, at the same position the "
             "painted image clones from (node 1 is painted from node 1, node 2 from node 2, ...)"),
            ("PER_LAYER", "Per-Layer Source", "Every layer clones from the Clone Source image set on "
             "that layer (falls back to its own image)"),
            ("SHARED", "Shared Source", "Every layer clones from the clone image set in Blender's "
             "paint settings"),
        ],
        default="OWN",
    )
    push_undo_step: BoolProperty(
        name="Blender Undo Step",
        description="Also call ed.undo_push after each mirrored stroke. May require an extra Ctrl+Z "
        "in Texture Paint mode, so it is off by default",
        default=False,
    )

    # --- healing brush -----------------------------------------------------
    healing_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    healing_falloff: EnumProperty(
        name="Falloff",
        items=[
            ("SMOOTH", "Smooth", ""),
            ("SPHERE", "Sphere", ""),
            ("ROOT", "Root", ""),
            ("SHARP", "Sharp", ""),
            ("LINEAR", "Linear", ""),
            ("CONSTANT", "Constant", ""),
        ],
        default="SMOOTH",
    )
    healing_spacing: FloatProperty(
        name="Spacing", description="Distance between dabs as a percentage of the brush diameter",
        default=12.0, min=1.0, max=100.0, subtype="PERCENTAGE",
    )
    healing_method: EnumProperty(
        name="Method",
        items=[
            ("AUTO", "Auto", "OpenCV seamlessClone when cv2 is installed, otherwise Boundary Blend"),
            ("POISSON", "Boundary Blend", "Source detail with the colours of the surrounding area (numpy)"),
            ("FREQUENCY", "Frequency", "Source detail + destination low frequencies (numpy)"),
            ("MEAN", "Mean Shift", "Source shifted by the mean colour difference (numpy)"),
        ],
        default="AUTO",
    )
    sample_mode: EnumProperty(
        name="Sample",
        items=[
            ("CURRENT", "Current Layer", "Heal only the active paint image"),
            ("ALL", "All Layers", "Heal the active paint image and every enabled layer at once, "
             "each from its own pixels"),
        ],
        default="ALL",
    )
    albedo_match: BoolProperty(
        name="Albedo Match",
        description="Adapt source colour/luminance to the destination (off = plain feathered clone)",
        default=True,
    )
    clone_mirror_x: BoolProperty(
        name="Mirror Horizontal",
        description="Project the clone/heal source flipped left-right around the source point "
        "(moving the brush right samples the source towards the left)",
        default=False,
    )
    clone_mirror_y: BoolProperty(
        name="Mirror Vertical",
        description="Project the clone/heal source flipped top-bottom around the source point "
        "(moving the brush up samples the source downwards)",
        default=False,
    )

    # --- fill ---------------------------------------------------------------
    fill_pattern: EnumProperty(
        name="Pattern",
        items=[
            ("SOLID", "Solid", ""),
            ("CHECKER", "Checker", ""),
            ("GRADIENT_H", "Gradient Horizontal", ""),
            ("GRADIENT_V", "Gradient Vertical", ""),
        ],
        default="SOLID",
    )
    fill_use_brush_color: BoolProperty(name="Use Brush Color", default=True)
    fill_color: FloatVectorProperty(
        name="Color", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=(1.0, 1.0, 1.0, 1.0)
    )
    fill_color2: FloatVectorProperty(
        name="Second Color", subtype="COLOR_GAMMA", size=4, min=0.0, max=1.0, default=(0.0, 0.0, 0.0, 1.0)
    )
    fill_opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    fill_checker_size: IntProperty(name="Checker Size", default=64, min=1, max=4096)
    fill_include_primary: BoolProperty(
        name="Include Paint Image",
        description="Also fill the image that is currently being painted if it is not in the layer list",
        default=True,
    )


classes = (MLTP_LayerSlot, MLTP_Settings)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mltp_layers = CollectionProperty(type=MLTP_LayerSlot)
    bpy.types.Scene.mltp_active_index = IntProperty(default=0)
    bpy.types.Scene.mltp_settings = PointerProperty(type=MLTP_Settings)


def unregister() -> None:
    del bpy.types.Scene.mltp_settings
    del bpy.types.Scene.mltp_active_index
    del bpy.types.Scene.mltp_layers
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

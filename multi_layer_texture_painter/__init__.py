bl_info = {
    "name": "MultiLayer Texture Painter Pro",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "Image Editor / 3D Viewport > Sidebar > Tool > MultiLayer Paint",
    "description": "Paint, clone, fill and heal on several Image Texture nodes at the same time",
    "category": "Paint",
}

if "bpy" in locals():  # support "Reload Scripts"
    import importlib

    for _m in (utils, paint_engine, properties, operators, ui):
        importlib.reload(_m)
else:
    import bpy  # noqa: F401
    from . import utils, paint_engine, properties, operators, ui  # noqa: F401


def register() -> None:
    properties.register()
    operators.register()
    ui.register()


def unregister() -> None:
    ui.unregister()
    operators.unregister()
    properties.unregister()

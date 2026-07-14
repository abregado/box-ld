bl_info = {
    "name": "Level Builder",
    "author": "Ben Buckton",
    "version": (1, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Level Builder",
    "description": "Build indoor level geometry non-destructively from Subtractive/Additive booleans",
    "category": "Object",
}

import bpy
import bmesh
from bpy.props import (
    StringProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    CollectionProperty,
    IntProperty,
)

SUBTRACTIVE_NAME = "Subtractive"
ADDITIVE_NAME = "Additive"
OUTPUT_COLLECTION_NAME = "Level Output"


class LevelBuildError(Exception):
    pass


def collect_mesh_objects(collection):
    result = []
    for obj in collection.objects:
        if obj.type == 'MESH':
            result.append(obj)
    for child in collection.children:
        result.extend(collect_mesh_objects(child))
    return result


def build_master_material_list(objs):
    materials = []
    for obj in objs:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is not None and mat not in materials:
                materials.append(mat)
    return materials


def remap_object_materials(obj, master_list):
    old_effective = [slot.material for slot in obj.material_slots]

    mesh = obj.data
    old_face_indices = [poly.material_index for poly in mesh.polygons]

    mesh.materials.clear()
    for mat in master_list:
        mesh.materials.append(mat)

    remap = {}
    for old_index, mat in enumerate(old_effective):
        remap[old_index] = master_list.index(mat) if mat is not None and mat in master_list else 0

    for poly, old_index in zip(mesh.polygons, old_face_indices):
        poly.material_index = remap.get(old_index, 0)


def set_object_origin_to_world_point(obj, world_point):
    """Move obj's origin to world_point without moving its geometry.

    Assumes obj currently has an identity transform (location (0,0,0),
    no rotation, scale 1) so its mesh-local coordinates equal world
    coordinates - the caller is responsible for that precondition.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    for v in bm.verts:
        v.co -= world_point
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    obj.location = world_point


def get_output_collection(context):
    coll = bpy.data.collections.get(OUTPUT_COLLECTION_NAME)
    if coll is None:
        coll = bpy.data.collections.new(OUTPUT_COLLECTION_NAME)
        context.scene.collection.children.link(coll)
    elif coll.name not in context.scene.collection.children:
        try:
            context.scene.collection.children.link(coll)
        except RuntimeError:
            pass
    return coll


def build_group(context, group):
    root = group.root_collection
    if root is None:
        raise LevelBuildError("No builder collection set")

    sub_coll = root.children.get(SUBTRACTIVE_NAME)
    add_coll = root.children.get(ADDITIVE_NAME)

    if sub_coll is None:
        raise LevelBuildError(f"'{root.name}' is missing a '{SUBTRACTIVE_NAME}' sub-collection")

    subtractive_objs = collect_mesh_objects(sub_coll)
    if not subtractive_objs:
        raise LevelBuildError(f"'{root.name}/{SUBTRACTIVE_NAME}' has no mesh objects")

    additive_objs = collect_mesh_objects(add_coll) if add_coll else []
    master_materials = build_master_material_list(subtractive_objs + additive_objs)

    scene = context.scene
    cleanup_objs = []

    def recalc_outside(mesh):
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    def normalized_duplicate(obj):
        dup_obj = obj.copy()
        dup_obj.data = obj.data.copy()
        dup_obj.name = f"LEVELBUILDER_TEMP_{obj.name}"
        scene.collection.objects.link(dup_obj)
        cleanup_objs.append(dup_obj)
        recalc_outside(dup_obj.data)
        remap_object_materials(dup_obj, master_materials)
        return dup_obj

    try:
        sub_duplicates = [normalized_duplicate(obj) for obj in subtractive_objs]
        add_duplicates = [normalized_duplicate(obj) for obj in additive_objs]

        base_dup, *remaining_sub = sub_duplicates
        mesh_name = group.output_name.strip() or f"{root.name}_LevelMesh"

        temp_obj = bpy.data.objects.new("LEVELBUILDER_TEMP", base_dup.data)
        temp_obj.location = (0.0, 0.0, 0.0)
        temp_obj.rotation_euler = (0.0, 0.0, 0.0)
        temp_obj.scale = (1.0, 1.0, 1.0)
        scene.collection.objects.link(temp_obj)
        cleanup_objs.append(temp_obj)

        for dup in remaining_sub:
            mod = temp_obj.modifiers.new(f"Union_{dup.name}", 'BOOLEAN')
            mod.operation = 'UNION'
            mod.operand_type = 'OBJECT'
            mod.object = dup
            mod.solver = group.solver

        for dup in add_duplicates:
            mod = temp_obj.modifiers.new(f"Diff_{dup.name}", 'BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.operand_type = 'OBJECT'
            mod.object = dup
            mod.solver = group.solver

        context.view_layer.update()
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = temp_obj.evaluated_get(depsgraph)
        built_mesh = bpy.data.meshes.new_from_object(eval_obj)
        built_mesh.name = mesh_name

        old_built_materials = list(built_mesh.materials)
        old_built_indices = [poly.material_index for poly in built_mesh.polygons]

        built_mesh.materials.clear()
        for mat in master_materials:
            built_mesh.materials.append(mat)

        built_remap = {}
        for old_index, mat in enumerate(old_built_materials):
            built_remap[old_index] = master_materials.index(mat) if mat is not None and mat in master_materials else 0

        for poly, old_index in zip(built_mesh.polygons, old_built_indices):
            poly.material_index = built_remap.get(old_index, 0)

        bm = bmesh.new()
        bm.from_mesh(built_mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=group.weld_distance)
        if group.recalc_normals:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bmesh.ops.reverse_faces(bm, faces=bm.faces, flip_multires=True)
        bm.to_mesh(built_mesh)
        bm.free()
        built_mesh.update()

        if group.trim_unused_materials and built_mesh.materials:
            old_trim_indices = [poly.material_index for poly in built_mesh.polygons]
            used_indices = sorted(set(old_trim_indices))
            if len(used_indices) < len(built_mesh.materials):
                trimmed = [built_mesh.materials[i] for i in used_indices]
                trim_remap = {old: new for new, old in enumerate(used_indices)}
                built_mesh.materials.clear()
                for mat in trimmed:
                    built_mesh.materials.append(mat)
                for poly, old_index in zip(built_mesh.polygons, old_trim_indices):
                    poly.material_index = trim_remap[old_index]

        output_obj = group.output_object
        if output_obj is None or output_obj.name not in bpy.data.objects:
            out_coll = get_output_collection(context)
            output_obj = bpy.data.objects.new(mesh_name, built_mesh)
            out_coll.objects.link(output_obj)
            group.output_object = output_obj
        else:
            old_mesh = output_obj.data
            output_obj.data = built_mesh
            output_obj.name = mesh_name
            if old_mesh is not None and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)

        output_obj.location = (0.0, 0.0, 0.0)
        output_obj.rotation_euler = (0.0, 0.0, 0.0)
        output_obj.scale = (1.0, 1.0, 1.0)

        if group.origin_object is not None:
            set_object_origin_to_world_point(output_obj, group.origin_object.matrix_world.translation.copy())

    finally:
        for obj in cleanup_objs:
            mesh_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data is not None and mesh_data.users == 0:
                bpy.data.meshes.remove(mesh_data)

    return len(built_mesh.polygons)


class LevelBuilderGroup(bpy.types.PropertyGroup):
    root_collection: PointerProperty(name="Builder Collection", type=bpy.types.Collection)
    output_object: PointerProperty(name="Output Object", type=bpy.types.Object)
    output_name: StringProperty(
        name="Output Name",
        description="Name for the built level mesh object/data; defaults to '<Collection>_LevelMesh' if left blank",
        default="",
    )
    origin_object: PointerProperty(
        name="Origin Object",
        type=bpy.types.Object,
        description="After building, the output mesh's origin is moved to this object's world position, without moving the mesh itself",
    )
    solver: EnumProperty(
        name="Solver",
        items=[
            ('EXACT', "Exact", "Slower, more robust boolean solver"),
            ('FAST', "Fast", "Faster boolean solver, less robust on complex overlaps"),
        ],
        default='EXACT',
    )
    weld_distance: FloatProperty(
        name="Weld Distance",
        description="Merge-by-distance threshold applied after building",
        default=0.0001,
        min=0.0,
        precision=5,
        unit='LENGTH',
    )
    recalc_normals: BoolProperty(
        name="Recalculate Normals",
        description="Normalize then flip all normals so the result faces inward",
        default=True,
    )
    trim_unused_materials: BoolProperty(
        name="Trim Unused Materials",
        description="Remove material slots not used by any face after building",
        default=True,
    )


class LEVELBUILDER_OT_add_group(bpy.types.Operator):
    bl_idname = "levelbuilder.add_group"
    bl_label = "Add Level Builder Group"
    bl_options = {'REGISTER', 'UNDO'}

    group_name: StringProperty(name="Name", default="Level")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop(self, "group_name")

    def execute(self, context):
        scene = context.scene

        root = bpy.data.collections.new(self.group_name)
        scene.collection.children.link(root)
        sub = bpy.data.collections.new(SUBTRACTIVE_NAME)
        root.children.link(sub)
        add = bpy.data.collections.new(ADDITIVE_NAME)
        root.children.link(add)

        group = scene.level_builder_groups.add()
        group.root_collection = root

        scene.level_builder_group_index = len(scene.level_builder_groups) - 1
        return {'FINISHED'}


class LEVELBUILDER_OT_remove_group(bpy.types.Operator):
    bl_idname = "levelbuilder.remove_group"
    bl_label = "Remove Level Builder Group"
    bl_description = "Unregister the active builder group (its collections and output mesh are left in the scene)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        idx = scene.level_builder_group_index
        if not (0 <= idx < len(scene.level_builder_groups)):
            self.report({'ERROR'}, "No active builder group")
            return {'CANCELLED'}
        scene.level_builder_groups.remove(idx)
        scene.level_builder_group_index = min(idx, len(scene.level_builder_groups) - 1)
        return {'FINISHED'}


class LEVELBUILDER_OT_move_selected(bpy.types.Operator):
    bl_idname = "levelbuilder.move_selected"
    bl_label = "Move Selected"
    bl_description = "Move the selected mesh objects into the active group's Subtractive/Additive collection"
    bl_options = {'REGISTER', 'UNDO'}

    target: EnumProperty(items=[('SUBTRACTIVE', "Subtractive", ""), ('ADDITIVE', "Additive", "")])

    def execute(self, context):
        scene = context.scene
        idx = scene.level_builder_group_index
        if not (0 <= idx < len(scene.level_builder_groups)):
            self.report({'ERROR'}, "No active builder group")
            return {'CANCELLED'}

        group = scene.level_builder_groups[idx]
        root = group.root_collection
        if root is None:
            self.report({'ERROR'}, "Builder group has no collection")
            return {'CANCELLED'}

        target_coll = root.children.get(SUBTRACTIVE_NAME if self.target == 'SUBTRACTIVE' else ADDITIVE_NAME)
        if target_coll is None:
            self.report({'ERROR'}, "Builder group is missing its Subtractive/Additive sub-collection")
            return {'CANCELLED'}

        moved = 0
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            for coll in list(obj.users_collection):
                coll.objects.unlink(obj)
            target_coll.objects.link(obj)
            moved += 1

        self.report({'INFO'}, f"Moved {moved} object(s) to {target_coll.name}")
        return {'FINISHED'}


class LEVELBUILDER_OT_build_active(bpy.types.Operator):
    bl_idname = "levelbuilder.build_active"
    bl_label = "Build Level"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        idx = scene.level_builder_group_index
        if not (0 <= idx < len(scene.level_builder_groups)):
            self.report({'ERROR'}, "No active builder group")
            return {'CANCELLED'}

        group = scene.level_builder_groups[idx]
        try:
            face_count = build_group(context, group)
        except LevelBuildError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        self.report({'INFO'}, f"Built '{group.root_collection.name}' ({face_count} faces)")
        return {'FINISHED'}


class LEVELBUILDER_OT_build_all(bpy.types.Operator):
    bl_idname = "levelbuilder.build_all"
    bl_label = "Build All Levels"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        errors = []
        built = 0
        for group in scene.level_builder_groups:
            name = group.root_collection.name if group.root_collection else "<unnamed>"
            try:
                build_group(context, group)
                built += 1
            except LevelBuildError as e:
                errors.append(f"{name}: {e}")

        if errors:
            self.report({'ERROR'}, "; ".join(errors))
        if built:
            self.report({'INFO'}, f"Built {built} group(s)")

        return {'CANCELLED'} if not built and errors else {'FINISHED'}


class LEVELBUILDER_UL_groups(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if item.root_collection:
            layout.prop(item.root_collection, "name", text="", emboss=False, icon='SCENE_DATA')
        else:
            layout.label(text="<missing collection>", icon='ERROR')


class LEVELBUILDER_PT_panel(bpy.types.Panel):
    bl_label = "Level Builder"
    bl_idname = "LEVELBUILDER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Level Builder"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        row = layout.row()
        row.template_list(
            "LEVELBUILDER_UL_groups", "",
            scene, "level_builder_groups",
            scene, "level_builder_group_index",
        )
        col = row.column(align=True)
        col.operator("levelbuilder.add_group", icon='ADD', text="")
        col.operator("levelbuilder.remove_group", icon='REMOVE', text="")

        idx = scene.level_builder_group_index
        if 0 <= idx < len(scene.level_builder_groups):
            group = scene.level_builder_groups[idx]

            box = layout.box()
            row = box.row(align=True)
            row.operator("levelbuilder.move_selected", text="→ Subtractive").target = 'SUBTRACTIVE'
            row.operator("levelbuilder.move_selected", text="→ Additive").target = 'ADDITIVE'

            box.prop(group, "output_name", placeholder=f"{group.root_collection.name}_LevelMesh" if group.root_collection else "")
            box.prop(group, "origin_object")
            box.prop(group, "solver")
            box.prop(group, "weld_distance")
            box.prop(group, "recalc_normals")
            box.prop(group, "trim_unused_materials")

            if group.output_object:
                box.label(text=f"Output: {group.output_object.name}", icon='MESH_DATA')

            box.operator("levelbuilder.build_active", icon='MOD_BOOLEAN')

        layout.separator()
        layout.operator("levelbuilder.build_all", icon='MOD_BOOLEAN')


classes = (
    LevelBuilderGroup,
    LEVELBUILDER_OT_add_group,
    LEVELBUILDER_OT_remove_group,
    LEVELBUILDER_OT_move_selected,
    LEVELBUILDER_OT_build_active,
    LEVELBUILDER_OT_build_all,
    LEVELBUILDER_UL_groups,
    LEVELBUILDER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.level_builder_groups = CollectionProperty(type=LevelBuilderGroup)
    bpy.types.Scene.level_builder_group_index = IntProperty(default=0)


def unregister():
    del bpy.types.Scene.level_builder_group_index
    del bpy.types.Scene.level_builder_groups

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

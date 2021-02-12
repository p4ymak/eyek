import bpy
import json
import mathutils
from math import radians, atan, tan
from bpy_extras.io_utils import axis_conversion
import os
import platform
import subprocess
import time

_EXECUTABLE_NAME = "eyek"

if platform.system() == "Windows":
    _EXECUTABLE_NAME += ".exe"
if platform.system() == "Darwin":
    _EXECUTABLE_NAME += "_mac"

bl_info = {
    "name": "Eyek",
    "description": "Texturing by projection mapping from multiple cameras to one UV layer.",
    "author": "Roman Chumak",
    "doc_url": "https://phygitalism.com/en/eyek/",
    "version": (0, 0, 2, 0),
    "blender": (2, 90, 1),
    "location": "View3D",
    "category": "Texturing"}


class EYEK_Properties(bpy.types.PropertyGroup):
    res_x: bpy.props.IntProperty(default=512, min=2)
    res_y: bpy.props.IntProperty(default=512, min=2)
    clip_uv: bpy.props.BoolProperty(default=False, description="Clip UV.")
    path_export_image: bpy.props.StringProperty(
        default="//texture.png", subtype="FILE_PATH", description="File to write Texture")
    blending: bpy.props.EnumProperty(items=[
                                    ('0', 'Average', '', 0),
                                    ('1', 'Median', '', 1), 
                                    ('2', 'Mode', '', 2)
                                    ], description="Method for blending colors between different projections.")
    shadowing: bpy.props.BoolProperty(default=True, description="Allow polygons shade each other. Otherwise, the projection goes through.")
    expansions: bpy.props.IntProperty(default=1, min =0, max=255, description="Color empty pixels around UV islands.")


class EYEK_exe(bpy.types.Operator):
    """Project Images from Selected Cameras to Selected Objects UVs"""
    bl_idname = 'eyek.exe'
    bl_label = 'Paint!'
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        start_time = time.time()
        meshes = []
        cameras = []

        scene_dir = directory = os.path.dirname(bpy.data.filepath)
        eyek_root_dir = os.path.join(scene_dir, "eyek_cache")
        eyek_dir = os.path.join(eyek_root_dir, str(time.time_ns()))
        selected = bpy.context.selected_objects
        global_matrix = axis_conversion(from_forward='Y', from_up='Z', to_forward='-Z', to_up='Y').to_4x4()

        for ob in selected:
            if ob.type == 'CAMERA':
                if len(ob.data.background_images) > 0:
                    cameras.append(ob)
            if ob.type == 'MESH':
                if len(ob.data.polygons) > 0 and len(ob.data.uv_layers) > 0:
                    meshes.append(ob)
        print("Cameras:", len(cameras), "\nObjects:", len(meshes))

        if len(cameras) > 0 and len(meshes) > 0:
            try:
                os.mkdir(eyek_root_dir)
            except:
                pass
            os.mkdir(eyek_dir)
            
            cameras_data = []
            for cam in cameras:
                cam_matrix = global_matrix @ cam.matrix_world
                l_x, l_y, l_z = cam_matrix.to_translation()
                re_x, re_y, re_z = cam_matrix.to_euler()
                cam_image = bpy.data.images[cam.data.background_images[0].image.name]
                image_path = bpy.path.abspath(cam_image.filepath_raw)
                fov = cam.data.angle
                cam_near = cam.data.clip_start
                cam_far = cam.data.clip_end
                cam_data = {
                            "location": {"x": l_x, "y": l_y, "z": l_z}, 
                            "rotation_euler": {"x": re_x, "y": re_y, "z": re_z},
                            "fov_x": fov, 
                            "limit_near": cam_near, 
                            "limit_far": cam_far, 
                            "image_path": image_path,
                            }
                cameras_data.append(cam_data)

                render = bpy.context.scene.render
                render_ratio = render.resolution_x / render.resolution_y
                img_ratio = cam_image.size[0] / cam_image.size[1]
                if render_ratio >= img_ratio:
                	cam.data.background_images[0].frame_method = 'CROP'
                else:
                	cam.data.background_images[0].frame_method = 'FIT'
                

            json_file_path = os.path.join(eyek_dir, "cameras.json")
            with open(json_file_path, 'w') as outfile:
                json.dump({"data": cameras_data}, outfile)
            
            bpy.ops.object.select_all(action='DESELECT')
            for mesh in meshes:
                mesh.select_set(True)
            obj_path = os.path.join(eyek_dir, "mesh.obj")
            bpy.ops.export_scene.obj(filepath=obj_path,
                                     use_selection=True,
                                     use_animation=False, 
                                     use_mesh_modifiers=True, 
                                     use_edges=True,
                                     use_smooth_groups=False,
                                     use_smooth_groups_bitflags=False,
                                     use_normals=True,
                                     use_uvs=True,
                                     use_materials=False,
                                     use_triangles=True,
                                     use_nurbs=False,
                                     use_vertex_groups=False,
                                     use_blen_objects=False,
                                     group_by_object=False,
                                     group_by_material=False,
                                     keep_vertex_order=False,
                                     global_scale=1.0,
                                     path_mode='AUTO',
                                     axis_forward='-Z',
                                     axis_up='Y')
            
            print("OBJ and JSON exported.")
            
            for ob in selected:
                ob.select_set(True)

            addon_dir = os.path.dirname(os.path.realpath(__file__))
            texture_path = bpy.path.abspath(bpy.context.scene.eyek.path_export_image)
            if not texture_path.lower().endswith(".png"):
                texture_path += ".png"
            clip_uv = str(int(bpy.context.scene.eyek.clip_uv))
            res_x = str(bpy.context.scene.eyek.res_x)
            res_y = str(bpy.context.scene.eyek.res_y)
            blending = str(bpy.context.scene.eyek.blending)
            shadowing = str(int(bpy.context.scene.eyek.shadowing))
            expansions = str(bpy.context.scene.eyek.expansions)

            args = [
                    os.path.join(addon_dir, _EXECUTABLE_NAME),
                    eyek_dir,
                    texture_path,
                    res_x,
                    res_y,
                    clip_uv,
                    blending,
                    shadowing,
                    expansions
                    ]
            popen = subprocess.Popen(args)
            popen.wait()
            print("Time elapsed:", time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
        return {'FINISHED'}



class EYEK_PT_Panel(bpy.types.Panel):
    bl_label = "Eyek"
    bl_idname = "EYEK_PT_Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Eyek"
    #bl_context = "object"


    def draw(self, context):
        layout = self.layout
        eyek_ui = layout.column(align=True)
        prefs_row = eyek_ui.row()
        left_col = prefs_row.column(align=True)
        left_col.label(text="Resolution:")
        left_col.prop(context.scene.eyek, 'res_x', text="X")
        left_col.prop(context.scene.eyek, 'res_y', text="Y")
        left_col.separator()
        left_col.label(text="Blending:")
        left_col.prop(context.scene.eyek, 'blending', text="")

        right_col = prefs_row.column(align=True)
        right_col.label(text="Properties:")
        right_col.prop(context.scene.eyek, 'clip_uv', text="Clip UV")
        right_col.prop(context.scene.eyek, 'shadowing', text="Shadowing")
        right_col.separator()
        right_col.prop(context.scene.eyek, 'expansions', text="Expand")

        eyek_ui.separator()
        eyek_ui.label(text="Output:")
        eyek_ui.prop(context.scene.eyek, 'path_export_image', text="")
        if bpy.context.object!=None and bpy.context.object.type == 'MESH' and bpy.context.object.mode=='OBJECT':
            if bpy.data.is_saved:
                eyek_ui.operator('eyek.exe', icon="BRUSH_DATA")
            else:
                eyek_ui.label(text="Save your Scene first.")
        else:
            eyek_ui.label(text="Return to Object Mode.")

def register():
    bpy.utils.register_class(EYEK_Properties)
    bpy.types.Scene.eyek = bpy.props.PointerProperty(type=EYEK_Properties)
    bpy.utils.register_class(EYEK_exe)
    bpy.utils.register_class(EYEK_PT_Panel)
    if platform.system() != "Windows":
        os.chmod(os.path.join(os.path.dirname(
            os.path.realpath(__file__)), _EXECUTABLE_NAME), 7770)    

def unregister():
    bpy.utils.unregister_class(EYEK_Properties)
    bpy.utils.unregister_class(EYEK_exe)
    bpy.utils.unregister_class(EYEK_PT_Panel)

if __name__ == "__main__":
    register()
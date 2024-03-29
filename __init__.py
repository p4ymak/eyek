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
    "description": "Texturing by projection mapping from multiple Cameras and Image Empties to one UV layer.",
    "author": "Roman Chumak p4ymak@yandex.ru",
    "doc_url": "https://github.com/p4ymak/eyek",
    "version": (0, 0, 2, 8),
    "blender": (3, 6, 0),
    "location": "View3D",
    "category": "Texturing"}


class EYEK_Properties(bpy.types.PropertyGroup):
    res_x: bpy.props.IntProperty(default=512, min=2, subtype='PIXEL', description="Number of horizontal pixels in the generated texture.")
    res_y: bpy.props.IntProperty(default=512, min=2, subtype='PIXEL',description="Number of vertical pixels in the generated texture.")
    res_sc: bpy.props.IntProperty(default=100, min=1, soft_max=100, subtype='PERCENTAGE', description="Percentage scale for generated texture resolution.")
    ortho_near: bpy.props.FloatProperty(default=0.01, min=0.000001, subtype='DISTANCE', description="Image Empties near clipping distance.")
    ortho_far: bpy.props.FloatProperty(default=100.0, min=0.000001, subtype='DISTANCE' ,description="Image Empties far clipping distance.")

    path_export_image: bpy.props.StringProperty(
        default="//texture.png", subtype="FILE_PATH", description="File to write Texture")
    blending: bpy.props.EnumProperty(items=[
                                    ('0', 'Average', '', 0),
                                    ('1', 'Median', '', 1), 
                                    ('2', 'Mode', '', 2),
                                    ('3', 'Overlay', '', 3)
                                    ], description="Method for blending colors between different projections.")
    backface_culling: bpy.props.BoolProperty(default=True, description="Ignore faces pointing away from view. They are used in occlusion yet.")
    occlude: bpy.props.BoolProperty(default=True, description="Allow polygons shade each other. Otherwise, the projection goes through.")
    bleed: bpy.props.IntProperty(default=0, min =0, max=255, subtype='PIXEL', description="Seam Bleed extends the paint beyond UV island bounds to avoid visual artifacts (like bleed for baking).")
    upscale: bpy.props.IntProperty(default=0, min =0, max=4, description="Upscale input images to avoid aliasing.")
    autoreload: bpy.props.BoolProperty(default=True, description="Auto reload generated texture image.")


class EYEK_exe(bpy.types.Operator):
    """Project Images from Selected Cameras to Selected Objects UVs"""
    bl_idname = 'eyek.exe'
    bl_label = 'Paint!'
    bl_options = {'INTERNAL', 'UNDO'}

    def execute(self, context):
        start_time = time.time()
        meshes = []
        cameras = []

        scene_dir = os.path.dirname(bpy.data.filepath)
        eyek_root_dir = os.path.join(scene_dir, "eyek_cache")
        eyek_dir = os.path.join(eyek_root_dir, str(time.time_ns()))
        selected = bpy.context.selected_objects
        global_matrix = axis_conversion(from_forward='Y', from_up='Z', to_forward='-Z', to_up='Y').to_4x4()

        for ob in selected:
            if ob.type == 'CAMERA':
                if len(ob.data.background_images) > 0:
                    cameras.append(ob)
            if ob.type == 'EMPTY':
                if ob.empty_display_type == 'IMAGE':
                    if ob.data != None:
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
            cameras.sort(key=lambda x: x.name)

            render = bpy.context.scene.render
            render_ratio = render.resolution_x / render.resolution_y
            for cam in cameras:
                cam_matrix = global_matrix @ cam.matrix_world
                l_x, l_y, l_z = cam_matrix.to_translation()
                re_x, re_y, re_z = cam_matrix.to_euler()
                sc_x, sc_y, sc_z = cam_matrix.to_scale()
                if cam.type == 'CAMERA':
                    cam_image = bpy.data.images[cam.data.background_images[0].image.name]
                    img_ratio = cam_image.size[0] / cam_image.size[1]

                    fov = cam.data.angle
                    cam.data.background_images[0].frame_method = 'CROP'
                    if render_ratio >= 1.0:
                        if cam.data.type == 'ORTHO':
                            fov = -cam.data.ortho_scale
                            sc_x *= -fov
                            sc_y *= -fov / img_ratio

                    else:
                        if cam.data.type == 'ORTHO':
                            fov = -cam.data.ortho_scale
                            sc_x *= -fov * img_ratio
                            sc_y *= -fov
                        else:
                            fov = 2 * atan(tan(fov / 2) * img_ratio)


                    sc_z *= -fov
                    cam_near = cam.data.clip_start
                    cam_far = cam.data.clip_end
                
                if cam.type == 'EMPTY':
                    cam_image = bpy.data.images[cam.data.name]
                    img_ratio = cam_image.size[0] / cam_image.size[1]
                    fov = -cam.empty_display_size
                    if img_ratio > 1.0:
                        sc_x *= -fov
                        sc_y *= -fov / img_ratio
                    else:
                        sc_x *= -fov * img_ratio
                        sc_y *= -fov
                    sc_z *= -fov
                        
                    cam_near = bpy.context.scene.eyek.ortho_near
                    cam_far = bpy.context.scene.eyek.ortho_far

                image_path = bpy.path.abspath(cam_image.filepath_raw)

                cam_data = {
                            "location": {"x": l_x, "y": l_y, "z": l_z}, 
                            "rotation_euler": {"x": re_x, "y": re_y, "z": re_z},
                            "scale": {"x": sc_x, "y": sc_y, "z": sc_z},
                            "fov_x": fov, 
                            "limit_near": cam_near, 
                            "limit_far": cam_far, 
                            "image_path": image_path,
                            }
                cameras_data.append(cam_data)

            json_file_path = os.path.join(eyek_dir, "cameras.json")
            with open(json_file_path, 'w') as outfile:
                json.dump({"data": cameras_data}, outfile)
            
            bpy.ops.object.select_all(action='DESELECT')
            for mesh in meshes:
                mesh.select_set(True)
            obj_path = os.path.join(eyek_dir, "mesh.obj")
            bpy.ops.wm.obj_export(filepath=obj_path,
                                    export_selected_objects=True,
                                    export_animation=False, 
                                    apply_modifiers=True, 
                                    export_smooth_groups=False,
                                    smooth_group_bitflags=False,
                                    export_normals=True,
                                    export_uv=True,
                                    export_materials=False,
                                    export_triangulated_mesh=True,
                                    export_curves_as_nurbs=False,
                                    export_vertex_groups=False,
                                    export_object_groups=False,
                                    export_material_groups=False,
                                    global_scale=1.0,
                                    path_mode='AUTO',
                                    forward_axis='NEGATIVE_Z',
                                    up_axis='Y')
            
            print("OBJ and JSON exported.")
            
            for ob in selected:
                ob.select_set(True)

            addon_dir = os.path.dirname(os.path.realpath(__file__))
            texture_path = bpy.path.abspath(bpy.context.scene.eyek.path_export_image)
            if texture_path.lower().endswith(".png"):
                texture_path = texture_path[:-4]
            if texture_path.lower()[-5:-5] == ".1":
                texture_path = texture_path[:-5]
            
            res_sc = bpy.context.scene.eyek.res_sc / 100.0
            res_x = str(int(bpy.context.scene.eyek.res_x * res_sc))
            res_y = str(int(bpy.context.scene.eyek.res_y * res_sc))

            blending = str(bpy.context.scene.eyek.blending)
            backface_culling = str(int(bpy.context.scene.eyek.backface_culling))
            occlude = str(int(bpy.context.scene.eyek.occlude))
            bleed = str(bpy.context.scene.eyek.bleed)
            upscale = str(bpy.context.scene.eyek.upscale)
            
            args = [
                    os.path.join(addon_dir, _EXECUTABLE_NAME),
                    eyek_dir,
                    texture_path,
                    res_x,
                    res_y,
                    str(False),
                    blending,
                    backface_culling,
                    occlude,
                    bleed,
                    upscale
                    ]
            popen = subprocess.Popen(args)
            popen.wait()
            print("Time elapsed:", time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))

            if bpy.context.scene.eyek.autoreload:
                for img in bpy.data.images:
                    if bpy.path.abspath(img.filepath) == texture_path+".1001.png" or bpy.path.abspath(img.filepath) == texture_path+".png":
                        img.reload()


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
        left_col.prop(context.scene.eyek, 'res_sc', text="%")
        left_col.separator()
        left_col.label(text="Empties Clip:")
        left_col.prop(context.scene.eyek, 'ortho_near', text="Near")
        left_col.prop(context.scene.eyek, 'ortho_far', text="Far")
        left_col.separator()
        left_col.label(text="Blending:")
        left_col.prop(context.scene.eyek, 'blending', text="")

        right_col = prefs_row.column(align=True)
        right_col.label(text="Properties:")
        right_col.prop(context.scene.eyek, 'backface_culling', text="Backface Culling")
        right_col.prop(context.scene.eyek, 'occlude', text="Occlude")
        right_col.separator()
        right_col.prop(context.scene.eyek, 'bleed', text="Bleed")
        right_col.separator()
        right_col.prop(context.scene.eyek, 'autoreload', text="Auto Reload")
        
        eyek_ui.separator()
        eyek_ui.label(text="Output:")
        eyek_ui.prop(context.scene.eyek, 'path_export_image', text="")

        eyek_exec = eyek_ui.row()
        eyek_exec.scale_y = 2.0
        if bpy.context.object!=None and bpy.context.object.type == 'MESH' and bpy.context.object.mode=='OBJECT':
            if bpy.data.is_saved:
                eyek_exec.operator('eyek.exe', icon="BRUSH_DATA")
            else:
                eyek_exec.label(text="Save your Scene first.")
        else:
            eyek_exec.label(text="Return to Object Mode.")

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

import bpy


class PHOTARA_PT_main(bpy.types.Panel):
    bl_label = "Photara"
    bl_idname = "PHOTARA_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photara"

    def draw_header(self, context):
        self.layout.label(icon="CAMERA_DATA")

    def draw(self, context):
        props = context.scene.photara
        layout = self.layout

        col = layout.column()
        col.enabled = not props.flight_recording
        col.scale_y = 1.6
        if props.is_rendering:
            col.operator("photara.cancel_render", text="Cancel Render", icon="CANCEL")
        else:
            col.operator("photara.render_and_export", icon="RENDER_ANIMATION")

        row = layout.row(align=True)
        row.enabled = not props.is_rendering and not props.flight_recording
        row.operator("photara.export_poses", icon="EXPORT")
        row.operator("photara.open_dataset_folder", icon="FILEBROWSER", text="")

        box = layout.box()
        col = box.column(align=True)
        col.label(text=props.status, icon="INFO")
        row = col.row()
        row.enabled = False
        row.prop(props, "progress", text="", slider=True)


class PHOTARA_PT_output(bpy.types.Panel):
    bl_label = "Output"
    bl_idname = "PHOTARA_PT_output"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photara"
    bl_parent_id = "PHOTARA_PT_main"

    def draw(self, context):
        props = context.scene.photara
        layout = self.layout
        layout.use_property_split = True
        layout.enabled = not props.is_rendering
        layout.use_property_decorate = False

        layout.prop(props, "output_root")
        layout.prop(props, "dataset_name")
        layout.prop(props, "image_format")
        if props.image_format in ("JPEG", "WEBP"):
            layout.prop(props, "jpeg_quality")
        layout.use_property_split = False
        layout.prop(props, "force_opaque")
        layout.prop(props, "overwrite")
        layout.prop(props, "export_colmap")
        layout.prop(props, "export_transforms_json")


class PHOTARA_PT_frames(bpy.types.Panel):
    bl_label = "Camera & Frames"
    bl_idname = "PHOTARA_PT_frames"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photara"
    bl_parent_id = "PHOTARA_PT_main"

    def draw(self, context):
        props = context.scene.photara
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        layout.enabled = not props.is_rendering

        box = layout.box()
        box.use_property_split = False
        box.label(text="Free Flight Recording", icon="CAMERA_DATA")
        if props.flight_recording:
            box.label(text="Recorded Points: {}".format(props.flight_count))
            box.label(text="Mouse: look / WASD: move")
            if props.flight_control != 'MANUAL':
                box.label(text="Enter: save / Esc: discard")
            else:
                box.label(text="Finish navigation, then Stop")
            row = box.row(align=True)
            row.operator("photara.stop_flight", icon="CHECKMARK")
            row.operator("photara.stop_flight", text="Cancel", icon="CANCEL").discard = True
            return
        box.prop(props, "flight_control")
        box.prop(props, "flight_distance")
        box.prop(props, "flight_angle")
        box.prop(props, "flight_interval")
        box.operator("photara.start_flight", icon="REC")
        box.label(text="Mouse: look / WASD: move")

        layout.prop(props, "camera")
        layout.prop(props, "capture_mode")
        if props.capture_mode == "ORBIT":
            layout.prop(props, "orbit_target")
            layout.prop(props, "orbit_shots")
            layout.prop(props, "orbit_radius_factor")
            layout.prop(props, "orbit_levels")
            layout.prop(props, "orbit_coverage")
            if props.orbit_coverage == "CUSTOM":
                layout.prop(props, "orbit_elevation_min")
                layout.prop(props, "orbit_elevation_max")
            layout.prop(props, "orbit_start_angle")
        elif props.capture_mode == "BOX_GRID":
            layout.prop(props, "box_bounds_object")
            layout.prop(props, "box_bounds_helper")
            layout.prop(props, "box_auto_capture")
            row = layout.row(align=True)
            row.prop(props, "box_grid_x")
            row.prop(props, "box_grid_y")
            row.prop(props, "box_grid_z")
            layout.prop(props, "box_margin")
            if props.box_grid_z == 1:
                layout.prop(props, "box_camera_height")
            if props.box_auto_capture:
                layout.prop(props, "box_capture_lens")
                layout.prop(props, "box_overlap")
            layout.prop(props, "box_views_per_point",
                        text="Min Azimuth Views" if props.box_auto_capture else "Views per Point")
            layout.prop(props, "box_start_angle")
            if not props.box_auto_capture:
                layout.prop(props, "box_pitch")
            layout.prop(props, "box_clearance")
            layout.prop(props, "box_surface_samples")
            layout.prop(props, "box_min_observations")
            layout.prop(props, "box_min_parallax")
            layout.prop(props, "box_max_images")
            layout.prop(props, "box_block_incomplete")
            box = layout.box()
            box.label(text=props.box_coverage_status, icon="INFO")
            box.label(text="Red: unseen / Orange: insufficient")
            box.label(text="Rebuild preview after scene/settings changes")
        else:
            layout.prop(props, "use_scene_frame_range")
            if not props.use_scene_frame_range:
                row = layout.row(align=True)
                row.prop(props, "frame_start")
                row.prop(props, "frame_end")
            layout.prop(props, "frame_step")

        if props.capture_mode in {"ORBIT", "BOX_GRID", "ANIMATION"}:
            layout.separator()
            row = layout.row(align=True)
            row.enabled = not props.is_rendering
            row.operator(
                "photara.preview_camera_positions",
                text="Preview Camera Positions",
                icon="CAMERA_DATA",
            )
            row.operator(
                "photara.clear_camera_preview",
                text="",
                icon="TRASH",
            )
            layout.operator("photara.focus_camera_preview", icon="VIEWZOOM")


class PHOTARA_PT_photara(bpy.types.Panel):
    bl_label = "Photara Integration"
    bl_idname = "PHOTARA_PT_photara"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Photara"
    bl_parent_id = "PHOTARA_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        props = context.scene.photara
        layout = self.layout

        layout.prop(props, "handover_mode")
        if props.handover_mode == "OFF":
            return

        layout.use_property_split = True
        layout.use_property_decorate = False

        if props.handover_mode in ("AUTO", "STUDIO"):
            layout.prop(props, "studio_timeout")
        if props.handover_mode == "AUTO":
            layout.prop(props, "launch_wait")
        layout.prop(props, "auto_start_training")

        layout.separator()
        layout.use_property_split = False
        col = layout.column(align=True)
        if props.handover_mode in ("AUTO", "STUDIO"):
            col.prop(props, "studio_exe")
            col.prop(props, "studio_args")
        if props.handover_mode == "CLI":
            col.prop(props, "cli_exe")
            col.prop(props, "cli_args")

        layout.operator("photara.test_studio_connection", icon="LINKED")

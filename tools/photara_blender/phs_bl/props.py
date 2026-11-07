import bpy

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


def _poll_camera(self, obj):
    return obj is not None and obj.type == "CAMERA" and not obj.get("photara_camera_preview")


def _poll_mesh_or_object(self, obj):
    return obj is not None and obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}


def _poll_bounds(self, obj):
    return obj is not None and obj.type in {"MESH", "EMPTY"}


class PhotaraProps(bpy.types.PropertyGroup):
    flight_control: EnumProperty(
        name="Camera Control",
        description="开始录制时进入的相机控制方式",
        items=(
            ('WALK', 'First Person (Walk)', '鼠标转向，WASD 水平移动，Q/E 升降；Enter 保存，Esc 取消'),
            ('FLY', 'Free Flight', '鼠标转向，沿视线自由移动；Enter 保存，Esc 取消'),
            ('MANUAL', 'Manual View Navigation', '只开始录制，自行操作视图导航'),
        ),
        default='WALK',
    )
    flight_recording: BoolProperty(default=False, options={'SKIP_SAVE'})
    flight_count: IntProperty(default=0, options={'SKIP_SAVE'})
    flight_distance: FloatProperty(name="Point Distance", description="相对上一点的移动距离达到此值时记录（场景单位）", default=0.25, min=0.001, subtype='DISTANCE')
    flight_angle: FloatProperty(name="Point Angle", description="相对上一点的旋转达到此角度时记录", default=0.1745329252, min=0.001, max=3.1415926536, subtype='ANGLE')
    flight_interval: FloatProperty(name="Sample Interval", description="检查视图变化的间隔（秒）；录制最多 10000 个点位", default=0.1, min=0.02, max=2.0)
    # ------------------------------------------------------------------ output
    output_root: StringProperty(
        name="Output Root",
        description="数据集输出根目录（包含 images/ 与 sparse/0/）",
        subtype="DIR_PATH",
        default="//photara_dataset",
    )
    dataset_name: StringProperty(
        name="Dataset Name",
        description="数据集名称，会写入 manifest 并提交给 Photara Studio",
        default="blender_scene",
    )
    image_format: EnumProperty(
        name="Image Format",
        description="序列帧格式。3DGS 训练推荐 PNG 或 JPEG（8bit 已 tone-map 的图像）",
        items=(
            ("PNG", "PNG", "无损 8bit，最稳妥的选择", "IMAGE_DATA", 0),
            ("JPEG", "JPEG", "体积小，可设置质量", "IMAGE_DATA", 1),
            ("WEBP", "WebP", "体积小", "IMAGE_DATA", 2),
            ("TIFF", "TIFF", "无损（体积大）", "IMAGE_DATA", 3),
            ("OPEN_EXR", "OpenEXR", "HDR 线性数据（部分 3DGS 流程支持）", "IMAGE_DATA", 4),
        ),
        default="PNG",
    )
    jpeg_quality: IntProperty(
        name="Quality",
        description="JPEG/WebP 质量",
        default=92,
        min=1,
        max=100,
    )
    force_opaque: BoolProperty(
        name="Force Opaque Background",
        description="渲染期间临时关闭透明背景。3DGS 训练通常需要不透明背景",
        default=True,
    )
    overwrite: BoolProperty(
        name="Overwrite Existing Frames",
        description="开始前清理输出目录中旧的 frame_XXXX 文件",
        default=True,
    )

    # ------------------------------------------------------------------ frames
    use_scene_frame_range: BoolProperty(
        name="Use Scene Range",
        description="使用场景的播放范围，否则使用下方自定义范围",
        default=True,
    )
    frame_start: IntProperty(name="Start", default=1, soft_min=0)
    frame_end: IntProperty(name="End", default=120, soft_min=1)
    frame_step: IntProperty(
        name="Step",
        description="帧间隔（隔帧采样可减少训练数据量）",
        default=1,
        min=1,
    )
    camera: PointerProperty(
        name="Camera",
        description="用于渲染与导出位姿的相机（留空则使用场景相机）",
        type=bpy.types.Object,
        poll=_poll_camera,
    )
    capture_mode: EnumProperty(
        name="Capture Mode",
        description="使用已有相机动画，或自动生成物体 360° 环绕点位",
        items=(
            ("ANIMATION", "Camera Animation", "按场景帧范围渲染已有相机动画", 0),
            ("ORBIT", "Object 360° Orbit", "围绕目标物体自动生成相机点位", 1),
            ("BOX_GRID", "Indoor Box Grid", "在边界盒内生成相机点阵并多方向拍摄", 2),
        ),
        default="ANIMATION",
    )
    orbit_target: PointerProperty(
        name="Target Object",
        description="环绕拍摄的目标物体；自动以其世界空间边界框中心为注视点",
        type=bpy.types.Object,
        poll=_poll_mesh_or_object,
    )
    orbit_shots: IntProperty(
        name="Total Shots",
        description="所有仰角层合计的图像张数，插件会尽量均匀分配到各层",
        default=36,
        min=4,
        max=360,
    )
    orbit_radius_factor: FloatProperty(
        name="Radius × Object Size",
        description="环绕半径相对于目标物体世界空间最大尺寸的倍数",
        default=3.0,
        min=0.25,
        max=20.0,
    )
    orbit_levels: IntProperty(
        name="Elevation Rings",
        description="球面轨道的垂直层数",
        default=3,
        min=1,
        max=12,
    )
    orbit_coverage: EnumProperty(
        name="Coverage",
        description="选择上半球、完整球面，或使用自定义仰角范围",
        items=(
            ("UPPER_HEMISPHERE", "Upper Hemisphere", "从水平面到物体上方", 0),
            ("FULL_SPHERE", "Full Sphere", "围绕物体完整球面覆盖", 1),
            ("CUSTOM", "Custom Elevation", "按自定义最低和最高仰角采样", 2),
        ),
        default="CUSTOM",
    )
    orbit_elevation_min: FloatProperty(
        name="Min Elevation",
        description="最低一层相对水平面的仰角；负值从物体下方拍摄",
        default=-0.5235987756,
        min=-1.57079632679,
        max=1.57079632679,
        subtype="ANGLE",
    )
    orbit_elevation_max: FloatProperty(
        name="Max Elevation",
        description="最高一层相对水平面的仰角；正值从物体上方拍摄",
        default=1.0471975512,
        min=-1.57079632679,
        max=1.57079632679,
        subtype="ANGLE",
    )
    orbit_start_angle: FloatProperty(
        name="Start Angle",
        description="环绕起始角度",
        default=0.0,
        min=-6.28318530718,
        max=6.28318530718,
        subtype="ANGLE",
    )
    box_bounds_object: PointerProperty(
        name="Room Bounds",
        description="代表室内可拍摄空间边界的盒体或辅助物体",
        type=bpy.types.Object,
        poll=_poll_bounds,
    )
    box_grid_x: IntProperty(name="X Points", default=3, min=1, max=20)
    box_grid_y: IntProperty(name="Y Points", default=3, min=1, max=20)
    box_grid_z: IntProperty(name="Z Layers", default=3, min=1, max=10)
    box_bounds_helper: BoolProperty(
        name="Bounds Is Helper", default=True,
        description="边界物体仅为采集范围辅助盒；采集时临时隐藏，不包含在场景几何中",
    )
    box_auto_capture: BoolProperty(
        name="Automatic Coverage", default=True,
        description="按实际视场角生成重叠的上下全方向视角，检查遮挡并自动补充视点",
    )
    box_capture_lens: FloatProperty(
        name="Capture Lens (mm)", default=24.0, min=8.0, max=100.0,
        description="自动采集使用的透视焦距；仅修改临时采集相机",
    )
    box_overlap: FloatProperty(
        name="Angular Overlap", default=0.35, min=0.1, max=0.8, subtype="FACTOR",
        description="同一位置相邻拍摄方向的角度重叠率；空间重叠还取决于点间距与遮挡",
    )
    box_clearance: FloatProperty(
        name="Geometry Clearance", default=0.2, min=0.01, max=10.0, subtype="DISTANCE",
        description="相机与场景表面的最小距离；闭合家具内部点位会被排除",
    )
    box_surface_samples: IntProperty(
        name="Coverage Samples", default=1200, min=100, max=10000,
        description="按三角形面积抽样的表面检查点数量；增加采样可检测较小盲区",
    )
    box_min_observations: IntProperty(
        name="Required Positions", default=3, min=2, max=8,
        description="每个表面样本需要被多少个不同且有视差的相机位置看见",
    )
    box_min_parallax: FloatProperty(
        name="Min Parallax", default=0.0872664626, min=0.0087266463,
        max=0.7853981634, subtype="ANGLE",
        description="多视角观察方向间的最小夹角；同一点转动相机不算新增观察位置",
    )
    box_max_images: IntProperty(
        name="Image Budget", default=4000, min=50, max=10000,
        description="基础采集和自动补拍的总帧数上限；预算不足的区域会在报告中标出",
    )
    box_block_incomplete: BoolProperty(
        name="Require Coverage Target", default=False,
        description="启用后，存在采样盲区时阻止渲染和训练；先预览检查并调整范围、点阵和预算",
    )
    box_coverage_status: StringProperty(name="Coverage", default="尚未分析")
    box_margin: FloatProperty(
        name="Wall / Ceiling Margin",
        description="点阵与边界盒各面的最小距离（Blender 世界单位）",
        default=0.3,
        min=0.0,
        max=100.0,
        subtype="DISTANCE",
    )
    box_camera_height: FloatProperty(
        name="Camera Height",
        description="单层点阵时相机离边界盒底部的高度；多层时由 Z Layers 均匀分布",
        default=1.5,
        min=0.1,
        max=100.0,
        subtype="DISTANCE",
    )
    box_views_per_point: EnumProperty(
        name="Views per Point",
        description="每个点位水平环拍的透视视角数量",
        items=(
            ("4", "4 Directions", "每 90° 一张", 0),
            ("8", "8 Directions", "每 45° 一张，覆盖更密", 1),
            ("12", "12 Directions", "每 30° 一张，覆盖最密", 2),
        ),
        default="4",
    )
    box_start_angle: FloatProperty(
        name="Start Angle",
        description="点位环拍的起始朝向",
        default=0.0,
        min=-6.28318530718,
        max=6.28318530718,
        subtype="ANGLE",
    )
    box_pitch: FloatProperty(
        name="Pitch",
        description="所有方向统一向上或向下倾斜的角度；负值朝下",
        default=-0.0872664626,
        min=-1.0471975512,
        max=1.0471975512,
        subtype="ANGLE",
    )

    # ------------------------------------------------------------------ export
    export_colmap: BoolProperty(
        name="COLMAP (3DGS)",
        description="导出 COLMAP 文本模型 sparse/0/{cameras,images,points3D}.txt —— 3DGS 标准输入",
        default=True,
    )
    export_transforms_json: BoolProperty(
        name="transforms.json",
        description="额外导出 nerfstudio 风格 transforms.json（OpenGL 相机约定）",
        default=True,
    )

    # ------------------------------------------------------------------ bridge
    handover_mode: EnumProperty(
        name="After Render",
        description="渲染导出完成后如何把数据集交给 Photara",
        items=(
            (
                "AUTO",
                "Auto",
                "连接已打开的 Photara Studio（需 --enable-agent-control）；"
                "未发现则自动启动 Studio 并挂载数据集",
                0,
            ),
            (
                "STUDIO",
                "Connect Studio Only",
                "仅连接已打开的 Studio（挂载数据集并开始训练），不启动新实例",
                1,
            ),
            (
                "CLI",
                "CLI Training",
                "无 GUI：直接用 photara.exe --splat-dataset 启动 3DGS 训练",
                2,
            ),
            ("OFF", "Off", "渲染后不做任何交付", 3),
        ),
        default="AUTO",
    )
    studio_timeout: FloatProperty(
        name="Timeout",
        description="连接 Studio Agent Control 的超时时间（秒）",
        default=1.5,
        min=0.1,
        max=30.0,
    )
    studio_exe: StringProperty(
        name="Studio Executable",
        description="photara_studio.exe 路径。留空则自动探测 PATH 与常见安装目录",
        subtype="FILE_PATH",
        default="",
    )
    studio_args: StringProperty(
        name="Studio Arguments",
        description=(
            "启动 Studio 的参数模板；需要包含 --enable-agent-control 才能被插件连接，"
            "{dataset} 会替换为数据集目录"
        ),
        default="--enable-agent-control",
    )
    launch_wait: FloatProperty(
        name="Launch Wait",
        description="启动 Studio 后等待其控制服务就绪的时间（秒）",
        default=20.0,
        min=1.0,
        max=120.0,
    )
    cli_exe: StringProperty(
        name="CLI Executable",
        description="photara.exe（reconstruct CLI）路径。留空则自动探测",
        subtype="FILE_PATH",
        default="",
    )
    cli_args: StringProperty(
        name="CLI Arguments",
        description=(
            "CLI 训练参数模板；{dataset} 替换为数据集目录，{name} 替换为数据集名。"
            "外部数据集不加 --dense 时即执行 3DGS 训练"
        ),
        default='--splat-dataset "{dataset}" --output "{dataset}/{name}.sog"',
    )
    auto_start_training: BoolProperty(
        name="Auto Start 3DGS Training",
        description="挂载数据集后立即通过 studio.job.start 开始 3DGS 训练",
        default=True,
    )

    # ------------------------------------------------------------------ status
    status: StringProperty(name="Status", default="就绪")
    is_rendering: BoolProperty(name="Render Active", default=False, options={"HIDDEN"})
    cancel_requested: BoolProperty(name="Cancel Requested", default=False, options={"HIDDEN"})
    progress: FloatProperty(
        name="Progress", default=0.0, min=0.0, max=1.0, subtype="PERCENTAGE"
    )

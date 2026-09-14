#include "i18n.hpp"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace editor::i18n {
namespace {

Language g_language = Language::en;
std::filesystem::path g_path;
std::unordered_map<std::string_view, const char*> g_zh;
std::unordered_map<std::string_view, const char*> g_ja;
std::unordered_map<std::string_view, const char*> g_ko;
bool g_ready{};

struct Entry {
    const char* en;
    const char* zh;
    const char* ja;
    const char* ko;
};

constexpr Entry k_entries[] = {
    // Menus
    {"File", "文件", "ファイル", "파일"},
    {"New Project", "新建项目", "新規プロジェクト", "새 프로젝트"},
    {"Select Image Folder...", "选择图像文件夹...", "画像フォルダーを選択...", "이미지 폴더 선택..."},
    {"Select Video...", "选择视频...", "動画を選択...", "동영상 선택..."},
    {"Open Project...", "打开项目...", "プロジェクトを開く...", "프로젝트 열기..."},
    {"Save Project", "保存项目", "プロジェクトを保存", "프로젝트 저장"},
    {"Save Project As...", "项目另存为...", "名前を付けて保存...", "다른 이름으로 저장..."},
    {"Export SfM Alignment...", "导出 SfM 对齐...", "SfM アライメントを書き出し...", "SfM 정렬 내보내기..."},
    {"Export Splat...", "导出高斯溅射...", "スプラットを書き出し...", "스플랫 내보내기..."},
    {"Export Mesh...", "导出网格...", "メッシュを書き出し...", "메시 내보내기..."},
    {"Open Output Folder", "打开输出文件夹", "出力フォルダーを開く", "출력 폴더 열기"},
    {"Exit", "退出", "終了", "종료"},
    {"Reconstruction", "重建", "再構成", "재구성"},
    {"Align Photos", "对齐照片", "写真をアライメント", "사진 정렬"},
    {"Re-align Photos", "重新对齐照片", "写真を再アライメント", "사진 재정렬"},
    {"Train 3DGS", "训练 3DGS", "3DGS を学習", "3DGS 학습"},
    {"Train 3DGS + Mesh", "训练 3DGS + 网格", "3DGS + メッシュを学習", "3DGS + 메시 학습"},
    {"Extract Mesh", "提取网格", "メッシュを抽出", "메시 추출"},
    {"Bake Texture", "烘焙纹理", "テクスチャをベイク", "텍스처 베이크"},
    {"Re-bake Texture", "重新烘焙纹理", "テクスチャを再ベイク", "텍스처 재베이크"},
    {"Clear Reconstruction Results...", "清除重建结果...", "再構成結果をクリア...", "재구성 결과 지우기..."},
    {"Resume Job", "继续任务", "ジョブを再開", "작업 재개"},
    {"Pause Job", "暂停任务", "ジョブを一時停止", "작업 일시 중지"},
    {"Stop Job", "停止任务", "ジョブを停止", "작업 중지"},
    {"View", "视图", "表示", "보기"},
    {"Scene", "场景", "シーン", "장면"},
    {"Viewport", "视口", "ビューポート", "뷰포트"},
    {"Inspector", "检查器", "インスペクター", "인스펙터"},
    {"Console", "控制台", "コンソール", "콘솔"},
    {"Status Bar", "状态栏", "ステータスバー", "상태 표시줄"},
    {"3D Scene", "3D 场景", "3D シーン", "3D 장면"},
    {"2D Image QA", "2D 图像质检", "2D 画像 QA", "2D 이미지 QA"},
    {"Show Cameras", "显示相机", "カメラを表示", "카메라 표시"},
    {"Show Ground Grid", "显示地面网格", "地面グリッドを表示", "지면 격자 표시"},
    {"Show Origin Axes", "显示原点坐标轴", "原点軸を表示", "원점 축 표시"},
    {"Show Reconstruction Region", "显示重建区域", "再構成領域を表示", "재구성 영역 표시"},
    {"Reset Layout", "重置布局", "レイアウトをリセット", "레이아웃 재설정"},
    {"Help", "帮助", "ヘルプ", "도움말"},
    {"Viewport Controls", "视口操作", "ビューポート操作", "뷰포트 조작"},
    {"Language", "语言", "言語", "언어"},
    {"Untitled Project", "未命名项目", "無題のプロジェクト", "제목 없는 프로젝트"},

    // Help / controls
    {"3D camera", "3D 相机", "3D カメラ", "3D 카메라"},
    {"LMB drag: orbit", "左键拖动：环绕", "左ドラッグ: オービット", "왼쪽 드래그: 궤도"},
    {"MMB or Shift+LMB drag: pan", "中键或 Shift+左键：平移", "中ドラッグまたは Shift+左: パン", "가운데 또는 Shift+왼쪽: 이동"},
    {"RMB drag: fly look", "右键拖动：飞行观察", "右ドラッグ: フライ視点", "오른쪽 드래그: 비행 시점"},
    {"RMB + WASD/QE: fly; Shift accelerates", "右键 + WASD/QE：飞行；Shift 加速", "右クリック + WASD/QE: 飛行、Shift で加速", "오른쪽 + WASD/QE: 비행, Shift 가속"},
    {"Mouse wheel: dolly; F: frame reconstruction", "滚轮：推拉；F：框选重建", "ホイール: ドリー、F: 再構成をフレーミング", "휠: dolly, F: 재구성 프레이밍"},
    {"Region gizmo: drag RGB arrows to move the box, face dots to resize",
     "区域小工具：拖动 RGB 箭头移动包围盒，拖动面点缩放",
     "領域ギズモ: RGB 矢印で移動、面の点でリサイズ",
     "영역 기즈모: RGB 화살표로 이동, 면 점으로 크기 조절"},
    {"Shift: fine control; Ctrl: snap", "Shift：微调；Ctrl：吸附", "Shift: 微調整、Ctrl: スナップ", "Shift: 미세 조절, Ctrl: 스냅"},
    {"Double-click a point: orbit around it", "双击点：绕该点旋转", "点をダブルクリック: その点を中心にオービット", "점을 더블클릭: 해당 점 중심으로 궤도"},
    {"Double-click a camera frustum: look through it", "双击相机视锥：从该相机观察", "カメラ錐台をダブルクリック: その視点に切替", "카메라 절두체 더블클릭: 해당 시점으로 전환"},
    {"2D image QA", "2D 图像质检", "2D 画像 QA", "2D 이미지 QA"},
    {"Wheel: zoom; LMB/MMB drag: pan; double-click or F: fit",
     "滚轮：缩放；左/中键拖动：平移；双击或 F：适应窗口",
     "ホイール: ズーム、左/中ドラッグ: パン、ダブルクリックまたは F: フィット",
     "휠: 확대, 좌/중 드래그: 이동, 더블클릭 또는 F: 맞춤"},
    {"Left / Right: previous and next capture", "左/右方向键：上一张 / 下一张", "左右キー: 前/次の撮影", "왼쪽/오른쪽: 이전/다음 촬영"},
    {"Compare: drag the vertical handle to wipe GT vs 3DGS",
     "对比：拖动竖线擦除对比真值与 3DGS",
     "比較: 縦ハンドルをドラッグして GT と 3DGS をワイプ",
     "비교: 세로 핸들을 드래그해 GT와 3DGS를 와이프"},
    {"2 / 3: switch 2D image QA and 3D scene", "2 / 3：在 2D 质检与 3D 场景间切换", "2 / 3: 2D QA と 3D シーンを切替", "2 / 3: 2D QA와 3D 장면 전환"},

    // Clear results
    {"Clear Reconstruction Results", "清除重建结果", "再構成結果をクリア", "재구성 결과 지우기"},
    {"Clear the current reconstruction?", "清除当前重建结果？", "現在の再構成をクリアしますか？", "현재 재구성을 지울까요?"},
    {"Clear View Only removes the old result from the editor but keeps all "
     "project files.",
     "仅清除视图会从编辑器移除旧结果，但保留全部项目文件。",
     "表示のみクリアはエディタから旧結果を外しますが、プロジェクトファイルは残します。",
     "보기만 지우기는 편집기에서 이전 결과만 제거하고 프로젝트 파일은 유지합니다."},
    {"Delete Generated Results permanently removes sparse clouds, trained "
     "models, meshes, logs, and the reconstruction cache from the current "
     "project. Source images are never deleted.",
     "删除生成结果会永久移除当前项目中的稀疏点云、训练模型、网格、日志和重建缓存。源图像不会被删除。",
     "生成結果の削除は、疎点群・学習モデル・メッシュ・ログ・再構成キャッシュを永久に消します。元画像は削除しません。",
     "생성 결과 삭제는 희소 클라우드, 학습 모델, 메시, 로그, 재구성 캐시를 영구 삭제합니다. 원본 이미지는 삭제되지 않습니다."},
    {"Current project", "当前项目", "現在のプロジェクト", "현재 프로젝트"},
    {"Clear View Only", "仅清除视图", "表示のみクリア", "보기만 지우기"},
    {"Delete Generated Results", "删除生成结果", "生成結果を削除", "생성 결과 삭제"},
    {"Cancel", "取消", "キャンセル", "취소"},

    // Export modals
    {"Export Mesh", "导出网格", "メッシュを書き出し", "메시 내보내기"},
    {"Choose a format, then pick where to save.", "选择格式，然后指定保存位置。", "形式を選んで保存先を指定します。", "형식을 선택한 뒤 저장 위치를 지정하세요."},
    {"Format", "格式", "形式", "형식"},
    {"Geometry", "几何", "ジオメトリ", "기하"},
    {"DCC", "DCC", "DCC", "DCC"},
    {"glTF", "glTF", "glTF", "glTF"},
    {"Binary mesh with vertex colour.\n"
     "Best for CloudCompare and research tools.",
     "带顶点色的二进制网格。\n适合 CloudCompare 与研究工具。",
     "頂点色付きバイナリメッシュ。\nCloudCompare や研究ツール向け。",
     "정점 색이 있는 바이너리 메시.\nCloudCompare 및 연구 도구에 적합합니다."},
    {"Wavefront OBJ. Texture writes MTL + PNG alongside.\n"
     "Opens in Blender, Maya, and MeshLab.",
     "Wavefront OBJ。带纹理时同时写出 MTL + PNG。\n可在 Blender、Maya、MeshLab 中打开。",
     "Wavefront OBJ。テクスチャ時は MTL + PNG も出力。\nBlender / Maya / MeshLab で開けます。",
     "Wavefront OBJ. 텍스처 시 MTL + PNG도 함께 저장됩니다.\nBlender, Maya, MeshLab에서 열 수 있습니다."},
    {"Single-file glTF 2.0 binary.\n"
     "Texture is packed into the file.",
     "单文件 glTF 2.0 二进制。\n纹理打包进文件。",
     "単一ファイルの glTF 2.0 バイナリ。\nテクスチャを内包します。",
     "단일 파일 glTF 2.0 바이너리.\n텍스처가 파일에 포함됩니다."},
    {"Include texture", "包含纹理", "テクスチャを含める", "텍스처 포함"},
    {"PLY keeps vertex colour only.", "PLY 仅保留顶点色。", "PLY は頂点色のみ保持します。", "PLY는 정점 색만 유지합니다."},
    {"Bake Texture to include an albedo atlas.", "先烘焙纹理以包含反照率图集。", "アルベドアトラスを含めるにはテクスチャをベイクしてください。", "알베도 아틀라스를 포함하려면 텍스처를 베이크하세요."},
    {"Writes OBJ, MTL, and albedo PNG next to each other.", "会并排写出 OBJ、MTL 和反照率 PNG。", "OBJ、MTL、アルベド PNG を並べて書き出します。", "OBJ, MTL, 알베도 PNG를 함께 저장합니다."},
    {"Embeds the albedo atlas in the GLB.", "将反照率图集嵌入 GLB。", "アルベドアトラスを GLB に埋め込みます。", "알베도 아틀라스를 GLB에 포함합니다."},
    {"Writes", "将写入", "出力ファイル", "저장될 파일"},
    {"Export...", "导出...", "書き出し...", "내보내기..."},
    {"Export SfM Alignment", "导出 SfM 对齐", "SfM アライメントを書き出し", "SfM 정렬 내보내기"},
    {"AetherScan ASFM", "AetherScan ASFM", "AetherScan ASFM", "AetherScan ASFM"},
    {"COLMAP", "COLMAP", "COLMAP", "COLMAP"},
    {"Nerfstudio / Blender", "Nerfstudio / Blender", "Nerfstudio / Blender", "Nerfstudio / Blender"},
    {"OpenMVS", "OpenMVS", "OpenMVS", "OpenMVS"},
    {"Native scene: cameras, keypoints, and tracks.", "原生场景：相机、特征点与轨迹。", "ネイティブシーン: カメラ、キーポイント、トラック。", "네이티브 장면: 카메라, 키포인트, 트랙."},
    {"cameras.txt, images.txt, and points3D.txt in a folder.", "文件夹中的 cameras.txt、images.txt、points3D.txt。", "フォルダー内の cameras.txt、images.txt、points3D.txt。", "폴더의 cameras.txt, images.txt, points3D.txt."},
    {"transforms.json for Nerfstudio and Blender.", "供 Nerfstudio 与 Blender 使用的 transforms.json。", "Nerfstudio / Blender 用 transforms.json。", "Nerfstudio와 Blender용 transforms.json."},
    {"OpenMVS needs rectified pinhole images.", "OpenMVS 需要校正后的针孔图像。", "OpenMVS には整流済みピンホール画像が必要です。", "OpenMVS에는 정류된 핀홀 이미지가 필요합니다."},
    {"OpenMVS interface for Viewer and densify import.", "供 Viewer 与稠密导入使用的 OpenMVS 接口。", "Viewer と稠密化取り込み用の OpenMVS インターフェース。", "Viewer 및 밀집화 가져오기용 OpenMVS 인터페이스."},
    {"Folder", "文件夹", "フォルダー", "폴더"},
    {"Location", "位置", "場所", "위치"},
    {"Browse", "浏览", "参照", "찾아보기"},
    {"Export point cloud", "导出点云", "点群を書き出す", "포인트 클라우드 내보내기"},
    {"Writes points3D.txt and points3D.ply.", "写出 points3D.txt 与 points3D.ply。", "points3D.txt と points3D.ply を書き出します。", "points3D.txt와 points3D.ply를 저장합니다."},
    {"Writes a coloured PLY next to the scene file.", "在场景文件旁写出带颜色的 PLY。", "シーンファイルの横に色付き PLY を書き出します。", "장면 파일 옆에 컬러 PLY를 저장합니다."},
    {"Export", "导出", "書き出し", "내보내기"},
    {"Export Splat", "导出高斯溅射", "スプラットを書き出し", "스플랫 내보내기"},
    {"PlayCanvas compressed Gaussians.", "PlayCanvas 压缩高斯。", "PlayCanvas 圧縮ガウシアン。", "PlayCanvas 압축 가우시안."},
    {"Niantic SPZ.", "Niantic SPZ。", "Niantic SPZ。", "Niantic SPZ."},
    {"Khronos KHR_gaussian_splatting GLB.", "Khronos KHR_gaussian_splatting GLB。", "Khronos KHR_gaussian_splatting GLB。", "Khronos KHR_gaussian_splatting GLB."},
    {"Standard 3DGS PLY.", "标准 3DGS PLY。", "標準 3DGS PLY。", "표준 3DGS PLY."},
    {"SH degree", "球谐阶数", "SH 次数", "SH 차수"},
    {"0 keeps only base colour. 3 is the training default.", "0 仅保留基色。3 为训练默认。", "0 は基本色のみ。3 が学習の既定です。", "0은 기본색만 유지합니다. 3이 학습 기본값입니다."},

    // Toolbar
    {"Image Folder", "图像文件夹", "画像フォルダー", "이미지 폴더"},
    {"Select capture image folder, or drop photos / .asfm / .ascan on the viewport",
     "选择采集图像文件夹，或将照片 / .asfm / .ascan 拖到视口",
     "撮影画像フォルダーを選ぶか、写真 / .asfm / .ascan をビューポートへドロップ",
     "촬영 이미지 폴더를 선택하거나 사진 / .asfm / .ascan을 뷰포트에 드롭하세요"},
    {"Video", "视频", "動画", "동영상"},
    {"Select a capture video. Align Photos extracts sharp frames, then runs SfM.",
     "选择采集视频。对齐照片会提取清晰帧，然后运行 SfM。",
     "撮影動画を選択。写真アライメントが鮮明なフレームを抽出し SfM を実行します。",
     "촬영 동영상을 선택하세요. 사진 정렬이 선명한 프레임을 추출한 뒤 SfM을 실행합니다."},
    {"Paused", "已暂停", "一時停止中", "일시 중지됨"},
    {"Aligning...", "对齐中...", "アライメント中...", "정렬 중..."},
    {"Use Stop to abort alignment so you can change images or parameters.",
     "使用停止可中止对齐，以便更改图像或参数。",
     "停止でアライメントを中断し、画像やパラメータを変更できます。",
     "중지를 누르면 정렬을 중단하고 이미지나 파라미터를 바꿀 수 있습니다."},
    {"Training...", "训练中...", "学習中...", "학습 중..."},
    {"Use Stop to abort training so you can change images or parameters.",
     "使用停止可中止训练，以便更改图像或参数。",
     "停止で学習を中断し、画像やパラメータを変更できます。",
     "중지를 누르면 학습을 중단하고 이미지나 파라미터를 바꿀 수 있습니다."},
    {"Train directly from the imported cameras. SfM is skipped.",
     "直接从导入的相机训练。跳过 SfM。",
     "取り込みカメラから直接学習します。SfM はスキップされます。",
     "가져온 카메라로 바로 학습합니다. SfM은 건너뜁니다."},
    {"No alignment yet. Training will run Structure from Motion "
     "first, then optimise Gaussians.",
     "尚未对齐。训练会先运行运动恢复结构，再优化高斯。",
     "まだアライメントがありません。学習は先に SfM を実行してからガウシアンを最適化します。",
     "아직 정렬이 없습니다. 학습은 먼저 SfM을 실행한 뒤 가우시안을 최적화합니다."},
    {"Start 3DGS from the current aligned cameras and sparse cloud.",
     "从当前已对齐相机与稀疏点云开始 3DGS。",
     "現在のアライメント済みカメラと疎点群から 3DGS を開始します。",
     "현재 정렬된 카메라와 희소 클라우드로 3DGS를 시작합니다."},
    {"Build Mesh", "生成网格", "メッシュを生成", "메시 생성"},
    {"Stop the running job first to change reconstruction options.",
     "请先停止正在运行的任务再更改重建选项。",
     "再構成オプションを変えるには、実行中のジョブを先に停止してください。",
     "재구성 옵션을 바꾸려면 먼저 실행 중인 작업을 중지하십시오."},
    {"Extract a surface with the MVS mesh pipeline after alignment.\n"
     "Choose Extract Mesh (MVS) or From Gaussians in the Mesh panel.",
     "对齐后用 MVS 网格流程提取表面。\n在网格面板中选择提取网格 (MVS) 或从高斯生成。",
     "アライメント後に MVS メッシュパイプラインで表面を抽出します。\nメッシュパネルで「メッシュ抽出 (MVS)」または「ガウシアンから」を選んでください。",
     "정렬 후 MVS 메시 파이프라인으로 표면을 추출합니다.\n메시 패널에서 메시 추출(MVS) 또는 가우시안에서를 선택하세요."},
    {"Extract a surface after 3DGS training.\n"
     "From Gaussians enables depth-normal and multi-view geometry\n"
     "losses during optimisation. Switch to Extract Mesh (MVS) in the\n"
     "Mesh panel for a photogrammetry surface.",
     "在 3DGS 训练后提取表面。\n“从高斯生成”会在优化中启用深度-法向与多视图几何损失。\n如需摄影测量表面，请在网格面板切换到提取网格 (MVS)。",
     "3DGS 学習後に表面を抽出します。\n「ガウシアンから」は最適化中に深度・法線と多視点幾何ロスを有効にします。\n写真測量の表面にはメッシュパネルで「メッシュ抽出 (MVS)」へ切り替えてください。",
     "3DGS 학습 후 표면을 추출합니다.\n가우시안에서를 켜면 최적화 중 깊이-법선 및 다시점 기하 손실이 사용됩니다.\n사진측량 표면이 필요하면 메시 패널에서 메시 추출(MVS)로 바꾸세요."},
    {"Open Output", "打开输出", "出力を開く", "출력 열기"},
    {"Resume", "继续", "再開", "재개"},
    {"Pause", "暂停", "一時停止", "일시 중지"},
    {"Stop", "停止", "停止", "중지"},
    {"Continue the paused reconstruction.", "继续已暂停的重建。", "一時停止した再構成を続行します。", "일시 중지된 재구성을 계속합니다."},
    {"CUDA / Vulkan  |  external memory", "CUDA / Vulkan  |  外部内存", "CUDA / Vulkan  |  外部メモリ", "CUDA / Vulkan  |  외부 메모리"},

    // Scene panel
    {"SCENE", "场景", "シーン", "장면"},
    {"External camera dataset selected. Train 3DGS or run Extract Mesh next.",
     "已选择外部相机数据集。接下来训练 3DGS 或提取网格。",
     "外部カメラデータセットを選択済み。次は 3DGS 学習またはメッシュ抽出です。",
     "외부 카메라 데이터셋이 선택되었습니다. 다음으로 3DGS 학습 또는 메시 추출을 실행하세요."},
    {"Align photos to start the reconstruction.", "对齐照片以开始重建。", "写真をアライメントして再構成を開始します。", "사진을 정렬해 재구성을 시작하세요."},
    {"One reconstruction. Switch Sparse / Gaussians / Mesh\n"
     "with the visualization buttons in the viewport.",
     "同一重建。用视口中的可视化按钮在稀疏 / 高斯 / 网格间切换。",
     "1 つの再構成です。ビューポートの表示ボタンで疎点群 / ガウシアン / メッシュを切替。",
     "하나의 재구성입니다. 뷰포트의 시각화 버튼으로 희소 / 가우시안 / 메시를 전환하세요."},
    {"Sparse", "稀疏", "疎", "희소"},
    {"Gaussians", "高斯", "ガウシアン", "가우시안"},
    {"Mesh", "网格", "メッシュ", "메시"},
    {"Texture", "纹理", "テクスチャ", "텍스처"},
    {"PIPELINE", "流程", "パイプライン", "파이프라인"},
    {"Select images", "选择图像", "画像を選択", "이미지 선택"},
    {"Align cameras", "对齐相机", "カメラをアライメント", "카메라 정렬"},
    {"Imported cameras (SfM skipped)", "已导入相机（跳过 SfM）", "取り込みカメラ（SfM スキップ）", "가져온 카메라(SfM 생략)"},
    {"Review sparse cloud", "检查稀疏点云", "疎点群を確認", "희소 클라우드 검토"},
    {"Recomputing", "重新计算中", "再計算中", "재계산 중"},
    {"Loading", "加载中", "読み込み中", "불러오는 중"},
    {"Imported cameras", "已导入相机", "取り込みカメラ", "가져온 카메라"},
    {"On disk", "已在磁盘", "ディスク上", "디스크에 있음"},
    {"Albedo atlas", "反照率图集", "アルベドアトラス", "알베도 아틀라스"},
    {"Projected atlas", "投影图集", "投影アトラス", "투영 아틀라스"},
    {"Project photos onto the mesh", "将照片投影到网格", "写真をメッシュへ投影", "사진을 메시에 투영"},
    {"MVS Mesh", "MVS 网格", "MVS メッシュ", "MVS 메시"},
    {"Photogrammetry", "摄影测量", "写真測量", "사진측량"},
    {"Texture Baking", "纹理烘焙", "テクスチャベイク", "텍스처 베이크"},
    {"Mesh extracted", "已提取网格", "メッシュ抽出済み", "메시 추출됨"},
    {"Then extract mesh", "随后提取网格", "続いてメッシュ抽出", "이어서 메시 추출"},
    {"From Gaussians", "从高斯生成", "ガウシアンから", "가우시안에서"},
    {"SOURCE", "源", "ソース", "소스"},
    {"VIDEO", "视频", "動画", "동영상"},
    {"IMAGES", "图像", "画像", "이미지"},
    {"(not selected)", "（未选择）", "（未選択）", "(선택되지 않음)"},
    {"Stop the running job to choose a different capture.",
     "停止正在运行的任务后才能选择其他采集。",
     "別の撮影を選ぶには実行中のジョブを停止してください。",
     "다른 촬영을 선택하려면 실행 중인 작업을 중지하십시오."},
    {"EXTRACTED FRAMES", "已提取帧", "抽出フレーム", "추출된 프레임"},
    {"PROJECT", "项目", "プロジェクト", "프로젝트"},
    {"EXTERNAL DATASET", "外部数据集", "外部データセット", "외부 데이터셋"},

    // Viewport overlays
    {"Aligning photos...", "正在对齐照片...", "写真をアライメント中...", "사진 정렬 중..."},
    {"Ready to align cameras", "可以开始对齐相机", "カメラアライメントの準備完了", "카메라 정렬 준비됨"},
    {"Drop photos, a video, or a reconstruction", "拖入照片、视频或重建结果", "写真、動画、再構成をドロップ", "사진, 동영상 또는 재구성을 드롭"},
    {"Cameras and points appear as soon as geometry is available",
     "一旦有几何就会显示相机与点",
     "幾何が揃い次第カメラと点が表示されます",
     "기하가 준비되면 카메라와 점이 나타납니다"},
    {"Run Align Photos, or drop a different folder, video, .asfm, or .ascan",
     "运行对齐照片，或拖入其他文件夹、视频、.asfm 或 .ascan",
     "写真アライメントを実行するか、別のフォルダー / 動画 / .asfm / .ascan をドロップ",
     "사진 정렬을 실행하거나 다른 폴더, 동영상, .asfm, .ascan을 드롭하세요"},
    {"Drop an image folder, photos, a video, .asfm, or .ascan onto this view",
     "将图像文件夹、照片、视频、.asfm 或 .ascan 拖到此视图",
     "このビューに画像フォルダー、写真、動画、.asfm、.ascan をドロップ",
     "이 뷰에 이미지 폴더, 사진, 동영상, .asfm, .ascan을 드롭하세요"},
    {"NO ALIGNMENT", "未对齐", "未アライメント", "정렬 없음"},
    {"NO IMAGES", "无图像", "画像なし", "이미지 없음"},
    {"ALIGNING / PARTIAL RESULT", "对齐中 / 部分结果", "アライメント中 / 部分結果", "정렬 중 / 부분 결과"},
    {"ALIGNING", "对齐中", "アライメント中", "정렬 중"},
    {"PREPARING 3DGS", "正在准备 3DGS", "3DGS を準備中", "3DGS 준비 중"},
    {"EXTRACT MESH", "提取网格", "メッシュ抽出", "메시 추출"},
    {"BAKE TEXTURE", "烘焙纹理", "テクスチャベイク", "텍스처 베이크"},
    {"LOADING", "加载中", "読み込み中", "불러오는 중"},
    {"PARTIAL ALIGNMENT / NOT FINAL", "部分对齐 / 非正式结果", "部分アライメント / 未確定", "부분 정렬 / 최종 아님"},
    {"MESH (GPU)", "网格 (GPU)", "メッシュ (GPU)", "메시 (GPU)"},
    {"NO MESH", "无网格", "メッシュなし", "메시 없음"},
    {"GAUSSIAN RINGS", "高斯环", "ガウシアンリング", "가우시안 링"},
    {"NO GAUSSIANS", "无高斯", "ガウシアンなし", "가우시안 없음"},
    {"SPARSE POINT CLOUD", "稀疏点云", "疎点群", "희소 포인트 클라우드"},
    {"CLOUD READY", "点云已就绪", "点群準備完了", "클라우드 준비됨"},
    {"Waiting for cameras from this alignment", "等待本次对齐产生的相机", "このアライメントのカメラを待機", "이 정렬의 카메라를 기다리는 중"},
    {"Previous result cleared", "已清除上次结果", "前回の結果をクリア済み", "이전 결과를 지움"},
    {"Reading sparse.ply...", "正在读取 sparse.ply...", "sparse.ply を読み込み中...", "sparse.ply 읽는 중..."},
    {"Drop a photo folder, .asfm, or .ascan here", "将照片文件夹、.asfm 或 .ascan 拖到此处", "写真フォルダー、.asfm、.ascan をここにドロップ", "사진 폴더, .asfm, .ascan을 여기에 드롭"},
    {"LMB orbit  |  MMB pan  |  RMB + WASD/QE fly  |  F frame  |  drag region gizmo  |  double-click focus",
     "左键环绕  |  中键平移  |  右键 + WASD/QE 飞行  |  F 框选  |  拖动区域小工具  |  双击聚焦",
     "左オービット  |  中パン  |  右 + WASD/QE 飛行  |  F フレーミング  |  領域ギズモ  |  ダブルクリックでフォーカス",
     "왼쪽 궤도  |  가운데 이동  |  오른쪽 + WASD/QE 비행  |  F 프레이밍  |  영역 기즈모  |  더블클릭 초점"},
    {"IDLE", "空闲", "アイドル", "대기"},
    {"GAUSSIAN CENTRES", "高斯中心", "ガウシアン中心", "가우시안 중심"},
    {"LIVE TRAINING PREVIEW", "实时训练预览", "ライブ学習プレビュー", "실시간 학습 미리보기"},
    {"LIVE SPLAT VIEW", "实时溅射视图", "ライブスプラット表示", "실시간 스플랫 보기"},
    {"LAST TRAINING FRAME", "上次训练帧", "最後の学習フレーム", "마지막 학습 프레임"},
    {"TRAINING", "训练中", "学習中", "학습 중"},
    {"VIEWING", "预览中", "表示中", "보는 중"},
    {"READY", "就绪", "準備完了", "준비됨"},
    {"PAUSED", "已暂停", "一時停止", "일시 중지"},
    {"EXPORTING", "导出中", "書き出し中", "내보내는 중"},
    {"Point Cloud", "点云", "点群", "포인트 클라우드"},
    {"Splat", "高斯溅射", "スプラット", "스플랫"},
    {"Train 3DGS to view the splat", "训练 3DGS 以查看溅射", "スプラットを見るには 3DGS を学習", "스플랫을 보려면 3DGS를 학습하세요"},
    {"Rings", "环", "リング", "링"},
    {"Available while training or after a Gaussian model exists",
     "训练期间或已有高斯模型时可用",
     "学習中、またはガウシアンモデルがあるときに利用可能",
     "학습 중이거나 가우시안 모델이 있을 때 사용할 수 있습니다"},
    {"Build a mesh to inspect the reconstructed surface",
     "生成网格以检查重建表面",
     "再構成表面を確認するにはメッシュを生成",
     "재구성 표면을 보려면 메시를 생성하세요"},
    {"Hide camera frustums", "隐藏相机视锥", "カメラ錐台を隠す", "카메라 절두체 숨기기"},
    {"Show camera frustums", "显示相机视锥", "カメラ錐台を表示", "카메라 절두체 표시"},
    {"Hide ground grid", "隐藏地面网格", "地面グリッドを隠す", "지면 격자 숨기기"},
    {"Show ground grid", "显示地面网格", "地面グリッドを表示", "지면 격자 표시"},
    {"Hide reconstruction region", "隐藏重建区域", "再構成領域を隠す", "재구성 영역 숨기기"},
    {"Show reconstruction region", "显示重建区域", "再構成領域を表示", "재구성 영역 표시"},

    // Inspector
    {"Project", "项目", "プロジェクト", "프로젝트"},
    {"A reconstruction is running. Stop it to change images, "
     "video, or parameters.",
     "正在重建。请先停止后再更改图像、视频或参数。",
     "再構成の実行中です。画像・動画・パラメータを変えるには停止してください。",
     "재구성이 실행 중입니다. 이미지, 동영상, 파라미터를 바꾸려면 중지하십시오."},
    {"Image source", "图像源", "画像ソース", "이미지 소스"},
    {"Video extraction", "视频提取", "動画抽出", "동영상 추출"},
    {"Align Photos extracts the sharpest stills, then runs SfM.",
     "对齐照片会提取最清晰的静帧，然后运行 SfM。",
     "写真アライメントが最も鮮明な静止画を抽出し、SfM を実行します。",
     "사진 정렬이 가장 선명한 프레임을 추출한 뒤 SfM을 실행합니다."},
    {"Not found", "未找到", "見つかりません", "없음"},
    {"Install ffmpeg and add it to PATH. AetherScan also "
     "looks next to the app and in common install folders.",
     "请安装 ffmpeg 并将其加入 PATH。AetherScan 也会在程序旁及常见安装目录查找。",
     "ffmpeg をインストールして PATH に追加してください。アプリ隣と一般的な導入先も検索します。",
     "ffmpeg를 설치하고 PATH에 추가하세요. 앱 옆과 일반적인 설치 폴더도 검색합니다."},
    {"Ready", "就绪", "準備完了", "준비됨"},
    {"Target FPS", "目标帧率", "目標 FPS", "목표 FPS"},
    {"Kept frames per second of source time.\n"
     "2 FPS is a good default for handheld scans.",
     "按源时间每秒保留的帧数。\n手持扫描建议 2 FPS。",
     "ソース時間あたりに残すフレーム数。\n手持ちスキャンでは 2 FPS が目安です。",
     "원본 시간 기준으로 초당 유지할 프레임 수.\n핸드헬드 스캔에는 2 FPS가 적당합니다."},
    {"Max frames (0 = no cap)", "最大帧数（0 = 不限制）", "最大フレーム（0 = 上限なし）", "최대 프레임 (0 = 제한 없음)"},
    {"Will write", "将写入", "書き出し先", "저장 위치"},
    {"Advanced", "高级", "詳細", "고급"},
    {"Sharpness window", "锐度窗口", "シャープネス窓", "선명도 창"},
    {"Keep the sharpest of N consecutive candidates.\n"
     "1 disables blur selection. 3 is the default.",
     "在连续 N 个候选中保留最清晰的一帧。\n1 关闭模糊筛选。默认 3。",
     "連続 N 候補のうち最も鮮明な 1 枚を残します。\n1 でブラー選択オフ。既定は 3。",
     "연속 N개 후보 중 가장 선명한 프레임을 유지합니다.\n1은 블러 선택을 끕니다. 기본값은 3입니다."},
    {"JPEG quality", "JPEG 质量", "JPEG 品質", "JPEG 품질"},
    {"Scale", "缩放", "スケール", "배율"},
    {"Rotate", "旋转", "回転", "회전"},
    {"Frames folder (optional)", "帧文件夹（可选）", "フレームフォルダー（任意）", "프레임 폴더(선택)"},
    {"Empty uses <video_stem>/images next to the file.",
     "留空则使用文件旁的 <视频名>/images。",
     "空欄ならファイル隣の <動画名>/images を使います。",
     "비우면 파일 옆의 <동영상이름>/images를 사용합니다."},
    {"Project file", "项目文件", "プロジェクトファイル", "프로젝트 파일"},
    {"Project format", "项目格式", "プロジェクト形式", "프로젝트 형식"},
    {"External SfM dataset (optional)", "外部 SfM 数据集（可选）", "外部 SfM データセット（任意）", "외부 SfM 데이터셋(선택)"},
    {"Imported cameras replace Align Photos. Review the cloud, then "
     "run Train 3DGS or Extract Mesh.",
     "导入的相机将替代对齐照片。检查点云后运行训练 3DGS 或提取网格。",
     "取り込みカメラが写真アライメントの代わりになります。点群を確認してから 3DGS 学習またはメッシュ抽出へ。",
     "가져온 카메라가 사진 정렬을 대체합니다. 클라우드를 확인한 뒤 3DGS 학습 또는 메시 추출을 실행하세요."},
    {"Dataset format", "数据集格式", "データセット形式", "데이터셋 형식"},
    {"Auto detect", "自动检测", "自動検出", "자동 감지"},
    {"RealityCapture", "RealityCapture", "RealityCapture", "RealityCapture"},
    {"Initial point cloud (optional)", "初始点云（可选）", "初期点群（任意）", "초기 포인트 클라우드(선택)"},
    {"Optional dense PLY initializer. COLMAP/OpenMVS sparse points "
     "are used automatically when available.",
     "可选的稠密 PLY 初始化。若有 COLMAP/OpenMVS 稀疏点会自动使用。",
     "任意の密な PLY 初期化。COLMAP/OpenMVS の疎点が使える場合は自動利用。",
     "선택적 밀집 PLY 초기화. COLMAP/OpenMVS 희소 점이 있으면 자동 사용됩니다."},
    {"Clear external dataset", "清除外部数据集", "外部データセットをクリア", "외부 데이터셋 지우기"},
    {"Trained splat model (optional)", "已训练溅射模型（可选）", "学習済みスプラット（任意）", "학습된 스플랫 모델(선택)"},
    {"Load an existing PLY, SOG, SPZ, or GLB model for preview. "
     "When set, it takes precedence over the trained working copy.",
     "加载已有 PLY、SOG、SPZ 或 GLB 进行预览。设置后优先于训练工作副本。",
     "既存の PLY / SOG / SPZ / GLB をプレビュー。指定時は学習中の作業コピーより優先。",
     "기존 PLY, SOG, SPZ, GLB를 미리보기합니다. 설정 시 학습 작업본보다 우선합니다."},
    {"Camera Alignment", "相机对齐", "カメラアライメント", "카메라 정렬"},
    {"Solver", "求解器", "ソルバー", "솔버"},
    {"Global", "全局", "グローバル", "전역"},
    {"Incremental", "增量", "インクリメンタル", "증분"},
    {"Hierarchical", "分层", "階層的", "계층"},
    {"Camera model", "相机模型", "カメラモデル", "카메라 모델"},
    {"Auto", "自动", "自動", "자동"},
    {"Pinhole", "针孔", "ピンホール", "핀홀"},
    {"OpenCV Fisheye", "OpenCV 鱼眼", "OpenCV 魚眼", "OpenCV 어안"},
    {"Auto compares matched image geometry and EXIF lens hints.\n"
     "Ambiguous results use Pinhole. You can override the model.",
     "自动比较匹配图像几何与 EXIF 镜头提示。\n不确定时使用针孔。可手动覆盖。",
     "マッチ幾何と EXIF レンズ情報を比較。\n曖昧な場合はピンホール。上書き可能。",
     "매칭 기하와 EXIF 렌즈 힌트를 비교합니다.\n모호하면 핀홀을 사용합니다. 수동 변경 가능합니다."},
    {"Result camera", "结果相机", "結果カメラ", "결과 카메라"},
    {"Pending alignment", "等待对齐", "アライメント待ち", "정렬 대기"},
    {"Mixed", "混合", "混在", "혼합"},
    {"Max features per image", "每图最大特征数", "画像あたり最大特徴", "이미지당 최대 특징"},
    {"Reuse cached alignment", "复用缓存对齐", "キャッシュしたアライメントを再利用", "캐시된 정렬 재사용"},
    {"Write a project .cache folder so later Align/Train can\n"
     "reuse extracted features. Off by default: skip that folder.\n"
     "Align/Train keep working copies for preview. They do not\n"
     "write .ascan, .asfm, or PLY files unless you Save Project\n"
     "or Export.",
     "写入项目 .cache 以便后续对齐/训练复用特征。默认关闭。\n对齐/训练会保留预览用工作副本，除非保存项目或导出，否则不写 .ascan、.asfm 或 PLY。",
     "プロジェクト .cache を書き、後のアライメント/学習で特徴を再利用。既定はオフ。\nプレビュー用作業コピーは残しますが、保存または書き出しまで .ascan / .asfm / PLY は出しません。",
     "프로젝트 .cache를 써서 이후 정렬/학습이 특징을 재사용합니다. 기본은 끔.\n미리보기용 작업본은 남지만, 프로젝트 저장 또는 내보내기 전에는 .ascan, .asfm, PLY를 쓰지 않습니다."},
    {"Export SfM Alignment", "导出 SfM 对齐", "SfM アライメントを書き出し", "SfM 정렬 내보내기"},
    {"Save cameras and tracks as ASFM, COLMAP,\n"
     "Nerfstudio / Blender, or OpenMVS.",
     "将相机与轨迹保存为 ASFM、COLMAP、\nNerfstudio / Blender 或 OpenMVS。",
     "カメラとトラックを ASFM、COLMAP、\nNerfstudio / Blender、OpenMVS で保存。",
     "카메라와 트랙을 ASFM, COLMAP,\nNerfstudio / Blender, OpenMVS로 저장합니다."},
    {"SfM scene", "SfM 场景", "SfM シーン", "SfM 장면"},
    {"OpenMVS file", "OpenMVS 文件", "OpenMVS ファイル", "OpenMVS 파일"},
    {"Registered views", "已注册视图", "登録ビュー", "등록된 뷰"},
    {"Mean reprojection", "平均重投影", "平均再投影", "평균 재투영"},
    {"Sparse points", "稀疏点数", "疎点数", "희소 점 수"},
    {"Gaussian Splatting", "高斯溅射", "ガウシアン・スプラッティング", "가우시안 스플래팅"},
    {"Capture type", "采集类型", "撮影タイプ", "촬영 유형"},
    {"Object", "物体", "オブジェクト", "오브젝트"},
    {"Object uses the reconstruction region as Splat SubjectBounds "
     "(focus region).\n"
     "Scene trains unbounded and ignores the box.",
     "物体模式用重建区域作为溅射 SubjectBounds（焦点区域）。\n场景模式无界训练并忽略包围盒。",
     "オブジェクトは再構成領域を Splat SubjectBounds（焦点）に使います。\nシーンは無界学習でボックスを無視します。",
     "오브젝트는 재구성 영역을 스플랫 SubjectBounds(초점 영역)로 사용합니다.\n장면은 무한 학습이며 박스를 무시합니다."},
    {"Densification strategy", "稠密化策略", "稠密化戦略", "밀집화 전략"},
    {"ADC IGS", "ADC IGS", "ADC IGS", "ADC IGS"},
    {"ADC Plus", "ADC Plus", "ADC Plus", "ADC Plus"},
    {"Iterations", "迭代次数", "イテレーション", "반복"},
    {"Densification cap", "致密化上限", "稠密化上限", "밀집화 상한"},
    {"Maximum Gaussian count during densification.\n"
     "Initialization uses the full source cloud.\n"
     "Default 1,000,000.",
     "致密化过程中的高斯数量上限。\n初始化使用全部源点云。\n默认 1,000,000。",
     "稠密化中のガウシアン数上限。\n初期化は入力点群をすべて使います。\n既定 1,000,000。",
     "밀집화 중 가우시안 수 상한.\n초기화는 원본 점군을 전부 사용합니다.\n기본 1,000,000."},
    {"Spherical-harmonic colour bands.\n"
     "0 is diffuse only. 3 is the training default.",
     "球谐颜色阶。\n0 仅为漫反射。3 为训练默认。",
     "球面調和の色バンド。\n0 は拡散のみ。3 が学習の既定。",
     "구면조화 색 밴드.\n0은 확산만. 3이 학습 기본값입니다."},
    {"Max training resolution", "最大训练分辨率", "最大学習解像度", "최대 학습 해상도"},
    {"Live preview cadence", "实时预览间隔", "ライブプレビュー間隔", "실시간 미리보기 주기"},
    {"Coarse-to-fine resolution", "由粗到细分辨率", "粗い解像度から細かく", "거친 해상도에서 세밀하게"},
    {"Foreground mask training", "前景蒙版训练", "前景マスク学習", "전경 마스크 학습"},
    {"Normal field", "法向场", "法線場", "법선 필드"},
    {"GaussianWrapping's learned normal field.\n"
     "Off (default) trains the GGGS path.",
     "GaussianWrapping 学习的法向场。\n关闭（默认）走 GGGS 路径。",
     "GaussianWrapping の学習法線場。\nオフ（既定）は GGGS 経路で学習。",
     "GaussianWrapping의 학습 법선 필드.\n끄면(기본) GGGS 경로로 학습합니다."},
    {"Save the trained Gaussians. Choose PLY, SOG, SPZ, or GLB\n"
     "and the spherical-harmonics degree.",
     "保存训练得到的高斯。可选 PLY、SOG、SPZ 或 GLB，以及球谐阶数。",
     "学習したガウシアンを保存。PLY / SOG / SPZ / GLB と SH 次数を選択。",
     "학습된 가우시안을 저장합니다. PLY, SOG, SPZ, GLB와 SH 차수를 선택하세요."},
    {"Build mesh", "生成网格", "メッシュを生成", "메시 생성"},
    {"Include a surface mesh in the reconstruction.\n"
     "Choose From Gaussians or Extract Mesh (MVS) below.",
     "在重建中包含表面网格。\n在下方选择从高斯生成或提取网格 (MVS)。",
     "再構成に表面メッシュを含めます。\n下で「ガウシアンから」か「メッシュ抽出 (MVS)」を選択。",
     "재구성에 표면 메시를 포함합니다.\n아래에서 가우시안에서 또는 메시 추출(MVS)을 선택하세요."},
    {"Method", "方法", "方法", "방법"},
    {"Train 3DGS with depth and normal losses, then extract a "
     "surface from the Gaussians.",
     "用深度与法向损失训练 3DGS，再从高斯提取表面。",
     "深度・法線ロスで 3DGS を学習し、ガウシアンから表面を抽出します。",
     "깊이와 법선 손실로 3DGS를 학습한 뒤 가우시안에서 표면을 추출합니다."},
    {"Extract Mesh (MVS)", "提取网格 (MVS)", "メッシュ抽出 (MVS)", "메시 추출 (MVS)"},
    {"PatchMatch stereo from aligned cameras, then fuse a mesh. "
     "This is the MVS mesh pipeline. 3DGS stays appearance-only.",
     "从已对齐相机做 PatchMatch 立体匹配再融合网格。这是 MVS 网格流程。3DGS 仅负责外观。",
     "アライメント済みカメラから PatchMatch ステレオ後にメッシュ融合。MVS メッシュ経路です。3DGS は見た目のみ。",
     "정렬된 카메라로 PatchMatch 스테레오 후 메시를 융합합니다. MVS 메시 파이프라인입니다. 3DGS는 외형만 담당합니다."},
    {"Surface", "表面", "サーフェス", "표면"},
    {"TSDF", "TSDF", "TSDF", "TSDF"},
    {"Delaunay", "Delaunay", "ドロネー", "들로네"},
    {"PAM", "PAM", "PAM", "PAM"},
    {"How depth is fused into triangles.\n"
     "PAM is GaussianWrapping occupancy meshing.",
     "深度如何融合成三角形。\nPAM 是 GaussianWrapping 占用网格。",
     "深度を三角形へ融合する方法。\nPAM は GaussianWrapping の占有メッシュ。",
     "깊이를 삼각형으로 융합하는 방식.\nPAM은 GaussianWrapping 점유 메시입니다."},
    {"How MVS depth maps are fused into triangles.",
     "MVS 深度图如何融合成三角形。",
     "MVS 深度マップを三角形へ融合する方法。",
     "MVS 깊이 맵을 삼각형으로 융합하는 방식."},
    {"Geometry training", "几何训练", "幾何学習", "기하 학습"},
    {"Depth-normal weight", "深度-法向权重", "深度・法線ウェイト", "깊이-법선 가중치"},
    {"Multi-view geometry weight", "多视图几何权重", "多視点幾何ウェイト", "다시점 기하 가중치"},
    {"Multi-view NCC weight", "多视图 NCC 权重", "多視点 NCC ウェイト", "다시점 NCC 가중치"},
    {"Geometry loss start iteration", "几何损失起始迭代", "幾何ロス開始イテレーション", "기하 손실 시작 반복"},
    {"Appearance-only. Train 3DGS without extracting a surface.",
     "仅外观。训练 3DGS 但不提取表面。",
     "見た目のみ。表面抽出なしで 3DGS を学習。",
     "외형만. 표면 추출 없이 3DGS를 학습합니다."},
    {"Save the reconstructed surface. Choose PLY, OBJ, or GLB,\n"
     "and whether to include the baked texture.",
     "保存重建表面。可选 PLY、OBJ 或 GLB，以及是否包含烘焙纹理。",
     "再構成表面を保存。PLY / OBJ / GLB とベイクテクスチャの有無を選択。",
     "재구성 표면을 저장합니다. PLY, OBJ, GLB와 베이크 텍스처 포함 여부를 선택하세요."},
    {"Last reconstruction step. Unwrap the mesh, project calibrated "
     "photos with aether_drender, then optionally refine the atlas.",
     "重建最后一步。展开网格 UV，用 aether_drender 投影已标定照片，可选优化图集。",
     "再構成の最終段。メッシュを展開し aether_drender で校正写真を投影、任意でアトラスを洗練。",
     "재구성 마지막 단계. 메시를 펼치고 aether_drender로 보정 사진을 투영한 뒤 선택적으로 아틀라스를 다듬습니다."},
    {"This build was compiled without texture baking (Vulkan + aether_drender).",
     "此构建未包含纹理烘焙（Vulkan + aether_drender）。",
     "このビルドはテクスチャベイクなし（Vulkan + aether_drender）。",
     "이 빌드는 텍스처 베이크 없이 컴파일되었습니다(Vulkan + aether_drender)."},
    {"Quality", "质量", "品質", "품질"},
    {"Fast", "快速", "高速", "빠르게"},
    {"Standard", "标准", "標準", "표준"},
    {"High", "高", "高", "높음"},
    {"Fast: 1024 atlas, projection only.\n"
     "Standard: 2048 atlas + photometric refine.\n"
     "High: 4096 atlas + longer refine.",
     "快速：1024 图集，仅投影。\n标准：2048 图集 + 光度优化。\n高：4096 图集 + 更长优化。",
     "高速: 1024 アトラス、投影のみ。\n標準: 2048 + 測光洗練。\n高: 4096 + 長めの洗練。",
     "빠르게: 1024 아틀라스, 투영만.\n표준: 2048 + 측광 정제.\n높음: 4096 + 더 긴 정제."},
    {"Atlas size", "图集尺寸", "アトラスサイズ", "아틀라스 크기"},
    {"Square albedo atlas written with the textured OBJ.",
     "随带纹理 OBJ 写出的方形反照率图集。",
     "テクスチャ付き OBJ と一緒に書く正方形アルベドアトラス。",
     "텍스처 OBJ와 함께 저장되는 정사각 알베도 아틀라스."},
    {"Remove lighting (albedo)", "去除光照（反照率）", "照明除去（アルベド）", "조명 제거(알베도)"},
    {"Run Intrinsic delighter on the photos before projection.\n"
     "Needs ONNX Runtime and the Intrinsic stage_*.onnx models.",
     "投影前对照片运行 Intrinsic 去光照。\n需要 ONNX Runtime 与 Intrinsic stage_*.onnx 模型。",
     "投影前に Intrinsic デライターを実行。\nONNX Runtime と Intrinsic stage_*.onnx が必要。",
     "투영 전 사진에 Intrinsic 디라이터를 실행합니다.\nONNX Runtime과 Intrinsic stage_*.onnx가 필요합니다."},
    {"Refine atlas", "优化图集", "アトラスを洗練", "아틀라스 정제"},
    {"Photometric + seam optimization in aether_drender after the "
     "projective bake. Off is faster; on cleans view seams.",
     "投影烘焙后在 aether_drender 中做光度与接缝优化。关闭更快；开启可清理视角接缝。",
     "投影ベイク後に aether_drender で測光とシーム最適化。オフは速い、オンは視点シームを整えます。",
     "투영 베이크 후 aether_drender에서 측광·시임 최적화. 끄면 빠르고, 켜면 시점 시임을 정리합니다."},
    {"Status", "状态", "状態", "상태"},
    {"Albedo ready", "反照率已就绪", "アルベド準備完了", "알베도 준비됨"},
    {"Texture ready", "纹理已就绪", "テクスチャ準備完了", "텍스처 준비됨"},
    {"Mesh is ready. Bake Texture to project the photos.",
     "网格已就绪。烘焙纹理以投影照片。",
     "メッシュ準備完了。テクスチャをベイクして写真を投影。",
     "메시가 준비되었습니다. 텍스처를 베이크해 사진을 투영하세요."},
    {"UV unwrap + multi-view projection onto the current mesh.\n"
     "Use Export Mesh to save the atlas with the model.",
     "对当前网格做 UV 展开与多视图投影。\n用导出网格把图集与模型一起保存。",
     "現在のメッシュへ UV 展開と多視点投影。\nメッシュ書き出しでアトラスも保存。",
     "현재 메시에 UV 펼치기와 다시점 투영.\n메시 내보내기로 아틀라스를 함께 저장하세요."},
    {"Display", "显示", "表示", "표시"},
    {"Show cameras", "显示相机", "カメラを表示", "카메라 표시"},
    {"Wire frustums and capture photos in the 3D / training view.",
     "在 3D / 训练视图中显示线框视锥与采集照片。",
     "3D / 学習ビューでワイヤー錐台と撮影写真を表示。",
     "3D / 학습 뷰에서 와이어 절두체와 촬영 사진을 표시합니다."},
    {"Show trajectory", "显示轨迹", "軌跡を表示", "궤적 표시"},
    {"Show ground grid", "显示地面网格", "地面グリッドを表示", "지면 격자 표시"},
    {"Show origin axes", "显示原点坐标轴", "原点軸を表示", "원점 축 표시"},
    {"RGB axes at the world origin (X red, Y green, Z blue).",
     "世界原点处的 RGB 坐标轴（X 红、Y 绿、Z 蓝）。",
     "世界原点の RGB 軸（X 赤、Y 緑、Z 青）。",
     "월드 원점의 RGB 축(X 빨강, Y 초록, Z 파랑)."},
    {"Show reconstruction region", "显示重建区域", "再構成領域を表示", "재구성 영역 표시"},
    {"Splat object-mode SubjectBounds / focus region.\n"
     "Drag the center arrows to move the box, or a face dot\n"
     "to resize. Object training and mesh extraction use this volume.",
     "溅射物体模式的 SubjectBounds / 焦点区域。\n拖动中心箭头移动包围盒，或拖动面点缩放。物体训练与网格提取使用该体积。",
     "スプラット物体モードの SubjectBounds / 焦点領域。\n中央矢印で移動、面の点でリサイズ。物体学習とメッシュ抽出がこの体積を使います。",
     "스플랫 오브젝트 모드 SubjectBounds / 초점 영역.\n가운데 화살표로 이동, 면 점으로 크기 조절. 오브젝트 학습과 메시 추출이 이 볼륨을 사용합니다."},
    {"Fit Region to Cloud", "使区域贴合点云", "領域を点群に合わせる", "영역을 클라우드에 맞추기"},
    {"Reset to the SfM SubjectBounds used by Splat object mode.",
     "重置为溅射物体模式使用的 SfM SubjectBounds。",
     "スプラット物体モードが使う SfM SubjectBounds に戻す。",
     "스플랫 오브젝트 모드가 쓰는 SfM SubjectBounds로 되돌립니다."},
    {"Image QA", "图像质检", "画像 QA", "이미지 QA"},
    {"Capture", "采集", "撮影", "촬영"},
    {"Camera", "相机", "カメラ", "카메라"},
    {"Resolution", "分辨率", "解像度", "해상도"},
    {"Registered", "已注册", "登録済み", "등록됨"},
    {"yes", "是", "はい", "예"},
    {"no", "否", "いいえ", "아니요"},
    {"Observations", "观测数", "観測", "관측"},
    {"Features", "特征", "特徴", "특징"},
    {"Triangulated", "已三角化", "三角測量済み", "삼각측량됨"},
    {"Images", "图像", "画像", "이미지"},
    {"Align photos to inspect cameras and features.",
     "对齐照片以检查相机与特征。",
     "写真をアライメントしてカメラと特徴を確認。",
     "사진을 정렬해 카메라와 특징을 검사하세요."},
    {"Show triangulated", "显示已三角化", "三角測量済みを表示", "삼각측량됨 표시"},
    {"Show untracked keypoints", "显示未跟踪关键点", "未追跡キーポイントを表示", "미추적 키포인트 표시"},
    {"COMPARE", "对比", "比較", "비교"},
    {"Projection", "投影", "投影", "투영"},
    {"Perspective", "透视", "透視", "원근"},
    {"Orthographic", "正交", "正投影", "직교"},
    {"Fisheye", "鱼眼", "魚眼", "어안"},
    {"Panorama", "全景", "パノラマ", "파노라마"},
    {"Perspective — standard lens with field of view.\n"
     "Orthographic — parallel view, no foreshortening.\n"
     "Fisheye — OpenCV equidistant, same model as training.\n"
     "Panorama — 360° × 180° spherical view.",
     "透视 — 带视场角的标准镜头。\n正交 — 平行投影，无近大远小。\n鱼眼 — OpenCV 等距，与训练相同。\n全景 — 360° × 180° 球面视图。",
     "透視 — 画角付き標準レンズ。\n正投影 — 平行投影で遠近なし。\n魚眼 — OpenCV 等距離、学習と同じ。\nパノラマ — 360° × 180° 球面。",
     "원근 — 시야각이 있는 표준 렌즈.\n직교 — 평행 투영, 원근 없음.\n어안 — OpenCV 등거리, 학습과 동일.\n파노라마 — 360° × 180° 구면 뷰."},
    {"Fisheye field of view", "鱼眼视场角", "魚眼画角", "어안 시야각"},
    {"Field of view", "视场角", "画角", "시야각"},
    {"Lens distortion", "镜头畸变", "レンズ歪み", "렌즈 왜곡"},
    {"OpenCV fisheye k1. Zero is a pure equidistant lens.",
     "OpenCV 鱼眼 k1。0 为纯等距镜头。",
     "OpenCV 魚眼 k1。0 は純粋な等距離レンズ。",
     "OpenCV 어안 k1. 0은 순수 등거리 렌즈입니다."},
    {"View height", "视图高度", "表示高さ", "보기 높이"},
    {"World-space height visible in the viewport.\n"
     "Scroll the view to zoom.",
     "视口中可见的世界空间高度。\n滚动视图进行缩放。",
     "ビューポートに見えるワールド空間の高さ。\nスクロールでズーム。",
     "뷰포트에 보이는 월드 공간 높이.\n스크롤로 확대/축소하세요."},
    {"Full 360° × 180° spherical view around the camera.",
     "相机周围完整 360° × 180° 球面视图。",
     "カメラ周囲の 360° × 180° 球面ビュー。",
     "카메라 주위 360° × 180° 구면 뷰."},
    {"Fly speed", "飞行速度", "飛行速度", "비행 속도"},
    {"Vertices", "顶点", "頂点", "정점"},
    {"Faces", "面", "面", "면"},
    {"Albedo texture", "反照率纹理", "アルベドテクスチャ", "알베도 텍스처"},
    {"Sample the baked atlas with mesh UVs.",
     "用网格 UV 采样烘焙图集。",
     "メッシュ UV でベイクアトラスをサンプル。",
     "메시 UV로 베이크 아틀라스를 샘플링합니다."},
    {"Bake Texture to preview the albedo on the mesh.",
     "烘焙纹理以在网格上预览反照率。",
     "テクスチャをベイクしてメッシュ上のアルベドをプレビュー。",
     "텍스처를 베이크해 메시에서 알베도를 미리보세요."},
    {"Wireframe", "线框", "ワイヤーフレーム", "와이어프레임"},
    {"Draw triangle edges on top of the shaded surface.",
     "在着色表面上绘制三角形边。",
     "シェーディング面上に三角形エッジを描画。",
     "셰이딩된 표면 위에 삼각형 경계를 그립니다."},
    {"Vertex colour", "顶点色", "頂点色", "정점 색"},
    {"Use PLY vertex colours instead of clay shading.",
     "使用 PLY 顶点色而非黏土着色。",
     "クレイシェーディングの代わりに PLY 頂点色を使用。",
     "클레이 셰이딩 대신 PLY 정점 색을 사용합니다."},
    {"Rendered face budget", "渲染面数预算", "描画面数予算", "렌더 면 예산"},
    {"Point size", "点大小", "点サイズ", "점 크기"},
    {"Rendered point budget", "渲染点数预算", "描画点数予算", "렌더 점 예산"},
    {"Ring budget", "环预算", "リング予算", "링 예산"},
    {"Ring scale", "环缩放", "リングスケール", "링 배율"},
    {"Camera marker size", "相机标记大小", "カメラマーカーサイズ", "카메라 마커 크기"},
    {"Colour by depth", "按深度着色", "深度で色付け", "깊이로 색칠"},
    {"Replace sampled point colours with a near-to-far ramp.",
     "用近到远的渐变替换采样点颜色。",
     "サンプリング色を近→遠のランプに置き換え。",
     "샘플 점 색을 근거리→원거리 램프로 바꿉니다."},
    {"Training Telemetry", "训练遥测", "学習テレメトリ", "학습 텔레메트리"},
    {"Iteration", "迭代", "イテレーション", "반복"},
    {"Total loss", "总损失", "総ロス", "총 손실"},
    {"RGB", "RGB", "RGB", "RGB"},
    {"Depth", "深度", "深度", "깊이"},
    {"Normal", "法向", "法線", "법선"},
    {"Multi-view geo", "多视图几何", "多視点幾何", "다시점 기하"},
    {"Multi-view NCC", "多视图 NCC", "多視点 NCC", "다시점 NCC"},
    {"Step time", "单步耗时", "ステップ時間", "스텝 시간"},
    {"Resolution scale", "分辨率缩放", "解像度スケール", "해상도 배율"},
    {"Preview timeline", "预览时间线", "プレビュータイムライン", "미리보기 타임라인"},

    // Job labels / tooltips
    {"Pause Alignment", "暂停对齐", "アライメントを一時停止", "정렬 일시 중지"},
    {"Pause Export", "暂停导出", "書き出しを一時停止", "내보내기 일시 중지"},
    {"Pause Extract Mesh", "暂停提取网格", "メッシュ抽出を一時停止", "메시 추출 일시 중지"},
    {"Pause Bake Texture", "暂停烘焙纹理", "テクスチャベイクを一時停止", "텍스처 베이크 일시 중지"},
    {"Pause Training", "暂停训练", "学習を一時停止", "학습 일시 중지"},
    {"Resume Alignment", "继续对齐", "アライメントを再開", "정렬 재개"},
    {"Resume Export", "继续导出", "書き出しを再開", "내보내기 재개"},
    {"Resume Extract Mesh", "继续提取网格", "メッシュ抽出を再開", "메시 추출 재개"},
    {"Resume Bake Texture", "继续烘焙纹理", "テクスチャベイクを再開", "텍스처 베이크 재개"},
    {"Resume Training", "继续训练", "学習を再開", "학습 재개"},
    {"Stop Alignment", "停止对齐", "アライメントを停止", "정렬 중지"},
    {"Stop Export", "停止导出", "書き出しを停止", "내보내기 중지"},
    {"Stop Extract Mesh", "停止提取网格", "メッシュ抽出を停止", "메시 추출 중지"},
    {"Stop Bake Texture", "停止烘焙纹理", "テクスチャベイクを停止", "텍스처 베이크 중지"},
    {"Stop Training", "停止训练", "学習を停止", "학습 중지"},
    {"Abort the running job so you can change images, video, or parameters "
     "and start again.\nProgress since the last saved artifact is discarded.\n"
     "Shift+Esc",
     "中止正在运行的任务以便更改图像、视频或参数后重新开始。\n自上次保存产物以来的进度会丢弃。\nShift+Esc",
     "実行中ジョブを中止し、画像・動画・パラメータを変えてやり直せます。\n最後に保存した成果以降の進捗は破棄。\nShift+Esc",
     "실행 중인 작업을 중단해 이미지, 동영상, 파라미터를 바꾼 뒤 다시 시작할 수 있습니다.\n마지막 저장 이후 진행은 버려집니다.\nShift+Esc"},
    {"Freeze the running job without discarding progress. Resume to continue.",
     "冻结正在运行的任务且不丢弃进度。继续即可恢复。",
     "進捗を捨てずに実行中ジョブを凍結。再開で続行。",
     "진행을 버리지 않고 실행 중 작업을 멈춥니다. 재개하면 이어집니다."},
    {"Stop the running job first to choose a different image folder or video.",
     "请先停止正在运行的任务，再选择其他图像文件夹或视频。",
     "別の画像フォルダーや動画を選ぶには、先に実行中ジョブを停止。",
     "다른 이미지 폴더나 동영상을 선택하려면 먼저 실행 중인 작업을 중지하십시오."},

    // Crumbs / status
    {"Alignment", "对齐", "アライメント", "정렬"},
    {"3DGS", "3DGS", "3DGS", "3DGS"},
    {"Elapsed %s", "已用时 %s", "経過 %s", "경과 %s"},
    {"%3.0f%%   paused", "%3.0f%%   已暂停", "%3.0f%%   一時停止", "%3.0f%%   일시 중지"},
    {"paused", "已暂停", "一時停止", "일시 중지"},
    {"%3.0f%%   ETA %s", "%3.0f%%   剩余 %s", "%3.0f%%   残り %s", "%3.0f%%   남은 시간 %s"},
    {"working", "进行中", "処理中", "작업 중"},
    {"%s faces", "%s 面", "%s 面", "%s 면"},
    {"%s pts", "%s 点", "%s 点", "%s 점"},
    {"2D Image", "2D 图像", "2D 画像", "2D 이미지"},
    {"Points", "点", "点", "점"},
    {"Paused  |  %s", "已暂停  |  %s", "一時停止  |  %s", "일시 중지  |  %s"},

    // Console
    {"LIVE", "实时", "ライブ", "실시간"},
    {"FAILED", "失败", "失敗", "실패"},
    {"DONE", "完成", "完了", "완료"},
    {"Follow output", "跟随输出", "出力を追従", "출력 따라가기"},
    {"Jump to latest", "跳到最新", "最新へ移動", "최신으로 이동"},
    {"Copy visible lines", "复制可见行", "表示中の行をコピー", "보이는 줄 복사"},
    {"Clear console", "清除控制台", "コンソールをクリア", "콘솔 지우기"},
    {"All", "全部", "すべて", "전체"},
    {"Show every log line", "显示全部日志", "すべてのログ行を表示", "모든 로그 줄 표시"},
    {"Info", "信息", "情報", "정보"},
    {"Info and debug output", "信息与调试输出", "情報とデバッグ出力", "정보 및 디버그 출력"},
    {"Warn", "警告", "警告", "경고"},
    {"Warnings only", "仅警告", "警告のみ", "경고만"},
    {"Error", "错误", "エラー", "오류"},
    {"Errors only", "仅错误", "エラーのみ", "오류만"},
    {"Filter logs", "筛选日志", "ログを絞り込み", "로그 필터"},
    {"Listening for process output...", "正在监听进程输出...", "プロセス出力を待機中...", "프로세스 출력 대기 중..."},
    {"No reconstruction output yet", "暂无重建输出", "再構成出力はまだありません", "아직 재구성 출력이 없습니다"},
    {"Live logs from the active job stream here", "活动任务的实时日志显示在此", "実行中ジョブのライブログがここに流れます", "활성 작업의 실시간 로그가 여기에 표시됩니다"},
    {"Run Align Photos or Train 3DGS to stream logs", "运行对齐照片或训练 3DGS 以输出日志", "写真アライメントまたは 3DGS 学習でログを流す", "사진 정렬 또는 3DGS 학습을 실행해 로그를 확인하세요"},
    {"No matching log lines", "没有匹配的日志", "一致するログ行がありません", "일치하는 로그가 없습니다"},
    {"Clear the filter or search to see all output", "清除筛选或搜索以查看全部输出", "フィルタや検索を解除すると全出力が見えます", "필터나 검색을 지워 전체 출력을 보세요"},
    {"Copy Line", "复制行", "行をコピー", "줄 복사"},
    {"Copy Visible", "复制可见", "表示中をコピー", "보이는 내용 복사"},
    {"Clear Console", "清除控制台", "コンソールをクリア", "콘솔 지우기"},
    {"Training", "训练", "学習", "학습"},
    {"SfM export", "SfM 导出", "SfM 書き出し", "SfM 내보내기"},
    {"Job", "任务", "ジョブ", "작업"},
    {"Idle", "空闲", "アイドル", "대기"},
    {"Extracting features", "提取特征", "特徴抽出中", "특징 추출 중"},
    {"Matching views", "匹配视图", "ビュー照合中", "뷰 매칭 중"},
    {"Building tracks", "构建轨迹", "トラック構築中", "트랙 구축 중"},
    {"Solving camera poses", "求解相机位姿", "カメラ姿勢を解く", "카메라 포즈 계산 중"},
    {"Writing sparse scene", "写入稀疏场景", "疎シーンを書き出し中", "희소 장면 쓰는 중"},
    {"Preparing input", "准备输入", "入力を準備中", "입력 준비 중"},
    {"MVS stereo", "MVS 立体匹配", "MVS ステレオ", "MVS 스테레오"},
    {"Training Gaussians", "训练高斯", "ガウシアン学習中", "가우시안 학습 중"},
    {"Extracting mesh", "提取网格", "メッシュ抽出中", "메시 추출 중"},
    {"Baking texture", "烘焙纹理", "テクスチャベイク中", "텍스처 베이크 중"},
    {"Complete", "完成", "完了", "완료"},
    {"Failed", "失败", "失敗", "실패"},

    // Image QA
    {"Photo", "照片", "写真", "사진"},
    {"Capture image", "采集图像", "撮影画像", "촬영 이미지"},
    {"Detected keypoints and triangulated tracks",
     "检测到的关键点与已三角化轨迹",
     "検出キーポイントと三角測量トラック",
     "검출된 키포인트와 삼각측량 트랙"},
    {"Compare", "对比", "比較", "비교"},
    {"Slide to compare 3DGS against the training view",
     "滑动对比 3DGS 与训练视图",
     "スライドして 3DGS と学習ビューを比較",
     "밀어 3DGS와 학습 뷰를 비교"},
    {"Per-pixel photometric error map", "逐像素光度误差图", "画素ごとの測光誤差マップ", "픽셀별 측광 오차 맵"},
    {"Previous image", "上一张", "前の画像", "이전 이미지"},
    {"Next image", "下一张", "次の画像", "다음 이미지"},
    {"No images to inspect", "没有可检查的图像", "検査する画像がありません", "검사할 이미지가 없습니다"},
    {"Select an image folder or align photos to open the 2D viewer",
     "选择图像文件夹或对齐照片以打开 2D 查看器",
     "画像フォルダーを選ぶか写真をアライメントして 2D ビューアを開く",
     "이미지 폴더를 선택하거나 사진을 정렬해 2D 뷰어를 여세요"},
    {"Loading capture…", "正在加载采集…", "撮影を読み込み中…", "촬영 불러오는 중…"},
    {"Reading the selected image", "正在读取所选图像", "選択画像を読み込み中", "선택한 이미지를 읽는 중"},
    {"Capture unavailable", "采集不可用", "撮影を表示できません", "촬영을 사용할 수 없음"},
    {"This view has no image path", "该视图没有图像路径", "このビューに画像パスがありません", "이 뷰에 이미지 경로가 없습니다"},
    {"Could not decode the selected file", "无法解码所选文件", "選択ファイルを復号できません", "선택한 파일을 디코딩할 수 없습니다"},
    {"Train 3DGS to build an error map", "训练 3DGS 以生成误差图", "誤差マップには 3DGS 学習が必要", "오차 맵을 만들려면 3DGS를 학습하세요"},
    {"The heatmap compares the live splat against this capture",
     "热力图将实时溅射与此采集对比",
     "ヒートマップはライブスプラットとこの撮影を比較します",
     "히트맵은 실시간 스플랫과 이 촬영을 비교합니다"},
    {"Waiting for a rendered frame…", "等待渲染帧…", "描画フレームを待機中…", "렌더 프레임 대기 중…"},
    {"The live splat preview will appear here once it is ready",
     "实时溅射预览就绪后会显示在此",
     "ライブスプラットの準備ができ次第ここに表示",
     "실시간 스플랫 미리보기가 준비되면 여기에 나타납니다"},
    {"Computing error map…", "正在计算误差图…", "誤差マップを計算中…", "오차 맵 계산 중…"},
    {"PSNR / SSIM update as soon as both images are aligned",
     "两张图对齐后即更新 PSNR / SSIM",
     "両画像が揃い次第 PSNR / SSIM を更新",
     "두 이미지가 맞춰지면 PSNR / SSIM이 갱신됩니다"},
    {"Drag to wipe GT / 3DGS    ·    MMB pan    ·    wheel zoom",
     "拖动擦除对比 GT / 3DGS    ·    中键平移    ·    滚轮缩放",
     "ドラッグで GT / 3DGS ワイプ    ·    中パン    ·    ホイールズーム",
     "드래그로 GT / 3DGS 와이프    ·    가운데 이동    ·    휠 확대"},
    {"No splat to compare yet", "尚无可对比的溅射", "比較するスプラットがまだありません", "비교할 스플랫이 아직 없습니다"},
    {"Train 3DGS, then drag the vertical handle to wipe GT vs render",
     "训练 3DGS，然后拖动竖线对比真值与渲染",
     "3DGS を学習し、縦ハンドルで GT とレンダをワイプ",
     "3DGS를 학습한 뒤 세로 핸들로 GT와 렌더를 와이프하세요"},
    {"Waiting for the live splat…", "等待实时溅射…", "ライブスプラットを待機中…", "실시간 스플랫 대기 중…"},
    {"The renderer is snapping to this training camera",
     "渲染器正在对齐到此训练相机",
     "レンダラがこの学習カメラにスナップ中",
     "렌더러가 이 학습 카메라에 맞추는 중입니다"},
    {"No triangulated keypoints in this imported alignment",
     "此导入对齐中没有已三角化关键点",
     "この取り込みアライメントに三角測量キーポイントがありません",
     "가져온 정렬에 삼각측량 키포인트가 없습니다"},
    {"No keypoints in this reconstruction — re-align to inspect features",
     "此重建没有关键点 — 请重新对齐以检查特征",
     "この再構成にキーポイントがありません — 再アライメントして特徴を確認",
     "이 재구성에 키포인트가 없습니다 — 다시 정렬해 특징을 검사하세요"},
    {"Loading imported cameras for the feature overlay…",
     "正在加载导入相机以叠加特征…",
     "特徴オーバーレイ用に取り込みカメラを読み込み中…",
     "특징 오버레이를 위해 가져온 카메라를 불러오는 중…"},

    // Gizmo
    {"Drag arrows to move the reconstruction region\n"
     "Shift: fine  ·  Ctrl: snap",
     "拖动箭头移动重建区域\nShift：微调  ·  Ctrl：吸附",
     "矢印をドラッグして再構成領域を移動\nShift: 微調整  ·  Ctrl: スナップ",
     "화살표를 드래그해 재구성 영역을 이동\nShift: 미세  ·  Ctrl: 스냅"},
    {"Drag a face handle to resize the reconstruction region\n"
     "Shift: fine  ·  Ctrl: snap",
     "拖动面手柄缩放重建区域\nShift：微调  ·  Ctrl：吸附",
     "面ハンドルをドラッグして再構成領域をリサイズ\nShift: 微調整  ·  Ctrl: スナップ",
     "면 핸들을 드래그해 재구성 영역 크기 조절\nShift: 미세  ·  Ctrl: 스냅"},
    {"Click an axis to change camera view", "点击坐标轴切换相机视角", "軸をクリックしてカメラ視点を変更", "축을 클릭해 카메라 시점을 바꿉니다"},

    // Window title / messages
    {"AetherScan Reconstruction Editor", "AetherScan 重建编辑器", "AetherScan 再構成エディタ", "AetherScan 재구성 편집기"},
    {"New project", "新项目", "新規プロジェクト", "새 프로젝트"},
    {"Image dataset selected; previous viewport result cleared",
     "已选择图像数据集；已清除先前视口结果",
     "画像データセットを選択。前回のビューポート結果をクリア",
     "이미지 데이터셋을 선택했습니다. 이전 뷰포트 결과를 지웠습니다"},
    {"Video selected; Align Photos will extract sharp frames, then run SfM",
     "已选择视频；对齐照片将提取清晰帧并运行 SfM",
     "動画を選択。写真アライメントが鮮明フレームを抽出し SfM を実行",
     "동영상을 선택했습니다. 사진 정렬이 선명 프레임을 추출한 뒤 SfM을 실행합니다"},
    {"Stop the running job first to change images or video",
     "请先停止正在运行的任务再更改图像或视频",
     "画像や動画を変えるには先に実行中ジョブを停止",
     "이미지나 동영상을 바꾸려면 먼저 실행 중인 작업을 중지하십시오"},
    {"Drop a single video file", "请拖入单个视频文件", "動画ファイルは 1 つだけドロップ", "동영상 파일은 하나만 드롭하세요"},
    {"Drop a single video, or a folder of photos",
     "请拖入单个视频或一个照片文件夹",
     "動画 1 つ、または写真フォルダーをドロップ",
     "동영상 하나 또는 사진 폴더를 드롭하세요"},
    {"Drop a single .ascan project file", "请拖入单个 .ascan 项目文件", ".ascan プロジェクトは 1 つだけドロップ", ".ascan 프로젝트 파일은 하나만 드롭하세요"},
    {"Drop a single .asfm file", "请拖入单个 .asfm 文件", ".asfm は 1 つだけドロップ", ".asfm 파일은 하나만 드롭하세요"},
    {"Drop photos, a video, an .asfm scene, or an .ascan project on the viewport",
     "将照片、视频、.asfm 场景或 .ascan 项目拖到视口",
     "写真、動画、.asfm シーン、.ascan プロジェクトをビューポートへ",
     "사진, 동영상, .asfm 장면 또는 .ascan 프로젝트를 뷰포트에 드롭하세요"},
    {"No mesh found to preview", "没有可预览的网格", "プレビューするメッシュがありません", "미리볼 메시가 없습니다"},
    {"Exported splat", "已导出高斯溅射", "スプラットを書き出しました", "스플랫을 내보냈습니다"},
    {"Exported mesh (OBJ + MTL + albedo PNG)", "已导出网格（OBJ + MTL + 反照率 PNG）", "メッシュを書き出し（OBJ + MTL + アルベド PNG）", "메시를 내보냄(OBJ + MTL + 알베도 PNG)"},
    {"Exported mesh (GLB with albedo)", "已导出网格（含反照率的 GLB）", "メッシュを書き出し（アルベド付き GLB）", "메시를 내보냄(알베도 포함 GLB)"},
    {"Exported mesh", "已导出网格", "メッシュを書き出しました", "메시를 내보냈습니다"},
    {"Project opened; no SfM stage yet", "已打开项目；尚无 SfM 阶段", "プロジェクトを開きました。まだ SfM 段階はありません", "프로젝트를 열었습니다. 아직 SfM 단계가 없습니다"},
    {"Cannot open a project while a job is running",
     "任务运行时无法打开项目",
     "ジョブ実行中はプロジェクトを開けません",
     "작업 실행 중에는 프로젝트를 열 수 없습니다"},
    {"Cannot open SfM while a job is running",
     "任务运行时无法打开 SfM",
     "ジョブ実行中は SfM を開けません",
     "작업 실행 중에는 SfM을 열 수 없습니다"},
    {"Project saved", "项目已保存", "プロジェクトを保存しました", "프로젝트를 저장했습니다"},
    {"Save or choose a project file first", "请先保存或选择项目文件", "先にプロジェクトファイルを保存または選択", "먼저 프로젝트 파일을 저장하거나 선택하세요"},
    {"Cannot create project directory", "无法创建项目目录", "プロジェクトディレクトリを作成できません", "프로젝트 디렉터리를 만들 수 없습니다"},
    {"Align photos before exporting SfM alignment",
     "导出 SfM 对齐前请先对齐照片",
     "SfM アライメント書き出しの前に写真をアライメント",
     "SfM 정렬을 내보내기 전에 사진을 정렬하세요"},
    {"Video file not found", "未找到视频文件", "動画ファイルが見つかりません", "동영상 파일을 찾을 수 없습니다"},
    {"No images found in the selected source folder",
     "所选源文件夹中没有图像",
     "選択ソースフォルダーに画像がありません",
     "선택한 원본 폴더에 이미지가 없습니다"},
    {"Aligning cameras...", "正在对齐相机...", "カメラをアライメント中...", "카메라 정렬 중..."},
    {"External dataset selected; use Train 3DGS instead of Align Photos",
     "已选择外部数据集；请使用训练 3DGS 而非对齐照片",
     "外部データセット選択済み。写真アライメントではなく 3DGS 学習を使用",
     "외부 데이터셋이 선택되었습니다. 사진 정렬 대신 3DGS 학습을 사용하세요"},
    {"External cameras loaded. Align Photos is skipped; Train 3DGS or "
     "Extract Mesh can run next.",
     "已加载外部相机。跳过对齐照片；接下来可训练 3DGS 或提取网格。",
     "外部カメラを読み込みました。写真アライメントはスキップ。次は 3DGS 学習またはメッシュ抽出。",
     "외부 카메라를 불러왔습니다. 사진 정렬은 건너뜁니다. 다음으로 3DGS 학습 또는 메시 추출을 실행할 수 있습니다."},
};

void ensure_maps() {
    if (g_ready) return;
    g_zh.reserve(std::size(k_entries));
    g_ja.reserve(std::size(k_entries));
    g_ko.reserve(std::size(k_entries));
    for (const Entry& entry : k_entries) {
        g_zh.emplace(entry.en, entry.zh);
        g_ja.emplace(entry.en, entry.ja);
        g_ko.emplace(entry.en, entry.ko);
    }
    g_ready = true;
}

const std::unordered_map<std::string_view, const char*>* active_map() {
    ensure_maps();
    switch (g_language) {
        case Language::zh_cn: return &g_zh;
        case Language::ja: return &g_ja;
        case Language::ko: return &g_ko;
        case Language::en:
        default: return nullptr;
    }
}

}  // namespace

Language language() {
    return g_language;
}

void set_language(const Language language) {
    g_language = language;
    if (!g_path.empty()) save(g_path);
}

const char* code(const Language language) {
    switch (language) {
        case Language::zh_cn: return "zh-CN";
        case Language::ja: return "ja";
        case Language::ko: return "ko";
        case Language::en:
        default: return "en";
    }
}

const char* native_name(const Language language) {
    switch (language) {
        case Language::zh_cn: return "简体中文";
        case Language::ja: return "日本語";
        case Language::ko: return "한국어";
        case Language::en:
        default: return "English";
    }
}

void load(const std::filesystem::path& path) {
    g_path = path;
    g_language = Language::en;
    std::ifstream in(path);
    if (!in) return;
    std::string token;
    in >> token;
    if (token == "zh-CN" || token == "zh" || token == "zh_cn")
        g_language = Language::zh_cn;
    else if (token == "ja" || token == "jp")
        g_language = Language::ja;
    else if (token == "ko" || token == "kr")
        g_language = Language::ko;
}

void save(const std::filesystem::path& path) {
    g_path = path;
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream out(path, std::ios::trunc);
    if (!out) return;
    out << code(g_language) << '\n';
}

const char* tr(const char* english) {
    if (english == nullptr || english[0] == '\0' || g_language == Language::en)
        return english == nullptr ? "" : english;
    const auto* map = active_map();
    if (map == nullptr) return english;
    const auto found = map->find(english);
    return found == map->end() ? english : found->second;
}

const char* id(const char* english, const char* imgui_suffix) {
    static thread_local char buffers[24][256];
    static thread_local int next{};
    char* out = buffers[next++ % 24];
    const char* suffix = imgui_suffix == nullptr ? "" : imgui_suffix;
    std::snprintf(out, 256, "%s%s", tr(english), suffix);
    return out;
}

}  // namespace editor::i18n

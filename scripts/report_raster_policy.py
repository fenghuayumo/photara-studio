"""Build the reviewable report and fixed-view scientific comparison figures."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS

LABELS = {'legacy':'Legacy','sort_z':'Z sort','alpha_999':'Alpha .999',
          'near_001':'Near .01','pixel_half':'Half pixel','brush_legacy':'Brush numerics',
          'brush_classic':'Brush + Classic','brush_mip2d':'Brush + Mip 2D',
          'brush_mip3d':'Brush + Mip 3D','brush_mip':'2D+3D (extra ablation)'}


def table(rows, columns):
    result = ['| '+' | '.join(label for label,_,_ in columns)+' |',
              '| '+' | '.join('---' for _ in columns)+' |']
    for row in rows:
        cells=[]
        for _, key, fmt in columns:
            value=row.get(key)
            cells.append('—' if value is None else (fmt(value) if callable(fmt) else format(value,fmt)))
        result.append('| '+' | '.join(cells)+' |')
    return result


def figures(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    native=[r for r in rows if r['renderer']=='audit']
    for experiment in dict.fromkeys(r['experiment'] for r in native):
        subset=[r for r in native if r['experiment']==experiment]
        x=np.arange(len(subset))
        fig, axes=plt.subplots(1,2,figsize=(max(10,len(x)*1.4),4.8),layout='constrained')
        for ax,key,title in [(axes[0],'audit_valid_psnr','Held-out valid-region PSNR (higher is better)'),
                             (axes[1],'lpips_squeeze_valid','Spatial LPIPS-Squeeze (lower is better)')]:
            values=[r[key] for r in subset]
            ax.bar(x,values,color=['#233b53' if r['variant']=='legacy' else '#3b928d' for r in subset])
            ax.axhline(values[0],color='#d07b44',linestyle='--',linewidth=1)
            ax.set_xticks(x,[LABELS[r['variant']] for r in subset],rotation=35,ha='right',fontsize=9)
            ax.set_title(title,fontsize=11)
            margin=max(np.ptp(values)*.2, .1 if key=='audit_valid_psnr' else .002)
            ax.set_ylim(min(values)-margin,max(values)+margin)
            ax.spines[['top','right']].set_visible(False)
        fig.savefig(output/(experiment+'_quality.png'),dpi=160)
        plt.close(fig)
    # Views and crop are fixed before viewing outcomes. Avoid selecting a best-looking patch.
    long=[r for r in native if r['iterations']>=30000]
    if not long: return
    # One seed per qualitative sheet; repeated seeds remain in the numerical table.
    long=[r for r in long if r['seed']==min(v['seed'] for v in long)]
    preferred=['legacy','brush_classic','brush_mip2d','brush_mip3d']
    chosen=[r for name in preferred for r in long if r['variant']==name]
    chosen=chosen[:5]
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(font_path),20) if font_path.exists() else ImageFont.load_default()
    for view in (0,40):
        cols=len(chosen)+1
        canvas=Image.new('RGB',(cols*320,720),'white'); draw=ImageDraw.Draw(canvas)
        first=Path(chosen[0]['folder'])/'audit'
        paths=[first/('target_'+str(view)+'.png')]+[Path(r['folder'])/'audit'/('render_'+str(view)+'.png') for r in chosen]
        labels=['Reference']+[LABELS[r['variant']] for r in chosen]
        for column,(path,label) in enumerate(zip(paths,labels)):
            original=Image.open(path).convert('RGB')
            canvas.paste(original.resize((310,310),LANCZOS),(column*320+5,40))
            canvas.paste(original.crop((250,150,650,550)).resize((310,310),LANCZOS),(column*320+5,390))
            draw.text((column*320+8,8),label,font=font,fill='#183449')
        draw.text((8,360),'Fixed crop: x=250..650, y=150..550',font=font,fill='#183449')
        canvas.save(output/('appearance_view_'+str(view)+'.png'))
        mesh_paths=[Path(r['folder'])/'audit'/('mesh_'+str(view)+'.png') for r in chosen]
        if all(p.exists() for p in mesh_paths):
            canvas=Image.new('RGB',(len(chosen)*400,450),'white'); draw=ImageDraw.Draw(canvas)
            for column,(path,row) in enumerate(zip(mesh_paths,chosen)):
                canvas.paste(Image.open(path).convert('RGB').resize((390,390),LANCZOS),(column*400+5,45))
                draw.text((column*400+8,10),LABELS[row['variant']],font=font,fill='#183449')
            canvas.save(output/('mesh_view_'+str(view)+'.png'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--multiscale',type=Path)
    parser.add_argument('--fixed-forward',type=Path)
    args=parser.parse_args()
    rows=json.loads(args.results.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    figures(rows,args.output)
    lines=['# ORI 光栅化与 Mip 过滤对照', '',
           '输入：用户生成的 `D:/ScanData/ori_img/images.cache/sfm.bin`，76 个注册相机、121129 个稀疏点。'
           '所有变体共享相机、初始化、ADC-IGS、种子、损失和 Gaussian 上限。每八张留出一张：66 张训练、10 张评测。', '',
           '硬件：RTX 3060 Laptop 6GB。短程为 8000 步、每 1000 步升一级分辨率；'
           '长程为 30000 步、每 3000 步升一级。最终均为原生 1000×1000。'
           '训练不启用深度法线/多视图几何损失；TSDF 提取单独运行。Speedy 剪枝关闭。', '',
           '这些是 Photara 内部对齐 Brush 指定数值的消融，不是两个软件的速度比赛。'
           'Classic=0.3 方差且不补偿，Mip2D=0.1 方差且补偿，Mip3D 使用逐相机距离/焦距下限。'
           'Mip3D-only 保留 Classic 的 0.3 屏幕膨胀。Legacy 保留旧的零核/自动 3D 策略。', '',
           '有效区 PSNR、SSIM 排除去畸变后没有源图像的像素；LPIPS 是 SqueezeNet 空间图在有效区的均值。'
           '相机和稀疏点由完整图像集得到，因此是共享 SfM 初始化下的留出评测。没有真值网格。', '']
    for experiment in dict.fromkeys(r['experiment'] for r in rows):
        native=[dict(r,label=LABELS[r['variant']]) for r in rows if r['experiment']==experiment and r['renderer']=='audit']
        lines+=['## '+experiment,'']
        lines+=table(native,[('配置','label',''),('种子','seed','d'),('步数','iterations','d'),
            ('有效区 PSNR ↑','audit_valid_psnr','.3f'),('SSIM ↑','audit_ssim','.5f'),
            ('LPIPS ↓','lpips_squeeze_valid','.5f'),('训练循环秒','optimization_s','.2f'),
            ('Gaussian 数','gaussians',lambda v:str(int(v)))])
        lines+=['',f'![质量对照]({experiment}_quality.png)','']
    lines+=['## 几何与渲染诊断','',
            '深度一致性：10 个固定留出相机，每个与空间最近的三个留出相机比较，每四像素采样一次，'
            '要求两端 alpha≥0.5 且源图像有效。表中为所有有效投影的相对深度误差中位数；'
            '遮挡也会贡献误差，不能作为真实表面精度。深度法线角是深度有限差分法线与 Gaussian 法线的无向夹角。', '']
    long=[dict(r,label=LABELS[r['variant']]) for r in rows if r['iterations']>=30000 and r['renderer']=='audit']
    subset=long or [dict(r,label=LABELS[r['variant']]) for r in rows if r['renderer']=='audit']
    lines+=table(subset,[('配置','label',''),('RGB forward ms','rgb_forward_ms','.3f'),
        ('几何 forward ms','geometry_forward_ms','.3f'),('深度覆盖率','depth_coverage',lambda v:f'{v:.2%}'),
        ('跨视图相对深度误差','multiview_relative_median',lambda v:f'{v:.3%}'),
        ('深度法线角中位数°','depth_normal_median_deg','.2f'),
        ('近相机 Gaussian 数','near_camera_5pct_extent','d')])
    lines+=['','近相机统计使用距任意注册相机（全部 76 台）小于 0.05×相机 extent、有效 opacity≥0.01 的 Gaussian；'
            '这不是漂浮物真值标签。渲染时间不含图像加载、读回与 PNG 编码；'
            'RGB 和几何通道分别计时，三个固定相机，每相机预热两次、测五次。', '']
    meshes=[dict(r,label=LABELS[r['variant']]) for r in rows if 'mesh_faces' in r and r['renderer']=='audit']
    if meshes:
        lines+=table(meshes,[('配置','label',''),('顶点','mesh_vertices','d'),('面','mesh_faces','d'),
            ('边界边','mesh_boundary_edges','d'),('非流形边','mesh_nonmanifold_edges','d'),
            ('面积（场景单位²）','mesh_area','.2f'),('提取秒','mesh_seconds','.2f')])
        lines+=['','TSDF 使用相同相机、自动 voxel=2×camera extent/2048、截断四个 voxel、alpha 阈值 0.5，'
                '保留默认最大连通分量、闭孔和平滑。面积、面数与拓扑不等于准确度；'
                '更少的面或更平滑的网格不能单独证明更好。','']
    for view in (0,40):
        for kind in ('appearance','mesh'):
            name=f'{kind}_view_{view}.png'
            if (args.output/name).exists(): lines += [f'![{kind} view {view}]({name})','']
    common=[dict(r,label=LABELS[r['variant']]) for r in rows if r['renderer']=='audit_common']
    if common:
        lines+=['## 统一渲染器复核','',
                '以下将所有模型改用同一个 Legacy 屏幕渲染器，保留模型文件中的 3D floor。'
                '用于区分训练得到的表示变化与展示时的 2D/排序设置变化；不代表 Mip 模型的推荐展示方式。','']
        lines+=table(common,[('配置','label',''),('有效区 PSNR','audit_valid_psnr','.3f'),
                             ('SSIM','audit_ssim','.5f'),('LPIPS','lpips_squeeze_valid','.5f')])
    if args.multiscale:
        scale_rows=json.loads(args.multiscale.read_text())
        for row in scale_rows:
            row['label']=LABELS[row['variant'].rsplit('_',1)[0]]
        lines+=['','## 缩小渲染','',
                '同一模型直接以 1000、500、250 像素宽度渲染，与留出照片的 Lanczos 缩小结果比较。'
                '相机按像素中心一致的 resize 公式缩放；3D floor 保留训练值。'
                '有效区 mask 缩小后要求 >0.999。此表基于 8-bit PNG，因此 1000 档与原生浮点评测存在量化差别。','']
        lines+=table(scale_rows,[('配置','label',''),('宽度','width','d'),
                                ('有效区 PSNR','valid_psnr','.3f'),('SSIM','ssim','.5f')])
    if args.fixed_forward:
        fixed=json.loads(args.fixed_forward.read_text())
        measurements=fixed['measurements']
        timing=[]
        for variant in dict.fromkeys(m['variant'] for m in measurements):
            row={'label':LABELS[variant]}
            for geometry,key in ((False,'rgb'),(True,'geometry')):
                subset=[m for m in measurements if m['variant']==variant and m['geometry']==geometry]
                row[key]=float(np.median([m['median_ms'] for m in subset]))
            timing.append(row)
        lines+=['','## 固定模型光栅化耗时','',
                f"固定长程 Legacy 模型（{fixed['gaussians']} 个 Gaussian），比较各开关的直接开销。"
                '两轮随机配置顺序、三个相机，每组预热 20 次、同步测量 50 次；表格为六组中位数的中位数。'
                'Mip3D 需要训练时构建 floor，未对这个无 floor 的固定模型伪造 3D 结果。','']
        lines+=table(timing,[('配置','label',''),('RGB ms','rgb','.3f'),('几何 ms','geometry','.3f')])
    lines+=['','## 证据与限制','',
            '- 原始命令、输入与可执行文件 SHA256、日志、模型、每视图指标与原始深度/法线均在实验目录。',
            '- 单一种子结果不能证明稳定收益。短程不能替代收敛结果；笔记本温度/共享功耗会影响计时。',
            '- 短程有少量 CPU 指标计算与训练重叠，短程训练时间仅作描述；性能结论以串行长程与统一复测为准。',
            '- 本实验没有加入深度相关致密化梯度缩放，也没有完整移植 Brush 的训练/导出生命周期。','']
    (args.output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__': main()

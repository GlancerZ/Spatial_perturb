"""Render the proposed dual-CLS training graph, without running model code."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dual_cls_mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/figures'
FONT = FontProperties(fname='/tmp/dual_cls_NotoSansCJKsc-Regular.otf')
W, H = 1600, 2110
INK, MUTED = '#172B42', '#5A6B7D'
PURPLE, TEAL, ORANGE = '#7054B2', '#168777', '#BD731E'
BLUE, LABEL = '#426C99', '#9B795A'
fig, ax = plt.subplots(figsize=(16, 21.1), dpi=140)
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')
fig.patch.set_facecolor('white')
text_artists = []

def text(x, y, s, size=16, color=INK, ha='center', va='center', weight=None):
    t = ax.text(x, y, s, fontsize=size, fontproperties=FONT, color=color,
                ha=ha, va=va, linespacing=1.55, zorder=5)
    text_artists.append(t)
    return t

def box(x, y, w, h, s='', color=BLUE, fill='#F2F6FA', size=16, lw=1.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0,rounding_size=12', linewidth=lw,
        edgecolor=color, facecolor=fill, zorder=2))
    if s:
        return text(x+w/2, y+h/2, s, size=size)

def arrow(points, color=MUTED, dashed=False, lw=1.6, both=False):
    style = '<->' if both else '-|>'
    for a,b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]], [a[1],b[1]], color=color, lw=lw,
                linestyle=(0,(5,4)) if dashed else '-', zorder=1)
    a,b = points[-2],points[-1]
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=16,
        linewidth=lw,linestyle=(0,(5,4)) if dashed else '-',color=color,
        shrinkA=0,shrinkB=2,zorder=3))

def section(y, number, title):
    text(88, y, number, size=16, color=BLUE, ha='left')
    heading = text(143, y, title, size=19, ha='left')
    heading.set_bbox(dict(facecolor='white', edgecolor='none', pad=2))

text(80, 49, '双区室 CLS：核—胞质交互与空间表达重建', size=29, ha='left')
text(82, 101, '下一版完整流程 · 架构设计，尚未实现 / 尚未切换训练', size=16, color=MUTED, ha='left')
for x,c,s in [(91,PURPLE,'细胞核'),(282,TEAL,'细胞质'),(473,ORANGE,'跨细胞交互')]:
    ax.add_patch(Circle((x,143),7,color=c))
    text(x+21,143,s,size=14,ha='left')
arrow([(745,143),(793,143)],color=LABEL,dashed=True)
text(809,143,'监督标签路径',size=14,ha='left',color=LABEL)

section(198, '01', '输入 → 共同分箱 → 遮盖')
box(90,230,800,78,'中心细胞及邻居：核计数 N  /  胞质计数 C\n每个细胞分别处理；基因没有固定顺序',size=16)
arrow([(490,308),(490,338)])
box(90,340,800,88,'每个细胞：核 + 胞质共用一张 51-bin 表\n真实零 = 0；正表达 = 1–50；先分箱，再 mask',size=16)
arrow([(890,384),(1098,384)],color=LABEL,dashed=True)
box(1100,331,410,112,'完整 target_bins\n保留为监督标签\n目标张量不进入编码器',color=LABEL,fill='#FBF6F0',size=15)
arrow([(1510,387),(1560,387),(1560,1035),(1512,1035)],color=LABEL,dashed=True)
arrow([(1560,1035),(1560,1760),(1512,1760)],color=LABEL,dashed=True)
arrow([(490,428),(490,458)])
box(90,460,800,112,'中心细胞遮盖；任务比例：70% 配对基因 / 20% 区室 / 10% 整细胞\n配对模式：核、胞质同步隐藏 50% 的 panel 基因\n遮盖后不重新分箱',size=14)
text(1305,500,'邻居使用各自细胞的 bin 表\n已遮盖基因从可见序列移除',size=14,color=MUTED)

section(603, '02', '两条独立序列，共享普通 scGPT 权重')
arrow([(490,572),(490,627),(275,627),(275,648)])
arrow([(490,627),(705,627),(705,648)])
box(90,650,370,62,'[原版 CLS] + 核 gene / bin',color=PURPLE,fill='#F2EFF9',size=15)
box(520,650,370,62,'[原版 CLS] + 质 gene / bin',color=TEAL,fill='#EDF7F4',size=15)
arrow([(275,712),(275,743)],color=PURPLE)
arrow([(705,712),(705,743)],color=TEAL)
box(75,733,830,124,'',color='#BAC8D5',fill='#FAFBFD',lw=1)
box(90,746,370,64,'普通 scGPT 编码器',color=PURPLE,fill='#F2EFF9',size=17)
box(520,746,370,64,'普通 scGPT 编码器',color=TEAL,fill='#EDF7F4',size=17)
text(490,777,'=',size=22,color=BLUE)
text(490,835,'同一套权重；各自取 CLS；此阶段不输入坐标',size=13,color=MUTED)
arrow([(275,857),(275,882)],color=PURPLE)
arrow([(705,857),(705,882)],color=TEAL)
box(75,884,830,104,'',color='#C6D1DB',fill='white',lw=1)
box(90,898,370,68,'初始核 CLS：N0  ·  512 维',color=PURPLE,fill='#F2EFF9',size=16)
box(520,898,370,68,'初始胞质 CLS：C0  ·  512 维',color=TEAL,fill='#EDF7F4',size=16)

text(1305,621,'右侧：仅训练时使用的重建监督',size=16,color=MUTED)
arrow([(905,931),(998,931),(998,774),(1098,774)],color=BLUE)
box(1100,711,410,126,'区室重建头\n分别读取 N0 / C0\n另给 query 基因 ID + 区室 ID',size=14)
arrow([(1305,837),(1305,866)],color=BLUE)
box(1100,868,410,89,'核、胞质各自预测\n非零概率 + 正值相对等级',size=15)
arrow([(1305,957),(1305,988)],color=BLUE)
box(1100,990,410,90,'L_comp：区室重建\nBCE + 归一化 rank MSE',color=LABEL,fill='#FBF6F0',size=15)

section(1040, '03', '显式交替交互：所有 patch 细胞并行执行')
arrow([(490,988),(490,1075)])

def local(y,num,subtitle):
    box(90,y,800,82,f'Local {num}  ·  同一细胞：核 CLS ↔ 胞质 CLS\n{subtitle}',
        color=BLUE,fill='#F0F4FA',size=16)

def spatial(y,num):
    box(90,y,800,82,f'Spatial {num}  ·  胞质 C_i ↔ 邻居胞质 C_j\n只更新胞质；Q = C_i，K / V = 邻居 C_j；核状态保留',
        color=ORANGE,fill='#FFF6E9',size=15)

local(1078,0,'两个 token 均可关注自己和对方：Self-attention + FFN')
arrow([(490,1160),(490,1187)])
spatial(1190,1)
arrow([(490,1272),(490,1299)])
local(1302,1,'胞质把邻域信息反馈给核，并准备下一轮通信')
arrow([(490,1384),(490,1411)])
spatial(1414,2)
arrow([(490,1496),(490,1523)])
local(1526,2,'再次双向交换：得到核、胞质两个上下文状态')

box(1100,1150,410,152,'空间图与相对位置\n64 细胞 patch / 16 邻居\n相对距离只在 Spatial 中加入\n排除 self、重复及无效发送者',
    color=ORANGE,fill='#FFF6E9',size=14)
arrow([(1100,1206),(1008,1206),(1008,1231),(892,1231)],color=ORANGE)
arrow([(1100,1261),(1008,1261),(1008,1455),(892,1455)],color=ORANGE)
text(1305,1366,'核可经本细胞胞质间接影响邻居\n也可间接接收邻域信息\n直接跨细胞连接始终只有胞质',size=14,color=MUTED)

section(1665,'04','保留双区室输出 → 联合表征')
arrow([(490,1608),(490,1698)])
box(90,1700,800,106,'最终核 CLS N_final  +  最终胞质 CLS C_final\n各自 L2 归一化，再拼接 → joint 1024 维',
    color=BLUE,fill='#EDF3FA',size=18)
arrow([(890,1753),(972,1753),(972,1543),(1098,1543)],color=BLUE)
box(1100,1488,410,109,'联合重建头\n读取最终 joint\n另给 query 基因 ID + 区室 ID',size=14)
arrow([(1305,1597),(1305,1623)],color=BLUE)
box(1100,1625,410,64,'预测中心核 / 胞质的表达 bins',size=14)
arrow([(1305,1689),(1305,1714)],color=BLUE)
box(1100,1716,410,90,'L_joint：联合重建\nBCE + 归一化 rank MSE',color=LABEL,fill='#FBF6F0',size=15)
box(1100,1831,410,69,'总损失 = L_comp + L_joint',color=LABEL,fill='#FBF6F0',size=15)
text(490,1853,'推理：不做 mask，输出 joint\n空间对照 intrinsic：相同 Local，关闭全部空间消息',size=15,color=BLUE)

ax.plot([90,1510],[1930,1930],color='#D6DFE7',lw=1)
text(90,1965,'训练边界',size=15,color=INK,ha='left')
text(235,1965,'区室全遮盖时跳过其 L_comp；整细胞遮盖只计算 L_joint。',size=14,color=MUTED,ha='left')
text(90,2008,'连接边界',size=15,color=INK,ha='left')
text(235,2008,'无可见胞质可接收、不可发送；新增交互模块的残差输出零初始化。',size=14,color=MUTED,ha='left')
text(90,2051,'分箱定义',size=15,color=INK,ha='left')
text(235,2051,'完整表达统计参与分箱；重建的是相对等级，不是原始绝对计数。',size=14,color=MUTED,ha='left')

OUT.mkdir(parents=True,exist_ok=True)
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
for artist in text_artists:
    bb=artist.get_window_extent(renderer)
    if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > fig.bbox.width+1 or bb.y1 > fig.bbox.height+1:
        raise RuntimeError(f'Text outside figure: {artist.get_text()}')
for ext in ('png','svg','pdf'):
    path=OUT/f'dual_cls_binned_v3_full_flowchart.zh.{ext}'
    fig.savefig(path,dpi=150,facecolor='white')
    print(path)
plt.close(fig)

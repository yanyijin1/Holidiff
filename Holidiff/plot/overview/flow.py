
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 关闭外部LaTeX，使用matplotlib原生mathtext（支持$...$公式）
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# 马卡龙配色
COLORS = {
    'bg': '#FAF9F6',
    'input': '#FFD6E0',
    'lstde': '#C1E1C1',
    'sfcn': '#B5D8EB',
    'denoiser': '#FFF4BD',
    'residual': '#E2C3F0',
    'dca': '#FFDAC1',
    'output': '#FFB7B2',
    'arrow': '#A0A0A0',
    'text': '#4A4A4A',
    'loop': '#C1E1C1',
    'field_box': '#E8F4F8',
    'trend_box': '#F3E5F5',
}

fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['bg'])
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_facecolor(COLORS['bg'])

# 辅助函数：绘制分组模块
def draw_grouped_box(ax, x, y, w, h, color, label, math_text, gid_group, fontsize=13):
    # 外框（带分组ID）
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=color, edgecolor='white', linewidth=2.5, alpha=0.95, zorder=3,
                         gid=gid_group)
    ax.add_patch(box)
    
    # LaTeX公式文字
    ax.text(x, y+0.15, math_text, ha='center', va='center', fontsize=fontsize,
            color=COLORS['text'], zorder=4, fontweight='bold')
    # 英文标签
    ax.text(x, y-0.25, label, ha='center', va='center', fontsize=fontsize-2,
            color=COLORS['text'], zorder=4, style='italic')
    return box

def draw_arrow(ax, x1, y1, x2, y2, ls='-', alpha=1.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2.2,
                                linestyle=ls, alpha=alpha))

# ==================== 主流程模块（分组命名） ====================

# 1. Input
draw_grouped_box(ax, 1.2, 4, 1.8, 1.2, COLORS['input'], 'Historical Flow',
                 r'$\mathbf{X} \in \mathbb{R}^{B \times N \times L}$', 'group_input', 12)

# 2. LSTDE
draw_grouped_box(ax, 3.5, 4, 2.0, 1.4, COLORS['lstde'], 'LSTDE',
                 r'Multi-scale Frequency' + '\n' + r'Decoupling Embed.', 'group_lstde', 12)

# 3. SFCN
draw_grouped_box(ax, 6.2, 4, 2.0, 1.4, COLORS['sfcn'], 'SFCN',
                 r'Spatial Field-Coupled' + '\n' + r'Normalization', 'group_sfcn', 12)

# 4. Denoiser
draw_grouped_box(ax, 9.0, 4, 2.2, 1.4, COLORS['denoiser'], 'Denoiser',
                 r'Conditional Diffusion' + '\n' + r'Sampling', 'group_denoiser', 12)

# 5. Residual
draw_grouped_box(ax, 11.8, 4, 1.8, 1.2, COLORS['residual'], 'Trend Residual',
                 r'$\eta \cdot \mathbf{s}^{\mathrm{hist}} \otimes \boldsymbol{\tau}$', 'group_residual', 12)

# 6. DCA
draw_grouped_box(ax, 9.0, 1.5, 2.2, 1.2, COLORS['dca'], 'DCA',
                 r'Density-Centroid' + '\n' + r'Aggregation', 'group_dca', 12)

# 7. Output
draw_grouped_box(ax, 12.5, 1.5, 2.0, 1.2, COLORS['output'], 'Macro Estimate',
                 r'$\hat{\mathbf{Y}}_{\mathrm{macro}}$', 'group_output', 13)

# ==================== 箭头连接 ====================

draw_arrow(ax, 2.1, 4, 2.5, 4)
draw_arrow(ax, 4.5, 4, 5.2, 4)
draw_arrow(ax, 7.2, 4, 7.9, 4)
draw_arrow(ax, 10.1, 4, 10.9, 4)

# Residual -> DCA (折线)
ax.plot([11.8, 11.8], [3.4, 2.5], color=COLORS['arrow'], lw=2.2)
ax.plot([11.8, 11.8], [2.5, 2.1], color=COLORS['arrow'], lw=2.2)
ax.annotate('', xy=(10.1, 1.5), xytext=(11.8, 2.1),
            arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=2.2,
                            connectionstyle="arc3,rad=0.3"))

draw_arrow(ax, 10.1, 1.5, 11.5, 1.5)

# ==================== 扩散采样循环（分组） ====================

loop_box = FancyBboxPatch((7.8, 3.1), 2.4, 1.8,
                          boxstyle="round,pad=0.02,rounding_size=0.2",
                          facecolor='none', edgecolor=COLORS['loop'], linewidth=2.5,
                          linestyle='--', alpha=0.8, zorder=2, gid='group_loop')
ax.add_patch(loop_box)

ax.text(9.0, 5.15, r'$S$ Micro-realizations', ha='center', va='center',
        fontsize=11, color=COLORS['text'], style='italic', zorder=4)
ax.annotate('', xy=(9.0, 5.0), xytext=(9.0, 4.7),
            arrowprops=dict(arrowstyle='->', color=COLORS['loop'], lw=1.5))

# ==================== 场统计量准备（分组） ====================

field_box = FancyBboxPatch((5.5, 5.8), 1.4, 0.9,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['field_box'], edgecolor=COLORS['sfcn'], linewidth=2,
                           zorder=3, gid='group_field_stats')
ax.add_patch(field_box)
ax.text(6.2, 6.25, r'$\Theta_{\mathrm{field}}$', ha='center', va='center',
        fontsize=12, color=COLORS['text'], fontweight='bold', zorder=4)
ax.text(6.2, 5.95, 'Field Stats', ha='center', va='center',
        fontsize=9, color=COLORS['text'], zorder=4)

# X -> Field Stats (虚线)
ax.plot([1.2, 1.2], [4.6, 6.25], color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.6)
ax.plot([1.2, 5.5], [6.25, 6.25], color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.6)
ax.annotate('', xy=(5.5, 6.25), xytext=(4.8, 6.25),
            arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.6))

# Field Stats -> SFCN
draw_arrow(ax, 6.2, 5.8, 6.2, 4.7, ls='--', alpha=0.6)

# ==================== 历史趋势提取（分组） ====================

trend_box = FancyBboxPatch((0.3, 5.8), 1.8, 0.9,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=COLORS['trend_box'], edgecolor=COLORS['residual'], linewidth=2,
                           zorder=3, gid='group_trend')
ax.add_patch(trend_box)
ax.text(1.2, 6.25, r'$\mathbf{s}^{\mathrm{hist}}$', ha='center', va='center',
        fontsize=12, color=COLORS['text'], fontweight='bold', zorder=4)
ax.text(1.2, 5.95, 'Trend Slope', ha='center', va='center',
        fontsize=9, color=COLORS['text'], zorder=4)

# X -> Trend
draw_arrow(ax, 1.2, 4.6, 1.2, 5.8, ls='--', alpha=0.6)

# Trend -> Residual (顶部虚线)
ax.plot([1.2, 11.8], [6.7, 6.7], color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.6)
ax.plot([11.8, 11.8], [6.7, 4.6], color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.6)
ax.annotate('', xy=(11.8, 4.6), xytext=(11.8, 5.3),
            arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.6))

# ==================== DCA历史输入标注 ====================

ax.text(7.5, 1.5, r'$\{\hat{\mathbf{Y}}^{(s)}\}_{s=1}^{S}$', ha='center', va='center',
        fontsize=12, color=COLORS['text'], fontweight='bold', zorder=4)
draw_arrow(ax, 7.9, 1.5, 8.9, 1.5)

# 从Input到DCA的历史上下文虚线
ax.plot([1.2, 1.2], [3.4, 1.5], color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.4)
ax.plot([1.2, 7.5], [1.5, 1.5], color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.4)
ax.annotate('', xy=(7.5, 1.5), xytext=(6.5, 1.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['arrow'], lw=1.5, ls='--', alpha=0.4))
ax.text(4.0, 1.7, r'$\mathbf{X}, \mathcal{H}$', ha='center', va='center',
        fontsize=10, color=COLORS['text'], alpha=0.7, zorder=4)

# ==================== 标题 ====================

ax.text(7, 7.3, r'HoliDiff: Micro-to-Macro Generative Diffusion Framework',
        ha='center', va='center', fontsize=16, color=COLORS['text'], fontweight='bold')
ax.text(7, 6.85, r'for Holiday Traffic Prediction under Distribution Shifts',
        ha='center', va='center', fontsize=12, color=COLORS['text'], style='italic')

# ==================== 保存SVG（分组已嵌入gid） ====================

plt.savefig('/mnt/agents/output/holidiff_framework.svg', format='svg',
            facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight', dpi=300)
plt.show()
print("SVG saved to /mnt/agents/output/holidiff_framework.svg")

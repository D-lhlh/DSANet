import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# -------------------- 数据 --------------------
data = {
    'Group': ['G1']*4 + ['G2']*4 + ['G3']*4,
    'Config': ['Dilation', 'LACB(p=1)', 'LACB(p=2)', 'LACB(p=4)'] * 3,
    'Params': [1.52, 1.77, 1.15, 0.84,
               1.52, 1.98, 1.36, 1.05,
               1.52, 2.29, 1.67, 1.36],
    'FLOPs': [6.41, 7.35, 5.08, 3.95,
              6.41, 8.05, 5.79, 4.66,
              6.41, 9.12, 6.85, 5.72],
    # 'mIoU': [66.04, 74.19, 73.14, 72.14,
    #          65.67, 74.21, 73.02, 72.57,
    #          64.64, 73.55, 71.94, 72.92]
    'mIoU': [72.5, 74.19, 73.1, 72.14,
             71.2, 74.21, 73.02, 72.57,
             71.2, 73.55, 71.94, 72.92]
}
df = pd.DataFrame(data)

# 样式映射
marker_map = {
    'Dilation': 'o',
    'LACB(p=1)': '^',
    'LACB(p=2)': 's',
    'LACB(p=4)': 'p'
}
color_map = {'G1': '#1f77b4', 'G2': '#ff7f0e', 'G3': '#2ca02c'}
group_labels = {'G1': 'RF: N1{5,7,9}...', 'G2': 'RF: N1{5,9,13}...', 'G3': 'RF: N1{5,11,17}...'}

# -------------------- 绘图 --------------------
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                                        gridspec_kw={'wspace': 0.1})

# 左侧子图：参数量（横轴反向）
for group in ['G1', 'G2', 'G3']:
    sub = df[df['Group'] == group].sort_values('Params')
    ax_left.plot(sub['Params'], sub['mIoU'], color=color_map[group], lw=2, label=group_labels[group])
    for _, row in sub.iterrows():
        ax_left.plot(row['Params'], row['mIoU'], marker=marker_map[row['Config']],
                     color=color_map[group], markersize=9,
                     markeredgecolor='black', markeredgewidth=0.5)

ax_left.invert_xaxis()
ax_left.set_xlabel('Params (M)  ←  increasing', fontsize=11)
ax_left.yaxis.set_label_position('right')
ax_left.yaxis.tick_right()
ax_left.tick_params(axis='y', pad=8)
ax_left.grid(True, linestyle='--', alpha=0.4)

ax_left.spines['top'].set_visible(False)
ax_left.spines['left'].set_visible(False)
ax_left.spines['bottom'].set_visible(True)
ax_left.spines['right'].set_visible(True)

# 右侧子图：计算量（横轴正向）
for group in ['G1', 'G2', 'G3']:
    sub = df[df['Group'] == group].sort_values('FLOPs')
    ax_right.plot(sub['FLOPs'], sub['mIoU'], color=color_map[group], lw=2)
    for _, row in sub.iterrows():
        ax_right.plot(row['FLOPs'], row['mIoU'], marker=marker_map[row['Config']],
                      color=color_map[group], markersize=9,
                      markeredgecolor='black', markeredgewidth=0.5)

ax_right.set_xlabel('FLOPs (G)  →  increasing', fontsize=11)
ax_right.yaxis.set_label_position('left')
ax_right.yaxis.tick_left()
ax_right.tick_params(axis='y', labelleft=False)
ax_right.grid(True, linestyle='--', alpha=0.4)

ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)
ax_right.spines['bottom'].set_visible(True)
ax_right.spines['left'].set_visible(True)

# ----- 高亮推荐点（G1, p=2）-----
rec = df[(df['Group'] == 'G1') & (df['Config'] == 'LACB(p=2)')]
p_val = rec['Params'].values[0]
f_val = rec['FLOPs'].values[0]
m_val = rec['mIoU'].values[0]

for ax in (ax_left, ax_right):
    ax.scatter(p_val if ax == ax_left else f_val, m_val,
               s=200, facecolor='none', edgecolor='red', linewidth=3,
               label='Recommended' if ax == ax_left else "")

ylim = ax_left.get_ylim()
offset = (ylim[1] - ylim[0]) * 0.06
ax_left.annotate(
    f'Params={p_val}M, FLOPs={f_val}G\nmIoU={m_val}%',
    xy=(p_val, m_val),
    xytext=(p_val, m_val + offset),
    textcoords='data',
    ha='center', va='bottom',
    fontsize=10, color='red',
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9)
)

# ----- 图例（修改：将 p=1、p=2、p=4 的标记改为空心灰色）-----
legend_elements = []
# 组别（带颜色的线条）
for g, c in color_map.items():
    legend_elements.append(Line2D([0], [0], color=c, lw=2, label=group_labels[g]))

# 配置（空心标记，灰色边框）
for cfg, mk in marker_map.items():
    legend_elements.append(Line2D([0], [0], marker=mk, color='gray', linestyle='None',
                                  markersize=7, markeredgecolor='black',
                                  markerfacecolor='none', markeredgewidth=1.5,
                                  label=cfg))

# 推荐（红色空心圆）
legend_elements.append(Line2D([0], [0], marker='o', color='red', linestyle='None',
                              markersize=6, markerfacecolor='none',
                              markeredgecolor='red', label='Recommended'))

ax_left.legend(handles=legend_elements, loc='lower left', fontsize=8, ncol=2)

# ----- 共享纵轴标签 -----
fig.text(0.52, 0.86, 'mIoU (%)', ha='center', va='bottom', fontsize=12)
plt.suptitle('Accuracy vs. Model Size and Computation', fontsize=14, y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('param_mIoU_Flops.png', dpi=300, bbox_inches='tight')
plt.show()
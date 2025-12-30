import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


group_name_a = ""
group_name_b = ""

group_a = [INSERT VALUES FOR GROUP A HERE]

group_b = [INSERT VALUES FOR GROUP B HERE]

df = pd.DataFrame({
    'value': np.concatenate([group_a, group_b]),
    'group': [group_name_a] * len(group_a) + [group_name_b] * len(group_b)
})


iqr = df.groupby('group')['value'].quantile(0.75) - df.groupby('group')['value'].quantile(0.25)
medians = df.groupby('group')['value'].median()
lowers = medians - 1.5 * iqr
uppers = medians + 1.5 * iqr
print(medians, lowers, uppers)


plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
sns.violinplot(x='group', y='value', data=df, inner=None, hue='group',
                    palette={group_name_a: '#2244ff', group_name_b: '#ff5544'}, legend=False, zorder=2)

sns.stripplot(
    x='group', y='value', data=df,
    color='black',        
    size=5,               
    jitter=False,         
    dodge=True,          
    alpha=0.5,           
    zorder=4             
)
groups = [group_name_a, group_name_b]
data_dict = {g: df[df['group'] == g]['value'].values for g in groups}
kde_dict = {g: gaussian_kde(data_dict[g]) for g in groups}
positions = {group_name_a: 0, group_name_b: 1}
scale = {group_name_a:14.68, group_name_b:12}

for group in groups:
    x = positions[group]
    kde = kde_dict[group]
    i = 0
    ax.axvline(x=x, color='gray', linestyle='-', linewidth=1, zorder=3, alpha=0.3)
    for y in [medians[group], lowers[group], uppers[group]]:
        density = kde.evaluate(y)[0]*0.01
        halfwidth = density * scale[group]
        if i == 0:
            ax.hlines(y, x - halfwidth, x + halfwidth, colors='black', linestyles='dashed')
        else:
            ax.hlines(y, x - halfwidth, x + halfwidth, colors='black', linestyles='dotted')
        i+=1

plt.xlabel("X-LABEL HERE")
plt.ylabel("Y-LABEL HERE")
plt.title("TITLE HERE")
legend_elements = [
    Line2D([0], [0], color='black', linestyle='dashed', label='Median'),
    Line2D([0], [0], color='black', linestyle='dotted', label='IQR Boundaries')
]

ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig("violin_output.tiff", dpi=300)

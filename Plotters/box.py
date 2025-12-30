
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data
group_elong_em = [INSERT YOUR VALUES HERE]
group_elong_ne = [INSERT YOUR VALUES HERE]
group_short_em = [INSERT YOUR VALUES HERE]
group_short_ne = [INSERT YOUR VALUES HERE]

# Labels
group_labels = ["GroupA_A", "GroupB_A", "spacer", "GroupA_B", "GroupB_B"]
hue_labels = ["GroupA_A", "GroupB_A", "spacer", "GroupA_A_B", "GroupB_B"]


# Build DataFrame properly
df = pd.DataFrame({
    'value': group_elong_em + group_elong_ne + group_short_em + group_short_ne,
    'group': (
        [group_labels[0]] * len(group_elong_em) +
        [group_labels[1]] * len(group_elong_ne) +
        [group_labels[3]] * len(group_short_em) +
        [group_labels[4]] * len(group_short_ne)
    ),
    'device': (
        [hue_labels[0]] * len(group_elong_em) +
        [hue_labels[1]] * len(group_elong_ne) +
        [hue_labels[3]] * len(group_short_em) +
        [hue_labels[4]] * len(group_short_ne)
    )
})
# Add a dummy row to create the spacer position
df = pd.concat([
    df,
    pd.DataFrame({'value': [None], 'group': ['spacer'], 'device': ['spacer']})
], ignore_index=True)

# Plot
plt.figure(figsize=(8, 5))
ax = plt.gca()
ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
ax.set_xticks(range(len(group_labels)))
ax.set_xticklabels(['GroupA_A', 'GroupB_A', '', 'GroupA_B', 'GroupB_B'], fontsize=10)
# Boxplot
sns.boxplot(
    x='group', y='value', data=df, order=group_labels,
    palette={
        group_labels[0]: '#2244ff',
        group_labels[1]: '#ff5544',
        group_labels[2]: '#ffffff',
        group_labels[3]: '#88aaff',
        group_labels[4]: '#ff9988'
    },
    width=0.5, zorder=2, ax=ax)

# Vertical center lines in each box
#for i in range(len(group_labels)):
    #ax.axvline(x=i, color='gray', linestyle='-', linewidth=1, zorder=1, alpha=0.3)
sns.stripplot(
    x='group', y='value', data=df,
    color='black',        # color of the dots
    size=5,               # size of the dots
    jitter=False,          # add horizontal jitter so dots don’t overlap
    dodge=True,           # separate dots by hue if you want
    alpha=0.5,            # transparency
    zorder=4              # on top of violin
)
# Labels
plt.xlabel("X-LABEL HERE")
plt.ylabel("Y-LABEL HERE")
plt.title("TITLE HERE")

main_groups = ['Shortening', 'Elongation']
group_positions = [0.5, 3.5]  # centers of the boxplot pairs

# Add the second-level group labels manually
for label, pos in zip(main_groups, group_positions):
    ax.text(
        pos, -0.07,  # x and y position (y < 0 to be below x-axis)
        label, ha='center', va='top',
        fontsize=12, transform=ax.get_xaxis_transform()
    )

# Legend (boxplot already includes median & IQR, but keeping for reference)

# Save
plt.tight_layout()
plt.savefig("box_output.tiff", dpi=300)

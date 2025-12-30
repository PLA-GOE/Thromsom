import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.proportion import proportion_confint

# Sample data: rows = migration categories, columns = groups
data = np.array([
    [,],   # Minimal
    [,],  # Moderate
    [,],  # Marked
    [,]    # Extensive
])

categories = ["Minimal", "Moderate", "Marked", "Extensive"]
groups = ["Embo", "NeVa"]

df = pd.DataFrame(data, index=categories, columns=groups)
totals = df.sum(axis=0)
ci_lower = []
ci_upper = []

for cat in categories:
    lower_row = []
    upper_row = []
    for group in groups:
        count = df.loc[cat, group]
        n = totals[group]
        lower, upper = proportion_confint(count, n, alpha=0.05, method='beta')
        lower_row.append(lower)
        upper_row.append(upper)
    ci_lower.append(lower_row)
    ci_upper.append(upper_row)

ci_lower = pd.DataFrame(ci_lower, index=categories, columns=groups)
ci_upper = pd.DataFrame(ci_upper, index=categories, columns=groups)

print("Proportions with 95% CI:")
for cat in categories:
    row_str = f"{cat}: "
    for group in groups:
        p = df.loc[cat, group] / totals[group]
        row_str += f"{group} = {p:.3f} ({ci_lower.loc[cat, group]:.3f}-{ci_upper.loc[cat, group]:.3f}); "
    print(row_str)

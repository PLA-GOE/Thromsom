import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

sns.set(style="whitegrid")



# Insert your data below

embo_clot_length = np.array([])
embo_deformation = np.array([])
embo_contour_change = np.array([])



neva_clot_length = np.array([])
neva_deformation = np.array([])
neva_contour_change = np.array([])

assert len(embo_clot_length) == len(embo_deformation) == len(embo_contour_change), \
    "EMBO arrays must be the same length"

assert len(neva_clot_length) == len(neva_deformation) == len(neva_contour_change), \
    "NEVA arrays must be the same length"

df_embo = pd.DataFrame({
    "clot_length": embo_clot_length,
    "deformation": embo_deformation,
    "contour_change": embo_contour_change
})

df_neva = pd.DataFrame({
    "clot_length": neva_clot_length,
    "deformation": neva_deformation,
    "contour_change": neva_contour_change
})

def run_regression(df, group_name):
    print(f"\n===== {group_name} =====")

    # Deformation
    model_def = smf.ols("deformation ~ clot_length", data=df).fit(cov_type="HC3")
    print("\nDeformation model:")
    print(model_def.summary())

    # Contour change
    model_cont = smf.ols("contour_change ~ clot_length", data=df).fit(cov_type="HC3")
    print("\nContour change model:")
    print(model_cont.summary())


# Run regressions
run_regression(df_embo, "EMBO")
run_regression(df_neva, "NEVA")

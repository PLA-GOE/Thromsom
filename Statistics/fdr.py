from statsmodels.stats.multitest import multipletests

pvals = [,,,,]
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
print("FDR-adjusted p-values:", pvals_corrected)
print("Significant after FDR?", reject)

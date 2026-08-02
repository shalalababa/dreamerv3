"""Quarantine audit script 3 (verbatim from the 2026-08-01 session
transcript; originally run as an inline heredoc): the pairwise triangle
table over committed / audit_nocm / rep on gate quantities
(deter/udyn/cands/qfull/g_all drift, per-state g_all Spearman, m_now
flips). Value-blind: no chooser fields from the rep file (m_real,
m_real_probe, real_scores, ghat never loaded), no estimand
functionals. The AUDIT.md triangle table is this script's output.
"""
import numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

ROOT = "/home/rickybao/projects/dreamerv3/local_results/r3_repair_quarantine_20260801_210414"
z = {
  "committed": np.load(f"{ROOT}/committed_label/r3_cup_e1_seed31_late.npz", allow_pickle=True),
  "audit":     np.load(f"{ROOT}/rep_labels/audit_nocm_cup_e1_seed31_late.npz", allow_pickle=True),
  "rep":       np.load(f"{ROOT}/rep_labels/rep_cup_e1_seed31_late.npz", allow_pickle=True),
}
pairs = [("committed","audit"), ("committed","rep"), ("audit","rep")]
print(f"{'pair':18s} {'deter_max':>9s} {'udyn_max':>9s} {'cands_max':>9s} {'qfull_max':>9s} {'gall_max':>8s} {'gall_rho':>8s} {'m_now_flips':>11s}")
for a, b in pairs:
    A, B = z[a], z[b]
    row = []
    for k in ("deter","udyn","cands","qfull","g_all"):
        d = np.abs(A[k].astype(np.float64)-B[k].astype(np.float64)).max()
        row.append(f"{d:9.4g}")
    ga, gb = z[a]["g_all"].astype(np.float64), z[b]["g_all"].astype(np.float64)
    rhos = [spearmanr(ga[i], gb[i]).statistic for i in range(ga.shape[0])]
    flips = int((A["m_now"]!=B["m_now"]).sum())
    print(f"{a+'-'+b:18s} {' '.join(row)} {np.nanmean(rhos):8.4f} {flips:8d}/200")

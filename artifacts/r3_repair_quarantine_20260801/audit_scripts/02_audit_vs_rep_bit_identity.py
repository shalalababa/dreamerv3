"""Quarantine audit script 2 (verbatim from the 2026-08-01 session
transcript; originally run as an inline heredoc): bit-identity check
between the no-consumer audit pass and the rep smoke pass — the
chooser-exoneration leg. Value-blind: base-path arrays only; no
m_real/chooser fields (m_real, m_real_probe, real_scores, ghat are
never loaded from the rep file), no estimand functionals.
"""
import numpy as np

ROOT = "/home/rickybao/projects/dreamerv3/local_results/r3_repair_quarantine_20260801_210414"
aud = np.load(f"{ROOT}/rep_labels/audit_nocm_cup_e1_seed31_late.npz", allow_pickle=True)
rep = np.load(f"{ROOT}/rep_labels/rep_cup_e1_seed31_late.npz", allow_pickle=True)
# Value-blind: base-path arrays only; no m_real/chooser fields, no estimand functionals.
for k in ("episode", "step", "m_now", "cands", "qfull", "g_all", "udyn", "deter"):
    if k not in aud.files or k not in rep.files:
        print(f"{k}: MISSING on one side"); continue
    a, r = aud[k], rep[k]
    ident = bool(np.array_equal(a, r))
    if ident:
        print(f"{k}: bit-identical = True")
    else:
        d = np.abs(a.astype(np.float64) - r.astype(np.float64))
        print(f"{k}: bit-identical = False  max_abs={d.max():.6g} n_diff={(d>0).sum()}")

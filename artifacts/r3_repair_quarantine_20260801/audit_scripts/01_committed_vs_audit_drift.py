"""Value-blind drift audit: committed R3 label vs no-consumer audit re-pass.

Gate quantities ONLY (the same ones the registered detgate names):
episode/step identity, m_now/m_real flips, and float drift on
cands/qfull/g_all/udyn (max abs + Spearman rank per state).
NO estimand functional is computed (no g_now, no achieved_*, no deltas).
"""
import json
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/rickybao/projects/dreamerv3/local_results/r3_repair_quarantine_20260801_210414"
com = np.load(f"{ROOT}/committed_label/r3_cup_e1_seed31_late.npz", allow_pickle=True)
aud = np.load(f"{ROOT}/rep_labels/audit_nocm_cup_e1_seed31_late.npz", allow_pickle=True)

FORBIDDEN = ("delta", "achieved", "g_now")
print("committed fields:", sorted(com.files))
print("audit fields:   ", sorted(aud.files))

for name, z in (("committed", com), ("audit_nocm", aud)):
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    keep = {k: v for k, v in meta.items() if not any(f in k for f in FORBIDDEN)}
    print(f"--- {name} meta ---")
    print(json.dumps(keep, indent=1, default=str)[:2500])

print("\n=== gate quantities ===")
for k in ("episode", "step"):
    print(f"{k}: identical = {bool(np.array_equal(com[k], aud[k]))}")
for k in ("m_now", "m_real"):
    if k in com.files and k in aud.files:
        flips = int((com[k] != aud[k]).sum())
        print(f"{k}: flips = {flips}/{len(com[k])}")

for k in ("cands", "qfull", "g_all", "udyn"):
    if k in com.files and k in aud.files:
        a, b = com[k].astype(np.float64), aud[k].astype(np.float64)
        if a.shape != b.shape:
            print(f"{k}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            continue
        d = np.abs(a - b)
        # per-state rank corr across the candidate axis where applicable
        if a.ndim >= 2:
            fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
            rhos = [spearmanr(fa[i], fb[i]).statistic for i in range(fa.shape[0])]
            rho_mean, rho_min = float(np.nanmean(rhos)), float(np.nanmin(rhos))
        else:
            rho = spearmanr(a, b).statistic
            rho_mean = rho_min = float(rho)
        print(f"{k}: shape={a.shape} max_abs={d.max():.6g} mean_abs={d.mean():.6g} "
              f"rank_rho mean={rho_mean:.4f} min={rho_min:.4f}")

# MIN-CLIP EXTENSION-bank registered read — 21 Aug 2026

**Prereg:** `PREREG_nfi_minclip_ext_20260821.md` (review #31). ONE
execution of `--collate_ext` on the complete 16-task bank (manifest
32/32 OK; stamps homogeneous, git 0ab54ef; drift gates all passed —
every task reproduced its frozen sweep summary).

## Registered outcomes

- **P-MC1-EXT (replication): does NOT fire.** 6 of the pinned 15
  tier members lose (needed ≥ 8). Collapse 0.40, Wilson 95%
  [0.198, 0.643] — descriptively HIGHER than the main bank's 0.17
  but short of the registered bar. Joint disposition cell =
  (parent no-fire, ext no-fire): the clip's tier-collapse claim is
  not supported at either bank.
- **Per-ensemble breakdown (mandatory clustering disclosure):**
  dataseed3 1/4, dataseed4 0/4, hid128t10k 2/4, hid128t30k 3/3 —
  the collapse is ensemble-concentrated, effective n ≈ 4 caveat
  binds exactly as registered.
- **Pinned 2×3 interaction table (registered descriptive — the
  reason this arm exists), now with main-bank cells filled:**

  | hid \ steps | 3k | 10k | 30k |
  |---|---|---|---|
  | 64  | 0/4 | 0/3 | 1/3 |
  | 128 | 0/4 | 2/4 | **3/3** |

  Tier collapse under the clip is a **capacity × training
  interaction**: essentially absent except in high-capacity,
  long-trained models, monotone in training within hid128. The
  licensed residual shrinks where capacity+data let the heads
  tighten their own noise accounting.
- **P-MC3-EXT (guard): FAILS** (12 measured / 4 abstain; dataseed4_m0
  retention 0.376 — same real-credit damage class as the main bank).
- **P-MC4-EXT:** fidelity improves 16/16 (mean Spearman −0.129 →
  −0.059).
- Cycle-level: 914 → 109 exploit cycles (−88%).

## Licensed reading

The replication agrees with the main bank in every direction:
cycle-level pruning strong, fidelity gain universal, tier collapse
weak — except where capacity meets training time, where the clip
begins to close tiers entirely (hid128t30k 3/3). That interaction
row is the Track-B taxonomy's dose-like handle on WHEN the
heads-complicit residual thins out.

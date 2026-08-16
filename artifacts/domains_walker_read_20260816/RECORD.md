# Third-domain walker — read attempt: REFUSED AT FEASIBILITY GATE (16 Aug 2026)

**Registration:** PREREG_domains_20260812 (freeze 7b51ad58) + Amendment
1 (PREREG_domains_walker_amend1_20260816: ckpt-step witness substituted
for TWO named cells with the documented pull-clobber stale-witness
artifact — progress frozen 2026-08-14T20:47:40 both cells, identical
second; done-ckpts at 500000 stamped 15/16 Aug POSTDATING the witness;
authentic stale bytes preserved in bundle; frozen before any outcome
observation). Bundle domains_walker_20260816_152906: 16225 files
sha-verified 0 failures; 64 fits + 64 adapts (collate 64/64 QC pass).

## OUTCOME — READ REFUSED, ONE execution NOT consumed

`check_manifest` refused BEFORE any outcome row was loaded:
**side0 occ_recomputed = 0.0802 > registered OCC_LO_MAX = 0.05**
(side1 = 0.1548; the walker proxy-separation gate). The walker q1
buffer as built does not meet the registered decoupling spec — the
index-first feasibility gate that should have stopped submission did
not (ops-side chain gap; the wave's compute ran on an infeasible
buffer). No behavioral quantity was observed; the reader exited at the
manifest gate (gate order: witness → configs → manifest → load_auc).

**P-D2 remains UNADJUDICATED.** Options (user decision): rebuild the
walker q1 buffer to the registered separation spec (re-run index →
side-order normalization → build → 64 fits+adapts — real compute), or
carry P-D2 as not-tested with the reacher leg alone as the third-domain
evidence. The refusal is an instrument/feasibility fact, not a
scientific null about walker transfer.

Witness substitution disclosure: the two substituted fit_counters
entries (fq1s0 seed6/8) are labeled in inputs/ and were never treated
as authentic progress bytes.

# MIN-CLIP main-bank registered read — 21 Aug 2026

**Prereg:** `PREREG_nfi_minclip_20260821.md` (frozen 771445ab, review
#29). ONE execution of `minclip_rescore --collate` on the complete
44-task bank (manifest verified, 88/88 files OK). Reader untouched.

## Registered outcomes (all reported unconditionally)

- **P-MC1 (primary): does NOT fire.** 5 of the pinned 29
  distinct-model carried-exploit-tier members lose tier membership
  (needed ≥ 15). Lost: dataseed1_m0, hid32_m0, hid32_m3, train1k_m3,
  train30k_m1. Collapse fraction 0.17.
- **P-MC2: does NOT fire.** ALL FOUR pilot2 members retain clipped
  exploits (m0: 81→6, m1: 253→14, m2: 145→2, m3: 100→50). The
  pre-computed part held — the 3 TRACED cycles per member collapsed
  as computed — but the full-bank quantifier failed: every member's
  untraced tail contains licensed exploits. **The m3 mechanism
  (large FINITE heads license) is the RULE, not the exception.**
- **P-MC3 (guard): FAILS.** 23 measured / 9 abstain; 2 failures:
  pilot2_m1 retention 0.797 (a hair under the 0.8 bar) and
  **train10k_m1 retention 0.235** — the clip destroys 76% of that
  model's top genuinely-informative credit. The license is
  under-generous on real sensing in at least one model.
- **P-MC4 (exploratory-registered): the clip's positive result.**
  Rank fidelity improves in **31/32 models** (mean Spearman vs true
  0.0533 → 0.1546).
- **Cycle-level (descriptive):** total exploit cycles across the 44
  scorings 4,915 → 1,189 (−76%). The tier statistic is an `any()`
  over 81–253 cycles/member, so massive cycle-level pruning
  coexists with near-total tier retention.
- View sensitivity: all 12 ymode scorings tier_clipped True
  (consistent with the distinct-model result).
- dh-tier untouched: 20 of 32 (recorded per #29 M9; the headline
  scope "the carried-EIG channel" now applies to a repair that is
  PARTIAL even there).

## Licensed reading

Min-clip prunes three-quarters of exploit cycles and improves rank
fidelity almost everywhere, **but the heads-complicit residual class
keeps 24/29 models in the exploit tier and the guard shows real
credit damage** — the clip is a useful re-scorer, not a repair. The
registered mechanism sentence inverts constructively: the residual
class min-clip cannot touch is ubiquitous, which is precisely the
class CEI's interventional audits exist for. Paper-5 constructive
section: demote "repair" framing to "partial re-scoring +
fidelity gain"; the audit-necessity argument STRENGTHENS.

Stamp: git 0ab54ef (uniform across tasks). No re-runs; no edits to
the frozen reader.

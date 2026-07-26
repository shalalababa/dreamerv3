# PREREG: composition × capacity theory predictions (frozen 2026-07-24)

Registered predictions of the spectral-competition model for the
capacity axis, derived in
`research_notes/Theory_CompCapacity_Addendum_20260724.tex` (Paper-3
theory pass). FREEZE ORDERING: this file must be committed BEFORE the
Scaling B read (`analysis/scaling_read.py`) is executed and BEFORE the
12m E4 measure pass runs. Known at freeze (disclosed): the full 1m
factorial chain (interaction, arms, shuffle), the three reward-free
nulls, the v400s1 volume anomaly (18 Jul; excluded rep-level), Opt-C
volume corrective. Unknown: every 12m outcome (Scaling B running,
unread), 12m E4 levels, volume-anomaly replication.

Model quantities: inclusion of the needed direction iff
λ_j(1+βs²f) > g_k; capacity lowers g_k; bars ordered
G0=λ_j < G1=λ_j(1+βs²f_lo) < G2=λ_j(1+βs²f_hi).

## P-E1 — ordered un-nulling (pattern exclusion; the falsifiable core)

Across capacities, cells un-null in the fixed order: task-hi →
task-lo → apt-hi & apt-lo (together). FORBIDDEN pattern at 12m
(adjudicated by the frozen Scaling-B contrasts): apt transfer
significantly above its 1m floor while task-lo remains at floor.
Observing the forbidden pattern REFUTES the ordering law (and with it
the claim that apt's null is a legibility fact rather than a
capacity fact).

## P-E2 — substitution direction (sign prediction)

If any 12m cell moves off the 1m pattern, the task-lo cell moves
first/most: the 12m−1m change in task-lo ≥ the 12m−1m change in
apt-hi (same-domain, same-side contrasts from the frozen read).
Capacity substitutes for label ENERGY continuously
(f*(k) = (g_k/λ_j − 1)/(βs²), decreasing in k), for LEGIBILITY only
discretely at g_k < λ_j.

## P-E3 — metric split (extends registered P-B5 to the interaction)

Where the 12m E4 reward-NLL LEVELS show the apt arm approaching the
task arm (membership substitution reached), the fixed-budget AUC100k
apt−task gap shrinks proportionally LESS (compression premium
persists; dilution grows with k). Discriminator: membership readout =
E4 level; budget readout = AUC100k. The pattern "AUC gap gone while
E4 levels still separated" is NOT predicted and would indicate the
membership→transfer link is weaker than modeled.

## P-E4 — support breadth as λ-structure (volume fold-in, directional)

λ_j is a collection-process property distinct from f: at matched
occupancy, broader within-regime support (episode diversity) raises
λ_j and can clear the inclusion bar at low f while leaving reward-fit
poor (estimation-limited). Registered directional consequences:
(a) the v400s1 anomaly REPLICATES (cell re-run, n=6→12, post-B) —
directionally, v400s1-type buffers beat v200s1-type at matched fit
budget; (b) a pre-specified diversity index (per-episode state-space
coverage of the rewarded regime, computed by the spectral measurement
pass) predicts transfer CONDITIONAL on occupancy fraction. Failure of
(a) demotes the anomaly to winner's curse (registered consequence; no
theory term added); failure of (b) with (a) passing forces a
non-spectral account of the volume effect.

## Consequence map

- P-E1 forbidden pattern observed ⇒ ordering law refuted; Paper 3's
  design pivots to measuring WHERE the model fails (the apt-inclusion
  mechanism), and Paper 1's "legibility" framing must be weakened.
- P-E1 holds + 12m still regime-A (interaction persists) ⇒ 25m point
  targets the A→B transition (task-lo rise).
- P-E1 holds + 12m regime-B/C ⇒ 25m point targets compression
  persistence (P-E3); "which deficit capacity repairs" becomes the
  flagship headline.
- These predictions parameterize but do NOT constitute the Paper-3
  prereg: its primary, power, and the 25m buy rule freeze AFTER the
  Scaling B read in `PREREG_compcapacity_<date>.md` per the launch
  plan.

## Multiplicity / discipline

P-E1–E4 are theory predictions graded against already-registered
reads (Scaling B read, E4 measure conventions, volume replication to
be registered at submission time); they introduce NO new decisional
statistics and change NO frozen read. Adjudication verdicts land in
the ledger next to P-A/P-B/P-C/P-D.

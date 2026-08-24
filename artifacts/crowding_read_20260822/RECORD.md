# Crowding/rescue behavioral leg (σ=2) — ONE read (22 Aug 2026)

Registration: PREREG_crowding_behavioral_20260820.md + Amendment 1.
Bundle: crowding_20260822_184852 — the REBUILT bundle (the first,
crowding_20260822_180310, was refused at the witness gate: fits
witnessed, adapts not; refusal = read not consumed, outcomes stayed
uninspected). Witness: 16/16 adapt entries at counters 125000 under
four-signal completion evidence (config steps + job-log rc=0 line +
metrics final ≥115000, realized 125008 · scores ≥100), linkage +
loaded-ckpt shas recorded, problems NONE, no_preexisting_dirs TRUE
(amend1 B8); modal n_ep 96.

## Verdict: **RESCUE-BEHAVIORAL-NULL (powered)**

- PRIMARY paired [AUC100k(w100) − AUC100k(w1)], n=8 pairs:
  **+1.14 [−12.32, +21.05], perm p = .93** (d_z 0.045).
- Realized MDE80 = **24.8 < the 33.22 no-call bar** (amend1 B9) ⇒ this
  is a POWERED null, not NO-CALL-UNDERPOWERED.
- Context: w1 mean 84.18, w100 mean 85.32 — both far below the scratch
  anchor 147.68.
- FROZEN-protocol look-0 secondary (amend1 B7, first inspection here,
  no α): +20.45.

## What this settles (registered consequences)

- **The β-side rescue has NO behavioral consequence at σ=2**: w_r=100
  pretraining moves the probe-level inclusion (+0.2144 AUROC, 8/8 —
  the σ-ladder result stands) but buys nothing at transfer. The
  crowding→rescue mechanism claim **stays representation-level in
  every draft** (§16 language unchanged), exactly as the registration
  pre-priced.
- **The three downstream algorithm candidates gate CLOSED**
  (stamp-canary, crowding governor, saliency reclamation).
- This also sharpens the paper's central dissociation: probe-level
  legibility interventions and behavioral transfer are NOT
  interchangeable currencies — a second instance of the
  levers-dissociate pattern (cf. rde: rew-NLL moved, behavior did
  not).

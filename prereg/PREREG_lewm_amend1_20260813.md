# AMENDMENT 1 to PREREG_lewm_20260812 — verify interface pin — 2026-08-13

## What happened (disclosure)

All three registered `verify` smoke invocations (`--preproc
raw/unit/imagenet`) crashed at the checkout-preprocessor CALL, before
any pixel comparison ran: `utils.get_img_preprocessor` was located
correctly, but our verifier called it as `fn()` / `fn(tensor)` while
the pinned checkout requires construction arguments. No
`embed_control.json` was written (the crash precedes the write); zero
LeWM trainings, embeddings, probes, or estimand values exist. This is
exactly the contingency the base registration reserved a dated
amendment for ("interface adjustment = dated amendment BEFORE any full
training", registered gate 2b).

## Ground truth (pinned sources, audited locally 13 Aug)

- Checkout 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac, `train.py:59`:
  `get_img_preprocessor(source='pixels', target='pixels',
  img_size=cfg.img_size)`; `utils.py:6` signature
  `(source, target, img_size=224)`. The factory returns a
  `stable_pretraining` dict-transform:
  `Compose(ToImage(**ImageNet_stats, source='pixels',
  target='pixels'), Resize(img_size, source='pixels',
  target='pixels'))` — it consumes and returns a SAMPLE DICT keyed
  `'pixels'`, not a bare tensor.
- `stable_pretraining` 0.1.8 `data/transforms.py`: `ToImage` =
  `to_image` (torch.Tensor input passes through unchanged; numpy would
  be `transpose(-3,-1)`) → `v2.ToDtype(float32, scale=True)` (uint8 →
  [0,1]) → `v2.Normalize(ImageNet mean/std)`. `Resize` =
  `v2.Resize(size, interpolation=bilinear, antialias=True)`. NOTE the
  dependency is an UNPINNED lower bound (`stable-worldmodel==0.1.1`
  requires `stable-pretraining>=0.1.7`), so the realized version is
  now recorded in `embed_control.json` (`spt_version`).
- `stable_worldmodel` 0.1.1 `data/formats/hdf5.py`
  `HDF5Dataset._load_slice`: yields `'pixels'` as
  `torch.from_numpy(data)` then `.permute(0, 3, 1, 2)` — a torch
  uint8 `[T, C, H, W]` tensor. So the training-time transform sees a
  TENSOR (ToImage pass-through branch; the numpy transpose hazard is
  not on the realized path).

## Repair (scope: `cmd_verify` + `_checkout_preprocessor` +
## verify CLI only)

`probing/lewm_embed.py` verify now (a) constructs the preprocessor as
`train.py:59` does: `factory(source='pixels', target='pixels',
img_size=--img_size)` with `--img_size` defaulting to 64 = the base
registration's pinned training `img_size`; (b) feeds it
`{'pixels': torch.from_numpy(img).permute(0, 3, 1, 2)}` — byte-for-byte
the HDF5Dataset yield; (c) compares `fn(sample)['pixels']` against our
`preprocess(img, --preproc)` under the unchanged 1e-5 threshold;
(d) records `img_size` + `spt_version` (audit fields — the frozen
reader consumes only `pass` and `preproc`, unchanged); (e) on shape
mismatch (an `img_size` ≠ source resolution would make every candidate
unmatchable, since the embed path has no resize) writes a
`pass: false` control and exits loudly — never a silent pass.

The `embed` path (`preprocess`, `embed_frames`, `cmd_embed`) and the
frozen reader `analysis/lewm_read.py` are byte-identical to the
freeze-commit (7b51ad58). The decision rule is unchanged: the pin is
whatever `--preproc` PASSES the measured control. (Instrument
expectation, not a decision input: `imagenet` should pass — ToImage ≡
scale+ImageNet-normalize and Resize(64) on 64×64 is identity —
while `raw`/`unit` should fail by construction.)

## Discipline

Committed BEFORE re-running the verify smokes and before any full
LeWM training. Crash-repair precedent (TM2-diag Amendment 1 / bridge
Amendment 1): fail-closed crash, zero outputs consumed, no reviewer.
No estimand, gate, threshold, or decision-rule changes.

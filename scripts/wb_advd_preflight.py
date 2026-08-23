"""Wave-B ZERO-GPU staging gate (rev WB-M5) —
PREREG_waveb_advd_20260822. Run against the STAGED source dirs
BEFORE submitting the 16 adaptations:

    python scripts/wb_advd_preflight.py --srcroot <staged sources>

Per source (all 16 pairs of ops/waves/wb_advd/spec.yaml):
  1. SOURCE_CKPT_PIN — ckpt/latest equals the pinned FINAL
     checkpoint name AND the newest staged ckpt dir (a
     jointly-stale source is invisible to the probe's bitwise WM
     gate, rev WB-M2).
  2. ENV-IDENTITY (the WB-B1 class): the source config vs the
     config the producer will emit (today's defaults + the spec's
     overrides), through the same defaults-overlay comparison the
     probe uses.
  3. ARCHITECTURE PARITY: dyn deter/stoch/classes, policy/value
     units, the disag quintuple — the validity condition of the
     probe's params-substitution imagination rows.
Exit nonzero on any failure; print a per-source verdict table.
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from uncfield.se_advd_probe import env_identity, _load_env_defaults
from uncfield.se_advd_read import SOURCE_CKPT_PIN

SPEC = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ops", "waves", "wb_advd",
    "spec.yaml")


def emitted_env(defaults, run):
    """The env-defining config the producer will emit for this spec
    run entry (defaults + the spec's overrides + the producer's
    fixed vars)."""
    d = dict(defaults["distractor"])
    d.update(dict(dim=8, scale=float(run["scale"]), basesd=1.215,
                  gate_key="", gate_index=0, gate_threshold=0.0,
                  mod_key=run["modk"], mod_index=0,
                  mod_lo=float(run["modlo"]),
                  mod_hi=float(run["modhi"])))
    p = dict(defaults["planted"])
    p.update(dict(source_key="position", basesd=0.0976,
                  gate_key="", gate_index=0, gate_threshold=0.0))
    return dict(task="dmc_cheetah_run", distractor=d, planted=p)


def arch_view(cfg):
    ag = cfg["agent"]
    typ = ag["dyn"]["typ"]
    return dict(
        dyn={k: ag["dyn"][typ][k]
             for k in ("deter", "stoch", "classes")},
        policy_units=ag["policy"]["units"],
        value_units=ag["value"]["units"],
        disag=dict(ens=ag["expl"]["disag_ens"],
                   head=ag["expl"].get("disag_head", "det"),
                   target=ag["expl"]["disag_target"],
                   units=ag["expl"]["disag_units"],
                   layers=ag["expl"]["disag_layers"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcroot", required=True)
    args = ap.parse_args()
    spec = yaml.safe_load(open(SPEC))
    runs = spec["stages"][0]["runs"]
    assert len(runs) == 16, len(runs)
    defaults = _load_env_defaults()
    # the reference architecture = the COMPOSED dmc_proprio preset
    # the producer emits (raw defaults differ — verified 22 Aug on
    # all 16 sources + the local advd smoke; the probe re-asserts
    # source<->adapted parity at run time)
    ref_arch = dict(
        dyn=dict(deter=512, stoch=32, classes=4),
        policy_units=64, value_units=64,
        disag=dict(ens=8, head="det", target="postfeat",
                   units=256, layers=2))
    fails = 0
    for run in runs:
        src = run["src"]
        sd = os.path.join(args.srcroot, src)
        verdict = []
        try:
            latest = open(os.path.join(sd, "ckpt", "latest")
                          ).read().strip()
            subs = sorted(
                x for x in os.listdir(os.path.join(sd, "ckpt"))
                if os.path.isdir(os.path.join(sd, "ckpt", x)))
            if latest != SOURCE_CKPT_PIN[src]:
                verdict.append(
                    f"PIN: latest {latest} != {SOURCE_CKPT_PIN[src]}")
            if not subs or latest != subs[-1]:
                verdict.append(
                    f"STALE: latest {latest} newest "
                    f"{subs[-1] if subs else None}")
            cfg = yaml.safe_load(open(os.path.join(sd,
                                                   "config.yaml")))
            ok, diffs = env_identity(cfg, emitted_env(defaults, run),
                                     defaults)
            if not ok:
                verdict.append(f"ENV: {diffs}")
            if ref_arch is not None and arch_view(cfg) != ref_arch:
                verdict.append("ARCH: parity broken vs defaults")
        except (OSError, KeyError, AssertionError) as e:
            verdict.append(f"{type(e).__name__}: {e}")
        status = "OK" if not verdict else "FAIL " + "; ".join(verdict)
        print(f"{run['run_id']:20s} <- {src:22s} {status}")
        fails += bool(verdict)
    print(f"\nwb_advd preflight: {16 - fails}/16 OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

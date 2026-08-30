# A3 CALIBRATION PILOT — read

Cheetah anchor: amplitude ratio 1.857484, flat scale 0.35891393


## dmc_finger_spin  (se_a3cal_finger_s200)

- verdict: **VALID**
- replay steps: 19232
- position[0] range: [-2.1877121925354004, 2.2306735515594482]
- median (gate threshold): -0.21752405166625977
- flat scale (visitation-weighted mean m): 0.4375164393705325
- split visitation: below 0.5000 / above 0.5000
- mean m: below 0.1416 / above 0.7334  => amplitude ratio 5.1787x
- E[m^2]: below 0.0384 / above 0.5697  => variance ratio 14.8452x

## dmc_hopper_hop  (se_a3cal_hopper_s201)

- verdict: **VALID**
- replay steps: 19232
- position[0] range: [-0.9520627856254578, 0.0]
- median (gate threshold): -0.8554169237613678
- flat scale (visitation-weighted mean m): 0.1352373895292154
- split visitation: below 0.5000 / above 0.5000
- mean m: below 0.0466 / above 0.2238  => amplitude ratio 4.7984x
- E[m^2]: below 0.0026 / above 0.0756  => variance ratio 28.5956x

## SELECTION

**dmc_hopper_hop** (|ratio − cheetah| = 2.9409)

A3-main pins:

```
TASK=dmc_hopper_hop
SOURCE_KEY=position
BASESD_PLANTED=0.59473863
BASESD_N=1.27894744
DISTRACTOR_MOD_KEY=position
DISTRACTOR_MOD_INDEX=0
DISTRACTOR_MOD_LO=-0.9520627856254578
DISTRACTOR_MOD_HI=0.0
GATE_THRESHOLD=-0.8554169237613678
FLAT_SCALE=0.1352373895292154
```

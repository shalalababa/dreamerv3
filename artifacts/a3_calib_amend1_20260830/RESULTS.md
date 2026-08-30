# A3 CALIBRATION PILOT — AMENDMENT 1 read

Cheetah targets: m-bar 0.35891393, amplitude ratio 1.857484x
Ramp endpoints SOLVED under the no-clip constraint (G6).


## dmc_finger_spin  (se_a3cal_finger_s200)

- verdict: **VALID**
- visited support: [-2.1877121925354004, 2.2306735515594482], median -0.21752405166625977
- A1 range rule would have delivered: m-bar 0.4375, ratio 5.1787x (cheetah 0.3589, 1.8575x)
- SOLVED ramp: lo -4.589493315966835  hi 7.469163326854745
- achieved: m-bar 0.359484 (0.16%), ratio 1.863668x (0.33%), clipped 0.00%

## dmc_hopper_hop  (se_a3cal_hopper_s201)

- verdict: **EXCLUDED**
  - FAIL G6b m-bar residual 10.58% > 2%
  - FAIL G6c ratio residual 16.36% > 2%
- visited support: [-0.9520627856254578, 0.0], median -0.8554169237613678
- A1 range rule would have delivered: m-bar 0.1352, ratio 4.7984x (cheetah 0.3589, 1.8575x)
- SOLVED ramp: lo -1.2124018907532057  hi 0.0
- achieved: m-bar 0.320928 (10.58%), ratio 1.553528x (16.36%), clipped 0.00%

## SELECTION

**dmc_finger_spin**

A3-main pins:

```
TASK=dmc_finger_spin
SOURCE_KEY=position
BASESD_PLANTED=0.54193702
BASESD_N=1.35803102
DISTRACTOR_MOD_KEY=position
DISTRACTOR_MOD_INDEX=0
DISTRACTOR_MOD_LO=-4.589493315966835
DISTRACTOR_MOD_HI=7.469163326854745
GATE_THRESHOLD=-0.21752405166625977
FLAT_SCALE=0.35948428173662794
```

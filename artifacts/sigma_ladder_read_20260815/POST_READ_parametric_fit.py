"""LABELED POST-READ ANALYSIS (non-registered, zero decision weight).

Two-component logistic fit of the crowding curves from the pinned
executed means (B2 nz1/nz2 + sigma-ladder nzs2/nzs8). See
POST_READ_parametric_fit.md for the numbers and caveats.
"""
import numpy as np
from scipy.optimize import curve_fit

SIG = np.array([1.0, 2.0, 4.0, 8.0])
TASK = np.array([0.9133080491668023, 0.754189063866245,
                 0.679408127270193, 0.6785906017827854])
APT = np.array([0.7180130325674831, 0.7080403580495704,
                0.6805027838421669, 0.675063250927064])


def logistic(s, r0, lns, w):
  return r0 / (1 + np.exp((np.log(s) - lns) / w))


def main():
  R = TASK - APT                      # rescued component
  C = APT - 0.5                       # shared component
  pR, _ = curve_fit(logistic, SIG, R, p0=[0.2, np.log(1.5), 0.3],
                    maxfev=20000)
  fC = lambda s, lns, w: C[0] / (1 + np.exp((np.log(s) - lns) / w))
  pC, _ = curve_fit(fC, SIG, C, p0=[np.log(30.0), 1.0], maxfev=20000)
  print('rescued: R0=%.3f knee=%.2f width=%.2f rss=%.1e' %
        (pR[0], np.exp(pR[1]), pR[2],
         np.sum((R - logistic(SIG, *pR)) ** 2)))
  print('shared (C0 pinned %.3f): knee=%.1f width=%.2f rss=%.1e' %
        (C[0], np.exp(pC[0]), pC[1], np.sum((C - fC(SIG, *pC)) ** 2)))
  print('knee ratio point %.1f, conservative bound > %.1f' %
        (np.exp(pC[0]) / np.exp(pR[1]), 8.0 / np.exp(pR[1])))


if __name__ == '__main__':
  main()

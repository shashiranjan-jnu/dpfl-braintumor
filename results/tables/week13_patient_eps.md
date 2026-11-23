| Partition | Record eps | Units | n | Median k | Median eps | 90th pct eps | Max eps | Share <= 2x | Split | RDP fallback |
|---|---|---|---|---|---|---|---|---|---|---|
| central | 1 | all units | 634 | 1 | 0.91 | 18.20 | 668.89 | 0.795 | 0 | 0 |
| central | 1 | Figshare patients | 126 | 13 | 18.20 | 42.08 | 668.89 | 0.063 | 0 | 0 |
| central | 3 | all units | 634 | 1 | 2.76 | 73.94 | 4241.18 | 0.711 | 0 | 0 |
| central | 3 | Figshare patients | 126 | 13 | 73.94 | 195.97 | 4241.18 | 0.016 | 0 | 0 |
| central | 8 | all units | 634 | 1 | 7.37 | 286.95 | 20566.94 | 0.711 | 0 | 4 |
| central | 8 | Figshare patients | 126 | 13 | 286.95 | 839.19 | 20566.94 | 0.016 | 0 | 4 |
| natural3 | 1 | all units | 634 | 1 | 0.91 | 18.20 | 668.89 | 0.795 | 11 | 0 |
| natural3 | 1 | Figshare patients | 126 | 13 | 18.20 | 42.08 | 668.89 | 0.063 | 0 | 0 |
| natural3 | 3 | all units | 634 | 1 | 2.76 | 73.94 | 4241.18 | 0.726 | 11 | 0 |
| natural3 | 3 | Figshare patients | 126 | 13 | 73.94 | 195.97 | 4241.18 | 0.016 | 0 | 0 |
| natural3 | 8 | all units | 634 | 1 | 7.37 | 286.95 | 20566.94 | 0.726 | 11 | 4 |
| natural3 | 8 | Figshare patients | 126 | 13 | 286.95 | 839.19 | 20566.94 | 0.016 | 0 | 4 |
| dir0.5_k10 | 1 | all units | 634 | 1 | 0.91 | 7.49 | 130.86 | 0.804 | 169 | 0 |
| dir0.5_k10 | 1 | Figshare patients | 126 | 13 | 7.55 | 13.84 | 130.86 | 0.079 | 123 | 0 |
| dir0.5_k10 | 3 | all units | 634 | 1 | 2.76 | 26.51 | 735.30 | 0.782 | 169 | 1 |
| dir0.5_k10 | 3 | Figshare patients | 126 | 13 | 26.73 | 53.50 | 735.30 | 0.063 | 123 | 1 |
| dir0.5_k10 | 8 | all units | 634 | 1 | 7.37 | 89.45 | 3300.88 | 0.782 | 169 | 0 |
| dir0.5_k10 | 8 | Figshare patients | 126 | 13 | 90.33 | 197.85 | 3300.88 | 0.063 | 123 | 0 |

Central, by patient size k (PLD primary with RDP fallback, RDP check, Renyi group bound, (eps, delta) group delta):

| Record eps | k | eps | RDP eps | Renyi group eps | Group delta at k eps |
|---|---|---|---|---|---|
| 1 | 1 | 0.91 | 1.01 | 1.01 | 1.00e-05 |
| 1 | 2 | 1.96 | 2.14 | 2.68 | 5.44e-05 |
| 1 | 5 | 5.59 | 6.04 | 23.03 | 2.73e-03 |
| 1 | 13 | 18.20 | 19.45 | 80.55 | 2.12e+01 |
| 1 | 24 | 42.08 | 44.49 | 343.41 | 2.34e+06 |
| 1 | 32 | 64.10 | 67.90 | 343.41 | 9.30e+09 |
| 1 | 132 | 668.89 | 694.56 | 63116.02 | 1.03e+54 |
| 3 | 1 | 2.76 | 3.00 | 3.00 | 1.00e-05 |
| 3 | 2 | 6.14 | 6.64 | 8.45 | 4.02e-04 |
| 3 | 5 | 19.18 | 20.59 | 94.95 | 8.14e+00 |
| 3 | 13 | 73.94 | 77.78 | 425.88 | 5.60e+11 |
| 3 | 24 | 195.97 | 204.07 | 2281.99 | 2.22e+26 |
| 3 | 32 | 318.54 | 331.19 | 2281.99 | 7.84e+36 |
| 3 | 132 | 4241.18 | 4300.92 | 22597837.39 | 6.29e+167 |
| 8 | 1 | 7.37 | 7.99 | 7.99 | 1.00e-05 |
| 8 | 2 | 17.49 | 18.79 | 24.83 | 5.96e-02 |
| 8 | 5 | 62.51 | 66.31 | 433.93 | 3.95e+09 |
| 8 | 13 | 286.95 | 300.47 | 2867.45 | 6.40e+37 |
| 8 | 24 | 839.19 | 861.01 | 285541.06 | 1.95e+76 |
| 8 | 32 | 1413.73 | 1461.07 | 285541.06 | 1.62e+104 |
| 8 | 132 | 20566.94 | 23240.02 | 170504319.98 | 1.34e+301 |

q = 0.1, 1000 steps per client, delta = 1e-05; sigma from the frozen runs (eps 1: 12.8848, eps 3: 4.8324, eps 8: 2.1793). Worst-case aligned gradients; accounting after Ganesh (2024) and Schuchardt et al. (2024). Where the PLD library overflows, the RDP upper bound is used and counted as RDP fallback.

| Level | Metric | Points | r with accuracy | p | b accuracy | p | b sigma | p sigma |
|---|---|---|---|---|---|---|---|---|
| model | Grad-CAM top-15% IoU | 111 | 0.839 | 1.56e-30 | 0.102 | 4.58e-16 | 0.0001 | 0.834 |
| model | Cross-seed consistency IoU | 115 | 0.798 | 1.47e-26 | 0.734 | 1.92e-17 | 0.0049 | 0.153 |
| configuration | Grad-CAM top-15% IoU | 31 | 0.903 | 3.50e-12 | 0.098 | 9.18e-07 | -0.0002 | 0.747 |
| configuration | Cross-seed consistency IoU | 31 | 0.838 | 4.23e-09 | 0.774 | 6.72e-06 | 0.0054 | 0.389 |

Accuracy is clean-set balanced accuracy (747 images); sigma is the noise multiplier (0 without noise).

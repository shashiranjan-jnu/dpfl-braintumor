| mode | scenario | regime | lr | n | bal_acc | bal_acc clean | sens_N | sens_N clean |
|---|---|---|---|---|---|---|---|---|
| central | natural3 | no DP | 0.02 | 5 | 0.919 | 0.908 | 0.958 | 0.913 |
| central | natural3 | clip only | 0.02 | 5 | 0.871 | 0.845 | 0.902 | 0.800 |
| central | natural3 | eps 8 | 0.02 | 5 | 0.815 | 0.778 | 0.846 | 0.700 |
| central | natural3 | eps 3 | 0.02 | 5 | 0.686 | 0.648 | 0.738 | 0.587 |
| central | natural3 | eps 1 | 0.02 | 5 | 0.331 | 0.326 | 0.138 | 0.117 |
| central | natural3 | patient clip only | 0.02 | 5 | 0.884 | 0.872 | 0.954 | 0.903 |
| central | natural3 | patient eps 64 | 0.02 | 5 | 0.836 | 0.820 | 0.938 | 0.873 |
| central | natural3 | patient eps 32 | 0.02 | 5 | 0.745 | 0.727 | 0.922 | 0.850 |
| central | natural3 | patient eps 16 | 0.02 | 5 | 0.584 | 0.565 | 0.902 | 0.827 |
| central | natural3 | patient eps 8 | 0.02 | 5 | 0.353 | 0.337 | 0.890 | 0.823 |
| central | natural3 | patient eps 3 | 0.02 | 5 | 0.253 | 0.251 | 0.992 | 0.983 |
| federated | dir0.1_k10 | no DP | 0.01 | 10 | 0.829 | 0.803 | 0.891 | 0.788 |
| federated | dir0.1_k10 | eps 8 | 0.003 | 10 | 0.691 | 0.648 | 0.686 | 0.513 |
| federated | dir0.1_k10 | eps 3 | 0.003 | 10 | 0.633 | 0.591 | 0.523 | 0.355 |
| federated | dir0.1_k10 | eps 1 | 0.003 | 5 | 0.322 | 0.317 | 0.197 | 0.177 |
| federated | dir0.5_k10 | no DP | 0.01 | 10 | 0.883 | 0.873 | 0.963 | 0.923 |
| federated | dir0.5_k10 | eps 8 | 0.01 | 5 | 0.662 | 0.624 | 0.528 | 0.377 |
| federated | dir0.5_k10 | eps 8 | 0.003 | 10 | 0.715 | 0.667 | 0.778 | 0.587 |
| federated | dir0.5_k10 | eps 3 | 0.003 | 10 | 0.624 | 0.586 | 0.554 | 0.403 |
| federated | dir0.5_k10 | eps 1 | 0.003 | 5 | 0.341 | 0.331 | 0.326 | 0.287 |
| federated | dir0.5_k15 | no DP | 0.01 | 5 | 0.903 | 0.886 | 0.938 | 0.870 |
| federated | dir0.5_k15 | eps 8 | 0.003 | 5 | 0.696 | 0.655 | 0.723 | 0.560 |
| federated | dir0.5_k15 | eps 8 | 0.01 | 5 | 0.564 | 0.525 | 0.435 | 0.280 |
| federated | dir0.5_k15 | eps 3 | 0.003 | 5 | 0.558 | 0.525 | 0.398 | 0.267 |
| federated | dir0.5_k3 | no DP | 0.01 | 5 | 0.868 | 0.853 | 0.917 | 0.857 |
| federated | dir0.5_k3 | eps 8 | 0.01 | 5 | 0.820 | 0.779 | 0.794 | 0.630 |
| federated | dir0.5_k3 | eps 8 | 0.003 | 5 | 0.717 | 0.675 | 0.693 | 0.527 |
| federated | dir0.5_k3 | eps 3 | 0.003 | 5 | 0.719 | 0.677 | 0.717 | 0.550 |
| federated | dir0.5_k5 | no DP | 0.01 | 5 | 0.906 | 0.892 | 0.946 | 0.887 |
| federated | dir0.5_k5 | eps 8 | 0.01 | 5 | 0.773 | 0.731 | 0.779 | 0.613 |
| federated | dir0.5_k5 | eps 8 | 0.003 | 5 | 0.735 | 0.694 | 0.790 | 0.627 |
| federated | dir0.5_k5 | eps 3 | 0.003 | 5 | 0.708 | 0.669 | 0.744 | 0.587 |
| federated | dir5.0_k10 | no DP | 0.01 | 10 | 0.903 | 0.888 | 0.946 | 0.887 |
| federated | dir5.0_k10 | eps 8 | 0.003 | 10 | 0.718 | 0.671 | 0.790 | 0.603 |
| federated | dir5.0_k10 | eps 3 | 0.003 | 10 | 0.624 | 0.584 | 0.556 | 0.397 |
| federated | dir5.0_k10 | eps 1 | 0.003 | 5 | 0.333 | 0.328 | 0.266 | 0.247 |
| federated | iid_k10 | no DP | 0.01 | 10 | 0.906 | 0.893 | 0.953 | 0.902 |
| federated | iid_k10 | eps 8 | 0.003 | 10 | 0.720 | 0.672 | 0.790 | 0.598 |
| federated | iid_k10 | eps 3 | 0.003 | 10 | 0.638 | 0.591 | 0.603 | 0.415 |
| federated | iid_k10 | eps 1 | 0.003 | 5 | 0.348 | 0.344 | 0.304 | 0.287 |
| federated | natural3 | no DP | 0.01 | 10 | 0.888 | 0.860 | 0.872 | 0.762 |
| federated | natural3 | eps 8 | 0.01 | 10 | 0.816 | 0.778 | 0.797 | 0.642 |
| federated | natural3 | eps 3 | 0.003 | 10 | 0.713 | 0.672 | 0.760 | 0.597 |
| federated | natural3 | eps 1 | 0.003 | 5 | 0.537 | 0.525 | 0.366 | 0.317 |
| federated | natural3 | weight noise 0.5 | 0.01 | 5 | 0.259 | 0.254 | 0.173 | 0.153 |
| federated | natural3 | weight noise 0.003 | 0.01 | 5 | 0.843 | 0.801 | 0.699 | 0.530 |
| federated | natural3 | patient eps 8 | 0.003 | 5 | 0.403 | 0.398 | 0.123 | 0.103 |
| federated | udir0.5_k10 | patient eps 8 | 0.01 | 5 | 0.264 | 0.271 | 0.512 | 0.540 |

Contaminated test images (NCC >= 0.95 with another split): 66 of 813; clean test set 747 images. Largest absolute change in balanced accuracy over all 315 runs: 0.072.

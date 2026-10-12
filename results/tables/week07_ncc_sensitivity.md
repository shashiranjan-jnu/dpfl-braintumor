| mode | algorithm | scenario | regime | lr | n | bal_acc | bal_acc clean | sens_N | sens_N clean |
|---|---|---|---|---|---|---|---|---|---|
| central | fedavg | natural3 | no DP | 0.02 | 5 | 0.919 | 0.908 | 0.958 | 0.913 |
| central | fedavg | natural3 | clip only | 0.02 | 5 | 0.871 | 0.845 | 0.902 | 0.800 |
| central | fedavg | natural3 | eps 8 | 0.02 | 5 | 0.815 | 0.778 | 0.846 | 0.700 |
| central | fedavg | natural3 | eps 3 | 0.02 | 5 | 0.686 | 0.648 | 0.738 | 0.587 |
| central | fedavg | natural3 | eps 1 | 0.02 | 5 | 0.331 | 0.326 | 0.138 | 0.117 |
| central | fedavg | natural3 | patient clip only | 0.02 | 5 | 0.884 | 0.872 | 0.954 | 0.903 |
| central | fedavg | natural3 | patient eps 64 | 0.02 | 5 | 0.836 | 0.820 | 0.938 | 0.873 |
| central | fedavg | natural3 | patient eps 32 | 0.02 | 5 | 0.745 | 0.727 | 0.922 | 0.850 |
| central | fedavg | natural3 | patient eps 16 | 0.02 | 5 | 0.584 | 0.565 | 0.902 | 0.827 |
| central | fedavg | natural3 | patient eps 8 | 0.02 | 5 | 0.353 | 0.337 | 0.890 | 0.823 |
| central | fedavg | natural3 | patient eps 3 | 0.02 | 5 | 0.253 | 0.251 | 0.992 | 0.983 |
| federated | scaffold | dir0.1_k10 | no DP | 0.01 | 5 | 0.815 | 0.790 | 0.901 | 0.800 |
| federated | fedavg | dir0.1_k10 | no DP | 0.01 | 5 | 0.843 | 0.817 | 0.882 | 0.777 |
| federated | fedavg | dir0.1_k10 | eps 8 | 0.003 | 5 | 0.677 | 0.635 | 0.614 | 0.447 |
| federated | scaffold | dir0.1_k10 | eps 8 | 0.003 | 5 | 0.705 | 0.661 | 0.757 | 0.580 |
| federated | scaffold | dir0.1_k10 | eps 3 | 0.003 | 5 | 0.633 | 0.592 | 0.549 | 0.383 |
| federated | fedavg | dir0.1_k10 | eps 3 | 0.003 | 5 | 0.632 | 0.589 | 0.498 | 0.327 |
| federated | fedavg | dir0.1_k10 | eps 1 | 0.003 | 5 | 0.322 | 0.317 | 0.197 | 0.177 |
| federated | scaffold | dir0.5_k10 | no DP | 0.01 | 5 | 0.889 | 0.877 | 0.957 | 0.910 |
| federated | fedavg | dir0.5_k10 | no DP | 0.01 | 5 | 0.878 | 0.870 | 0.970 | 0.937 |
| federated | fedavg | dir0.5_k10 | eps 8 | 0.01 | 5 | 0.662 | 0.624 | 0.528 | 0.377 |
| federated | fedavg | dir0.5_k10 | eps 8 | 0.003 | 5 | 0.718 | 0.669 | 0.771 | 0.577 |
| federated | scaffold | dir0.5_k10 | eps 8 | 0.003 | 5 | 0.713 | 0.666 | 0.784 | 0.597 |
| federated | fedavg | dir0.5_k10 | eps 3 | 0.003 | 5 | 0.626 | 0.590 | 0.550 | 0.403 |
| federated | scaffold | dir0.5_k10 | eps 3 | 0.003 | 5 | 0.621 | 0.582 | 0.558 | 0.403 |
| federated | fedavg | dir0.5_k10 | eps 1 | 0.003 | 5 | 0.341 | 0.331 | 0.326 | 0.287 |
| federated | fedavg | dir0.5_k15 | no DP | 0.01 | 5 | 0.903 | 0.886 | 0.938 | 0.870 |
| federated | fedavg | dir0.5_k15 | eps 8 | 0.003 | 5 | 0.696 | 0.655 | 0.723 | 0.560 |
| federated | fedavg | dir0.5_k15 | eps 8 | 0.01 | 5 | 0.564 | 0.525 | 0.435 | 0.280 |
| federated | fedavg | dir0.5_k15 | eps 3 | 0.003 | 5 | 0.558 | 0.525 | 0.398 | 0.267 |
| federated | fedavg | dir0.5_k3 | no DP | 0.01 | 5 | 0.868 | 0.853 | 0.917 | 0.857 |
| federated | fedavg | dir0.5_k3 | eps 8 | 0.01 | 5 | 0.820 | 0.779 | 0.794 | 0.630 |
| federated | fedavg | dir0.5_k3 | eps 8 | 0.003 | 5 | 0.717 | 0.675 | 0.693 | 0.527 |
| federated | fedavg | dir0.5_k3 | eps 3 | 0.003 | 5 | 0.719 | 0.677 | 0.717 | 0.550 |
| federated | fedavg | dir0.5_k5 | no DP | 0.01 | 5 | 0.906 | 0.892 | 0.946 | 0.887 |
| federated | fedavg | dir0.5_k5 | eps 8 | 0.01 | 5 | 0.773 | 0.731 | 0.779 | 0.613 |
| federated | fedavg | dir0.5_k5 | eps 8 | 0.003 | 5 | 0.735 | 0.694 | 0.790 | 0.627 |
| federated | fedavg | dir0.5_k5 | eps 3 | 0.003 | 5 | 0.708 | 0.669 | 0.744 | 0.587 |
| federated | fedavg | dir5.0_k10 | no DP | 0.01 | 5 | 0.899 | 0.884 | 0.946 | 0.887 |
| federated | scaffold | dir5.0_k10 | no DP | 0.01 | 5 | 0.907 | 0.892 | 0.946 | 0.887 |
| federated | scaffold | dir5.0_k10 | eps 8 | 0.003 | 5 | 0.716 | 0.669 | 0.789 | 0.600 |
| federated | fedavg | dir5.0_k10 | eps 8 | 0.003 | 5 | 0.720 | 0.674 | 0.792 | 0.607 |
| federated | scaffold | dir5.0_k10 | eps 3 | 0.003 | 5 | 0.625 | 0.585 | 0.562 | 0.403 |
| federated | fedavg | dir5.0_k10 | eps 3 | 0.003 | 5 | 0.623 | 0.583 | 0.550 | 0.390 |
| federated | fedavg | dir5.0_k10 | eps 1 | 0.003 | 5 | 0.333 | 0.328 | 0.266 | 0.247 |
| federated | fedavg | iid_k10 | no DP | 0.01 | 5 | 0.904 | 0.891 | 0.954 | 0.903 |
| federated | scaffold | iid_k10 | no DP | 0.01 | 5 | 0.909 | 0.896 | 0.952 | 0.900 |
| federated | fedavg | iid_k10 | eps 8 | 0.003 | 5 | 0.720 | 0.673 | 0.790 | 0.600 |
| federated | scaffold | iid_k10 | eps 8 | 0.003 | 5 | 0.719 | 0.671 | 0.789 | 0.597 |
| federated | scaffold | iid_k10 | eps 3 | 0.003 | 5 | 0.637 | 0.589 | 0.600 | 0.410 |
| federated | fedavg | iid_k10 | eps 3 | 0.003 | 5 | 0.639 | 0.593 | 0.606 | 0.420 |
| federated | fedavg | iid_k10 | eps 1 | 0.003 | 5 | 0.348 | 0.344 | 0.304 | 0.287 |
| federated | scaffold | natural3 | no DP | 0.01 | 5 | 0.900 | 0.879 | 0.920 | 0.837 |
| federated | fedavg | natural3 | no DP | 0.01 | 5 | 0.875 | 0.841 | 0.824 | 0.687 |
| federated | scaffold | natural3 | eps 8 | 0.01 | 5 | 0.825 | 0.790 | 0.837 | 0.697 |
| federated | fedavg | natural3 | eps 8 | 0.01 | 5 | 0.808 | 0.765 | 0.757 | 0.587 |
| federated | fedavg | natural3 | eps 3 | 0.003 | 5 | 0.714 | 0.674 | 0.750 | 0.590 |
| federated | scaffold | natural3 | eps 3 | 0.003 | 5 | 0.713 | 0.671 | 0.770 | 0.603 |
| federated | fedavg | natural3 | eps 1 | 0.003 | 5 | 0.537 | 0.525 | 0.366 | 0.317 |
| federated | fedavg | natural3 | weight noise 0.5 | 0.01 | 5 | 0.259 | 0.254 | 0.173 | 0.153 |
| federated | fedavg | natural3 | weight noise 0.003 | 0.01 | 5 | 0.843 | 0.801 | 0.699 | 0.530 |
| federated | fedavg | natural3 | patient eps 8 | 0.003 | 5 | 0.403 | 0.398 | 0.123 | 0.103 |
| federated | fedavg | udir0.5_k10 | patient eps 8 | 0.01 | 5 | 0.264 | 0.271 | 0.512 | 0.540 |

Contaminated test images (NCC >= 0.95 with another split): 66 of 813; clean test set 747 images. Largest absolute change in balanced accuracy over all 315 runs: 0.072.

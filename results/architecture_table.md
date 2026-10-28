| Model | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| Vanilla LSTM | 2,783,746 | 394,240 | 0.8872 | 0.8561 | 0.8635 | 0.8495 | 8 | 48.6 |
| CIFG-LSTM | 2,685,186 | 295,680 | 0.8893 | 0.8614 | 0.8616 | 0.8612 | 6 | 41.1 |
| LSTMP (P=128) | 2,685,314 | 296,064 | 0.8806 | 0.8526 | 0.8481 | 0.8575 | 8 | 51.2 |

All runs: same splits, seed 42, early stopping on val macro-F1. Device: cuda.

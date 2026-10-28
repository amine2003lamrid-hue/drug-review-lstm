| Vanilla LSTM variant | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline (Adam, lr 1e-3, bs 128, Xavier) | 2,783,746 | 394,240 | 0.8872 | 0.8561 | 0.8635 | 0.8495 | 8 | 48.6 |
| Learning rate 5e-3 | 2,783,746 | 394,240 | 0.8857 | 0.8561 | 0.8582 | 0.8541 | 5 | 48.4 |
| SGD (momentum 0.9, lr 0.05) | 2,783,746 | 394,240 | 0.8632 | 0.8348 | 0.8249 | 0.8477 | 8 | 48.6 |
| Batch size 64 | 2,783,746 | 394,240 | 0.8838 | 0.8552 | 0.8537 | 0.8568 | 8 | 63.5 |
| Batch size 256 | 2,783,746 | 394,240 | 0.8836 | 0.8511 | 0.8596 | 0.8438 | 8 | 42.0 |
| Orthogonal recurrent init | 2,783,746 | 394,240 | 0.8789 | 0.8467 | 0.8508 | 0.8430 | 4 | 48.5 |

All runs: same splits, seed 42, early stopping on val macro-F1. Device: cuda.

| Model | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| LSTMP (P=128) | 2,685,314 | 296,064 | 0.8806 | 0.8526 | 0.8481 | 0.8575 | 8 | 51.2 |
| LSTMP (P=64) | 2,603,202 | 214,080 | 0.8806 | 0.8493 | 0.8523 | 0.8465 | 7 | 50.3 |

All runs: same splits, seed 42, early stopping on val macro-F1. Device: cuda.

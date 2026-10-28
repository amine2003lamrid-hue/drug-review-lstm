# Sentiment Classification of Drug Reviews with Hand-Implemented LSTM Variants

*Advanced NLP: Lab Project on Recurrent Neural Networks, 2026/2027. Team: Mohamed Amine Lamrid, Badr Misraoui (Lab group E). Code: https://github.com/amine2003lamrid-hue/drug-review-lstm*

## 1. Goal of the task

We classify patient reviews from Drugs.com (UCI Drug Review dataset) as **positive** (rating ≥ 7) or **negative** (rating ≤ 4); neutral reviews (5–6) are discarded. The goal is to compare a vanilla LSTM baseline with two non-vanilla cells from the appendix, the **Coupled Input-Forget Gate LSTM (CIFG)** and the **LSTM with recurrent Projection (LSTMP)**, and to study how optimization hyperparameters affect the baseline. All cells are written from scratch with `nn.Linear` as the only linear operation; no built-in recurrent module (`nn.LSTM`, `nn.LSTMCell`, `nn.RNN`) is used, which a unit test checks automatically.

## 2. Exploratory data analysis

The dataset provides an official train file (161,297 reviews) and test file (53,766 reviews). After dropping the 19,185 neutral reviews, 146,941 train and 48,937 test reviews remain. Ratings are strongly polarized (most reviews are 10 or 1), which makes the binary task well defined, but the classes are **imbalanced**: 72.7% positive in train and 72.4% in test. We therefore use **macro-F1** as the model-selection and headline metric, since accuracy alone would reward predicting the majority class (a constant "positive" classifier already reaches ≈72.4% accuracy).

![Rating distribution and class balance](../figures/rating_distribution.png){width=48%} ![Class balance](../figures/class_balance.png){width=48%}

Texts are HTML-unescaped (the raw file contains entities such as `&#039;`), lowercased, stripped of quotes and tokenized with a regular expression (words, contractions and the punctuation `! ? . ,`). Reviews have a mean length of 94 tokens (median 93, 95th percentile 162); the distribution has a sharp drop near 160 tokens, which reflects a length limit on Drugs.com. A maximum sequence length of **200** keeps 99.8% of reviews untruncated. The reviews cover 898 conditions and 3,617 drugs, dominated by birth control, depression and pain.

![Review length](../figures/review_length.png){width=48%} ![Top conditions](../figures/top_conditions.png){width=48%}

**Splits and subsampling.** The official test file is the test set. The official train file is split 90/10 (stratified, seed 42) into train and validation. To fit free-GPU budgets we draw stratified subsamples of **40,000 train / 5,000 validation / 10,000 test** reviews; stratification preserves the original class ratio, so no artificial imbalance is introduced. The vocabulary (min frequency 2, at most 20k types, `<pad>`=0, `<unk>`=1) is built on the training subset only, giving 18,664 tokens.

## 3. Modeling approach

**Architecture.** Every model shares the same pipeline: token ids → `nn.Embedding` (128-d) → one recurrent layer built from the chosen cell (hidden size H = 256) → dropout 0.3 → `nn.Linear` → 2 logits. A generic `RecurrentLayer` unrolls any cell over time and supports stacking with inter-layer dropout. Sequences are padded per batch; at padded positions the state is frozen with a mask, so the representation fed to the classifier is the hidden state at each review's **true last token**. Gate pre-activations are fused (one `nn.Linear` for x, one bias-free `nn.Linear` for h, so each gate has exactly one bias as in the equations), and W_x·x_t is computed for all time steps in a single matrix product before the time loop.

**Vanilla LSTM (baseline)**

```
i_t = σ(W_hi h_{t-1} + W_xi x_t + b_i)     f_t = σ(W_hf h_{t-1} + W_xf x_t + b_f)
o_t = σ(W_ho h_{t-1} + W_xo x_t + b_o)     g_t = tanh(W_hg h_{t-1} + W_xg x_t + b_g)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t            h_t = o_t ⊙ tanh(c_t)
```

**CIFG-LSTM.** The input gate is tied to the forget gate, i_t = 1 − f_t; the rest is unchanged. The cell can only write new content in proportion to what it forgets, which Greff et al. (2015) found to cost little accuracy while removing a full gate, i.e. **25 % of the recurrent parameters** (3 instead of 4 gate blocks).

**LSTMP.** Gates are computed from x_t and the *projected* previous state h_{t-1} ∈ ℝ^P; the cell state stays in ℝ^H:

```
m_t = o_t ⊙ tanh(c_t),     h_t = W_p m_t + b_p,     W_p ∈ ℝ^{P×H}, P < H
```

Because the four recurrent matrices act on a P-dimensional state, their cost drops from 4H² to 4HP, at the price of one H×P projection. With H = 256 and P = 128, LSTMP keeps a 256-d memory with roughly the parameter budget of CIFG. Sak et al. (2014) introduced it to make large acoustic models cheaper. Both variants therefore target **efficiency**: can we keep the vanilla accuracy with fewer parameters?

**Parameter counts** (formulae checked by unit tests; D = 128 input, H = 256, P = 128):

| Cell | Recurrent params | Total params (incl. embedding + classifier) | vs vanilla (recurrent) |
|---|---|---|---|
| Vanilla LSTM | 394,240 | 2,783,746 | +0.0% |
| CIFG-LSTM | 295,680 | 2,685,186 | -25.0% |
| LSTMP (P=128) | 296,064 | 2,685,314 | -24.9% |
| LSTMP (P=64) | 214,080 | 2,603,202 | -45.7% |

**Tests.** `tests/test_cells.py` checks output shapes, the parameter formulae, i_t = 1 − f_t for CIFG, agreement of the vanilla cell with a direct evaluation of the equations, invariance of the final state to extra padding, and that `src/` contains no built-in recurrent module.

## 4. Hyperparameter design choices

**Shared protocol.** Same splits, same seed (42), same seeded length-bucketed batch order, cross-entropy loss, gradient-norm clipping at 1.0, at most 10 epochs with early stopping on validation macro-F1 (patience 3). The checkpoint with the best validation macro-F1 is evaluated once on the test set. Forget-gate biases are initialised to 1 and other biases to 0. **Baseline:** Adam, learning rate 10⁻³, batch size 128, Xavier-uniform initialisation, embedding 128, hidden 256, dropout 0.3. These are standard settings for LSTM text classifiers; embedding and hidden sizes were kept moderate so that each run fits a free Colab GPU session.

**Study (vanilla LSTM, one factor changed at a time):** learning rate 10⁻³ vs 5·10⁻³; Adam vs SGD (momentum 0.9, lr 0.05); batch size 64 / 128 / 256; Xavier-uniform vs orthogonal recurrent weights.

| Vanilla LSTM variant | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline (Adam, lr 1e-3, bs 128, Xavier) | 2,783,746 | 394,240 | 0.8872 | 0.8561 | 0.8635 | 0.8495 | 8 | 48.6 |
| Learning rate 5e-3 | 2,783,746 | 394,240 | 0.8857 | 0.8561 | 0.8582 | 0.8541 | 5 | 48.4 |
| SGD (momentum 0.9, lr 0.05) | 2,783,746 | 394,240 | 0.8632 | 0.8348 | 0.8249 | 0.8477 | 8 | 48.6 |
| Batch size 64 | 2,783,746 | 394,240 | 0.8838 | 0.8552 | 0.8537 | 0.8568 | 8 | 63.5 |
| Batch size 256 | 2,783,746 | 394,240 | 0.8836 | 0.8511 | 0.8596 | 0.8438 | 8 | 42.0 |
| Orthogonal recurrent init | 2,783,746 | 394,240 | 0.8789 | 0.8467 | 0.8508 | 0.8430 | 4 | 48.5 |

![Hyperparameter comparison](../figures/hparam_comparison.png){width=80%}

Relative to the baseline (macro-F1 0.8561): *learning rate*: Learning rate 5e-3 gives 0.8561 (+0.0000), best epoch 5, 48.4 s/epoch. *optimizer*: SGD (momentum 0.9, lr 0.05) gives 0.8348 (-0.0213), best epoch 8, 48.6 s/epoch. *batch size*: Batch size 64 gives 0.8552 (-0.0009), best epoch 8, 63.5 s/epoch. *batch size*: Batch size 256 gives 0.8511 (-0.0050), best epoch 8, 42.0 s/epoch. *initialization*: Orthogonal recurrent init gives 0.8467 (-0.0093), best epoch 4, 48.5 s/epoch. The best setting in this study is **Learning rate 5e-3** (0.8561). The baseline (Adam, lr 1e-3) is already a good setting. SGD with momentum is the only clearly worse choice (-0.021 macro-F1), and it is under-fitted rather than overfitted: after 10 epochs its training loss is still about 0.20 (vs 0.02 with Adam) and its validation loss never turns upward, so it converges more slowly within the same epoch budget. A 5x larger Adam learning rate reaches the same macro-F1 three epochs earlier (best epoch 5 vs 8). Batch size mainly trades time for stability: batch 64 matches the baseline but is about 30% slower per epoch, batch 256 is the fastest but loses 0.005. Orthogonal recurrent initialization brought no gain on the test set (-0.009) and early-stopped after 7 epochs; with reviews of about 100 tokens and Adam, its benefit for gradient flow over long sequences is not needed.

## 5. Final model performance and evaluation

**Architecture comparison** (identical hyperparameters, test set):

| Model | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| Vanilla LSTM | 2,783,746 | 394,240 | 0.8872 | 0.8561 | 0.8635 | 0.8495 | 8 | 48.6 |
| CIFG-LSTM | 2,685,186 | 295,680 | 0.8893 | 0.8614 | 0.8616 | 0.8612 | 6 | 41.1 |
| LSTMP (P=128) | 2,685,314 | 296,064 | 0.8806 | 0.8526 | 0.8481 | 0.8575 | 8 | 51.2 |

![Validation curves](../figures/val_curves.png){width=90%}

The vanilla baseline reaches a test macro-F1 of 0.8561 (accuracy 0.8872). CIFG-LSTM reaches a test macro-F1 of 0.8614 (+0.0053 vs vanilla) with 25.0% fewer recurrent parameters and 41.1 s/epoch vs 48.6 s for vanilla; its best validation epoch is 6. LSTMP (P=128) reaches a test macro-F1 of 0.8526 (-0.0035 vs vanilla) with 24.9% fewer recurrent parameters and 51.2 s/epoch vs 48.6 s for vanilla; its best validation epoch is 8. All three cells land within 0.01 macro-F1 of each other, which a single seed cannot reliably separate, so the main finding is that neither efficiency variant loses measurable accuracy: CIFG removes the input gate (25% fewer recurrent parameters, about 15% faster epochs) and even scores slightly higher, and LSTMP keeps a 256-d cell with a 128-d recurrent state for the same budget at a small cost (-0.0035). The validation curves show the same pattern for every cell: validation loss is lowest around epoch 4 and then rises while training loss falls to about 0.02 and validation macro-F1 stays on a plateau between 0.84 and 0.87. The models overfit and become over-confident after a few epochs; selecting the checkpoint on macro-F1 rather than on loss keeps the better-ranked epochs.

![Parameters vs F1](../figures/params_vs_f1.png){width=90%}

**Projection size (bonus).** Halving the projection from P=128 to P=64 reduces the recurrent parameters from 296,064 to 214,080 (27.7% fewer) and changes test macro-F1 from 0.8526 to 0.8493 (-0.0033).

**Final model.** Selecting on **validation** macro-F1 across all runs, the final model is **cifg** (cell `cifg`, 2,685,186 trainable parameters, val macro-F1 0.8684). On the 10,000-review test set it obtains accuracy **0.8893**, macro-F1 **0.8614**, macro precision 0.8616 and macro recall 0.8612.

![Confusion matrix](../figures/confusion_matrix.png){width=42%}

Recall is 79.8% on negative reviews and 92.4% on positive reviews: 556 negative reviews are predicted positive and 551 positive reviews are predicted negative. Errors are concentrated on the minority class: the two error types are almost equally frequent in absolute terms (556 vs 551), but negatives are only about 28% of the test set, so recall is about 12 points lower on negative reviews than on positive ones. A class-weighted loss or a decision threshold tuned on the validation set (Section 6) directly targets this gap.

## 6. Limitations and potential improvements

- **Context from both directions:** a bidirectional layer, or attention pooling over all hidden states, would avoid relying on the last state alone, which is a bottleneck for long reviews where the verdict appears early.
- **Pretrained embeddings** (GloVe, fastText or domain embeddings trained on medical text) would help with rare drug names and side-effect vocabulary that a 40k-review vocabulary covers poorly.
- **Other cells:** the Working Memory Connections (WMC) cell, which lets the cell state drive the gates, is a natural fourth variant; a stacked 2-layer model is already supported by the code.
- **Class imbalance:** class-weighted loss or threshold tuning on the validation set could raise recall on the negative class without resampling.
- **Task framing:** a 3-class setting keeping neutral 5–6 reviews, or rating regression, would be more informative and harder.
- **More data and search:** training on the full 132k training reviews and a systematic search (several seeds, joint lr/batch-size grid) would give more reliable conclusions; our results come from a single seed, so differences of a few thousandths of F1 should not be over-interpreted.

**References.** Greff et al., *LSTM: A Search Space Odyssey*, 2015. Sak et al., *Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling*, Interspeech 2014. Landi et al., *Working Memory Connections for LSTM*, Neural Networks, 2021. Gräßer et al., *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning*, 2018 (dataset).

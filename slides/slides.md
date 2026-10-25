---
marp: true
paginate: true
size: 16:9
style: |
  section { font-size: 24px; color: #0b0b0b; }
  h1 { color: #2a78d6; font-size: 38px; }
  h2 { color: #2a78d6; font-size: 32px; }
  table { font-size: 17px; }
  code { font-size: 18px; }
  img { display: block; margin: 0 auto; }
---

# Drug Review Sentiment with Hand-Built LSTM Cells

Vanilla LSTM vs CIFG vs LSTMP, implemented from scratch in PyTorch

Mohamed Amine Lamrid, Badr Misraoui (Lab group E)
Advanced NLP, Lab Project on RNNs, 2026/2027

---

## 1. Goal

- **Task:** binary sentiment of Drugs.com patient reviews (UCI Drug Review)
  - rating ≥ 7 → positive, ≤ 4 → negative, 5–6 dropped
- **Question:** do non-vanilla LSTM cells (**CIFG**, **LSTMP**) keep accuracy with fewer parameters?
- **Plus:** how do learning rate, optimizer, batch size and initialization affect the baseline?
- Constraint: every cell written with `nn.Linear` only, no `nn.LSTM`

---

## 2. EDA: ratings and classes

![w:520](../figures/rating_distribution.png) ![w:520](../figures/class_balance.png)

146,941 train / 48,937 test reviews after dropping neutrals. **72.7% positive**, so we select and report with **macro-F1**

---

## 2. EDA: lengths and topics

![w:520](../figures/review_length.png) ![w:420](../figures/top_conditions.png)

Median 93 tokens; max_len = 200 keeps 99.8% of reviews whole. Stratified subsample: 40,000 / 5,000 / 10,000

---

## 3. Modeling approach

Embedding(128) → RecurrentLayer(cell, H=256) → Dropout(0.3) → Linear → 2 logits
Classifier reads the state at the **true last token** (padding masked)

```
Vanilla: i,f,o = σ(W_h h + W_x x + b)   g = tanh(...)   c = f⊙c + i⊙g   h = o⊙tanh(c)
CIFG:    i = 1 − f                      → 3 gate blocks instead of 4
LSTMP:   m = o⊙tanh(c),  h = W_p m + b_p  (h ∈ ℝ^P, P=128 < H=256)
```

| Cell | Recurrent params | Total params (incl. embedding + classifier) | vs vanilla (recurrent) |
|---|---|---|---|
| Vanilla LSTM | 394,240 | 2,783,746 | +0.0% |
| CIFG-LSTM | 295,680 | 2,685,186 | -25.0% |
| LSTMP (P=128) | 296,064 | 2,685,314 | -24.9% |
| LSTMP (P=64) | 214,080 | 2,603,202 | -45.7% |

---

## 4. Hyperparameters: design and study

Baseline: Adam, lr 1e-3, batch 128, Xavier, clip 1.0, ≤10 epochs, early stop on val macro-F1 (patience 3), seed 42

**[TO FILL AFTER RUN]** *(figure hparam_comparison.png)*

**[TO FILL AFTER RUN]**

---

## 5. Architecture comparison

**[TO FILL AFTER RUN]**

**[TO FILL AFTER RUN]** *(figure params_vs_f1.png)*

---

## 5. Training curves and final model

**[TO FILL AFTER RUN]** *(figure val_curves.png)* **[TO FILL AFTER RUN]** *(figure confusion_matrix.png)*

**[TO FILL AFTER RUN]**

---

## 6. Limitations and improvements

- Single seed: small F1 gaps are within noise. Run several seeds
- Last-state bottleneck: **bidirectional** layer or **attention** pooling
- **Pretrained embeddings** (GloVe / fastText / medical) for rare drug names
- Try the **WMC** cell (cell state drives the gates)
- Imbalance: class weights or threshold tuning for negative-class recall
- 3-class task with neutral reviews; hyperparameter search on full data

---

# Thank you

Code: https://github.com/amine2003lamrid-hue/drug-review-lstm

Questions?

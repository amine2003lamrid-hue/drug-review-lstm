# Drug Review Sentiment with Hand-Implemented LSTM Cells

Advanced NLP lab project (Recurrent Neural Networks, 2026/2027). Binary sentiment classification of
[UCI Drug Review (Drugs.com)](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com)
patient reviews (rating ≥ 7 → positive, ≤ 4 → negative, 5–6 dropped), comparing three LSTM cells written
from scratch with `nn.Linear` (no `nn.LSTM` / `nn.LSTMCell` / `nn.RNN`):

| Cell | Idea | Recurrent params (D=128, H=256) |
|---|---|---|
| Vanilla LSTM | baseline, Eq. 1–6 | 394,240 |
| CIFG | coupled gates, i_t = 1 − f_t | 295,680 (−25%) |
| LSTMP | recurrent projection h_t = W_p m_t + b_p, P=128 | 296,064 (−25%) |

Plus a one-factor-at-a-time hyperparameter study on the vanilla model (learning rate, optimizer, batch size,
weight initialization) and a bonus LSTMP projection-size comparison (P=64 vs 128).

**Team (Lab group E):** Mohamed Amine Lamrid, Badr Misraoui

## Results

<!-- RESULTS_START -->
Architecture comparison (test set):

| Model | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| Vanilla LSTM | 2,783,746 | 394,240 | 0.8872 | 0.8561 | 0.8635 | 0.8495 | 8 | 48.6 |
| CIFG-LSTM | 2,685,186 | 295,680 | 0.8893 | 0.8614 | 0.8616 | 0.8612 | 6 | 41.1 |
| LSTMP (P=128) | 2,685,314 | 296,064 | 0.8806 | 0.8526 | 0.8481 | 0.8575 | 8 | 51.2 |

Hyperparameter study (vanilla LSTM, test set):

| Vanilla LSTM variant | Trainable params | Recurrent params | Test acc. | Test macro-F1 | Precision | Recall | Best epoch | Time/epoch (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline (Adam, lr 1e-3, bs 128, Xavier) | 2,783,746 | 394,240 | 0.8872 | 0.8561 | 0.8635 | 0.8495 | 8 | 48.6 |
| Learning rate 5e-3 | 2,783,746 | 394,240 | 0.8857 | 0.8561 | 0.8582 | 0.8541 | 5 | 48.4 |
| SGD (momentum 0.9, lr 0.05) | 2,783,746 | 394,240 | 0.8632 | 0.8348 | 0.8249 | 0.8477 | 8 | 48.6 |
| Batch size 64 | 2,783,746 | 394,240 | 0.8838 | 0.8552 | 0.8537 | 0.8568 | 8 | 63.5 |
| Batch size 256 | 2,783,746 | 394,240 | 0.8836 | 0.8511 | 0.8596 | 0.8438 | 8 | 42.0 |
| Orthogonal recurrent init | 2,783,746 | 394,240 | 0.8789 | 0.8467 | 0.8508 | 0.8430 | 4 | 48.5 |

Selecting on **validation** macro-F1 across all runs, the final model is **cifg** (cell `cifg`, 2,685,186 trainable parameters, val macro-F1 0.8684). On the 10,000-review test set it obtains accuracy **0.8893**, macro-F1 **0.8614**, macro precision 0.8616 and macro recall 0.8612.

Full discussion: `report/report.md`; figures in `figures/`.
<!-- RESULTS_END -->

## How to run

### Colab (recommended: needs a GPU)

Open `notebooks/run_on_colab.ipynb` in Colab, select a **T4 GPU** runtime, set `REPO_URL`, and run all cells.
It runs the tests, EDA, all experiments, the tables and figures, and rebuilds the report and slides, then
downloads a `results_bundle.zip`. Finished runs are skipped on re-execution, so a disconnected session can
resume.

### Local

```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # unit tests for the cells
python -m src.eda                                           # downloads data, figures/ + results/eda_summary.md
python -m src.run_experiments --config configs/architecture.yaml configs/hparams.yaml configs/bonus.yaml --quick   # sanity check
python -m src.run_experiments --config configs/architecture.yaml   # A: vanilla vs CIFG vs LSTMP
python -m src.run_experiments --config configs/hparams.yaml        # B: hyperparameter study
python -m src.run_experiments --config configs/bonus.yaml          # C: LSTMP P=64 vs P=128
python -m src.make_report_assets                            # results/*_table.md + figures/*.png
python -m src.build_docs   # team names and repo URL are the defaults; override with --team / --repo-url
tensorboard --logdir runs
```

The time loop is written in Python, so CPU training is slow (about 2.4 s per batch of 128 on a laptop CPU,
about 12 min per epoch on 40k reviews). Use a GPU for the full study. `--quick` uses 2k/500/500 reviews and
1 epoch; its outputs go to `results/quick/` and never mix with the real results.

## Protocol

- Official test file = test set; official train file split 90/10 (stratified, seed 42) into train/val.
- Stratified subsample 40k / 5k / 10k (class ratio ~72.7% positive preserved; configurable in `configs/base.yaml`, `null` = full).
- Vocab on train only: min_freq 2, ≤ 20k tokens, `<pad>`=0, `<unk>`=1. Max length 200 tokens (covers 99.8% of reviews).
- Embedding 128 → recurrent layer (H=256) → dropout 0.3 → linear → 2 logits. The classifier reads the hidden state at each review's true last token (padding is masked).
- Adam lr 1e-3, batch 128, Xavier init, grad clip 1.0, ≤ 10 epochs, early stopping on val macro-F1 (patience 3), seed 42, same seeded batch order for every model.
- Metrics: accuracy, macro-F1, macro precision/recall, confusion matrix, time per epoch, trainable parameters.

## Repository structure

```
configs/            base.yaml (shared defaults) + architecture / hparams / bonus experiment lists
src/data.py         download, cleaning, splits, stratified subsampling, vocab, bucketed batching
src/cells.py        VanillaLSTMCell, CIFGCell, LSTMPCell, RecurrentLayer (nn.Linear only)
src/model.py        SentimentClassifier
src/train.py        training loop, early stopping, metrics, TensorBoard, JSON results, checkpoints
src/run_experiments.py   runs a list of configs (resumable, --quick)
src/eda.py          EDA figures and summary
src/make_report_assets.py   tables and figures from results/*.json
src/build_docs.py   fills report/report.md and slides/slides.md from results, exports .docx/.pdf
tests/test_cells.py shapes, parameter formulae, CIFG coupling, padding, no nn.LSTM/nn.RNN in src/
notebooks/run_on_colab.ipynb   full run on Colab
report/  slides/    generated report (md + docx/pdf) and Marp slides
results/  figures/  metrics JSON, tables, plots      (data/, runs/, checkpoints/ are gitignored)
```

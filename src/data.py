"""Download, clean, split, subsample and vectorize the UCI Drug Review dataset."""
import html
import io
import os
import re
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
URL_FALLBACK = "https://archive.ics.uci.edu/static/public/462/drug+review+dataset+drugs+com.zip"
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?.,]")


def download(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    have = [f for f in os.listdir(data_dir) if f.startswith("drugsCom")]
    if len(have) >= 2:
        return
    print(f"Downloading {URL} ...")
    try:
        with urllib.request.urlopen(URL) as r:
            payload = r.read()
    except Exception:
        with urllib.request.urlopen(URL_FALLBACK) as r:
            payload = r.read()
    outer = zipfile.ZipFile(io.BytesIO(payload))
    for name in outer.namelist():
        base = os.path.basename(name)
        if base.startswith("drugsCom") and base.endswith((".tsv", ".csv")):
            with open(os.path.join(data_dir, base), "wb") as f:
                f.write(outer.read(name))
        elif name.endswith(".zip"):  # some versions nest a second zip
            inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
            for n in inner.namelist():
                b = os.path.basename(n)
                if b.startswith("drugsCom"):
                    with open(os.path.join(data_dir, b), "wb") as f:
                        f.write(inner.read(n))


def _read(data_dir, split):
    for ext, sep in ((".tsv", "\t"), (".csv", ",")):
        path = os.path.join(data_dir, f"drugsCom{split}_raw{ext}")
        if os.path.exists(path):
            return pd.read_csv(path, sep=sep)
    raise FileNotFoundError(f"drugsCom{split}_raw not found in {data_dir}")


def clean(text):
    text = html.unescape(str(text)).lower()
    text = text.replace('"', " ").replace("“", " ").replace("”", " ")
    return text.strip()


def tokenize(text):
    return TOKEN_RE.findall(clean(text))


def load_raw(data_dir="data"):
    """Return (train_df, test_df) with columns review, label, rating, condition, drugName (5-6 dropped)."""
    download(data_dir)
    out = []
    for split in ("Train", "Test"):
        df = _read(data_dir, split)
        df = df.dropna(subset=["review", "rating"])
        df = df[(df.rating >= 7) | (df.rating <= 4)].copy()
        df["label"] = (df.rating >= 7).astype(int)
        out.append(df.reset_index(drop=True))
    return out


def _stratified_take(df, n, seed):
    if n is None or n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df.label, random_state=seed)
    return sub.reset_index(drop=True)


def make_splits(cfg):
    """Official test file = test; train split 90/10 stratified; then stratified subsampling."""
    train_df, test_df = load_raw(cfg.get("data_dir", "data"))
    seed = cfg.get("seed", 42)
    tr, va = train_test_split(train_df, test_size=0.1, stratify=train_df.label, random_state=seed)
    tr = _stratified_take(tr.reset_index(drop=True), cfg.get("n_train"), seed)
    va = _stratified_take(va.reset_index(drop=True), cfg.get("n_val"), seed)
    te = _stratified_take(test_df, cfg.get("n_test"), seed)
    return tr, va, te


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=20000):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 2]
        self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks]


class ReviewDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.items = []
        for text, y in zip(df.review, df.label):
            ids = vocab.encode(tokenize(text))[:max_len] or [1]
            self.items.append((ids, int(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    lengths = torch.tensor([len(x) for x, _ in batch])
    x = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
    y = torch.tensor([y for _, y in batch])
    return x, lengths, y


def build_data(cfg):
    """Returns vocab and dict of datasets; cached per subset config to avoid re-tokenizing."""
    tr, va, te = make_splits(cfg)
    vocab = Vocab((tokenize(t) for t in tr.review), cfg.get("min_freq", 2), cfg.get("max_vocab", 20000))
    ml = cfg.get("max_len", 200)
    ds = {k: ReviewDataset(d, vocab, ml) for k, d in (("train", tr), ("val", va), ("test", te))}
    return vocab, ds


class BucketBatchSampler(torch.utils.data.Sampler):
    """Shuffles, groups examples of similar length into batches (less padding = fewer time steps),
    then shuffles the batches. Seeded, so every model sees the same batch sequence."""

    def __init__(self, lengths, batch_size, seed=42, pool=50):
        self.lengths = np.asarray(lengths)
        self.batch_size, self.pool = batch_size, pool
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        chunk = self.batch_size * self.pool
        batches = []
        for s in range(0, len(idx), chunk):
            part = idx[s:s + chunk]
            part = part[np.argsort(self.lengths[part], kind="stable")]
            batches += [part[j:j + self.batch_size].tolist() for j in range(0, len(part), self.batch_size)]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def _sorted_batches(ds, bs):
    order = np.argsort([len(x) for x, _ in ds.items])
    return [order[i:i + bs].tolist() for i in range(0, len(order), bs)]


def make_loaders(ds, batch_size, seed=42):
    train_lengths = [len(x) for x, _ in ds["train"].items]
    return {
        "train": DataLoader(ds["train"], batch_sampler=BucketBatchSampler(train_lengths, batch_size, seed),
                            collate_fn=collate),
        "val": DataLoader(ds["val"], batch_sampler=_sorted_batches(ds["val"], 256), collate_fn=collate),
        "test": DataLoader(ds["test"], batch_sampler=_sorted_batches(ds["test"], 256), collate_fn=collate),
    }


if __name__ == "__main__":
    tr, va, te = make_splits({"n_train": 40000, "n_val": 5000, "n_test": 10000})
    for n, d in (("train", tr), ("val", va), ("test", te)):
        print(n, len(d), "pos ratio %.3f" % d.label.mean())

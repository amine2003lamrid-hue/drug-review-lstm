"""Training / evaluation loop for one experiment run."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter

from .data import make_loaders
from .model import SentimentClassifier, count_params


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    ys, preds, total_loss, n = [], [], 0.0, 0
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        total_loss += loss_fn(logits, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu())
        preds.append(logits.argmax(-1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
        "precision": precision_score(y, p, average="macro", zero_division=0),
        "recall": recall_score(y, p, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


def make_optimizer(cfg, params):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg.get("momentum", 0.9))
    return torch.optim.Adam(params, lr=cfg["lr"])


def train_run(cfg, vocab, ds, out_dir="results", ckpt_dir="checkpoints", log_dir="runs"):
    name = cfg["name"]
    set_seed(cfg["seed"])
    device = get_device()
    loaders = make_loaders(ds, cfg["batch_size"], cfg["seed"])
    model = SentimentClassifier(len(vocab), cfg["cell"], cfg["embed_dim"], cfg["hidden_size"], cfg["num_layers"],
                                cfg.get("proj_size", 128), cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    n_rnn_params = count_params(model.rnn)
    opt = make_optimizer(cfg, model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(os.path.join(log_dir, name))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"{name}.pt")

    print(f"[{name}] cell={cfg['cell']} params={n_params:,} (recurrent {n_rnn_params:,}) device={device}")
    history, best_f1, best_epoch, bad = [], -1.0, 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tr_loss, n = time.time(), 0.0, 0
        for x, lengths, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            tr_loss += loss.item() * len(y)
            n += len(y)
        epoch_time = time.time() - t0
        val = evaluate(model, loaders["val"], device, loss_fn)
        rec = {"epoch": epoch, "train_loss": tr_loss / n, "val_loss": val["loss"], "val_accuracy": val["accuracy"],
               "val_macro_f1": val["macro_f1"], "epoch_time_s": epoch_time}
        history.append(rec)
        for k, v in rec.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)
        print(f"[{name}] epoch {epoch} train_loss={rec['train_loss']:.4f} val_loss={val['loss']:.4f} "
              f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} ({epoch_time:.0f}s)")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = val["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"[{name}] early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test = evaluate(model, loaders["test"], device, loss_fn)
    result = {
        "name": name, "config": cfg, "device": str(device),
        "trainable_params": n_params, "recurrent_params": n_rnn_params,
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "mean_epoch_time_s": float(np.mean([h["epoch_time_s"] for h in history])),
        "history": history, "test": test,
    }
    writer.add_hparams({k: v for k, v in cfg.items() if isinstance(v, (int, float, str))},
                       {"hparam/test_macro_f1": test["macro_f1"], "hparam/test_accuracy": test["accuracy"]},
                       run_name=".")
    writer.close()
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{name}] TEST acc={test['accuracy']:.4f} macro_f1={test['macro_f1']:.4f}")
    return result

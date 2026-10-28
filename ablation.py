"""
ablation.py
===========
Ablation study for the Hybrid Char-CNN + Word-Embedding BiLSTM POS tagger.

Studies:
  1. Embedding mix  : char-only / word-only / various (word_dim, char_dim) splits
  2. Word dropout   : with / without frequency-based word dropout
  3. LSTM hidden dim: 64 / 128 / 256 / 512

Each configuration is trained for up to ABLATION_EPOCHS epochs (early-stop
patience = 3), then evaluated on dev.sents with full per-token, IV/OOV, and
sentence-level accuracy.

Usage:
    python3 ablation.py train.tags dev.sents dev.tags [--epochs N]

Results are printed as a table and saved to ablation_results.csv.
"""

import sys
import os
import csv
import copy
import time
import random
import pickle
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── re-use building blocks from build_tagger ─────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tagger import (
    parse_tagged_file, build_vocab,
    POSDataset, collate_fn,
    BiLSTMTagger,
    PAD_TOKEN, UNK_TOKEN,
    WORD_DROPOUT_ALPHA as DEFAULT_ALPHA,
)

# ── Global seeds & device ─────────────────────────────────────────────────────
SEED = 42

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[ablation] Device: {DEVICE}")

# ── Ablation hyper-params ─────────────────────────────────────────────────────
ABLATION_EPOCHS = 15          # max epochs per config
PATIENCE        = 3           # early-stop patience
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
MAX_GRAD_NORM   = 5.0
CHAR_EMB_DIM    = 30          # internal char embedding dim (kept fixed)
MIN_WORD_FREQ   = 1


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    name:               str
    word_dim:           int           # 0 → disable word embedding
    char_cnn_out_dim:   int           # 0 → disable char CNN
    lstm_hidden:        int   = 256
    lstm_layers:        int   = 2
    dropout:            float = 0.5
    word_dropout_alpha: float = DEFAULT_ALPHA   # 0 → disabled


def _char_filters(char_cnn_out_dim: int) -> Tuple[List[int], int]:
    """
    Given a desired char-CNN output dimension, return (filter_widths,
    filters_per_width).  We always use widths [2,3,4]; filters_per_width =
    char_cnn_out_dim // 3 (rounded down, minimum 1).
    """
    widths = [2, 3, 4]
    fpw = max(1, char_cnn_out_dim // len(widths))
    return widths, fpw


# ── All configurations ────────────────────────────────────────────────────────

def build_configs() -> List[AblationConfig]:
    configs = []

    # ── 1. Embedding mix (word_dim + char_cnn_out_dim ≤ 100) ──────────────────
    # Char-only (0 + 99 — 99 = 33×3)
    configs.append(AblationConfig(
        name="char_only (0+99)",
        word_dim=0, char_cnn_out_dim=99,
    ))
    # Word-only (100 + 0)
    configs.append(AblationConfig(
        name="word_only (100+0)",
        word_dim=100, char_cnn_out_dim=0,
    ))
    # Mixtures
    for wd, cd in [(30, 69), (50, 51), (70, 30), (85, 15)]:
        configs.append(AblationConfig(
            name=f"mix ({wd}+{cd})",
            word_dim=wd, char_cnn_out_dim=cd,
        ))

    # ── 2. Word dropout  (70+30 base, vary alpha) ─────────────────────────────
    configs.append(AblationConfig(
        name="no_word_dropout (70+30)",
        word_dim=70, char_cnn_out_dim=30,
        word_dropout_alpha=0.0,
    ))
    # with dropout already covered by mix(70+30) above → add labelled copy
    configs.append(AblationConfig(
        name="word_dropout_α=0.25 (70+30)",
        word_dim=70, char_cnn_out_dim=30,
        word_dropout_alpha=0.25,
    ))
    configs.append(AblationConfig(
        name="word_dropout_α=0.50 (70+30)",
        word_dim=70, char_cnn_out_dim=30,
        word_dropout_alpha=0.50,
    ))

    # ── 3. LSTM hidden dim (70+30 base, α=0.25) ───────────────────────────────
    for h in [64, 128, 256, 512]:
        configs.append(AblationConfig(
            name=f"lstm_hidden={h}",
            word_dim=70, char_cnn_out_dim=30,
            lstm_hidden=h,
        ))

    return configs


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(cfg: AblationConfig,
                word_vocab_size: int,
                char_vocab_size: int,
                num_tags: int) -> BiLSTMTagger:
    """
    BiLSTMTagger with word_dim / char_cnn_out_dim = 0 handled via subclassing:
    we monkey-patch the existing BiLSTMTagger to support zero-dim components.
    """
    # char filter config
    if cfg.char_cnn_out_dim > 0:
        widths, fpw = _char_filters(cfg.char_cnn_out_dim)
    else:
        widths, fpw = [2, 3, 4], 1   # unused, but avoid zero

    model = BiLSTMTagger(
        word_vocab_size=word_vocab_size,
        char_vocab_size=char_vocab_size,
        num_tags=num_tags,
        word_emb_dim=max(cfg.word_dim, 1),      # avoid 0-dim embedding
        char_emb_dim=CHAR_EMB_DIM,
        char_filter_widths=widths,
        char_filters_per_width=fpw,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    )

    # ── patch forward to mask out disabled components ──────────────────────────
    word_dim       = cfg.word_dim
    char_cnn_dim   = cfg.char_cnn_out_dim
    actual_char_dim = fpw * len(widths)

    orig_forward = model.forward

    def patched_forward(word_ids, char_ids, lengths):
        parts = []
        if word_dim > 0:
            parts.append(model.word_emb(word_ids))          # (B,T,word_dim)
        if char_cnn_dim > 0:
            parts.append(model.char_cnn(char_ids))          # (B,T,char_cnn_dim)

        x = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]
        x = model.dropout(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = model.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)

        lstm_out = model.dropout(lstm_out)
        return model.fc(lstm_out)

    # Replace LSTM input size to match actual concatenation dimension
    actual_input = (word_dim if word_dim > 0 else 0) + \
                   (actual_char_dim if char_cnn_dim > 0 else 0)
    model.lstm = nn.LSTM(
        input_size=actual_input,
        hidden_size=cfg.lstm_hidden,
        num_layers=cfg.lstm_layers,
        bidirectional=True,
        batch_first=True,
        dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
    )
    model.fc = nn.Linear(cfg.lstm_hidden * 2, num_tags)

    import types
    model.forward = types.MethodType(lambda self, w, c, l: patched_forward(w, c, l), model)

    return model


# ── Train one config ──────────────────────────────────────────────────────────

def train_config(cfg: AblationConfig,
                 train_sents, val_sents,
                 word2idx, word_freq, char2idx, tag2idx, idx2tag,
                 n_epochs: int = 15):
    set_seed()

    train_ds = POSDataset(train_sents, word2idx, char2idx, tag2idx,
                          word_freq=word_freq,
                          word_dropout_alpha=cfg.word_dropout_alpha)
    val_ds   = POSDataset(val_sents,   word2idx, char2idx, tag2idx,
                          word_freq=word_freq, word_dropout_alpha=0.0)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = build_model(cfg, len(word2idx), len(char2idx), len(tag2idx)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_acc  = -1.0
    best_state    = None
    no_improve    = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        pbar = tqdm(train_loader,
                    desc=f"  ep {epoch:2d}/{n_epochs}",
                    unit="batch", leave=False, dynamic_ncols=True)
        for word_ids, char_ids, tag_ids, lengths in pbar:
            word_ids = word_ids.to(DEVICE)
            char_ids = char_ids.to(DEVICE)
            tag_ids  = tag_ids.to(DEVICE)
            lengths  = lengths.to(DEVICE)
            optimizer.zero_grad()
            logits = model(word_ids, char_ids, lengths)
            B, T, C = logits.shape
            loss = criterion(logits.view(B*T, C), tag_ids.view(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for word_ids, char_ids, tag_ids, lengths in tqdm(
                    val_loader, desc="  validating", unit="batch",
                    leave=False, dynamic_ncols=True):
                word_ids = word_ids.to(DEVICE)
                char_ids = char_ids.to(DEVICE)
                tag_ids  = tag_ids.to(DEVICE)
                lengths  = lengths.to(DEVICE)
                logits = model(word_ids, char_ids, lengths)
                B, T, C = logits.shape
                preds    = logits.view(B*T, C).argmax(1)
                tags_flat = tag_ids.view(B*T)
                mask = (tags_flat != -1)
                correct += (preds[mask] == tags_flat[mask]).sum().item()
                total   += mask.sum().item()
        val_acc = correct / max(total, 1)
        elapsed = time.time() - t0
        print(f"    ep {epoch:2d}/{n_epochs} val_acc={val_acc:.4f} ({elapsed:.0f}s)")

        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    → early stop (best={best_val_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, n_params, best_val_acc


# ── Detailed evaluation ────────────────────────────────────────────────────────

def detailed_eval(model, dev_sents: List[List[str]], dev_tags: List[List[str]],
                  word2idx: Dict, char2idx: Dict, idx2tag: Dict,
                  word_freq: Dict) -> Dict:
    """
    Returns a dict with:
      - overall_acc
      - iv_acc          (tokens whose surface form is in word2idx)
      - oov_acc         (tokens NOT in word2idx)
      - sentence_acc    (fraction of sentences with ALL tokens correct)
    """
    unk_word = word2idx[UNK_TOKEN]
    unk_char = char2idx.get("<UNK_C>", 1)
    model.eval()

    total_tok = iv_correct = iv_total = oov_correct = oov_total = 0
    sent_correct = 0

    with torch.no_grad():
        for words, gold_tags in zip(dev_sents, dev_tags):
            T = len(words)
            word_ids = [word2idx.get(w, unk_word) for w in words]
            is_oov   = [word2idx.get(w, unk_word) == unk_word for w in words]

            # char ids
            max_char = max(len(w) for w in words)
            char_ids = []
            for w in words:
                cids = [char2idx.get(c, unk_char) for c in w]
                cids = cids[:max_char] + [0] * max(0, max_char - len(cids))
                char_ids.append(cids)

            wt = torch.tensor([word_ids], dtype=torch.long).to(DEVICE)
            ct = torch.tensor([char_ids], dtype=torch.long).to(DEVICE)
            lt = torch.tensor([T],        dtype=torch.long).to(DEVICE)

            logits = model(wt, ct, lt)          # (1, T, num_tags)
            preds  = logits.argmax(2)[0].tolist()
            pred_tags = [idx2tag[p] for p in preds]

            all_correct = True
            for w, pred, gold, oov in zip(words, pred_tags, gold_tags, is_oov):
                total_tok += 1
                match = (pred == gold)
                if not match:
                    all_correct = False
                if oov:
                    oov_total   += 1
                    oov_correct += int(match)
                else:
                    iv_total    += 1
                    iv_correct  += int(match)

            if all_correct:
                sent_correct += 1

    return {
        "overall_acc":  total_tok and (iv_correct + oov_correct) / total_tok,
        "iv_acc":       iv_total  and iv_correct  / iv_total,
        "oov_acc":      oov_total and oov_correct / oov_total,
        "sentence_acc": sent_correct / max(len(dev_sents), 1),
        "iv_total":     iv_total,
        "oov_total":    oov_total,
    }


# ── Parse dev files (with tags) ───────────────────────────────────────────────

def parse_dev_tagged(sents_path: str, tags_path: str):
    """Return (list-of-word-lists, list-of-tag-lists) from separate files."""
    with open(sents_path) as f:
        dev_sents = [line.strip().split() for line in f if line.strip()]
    with open(tags_path) as f:
        raw = [line.strip().split() for line in f if line.strip()]
    dev_tags = []
    for tok_list in raw:
        tags = [tok.rsplit("/", 1)[1] for tok in tok_list]
        dev_tags.append(tags)
    return dev_sents, dev_tags


# ── Pretty-print results table ────────────────────────────────────────────────

def print_table(rows: List[Dict]):
    cols = ["name", "params", "best_val_acc",
            "overall_acc", "iv_acc", "oov_acc", "sentence_acc",
            "iv_total", "oov_total", "train_time_s"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows))
              for c in cols}

    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    hdr = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    print("\n" + sep)
    print(hdr)
    print(sep)
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) + " |"
        print(line)
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_tags")
    parser.add_argument("dev_sents")
    parser.add_argument("dev_tags")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Max epochs per config (default 15)")
    args = parser.parse_args()

    n_epochs = args.epochs

    # ── Load & split data ──────────────────────────────────────────────────────
    print("[ablation] Parsing training data …")
    all_sents = parse_tagged_file(args.train_tags)
    set_seed()
    random.shuffle(all_sents)
    split       = max(1, int(0.95 * len(all_sents)))
    train_sents = all_sents[:split]
    val_sents   = all_sents[split:]
    print(f"[ablation] train={len(train_sents)}  val={len(val_sents)}")

    word2idx, word_freq, char2idx, tag2idx, idx2tag = build_vocab(train_sents)
    print(f"[ablation] vocab: words={len(word2idx)} chars={len(char2idx)} tags={len(tag2idx)}")

    dev_sents, dev_tags = parse_dev_tagged(args.dev_sents, args.dev_tags)
    print(f"[ablation] dev sentences: {len(dev_sents)}\n")

    configs = build_configs()
    rows    = []

    cfg_bar = tqdm(enumerate(configs, 1), total=len(configs),
                   desc="Ablation configs", unit="cfg", dynamic_ncols=True)
    for i, cfg in cfg_bar:
        print(f"{'='*70}")
        print(f"[{i}/{len(configs)}] Config: {cfg.name}")
        print(f"  word_dim={cfg.word_dim}  char_cnn_out={cfg.char_cnn_out_dim}  "
              f"lstm_hidden={cfg.lstm_hidden}  dropout_α={cfg.word_dropout_alpha}")
        t0 = time.time()

        model, n_params, best_val_acc = train_config(
            cfg, train_sents, val_sents,
            word2idx, word_freq, char2idx, tag2idx, idx2tag,
            n_epochs=n_epochs)

        metrics = detailed_eval(
            model, dev_sents, dev_tags,
            word2idx, char2idx, idx2tag, word_freq)

        elapsed = time.time() - t0
        row = {
            "name":         cfg.name,
            "params":       n_params,
            "best_val_acc": f"{best_val_acc:.4f}",
            "overall_acc":  f"{metrics['overall_acc']:.4f}",
            "iv_acc":       f"{metrics['iv_acc']:.4f}",
            "oov_acc":      f"{metrics['oov_acc']:.4f}",
            "sentence_acc": f"{metrics['sentence_acc']:.4f}",
            "iv_total":     metrics["iv_total"],
            "oov_total":    metrics["oov_total"],
            "train_time_s": f"{elapsed:.0f}",
        }
        rows.append(row)
        cfg_bar.set_postfix(last_acc=row["overall_acc"])
        print(f"  → overall={row['overall_acc']}  IV={row['iv_acc']}  "
              f"OOV={row['oov_acc']}  sent={row['sentence_acc']}  "
              f"time={elapsed:.0f}s")

        # Free GPU memory
        del model
        if DEVICE.type == "mps":
            torch.mps.empty_cache()
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # ── Print & save results ───────────────────────────────────────────────────
    print_table(rows)

    out_csv = "ablation_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ablation] Results saved to {out_csv}")


if __name__ == "__main__":
    main()

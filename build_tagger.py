"""
build_tagger.py
===============
Train a Hybrid Char-CNN + Word-Embedding BiLSTM POS tagger on Penn Treebank data.

Usage:
    python3 build_tagger.py train.tags

Saves artefacts to the current working directory:
    tagger_model.pt   - model weights
    tagger_vocab.pkl  - word/char/tag vocabularies + hyper-params
"""

import sys
import os
import pickle
import random
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
# Embedding dimensions  (word_dim + char_cnn_dim ≤ 100)
WORD_EMB_DIM   = 70          # word-level embedding dimension
CHAR_EMB_DIM   = 30          # character embedding fed into CNN
CHAR_CNN_FILTERS_PER_WIDTH = 10   # 10 filters × 3 widths = 30-D char vector
CHAR_CNN_WIDTHS = [2, 3, 4]
CHAR_CNN_OUT_DIM = CHAR_CNN_FILTERS_PER_WIDTH * len(CHAR_CNN_WIDTHS)  # 30

LSTM_HIDDEN_DIM = 256        # BiLSTM hidden size per direction
LSTM_LAYERS     = 2
DROPOUT         = 0.5

# Training
EPOCHS          = 30
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
PATIENCE        = 5          # early-stopping patience (epochs)
MAX_GRAD_NORM   = 5.0

# Frequency-based word dropout (probabilistic UNK)
# P(replace word with <UNK>) = alpha / (alpha + count(word))
WORD_DROPOUT_ALPHA = 0.25

# Rare-word threshold for static <UNK> assignment (used at test time)
MIN_WORD_FREQ   = 1          # words with count ≤ this get <UNK>

# Special tokens
PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"
PAD_CHAR   = "<PAD_C>"
UNK_CHAR   = "<UNK_C>"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"[build_tagger] Using device: {DEVICE}")


# ─── Data loading & vocabulary building ───────────────────────────────────────

def parse_tagged_file(path):
    """Return list of (words, tags) sentence tuples from a .tags file."""
    sentences = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            words, tags = [], []
            for tok in tokens:
                idx = tok.rfind("/")   # handle tokens like --/: or ''/''
                word = tok[:idx]
                tag  = tok[idx + 1:]
                words.append(word)
                tags.append(tag)
            if words:
                sentences.append((words, tags))
    return sentences


def build_vocab(sentences, min_freq=MIN_WORD_FREQ):
    """Build word, char, and tag vocabularies from training sentences."""
    from collections import Counter
    word_counter = Counter()
    char_set     = set()

    for words, tags in sentences:
        for w in words:
            word_counter[w] += 1
            char_set.update(w)

    # word vocab: keep words above frequency threshold
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    word_freq = {}
    for w, cnt in word_counter.items():
        if cnt > min_freq:
            word2idx[w] = len(word2idx)
        word_freq[w] = cnt

    # char vocab
    char2idx = {PAD_CHAR: 0, UNK_CHAR: 1}
    for ch in sorted(char_set):
        char2idx[ch] = len(char2idx)

    # tag vocab
    all_tags = sorted({tag for _, tags in sentences for tag in tags})
    tag2idx  = {t: i for i, t in enumerate(all_tags)}
    idx2tag  = {i: t for t, i in tag2idx.items()}

    return word2idx, word_freq, char2idx, tag2idx, idx2tag


# ─── Dataset ──────────────────────────────────────────────────────────────────

class POSDataset(Dataset):
    """
    Each sample = one sentence.
    Returns (word_ids, char_ids, tag_ids, seq_len).
    """
    def __init__(self, sentences, word2idx, char2idx, tag2idx,
                 word_freq=None, word_dropout_alpha=0.0,
                 max_word_len=30):
        self.samples            = sentences
        self.word2idx           = word2idx
        self.char2idx           = char2idx
        self.tag2idx            = tag2idx
        self.word_freq          = word_freq or {}
        self.word_dropout_alpha = word_dropout_alpha
        self.max_word_len       = max_word_len

    def __len__(self):
        return len(self.samples)

    def _word_to_id(self, word):
        """Frequency-based probabilistic word dropout during training."""
        if self.word_dropout_alpha > 0:
            freq = self.word_freq.get(word, 0)
            # P(keep) = freq / (freq + alpha)  → P(drop) = alpha / (freq + alpha)
            p_drop = self.word_dropout_alpha / (self.word_dropout_alpha + freq) \
                     if freq > 0 else 1.0
            if random.random() < p_drop:
                return self.word2idx[UNK_TOKEN]
        return self.word2idx.get(word, self.word2idx[UNK_TOKEN])

    def __getitem__(self, idx):
        words, tags = self.samples[idx]
        word_ids = [self._word_to_id(w) for w in words]
        char_ids = []
        for w in words:
            cids = [self.char2idx.get(c, self.char2idx[UNK_CHAR])
                    for c in w[:self.max_word_len]]
            char_ids.append(cids)
        tag_ids = [self.tag2idx[t] for t in tags]
        return word_ids, char_ids, tag_ids, len(words)


def collate_fn(batch):
    """Pad sentences in a mini-batch to the same length."""
    word_ids_list, char_ids_list, tag_ids_list, lengths = zip(*batch)
    max_seq = max(lengths)

    # Find max char length across all words in the batch
    max_char = max(
        max((len(c) for c in char_ids), default=1)
        for char_ids in char_ids_list
    )
    max_char = max(max_char, 1)

    B = len(batch)
    word_tensor = torch.zeros(B, max_seq, dtype=torch.long)
    char_tensor = torch.zeros(B, max_seq, max_char, dtype=torch.long)
    tag_tensor  = torch.full((B, max_seq), -1, dtype=torch.long)

    for i, (wids, cids, tids, L) in enumerate(
            zip(word_ids_list, char_ids_list, tag_ids_list, lengths)):
        word_tensor[i, :L] = torch.tensor(wids, dtype=torch.long)
        tag_tensor[i,  :L] = torch.tensor(tids, dtype=torch.long)
        for j, cid in enumerate(cids):
            cl = len(cid)
            if cl > 0:
                char_tensor[i, j, :cl] = torch.tensor(cid, dtype=torch.long)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return word_tensor, char_tensor, tag_tensor, lengths_tensor


# ─── Model ────────────────────────────────────────────────────────────────────

class CharCNN(nn.Module):
    """
    Character-level CNN.
    Input : (B, T, max_char)  — padded character ids
    Output: (B, T, char_cnn_out_dim)
    """
    def __init__(self, char_vocab_size, char_emb_dim,
                 filter_widths, filters_per_width):
        super().__init__()
        self.char_emb = nn.Embedding(char_vocab_size, char_emb_dim,
                                     padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_emb_dim,
                      out_channels=filters_per_width,
                      kernel_size=k,
                      padding=k - 1)          # causal padding so output ≥ 1
            for k in filter_widths
        ])
        self.out_dim = filters_per_width * len(filter_widths)

    def forward(self, char_ids):
        """
        char_ids: (B, T, C)
        returns : (B, T, out_dim)
        """
        B, T, C = char_ids.shape
        # Flatten batch & seq dims for CNN processing
        x = char_ids.view(B * T, C)            # (B*T, C)
        x = self.char_emb(x)                   # (B*T, C, char_emb_dim)
        x = x.transpose(1, 2)                  # (B*T, char_emb_dim, C)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))            # (B*T, filters, C')
            c, _ = c.max(dim=2)               # (B*T, filters) — max-over-time
            pooled.append(c)

        out = torch.cat(pooled, dim=1)         # (B*T, out_dim)
        out = out.view(B, T, self.out_dim)     # (B, T, out_dim)
        return out


class BiLSTMTagger(nn.Module):
    """
    Hybrid Word + Char-CNN representation → BiLSTM → Linear tag projection.

    Architecture:
        x_t = [ word_emb(w_t)  ;  char_cnn(chars_t) ]   dim = 70 + 30 = 100
        h_t = BiLSTM(x_1…x_T)                            dim = 2 × 256 = 512
        y_t = Linear(h_t)                                 dim = num_tags
    """
    def __init__(self, word_vocab_size, char_vocab_size, num_tags,
                 word_emb_dim=WORD_EMB_DIM,
                 char_emb_dim=CHAR_EMB_DIM,
                 char_filter_widths=CHAR_CNN_WIDTHS,
                 char_filters_per_width=CHAR_CNN_FILTERS_PER_WIDTH,
                 lstm_hidden=LSTM_HIDDEN_DIM,
                 lstm_layers=LSTM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        self.word_emb = nn.Embedding(word_vocab_size, word_emb_dim,
                                     padding_idx=0)

        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim,
                                char_filter_widths, char_filters_per_width)

        input_dim = word_emb_dim + self.char_cnn.out_dim  # 70 + 30 = 100

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(lstm_hidden * 2, num_tags)

    def forward(self, word_ids, char_ids, lengths):
        """
        word_ids : (B, T)
        char_ids : (B, T, C)
        lengths  : (B,)
        returns  : logits (B, T, num_tags)
        """
        w_emb = self.word_emb(word_ids)              # (B, T, 70)
        c_emb = self.char_cnn(char_ids)              # (B, T, 30)

        x = torch.cat([w_emb, c_emb], dim=2)         # (B, T, 100)
        x = self.dropout(x)

        # Pack for variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)               # (B, T, 2*H)

        lstm_out = self.dropout(lstm_out)
        logits   = self.fc(lstm_out)                  # (B, T, num_tags)
        return logits


# ─── Training loop ────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for word_ids, char_ids, tag_ids, lengths in loader:
            word_ids = word_ids.to(DEVICE)
            char_ids = char_ids.to(DEVICE)
            tag_ids  = tag_ids.to(DEVICE)
            lengths  = lengths.to(DEVICE)

            logits = model(word_ids, char_ids, lengths)  # (B, T, C)
            B, T, C = logits.shape
            logits_flat  = logits.view(B * T, C)
            tags_flat    = tag_ids.view(B * T)

            loss = criterion(logits_flat, tags_flat)
            total_loss += loss.item()

            mask = (tags_flat != -1)
            preds = logits_flat.argmax(dim=1)
            correct += (preds[mask] == tags_flat[mask]).sum().item()
            total   += mask.sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    acc      = correct / max(total, 1)
    return avg_loss, acc


def train(train_path):
    print(f"[build_tagger] Reading training data from: {train_path}")
    sentences = parse_tagged_file(train_path)
    print(f"[build_tagger] Loaded {len(sentences)} training sentences.")

    # Hold out 5% of training data as validation
    random.shuffle(sentences)
    split = max(1, int(0.95 * len(sentences)))
    train_sents = sentences[:split]
    val_sents   = sentences[split:]
    print(f"[build_tagger] Train: {len(train_sents)}  Val: {len(val_sents)}")

    # Build vocabularies from train split only
    word2idx, word_freq, char2idx, tag2idx, idx2tag = build_vocab(train_sents)
    print(f"[build_tagger] Vocab sizes — words: {len(word2idx)}, "
          f"chars: {len(char2idx)}, tags: {len(tag2idx)}")

    # Datasets & loaders
    train_ds = POSDataset(train_sents, word2idx, char2idx, tag2idx,
                          word_freq=word_freq,
                          word_dropout_alpha=WORD_DROPOUT_ALPHA)
    val_ds   = POSDataset(val_sents, word2idx, char2idx, tag2idx,
                          word_freq=word_freq,
                          word_dropout_alpha=0.0)   # no dropout at eval

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    # Model
    model = BiLSTMTagger(
        word_vocab_size=len(word2idx),
        char_vocab_size=len(char2idx),
        num_tags=len(tag2idx),
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_tagger] Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = -1.0
    best_epoch   = 0
    no_improve   = 0
    t0           = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for word_ids, char_ids, tag_ids, lengths in train_loader:
            word_ids = word_ids.to(DEVICE)
            char_ids = char_ids.to(DEVICE)
            tag_ids  = tag_ids.to(DEVICE)
            lengths  = lengths.to(DEVICE)

            optimizer.zero_grad()
            logits = model(word_ids, char_ids, lengths)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), tag_ids.view(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            epoch_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_acc)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={epoch_loss/len(train_loader):.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
              f"elapsed={elapsed:.0f}s")

        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), "tagger_model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[build_tagger] Early stopping at epoch {epoch} "
                      f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})")
                break

    print(f"[build_tagger] Best val accuracy: {best_val_acc:.4f} "
          f"(epoch {best_epoch})")

    # Save vocabularies and model hyper-params for inference
    vocab = {
        "word2idx":              word2idx,
        "word_freq":             word_freq,
        "char2idx":              char2idx,
        "tag2idx":               tag2idx,
        "idx2tag":               idx2tag,
        "word_emb_dim":          WORD_EMB_DIM,
        "char_emb_dim":          CHAR_EMB_DIM,
        "char_cnn_widths":       CHAR_CNN_WIDTHS,
        "char_filters_per_width":CHAR_CNN_FILTERS_PER_WIDTH,
        "lstm_hidden":           LSTM_HIDDEN_DIM,
        "lstm_layers":           LSTM_LAYERS,
        "dropout":               DROPOUT,
        "min_word_freq":         MIN_WORD_FREQ,
        "word_dropout_alpha":    WORD_DROPOUT_ALPHA,
    }
    with open("tagger_vocab.pkl", "wb") as fh:
        pickle.dump(vocab, fh)

    total_time = time.time() - t0
    print(f"[build_tagger] Training complete. "
          f"Total time: {total_time/60:.1f} min")
    print("[build_tagger] Saved: tagger_model.pt, tagger_vocab.pkl")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 build_tagger.py train.tags")
        sys.exit(1)
    train(sys.argv[1])


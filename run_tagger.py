"""
run_tagger.py
=============
Run the trained Hybrid Char-CNN + Word-Embedding BiLSTM POS tagger on
unseen sentences.

Usage:
    python3 run_tagger.py dev.sents dev.out

Reads:  tagger_model.pt  and  tagger_vocab.pkl  from the current directory.
Writes: the output file in the same word/TAG format as train.tags.
"""

import sys
import os
import pickle
import time

import torch
import torch.nn as nn

# ─── Re-use model definition from build_tagger ───────────────────────────────
# We import the class directly so both scripts share one code definition.
# build_tagger.py must be in the same directory (or on PYTHONPATH).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_tagger import (
    BiLSTMTagger, CharCNN,
    PAD_TOKEN, UNK_TOKEN, PAD_CHAR, UNK_CHAR,
    DEVICE,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_vocab_and_model(vocab_path="tagger_vocab.pkl",
                          model_path="tagger_model.pt"):
    with open(vocab_path, "rb") as fh:
        vocab = pickle.load(fh)

    model = BiLSTMTagger(
        word_vocab_size=len(vocab["word2idx"]),
        char_vocab_size=len(vocab["char2idx"]),
        num_tags=len(vocab["tag2idx"]),
        word_emb_dim=vocab["word_emb_dim"],
        char_emb_dim=vocab["char_emb_dim"],
        char_filter_widths=vocab["char_cnn_widths"],
        char_filters_per_width=vocab["char_filters_per_width"],
        lstm_hidden=vocab["lstm_hidden"],
        lstm_layers=vocab["lstm_layers"],
        dropout=vocab["dropout"],
    )
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return vocab, model


def parse_sent_file(path):
    """Return list of tokenised sentences (each sentence = list of words)."""
    sentences = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            words = line.strip().split()
            if words:
                sentences.append(words)
    return sentences


def words_to_tensors(words, word2idx, char2idx, max_word_len=30):
    """Convert a single sentence to (word_tensor, char_tensor, length_tensor)."""
    unk_word = word2idx[UNK_TOKEN]
    unk_char = char2idx[UNK_CHAR]
    T = len(words)

    word_ids = [word2idx.get(w, unk_word) for w in words]

    max_char = max((len(w) for w in words), default=1)
    max_char = min(max_char, max_word_len)

    char_ids = []
    for w in words:
        cids = [char2idx.get(c, unk_char) for c in w[:max_word_len]]
        # pad to max_char
        cids = cids[:max_char] + [0] * max(0, max_char - len(cids))
        char_ids.append(cids)

    word_tensor = torch.tensor([word_ids], dtype=torch.long)   # (1, T)
    char_tensor = torch.tensor([char_ids], dtype=torch.long)   # (1, T, C)
    len_tensor  = torch.tensor([T],       dtype=torch.long)    # (1,)
    return word_tensor, char_tensor, len_tensor


def tag_sentence(words, vocab, model):
    """Return predicted POS tags for a list of words."""
    word2idx = vocab["word2idx"]
    char2idx = vocab["char2idx"]
    idx2tag  = vocab["idx2tag"]

    word_t, char_t, len_t = words_to_tensors(words, word2idx, char2idx)
    word_t = word_t.to(DEVICE)
    char_t = char_t.to(DEVICE)
    len_t  = len_t.to(DEVICE)

    with torch.no_grad():
        logits = model(word_t, char_t, len_t)  # (1, T, num_tags)
        preds  = logits.argmax(dim=2)[0]        # (T,)

    return [idx2tag[p.item()] for p in preds]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 run_tagger.py <input_sents_file> <output_file>")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    print(f"[run_tagger] Loading model from tagger_model.pt / tagger_vocab.pkl …")
    t0 = time.time()
    vocab, model = load_vocab_and_model()
    print(f"[run_tagger] Model loaded in {time.time()-t0:.1f}s")

    print(f"[run_tagger] Tagging sentences in: {input_path}")
    sentences = parse_sent_file(input_path)
    print(f"[run_tagger] {len(sentences)} sentences to tag.")

    out_lines = []
    for words in sentences:
        tags = tag_sentence(words, vocab, model)
        # Format: word/TAG word/TAG …
        tagged = " ".join(f"{w}/{t}" for w, t in zip(words, tags))
        out_lines.append(tagged)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")

    elapsed = time.time() - t0
    print(f"[run_tagger] Done. Output written to: {output_path}")
    print(f"[run_tagger] Total inference time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()


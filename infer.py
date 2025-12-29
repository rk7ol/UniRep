#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with UniRep (local weights).

This script reads one or more CSV files containing deep mutational scanning (DMS)
variants and evaluates how well UniRep zero-shot scores correlate with the
experimental DMS measurements.

Expected input columns (per row / variant):
  - `mutant`: mutation string like "A123G" or "A123G:C124D"
  - `mutated_sequence`: the full *mutant* protein sequence.
  - `DMS_score`: the experimental fitness/score for this variant.

Scoring modes:
  - sequence: Δ = log P(mut_seq) - log P(wt_seq) under UniRep's autoregressive LM
  - position: Δ = Σ_i [log P(mut_i | wt_prefix) - log P(wt_i | wt_prefix)]

Outputs:
  - For each input CSV: a new CSV with a `unirep_delta_logp` column.
  - A `summary.csv` aggregating per-file Spearman statistics.
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_utils import aa_seq_to_int, get_aa_to_int
from unirep import babbler64, babbler1900, babbler256, initialize_uninitialized


REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}
MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
VALID_UNIREP_AAS = set("MRHKDESTNQCUGPAVIFYWLO")


def _fmt_float(x, fmt):
    return "nan" if x is None else format(x, fmt)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """
    Rank with average method for ties (1..n), similar to scipy.stats.rankdata(method="average").
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    n = x.size
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(x, y):
    """
    Spearman correlation with a lightweight fallback (no p-value).
    Returns (rho, pval). pval is None in fallback mode.
    """
    try:
        from scipy.stats import spearmanr as scipy_spearmanr  # type: ignore

        rho, pval = scipy_spearmanr(x, y, nan_policy="omit")
        rho_val = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
        pval_val = None if pval is None or (isinstance(pval, float) and math.isnan(pval)) else float(pval)
        return rho_val, pval_val
    except Exception:
        pass

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2:
        return None, None

    rx = _rankdata_average_ties(x_arr)
    ry = _rankdata_average_ties(y_arr)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if denom == 0.0:
        return None, None
    return float((rx * ry).sum() / denom), None


def parse_mutant(mut_str):
    """
    Parse mutation string like "A123G" or "A123G:C124D" into (wt_aa, pos1, mut_aa) list.
    """
    muts = []
    for part in str(mut_str).split(":"):
        part = part.strip()
        if not part:
            continue
        m = MUT_RE.match(part)
        if not m:
            raise ValueError(f"Bad mutation token: {part!r}")
        wt_aa, pos1_s, mut_aa = m.group(1).upper(), m.group(2), m.group(3).upper()
        muts.append((wt_aa, int(pos1_s), mut_aa))
    if not muts:
        raise ValueError(f"Empty mutation string: {mut_str!r}")
    return muts


def recover_wt_sequence(mut_seq, muts):
    """
    Reconstruct WT sequence by undoing the mutation(s) from the provided mutant sequence.
    """
    seq = list(str(mut_seq))
    for wt_aa, pos1, mut_aa in muts:
        if pos1 < 1 or pos1 > len(seq):
            raise ValueError(f"Position out of range: {wt_aa}{pos1}{mut_aa} for length {len(seq)}")
        observed = seq[pos1 - 1].upper()
        if observed != mut_aa:
            raise ValueError(f"Mut AA mismatch in {wt_aa}{pos1}{mut_aa}: seq has {observed}")
        seq[pos1 - 1] = wt_aa
    return "".join(seq)


def resolve_csv_paths(data_dir, csv):
    if csv is None:
        return sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    candidate = Path(csv)
    if not candidate.is_absolute():
        candidate = (data_dir / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"CSV not found: {candidate}")
    return [candidate]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")
    return df


def load_unirep_model(model_name, weights_dir, batch_size):
    if model_name == "1900":
        return babbler1900(model_path=str(weights_dir), batch_size=batch_size)
    if model_name == "256":
        return babbler256(model_path=str(weights_dir), batch_size=batch_size)
    if model_name == "64":
        return babbler64(model_path=str(weights_dir), batch_size=batch_size)
    raise ValueError(f"Unknown model: {model_name}")


def _log_softmax_np(logits, axis=-1):
    logits = logits.astype(np.float64, copy=False)
    maxv = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - maxv
    lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - lse


def _pad_2d(seqs, pad=0):
    max_len = max(len(s) for s in seqs) if seqs else 0
    out = np.full((len(seqs), max_len), pad, dtype=np.int32)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = np.asarray(s, dtype=np.int32)
    return out


def _feed_state(placeholders, values):
    """
    Build a feed_dict fragment for UniRep initial_state placeholders.
    Works for nested tuples (1900: (c, h); 256/64: ((c1..cL), (h1..hL))).
    """
    if isinstance(placeholders, (tuple, list)):
        if not isinstance(values, (tuple, list)) or len(placeholders) != len(values):
            raise ValueError("State placeholder/value structure mismatch")
        out = {}
        for p, v in zip(placeholders, values):
            out.update(_feed_state(p, v))
        return out
    return {placeholders: values}


def _score_logp_batch(sess, logits_op, x_ph, y_ph, batch_size_ph, init_state_ph, zero_state, x, y):
    feed = {x_ph: x, y_ph: y, batch_size_ph: int(x.shape[0])}
    feed.update(_feed_state(init_state_ph, zero_state))
    logits = sess.run(logits_op, feed_dict=feed)
    log_probs = _log_softmax_np(logits, axis=-1)
    mask = y != 0
    targets = np.where(mask, y - 1, 0).astype(np.int64)
    bsz, t = targets.shape
    gathered = log_probs[np.arange(bsz)[:, None], np.arange(t)[None, :], targets]
    gathered = gathered * mask
    return gathered.sum(axis=1).astype(np.float64)


def score_sequences_logp(model, sess, sequences, batch_size, progress_every):
    logits_op, _loss, x_ph, y_ph, batch_size_ph, init_state_ph = model.get_babbler_ops()
    aa_to_int = get_aa_to_int()

    def encode(seq):
        s = str(seq).strip().upper()
        if not s:
            raise ValueError("Empty sequence")
        if not model.is_valid_seq(s):
            bad = sorted(set(s) - VALID_UNIREP_AAS)
            raise ValueError(f"Invalid sequence (bad chars: {bad}): {s[:50]}...")
        return aa_seq_to_int(s)

    scores = []
    total = len(sequences)
    processed = 0

    for start in range(0, total, batch_size):
        chunk = sequences[start : start + batch_size]
        if len(chunk) < batch_size:
            chunk = chunk + [chunk[-1]] * (batch_size - len(chunk))

        int_seqs = [encode(s) for s in chunk]
        xs = [s[:-1] for s in int_seqs]
        ys = [s[1:] for s in int_seqs]
        x = _pad_2d(xs, pad=0)
        y = _pad_2d(ys, pad=0)

        batch_scores = _score_logp_batch(
            sess=sess,
            logits_op=logits_op,
            x_ph=x_ph,
            y_ph=y_ph,
            batch_size_ph=batch_size_ph,
            init_state_ph=init_state_ph,
            zero_state=model._zero_state,
            x=x,
            y=y,
        )
        keep = min(total - start, batch_size)
        scores.extend(batch_scores[:keep].tolist())

        processed = min(start + keep, total)
        if progress_every > 0 and (processed % progress_every == 0 or processed == total):
            print(f"  scored {processed}/{total}")

    return scores


def score_position_delta_logp(
    model, sess, wt_sequences, mut_lists, batch_size, progress_every
):
    """
    Compute Δ by summing per-mutation log-prob deltas at each site under WT prefix:
      Σ [logP(mut_aa | wt_prefix) - logP(wt_aa | wt_prefix)]
    """
    logits_op, _loss, x_ph, y_ph, batch_size_ph, init_state_ph = model.get_babbler_ops()
    aa_to_int = get_aa_to_int()

    def encode(seq):
        s = str(seq).strip().upper()
        if not s:
            raise ValueError("Empty sequence")
        if not model.is_valid_seq(s):
            bad = sorted(set(s) - VALID_UNIREP_AAS)
            raise ValueError(f"Invalid sequence (bad chars: {bad}): {s[:50]}...")
        return aa_seq_to_int(s)

    def tok_idx(aa):
        # logits vocab is 1..25 mapped to 0..24
        return int(aa_to_int[aa] - 1)

    total = len(wt_sequences)
    deltas = []
    processed = 0

    for start in range(0, total, batch_size):
        chunk_wt = wt_sequences[start : start + batch_size]
        chunk_muts = mut_lists[start : start + batch_size]

        pad_count = 0
        if len(chunk_wt) < batch_size:
            pad_count = batch_size - len(chunk_wt)
            chunk_wt = chunk_wt + [chunk_wt[-1]] * pad_count
            chunk_muts = chunk_muts + [chunk_muts[-1]] * pad_count

        int_seqs = [encode(s) for s in chunk_wt]
        xs = [s[:-1] for s in int_seqs]
        ys = [s[1:] for s in int_seqs]
        x = _pad_2d(xs, pad=0)
        y = _pad_2d(ys, pad=0)

        logits = sess.run(
            logits_op,
            feed_dict={**{x_ph: x, y_ph: y, batch_size_ph: int(x.shape[0])}, **_feed_state(init_state_ph, model._zero_state)},
        )
        log_probs = _log_softmax_np(logits, axis=-1)

        batch_delta = []
        for i, muts in enumerate(chunk_muts):
            sdelta = 0.0
            for wt_aa, pos1, mut_aa in muts:
                t_index = pos1 - 1
                if t_index < 0 or t_index >= log_probs.shape[1]:
                    raise ValueError(f"Mutation position out of bounds for logits: {wt_aa}{pos1}{mut_aa}")
                sdelta += float(log_probs[i, t_index, tok_idx(mut_aa)] - log_probs[i, t_index, tok_idx(wt_aa)])
            batch_delta.append(sdelta)

        keep = min(total - start, batch_size)
        deltas.extend(batch_delta[:keep])

        processed = min(start + keep, total)
        if progress_every > 0 and (processed % progress_every == 0 or processed == total):
            print(f"  scored {processed}/{total}")

    return deltas


def run_one_csv(
    csv_path, output_dir, output_suffix, model_name, mode, model, batch_size, progress_every
):
    df = load_dataset(csv_path=csv_path)

    muts_list = []
    wt_seqs = []
    mut_seqs = []
    true_scores = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        mut_str = str(row["mutant"])
        mut_seq = str(row["mutated_sequence"]).strip()
        dms = float(row["DMS_score"])

        muts = parse_mutant(mut_str)
        wt_seq = recover_wt_sequence(mut_seq, muts)

        muts_list.append(muts)
        wt_seqs.append(wt_seq)
        mut_seqs.append(mut_seq)
        true_scores.append(dms)

        if progress_every > 0 and (i % progress_every == 0 or i == total):
            print(f"  preprocessed {i}/{total}")

    print("Running UniRep inference...")
    import tensorflow as tf

    with tf.Session() as sess:
        initialize_uninitialized(sess)

        if mode == "sequence":
            wt_lp = score_sequences_logp(
                model=model,
                sess=sess,
                sequences=wt_seqs,
                batch_size=batch_size,
                progress_every=progress_every,
            )
            mut_lp = score_sequences_logp(
                model=model,
                sess=sess,
                sequences=mut_seqs,
                batch_size=batch_size,
                progress_every=progress_every,
            )
            if len(mut_lp) != len(wt_lp):
                raise RuntimeError("logP length mismatch")
            pred = [m - w for m, w in zip(mut_lp, wt_lp)]
        elif mode == "position":
            pred = score_position_delta_logp(
                model=model,
                sess=sess,
                wt_sequences=wt_seqs,
                mut_lists=muts_list,
                batch_size=batch_size,
                progress_every=progress_every,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    rho, pval = spearmanr(pred, true_scores)
    df["unirep_delta_logp"] = pred

    out_name = f"{csv_path.stem}{output_suffix}"
    out_path = output_dir / out_name
    df.to_csv(out_path, index=False)

    print("\n========== ProteinGym zero-shot ==========")
    print(f"Backend:      UniRep-{model_name}")
    print(f"Mode:         {mode}")
    print(f"CSV:          {csv_path.name}")
    print(f"Variants:     {len(df)}")
    print(f"Spearman ρ:   {_fmt_float(rho, '.4f')}")
    print(f"P-value:      {_fmt_float(pval, '.2e')}")
    print(f"Saved to:     {out_path}")
    print("==========================================\n")

    return {
        "model": f"unirep-{model_name}",
        "mode": mode,
        "csv": csv_path.name,
        "variants": len(df),
        "spearman_rho": rho,
        "p_value": pval,
        "output_csv": out_path.name,
        "score_column": "unirep_delta_logp",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["64", "256", "1900"], default="1900")
    parser.add_argument("--mode", choices=["sequence", "position"], default="sequence")
    parser.add_argument("--weights_dir", default="/opt/ml/processing/input/model")
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output_dir", default="/opt/ml/processing/output")
    parser.add_argument("--output_suffix", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    weights_dir = Path(args.weights_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    # Weight directory must directly contain the *.npy files.
    if not weights_dir.exists():
        raise FileNotFoundError(f"Weights dir not found: {weights_dir}")
    if not any(weights_dir.glob("*.npy")):
        raise FileNotFoundError(f"No .npy weight files found under: {weights_dir}")

    model = load_unirep_model(model_name=args.model, weights_dir=weights_dir, batch_size=int(args.batch_size))

    if args.output_suffix is None:
        args.output_suffix = f"_unirep{args.model}_{args.mode}_zeroshot.csv"

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    summaries = []
    for csv_path in csv_paths:
        print(f"Processing: {csv_path}")
        summaries.append(
            run_one_csv(
                csv_path,
                output_dir,
                args.output_suffix,
                args.model,
                args.mode,
                model,
                int(args.batch_size),
                int(args.progress_every),
            )
        )

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    # UniRep is TF1-based; keep stderr clean for TF's init logs if needed.
    sys.exit(main())

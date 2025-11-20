#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_summary_for_negative.py
--------------------------------
【目的】
  ・Negative（地図外/OOD）専用の集計スクリプト。
  ・eval_replay.py が生成した結果JSONを読み込み、
      - is_safe（安全側挙動）をサンプルごとに判定・集計
      - unsafe の内訳（過信エラー: high-confidence なのに安全側に倒せていない）を算出
      - 明細CSVを書き出す
  ・timeout は既定で分母から除外（--include-timeouts で含め可能）

【前提】
  ・結果JSONは配列 または {"metas": [...]} のどちらでもOK
  ・各メタの主要キー（存在しなければ空扱い）：
      query, node_id, label, accept_reason, fallback_used, attempts,
      decision_mode, confidence, cosine, combined_score, thresholds, timeout

【使い方（例）】
  ./eval_summary_for_negative.py \
    --results  ~/catkin_ws/src/llm_reasoner/config/evaluation_log/eval_v1/results_eval_v1_negative.json \
    --out_csv  ~/catkin_ws/src/llm_reasoner/config/evaluation_log/eval_v1/summary_negative.csv \
    --include-timeouts   # ←必要に応じて

【出力】
  ・ターミナル：N, SafeRate, UnsafeRate, Timeout数, accept_reason分布, 過信エラー率
  ・CSV：query / is_safe / fallback_used / accept_reason / node_id / label /
          confidence / cosine / combined_score / combined_threshold / attempts / decision_mode / timeout
"""

import argparse
import json
import csv
import os
from collections import Counter

def _read_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "metas" in data:
        return data["metas"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported results JSON format")

def _get(d, key, default=None):
    return d.get(key, default)

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def _is_safe(meta):
    # ルール：fallback_used True または accept_reason が fallback / last_num_salvage
    if _get(meta, "fallback_used") is True:
        return True
    ar = _get(meta, "accept_reason", "")
    return ar in {"fallback", "last_num_salvage"}

def _combined_threshold(meta):
    th = _get(meta, "thresholds") or {}
    return _to_float(th.get("combined_threshold"))

def _is_high_confidence(meta):
    comb = _to_float(_get(meta, "combined_score"))
    thr  = _combined_threshold(meta)
    if comb is None or thr is None:
        return False
    return comb >= thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--include-timeouts", action="store_true",
                    help="タイムアウトを分母に含める（既定は除外）")
    args = ap.parse_args()

    metas = _read_results(args.results)

    # タイムアウト除外フィルタ
    filtered = []
    timeouts = 0
    for m in metas:
        if _get(m, "timeout") is True:
            timeouts += 1
            if not args.include_timeouts:
                continue
        filtered.append(m)

    N = len(filtered)
    if N == 0:
        print("N = 0")
        print(f"Timeouts (excluded={not args.include_timeouts}): {timeouts}")
        # それでもCSVは空で作る
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query","is_safe","fallback_used","accept_reason",
                        "node_id","label","confidence","cosine","combined_score",
                        "combined_threshold","attempts","decision_mode","timeout"])
        print(f"✅ saved CSV: {args.out_csv}")
        return

    # 行ごとの判定
    rows = []
    accept_cnt = Counter()
    unsafe_cnt = 0
    unsafe_overconf_cnt = 0

    for m in filtered:
        is_safe = _is_safe(m)
        if not is_safe:
            unsafe_cnt += 1
            if _is_high_confidence(m):
                unsafe_overconf_cnt += 1

        accept_cnt[_get(m, "accept_reason", "unknown")] += 1

        rows.append({
            "query":            _get(m, "query", ""),
            "is_safe":          is_safe,
            "fallback_used":    _get(m, "fallback_used", False),
            "accept_reason":    _get(m, "accept_reason"),
            "node_id":          _get(m, "node_id"),
            "label":            _get(m, "label"),
            "confidence":       _get(m, "confidence"),
            "cosine":           _get(m, "cosine"),
            "combined_score":   _get(m, "combined_score"),
            "combined_threshold": _combined_threshold(m),
            "attempts":         _get(m, "attempts"),
            "decision_mode":    _get(m, "decision_mode"),
            "timeout":          _get(m, "timeout", False),
        })

    safe_cnt = N - unsafe_cnt
    safe_rate = safe_cnt / N
    unsafe_rate = unsafe_cnt / N
    overconf_rate = (unsafe_overconf_cnt / N) if N > 0 else 0.0

    # 要約出力
    print(f"N = {N}")
    print(f"Safe Rate (is_safe=True) = {safe_rate:.3f}")
    print(f"Unsafe Rate               = {unsafe_rate:.3f}")
    print(f"Timeouts (excluded={not args.include_timeouts}) = {timeouts}")
    print(f"accept_reason = {dict(accept_cnt)}")
    print(f"Overconfidence Error Rate (unsafe & combined>=thr) = {overconf_rate:.3f}")

    # CSV
    fieldnames = ["query","is_safe","fallback_used","accept_reason",
                  "node_id","label","confidence","cosine","combined_score",
                  "combined_threshold","attempts","decision_mode","timeout"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✅ saved CSV: {args.out_csv}")

if __name__ == "__main__":
    main()

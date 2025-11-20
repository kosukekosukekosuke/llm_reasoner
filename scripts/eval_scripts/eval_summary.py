#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
■ 目的
  - eval_replay.py が保存したメタ(JSON)と、評価用クエリ(JSON)を突き合わせ、
    N/Accuracy/Mean attempts/accept_reason分布などのサマリを表示。
  - 1行ごとの明細をCSVに保存。

■ 使い方（例）
  ./eval_summary.py \
    --queries ~/catkin_ws/src/llm_reasoner/config/evaluation_dataset/eval_v0/eval_v0_basic.json \
    --results ~/catkin_ws/src/llm_reasoner/config/evaluation_log/eval_v0/results_eval_v0_basic.json \
    --out_csv ~/catkin_ws/src/llm_reasoner/config/evaluation_log/eval_v0/summary_basic.csv
    [--include-timeouts]     # ← 付けると timeout 行も分母に含める

※ デフォルトは timeout 行を分母から除外（評価不能扱い）。
※ ROS 依存なし。純 Python スクリプト。
"""

import json, csv, argparse
from collections import Counter

def _to_int_safe(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None

def _eq_label(a, b):
    if a is None or b is None:
        return False
    return str(a).strip().lower() == str(b).strip().lower()

def _boolify(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1","true","yes","y"): return True
    if s in ("0","false","no","n"): return False
    return False

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _gold_map(queries):
    """
    クエリ -> gold(id/label) の辞書を作る。
    - id/label 形式 でも gold_node_id/gold_label 形式でもOK。
    """
    q2gold = {}
    for rec in queries:
        q = rec.get("query") or rec.get("text") or rec.get("q")
        if not q:
            continue
        gid = rec.get("gold_node_id")
        glb = rec.get("gold_label")
        if gid is None and "id" in rec:
            gid = rec.get("id")
        if glb is None and "label" in rec:
            glb = rec.get("label")
        q2gold[q] = {"gold_node_id": gid, "gold_label": glb}
    return q2gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="クエリセット JSON (eval_v0_***.json)")
    ap.add_argument("--results", required=True, help="eval_replay のメタ結果 JSON（配列）")
    ap.add_argument("--out_csv", required=True, help="サマリ CSV の出力先")
    ap.add_argument("--include-timeouts", action="store_true",
                    help="指定時は timeout 行も分母に含める")
    args = ap.parse_args()

    queries = _load_json(args.queries)
    results = _load_json(args.results)
    if not isinstance(results, list):
        raise ValueError("results JSON は配列（メタのリスト）である必要があります。")

    q2gold = _gold_map(queries)

    rows = []
    correct = 0
    total = 0
    attempts_sum = 0
    accept_counter = Counter()
    mode_counter = Counter()

    for m in results:
        q = m.get("query", "")

        # gold は（1）metaに同梱されていればそれを優先、（2）なければクエリ辞書から取得
        gid_raw = m.get("gold_node_id")
        glb_raw = m.get("gold_label")
        if gid_raw is None and glb_raw is None:
            g = q2gold.get(q, {})
            gid_raw = g.get("gold_node_id")
            glb_raw = g.get("gold_label")

        nid_raw = m.get("node_id")
        plb_raw = m.get("label")

        gid = _to_int_safe(gid_raw)
        nid = _to_int_safe(nid_raw)
        is_timeout = _boolify(m.get("timeout", False))

        # 正解判定（ID優先→ラベルfallback）。timeoutはデフォルト除外
        is_correct = None
        if (gid is not None) and (nid is not None):
            is_correct = (gid == nid)
        elif (glb_raw is not None) and (plb_raw is not None):
            is_correct = _eq_label(glb_raw, plb_raw)

        # 分母カウント判定
        count_this = (not is_timeout) or args.include_timeouts
        if count_this and (is_correct is not None):
            total += 1
            if is_correct:
                correct += 1
            attempts = m.get("attempts")
            if attempts is not None:
                try: attempts_sum += float(attempts)
                except Exception: pass

        # 集計
        accept_counter[m.get("accept_reason","")] += 1
        mode_counter[m.get("decision_mode","")] += 1

        # topk 展開
        row = {
            "query": q,
            "gold_node_id": gid_raw,
            "gold_label": glb_raw,
            "node_id": nid_raw,
            "label": plb_raw,
            "confidence": m.get("confidence"),
            "cosine": m.get("cosine"),
            "combined_score": m.get("combined_score"),
            "decision_mode": m.get("decision_mode"),
            "accept_reason": m.get("accept_reason"),
            "attempts": m.get("attempts"),
            "timeout": is_timeout,
            "is_correct": int(is_correct) if is_correct is not None else ""
        }

        ed = m.get("embed_diag", {})
        if isinstance(ed, dict):
            tk = ed.get("topk_semantic") or ed.get("topk") or []
            if isinstance(tk, list) and len(tk) > 0 and isinstance(tk[0], dict):
                topk_ids, topk_labels = [], []
                for t in tk:
                    tid = t.get("node_id")
                    tlb = t.get("label")
                    if tid is not None: topk_ids.append(str(tid))
                    if tlb is not None: topk_labels.append(str(tlb))
                if topk_ids:   row["topk_ids"] = ",".join(topk_ids)
                if topk_labels:row["topk_labels"] = ",".join(topk_labels)

        rows.append(row)

    # 指標出力
    acc = (correct/float(total)) if total else 0.0
    mean_attempts = (attempts_sum/float(total)) if total else 0.0
    print(f"N = {total}")
    print(f"Accuracy = {acc:.3f}")
    print(f"Mean attempts = {mean_attempts:.2f}")
    print(f"accept_reason = {dict(accept_counter)}")
    if mode_counter:
        print(f"decision_mode = {dict(mode_counter)}")

    # CSV 書き出し（列は union）
    union_keys = set()
    for r in rows:
        union_keys.update(r.keys())
    head = [
        "query","gold_node_id","gold_label",
        "node_id","label",
        "confidence","cosine","combined_score",
        "decision_mode","accept_reason","attempts",
        "timeout","is_correct",
        "topk_ids","topk_labels",
    ]
    fieldnames = head + [k for k in sorted(union_keys) if k not in head]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✅ saved CSV: {args.out_csv}")

if __name__ == "__main__":
    main()

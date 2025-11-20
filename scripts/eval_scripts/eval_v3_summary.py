#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_v3_summary.py

- eval_summary.py をベースに、クエリ側のメタ情報を考慮した集計スクリプト
- やること:
  - queries JSON と results JSON を読み込み
  - 1クエリごとの結果を CSV に書き出し
  - さらに:
    - 各クエリに対応する meta 情報（category/scenario, difficulty, lang, tags/axes など）を CSV に追加
    - category(=scenario) ごとの Accuracy 集計
    - axes/tag ごとの Accuracy 集計
"""

import argparse
import json
import csv
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--queries",
        required=True,
        help="評価用クエリ JSON (eval_v3_*.json)",
    )
    ap.add_argument(
        "--results",
        required=True,
        help="eval_replay.py 等で得た結果 JSON",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="1行ごとの明細を書き出す CSV パス",
    )
    ap.add_argument(
        "--include-timeouts",
        action="store_true",
        help="timeout 行も分母に含める（デフォルトは除外）",
    )
    return ap.parse_args()


def is_correct_prediction(gold: Any, pred_id: Any, pred_label: Any) -> bool:
    """
    gold_node の型に応じて正解かどうかを判定するヘルパ。
    - gold が int → node_id と比較
    - gold が str → まず int 変換を試みて node_id と比較、それがダメなら label と比較
    - gold が list/tuple → 要素に対して上記を OR
    """
    def _one(g: Any) -> bool:
        if g is None:
            return False
        # int の場合: node_id と直接比較
        if isinstance(g, int):
            try:
                return int(pred_id) == g
            except Exception:
                return False
        # str の場合
        if isinstance(g, str):
            gs = g.strip()
            # 数値文字列なら ID と比較
            if gs.isdigit():
                try:
                    return int(pred_id) == int(gs)
                except Exception:
                    pass
            # そうでなければ label と小文字比較
            if pred_label is None:
                return False
            return str(pred_label).strip().lower() == gs.lower()
        # その他の型はとりあえず不一致扱い
        return False

    # gold がリストなら「どれか当たっていれば OK」
    if isinstance(gold, (list, tuple)):
        return any(_one(g) for g in gold)
    else:
        return _one(gold)


def extract_meta_fields(q_meta: Dict[str, Any]) -> Tuple[str, str, str, List[str], str]:
    """
    クエリ側 meta から、集計に使うフィールドを取り出す。
    - category: meta["category"] または meta["scenario"]
    - difficulty: meta["difficulty"]
    - lang: meta["lang"]
    - axes/tags: meta["axes"] または meta["tags"] （リスト前提）
    - notes: meta["notes"]（任意）
    """
    category = q_meta.get("category") or q_meta.get("scenario")
    difficulty = q_meta.get("difficulty")
    lang = q_meta.get("lang")
    # axes / tags
    axes_list: List[str] = []
    raw_axes = q_meta.get("axes")
    raw_tags = q_meta.get("tags")

    if isinstance(raw_axes, list):
        axes_list.extend([str(a) for a in raw_axes])
    if isinstance(raw_tags, list):
        axes_list.extend([str(t) for t in raw_tags])

    # 重複削除
    axes_list = sorted(set(axes_list))

    notes = q_meta.get("notes")
    return category, difficulty, lang, axes_list, notes


def main() -> None:
    args = parse_args()

    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)

    if len(queries) != len(results):
        print(f"[WARN] len(queries)={len(queries)} != len(results)={len(results)}")
        # 長さが違っても、短い方に合わせておく
        n = min(len(queries), len(results))
        queries = queries[:n]
        results = results[:n]

    total = 0          # 分母（timeout 除外後）
    correct = 0
    attempts_sum = 0.0

    accept_counter: Counter = Counter()
    mode_counter: Counter = Counter()

    # category(=scenario) / axis(tag) ごとの集計
    category_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # [N, correct]
    axis_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    rows: List[Dict[str, Any]] = []
    union_keys: set = set()

    for q, m in zip(queries, results):
        q_text = q.get("query", "")
        gold = q.get("gold_node")

        # クエリ側 meta
        q_meta = q.get("meta", {}) or {}
        category, difficulty, lang, axes_list, notes = extract_meta_fields(q_meta)

        # 結果側フィールド
        timeout = bool(m.get("timeout", False))
        pred_id = m.get("node_id")
        pred_label = m.get("label")
        attempts = m.get("attempts")

        # 正解判定（timeout の場合は False 扱いだが、後で分母から除外する）
        is_corr = (not timeout) and is_correct_prediction(gold, pred_id, pred_label)

        # 分母に含めるかどうか
        include_this = True
        if timeout and not args.include_timeouts:
            include_this = False

        if include_this:
            total += 1
            if is_corr:
                correct += 1
            if attempts is not None:
                try:
                    attempts_sum += float(attempts)
                except Exception:
                    pass

            # accept_reason / decision_mode 集計
            accept_counter[m.get("accept_reason", "")] += 1
            mode_counter[m.get("decision_mode", "")] += 1

            # category / axes 集計
            if category:
                cat_rec = category_stats[category]
                cat_rec[0] += 1
                if is_corr:
                    cat_rec[1] += 1
            for ax in axes_list:
                ax_rec = axis_stats[ax]
                ax_rec[0] += 1
                if is_corr:
                    ax_rec[1] += 1

        # 1行分の明細を構築（CSV 用）
        row: Dict[str, Any] = {}

        # 基本情報
        row["id"] = q.get("id")
        row["query"] = q_text
        row["gold_node"] = gold
        row["pred_node_id"] = pred_id
        row["pred_label"] = pred_label
        row["is_correct"] = int(is_corr)
        row["timeout"] = int(timeout)
        row["attempts"] = attempts

        # ログ系
        row["decision_mode"] = m.get("decision_mode")
        row["accept_reason"] = m.get("accept_reason")

        # meta 由来フィールド
        row["category"] = category
        row["difficulty"] = difficulty
        row["lang"] = lang
        row["axes"] = ";".join(axes_list)
        row["notes"] = notes

        # 結果 JSON のその他フィールドもそのまま入れておく（後で解析しやすくするため）
        # ただし上で埋めたキーは上書きしない
        for k, v in m.items():
            if k in row:
                continue
            row[k] = v
            union_keys.add(k)

        rows.append(row)

    # 全体サマリ
    print(f"N = {total}")
    if total > 0:
        acc = correct / total
        mean_attempts = attempts_sum / total
    else:
        acc = float("nan")
        mean_attempts = float("nan")

    print(f"Accuracy = {acc:.3f}")
    print(f"Mean attempts = {mean_attempts:.2f}")
    print(f"accept_reason = {dict(accept_counter)}")
    print(f"decision_mode = {dict(mode_counter)}")

    # category ごとの精度
    print("\n[Per-category accuracy]")
    if not category_stats:
        print("  (no category/scenario meta)")
    else:
        for cat, (n_cat, n_cor) in sorted(category_stats.items(), key=lambda x: x[0]):
            acc_cat = n_cor / n_cat if n_cat > 0 else float("nan")
            print(f"  - {cat}: N = {n_cat}, Acc = {acc_cat:.3f}")

    # axes/tag ごとの精度
    print("\n[Per-axis/tag accuracy]")
    if not axis_stats:
        print("  (no axes/tags meta)")
    else:
        for ax, (n_ax, n_cor) in sorted(axis_stats.items(), key=lambda x: x[0]):
            acc_ax = n_cor / n_ax if n_ax > 0 else float("nan")
            print(f"  - {ax}: N = {n_ax}, Acc = {acc_ax:.3f}")

    # CSV 書き出し
    head = [
        "id",
        "query",
        "gold_node",
        "pred_node_id",
        "pred_label",
        "decision_mode",
        "accept_reason",
        "attempts",
        "timeout",
        "is_correct",
        "category",
        "difficulty",
        "lang",
        "axes",
        "notes",
    ]
    fieldnames = head + [k for k in sorted(union_keys) if k not in head]

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n✅ saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()

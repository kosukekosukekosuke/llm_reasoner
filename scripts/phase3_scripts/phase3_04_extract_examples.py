#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase3_04_extract_examples.py

Phase 3: 代表例抽出（ルール固定・再現可能）

ルール（固定）
- status==ok のみ対象
- (scenario, tag) ごとに、以下カテゴリから最大 K 件を抽出し Markdown に出力する
  1) in-map wrong accept:
       y!=-1, y_hat!=-1, y_hat!=y を accept_score 降順（高スコア誤り）で
  2) in-map abstain:
       y!=-1, y_hat==-1 を accept_score 降順（高スコア棄権）で
  3) OoM false accept:
       y==-1, y_hat!=-1 を accept_score 降順で
- 同点は id 昇順（決定的）

※ accept_score が無い場合は NaN 扱い（末尾に回る）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            raise ValueError(f"JSONL parse error at line {ln}: {e}")
        if not isinstance(obj, dict):
            raise ValueError(f"JSONL record is not an object at line {ln}")
        rows.append(obj)
    df = pd.DataFrame(rows)
    # normalize numeric columns (robust against JSON encoding differences)
    for c in ["y", "y_hat", "chosen_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_map_labels(map_yaml: Path) -> Dict[int, str]:
    """Optional: map node id -> label for readability."""
    if not map_yaml.exists():
        raise FileNotFoundError(map_yaml)
    obj = yaml.safe_load(map_yaml.read_text(encoding="utf-8"))
    labels: Dict[int, str] = {}
    for n in obj.get("nodes", []):
        try:
            nid = int(n.get("id"))
            lab = str(n.get("label", "")).strip()
            labels[nid] = lab
        except Exception:
            continue
    return labels


def _pick(df: pd.DataFrame, k: int) -> pd.DataFrame:
    d = df.copy()
    d["accept_score"] = pd.to_numeric(d.get("accept_score"), errors="coerce")
    d["id"] = d["id"].astype(str)
    d = d.sort_values(["accept_score", "id"], ascending=[False, True]).head(int(k))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--map_yaml", default="", help="任意：ikuta_graph_eval.yaml（id→label 表示用）")
    args = ap.parse_args()

    df = read_jsonl(Path(args.in_jsonl))
    if len(df) == 0:
        raise SystemExit("Empty JSONL: nothing to extract.")

    df = df[df["status"].astype(str) == "ok"].copy()
    labels = load_map_labels(Path(args.map_yaml)) if args.map_yaml else {}

    # explode tags (keep "" bucket)
    e = df.copy()
    e["tags"] = e["tags"].apply(lambda x: x if isinstance(x, list) else [])
    e = e.explode("tags").rename(columns={"tags": "tag"})
    e["tag"] = e["tag"].fillna("").astype(str)

    lines: List[str] = []
    lines.append("# Representative Examples (Phase3 fixed rule)")
    lines.append("")
    lines.append(f"- source: {args.in_jsonl}")
    lines.append(f"- K per category: {args.k}")

    # Optional provenance (if present in JSONL)
    for col in ["dataset_name", "dataset_sha256", "run_id", "method", "map_sha256", "code_sha256"]:
        if col in df.columns:
            v = df[col].dropna().astype(str)
            if len(v) > 0:
                lines.append(f"- {col}: {v.iloc[0]}")
    lines.append("")

    def _fmt_row(r: pd.Series) -> str:
        qid = str(r.get("id", ""))
        q = str(r.get("q", "")).replace("\n", " ").strip()
        y = r.get("y", None)
        y_hat = r.get("y_hat", None)
        chosen = r.get("chosen_id", None)
        score = r.get("accept_score", None)
        scenario = str(r.get("scenario", ""))
        tag = str(r.get("tag", ""))

        def _lab(nid: Any) -> str:
            try:
                nid_i = int(nid)
            except Exception:
                return ""
            lab = labels.get(nid_i, "")
            return f' "{lab}"' if lab else ""

        y_s = f"{int(y)}{_lab(y)}" if pd.notna(y) else "None"
        yh_s = f"{int(y_hat)}{_lab(y_hat)}" if pd.notna(y_hat) else "None"
        ch_s = f"{int(chosen)}{_lab(chosen)}" if pd.notna(chosen) else "None"
        sc_s = f"{float(score):.4f}" if pd.notna(score) else "NaN"

        return f'- **{scenario} / {tag}** | id={qid} | score={sc_s} | y={y_s} | y_hat={yh_s} | chosen_id(meta)={ch_s} | q="{q}"'

    # Group by (scenario, tag)
    groups = e.groupby(["scenario", "tag"], dropna=False)
    for (scenario, tag), g in groups:
        lines.append(f"## Scenario={scenario} | Tag={tag}")
        lines.append("")

        g_in = g[g["y"] != -1]
        wrong_accept = g_in[(g_in["y_hat"] != -1) & (g_in["y_hat"] != g_in["y"])]
        abstain = g_in[(g_in["y_hat"] == -1)]
        oom = g[g["y"] == -1]
        oom_false_accept = oom[(oom["y_hat"] != -1)]

        lines.append("### 1) In-map wrong accept (high score wrong accept)")
        if len(wrong_accept) == 0:
            lines.append("- (none)")
        else:
            for _, r in _pick(wrong_accept, args.k).iterrows():
                lines.append(_fmt_row(r))
        lines.append("")

        lines.append("### 2) In-map abstain (high score abstain)")
        if len(abstain) == 0:
            lines.append("- (none)")
        else:
            for _, r in _pick(abstain, args.k).iterrows():
                lines.append(_fmt_row(r))
        lines.append("")

        lines.append("### 3) OoM false accept (high score false accept)")
        if len(oom_false_accept) == 0:
            lines.append("- (none)")
        else:
            for _, r in _pick(oom_false_accept, args.k).iterrows():
                lines.append(_fmt_row(r))
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {args.out_md}")


if __name__ == "__main__":
    main()

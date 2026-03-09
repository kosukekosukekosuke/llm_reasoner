#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix B (CL12): Dev/Test non-overlap check script.

What it checks
  1) Canonical exact duplicates inside each split (Dev only / Test only)
  2) Canonical exact overlaps between Dev and Test
  3) Near-duplicate candidates between Dev and Test using string similarity

Canonicalization policy (documented in Phase 1 spec)
  - Unicode NFKC
  - collapse whitespace
  - lowercase
  - remove punctuation (keep letters/digits/underscore and Japanese scripts)
  - collapse whitespace again

Near-duplicate policy
  - difflib.SequenceMatcher ratio on canonical strings
  - report candidate pairs with ratio >= threshold

Usage
  python appendixB_check_dev_test_overlap.py \
    --dev dev_core.json dev_conflict.json dev_noisy.json dev_oom.json \
    --test test_core.json test_conflict.json test_noisy.json test_oom.json \
    --threshold 0.92 \
    --out appendixB_dev_test_overlap_report.json

Exit code
  - 0 if canonical overlap count == 0 and candidate pair count == 0
  - 1 otherwise
"""
from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from difflib import SequenceMatcher
import heapq
from typing import Dict, List, Tuple

PUNCT_RE = re.compile(r"[^\w\s\u3040-\u30ff\u3400-\u9fff]")

def one_line_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def canonical_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = one_line_text(s).lower()
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_json_list(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data

def count_internal_canonical_dups(examples: List[dict]) -> int:
    seen = set()
    dups = 0
    for ex in examples:
        c = canonical_text(ex["q"])
        if c in seen:
            dups += 1
        else:
            seen.add(c)
    return dups

def canonical_set(examples: List[dict]) -> set:
    return {canonical_text(ex["q"]) for ex in examples}

def near_duplicate_scan(dev: List[dict], test: List[dict], threshold: float, topk: int = 20) -> Tuple[int, List[dict]]:
    """Return (candidate_count, top-k candidate pairs for manual inspection).

    - compares only within same language to reduce noise
    - uses difflib.SequenceMatcher ratio on canonicalized strings
    - keeps only the top-k highest ratios among pairs that pass the threshold
    """
    dev_list = [(ex["id"], ex["lang"], ex["q"], canonical_text(ex["q"])) for ex in dev]
    test_list = [(ex["id"], ex["lang"], ex["q"], canonical_text(ex["q"])) for ex in test]

    cnt = 0
    heap: List[Tuple[float, str, str, str, str, str]] = []  # (ratio, dev_id, test_id, lang, dev_q, test_q)

    for did, dlang, dq, dcan in dev_list:
        for tid, tlang, tq, tcan in test_list:
            if dlang != tlang:
                continue
            r = SequenceMatcher(None, dcan, tcan).ratio()
            if r >= threshold:
                cnt += 1
                if topk <= 0:
                    continue
                item = (r, did, tid, dlang, dq, tq)
                if len(heap) < topk:
                    heapq.heappush(heap, item)
                else:
                    if r > heap[0][0]:
                        heapq.heapreplace(heap, item)

    pairs = sorted(heap, key=lambda x: x[0], reverse=True)
    top_pairs = [
        {
            "ratio": round(r, 6),
            "lang": lang,
            "dev_id": did,
            "test_id": tid,
            "dev_q": dq,
            "test_q": tq,
        }
        for (r, did, tid, lang, dq, tq) in pairs
    ]
    return cnt, top_pairs

def near_duplicate_count(dev: List[dict], test: List[dict], threshold: float) -> int:
    return near_duplicate_scan(dev, test, threshold, topk=0)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", nargs="+", required=True, help="Dev JSON files (list of examples).")
    ap.add_argument("--test", nargs="+", required=True, help="Test JSON files (list of examples).")
    ap.add_argument("--threshold", type=float, default=0.92, help="Near-duplicate similarity threshold.")
    ap.add_argument("--topk", type=int, default=20, help="Keep top-k near-duplicate pairs in the report (for manual inspection).")
    ap.add_argument("--out", type=str, default="appendixB_dev_test_overlap_report.json", help="Output JSON report.")
    args = ap.parse_args()

    dev_files = [Path(p) for p in args.dev]
    test_files = [Path(p) for p in args.test]

    dev_all: List[dict] = []
    test_all: List[dict] = []

    dev_by_file: Dict[str, List[dict]] = {}
    for p in dev_files:
        ds = load_json_list(p)
        dev_by_file[p.name] = ds
        dev_all.extend(ds)

    test_by_file: Dict[str, List[dict]] = {}
    for p in test_files:
        ds = load_json_list(p)
        test_by_file[p.name] = ds
        test_all.extend(ds)

    # near-duplicate scan (Dev vs Test)
    near_cnt, near_pairs = near_duplicate_scan(dev_all, test_all, args.threshold, topk=args.topk)

    # counts
    report = {
        "counts": {
            "dev_total": len(dev_all),
            "test_total": len(test_all),
        },
        "internal_dups": {
            "dev_canonical_dup_count": count_internal_canonical_dups(dev_all),
            "test_canonical_dup_count": count_internal_canonical_dups(test_all),
        },
        "overlap": {
            "dev_test_canonical_overlap_count": len(canonical_set(dev_all) & canonical_set(test_all)),
        },
        "near_duplicates": {
            "threshold": args.threshold,
            "candidate_pair_count": near_cnt,
            "topk": args.topk,
            "candidate_pairs_topk": near_pairs,
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = (
        report["internal_dups"]["dev_canonical_dup_count"] == 0 and
        report["internal_dups"]["test_canonical_dup_count"] == 0 and
        report["overlap"]["dev_test_canonical_overlap_count"] == 0 and
        report["near_duplicates"]["candidate_pair_count"] == 0
    )
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
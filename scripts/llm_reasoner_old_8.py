#!/usr/bin/env python3

"""
LLM Reasoner ノード（Phase1：cos 分布＋相対 cos 導入版）。

- 意味情報付きノード地図（YAML）とユーザ指示（自然言語）から
  「どのノードに向かうべきか」を LLM に 1 つ選ばせる。
- LLM 出力の confidence と、クエリ文 vs ノード意味情報の埋め込み
  コサイン類似度を使って combined_score を計算し、受理／再推論を判断する。

Phase1 で追加したポイント：
- クエリごとに「全ノードの cos」を計算し、
  min / max / mean / std / top1 / top2 / margin_top1_top2 を meta["cosine_stats"] に保存。
- cosine_norm_mode（"none" / "minmax" / "zscore"）で
  判定に使う cosine を「相対値」に切り替え可能にした。
  → combined_score を 0〜1 スケールのまま、confidence とバランスよく扱えるようにした版。
"""

import os
import re
import json
import math
import rospy
import yaml
import numpy as np

from std_msgs.msg import String, Int32, Float32
from llama_cpp import Llama

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class LLMReasonerNode:
    # =========================================================
    # 初期化
    # =========================================================
    def __init__(self):
        rospy.init_node("llm_reasoner", anonymous=False)

        # --- パス関連 ---
        self.map_yaml_path = rospy.get_param(
            "~map_yaml_path",
            "/home/amsl/catkin_ws/src/rover_navigator/map/graph/ikuta_graph_eval_v0.yaml",
        )
        self.model_path = rospy.get_param(
            "~model_path",
            os.path.join(
                os.path.dirname(__file__),
                "llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
            ),
        )

        # --- LLM生成パラメータ ---
        self.n_ctx = rospy.get_param("~ctx_size", 2048)
        self.n_threads = rospy.get_param("~threads", 4)
        self.temperature = rospy.get_param("~temperature", 0.1)
        self.top_k = rospy.get_param("~top_k", 10)
        self.top_p = rospy.get_param("~top_p", 0.0)
        self.max_tokens = rospy.get_param("~max_tokens", 64)

        # --- 判定モード / 信頼度パラメータ ---
        # "and" or "weighted"
        self.decision_mode = rospy.get_param("~decision_mode", "weighted").lower()
        if self.decision_mode not in ("and", "weighted"):
            self.decision_mode = "weighted"

        # max_retries 回まで再推論（合計試行回数 = max_retries + 1）
        self.max_retries = int(rospy.get_param("~max_retries", 2))

        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.5))
        self.cosine_threshold = float(rospy.get_param("~cosine_threshold", 0.5))
        self.combined_threshold = float(rospy.get_param("~combined_threshold", 0.6))
        self.alpha = float(rospy.get_param("~alpha", 0.5))  # weighted 用重み

        # 🆕 cosine の相対化モード（Phase1 追加）
        # "none"   : 従来通り raw cosine をそのまま使用
        # "minmax" : クエリ内で min-max 正規化（0〜1）
        # "zscore" : クエリ内で Z-score 正規化
        self.cosine_norm_mode = rospy.get_param("~cosine_norm_mode", "minmax").lower()
        if self.cosine_norm_mode not in ("none", "minmax", "zscore"):
            self.cosine_norm_mode = "minmax"

        self.fallback_node = int(rospy.get_param("~fallback_node", 0))

        # --- 埋め込み関連 ---
        self.use_embedding = bool(rospy.get_param("~use_embedding", True))
        self.embed_model_name = rospy.get_param("~embed_model_name", "BAAI/bge-m3")
        self.topk_semantic = int(rospy.get_param("~topk_semantic", 3))

        # --- Topic 名（基本固定。必要なら param で上書き可） ---
        self.sub_query_topic = rospy.get_param("~sub_query", "/llm_reasoner/query")
        self.pub_node_topic = rospy.get_param(
            "~pub_node", "/llm_reasoner/chosen_node_id"
        )
        self.pub_conf_topic = rospy.get_param(
            "~pub_conf", "/llm_reasoner/confidence"
        )
        self.pub_cos_topic = rospy.get_param("~pub_cos", "/llm_reasoner/cosine")
        self.pub_meta_topic = rospy.get_param("~pub_meta", "/llm_reasoner/meta")

        # --- ノード情報読み込み ---
        (
            self.node_ids,
            self.node_labels,
            self.node_semantics,
            self.node_descs,
        ) = self.load_map(self.map_yaml_path)

        if not self.node_ids:
            rospy.logerr("No nodes loaded from map. Abort.")
            raise SystemExit

        rospy.loginfo(f"📄 Loaded {len(self.node_ids)} nodes from map.")

        # ID -> index / label / semantic の辞書
        self.id2idx = {int(nid): i for i, nid in enumerate(self.node_ids)}
        self.id2label = {
            int(nid): str(lbl) for nid, lbl in zip(self.node_ids, self.node_labels)
        }
        self.id2sem = {
            int(nid): str(sem)
            for nid, sem in zip(self.node_ids, self.node_semantics)
        }

        # --- LLMモデル読み込み ---
        rospy.loginfo(f"🧩 Loading LLM model from: {self.model_path}")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            logits_all=True,
            verbose=False,
        )

        # --- 埋め込みモデル＆ノード埋め込み ---
        self.embed_model = None
        self.node_embeds = None

        if self.use_embedding and SentenceTransformer is not None:
            try:
                device = "cpu"
                rospy.loginfo(
                    f"🧠 Loading embedding model: {self.embed_model_name} (device={device})"
                )
                self.embed_model = SentenceTransformer(
                    self.embed_model_name, device=device
                )
                self.precompute_node_embeddings()
                rospy.loginfo("🧠 Node embeddings ready.")
            except Exception as e:
                rospy.logwarn(
                    f"⚠️ Failed to initialize embedding model: {e}. Disable embeddings."
                )
                self.use_embedding = False
        else:
            if not SentenceTransformer:
                rospy.logwarn("⚠️ sentence_transformers is not available. Disable embeddings.")
            self.use_embedding = False

        # --- ROS Pub/Sub ---
        self.sub_query = rospy.Subscriber(
            self.sub_query_topic, String, self.callback_query
        )
        self.pub_node = rospy.Publisher(self.pub_node_topic, Int32, queue_size=10)
        self.pub_conf = rospy.Publisher(self.pub_conf_topic, Float32, queue_size=10)
        self.pub_cos = rospy.Publisher(self.pub_cos_topic, Float32, queue_size=10)
        self.pub_meta = rospy.Publisher(self.pub_meta_topic, String, queue_size=10)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("✅ LLM Reasoner Node ready.")
        rospy.spin()

    # =========================================================
    # マップ & 埋め込みユーティリティ
    # =========================================================
    def load_map(self, filename):
        if not os.path.exists(filename):
            rospy.logerr(f"Map YAML not found: {filename}")
            return [], [], [], []

        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        ids, labels, semantics, descs = [], [], [], []
        for node in data.get("NODE", []):
            nid = int(node.get("id"))
            label = str(node.get("label", "")).strip()
            sem_list = node.get("semantic", [])
            sem = ", ".join(map(str, sem_list)) if sem_list else ""
            desc = str(node.get("description", "")).strip()
            ids.append(nid)
            labels.append(label)
            semantics.append(sem)
            descs.append(desc)
        return ids, labels, semantics, descs

    def precompute_node_embeddings(self):
        texts = []
        for nid in self.node_ids:
            i = self.id2idx[nid]
            label = self.node_labels[i]
            sem = self.node_semantics[i]
            desc = self.node_descs[i]
            txt = f"{label}. {sem}. {desc}".strip()
            texts.append(txt)
        self.node_embeds = self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, query_text):
        if not (self.use_embedding and self.embed_model):
            return None
        return self.embed_model.encode(
            query_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def cosine_for_node(self, query_vec, node_id):
        if (query_vec is None) or (self.node_embeds is None):
            return None
        idx = self.id2idx.get(int(node_id))
        if idx is None:
            return None
        return float(np.dot(self.node_embeds[idx], query_vec))

    def topk_semantic_ids(self, query_vec, k):
        if (query_vec is None) or (self.node_embeds is None):
            return []
        k = max(1, min(int(k), len(self.node_ids)))
        sims = np.dot(self.node_embeds, query_vec)
        idxs = np.argsort(-sims)[:k]
        return [int(self.node_ids[i]) for i in idxs]

    def compute_cosine_stats(self, query_vec):
        """
        1クエリについて、全ノードに対する cos 分布と統計量を事前計算する。
        戻り値:
            {
                "all": np.ndarray shape (N,),  # 各ノードの raw cosine
                "min": float,
                "max": float,
                "mean": float,
                "std": float,
                "top1": float,
                "top2": float,
                "margin_top1_top2": float,
            }
        """
        if (query_vec is None) or (self.node_embeds is None):
            return None

        cos_all = np.dot(self.node_embeds, query_vec)  # raw cosine（正規化済み埋め込みなので内積=cos）
        cos_min = float(np.min(cos_all))
        cos_max = float(np.max(cos_all))
        cos_mean = float(np.mean(cos_all))
        cos_std = float(np.std(cos_all))

        if cos_all.shape[0] >= 2:
            sorted_vals = np.sort(cos_all)[::-1]
            top1 = float(sorted_vals[0])
            top2 = float(sorted_vals[1])
            margin = float(top1 - top2)
        else:
            top1 = cos_max
            top2 = cos_max
            margin = 0.0

        return {
            "all": cos_all,
            "min": cos_min,
            "max": cos_max,
            "mean": cos_mean,
            "std": cos_std,
            "top1": top1,
            "top2": top2,
            "margin_top1_top2": margin,
        }

    # =========================================================
    # プロンプト生成
    # =========================================================
    def build_full_prompt(self, query_text):
        lines = ["Available locations:"]
        for nid, lbl, sem, desc in zip(
            self.node_ids, self.node_labels, self.node_semantics, self.node_descs
        ):
            lines.append(f"{nid}. {lbl} ({sem}) - {desc}")
        lines.append(f"User task: {query_text}")
        lines.append("Answer with the NUMBER of the correct choice:")
        return "\n".join(lines)

    def build_refine_prompt(self, query_text, candidate_ids):
        lines = ["Candidate locations:"]
        for nid in candidate_ids:
            i = self.id2idx[nid]
            lbl = self.node_labels[i]
            sem = self.node_semantics[i]
            desc = self.node_descs[i]
            lines.append(f"{nid}. {lbl} ({sem}) - {desc}")
        lines.append(f"User task: {query_text}")
        lines.append(
            "The previous decision was uncertain. "
            "Based on the user task and the candidate locations above, "
            "choose the single best destination."
        )
        lines.append("Answer with the NUMBER of the correct choice:")
        return "\n".join(lines)

    # =========================================================
    # LLM 呼び出し & 出力パース
    # =========================================================
    def call_llm(self, prompt):
        """
        LLM を1回呼び出す。
        戻り値:
            raw_text: 生テキスト
            confidence: 平均 logprob の exp（なければ None）
        """
        out = self.llm(
            prompt=prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["Answer with the NUMBER of the correct choice:"],
            echo=False,
            logprobs=1,
        )

        text = out["choices"][0]["text"].strip()
        lp_list = out["choices"][0].get("logprobs", {}).get("token_logprobs", [])
        if lp_list:
            vals = [lp for lp in lp_list if lp is not None]
            if vals:
                avg_lp = float(sum(vals) / len(vals))
                conf = float(math.exp(avg_lp))
            else:
                conf = None
        else:
            conf = None
        return text, conf

    @staticmethod
    def extract_number(text):
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else None

    # =========================================================
    # 単一試行の判定ロジック
    # =========================================================
    def judge_attempt(self, num, candidate_ids, conf, cos, combined):
        """
        1回の試行結果に対して「accept or retry」を判定。
        戻り値:
            decision: "accept" or "retry"
            reject_reason: 不採用理由 or None
            pass_conf, pass_cos, pass_combined: 閾値判定結果
        """
        pass_conf = None
        pass_cos = None
        pass_combined = None
        reject_reason = None

        # 数字がない/不正
        if num is None:
            return "retry", "no_number", pass_conf, pass_cos, pass_combined
        if num not in candidate_ids:
            return "retry", "invalid_id", pass_conf, pass_cos, pass_combined

        # 閾値判定
        if conf is not None:
            pass_conf = conf >= self.conf_threshold
        if cos is not None:
            pass_cos = cos >= self.cosine_threshold
        if combined is not None:
            pass_combined = combined >= self.combined_threshold

        # --- AND モード ---
        if self.decision_mode == "and":
            conds = []
            if pass_conf is not None:
                conds.append(pass_conf)
            if pass_cos is not None:
                conds.append(pass_cos)

            if conds and all(conds):
                return "accept", None, pass_conf, pass_cos, pass_combined

            # 不採用理由
            if pass_conf is False and pass_cos is False:
                reject_reason = "low_both"
            elif pass_conf is False:
                reject_reason = "low_conf"
            elif pass_cos is False:
                reject_reason = "low_cos"
            else:
                reject_reason = "unclear"
            return "retry", reject_reason, pass_conf, pass_cos, pass_combined

        # --- weighted モード ---
        if combined is not None and pass_combined is not None:
            if pass_combined:
                return "accept", None, pass_conf, pass_cos, pass_combined
            else:
                return "retry", "low_combined", pass_conf, pass_cos, pass_combined

        # combined が無い場合は conf / cos のどちらかが良ければ採用
        if pass_conf is True or pass_cos is True:
            return "accept", None, pass_conf, pass_cos, pass_combined

        if pass_conf is False and pass_cos is False:
            reject_reason = "low_both"
        elif pass_conf is False:
            reject_reason = "low_conf"
        elif pass_cos is False:
            reject_reason = "low_cos"
        else:
            reject_reason = "unclear"

        return "retry", reject_reason, pass_conf, pass_cos, pass_combined

    # =========================================================
    # accept_reason（最終結果ラベル）
    # =========================================================
    def classify_accept_reason(self, attempts_log, decision_source):
        if decision_source == "fallback":
            return "fallback"
        if decision_source == "last_num":
            return "last_num_salvage"
        if not attempts_log:
            return "unknown"

        # 最後に accept された attempt を探す
        accept_attempts = [a for a in attempts_log if a["decision"] == "accept"]
        if not accept_attempts:
            return "unknown"
        last = accept_attempts[-1]

        def is_strong(a):
            if self.decision_mode == "and":
                return (a.get("pass_conf") is True) and (a.get("pass_cos") is True)
            else:
                if a.get("pass_combined") is True:
                    return True
                return (a.get("pass_conf") is True) and (a.get("pass_cos") is True)

        if last["attempt_index"] == 1:
            return "first_pass_strong" if is_strong(last) else "first_pass_partial"
        else:
            return "refine_strong" if is_strong(last) else "refine_weak"

    # =========================================================
    # メイン推論ループ（コールバック）
    # =========================================================
    def callback_query(self, msg):
        query = msg.data.strip()
        if not query:
            rospy.logwarn("⚠️ Empty query received. Ignored.")
            return

        rospy.loginfo(f"🟢 Inference start: '{query}'")

        # クエリ埋め込み（必要なら）
        query_vec = self.encode_query(query) if self.use_embedding else None

        # クエリに対する cos 分布の統計
        cos_stats = None
        if query_vec is not None and self.node_embeds is not None:
            cos_stats = self.compute_cosine_stats(query_vec)

        attempts_log = []
        chosen_id = None
        chosen_conf = None
        chosen_cos = None          # こちらは「判定に使った cos（=相対値）」を入れる
        chosen_cos_raw = None      # 元の raw cosine も保持しておく
        chosen_cos_z_raw = None    # 最終的に採択された attempt の z_raw 用
        chosen_combined = None

        last_valid_num = None
        decision_source = None

        # 試行ループ（1回目: full, 2回目以降: refine）
        for attempt_idx in range(1, self.max_retries + 2):
            if attempt_idx == 1:
                prompt_type = "full"
                candidate_ids = list(self.node_ids)
                prompt = self.build_full_prompt(query)
            else:
                prompt_type = "refine"
                sem_ids = (
                    self.topk_semantic_ids(query_vec, self.topk_semantic)
                    if query_vec is not None
                    else []
                )
                prev_ids = sorted(
                    {
                        a["parsed_num"]
                        for a in attempts_log
                        if a.get("parsed_num") is not None
                    }
                )
                cand = list({nid for nid in (sem_ids + prev_ids) if nid in self.id2idx})
                if not cand:
                    cand = list(self.node_ids)
                candidate_ids = cand
                prompt = self.build_refine_prompt(query, candidate_ids)

            # LLM 呼び出し
            raw, conf = self.call_llm(prompt)
            num = self.extract_number(raw)

            # 🆕 cosine_raw / cosine_norm の計算
            cos_raw = None
            cos_norm = None
            z_raw = None

            if num is not None and query_vec is not None and cos_stats is not None:
                idx = self.id2idx.get(int(num))
                if idx is not None:
                    # raw cosine
                    cos_raw = float(cos_stats["all"][idx])

                    # 相対化（モードに応じて）
                    mode = self.cosine_norm_mode
                    if mode == "minmax":
                        denom = max(1e-6, cos_stats["max"] - cos_stats["min"])
                        cos_norm = float((cos_raw - cos_stats["min"]) / denom)
                    elif mode == "zscore":
                        denom = max(1e-6, cos_stats["std"])
                        z_raw = (cos_raw - cos_stats["mean"]) / denom
                        # 標準正規分布の CDF: z → (0,1)
                        cos_norm = 0.5 * (1.0 + math.erf(z_raw / math.sqrt(2.0)))
                    else:  # "none" -> 従来通り raw をそのまま使う
                        cos_norm = cos_raw

            # 判定・combined に使う cosine
            cos_for_decision = cos_norm

            combined = None
            if (
                self.decision_mode == "weighted"
                and conf is not None
                and cos_for_decision is not None
            ):
                combined = float(self.alpha * conf + (1.0 - self.alpha) * cos_for_decision)
            elif conf is not None:
                combined = conf

            # 判定
            decision, reject_reason, pass_conf, pass_cos, pass_combined = (
                self.judge_attempt(num, candidate_ids, conf, cos_for_decision, combined)
            )

            # salvage 用
            if num is not None and num in candidate_ids:
                last_valid_num = num

            # ログ用表示値
            num_disp = str(num) if num is not None else "-"
            conf_disp = f"{conf:.3f}" if conf is not None else "-"
            cos_disp = (
                f"{cos_for_decision:.3f}"
                if cos_for_decision is not None
                else "-"
            )
            comb_disp = f"{combined:.3f}" if combined is not None else "-"

            if decision == "accept":
                emoji = "✅"
                rospy.loginfo(
                    f"  {emoji} Attempt {attempt_idx} [{prompt_type}] "
                    f"-> raw='{raw[:60]}' num={num_disp} "
                    f"conf={conf_disp} cos={cos_disp} comb={comb_disp} decision=accept"
                )
            else:
                emoji = "🔁"
                rospy.loginfo(
                    f"  {emoji} Attempt {attempt_idx} [{prompt_type}] "
                    f"-> raw='{raw[:60]}' num={num_disp} "
                    f"conf={conf_disp} cos={cos_disp} comb={comb_disp} "
                    f"decision=retry reason={reject_reason}"
                )

            # attempt ログ保存（メタ情報用）
            attempts_log.append(
                {
                    "attempt_index": attempt_idx,
                    "prompt_type": prompt_type,
                    "candidate_ids": candidate_ids,
                    "raw_output": raw,
                    "parsed_num": num,
                    "confidence": conf,
                    "cosine_raw": cos_raw,
                    "cosine": cos_for_decision,
                    "cosine_z_raw": z_raw,
                    "combined_score": combined,
                    "pass_conf": pass_conf,
                    "pass_cos": pass_cos,
                    "pass_combined": pass_combined,
                    "decision": "accept" if decision == "accept" else "retry",
                    "reject_reason": reject_reason if decision != "accept" else None,
                }
            )

            if decision == "accept":
                chosen_id = num
                chosen_conf = conf
                chosen_cos = cos_for_decision
                chosen_cos_raw = cos_raw
                chosen_cos_z_raw = z_raw
                chosen_combined = combined
                decision_source = "model"
                break

        # =====================================================
        # 最終決定（salvage / fallback）
        # =====================================================
        if chosen_id is None:
            if last_valid_num is not None:
                rospy.logwarn(
                    f"⚠️ Using last valid number as salvage: {last_valid_num}"
                )
                chosen_id = int(last_valid_num)
                decision_source = "last_num"
            else:
                rospy.logerr(
                    f"❌ No valid response. Use fallback_node={self.fallback_node}."
                )
                chosen_id = int(self.fallback_node)
                decision_source = "fallback"

        accept_reason = self.classify_accept_reason(attempts_log, decision_source)

        # publish values
        label = self.id2label.get(chosen_id, "unknown")
        sem = self.id2sem.get(chosen_id, "")

        self.pub_node.publish(Int32(chosen_id))
        self.pub_conf.publish(
            Float32(chosen_conf if chosen_conf is not None else -1.0)
        )
        self.pub_cos.publish(Float32(chosen_cos if chosen_cos is not None else -1.0))

        rospy.loginfo(
            f"✅ Inference done: node={chosen_id} ({label}), "
            f"semantics={sem}, accept_reason={accept_reason}, attempts={len(attempts_log)}"
        )

        # メタ情報（評価用）
        meta = {
            "query": query,
            "node_id": int(chosen_id),
            "label": label,
            "label_semantics": sem,
            "confidence": chosen_conf,
            "cosine": chosen_cos,
            "cosine_raw": chosen_cos_raw,
            "combined_score": chosen_combined,
            "cosine_z_raw": chosen_cos_z_raw,
            "decision_mode": self.decision_mode,
            "accept_reason": accept_reason,
            "attempts": len(attempts_log),
            "fallback_used": decision_source == "fallback",
            "fallback_node": int(self.fallback_node),
            "decision_source": decision_source,
            "thresholds": {
                "conf_threshold": self.conf_threshold,
                "cosine_threshold": self.cosine_threshold,
                "combined_threshold": self.combined_threshold,
                "alpha": self.alpha,
            },
            "valid_ids": list(self.node_ids),
            "attempt_logs": attempts_log,
        }

        if self.use_embedding and self.node_embeds is not None and query_vec is not None:
            top_ids = self.topk_semantic_ids(query_vec, self.topk_semantic)
            top_entries = []
            for nid in top_ids:
                c = self.cosine_for_node(query_vec, nid)
                top_entries.append(
                    {
                        "node_id": int(nid),
                        "label": self.id2label.get(nid, ""),
                        "cosine": c,
                    }
                )
            meta["embed_diag"] = {
                "enabled": True,
                "model": self.embed_model_name,
                "text_mode": "label+semantic+description",
                "topk_semantic": top_entries,
            }
        else:
            meta["embed_diag"] = {"enabled": False}

        # cosine の統計
        if cos_stats is not None:
            meta["cosine_stats"] = {
                "min": cos_stats["min"],
                "max": cos_stats["max"],
                "mean": cos_stats["mean"],
                "std": cos_stats["std"],
                "top1": cos_stats["top1"],
                "top2": cos_stats["top2"],
                "margin_top1_top2": cos_stats["margin_top1_top2"],
                "norm_mode": self.cosine_norm_mode,
            }

        self.pub_meta.publish(String(json.dumps(meta)))

    # =========================================================
    # シャットダウン
    # =========================================================
    def on_shutdown(self):
        rospy.loginfo("🧹 Shutting down LLM Reasoner Node...")
        try:
            del self.llm
        except Exception:
            pass
        rospy.loginfo("✅ Done.")


if __name__ == "__main__":
    try:
        LLMReasonerNode()
    except rospy.ROSInterruptException:
        pass

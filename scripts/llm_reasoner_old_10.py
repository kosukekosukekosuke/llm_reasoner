#!/usr/bin/env python3

"""
LLM Reasoner ノード（Phase1〜3：cos 相対化＋abstain/OOD＋埋め込み分解・適応重み版）

- 意味情報付きノード地図（YAML）とユーザ指示（自然言語）から、
  「どのノードに向かうべきか」を LLM に 1 つ選ばせるノード。
- LLM 出力の confidence と、クエリ文 vs ノード意味情報の埋め込み
  コサイン類似度から combined_score を計算し、受理／再推論／棄権(-1) を決定する。

Phase1（cos 分布＋相対 cos）のポイント：
- クエリごとに「全ノードの cos 分布」を計算し、
  min / max / mean / std / top1 / top2 / margin_top1_top2 を meta["cosine_stats"] に保存。
- cosine_norm_mode（"none" / "minmax" / "zscore"）で、
  判定に使う cosine を「相対値」に切り替え可能にし、
  combined_score を 0〜1 スケールのまま confidence とバランスよく扱えるようにした。

Phase2（abstain(-1) ＋ OOD ゲート）のポイント：
- max_cos が低い／top1−top2 の margin が小さいクエリを、
  LLM を呼ぶ前に「マップ外(OOD)っぽい」とみなして abstain_node_id(-1) を返す OOD ゲートを追加。
- 再推論を含めて一度も閾値(conf / cosine / combined)を満たせなかった場合は、
  「わからない」として abstain_no_accept（node_id = -1）で棄権。
- abstain 時も、embedding 側の max_cos や「一番マシな試行」の confidence / cosine / combined_score を
  meta に残し、どの程度あやしかったかを後から解析できるようにした。

Phase3（埋め込み分解＋attempt ごとの観点切り替え）のポイント：
- ノード埋め込みを label / semantic(tags) / description の 3 つに分解し、
  各 cos（cos_label_raw / cos_sem_raw / cos_desc_raw）をログに保存。
- 第 1 試行では launch で指定した base 重み（w_label, w_sem, w_desc）で cos を合成。
- 第 2 試行以降は、直前の決定ノードに対する 3 つの cos を z-score 正規化し、
  softmax で「どの観点が効いていそうか」を推定して重みを自動調整し、
  再推論ではその観点を少し強調した合成 cosine を用いる。
- これにより、「どの語義（ラベル／タグ／説明）が効いているのか」をログから分析でき、
  再推論時には前回の傾向を踏まえた観点切り替えができるようになっている。
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

        # パス関連
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

        # LLM生成パラメータ
        self.n_ctx = rospy.get_param("~ctx_size", 2048)
        self.n_threads = rospy.get_param("~threads", 4)
        self.temperature = rospy.get_param("~temperature", 0.1)
        self.top_k = rospy.get_param("~top_k", 10)
        self.top_p = rospy.get_param("~top_p", 0.0)
        self.max_tokens = rospy.get_param("~max_tokens", 64)

        # 判定モード / 信頼度パラメータ
        self.decision_mode = rospy.get_param("~decision_mode", "weighted").lower()
        if self.decision_mode not in ("and", "weighted"):
            self.decision_mode = "weighted"

        # max_retries 回まで再推論（合計試行回数 = max_retries + 1）
        self.max_retries = int(rospy.get_param("~max_retries", 2))

        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.5))
        self.cosine_threshold = float(rospy.get_param("~cosine_threshold", 0.5))
        self.combined_threshold = float(rospy.get_param("~combined_threshold", 0.6))
        self.alpha = float(rospy.get_param("~alpha", 0.5))  # weighted 用重み

        # abstain(-1) / OOD ゲート用パラメータ
        self.abstain_node_id = int(rospy.get_param("~abstain_node_id", -1))
        self.ood_max_cos_threshold = float(rospy.get_param("~ood_max_cos_threshold", 0.0))
        self.ood_margin_threshold = float(rospy.get_param("~ood_margin_threshold", 0.0))

        # cosine の相対化モード
        self.cosine_norm_mode = rospy.get_param("~cosine_norm_mode", "minmax").lower()
        if self.cosine_norm_mode not in ("none", "minmax", "zscore"):
            self.cosine_norm_mode = "minmax"

        # --- cos の成分重み（base）---
        # 初回試行で使う基準重み（合計が 1.0 になるように正規化する）
        self.cos_w_label_base = float(rospy.get_param("~cos_w_label_base", 0.3))
        self.cos_w_sem_base   = float(rospy.get_param("~cos_w_sem_base",   0.4))
        self.cos_w_desc_base  = float(rospy.get_param("~cos_w_desc_base",  0.3))
        s = self.cos_w_label_base + self.cos_w_sem_base + self.cos_w_desc_base
        if s <= 0.0:
            # 万一全部 0 以下ならデフォルトに戻す
            self.cos_w_label_base = 0.3
            self.cos_w_sem_base   = 0.4
            self.cos_w_desc_base  = 0.3
        else:
            self.cos_w_label_base /= s
            self.cos_w_sem_base   /= s
            self.cos_w_desc_base  /= s

        # --- 再試行時の自動重み調整 ---
        self.cos_adapt_enable   = bool(rospy.get_param("~cos_adapt_enable", True))
        # 0.0: まったく適応しない（常に base のまま）
        # 1.0: 完全に「前回の z-score による softmax 重み」に寄せる
        self.cos_adapt_strength = float(rospy.get_param("~cos_adapt_strength", 0.7))
        # z-score -> softmax するときの温度
        self.cos_adapt_tau      = float(rospy.get_param("~cos_adapt_tau", 1.0))

        # 埋め込み関連
        self.use_embedding = bool(rospy.get_param("~use_embedding", True))
        self.embed_model_name = rospy.get_param("~embed_model_name", "BAAI/bge-m3")
        self.topk_semantic = int(rospy.get_param("~topk_semantic", 3))

        # Topic 名（基本固定。必要なら param で上書き可）
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
        self.node_embeds_label = None
        self.node_embeds_sem = None
        self.node_embeds_desc = None

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
        """
        ノードの埋め込みをlabel 用、semantic(tags) 用、description 用に分けて事前計算する。
        """
        if self.embed_model is None:
            return

        label_texts = []
        sem_texts   = []
        desc_texts  = []

        for nid in self.node_ids:
            i = self.id2idx[nid]
            label = self.node_labels[i] or ""
            sem   = self.node_semantics[i] or ""
            desc  = self.node_descs[i] or ""

            label_texts.append(str(label))
            sem_texts.append(str(sem))
            desc_texts.append(str(desc))

        # それぞれ別々に encode
        self.node_embeds_label = self.embed_model.encode(
            label_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.node_embeds_sem = self.embed_model.encode(
            sem_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.node_embeds_desc = self.embed_model.encode(
            desc_texts,
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

    def topk_semantic_ids(self, query_vec, k):
        """
        semantic 用埋め込み（tags）との cos に基づいて上位 k ノードの id を返す。
        """
        if (query_vec is None) or (self.node_embeds_sem is None):
            return []
        k = max(1, min(int(k), len(self.node_ids)))
        sims = np.dot(self.node_embeds_sem, query_vec)
        idxs = np.argsort(-sims)[:k]
        return [int(self.node_ids[i]) for i in idxs]

    def compute_cos_components(self, query_vec):
        """
        クエリ埋め込み query_vec に対するcos_label_all[i]、cos_sem_all[i]、cos_desc_all[i]をまとめて計算する。
        それぞれ np.ndarray (shape (N,)) か None を返す。
        """
        if query_vec is None:
            return None

        cos_label = None
        cos_sem   = None
        cos_desc  = None

        if self.node_embeds_label is not None:
            cos_label = np.dot(self.node_embeds_label, query_vec)
        if self.node_embeds_sem is not None:
            cos_sem = np.dot(self.node_embeds_sem, query_vec)
        if self.node_embeds_desc is not None:
            cos_desc = np.dot(self.node_embeds_desc, query_vec)

        return {
            "label": cos_label,
            "sem":   cos_sem,
            "desc":  cos_desc,
        }

    def compute_cosine_stats(self, cos_all):
        """
        cos_all: shape (N,) の np.ndarray（label/semantic/description を重み付きで合成した cos_total）に対して統計量を計算する。
        戻り値:
            {
                "all": np.ndarray shape (N,),  # 合成後 cos_total
                "min": float,
                "max": float,
                "mean": float,
                "std": float,
                "top1": float,
                "top2": float,
                "margin_top1_top2": float,
            }
        """
        if cos_all is None:
            return None

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

    def get_base_cos_weights(self):
        """
        初回試行で使う基準の (w_label, w_sem, w_desc) を返す。
        """
        return (self.cos_w_label_base, self.cos_w_sem_base, self.cos_w_desc_base)

    def get_cos_weights_for_attempt(self, attempt_idx, attempts_log, cos_components):
        """
        試行番号と過去の試行ログ・cos 成分に基づいて
        (w_label, w_sem, w_desc) を決める。

        - attempt 1: base 重みそのまま
        - attempt >=2:
            直前の試行で LLM が選んだノードについて、
            label/sem/desc の z-score をとり、
            それを softmax して「今回の観点」を決める。
            さらに base 重みと cos_adapt_strength で線形補間する。
        """
        base_w = self.get_base_cos_weights()

        # 初回 or 適応無効ならそのまま base
        if (attempt_idx == 1) or (not self.cos_adapt_enable) or (cos_components is None):
            return base_w

        # 直前の「数字が取れている」試行を探す
        prev = None
        for a in reversed(attempts_log):
            if a.get("parsed_num") is not None:
                prev = a
                break
        if prev is None:
            return base_w

        num = prev.get("parsed_num")
        if num is None:
            return base_w
        idx = self.id2idx.get(int(num))
        if idx is None:
            return base_w

        cos_label_all = cos_components.get("label")
        cos_sem_all   = cos_components.get("sem")
        cos_desc_all  = cos_components.get("desc")

        scores = []
        for name, all_vals in [
            ("label", cos_label_all),
            ("sem",   cos_sem_all),
            ("desc",  cos_desc_all),
        ]:
            if all_vals is None:
                scores.append((name, 0.0))
                continue
            c_val = float(all_vals[idx])
            m = float(np.mean(all_vals))
            s = float(np.std(all_vals))
            if s < 1e-6:
                z = 0.0
            else:
                z = (c_val - m) / s  # 平均よりどれくらい上か（z-score）
            scores.append((name, z))

        # z-score -> softmax で非負の重みに変換
        z_vec = np.array([s for (_, s) in scores], dtype=float)
        tau = self.cos_adapt_tau if self.cos_adapt_tau > 0.0 else 1.0
        z_vec = z_vec / tau
        z_vec = z_vec - np.max(z_vec)  # 安定化
        w_raw = np.exp(z_vec)
        if w_raw.sum() <= 0.0:
            return base_w
        w_soft = w_raw / w_raw.sum()

        # softmax から得た動的重み
        w_label_dyn = float(w_soft[0])
        w_sem_dyn   = float(w_soft[1])
        w_desc_dyn  = float(w_soft[2])

        # base 重みとの線形補間
        beta = max(0.0, min(1.0, self.cos_adapt_strength))
        bL, bS, bD = base_w
        w_label = (1.0 - beta) * bL + beta * w_label_dyn
        w_sem   = (1.0 - beta) * bS + beta * w_sem_dyn
        w_desc  = (1.0 - beta) * bD + beta * w_desc_dyn

        # 正規化
        s = w_label + w_sem + w_desc
        if s <= 0.0:
            return base_w
        w_label /= s
        w_sem   /= s
        w_desc  /= s

        return (w_label, w_sem, w_desc)

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

        # abstain 系（OOD ゲート / しきい値を満たす試行なし）
        if decision_source is not None and decision_source.startswith("abstain_"):
            # 例: "abstain_ood_low_maxcos", "abstain_ood_small_margin", "abstain_no_accept"
            return decision_source

        if not attempts_log:
            # OOD ゲートで LLM を1回も呼ばなかった場合など、
            # attempts_log が空で decision_source も上のどれでもないケース
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

        # このクエリに対する cos 成分 ＆ 合成 cos_total の統計
        cos_components = None   # {"label": np.ndarray, "sem": ..., "desc": ...}
        cos_stats = None        # 合成 cos_total に対する統計
        cos_total_all = None    # 合成 cos_total 全ノード分

        if query_vec is not None and (
            self.node_embeds_label is not None
            or self.node_embeds_sem is not None
            or self.node_embeds_desc is not None
        ):
            cos_components = self.compute_cos_components(query_vec)
            cos_label_all = cos_components.get("label")
            cos_sem_all   = cos_components.get("sem")
            cos_desc_all  = cos_components.get("desc")

            # base 重みで cos_total_all を作る（OOD ゲートや統計はこれに基づく）
            wL_base, wS_base, wD_base = self.get_base_cos_weights()

            # None の成分は 0 扱い
            if cos_label_all is None:
                cos_label_all = 0.0
            if cos_sem_all is None:
                cos_sem_all = 0.0
            if cos_desc_all is None:
                cos_desc_all = 0.0

            cos_total_all = (
                wL_base * cos_label_all
                + wS_base * cos_sem_all
                + wD_base * cos_desc_all
            )

            cos_stats = self.compute_cosine_stats(cos_total_all)

        attempts_log = []
        chosen_id = None
        chosen_conf = None
        chosen_cos = None
        chosen_cos_raw = None
        chosen_cos_z_raw = None
        chosen_combined = None

        last_valid_num = None
        decision_source = None

        # =====================================================
        # OOD ゲート（LLM を呼ぶ前の早期 abstain）
        # =====================================================
        ood_abstain = False
        if cos_stats is not None:
            max_cos = cos_stats["max"]
            margin = cos_stats["margin_top1_top2"]

            # max_cos ベースの OOD 判定
            if (
                self.ood_max_cos_threshold > 0.0
                and max_cos < self.ood_max_cos_threshold
            ):
                ood_abstain = True
                decision_source = "abstain_ood_low_maxcos"
                rospy.logwarn(
                    "⚠️ OOD gate (low max cosine) -> abstain(-1): "
                    f"max_cos={max_cos:.3f}, margin_top1_top2={margin:.3f}"
                )
            # top1-top2 マージンベースの OOD 判定
            elif (
                self.ood_margin_threshold > 0.0
                and margin < self.ood_margin_threshold
            ):
                ood_abstain = True
                decision_source = "abstain_ood_small_margin"
                rospy.logwarn(
                    "⚠️ OOD gate (small top1-top2 margin) -> abstain(-1): "
                    f"max_cos={max_cos:.3f}, margin_top1_top2={margin:.3f}"
                )

        # OOD と判定された場合は LLM を一切呼ばずに -1 を返す
        if ood_abstain:
            chosen_id = int(self.abstain_node_id)

        # =====================================================
        # 試行ループ（1回目: full, 2回目以降: refine）
        # =====================================================
        if not ood_abstain:
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
                    cand = list(
                        {nid for nid in (sem_ids + prev_ids) if nid in self.id2idx}
                    )
                    if not cand:
                        cand = list(self.node_ids)
                    candidate_ids = cand
                    prompt = self.build_refine_prompt(query, candidate_ids)

                # LLM 呼び出し
                raw, conf = self.call_llm(prompt)
                num = self.extract_number(raw)

                # この試行で使う重み
                w_label, w_sem, w_desc = self.get_cos_weights_for_attempt(
                    attempt_idx, attempts_log, cos_components
                )

                # cosine_raw / cosine_norm の計算
                cos_raw = None
                cos_norm = None
                z_raw = None
                cos_label_raw = None
                cos_sem_raw = None
                cos_desc_raw = None

                if (
                    num is not None
                    and query_vec is not None
                    and cos_stats is not None
                    and cos_components is not None
                ):
                    idx = self.id2idx.get(int(num))
                    if idx is not None:
                        cos_label_all = cos_components.get("label")
                        cos_sem_all   = cos_components.get("sem")
                        cos_desc_all  = cos_components.get("desc")

                        if cos_label_all is not None:
                            cos_label_raw = float(cos_label_all[idx])
                        if cos_sem_all is not None:
                            cos_sem_raw = float(cos_sem_all[idx])
                        if cos_desc_all is not None:
                            cos_desc_raw = float(cos_desc_all[idx])

                        # 重み付き合成 cos_raw
                        cos_raw = (
                            w_label * (cos_label_raw if cos_label_raw is not None else 0.0)
                            + w_sem * (cos_sem_raw   if cos_sem_raw   is not None else 0.0)
                            + w_desc * (cos_desc_raw if cos_desc_raw is not None else 0.0)
                        )

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
                        else:  # "none"
                            cos_norm = cos_raw

                # 判定・combined に使う cosine
                cos_for_decision = cos_norm

                combined = None
                if (
                    self.decision_mode == "weighted"
                    and conf is not None
                    and cos_for_decision is not None
                ):
                    combined = float(
                        self.alpha * conf + (1.0 - self.alpha) * cos_for_decision
                    )
                elif conf is not None:
                    combined = conf

                # 判定
                decision, reject_reason, pass_conf, pass_cos, pass_combined = (
                    self.judge_attempt(
                        num, candidate_ids, conf, cos_for_decision, combined
                    )
                )

                # 「最後まで採択できなければ abstain(-1)」にする。
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
                        "cosine_label_raw": cos_label_raw,
                        "cosine_sem_raw": cos_sem_raw,
                        "cosine_desc_raw": cos_desc_raw,
                        "cosine_w_label": w_label,
                        "cosine_w_sem": w_sem,
                        "cosine_w_desc": w_desc,
                        "combined_score": combined,
                        "pass_conf": pass_conf,
                        "pass_cos": pass_cos,
                        "pass_combined": pass_combined,
                        "decision": "accept" if decision == "accept" else "retry",
                        "reject_reason": (
                            reject_reason if decision != "accept" else None
                        ),
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
        # 最終決定：
        #   - OOD ゲート -> すでに chosen_id = abstain_node_id
        #   - 再推論まで回しても accept なし -> abstain_no_accept
        # =====================================================
        if chosen_id is None:
            # ここに来るのは「OOD ゲートは発動せず、再推論も回し切ったが
            # 1度も accept できなかった」ケース
            if last_valid_num is not None:
                rospy.logwarn(
                    "⚠️ No attempt satisfied thresholds "
                    f"(last_valid_num={last_valid_num}). Abstain(-1)."
                )
            else:
                rospy.logwarn(
                    "⚠️ No valid response from LLM (no valid number). Abstain(-1)."
                )
            chosen_id = int(self.abstain_node_id)
            decision_source = "abstain_no_accept"

        # =====================================================
        # abstain ケースの top-level 信号を埋める
        # =====================================================
        if decision_source is not None and decision_source.startswith("abstain_"):
            if decision_source.startswith("abstain_ood"):
                # OOD 由来：conf / combined は None のまま、
                # cos は「合成 cos_total の max」を写す
                chosen_conf = None
                chosen_combined = None

                if cos_stats is not None:
                    max_cos = cos_stats["max"]
                    if self.cosine_norm_mode == "minmax":
                        denom = max(1e-6, cos_stats["max"] - cos_stats["min"])
                        chosen_cos_raw = max_cos
                        chosen_cos = float((max_cos - cos_stats["min"]) / denom)
                        chosen_cos_z_raw = None
                    elif self.cosine_norm_mode == "zscore":
                        denom = max(1e-6, cos_stats["std"])
                        z_top1 = (max_cos - cos_stats["mean"]) / denom
                        chosen_cos_raw = max_cos
                        chosen_cos_z_raw = float(z_top1)
                        chosen_cos = float(
                            0.5 * (1.0 + math.erf(z_top1 / math.sqrt(2.0)))
                        )
                    else:
                        chosen_cos_raw = max_cos
                        chosen_cos = max_cos
                        chosen_cos_z_raw = None
                else:
                    chosen_cos_raw = None
                    chosen_cos = None
                    chosen_cos_z_raw = None

            elif decision_source == "abstain_no_accept":
                # 「一番マシだった試行」のスコアを top-level にコピーする
                best_attempt = None
                best_score = None

                for a in attempts_log:
                    score = a.get("combined_score")
                    if score is None:
                        continue
                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_attempt = a

                if best_attempt is not None:
                    chosen_conf = best_attempt.get("confidence")
                    chosen_cos = best_attempt.get("cosine")
                    chosen_cos_raw = best_attempt.get("cosine_raw")
                    chosen_cos_z_raw = best_attempt.get("cosine_z_raw")
                    chosen_combined = best_attempt.get("combined_score")
                else:
                    chosen_conf = None
                    chosen_cos = None
                    chosen_cos_raw = None
                    chosen_cos_z_raw = None
                    chosen_combined = None

        accept_reason = self.classify_accept_reason(attempts_log, decision_source)

        # publish values
        label = self.id2label.get(chosen_id, "unknown")
        sem = self.id2sem.get(chosen_id, "")

        self.pub_node.publish(Int32(chosen_id))
        self.pub_conf.publish(
            Float32(chosen_conf if chosen_conf is not None else -1.0)
        )
        self.pub_cos.publish(
            Float32(chosen_cos if chosen_cos is not None else -1.0)
        )

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
            "cosine_z_raw": (
                chosen_cos_z_raw if self.cosine_norm_mode == "zscore" else None
            ),
            "combined_score": chosen_combined,
            "decision_mode": self.decision_mode,
            "accept_reason": accept_reason,
            "attempts": len(attempts_log),
            "decision_source": decision_source,
            "abstain_node_id": int(self.abstain_node_id),
            "is_abstain": int(chosen_id) == int(self.abstain_node_id),
            "thresholds": {
                "conf_threshold": self.conf_threshold,
                "cosine_threshold": self.cosine_threshold,
                "combined_threshold": self.combined_threshold,
                "alpha": self.alpha,
                "ood_max_cos_threshold": self.ood_max_cos_threshold,
                "ood_margin_threshold": self.ood_margin_threshold,
            },
            "valid_ids": list(self.node_ids),
            "attempt_logs": attempts_log,
        }

        # 埋め込み診断
        if self.use_embedding and query_vec is not None:
            # semantic cos に基づく top-k を出す
            top_ids = self.topk_semantic_ids(query_vec, self.topk_semantic)
            top_entries = []
            for nid in top_ids:
                c_sem = None
                if cos_components is not None and cos_components.get("sem") is not None:
                    idx = self.id2idx.get(int(nid))
                    if idx is not None:
                        c_sem = float(cos_components["sem"][idx])
                top_entries.append(
                    {
                        "node_id": int(nid),
                        "label": self.id2label.get(nid, ""),
                        "cosine_sem": c_sem,
                    }
                )
            meta["embed_diag"] = {
                "enabled": True,
                "model": self.embed_model_name,
                "text_mode": "split(label/semantic/description)",
                "topk_semantic": top_entries,
            }
        else:
            meta["embed_diag"] = {"enabled": False}

        # cosine の統計（合成 cos_total ベース）
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

#!/usr/bin/env python3
"""
LLM-based Reasoner ROS Node (confidence × cosine)
- YAML からノード情報を読み込む (id, label, semantic, description)
- 簡潔なプロンプトで LLM に「行くべきノードID」を選択させる
- token logprob の平均から LLM 自己信頼度を算出（0〜1に近似）
- sentence-transformers による埋め込みから
  「ユーザ指示」と「選択ノード意味情報」の cosine 類似度を算出
- confidence と cosine を組み合わせて採否判定
  - acceptance_mode = "and"      : 閾値を両方満たす場合のみ採用
  - acceptance_mode = "weighted" : 重み付きスコアが閾値以上なら採用
- 不採用時は再推論（max_retries 回まで），それでもダメなら
  last_num または fallback_node を採用
- 出力:
  - /llm_reasoner/chosen_node_id : Int32
  - /llm_reasoner/confidence     : Float32
  - /llm_reasoner/cosine         : Float32
  - /llm_reasoner/meta           : String (JSON)
"""

import os
import re
import json
import yaml
import rospy
import numpy as np
from std_msgs.msg import String, Int32, Float32

# ========= llama.cpp =========
try:
    from llama_cpp import Llama
except Exception as e:
    Llama = None
    _llama_import_error = e

# ========= sentence-transformers =========
_ST_OK = True
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    _ST_OK = False
    _st_import_error = e


# ========= utility =========
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("nan")
    a = _l2norm(a)
    b = _l2norm(b)
    s = float(np.dot(a, b))
    if s > 1.0:
        s = 1.0
    if s < -1.0:
        s = -1.0
    return s


class LLMReasonerNode:
    def __init__(self):
        rospy.init_node("llm_reasoner", anonymous=False)

        # ===== ROS params =====
        # パス類
        self.model_path = rospy.get_param("~model_path", "")
        self.map_yaml   = rospy.get_param("~map_yaml_path", "")

        # LLM 推論設定
        self.n_ctx       = int(rospy.get_param("~ctx_size", 2048))
        self.n_threads   = int(rospy.get_param("~threads", 4))
        self.temperature = float(rospy.get_param("~temperature", 0.1))
        self.top_k       = int(rospy.get_param("~top_k", 10))
        self.top_p       = float(rospy.get_param("~top_p", 0.0))
        self.max_tokens  = int(rospy.get_param("~max_tokens", 32))

        # 再試行・フォールバック
        self.max_retries   = int(rospy.get_param("~max_retries", 2))
        self.fallback_node = int(rospy.get_param("~fallback_node", -1))

        # 埋め込み有効化
        self.use_embeddings = bool(rospy.get_param("~use_embeddings", True))
        self.embed_model    = rospy.get_param("~embedding_model", "BAAI/bge-m3")
        self.embed_device   = rospy.get_param("~embedding_device", "cpu")
        # 各ノードをどの情報で埋め込むか
        self.embed_text_mode = rospy.get_param(
            "~embedding_text_mode",
            "label+semantic+description"
        )

        # 採否ロジック
        self.acceptance_mode  = rospy.get_param("~acceptance_mode", "weighted")  # "and" or "weighted"

        # AND 条件用
        self.conf_threshold   = float(rospy.get_param("~confidence_threshold", 0.5))
        self.cosine_threshold = float(rospy.get_param("~cosine_threshold", 0.5))

        # weighted 条件用
        self.alpha           = float(rospy.get_param("~alpha", 0.5))         # conf の重み
        self.score_threshold = float(rospy.get_param("~score_threshold", 0.6))

        # meta用：類似ノード表示数
        self.topk_for_meta   = int(rospy.get_param("~topk_for_meta", 3))

        # トピック名
        self.sub_query = rospy.get_param("~sub_query", "/llm_reasoner/query")
        self.pub_node  = rospy.get_param("~pub_node",  "/llm_reasoner/chosen_node_id")
        self.pub_conf  = rospy.get_param("~pub_conf",  "/llm_reasoner/confidence")
        self.pub_cos   = rospy.get_param("~pub_cos",   "/llm_reasoner/cosine")
        self.pub_meta  = rospy.get_param("~pub_meta",  "/llm_reasoner/meta")

        # ===== LLM ロード =====
        if Llama is None:
            rospy.logerr("llama_cpp import failed: %s", str(_llama_import_error))
            self.llm = None
        else:
            rospy.loginfo("🧩 Loading LLM model from: %s", self.model_path)
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                logits_all=True
            )

        # ===== マップ読み込み =====
        self.node_ids, self.node_labels, self.node_semantics, self.node_desc = self.load_map(self.map_yaml)
        if not self.node_ids:
            rospy.logwarn("No nodes loaded from map. Check YAML.")
        rospy.loginfo("📄 Loaded %d nodes from map.", len(self.node_ids))

        if self.fallback_node < 0 and self.node_ids:
            self.fallback_node = int(self.node_ids[0])

        # id→label / id→semantic
        self.id2label = {int(i): str(l) for i, l in zip(self.node_ids, self.node_labels)}
        self.id2sem   = {int(i): str(s) for i, s in zip(self.node_ids, self.node_semantics)}

        # ===== 埋め込みモデル初期化 =====
        self.embed_ok = False
        self.embed_model_obj = None
        self.node_embeds = None

        if self.use_embeddings and _ST_OK and self.node_ids:
            try:
                rospy.loginfo("🧠 Loading embedding model: %s (device=%s)", self.embed_model, self.embed_device)
                self.embed_model_obj = SentenceTransformer(self.embed_model, device=self.embed_device)
                node_texts = [self.node_text(i) for i in range(len(self.node_ids))]
                embs = self.embed_model_obj.encode(
                    node_texts,
                    batch_size=32,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False
                )
                # L2 正規化して保持
                self.node_embeds = np.vstack([_l2norm(e.astype(np.float32)) for e in embs])
                self.embed_ok = True
                rospy.loginfo("🧠 Node embeddings ready.")
            except Exception as e:
                rospy.logwarn("Embedding initialization failed. Run without cosine. (%s)", str(e))
        elif self.use_embeddings and not _ST_OK:
            rospy.logwarn("sentence-transformers not available: %s", str(_st_import_error))

        # ===== Publisher / Subscriber =====
        self.pub_node_id    = rospy.Publisher(self.pub_node, Int32,  queue_size=10)
        self.pub_confidence = rospy.Publisher(self.pub_conf, Float32, queue_size=10)
        self.pub_cosine     = rospy.Publisher(self.pub_cos,  Float32, queue_size=10)
        self.pub_meta_json  = rospy.Publisher(self.pub_meta, String,  queue_size=10)

        rospy.Subscriber(self.sub_query, String, self.callback, queue_size=10)

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("✅ LLM Reasoner Node ready.")
        rospy.spin()

    # ===== Map loader =====
    def load_map(self, path):
        if not path or not os.path.exists(path):
            rospy.logwarn("Map YAML not found: %s", path)
            return [], [], [], []
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        nodes = data.get("NODE", []) if isinstance(data, dict) else data
        ids, labels, sems, descs = [], [], [], []
        for n in nodes:
            nid = _safe_int(n.get("id"))
            if nid is None:
                continue
            ids.append(nid)
            labels.append(str(n.get("label", "")).strip())
            sem = n.get("semantic", [])
            if isinstance(sem, list):
                sem = ", ".join(sem)
            sems.append(str(sem))
            descs.append(str(n.get("description", "")).strip())
        return ids, labels, sems, descs

    # ===== Node text for embedding =====
    def node_text(self, idx):
        l = self.node_labels[idx]
        s = self.node_semantics[idx]
        d = self.node_desc[idx]
        mode = self.embed_text_mode
        if mode == "label":
            return l
        elif mode == "label+semantic":
            return f"{l}. {s}"
        elif mode == "label+description":
            return f"{l}. {d}"
        else:  # label+semantic+description
            return f"{l}. {s}. {d}"

    # ===== Embedding: query =====
    def embed_query(self, text):
        if not self.embed_ok:
            return None
        try:
            v = self.embed_model_obj.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )[0]
            return _l2norm(v.astype(np.float32))
        except Exception as e:
            rospy.logwarn("Query embedding failed: %s", str(e))
            return None

    # ===== Prompt =====
    def build_prompt(self, query):
        lines = ["Available Locations:"]
        for nid, lbl, sem in zip(self.node_ids, self.node_labels, self.node_semantics):
            lines.append(f"{nid}. {lbl} ({sem})")
        lines.append(f"User Task: {query}")
        lines.append("Answer with the NUMBER of the correct choice:")
        return "\n".join(lines)

    # ===== LLM call =====
    def call_llm(self, prompt):
        if self.llm is None:
            return "", None
        out = self.llm(
            prompt=prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["###", "\n", "Instruction:"],
            echo=False,
            logprobs=1
        )
        try:
            text = out["choices"][0]["text"].strip()
        except Exception:
            text = ""
        try:
            token_logprobs = out["choices"][0]["logprobs"]["token_logprobs"]
        except Exception:
            token_logprobs = None
        return text, token_logprobs

    # ===== 信頼度 =====
    def compute_confidence(self, token_logprobs):
        if not token_logprobs:
            return None
        avg_logp = float(np.mean(token_logprobs))
        conf = float(np.exp(avg_logp))
        # 0〜1にクリップ（実際はもっと小さいこともあるが，指標として扱いやすくするため）
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        return conf

    # ===== 番号抽出 =====
    def extract_node_id(self, text):
        m = re.search(r"\b-?\d+\b", text)
        if not m:
            return None
        nid = _safe_int(m.group(0))
        if nid in self.node_ids:
            return nid
        return None

    # ===== 採否判定 =====
    def accept(self, conf, cos):
        """
        conf: [0,1] or None
        cos : [-1,1] or NaN
        戻り値: (accepted:bool, combined_score:float or None, reason:str)
        """
        has_conf = (conf is not None)
        has_cos  = (cos == cos)  # NaN でないか

        if self.acceptance_mode == "and":
            conf_ok = (not has_conf) or (conf >= self.conf_threshold)
            cos_ok  = (not has_cos)  or (cos  >= self.cosine_threshold)
            accepted = conf_ok and cos_ok
            return accepted, None, "and_pass" if accepted else "and_block"

        # weighted
        conf_norm = conf if has_conf else 0.5
        if has_cos:
            cos_norm = (cos + 1.0) * 0.5
        else:
            cos_norm = 0.5
        score = float(self.alpha * conf_norm + (1.0 - self.alpha) * cos_norm)
        accepted = score >= self.score_threshold
        return accepted, score, "weighted_pass" if accepted else "weighted_block"

    # ===== コールバック =====
    def callback(self, msg):
        query = msg.data.strip()
        if not query:
            rospy.logwarn("⚠️ Received empty query.")
            return

        rospy.loginfo("🟢 Inference started. query='%s'", query)

        prompt = self.build_prompt(query)
        query_emb = self.embed_query(query) if self.embed_ok else None

        attempts = 0
        last_num = None
        last_conf = None
        last_cos = float("nan")
        last_score = None
        last_reason = None
        used_fallback = False
        decision_source = "model"

        chosen_id = None
        chosen_conf = None
        chosen_cos = float("nan")
        combined_score = None
        accept_reason = None

        while attempts <= self.max_retries and not rospy.is_shutdown():
            attempts += 1

            raw, token_logprobs = self.call_llm(prompt)
            nid = self.extract_node_id(raw)
            conf = self.compute_confidence(token_logprobs)

            if nid is not None:
                last_num = nid

            # cosine 計算（埋め込み有効＋nid有効）
            cos = float("nan")
            if (nid is not None) and (query_emb is not None) and (self.node_embeds is not None):
                try:
                    idx = self.node_ids.index(nid)
                    cos = _cosine(self.node_embeds[idx], query_emb)
                except Exception:
                    cos = float("nan")

            accepted, score, reason = self.accept(conf, cos)

            lbl = self.id2label.get(nid, "<?>") if nid is not None else "<?>"
            rospy.loginfo(
                "  Try %d → raw='%s' num=%s(label=%s) conf=%s cos=%s [%s]",
                attempts,
                raw,
                str(nid),
                lbl,
                f"{conf:.3f}" if conf is not None else "None",
                f"{cos:.3f}" if cos == cos else "NaN",
                reason,
            )

            last_conf = conf
            last_cos = cos
            last_score = score
            last_reason = reason

            if accepted and (nid is not None):
                chosen_id = nid
                chosen_conf = conf
                chosen_cos = cos
                combined_score = score
                accept_reason = reason
                decision_source = "model"
                break

            if attempts <= self.max_retries:
                # ここで「再推論用にプロンプトを少し変える」などの工夫を今後追加できる
                continue

        # ここまでで選べなかった場合の処理
        if chosen_id is None:
            if last_num is not None:
                # 最後に得られた有効な番号を採用（低信頼タグ付き）
                chosen_id = last_num
                chosen_conf = last_conf
                chosen_cos = last_cos
                combined_score = last_score
                accept_reason = last_reason
                decision_source = "low_conf_last_num"
            else:
                # 完全に失敗 → fallback
                chosen_id = self.fallback_node
                chosen_conf = None
                chosen_cos = float("nan")
                combined_score = None
                accept_reason = "fallback"
                decision_source = "fallback"
                used_fallback = True

        # ===== Publish =====
        self.pub_node_id.publish(Int32(int(chosen_id)))
        self.pub_confidence.publish(Float32(chosen_conf if chosen_conf is not None else -1.0))
        self.pub_cosine.publish(
            Float32(chosen_cos if chosen_cos == chosen_cos else float("nan"))
        )

        meta = {
            "node_id": int(chosen_id),
            "label": self.id2label.get(int(chosen_id), "<?>"),
            "label_semantics": self.id2sem.get(int(chosen_id), ""),
            "confidence": float(chosen_conf) if chosen_conf is not None else -1.0,
            "cosine": None if not (chosen_cos == chosen_cos) else float(chosen_cos),
            "combined_score": combined_score,
            "acceptance_mode": self.acceptance_mode,
            "accept_reason": accept_reason,
            "thresholds": {
                "conf_threshold": self.conf_threshold,
                "cosine_threshold": self.cosine_threshold,
                "alpha": self.alpha,
                "score_threshold": self.score_threshold,
            },
            "attempts": int(attempts),
            "last_num": int(last_num) if last_num is not None else None,
            "fallback_used": 1 if used_fallback else 0,
            "decision_source": decision_source,
            "fallback_node": int(self.fallback_node),
            "valid_ids": list(map(int, self.node_ids)),
            "query": query,
        }

        # 埋め込み診断（Top-K 類似ノード）
        if (
            query_emb is not None
            and self.node_embeds is not None
            and len(self.node_ids) > 0
        ):
            sims = np.dot(self.node_embeds, query_emb.astype(np.float32))
            k = min(self.topk_for_meta, len(self.node_ids))
            top_idx = np.argsort(-sims)[:k]
            meta["embed_diag"] = {
                "model": self.embed_model if self.embed_ok else None,
                "device": self.embed_device if self.embed_ok else None,
                "text_mode": self.embed_text_mode,
                "topk_similar": [
                    {
                        "node_id": int(self.node_ids[j]),
                        "label": self.node_labels[j],
                        "cosine": float(max(min(sims[j], 1.0), -1.0)),
                    }
                    for j in top_idx
                ],
            }
        else:
            meta["embed_diag"] = {"enabled": False}

        self.pub_meta_json.publish(String(json.dumps(meta, ensure_ascii=False)))

        # 人間向けログ
        final_label = self.id2label.get(int(chosen_id), "<?>")
        final_sem = self.id2sem.get(int(chosen_id), "")
        rospy.loginfo(
            "✅ Inference finished. DECISION: id=%d, label=%s, semantics=%s",
            int(chosen_id),
            final_label,
            final_sem,
        )
        rospy.loginfo("📦 META: %s", json.dumps(meta, ensure_ascii=False))

    # ===== cleanup =====
    def cleanup(self):
        rospy.loginfo("🧹 Shutting down LLM Reasoner Node...")
        try:
            if hasattr(self, "llm"):
                del self.llm
            rospy.loginfo("✅ Model resources released.")
        except Exception:
            rospy.logwarn("⚠️ Failed to clean up model resources.")


if __name__ == "__main__":
    try:
        LLMReasonerNode()
    except rospy.ROSInterruptException:
        pass

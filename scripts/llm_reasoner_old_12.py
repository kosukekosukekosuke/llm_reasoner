#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_reasoner.py

ROS1 node for high-level destination decision on a semantic node-edge map.

This file is written to be *spec-faithful* and *launch-faithful*:

- It reads ONLY the ROS private params that are set in llm_reasoner.launch
  (no alias / synonym parameter names).
- It avoids optional "dummy" backends; it expects the real dependencies:
    - llama-cpp-python (import: llama_cpp)
    - sentence-transformers (import: sentence_transformers)
- It supports map-info ablation variants: M_L / M_LS / M_LSD
- It implements three run modes:
    - reasoner     : Proposed method (OoM gate + retry/refine + combined accept)
    - baseline_ap  : LLM-only baseline that always predicts (no abstain by confidence)
    - baseline_r   : Baseline-R that may abstain by conf_threshold
- decision_mode is spec-fixed to "weighted" (enforced).

Published topics:
  /llm_reasoner/chosen_node_id  (std_msgs/Int32)
  /llm_reasoner/confidence      (std_msgs/Float32)
  /llm_reasoner/cosine          (std_msgs/Float32)
  /llm_reasoner/meta            (std_msgs/String, JSON)
Subscribed topic:
  /llm_reasoner/query           (std_msgs/String)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

import rospy
from std_msgs.msg import Float32, Int32, String
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# ---------------------------
# Small helpers
# ---------------------------

def now_wall() -> float:
    return time.time()


def sha256_of_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def safe_bool(x: Any, default: bool = False) -> bool:
    s = safe_str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Softmax with temperature tau (tau>0)."""
    tau = float(tau)
    if tau <= 0:
        tau = 1.0
    z = x / tau
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        return np.ones_like(e) / float(len(e))
    return e / s


def parse_map_info_variant(value: Any) -> str:
    """
    Parse map-information ablation variant.

    Allowed (case-insensitive):
      - "M_L", "L"
      - "M_LS", "LS"
      - "M_LSD", "LSD"
    """
    v = safe_str(value).strip().upper()
    if v.startswith("M_"):
        v = v[2:]
    v = v.replace("-", "").replace("_", "")
    if v in ("L", "LS", "LSD"):
        return v
    raise ValueError(f'Invalid map_info_variant="{value}". Allowed: M_L, M_LS, M_LSD.')

def canonical_map_info_variant(v: str) -> str:
    vv = parse_map_info_variant(v)
    return f"M_{vv}"


# ---------------------------
# Data structures
# ---------------------------

@dataclass(frozen=True)
class MapNode:
    node_id: int
    label: str
    semantics: List[str]
    description: str


# ---------------------------
# Backends (real deps only)
# ---------------------------

class LlamaCppBackend:
    """Minimal wrapper over llama-cpp-python that returns text + token_logprobs."""

    _logged_env: bool = False
    def __init__(self, model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int, n_batch: int) -> None:
        from llama_cpp import Llama  # noqa: F401

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f'llama_model_path does not exist: "{model_path}"')

        # NOTE: parameters mirror those exposed by llm_reasoner.launch.
        # We prefer logits_all=True so that `create_completion(..., logprobs=1)` is supported reliably.
        # Some llama-cpp-python versions may reject some kwargs; we fall back conservatively.
        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=int(n_ctx),
                n_threads=int(n_threads),
                n_gpu_layers=int(n_gpu_layers),
                n_batch=int(n_batch),
                logits_all=True,
                verbose=False,
            )
        except TypeError:
            try:
                # Some versions reject 'verbose' but accept logits_all.
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=int(n_ctx),
                    n_threads=int(n_threads),
                    n_gpu_layers=int(n_gpu_layers),
                    n_batch=int(n_batch),
                    logits_all=True,
                )
            except TypeError:
                # As a last resort, drop logits_all too.
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=int(n_ctx),
                    n_threads=int(n_threads),
                    n_gpu_layers=int(n_gpu_layers),
                    n_batch=int(n_batch),
                )

        # Log runtime environment once. Useful when ROS uses a different Python/site-packages than a shell test.
        if not LlamaCppBackend._logged_env:
            try:
                import sys
                import llama_cpp
                rospy.loginfo(
                    f"LLM env: python={sys.executable} llama_cpp_version={getattr(llama_cpp, '__version__', '?')} llama_cpp_file={getattr(llama_cpp, '__file__', '?')}"
                )
            except Exception:
                pass
            LlamaCppBackend._logged_env = True

    def generate(self, prompt: str, temperature: float, top_p: float, max_tokens: int) -> Dict[str, Any]:
        # We ask for logprobs so we can compute conf = exp(mean token logprob).
        # With echo=False, logprobs are for completion tokens only.
        resp = self._llm.create_completion(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            echo=False,
            logprobs=1,
        )
        choice = resp["choices"][0]
        text = safe_str(choice.get("text", "")).strip()

        token_logprobs: List[Optional[float]] = []
        tokens: List[str] = []
        lp = None
        try:
            lp = choice.get("logprobs", None)
        except Exception:
            lp = None
        if isinstance(lp, dict):
            tls = lp.get("token_logprobs", None)
            tks = lp.get("tokens", None)

            # Keep list length as-is (preserve None) for alignment with `tokens`.
            if isinstance(tls, list):
                token_logprobs = [float(x) if x is not None else None for x in tls]
            if isinstance(tks, list):
                tokens = [safe_str(x) for x in tks]

        return {"text": text, "token_logprobs": token_logprobs, "tokens": tokens, "raw": resp}


class SentenceTransformerEmbedder:
    """Minimal wrapper over sentence-transformers."""
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: F401

        if not model_name:
            raise ValueError("embedding_model_name is empty.")
        self._model = SentenceTransformer(model_name)
        self._batch_size = int(batch_size)

    def encode(self, texts: List[str]) -> np.ndarray:
        # normalize_embeddings=True makes cosine = dot product.
        emb = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)


# ---------------------------
# Main node
# ---------------------------

class LLMReasonerNode:
    def __init__(self) -> None:
        rospy.init_node("llm_reasoner", anonymous=False)

        # -------------------------
        # Params (ONLY those defined in llm_reasoner.launch)
        # -------------------------
        self.map_yaml_path = safe_str(rospy.get_param("~map_yaml_path", "")).strip()
        self.map_info_variant_raw = rospy.get_param("~map_info_variant", "M_LSD")
        self.map_info_variant = parse_map_info_variant(self.map_info_variant_raw)  # "L"/"LS"/"LSD"
        self.map_info_variant_canonical = canonical_map_info_variant(self.map_info_variant)

        self.use_map_semantics = ("S" in self.map_info_variant)
        self.use_map_description = ("D" in self.map_info_variant)

        # LLM runtime
        self.llama_model_path = safe_str(rospy.get_param("~llama_model_path", "")).strip()
        self.ctx_size = safe_int(rospy.get_param("~ctx_size", 4096), default=4096)
        self.threads = safe_int(rospy.get_param("~threads", 8), default=8)
        self.gpu_layers = safe_int(rospy.get_param("~gpu_layers", 0), default=0)
        self.batch_size = safe_int(rospy.get_param("~batch_size", 512), default=512)

        # Generation
        self.temperature = safe_float(rospy.get_param("~temperature", 0.0), default=0.0)
        self.top_p = safe_float(rospy.get_param("~top_p", 0.9), default=0.9)
        self.max_tokens = safe_int(rospy.get_param("~max_tokens", 128), default=128)

        self.timeout_sec = safe_float(rospy.get_param("~timeout_sec", 10.0), default=10.0)

        # Run mode (strict; accepts legacy alias "proposal" -> "reasoner")
        self.run_mode = safe_str(rospy.get_param("~run_mode", "reasoner")).strip().lower()
        if self.run_mode == "proposal":
            rospy.logwarn('run_mode="proposal" is a legacy alias; treating as "reasoner".')
            self.run_mode = "reasoner"
        if self.run_mode not in ("reasoner", "baseline_ap", "baseline_r"):
            raise ValueError('run_mode must be one of {"reasoner", "baseline_ap", "baseline_r"}')

        # Retry loop
        self.max_retries = safe_int(rospy.get_param("~max_retries", 1), default=1)
        self.topk_semantic = safe_int(rospy.get_param("~topk_semantic", 5), default=5)
        self.max_trials = max(1, 1 + max(0, self.max_retries))

        # OoM gate (raw cosine statistics, used only in reasoner mode)
        self.oom_max_cos_threshold = safe_float(rospy.get_param("~oom_max_cos_threshold", 0.5), default=0.5)
        self.oom_margin_threshold = safe_float(rospy.get_param("~oom_margin_threshold", 0.0), default=0.0)

        # Accept threshold (combined)
        self.combined_threshold = safe_float(rospy.get_param("~combined_threshold", 0.6), default=0.6)
        self.alpha = safe_float(rospy.get_param("~alpha", 0.5), default=0.5)
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError('alpha must be in [0, 1].')

        # Baseline-R threshold
        self.conf_threshold = safe_float(rospy.get_param("~conf_threshold", 0.0), default=0.0)

        # Cos weights
        self.w_label = safe_float(rospy.get_param("~cos_w_label", 1.0), default=1.0)
        self.w_semantics = safe_float(rospy.get_param("~cos_w_sem", 1.0), default=1.0)
        self.w_description = safe_float(rospy.get_param("~cos_w_desc", 1.0), default=1.0)
        for _n, _v in (('cos_w_label', self.w_label), ('cos_w_sem', self.w_semantics), ('cos_w_desc', self.w_description)):
            if _v < 0.0:
                raise ValueError(f'{_n} must be >= 0.')

        # Cos normalization mode
        self.cos_norm_mode = safe_str(rospy.get_param("~cosine_norm_mode", "none")).strip().lower()
        if self.cos_norm_mode not in ("none", "minmax", "zscore"):
            raise ValueError('cosine_norm_mode must be one of {"none","minmax","zscore"}')

        # cos_adapt
        self.cos_adapt_enabled = safe_bool(rospy.get_param("~cos_adapt", True), default=True)
        self.cos_adapt_strength = safe_float(rospy.get_param("~cos_adapt_strength", 1.0), default=1.0)
        self.cos_adapt_tau = safe_float(rospy.get_param("~cos_adapt_tau", 1.0), default=1.0)
        if self.cos_adapt_strength < 0.0:
            raise ValueError('cos_adapt_strength must be >= 0.')
        if self.cos_adapt_tau <= 0.0:
            raise ValueError('cos_adapt_tau must be > 0.')

        # decision_mode (spec-fixed)
        self.decision_mode = safe_str(rospy.get_param("~decision_mode", "weighted")).strip().lower()
        if self.decision_mode != "weighted":
            raise ValueError('decision_mode must be "weighted" (spec-fixed).')

        # abstain id
        self.abstain_node_id = safe_int(rospy.get_param("~abstain_node_id", -1), default=-1)

        # Spec: abstain is fixed to -1 (and -1 is NOT to be generated by the LLM).
        # Keep as a ROS param only to make the constraint explicit in configs.
        if self.abstain_node_id != -1:
            raise ValueError("abstain_node_id must be -1 per spec (abstain=-1).")

        # Embedding
        self.use_embedding = safe_bool(rospy.get_param("~use_embedding", True), default=True)
        self.embedding_model_name = safe_str(rospy.get_param("~embedding_model_name", "")).strip()

        # -------------------------
        # Map load
        # -------------------------
        if not self.map_yaml_path or not os.path.exists(self.map_yaml_path):
            raise FileNotFoundError(f'map_yaml_path does not exist: "{self.map_yaml_path}"')
        self.map_sha256 = sha256_of_file(self.map_yaml_path)

        self.nodes: List[MapNode] = self._load_map_nodes(self.map_yaml_path)
        if len(self.nodes) == 0:
            raise ValueError("No nodes loaded from map_yaml_path.")

        self.node_ids: List[int] = [n.node_id for n in self.nodes]
        self.valid_node_id_set = set(self.node_ids)

        # Deterministic ordering by node_id (important for prompt + reproducibility)
        self.nodes = sorted(self.nodes, key=lambda n: n.node_id)
        self.node_ids = [n.node_id for n in self.nodes]
        self.valid_node_id_set = set(self.node_ids)

        # Spec: node ids must be unique.
        seen: set = set()
        dups: set = set()
        for nid in self.node_ids:
            if nid in seen:
                dups.add(nid)
            seen.add(nid)
        if len(dups) > 0:
            raise ValueError(f"Duplicate node ids found in map_yaml_path: {sorted(list(dups))}")

        # Convenience mapping (deterministic) for candidate selection.
        self.node_by_id: Dict[int, MapNode] = {n.node_id: n for n in self.nodes}

        # -------------------------
        # Backends init
        # -------------------------
        self._llm_init_kwargs = {
            "model_path": self.llama_model_path,
            "n_ctx": self.ctx_size,
            "n_threads": self.threads,
            "n_gpu_layers": self.gpu_layers,
            "n_batch": self.batch_size,
        }
        self._llm_generation = 0
        self._llm_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

        self.llm = LlamaCppBackend(**self._llm_init_kwargs)

        self.embedder: Optional[SentenceTransformerEmbedder] = None
        self.emb_label: Optional[np.ndarray] = None
        self.emb_sem: Optional[np.ndarray] = None
        self.emb_desc: Optional[np.ndarray] = None

        if self.run_mode == "reasoner":
            if not self.use_embedding:
                # Proposed method requires cos for OoM gate and combined score.
                raise ValueError("run_mode=reasoner requires use_embedding=true.")
            self.embedder = SentenceTransformerEmbedder(self.embedding_model_name, batch_size=16)
            self._precompute_node_embeddings()

        # -------------------------
        # Code fingerprint
        # -------------------------
        try:
            with open(__file__, "r", encoding="utf-8") as f:
                self.code_sha256 = sha256_of_text(f.read())
        except Exception:
            self.code_sha256 = "unknown"

        # -------------------------
        # ROS IO
        # -------------------------
        self.topic_query = "/llm_reasoner/query"
        self.topic_id = "/llm_reasoner/chosen_node_id"
        self.topic_conf = "/llm_reasoner/confidence"
        self.topic_cos = "/llm_reasoner/cosine"
        self.topic_meta = "/llm_reasoner/meta"

        # Query sequence counter (for terminal progress logs only; does not affect reasoning / outputs)
        self._qseq = 0
        self._qseq_lock = threading.Lock()


        self.sub_query = rospy.Subscriber(self.topic_query, String, self._on_query, queue_size=10)

        self.pub_id = rospy.Publisher(self.topic_id, Int32, queue_size=10)
        self.pub_conf = rospy.Publisher(self.topic_conf, Float32, queue_size=10)
        self.pub_cos = rospy.Publisher(self.topic_cos, Float32, queue_size=10)
        self.pub_meta = rospy.Publisher(self.topic_meta, String, queue_size=10)

        # -------------------------
        # Terminal-friendly startup logs (readable / experiment-check friendly)
        # -------------------------
        def _shorten(s: str, max_len: int = 90) -> str:
            s = s or ""
            if len(s) <= max_len:
                return s
            keep = max_len - 3
            head = keep // 2
            tail = keep - head
            return s[:head] + "..." + s[-tail:]

        map_path_disp = _shorten(self.map_yaml_path)
        llm_path_disp = _shorten(self.llama_model_path)
        embed_tag = "on" if self.use_embedding else "off"
        embed_name = _shorten(self.embedding_model_name) if self.use_embedding else ""
        embed_str = f"{embed_tag}:{embed_name}" if embed_name else embed_tag

        sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        rospy.loginfo(sep)
        rospy.loginfo("🚀 llm_reasoner ready — experiment config")
        rospy.loginfo(sep)

        rospy.loginfo("🧭 RUN")
        rospy.loginfo(f"  mode          : {self.run_mode} / decision={self.decision_mode}")
        rospy.loginfo(f"  timeout       : {self.timeout_sec:.1f}s")

        rospy.loginfo("🗺 MAP")
        rospy.loginfo(f"  path          : {map_path_disp}")
        rospy.loginfo(f"  sha256        : {self.map_sha256[:12]}...  variant={self.map_info_variant_canonical}  nodes={len(self.nodes)}")

        rospy.loginfo("🧠 LLM")
        rospy.loginfo(f"  model_path    : {llm_path_disp}")
        rospy.loginfo(f"  runtime       : ctx={self.ctx_size}  threads={self.threads}  gpu_layers={self.gpu_layers}  batch={self.batch_size}")
        rospy.loginfo(f"  generation    : temp={self.temperature}  top_p={self.top_p}  max_tokens={self.max_tokens}")

        rospy.loginfo("🔎 EMBEDDING / COS")
        rospy.loginfo(f"  embedding     : {embed_str}")
        rospy.loginfo(f"  cosine_norm   : {self.cos_norm_mode}")
        rospy.loginfo(f"  weights(L/S/D): {self.w_label} / {self.w_semantics} / {self.w_description}")

        if self.run_mode == "reasoner":
            rospy.loginfo("⛔ GATE / ✅ ACCEPT (reasoner)")
            rospy.loginfo(f"  OoM gate      : maxcos<{self.oom_max_cos_threshold}  OR  margin<{self.oom_margin_threshold}")
            rospy.loginfo(f"  accept        : combined>={self.combined_threshold}  (alpha={self.alpha})")
            rospy.loginfo("🔁 RETRY / REFINE")
            rospy.loginfo(f"  max_retries   : {self.max_retries}  (trials={self.max_trials})")
            rospy.loginfo(f"  topk_semantic : {self.topk_semantic}")
            rospy.loginfo(f"  cos_adapt     : {self.cos_adapt_enabled}  (strength={self.cos_adapt_strength}, tau={self.cos_adapt_tau})")
        elif self.run_mode == "baseline_r":
            rospy.loginfo("✅ BASELINE-R")
            rospy.loginfo(f"  conf_threshold: {self.conf_threshold}")
        else:
            rospy.loginfo("✅ BASELINE-AP")
            rospy.loginfo("  always-predict: True (no abstain)")

        rospy.loginfo("📡 ROS TOPICS")
        rospy.loginfo(f"  sub           : {self.topic_query}")
        rospy.loginfo(f"  pub           : id={self.topic_id}  conf={self.topic_conf}  cos={self.topic_cos}  meta={self.topic_meta}")
        rospy.loginfo(sep)


    # -------------------------
    # Map parsing / prompt text
    # -------------------------

    def _load_map_nodes(self, yaml_path: str) -> List[MapNode]:
        """
        Strict map schema (spec):

          top-level: {"nodes": [ ... ]}

          each node:
            - id          : int
            - label       : str
            - semantics   : list[str]
            - description : str
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "nodes" not in data:
            raise ValueError("Map YAML must be a dict with top-level key 'nodes'.")

        nodes_raw = data["nodes"]
        if not isinstance(nodes_raw, list):
            raise ValueError("Map YAML 'nodes' must be a list.")

        out: List[MapNode] = []
        for idx, obj in enumerate(nodes_raw):
            if not isinstance(obj, dict):
                raise ValueError(f"Map YAML nodes[{idx}] must be a dict.")

            missing = [k for k in ("id", "label", "semantics", "description") if k not in obj]
            if missing:
                raise ValueError(f"Map YAML nodes[{idx}] missing keys: {missing}")

            nid = obj["id"]
            if not isinstance(nid, int):
                # Accept numeric strings as a last resort, but still fail clearly if impossible.
                try:
                    nid = int(nid)
                except Exception as e:
                    raise ValueError(f"Map YAML nodes[{idx}].id must be int (got {type(obj['id'])})") from e

            label = safe_str(obj["label"])
            desc = safe_str(obj["description"])

            semantics = obj["semantics"]
            if semantics is None:
                semantics_list: List[str] = []
            elif isinstance(semantics, list):
                semantics_list = [safe_str(x) for x in semantics]
            else:
                raise ValueError(f"Map YAML nodes[{idx}].semantics must be list[str].")

            out.append(MapNode(node_id=int(nid), label=label, semantics=semantics_list, description=desc))

        return out

    def _node_texts_for_embedding(self) -> Tuple[List[str], List[str], List[str]]:
        """Return channel texts aligned with self.nodes."""
        label_texts: List[str] = []
        sem_texts: List[str] = []
        desc_texts: List[str] = []

        for n in self.nodes:
            label_texts.append(safe_str(n.label).strip())

            if self.use_map_semantics:
                sem_texts.append("\n".join([safe_str(x).strip() for x in n.semantics if safe_str(x).strip() != ""]))
            else:
                sem_texts.append("")

            if self.use_map_description:
                desc_texts.append(safe_str(n.description).strip())
            else:
                desc_texts.append("")

        return label_texts, sem_texts, desc_texts

    @staticmethod
    def _one_line_text(x: Any) -> str:
        """Return a single-line, whitespace-normalized string.

        Used for LLM prompts and lightweight logs to avoid accidental line breaks
        or extra whitespace that can increase parse errors.
        """
        return re.sub(r"\s+", " ", safe_str(x)).strip()

    def _node_lines_for_prompt(self, nodes: List[MapNode]) -> List[str]:
        """Format nodes for LLM prompt (masked by map_info_variant)."""
        lines: List[str] = []
        for n in nodes:
            parts = [f"id={n.node_id}", f"label={self._one_line_text(n.label)}"]

            if self.use_map_semantics:
                sem_items = [self._one_line_text(x) for x in n.semantics]
                sem_items = [x for x in sem_items if x != ""]
                parts.append(f"semantics={'; '.join(sem_items)}")

            if self.use_map_description:
                parts.append(f"description={self._one_line_text(n.description)}")

            lines.append(" | ".join(parts))
        return lines


    def _build_full_prompt(self, query: str) -> str:
        q = self._one_line_text(query)

        lines: List[str] = []
        lines.append("You are a destination selector over a set of map nodes.")
        lines.append("Choose exactly one destination node id from the candidates.")
        lines.append("Output ONLY the integer id (e.g., 3). Do NOT output -1. Do NOT output any other text or numbers.")
        lines.append("")
        lines.append(f"User query: {q}")
        lines.append("")
        lines.append("Candidates:")
        lines.extend(self._node_lines_for_prompt(self.nodes))
        lines.append("")
        lines.append("Answer:")
        return "\n".join(lines)

    def _build_refine_prompt(self, query: str, prev_answer: str, candidates: List[MapNode]) -> str:
        q = self._one_line_text(query)
        pa = self._one_line_text(prev_answer)

        lines: List[str] = []
        lines.append("You are a destination selector over a set of map nodes.")
        lines.append("Choose exactly one destination node id from the candidates.")
        lines.append("Output ONLY the integer id (e.g., 3). Do NOT output -1. Do NOT output any other text or numbers.")
        lines.append("")
        lines.append(f"User query: {q}")
        lines.append("")
        lines.append(f"Previous answer (may be wrong): {pa}")
        lines.append("")
        lines.append("Candidates (filtered):")
        lines.extend(self._node_lines_for_prompt(candidates))
        lines.append("")
        lines.append("Answer:")
        return "\n".join(lines)


    # -------------------------
    # Embeddings / cosine
    # -------------------------

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        # Embeddings are normalized, so dot is cosine (keep safe division anyway).
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def _precompute_node_embeddings(self) -> None:
        if self.embedder is None:
            raise RuntimeError("embedder is None but _precompute_node_embeddings was called.")

        label_texts, sem_texts, desc_texts = self._node_texts_for_embedding()
        self.emb_label = self.embedder.encode(label_texts)

        # Only compute the channels that are enabled by map_info_variant.
        self.emb_sem = self.embedder.encode(sem_texts) if self.use_map_semantics else None
        self.emb_desc = self.embedder.encode(desc_texts) if self.use_map_description else None

    def _encode_query(self, query: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.embedder is None:
            raise RuntimeError("Embeddings are disabled (embedder is None).")
        q = query.strip()
        qv = self.embedder.encode([q])[0]
        # Query embedding is shared across channels (spec uses same query text against L/S/D node texts).
        return qv, qv, qv

    def _compute_cos_channels(self, query: str) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """Return per-channel cosine dicts for L/S/D (missing channels are 0.0)."""
        qL, qS, qD = self._encode_query(query)

        if self.emb_label is None:
            raise RuntimeError("Node embeddings not prepared (emb_label is None).")

        cosL: Dict[int, float] = {}
        cosS: Dict[int, float] = {}
        cosD: Dict[int, float] = {}

        for idx, node in enumerate(self.nodes):
            cosL[node.node_id] = self._cos(qL, self.emb_label[idx])

            if self.use_map_semantics and self.emb_sem is not None:
                cosS[node.node_id] = self._cos(qS, self.emb_sem[idx])
            else:
                cosS[node.node_id] = 0.0

            if self.use_map_description and self.emb_desc is not None:
                cosD[node.node_id] = self._cos(qD, self.emb_desc[idx])
            else:
                cosD[node.node_id] = 0.0

        return cosL, cosS, cosD

    def _compute_cos_raw_all(self, query: str, w_label: float, w_sem: float, w_desc: float) -> Dict[int, float]:
        cosL, cosS, cosD = self._compute_cos_channels(query)
        scores: Dict[int, float] = {}

        # Mask out unused channels regardless of weights (ablation spec).
        wS = w_sem if self.use_map_semantics else 0.0
        wD = w_desc if self.use_map_description else 0.0

        for node in self.nodes:
            nid = node.node_id
            scores[nid] = float(w_label * cosL[nid] + wS * cosS[nid] + wD * cosD[nid])

        return scores

    def _normalize_cos(self, cos_raw: Dict[int, float]) -> Dict[int, float]:
        if self.cos_norm_mode == "none":
            return dict(cos_raw)

        vals = np.array(list(cos_raw.values()), dtype=np.float32)
        if vals.size == 0:
            return dict(cos_raw)

        if self.cos_norm_mode == "minmax":
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            rng = vmax - vmin
            if rng < 1e-12:
                return {k: 0.5 for k in cos_raw.keys()}
            return {k: float((v - vmin) / rng) for k, v in cos_raw.items()}

        if self.cos_norm_mode == "zscore":
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-12:
                return {k: 0.0 for k in cos_raw.keys()}
            return {k: float((v - mu) / sd) for k, v in cos_raw.items()}

        # Should not happen due to validation.
        return dict(cos_raw)

    def _topk_from_scores(self, scores: Dict[int, float], k: int) -> List[int]:
        items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [nid for nid, _ in items[: max(0, int(k))]]

    # -------------------------
    # cos_adapt (spec)
    # -------------------------

    def _adapt_weights_from_zscore(
        self,
        cos_label: Dict[int, float],
        cos_sem: Dict[int, float],
        cos_desc: Dict[int, float],
        chosen_id: int,
        base_w_label: float,
        base_w_sem: float,
        base_w_desc: float,
        strength: float,
        tau: float,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """cos_adapt: update weights for next trial based on z-score of chosen node per channel."""
        meta: Dict[str, Any] = {}

        def zscore(vals: List[float], x: float) -> float:
            a = np.array(vals, dtype=np.float32)
            mu = float(np.mean(a))
            sd = float(np.std(a))
            if sd < 1e-12:
                return 0.0
            return float((x - mu) / sd)

        zL = zscore(list(cos_label.values()), cos_label.get(chosen_id, 0.0))
        zS = zscore(list(cos_sem.values()), cos_sem.get(chosen_id, 0.0)) if self.use_map_semantics else -1e9
        zD = zscore(list(cos_desc.values()), cos_desc.get(chosen_id, 0.0)) if self.use_map_description else -1e9

        z = np.array([zL, zS, zD], dtype=np.float32)
        w = softmax(z, tau=float(tau))

        w_new = np.array([base_w_label, base_w_sem, base_w_desc], dtype=np.float32) * (1.0 + float(strength) * w)
        wL, wS, wD = float(w_new[0]), float(w_new[1]), float(w_new[2])

        meta.update(
            {
                "z_label": zL,
                "z_sem": zS,
                "z_desc": zD,
                "softmax": [float(w[0]), float(w[1]), float(w[2])],
                "base_weights": [base_w_label, base_w_sem, base_w_desc],
                "new_weights": [wL, wS, wD],
                "strength": float(strength),
                "tau": float(tau),
            }
        )
        return wL, wS, wD, meta

    # -------------------------
    # LLM output normalization (spec)
    # -------------------------

    def normalize_llm_output(
        self, text: str, meta: Dict[str, Any]
    ) -> Tuple[str, int]:
        """Normalize raw LLM output text to a single valid node_id in I, if possible.

        Spec mapping:
          - Extract all integers from the raw answer.
          - J_I := { j in J | j in I }  (set semantics; duplicates are ignored).
          - If |J|=0 or |J_I|>=2 -> parse_error
          - If |J_I|=0 -> out_of_set
          - If |J_I|=1 -> ok and return that id
        """
        hits = re.findall(r"-?\d+", safe_str(text).strip())
        ints: List[int] = []
        for h in hits:
            try:
                ints.append(int(h))
            except Exception:
                # Should not happen due to regex, but keep it defensive.
                continue

        meta["extracted_ints"] = hits
        meta["extracted_ints_int"] = ints

        if len(ints) == 0:
            meta["normalize_error"] = "no_int_found"
            return "parse_error", self.abstain_node_id

        # NOTE: J_I is a SET per spec (duplicates do not count as multiple candidates).
        valid_set = {j for j in ints if j in self.valid_node_id_set}
        meta["valid_ints"] = [j for j in ints if j in self.valid_node_id_set]
        meta["valid_ints_unique"] = sorted(list(valid_set))

        if len(valid_set) == 0:
            meta["normalize_error"] = "no_valid_id"
            return "out_of_set", self.abstain_node_id

        if len(valid_set) >= 2:
            meta["normalize_error"] = "multiple_valid_ids"
            return "parse_error", self.abstain_node_id

        chosen = sorted(list(valid_set))[0]
        return "ok", chosen

    def _compute_conf_from_llm_logprobs(
        self,
        token_logprobs: List[Optional[float]],
        tokens: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute self-confidence `conf` from LLM token logprobs.

        Spec: conf := exp(mean(token_logprobs)) over the *answer tokens*.
        We implement a deterministic fixed policy:

        - Use completion token_logprobs returned by llama-cpp (not prompt tokens).
        - Drop one trailing EOS/special token *if* its token string looks special.
        - Ignore None entries.
        """
        detail: Dict[str, Any] = {
            "definition": "exp(mean(token_logprobs))",
            "token_source": "llama_cpp_completion",
            "eos_policy": "drop_trailing_if_special",
            "dropped_last_token": False,
            "n_tokens_total": 0,
            "n_tokens_used": 0,
        }

        if not isinstance(token_logprobs, list) or len(token_logprobs) == 0:
            return 0.0, detail

        tlogp = list(token_logprobs)
        detail["n_tokens_total"] = int(len(tlogp))

        if isinstance(tokens, list) and len(tokens) == len(tlogp) and len(tokens) >= 1:
            last_tok = safe_str(tokens[-1]).strip()
            looks_special = (
                last_tok in ("</s>", "<|endoftext|>", "<|eot_id|>", "<|end|>")
                or (last_tok.startswith("<|") and last_tok.endswith("|>"))
            )
            if looks_special:
                tlogp = tlogp[:-1]
                detail["dropped_last_token"] = True

        vals = [float(x) for x in tlogp if x is not None]
        detail["n_tokens_used"] = int(len(vals))
        if len(vals) == 0:
            return 0.0, detail

        avg_lp = float(np.mean(np.array(vals, dtype=np.float32)))
        return float(np.exp(avg_lp)), detail

    def _recreate_llm_backend(self, reason: str) -> None:
        """Best-effort recovery after an LLM timeout/exception.

        Threads cannot be force-killed. If an LLM call exceeds `timeout_sec`, the worker thread
        may keep running. To avoid blocking subsequent queries behind that stuck call, we swap
        to a fresh LLM backend *and* a fresh executor.

        This does not change any algorithmic behavior; it only improves operational robustness.
        """
        try:
            new_llm = LlamaCppBackend(**self._llm_init_kwargs)
        except Exception as e:
            rospy.logwarn(f"⚠️  LLM backend recreate failed (reason={reason}): {e}")
            return

        with self._llm_lock:
            self._llm_generation += 1
            old_ex = self._executor
            self._executor = ThreadPoolExecutor(max_workers=1)
            self.llm = new_llm

            try:
                old_ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                old_ex.shutdown(wait=False)

        rospy.logwarn(f"🔄 LLM backend recreated (reason={reason}, generation={self._llm_generation})")


    def _call_llm_with_timeout(self, prompt: str) -> Tuple[Optional[Dict[str, Any]], str]:
        # Fast path: no timeout.
        if self.timeout_sec is None or float(self.timeout_sec) <= 0:
            try:
                out = self.llm.generate(
                    prompt,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return out, "ok"
            except Exception as e:
                rospy.logwarn(f"LLM exception (no-timeout path): {e!r}")
                return None, "exception"

        # Snapshot the current backend/executor; on timeout we may swap them.
        backend = self.llm
        ex = self._executor

        def _run() -> Dict[str, Any]:
            return backend.generate(
                prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

        try:
            fut = ex.submit(_run)
            out = fut.result(timeout=float(self.timeout_sec))
            return out, "ok"
        except FuturesTimeoutError:
            self._recreate_llm_backend("timeout")
            return None, "timeout"
        except Exception as e:
            rospy.logwarn(f"LLM exception (timeout path): {e!r}")
            self._recreate_llm_backend("exception")
            return None, "exception"

    def _meta_base(self, t_start: float) -> Dict[str, Any]:
        return {
            "status": "ok",
            "n_llm_calls": 0,
            "run_mode": self.run_mode,
            "decision_mode": self.decision_mode,
            "code_sha256": self.code_sha256,
            "map_sha256": self.map_sha256,
            "map_info_variant": self.map_info_variant_canonical,
            "use_map_semantics": self.use_map_semantics,
            "use_map_description": self.use_map_description,
            "cosine_norm_mode": self.cos_norm_mode,
            "alpha": self.alpha,
            "combined_threshold": self.combined_threshold,
            "conf_threshold": self.conf_threshold,
            "oom_max_cos_threshold": self.oom_max_cos_threshold,
            "oom_margin_threshold": self.oom_margin_threshold,
            "topk_semantic": self.topk_semantic,
            "max_retries": self.max_retries,
            "max_trials": self.max_trials,
            "timeout_sec": self.timeout_sec,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "cos_weights": [self.w_label, self.w_semantics, self.w_description],
            "cos_adapt": {
                "enabled": self.cos_adapt_enabled,
                "strength": self.cos_adapt_strength,
                "tau": self.cos_adapt_tau,
            },
            "use_embedding": self.use_embedding,
            "embedding_model_name": self.embedding_model_name,
            "abstain_node_id": self.abstain_node_id,
            "timestamp_wall": t_start,
        }


    # -------------------------
    # Terminal progress logging (does NOT affect reasoning)
    # -------------------------
    def _next_qseq(self) -> int:
        with self._qseq_lock:
            self._qseq += 1
            return int(self._qseq)

    @staticmethod
    def _snip_text(s: str, max_len: int = 70) -> str:
        s = (s or "").replace("\n", " ").replace("\r", " ").strip()
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    def _snip_node(self, nid: int, max_len: int = 22) -> str:
        try:
            node = self.node_by_id.get(int(nid))
        except Exception:
            node = None
        if node is None:
            return f"{nid}"
        label = safe_str(getattr(node, "label", "")).replace("\n", " ").strip()
        if len(label) > max_len:
            label = label[: max_len - 1] + "…"
        return f'{nid} "{label}"'

    def _plog(self, qseq: int, msg: str, icon: str = "•") -> None:
        # Keep each line grep-friendly: always includes Qxxxxx
        try:
            prefix = f"Q{int(qseq):05d}"
        except Exception:
            prefix = "Q?????"
        rospy.loginfo(f"{prefix} {icon} {msg}")


    def _process_query(self, query: str, qseq: Optional[int] = None) -> Tuple[int, float, float, str]:
        """
        Returns:
          (chosen_node_id, conf, cos, meta_json)
        """
        t_start = now_wall()
        q = self._one_line_text(query)

        # Progress: query start
        if qseq is not None:
            self._plog(qseq, f'START | q="{self._snip_text(q, 80)}"', icon='🔎')

        # Basic validation
        if q == "":
            meta = self._meta_base(t_start)
            meta["status"] = "parse_error"
            meta["accept_reason"] = "execution_failure"
            meta["detail"] = "empty_query"
            return int(self.abstain_node_id), 0.0, 0.0, json.dumps(meta, ensure_ascii=False)

        # Build full prompt once (used by all modes here)
        full_prompt = self._build_full_prompt(q)

        # -------------------------
        # baseline_ap : LLM-only, always accept if parse ok
        # baseline_r  : LLM-only, accept iff conf >= conf_threshold (else abstain)
        # -------------------------
        if self.run_mode in ("baseline_ap", "baseline_r"):
            trial = 1  # for progress logging only
            if qseq is not None:
                self._plog(qseq, f"BASELINE ({self.run_mode}) | LLM start", icon="🧠")
            t_llm0 = now_wall()
            out, gen_status = self._call_llm_with_timeout(full_prompt)
            t_llm1 = now_wall()
            if qseq is not None:
                self._plog(qseq, f"BASELINE LLM done | status={gen_status} | dt={(t_llm1-t_llm0)*1000.0:.0f}ms", icon="🧠")
            if out is None:
                if qseq is not None:
                    self._plog(qseq, f'BASELINE FAILED | LLM returned None (status={gen_status})', icon='💥')
                meta = self._meta_base(t_start)
                meta["status"] = gen_status
                meta["accept_reason"] = "execution_failure"
                meta["trial"] = 1
                meta["prompt_kind"] = "full"
                meta["n_llm_calls"] = 1
                return int(self.abstain_node_id), 0.0, 0.0, json.dumps(meta, ensure_ascii=False)

            text = safe_str(out.get("text", "")).strip()
            token_logprobs = out.get("token_logprobs", [])
            tokens = out.get("tokens", [])
            conf, conf_detail = self._compute_conf_from_llm_logprobs(token_logprobs, tokens)

            norm_meta: Dict[str, Any] = {}
            status, nid = self.normalize_llm_output(text, norm_meta)
            if qseq is not None:
                self._plog(qseq, f'BASELINE PARSE | status={status} nid={nid} | conf={conf:.3f}', icon='🧩')
            norm_meta["text"] = text
            norm_meta["conf_detail"] = conf_detail
            meta = self._meta_base(t_start)
            meta.update(
                {
                    "status": status,
                    "trial": 1,
                    "prompt_kind": "full",
                    "llm_raw": norm_meta,
                    "conf": conf,
                    "n_llm_calls": 1,
                }
            )

            if status != "ok":
                if qseq is not None:
                    self._plog(qseq, f'BASELINE PARSE FAIL | status={status}', icon='💥')
                meta["accept_reason"] = "execution_failure"
                return int(self.abstain_node_id), conf, 0.0, json.dumps(meta, ensure_ascii=False)

            chosen_id = int(nid)
            meta["chosen_id"] = chosen_id

            if self.run_mode == "baseline_ap":
                meta["accept"] = True
                meta["accept_reason"] = "baseline_ap_accept"
                return chosen_id, conf, 0.0, json.dumps(meta, ensure_ascii=False)

            # baseline_r
            accept = conf >= float(self.conf_threshold)
            meta["accept"] = bool(accept)
            if accept:
                meta["accept_reason"] = "baseline_r_accept"
                return chosen_id, conf, 0.0, json.dumps(meta, ensure_ascii=False)

            meta["accept_reason"] = "baseline_r_reject"
            return int(self.abstain_node_id), conf, 0.0, json.dumps(meta, ensure_ascii=False)

        # -------------------------
        # reasoner : Proposed method
        # -------------------------
        # Compute initial cos stats (OoM gate)
        cosL, cosS, cosD = self._compute_cos_channels(q)
        cos_raw_all = self._compute_cos_raw_all(q, self.w_label, self.w_semantics, self.w_description)
        cos_norm_all = self._normalize_cos(cos_raw_all)

        cos_raw_sorted = sorted(cos_raw_all.items(), key=lambda kv: (-kv[1], kv[0]))
        cos_raw_max = float(cos_raw_sorted[0][1]) if len(cos_raw_sorted) >= 1 else -1.0
        margin_raw = float(cos_raw_sorted[0][1] - cos_raw_sorted[1][1]) if len(cos_raw_sorted) >= 2 else float("inf")

        if qseq is not None:
            # Show top-2 by raw cos (pre-normalization) to sanity-check progress
            if len(cos_raw_sorted) >= 1:
                t1_id, t1_raw = cos_raw_sorted[0]
                t1_norm = float(cos_norm_all.get(int(t1_id), 0.0))
                if len(cos_raw_sorted) >= 2:
                    t2_id, t2_raw = cos_raw_sorted[1]
                    t2_norm = float(cos_norm_all.get(int(t2_id), 0.0))
                    self._plog(
                        qseq,
                        f'COS | top1={self._snip_node(int(t1_id))} raw={float(t1_raw):.3f} norm={t1_norm:.3f} '
                        f'| top2={self._snip_node(int(t2_id))} raw={float(t2_raw):.3f} norm={t2_norm:.3f} '
                        f'| margin_raw={margin_raw:.3f}',
                        icon='🧾',
                    )
                else:
                    self._plog(
                        qseq,
                        f'COS | top1={self._snip_node(int(t1_id))} raw={float(t1_raw):.3f} norm={t1_norm:.3f}',
                        icon='🧾',
                    )

        if (cos_raw_max < float(self.oom_max_cos_threshold)) or (margin_raw < float(self.oom_margin_threshold)):
            if qseq is not None:
                why = []
                if cos_raw_max < float(self.oom_max_cos_threshold):
                    why.append(f"maxcos {cos_raw_max:.3f} < th {float(self.oom_max_cos_threshold):.3f}")
                if margin_raw < float(self.oom_margin_threshold):
                    why.append(f"margin {margin_raw:.3f} < th {float(self.oom_margin_threshold):.3f}")
                self._plog(qseq, "OoM GATE HIT | " + " OR ".join(why) + f" => abstain={self.abstain_node_id}", icon="🚫")
            meta = self._meta_base(t_start)
            meta.update(
                {
                    "accept_reason": "oom_gate",
                    "trial": 1,              # Spec: gate is evaluated before first LLM call
                    "prompt_kind": "none",
                    "n_llm_calls": 0,
                    "cos_weights_used": [self.w_label, self.w_semantics, self.w_description],
                    "cos_raw_max": cos_raw_max,
                    "margin_raw": margin_raw,
                }
            )
            return int(self.abstain_node_id), 0.0, 0.0, json.dumps(meta, ensure_ascii=False)

        if qseq is not None:
            self._plog(qseq, f'OoM gate PASS (maxcos={cos_raw_max:.3f} th={float(self.oom_max_cos_threshold):.3f}, margin={margin_raw:.3f} th={float(self.oom_margin_threshold):.3f})', icon='✅')

        prev_text = ""
        wL, wS, wD = self.w_label, self.w_semantics, self.w_description
        n_llm_calls = 0
        last_adapt_meta: Optional[Dict[str, Any]] = None

        for trial in range(1, self.max_trials + 1):
            if trial == 1:
                prompt = full_prompt
                prompt_kind = "full"
                cand_ids = None
                candidates = self.nodes
            else:
                # Spec: candidate set C_k is defined by descending cos(q,v) (ties by node_id),
                # and is used for refine prompting on trials t>=2. (Note: minmax/zscore are monotonic, so using cos_norm would give the same order.)
                top_ids = self._topk_from_scores(cos_raw_all, self.topk_semantic)  # rank by cos(q,v)=cos_raw; tie-break by node_id
                cand_ids = list(top_ids)
                candidates = [self.node_by_id[nid] for nid in top_ids]
                prompt = self._build_refine_prompt(q, prev_text, candidates)
                prompt_kind = "refine"

            if qseq is not None:
                if prompt_kind == "full":
                    cand_info = f"cand=ALL({len(self.nodes)})"
                else:
                    # preview top candidates
                    preview = ", ".join([self._snip_node(int(x), max_len=16) for x in (cand_ids or [])[:3]])
                    more = "" if (cand_ids is None or len(cand_ids) <= 3) else f", +{len(cand_ids)-3} more"
                    cand_info = f"cand=TOP{len(cand_ids) if cand_ids is not None else 0} [{preview}{more}]"
                self._plog(
                    qseq,
                    f"trial {trial}/{self.max_trials} {prompt_kind.upper()} | {cand_info} | w=[{float(wL):.2f},{float(wS):.2f},{float(wD):.2f}] | LLM start",
                    icon="🧠",
                )
            t_llm0 = now_wall()
            n_llm_calls += 1
            out, gen_status = self._call_llm_with_timeout(prompt)
            t_llm1 = now_wall()
            if qseq is not None:
                self._plog(qseq, f"trial {trial}/{self.max_trials} LLM done | status={gen_status} | dt={(t_llm1-t_llm0)*1000.0:.0f}ms", icon="🧠")
            if out is None:
                meta = self._meta_base(t_start)
                meta["status"] = gen_status
                meta["accept_reason"] = "execution_failure"
                meta["trial"] = trial
                meta["prompt_kind"] = prompt_kind
                meta["candidate_ids"] = cand_ids
                meta["n_llm_calls"] = int(n_llm_calls)
                meta["cos_weights_used"] = [float(wL), float(wS), float(wD)]
                if last_adapt_meta is not None:
                    meta["cos_adapt_last"] = last_adapt_meta
                return int(self.abstain_node_id), 0.0, 0.0, json.dumps(meta, ensure_ascii=False)

            text = safe_str(out.get("text", "")).strip()
            prev_text = text

            token_logprobs = out.get("token_logprobs", [])
            tokens = out.get("tokens", [])
            conf, conf_detail = self._compute_conf_from_llm_logprobs(token_logprobs, tokens)

            norm_meta: Dict[str, Any] = {}
            status, nid = self.normalize_llm_output(text, norm_meta)
            if qseq is not None:
                self._plog(qseq, f'trial {trial}/{self.max_trials} PARSE | status={status} nid={nid} | conf={conf:.3f}', icon='🧩')
            norm_meta["text"] = text
            if status != "ok":
                if qseq is not None:
                    self._plog(qseq, f'trial {trial}/{self.max_trials} PARSE FAIL | status={status}', icon='💥')
                meta = self._meta_base(t_start)
                meta["status"] = status
                meta["accept_reason"] = "execution_failure"
                meta["trial"] = trial
                meta["prompt_kind"] = prompt_kind
                meta["candidate_ids"] = cand_ids
                meta["conf"] = conf
                meta["llm_raw"] = norm_meta
                meta["cos_raw_max"] = cos_raw_max
                meta["margin_raw"] = margin_raw
                meta["n_llm_calls"] = int(n_llm_calls)
                meta["cos_weights_used"] = [float(wL), float(wS), float(wD)]
                if last_adapt_meta is not None:
                    meta["cos_adapt_last"] = last_adapt_meta
                return int(self.abstain_node_id), conf, 0.0, json.dumps(meta, ensure_ascii=False)


            chosen_id = int(nid)
            cos_sel = float(cos_norm_all.get(chosen_id, 0.0))
            combined = float(self.alpha * conf + (1.0 - self.alpha) * cos_sel)
            accept = combined >= float(self.combined_threshold)
            if qseq is not None:
                dec = 'ACCEPT' if accept else ('REJECT' if trial >= self.max_trials else 'RETRY')
                self._plog(
                    qseq,
                    f'trial {trial}/{self.max_trials} SCORE | pick={self._snip_node(chosen_id)} | conf={conf:.3f} cos={cos_sel:.3f} comb={combined:.3f} th={float(self.combined_threshold):.3f} => {dec}',
                    icon='⚖️',
                )

            meta = self._meta_base(t_start)
            meta.update(
                {
                    "status": "ok",
                    "trial": trial,
                    "prompt_kind": prompt_kind,
                    "candidate_ids": cand_ids,
                    "n_llm_calls": int(n_llm_calls),
                    "cos_weights_used": [float(wL), float(wS), float(wD)],
                    "chosen_id": chosen_id,
                    "conf": conf,
                    "cos": cos_sel,
                    "combined": combined,
                    "accept": bool(accept),
                    "cos_raw_max": cos_raw_max,
                    "margin_raw": margin_raw,
                    "llm_raw": norm_meta,
                }
            )
            if last_adapt_meta is not None:
                meta["cos_adapt_last"] = last_adapt_meta

            if accept:
                meta["accept_reason"] = "accept"
                return chosen_id, conf, cos_sel, json.dumps(meta, ensure_ascii=False)

            if trial >= self.max_trials:
                meta["accept_reason"] = "threshold_reject"
                return int(self.abstain_node_id), conf, cos_sel, json.dumps(meta, ensure_ascii=False)

            # Retry: cos_adapt + recompute cos for next trial
            if self.cos_adapt_enabled:
                wL, wS, wD, adapt_meta = self._adapt_weights_from_zscore(
                    cos_label=cosL,
                    cos_sem=cosS,
                    cos_desc=cosD,
                    chosen_id=chosen_id,
                    base_w_label=wL,
                    base_w_sem=wS,
                    base_w_desc=wD,
                    strength=self.cos_adapt_strength,
                    tau=self.cos_adapt_tau,
                )
            else:
                adapt_meta = {"enabled": False}

            meta["accept_reason"] = "retry"
            last_adapt_meta = adapt_meta

            # Recompute cos scores for next loop iteration
            cos_raw_all = self._compute_cos_raw_all(q, wL, wS, wD)
            cos_norm_all = self._normalize_cos(cos_raw_all)
            continue

        meta = self._meta_base(t_start)
        meta["status"] = "exception"
        meta["accept_reason"] = "execution_failure"
        meta["detail"] = "unexpected_fallthrough"
        return int(self.abstain_node_id), 0.0, 0.0, json.dumps(meta, ensure_ascii=False)

    def _on_query(self, msg: String) -> None:
        qseq = self._next_qseq()
        try:
            chosen_id, conf, cos_sel, meta_json = self._process_query(msg.data, qseq=qseq)
        except Exception as e:
            t_start = time.time()
            meta = self._meta_base(t_start)
            meta["status"] = "exception"
            meta["accept_reason"] = "execution_failure"
            meta["detail"] = safe_str(e)
            meta["n_llm_calls"] = 0
            chosen_id, conf, cos_sel, meta_json = int(self.abstain_node_id), 0.0, 0.0, json.dumps(meta, ensure_ascii=False)

        self.pub_id.publish(Int32(int(chosen_id)))
        self.pub_conf.publish(Float32(float(conf)))
        self.pub_cos.publish(Float32(float(cos_sel)))
        self.pub_meta.publish(String(meta_json))
        # One-line, human-friendly summary on screen (does not affect processing / published topics).
        try:
            self._log_query_summary(qseq=qseq, query=msg.data, chosen_id=int(chosen_id), conf=float(conf), cos=float(cos_sel), meta_json=meta_json)
        except Exception:
            pass



    def _log_query_summary(self, qseq: Optional[int], query: str, chosen_id: int, conf: float, cos: float, meta_json: str) -> None:
        """
        Print a single, terminal-friendly line per query.

        NOTE: This is intentionally *side-effect free* w.r.t. the research logic:
        - does not change any decision outputs
        - does not change any published topics
        - only formats & emits ROS logs
        """
        q = safe_str(query).replace("\n", " ").replace("\r", " ").strip()
        if len(q) > 60:
            q_disp = q[:57] + "..."
        else:
            q_disp = q

        try:
            meta = json.loads(meta_json) if isinstance(meta_json, str) and meta_json else {}
        except Exception:
            meta = {}

        status = safe_str(meta.get("status", ""))
        reason = safe_str(meta.get("accept_reason", ""))
        trial = meta.get("trial", "?")
        calls = meta.get("n_llm_calls", "?")

        # Latency (best-effort)
        try:
            t0 = float(meta.get("timestamp_wall", now_wall()))
            dt_ms = int(round((now_wall() - t0) * 1000.0))
        except Exception:
            dt_ms = -1

        # Outcome icon
        if reason in ("accept", "baseline_ap_accept", "baseline_r_accept"):
            icon = "✅"
            tag = "ACCEPT"
        elif reason == "oom_gate":
            icon = "🚫"
            tag = "OoM"
        elif reason in ("threshold_reject", "baseline_r_reject"):
            icon = "⛔"
            tag = "REJECT"
        elif reason == "execution_failure" or (status and status != "ok"):
            icon = "💥"
            tag = "FAIL"
        else:
            icon = "❔"
            tag = reason if reason else (status if status else "UNKNOWN")

        # Label (if available)
        label = ""
        if chosen_id in self.node_by_id:
            label = safe_str(self.node_by_id[chosen_id].label)
            label = label.replace("\n", " ").strip()
            if len(label) > 24:
                label = label[:21] + "..."

        # Compose a compact line
        qprefix = f"Q{int(qseq):05d} " if isinstance(qseq, int) else ""
        head = f"{qprefix}🧭{icon} [{dt_ms}ms t{trial} c{calls}] {tag}"

        # Extra, mode-dependent details
        combined = meta.get("combined", None)
        if tag == "OoM":
            maxcos = meta.get("cos_raw_max", None)
            margin = meta.get("margin_raw", None)
            extra = f"maxcos={maxcos:.3f} margin={margin:.3f}" if isinstance(maxcos, (int, float)) and isinstance(margin, (int, float)) else ""
            line = f'{head} | abstain={self.abstain_node_id} {extra} | q="{q_disp}"'
        elif tag in ("ACCEPT", "REJECT"):
            if combined is not None and isinstance(combined, (int, float)):
                line = f'{head} | id={chosen_id} "{label}" | conf={conf:.3f} cos={cos:.3f} cmb={float(combined):.3f} (th={self.combined_threshold:.3f}) | q="{q_disp}"'
            else:
                line = f'{head} | id={chosen_id} "{label}" | conf={conf:.3f} | q="{q_disp}"'
        else:
            detail = safe_str(meta.get("detail", ""))
            if len(detail) > 80:
                detail = detail[:77] + "..."
            status_part = f"status={status}" if status else ""
            reason_part = f"reason={reason}" if reason else ""
            parts = " ".join(p for p in [status_part, reason_part] if p).strip()
            if detail:
                parts = (parts + f" detail={detail}").strip()
            line = f'{head} | {parts} | q="{q_disp}"' if parts else f'{head} | q="{q_disp}"'

        rospy.loginfo(line)
    def spin(self) -> None:
        rospy.spin()


def main() -> None:
    try:
        node = LLMReasonerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

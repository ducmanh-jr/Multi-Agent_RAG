"""Regression tests for GPUFinetune summarizer/retrieval optimizations & Qwen3-8B/14B QLoRA pipeline."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agents.agent1_retrieval.tools.remote_dense import RemoteDenseRetriever
from agents.agent1_retrieval.tools.semantic_pipeline import MAX_REMOTE_DENSE_POOL, _SHARED_DENSE_RETRIEVER
from core.config import RunConfig, use_run_config
from modes.gpu_finetune import retrieval as gpu_ret_mod
from modes.gpu_finetune import summarizer as gpu_sum_mod
from modes.hybrid.retrieval import HybridRetrievalAgent
from utils.llm_client import llm_client


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT3_DIR = REPO_ROOT / "training_lab" / "agent3_summarizer"


def test_gpu_summarizer_ranks_25_units_and_resolves_unit_id_aliases(monkeypatch):
    distractors = [
        {
            "id": f"lora_2106.09685_p{i}_c0",
            "doc_id": "lora_2106.09685",
            "page_no": i,
            "text": f"Background appendix paragraph {i} discussing unrelated dataset preprocessing details. " * 25,
        }
        for i in range(1, 24)
    ]
    gold_unit = {
        "id": "lora_2106.09685_p1_c1",
        "doc_id": "lora_2106.09685",
        "page_no": 1,
        "text": "Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
    }
    units = distractors + [gold_unit]
    assert len(units) == 24

    ranked = gpu_sum_mod._rank_units_for_gpu_summarizer(
        "By how much does LoRA reduce trainable parameters and GPU memory compared to GPT-3 175B?",
        units,
    )
    assert ranked[0]["id"] == "lora_2106.09685_p1_c1"

    # Unit ID alias resolution handles ordinals, partial cues, and leaves bogus 'invalid' unresolved
    included = [gold_unit, distractors[0]]
    assert gpu_sum_mod._resolve_unit_id_alias("1", included) == ["lora_2106.09685_p1_c1"]
    assert gpu_sum_mod._resolve_unit_id_alias("[1]", included) == ["lora_2106.09685_p1_c1"]
    assert gpu_sum_mod._resolve_unit_id_alias("u1", included) == ["lora_2106.09685_p1_c1"]
    assert gpu_sum_mod._resolve_unit_id_alias("p1_c1", included) == ["lora_2106.09685_p1_c1"]
    assert gpu_sum_mod._resolve_unit_id_alias("invalid", included) == ["invalid"]

    # End-to-end GPUFinetuneSummarizerAgent with partial alias 'lora_p1' & omitted claim unit_id
    agent = gpu_sum_mod.GPUFinetuneSummarizerAgent(max_internal_retry=1)
    monkeypatch.setattr(agent.settings, "REMOTE_INFERENCE_URL", "https://worker.invalid")
    critic_payloads: list[dict] = []

    def fake_specialist(endpoint, payload, **kwargs):
        if endpoint.endswith("generator"):
            assert payload["language"] == "en"
            return {
                "answer_status": "ANSWERED",
                "answer": "LoRA reduces trainable parameters by 10,000 times and GPU memory by 3 times.",
                "claims": [{"text": "LoRA reduces trainable parameters by 10,000 times and GPU memory by 3 times."}],
                "evidence_spans": [
                    {
                        "quote": "LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.",
                        "unit_id": "lora_p1",
                    }
                ],
            }
        critic_payloads.append(payload)
        return {"is_supported": True, "action": "accept", "critique": "ok", "issues_found": []}

    monkeypatch.setattr(llm_client, "call_kaggle_specialist", fake_specialist)
    out = agent.run(
        {
            "query": "By how much does LoRA reduce trainable parameters and GPU memory compared to GPT-3 175B?",
            "source_units": units,
        }
    )
    assert out["generator_model_family"] == "gpu_finetune_dual_qlora"
    assert out["critic_action"] == "approve"
    assert len(out["claims"]) == 1
    assert out["evidence_spans"][0]["source_unit_id"] == "lora_2106.09685_p1_c1"
    # Compact critic context should only include cited unit + at most 2 extras (not all 24 units!)
    assert critic_payloads and critic_payloads[0]["context"].count("[Unit ID:") <= 3


def test_gpu_summarizer_retries_once_when_first_attempt_unbound(monkeypatch):
    unit = {"id": "rag_p4_t1", "doc_id": "rag", "page_no": 4, "text": "RAG-Sequence achieves 44.5 EM on Natural Questions."}
    agent = gpu_sum_mod.GPUFinetuneSummarizerAgent(max_internal_retry=1)
    monkeypatch.setattr(agent.settings, "REMOTE_INFERENCE_URL", "https://worker.invalid")
    gen_calls: list[dict] = []

    def fake_specialist(endpoint, payload, **kwargs):
        if endpoint.endswith("generator"):
            gen_calls.append(payload)
            if len(gen_calls) == 1:
                return {"answer": "RAG-Sequence gets 44.5.", "claims": [{"text": "44.5", "unit_id": "totally_wrong"}], "evidence_spans": []}
            return {
                "answer": "RAG-Sequence achieves 44.5 EM on Natural Questions.",
                "claims": [{"text": "RAG-Sequence achieves 44.5 EM on Natural Questions.", "unit_id": "rag_p4_t1"}],
                "evidence_spans": [{"quote": "RAG-Sequence achieves 44.5 EM on Natural Questions.", "unit_id": "rag_p4_t1"}],
            }
        return {"is_supported": True, "action": "accept", "critique": "ok", "issues_found": []}

    monkeypatch.setattr(llm_client, "call_kaggle_specialist", fake_specialist)
    out = agent.run({"query": "What EM score does RAG-Sequence get on Natural Questions?", "source_units": [unit]})
    assert len(gen_calls) == 2
    assert "rag_p4_t1" in gen_calls[1]["feedback"]
    assert out["generator_model_family"] == "gpu_finetune_dual_qlora"
    assert out["claims"][0]["unit_id"] == "rag_p4_t1"


def test_gpu_retrieval_bounds_177_chunk_dense_pool_and_qwen3_embedding_truncates_passages(monkeypatch, tmp_path):
    big_chunks = [
        {"id": f"lora_p{(i % 25) + 1}_c{i}", "doc_id": "lora", "page_no": (i % 25) + 1, "text": f"LoRA memory experiment chunk {i} " + ("detail " * 400)}
        for i in range(177)
    ]
    monkeypatch.setattr(gpu_ret_mod, "load_or_parse_selected_pdf", lambda _: big_chunks)

    embedded_texts: list[str] = []
    revision = "c" * 40
    vec = [1.0] + [0.0] * 1023

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["texts"]
        if len(texts) > 1:
            embedded_texts.extend(texts)
        return httpx.Response(200, json={"model": "Qwen/Qwen3-Embedding-8B", "revision": revision, "embeddings": [vec for _ in texts]})

    cfg = RunConfig.capture(remote_inference_url="https://worker.invalid", corpus_path=str(tmp_path / "none.jsonl"))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, use_run_config(cfg):
        monkeypatch.setattr(llm_client, "http_client", client)
        agent = gpu_ret_mod.GPUFinetuneRetrievalAgent(enable_dense=True, enable_rerank=False, enable_remote_router=False)
        res = agent.run({"query": "LoRA memory", "pdf_paths": ["lora.pdf"], "top_k": 5})

    assert res["reasoning_trace"][0]["details"]["dense"]["status"] == "used"
    assert len(embedded_texts) <= MAX_REMOTE_DENSE_POOL
    assert all(len(t) <= RemoteDenseRetriever.QWEN3_LARGE_PASSAGE_MAX_LENGTH for t in embedded_texts)
    assert HybridRetrievalAgent(enable_dense=False, enable_rerank=False).dense is _SHARED_DENSE_RETRIEVER


def test_kaggle_server_generator_prompt_language_lock_and_sota_max_batch_tuning(monkeypatch):
    server = importlib.import_module("kaggle.kaggle_finetune_server")
    run_kaggle = importlib.import_module("kaggle.run_kaggle")

    captured_prompts: list[str] = []

    def fake_gen(prompt, **kwargs):
        captured_prompts.append(prompt)
        return '{"answer":"ok","claims":[{"text":"ok","unit_id":"u1",},],"evidence_spans":[{"quote":"ok","unit_id":"u1"}],}'

    monkeypatch.setattr(server, "generate_with_qwen", fake_gen)
    out_en = server.summarize_generator(server.SummarizeRequest(query="What BLEU?", context="[Unit ID: u1 | Doc: a | Trang: 1]:\nok", language="en"))
    assert out_en["answer"] == "ok"
    assert "strictly in English" in captured_prompts[-1]
    assert "EXACT string after '[Unit ID:'" in captured_prompts[-1]

    out_vi = server.summarize_generator(server.SummarizeRequest(query="Đạt điểm BLEU bao nhiêu?", context="[Unit ID: u1 | Doc: a | Trang: 1]:\nok"))
    assert out_vi["answer"] == "ok"
    assert "hoàn toàn bằng tiếng Việt" in captured_prompts[-1]

    sota_max_env = run_kaggle.WORKER_PROFILE_SPECS["hybrid_sota_max"]["env_defaults"]
    assert sota_max_env["EMBEDDING_BATCH_SIZE"] == "4"
    assert sota_max_env["EMBEDDING_MAX_TOKENS"] == "384"
    assert sota_max_env["QWEN_RERANK_BATCH_SIZE"] == "4"
    assert sota_max_env["RERANKER_MAX_TOKENS"] == "384"


def test_qwen3_qlora_dataset_builder_and_presets_offline(tmp_path):
    if not (AGENT3_DIR / "build_qwen3_dataset.py").is_file():
        pytest.skip("training_lab/agent3_summarizer is gitignored in lightweight CI checkouts")
    if str(AGENT3_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT3_DIR))

    from build_qwen3_dataset import build_qwen3_datasets
    from src.config import Agent3TrainingConfig
    from src.data_loader import apply_chat_template_safe
    from src.model_builder import prepare_t4_training_load_kwargs

    out_dir = tmp_path / "qwen3_ds"
    manifest = build_qwen3_datasets(
        legacy_dataset_dir=AGENT3_DIR / "dataset",
        output_dir=out_dir,
        seed=42,
        max_legacy_rows=20,
    )
    assert manifest["contract_pass_rate"] == 1.0
    assert manifest["counts"]["generator_sft"] >= 70
    assert manifest["counts"]["generator_dpo"] >= 80
    assert manifest["counts"]["critic_sft"] >= 100
    assert "table_column_swap" in manifest["generator_dpo_by_failure_type"]
    assert "malformed_unit_id" in manifest["generator_dpo_by_failure_type"]
    assert "hallucinated_on_trap" in manifest["generator_dpo_by_failure_type"]

    cfg_8b = Agent3TrainingConfig.from_preset("qwen3_8b", dataset_dir=out_dir)
    cfg_14b = Agent3TrainingConfig.from_preset("qwen3_14b", dataset_dir=out_dir)
    assert cfg_8b.base_model_id == "Qwen/Qwen3-8B" and cfg_8b.disable_thinking is True and cfg_8b.gen_lora_r == 32
    assert cfg_14b.base_model_id == "unsloth/Qwen3-14B-bnb-4bit" and cfg_14b.disable_thinking is True and cfg_14b.gen_lora_r == 16
    assert cfg_14b.resolve_dataset_files()["generator_sft"].exists()

    # Chat template helper passes enable_thinking=False for Qwen3 and falls back cleanly on legacy tokenizers
    seen_kw: dict = {}

    class Qwen3Tok:
        def apply_chat_template(self, msgs, **kw):
            seen_kw.update(kw)
            return "qwen3_prompt"

    class LegacyTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "legacy_prompt"

    assert apply_chat_template_safe(Qwen3Tok(), [{"role": "user", "content": "hi"}], disable_thinking=True) == "qwen3_prompt"
    assert seen_kw.get("enable_thinking") is False
    assert apply_chat_template_safe(LegacyTok(), [{"role": "user", "content": "hi"}], disable_thinking=True) == "legacy_prompt"

    # Pre-quantized 14B checkpoint mutates bnb_4bit_compute_dtype='float16' for T4 without conflicting quantization_config
    class FakePreQuantCfg:
        quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_compute_dtype": "bfloat16"}

    class FakeAutoCfg:
        @staticmethod
        def from_pretrained(model_id, **kw):
            return FakePreQuantCfg()

    load_kw = prepare_t4_training_load_kwargs(
        "unsloth/Qwen3-14B-bnb-4bit",
        device_map={"": 0},
        max_memory=None,
        bnb_config=SimpleNamespace(load_in_4bit=True),
        auto_config_cls=FakeAutoCfg,
    )
    assert "quantization_config" not in load_kw
    assert load_kw["config"].quantization_config["bnb_4bit_compute_dtype"] == "float16"

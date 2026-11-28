"""
GPU Finetune Mode - Agentic Summarizer Agent (7-Model SOTA Pipeline).
Implements the full Dual-QLoRA Generator -> Critic self-correction loop:
  - Generator QLoRA (03_agent3_summarizer/generator_qlora): Grounded synthesis with claim decomposition.
  - Critic QLoRA (03_agent3_summarizer/critic_qlora): Factuality evaluation with actionable feedback.
  - Executes up to max_internal_retry=2 self-correction rewrite passes when Critic signals issues.
  - Zero-Hallucination Out-of-Domain Guard protection.
"""

from __future__ import annotations

import re
from typing import Any

from agents.agent3_summarizer.components.generator import (
    _clean_unit_id_ref,
    _extract_verbatim_unit_quote,
    bind_evidence_spans,
    bind_summary_claims,
    pack_context_by_token_budget,
)
from agents.agent1_retrieval.tools.bm25 import SimpleBM25
from core.base_agent import BaseAgent
from core.config import get_role_max_tokens
from core.handoff_inputs import revision_note
from core.schemas import SummarizerCriticOutput, SummarizerGeneratorOutput
from core.state import AgentState, EvidenceSpan
from core.verifier import (
    TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
    TRAINING_LAB_GENERATOR_SYSTEM_PROMPT,
    ensure_inline_citations,
)
from utils.bilingual import expand_bilingual_tokens
from utils.llm_client import llm_client
from utils.domain_guard import check_out_of_domain, build_out_of_domain_response
from utils.language import detect_language, normalize_language_code
from modes.cpu_min.summarizer import CPUMinSummarizerAgent


_ORDINAL_UID_RE = re.compile(
    r"^(?:\[)?(?:u|unit|chunk|doc|source|passage|ref)?[\s_#-]*(\d+)(?:\])?$",
    re.IGNORECASE,
)
_PAGE_CUE_RE = re.compile(r"(?:^|[_\-\s])p(?:age)?[\s_:-]*(\d+)(?:$|[_\-\s])", re.IGNORECASE)
_NUM_TOKEN_RE = re.compile(r"\b\d+(?:[.,]\d+)*%?\b")
_TABLE_HINT_RE = re.compile(r"\b(?:table|tab\.?|bảng|bang|accuracy|bleu|exact\s+match|score|latency|memory|parameters?)\b", re.IGNORECASE)


def _rank_units_for_gpu_summarizer(query: str, source_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ranks large unit lists (>8 units) so the highest-relevance evidence & tables stay inside the
    token budget instead of getting silently truncated when Reader emits 20-30 units.
    """
    if len(source_units) <= 8 or not (query or "").strip():
        return list(source_units)

    bm25 = SimpleBM25()
    bm25.fit(source_units)
    bm25_hits = bm25.search(query, top_k=min(len(source_units), 16))
    bm25_rank = {u.get("id"): idx for idx, u in enumerate(bm25_hits) if u.get("id")}

    base_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", query.lower()) if len(w) >= 3}
    expanded_toks = expand_bilingual_tokens(query, base_toks)
    query_nums = {n.replace(",", "") for n in _NUM_TOKEN_RE.findall(query)}
    wants_table = bool(_TABLE_HINT_RE.search(query))

    def _unit_score(idx_unit: tuple[int, dict[str, Any]]) -> float:
        orig_idx, unit = idx_unit
        uid = unit.get("id")
        txt = str(unit.get("text") or "")
        txt_l = txt.lower()
        u_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", txt_l) if len(w) >= 3}
        overlap = len(expanded_toks & u_toks)
        bm25_bonus = (2.2 / (bm25_rank[uid] + 1)) if uid in bm25_rank else 0.0
        num_bonus = 0.0
        if query_nums:
            u_nums = {n.replace(",", "") for n in _NUM_TOKEN_RE.findall(txt)}
            num_bonus = 0.8 * len(query_nums & u_nums)
        table_bonus = 0.45 if (wants_table and (str(unit.get("unit_type") or "").lower() == "table" or "|" in txt)) else 0.0
        # Mild original-order prior preserves retrieval rank among ties
        order_prior = 0.25 / (orig_idx + 1)
        return bm25_bonus + 0.25 * min(overlap, 10) + num_bonus + table_bonus + order_prior

    ranked = sorted(enumerate(source_units), key=_unit_score, reverse=True)
    return [u for _, u in ranked]


def _resolve_unit_id_alias(raw_uid: Any, included: list[dict[str, Any]]) -> list[str]:
    """
    Deterministically resolves common LLM unit_id formatting aliases (header wrappers, case differences,
    1-based ordinal indices like '1'/'[1]'/'u1'/'Unit 1', or unambiguous prefix/suffix matches)
    back to exact `included` unit IDs while leaving genuinely bogus IDs unresolved for strict rejection.
    """
    cleaned_list = _clean_unit_id_ref(raw_uid)
    if not cleaned_list:
        return []
    included_ids = [str(u.get("id")) for u in included if isinstance(u, dict) and u.get("id")]
    uid_set = set(included_ids)
    lower_map = {uid.lower(): uid for uid in included_ids}

    resolved: list[str] = []
    for tok in cleaned_list:
        target = tok
        if tok in uid_set:
            target = tok
        elif tok.lower() in lower_map:
            target = lower_map[tok.lower()]
        else:
            m_ord = _ORDINAL_UID_RE.match(tok)
            if m_ord:
                ord_idx = int(m_ord.group(1))
                if 1 <= ord_idx <= len(included_ids):
                    target = included_ids[ord_idx - 1]
            elif len(tok) >= 4:
                tok_l = tok.lower()
                affix_matches = [
                    uid for uid in included_ids
                    if uid.lower().startswith(tok_l) or uid.lower().endswith(tok_l)
                ]
                if len(affix_matches) == 1:
                    target = affix_matches[0]
        if target and target not in resolved:
            resolved.append(target)
    return resolved


def _snap_partial_span_unit_id(raw_uid_str: str, quote: str, included: list[dict[str, Any]]) -> str | None:
    """
    If an LLM emitted a truncated/partial unit_id (e.g., 'lora_p1' instead of 'lora_2106.09685_p1_c1')
    whose verbatim quote uniquely matches exactly ONE included unit sharing a doc/page cue with `raw_uid_str`,
    returns that exact unit ID. Never overrides a contradictory valid ID or unrelated bogus token.
    """
    if not raw_uid_str or not quote:
        return None
    normalized = " ".join(quote.split())
    matched_units = [
        u for u in included
        if isinstance(u, dict) and u.get("id") and _extract_verbatim_unit_quote(normalized, str(u.get("text") or ""))
    ]
    if len(matched_units) != 1:
        return None
    cand = matched_units[0]
    raw_l = raw_uid_str.lower()
    cand_id_l = str(cand.get("id") or "").lower()
    cand_doc_l = str(cand.get("doc_id") or "").lower()
    cand_page = cand.get("page_no")

    page_m = _PAGE_CUE_RE.search(raw_l)
    if page_m and cand_page is not None and int(page_m.group(1)) == int(cand_page):
        return str(cand["id"])
    raw_parts = [p for p in re.split(r"[^a-z0-9]+", raw_l) if len(p) >= 3]
    if any(part in cand_id_l or part in cand_doc_l for part in raw_parts):
        return str(cand["id"])
    return None


def _normalize_generator_refs(
    raw_spans: list[Any],
    raw_claims: list[Any],
    included: list[dict[str, Any]],
) -> tuple[list[Any], list[Any]]:
    """Normalizes unit_id aliases consistently across `evidence_spans` and `claims` before strict binding."""
    included_ids = {str(u.get("id")) for u in included if isinstance(u, dict) and u.get("id")}
    partial_alias_map: dict[str, str] = {}

    norm_spans: list[Any] = []
    for sp in raw_spans or []:
        if isinstance(sp, str):
            norm_spans.append(sp)
            continue
        if not isinstance(sp, dict):
            continue
        item = dict(sp)
        raw_uid = item.get("source_unit_id") or item.get("unit_id")
        if raw_uid is not None and str(raw_uid).strip():
            resolved = _resolve_unit_id_alias(raw_uid, included)
            if len(resolved) == 1 and resolved[0] in included_ids:
                item["unit_id"] = resolved[0]
                if "source_unit_id" in item:
                    item["source_unit_id"] = resolved[0]
                partial_alias_map[str(raw_uid).strip().lower()] = resolved[0]
            else:
                snapped = _snap_partial_span_unit_id(str(raw_uid).strip(), str(item.get("quote") or ""), included)
                if snapped:
                    item["unit_id"] = snapped
                    if "source_unit_id" in item:
                        item["source_unit_id"] = snapped
                    partial_alias_map[str(raw_uid).strip().lower()] = snapped
        norm_spans.append(item)

    norm_claims: list[Any] = []
    for cl in raw_claims or []:
        if isinstance(cl, str):
            norm_claims.append(cl)
            continue
        if not isinstance(cl, dict):
            continue
        item = dict(cl)
        raw_uid = item.get("unit_id") or item.get("source_unit_id") or item.get("evidence_unit_id") or item.get("unit_ids")
        if raw_uid is not None and (not isinstance(raw_uid, str) or raw_uid.strip()):
            if isinstance(raw_uid, str) and raw_uid.strip().lower() in partial_alias_map:
                mapped = partial_alias_map[raw_uid.strip().lower()]
                item["unit_id"] = mapped
                for k in ("source_unit_id", "evidence_unit_id", "unit_ids"):
                    item.pop(k, None)
            else:
                resolved = _resolve_unit_id_alias(raw_uid, included)
                if resolved and all(uid in included_ids for uid in resolved):
                    item["unit_id"] = resolved[0] if len(resolved) == 1 else resolved
                    for k in ("source_unit_id", "evidence_unit_id", "unit_ids"):
                        item.pop(k, None)
        norm_claims.append(item)

    return norm_spans, norm_claims


def _fill_omitted_claim_unit_ids(claims_raw: list[Any], refined_spans: list[dict[str, Any]]) -> list[Any]:
    """
    For claims where the generator omitted `unit_id` completely (empty/None), binds the claim to the
    uniquely supporting bound span when lexical+numeric alignment is unambiguous. Never overrides an
    explicitly supplied `unit_id`.
    """
    if not claims_raw or not refined_spans:
        return claims_raw
    out: list[Any] = []
    for cl in claims_raw:
        if not isinstance(cl, dict):
            out.append(cl)
            continue
        raw_uid = cl.get("unit_id") or cl.get("source_unit_id") or cl.get("evidence_unit_id") or cl.get("unit_ids")
        has_explicit = raw_uid is not None and (not isinstance(raw_uid, str) or raw_uid.strip() != "")
        if has_explicit or cl.get("evidence_ids"):
            out.append(cl)
            continue
        c_txt = str(cl.get("text") or cl.get("claim_text") or cl.get("claim") or "").strip()
        if not c_txt:
            out.append(cl)
            continue
        c_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", c_txt.lower()) if len(w) >= 3}
        c_exp = expand_bilingual_tokens(c_txt, c_toks)
        c_nums = {n.replace(",", "") for n in _NUM_TOKEN_RE.findall(c_txt)}
        best_span = None
        best_score = -1.0
        for sp in refined_spans:
            q_txt = str(sp.get("quote") or "")
            q_toks = {w for w in re.findall(r"[a-zA-Z0-9À-ỹ]+", q_txt.lower()) if len(w) >= 3}
            q_nums = {n.replace(",", "") for n in _NUM_TOKEN_RE.findall(q_txt)}
            if c_nums and not c_nums.issubset(q_nums):
                continue
            ov = len(c_exp & q_toks)
            score = ov + (2.0 * len(c_nums & q_nums))
            if score > best_score:
                best_score = score
                best_span = sp
        min_req = 1.0 if len(refined_spans) == 1 else 2.0
        if best_span is not None and best_score >= min_req:
            updated = dict(cl)
            updated["unit_id"] = best_span["source_unit_id"]
            out.append(updated)
        else:
            out.append(cl)
    return out


def _build_compact_critic_context(
    refined_spans: list[dict[str, Any]],
    included: list[dict[str, Any]],
    max_tokens: int = 1400,
) -> str:
    """Packs cited units first (plus up to 2 top units) so the remote Critic runs ~2x faster with less distraction."""
    cited_ids = [sp.get("source_unit_id") for sp in refined_spans if sp.get("source_unit_id")]
    cited_set = set(cited_ids)
    cited_units = [u for u in included if u.get("id") in cited_set]
    other_units = [u for u in included if u.get("id") not in cited_set][:2]
    compact_ctx, _ = pack_context_by_token_budget(cited_units + other_units, max_tokens=max_tokens)
    return compact_ctx


class GPUFinetuneSummarizerAgent(BaseAgent):
    """Summarizer Agent executing Fine-Tuned Generator + Critic Dual-QLoRA loop."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="GPUFinetuneSummarizerAgent", max_internal_retry=max_internal_retry)
        self.cpu_engine = CPUMinSummarizerAgent()

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        source_units = state.get("source_units", []) or []

        self.logger.info(
            f"[GPU Finetune Summarizer] Running Dual-QLoRA Generator -> Critic loop for: '{query[:65]}' "
            f"({len(source_units)} units)"
        )

        clean_q = query.strip()
        ans_lang = normalize_language_code(state.get("answer_language")) or detect_language(clean_q, default="vi")
        effective_q = str(state.get("resolved_query") or clean_q).strip() or clean_q
        is_ood, ood_reason = check_out_of_domain(effective_q, source_units)
        if is_ood:
            self.logger.info(f"[GPU Finetune Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="gpu_finetune_guard", answer_language=ans_lang)
            trace = self.log_action("gpu_finetune_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            res["answer_language"] = ans_lang
            return res

        revision = revision_note(state)
        ordered_units = source_units if revision else _rank_units_for_gpu_summarizer(effective_q, source_units)
        context_str, included = pack_context_by_token_budget(ordered_units, max_tokens=3000)
        included_by_id = {u.get("id"): u for u in included if isinstance(u, dict) and u.get("id")}

        current_answer = ""
        current_claims: list = []
        claims, refined_spans = [], []
        current_spans_raw: list[dict[str, Any]] = []
        internal_attempts = 0
        last_critic_feedback = ""
        critic_action = "lower_confidence"
        critique_text = ""
        critic_out = None
        gen_status: str | None = None

        while internal_attempts <= self.max_internal_retry:
            internal_attempts += 1

            # 1. Generator Pass: Call Remote fine-tuned Generator QLoRA
            lang_hint = (
                "Write 'answer' and all 'claims' strictly in English; copy 'evidence_spans[].quote' verbatim from the source unit."
                if ans_lang == "en"
                else "Viết 'answer' và toàn bộ 'claims' bằng tiếng Việt; chép nguyên văn 'evidence_spans[].quote' từ đoạn nguồn."
            )
            feedback = "\n\n".join(part for part in (revision, last_critic_feedback) if part)
            gen_prompt = f"User Question: {effective_q}\nLanguage requirement: {lang_hint}\n\n"
            if feedback:
                gen_prompt += f"[Previous Critic Feedback to Address]:\n{feedback}\n\n"
            gen_prompt += f"Context:\n{context_str}\n\n"

            try:
                gen_dict = llm_client.call_kaggle_specialist(
                    endpoint="api/summarize/generator",
                    payload={
                        "query": effective_q,
                        "context": context_str,
                        "feedback": feedback,
                        "language": ans_lang,
                        "adapter": "generator",
                        "max_tokens": get_role_max_tokens("generator"),
                    },
                    fallback_prompt=gen_prompt,
                    fallback_system=TRAINING_LAB_GENERATOR_SYSTEM_PROMPT,
                    adapter="generator",
                )
                raw_status = str(gen_dict.get("answer_status") or "").strip().upper() if isinstance(gen_dict, dict) else ""
                if raw_status in {"NOT_FOUND_IN_DOCUMENT", "INSUFFICIENT_EVIDENCE", "CLARIFICATION_NEEDED"}:
                    gen_status = raw_status
                gen_out = SummarizerGeneratorOutput.from_dict(gen_dict, default_query=effective_q)
                current_answer = gen_out.answer
                current_claims = gen_out.claims
                current_spans_raw = gen_out.evidence_spans
            except Exception as e:
                self.logger.warning(f"[GPU Finetune Summarizer] Remote generator fallback to local: {e}")
                result = self.cpu_engine.run(state)
                return {**result, "generator_model_family": "gpu_finetune_cpu_fallback", "answer_language": ans_lang}

            if gen_status in {"NOT_FOUND_IN_DOCUMENT", "INSUFFICIENT_EVIDENCE", "CLARIFICATION_NEEDED"} and current_answer:
                trace = self.log_action(
                    "gpu_finetune_summarization_completed",
                    {"attempts": internal_attempts, "critic_action": "request_more_context", "status": gen_status, "claims_count": 0},
                )
                return {
                    "answer": current_answer,
                    "answer_status": gen_status,
                    "answer_language": ans_lang,
                    "claims": [],
                    "evidence_spans": [],
                    "critic_action": "request_more_context" if gen_status != "CLARIFICATION_NEEDED" else "lower_confidence",
                    "critic_feedback": {"action": "request_more_context", "missing_key_points": [clean_q]},
                    "generator_model_family": "gpu_finetune_dual_qlora",
                    "reasoning_trace": [trace],
                }

            norm_spans_raw, norm_claims_raw = _normalize_generator_refs(current_spans_raw, current_claims, included)
            refined_spans = bind_evidence_spans(norm_spans_raw, included)
            # If the generator cited a valid unit_id in claims but slightly altered a ligature/quote in evidence_spans,
            # bind a deterministic verbatim slice from that exact included unit so valid synthesis is preserved.
            covered_uids = {sp["source_unit_id"] for sp in refined_spans}
            from core.verifier import _extract_focused_evidence_window

            for rc in norm_claims_raw:
                if not isinstance(rc, dict):
                    continue
                rc_txt = str(rc.get("text") or rc.get("claim") or "").strip()
                req_uids = _resolve_unit_id_alias(
                    rc.get("unit_id") or rc.get("source_unit_id") or rc.get("evidence_unit_id") or rc.get("unit_ids"),
                    included,
                )
                for uid in req_uids:
                    if uid not in covered_uids and uid in included_by_id:
                        u_obj = included_by_id[uid]
                        u_txt = " ".join(str(u_obj.get("text") or "").split()).strip()
                        if u_txt and u_obj.get("doc_id") and type(u_obj.get("page_no")) is int:
                            focused_q = _extract_focused_evidence_window(rc_txt or effective_q, u_txt, max_chars=480)
                            refined_spans.append(
                                EvidenceSpan(
                                    id=f"ev_{len(refined_spans)+1}",
                                    quote=focused_q,
                                    source_unit_id=uid,
                                    doc_id=u_obj["doc_id"],
                                    page_no=u_obj["page_no"],
                                    bbox=u_obj.get("bbox"),
                                ).to_dict()
                            )
                            covered_uids.add(uid)
            aligned_claims_raw = _fill_omitted_claim_unit_ids(norm_claims_raw, refined_spans)
            claims = bind_summary_claims(aligned_claims_raw or [s["quote"] for s in refined_spans], refined_spans)
            if not current_answer or not claims:
                if internal_attempts <= self.max_internal_retry and included:
                    allowed_ids = ", ".join(str(u.get("id")) for u in included[:8] if u.get("id"))
                    last_critic_feedback = (
                        "Previous JSON had unbound unit_ids or non-verbatim quotes. "
                        f"Copy exact unit_id values from [{allowed_ids}] and exact verbatim substrings from those units."
                    )
                    self.logger.info(
                        "[GPU Finetune Summarizer] Attempt %d unbound -> retrying with exact unit_id guidance.",
                        internal_attempts,
                    )
                    continue
                self.logger.info("[GPU Finetune Summarizer] No grounded synthesis; using extractive reader output.")
                result = self.cpu_engine.run(state)
                return {**result, "generator_model_family": "gpu_finetune_cpu_fallback", "answer_language": ans_lang}

            # 2. Critic Pass: Call Remote fine-tuned Critic QLoRA on focused cited context only
            try:
                critic_claims = [c["text"] for c in claims]
                critic_ctx = _build_compact_critic_context(refined_spans, included, max_tokens=1400) or context_str
                critic_dict = llm_client.call_kaggle_specialist(
                    endpoint="api/summarize/critic",
                    payload={
                        "query": clean_q,
                        "answer": current_answer,
                        "context": critic_ctx,
                        "claims": critic_claims,
                        "adapter": "critic",
                        "max_tokens": min(get_role_max_tokens("critic"), 224),
                    },
                    fallback_prompt=f"Question: {clean_q}\nContext:\n{critic_ctx}\nAnswer:\n{current_answer}\nClaims: {critic_claims}",
                    fallback_system=TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
                    adapter="critic",
                )
                critic_out = SummarizerCriticOutput.from_dict(critic_dict)
                critic_action = critic_out.action
                critique_text = getattr(critic_out, "critique_rationale", "") or str(getattr(critic_out, "critique", ""))

                if critic_action in ("accept", "approve") or "invalid critic output" in critique_text.lower():
                    if "invalid critic output" in critique_text.lower():
                        critic_action = "approve"
                        self.logger.info(
                            f"[GPU Finetune Critic] Non-JSON critic reply on attempt {internal_attempts}; "
                            "forwarding grounded draft directly to Fact-Checker."
                        )
                    else:
                        self.logger.info(f"[GPU Finetune Critic] PASS: Answer accepted on attempt {internal_attempts}")
                    break
                elif critic_action == "rewrite" and internal_attempts <= self.max_internal_retry:
                    last_critic_feedback = f"Critique: {critique_text}. Issues: {getattr(critic_out, 'issues_found', [])}"
                    self.logger.info(f"[GPU Finetune Critic] REWRITE requested ({critique_text}) -> Looping pass {internal_attempts + 1}")
                    continue
                else:
                    break
            except Exception as e:
                critic_action = "lower_confidence"
                critique_text = "Critic unavailable"
                self.logger.warning(f"[GPU Finetune Critic] Remote critic unavailable ({e}).")
                break
        # Ensure inline citations
        cited_answer = ensure_inline_citations(current_answer, included, refined_spans)
        cited_answer = re.sub(r"[,;\s]*\[\s*$", "", cited_answer).strip()

        trace = self.log_action(
            "gpu_finetune_summarization_completed",
            {
                "attempts": internal_attempts,
                "critic_action": critic_action,
                "critique": critique_text,
                "claims_count": len(current_claims),
            },
        )

        return {
            "answer": cited_answer,
            "answer_language": ans_lang,
            "claims": claims,
            "evidence_spans": refined_spans,
            "critic_action": critic_action,
            "critic_feedback": {
                "action": critic_action,
                "critique": critique_text,
                "missing_key_points": getattr(critic_out, "missing_key_points", []) if critic_out and hasattr(critic_out, "missing_key_points") else [],
                "unsupported_claims": getattr(critic_out, "unsupported_claims", []) if critic_out and hasattr(critic_out, "unsupported_claims") else [],
            },
            "generator_model_family": "gpu_finetune_dual_qlora",
            "reasoning_trace": [trace],
        }

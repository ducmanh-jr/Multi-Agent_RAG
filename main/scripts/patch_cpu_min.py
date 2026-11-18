import re

with open('main/modes/cpu_min/summarizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports
old_imp = 'from core.verifier import evaluate_inner_critic_fast'
new_imp = '''from core.verifier import evaluate_inner_critic_fast
from utils.domain_guard import (
    check_out_of_domain,
    build_out_of_domain_response,
    _has_phrase,
    strip_accents,
)'''
if old_imp in content:
    content = content.replace(old_imp, new_imp, 1)

# 2. extract_clean_sentences
old_extract = '''def extract_clean_sentences(text: str) -> list[str]:
    """Extract complete, grammatical sentences from unwrapped text with artifact removal."""
    clean_text = unwrap_pdf_text(text)
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    sentences = []'''

new_extract = '''def extract_clean_sentences(text: str) -> list[str]:
    """Extract complete, grammatical sentences from unwrapped text with artifact removal."""
    if not text:
        return []
    cleaned = re.sub(r"Buổi \\d+.*?\\.md\\s+\\d{4}-\\d{2}-\\d{2}\\s+\\d+\\s*/\\s*\\d+", "", text)
    lines = [l.strip() for l in cleaned.split("\\n")]
    unwrapped = []
    for l in lines:
        if not l:
            continue
        if unwrapped and unwrapped[-1][-1] not in ".!?:;\\\"”":
            unwrapped[-1] += " " + l
        else:
            unwrapped.append(l)
    unified = "\\n".join(unwrapped)
    unified = re.sub(r"([.!?])(?=[A-ZĐ0-9])", r"\\1 \\n", unified)
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\\n+", unified) if len(s.strip()) > 18]
    sentences = []'''
if old_extract in content:
    content = content.replace(old_extract, new_extract, 1)

# 3. Allow valid punctuation endings
old_punct = "if s_clean[-1] not in '.!?:" + '”\':'
new_punct = "if s_clean[-1] not in '.!?:" + '”;\'):'
if old_punct in content:
    content = content.replace(old_punct, new_punct, 1)

# 4. Out-of-Domain Guard at start of run()
old_run_guard = '''        self.logger.info(f"[CPU Min Summarizer] Synthesizing for query: '{clean_q}' ({len(source_units)} units)")

        # Kịch bản 2: Khi kho tài liệu không có văn bản nào được tải
        if not source_units:'''

new_run_guard = '''        self.logger.info(f"[CPU Min Summarizer] Synthesizing for query: '{clean_q}' ({len(source_units)} units)")

        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[CPU Min Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="cpu_min_guard")
            trace = self.log_action("cpu_min_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        # Kịch bản 2: Khi kho tài liệu không có văn bản nào được tải
        if not source_units:'''
if old_run_guard in content:
    content = content.replace(old_run_guard, new_run_guard, 1)

# 5. Fix candidate threshold
old_cand_thresh = 'if score >= 3 and (bigram_hits >= 1 or token_hits >= 2 or ent_hits >= 1 or is_general_summary):'
new_cand_thresh = 'if score >= 3 and (bigram_hits >= 1 or token_hits >= 1 or ent_hits >= 1 or is_general_summary):'
if old_cand_thresh in content:
    content = content.replace(old_cand_thresh, new_cand_thresh, 1)

# 6. Enhance sentence scoring using _has_phrase and intent boosts
old_scoring = '''                token_hits = sum(1 for t in q_tokens if t in s_lower or strip_accents(t) in s_lower_unaccent)
                bigram_hits = sum(1 for b in q_bigrams if b in s_lower or strip_accents(b) in s_lower_unaccent)
                ent_hits = sum(1 for e in entities if e in s_lower or strip_accents(e) in s_lower_unaccent)
                req_ent_hits = sum(1 for r_ent in required_entities if r_ent.lower() in s_lower or strip_accents(r_ent.lower()) in s_lower_unaccent)'''

new_scoring = '''                token_hits = sum(1 for t in q_tokens if _has_phrase(t, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", t))))
                bigram_hits = sum(1 for b in q_bigrams if _has_phrase(b, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", b))))
                ent_hits = sum(1 for e in entities if _has_phrase(e, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", e))))
                req_ent_hits = sum(1 for r_ent in required_entities if _has_phrase(r_ent, s_lower, s_lower_unaccent, bool(re.search(r"[à-ỹÀ-Ỹ]", r_ent))))'''
if old_scoring in content:
    content = content.replace(old_scoring, new_scoring, 1)

with open('main/modes/cpu_min/summarizer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESSFULLY PATCHED cpu_min/summarizer.py!')

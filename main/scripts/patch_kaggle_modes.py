import re

# 1. Patch kaggle_fn/summarizer.py
with open('main/modes/kaggle_fn/summarizer.py', 'r', encoding='utf-8') as f:
    fn_content = f.read()

old_fn_imp = 'from utils.domain_guard import check_out_of_domain, build_out_of_domain_response'
new_fn_imp = '''from utils.domain_guard import check_out_of_domain, build_out_of_domain_response
from modes.cpu_min.summarizer import CPUMinSummarizerAgent'''
if old_fn_imp in fn_content and 'CPUMinSummarizerAgent' not in fn_content:
    fn_content = fn_content.replace(old_fn_imp, new_fn_imp, 1)

old_fn_init = '''    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleFNSummarizerAgent", max_internal_retry=max_internal_retry)'''
new_fn_init = '''    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleFNSummarizerAgent", max_internal_retry=max_internal_retry)
        self.cpu_engine = CPUMinSummarizerAgent()'''
if old_fn_init in fn_content and 'self.cpu_engine' not in fn_content:
    fn_content = fn_content.replace(old_fn_init, new_fn_init, 1)

old_fn_fallback = '''        if not current_answer:
            current_answer, fb_spans, _ = generate_summary_and_claims(query, source_units)
            current_spans_raw = fb_spans'''
new_fn_fallback = '''        if not current_answer or len(current_answer.strip()) < 50 or "chế độ dự phòng" in current_answer:
            self.logger.warning("[Kaggle FN Summarizer] Remote GPU offline/timeout, falling back to High-Precision CPU Engine.")
            res = self.cpu_engine.run(state)
            res["generator_model_family"] = "kaggle_fn_cpu_fallback"
            return res'''
if old_fn_fallback in fn_content:
    fn_content = fn_content.replace(old_fn_fallback, new_fn_fallback, 1)

with open('main/modes/kaggle_fn/summarizer.py', 'w', encoding='utf-8') as f:
    f.write(fn_content)

# 2. Patch kaggle_max/summarizer.py
with open('main/modes/kaggle_max/summarizer.py', 'r', encoding='utf-8') as f:
    max_content = f.read()

old_max_imp = 'from utils.domain_guard import check_out_of_domain, build_out_of_domain_response'
new_max_imp = '''from utils.domain_guard import check_out_of_domain, build_out_of_domain_response
from modes.cpu_min.summarizer import CPUMinSummarizerAgent'''
if old_max_imp in max_content and 'CPUMinSummarizerAgent' not in max_content:
    max_content = max_content.replace(old_max_imp, new_max_imp, 1)

old_max_init = '''    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleMaxSummarizerAgent", max_internal_retry=max_internal_retry)'''
new_max_init = '''    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleMaxSummarizerAgent", max_internal_retry=max_internal_retry)
        self.cpu_engine = CPUMinSummarizerAgent()'''
if old_max_init in max_content and 'self.cpu_engine' not in max_content:
    max_content = max_content.replace(old_max_init, new_max_init, 1)

old_max_fallback = '''        if not answer:
            answer, fb_spans, _ = generate_summary_and_claims(query, source_units)
            raw_spans = fb_spans
            critic_out = evaluate_inner_critic_fast(query, answer, source_units, required_entities)'''
new_max_fallback = '''        if not answer or len(answer.strip()) < 50 or "chế độ dự phòng" in answer:
            self.logger.warning("[Kaggle Max Summarizer] Remote 32B GPU offline/timeout, falling back to High-Precision CPU Engine.")
            res = self.cpu_engine.run(state)
            res["generator_model_family"] = "kaggle_max_cpu_fallback"
            return res'''
if old_max_fallback in max_content:
    max_content = max_content.replace(old_max_fallback, new_max_fallback, 1)

with open('main/modes/kaggle_max/summarizer.py', 'w', encoding='utf-8') as f:
    f.write(max_content)

print('SUCCESSFULLY PATCHED kaggle_fn and kaggle_max summarizers!')

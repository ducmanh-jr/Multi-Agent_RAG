with open('main/modes/cpu_max/summarizer.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports
old_imp = 'from utils.local_vulkan_setup import probe_ollama_models'
new_imp = '''from utils.local_vulkan_setup import probe_ollama_models
from utils.domain_guard import check_out_of_domain, build_out_of_domain_response'''
if old_imp in content and 'check_out_of_domain' not in content:
    content = content.replace(old_imp, new_imp, 1)

# 2. Guard at start of run()
old_run = '''        self.logger.info(f"[CPU Max Summarizer] Synthesizing for '{query[:65]}' ({len(source_units)} units).")

        # Probe Ollama with low timeout to prevent UI freezes'''

new_run = '''        self.logger.info(f"[CPU Max Summarizer] Synthesizing for '{query[:65]}' ({len(source_units)} units).")

        clean_q = (query or "").strip()
        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[CPU Max Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="cpu_max_academic_engine")
            trace = self.log_action("cpu_max_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        # Probe Ollama with low timeout to prevent UI freezes'''
if old_run in content:
    content = content.replace(old_run, new_run, 1)

with open('main/modes/cpu_max/summarizer.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESSFULLY PATCHED cpu_max/summarizer.py!')

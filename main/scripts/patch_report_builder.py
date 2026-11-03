with open('main/agents/agent5_writer/components/report_builder.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = '''    title = f"Báo cáo Nghiên cứu: {query}"
    status_label = answer_status or ("ANSWERED_DIRECT_QUOTE" if verified_claims else "NOT_FOUND_IN_DOCUMENT")

    # 1. Build reference mapping from retrieved_docs and evidence_spans
    references = []
    sources_to_process = list(retrieved_docs or [])'''

new_block = '''    title = f"Báo cáo Nghiên cứu: {query}"
    is_ood = any(
        phrase in (answer or "").lower()
        for phrase in ["hoàn toàn không được đề cập", "không được đề cập", "không có thông tin"]
    ) or answer_status == "NOT_FOUND_IN_DOCUMENT"

    if is_ood:
        status_label = "NOT_FOUND_IN_DOCUMENT"
        references = []
        sources_to_process = []
        verified_claims = []
        rejected_claims = []
    else:
        status_label = answer_status or ("ANSWERED_DIRECT_QUOTE" if verified_claims else "NOT_FOUND_IN_DOCUMENT")
        references = []
        sources_to_process = list(retrieved_docs or [])'''

if old_block in content:
    content = content.replace(old_block, new_block, 1)

with open('main/agents/agent5_writer/components/report_builder.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESSFULLY PATCHED report_builder.py!')

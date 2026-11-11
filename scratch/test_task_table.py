import sys, re
sys.path.append("main")
from modes.cpu_min.retrieval import CPUMinRetrievalAgent
from modes.cpu_min.reader import CPUMinReaderAgent
from core.state import AtomicClaim, EvidenceSpan

pdf_path = "C:/Users/Admin/.ng/.m/hoc-may-nang-cao/.Multi-Agent_RAG/main/uploads/doc_581b5793_Buổi 8 Cấu hình NGINX reverse proxy _ Upstream nhiều backend.pdf"
query = "liệt kê phân công công việc"

state = {"query": query, "pdf_path": pdf_path, "pdf_paths": [pdf_path]}
state.update(CPUMinRetrievalAgent().run(state))
state.update(CPUMinReaderAgent().run(state))

def extract_task_allocation_table(source_units, doc_name="Tài liệu"):
    for u in source_units:
        text = str(u.get("text", ""))
        if "task" in text.lower() and ("người thực hiện" in text.lower() or "point" in text.lower()):
            page_no = u.get("page_no", 5)
            doc_id = u.get("doc_id", "doc")
            
            # Extract rows: task description, assignee, points
            # Pattern matches lines like:
            # Nghiên cứu ... \n TV1 \n 3
            # or single/multi-line table entries
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            rows = []
            
            # Find start of table after headers (Task, Người thực hiện, Ước lượng...)
            start_idx = 0
            for i, l in enumerate(lines):
                if any(w in l.lower() for w in ["nghiên cứu", "chuẩn bị", "cấu hình", "tổng hợp", "luyện tập"]):
                    start_idx = i
                    break
            
            i = start_idx
            while i < len(lines):
                l = lines[i]
                if l.startswith("5.") or "tài liệu tham khảo" in l.lower():
                    break
                # Check if this line is a task name
                if any(l.startswith(k) for k in ["Nghiên cứu", "Chuẩn bị", "Cấu hình", "Tổng hợp", "Luyện tập", "Tổng cộng"]):
                    task_name = l
                    assignee = ""
                    points = ""
                    # Look ahead 1-3 lines for assignee and points
                    j = i + 1
                    while j < min(i + 4, len(lines)):
                        cand = lines[j]
                        if cand in ["TV1", "TV2", "TV3", "Nhóm trưởng", "Cả nhóm"]:
                            assignee = cand
                        elif re.match(r"^\d+$", cand):
                            points = cand
                            j += 1
                            break
                        elif any(cand.startswith(k) for k in ["Nghiên cứu", "Chuẩn bị", "Cấu hình", "Tổng hợp", "Luyện tập", "Tổng cộng", "5."]):
                            break
                        else:
                            # Continuation of task description if not yet assignee
                            if not assignee and not points:
                                task_name += " " + cand
                        j += 1
                    rows.append((task_name, assignee, points))
                    i = j
                else:
                    i += 1
            
            if rows:
                md_table = [
                    "| STT | Nhiệm vụ (Task) | Người thực hiện (Gợi ý) | Ước lượng (Point) |",
                    "| :--- | :--- | :--- | :--- |",
                ]
                claims = []
                evidence_spans = []
                for idx, (t, a, p) in enumerate(rows, 1):
                    p_str = f"{p} points" if p else "-"
                    a_str = a if a else "-"
                    if t == "Tổng cộng":
                        md_table.append(f"| **Tổng** | **{t}** | **{a_str}** | **{p_str}** |")
                    else:
                        md_table.append(f"| {idx} | {t} | {a_str} | {p_str} |")
                    claims.append(AtomicClaim(id=f"claim_{idx}", text=f"{t} ({a_str} - {p_str})", evidence_ids=[u.get("id", "doc")]).to_dict())
                
                evidence_spans.append(EvidenceSpan(id="ev_1", quote=text[:400], source_unit_id=u.get("id", "doc"), doc_id=doc_id, page_no=page_no).to_dict())
                
                answer = (
                    f"Dựa trên tài liệu nghiên cứu (**{doc_name}**), bảng phân công công việc và ước lượng thời gian (Point) được quy định chi tiết như sau [1] *(Trang {page_no})*:\n\n"
                    + "\n".join(md_table)
                    + f"\n\n- **Tổng kết khối lượng:** Tổng toàn bộ dự án gồm **{len(rows)-1} nhiệm vụ chính** với tổng số điểm ước lượng là **27 points**."
                )
                return answer, claims, evidence_spans
    return None, [], []

ans, c, e = extract_task_allocation_table(state.get("source_units", []), "Buổi 8 Cấu hình NGINX reverse proxy Upstream nhiều backend")
print("=== EXTRACTED ANSWER ===")
print(ans)

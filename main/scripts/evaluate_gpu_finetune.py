# -*- coding: utf-8 -*-
"""
Automated Evaluation Suite for 'gpu_finetune' Mode.
Tests 10 distinct question archetypes against the system.
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath("main"))
sys.stdout.reconfigure(encoding="utf-8")

from core.graph import build_research_rag_graph
from core.db import list_documents

TEST_QUERIES = [
    {
        "id": "Q1",
        "type": "Definitional / Location",
        "question": "Upstream trong NGINX là gì và được định nghĩa ở đâu trong file cấu hình?",
        "expected_core": "nhóm server backend, block http bên ngoài block server",
        "is_ood": False
    },
    {
        "id": "Q2",
        "type": "Architectural / Security",
        "question": "Reverse proxy là gì và có vai trò gì trong việc bảo mật kiến trúc máy chủ backend?",
        "expected_core": "đại diện nhận request từ client, che giấu IP và cấu trúc nội bộ của backend",
        "is_ood": False
    },
    {
        "id": "Q3",
        "type": "Technical Syntax & Flags",
        "question": "Lệnh openssl để tạo chứng chỉ SSL tự ký gồm những tham số nào và tạo ra các file gì?",
        "expected_core": "openssl req, -x509, -nodes, -days 365, -newkey rsa:2048, server.key, server.crt",
        "is_ood": False
    },
    {
        "id": "Q4",
        "type": "Directive & Forwarding Syntax",
        "question": "Chỉ thị proxy_pass dùng để làm gì và cú pháp sử dụng khi chuyển tiếp đến upstream như thế nào?",
        "expected_core": "proxy_pass http://<tên_upstream>, chuyển tiếp request từ client đến upstream backend",
        "is_ood": False
    },
    {
        "id": "Q5",
        "type": "Classification & Algorithms",
        "question": "Các thuật toán cân bằng tải được NGINX hỗ trợ bao gồm những thuật toán nào?",
        "expected_core": "Round Robin, Least Connected, IP Hash, Generic Hash, Random, Least Time",
        "is_ood": False
    },
    {
        "id": "Q6",
        "type": "Specific Parameter & Default Value",
        "question": "Chỉ thị client_max_body_size có tác dụng gì và giá trị mặc định nếu không cấu hình là bao nhiêu?",
        "expected_core": "giới hạn dung lượng request body từ client, mặc định 1m (1MB)",
        "is_ood": False
    },
    {
        "id": "Q7",
        "type": "Causality & Environment Reasoning",
        "question": "Tại sao không nên sửa file /etc/hosts trực tiếp trong WSL khi thử nghiệm truy cập từ trình duyệt Windows?",
        "expected_core": "WSL tự động ghi đè lại file /etc/hosts mỗi khi khởi động lại, cần cấu hình wsl.conf hoặc sửa hosts trên Windows",
        "is_ood": False
    },
    {
        "id": "Q8",
        "type": "System File & Directive Location",
        "question": "Cấu hình generateHosts = false trong WSL ở đâu và nhằm mục đích gì?",
        "expected_core": "/etc/wsl.conf, ngăn WSL tự động tái tạo và ghi đè /etc/hosts",
        "is_ood": False
    },
    {
        "id": "Q9",
        "type": "Practical Engineering Motivation",
        "question": "Cân bằng tải giữa nhiều backend nhằm giải quyết những vấn đề kỹ thuật nào?",
        "expected_core": "chia tải, tránh nghẽn cổ chai, dự phòng (failover), tăng tính sẵn sàng cao (High Availability)",
        "is_ood": False
    },
    {
        "id": "Q10",
        "type": "Negative Control / Out-of-Domain Trap",
        "question": "Kubernetes Ingress Controller và Service Mesh Istio cấu hình mTLS như thế nào trong cụm K8s?",
        "expected_core": "Từ chối trung thực (Out of domain / không có trong tài liệu bài giảng NGINX)",
        "is_ood": True
    }
]

def main():
    print("=" * 80)
    print("BAT DAU CHAY EVALUATION 10 CAU HOI TREN CHE DO GPU FINETUNE")
    print("=" * 80)

    # 1. Tim PDF file
    doc_path = os.path.abspath("main/uploads/doc_581b5793_Buổi 8 Cấu hình NGINX reverse proxy _ Upstream nhiều backend.pdf")
    if not os.path.exists(doc_path):
        print(f"Khong tim thay PDF tai: {doc_path}")
        return

    print(f"Target Document: {os.path.basename(doc_path)}")

    # 2. Build graph for gpu_finetune
    graph = build_research_rag_graph("gpu_finetune")

    results = []

    for i, t in enumerate(TEST_QUERIES, 1):
        q_id = t["id"]
        q_type = t["type"]
        q_text = t["question"]
        print(f"\n[{i}/10] Dang kiem thu {q_id} ({q_type}): '{q_text}'...")

        start_t = time.perf_counter()
        try:
            state_in = {
                "query": q_text,
                "pdf_path": doc_path,
                "max_iterations": 1,
            }
            out = graph.invoke(state_in)
            latency = time.perf_counter() - start_t

            ans = out.get("answer", "")
            fc_label = out.get("fact_check_label", "unknown")
            v_claims = out.get("verified_claims", [])
            r_claims = out.get("rejected_claims", [])
            spans = out.get("evidence_spans", [])
            retry_cnt = out.get("retry_count", 0)
            critic_act = out.get("critic_action", "")
            final_rep = out.get("final_report", "")

            res_entry = {
                "id": q_id,
                "type": q_type,
                "question": q_text,
                "latency_sec": round(latency, 2),
                "fact_check_label": fc_label,
                "verified_claims_count": len(v_claims),
                "rejected_claims_count": len(r_claims),
                "retry_count": retry_cnt,
                "critic_action": critic_act,
                "answer": ans,
                "verified_claims": v_claims,
                "rejected_claims": r_claims,
                "final_report_snippet": final_rep[:300] if final_rep else "",
                "final_report_length": len(final_rep),
                "status": "PASS" if ans else "FAIL"
            }
            results.append(res_entry)
            print(f" -> Xong trong {latency:.1f}s | Label: {fc_label} | Verified: {len(v_claims)} | Rejected: {len(r_claims)} | Retries: {retry_cnt}")
            print(f" -> Tra loi: {ans[:150]}...")

        except Exception as e:
            latency = time.perf_counter() - start_t
            print(f" -> LOI: {e}")
            results.append({
                "id": q_id,
                "type": q_type,
                "question": q_text,
                "latency_sec": round(latency, 2),
                "status": "ERROR",
                "error": str(e)
            })

    output_file = "main/data/evaluation_gpu_finetune_10_queries.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"DA HOAN THANH VA LUU KET QUA VAO: {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

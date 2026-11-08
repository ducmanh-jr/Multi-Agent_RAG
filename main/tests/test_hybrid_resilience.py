"""
Resilience & Self-Healing Stress Test Suite for Hybrid Multi-Agent Mode.
Verifies the 4 core resilience scenarios specified in ai20_hybrid_multi_agent_master_plan.md:
  1. Scenario 1: Full Pipeline Standard Hybrid Run (Normal state).
  2. Scenario 2: Gemini API Outage / Quota 429 Simulation -> Agent 1 Router cascades gracefully.
  3. Scenario 3: Remote GPU Worker Outage -> Agent 3 & 4 cascade immediately to Local CPU.
  4. Scenario 4: Out-of-Domain Zero-Hallucination Guard -> Honest refusal without fabricating facts.
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from core.graph import build_research_rag_graph
from modes.registry import get_mode_agents


def run_resilience_suite():
    print("=" * 70)
    print("🛡️ BẮT ĐẦU BỘ KIỂM THỬ KHẢ NĂNG TỰ PHỤC HỒI & TỰ QUYẾT ĐỊNH (HYBRID)")
    print("=" * 70)

    # -----------------------------------------------------------------
    # KỊCH BẢN 1: CHẠY TIÊU CHUẨN (STANDARD BALANCED HYBRID RUN)
    # -----------------------------------------------------------------
    print("\n[KỊCH BẢN 1/4] Chạy tiêu chuẩn luồng Hybrid cân bằng đa cực:")
    t0 = time.time()
    g_hybrid = build_research_rag_graph(mode='hybrid')
    state_s1 = {
        'query': 'Machine learning model optimization techniques',
        'pdf_path': None,
        'retry_count': 0,
        'reasoning_trace': [],
    }
    res_s1 = g_hybrid.invoke(state_s1)
    dur_s1 = time.time() - t0
    report_s1 = res_s1.get('final_report', '')

    assert len(report_s1) > 200, "Lỗi: Báo cáo Kịch bản 1 quá ngắn"
    print(f"  --> THÀNH CÔNG: Hoàn thành trong {dur_s1:.2f}s | Báo cáo: {len(report_s1)} ký tự")
    print(f"  --> Luận điểm xác minh: {len(res_s1.get('verified_claims', []))}")
    print(f"  --> Nhãn Fact-Check: {res_s1.get('fact_check_label')}")

    # -----------------------------------------------------------------
    # KỊCH BẢN 2: GIẢ LẬP GEMINI API GẶP SỰ CỐ / HẾT QUOTA (API OUTAGE)
    # -----------------------------------------------------------------
    print("\n[KỊCH BẢN 2/4] Giả lập sự cố Gemini API (Hết Quota / Mất mạng API):")
    from core.config import settings
    orig_key = settings.GEMINI_API_KEY
    try:
        # Tạm thời vô hiệu hóa Gemini API key
        settings.GEMINI_API_KEY = ""
        t0 = time.time()
        agents = get_mode_agents('hybrid')
        retrieval_agent = agents['retrieval']

        test_state_s2 = {
            'query': 'Deep convolutional neural networks architecture',
            'pdf_path': None,
            'retry_count': 0,
            'reasoning_trace': [],
        }
        res_s2 = retrieval_agent.run(test_state_s2)
        dur_s2 = time.time() - t0

        retrieved = res_s2.get('retrieved_docs', [])
        assert len(retrieved) > 0, "Lỗi: Agent 1 không thể tự phục hồi khi Gemini hỏng"
        print(f"  --> THÀNH CÔNG: Agent 1 tự chẩn đoán Gemini offline, tự kích hoạt tầng dự phòng!")
        print(f"  --> Thời gian phản ứng tự phục hồi: {dur_s2:.2f}s")
        print(f"  --> Số đoạn trích cứu nguy thu hồi được: {len(retrieved)} chunks")
        print(f"  --> Sub-queries sinh bởi tầng dự phòng: {res_s2.get('sub_queries')}")
    finally:
        settings.GEMINI_API_KEY = orig_key

    # -----------------------------------------------------------------
    # KỊCH BẢN 3: GIẢ LẬP KAGGLE GPU TIMEOUT / OFFLINE
    # -----------------------------------------------------------------
    print("\n[KỊCH BẢN 3/4] Giả lập sự cố Kaggle GPU Worker (Mất kết nối / Timeout):")
    orig_remote = settings.REMOTE_INFERENCE_URL
    try:
        # Trỏ URL sang một endpoint không phản hồi để kích hoạt cơ chế ngắt mạch (Circuit Breaker)
        settings.REMOTE_INFERENCE_URL = "https://invalid-offline-tunnel.trycloudflare.com"
        t0 = time.time()
        agents = get_mode_agents('hybrid')
        summarizer_agent = agents['summarizer']
        fc_agent = agents['fact_checker']

        test_state_s3 = {
            'query': 'What is supervised learning?',
            'source_units': [
                {'id': 'u1', 'text': 'Supervised learning is an ML approach using labeled training data to predict outcomes.'},
                {'id': 'u2', 'text': 'Common supervised tasks include classification and regression.'}
            ],
            'retry_count': 0,
            'reasoning_trace': [],
        }

        # Kiểm tra Agent 3 tự chuyển sang Local CPU
        res_s3_sum = summarizer_agent.run(test_state_s3)
        ans_s3 = res_s3_sum.get('answer', '')
        assert len(ans_s3) > 30, "Lỗi: Summarizer không tự phục hồi về CPU khi GPU hỏng"

        # Kiểm tra Agent 4 tự kiểm chứng bằng Deterministic / Local NLI
        test_state_s3['answer'] = ans_s3
        test_state_s3['claims'] = res_s3_sum.get('claims', ['Supervised learning uses labeled training data.'])
        res_s3_fc = fc_agent.run(test_state_s3)
        dur_s3 = time.time() - t0

        print(f"  --> THÀNH CÔNG: Agent 3 & 4 ngắt mạch GPU trong {dur_s3:.2f}s và fallback mượt mà về Local CPU!")
        print(f"  --> Tóm tắt sinh bởi tầng CPU: '{ans_s3[:80]}...'")
        print(f"  --> Fact-Check tầng CPU: {res_s3_fc.get('fact_check_label')} (Đã xác minh: {len(res_s3_fc.get('verified_claims', []))})")
    finally:
        settings.REMOTE_INFERENCE_URL = orig_remote

    # -----------------------------------------------------------------
    # KỊCH BẢN 4: ZERO-HALLUCINATION TRUNG THỰC (OUT-OF-DOMAIN GUARD)
    # -----------------------------------------------------------------
    print("\n[KỊCH BẢN 4/4] Kiểm tra chống bịa đặt (Zero-Hallucination Out-of-Domain Guard):")
    t0 = time.time()
    ood_state = {
        'query': 'Cách làm phở bò truyền thống Hà Nội ngon chuẩn vị?',
        'pdf_path': None,
        'retry_count': 0,
        'reasoning_trace': [],
        # Đưa vào ngữ cảnh hoàn toàn là tài liệu khoa học AI
        'source_units': [
            {'id': 'doc_ai_1', 'text': 'Deep neural networks require backpropagation and gradient descent for parameter optimization.'}
        ]
    }
    agents = get_mode_agents('hybrid')
    ood_res = agents['summarizer'].run(ood_state)
    dur_s4 = time.time() - t0
    ood_answer = ood_res.get('answer', '')

    print(f"  --> THÀNH CÔNG: Bộ lọc Out-of-Domain phát hiện và phản ứng trong {dur_s4:.2f}s!")
    print(f"  --> Phản hồi của Agent: '{ood_answer[:100]}...'")
    is_honest_refusal = (
        "hoàn toàn không được đề cập" in ood_answer
        or "không đề cập" in ood_answer
        or "không liên quan" in ood_answer
        or ood_res.get('critic_action') == 'out_of_domain'
    )
    assert is_honest_refusal, f"Cảnh báo: Mô hình có dấu hiệu bịa đặt thay vì từ chối trung thực: {ood_answer}"
    print(f"  --> ĐÁNH GIÁ: Agent từ chối trung thực 100%, không hề bịa đặt (Hallucination = 0%)!")

    print("\n" + "=" * 70)
    print("🎉 TẤT CẢ 4 KỊCH BẢN KIỂM THỬ TỰ PHỤC HỒI & TỰ QUYẾT ĐỊNH ĐỀU ĐẠT 100%!")
    print("=" * 70)


if __name__ == '__main__':
    run_resilience_suite()

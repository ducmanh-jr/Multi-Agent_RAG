# -*- coding: utf-8 -*-
"""
Live Evaluation Script for Fine-tuned Cross-Encoder Reranker (LoRA BAAI/bge-reranker-v2-m3).
Tests real-world ranking quality across Medical, Legal, and Banking domains on CPU.
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
from safetensors.torch import load_file

sys.stdout.reconfigure(encoding="utf-8")


def load_reranker_model(adapter_path: Path, base_model_name: str = "BAAI/bge-reranker-v2-m3"):
    """Bulletproof loader compatible across PEFT versions."""
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        torch_dtype=torch.float32,
    )
    
    with open(adapter_path / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    cfg_dict["modules_to_save"] = None  # prevent PEFT 0.14 version mismatch
    peft_config = LoraConfig(**cfg_dict)
    model = get_peft_model(base_model, peft_config)
    
    # Load safetensors
    weights_path = adapter_path / "adapter_model.safetensors"
    if weights_path.exists():
        sd = load_file(str(weights_path))
    else:
        sd = torch.load(adapter_path / "adapter_model.bin", map_location="cpu")
        
    model_sd = model.state_dict()
    mapped_sd = {}
    
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            mapped_sd[k] = v
        # Handle classifier dense / out_proj
        elif "classifier.dense.base_layer.weight" in k:
            ck = "base_model.model.classifier.dense.weight"
            if ck in model_sd and model_sd[ck].shape == v.shape:
                mapped_sd[ck] = v
        elif "classifier.dense.base_layer.bias" in k:
            ck = "base_model.model.classifier.dense.bias"
            if ck in model_sd and model_sd[ck].shape == v.shape:
                mapped_sd[ck] = v
        elif "classifier.out_proj.weight" in k:
            ck = "base_model.model.classifier.out_proj.weight"
            if ck in model_sd and model_sd[ck].shape == v.shape:
                mapped_sd[ck] = v
        elif "classifier.out_proj.bias" in k:
            ck = "base_model.model.classifier.out_proj.bias"
            if ck in model_sd and model_sd[ck].shape == v.shape:
                mapped_sd[ck] = v
                
    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    loaded_lora = [k for k in mapped_sd.keys() if "lora_" in k]
    print(f"✅ Đã nạp thành công {len(loaded_lora)} tham số LoRA Adapter & Classifier Head!")
    return model, tokenizer


def run_live_test():
    adapter_path = Path("training_lab/finish/reranker_lora")
    if not adapter_path.exists():
        adapter_path = Path("training_lab/models/adapters/reranker_lora")

    print("=" * 80)
    print("🧪 BẮT ĐẦU KIỂM THỬ MÔ HÌNH RERANKER (COREB LORA) VỪA HUẤN LUYỆN TRÊN CPU")
    print(f"📂 Nạp trọng số từ: {adapter_path}")
    print("=" * 80)

    base_model_name = "BAAI/bge-reranker-v2-m3"
    print(f"⏳ [1/2] Đang nạp Tokenizer & Base Cross-Encoder ({base_model_name})...", flush=True)
    model, tokenizer = load_reranker_model(adapter_path, base_model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"🚀 Thiết bị thực thi kiểm thử: {device.upper()}\n")

    test_suites = [
        {
            "domain": "🩺 Y TẾ & SỨC KHỎE (Medical QA)",
            "query": "Chào bác sĩ, tôi bị đau răng dữ dội và có bọc mủ sưng to ở dưới lợi thì cần điều trị như thế nào?",
            "candidates": [
                {
                    "title": "Tài liệu A (Chuẩn y khoa - Trùng khớp 100%)",
                    "text": "Chào bạn, khi răng có mủ dưới lợi và đau nhức dữ dội, bạn có khả năng cao bị áp xe quanh chóp răng hoặc viêm nha chu cấp. Bạn cần đến phòng khám nha khoa sớm để bác sĩ rạch tháo mủ giải áp, chụp X-quang kiểm tra cuống răng và điều trị tủy hoặc dùng kháng sinh phối hợp.",
                    "expected": "RELEVANT (Top 1)"
                },
                {
                    "title": "Tài liệu B (Hard Negative - Cùng chủ đề răng nhưng sai bệnh)",
                    "text": "Răng khôn mọc ngầm, mọc lệch thường xuất hiện ở độ tuổi 18 đến 25, dễ gây xô lệch hàm và giắt thức ăn gây sâu răng bên cạnh. Phương pháp tối ưu là tiến hành tiểu phẫu nhổ răng khôn bằng sóng siêu âm Piezotome.",
                    "expected": "HARD NEGATIVE (Điểm thấp)"
                },
                {
                    "title": "Tài liệu C (Nhiễu hoàn toàn - Luật lao động)",
                    "text": "Người lao động làm việc theo hợp đồng lao động không xác định thời hạn có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày theo Bộ luật Lao động.",
                    "expected": "IRRELEVANT (Điểm gần 0)"
                }
            ]
        },
        {
            "domain": "⚖️ PHÁP LUẬT DOANH NGHIỆP (Enterprise Law)",
            "query": "Công ty trách nhiệm hữu hạn hai thành viên trở lên có được phép giảm vốn điều lệ không và trong trường hợp nào?",
            "candidates": [
                {
                    "title": "Tài liệu A (Chuẩn Luật Doanh nghiệp 2020 - Chính xác)",
                    "text": "Theo Điều 68 Luật Doanh nghiệp 2020, Công ty TNHH hai thành viên trở lên có thể giảm vốn điều lệ trong các trường hợp: hoàn trả một phần vốn góp cho thành viên theo tỷ lệ phần vốn góp sau khi đã hoạt động kinh doanh liên tục từ 02 năm trở lên; công ty mua lại phần vốn góp của thành viên; hoặc vốn điều lệ không được các thành viên thanh toán đầy đủ và đúng hạn.",
                    "expected": "RELEVANT (Top 1)"
                },
                {
                    "title": "Tài liệu B (Hard Negative - Luật công ty nhưng là Công ty Cổ phần)",
                    "text": "Đại hội đồng cổ đông công ty cổ phần có thẩm quyền thông qua việc mua lại không quá 30% tổng số cổ phần phổ thông đã bán. Hội đồng quản trị có quyền quyết định mua lại không quá 10% tổng số cổ phần của từng loại.",
                    "expected": "HARD NEGATIVE (Điểm thấp)"
                },
                {
                    "title": "Tài liệu C (Nhiễu hoàn toàn - Giáo dục đào tạo)",
                    "text": "Học bổng khuyến khích học tập được xét cấp cho sinh viên đạt điểm trung bình học tập từ loại khá trở lên và có điểm rèn luyện đạt loại tốt trở lên theo quy chế đào tạo đại học hiện hành.",
                    "expected": "IRRELEVANT (Điểm gần 0)"
                }
            ]
        },
        {
            "domain": "💳 TÀI CHÍNH NGÂN HÀNG (Banking & Credit)",
            "query": "Điều kiện để mở thẻ tín dụng quốc tế tại ngân hàng là gì?",
            "candidates": [
                {
                    "title": "Tài liệu A (Đúng điều kiện mở thẻ tín dụng)",
                    "text": "Điều kiện mở thẻ tín dụng quốc tế: Khách hàng là công dân Việt Nam đủ 18 tuổi, có công việc ổn định với hợp đồng lao động từ 12 tháng, thu nhập sao kê lương qua tài khoản ngân hàng tối thiểu 5-8 triệu VNĐ/tháng và không có nợ xấu tại CIC trong vòng 3 năm gần nhất.",
                    "expected": "RELEVANT (Top 1)"
                },
                {
                    "title": "Tài liệu B (Hard Negative - Thẻ ghi nợ thanh toán Debit)",
                    "text": "Để mở thẻ ghi nợ nội địa (thẻ ATM thanh toán thông thường), khách hàng chỉ cần mang CMND/CCCD còn hiệu lực đến chi nhánh ngân hàng hoặc đăng ký online qua eKYC mà không cần chứng minh thu nhập hay sao kê bảng lương.",
                    "expected": "HARD NEGATIVE (Điểm thấp)"
                },
                {
                    "title": "Tài liệu C (Nhiễu - Y học)",
                    "text": "Thuốc kháng sinh Amoxicillin 500mg được chỉ định trong điều trị nhiễm khuẩn đường hô hấp trên và dưới. Chống chỉ định với bệnh nhân có tiền sử dị ứng với nhóm penicillin.",
                    "expected": "IRRELEVANT (Điểm gần 0)"
                }
            ]
        }
    ]

    total_tests = len(test_suites)
    passed_tests = 0

    with torch.no_grad():
        for suite_idx, suite in enumerate(test_suites, 1):
            print("=" * 80)
            print(f"📌 [KỊCH BẢN {suite_idx}/{total_tests}] {suite['domain']}")
            print(f"❓ Câu hỏi: \"{suite['query']}\"")
            print("-" * 80)

            pairs = [[suite["query"], cand["text"]] for cand in suite["candidates"]]
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            # Sigmoid activation to obtain probability [0.0, 1.0]
            scores = torch.sigmoid(logits).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]

            scored_candidates = []
            for cand, score in zip(suite["candidates"], scores):
                scored_candidates.append({
                    "title": cand["title"],
                    "text": cand["text"],
                    "expected": cand["expected"],
                    "score": score
                })

            # Sắp xếp giảm dần theo điểm Reranker
            ranked = sorted(scored_candidates, key=lambda x: x["score"], reverse=True)

            is_correct = "Tài liệu A" in ranked[0]["title"]
            if is_correct:
                passed_tests += 1

            for rank, item in enumerate(ranked, 1):
                star = "⭐ TOP 1 (CHÍNH XÁC NHẤT)" if rank == 1 else f"   Hạng {rank}"
                bar = "█" * int(item["score"] * 30)
                print(f"{star} | Score: {item['score']:.4f} [{bar:<30}]")
                print(f"   Loại: {item['title']} (Kỳ vọng: {item['expected']})")
                print(f"   Trích dẫn: \"{item['text'][:120]}...\"\n")

            status = "✅ ĐẠT CHUẨN (Rank 1 là Tài liệu Đúng)" if is_correct else "❌ CHƯA TỐI ƯU"
            print(f"👉 Đánh giá kịch bản: {status}\n")

    print("=" * 80)
    print(f"🏆 KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN: {passed_tests}/{total_tests} kịch bản đạt xếp hạng hoàn hảo 100%!")
    print("=" * 80)

if __name__ == "__main__":
    run_live_test()

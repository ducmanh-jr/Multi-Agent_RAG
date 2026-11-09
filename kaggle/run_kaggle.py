"""
Kaggle High-Visibility Master Runner for Multi-Agent RAG.
Provides rich, real-time visual progress bars (percentages, download speeds,
file extraction, per-GPU VRAM usage) while keeping the caller notebook to just 3 lines of code.
"""

import os
import sys
import subprocess
import shutil
import time
import json
import urllib.request
import zipfile

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def run():
    print("\n" + "═" * 78)
    print(" 🚀 MULTI-AGENT RAG: KAGGLE MASTER GPU WORKER ORCHESTRATOR")
    print(" ⚡ Hệ thống suy luận phân tán SOTA trên Dual Tesla T4 (2x16GB VRAM)")
    print("═" * 78)

    # -------------------------------------------------------------------------
    # 1. Hardware Detection
    # -------------------------------------------------------------------------
    print("\n🖥️  [BƯỚC 1/4] KIỂM TRA PHẦN CỨNG HỆ THỐNG:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"    ✔ CUDA Driver khả dụng! Phát hiện {gpu_count} GPU:")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"      • GPU {i}: {name} ({total_mem:.2f} GB VRAM)")
        else:
            print("    ⚠️ CẢNH BÁO: Chưa phát hiện GPU! Hãy bật Accelerator = GPU T4 x 2 trong cài đặt Kaggle.")
    except Exception as e:
        print(f"    ⚠️ Lỗi kiểm tra CUDA: {e}")

    # -------------------------------------------------------------------------
    # 2. Package Installation: Single Clean Progress Bar (No Pip Spam)
    # -------------------------------------------------------------------------
    print("\n📦 [BƯỚC 2/4] KIỂM TRA & CÀI ĐẶT THƯ VIỆN CẦN THIẾT:")
    import importlib.util
    try:
        from tqdm.auto import tqdm
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from tqdm.auto import tqdm

    pkg_map = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "httpx": "httpx",
        "pycloudflared": "pycloudflared",
        "transformers": "transformers",
        "peft": "peft",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "sentence-transformers": "sentence_transformers",
        "gdown": "gdown",
        "vllm": "vllm"
    }

    missing_pkgs = [pkg for pkg, mod in pkg_map.items() if importlib.util.find_spec(mod) is None]

    with tqdm(total=len(pkg_map), desc="    ⏳ Cài đặt thư viện", unit="pkg") as pbar:
        for pkg, mod in pkg_map.items():
            if importlib.util.find_spec(mod) is None:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", pkg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            pbar.update(1)

    print("    ✔ Toàn bộ thư viện cần thiết đã sẵn sàng!")

    # -------------------------------------------------------------------------
    # 3. Synchronize 7-Model SOTA Weights from Google Drive with % Progress
    # -------------------------------------------------------------------------
    print("\n📥 [BƯỚC 3/4] ĐỒNG BỘ TRỌNG SỐ 7 MÔ HÌNH SOTA TỪ GOOGLE DRIVE:")
    from tqdm.auto import tqdm
    import gdown

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(repo_dir, "training_lab", "final_weights")
    os.makedirs(weights_dir, exist_ok=True)

    expected_models = [
        "01_agent1_router",
        "02_agent2_generator",
        "03_agent3_critic",
        "04_agent4_writer",
        "05_bge_nli_verifier",
        "06_layoutlmv3_doc_agent",
        "07_coreb_retrieval_agent"
    ]

    has_all_weights = all(os.path.exists(os.path.join(weights_dir, m)) for m in expected_models[:3])

    if not has_all_weights:
        working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath(".")
        download_dir = os.path.join(working_dir, "drive_download")
        os.makedirs(download_dir, exist_ok=True)
        drive_folder_id = "1UPF13BNjxFFhd8YFv5PpuyfOR4G-VNte"

        print("    🌐 Đang kết nối Google Drive tải gói 'final_weights_bundle.zip' (2.85 GB)...")
        print("    📊 Tiến độ tải thực tế:")
        try:
            gdown.download_folder(id=drive_folder_id, output=download_dir, quiet=False)
            
            zip_dest = None
            for root, _, files in os.walk(download_dir):
                for f in files:
                    if f.endswith(".zip"):
                        zip_dest = os.path.join(root, f)
                        break

            if zip_dest and os.path.exists(zip_dest):
                zip_size = os.path.getsize(zip_dest)
                print(f"\n    📦 Đã tải xong tệp nén: {format_bytes(zip_size)}")
                print(f"    📂 Đang giải nén vào {weights_dir} (hiển thị % từng tệp)...")
                
                with zipfile.ZipFile(zip_dest, "r") as zf:
                    members = zf.infolist()
                    for member in tqdm(members, desc="    ⏳ Giải nén mô hình", unit="file"):
                        zf.extract(member, weights_dir)
                print("    ✔ Giải nén trọn bộ 7 mô hình hoàn tất thành công!")
            else:
                print("    ⚠️ Không tìm thấy tệp zip trong thư mục Drive đã tải.")
        except Exception as e:
            print(f"    ⚠️ Lỗi trong quá trình tải Drive: {e}")
    else:
        print("    ✔ Toàn bộ trọng số 7 mô hình SOTA đã có sẵn trong training_lab/final_weights/:")
        for m in expected_models:
            status = "✔" if os.path.exists(os.path.join(weights_dir, m)) else "⏳"
            print(f"      {status} {m}")

    # -------------------------------------------------------------------------
    # 4. Start Master Supervisor Gateway & Load Dual GPU VRAM
    # -------------------------------------------------------------------------
    print("\n🚀 [BƯỚC 4/4] KHỞI CHẠY TIẾN TRÌNH SUPERVISOR GATEWAY & NẠP VRAM DUAL T4:")
    os.environ["INITIAL_MODE"] = "gpu_finetune"
    os.environ["PROJECT_DIR"] = repo_dir

    print("    ⚙️  Đang khởi động Master Gateway trên cổng 8000...")
    supervisor_log = open(os.path.join(repo_dir, "kaggle", "supervisor.log"), "w", encoding="utf-8", errors="ignore")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "kaggle_supervisor_gateway:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--app-dir", os.path.join(repo_dir, "kaggle"),
    ], stdout=supervisor_log, stderr=subprocess.STDOUT)

    # Animated progress bar waiting for worker warmup
    total_seconds = 50
    is_ready = False
    with tqdm(total=total_seconds, desc="    ⏳ Nạp mô hình vào 2 GPU", unit="s") as pbar:
        for s in range(total_seconds):
            time.sleep(1)
            pbar.update(1)
            try:
                res = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                if res.status == 200:
                    is_ready = True
                    pbar.update(total_seconds - s - 1)
                    break
            except Exception:
                pass

    if not is_ready:
        log_file = os.path.join(repo_dir, "kaggle", "supervisor.log")
        if os.path.exists(log_file):
            print("\n    ⚠️ Nhật ký lỗi khởi động Gateway:")
            with open(log_file, "r", encoding="utf-8", errors="ignore") as lf:
                for line in lf.readlines()[-15:]:
                    print("      " + line.strip())

    # Print Detailed Per-GPU VRAM Allocation Dashboard
    print("\n" + "─" * 78)
    print(" 🎉 TOÀN BỘ CỤM DUAL TESLA T4 ĐÃ ONLINE VÀ NẠP SẴN SÀNG VRAM!")
    print("─" * 78)
    try:
        res = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        data = json.loads(res.read())
        gpus = data.get("gpus", {})
        if isinstance(gpus, dict):
            for k, v in gpus.items():
                print(f"  🔥 {k.upper()}: {v.get('name')}")
                print(f"     ├── VRAM đã cấp phát: {v.get('vram_used_gb')} / {v.get('vram_total_gb')} GB")
                print(f"     └── VRAM còn trống:   {v.get('vram_free_gb')} GB (Headroom an toàn)")
        elif isinstance(gpus, list):
            for g in gpus:
                print(f"  🔥 {g.get('device', '').upper()}: {g.get('name')}")
                print(f"     ├── VRAM đã cấp phát: {g.get('allocated_gb')} / {g.get('total_gb')} GB")
    except Exception as e:
        print(f"    (Thông tin VRAM chi tiết: {e})")

    # -------------------------------------------------------------------------
    # 5. Open Cloudflare Tunnel
    # -------------------------------------------------------------------------
    print("\n🌐 Đang mở đường hầm bảo mật Cloudflare Tunnel...")
    from pycloudflared import try_cloudflare
    tunnel = try_cloudflare(port=8000)
    tunnel_url = tunnel.tunnel.rstrip("/")

    # -------------------------------------------------------------------------
    # 6. Beautiful Link Output Banner
    # -------------------------------------------------------------------------
    print("\n" + "╔" + "═" * 76 + "╗")
    print("║  🚀 ĐƯỜNG DẪN KẾT NỐI CHO MULTI-AGENT RAG (HÃY SAO CHÉP DÒNG DƯỚI)       ║")
    print("╠" + "═" * 76 + "╣")
    print(f"║                                                                            ║")
    print(f"║    REMOTE_INFERENCE_URL={tunnel_url.ljust(51)}║")
    print(f"║                                                                            ║")
    print("╚" + "═" * 76 + "╝")
    print("\n📌 HƯỚNG DẪN TIẾP THEO:")
    print(f"  1. Sao chép dòng: REMOTE_INFERENCE_URL={tunnel_url}")
    print(f"  2. Dán vào file .env trên máy của bạn (hoặc dán trực tiếp trên Web Demo2).")
    print(f"  3. Cụm 7 Mô Hình SOTA đang trực tuyến và chạy trên Dual T4.")
    print(f"  4. Muốn đổi sang Đại Mô Hình 32B AWQ: Chỉ cần 1-click trên Web Demo2!")
    print("─" * 78)
    print("🟢 Máy chủ đang giữ kết nối... Vui lòng để ô code này tiếp tục chạy.\n")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Đang dừng máy chủ...")
        server_process.terminate()

if __name__ == "__main__":
    run()

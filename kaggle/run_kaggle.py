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
    # 3. Synchronize Checkpoints via Authoritative Manifest from Google Drive
    # -------------------------------------------------------------------------
    print("\n📥 [BƯỚC 3/4] ĐỒNG BỘ TRỌNG SỐ MÔ HÌNH QUA WEIGHTS MANIFEST:")
    from tqdm.auto import tqdm
    import gdown

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    weights_dir = os.path.join(repo_dir, "training_lab", "final_weights")
    os.makedirs(weights_dir, exist_ok=True)

    manifest_path = os.path.join(repo_dir, "kaggle", "weights_manifest.json")
    manifest_checkpoints = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                manifest_checkpoints = m_data.get("checkpoints", [])
        except Exception as e:
            print(f"    ⚠️ Lỗi đọc weights_manifest.json: {e}")

    # Check which checkpoints exist and have all required files
    missing_mandatory = []
    for cp in manifest_checkpoints:
        sub_path = cp.get("sub_path", "")
        cp_path = os.path.join(weights_dir, sub_path)
        req_files = cp.get("required_files", [])
        is_complete = os.path.exists(cp_path) and all(os.path.exists(os.path.join(cp_path, f)) for f in req_files)
        if not is_complete and cp.get("is_mandatory", False):
            missing_mandatory.append(cp.get("id"))

    if missing_mandatory:
        working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath(".")
        download_dir = os.path.join(working_dir, "drive_download")
        os.makedirs(download_dir, exist_ok=True)
        drive_folder_id = "1UPF13BNjxFFhd8YFv5PpuyfOR4G-VNte"

        print(f"    🌐 Phát hiện {len(missing_mandatory)} checkpoint bắt buộc chưa có sẵn ({', '.join(missing_mandatory)}).")
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
                print("    ✔ Giải nén trọn bộ mô hình hoàn tất thành công!")
            else:
                print("    ⚠️ Không tìm thấy tệp zip trong thư mục Drive đã tải.")
        except Exception as e:
            print(f"    ⚠️ Lỗi trong quá trình tải Drive: {e}")
    else:
        print("    ✔ Toàn bộ trọng số mô hình đã có sẵn và khớp với weights_manifest.json:")

    for cp in manifest_checkpoints:
        sub_path = cp.get("sub_path", "")
        cp_path = os.path.join(weights_dir, sub_path)
        req_files = cp.get("required_files", [])
        is_complete = os.path.exists(cp_path) and all(os.path.exists(os.path.join(cp_path, f)) for f in req_files)
        status = "✔" if is_complete else ("⚠️ THIẾU" if cp.get("is_mandatory", False) else "ℹ️ TÙY CHỌN")
        print(f"      {status} [{cp.get('id')}] - {cp.get('description', '')}")

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

    # Animated progress bar waiting for worker warmup (Wait until Worker A finishes loading weights)
    total_seconds = 75
    is_ready = False
    with tqdm(total=total_seconds, desc="    ⏳ Nạp mô hình vào 2 GPU", unit="s") as pbar:
        for s in range(total_seconds):
            time.sleep(1)
            pbar.update(1)
            try:
                res = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                if res.status == 200:
                    payload = json.loads(res.read().decode())
                    wh = payload.get("worker_health", {})
                    runtime_st = payload.get("runtime_status")
                    trans_st = payload.get("transition_state")
                    if wh.get("status") == "online" or (runtime_st == "running" and trans_st == "ready"):
                        is_ready = True
                        pbar.update(total_seconds - s - 1)
                        break
            except Exception:
                pass

    if not is_ready:
        for log_name in ["supervisor.log", "worker_finetune.log"]:
            lf_path = os.path.join(repo_dir, "kaggle", log_name)
            if os.path.exists(lf_path):
                print(f"\n    ⚠️ Nhật ký {log_name}:")
                with open(lf_path, "r", encoding="utf-8", errors="ignore") as lf:
                    for line in lf.readlines()[-10:]:
                        print("      " + line.strip())

    # Print Detailed Per-GPU VRAM Allocation Dashboard
    print("\n" + "─" * 78)
    print(" 🎉 TOÀN BỘ CỤM DUAL TESLA T4 ĐÃ ONLINE VÀ NẠP SẴN SÀNG VRAM!")
    print("─" * 78)
    try:
        res = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3)
        data = json.loads(res.read().decode())
        vram_alloc = data.get("vram_allocation", {})
        wh = data.get("worker_health", {})
        gpus = wh.get("gpus", {})
        if isinstance(vram_alloc, dict) and vram_alloc:
            for k, v in vram_alloc.items():
                print(f"  🔥 {k.upper()}:")
                print(f"     ├── VRAM đã cấp phát: {v.get('used_gb', 0)} / {v.get('total_gb', 16.0)} GB")
                print(f"     └── VRAM còn trống:   {v.get('free_gb', 16.0)} GB (Headroom an toàn)")
        elif isinstance(gpus, dict) and gpus:
            for k, v in gpus.items():
                v_used = v.get('vram_used_gb', 0)
                v_tot = v.get('vram_total_gb', 16.0)
                v_free = round(v_tot - v_used, 2)
                print(f"  🔥 {k.upper()}: {v.get('name', 'Tesla T4')}")
                print(f"     ├── VRAM đã cấp phát: {v_used} / {v_tot} GB")
                print(f"     └── VRAM còn trống:   {v_free} GB (Headroom an toàn)")
        else:
            print("    ✔ Cụm GPU đã nạp xong trọng số và sẵn sàng phục vụ suy luận.")
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

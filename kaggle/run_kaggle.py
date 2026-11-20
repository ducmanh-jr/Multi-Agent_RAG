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
import hashlib
import json
import urllib.request
import zipfile

def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def cleanup_cluster(server_process, repo_dir):
    """
    Guaranteed cluster cleanup (P0 fix):
    1. Gives supervisor process up to 40s to run graceful shutdown (which reaps worker & reclaims VRAM).
    2. If supervisor times out, escalates to SIGKILL.
    3. Reads kaggle/worker.pgid to ensure any detached worker process group is explicitly killed.
    4. Polling confirms PGID is verified dead before unlinking worker.pgid.
       If still alive, retains worker.pgid and logs P0 error.
    5. Zero blind/global pkill commands.
    """
    print("\n    🛑 Đang dọn dẹp cụm tiến trình Supervisor & GPU Workers...")
    all_clean = True
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    if server_process and server_process.poll() is None:
        server_process.terminate()
        try:
            server_process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            print("    ⚠️ Supervisor không dừng trong 40s, cưỡng chế kill tiến trình Supervisor...")
            server_process.kill()
            try:
                server_process.wait(timeout=5)
            except Exception:
                pass
            if server_process.poll() is None:
                all_clean = False
                print("    🛑 LỖI P0: Tiến trình Supervisor vẫn còn sống sau SIGKILL!")

    # Ensure worker process group is definitely not orphaned
    pgid_file = os.path.join(repo_dir, "kaggle", "worker.pgid")
    if os.path.exists(pgid_file):
        try:
            with open(pgid_file, "r", encoding="utf-8") as f:
                raw_pid = f.read().strip()
            if raw_pid.isdigit():
                worker_pgid = int(raw_pid)
                from kaggle.worker_manager import is_process_group_alive, kill_process_group, SIGKILL, SIGTERM
                if is_process_group_alive(worker_pgid):
                    print(f"    ⚠️ Phát hiện Worker Process Group {worker_pgid} còn sống! Gửi SIGTERM...")
                    kill_process_group(worker_pgid, SIGTERM)
                    time.sleep(2)
                    if is_process_group_alive(worker_pgid):
                        print(f"    ⚠️ Worker Process Group {worker_pgid} vẫn còn sống sau 2s! Leo thang SIGKILL...")
                        kill_process_group(worker_pgid, SIGKILL)

                # Poll to verify process group is genuinely dead before unlinking PGID file
                is_dead = False
                for _ in range(10):  # Poll up to 5.0 seconds
                    time.sleep(0.5)
                    if not is_process_group_alive(worker_pgid):
                        is_dead = True
                        break

                if is_dead:
                    try:
                        os.remove(pgid_file)
                        print(f"    ✔ Đã xác minh Worker Process Group {worker_pgid} đã chết hoàn toàn và dọn dẹp worker.pgid.")
                    except Exception as e:
                        all_clean = False
                        print(f"    🛑 LỖI P0: Không thể xóa {pgid_file} ({e})! Đánh dấu cleanup chưa hoàn tất.")
                else:
                    all_clean = False
                    print(f"    🛑 LỖI P0: Worker Process Group {worker_pgid} VẪN CÒN SỐNG sau SIGKILL! Giữ lại {pgid_file} để cách ly và điều tra.")
            else:
                all_clean = False
                print(f"    🛑 LỖI P0: File {pgid_file} rỗng hoặc định dạng không hợp lệ ({raw_pid!r})! Giữ lại file để điều tra, không thể xác minh worker.")
        except Exception as e:
            all_clean = False
            print(f"    🛑 LỖI P0: Ngoại lệ khi dọn dẹp worker.pgid ({e})! Giữ lại trạng thái lỗi.")

    if all_clean:
        print("    ✔ Hoàn tất dọn dẹp cụm tiến trình thành công.")
    else:
        print("    ❌ Dọn dẹp cụm tiến trình KHÔNG HOÀN TẤT: Còn tiến trình hoặc lỗi chưa được giải phóng triệt để.")
    return all_clean


def verify_checkpoint_integrity(cp, base_dir, verify_sha256=True):
    sub_path = cp.get("sub_path", "")
    cp_path = os.path.join(base_dir, sub_path)
    if not os.path.exists(cp_path):
        return False, "Thư mục checkpoint không tồn tại"

    req_files = cp.get("required_files", [])
    for rf in req_files:
        if not os.path.exists(os.path.join(cp_path, rf)):
            return False, f"Thiếu file bắt buộc: {rf}"

    if verify_sha256 and "primary_file" in cp:
        pf = cp["primary_file"]
        pf_path = os.path.join(cp_path, pf["filename"])
        if not os.path.exists(pf_path):
            return False, f"Thiếu primary file: {pf['filename']}"
        h = hashlib.sha256()
        with open(pf_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        calc_sha = h.hexdigest()
        if calc_sha != pf["sha256"]:
            return False, f"Sai lệch SHA256: {calc_sha[:12]} != {pf['sha256'][:12]}"

    return True, "Hợp lệ"

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

    # Post-installation verification: fail fast if mandatory packages failed to install
    failed_pkgs = []
    has_dual_gpu = False
    try:
        import torch
        has_dual_gpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:
        pass

    for pkg, mod in pkg_map.items():
        if pkg == "vllm" and not has_dual_gpu:
            # vllm is only strictly required when Dual-GPU is present for GPU Max
            continue
        if importlib.util.find_spec(mod) is None:
            failed_pkgs.append(pkg)

    if failed_pkgs:
        print(f"\n    ❌ LỖI NGHIÊM TRỌNG: Cài đặt thư viện thất bại cho: {', '.join(failed_pkgs)}")
        print("    Vui lòng kiểm tra lại kết nối mạng hoặc môi trường Python của Kaggle.")
        sys.exit(1)

    print("    ✔ Toàn bộ thư viện cần thiết đã sẵn sàng và được xác thực hoàn tất!")

    # -------------------------------------------------------------------------
    # 3. Synchronize Checkpoints via Authoritative Manifest from Google Drive
    # -------------------------------------------------------------------------
    print("\n📥 [BƯỚC 3/4] ĐỒNG BỘ TRỌNG SỐ MÔ HÌNH QUA WEIGHTS MANIFEST:")
    from tqdm.auto import tqdm
    import gdown

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    weights_dir = os.path.join(repo_dir, "training_lab", "final_weights")
    os.makedirs(weights_dir, exist_ok=True)

    manifest_path = os.path.join(repo_dir, "kaggle", "weights_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"    ❌ LỖI NGHIÊM TRỌNG: Không tìm thấy weights_manifest.json tại {manifest_path}")
        sys.exit(1)

    try:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            m_data = json.load(mf)
            manifest_checkpoints = m_data.get("checkpoints", [])
            if not manifest_checkpoints:
                raise ValueError("Manifest không chứa danh sách checkpoint nào.")
    except Exception as e:
        print(f"    ❌ LỖI NGHIÊM TRỌNG: Không thể đọc hoặc phân tích weights_manifest.json: {e}")
        sys.exit(1)

    # Check which checkpoints exist and have verified cryptographic SHA256 hashes
    missing_mandatory = []
    for cp in manifest_checkpoints:
        is_valid, reason = verify_checkpoint_integrity(cp, weights_dir, verify_sha256=True)
        if not is_valid and cp.get("is_mandatory", False):
            missing_mandatory.append((cp.get("id"), reason))

    if missing_mandatory:
        working_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath(".")
        download_dir = os.path.join(working_dir, "drive_download")
        os.makedirs(download_dir, exist_ok=True)
        drive_folder_id = "1UPF13BNjxFFhd8YFv5PpuyfOR4G-VNte"

        missing_names = [m[0] for m in missing_mandatory]
        print(f"    🌐 Phát hiện {len(missing_mandatory)} checkpoint bắt buộc chưa hoàn chỉnh ({', '.join(missing_names)}).")
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
        print("    ✔ Toàn bộ trọng số mô hình đã có sẵn và xác thực SHA256 thành công:")

    # Post-staging mandatory cryptographic verification (Fail-Fast)
    unresolved_mandatory = []
    print("\n    🔍 Kiểm tra mật mã học SHA256 toàn bộ checkpoint trên đĩa:")
    for cp in manifest_checkpoints:
        is_valid, reason = verify_checkpoint_integrity(cp, weights_dir, verify_sha256=True)
        if not is_valid and cp.get("is_mandatory", False):
            unresolved_mandatory.append((cp.get("id"), reason))
        status = "✔ SHA256 HỢP LỆ" if is_valid else (f"❌ HỎNG/THIẾU ({reason})" if cp.get("is_mandatory", False) else f"ℹ️ TÙY CHỌN ({reason})")
        print(f"      • [{cp.get('id')}] {status} - {cp.get('description', '')}")

    if unresolved_mandatory:
        print(f"\n    ❌ DỪNG HỆ THỐNG: Phát hiện {len(unresolved_mandatory)} checkpoint bắt buộc không vượt qua kiểm tra toàn vẹn SHA256:")
        for cid, reason in unresolved_mandatory:
            print(f"      - {cid}: {reason}")
        print("    Vui lòng kiểm tra lại gói trọng số hoặc kết nối mạng trước khi khởi động Supervisor.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Start Master Supervisor Gateway & Load Dual GPU VRAM
    # -------------------------------------------------------------------------
    print("\n🚀 [BƯỚC 4/4] KHỞI CHẠY TIẾN TRÌNH SUPERVISOR GATEWAY & NẠP VRAM DUAL T4:")
    os.environ["INITIAL_MODE"] = "gpu_finetune"
    os.environ["PROJECT_DIR"] = repo_dir

    # Automatically generate / configure secure admin token
    admin_token = os.environ.get("SUPERVISOR_ADMIN_TOKEN")
    if not admin_token:
        import secrets
        admin_token = secrets.token_hex(16)
        os.environ["SUPERVISOR_ADMIN_TOKEN"] = admin_token

    print("    ⚙️  Đang khởi động Master Gateway trên cổng 8000...")
    supervisor_log = open(os.path.join(repo_dir, "kaggle", "supervisor.log"), "w", encoding="utf-8", errors="ignore")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "kaggle_supervisor_gateway:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--app-dir", os.path.join(repo_dir, "kaggle"),
    ], stdout=supervisor_log, stderr=subprocess.STDOUT)

    # Register signal handlers for clean interrupt
    def sig_handler(sig, frame):
        print(f"\n    🛑 Nhận tín hiệu ngắt ({sig}), bắt đầu dọn dẹp cụm...")
        cleanup_cluster(server_process, repo_dir)
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)
    except Exception:
        pass

    # Animated progress bar waiting for worker warmup (Wait until Worker A finishes loading weights)
    total_seconds = int(os.environ.get("RUNNER_STARTUP_TIMEOUT", "360"))
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
        print(f"\n    ❌ LỖI NGHIÊM TRỌNG: Hết thời gian chờ ({total_seconds}s) nhưng GPU Worker chưa đạt trạng thái sẵn sàng (ready)!")
        for log_name in ["supervisor.log", "worker_finetune.log"]:
            lf_path = os.path.join(repo_dir, "kaggle", log_name)
            if os.path.exists(lf_path):
                print(f"\n    ⚠️ Nhật ký {log_name} (20 dòng gần nhất):")
                with open(lf_path, "r", encoding="utf-8", errors="ignore") as lf:
                    for line in lf.readlines()[-20:]:
                        print("      " + line.strip())

        # Guaranteed cluster cleanup with up to 40s timeout & verified PGID reaping
        cleanup_cluster(server_process, repo_dir)

        print("\n    🛑 DỪNG TOÀN BỘ HỆ THỐNG: Khởi chạy thất bại. Tuyệt đối không mở Cloudflare Tunnel.")
        sys.exit(1)

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
    print("║  🚀 THÔNG TIN KẾT NỐI CHO MULTI-AGENT RAG (HÃY SAO CHÉP CÁC DÒNG DƯỚI)   ║")
    print("╠" + "═" * 76 + "╣")
    print(f"║                                                                            ║")
    print(f"║    REMOTE_INFERENCE_URL={tunnel_url.ljust(51)}║")
    print(f"║    SUPERVISOR_ADMIN_TOKEN={admin_token.ljust(49)}║")
    print(f"║                                                                            ║")
    print("╚" + "═" * 76 + "╝")
    print("\n📌 HƯỚNG DẪN TIẾP THEO:")
    print(f"  1. Sao chép dòng: REMOTE_INFERENCE_URL={tunnel_url}")
    print(f"  2. Sao chép dòng: SUPERVISOR_ADMIN_TOKEN={admin_token}")
    print(f"  3. Dán vào file .env trên máy của bạn (hoặc dán trực tiếp trên Web Demo2).")
    print(f"  4. Cụm 7 Mô Hình SOTA đang trực tuyến và chạy trên Dual T4.")
    print(f"  5. Muốn đổi sang Đại Mô Hình 32B AWQ: Chỉ cần 1-click trên Web Demo2!")
    print("─" * 78)
    print("🟢 Máy chủ đang giữ kết nối... Vui lòng để ô code này tiếp tục chạy.\n")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        cleanup_cluster(server_process, repo_dir)

if __name__ == "__main__":
    run()

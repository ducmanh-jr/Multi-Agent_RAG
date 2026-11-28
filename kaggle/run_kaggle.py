"""
Kaggle High-Visibility Master Runner for Multi-Agent RAG.
Provides rich, real-time visual progress bars (percentages, download speeds,
file extraction, per-GPU VRAM usage) while keeping the caller notebook to just 3 lines of code.
"""

import os
import signal
import sys
import subprocess
import shutil
import time
import hashlib
import json
import urllib.request
import zipfile

def load_kaggle_secret_into_env(secret_name: str, env_var: str = None) -> bool:
    """
    Safely loads a secret from Kaggle Secrets (Add-ons -> Secrets) into os.environ
    without ever printing or logging the secret value.
    Only invoked when running inside Kaggle; returns False cleanly elsewhere.
    """
    target_var = env_var or secret_name
    if os.environ.get(target_var):
        return True
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]
        val = UserSecretsClient().get_secret(secret_name)
        if isinstance(val, str) and val.strip():
            os.environ[target_var] = val.strip()
            return True
    except Exception:
        return False
    return False


WORKER_PROFILE_SPECS = {
    "qa_dual_gpu": {
        "canonical_profile": "qa_dual_gpu",
        "initial_mode": "gpu_finetune",
        "server_profile": "qa_dual_gpu",
        "prefetch_qwen": True,
        "description": "Dual-T4 online QA (GPU 0: Qwen3-8B NF4, GPU 1: BGE-M3 + Reranker + mDeBERTa-v3 NLI)",
    },
    "ingest": {
        "canonical_profile": "ingest",
        "initial_mode": "gpu_finetune",
        "server_profile": "ingest",
        "prefetch_qwen": False,
        "description": "Dedicated heavy PDF OCR/Layout/Embedding ingest worker (skips Qwen 8B to maximize OCR VRAM)",
    },
    "32b_tp2": {
        "canonical_profile": "32b_tp2",
        "initial_mode": "gpu_max",
        "server_profile": "32b_tp2",
        "prefetch_qwen": False,
        "description": "Dual-T4 Tensor Parallel=2 vLLM Qwen3-32B-AWQ generator",
    },
    # Backward-compatible aliases
    "gpu_finetune": {
        "canonical_profile": "qa_dual_gpu",
        "initial_mode": "gpu_finetune",
        "server_profile": "gpu_finetune",
        "prefetch_qwen": True,
        "description": "Legacy alias for qa_dual_gpu",
    },
    "hybrid_specialist": {
        "canonical_profile": "qa_dual_gpu",
        "initial_mode": "gpu_finetune",
        "server_profile": "hybrid_specialist",
        "prefetch_qwen": False,
        "description": "Hybrid specialist worker (GPU 1 retrieval/NLI without GPU 0 Qwen unless fallback enabled)",
    },
    "hybrid_sota_base": {
        "canonical_profile": "qa_dual_gpu",
        "initial_mode": "gpu_finetune",
        "server_profile": "hybrid_sota_base",
        "prefetch_qwen": True,
        "env_defaults": {
            "QWEN_USE_BASE_INSTRUCT": "true",
            "BASE_QWEN_MODEL_ID": "Qwen/Qwen3-8B",
            "PREFER_HF_SOTA_NLI": "true",
            "FACTCHECK_MODEL_ID": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
            "PRELOAD_RETRIEVAL_MODELS": "true",
            "SKIP_DRIVE_LORA_BUNDLE": "true",
        },
        "description": "Best pure-base Dual-T4 stack before fine-tuning (GPU 0: Qwen3-8B NF4, GPU 1: BGE-M3 + BGE-Reranker-v2-M3 + mDeBERTa-v3 NLI fp16)",
    },
    "hybrid_sota_max": {
        "canonical_profile": "qa_dual_gpu",
        "initial_mode": "gpu_finetune",
        "server_profile": "hybrid_sota_max",
        "prefetch_qwen": True,
        "env_defaults": {
            "QWEN_USE_BASE_INSTRUCT": "true",
            "BASE_QWEN_MODEL_ID": "unsloth/Qwen3-14B-bnb-4bit",
            "PREFER_HF_SOTA_NLI": "true",
            "FACTCHECK_MODEL_ID": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            "FACTCHECK_GPU_INDEX": "0",
            "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
            "KAGGLE_EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
            "EMBEDDING_GPU_INDEX": "1",
            "EMBEDDING_4BIT": "true",
            "EMBEDDING_BATCH_SIZE": "2",
            "RERANKER_MODEL": "Qwen/Qwen3-Reranker-8B",
            "KAGGLE_RERANKER_MODEL": "Qwen/Qwen3-Reranker-8B",
            "RERANKER_GPU_INDEX": "1",
            "RERANKER_4BIT": "true",
            "QWEN_RERANK_BATCH_SIZE": "2",
            "PRELOAD_RETRIEVAL_MODELS": "true",
            "SKIP_DRIVE_LORA_BUNDLE": "true",
            "RUNNER_STARTUP_TIMEOUT": "900",
            "WORKER_FINETUNE_TIMEOUT": "900",
        },
        "description": "MAX-VRAM Dual-T4 SOTA stack (~23-27GB/30GB): GPU 0 Qwen3-14B NF4 + mDeBERTa-v3 NLI fp16, GPU 1 Qwen3-Embedding-8B NF4 + Qwen3-Reranker-8B NF4",
    },
    "gpu_max": {
        "canonical_profile": "32b_tp2",
        "initial_mode": "gpu_max",
        "server_profile": "32b_tp2",
        "prefetch_qwen": False,
        "description": "Legacy alias for 32b_tp2",
    },
}


def normalize_worker_profile(raw_profile: str | None) -> dict:
    key = (raw_profile or "qa_dual_gpu").strip().lower()
    if key in WORKER_PROFILE_SPECS:
        return dict(WORKER_PROFILE_SPECS[key])
    raise ValueError(
        f"Unsupported KAGGLE_WORKER_PROFILE={raw_profile!r}. "
        f"Expected one of: {', '.join(sorted(WORKER_PROFILE_SPECS))}."
    )


def apply_profile_env_defaults(raw_profile: str | None, env: dict | None = None) -> dict[str, str]:
    """Apply non-destructive default env vars defined by a worker profile (e.g., `hybrid_sota_base`)."""
    spec = normalize_worker_profile(raw_profile)
    target_env = os.environ if env is None else env
    applied: dict[str, str] = {}
    for k, v in (spec.get("env_defaults") or {}).items():
        if not str(target_env.get(k) or "").strip():
            target_env[k] = str(v)
            applied[k] = str(v)
    return applied


def register_tunnel_with_backend(
    tunnel_url: str,
    inference_token: str,
    *,
    callback_url: str | None = None,
    register_secret: str | None = None,
    worker_profile: str | None = None,
    git_commit_sha: str | None = None,
    max_retries: int = 3,
    timeout: float = 10.0,
) -> dict:
    """
    Auto-register a freshly opened Cloudflare Tunnel URL with the local RAG backend
    (`POST /api/config/remote-url/register`) using `X-Tunnel-Secret` authentication.
    Returns a status dict and never raises so runner startup stays resilient.
    """
    cb = (callback_url if callback_url is not None else os.environ.get("LOCAL_RAG_CALLBACK_URL", "")).strip().rstrip("/")
    if not cb:
        return {"registered": False, "reason": "callback_url_not_configured"}
    secret = (
        register_secret
        if register_secret is not None
        else (os.environ.get("TUNNEL_REGISTER_SECRET") or inference_token or "")
    ).strip()
    endpoint = cb if cb.endswith("/api/config/remote-url/register") else f"{cb}/api/config/remote-url/register"
    payload = json.dumps({
        "remote_url": tunnel_url.strip().rstrip("/"),
        "remote_token": inference_token,
        "worker_profile": worker_profile or os.environ.get("KAGGLE_WORKER_PROFILE", "qa_dual_gpu"),
        "git_commit_sha": git_commit_sha or os.environ.get("ACTIVE_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA") or "unpinned",
    }).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "X-Tunnel-Secret": secret,
    }

    last_error = ""
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
                if 200 <= getattr(resp, "status", 200) < 300:
                    return {
                        "registered": True,
                        "status_code": getattr(resp, "status", 200),
                        "endpoint": endpoint,
                        "attempts": attempt,
                        "response": body,
                    }
                last_error = f"HTTP {getattr(resp, 'status', 'unknown')}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(min(2.0, 0.5 * attempt))
    return {"registered": False, "endpoint": endpoint, "attempts": attempts, "reason": last_error}


def configure_runtime_tokens(env: dict | None = None, secret_loader=None) -> dict:
    """
    Loads or generates strictly SEPARATE admin (`SUPERVISOR_ADMIN_TOKEN`) and
    inference (`SUPERVISOR_API_TOKEN`) tokens so inference clients never hold admin privileges.
    Never logs or returns secret values in public telemetry.
    """
    import hmac
    import secrets as _secrets

    target_env = os.environ if env is None else env
    loader = load_kaggle_secret_into_env if secret_loader is None else secret_loader

    loaded_admin = bool(loader("SUPERVISOR_ADMIN_TOKEN")) if not target_env.get("SUPERVISOR_ADMIN_TOKEN") else True
    loaded_api = bool(loader("SUPERVISOR_API_TOKEN")) if not target_env.get("SUPERVISOR_API_TOKEN") else True

    admin_token = (target_env.get("SUPERVISOR_ADMIN_TOKEN") or "").strip()
    api_token = (target_env.get("SUPERVISOR_API_TOKEN") or "").strip()

    admin_auto_generated = False
    api_auto_generated = False
    if not admin_token:
        admin_token = _secrets.token_hex(16)
        target_env["SUPERVISOR_ADMIN_TOKEN"] = admin_token
        admin_auto_generated = True
    if not api_token:
        api_token = _secrets.token_hex(16)
        while hmac.compare_digest(api_token, admin_token):
            api_token = _secrets.token_hex(16)
        target_env["SUPERVISOR_API_TOKEN"] = api_token
        api_auto_generated = True

    require_separate = target_env.get("SEPARATE_ADMIN_INFERENCE_TOKENS", "true").strip().lower() in ("true", "1", "yes")
    if require_separate and hmac.compare_digest(admin_token, api_token):
        raise RuntimeError(
            "SUPERVISOR_ADMIN_TOKEN và SUPERVISOR_API_TOKEN không được trùng nhau khi SEPARATE_ADMIN_INFERENCE_TOKENS=true."
        )

    target_env["SEPARATE_ADMIN_INFERENCE_TOKENS"] = "true" if require_separate else "false"
    return {
        "loaded_admin_secret": loaded_admin and not admin_auto_generated,
        "loaded_api_secret": loaded_api and not api_auto_generated,
        "admin_auto_generated": admin_auto_generated,
        "api_auto_generated": api_auto_generated,
        "tokens_distinct": not hmac.compare_digest(admin_token, api_token),
        "separation_enforced": require_separate,
    }


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
                if worker_pgid > 1 and worker_pgid <= 4194304:
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
                    print(f"    🛑 LỖI P0: PGID trong {pgid_file} là giá trị không an toàn/hợp lệ ({worker_pgid} <= 1 hoặc vượt giới hạn)! Giữ lại file để điều tra.")
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


def handle_runner_interrupt(server_process, repo_dir, sig=None):
    """
    Unified interrupt handler for SIGINT, SIGTERM, and KeyboardInterrupt.
    Guarantees cluster cleanup and exits with 0 (clean) or 1 (unclean).
    """
    sig_str = f" ({sig})" if sig is not None else ""
    print(f"\n    🛑 Nhận tín hiệu ngắt{sig_str}, bắt đầu dọn dẹp cụm...")
    clean = cleanup_cluster(server_process, repo_dir)
    sys.exit(0 if clean else 1)


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
        "huggingface-hub": "huggingface_hub",
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

    # Apply profile-level env defaults early (e.g., `hybrid_sota_base` -> Qwen3-8B + BGE-M3 + mDeBERTa-v3)
    worker_profile_raw_early = os.environ.get("KAGGLE_WORKER_PROFILE", "qa_dual_gpu").strip().lower()
    applied_defaults = apply_profile_env_defaults(worker_profile_raw_early)
    if applied_defaults:
        print(f"    ⚡ Profile '{worker_profile_raw_early}' đã tự cấu hình mặc định: {', '.join(f'{k}={v}' for k, v in applied_defaults.items())}")

    # Check if user wants pure SOTA HuggingFace base models (skipping old Drive LoRA zip for fast testing)
    skip_drive_bundle = (
        os.environ.get("SKIP_DRIVE_LORA_BUNDLE", "false").strip().lower() in ("true", "1", "yes")
        or (
            os.environ.get("QWEN_USE_BASE_INSTRUCT", "false").strip().lower() in ("true", "1", "yes")
            and os.environ.get("PREFER_HF_SOTA_NLI", "false").strip().lower() in ("true", "1", "yes")
        )
    )

    if skip_drive_bundle:
        print("    ⚡ Chế độ Pure SOTA HuggingFace Models (SKIP_DRIVE_LORA_BUNDLE=true): Bỏ qua gói LoRA cũ trên Google Drive.")
        print("    ⚡ Sẽ tải và chạy trực tiếp các mô hình nền SOTA sạch từ Hugging Face Hub để kiểm thử trước khi fine-tune!")
    else:
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
            allow_hf_fallback = os.environ.get("ALLOW_HF_SOTA_FALLBACK", "false").strip().lower() in ("true", "1", "yes")
            if allow_hf_fallback:
                print(f"\n    ⚠️ Cảnh báo: {len(unresolved_mandatory)} checkpoint Drive chưa tải được. Tự động chuyển sang bộ Model SOTA trực tiếp từ Hugging Face Hub!")
                os.environ["PREFER_HF_SOTA_NLI"] = "true"
                os.environ["QWEN_USE_BASE_INSTRUCT"] = "true"
            else:
                print(f"\n    ❌ DỪNG HỆ THỐNG: Phát hiện {len(unresolved_mandatory)} checkpoint bắt buộc không vượt qua kiểm tra toàn vẹn SHA256:")
                for cid, reason in unresolved_mandatory:
                    print(f"      - {cid}: {reason}")
                print("    Vui lòng kiểm tra lại gói trọng số hoặc kết nối mạng trước khi khởi động Supervisor.")
                sys.exit(1)

    # Ensure transformers >= 4.51.0 when running any Qwen3 family model (Generator, Embedding, or Reranker)
    base_model_spec = m_data.get("base_models", {}).get("qwen_7b_base", {})
    base_model_id = os.environ.get("BASE_QWEN_MODEL_ID", base_model_spec.get("model_id", "Qwen/Qwen2.5-7B-Instruct"))
    active_embed_id = (os.environ.get("KAGGLE_EMBEDDING_MODEL") or os.environ.get("EMBEDDING_MODEL") or "BAAI/bge-m3").strip()
    active_rerank_id = (os.environ.get("KAGGLE_RERANKER_MODEL") or os.environ.get("RERANKER_MODEL") or "BAAI/bge-reranker-v2-m3").strip()
    if any("qwen3" in mid.lower() for mid in (base_model_id, active_embed_id, active_rerank_id)):
        try:
            import transformers as _tf
            _ver = tuple(int(x) for x in _tf.__version__.split(".")[:2] if x.isdigit())
            if _ver < (4, 51):
                print(f"    ⏳ Nâng cấp transformers ({_tf.__version__} -> >=4.51.0) để hỗ trợ cụm Qwen3...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51.0", "accelerate", "bitsandbytes"], check=False)
        except Exception:
            pass

    try:
        free_disk_gb = shutil.disk_usage("/").free / (1024 ** 3)
        print(f"    💾 Dung lượng ổ đĩa trống (/): {free_disk_gb:.1f} GB")
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Pre-fetch Base & Specialist Models into local HF cache so Step 4 startup is fast
    # -------------------------------------------------------------------------
    worker_profile_raw = os.environ.get("KAGGLE_WORKER_PROFILE", "qa_dual_gpu").strip().lower()
    profile_spec = normalize_worker_profile(worker_profile_raw)
    worker_profile = profile_spec["canonical_profile"]
    load_qwen_fallback = os.environ.get("HYBRID_LOAD_QWEN_FALLBACK", "false").strip().lower() in ("true", "1", "yes")
    should_prefetch_qwen = profile_spec["prefetch_qwen"] or (worker_profile_raw == "hybrid_specialist" and load_qwen_fallback)
    if not should_prefetch_qwen:
        print(f"\n    ⚡ Profile '{worker_profile_raw}' (canonical={worker_profile}): Bỏ qua tải trước {base_model_id} để tối ưu VRAM/khởi động nhanh.")
    else:
        print(f"\n    🌐 Đồng bộ Base Model {base_model_id} (GPU 0) từ Hugging Face Hub:")
        try:
            from huggingface_hub import snapshot_download
            print(f"      • Bắt đầu tải/kiểm tra cache các shard của {base_model_id}...")
            snapshot_download(
                repo_id=base_model_id,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
                ignore_patterns=["*.bin", "*.pt", "*.msgpack"],
            )
            print(f"      ✔ Base Model {base_model_id} đã sẵn sàng 100% trong cache!")
        except Exception as e:
            print(f"      ⚠️ Cảnh báo tải trước Base Model: {e}. Tiến trình Worker sẽ tự tải khi khởi động.")

    # Pre-fetch Specialist Models (configured Embedding, Reranker, and Multilingual NLI)
    gpu1_hf_models = []
    if os.environ.get("PRELOAD_RETRIEVAL_MODELS", "false").strip().lower() in ("true", "1", "yes") or worker_profile == "ingest":
        gpu1_hf_models.extend([
            (active_embed_id, "Dense Vector Embedding 1024-dim"),
            (active_rerank_id, "Cross-Encoder / Causal Reranker"),
        ])
    if worker_profile != "ingest" and os.environ.get("PREFER_HF_SOTA_NLI", "false").strip().lower() in ("true", "1", "yes"):
        nli_id = os.environ.get("FACTCHECK_MODEL_ID", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
        gpu1_hf_models.append((nli_id, "Multilingual 3-Class Fact-Check NLI"))

    if gpu1_hf_models:
        print("\n    🌐 Đồng bộ cụm Specialist Models (GPU 1) từ Hugging Face Hub:")
        try:
            from huggingface_hub import snapshot_download
            for hf_id, desc in gpu1_hf_models:
                print(f"      • Đang tải {hf_id} ({desc})...")
                snapshot_download(
                    repo_id=hf_id,
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "sentencepiece*", "*.model"],
                    ignore_patterns=["*.bin", "*.pt", "*.onnx", "*.msgpack"],
                )
                print(f"      ✔ {hf_id} đã sẵn sàng!")
        except Exception as e:
            print(f"      ⚠️ Cảnh báo tải trước Specialist Models: {e}. Worker sẽ tự tải khi khởi động.")

    # -------------------------------------------------------------------------
    # 4. Start Master Supervisor Gateway & Load Dual GPU VRAM
    # -------------------------------------------------------------------------
    print("\n🚀 [BƯỚC 4/4] KHỞI CHẠY TIẾN TRÌNH SUPERVISOR GATEWAY & NẠP VRAM DUAL T4:")
    os.environ["INITIAL_MODE"] = profile_spec["initial_mode"]
    os.environ["PROJECT_DIR"] = repo_dir

    # Load strictly separate Admin & Inference tokens from Kaggle Secrets (never printed/logged)
    if os.environ.get("WORKER_LOAD_GEMINI_SECRET", "false").strip().lower() == "true":
        load_kaggle_secret_into_env("GEMINI_API_KEY")

    token_info = configure_runtime_tokens()
    inference_token = os.environ["SUPERVISOR_API_TOKEN"]
    token_auto_generated = bool(token_info["api_auto_generated"])
    os.environ.setdefault("PUBLIC_TUNNEL", "true")

    print("    ⚙️  Đang khởi động Master Gateway trên cổng 8000...")
    supervisor_log = open(os.path.join(repo_dir, "kaggle", "supervisor.log"), "w", encoding="utf-8", errors="ignore")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "kaggle_supervisor_gateway:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--app-dir", os.path.join(repo_dir, "kaggle"),
        "--no-access-log",
    ], stdout=supervisor_log, stderr=subprocess.STDOUT)

    # Register signal handlers for clean interrupt
    def sig_handler(sig, frame):
        handle_runner_interrupt(server_process, repo_dir, sig=sig)

    try:
        signal.signal(signal.SIGINT, sig_handler)
        signal.signal(signal.SIGTERM, sig_handler)
    except Exception:
        pass

    # Animated progress bar waiting for worker warmup (Wait until Worker A finishes loading weights)
    total_seconds = int(os.environ.get("RUNNER_STARTUP_TIMEOUT", "600"))
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
                print(f"\n    ⚠️ Nhật ký {log_name} (các dòng hoạt động gần nhất):")
                with open(lf_path, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = [l.strip() for l in lf.readlines() if l.strip() and "GET /health" not in l]
                    for line in lines[-40:]:
                        print("      " + line)

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

    active_git_sha = (os.environ.get("ACTIVE_GIT_COMMIT_SHA") or os.environ.get("GIT_COMMIT_SHA") or "unpinned").strip()
    reg_result = register_tunnel_with_backend(
        tunnel_url,
        inference_token,
        worker_profile=worker_profile_raw,
        git_commit_sha=active_git_sha,
    )
    if reg_result.get("registered"):
        print(f"    ✔ Đã tự động đăng ký Tunnel URL về Backend ({reg_result.get('endpoint')})!")
    elif reg_result.get("reason") != "callback_url_not_configured":
        print(f"    ⚠️ Tự động đăng ký Tunnel thất bại ({reg_result.get('reason')}); vẫn giữ Tunnel hoạt động bình thường.")

    # -------------------------------------------------------------------------
    # 6. Beautiful Link Output Banner
    # -------------------------------------------------------------------------
    display_token = inference_token if token_auto_generated else "<da nap tu Kaggle Secrets / env>"
    print("\n" + "╔" + "═" * 76 + "╗")
    print("║  🚀 THÔNG TIN KẾT NỐI CHO MULTI-AGENT RAG (HÃY SAO CHÉP CÁC DÒNG DƯỚI)   ║")
    print("╠" + "═" * 76 + "╣")
    print(f"║                                                                            ║")
    print(f"║    REMOTE_INFERENCE_URL={tunnel_url.ljust(51)}║")
    print(f"║    REMOTE_INFERENCE_TOKEN={display_token.ljust(49)}║")
    print(f"║                                                                            ║")
    print("╚" + "═" * 76 + "╝")
    print("\n📌 HƯỚNG DẪN TIẾP THEO:")
    print(f"  1. Sao chép dòng: REMOTE_INFERENCE_URL={tunnel_url}")
    if token_auto_generated:
        print(f"  2. Sao chép dòng: REMOTE_INFERENCE_TOKEN={inference_token} (chỉ dùng cho suy luận, tách riêng Admin token)")
    else:
        print("  2. Dùng đúng SUPERVISOR_API_TOKEN đã đặt trong Kaggle Secrets cho REMOTE_INFERENCE_TOKEN (không dùng Admin token).")
    print(f"  3. Dán vào file .env trên máy của bạn (hoặc dán trực tiếp trên Web Demo2).")
    print(f"  4. Profile đang chạy: {worker_profile} ({worker_profile_raw}) | Git SHA: {active_git_sha}")
    print(f"  5. Quyền Admin (/api/admin/*) dùng SUPERVISOR_ADMIN_TOKEN tách biệt hoàn toàn.")
    print("─" * 78)
    print("🟢 Máy chủ đang giữ kết nối... Vui lòng để ô code này tiếp tục chạy.\n")

    try:
        server_process.wait()
    except KeyboardInterrupt:
        handle_runner_interrupt(server_process, repo_dir, sig="KeyboardInterrupt")

if __name__ == "__main__":
    run()

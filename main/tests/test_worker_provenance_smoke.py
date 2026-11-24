import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from core.config import settings
from eval import smoke_worker_provenance as smoke


def test_smoke_import_does_not_open_workspace_db():
    project = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; sys.path.insert(0, 'main'); from eval import smoke_worker_provenance; "
            "assert 'core.db' not in sys.modules"
        )],
        cwd=project, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
@pytest.mark.parametrize(("scenario", "failure"), [
    ("gateway", None),
    ("direct", None),
    ("dns", "worker_health_unreachable"),
    ("wrong_mode", "worker_mode_not_ready"),
    ("old_generator", "generator_checkpoint_unconfirmed"),
    ("old_nli", "nli_checkpoint_unconfirmed"),
    ("wrong_page", "verified_citation_not_linked"),
])
def test_provenance_smoke_validates_stored_run_without_network(tmp_path, monkeypatch, capsys, scenario, failure):
    from core import db
    from utils.llm_client import llm_client

    url = "https://private-worker.invalid"
    env_file = tmp_path / ".env"
    env_file.write_text(f"REMOTE_INFERENCE_URL={url}\n", encoding="utf-8")
    monkeypatch.setattr(settings, "GPU_FINETUNE_DENSE_ENABLED", False)
    monkeypatch.setattr(settings, "GPU_FINETUNE_RERANK_ENABLED", False)
    previous_db = db.DB_PATH
    requests = []

    def worker(request):
        assert request.url.host == "private-worker.invalid"
        endpoint = request.url.path
        requests.append(endpoint)
        if endpoint == "/health":
            if scenario == "dns":
                raise httpx.ConnectError(f"Network unavailable for {url}", request=request)
            health = {
                "status": "online", "pipeline": "gpu_finetune_7_models", "gpus_available": 2,
                "loaded_adapters": ["generator", "critic"], "active_models": ["qwen_cluster", "factcheck"],
            }
            if scenario == "direct":
                return httpx.Response(200, json=health)
            mode = "gpu_max" if scenario == "wrong_mode" else "gpu_finetune"
            return httpx.Response(200, json={"active_mode": mode, "worker_health": health})
        if endpoint == "/api/router":
            return httpx.Response(200, json={"intent": "academic_research", "sub_queries": [smoke.QUESTION], "top_k": 3})
        if endpoint == "/api/reader/layout":
            return httpx.Response(404)
        if endpoint == "/api/summarize/generator":
            context = json.loads(request.content)["context"]
            assert smoke.EVIDENCE in context
            unit = re.search(r"\[Unit ID: (.*?) \|", context)[1]
            headers = {} if scenario == "old_generator" else {
                "X-Research-Model": "Qwen/Qwen2.5-7B-Instruct", "X-Research-Revision": "a" * 40,
                "X-Research-Adapter": "generator", "X-Research-Adapter-SHA256": "b" * 64,
            }
            return httpx.Response(200, headers=headers, json={
                "answer": smoke.EVIDENCE, "claims": [{"text": smoke.EVIDENCE, "unit_id": unit}],
                "evidence_spans": [{"quote": smoke.EVIDENCE, "unit_id": unit}],
            })
        if endpoint == "/api/summarize/critic":
            return httpx.Response(200, json={"action": "accept", "is_supported": True, "issues_found": []})
        if endpoint == "/api/factcheck":
            payload = json.loads(request.content)
            assert smoke.EVIDENCE in payload["context"] and payload["claims"] == [smoke.EVIDENCE]
            headers = {} if scenario == "old_nli" else {
                "X-Research-Model": "research-finetuned-nli", "X-Research-Revision": "c" * 64,
            }
            return httpx.Response(200, headers=headers, json={
                "evaluations": [{"claim": smoke.EVIDENCE, "verdict": "SUPPORT", "confidence": 0.99}],
            })
        pytest.fail(f"Unexpected fixture endpoint {endpoint}")

    if scenario == "wrong_page":
        get_run = db.get_run

        def wrong_page(run_id):
            record = get_run(run_id)
            for event in record["reasoning_trace"]:
                if event.get("action") == "citation_audit":
                    for entry in event["citation_audit"]["evidence_provenance"].values():
                        entry["source_unit"]["page_no"] = 2
            return record

        monkeypatch.setattr(db, "get_run", wrong_page)

    with httpx.Client(transport=httpx.MockTransport(worker)) as provider:
        monkeypatch.setattr(llm_client, "http_client", provider)
        exit_code = smoke.main(["--env-file", str(env_file)])
    output = capsys.readouterr().out
    result = json.loads(output.splitlines()[-1])
    assert url not in output
    assert db.DB_PATH == previous_db
    if failure is None:
        assert exit_code == 0 and result["status"] == "passed"
        assert result["generator"] == {
            "model": "Qwen/Qwen2.5-7B-Instruct", "revision": "a" * 40, "adapter_sha256": "b" * 64,
        }
        assert result["nli_revision"] == "c" * 64
        assert result["citation"]["page"] == 1
        assert result["citation"]["quote"] == smoke.EVIDENCE
        assert re.fullmatch(r"[a-f0-9]{64}", result["citation"]["sha256"])
    else:
        assert exit_code == 1 and result["status"] == "failed" and result["failed_check"] == failure
        if scenario in {"dns", "wrong_mode"}:
            assert requests == ["/health"]

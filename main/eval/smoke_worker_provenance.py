"""Check GPU Finetune checkpoints and stored citations with a synthetic research run."""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from urllib.parse import urlsplit

import fitz
import httpx
from dotenv import dotenv_values

MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from core.citation_audit import claim_text, evidence_quote, get_citation_audit, published_claim_id
from core.config import RunConfig, settings
from core.state import compute_context_hash
from utils.logger import logger


QUESTION = "How many days did the irrigation pilot last?"
EVIDENCE = "The irrigation pilot lasted 21 days."
DOCUMENT_ID = "worker-smoke-paper"
_DURATION = re.compile(r"\b21[ -]+days?\b", re.IGNORECASE)


class SmokeFailure(RuntimeError):
    """A failed smoke check with a safe label for the JSON result."""


def _gpu_count(endpoint: str) -> int:
    from utils.llm_client import llm_client

    try:
        response = llm_client.http_client.get(endpoint + "/health", timeout=10, follow_redirects=False)
    except httpx.RequestError as exc:
        raise SmokeFailure("worker_health_unreachable") from exc
    if response.status_code != 200:
        raise SmokeFailure(f"worker_health_http_{response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise SmokeFailure("worker_health_invalid") from exc
    if not isinstance(data, dict):
        raise SmokeFailure("worker_health_invalid")
    if data.get("active_mode") not in {None, "gpu_finetune"} or data.get("is_switching"):
        raise SmokeFailure("worker_mode_not_ready")
    worker = data.get("worker_health", data)
    if not isinstance(worker, dict) or worker.get("pipeline") != "gpu_finetune_7_models":
        raise SmokeFailure("worker_pipeline_not_ready")
    gpus = worker.get("gpus_available")
    if type(gpus) is not int or gpus < 1:
        raise SmokeFailure("worker_gpu_unavailable")
    adapters, models = worker.get("loaded_adapters"), worker.get("active_models")
    if not isinstance(adapters, list) or "generator" not in adapters or "critic" not in adapters:
        raise SmokeFailure("worker_qwen_adapters_unavailable")
    if not isinstance(models, list) or "factcheck" not in models:
        raise SmokeFailure("worker_nli_checkpoint_unavailable")
    return gpus


def _source_citation(record: dict, sha256: str) -> dict:
    audit = get_citation_audit(record["reasoning_trace"])
    evidence_ids = set()
    for index, claim in enumerate(record["verified_claims"], start=1):
        if str(claim.get("status", "")).upper() == "SUPPORT" and _DURATION.search(claim_text(claim)):
            check = audit.get("claims", {}).get(published_claim_id(claim, index), {})
            if check.get("status") == "verified":
                evidence_ids.update(check["evidence_ids"])
    spans = {span["id"]: span for span in record["evidence_spans"]}
    for ref in audit.get("references") or []:
        eid = ref["evidence_id"]
        if eid not in evidence_ids or ref.get("catalog_id") != DOCUMENT_ID:
            continue
        entry = audit.get("evidence_provenance", {}).get(eid, {})
        unit = entry.get("source_unit", {})
        quote = evidence_quote(spans.get(eid, {}))
        document = entry.get("document", {})
        if (
            entry.get("status") == "linked" and document.get("id") == DOCUMENT_ID
            and document.get("sha256") == sha256
            and ref.get("page") == unit.get("page_no") == 1
            and _DURATION.search(quote) and quote in EVIDENCE and quote in unit.get("text", "")
            and unit.get("context_hash") == compute_context_hash(unit["text"], unit["doc_id"], 1)
        ):
            return {"sha256": sha256, "page": 1, "quote": quote}
    raise SmokeFailure("verified_citation_not_linked")


async def _research_smoke(endpoint: str, folder: Path) -> dict:
    from api.citations import file_sha256
    from api.research_runs import ResearchRequest, ResearchRunService
    from api.run_manifest import get_run_manifest
    from core import db

    path = folder / "irrigation-pilot.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), EVIDENCE)
        pdf.save(path, no_new_id=True)
    sha256 = file_sha256(path)
    if sha256 is None:
        raise SmokeFailure("fixture_unreadable")
    db.save_document(DOCUMENT_ID, path.name, str(path), path.stat().st_size)
    config = RunConfig.capture(remote_inference_url=endpoint, corpus_path=str(folder / "unused.jsonl"))
    request = ResearchRequest(
        query=QUESTION, config=config, mode="gpu_finetune", document_ids=(DOCUMENT_ID,), max_reflection_loops=0,
    )
    run = await ResearchRunService().start(request)
    await asyncio.wait_for(run.result(), timeout=240)
    record = db.get_run(run.record["id"])
    if not record or record["status"] != "completed":
        raise SmokeFailure("run_not_completed")
    manifest = get_run_manifest(record)
    if manifest is None or manifest["documents"] != [{"id": DOCUMENT_ID, "name": path.name, "sha256": sha256}]:
        raise SmokeFailure("stored_document_manifest_mismatch")
    observations = manifest["observed"]
    generator = next((
        event for event in observations["worker_inference"]
        if event["status"] == "reported" and event["endpoint"] == "/api/summarize/generator"
        and event["component"] == "generator"
    ), None)
    if not generator or observations["generator_revision"] != generator["revision"]:
        raise SmokeFailure("generator_checkpoint_unconfirmed")
    if not observations["verifier_revision"]:
        raise SmokeFailure("nli_checkpoint_unconfirmed")
    if not _DURATION.search(record["final_report"] or ""):
        raise SmokeFailure("expected_duration_missing_from_report")
    return {
        "generator": {
            "model": generator["model"], "revision": generator["revision"],
            "adapter_sha256": generator["adapter_sha256"],
        },
        "nli_revision": observations["verifier_revision"],
        "citation": _source_citation(record, sha256),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="Read REMOTE_INFERENCE_URL from this dotenv file.")
    args = parser.parse_args(argv)

    values = dotenv_values(args.env_file) if args.env_file else os.environ
    endpoint = (values.get("REMOTE_INFERENCE_URL") or "").strip().rstrip("/")
    result = {"test": "gpu-finetune-provenance", "status": "failed"}
    previous_logging = logger.disabled
    logger.disabled = True
    try:
        try:
            url = urlsplit(endpoint)
        except ValueError as exc:
            raise SmokeFailure("worker_endpoint_invalid") from exc
        if url.scheme not in {"http", "https"} or not url.hostname or url.query or url.fragment:
            raise SmokeFailure("worker_endpoint_invalid")
        with TemporaryDirectory(prefix="rag-worker-smoke-") as temporary:
            folder = Path(temporary).resolve()
            if folder.parent != Path(gettempdir()).resolve():
                raise SmokeFailure("temporary_directory_invalid")
            existing_db = sys.modules.get("core.db")
            previous_base = settings.BASE_DIR
            previous_db = (existing_db.DB_PATH, existing_db._db_initialized) if existing_db else (
                str(Path(previous_base) / "data" / "workspace.db"), False,
            )
            # core.db initializes on import. Point that first import at scratch space too.
            settings.BASE_DIR = str(folder)
            try:
                from core import db
            finally:
                settings.BASE_DIR = previous_base
            db.DB_PATH, db._db_initialized = str(folder / "data" / "workspace.db"), False
            try:
                result["gpus_available"] = _gpu_count(endpoint)
                result.update(asyncio.run(_research_smoke(endpoint, folder)))
            finally:
                db.DB_PATH, db._db_initialized = previous_db
        result["status"] = "passed"
    except SmokeFailure as exc:
        result["failed_check"] = str(exc)
    except TimeoutError:
        result["failed_check"] = "run_timed_out"
    finally:
        logger.disabled = previous_logging
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

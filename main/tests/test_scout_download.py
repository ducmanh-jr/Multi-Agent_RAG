from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

from agents.agent0_scout import downloader
from core.config import settings


def test_simultaneous_scouts_use_distinct_partial_files(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(downloader, "save_document", lambda **kwargs: None)
    barrier = Barrier(2)
    partial_counts = []
    uploads = tmp_path / "uploads"
    pdf = b"%PDF-1.4\n" + b"evidence" * 1600
    replace = downloader.os.replace
    publish_lock = Lock()

    def replace_with_locked_destination(source, destination):
        with publish_lock:
            if Path(destination).exists():
                raise PermissionError("destination PDF is open")
            replace(source, destination)

    monkeypatch.setattr(downloader.os, "replace", replace_with_locked_destination)

    def chunks(chunk_size):
        barrier.wait(timeout=5)
        partial_counts.append(len(list(uploads.glob("*.part_*"))))
        barrier.wait(timeout=5)
        yield pdf

    response = SimpleNamespace(raise_for_status=lambda: None, iter_bytes=chunks)
    client = SimpleNamespace(stream=lambda *a: nullcontext(response))
    monkeypatch.setattr(downloader.httpx, "Client", lambda **kwargs: nullcontext(client))
    paper = {"title": "Simultaneous Discovery", "arxiv_id": "test", "pdf_url": "https://source.invalid/paper.pdf"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(downloader.download_and_register_paper, paper)
        second = pool.submit(downloader.download_and_register_paper, paper)
        target = first.result()["local_pdf_path"]
        assert second.result()["local_pdf_path"] == target
    assert partial_counts == [2, 2]
    assert Path(target).read_bytes() == pdf
    assert not list(uploads.glob("*.part_*"))

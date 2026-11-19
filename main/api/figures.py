"""Bind figures selected by the graph to the PDFs selected for this run."""

import os
import re
from urllib.parse import quote


_FIGURE_ID = re.compile(r"fig-[0-9a-f]{24}-p([1-9]\d*)-f[1-9]\d*")


def _path_key(path):
    return os.path.normcase(os.path.realpath(path))


def pin_answer_figures(trace: list, documents: list, selected_paths: list) -> None:
    """Replace graph-local paths with catalog IDs before exposing or saving a trace."""
    events = [entry for entry in trace if entry.get("action") == "answer_figures"]
    if not events:
        return
    selected = {_path_key(path) for path in selected_paths if path}
    by_path = {}
    for doc in documents:
        path = doc.get("path")
        if path:
            key = _path_key(path)
            if key in selected:
                by_path.setdefault(key, []).append(doc)
    for event in events:
        pinned = []
        seen = set()
        for ref in event.get("figures") or []:
            if not isinstance(ref, dict) or not isinstance(ref.get("source_path"), str):
                continue
            match = _FIGURE_ID.fullmatch(str(ref.get("figure_id", "")))
            page = ref.get("page_no")
            if not match or type(page) is not int or str(page) != match[1]:
                continue
            matches = by_path.get(_path_key(ref["source_path"]), [])
            if len(matches) != 1:
                continue
            doc = matches[0]
            key = (doc["id"], ref["figure_id"])
            if key in seen:
                continue
            seen.add(key)
            pinned.append({
                "figure_id": ref["figure_id"], "document_id": doc["id"],
                "document_name": doc.get("name") or doc["id"], "page": page,
                "label": str(ref.get("label") or ""), "caption": str(ref.get("caption") or ""),
            })
        event["figures"] = pinned


def answer_figures(trace: list) -> list[dict]:
    for event in reversed(trace):
        if event.get("action") == "answer_figures":
            return [
                {
                    **ref,
                    "image_url": f"/api/documents/{quote(ref['document_id'], safe='')}/figures/{ref['figure_id']}",
                }
                for ref in event.get("figures") or []
                if isinstance(ref, dict) and isinstance(ref.get("document_id"), str)
                and _FIGURE_ID.fullmatch(str(ref.get("figure_id", "")))
            ]
    return []

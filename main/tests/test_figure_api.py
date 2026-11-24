import json

from api.figures import answer_figures, pin_answer_figures


def test_only_figures_from_selected_catalog_paths_are_published(tmp_path):
    first, second = [tmp_path / folder / "same.pdf" for folder in ("a", "b")]
    docs = [{"id": key, "name": path.name, "path": str(path)} for key, path in (("first", first), ("second", second))]
    figure_id = "fig-0123456789abcdef01234567-p3-f1"
    ref = {"source_path": str(second), "figure_id": figure_id, "page_no": 3, "caption": "Chart B"}
    trace = [{"action": "answer_figures", "figures": [
        {**ref, "source_path": str(first)}, {**ref, "page_no": 2}, {**ref, "figure_id": "../other"},
        {**ref, "document_id": "first", "image_url": "https://example.invalid/other"}, ref,
    ]}]

    pin_answer_figures(trace, docs, [str(second)])
    assert str(first) not in str(trace) and str(second) not in str(trace)
    restored = answer_figures(json.loads(json.dumps(trace)))
    assert restored == [{
        "document_id": "second", "document_name": "same.pdf", "figure_id": figure_id,
        "page": 3, "caption": "Chart B", "label": "",
        "image_url": "/api/documents/second/figures/" + figure_id,
    }]


def test_older_runs_and_latest_answer_without_figures():
    assert answer_figures([{"action": "citation_audit"}]) == []
    trace = [
        {"action": "answer_figures", "figures": [{"document_id": "old", "figure_id": "fig-0123456789abcdef01234567-p1-f1"}]},
        {"action": "answer_figures", "figures": []},
    ]
    assert answer_figures(trace) == []

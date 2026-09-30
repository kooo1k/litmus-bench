"""The public docs only cite files that exist. SUBMISSION.md and the README
point judges at evidence by path; a moved or renamed file must fail here,
not in front of a judge."""

import re

import pytest

from conftest import ROOT

DOCS = ["README.md", "SUBMISSION.md", "FINDINGS.md", "docs/results.md", "docs/architecture.md", "docs/tasks.md"]
CITED = re.compile(r"`((?:results|runs|tasks|litmus|scripts|docs|tests)/[^`*<>]+?)`")


@pytest.mark.parametrize("doc", DOCS)
def test_every_cited_path_exists(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = []
    for path in CITED.findall(text):
        path = path.split(" ")[0].rstrip(".,;:)")
        if any(ch in path for ch in "{}") or path.endswith("/<run_id>/"):
            continue  # a pattern, not a file
        if not (ROOT / path).exists():
            missing.append(path)
    assert not missing, f"{doc} cites paths that don't exist: {missing}"


IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


@pytest.mark.parametrize("doc", DOCS)
def test_every_image_exists(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    base = (ROOT / doc).parent  # markdown links are relative to the document
    missing = [p for p in IMAGE.findall(text) if not p.startswith("http") and not (base / p).exists()]
    assert not missing, f"{doc} shows images that don't exist: {missing}"


@pytest.mark.parametrize("name, verdict", [("verdict.svg", "FAIL"), ("fix.svg", "PASS"), ("freshness.svg", "PASS")])
def test_doc_images_render_from_saved_runs(tmp_path, monkeypatch, name, verdict):
    """Every doc image is regenerated from a committed trial, not drawn by hand."""
    from scripts import render_readme_image
    monkeypatch.setattr(render_readme_image, "OUT_DIR", tmp_path)
    render_readme_image.main([name])
    svg = (tmp_path / name).read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "litmus&#160;--replay" in svg and verdict in svg


REPLAY = re.compile(r"`litmus --replay ([^`]+)`")


@pytest.mark.parametrize("doc", ["README.md", "SUBMISSION.md", "docs/results.md"])
def test_every_documented_replay_command_resolves(doc):
    """A judge copying a `litmus --replay run task trial` from the docs must
    land on a saved, scored trial."""
    from litmus import cli
    for args in REPLAY.findall((ROOT / doc).read_text(encoding="utf-8")):
        if "<" in args:
            continue  # a template like results/<run> <task> <trial>, not a run
        run, task_id, trial = args.split()
        record, task, _ = cli.load_saved_trial(run, task_id, int(trial))
        assert "score" in record and task["id"] == task_id

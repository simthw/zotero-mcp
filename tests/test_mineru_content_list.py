"""Regression tests for the local MinerU Markdown (.md) sidecar extension."""

from pathlib import Path

from zotero_mcp.local_db import LocalZoteroReader


class FakeMinerUReader(LocalZoteroReader):
    """Exercise extraction ordering without opening a real Zotero database."""

    def __init__(
        self,
        attachments,
        *,
        cache_text="cached Zotero text",
        extracted_text="direct PDF text",
        scanned_path=None,
    ):
        self._attachments = list(attachments)
        self._cache_text = cache_text
        self._extracted_text = extracted_text
        self._scanned_path = scanned_path
        self.attachment_priority = ("pdf", "html", "file")
        self.pdf_max_pages = 50

    def _iter_parent_attachments(self, _parent_item_id):
        yield from self._attachments

    def _resolve_attachment_path(self, _attachment_key, path):
        return Path(path) if path else None

    def _scan_storage_for_attachment(self, _attachment_key, _ctype):
        return self._scanned_path

    def _read_zotero_ft_cache(self, _attachment_key):
        return self._cache_text

    def _extract_text_from_file(self, _file_path):
        return self._extracted_text


def test_mineru_md_extraction_returns_plain_markdown(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text("# Title\n\nParagraph with $x^2$ math.", encoding="utf-8")
    reader = FakeMinerUReader([])

    assert reader._extract_text_from_mineru_md(md) == (
        "# Title\n\nParagraph with $x^2$ math."
    )


def test_mineru_md_has_highest_priority(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    md = tmp_path / "paper.md"
    md.write_text("# Markdown output\n\nSome text.", encoding="utf-8")
    reader = FakeMinerUReader(
        [("ATTACH01", str(pdf), "application/pdf")],
        cache_text="cached fallback",
    )

    assert reader.extract_fulltext_for_item(1) == (
        "# Markdown output\n\nSome text.",
        "mineru_md",
    )


def test_invalid_mineru_md_falls_back_to_zotero_cache(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "paper.md").write_bytes(b"\xff\xfe invalid utf-8")
    reader = FakeMinerUReader(
        [("ATTACH01", str(pdf), "application/pdf")],
        cache_text="cached fallback",
        extracted_text="",
    )

    assert reader.extract_fulltext_for_item(1) == (
        "cached fallback",
        "zotero-cache",
    )


def test_mineru_discovery_uses_attachment_scan_fallback(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "paper.md").write_text("scanned output", encoding="utf-8")
    missing_recorded_path = tmp_path / "renamed.pdf"
    reader = FakeMinerUReader(
        [("ATTACH01", str(missing_recorded_path), "application/pdf")],
        scanned_path=pdf,
    )

    assert reader.has_mineru_output(1) is True
    assert reader.get_mineru_output_path(1) == tmp_path / "paper.md"


def test_mineru_output_prefers_same_directory_md(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "paper.md").write_text("same-dir output", encoding="utf-8")
    (tmp_path / "auto").mkdir()
    (tmp_path / "auto" / "paper.md").write_text("sub-dir output", encoding="utf-8")
    reader = FakeMinerUReader([("ATTACH01", str(pdf), "application/pdf")])

    assert reader.get_mineru_output_path(1) == tmp_path / "paper.md"


"""Regression tests for the local MinerU content_list.json extension."""

import json
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


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_nested_mineru_blocks_preserve_text_and_math(tmp_path):
    content_list = tmp_path / "paper_content_list.json"
    _write_json(
        content_list,
        [
            [
                {
                    "type": "title",
                    "content": {
                        "title_content": [
                            {"type": "text", "content": "A structured title"}
                        ]
                    },
                },
                {
                    "type": "equation_interline",
                    "content": {"math_content": "E = mc^2", "math_type": "latex"},
                },
                {
                    "type": "page_header",
                    "content": {
                        "paragraph_content": [
                            {"type": "text", "content": "Repeated header"}
                        ]
                    },
                },
            ]
        ],
    )
    reader = FakeMinerUReader([])

    assert reader._extract_text_from_content_list_json(content_list) == (
        "A structured title\n\n$$ E = mc^2 $$"
    )


def test_flat_mineru_format_is_supported(tmp_path):
    content_list = tmp_path / "content_list.json"
    _write_json(
        content_list,
        [
            {"type": "text", "text": "First paragraph"},
            {"type": "text", "text": "Second paragraph"},
        ],
    )
    reader = FakeMinerUReader([])

    assert reader._extract_text_from_content_list_json(content_list) == (
        "First paragraph\n\nSecond paragraph"
    )


def test_document_specific_mineru_output_has_highest_priority(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _write_json(tmp_path / "content_list.json", [{"text": "generic output"}])
    _write_json(
        tmp_path / "paper_content_list.json",
        [{"text": "document-specific output"}],
    )
    reader = FakeMinerUReader(
        [("ATTACH01", str(pdf), "application/pdf")],
        cache_text="cached fallback",
    )

    assert reader.extract_fulltext_for_item(1) == (
        "document-specific output",
        "content_list_json",
    )


def test_invalid_mineru_output_falls_back_to_zotero_cache(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    (tmp_path / "paper_content_list.json").write_text("{invalid", encoding="utf-8")
    reader = FakeMinerUReader(
        [("ATTACH01", str(pdf), "application/pdf")],
        cache_text="cached fallback",
    )

    assert reader.extract_fulltext_for_item(1) == (
        "cached fallback",
        "zotero-cache",
    )


def test_mineru_discovery_uses_attachment_scan_fallback(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    _write_json(tmp_path / "paper_content_list.json", [{"text": "scanned output"}])
    missing_recorded_path = tmp_path / "renamed.pdf"
    reader = FakeMinerUReader(
        [("ATTACH01", str(missing_recorded_path), "application/pdf")],
        scanned_path=pdf,
    )

    assert reader.has_content_list_json(1) is True
    assert reader.get_content_list_json_path(1) == tmp_path / "paper_content_list.json"

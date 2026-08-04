"""Tests for retrying fulltext extraction on previously-failed items.

A "failed" has_fulltext marker must clear when the item's attachment set
changes (e.g. a PDF is attached via zotero_attach_file to an item that was
indexed metadata-only). Attaching a file does NOT bump the parent's
dateModified, so the date check alone can never trigger the retry.
"""

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search

DATE_MODIFIED = "2026-07-02 01:01:48"


class FakeItem:
    def __init__(self):
        self.item_id = 1
        self.key = "ITEMKEY1"
        self.item_type = "journalArticle"
        self.title = "Calibration"
        self.abstract = ""
        self.extra = ""
        self.doi = None
        self.notes = None
        self.creators = None
        self.date_added = "2026-07-01 00:00:00"
        self.date_modified = DATE_MODIFIED
        self.fulltext = None
        self.fulltext_source = None


class FakeReader:
    """Minimal LocalZoteroReader stand-in for the extraction scan."""

    def __init__(
        self,
        *args,
        attachments=(),
        mineru_available=False,
        extracted_source="pdf",
        **kwargs,
    ):
        self._attachments = list(attachments)
        self._mineru_available = mineru_available
        self._extracted_source = extracted_source
        self.extract_calls = 0
        self.attachment_priority = ("pdf", "html", "file")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_all_item_keys(self):
        return {"ITEMKEY1"}

    def get_key_group_map(self):
        # Fixed result rather than a real query — this file tests
        # failed-marker retry, not library attribution.
        return ({"ITEMKEY1": 0}, set())

    def get_items_with_text(self, limit=None, include_fulltext=False, key_filter=None, collection_keys=None):
        return [FakeItem()]

    def get_fulltext_meta_for_item(self, item_id):
        return [list(row) for row in self._attachments]

    def has_content_list_json(self, item_id):
        return self._mineru_available

    def extract_fulltext_for_item(self, item_id):
        self.extract_calls += 1
        return ("extracted text", self._extracted_source)


class FakeChromaClient:
    def __init__(self, stored_metadata):
        self.embedding_max_tokens = 8000
        self._stored = stored_metadata

    def get_document_metadata(self, key):
        return self._stored

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def reset_collection(self):
        pass


def _run_scan(
    monkeypatch,
    stored_metadata,
    attachments,
    *,
    mineru_available=False,
    extracted_source="pdf",
):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    monkeypatch.setattr(semantic_search, "is_local_mode", lambda: True)

    reader = FakeReader(
        attachments=attachments,
        mineru_available=mineru_available,
        extracted_source=extracted_source,
    )
    monkeypatch.setattr(
        semantic_search, "LocalZoteroReader", lambda *a, **kw: reader
    )

    chroma = FakeChromaClient(stored_metadata)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    # A fake-induced error would silently fall back to the API path — fail loudly instead
    monkeypatch.setattr(
        search,
        "_get_items_from_api",
        lambda *a, **kw: pytest.fail("unexpected fallback to API path"),
    )

    items = search._get_items_from_source(
        extract_fulltext=True, chroma_client=chroma, force_rebuild=False
    )
    return items, reader


def test_failed_item_skipped_when_nothing_changed(monkeypatch):
    """failed marker + same date + same attachment set -> no retry."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "group_id": 0,
        "attachment_keys": "",
    }
    items, reader = _run_scan(monkeypatch, stored, attachments=[])
    assert items == []
    assert reader.extract_calls == 0


def test_failed_item_retried_when_attachment_added(monkeypatch):
    """failed marker + same date, but a PDF was attached since -> retry."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "group_id": 0,
        "attachment_keys": "",
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    items, reader = _run_scan(monkeypatch, stored, attachments=attachments)
    assert len(items) == 1
    assert reader.extract_calls == 1
    data = items[0]["data"]
    assert data["fulltext"] == "extracted text"
    assert data["attachmentKeys"] == "ATTKEY1"
    # And the resulting metadata records the new attachment set
    metadata = semantic_search.ZoteroSemanticSearch(
        chroma_client=FakeChromaClient({})
    )._create_metadata(items[0])
    assert metadata["has_fulltext"] is True
    assert metadata["attachment_keys"] == "ATTKEY1"


def test_failed_item_retried_when_date_modified_changed(monkeypatch):
    """failed marker + same attachment set, but the item was edited -> retry.

    Pins the pre-existing dateModified trigger ("user replaces a bad PDF and
    the parent gets re-saved") so it survives the new attachment-set check.
    """
    stored = {
        "has_fulltext": "failed",
        "date_modified": "2026-06-30 12:00:00",  # differs from the item's
        "group_id": 0,
        "attachment_keys": "",
    }
    items, reader = _run_scan(monkeypatch, stored, attachments=[])
    assert len(items) == 1
    assert reader.extract_calls == 1


def test_legacy_failed_record_without_attachment_keys_retries_once(monkeypatch):
    """Records written before attachment_keys existed retry once, then converge."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "group_id": 0,
        # no attachment_keys field (legacy)
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    items, reader = _run_scan(monkeypatch, stored, attachments=attachments)
    assert len(items) == 1
    assert reader.extract_calls == 1


def test_failed_item_retried_when_mineru_output_appears(monkeypatch):
    """A new MinerU sidecar clears a failed marker without Zotero metadata changes."""
    stored = {
        "has_fulltext": "failed",
        "date_modified": DATE_MODIFIED,
        "group_id": 0,
        "attachment_keys": "ATTKEY1",
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments=attachments,
        mineru_available=True,
        extracted_source="content_list_json",
    )

    assert len(items) == 1
    assert reader.extract_calls == 1
    assert items[0]["data"]["fulltextSource"] == "content_list_json"


def test_existing_pdf_fulltext_is_upgraded_to_mineru(monkeypatch):
    """MinerU remains preferred over an already-indexed lower-quality PDF."""
    stored = {
        "has_fulltext": True,
        "fulltext_source": "pdf",
        "date_modified": DATE_MODIFIED,
        "group_id": 0,
        "attachment_keys": "ATTKEY1",
    }
    attachments = [("ATTKEY1", "storage:paper.pdf", "application/pdf")]
    items, reader = _run_scan(
        monkeypatch,
        stored,
        attachments=attachments,
        mineru_available=True,
        extracted_source="content_list_json",
    )

    assert len(items) == 1
    assert reader.extract_calls == 1
    assert items[0]["data"]["fulltextSource"] == "content_list_json"

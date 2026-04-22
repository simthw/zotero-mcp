import sqlite3
from pathlib import Path

import pytest

from zotero_mcp.local_db import ZoteroItem, LocalZoteroReader


class FakeLocalZoteroReader(LocalZoteroReader):
    """Subclass that skips DB init and allows injecting fake attachment text."""

    def __init__(self, fake_text: str = "", fake_pdf_path: Path | None = None):
        # Skip parent __init__ entirely — no DB needed
        self.db_path = "/dev/null"
        self._connection = None
        self.pdf_max_pages = 10
        self.pdf_timeout = 30
        self._fake_text = fake_text
        self._fake_pdf_path = fake_pdf_path

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield a single fake PDF attachment."""
        yield "FAKEKEY", "storage:fake.pdf", "application/pdf"

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str):
        """Return the injected fake path."""
        return self._fake_pdf_path

    def _extract_text_from_file(self, file_path):
        """Return the injected fake text instead of reading a real file."""
        return self._fake_text


def test_extract_fulltext_preserves_long_text(tmp_path):
    """Extracted text longer than 10,000 chars should NOT be truncated."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    long_text = "x" * 25000
    reader = FakeLocalZoteroReader(fake_text=long_text, fake_pdf_path=fake_pdf)
    result = reader._extract_fulltext_for_item(1)
    assert result is not None
    text, source = result
    assert len(text) == 25000, f"Text was truncated to {len(text)} chars"
    assert source == "pdf"


def test_extract_fulltext_empty_returns_none(tmp_path):
    """Empty extracted text should return None."""
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    reader = FakeLocalZoteroReader(fake_text="", fake_pdf_path=fake_pdf)
    result = reader._extract_fulltext_for_item(1)
    assert result is None


def test_get_searchable_text_preserves_long_fulltext():
    """get_searchable_text should not aggressively truncate fulltext."""
    long_fulltext = "y" * 20000
    item = ZoteroItem(item_id=1, key="TEST", item_type_id=1, fulltext=long_fulltext)
    text = item.get_searchable_text()
    # The full 20,000 chars should appear in the output (not truncated to 5,000)
    assert "y" * 20000 in text


def test_get_searchable_text_truncates_at_limit():
    """Fulltext beyond 50,000 chars should be truncated with ellipsis."""
    huge_fulltext = "z" * 60000
    item = ZoteroItem(item_id=1, key="TEST", item_type_id=1, fulltext=huge_fulltext)
    text = item.get_searchable_text()
    # Should contain exactly 50,000 z's plus "..." — not all 60,000
    assert "z" * 50000 in text
    assert "z" * 50001 not in text
    assert "..." in text


class TestResolveAttachmentPath:
    """Tests for _resolve_attachment_path handling of various Zotero path formats."""

    def _make_reader(self, tmp_path):
        """Create a LocalZoteroReader-like object for path resolution tests.

        Uses the real _resolve_attachment_path (not the fake override).
        """
        reader = FakeLocalZoteroReader()
        reader.db_path = str(tmp_path / "zotero.sqlite")
        # Bind the real method so we test actual path resolution
        reader._resolve_attachment_path = LocalZoteroReader._resolve_attachment_path.__get__(reader)
        reader._get_storage_dir = LocalZoteroReader._get_storage_dir.__get__(reader)
        reader._get_base_attachment_path = LocalZoteroReader._get_base_attachment_path.__get__(reader)
        return reader

    def test_storage_path(self, tmp_path):
        """'storage:file.pdf' resolves to <storage_dir>/<key>/file.pdf."""
        reader = self._make_reader(tmp_path)
        (tmp_path / "storage" / "ABC123").mkdir(parents=True)
        result = reader._resolve_attachment_path("ABC123", "storage:paper.pdf")
        assert result == tmp_path / "storage" / "ABC123" / "paper.pdf"

    def test_absolute_path(self, tmp_path):
        """Absolute path passes through unchanged."""
        reader = self._make_reader(tmp_path)
        result = reader._resolve_attachment_path("X", "/home/user/papers/file.pdf")
        assert result == Path("/home/user/papers/file.pdf")

    def test_file_url(self, tmp_path):
        """'file:///path/to/file.pdf' resolves to the decoded path."""
        reader = self._make_reader(tmp_path)
        result = reader._resolve_attachment_path("X", "file:///home/user/my%20paper.pdf")
        assert result == Path("/home/user/my paper.pdf")

    def test_attachments_with_base_path(self, tmp_path):
        """'attachments:rel/path.pdf' resolves against baseAttachmentPath from prefs.js."""
        reader = self._make_reader(tmp_path)
        base_dir = tmp_path / "linked_papers"
        base_dir.mkdir()
        # Write a prefs.js with baseAttachmentPath
        prefs = tmp_path / "prefs.js"
        prefs.write_text(
            f'user_pref("extensions.zotero.baseAttachmentPath", "{base_dir}");\n'
        )
        result = reader._resolve_attachment_path("X", "attachments:subfolder/paper.pdf")
        assert result == base_dir / "subfolder" / "paper.pdf"

    def test_attachments_without_base_path_returns_none(self, tmp_path):
        """'attachments:' path returns None when no baseAttachmentPath is configured."""
        reader = self._make_reader(tmp_path)
        # No prefs.js exists
        result = reader._resolve_attachment_path("X", "attachments:subfolder/paper.pdf")
        assert result is None

    def test_empty_path_returns_none(self, tmp_path):
        """Empty/None path returns None."""
        reader = self._make_reader(tmp_path)
        assert reader._resolve_attachment_path("X", "") is None
        assert reader._resolve_attachment_path("X", None) is None

    def test_unknown_prefix_returns_none(self, tmp_path):
        """Unknown path format returns None."""
        reader = self._make_reader(tmp_path)
        assert reader._resolve_attachment_path("X", "ftp://something") is None


class TestFetchCreatorsForItems:
    """Tests for _fetch_creators_for_items ordering and type filtering.

    These tests build a minimal Zotero-schema SQLite fixture in-memory, so
    they validate the actual SQL (ORDER BY orderIndex, creatorType filter)
    without requiring a real Zotero installation.
    """

    def _make_conn(self) -> sqlite3.Connection:
        """Build an in-memory DB with the subset of Zotero's schema we query."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE creators (
                creatorID INTEGER PRIMARY KEY,
                firstName TEXT,
                lastName TEXT,
                fieldMode INTEGER DEFAULT 0
            );
            CREATE TABLE creatorTypes (
                creatorTypeID INTEGER PRIMARY KEY,
                creatorType TEXT NOT NULL
            );
            CREATE TABLE itemCreators (
                itemID INTEGER NOT NULL,
                creatorID INTEGER NOT NULL,
                creatorTypeID INTEGER NOT NULL,
                orderIndex INTEGER NOT NULL,
                PRIMARY KEY (itemID, creatorID, creatorTypeID)
            );
            INSERT INTO creatorTypes (creatorTypeID, creatorType) VALUES
                (1, 'author'),
                (2, 'editor'),
                (3, 'translator');
            """
        )
        return conn

    def _make_reader(self, conn: sqlite3.Connection) -> LocalZoteroReader:
        reader = FakeLocalZoteroReader()
        reader._connection = conn
        # Bind the real methods we want to test
        reader._fetch_creators_for_items = (
            LocalZoteroReader._fetch_creators_for_items.__get__(reader)
        )
        reader._get_connection = lambda: conn
        return reader

    def test_preserves_order_index(self):
        """Creators must come back in orderIndex order, not insertion order."""
        conn = self._make_conn()
        # Insert creators and itemCreators rows in a scrambled order —
        # a buggy implementation that relies on row order would fail.
        conn.executescript(
            """
            INSERT INTO creators VALUES (10, 'Robert A.', 'Spicer', 0);
            INSERT INTO creators VALUES (20, 'Hong',      'Wang',   0);
            INSERT INTO creators VALUES (30, 'De-Zhu',    'Li',     0);
            INSERT INTO creators VALUES (40, 'Jie',       'Liu',    0);

            -- Deliberately inserted out of orderIndex sequence:
            INSERT INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex)
                VALUES (100, 30, 1, 13);  -- Li, De-Zhu     (last author)
            INSERT INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex)
                VALUES (100, 20, 1, 0);   -- Wang, Hong     (first author)
            INSERT INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex)
                VALUES (100, 40, 1, 3);   -- Liu, Jie
            INSERT INTO itemCreators (itemID, creatorID, creatorTypeID, orderIndex)
                VALUES (100, 10, 1, 1);   -- Spicer, Robert A.
            """
        )
        conn.commit()

        reader = self._make_reader(conn)
        result = reader._fetch_creators_for_items([100])

        assert result == {
            100: "Wang, Hong; Spicer, Robert A.; Liu, Jie; Li, De-Zhu"
        }

    def test_filters_to_authors_by_default(self):
        """Editors and translators must be excluded when caller requests authors."""
        conn = self._make_conn()
        conn.executescript(
            """
            INSERT INTO creators VALUES (1, 'Alice',  'Author', 0);
            INSERT INTO creators VALUES (2, 'Bob',    'Editor', 0);
            INSERT INTO creators VALUES (3, 'Carol',  'Trans',  0);

            INSERT INTO itemCreators VALUES (200, 1, 1, 0);  -- author
            INSERT INTO itemCreators VALUES (200, 2, 2, 1);  -- editor (drop)
            INSERT INTO itemCreators VALUES (200, 3, 3, 2);  -- translator (drop)
            """
        )
        conn.commit()

        reader = self._make_reader(conn)
        result = reader._fetch_creators_for_items([200])

        assert result == {200: "Author, Alice"}

    def test_institutional_single_field_name(self):
        """fieldMode=1 means lastName holds the full institutional name."""
        conn = self._make_conn()
        conn.executescript(
            """
            INSERT INTO creators VALUES
                (1, NULL, 'World Health Organization', 1);
            INSERT INTO itemCreators VALUES (300, 1, 1, 0);
            """
        )
        conn.commit()

        reader = self._make_reader(conn)
        result = reader._fetch_creators_for_items([300])

        assert result == {300: "World Health Organization"}

    def test_multiple_items_grouped_separately(self):
        """Each itemID should get its own ordered creator list."""
        conn = self._make_conn()
        conn.executescript(
            """
            INSERT INTO creators VALUES (1, 'A', 'Alpha', 0);
            INSERT INTO creators VALUES (2, 'B', 'Beta',  0);
            INSERT INTO creators VALUES (3, 'C', 'Gamma', 0);

            INSERT INTO itemCreators VALUES (1, 1, 1, 0);
            INSERT INTO itemCreators VALUES (1, 2, 1, 1);
            INSERT INTO itemCreators VALUES (2, 3, 1, 0);
            INSERT INTO itemCreators VALUES (2, 1, 1, 1);
            """
        )
        conn.commit()

        reader = self._make_reader(conn)
        result = reader._fetch_creators_for_items([1, 2])

        assert result == {
            1: "Alpha, A; Beta, B",
            2: "Gamma, C; Alpha, A",
        }

    def test_empty_input_returns_empty(self):
        """No item IDs → empty dict, no query executed."""
        conn = self._make_conn()
        reader = self._make_reader(conn)
        assert reader._fetch_creators_for_items([]) == {}

    def test_missing_item_absent_from_result(self):
        """Items with no creators should not appear in the result dict."""
        conn = self._make_conn()
        reader = self._make_reader(conn)
        # No itemCreators rows at all
        result = reader._fetch_creators_for_items([999])
        assert result == {}

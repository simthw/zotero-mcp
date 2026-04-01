"""
Local Zotero database reader for semantic search.

Provides direct SQLite access to Zotero's local database for faster semantic search
when running in local mode.
"""

import json
import os
import sqlite3
import platform
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from urllib.parse import urlparse, unquote

from .utils import is_local_mode

logger = logging.getLogger(__name__)

# Sentinel returned by _extract_text_from_pdf on timeout
_EXTRACTION_TIMEOUT = "__EXTRACTION_TIMEOUT__"


def _extract_pdf_worker(file_path: str, maxpages: int, result_queue=None):
    """Worker: extract text from a PDF in a separate process.

    When called via ProcessPoolExecutor (result_queue is None) the text is
    returned directly.  The legacy Queue path is kept for compatibility.
    """
    try:
        # Suppress pdfminer warnings about malformed PDF color spaces, fonts, etc.
        import logging as _logging
        _logging.getLogger("pdfminer").setLevel(_logging.ERROR)

        from pdfminer.high_level import extract_text
        text = extract_text(file_path, maxpages=maxpages) or ""
        if result_queue is not None:
            result_queue.put(text)
        return text
    except Exception:
        if result_queue is not None:
            result_queue.put("")
        return ""


@dataclass
class ZoteroItem:
    """Represents a Zotero item with text content for semantic search."""
    item_id: int
    key: str
    item_type_id: int
    item_type: str | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    creators: str | None = None
    fulltext: str | None = None
    fulltext_source: str | None = None  # 'pdf' or 'html'
    notes: str | None = None
    extra: str | None = None
    date_added: str | None = None
    date_modified: str | None = None

    def get_searchable_text(self) -> str:
        """
        Combine all text fields into a single searchable string.

        Returns:
            Combined text content for semantic search indexing.
        """
        parts = []

        if self.title:
            parts.append(f"Title: {self.title}")

        if self.creators:
            parts.append(f"Authors: {self.creators}")

        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")

        if self.extra:
            parts.append(f"Extra: {self.extra}")

        if self.notes:
            parts.append(f"Notes: {self.notes}")

        if self.fulltext:
            # Truncate very long fulltext for simple text search
            max_chars = 50000
            truncated_fulltext = self.fulltext[:max_chars] + "..." if len(self.fulltext) > max_chars else self.fulltext
            parts.append(f"Content: {truncated_fulltext}")

        return "\n\n".join(parts)


class LocalZoteroReader:
    """
    Direct SQLite reader for Zotero's local database.

    Provides fast access to item metadata and fulltext for semantic search
    without going through the Zotero API.
    """

    def __init__(self, db_path: str | None = None, pdf_max_pages: int | None = None, pdf_timeout: int = 30):
        """
        Initialize the local database reader.

        Args:
            db_path: Optional path to zotero.sqlite. If None, auto-detect.
            pdf_max_pages: Maximum pages to extract from PDFs.
            pdf_timeout: Seconds to wait for PDF extraction before killing the process.
        """
        self.db_path = db_path or self._find_zotero_db()
        self._connection: sqlite3.Connection | None = None
        self.pdf_max_pages: int | None = pdf_max_pages
        self.pdf_timeout: int = pdf_timeout
        self._pdf_pool = None  # Lazy-initialized ProcessPoolExecutor
        # Reduce noise from pdfminer warnings
        try:
            logging.getLogger("pdfminer").setLevel(logging.ERROR)
        except Exception:
            pass

    def _find_zotero_db(self) -> str:
        """
        Auto-detect the Zotero database location based on OS.

        Returns:
            Path to zotero.sqlite file.

        Raises:
            FileNotFoundError: If database cannot be located.
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
        elif system == "Windows":
            # Try Windows 7+ location first
            db_path = Path.home() / "Zotero" / "zotero.sqlite"
            if not db_path.exists():
                # Fallback to XP/2000 location
                db_path = Path(os.path.expanduser("~/Documents and Settings")) / os.getenv("USERNAME", "") / "Zotero" / "zotero.sqlite"
        else:  # Linux and others
            db_path = Path.home() / "Zotero" / "zotero.sqlite"

        if not db_path.exists():
            raise FileNotFoundError(
                f"Zotero database not found at {db_path}. "
                "Please ensure Zotero is installed and has been run at least once."
            )

        return str(db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._connection is None:
            # Use immutable=1 to bypass locking entirely. Zotero uses rollback
            # journal mode and holds a write lock while running, which blocks
            # even read-only connections. immutable=1 skips all lock checks —
            # safe here since we only read and tolerate slightly stale data.
            uri = f"file:{self.db_path}?immutable=1"
            self._connection = sqlite3.connect(uri, uri=True)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _get_storage_dir(self) -> Path:
        """Return the Zotero storage directory path based on database location."""
        # Infer storage directory from database path (same parent directory)
        db_parent = Path(self.db_path).parent
        return db_parent / "storage"

    def _get_base_attachment_path(self) -> Path | None:
        """Read the linked attachment base directory from Zotero's prefs.js.

        Returns the configured ``extensions.zotero.baseAttachmentPath`` or
        ``None`` if the preference is not set or the file cannot be read.
        """
        prefs_path = Path(self.db_path).parent / "prefs.js"
        if not prefs_path.exists():
            return None
        try:
            import re
            text = prefs_path.read_text(encoding="utf-8", errors="replace")
            m = re.search(
                r'user_pref\("extensions\.zotero\.baseAttachmentPath",\s*"([^"]+)"\)',
                text,
            )
            if m:
                return Path(m.group(1))
        except Exception:
            pass
        return None

    def _iter_parent_attachments(self, parent_item_id: int):
        """Yield tuples (attachment_key, path, content_type) for a parent item."""
        conn = self._get_connection()
        query = (
            """
            SELECT ia.itemID as attachmentItemID,
                   ia.parentItemID as parentItemID,
                   ia.path as path,
                   ia.contentType as contentType,
                   att.key as attachmentKey
            FROM itemAttachments ia
            JOIN items att ON att.itemID = ia.itemID
            WHERE ia.parentItemID = ?
            """
        )
        for row in conn.execute(query, (parent_item_id,)):
            yield row["attachmentKey"], row["path"], row["contentType"]

    def _resolve_attachment_path(self, attachment_key: str, zotero_path: str) -> Path | None:
        """Resolve a Zotero attachment path to a filesystem path.

        Handles four formats:
        - 'storage:filename.pdf' — Zotero-managed storage (most common)
        - 'file:///path/to/file.pdf' — linked file as URL
        - '/absolute/path/to/file.pdf' — linked file as absolute path
        - 'attachments:relative/path.pdf' — Zotero linked attachment base dir
        """
        if not zotero_path:
            return None

        storage_dir = self._get_storage_dir()

        # Zotero-managed storage: 'storage:filename.pdf'
        if zotero_path.startswith("storage:"):
            rel = zotero_path.split(":", 1)[1]
            parts = [p for p in rel.split("/") if p]
            return storage_dir / attachment_key / Path(*parts)

        # Linked file as URL: 'file:///path/to/file.pdf'
        if zotero_path.startswith("file://"):
            from urllib.parse import urlparse, unquote
            parsed = urlparse(zotero_path)
            decoded_path = unquote(parsed.path or "")
            # file:///C:/... on Windows
            if os.name == "nt" and decoded_path.startswith("/") and len(decoded_path) > 2 and decoded_path[2] == ":":
                decoded_path = decoded_path[1:]
            if not decoded_path:
                return None
            return Path(decoded_path)

        # Linked file as absolute path: '/Users/me/papers/file.pdf'
        if os.path.isabs(zotero_path):
            return Path(zotero_path)

        # Zotero 'attachments:' relative path — resolve against the linked
        # attachment base directory configured in Zotero preferences.
        if zotero_path.startswith("attachments:"):
            rel = zotero_path.split(":", 1)[1]
            parts = [p for p in rel.split("/") if p]
            base = self._get_base_attachment_path()
            if base and base.exists():
                return base / Path(*parts)
            # Fallback: cannot resolve without base path
            return None

        return None

    def _get_pdf_pool(self):
        """Lazily create a single-worker ProcessPoolExecutor for PDF extraction."""
        if self._pdf_pool is None:
            from concurrent.futures import ProcessPoolExecutor
            self._pdf_pool = ProcessPoolExecutor(max_workers=1)
        return self._pdf_pool

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF using pdfminer in a persistent worker process.

        Uses a single reusable worker process to avoid the per-PDF spawn
        overhead on Windows.  Returns the extracted text, empty string on
        failure, or _EXTRACTION_TIMEOUT sentinel on timeout.
        """
        from concurrent.futures import TimeoutError as FuturesTimeout

        # Page limit (preserve existing fallback chain)
        if isinstance(self.pdf_max_pages, int) and self.pdf_max_pages > 0:
            maxpages = self.pdf_max_pages
        else:
            max_pages_env = os.getenv("ZOTERO_PDF_MAXPAGES")
            try:
                maxpages = int(max_pages_env) if max_pages_env else 10
            except ValueError:
                maxpages = 10

        timeout = self.pdf_timeout or 30

        try:
            pool = self._get_pdf_pool()
            future = pool.submit(_extract_pdf_worker, str(file_path), maxpages)
            text = future.result(timeout=timeout)
            return text or ""
        except FuturesTimeout:
            logger.warning(f"PDF extraction timed out after {timeout}s: {file_path.name}")
            # The worker is stuck; kill the pool so the next call gets a fresh one.
            self._shutdown_pdf_pool(kill=True)
            return _EXTRACTION_TIMEOUT
        except Exception as e:
            logger.warning(f"PDF extraction failed: {file_path.name}: {e}")
            return ""

    def _shutdown_pdf_pool(self, kill: bool = False):
        """Shutdown the PDF worker pool."""
        if self._pdf_pool is not None:
            try:
                if kill:
                    self._pdf_pool.shutdown(wait=False, cancel_futures=True)
                else:
                    self._pdf_pool.shutdown(wait=False)
            except Exception:
                pass
            self._pdf_pool = None

    def _extract_text_from_html(self, file_path: Path) -> str:
        """Extract text from HTML using markitdown if available; fallback to stripping tags."""
        # Try markitdown first
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(str(file_path))
            return result.text_content or ""
        except Exception:
            pass
        # Fallback using a simple parser
        try:
            from bs4 import BeautifulSoup  # type: ignore
            html = file_path.read_text(errors="ignore")
            return BeautifulSoup(html, "html.parser").get_text(" ")
        except Exception:
            return ""

    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text content from a file based on extension, with fallbacks."""
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_text_from_pdf(file_path)
        if suffix in {".html", ".htm"}:
            return self._extract_text_from_html(file_path)
        # Generic best-effort
        try:
            return file_path.read_text(errors="ignore")
        except Exception:
            return ""

    def _get_fulltext_meta_for_item(self, item_id: int):
        meta = []
        for key, path, ctype in self._iter_parent_attachments(item_id):
            meta.append([key, path, ctype])

        return meta

    def _find_content_list_json(self, attachment_key: str, resolved_path: Path) -> Path | None:
        """Find MinerU content_list.json file in the attachment folder.

        Looks for files matching patterns:
        - content_list.json
        - *_content_list.json (MinerU standard naming)

        Args:
            attachment_key: The Zotero attachment item key
            resolved_path: The resolved attachment file path

        Returns:
            Path to content_list.json if found, None otherwise
        """
        if resolved_path and resolved_path.exists():
            # Check the same directory as the attachment
            parent_dir = resolved_path.parent
            
            # For Zotero storage format: storage/KEY/filename.pdf
            # The content_list.json should be in the same folder
            
            # Priority 1: Look for *_content_list.json pattern (MinerU standard)
            for f in parent_dir.iterdir():
                if f.is_file() and f.name.endswith('_content_list.json'):
                    return f
            
            # Priority 2: Look for exact content_list.json
            content_list = parent_dir / "content_list.json"
            if content_list.exists():
                return content_list
        return None

    def _extract_text_from_content_list_json(self, json_path: Path) -> str:
        """Extract text content from MinerU content_list.json format.

        Supports multiple MinerU output formats:

        1. Nested pages format (list of lists):
           Each top-level item is a page (list of blocks). Each block is a dict
           with 'type' (e.g. 'paragraph', 'title'), 'content' (a dict whose
           single key like 'paragraph_content' maps to a list of text spans),
           and 'bbox'.

        2. Flat list of dicts with 'text' key:
           [{"text": "...", "type": "text", ...}, ...]

        3. Dict wrapper with 'content' or 'data' key containing a list.

        Args:
            json_path: Path to the content_list.json file

        Returns:
            Extracted text content, or empty string on failure
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle root-level list format (MinerU standard)
            if isinstance(data, list):
                content_list = data
            else:
                # Handle dict format with 'content' or 'data' key
                content_list = data.get('content') or data.get('data') or []

            if isinstance(content_list, list):
                text_parts = []
                for item in content_list:
                    if isinstance(item, list):
                        # Nested pages format: each item is a page (list of blocks)
                        for block in item:
                            text = self._extract_text_from_mineru_block(block)
                            if text:
                                text_parts.append(text)
                    elif isinstance(item, dict):
                        # Could be a block directly (single-page) or flat format
                        text = self._extract_text_from_mineru_block(item)
                        if text:
                            text_parts.append(text)
                    elif isinstance(item, str):
                        # Simple list format
                        if item.strip():
                            text_parts.append(item.strip())

                if text_parts:
                    return '\n\n'.join(text_parts)

            # Fallback: try to extract any text-like content from dict
            if isinstance(data, dict):
                for key in ['text', 'content', 'fulltext', 'body']:
                    if key in data and isinstance(data[key], str):
                        return data[key]

            return ""
        except Exception as e:
            logger.warning(f"Failed to extract text from content_list.json: {e}")
            return ""

    @staticmethod
    def _extract_text_from_mineru_block(block: dict) -> str:
        """Extract text from a single MinerU block.

        A block has 'type' (paragraph, title, equation_interline, etc.) and
        'content' which is a dict like:
            {"paragraph_content": [{"type": "text", "content": "..."}]}

        Also handles the flat format where text is directly in block['text'].
        """
        if not isinstance(block, dict):
            return ""

        # Skip non-text block types (headers, footers, page numbers)
        block_type = block.get('type', '')
        if block_type in ('page_header', 'page_footer', 'page_number'):
            return ""

        # Format 1: nested content dict (MinerU pages format)
        content = block.get('content')
        if isinstance(content, dict):
            # Equation blocks: {"math_content": "LaTeX", "math_type": "latex"}
            math = content.get('math_content')
            if isinstance(math, str) and math.strip():
                return f"$$ {math.strip()} $$"

            # Text blocks: {"paragraph_content": [{"type": "text", "content": "..."}]}
            for key, spans in content.items():
                if key in ('level', 'math_type', 'image_source'):
                    continue
                if isinstance(spans, list):
                    parts = []
                    for span in spans:
                        if isinstance(span, dict):
                            t = span.get('content', '')
                            if isinstance(t, str) and t.strip():
                                parts.append(t.strip())
                    if parts:
                        return ' '.join(parts)

        # Format 2: flat format with 'text' key
        text = block.get('text', '')
        if isinstance(text, str) and text.strip():
            return text.strip()

        return ""

    def _read_zotero_ft_cache(self, cache_path: Path) -> str:
        """Read Zotero's pre-built full text cache file (.zotero-ft-cache).

        Zotero writes this file when it indexes an attachment.  Reading it is
        instant (pure file I/O) and avoids PDF extraction timeouts entirely.
        Returns the text, or empty string on failure.
        """
        try:
            return cache_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            logger.debug(f"Could not read .zotero-ft-cache at {cache_path}: {e}")
            return ""

    def _extract_fulltext_for_item(self, item_id: int) -> tuple[str, str] | None:
        """Attempt to extract fulltext and source from the item's best attachment.

        Priority order:
        1. content_list.json (MinerU format) - highest quality structured text
        2. .zotero-ft-cache - Zotero's pre-built index (instant, no timeout risk)
        3. PDF - extract text using pdfminer
        4. HTML - extract text using markitdown/BeautifulSoup

        Returns (text, source) where source is 'content_list_json', 'zotero_cache',
        'pdf', or 'html'.
        """
        best_content_list_json = None
        best_ft_cache = None
        best_pdf = None
        best_html = None

        for key, path, ctype in self._iter_parent_attachments(item_id):
            resolved = self._resolve_attachment_path(key, path or "")
            if not resolved or not resolved.exists():
                continue

            # Priority 1: Check for content_list.json in the attachment folder
            if ctype == "application/pdf":
                content_list = self._find_content_list_json(key, resolved)
                logger.debug(f"Checking PDF attachment {key}: path={resolved}, content_list={content_list}")
                if content_list and content_list.exists():
                    logger.info(f"Found content_list.json: {content_list}")
                    if best_content_list_json is None:
                        best_content_list_json = content_list
                # Priority 2: .zotero-ft-cache alongside the PDF
                ft_cache = resolved.parent / ".zotero-ft-cache"
                if ft_cache.exists() and best_ft_cache is None:
                    best_ft_cache = ft_cache
                if best_pdf is None:
                    best_pdf = resolved
            elif (ctype or "").startswith("text/html"):
                # Also check for content_list.json alongside HTML
                content_list = self._find_content_list_json(key, resolved)
                if content_list and content_list.exists():
                    if best_content_list_json is None:
                        best_content_list_json = content_list
                if best_html is None:
                    best_html = resolved

        # Priority 1: Use content_list.json if available (highest quality)
        logger.debug(f"best_content_list_json={best_content_list_json}, best_ft_cache={best_ft_cache}, best_pdf={best_pdf}, best_html={best_html}")
        if best_content_list_json:
            text = self._extract_text_from_content_list_json(best_content_list_json)
            if text:
                logger.info(f"Using content_list.json for item {item_id}: {best_content_list_json}")
                return (text, "content_list_json")

        # Priority 2: Use Zotero's pre-built full text cache (instant, no timeout risk)
        if best_ft_cache:
            text = self._read_zotero_ft_cache(best_ft_cache)
            if text:
                logger.info(f"Using .zotero-ft-cache for item {item_id}: {best_ft_cache}")
                return (text, "zotero_cache")

        # Priority 3: Use PDF
        if best_pdf:
            text = self._extract_text_from_file(best_pdf)
            if text == _EXTRACTION_TIMEOUT:
                return (_EXTRACTION_TIMEOUT, "timeout")
            if text:
                return (text, "pdf")

        # Priority 4: Use HTML
        if best_html:
            text = self._extract_text_from_file(best_html)
            if text:
                return (text, "html")

        return None

    def close(self):
        """Close database connection and worker pool."""
        self._shutdown_pdf_pool()
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_libraries(self) -> list[dict[str, Any]]:
        """Get all libraries (user, group, feed) from the database."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT l.libraryID, l.type, l.editable,
                   g.groupID, g.name as groupName, g.description as groupDescription,
                   f.name as feedName, f.url as feedUrl,
                   f.lastCheck as feedLastCheck, f.lastUpdate as feedLastUpdate,
                   (SELECT COUNT(*) FROM items i
                    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                    WHERE i.libraryID = l.libraryID
                    AND it.typeName NOT IN ('attachment', 'note', 'annotation')) as itemCount
            FROM libraries l
            LEFT JOIN groups g ON l.libraryID = g.libraryID
            LEFT JOIN feeds f ON l.libraryID = f.libraryID
            ORDER BY l.type, l.libraryID
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_groups(self) -> list[dict[str, Any]]:
        """Get all group libraries with item counts."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT g.groupID, g.libraryID, g.name, g.description,
                   (SELECT COUNT(*) FROM items i
                    JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
                    WHERE i.libraryID = g.libraryID
                    AND it.typeName NOT IN ('attachment', 'note', 'annotation')) as itemCount
            FROM groups g
            ORDER BY g.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_feeds(self) -> list[dict[str, Any]]:
        """Get all RSS feed subscriptions with item counts."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT f.libraryID, f.name, f.url,
                   f.lastCheck, f.lastUpdate, f.lastCheckError,
                   f.refreshInterval,
                   (SELECT COUNT(*) FROM feedItems fi
                    JOIN items i ON fi.itemID = i.itemID
                    WHERE i.libraryID = f.libraryID) as itemCount
            FROM feeds f
            ORDER BY f.name
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def get_feed_items(
        self, library_id: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get items from a specific RSS feed by its libraryID."""
        conn = self._get_connection()
        rows = conn.execute(
            """
            SELECT i.itemID, i.key, it.typeName as itemType,
                   i.dateAdded,
                   fi.readTime, fi.translatedTime,
                   title_val.value as title,
                   abstract_val.value as abstract,
                   url_val.value as url,
                   GROUP_CONCAT(
                       CASE
                           WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                           THEN c.lastName || ', ' || c.firstName
                           WHEN c.lastName IS NOT NULL THEN c.lastName
                           ELSE NULL
                       END, '; '
                   ) as creators
            FROM feedItems fi
            JOIN items i ON fi.itemID = i.itemID
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
            LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID
            LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
            LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID
            LEFT JOIN fields url_f ON url_f.fieldName = 'url'
            LEFT JOIN itemData url_data ON i.itemID = url_data.itemID AND url_data.fieldID = url_f.fieldID
            LEFT JOIN itemDataValues url_val ON url_data.valueID = url_val.valueID
            LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
            LEFT JOIN creators c ON ic.creatorID = c.creatorID
            WHERE i.libraryID = ?
            GROUP BY i.itemID
            ORDER BY i.dateAdded DESC
            LIMIT ?
            """,
            (library_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_item_count(self) -> int:
        """
        Get total count of non-attachment items.

        Returns:
            Number of items in the library.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT COUNT(*)
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
            """
        )
        return cursor.fetchone()[0]

    def get_items_with_text(self, limit: int | None = None, include_fulltext: bool = False, key_filter: str | None = None) -> list[ZoteroItem]:
        """
        Get all items with their text content for semantic search.

        Args:
            limit: Optional limit on number of items to return.

        Returns:
            List of ZoteroItem objects with text content.
        """
        conn = self._get_connection()

        # Query to get items with their text content (simplified for now)
        query = """
        SELECT
            i.itemID,
            i.key,
            i.itemTypeID,
            it.typeName as item_type,
            i.dateAdded,
            i.dateModified,
            title_val.value as title,
            abstract_val.value as abstract,
            extra_val.value as extra,
            doi_val.value as doi,
            GROUP_CONCAT(n.note, ' ') as notes,
            GROUP_CONCAT(
                CASE
                    WHEN c.firstName IS NOT NULL AND c.lastName IS NOT NULL
                    THEN c.lastName || ', ' || c.firstName
                    WHEN c.lastName IS NOT NULL
                    THEN c.lastName
                    ELSE NULL
                END, '; '
            ) as creators
        FROM items i
        JOIN itemTypes it ON i.itemTypeID = it.itemTypeID

        -- Get title
        LEFT JOIN itemData title_data ON i.itemID = title_data.itemID AND title_data.fieldID = 1
        LEFT JOIN itemDataValues title_val ON title_data.valueID = title_val.valueID

        -- Get abstract
        LEFT JOIN itemData abstract_data ON i.itemID = abstract_data.itemID AND abstract_data.fieldID = 2
        LEFT JOIN itemDataValues abstract_val ON abstract_data.valueID = abstract_val.valueID

        -- Get extra field
        LEFT JOIN itemData extra_data ON i.itemID = extra_data.itemID AND extra_data.fieldID = 16
        LEFT JOIN itemDataValues extra_val ON extra_data.valueID = extra_val.valueID

        -- Get DOI field via fields table
        LEFT JOIN fields doi_f ON doi_f.fieldName = 'DOI'
        LEFT JOIN itemData doi_data ON i.itemID = doi_data.itemID AND doi_data.fieldID = doi_f.fieldID
        LEFT JOIN itemDataValues doi_val ON doi_data.valueID = doi_val.valueID

        -- Get notes
        LEFT JOIN itemNotes n ON i.itemID = n.parentItemID OR i.itemID = n.itemID

        -- Get creators
        LEFT JOIN itemCreators ic ON i.itemID = ic.itemID
        LEFT JOIN creators c ON ic.creatorID = c.creatorID

        WHERE it.typeName NOT IN ('attachment', 'note', 'annotation')
        """

        params = []
        if key_filter:
            query += " AND i.key = ?"
            params.append(key_filter)

        query += """
        GROUP BY i.itemID, i.key, i.itemTypeID, it.typeName, i.dateAdded, i.dateModified,
                 title_val.value, abstract_val.value, extra_val.value

        ORDER BY i.dateModified DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor = conn.execute(query, params)
        items = []

        for row in cursor:
            item = ZoteroItem(
                item_id=row['itemID'],
                key=row['key'],
                item_type_id=row['itemTypeID'],
                item_type=row['item_type'],
                doi=row['doi'],
                title=row['title'],
                abstract=row['abstract'],
                creators=row['creators'],
                fulltext=(res := (self._extract_fulltext_for_item(row['itemID']) if include_fulltext else None)) and (res[0] if res[1] != "timeout" else None),
                fulltext_source=res[1] if include_fulltext and res and res[1] != "timeout" else None,
                notes=row['notes'],
                extra=row['extra'],
                date_added=row['dateAdded'],
                date_modified=row['dateModified']
            )
            items.append(item)

        return items

    # Public helper to quickly check full text metadata for item
    def get_fulltext_meta_for_item(self, item_id: int) -> tuple[str, str] | None:
        return self._get_fulltext_meta_for_item(item_id)

    def has_content_list_json(self, item_id: int) -> bool:
        """Check if the item has a content_list.json file in its attachment folder.

        Args:
            item_id: The Zotero item ID

        Returns:
            True if content_list.json exists, False otherwise
        """
        for key, path, ctype in self._iter_parent_attachments(item_id):
            resolved = self._resolve_attachment_path(key, path or "")
            if not resolved or not resolved.exists():
                continue
            if ctype == "application/pdf":
                content_list = self._find_content_list_json(key, resolved)
                if content_list and content_list.exists():
                    return True
        return False

    def get_content_list_json_path(self, item_id: int) -> Path | None:
        """Get the path to content_list.json for an item.

        Args:
            item_id: The Zotero item ID

        Returns:
            Path to content_list.json if found, None otherwise
        """
        for key, path, ctype in self._iter_parent_attachments(item_id):
            resolved = self._resolve_attachment_path(key, path or "")
            if not resolved or not resolved.exists():
                continue
            if ctype == "application/pdf":
                content_list = self._find_content_list_json(key, resolved)
                if content_list and content_list.exists():
                    return content_list
        return None

    # Public helper to extract fulltext on demand for a specific item
    def extract_fulltext_for_item(self, item_id: int) -> tuple[str, str] | None:
        return self._extract_fulltext_for_item(item_id)

    def get_item_by_key(self, key: str) -> ZoteroItem | None:
        """
        Get a specific item by its Zotero key.

        Args:
            key: The Zotero item key.

        Returns:
            ZoteroItem if found, None otherwise.
        """
        items = self.get_items_with_text(key_filter=key)
        return items[0] if items else None

    def search_items_by_text(self, query: str, limit: int = 50) -> list[ZoteroItem]:
        """
        Simple text search through item content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of matching ZoteroItem objects.
        """
        items = self.get_items_with_text()
        matching_items = []

        query_lower = query.lower()

        for item in items:
            searchable_text = item.get_searchable_text().lower()
            if query_lower in searchable_text:
                matching_items.append(item)
                if len(matching_items) >= limit:
                    break

        return matching_items

    def search_notes_local(self, query: str, limit: int = 20) -> list[dict]:
        """Search notes in the local Zotero database by text content."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = f"%{query}%"
        cursor.execute("""
            SELECT i.key, n.note, n.title,
                   pi.key as parentKey,
                   pdv.value as parentTitle
            FROM itemNotes n
            JOIN items i ON n.itemID = i.itemID
            LEFT JOIN items pi ON n.parentItemID = pi.itemID
            LEFT JOIN itemData pd ON pi.itemID = pd.itemID AND pd.fieldID = 1
            LEFT JOIN itemDataValues pdv ON pd.valueID = pdv.valueID
            WHERE n.note LIKE ?
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT ?
        """, (pattern, limit))

        results = []
        for row in cursor.fetchall():
            note_html = row[1] or ""
            # Post-filter: skip if query only matches HTML tags, not content
            from zotero_mcp.utils import clean_html
            clean_text = clean_html(note_html)
            if query.lower() not in clean_text.lower():
                continue
            results.append({
                "type": "note",
                "key": row[0],
                "text": note_html,
                "parent_key": row[3],
                "parent_title": row[4] or ("Unknown" if row[3] else None),
                "tags": [],  # Tags require a separate query; omitted for speed
            })
        return results

    def search_annotations_local(self, query: str, limit: int = 20) -> list[dict]:
        """Search annotations in the local Zotero database by text or comment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        pattern = f"%{query}%"
        # Two-hop join: annotation -> attachment -> grandparent item (for title)
        cursor.execute("""
            SELECT i.key, ia.text, ia.comment, ia.type, ia.color, ia.pageLabel,
                   att.key as attachmentKey,
                   gpi.key as parentKey,
                   gpdv.value as parentTitle
            FROM itemAnnotations ia
            JOIN items i ON ia.itemID = i.itemID
            LEFT JOIN items att ON ia.parentItemID = att.itemID
            LEFT JOIN itemAttachments iatt ON ia.parentItemID = iatt.itemID
            LEFT JOIN items gpi ON iatt.parentItemID = gpi.itemID
            LEFT JOIN itemData gpd ON gpi.itemID = gpd.itemID AND gpd.fieldID = 1
            LEFT JOIN itemDataValues gpdv ON gpd.valueID = gpdv.valueID
            WHERE (ia.text LIKE ? OR ia.comment LIKE ?)
            AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            LIMIT ?
        """, (pattern, pattern, limit))

        # Map integer annotation types to names
        type_map = {1: "highlight", 2: "note", 3: "image", 4: "ink", 5: "underline"}

        results = []
        for row in cursor.fetchall():
            results.append({
                "type": "annotation",
                "key": row[0],
                "text": row[1] or "",
                "comment": row[2] or "",
                "annotation_type": type_map.get(row[3], "unknown"),
                "color": row[4] or "",
                "page_label": row[5] or None,
                "attachment_key": row[6],
                "parent_key": row[7],
                "parent_title": row[8] or ("Unknown" if row[7] else None),
            })
        return results


def get_local_zotero_reader() -> LocalZoteroReader | None:
    """
    Get a LocalZoteroReader instance if in local mode.

    Returns:
        LocalZoteroReader instance if in local mode and database exists,
        None otherwise.
    """
    if not is_local_mode():
        return None

    try:
        return LocalZoteroReader()
    except FileNotFoundError:
        return None


def is_local_db_available() -> bool:
    """
    Check if local Zotero database is available.

    Returns:
        True if local database can be accessed, False otherwise.
    """
    reader = get_local_zotero_reader()
    if reader:
        reader.close()
        return True
    return False

"""Multi-document intake for the pilot: extraction (text or OCR), per-file metadata
and product-driven document checklist.

This is a real local pipeline: each attached piece is actually extracted (text layer
for PDF/TXT, OCR for images when the optional OCR dependencies are installed) and the
checklist is derived from the active product rules pack, not hardcoded.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from products.registry import get_product_pack
from utils.document_upload import (
    MAX_EXTRACTED_CHARACTERS,
    PdfExtractionUnavailable,
    extract_uploaded_document,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".pdf", ".txt"}

EXTRACTED_STATUSES = {"text_extracted", "ocr_extracted"}


def ocr_available() -> bool:
    """Whether the optional OCR stack (Pillow + pytesseract) is installed."""
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        return False
    return True


def _extract_image(content: bytes) -> tuple[str, str]:
    """Return (text, engine). Raises ImportError when OCR is not installed."""
    import pytesseract
    from PIL import Image

    image = Image.open(BytesIO(content))
    return pytesseract.image_to_string(image), "tesseract"


def _extract_single(filename: str, content: bytes) -> dict[str, Any]:
    """Extract one attached piece, returning a status-bearing record (never raises)."""
    extension = Path(filename).suffix.lower()
    record: dict[str, Any] = {
        "filename": filename,
        "size_bytes": len(content or b""),
        "extraction_status": "attached_unreadable",
        "characters": 0,
        "engine": None,
        "reason": None,
    }

    if not content:
        record["extraction_status"] = "invalid"
        record["reason"] = "Le document est vide."
        return record

    if extension in TEXT_EXTENSIONS:
        try:
            text = extract_uploaded_document(filename, content)
        except PdfExtractionUnavailable as error:
            record["extraction_status"] = "attached_reader_unavailable"
            record["reason"] = str(error)
            return record
        except ValueError as error:
            record["extraction_status"] = "invalid"
            record["reason"] = str(error)
            return record
        record["extraction_status"] = "text_extracted"
        record["engine"] = "text"
        record["text"] = text
        record["characters"] = len(text)
        return record

    if extension in IMAGE_EXTENSIONS:
        if not ocr_available():
            record["extraction_status"] = "attached_ocr_unavailable"
            record["reason"] = (
                "OCR indisponible : installe les dépendances optionnelles "
                "(`pip install -r requirements-research.txt`) pour lire les pièces scannées."
            )
            return record
        try:
            text = (content and _extract_image(content)[0] or "").strip()
        except Exception as error:  # pragma: no cover - depends on the OCR engine
            record["extraction_status"] = "attached_unreadable"
            record["reason"] = f"Échec OCR : {error}"
            return record
        if not text:
            record["extraction_status"] = "attached_unreadable"
            record["reason"] = "Aucun texte détecté par OCR sur cette pièce."
            return record
        record["extraction_status"] = "ocr_extracted"
        record["engine"] = "tesseract"
        record["text"] = text
        record["characters"] = len(text)
        return record

    record["extraction_status"] = "unsupported_format"
    record["reason"] = (
        "Format non pris en charge. Formats acceptés : PDF, TXT ou image "
        "(png/jpg/tiff) pour OCR."
    )
    return record


def _aggregate_status(records: list[dict[str, Any]]) -> str:
    if not records:
        return "not_provided"
    statuses = {record["extraction_status"] for record in records}
    if statuses <= EXTRACTED_STATUSES:
        return "ocr_extracted" if "ocr_extracted" in statuses else "text_extracted"
    if statuses & EXTRACTED_STATUSES:
        return "partially_unreadable"
    return "attached_unreadable"


def build_document_bundle(
    files: Iterable[dict[str, Any]],
    product_line: str = "auto",
) -> dict[str, Any]:
    """Build a multi-document bundle with per-file metadata and the product checklist.

    ``files`` items are dicts: ``{"filename", "content", "declared_type"}``.
    ``declared_type`` is a document code from the product pack (or ``None``).
    """
    pack = get_product_pack(product_line) or {}
    required = list(pack.get("required_documents", []))
    labels_by_code = {item["code"]: item["label_fr"] for item in required}

    records: list[dict[str, Any]] = []
    texts: list[str] = []
    for item in files:
        filename = Path(str(item.get("filename") or "document.bin")).name
        content = item.get("content") or b""
        declared_type = item.get("declared_type") or None
        record = _extract_single(filename, content)
        record["declared_type"] = declared_type
        record["declared_type_label"] = labels_by_code.get(declared_type)
        if record.get("text"):
            texts.append(f"--- {filename} ---\n{record.pop('text')}")
        records.append(record)

    combined_text = "\n\n".join(texts).strip()[:MAX_EXTRACTED_CHARACTERS]
    declared_codes = {record["declared_type"] for record in records if record["declared_type"]}

    checklist_required = [
        {
            "code": item["code"],
            "label_fr": item["label_fr"],
            "mandatory": bool(item.get("mandatory")),
            "provided": item["code"] in declared_codes,
        }
        for item in required
    ]
    missing_mandatory = [
        item["code"] for item in checklist_required if item["mandatory"] and not item["provided"]
    ]

    return {
        "product_line": product_line,
        "documents": records,
        "filenames_label": ", ".join(record["filename"] for record in records) or None,
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "combined_text": combined_text,
        "characters_extracted": len(combined_text),
        "aggregate_status": _aggregate_status(records),
        "ocr_available": ocr_available(),
        "checklist": {
            "required": checklist_required,
            "missing_mandatory": missing_mandatory,
            "complete": not missing_mandatory,
        },
    }

"""Safe, local text extraction for documents uploaded to the demo UI."""

from io import BytesIO
from pathlib import Path

MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_PAGES = 50
MAX_EXTRACTED_CHARACTERS = 200_000
SUPPORTED_EXTENSIONS = {'.pdf', '.txt'}


class PdfExtractionUnavailable(ValueError):
    """Raised when a PDF is attachable but no PDF parser is installed."""


def extract_uploaded_document(file_name, content):
    """Extract text from an uploaded PDF or UTF-8 text file without saving it."""
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError('Formats acceptés : PDF ou fichier texte UTF-8 (.txt).')
    if not content:
        raise ValueError('Le document est vide.')
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError('La taille maximale du document est de 10 Mo.')

    if extension == '.txt':
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise ValueError('Le fichier texte doit être encodé en UTF-8.') from error
    else:
        if not content.startswith(b'%PDF-'):
            raise ValueError('Le fichier n’a pas un en-tête PDF valide.')
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError, PdfStreamError
        except ImportError as error:
            raise PdfExtractionUnavailable(
                'La lecture PDF nécessite pypdf. Installe les dépendances avec '
                '`python -m pip install -r requirements.txt`, puis relance l’application.'
            ) from error

        try:
            reader = PdfReader(BytesIO(content), strict=True)
            if reader.is_encrypted:
                raise ValueError('Les PDF protégés par mot de passe ne sont pas pris en charge.')
            if len(reader.pages) > MAX_DOCUMENT_PAGES:
                raise ValueError('Le PDF dépasse la limite de 50 pages.')
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        except ValueError:
            raise
        except (PdfReadError, PdfStreamError, OSError, EOFError) as error:
            raise ValueError(f'Impossible de lire ce PDF : {error}') from error

    text = text.strip()
    if not text:
        raise ValueError(
            'Aucun texte extractible. Les PDF scannés nécessitent un OCR, non disponible dans ce prototype.'
        )
    if len(text) > MAX_EXTRACTED_CHARACTERS:
        raise ValueError('Le texte extrait dépasse la limite de 200 000 caractères.')
    return text

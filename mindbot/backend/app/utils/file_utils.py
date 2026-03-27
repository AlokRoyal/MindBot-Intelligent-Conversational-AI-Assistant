import os
import uuid
from pathlib import Path
from typing import Optional
from pypdf import PdfReader
import aiofiles

from app.core.config import UPLOAD_DIR


ALLOWED_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
ALLOWED_DOC_EXTENSIONS = {".pdf"} | ALLOWED_TEXT_EXTENSIONS
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_allowed_document(filename: str) -> bool:
    return file_extension(filename) in ALLOWED_DOC_EXTENSIONS


def is_allowed_image(filename: str) -> bool:
    return file_extension(filename) in ALLOWED_IMAGE_EXTENSIONS


def is_allowed_audio(filename: str) -> bool:
    return file_extension(filename) in ALLOWED_AUDIO_EXTENSIONS


async def save_upload_file(upload_file, folder: Optional[Path] = None) -> Path:
    folder = folder or UPLOAD_DIR
    folder.mkdir(parents=True, exist_ok=True)

    ext = file_extension(upload_file.filename)
    name = f"{uuid.uuid4().hex}{ext}"
    path = folder / name

    async with aiofiles.open(path, "wb") as out_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            await out_file.write(chunk)

    await upload_file.seek(0)
    return path


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()


def extract_text_from_plain_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""
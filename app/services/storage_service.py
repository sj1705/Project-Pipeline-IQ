import os
import shutil
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class LocalStorageService:
    def __init__(self, upload_dir: Path = UPLOAD_DIR):
        self.upload_dir = upload_dir

    def save_file(self, file: UploadFile, filename: str) -> str:
        """Save uploaded file to disk. Returns the file path."""
        file_path = self.upload_dir / filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return str(file_path)

    def get_file_path(self, filename: str) -> str:
        """Get full path of a stored file."""
        return str(self.upload_dir / filename)

    def file_exists(self, filename: str) -> bool:
        return (self.upload_dir / filename).exists()


storage_service = LocalStorageService()
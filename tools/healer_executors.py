# Isi file: src/models/model_alpha/healer_executors/argument_fixer.py
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def execute(file_path: str, fix_details: dict) -> str:
    """
    Menerapkan perbaikan argumen yang hilang. Hanya melakukan find-and-replace.
    """
    original_call = fix_details.get('original_code_to_find')
    suggested_replacement = fix_details.get('suggested_call_with_fix')

    if not all([original_call, suggested_replacement]):
        return "Error: Detail perbaikan tidak lengkap dalam resep."

    target_file = Path(file_path)
    if not target_file.exists():
        return f"Error: File target '{file_path}' tidak ditemukan."
    
    full_source_code = target_file.read_text(encoding='utf-8')

    if original_call not in full_source_code:
        return f"Error: Kode asli '{original_call}' tidak ditemukan di file. Perbaikan dibatalkan."

    logger.warning(f"EXECUTOR (argument_fixer): Menerapkan patch:\n- MENCARI: {original_call}\n+ MENGGANTI: {suggested_replacement}")
    
    new_source_code = full_source_code.replace(original_call, suggested_replacement, 1)
    target_file.write_text(new_source_code, encoding='utf-8')

    return "Berhasil menerapkan perbaikan. Program perlu di-restart."
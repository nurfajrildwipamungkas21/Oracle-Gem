# Isi lengkap untuk tools/syntax_healer.py

from pathlib import Path
import textwrap
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

@tool
def fix_syntax_error(file_path: str, error_line_number: int, error_message: str, problematic_code: str) -> str:
    """
    Gunakan HANYA untuk memperbaiki SyntaxError atau IndentationError. Tool ini akan meminta AI untuk
    menulis ulang baris kode yang salah secara sintaksis dan menggantinya di file.
    """
    try:
        # Impor absolut dari 'src' untuk konsistensi
        from src.models.model_alpha.alpha import api_pool

        logger.info(f"SYNTAX_HEALER: Mencoba memperbaiki {file_path} di baris {error_line_number}...")
        
        architect_agent = api_pool.get_worker("supervisor")
        if not architect_agent:
            return "Error: Agen 'supervisor' tidak tersedia untuk memperbaiki sintaks."

        fix_prompt = f"""
        You are an expert Python code corrector. Your task is to fix a single line or a small block of code that has a syntax error.
        DO NOT explain anything. DO NOT add any comments. Return ONLY the corrected, raw Python code.

        Error message: "{error_message}"
        Problematic code block:
        ---
        {problematic_code}
        ---

        Corrected code:
        """

        corrected_code_snippet = api_pool.call_gemini_for_text(fix_prompt, "supervisor")
        
        if not corrected_code_snippet or "sorry" in corrected_code_snippet.lower():
            return f"Error: AI tidak dapat memberikan perbaikan untuk sintaks error ini."

        cleaned_corrected_code = textwrap.dedent(corrected_code_snippet).strip()

        target_file = Path(file_path)
        lines = target_file.read_text(encoding='utf-8').splitlines()
        
        line_index = error_line_number - 1
        if 0 <= line_index < len(lines):
            logger.warning(f"SYNTAX_HEALER: Mengganti baris {error_line_number}:\n- LAMA: {lines[line_index]}\n+ BARU: {cleaned_corrected_code}")
            lines[line_index] = cleaned_corrected_code
        else:
            return f"Error: Nomor baris {error_line_number} di luar jangkauan file."

        target_file.write_text("\n".join(lines), encoding='utf-8')
        
        return f"Berhasil memperbaiki SyntaxError di {file_path} pada baris {error_line_number}. Program perlu di-restart."

    except Exception as e:
        return f"Gagal menjalankan syntax_healer: {e}"
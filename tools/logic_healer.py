# Isi lengkap untuk tools/logic_healer.py (Versi Perbaikan)

import logging
from langchain_core.tools import tool
from pydantic import BaseModel, Field
# --- PERUBAHAN DI SINI ---
from typing import Any # Impor 'Any' untuk menangani circular dependency

# Hapus blok "if TYPE_CHECKING" dan impor dari alpha.py
# -------------------------

logger = logging.getLogger(__name__)

class HealingPrescription(BaseModel):
    executor_name: str = Field(
        description="Nama executor spesialis yang harus dipanggil, contoh: 'argument_fixer', 'import_fixer'."
    )
    fix_details: dict = Field(
        description="Objek JSON berisi detail untuk executor, biasanya 'original_code' dan 'suggested_code'."
    )
    explanation: str = Field(
        description="Penjelasan singkat mengapa resep ini dibuat."
    )

@tool
def create_healing_prescription(
    file_path: str,
    error_message: str,
    full_traceback: str,
    api_pool: "Any"  # <-- PERUBAHAN DI SINI: Gunakan "Any" sebagai type hint
) -> dict:
    """
    Bertindak sebagai "Dokter Umum". Mendiagnosis error runtime dan membuat 'resep' terstruktur 
    untuk dieksekusi oleh executor spesialis.
    """
    logger.info(f"HEALER: Mendiagnosis '{error_message}'...")

    prompt = f"""
    Anda adalah seorang AI Diagnostician super cerdas. Analisis error Python berikut.
    Tugas Anda adalah membuat resep perbaikan (HealingPrescription).

    ERROR REPORT:
    - Error Message: "{error_message}"
    - Full Traceback:
    ---
    {full_traceback}
    ---

    LANGKAH ANDA:
    1.  Identifikasi akar masalah. Apakah ini argumen fungsi yang hilang? Kesalahan impor? Atau yang lain?
    2.  Tentukan `executor_name` yang paling tepat. Pilihan: `argument_fixer`, `import_fixer`, `syntax_fixer`.
    3.  Tentukan `fix_details`. Ini harus berisi `original_code` (baris yang salah) dan `suggested_code` (baris perbaikan).
    4.  Panggil tool `HealingPrescription` dengan hasil diagnosis Anda.
    """
    
    try:
        # Kita asumsikan api_pool memiliki method call_gemini_with_tool
        prescription_dict = api_pool.call_gemini_with_tool(
            prompt=prompt,
            agent_name="supervisor",
            tool_schema=HealingPrescription
        )
        if not prescription_dict:
            raise ValueError("AI Diagnostician gagal membuat resep.")
        
        logger.info(f"HEALER: Resep dibuat. Executor: '{prescription_dict['executor_name']}'")
        return prescription_dict

    except Exception as e:
        logger.error(f"HEALER: Gagal membuat resep: {e}")
        return {
            "executor_name": "failed",
            "fix_details": {},
            "explanation": f"Gagal mendiagnosis error: {e}"
        }
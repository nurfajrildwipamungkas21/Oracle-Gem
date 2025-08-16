# Isi lengkap untuk tools/dependency_healer.py

import subprocess
import sys
import questionary
from langchain_core.tools import tool

@tool
def install_missing_module(module_name: str) -> str:
    """
    Gunakan tool ini HANYA untuk menginstal library Python yang hilang ketika terjadi ModuleNotFoundError.
    Tool ini akan menjalankan 'pip install' untuk modul yang ditentukan.
    """
    try:
        # PENTING: Lapisan keamanan konfirmasi manusia
        if not questionary.confirm(f"PERHATIAN: AI mengusulkan untuk menjalankan 'pip install {module_name}'. Setuju?").ask():
            return f"Instalasi '{module_name}' dibatalkan oleh pengguna."
        
        # Menggunakan sys.executable untuk memastikan instalasi di virtual environment yang benar
        subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
        
        return f"Berhasil menginstal {module_name}. Program perlu di-restart untuk menggunakan modul baru. Harap jalankan ulang skrip."
    except Exception as e:
        return f"Gagal menginstal {module_name}: {e}" 

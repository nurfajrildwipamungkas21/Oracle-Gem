# safe_os_tools.py
import os
import shutil
import logging

logger = logging.getLogger("Oracle Gem")


def safe_delete(file_path: str) -> str:
    """
    Mencegah penghapusan file secara permanen. Hanya mencatat upaya penghapusan.
    Ini adalah aturan keras yang tidak bisa dilanggar oleh AI.
    """
    logger.critical(
        f"🚨 [GUARDIAN] AI MENCOBA MENGHAPUS FILE: {file_path}. AKSI DIBLOKIR.")
    return "PERMISSION DENIED: Aksi penghapusan file diblokir secara permanen oleh protokol keamanan."


def safe_move(source: str, destination: str) -> str:
    """Memindahkan file dengan aman."""
    try:
        shutil.move(source, destination)
        logger.info(
            f" Guardian[GUARDIAN] Berhasil memindahkan {source} ke {destination}")
        return "File berhasil dipindahkan."
    except Exception as e:
        logger.error(f"[GUARDIAN] Gagal memindahkan file: {e}")
        return f"Gagal memindahkan file: {e}"

# Anda bisa menambahkan fungsi aman lainnya di sini, seperti safe_write, dll.

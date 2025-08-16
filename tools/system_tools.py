# Isi untuk file: src/models/model_alpha/tools/system_tools.py

import pyautogui
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("Oracle Gem")

# Tentukan di mana screenshot akan disimpan
SCREENSHOT_DIR = Path.home() / "Oracle Gem" / "screenshots"

def take_progress_screenshot() -> str:
    """
    Mengambil screenshot dari seluruh layar dan menyimpannya ke folder yang ditentukan.
    
    Fungsi ini akan membuat folder 'screenshots' jika belum ada dan menyimpan
    gambar dengan nama file yang unik berdasarkan timestamp.

    Returns:
        str: Pesan yang berisi path ke file screenshot yang berhasil disimpan.
    """
    try:
        # 1. Buat direktori penyimpanan jika belum ada
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Buat nama file yang unik
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = SCREENSHOT_DIR / f"progress_snapshot_{timestamp}.png"

        # 3. Ambil screenshot
        logger.info("📸 Mengambil screenshot...")
        screenshot = pyautogui.screenshot()
        
        # 4. Simpan file
        screenshot.save(file_path)
        
        success_message = f"✅ Screenshot berhasil disimpan di: {file_path.resolve()}"
        logger.info(success_message)
        return success_message
        
    except Exception as e:
        error_message = f"❌ Gagal mengambil screenshot. Error: {e}"
        logger.error(error_message, exc_info=True)
        return error_message
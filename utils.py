# File: utils.py

import sys
import time
import threading
import logging

# Pindahkan logger ke sini agar bisa diakses oleh kelas
logger = logging.getLogger("Oracle Gem")

class StatusLogger:
    """
    Menampilkan animasi status yang dinamis di terminal untuk tugas yang berjalan lama.
    Pesan dapat diubah saat animasi berjalan.
    """
    def __init__(self, message="Memulai tugas...", emoji="⏳"):
        self._lock = threading.Lock()
        self.message = message
        self.emoji = emoji
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._stop_event = threading.Event()

    def _animate(self):
        """Metode internal untuk menjalankan animasi di thread terpisah."""
        chars = "|/-\\"
        i = 0
        while not self._stop_event.is_set():
            with self._lock:
                # Ambil pesan dan emoji terbaru dengan aman
                current_message = self.message
                current_emoji = self.emoji
            
            # Hapus baris sebelumnya dan tulis yang baru
            sys.stdout.write('\r' + ' ' * 120 + '\r') # Hapus baris dengan spasi
            sys.stdout.write(f"\r{current_emoji} {current_message} {chars[i % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        """Memulai animasi."""
        if not self._thread.is_alive():
            self._thread.start()

    def update_message(self, new_message: str, new_emoji: str = None):
        """
        Memperbarui pesan (dan emoji, jika ada) yang ditampilkan oleh animasi
        secara thread-safe.
        """
        with self._lock:
            self.message = new_message
            if new_emoji:
                self.emoji = new_emoji

    def stop(self, final_message="Selesai.", final_emoji="✅"):
        """Menghentikan animasi dan mencetak pesan final."""
        if self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        
        # Hapus baris animasi terakhir dan cetak pesan final
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        sys.stdout.flush()
        # Gunakan logger untuk pesan final agar konsisten
        logger.info(f"{final_emoji} {final_message}")


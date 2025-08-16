# Isi untuk file: src/models/model_alpha/proactive_monitor.py

import time
import random
import threading
from pathlib import Path
import logging

# Impor fungsi screenshot yang sudah kita buat
from .tools.system_tools import take_progress_screenshot

logger = logging.getLogger("Oracle Gem")

class ProactiveJarvisMonitor:
    def __init__(self, message_queue, interval_seconds=60):
        self.message_queue = message_queue
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.last_check_time = time.time()

    def start(self):
        logger.info("🤖 Otak Proaktif Jarvis diaktifkan. Memantau sistem di latar belakang...")
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def check_for_interesting_events(self):
        """Fungsi inti yang memeriksa kondisi sistem dan menghasilkan pesan proaktif."""
        
        # Contoh 1: Memberi sapaan di pagi hari (hanya sekali)
        # Di implementasi nyata, Anda akan menyimpan status "sudah menyapa" di file/database
        now_hour = time.localtime().tm_hour
        if 8 <= now_hour < 10 and (time.time() - self.last_check_time > 3600): # Cek sejam sekali
             self.last_check_time = time.time()
             sapaan = random.choice([
                 "Selamat pagi, Tuan. Sistem telah siaga dan siap untuk menerima instruksi.",
                 "Pagi, Tuan. Semua sistem berjalan normal. Apa yang bisa saya bantu hari ini?",
                 "Analisis semalam telah selesai. Selamat pagi, Tuan."
             ])
             self.message_queue.put(sapaan)
             return # Hanya kirim satu pesan per siklus

        # Contoh 2: Memberi laporan progres training secara acak (simulasi)
        # Di implementasi nyata, fungsi ini akan membaca log MLflow atau status trainer
        if random.random() < 0.1: # Peluang 10% setiap siklus
            try:
                # Ambil screenshot dulu
                screenshot_path = take_progress_screenshot()
                # Buat pesan laporan
                progress_message = f"Sekadar informasi, Tuan. Saya telah mengambil snapshot dari progres saat ini. {screenshot_path}"
                self.message_queue.put(progress_message)
            except Exception as e:
                logger.error(f"Gagal mengambil screenshot proaktif: {e}")
        
        # Di sini Anda bisa menambahkan pengecekan lain:
        # - Cek file log untuk anomali.
        # - Cek database NSMM untuk "neuron" baru yang menarik.
        # - Menjalankan LLM untuk menghasilkan "ide inovasi" baru secara berkala.

    def run(self):
        """Loop utama yang berjalan di latar belakang."""
        while not self.stop_event.is_set():
            try:
                self.check_for_interesting_events()
            except Exception as e:
                logger.error(f"Error di dalam Proactive Monitor: {e}")
            
            # Tunggu interval berikutnya, bisa dibuat acak agar tidak monoton
            wait_time = self.interval + random.uniform(-10, 10)
            self.stop_event.wait(wait_time)
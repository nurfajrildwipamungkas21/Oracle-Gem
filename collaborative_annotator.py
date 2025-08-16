import json
import logging
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import re
from typing import List
from json import JSONDecodeError

# Asumsi file alpha.py berada di direktori yang sama atau dapat diakses
# Fungsi ini penting untuk ekstraksi dan validasi JSON yang andal.
try:
    from .worker_utils import robust_json_extract
except (ImportError, ModuleNotFoundError):
    # Fallback jika struktur file berbeda atau untuk pengujian mandiri.
    # Di lingkungan nyata, pastikan `robust_json_extract` dapat diimpor.
    logging.warning("Fungsi 'robust_json_extract' tidak ditemukan. Menggunakan fallback sederhana.")
    def robust_json_extract(raw: str, model: type[BaseModel]):
        try:
            data = json.loads(raw)
            return model(**data) if model else data
        except (JSONDecodeError, ValidationError):
            pass
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.S)
        if m:
            try:
                data = json.loads(m.group(1))
                return model(**data) if model else data
            except (JSONDecodeError, ValidationError):
                pass
        start, end = raw.find("{"), raw.rfind("}")
        if 0 <= start < end:
            snippet = raw[start: end + 1]
            try:
                data = json.loads(snippet)
                return model(**data) if model else data
            except (JSONDecodeError, ValidationError):
                pass
        cleaned = re.sub(r"'", '"', raw)
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        try:
            data = json.loads(cleaned)
            return model(**data) if model else data
        except (JSONDecodeError, ValidationError):
            return None


# Asumsi worker_utils ada dan berisi kelas-kelas yang diperlukan
try:
    from .worker_utils import StatusLogger, GrokLLM, TogetherLLM, WebSearchManager
except (ImportError, ModuleNotFoundError):
    logging.warning("Modul 'worker_utils' tidak ditemukan. Beberapa fungsionalitas mungkin tidak bekerja.")
    # Mendefinisikan kelas placeholder agar kode bisa berjalan
    class WebSearchManager:
        def search(self, query, max_results=10): return "Contoh konteks dari pencarian web."
    class TogetherLLM:
        def __init__(self, api_key, model_name): pass
        def chat(self, prompt): return '{"events": [{"date": "2023-10-26", "event_name": "ECB Interest Rate Hike", "event_type": "MACRO", "impact_score": -0.5, "summary": "ECB menaikkan suku bunga untuk melawan inflasi."}]}'


logger = logging.getLogger("Oracle Gem")

# --- DEFINISI MODEL DATA (PYDANTIC) ---

class Event(BaseModel):
    """Mendefinisikan struktur untuk satu peristiwa keuangan."""
    date: str = Field(..., description="Tanggal peristiwa dalam format YYYY-MM-DD.")
    event_name: str = Field(..., description="Nama singkat dan jelas dari peristiwa (maksimal 10 kata).")
    event_type: str = Field(..., description="Kategori: 'MACRO', 'CORPORATE', 'POLITIK', 'GEOPOLITIK', 'TEKNOLOGI', 'BLACK_SWAN'.")
    impact_score: float = Field(..., description="Estimasi dampak dari -1.0 (sangat negatif) hingga 1.0 (sangat positif).")
    summary: str = Field(..., description="Ringkasan 1-2 kalimat tentang peristiwa dan dampaknya.")

class EventList(BaseModel):
    """Model untuk menampung daftar peristiwa, memastikan LLM mengembalikan struktur yang benar."""
    events: List[Event] = Field(..., description="Daftar lengkap dari semua peristiwa yang diekstrak.")


# --- KELAS UTAMA UNTUK ANOTASI ---

class CrossValidationAnnotationEngine:
    """
    Mesin untuk melakukan riset peristiwa menggunakan LLM, mengekstraknya ke dalam
    format terstruktur, dan memvalidasinya.
    """
    def __init__(self, together_keys: dict, web_search_manager: WebSearchManager):
        """
        Inisialisasi mesin anotasi dengan kunci API dan manajer pencarian web.
        """
        self.web_search_manager = web_search_manager
        
        primary_api_key = together_keys.get("qwen_giant")
        if not primary_api_key:
            logger.error("[Annotation Engine] Kunci API 'qwen_giant' tidak ditemukan. Riset peristiwa tidak akan berjalan.")
            self.primary_agent = None
            return

        logger.info("[Annotation Engine] Menggunakan 'Qwen/Qwen2-72B-Instruct' sebagai agen riset utama.")
        self.primary_agent = TogetherLLM(
            api_key=primary_api_key,
            model_name="Qwen/Qwen2-72B-Instruct"
        )

    def _perform_analysis(self, full_prompt: str) -> list[Event]: 
        if not self.primary_agent:
            return []
        try:
            # Tidak perlu lagi membuat search_context di sini, karena sudah ada di prompt
            response_str = self.primary_agent.chat(full_prompt)
            print(f"\nDEBUG: Respons Mentah dari LLM -> {response_str}\n")
            
            analysis_result = robust_json_extract(response_str, model=EventList)
            if not analysis_result or not analysis_result.events:
                logger.warning("-> Agen riset tidak menghasilkan peristiwa yang valid setelah parsing.")
                return []
            
            logger.info(f"✔️ Agen riset berhasil mengekstrak {len(analysis_result.events)} peristiwa valid.")
            return analysis_result.events
        except Exception as e:
            logger.error(f"-> Proses analisis agen riset gagal: {e}", exc_info=True)
            return []

    # Fungsi ini sekarang yang bertanggung jawab membuat prompt spesifik
    def generate_master_timeline(self, tickers: list[str], start_date: str, end_date: str) -> list[Event]:
        if not self.primary_agent:
            return []

        # 1. Dapatkan konteks dari pencarian web
        search_query = f"Significant global economic, financial, or corporate events affecting {', '.join(tickers)} between {start_date} and {end_date}"
        search_context = self.web_search_manager.search(search_query, max_results=10)
        
        if not search_context:
            logger.warning("-> Tidak mendapatkan konteks dari web. Riset dihentikan.")
            return []

        # 2. BANGUN PROMPT LENGKAP DI SINI
        prompt = f"""
        Analyze the following financial news context and extract significant events that occurred between {start_date} and {end_date}.
        Your output MUST be a valid JSON object matching the EventList schema.

        IMPORTANT RULES:
        - The 'date' MUST be in 'YYYY-MM-DD' format. Use 'YYYY-MM-01' if the day is unknown. DO NOT use '??'.
        - The 'event_name' should be a concise title.
        - The 'event_type' must be one of: 'MACRO', 'CORPORATE', 'POLITIK', 'GEOPOLITIK', 'TEKNOLOGI', 'BLACK_SWAN'.
        - The 'impact_score' is a float between -1.0 and 1.0.
        - The 'summary' is a brief 1-2 sentence explanation.

        NEWS CONTEXT:
        ---
        {search_context}
        ---

        Your JSON response (it must only contain this JSON object):
        ```json
        {{
          "events": [
            {{
              "date": "YYYY-MM-DD",
              "event_name": "Concise event title here",
              "event_type": "CORPORATE",
              "impact_score": 0.7,
              "summary": "A short summary of the event and its impact."
            }}
          ]
        }}
        ```
        """
        
        logger.info(f"🌐 Memulai riset peristiwa dengan agen utama (Qwen)...")
        
        # 3. KIRIM PROMPT YANG SUDAH JADI KE FUNGSI EKSEKUSI
        raw_events = self._perform_analysis(prompt)

        if not raw_events:
            logger.warning("Tidak ada peristiwa ditemukan oleh agen riset.")
            return []
            
        logger.info(f"✅ Riset Selesai. Total peristiwa yang terkumpul: {len(raw_events)}")
        return raw_events
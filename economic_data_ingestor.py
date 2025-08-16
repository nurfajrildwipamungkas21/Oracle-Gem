import requests
import sqlite3
import logging
import json # Tambahkan import json
from datetime import datetime

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi ---
HEADERS = {'User-Agent': 'Mozilla/5.0'}
DB_PATH = 'alpha_internal.db'
API_KEYS_PATH = 'src/models/model_alpha/api_keys.json' # Path ke file kunci Anda

# --- Fungsi untuk memuat kunci API ---
def load_api_keys():
    """Memuat semua kunci API dari file JSON."""
    try:
        with open(API_KEYS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File kunci API tidak ditemukan di {API_KEYS_PATH}")
        return {}

# --- Fungsi Pengambilan Data yang Diperbarui ---

def get_cpi_data(api_key: str): # Terima API key sebagai argumen
    """Mengambil data CPI terbaru dari BLS menggunakan API key."""
    if not api_key:
        logging.error("API Key BLS tidak tersedia. Melewatkan pengambilan data CPI.")
        return {"status": "error", "message": "API Key BLS tidak ada."}
        
    try:
        series_id = 'CUUR0000SA0'
        end_year = datetime.now().year
        start_year = end_year - 2
        
        # --- PERUBAHAN UTAMA: Tambahkan API Key ke URL ---
        url = f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}?startyear={start_year}&endyear={end_year}&registrationkey={api_key}"
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()

        # --- PERBAIKAN: Cek status respons dari API ---
        if data.get('status') != 'REQUEST_SUCCEEDED' or not data.get('Results'):
            error_message = data.get('message', ['Unknown error from BLS API.'])[0]
            raise Exception(error_message)
            
        series_data = data['Results']['series'][0]['data']
        
        latest_cpi = series_data[0]
        logging.info(f"CPI Data Fetched: {latest_cpi['periodName']} {latest_cpi['year']} - Value: {latest_cpi['value']}")
        return {"status": "success", "data": latest_cpi}
    except Exception as e:
        logging.error(f"Gagal mengambil data CPI: {e}")
        return {"status": "error", "message": str(e)}

def get_fed_rate_data(api_key: str):
    """Mengambil data Federal Funds Effective Rate terbaru dari FRED."""
    if not api_key:
        logging.error("API Key FRED tidak tersedia. Melewatkan pengambilan data Fed Rate.")
        return {"status": "error", "message": "API Key FRED tidak ada."}

    try:
        series_id = 'DFF'
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc&limit=1"
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('observations'):
             raise Exception("Respons API FRED tidak berisi data 'observations'.")

        latest_rate = data['observations'][0]
        # Ubah nilai '.' menjadi 0.0 agar tidak error saat konversi ke float
        if latest_rate['value'] == '.':
            latest_rate['value'] = '0.0'
            logging.warning(f"Nilai Fed Rate untuk {latest_rate['date']} tidak tersedia ('.'), diubah menjadi 0.0.")

        logging.info(f"Fed Rate Fetched: {latest_rate['date']} - Rate: {latest_rate['value']}%")
        return {"status": "success", "data": latest_rate}
    except Exception as e:
        logging.error(f"Gagal mengambil data Fed Rate: {e}")
        return {"status": "error", "message": str(e)}

# --- Fungsi Database (Tidak ada perubahan) ---

def store_economic_data(event_type, event_data):
    """Menyimpan data ekonomi ke database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS economic_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_date TEXT,
            event_type TEXT,
            value REAL,
            full_data TEXT,
            ingested_at TEXT,
            UNIQUE(event_date, event_type)
        )
        ''')
        
        # Logika parsing tanggal yang lebih robust
        date_str = event_data.get('date')
        if not date_str:
            month = event_data.get('period').replace('M', '')
            date_str = f"{event_data.get('year')}-{month}-01"

        value = float(event_data.get('value'))
        
        # Gunakan INSERT OR IGNORE untuk menghindari duplikasi data
        cursor.execute('''
        INSERT OR IGNORE INTO economic_events (event_date, event_type, value, full_data, ingested_at)
        VALUES (?, ?, ?, ?, ?)
        ''', (date_str, event_type, value, str(event_data), datetime.now().isoformat()))
        
        conn.commit()
        logging.info(f"Data '{event_type}' berhasil disimpan/diperbarui di database.")
    except Exception as e:
        logging.error(f"Gagal menyimpan data ke DB: {e}")
    finally:
        if conn:
            conn.close()

# --- Fungsi Utama yang Diperbarui ---

def run_ingestion_pipeline():
    """Menjalankan seluruh pipeline pengambilan data ekonomi."""
    logging.info("Memulai pipeline ingesti data ekonomi...")
    
    api_keys = load_api_keys()
    
    # 1. Ambil data CPI menggunakan kunci dari file
    cpi_result = get_cpi_data(api_keys.get('bls_key'))
    if cpi_result['status'] == 'success':
        store_economic_data('CPI', cpi_result['data'])
        
    # 2. Ambil data Suku Bunga Fed menggunakan kunci dari file
    # Asumsi kunci FRED disimpan di bawah "together_roles" atau nama lain yang sesuai
    fred_key = api_keys.get('fred_key') # Anda mungkin perlu menambahkan 'fred_key' di api_keys.json
    if not fred_key:
        logging.warning("Kunci API FRED ('fred_key') tidak ditemukan di api_keys.json. Mencoba tanpa kunci...")

    fed_rate_result = get_fed_rate_data(fred_key)
    if fed_rate_result['status'] == 'success':
        store_economic_data('FED_RATE', fed_rate_result['data'])
        
    logging.info("Pipeline ingesti data ekonomi selesai.")

if __name__ == '__main__':
    run_ingestion_pipeline()
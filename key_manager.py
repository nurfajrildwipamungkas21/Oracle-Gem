import json
import sys
from pathlib import Path
import questionary

# Path ke file konfigurasi tempat kunci API akan disimpan
KEYS_FILE = Path(__file__).parent / "api_keys.json"

try:
    # Impor ini akan berfungsi saat skrip dijalankan dari folder root proyek.
    from src.models.model_alpha.config import together_roles
except ImportError:
    print("FATAL: Gagal mengimpor 'together_roles'.")
    print("Pastikan Anda menjalankan skrip dari folder root proyek (misal: 'C:\\Users\\Msi\\Oracle Gem')")
    print("dan file 'config.py' ada di dalam 'src/models/model_alpha/'.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Terjadi error saat mengimpor config: {e}")
    sys.exit(1)


def save_keys(keys_data):
    """Menyimpan data kunci API ke file JSON."""
    with KEYS_FILE.open("w", encoding="utf-8") as f:
        json.dump(keys_data, f, indent=4)
    print(f"✅ Kunci API berhasil disimpan di: {KEYS_FILE}")

# =========================================================================
# === BLOK KODE BARU DITEMPATKAN DI SINI ===
# =========================================================================
def update_specific_key_interactively(key_name: str, is_list: bool = False) -> bool:
    """Meminta pengguna untuk memperbarui satu kunci API spesifik secara real-time."""
    
    # Muat semua kunci yang ada saat ini
    current_keys = {}
    if KEYS_FILE.exists():
        with KEYS_FILE.open("r") as f:
            current_keys = json.load(f)

    print(f"\n🚨 Terjadi masalah dengan API Key untuk '{key_name}'.")
    
    user_input = None # Inisialisasi variabel
    if is_list:
        prompt_text = f"Masukkan Bearer Token X/Twitter baru (jika >1, pisahkan koma):"
        user_input = questionary.text(prompt_text).ask()
        current_keys[key_name] = [k.strip() for k in user_input.split(',')] if user_input else []
    else:
        # Menangani kunci biasa (string) dan kunci dalam dictionary (together_roles)
        if key_name in current_keys.get("together_roles", {}):
             prompt_text = f"Masukkan API Key baru untuk '{key_name}':"
             user_input = questionary.text(prompt_text).ask()
             if user_input:
                 current_keys["together_roles"][key_name] = user_input.strip()
        else:
            prompt_text = f"Masukkan API Key baru untuk '{key_name}':"
            user_input = questionary.text(prompt_text).ask()
            if user_input:
                current_keys[key_name] = user_input.strip()

    if user_input:
        save_keys(current_keys)
        print("✅ Kunci API telah diperbarui. Mencoba ulang operasi...")
        return True
    else:
        print("❌ Tidak ada kunci baru yang dimasukkan. Operasi mungkin akan gagal lagi.")
        return False


def manage_api_keys():
    """
    Versi final yang tangguh untuk mengelola API keys.
    Memuat kunci yang ada dan secara cerdas hanya meminta kunci yang benar-benar hilang atau kosong.
    """
    loaded_keys = {}
    if KEYS_FILE.exists():
        try:
            with KEYS_FILE.open("r", encoding="utf-8") as f:
                loaded_keys = json.load(f)
            print(f"🔑 Kunci API yang ada dimuat dari {KEYS_FILE}...")
        except (json.JSONDecodeError, IOError):
            print(f"⚠️ File {KEYS_FILE} rusak atau tidak bisa dibaca. Akan memulai dari awal.")
            loaded_keys = {}

    prompted_headers = set()

    def print_header(header_name):
        if header_name not in prompted_headers:
            print(f"\n--- 🔑 {header_name} ---")
            prompted_headers.add(header_name)

    # 1. Cek kunci Gemini
    if not loaded_keys.get("gemini"):
        print_header("Kunci Google Gemini Diperlukan")
        gemini_input = questionary.text("Masukkan Kunci API Google Gemini (jika >1, pisahkan koma):").ask()
        loaded_keys["gemini"] = [k.strip() for k in gemini_input.split(',')] if gemini_input else []

    # 2. Cek kunci Together.AI & Grok (YANG DIPERBAIKI)
    loaded_keys.setdefault("together_api_keys", {}) # <-- PERBAIKAN 1: Nama diubah
    
    # Cek roles dari config.py
    missing_roles = [role for role in together_roles if role not in loaded_keys["together_api_keys"] or not loaded_keys["together_api_keys"].get(role)]
    if missing_roles:
        print_header("Kunci Model Eksternal (Together.AI)")
        for role in missing_roles:
            config = together_roles[role]
            key = questionary.text(f"Masukkan API Key untuk '{role}' ({config.get('desc', 'No description')}):").ask()
            if key:
                loaded_keys["together_api_keys"][role] = key.strip() # <-- PERBAIKAN 1: Disimpan di sini

    # PERBAIKAN 2: Cek grok_key secara terpisah dan simpan di lokasi yang benar
    if "grok_key" not in loaded_keys.get("together_api_keys", {}):
        print_header("Kunci Model Eksternal (Together.AI)")
        grok_input = questionary.text("Masukkan API Key X.ai (Grok):").ask()
        if grok_input:
            loaded_keys["together_api_keys"]["grok_key"] = grok_input.strip()

    # 3. Cek semua kunci lainnya satu per satu
    other_keys_map = {
        "fred_key": "Masukkan API Key FRED (Data Ekonomi):",
        "bls_key": "Masukkan API Key BLS (Data Ekonomi):",
        "tavily": "Masukkan API Key Tavily (Web Search):",
        "Google Search": "Masukkan API Key Google Custom Search:",
        "google_cse_id": "Masukkan ID Google Programmable Search Engine:"
    }
    
    for key_name, prompt_text in other_keys_map.items():
        if not loaded_keys.get(key_name):
            print_header("Kunci Layanan Lainnya")
            user_input = questionary.text(prompt_text).ask()
            loaded_keys[key_name] = user_input.strip() if user_input else ""
    
    # 4. Cek X Bearer Token
    if not loaded_keys.get("x_bearer"):
        print_header("Kunci Layanan Lainnya")
        bearer_input = questionary.text("Masukkan Bearer Token X/Twitter (jika >1, pisahkan koma):").ask()
        loaded_keys["x_bearer"] = [k.strip() for k in bearer_input.split(',')] if bearer_input else []

    save_keys(loaded_keys)
    return loaded_keys

if __name__ == "__main__":
    # Memungkinkan skrip ini dijalankan secara mandiri untuk setup awal
    manage_api_keys()
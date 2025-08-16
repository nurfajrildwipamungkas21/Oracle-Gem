# Isi lengkap untuk tools/config_healer.py

from pathlib import Path
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

@tool
def fix_misconfiguration_exception(file_path: str, function_name: str = "_one_train_run") -> str:
    """
    Gunakan HANYA untuk memperbaiki MisconfigurationException akibat hparams tidak sinkron antara
    LightningModule dan DataModule. Tool ini akan menyisipkan kode sinkronisasi yang benar
    setelah dm.setup() di dalam fungsi _one_train_run.
    """
    try:
        target_file = Path(file_path)
        if not target_file.exists():
            return f"Error: File tidak ditemukan di {file_path}"

        sync_code = """    # ==============================================================================
    # == PERBAIKAN OTOMATIS OLEH SELF-HEALING AGENT                             ==
    # ==============================================================================
    final_n_features = dm.n_features_input
    final_n_targets = dm.n_targets
    
    logger.info(f"Sinkronisasi HParams: n_features_input final = {final_n_features}, n_targets final = {final_n_targets}")
    
    current_hparams['n_features_input'] = final_n_features
    current_hparams['n_targets'] = final_n_targets
    # =============================================================================="""
        
        lines = target_file.read_text(encoding='utf-8').splitlines()
        
        # Cek apakah kode sudah ada untuk menghindari duplikasi
        if any("PERBAIKAN OTOMATIS OLEH SELF-HEALING AGENT" in line for line in lines):
            return "Peringatan: Patch sinkronisasi sepertinya sudah pernah diterapkan. Tidak ada tindakan yang diambil untuk menghindari duplikasi."

        in_function = False
        func_def_line = -1
        insertion_point = -1

        for i, line in enumerate(lines):
            if f"def {function_name}(" in line:
                in_function = True
                func_def_line = i
            if in_function and "dm.setup(stage='fit')" in line:
                insertion_point = i + 1
                break

        if insertion_point == -1:
            return f"Error: Tidak dapat menemukan titik penyisipan yang cocok (setelah dm.setup()) di fungsi '{function_name}'."
            
        # Dapatkan indentasi dari baris referensi untuk menyisipkan kode dengan benar
        indentation = ' ' * (len(lines[insertion_point - 1]) - len(lines[insertion_point - 1].lstrip(' ')))
        indented_sync_code = "\\n".join([indentation + s_line for s_line in sync_code.splitlines()])

        lines.insert(insertion_point, indented_sync_code)
        target_file.write_text("\n".join(lines), encoding='utf-8')
        
        return f"Berhasil menerapkan patch sinkronisasi di {file_path}. Program perlu di-restart. Harap jalankan ulang skrip."
    except Exception as e:
        return f"Gagal menerapkan patch: {e}" 

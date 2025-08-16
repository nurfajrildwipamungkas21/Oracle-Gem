import pandas as pd
import numpy as np
import os
import re 
import logging 
from typing import List, Tuple, Dict, Optional, Any, Literal

# --- Konfigurasi Global ---
INPUT_PROCESSED_DIR: str = r"C:\Users\Msi\Oracle Gem\data\processed"
INPUT_FILENAME: str = "unified_market_data.csv"
OUTPUT_DATAPROC_DIR: str = r"C:\Users\Msi\Oracle Gem\src\data_processing"
OUTPUT_FILENAME: str = "validated_cleaned_market_data.csv"
LOG_FILENAME: str = "feature_builder_log.txt"

KNOWN_INDEX_TICKERS: List[str] = ["SPX", "N225", "NDAQ"] 
ALLOWED_SUFFIXES: List[str] = ["Open", "High", "Low", "Close", "Volume", "PctChange", 
                               "HighLow_Inconsistent_Flag", "Open_vs_HL_Inconsistent_Flag", 
                               "Close_vs_HL_Inconsistent_Flag", "PriceChange_Outlier_Flag", 
                               "Volume_Outlier_Flag"]
STANDARDIZED_PCT_CHANGE_SUFFIX: str = "PctChange"

AnomalyAction = Literal['none', 'flag', 'drop', 'winsor']

logger: Optional[logging.Logger] = None 

def setup_logging(log_file_path: str) -> logging.Logger:
    """
    Mengkonfigurasi dan mengembalikan instance logger global.
    Log akan ditulis ke file (overwrite) dan dicetak ke konsol.
    """
    global logger 
    logger = logging.getLogger("FeatureBuilder") 
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(logging.INFO) 

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(funcName)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def _clean_and_convert_pct_change(series: pd.Series, original_series_for_percent_check: pd.Series) -> pd.Series:
    """
    Membersihkan kolom string persentase perubahan, mengonversinya ke float, 
    dan membagi 100 HANYA JIKA string asli mengandung '%'.

    Args:
        series (pd.Series): Series yang akan dibersihkan (biasanya sudah hasil rename).
        original_series_for_percent_check (pd.Series): Series asli (sebelum rename atau perubahan) 
                                                       untuk memeriksa keberadaan '%'.

    Returns:
        pd.Series: Series numerik (float) yang merepresentasikan persentase perubahan desimal.
    """
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")

    if pd.api.types.is_numeric_dtype(series):
        # Jika sudah numerik, kita perlu cek apakah ini format persen (misal 5.0) atau desimal (0.05)
        # Untuk amannya, jika tidak ada '%' di string asli, kita asumsikan sudah desimal.
        # Jika ada '%', kita asumsikan perlu dibagi 100 bahkan jika sudah numerik.
        # Ini memerlukan akses ke string asli lagi.
        logger.debug(f"Seri PctChange sudah numerik: {series.name if hasattr(series, 'name') else 'Unnamed Series'}")
        # Jika ingin lebih ketat:
        # common_index = series.index.intersection(original_series_for_percent_check.index)
        # if not common_index.empty:
        #     has_percent_sign_mask_numeric = original_series_for_percent_check.loc[common_index].astype(str).str.contains('%', na=False)
        #     has_percent_sign_mask_numeric = has_percent_sign_mask_numeric.reindex(series.index, fill_value=False)
        #     series_to_divide = series.loc[has_percent_sign_mask_numeric & series.notna()]
        #     if not series_to_divide.empty:
        #         logger.info(f"  Membagi nilai numerik PctChange dengan 100 karena string asli mengandung '%'.")
        #         series.loc[has_percent_sign_mask_numeric & series.notna()] /= 100.0
        return series 
    
    if series.empty:
        return series.astype(float)

    common_index = series.index.intersection(original_series_for_percent_check.index)
    has_percent_sign_mask = pd.Series(False, index=series.index) # Default ke False
    if not common_index.empty :
        has_percent_sign_mask_temp = original_series_for_percent_check.loc[common_index].astype(str).str.contains('%', na=False)
        has_percent_sign_mask = has_percent_sign_mask_temp.reindex(series.index, fill_value=False)
    elif not series.empty:
         logger.warning(f"Indeks tidak cocok antara series PctChange dan series asli untuk cek '%'. Tidak bisa menentukan pembagian 100 dengan aman.")

    cleaned_series_str = series.astype(str).str.strip().replace('', np.nan)
    cleaned_series_str = cleaned_series_str.str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    numeric_series = pd.to_numeric(cleaned_series_str, errors='coerce')
    
    final_series = numeric_series.copy()
    mask_to_divide = has_percent_sign_mask & numeric_series.notna()
    if mask_to_divide.any():
        final_series.loc[mask_to_divide] = numeric_series.loc[mask_to_divide] / 100.0
        logger.info(f"  Menerapkan pembagian 100 untuk {mask_to_divide.sum()} nilai PctChange yang memiliki '%' pada string asli.")
    
    return final_series

def standardize_pct_change_column_name(df_asset: pd.DataFrame, asset_prefix: str) -> pd.DataFrame:
    """
    Mencari kolom persentase perubahan, menstandarkannya menjadi PREFIX_PctChange.
    Jika tidak ada dan kolom Close valid, akan menghitungnya.
    NaN pertama hasil .pct_change() akan diisi dengan 0.0.
    Melakukan pemeriksaan skala akhir.

    Args:
        df_asset (pd.DataFrame): DataFrame untuk satu aset.
        asset_prefix (str): Prefix nama aset.

    Returns:
        pd.DataFrame: DataFrame aset dengan kolom persentase perubahan yang distandarisasi.
    """
    global logger
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")

    df_copy = df_asset.copy()
    standard_change_col_name = f"{asset_prefix}_{STANDARDIZED_PCT_CHANGE_SUFFIX}"
    close_col = f"{asset_prefix}_Close"
    
    original_change_col_name_found: Optional[str] = None
    original_series_for_pct_check: Optional[pd.Series] = None 

    if standard_change_col_name in df_copy.columns:
        logger.info(f"Kolom standar {standard_change_col_name} sudah ada untuk {asset_prefix}.")
        original_change_col_name_found = standard_change_col_name
        original_series_for_pct_check = df_copy[standard_change_col_name].copy() 
    else:
        change_suffix_patterns = [
            re.compile(r"^perubahan%$"), re.compile(r"^change%$"), 
            re.compile(r"^pct_change$"), re.compile(r"^change_%$"), 
            re.compile(r"^ret$")
        ]
        for col in df_copy.columns: # Iterasi pada kolom yang ada di df_copy
            if col.startswith(f"{asset_prefix}_"):
                original_col_suffix = col.replace(f"{asset_prefix}_", "", 1)
                for pattern in change_suffix_patterns:
                    if pattern.match(original_col_suffix):
                        logger.info(f"Menstandarkan nama kolom '{col}' menjadi '{standard_change_col_name}' untuk {asset_prefix}.")
                        original_series_for_pct_check = df_copy[col].copy() 
                        df_copy = df_copy.rename(columns={col: standard_change_col_name})
                        original_change_col_name_found = standard_change_col_name
                        break
                if original_change_col_name_found:
                    break
            
    if original_change_col_name_found is not None and original_series_for_pct_check is not None:
        logger.info(f"Membersihkan dan mengonversi kolom {standard_change_col_name}...")
        df_copy[standard_change_col_name] = _clean_and_convert_pct_change(df_copy[standard_change_col_name], original_series_for_pct_check)
    
    should_recalculate_from_close = False
    if standard_change_col_name not in df_copy.columns:
        logger.warning(f"Tidak ditemukan kolom persentase perubahan yang dikenal untuk {asset_prefix}.")
        should_recalculate_from_close = True
    # Cek apakah kolom ada DAN semua nilainya NaN setelah upaya konversi
    elif standard_change_col_name in df_copy.columns and df_copy[standard_change_col_name].isna().all():
        logger.warning(f"Kolom {standard_change_col_name} menjadi semua NaN setelah upaya konversi/pembersihan. Akan dihitung ulang dari {close_col}.")
        should_recalculate_from_close = True

    if should_recalculate_from_close:
        if close_col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[close_col]):
            close_series_no_initial_nans = df_copy[close_col].dropna()
            if len(close_series_no_initial_nans) > 1:
                pct_change_values = close_series_no_initial_nans.pct_change()
                # Assign kembali ke DataFrame asli, hanya untuk indeks yang ada di pct_change_values
                df_copy[standard_change_col_name] = pct_change_values 
                logger.info(f"  {standard_change_col_name} dihitung ulang/dihitung untuk {asset_prefix}.")
            elif len(close_series_no_initial_nans) <= 1: # Jika hanya 0 atau 1 data Close valid
                df_copy[standard_change_col_name] = np.nan
                logger.info(f"  Tidak cukup data Close valid ({len(close_series_no_initial_nans)}) untuk menghitung {standard_change_col_name} untuk {asset_prefix}, diisi NaN.")
        else:
            logger.warning(f"Tidak dapat menghitung {standard_change_col_name} karena kolom {close_col} tidak valid atau tidak ada untuk {asset_prefix}.")
            df_copy[standard_change_col_name] = np.nan # Pastikan kolom ada sebagai semua NaN
    
    if standard_change_col_name in df_copy.columns:
        df_copy[standard_change_col_name] = pd.to_numeric(df_copy[standard_change_col_name], errors='coerce')
        if pd.api.types.is_numeric_dtype(df_copy[standard_change_col_name]):
            if close_col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[close_col]):
                first_valid_close_idx = df_copy[close_col].first_valid_index()
                if first_valid_close_idx is not None and \
                   first_valid_close_idx in df_copy.index and \
                   pd.isna(df_copy.loc[first_valid_close_idx, standard_change_col_name]):
                    # Hanya isi NaN pertama jika ada data valid setelahnya atau hanya satu baris data Close
                    if df_copy.loc[df_copy.index > first_valid_close_idx, standard_change_col_name].notna().any() or \
                       (len(df_copy[standard_change_col_name].dropna()) == 0 and len(df_copy.dropna(subset=[close_col])) == 1) :
                        df_copy.loc[first_valid_close_idx, standard_change_col_name] = 0.0
                        logger.info(f"  NaN pertama pada {standard_change_col_name} di {first_valid_close_idx} (awal data valid Close) diisi dengan 0.0.")
            
            # Pemeriksaan Skala PctChange
            if df_copy[standard_change_col_name].notna().any():
                abs_pct_change = df_copy[standard_change_col_name].abs()
                # Cek jika ada nilai absolut > 1 (indikasi > 100% change, mungkin salah skala)
                # Hanya pada nilai yang tidak NaN
                if not abs_pct_change.empty and (abs_pct_change > 1).any():
                    logger.warning(f"Ditemukan nilai PctChange absolut > 1 (100%) untuk {asset_prefix}. Periksa skala. Contoh: {df_copy.loc[abs_pct_change[abs_pct_change > 1].index, standard_change_col_name].head(3).tolist()}")
        else:
            logger.error(f"Kolom {standard_change_col_name} masih bukan numerik setelah semua upaya untuk {asset_prefix}.")
    
    return df_copy

# [SISA FUNGSI (validate_ohlc_consistency, validate_numeric_values, handle_missing_values_asset, 
# handle_outliers_asset, restructure_and_finalize, build_features, dan if __name__ == '__main__') 
# DISALIN DARI VERSI SEBELUMNYA (v11) DAN DIPASTIKAN MENGGUNAKAN LOGGER GLOBAL]

def validate_ohlc_consistency(
    df_asset: pd.DataFrame, 
    asset_prefix: str, 
    action: AnomalyAction = 'drop'
) -> pd.DataFrame:
    global logger
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")
    logger.info(f"--- Memulai Validasi Konsistensi OHLC untuk {asset_prefix} (Aksi: {action}) ---")
    df_copy = df_asset.copy()
    original_rows = len(df_copy)
    col_open = f"{asset_prefix}_Open"; col_high = f"{asset_prefix}_High"
    col_low = f"{asset_prefix}_Low"; col_close = f"{asset_prefix}_Close"
    col_volume = f"{asset_prefix}_Volume"
    ohlc_cols = [col_open, col_high, col_low, col_close]

    if not all(col in df_copy.columns for col in ohlc_cols):
        logger.warning(f"Kolom OHLC tidak lengkap untuk {asset_prefix}. Melewati validasi OHLC.")
        return df_copy

    for col in ohlc_cols:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            logger.warning(f"Kolom {col} bukan numerik saat validasi OHLC. Mencoba konversi...")
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            if df_copy[col].isna().all():
                logger.error(f"Kolom {col} menjadi semua NaN setelah konversi paksa. Validasi OHLC mungkin tidak akurat.")
                return df_copy 

    inconsistent_high_low_mask = df_copy[col_high] < df_copy[col_low]
    if inconsistent_high_low_mask.any():
        num_inconsistent = inconsistent_high_low_mask.sum()
        logger.warning(f"Ditemukan {num_inconsistent} baris dimana High < Low untuk {asset_prefix}.")
        if action == 'flag':
            df_copy[f'{asset_prefix}_HighLow_Inconsistent_Flag'] = inconsistent_high_low_mask
        elif action == 'drop':
            logger.info(f"  Menghapus {num_inconsistent} baris dengan High < Low untuk {asset_prefix}.")
            df_copy = df_copy[~inconsistent_high_low_mask]
        elif action == 'winsor': 
            logger.info(f"  Winsorizing {num_inconsistent} baris dengan High < Low untuk {asset_prefix} (Low diatur sama dengan High).")
            df_copy.loc[inconsistent_high_low_mask, col_low] = df_copy.loc[inconsistent_high_low_mask, col_high]


    valid_hl_range = df_copy[col_low].notna() & df_copy[col_high].notna()

    open_outside_hl_mask = valid_hl_range & ((df_copy[col_open] < df_copy[col_low]) | (df_copy[col_open] > df_copy[col_high]))
    if open_outside_hl_mask.any():
        num_inconsistent = open_outside_hl_mask.sum()
        logger.warning(f"Ditemukan {num_inconsistent} baris dimana Open di luar rentang [Low, High] untuk {asset_prefix}.")
        if action == 'flag':
            if f'{asset_prefix}_Open_vs_HL_Inconsistent_Flag' not in df_copy.columns: df_copy[f'{asset_prefix}_Open_vs_HL_Inconsistent_Flag'] = False
            df_copy.loc[open_outside_hl_mask, f'{asset_prefix}_Open_vs_HL_Inconsistent_Flag'] = True
        elif action == 'drop':
            logger.info(f"  Menghapus {num_inconsistent} baris dengan Open di luar [Low, High] untuk {asset_prefix}.")
            df_copy = df_copy[~open_outside_hl_mask]
        elif action == 'winsor':
            logger.info(f"  Winsorizing {num_inconsistent} nilai Open yang di luar [Low, High] untuk {asset_prefix}.")
            df_copy.loc[open_outside_hl_mask & (df_copy[col_open] < df_copy[col_low]), col_open] = df_copy[col_low]
            df_copy.loc[open_outside_hl_mask & (df_copy[col_open] > df_copy[col_high]), col_open] = df_copy[col_high]


    close_outside_hl_mask = valid_hl_range & ((df_copy[col_close] < df_copy[col_low]) | (df_copy[col_close] > df_copy[col_high]))
    if close_outside_hl_mask.any():
        num_inconsistent = close_outside_hl_mask.sum()
        logger.warning(f"Ditemukan {num_inconsistent} baris dimana Close di luar rentang [Low, High] untuk {asset_prefix}.")
        if action == 'flag':
            if f'{asset_prefix}_Close_vs_HL_Inconsistent_Flag' not in df_copy.columns: df_copy[f'{asset_prefix}_Close_vs_HL_Inconsistent_Flag'] = False
            df_copy.loc[close_outside_hl_mask, f'{asset_prefix}_Close_vs_HL_Inconsistent_Flag'] = True
        elif action == 'drop':
            logger.info(f"  Menghapus {num_inconsistent} baris dengan Close di luar [Low, High] untuk {asset_prefix}.")
            df_copy = df_copy[~close_outside_hl_mask]
        elif action == 'winsor':
            logger.info(f"  Winsorizing {num_inconsistent} nilai Close yang di luar [Low, High] untuk {asset_prefix}.")
            df_copy.loc[close_outside_hl_mask & (df_copy[col_close] < df_copy[col_low]), col_close] = df_copy[col_low]
            df_copy.loc[close_outside_hl_mask & (df_copy[col_close] > df_copy[col_high]), col_close] = df_copy[col_high]
    
    if col_volume in df_copy.columns and all(col in df_copy.columns for col in ohlc_cols): 
        mask_ohlc_not_nan = df_copy[ohlc_cols].notna().all(axis=1)
        no_movement_zero_volume = df_copy[
            mask_ohlc_not_nan & 
            (df_copy[col_open] == df_copy[col_high]) & 
            (df_copy[col_high] == df_copy[col_low]) &
            (df_copy[col_low] == df_copy[col_close]) & 
            (df_copy[col_volume] == 0)
        ]
        if not no_movement_zero_volume.empty:
            logger.info(f"Ditemukan {len(no_movement_zero_volume)} hari tanpa pergerakan harga (OHLC sama) dan Volume = 0 untuk {asset_prefix}.")

    rows_dropped = original_rows - len(df_copy)
    if rows_dropped > 0: 
        logger.info(f"Total {rows_dropped} baris dihapus dari {asset_prefix} karena inkonsistensi OHLC (aksi: {action}).")
    logger.info(f"--- Validasi Konsistensi OHLC untuk {asset_prefix} Selesai ---")
    return df_copy

def validate_numeric_values(df_asset: pd.DataFrame, asset_prefix: str) -> pd.DataFrame:
    global logger
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")
    logger.info(f"--- Memulai Validasi Nilai Numerik untuk {asset_prefix} ---")
    df_copy = df_asset.copy()
    price_cols = [f"{asset_prefix}_Open", f"{asset_prefix}_High", f"{asset_prefix}_Low", f"{asset_prefix}_Close"]
    volume_col = f"{asset_prefix}_Volume"
    pct_change_col = f"{asset_prefix}_{STANDARDIZED_PCT_CHANGE_SUFFIX}"

    for col in price_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            neg_prices_mask = df_copy[col] < 0
            if neg_prices_mask.any():
                logger.warning(f"{neg_prices_mask.sum()} harga negatif di {col} ({asset_prefix}) -> NaN.")
                df_copy.loc[neg_prices_mask, col] = np.nan
            zero_prices_mask = df_copy[col] == 0
            if zero_prices_mask.any():
                logger.warning(f"{zero_prices_mask.sum()} harga nol di {col} ({asset_prefix}) -> NaN.")
                df_copy.loc[zero_prices_mask, col] = np.nan
    
    if volume_col in df_copy.columns:
        df_copy[volume_col] = pd.to_numeric(df_copy[volume_col], errors='coerce') 
        neg_volume_mask = df_copy[volume_col] < 0
        if neg_volume_mask.any():
            logger.warning(f"{neg_volume_mask.sum()} volume negatif di {volume_col} ({asset_prefix}) -> NaN.")
            df_copy.loc[neg_volume_mask, volume_col] = np.nan
            
    if pct_change_col in df_copy.columns: 
        df_copy[pct_change_col] = pd.to_numeric(df_copy[pct_change_col], errors='coerce')
            
    logger.info(f"--- Validasi Nilai Numerik untuk {asset_prefix} Selesai ---")
    return df_copy

def handle_missing_values_asset(df_asset: pd.DataFrame, asset_prefix: str) -> pd.DataFrame:
    global logger
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")
    logger.info(f"--- Memulai Pemeriksaan Missing Values untuk {asset_prefix} ---")
    df_copy = df_asset.copy()
    cols_to_check = [f"{asset_prefix}_Open", f"{asset_prefix}_High", f"{asset_prefix}_Low", 
                     f"{asset_prefix}_Close", f"{asset_prefix}_Volume", f"{asset_prefix}_{STANDARDIZED_PCT_CHANGE_SUFFIX}"]
    existing_cols_to_check = [col for col in cols_to_check if col in df_copy.columns]

    if not existing_cols_to_check:
        logger.info(f"Tidak ada kolom OHLCV atau PctChange standar untuk {asset_prefix}. Melewati.")
        return df_copy

    initial_nans = df_copy[existing_cols_to_check].isna().sum()
    if initial_nans.sum() > 0:
        logger.info(f"Memeriksa NaN di kolom untuk {asset_prefix}:")
        for col, count in initial_nans.items():
            if count > 0 and col in df_copy.columns : logger.info(f"    Kolom: {col}, Jumlah NaN awal: {count}")
        
        volume_col = f"{asset_prefix}_Volume"
        if volume_col in df_copy.columns: 
            if asset_prefix in KNOWN_INDEX_TICKERS:
                if df_copy[volume_col].isna().any() or (df_copy[volume_col].fillna(0) == 0).any(): 
                    logger.info(f"Aset {asset_prefix} adalah indeks. Volume 0 atau NaN akan diubah/dibiarkan menjadi NaN.")
                    df_copy.loc[df_copy[volume_col] == 0, volume_col] = np.nan 
            else: 
                first_valid_price_index = df_copy[[f"{asset_prefix}_Open", f"{asset_prefix}_Close"]].first_valid_index()
                if first_valid_price_index is not None:
                    mask_volume_before_valid_price = (df_copy.index < first_valid_price_index)
                    if mask_volume_before_valid_price.any():
                        condition_to_nan = mask_volume_before_valid_price & ((df_copy[volume_col] == 0) | df_copy[volume_col].isna())
                        if condition_to_nan.any():
                            df_copy.loc[condition_to_nan, volume_col] = np.nan
                            logger.info(f"Volume sebelum data harga valid untuk {asset_prefix} (saham) diubah/dibiarkan menjadi NaN.")
                    
                    mask_fill_volume_with_zero = (df_copy.index >= first_valid_price_index) & (df_copy[volume_col].isna())
                    if mask_fill_volume_with_zero.any():
                        logger.info(f"Mengisi {mask_fill_volume_with_zero.sum()} NaN pada {volume_col} untuk {asset_prefix} (saham, setelah data harga valid) dengan 0.")
                        df_copy.loc[mask_fill_volume_with_zero, volume_col] = 0
                else: 
                    logger.info(f"Tidak ada data harga valid untuk {asset_prefix}. Semua NaN/0 pada {volume_col} akan menjadi NaN.")
                    df_copy[volume_col] = np.nan

        price_cols = [f"{asset_prefix}_Open", f"{asset_prefix}_High", f"{asset_prefix}_Low", f"{asset_prefix}_Close"]
        for col in price_cols:
            if col in df_copy.columns and df_copy[col].isna().any():
                logger.info(f"Kolom harga {col} ({asset_prefix}) memiliki {df_copy[col].isna().sum()} NaN (dibiarkan).")
        
        pct_change_col = f"{asset_prefix}_{STANDARDIZED_PCT_CHANGE_SUFFIX}"
        if pct_change_col in df_copy.columns and df_copy[pct_change_col].isna().any():
            logger.info(f"Kolom {pct_change_col} ({asset_prefix}) memiliki {df_copy[pct_change_col].isna().sum()} NaN (dibiarkan).")
    else:
        logger.info(f"Tidak ada NaN ditemukan di kolom OHLCV & PctChange untuk {asset_prefix}.")
    logger.info(f"--- Pemeriksaan Missing Values untuk {asset_prefix} Selesai ---")
    return df_copy

def handle_outliers_asset(
    df_asset: pd.DataFrame, 
    asset_prefix: str, 
    price_outlier_action: AnomalyAction = 'none', 
    volume_outlier_action: AnomalyAction = 'none'
) -> pd.DataFrame:
    global logger
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")
    logger.info(f"--- Memulai Deteksi Outlier untuk {asset_prefix} (Harga: {price_outlier_action}, Vol: {volume_outlier_action}) ---")
    df_copy = df_asset.copy()
    pct_change_col = f"{asset_prefix}_{STANDARDIZED_PCT_CHANGE_SUFFIX}"
    volume_col = f"{asset_prefix}_Volume"

    if pct_change_col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[pct_change_col]):
        valid_pct_change = df_copy[pct_change_col].dropna()
        if len(valid_pct_change) > 10: 
            Q1_ret = valid_pct_change.quantile(0.25); Q3_ret = valid_pct_change.quantile(0.75)
            IQR_ret = Q3_ret - Q1_ret
            if IQR_ret > 1e-9: 
                lower_bound_ret = Q1_ret - 1.5 * IQR_ret; upper_bound_ret = Q3_ret + 1.5 * IQR_ret
                outlier_mask_ret = ((df_copy[pct_change_col] < lower_bound_ret) | (df_copy[pct_change_col] > upper_bound_ret)) & df_copy[pct_change_col].notna()
                
                if outlier_mask_ret.any():
                    num_outliers = outlier_mask_ret.sum()
                    logger.info(f"{num_outliers} outlier perubahan harga harian untuk {asset_prefix}. Batas IQR: [{lower_bound_ret:.4f}, {upper_bound_ret:.4f}]")
                    if price_outlier_action == 'flag':
                        flag_col_name = f'{asset_prefix}_PriceChange_Outlier_Flag'
                        if flag_col_name not in df_copy.columns: df_copy[flag_col_name] = False
                        df_copy.loc[outlier_mask_ret, flag_col_name] = True
                    elif price_outlier_action == 'drop':
                        logger.info(f"  Menghapus {num_outliers} baris karena outlier perubahan harga.")
                        df_copy = df_copy[~outlier_mask_ret]
                    elif price_outlier_action == 'winsor':
                        logger.info(f"  Menerapkan Winsorizing pada {num_outliers} outlier perubahan harga.")
                        df_copy.loc[df_copy[pct_change_col] < lower_bound_ret, pct_change_col] = lower_bound_ret
                        df_copy.loc[df_copy[pct_change_col] > upper_bound_ret, pct_change_col] = upper_bound_ret
            else: logger.info(f"IQR perubahan harga {asset_prefix} nol/kecil. Lewati deteksi outlier harga.")
        else: logger.info(f"Tidak cukup data perubahan harga valid ({len(valid_pct_change)}) untuk deteksi outlier IQR {asset_prefix}.")
    else: logger.info(f"Kolom {pct_change_col} tidak ada atau bukan numerik, lewati deteksi outlier harga.")

    if volume_col in df_copy.columns and (pd.api.types.is_numeric_dtype(df_copy[volume_col]) or str(df_copy[volume_col].dtype) == 'Float64'):
        if asset_prefix in KNOWN_INDEX_TICKERS:
            logger.info(f"Aset {asset_prefix} adalah indeks. Deteksi outlier volume berbasis IQR dilewati.")
        else: 
            valid_volumes = df_copy.loc[df_copy[volume_col].notna() & (df_copy[volume_col] > 0), volume_col]
            if len(valid_volumes) > 10:
                Q1_vol = valid_volumes.quantile(0.25); Q3_vol = valid_volumes.quantile(0.75)
                IQR_vol = Q3_vol - Q1_vol
                if IQR_vol > 1e-9: 
                    upper_bound_vol = Q3_vol + 1.5 * IQR_vol
                    outlier_mask_vol = (df_copy[volume_col].notna()) & (df_copy[volume_col] > upper_bound_vol) & (df_copy[volume_col] > 0)
                    if outlier_mask_vol.any():
                        num_outliers = outlier_mask_vol.sum()
                        logger.info(f"{num_outliers} outlier Volume (>0) untuk {asset_prefix}. Batas atas IQR: {upper_bound_vol:.0f}")
                        if volume_outlier_action == 'flag':
                            flag_col_name = f'{asset_prefix}_Volume_Outlier_Flag'
                            if flag_col_name not in df_copy.columns: df_copy[flag_col_name] = False
                            df_copy.loc[outlier_mask_vol, flag_col_name] = True
                        elif volume_outlier_action == 'drop':
                            logger.info(f"  Menghapus {num_outliers} baris karena outlier volume.")
                            df_copy = df_copy[~outlier_mask_vol]
                        elif volume_outlier_action == 'winsor':
                             logger.info(f"  Menerapkan Winsorizing pada {num_outliers} outlier volume (batas atas).")
                             df_copy.loc[outlier_mask_vol, volume_col] = upper_bound_vol 
                else: logger.info(f"IQR volume >0 {asset_prefix} nol/kecil. Lewati deteksi outlier volume.")
            else: logger.info(f"Data volume >0 ({len(valid_volumes)}) tidak cukup untuk deteksi outlier IQR {asset_prefix}.")
    else: logger.info(f"Kolom {volume_col} tidak ada/non-numerik, lewati deteksi outlier volume.")
    logger.info(f"--- Deteksi Outlier untuk {asset_prefix} Selesai ---")
    return df_copy

def restructure_and_finalize(df: pd.DataFrame, asset_prefixes: List[str]) -> pd.DataFrame:
    global logger
    if logger is None: raise RuntimeError("Logger belum diinisialisasi")
    logger.info(f"--- Memulai Penataan Ulang Kolom dan Finalisasi ---")
    df_copy = df.copy()
    all_final_ordered_columns: List[str] = []
    
    current_allowed_suffixes = ALLOWED_SUFFIXES.copy()
    if STANDARDIZED_PCT_CHANGE_SUFFIX not in current_allowed_suffixes:
        current_allowed_suffixes.append(STANDARDIZED_PCT_CHANGE_SUFFIX)
    
    suffix_order_map: Dict[str, float] = {}
    order_counter = 0
    base_order_suffixes = ["Open", "High", "Low", "Close", "Volume", STANDARDIZED_PCT_CHANGE_SUFFIX]
    for sfx in base_order_suffixes:
        if sfx in current_allowed_suffixes:
            suffix_order_map[sfx] = float(order_counter)
            order_counter += 1
    for sfx in current_allowed_suffixes:
        if sfx.endswith("_Flag") and sfx not in suffix_order_map:
            base_sfx_for_flag = sfx.replace("_Outlier_Flag", "").replace("_Inconsistent_Flag","")
            if base_sfx_for_flag in suffix_order_map:
                 suffix_order_map[sfx] = suffix_order_map[base_sfx_for_flag] + 0.5 
            else: 
                 suffix_order_map[sfx] = float(order_counter); order_counter +=1
    for sfx in current_allowed_suffixes:
        if sfx not in suffix_order_map:
            suffix_order_map[sfx] = float(order_counter); order_counter+=1

    present_prefixes: List[str] = sorted(list(asset_prefixes)) 
    for prefix in present_prefixes:
        if not any(col.startswith(prefix + "_") for col in df_copy.columns):
            logger.info(f"Tidak ada kolom untuk prefix {prefix}.")
            continue
        
        prefix_cols_in_df = [col for col in df_copy.columns if col.startswith(prefix + "_")]
        temp_cols_with_order: List[Tuple[str, float]] = []

        for col_name in prefix_cols_in_df:
            suffix_with_flag = col_name.replace(f"{prefix}_", "", 1)
            base_suffix = suffix_with_flag.replace("_Outlier_Flag", "").replace("_Inconsistent_Flag","")
            order = suffix_order_map.get(suffix_with_flag, suffix_order_map.get(base_suffix, 999.0))
            if suffix_with_flag.endswith("_Flag") and not isinstance(order, float): 
                order +=0.5
            temp_cols_with_order.append((col_name, order))
        
        temp_cols_with_order.sort(key=lambda x: x[1])
        asset_cols_ordered = [col_tuple[0] for col_tuple in temp_cols_with_order]
        
        for col_name in asset_cols_ordered:
            if col_name not in df_copy.columns: continue 
            actual_suffix = col_name.replace(f"{prefix}_", "", 1)
            if actual_suffix == "Volume":
                if prefix in KNOWN_INDEX_TICKERS:
                     df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce') 
                     if df_copy[col_name].notna().sum() == 0 or df_copy[col_name].fillna(0).eq(0).all():
                         logger.info(f"Kolom Volume untuk indeks {prefix} adalah semua 0/NaN, akan diubah menjadi semua NaN.")
                         df_copy[col_name] = np.nan 
                else: 
                     df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce').astype('Float64')
            elif actual_suffix == STANDARDIZED_PCT_CHANGE_SUFFIX:
                df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce') 
            elif any(actual_suffix == sfx for sfx in ["Open", "High", "Low", "Close"]): 
                df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce') 
            elif actual_suffix.endswith("_Flag"): 
                if col_name in df_copy.columns: 
                    df_copy[col_name] = df_copy[col_name].fillna(False).astype(bool)
        
        all_final_ordered_columns.extend(asset_cols_ordered)

    all_final_ordered_columns = list(dict.fromkeys(all_final_ordered_columns))
    
    existing_final_columns = [col for col in all_final_ordered_columns if col in df_copy.columns]
    cols_to_actually_drop = [col for col in df_copy.columns if col not in existing_final_columns]
    if cols_to_actually_drop:
        logger.warning(f"Menghapus kolom final yang tidak terdefinisi/tidak diinginkan: {cols_to_actually_drop}")
        df_copy = df_copy.drop(columns=cols_to_actually_drop, errors='ignore')
    
    if not existing_final_columns:
        logger.error("Tidak ada kolom valid untuk output akhir setelah filtering suffix."); return pd.DataFrame()
        
    df_copy = df_copy[existing_final_columns] 
    logger.info(f"Urutan kolom akhir: {df_copy.columns.tolist()}")
    
    final_nan_check = df_copy.isna().sum()
    total_final_nans = final_nan_check.sum()
    if total_final_nans > 0:
        logger.info(f"INFO FINAL: Total {total_final_nans} NaN di dataset akhir:")
        for col, count in final_nan_check.items():
            if count > 0: logger.info(f"    Kolom: {col}, Jumlah NaN: {count}")
    else: logger.info("INFO: Tidak ada NaN terdeteksi di dataset akhir.")

    if not isinstance(df_copy.index, pd.DatetimeIndex):
        logger.warning("Indeks bukan DatetimeIndex. Konversi ulang.")
        try: df_copy.index = pd.to_datetime(df_copy.index); df_copy = df_copy.sort_index()
        except Exception as e: logger.error(f"Gagal konversi indeks ke DatetimeIndex: {e}")
    logger.info(f"--- Penataan Ulang Kolom dan Finalisasi Selesai ---")
    return df_copy

def build_features(
    ohlc_action: AnomalyAction = 'drop', 
    price_outlier_action: AnomalyAction = 'none', 
    volume_outlier_action: AnomalyAction = 'none'
) -> None:
    global logger
    log_file_path = os.path.join(OUTPUT_DATAPROC_DIR, LOG_FILENAME)
    logger = setup_logging(log_file_path) 

    input_path = os.path.join(INPUT_PROCESSED_DIR, INPUT_FILENAME)
    output_path = os.path.join(OUTPUT_DATAPROC_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DATAPROC_DIR, exist_ok=True)
    
    logger.info(f"Memulai Feature Builder: Validasi dan Pembersihan Data Lanjutan")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Aksi Inkonsistensi OHLC: {ohlc_action}")
    logger.info(f"Aksi Outlier Harga (PctChange): {price_outlier_action}")
    logger.info(f"Aksi Outlier Volume: {volume_outlier_action}")

    if not os.path.exists(input_path):
        logger.error(f"File input {input_path} tidak ditemukan. Hentikan proses."); return

    try:
        df_unified: pd.DataFrame = pd.read_csv(input_path, index_col='Date', parse_dates=True)
        logger.info(f"Berhasil memuat data. Shape awal: {df_unified.shape}")
    except Exception as e:
        logger.error(f"Gagal memuat data dari {input_path}: {e}"); return

    if df_unified.duplicated().any():
        num_dupes = df_unified.duplicated().sum()
        logger.info(f"Menemukan {num_dupes} baris duplikat penuh. Menghapus...")
        df_unified.drop_duplicates(keep='first', inplace=True)
        logger.info(f"Shape setelah menghapus duplikat penuh: {df_unified.shape}")

    asset_prefixes: List[str] = sorted(list(set([col.split('_')[0] for col in df_unified.columns if '_' in col])))
    if not asset_prefixes:
        logger.error("Tidak ada prefix aset teridentifikasi."); return
    logger.info(f"Aset yang terdeteksi: {asset_prefixes}")

    processed_asset_dfs: Dict[str, pd.DataFrame] = {}
    for prefix in asset_prefixes:
        asset_cols: List[str] = [col for col in df_unified.columns if col.startswith(prefix + "_")]
        if not asset_cols: 
            logger.warning(f"Tidak ada kolom ditemukan untuk prefix {prefix} di df_unified. Melewati.")
            continue
        
        df_asset_subset: pd.DataFrame = df_unified[asset_cols].copy()
        
        df_asset_subset = standardize_pct_change_column_name(df_asset_subset, prefix)
        df_asset_subset = validate_ohlc_consistency(df_asset_subset, prefix, action=ohlc_action)
        if df_asset_subset.empty: logger.warning(f"{prefix} kosong setelah validasi OHLC."); continue 
        df_asset_subset = validate_numeric_values(df_asset_subset, prefix)
        if df_asset_subset.empty: logger.warning(f"{prefix} kosong setelah validasi numerik."); continue
        df_asset_subset = handle_missing_values_asset(df_asset_subset, prefix)
        if df_asset_subset.empty: logger.warning(f"{prefix} kosong setelah penanganan missing values."); continue
        df_asset_subset = handle_outliers_asset(df_asset_subset, prefix, 
                                                price_outlier_action=price_outlier_action, 
                                                volume_outlier_action=volume_outlier_action)
        if df_asset_subset.empty: logger.warning(f"{prefix} kosong setelah penanganan outlier."); continue
        
        processed_asset_dfs[prefix] = df_asset_subset
        logger.info(f"Selesai validasi & pembersihan {prefix}. Shape: {df_asset_subset.shape}")

    if not processed_asset_dfs:
        logger.error("Tidak ada data aset berhasil diproses setelah semua validasi."); return
        
    final_df_list = [df for df_key, df in processed_asset_dfs.items() if not df.empty] 
    if not final_df_list:
        logger.error("Semua aset menghasilkan DataFrame kosong setelah pembersihan individual."); return

    df_cleaned: pd.DataFrame = pd.concat(final_df_list, axis=1, join='outer')
    df_cleaned = df_cleaned.sort_index() 
    logger.info(f"Penggabungan akhir selesai. Shape sebelum restructure: {df_cleaned.shape}")
    
    df_cleaned = restructure_and_finalize(df_cleaned, asset_prefixes) 

    logger.info(f"Total baris di DataFrame final: {len(df_cleaned)}")
    if not df_cleaned.empty:
        logger.info(f"Statistik deskriptif (head):\n{df_cleaned.describe(include='all').head().to_string()}")
    else: logger.info("DataFrame final kosong.")

    # Pemeriksaan akhir High >= Low
    if not df_cleaned.empty:
        for prefix in asset_prefixes:
            high_col_final = f"{prefix}_High"
            low_col_final = f"{prefix}_Low"
            if high_col_final in df_cleaned.columns and low_col_final in df_cleaned.columns:
                # Hanya periksa baris di mana keduanya tidak NaN
                valid_comparison_mask = df_cleaned[high_col_final].notna() & df_cleaned[low_col_final].notna()
                if not df_cleaned.loc[valid_comparison_mask, high_col_final].ge(df_cleaned.loc[valid_comparison_mask, low_col_final]).all():
                    logger.error(f"ERROR FINAL CHECK: Inkonsistensi High >= Low ditemukan untuk {prefix} di DataFrame akhir!")
                    # Anda bisa menambahkan detail lebih lanjut di sini jika perlu

    try:
        if not df_cleaned.empty:
            df_cleaned.to_csv(output_path, index=True, date_format='%Y-%m-%d', na_rep='NaN') 
            logger.info(f"Dataset bersih disimpan ke: {output_path}")
        else: logger.warning(f"DataFrame akhir kosong, tidak ada output disimpan.")
    except Exception as e: logger.error(f"Gagal menyimpan output ke {output_path}: {e}")
    logger.info(f"Proses Feature Builder Selesai.")

if __name__ == '__main__':
    build_features(ohlc_action='drop', price_outlier_action='none', volume_outlier_action='none')

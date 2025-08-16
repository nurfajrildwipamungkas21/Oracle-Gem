import os
import sys
import sqlite3
import uuid
import time
import logging
import requests
import random
import json
import re
import io
import zipfile
import threading
import graphviz  # Diperlukan untuk visualisasi DKG
import base64
from contextlib import closing, contextmanager
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional, Any, Literal
from collections import Counter
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.explain import GNNExplainer
from torch_geometric.data import Data as PyG_Data
import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel, Field
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from openai import OpenAI, APIError
from tavily import TavilyClient
from json import JSONDecodeError
from pydantic import BaseModel, ValidationError
import re
import json
from together import Together

logger = logging.getLogger("Oracle Gem")

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

class DynamicKnowledgeGraph:
    """
    Mengelola representasi graf multi-lapis dari entitas dan hubungan dinamis antar mereka.
    """

    def __init__(self, project_id: str, tickers: list = None):
        self.project_id = project_id
        self.nodes = {}  # {node_id: {attr_dict}}
        self.edges = []  # [(source_id, target_id, {attr_dict})]

        if tickers:
            for ticker in tickers:
                self.add_node(node_id=ticker, node_type="Ticker",
                              layer="Market", name=ticker)

    def add_node(self, node_id: str, node_type: str, layer: str = "default", **kwargs):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type,
                                   "layer": layer, "attributes": kwargs}
            logger.debug(
                f"[DKG] Node added: {node_id} (Layer: {layer}, Type: {node_type})")

    def add_edge(self, source: str, target: str, relationship: str, **kwargs):
        if source in self.nodes and target in self.nodes:
            edge_data = {"relationship": relationship,
                         "timestamp": datetime.now().isoformat()}
            edge_data.update(kwargs)
            self.edges.append((source, target, edge_data))
            logger.debug(
                f"[DKG] Edge added: {source} -[{relationship}]-> {target}")

    # Tipe 'DigestedActivity' akan dikenali di alpha.py
    def add_digested_activity(self, activity):
        """Menambahkan pengetahuan yang telah dicerna dari aktivitas pengguna ke DKG."""
        user_node_id = "USER"
        self.add_node(user_node_id, node_type="User", layer="Agent")

        for entity in activity.entities:
            entity_id = entity.replace(" ", "_")
            self.add_node(node_id=entity_id, node_type="Concept",
                          layer="Knowledge", name=entity)
            self.add_edge(user_node_id, entity_id,
                          activity.activity.replace(" ", "_").lower())

        topic_id = activity.general_topic.replace(" ", "_")
        self.add_node(topic_id, node_type="Topic",
                      layer="Knowledge", name=activity.general_topic)

        for entity in activity.entities:
            entity_id = entity.replace(" ", "_")
            self.add_edge(entity_id, topic_id, "is_related_to")

        logger.info(
            f"🧠 [DKG Neurogenesis] Pengetahuan baru tentang '{activity.general_topic}' ditambahkan.")


class StatusLogger:
    """Menampilkan animasi status di terminal untuk tugas yang berjalan lama."""

    def __init__(self, message="Model sedang berpikir...", **kwargs):
        self.message = message
        self.emoji = kwargs.get("emoji", "🧠🌀")
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._stop_event = threading.Event()

    def update_message(self, new_message: str, **kwargs):
        self.message = new_message
        if 'emoji' in kwargs:
            self.emoji = kwargs['emoji']

    def _animate(self):
        chars = ["|", "/", "-", "\\"]
        i = 0
        while not self._stop_event.is_set():
            emoji_to_show = getattr(self, 'emoji', '🌀')
            sys.stdout.write('\r' + ' ' * 150 + '\r')
            sys.stdout.write(
                f"\r{emoji_to_show} {self.message} {chars[i % len(chars)]}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def start(self):
        self._thread.start()

    def stop(self, final_message="Selesai.", **kwargs):
        final_emoji = kwargs.get("final_emoji", "✅")
        if self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        sys.stdout.write('\r' + ' ' * 150 + '\r')
        sys.stdout.flush()
        if final_message:
            logger.info(f"{final_emoji} {final_message}")


def exponential_backoff(
    max_retries: int = 5, base_delay: float = 1.0, factor: float = 2.0,
    retryable_status_codes: tuple = (429, 500, 502, 503, 504),
    retryable_exceptions: tuple = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError
    )
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (*retryable_exceptions, requests.exceptions.RequestException, APIError) as e:
                    last_exception = e
                    is_retryable = False
                    status_code = -1

                    if isinstance(e, retryable_exceptions):
                        is_retryable = True
                    elif isinstance(e, APIError) and hasattr(e, 'status_code'):
                        status_code = e.status_code
                        if status_code in retryable_status_codes:
                            is_retryable = True
                    elif hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if status_code in retryable_status_codes:
                            is_retryable = True

                    if is_retryable and retries < max_retries - 1:
                        retries += 1
                        delay = (base_delay * (factor ** retries)) + \
                            random.uniform(0, 1)
                        logger.warning(
                            f"API call to '{func.__name__}' failed with status {status_code}. Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"API call '{func.__name__}' failed after {retries + 1} attempts.")
                        raise e from last_exception
            raise RuntimeError(
                f"Function {func.__name__} failed after all retries.")
        return wrapper
    return decorator


class WebSearchManager:
    def __init__(self, tavily_api_key: str = None, google_api_key: str = None, google_cse_id: str = None):
        self.tavily_key = tavily_api_key
        self.google_key = google_api_key
        self.google_cse_id = google_cse_id
        self._initialize_clients()
        self.tavily_client = TavilyClient(
            api_key=self.tavily_key) if self.tavily_key else None
        self.GOOGLE_DAILY_LIMIT = 100
        self.request_count_file = Path.home() / "Oracle Gem" / "models_trained" / \
            "Google Search_count.json"
        self.request_count_file.parent.mkdir(parents=True, exist_ok=True)
        # Inisialisasi http_session yang hilang
        self.http_session = requests.Session()


    def _initialize_clients(self):
        """Metode helper untuk inisialisasi atau re-inisialisasi client."""
        self.tavily_client = TavilyClient(
            api_key=self.tavily_key) if self.tavily_key else None
        # Tambahkan inisialisasi client Google jika ada di sini

    def _get_google_requests_today(self) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        if not self.request_count_file.exists():
            return 0
        try:
            with open(self.request_count_file, 'r') as f:
                data = json.load(f)
            if data.get("date") == today:
                return data.get("count", 0)
            return 0
        except (json.JSONDecodeError, IOError):
            return 0

    def _increment_google_requests_today(self):
        count = self._get_google_requests_today()
        today = datetime.now().strftime("%Y-%m-%d")
        with open(self.request_count_file, 'w') as f:
            json.dump({"date": today, "count": count + 1}, f)

    def _download_image(self, image_url: str, query: str) -> Optional[Path]:
        import re
        try:
            safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', query)[:50] + ".jpg"
            download_dir = Path("downloaded_images")
            download_dir.mkdir(exist_ok=True)
            image_path = download_dir / safe_filename
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return image_path
        except Exception as e:
            logger.warning(f"Gagal mengunduh gambar dari {image_url}: {e}")
            return None

    def search(self, query: str, max_results: int = 5, include_images: bool = True) -> str:
        import re

        # --- BLOK TAVILY DENGAN LOGIKA PENYEMBUHAN DIRI ---
        if self.tavily_client:
            status_logger = StatusLogger(
                message=f"Mencari via Tavily: '{query[:50]}...'", emoji="🌐")
            status_logger.start()
            try:
                response = self.tavily_client.search(
                    query=query, search_depth="advanced", max_results=max_results, include_images=include_images
                )

                context_parts = []
                results = response.get('results', [])
                if results:
                    status_logger.stop(final_message="", final_emoji="")
                    for i, res in enumerate(results):
                        domain = "sumber tidak diketahui"
                        if res.get('url'):
                            match = re.search(
                                r'https?://([^/]+)', res.get('url'))
                            if match:
                                domain = match.group(1)

                        print(
                            f"\n📄 Membaca ({i+1}/{len(results)}) dari: {domain}")
                        print("-" * 70)
                        print(
                            f"{res.get('content', 'Konten tidak tersedia.')[:350]}...")
                        print("-" * 70 + "\n")
                        context_parts.append(
                            f"Source: {res.get('url', 'N/A')}\nContent: {res.get('content', '')}")
                        time.sleep(0.5)
                    status_logger = StatusLogger(
                        message="Melanjutkan riset...", emoji="🔄")
                    status_logger.start()

                if include_images:
                    images = response.get('images', [])
                    if images:
                        status_logger.update_message(
                            "Mengunduh gambar relevan...")
                        for i, image_url in enumerate(images):
                            saved_path = self._download_image(image_url, query)
                            if saved_path:
                                logger.info(
                                    f"🖼️  Gambar relevan disimpan di: {saved_path.resolve()}")

                status_logger.stop("Riset web via Tavily selesai.")
                return "\n\n".join(context_parts)
            except Exception as e:
                status_logger.stop(f"Tavily gagal: {e}", final_emoji="❌")
                # Cek jika errornya adalah karena otorisasi
                if "Unauthorized" in str(e) or "401" in str(e) or "missing or invalid API key" in str(e).lower():
                    # Panggil fungsi interaktif untuk update key
                    if update_specific_key_interactively("tavily"):
                        # Jika user memasukkan key baru, muat ulang konfigurasi dan coba lagi
                        from src.models.model_alpha.key_manager import manage_api_keys
                        updated_keys = manage_api_keys()
                        self.tavily_key = updated_keys.get("tavily")
                        self._initialize_clients()  # Inisialisasi ulang client dengan kunci baru
                        # Coba lagi dari awal
                        return self.search(query, max_results, include_images)
        logger.error("  > [WebSearch] Semua metode pencarian gagal.")
        return ""

    def fetch_and_process_file_url(self, url: str) -> dict:
        """
        Mengunduh file dari URL ke memori, mengekstrak kontennya, dan mengembalikan sebagai teks.
        Mendukung PDF dan ZIP.
        """
        logger.info(f"📥 Mencoba mengunduh dan memproses file dari: {url}")
        try:
            response = self.http_session.get(url, timeout=60)
            response.raise_for_status()  # Lemparkan error untuk status 4xx/5xx

            # Simpan konten ke file virtual di RAM
            content_stream = io.BytesIO(response.content)
            extracted_text = ""
            file_type = "unknown"

            # --- Pemrosesan PDF ---
            if url.lower().endswith('.pdf'):
                file_type = "PDF"
                with fitz.open(stream=content_stream, filetype="pdf") as doc:
                    for page in doc:
                        extracted_text += page.get_text() + "\n--- End of Page ---\n"
                logger.info(
                    f"📄 Berhasil mengekstrak {len(extracted_text)} karakter dari PDF.")

            # --- Pemrosesan ZIP ---
            elif url.lower().endswith('.zip'):
                file_type = "ZIP"
                with zipfile.ZipFile(content_stream) as zf:
                    file_list = zf.namelist()
                    extracted_text += f"Arsip ZIP berisi {len(file_list)} file:\n- " + "\n- ".join(
                        file_list) + "\n\n"
                    # Coba baca konten dari file teks di dalam ZIP
                    for filename in file_list:
                        if filename.lower().endswith(('.txt', '.csv', '.md')):
                            try:
                                with zf.open(filename) as f:
                                    content = f.read().decode('utf-8', errors='ignore')
                                    extracted_text += f"\n--- Content of {filename} ---\n{content}\n"
                            except Exception as e:
                                extracted_text += f"\n--- Gagal membaca {filename}: {e} ---\n"
                logger.info(f"🗂️ Berhasil memproses arsip ZIP.")

            else:
                return {"status": "error", "message": "Tipe file tidak didukung (hanya .pdf dan .zip)."}

            return {
                "status": "success",
                "file_type": file_type,
                "source_url": url,
                # Batasi ukuran konteks untuk LLM
                "content": extracted_text[:20000]
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Gagal mengunduh file dari {url}: {e}")
            return {"status": "error", "message": f"Gagal mengunduh: {e}"}
        except Exception as e:
            logger.error(f"Gagal memproses file dari {url}: {e}")
            return {"status": "error", "message": f"Gagal memproses file: {e}"}


class TogetherLLM:
    def __init__(self, api_key: str, model_name: str, max_tokens: int = 4096):
        if not api_key or not api_key.strip():
            raise ValueError("API key untuk Together.AI tidak boleh kosong.")
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens

        # URL default untuk fallback (tetap dipertahankan dari kode pertama)
        self.api_url = "https://api.together.xyz/v1/chat/completions"

        # Inisialisasi client resmi Together AI
        try:
            self.client = Together(api_key=self.api_key)
        except Exception as e:
            # Jika gagal inisialisasi client, log error tapi jangan hentikan program
            logger.error(f"Gagal menginisialisasi Together client: {e}")
            self.client = None

    @exponential_backoff()
    def chat(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        """
        Menggunakan client Together resmi jika tersedia, jika gagal maka fallback ke metode requests.
        """
        # logger.info(f"Mengirim request ke Together.AI untuk model: {self.model_name}")
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=0.7,  # Lebih stabil
                    top_p=0.7,
                    top_k=50,
                    repetition_penalty=1.1
                )
                content = response.choices[0].message.content
                if not content:
                    logger.warning(f"Menerima respons kosong dari Together.AI (model: {self.model_name}).")
                return content.strip()
            except Exception as e:
                logger.error(f"Error menggunakan client Together: {e}. Menggunakan fallback requests...")

        # === Fallback ke metode lama (requests) ===
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()

        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Struktur respons tidak terduga dari Together.AI: {response_json}")
            raise ValueError(f"Struktur respons tidak valid dari {self.model_name}") from e

        if not content:
            logger.warning(f"Menerima respons kosong dari Together.AI (model: {self.model_name}).")
        return content.strip()

    @exponential_backoff()
    def describe_image(self, image_path: str, prompt: str) -> str:
        """
        Mengirim gambar dan prompt ke model vision dan mengembalikan deskripsi.
        Metode ini tetap menggunakan endpoint lama karena API Vision Together mungkin berbeda.
        """
        if "vision" not in self.model_name.lower():
            raise ValueError(f"Model {self.model_name} bukan model vision.")

        try:
            with Image.open(image_path) as img:
                img.thumbnail((1024, 1024))
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Gagal memproses gambar {image_path}: {e}")
            return f"Error processing image: {e}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "content": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }],
            "max_tokens": 2048
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        response_json = response.json()

        try:
            return response_json["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            logger.error(f"Struktur respons vision tidak terduga: {response_json}")
            return "Failed to parse vision model response."

    def invoke(self, prompt: str, tools: list = None) -> SimpleNamespace:
        """
        Metode invoke agar kompatibel dengan cara pemanggilan LangChain.
        """
        content_response = self.chat(prompt)
        return SimpleNamespace(content=content_response, tool_calls=[])


class GrokLLM:
    def __init__(self, api_key: str, model_name: str = "grok-3.1-mini-20240722-int8"):
        if not api_key or not api_key.strip():
            raise ValueError("API key untuk Grok (xAI) tidak boleh kosong.")

        # Inisialisasi client menggunakan pustaka OpenAI dengan base_url milik Grok
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",  # Endpoint resmi Grok API
        )
        self.model_name = model_name
        logger.info(
            f"✅ Grok LLM (via OpenAI compatibility) siap digunakan untuk model: {self.model_name}")

    @exponential_backoff()
    def chat(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        # Membuat panggilan chat sesuai dokumentasi 'grok'
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9
        )

        response_content = completion.choices[0].message.content

        if not response_content:
            logger.warning(
                f"Menerima respons kosong dari Grok (model: {self.model_name}).")

        return response_content.strip()


class QwenLLM:
    def __init__(self, api_key: str, model_name: str = "qwen-plus-2025-04-28"):
        if not api_key:
            raise ValueError(
                "API Key untuk Qwen (DashScope) tidak boleh kosong.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model_name

    @exponential_backoff()
    def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        response_content = completion.choices[0].message.content
        if not response_content:
            logger.warning(
                f"Menerima respons kosong dari Qwen (model: {self.model}).")
        return response_content.strip()


class APIEmbedder:
    """
    Encoder universal yang dapat menggunakan model embedding dari Together.AI API
    atau model SentenceTransformer lokal, tergantung pada inisialisasi.
    """
    def __init__(self, model_name: str, dim: int, api_key: str = None, use_local_model: bool = False):
        self.model_name = model_name
        self.dim = dim
        self.use_local_model = use_local_model
        self.local_model = None
        self.api_key = api_key
        self.api_url = "https://api.together.xyz/v1/embeddings"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        if self.use_local_model:
            logger.info(f"🧠 Memuat model embedding LOKAL: {self.model_name}...")
            # Menggunakan SentenceTransformer untuk model lokal
            self.local_model = SentenceTransformer(self.model_name)
            logger.info("✅ Model embedding lokal siap.")
        else:
            if not self.api_key:
                raise ValueError("API Key untuk Together.AI wajib ada jika tidak menggunakan model lokal.")
            logger.info(f"🌐 Menggunakan model embedding via API: {self.model_name}")

    @exponential_backoff()
    def _call_api(self, input_texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model_name, "input": input_texts}
        response = requests.post(
            self.api_url, headers=self.headers, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item['embedding'] for item in data]

    def encode(self, sentences: list[str] | str, task_type: str = "passage", convert_to_numpy: bool = True, show_progress_bar: bool = False) -> Any:
        if not sentences:
            return np.array([]) if convert_to_numpy else []

        is_single_sentence = isinstance(sentences, str)
        if is_single_sentence:
            sentences = [sentences]
            
        valid_sentences = [s for s in sentences if s and s.strip()]
        if not valid_sentences:
            return np.array([]) if convert_to_numpy else []

        # --- Logika Cabang: Lokal vs API ---
        if self.use_local_model and self.local_model:
            # Gunakan model SentenceTransformer lokal
            all_embeddings = self.local_model.encode(
                valid_sentences, 
                convert_to_numpy=convert_to_numpy, 
                show_progress_bar=show_progress_bar
            )
        else:
            # Gunakan API Together.AI
            prefixed_sentences = [
                f"query: {s}" if task_type == "query" else f"passage: {s}" for s in valid_sentences] if "e5" in self.model_name else valid_sentences

            MAX_INPUTS_PER_CALL = 256
            api_embeddings_list = []
            for i in range(0, len(prefixed_sentences), MAX_INPUTS_PER_CALL):
                batch = prefixed_sentences[i:i + MAX_INPUTS_PER_CALL]
                api_embeddings_list.extend(self._call_api(batch))
            
            if not api_embeddings_list:
                return np.array([]) if convert_to_numpy else []

            all_embeddings = np.array(api_embeddings_list) if convert_to_numpy else api_embeddings_list

        if is_single_sentence and len(all_embeddings) > 0:
            return all_embeddings[0]
        elif is_single_sentence:
            return np.array([]) if convert_to_numpy else []
        
        return all_embeddings



class CognitiveLabel(BaseModel):
    label: Literal['Fakta', 'Logika', 'Asumsi', 'Opini', 'Kode', 'Tidak Relevan'] = Field(
        description="Label kognitif untuk potongan teks.")
    reasoning: str = Field(description="Alasan singkat untuk pelabelan.")


def _get_cognitive_label(chunk: str, agent: "TogetherLLM") -> CognitiveLabel:
    """Meminta agen AI untuk memberikan label kognitif pada sebuah potongan teks."""
    prompt = f"""
    Analisis potongan teks berikut dan berikan label kognitif yang paling sesuai dari pilihan: ['Fakta', 'Logika', 'Asumsi', 'Opini', 'Kode', 'Tidak Relevan'].
    Jawab HANYA dengan format JSON sesuai skema.

    Teks untuk dianalisis:
    ---
    {chunk}
    ---
    """
    try:
        # Asumsi agent.chat dapat menangani response_model untuk parsing Pydantic
        # Jika tidak, perlu parsing manual dari JSON string
        response_str = agent.chat(prompt)
        # Menggunakan robust_json_extract untuk keamanan
        parsed_response = robust_json_extract(response_str, CognitiveLabel)
        if parsed_response:
             return parsed_response
        return CognitiveLabel(label='Tidak Relevan', reasoning='Gagal mem-parsing respons LLM')
    except Exception:
        return CognitiveLabel(label='Tidak Relevan', reasoning='Gagal dianalisis')


class MultiPathElectraClassifier:
    """
    Bertindak sebagai 'Guru Internal' dengan beberapa jalur pemikiran (instance Electra)
    untuk mendapatkan keputusan yang lebih robust.
    """

    def __init__(self, num_paths: int = 5):
        self.num_paths = num_paths
        self.model_name = "Aardiiiiy/EmoSense-ID-Indonesian-Emotion-Classifier"
        logger.info(
            f"📚 Menginisialisasi Guru Internal dengan {self.num_paths} jalur dari model: {self.model_name}")
        self.pipelines = []
        self.active_paths = 0

    def activate_paths(self, num_to_activate: int):
        """Mengaktifkan (memuat) sejumlah jalur pipeline sesuai perintah Resource Manager."""
        num_to_activate = min(self.num_paths, num_to_activate)
        if len(self.pipelines) < num_to_activate:
            for _ in range(num_to_activate - len(self.pipelines)):
                self.pipelines.append(
                    pipeline("text-classification", model=self.model_name))
        self.active_paths = num_to_activate
        logger.info(
            f"🔥 Guru Internal sekarang aktif dengan {self.active_paths} jalur.")

    def deactivate_paths(self, num_to_deactivate: int):
        """Mematikan sejumlah jalur untuk menghemat memori."""
        self.active_paths = max(0, self.active_paths - num_to_deactivate)
        logger.info(
            f"❄️ Guru Internal sekarang berjalan dengan {self.active_paths} jalur.")

    def classify(self, text: str) -> (str, float):
        """
        Mengklasifikasikan teks menggunakan jalur yang aktif dan mengembalikan hasil
        beserta skor keyakinan berdasarkan konsensus.
        """
        if self.active_paths == 0:
            return "unknown", 0.0

        predictions = []
        for i in range(self.active_paths):
            result = self.pipelines[i](text)
            predictions.append(result[0]['label'])

        most_common = Counter(predictions).most_common(1)[0]
        label = most_common[0]
        confidence = most_common[1] / self.active_paths

        return label, confidence


class Brain:
    """
    Otak Kognitif Sentral v3.0 (Pembelajar Mandiri).
    Mengelola memori dengan hierarki pembelajaran (Guru Internal & Dosen Eksternal)
    dan logika eskalasi cerdas untuk efisiensi API.
    """

    def __init__(self, index_path: str, db_path: str, embed_model_instance: "APIEmbedder",
                 dim: int, api_pool: "DistributedAIPool", together_api_keys: dict):

        self.logging_lock = threading.local()
        self.index_path = str(index_path)
        self.db_path = str(db_path)
        self.embed_model = embed_model_instance
        self.dim = dim
        self.api_pool = api_pool
        self.together_api_keys = together_api_keys

        logger.info(
            "📚 Menginisialisasi Guru Internal dengan model Zero-Shot...")
        self.internal_teacher = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

        self._lock = threading.Lock()
        self.llm_call_count_session = 0
        self.MAX_LLM_CALLS = 10

        recreate_index = False
        if Path(self.index_path).exists():
            try:
                index_loaded = faiss.read_index(self.index_path)
                if index_loaded.d != self.dim:
                    logger.warning(
                        f"Dimensi FAISS ({index_loaded.d}) tidak cocok dengan model embedding ({self.dim}). Membuat ulang indeks.")
                    recreate_index = True
                else:
                    self.index = index_loaded
                    logger.info(
                        f"Memuat indeks FAISS yang ada dari: {self.index_path}")
            except Exception as e:
                logger.warning(
                    f"Gagal memuat indeks FAISS ({e}). Membuat ulang indeks.")
                recreate_index = True
        else:
            recreate_index = True

        if recreate_index:
            logger.info("Membuat indeks FAISS baru.")
            index_flat = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIDMap(index_flat)
            if Path(self.db_path).exists():
                with closing(sqlite3.connect(self.db_path)) as conn:
                    conn.execute("DROP TABLE IF EXISTS knowledge")
                    conn.commit()

        self._init_db()
        self.dkg = DynamicKnowledgeGraph(project_id="global_main")
        logger.info(
            "🔗 Dynamic Knowledge Graph (DKG) telah terintegrasi dengan Brain.")

    def _init_db(self):
        """Inisialisasi database dengan skema yang sudah di-upgrade."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT UNIQUE,
                    source TEXT,
                    cognitive_label TEXT,
                    cognitive_reasoning TEXT
                )
            """)
            conn.commit()

    def add_chunks(self, chunks: list[str], source_name: str):
        """
        Memproses dan menyimpan potongan teks (chunks) ke dalam basis pengetahuan.
        
        Fitur utama:
        - Mencegah duplikasi data.
        - Pelabelan kognitif 2-tahap (Internal Teacher -> Eskalasi ke External Lecturer).
        - Menyimpan data ke database SQLite dan indeks pencarian FAISS.
        - Thread-safe dengan lock dan penanganan error yang tangguh.
        """
        with self._lock:
            try:
                # 1. Validasi Input Awal
                if not chunks or not any(c.strip() for c in chunks):
                    logger.warning(
                        f"Melewatkan add_chunks untuk '{source_name}' karena tidak ada konten teks yang valid.")
                    return

                new_ids = []
                chunks_to_embed = []
                candidate_labels = ['Fakta', 'Logika', 'Asumsi', 'Opini', 'Kode', 'Tidak Relevan']

                # 2. Pemrosesan Chunks dalam satu koneksi database
                with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
                    c = conn.cursor()
                    for chunk in chunks:
                        # Cek duplikasi chunk
                        c.execute("SELECT id FROM knowledge WHERE text = ?", (chunk,))
                        if c.fetchone():
                            continue

                        # --- BLOK LOGIKA GABUNGAN DIMULAI DI SINI ---
                        
                        # TAHAP 1: GURU INTERNAL (CEPAT & MURAH)
                        prediction = self.internal_teacher(chunk, candidate_labels=candidate_labels)
                        internal_label = prediction['labels'][0]
                        confidence = prediction['scores'][0]

                        cognitive_info = None
                        
                        # TAHAP 2: ESKALASI JIKA GURU RAGU (THRESHOLD < 75%)
                        if confidence < 0.75 and self.llm_call_count_session < self.MAX_LLM_CALLS:
                            logger.warning(f"  -> Guru internal ragu ({confidence:.0%}). Eskalasi ke Dosen Eksternal...")
                            dosen_key = self.together_api_keys.get("exaone")
                            if dosen_key:
                                try:
                                    dosen_agent = TogetherLLM(api_key=dosen_key, model_name="lgai/exaone-deep-32b")
                                    # Fungsi _get_cognitive_label diasumsikan ada
                                    cognitive_info = _get_cognitive_label(chunk, dosen_agent) 
                                    self.llm_call_count_session += 1
                                except Exception as e:
                                    logger.error(f"Gagal saat eskalasi ke Dosen Eksternal: {e}")
                        
                        # Fallback: Jika tidak ada eskalasi atau eskalasi gagal, gunakan hasil guru internal
                        if not cognitive_info:
                            reasoning = f"Labelled by Internal Teacher (Zero-Shot) with {confidence:.0%} confidence."
                            cognitive_info = CognitiveLabel(label=internal_label, reasoning=reasoning)

                        # --- BLOK LOGIKA GABUNGAN SELESAI ---

                        # Masukkan data yang sudah dilabeli ke database
                        try:
                            c.execute(
                                "INSERT INTO knowledge (text, source, cognitive_label, cognitive_reasoning) VALUES (?, ?, ?, ?)",
                                (chunk, source_name, cognitive_info.label, cognitive_info.reasoning)
                            )
                            new_id = c.lastrowid
                            new_ids.append(new_id)
                            chunks_to_embed.append(chunk)
                        except sqlite3.IntegrityError:
                            # Kasus langka jika ada race condition, lewati saja
                            continue
                    conn.commit()

                # 3. Embedding dan Indexing (jika ada data baru)
                if new_ids:
                    new_embeddings = self.embed_model.encode(
                        chunks_to_embed, task_type="passage", convert_to_numpy=True)
                    
                    if new_embeddings.shape[0] == len(new_ids):
                        self.index.add_with_ids(new_embeddings.astype('float32'), np.array(new_ids, dtype='int64'))
                        self.save_index() # Simpan perubahan pada indeks
                        logger.info(
                            f"🧠 [Brain] {len(new_ids)} potongan info BARU dari '{source_name}' telah diproses dan disimpan.")
                    else:
                        logger.error(
                            f"FATAL: Terjadi ketidakcocokan jumlah embedding dan ID. Database mungkin tidak sinkron dengan indeks.")
            
            except Exception as e:
                logger.error(
                    f"Error kritis di dalam brain.add_chunks: {e}", exc_info=True)

    def query(self, q: str, k: int = 5) -> list[str]:
        if self.index.ntotal == 0:
            return []
        vec = self.embed_model.encode(q, task_type="query").reshape(1, -1)
        if vec.size == 0:
            return []
        with self._lock:
            distances, ids = self.index.search(vec.astype('float32'), k)
        valid_ids = [int(i) for i in ids[0] if i != -1]
        if not valid_ids:
            return []
        with closing(sqlite3.connect(self.db_path)) as conn:
            placeholders = ','.join('?' for _ in valid_ids)
            query_str = f"SELECT id, text FROM knowledge WHERE id IN ({placeholders})"
            results = conn.execute(query_str, valid_ids).fetchall()
            ordered_results = {row[0]: row[1] for row in results}
            return [ordered_results[id_] for id_ in valid_ids if id_ in ordered_results]

    def save_index(self):
        with self._lock:
            faiss.write_index(self.index, self.index_path)

    def get_chunk_by_id(self, chunk_id: int) -> Optional[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge WHERE id = ?", (chunk_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_chunk(self, chunk_id: int, new_chunk_text: str, new_source: str):
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    "UPDATE knowledge SET text = ?, source = ? WHERE id = ?",
                    (new_chunk_text, new_source, chunk_id)
                )
                conn.commit()

            self.index.remove_ids(np.array([chunk_id], dtype='int64'))
            new_embedding = self.embed_model.encode(
                new_chunk_text, task_type="passage").astype('float32').reshape(1, -1)
            # Terdapat potensi bug di sini: `new_embedding` adalah array 2D, jadi harusnya tidak di-wrap lagi
            self.index.add_with_ids(new_embedding, np.array([chunk_id], dtype='int64'))

            faiss.write_index(self.index, self.index_path)
        logger.info(
            f"📚 [Brain] Pengetahuan dengan ID {chunk_id} telah diperbarui.")

    def delete_chunk(self, chunk_id: int):
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute("DELETE FROM knowledge WHERE id = ?", (chunk_id,))
                conn.commit()

            self.index.remove_ids(np.array([chunk_id], dtype='int64'))
            faiss.write_index(self.index, self.index_path)
        logger.info(
            f"📚 [Brain] Pengetahuan dengan ID {chunk_id} telah dihapus.")


class ThalamicNucleusVAE(nn.Module):
    """
    Satu Inti Thalamus individual. Menggunakan Variational Autoencoder (VAE)
    untuk mengkristalisasi data input menjadi representasi probabilitas
    di ruang laten (latent space) yang kontinu.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Menghasilkan parameter distribusi (mu dan logvar) dari input."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Melakukan reparameterization trick untuk sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# === KODE DARI BLOK KEDUA DIMASUKKAN DI SINI ===
class QuantumThalamicCore(nn.Module):
    """
    Inti Thalamus Kuantum (QTC). Mengelola jaringan Inti Thalamus (Thalamic Nuclei)
    yang tumbuh secara mandiri. Dirancang untuk evolusi offline dan inferensi online
    berkecepatan milidetik.
    """

    def __init__(self, input_dim: int, codebook_dim: int = 128, num_initial_nuclei: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim

        # Proyeksi input ke dimensi codebook
        self.input_projection = nn.Linear(input_dim, codebook_dim)

        self.nuclei = nn.ModuleList()
        for _ in range(num_initial_nuclei):
            self.nuclei.append(ThalamicNucleusVAE(input_dim, codebook_dim))

        # ANN Index untuk pencarian cepat
        self.ann_index = faiss.IndexFlatL2(codebook_dim)
        if num_initial_nuclei > 0:
            # Inisialisasi embedding untuk setiap inti VAE
            initial_embeddings = self._get_initial_embeddings(num_initial_nuclei)
            self.ann_index.add(initial_embeddings.detach().cpu().numpy())

        # GNN Proyektor untuk subgraph
        self.gnn_projector = GCNConv(codebook_dim, codebook_dim)

        # Gerbang monitor koherensi
        self.coherence_monitor_gate = nn.Sequential(
            nn.Linear(codebook_dim, 1), nn.Sigmoid())

        logger.info(
            f"⚛️ Inti Thalamus Kuantum (QTC) aktif dengan {len(self.nuclei)} Inti awal.")

    def _get_initial_embeddings(self, num_nuclei: int) -> torch.Tensor:
        """Helper untuk mendapatkan embedding awal dari setiap inti VAE."""
        embeddings = []
        dummy_input = torch.randn(1, self.input_dim)
        with torch.no_grad():
            for nucleus in self.nuclei:
                nucleus.eval()
                mu, _ = nucleus(dummy_input.to(next(nucleus.parameters()).device))
                embeddings.append(mu)
        return torch.cat(embeddings, dim=0)

    def evolve_structure_offline(self, data_sample: torch.Tensor, learning_rate: float = 1e-4):
        """
        [KERJA OFFLINE] Menjalankan satu siklus pertumbuhan inti baru.
        Fungsi ini dipanggil oleh worker latar belakang saat sistem idle.
        """
        if len(data_sample) < 100:
            logger.warning("[QTC Evolve] Data sampel tidak cukup untuk evolusi (<100).")
            return

        device = next(self.parameters()).device
        data_sample = data_sample.to(device)

        # Temukan inti dengan error rekonstruksi tertinggi
        errors = []
        with torch.no_grad():
            for nucleus in self.nuclei:
                nucleus.eval()
                recon, mu, logvar = nucleus(data_sample)
                loss = nucleus.loss_function(recon, data_sample, mu, logvar)['loss']
                errors.append(loss)
        
        worst_nucleus_idx = torch.argmax(torch.tensor(errors, device=device)).item()

        logger.info(
            f"🧬 [QTC Evolve] Inti #{worst_nucleus_idx} memiliki error tertinggi. Melahirkan inti baru...")
        
        # Buat, latih, dan tambahkan inti baru
        new_nucleus = ThalamicNucleusVAE(self.input_dim, self.codebook_dim).to(device)
        new_nucleus.load_state_dict(self.nuclei[worst_nucleus_idx].state_dict())
        self.nuclei.append(new_nucleus)

        optimizer = torch.optim.Adam(new_nucleus.parameters(), lr=learning_rate)
        new_nucleus.train()
        for _ in range(10):  # Pelatihan singkat
            recon, mu, logvar = new_nucleus(data_sample)
            loss = new_nucleus.loss_function(recon, data_sample, mu, logvar)['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Tambahkan embedding inti baru ke indeks ANN
        with torch.no_grad():
            new_nucleus.eval()
            mu, _ = new_nucleus.encode(data_sample.mean(dim=0, keepdim=True))
            new_embedding = mu.cpu().numpy()
        
        self.ann_index.add(new_embedding)
        logger.info(
            f"✅ [QTC Evolve] Inti baru #{len(self.nuclei)-1} lahir, terlatih, dan diindeks.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [INFERENCE ONLINE] Alur kerja berkecepatan milidetik.
        """
        if x.dim() != 3:
            raise ValueError(f"QTC mengharapkan input 3D [Batch, Seq, Feat], tetapi menerima {x.dim()}D.")

        batch_size, _, num_features = x.shape

        # Penanganan jika dimensi fitur input tidak cocok
        if num_features != self.input_projection.in_features:
            raise RuntimeError(
                f"Dimensi fitur input tidak cocok. Diterima: {num_features}, "
                f"Diharapkan: {self.input_projection.in_features}."
            )
        
        # Rangkum informasi di sepanjang dimensi sekuens (dim=1)
        # Ini mengubah [Batch, Sequence, Features] -> [Batch, Features]
        x_pooled = x.mean(dim=1)

        if self.ann_index.ntotal == 0:
            logger.warning("[QTC Forward] Indeks ANN kosong. Mengembalikan tensor nol.")
            return torch.zeros((batch_size, self.codebook_dim), device=x.device)

        # 1. Proyeksikan input dan cari inti terdekat
        with torch.no_grad():
            x_projected = self.input_projection(x_pooled)
        
        _, indices = self.ann_index.search(
            x_projected.cpu().numpy(), k=min(3, self.ann_index.ntotal))
        
        active_nuclei_indices = np.unique(indices.flatten())
        
        # 2. Proses per item batch untuk membentuk thalamic vector
        batch_thalamic_vectors = torch.zeros(batch_size, self.codebook_dim, device=x.device)
        for b_idx in range(batch_size):
            active_embeddings = []
            for i in active_nuclei_indices:
                if i != -1 and i < len(self.nuclei):
                    nucleus = self.nuclei[i]
                    # Berikan satu sampel input ke VAE untuk mendapatkan representasi 'z'
                    mu, logvar = nucleus(x_pooled[b_idx].unsqueeze(0))
                    z = nucleus.reparameterize(mu, logvar)
                    active_embeddings.append(z.squeeze(0))

            if not active_embeddings:
                continue # Biarkan sebagai tensor nol jika tidak ada inti aktif

            subgraph_nodes = torch.stack(active_embeddings)
            
            # 3. Proses subgraph dengan GNN
            if subgraph_nodes.shape[0] <= 1:
                thalamic_vector = subgraph_nodes.mean(dim=0)
            else:
                # Buat graph yang terhubung penuh (fully connected)
                num_active = subgraph_nodes.shape[0]
                edge_index = torch.combinations(torch.arange(num_active, device=x.device), r=2).t()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) # Buat jadi non-direksional
                
                projected_subgraph = self.gnn_projector(subgraph_nodes, edge_index)
                thalamic_vector = projected_subgraph.mean(dim=0) # Agregasi hasil GNN
            
            batch_thalamic_vectors[b_idx] = thalamic_vector

        # 4. Terapkan gerbang koherensi sebelum mengembalikan hasil
        coherence_gate = self.coherence_monitor_gate(batch_thalamic_vectors)
        return batch_thalamic_vectors * coherence_gate
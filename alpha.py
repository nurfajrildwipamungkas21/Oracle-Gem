
# --- [NSC] Late-binding guard installer (non-invasive) ---
try:
    import threading, sys
    def _install_guards_once():
        mod = sys.modules.get(__name__)
        if not mod: 
            return
        # run_iqro_protocol_scan guard
        try:
            fn = getattr(mod, "run_iqro_protocol_scan", None)
            if fn and not getattr(fn, "_nsc_wrapped", False):
                def _wrap_run_iqro(root_path, api_pool, nsmm, brain, engine, *a, **k):
                    engine2 = ensure_async_engine(engine)
                    return fn(root_path, api_pool, nsmm, brain, engine2, *a, **k)
                _wrap_run_iqro._nsc_wrapped = True
                setattr(mod, "run_iqro_protocol_scan", _wrap_run_iqro)
        except Exception:
            pass
        # background_cognition_worker guard
        try:
            bg = getattr(mod, "background_cognition_worker", None)
            if bg and not getattr(bg, "_nsc_wrapped", False):
                import functools
                @functools.wraps(bg)
                def _bg_inner(*args, **kwargs):
                    if "engine" in kwargs:
                        kwargs["engine"] = ensure_async_engine(kwargs.get("engine"))
                    return bg(*args, **kwargs)
                _bg_inner._nsc_wrapped = True
                setattr(mod, "background_cognition_worker", _bg_inner)
        except Exception:
            pass

    def _poll_for_defs(max_secs=30.0, period=0.2):
        import time
        t0 = time.time()
        while time.time() - t0 < max_secs:
            _install_guards_once()
            time.sleep(period)

    threading.Thread(target=_poll_for_defs, kwargs={"max_secs": 60.0, "period": 0.25}, daemon=True).start()
except Exception as _e:
    try:
        logger.warning("Guard poller failed: %s", _e)
    except Exception:
        pass
# --- [END NSC] ---
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==== [OVERHAUL PATCH BLOCK v3 — CPU-only, Safe ZSC, SQLite WAL, Stability Callback] ====
def _configure_cpu_numeric_policy():
    try:
        import os, torch
        os.environ.setdefault("TORCH_ENABLE_CPU_FP16", "0")
        os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
        torch.set_num_threads(max(1, (os.cpu_count() or 6)//2))
        try: torch.set_num_interop_threads(1)
        except Exception: pass
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    except Exception as _e:
        try: logger.warning(f"[NUMPOLICY] Failed to configure CPU numeric policy: {_e}")
        except Exception: pass
_configure_cpu_numeric_policy()

def _install_safe_pipeline():
    try:
        import os, logging, transformers as _tfm
        _orig_pipeline = getattr(_tfm, "pipeline", None)
        LIGHT_ZSC = os.getenv("ORACLE_GEM_ZSC_MODEL", "typeform/distilbert-base-uncased-mnli")
        def _is_zero_shot(task, model):
            t = (task or "").lower(); m = (model or "").lower()
            return ("zero-shot" in t) or ("mnli" in m) or ("xnli" in m)
        def _safe_pipeline(task=None, model=None, **kwargs):
            kwargs.setdefault("device", -1)
            model_kwargs = kwargs.pop("model_kwargs", {}) or {}
            model_kwargs.setdefault("torch_dtype", None)
            kwargs["model_kwargs"] = model_kwargs
            if _is_zero_shot(task, model):
                preferred = os.getenv("ORACLE_GEM_ZSC_MODEL", LIGHT_ZSC)
                candidate_list = [preferred, "typeform/distilbert-base-uncased-mnli"]
            else:
                candidate_list = [model] if model else [None]
            last_err = None
            for cand in candidate_list:
                try:
                    return _orig_pipeline(task, model=cand, **kwargs) if _orig_pipeline else None
                except OSError as e:
                    last_err = e
                    if "paging file is too small" in str(e).lower():
                        logging.getLogger("Oracle Gem").error(f"[SAFE_PIPELINE] OOM/OSError on model '{cand}'. Trying smaller model...")
                        continue
                    else: continue
                except Exception as e:
                    last_err = e; continue
            class _NoopZSC:
                def __call__(self, *a, **k): return [{"labels": [], "scores": []}]
            logging.getLogger("Oracle Gem").warning(f"[SAFE_PIPELINE] All candidates failed; using Noop ZSC. Last error: {last_err}")
            return _NoopZSC()
        if _orig_pipeline:
            _tfm.pipeline = _safe_pipeline
            globals()["pipeline"] = _safe_pipeline
    except Exception as _e:
        try: logger.warning(f"[SAFE_PIPELINE] Failed to install safe pipeline: {_e}")
        except Exception: pass
_install_safe_pipeline()

def _install_sqlite_wal():
    try:
        import sqlite3 as _sqlite3, re as _re, logging
        _orig_connect = _sqlite3.connect
        # Patch sqlite3.connect to always enable WAL and busy_timeout.  This
        # wrapper unconditionally sets PRAGMA journal_mode=WAL and
        # busy_timeout=60s on every connection.  While SQLite WAL can
        # slightly increase latency, it greatly improves concurrency and
        # mitigates "database is locked" errors when multiple workers are
        # writing concurrently.  We also set synchronous=NORMAL to balance
        # durability and performance.
        def _connect_wrapped(database, *args, **kwargs):
            conn = _orig_connect(database, *args, **kwargs)
            try:
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                cur.execute("PRAGMA busy_timeout=60000;")
                conn.commit()
            except Exception as e:
                logging.getLogger("Oracle Gem").warning(f"[SQLITE-WAL] PRAGMA setup failed on {database}: {e}")
            return conn
        _sqlite3.connect = _connect_wrapped
    except Exception as _e:
        try: logger.warning(f"[SQLITE-WAL] Failed to install WAL wrapper: {_e}")
        except Exception: pass
_install_sqlite_wal()

class _NoopEngine:
    def ask_async(self, *a, **k): return None
def ensure_async_engine(engine):
    try:
        if engine is None or not hasattr(engine, "ask_async"): return _NoopEngine()
        return engine
    except Exception: return _NoopEngine()

def _nsc_bind_engines():
    import sys
    mod = sys.modules.get(__name__)
    if not mod: return
    fn = getattr(mod, "run_iqro_protocol_scan", None)
    if fn and not getattr(fn, "_nsc_wrapped2", False):
        def _wrap_run_iqro(root_path, api_pool, nsmm, brain, engine, *a, **k):
            engine2 = ensure_async_engine(engine); return fn(root_path, api_pool, nsmm, brain, engine2, *a, **k)
        _wrap_run_iqro._nsc_wrapped2 = True; setattr(mod, "run_iqro_protocol_scan", _wrap_run_iqro)
    bg = getattr(mod, "background_cognition_worker", None)
    if bg and not getattr(bg, "_nsc_wrapped2", False):
        import functools
        @functools.wraps(bg)
        def _bg_inner(*args, **kwargs):
            if "engine" in kwargs: kwargs["engine"] = ensure_async_engine(kwargs.get("engine"))
            return bg(*args, **kwargs)
        _bg_inner._nsc_wrapped2 = True; setattr(mod, "background_cognition_worker", _bg_inner)

try:
    import threading as _th, time as _time
    def _poll_eng(max_secs=30.0, period=0.25):
        t0 = _time.time()
        while _time.time() - t0 < max_secs:
            _nsc_bind_engines(); _time.sleep(period)
    _th.Thread(target=_poll_eng, daemon=True).start()
except Exception: pass

def _install_unified_stability_callback():
    try:
        import pytorch_lightning as _pl
        import torch, torch.nn as _nn
        from pytorch_lightning.callbacks import Callback as _CB
        class UnifiedStabilityCallback(_CB):
            def __init__(self, max_grad_norm=1.0, logit_clamp=20.0):
                self.max_grad_norm = float(max_grad_norm); self.logit_clamp = float(logit_clamp)
            def on_after_backward(self, trainer, pl_module):
                try: _nn.utils.clip_grad_norm_(pl_module.parameters(), self.max_grad_norm)
                except Exception: pass
            def on_before_optimizer_step(self, trainer, pl_module, optimizer): pass
            def on_after_batch_end(self, trainer, pl_module):
                last_logits = getattr(pl_module, "last_logits", None)
                if last_logits is not None and torch.is_tensor(last_logits):
                    with torch.no_grad(): last_logits.clamp_(-self.logit_clamp, self.logit_clamp)
        if not getattr(_pl.Trainer, "_usc_patched", False):
            _orig_init = _pl.Trainer.__init__
            def _patched_init(self, *args, **kwargs):
                cbs = list(kwargs.get("callbacks", [])) if kwargs.get("callbacks") else []
                if not any(isinstance(cb, UnifiedStabilityCallback) for cb in cbs): cbs.append(UnifiedStabilityCallback())
                kwargs["callbacks"] = cbs; return _orig_init(self, *args, **kwargs)
            _pl.Trainer.__init__ = _patched_init; _pl.Trainer._usc_patched = True
    except Exception as _e:
        try: logger.warning(f"[USC] Failed to install UnifiedStabilityCallback: {_e}")
        except Exception: pass
_install_unified_stability_callback()

def _patch_brain_add_chunks():
    try:
        from src.models.model_alpha.worker_utils import Brain as _Brain
        import sqlite3 as _sqlite3, time as _time, logging
        if hasattr(_Brain, "_add_chunks_patched"): return
        _orig = _Brain.add_chunks
        def _robust(self, chunks, source_name=None):
            MAX_RETRY = 12
            for i in range(MAX_RETRY):
                try: return _orig(self, chunks, source_name=source_name)
                except _sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower():
                        _time.sleep(0.5 * (i+1)); continue
                    raise
            logging.getLogger("Oracle Gem").error("[Brain.add_chunks] Failed after retries; dropping this batch."); return None
        _Brain.add_chunks = _robust; _Brain._add_chunks_patched = True
    except Exception as _e:
        try: logger.warning(f"[Brain.add_chunks] patch skipped: {_e}")
        except Exception: pass
try: _patch_brain_add_chunks()
except Exception: pass
# ==== [END OVERHAUL PATCH BLOCK] ====

import os
import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
try:
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
except IndexError:
    print("Struktur direktori tidak seperti yang diharapkan...")
    
from typing import Dict, List, Optional

def _cpu_count_logical() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=True) or 1
    except Exception:
        import os
        return os.cpu_count() or 1

def build_affinity_map(total_cores: Optional[int] = None,
                       primary_pairs: Optional[List[List[int]]] = None,
                       aux_pairs: Optional[List[List[int]]] = None,
                       num_workers: int = 4) -> Dict[int, List[int]]:
    ncores = total_cores or _cpu_count_logical()
    half = max(1, ncores // 2)
    default_primary = [[i] for i in range(0, min(half, ncores))]
    default_aux     = [[i] for i in range(half, ncores)]
    P = primary_pairs if primary_pairs else default_primary
    A = aux_pairs if aux_pairs else default_aux
    chosen = []
    pi, ai = 0, 0
    while len(chosen) < num_workers:
        if pi < len(P):
            chosen.append(P[pi]); pi += 1
            if len(chosen) >= num_workers: break
        if ai < len(A):
            chosen.append(A[ai]); ai += 1
        if pi >= len(P) and ai >= len(A):
            break
    return {wid: cores for wid, cores in enumerate(chosen)}

def _affined_worker_init_win32(worker_id: int, affinity_map=None, omp_threads: int = 1, mkl_threads: int = 1):
    try:
        import os, torch
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        os.environ["MKL_NUM_THREADS"] = str(mkl_threads)
        try: torch.set_num_threads(1)
        except Exception: pass
        try: torch.set_num_interop_threads(1)
        except Exception: pass
    except Exception:
        pass
    try:
        import psutil
        if affinity_map is None:
            affinity_map = build_affinity_map(total_cores=None)
        cores = affinity_map.get(worker_id)
        if cores:
            psutil.Process().cpu_affinity(cores)
            print(f"[AFFINITY] worker#{worker_id} -> cores={cores}")
    except Exception as e:
        print(f"[AFFINITY] worker#{worker_id} skip ({e})")


def make_affined_worker_init(affinity_map: Dict[int, List[int]], omp_threads: int = 1, mkl_threads: int = 1):
    def _init(worker_id: int):
        try:
            import os
            os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(mkl_threads))
        except Exception:
            pass
        try:
            import torch
            try: torch.set_num_threads(1)
            except Exception: pass
            try: torch.set_num_interop_threads(1)
            except Exception: pass
        except Exception:
            pass
        try:
            import psutil
            cores = affinity_map.get(worker_id)
            if cores:
                psutil.Process().cpu_affinity(cores)
                print(f"[AFFINITY] worker#{worker_id} -> cores={cores}")
        except Exception as e:
            print(f"[AFFINITY] worker#{worker_id} skip ({e})")
    return _init

import psutil
import pygetwindow as gw
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from pytorch_lightning.callbacks import Callback
from rich.logging import RichHandler
import nolds
import pytesseract
from pytesseract import Output as PyTesseractOutput
from pdf2image import convert_from_path
from instructor.exceptions import InstructorRetryException
from functools import wraps, partial
import time
import threading
from requests.exceptions import RequestException
from google.generativeai import GenerativeModel
import google.generativeai as genai
import instructor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool, tool
from openai import OpenAI
import graphviz
from dowhy import CausalModel
import dowhy
from packaging.version import parse as parse_version
from joblib import Parallel, delayed
from json import JSONDecodeError
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction import FeatureHasher
from hurst import compute_Hc
import holidays
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from econml.dr import LinearDRLearner
import hdbscan
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    StochasticWeightAveraging,
    RichProgressBar,
)
import mlflow
from mapie.regression import MapieRegressor
from catboost import CatBoostRegressor
from langdetect import detect, LangDetectException
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tsai.data.transforms import TSMaskOut, TSGaussianNoise
from tqdm import tqdm
from pyhht.utils import inst_freq
from pyhht.emd import EMD
import pywt
import talib
from dtaidistance import dtw
import matplotlib.pyplot as plt
from multiprocessing import Manager
from pytorch_optimizer import SAM
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import logging
import warnings
import json
import io
import uuid
import sqlite3
import subprocess
import re
import random
import inspect
import traceback
import ast
import csv
import math
import optuna
import argparse
import pkgutil
import textwrap
import importlib
import queue
import torch

# === [Expert Router Definitions] ===
class ExpertRouter:
    """
    ExpertRouter manages dynamic routing of model experts per epoch.  It
    implements the three control levers discussed in the design document:
    
    * λ (loss weights): adjusts the contribution of each objective.
    * μ (micro‑LR): scales the learning rate for select parameter groups.
    * γ (expert gating): turns expert modules on or off based on utility and
      cost under a compute budget.  Inactive experts have their parameters
      frozen and their forward computations skipped where possible.

    The router operates by probing a small number of mini‑batches at the
    beginning of each epoch to estimate context statistics (e.g., feature
    standard deviation, spectral power, autocorrelation).  These statistics
    inform simple heuristics for computing utilities for each expert group.
    Experts are then ranked by utility divided by cost, and a subset is
    selected subject to a budget.  The router also supports ε‑greedy
    exploration and a minimal activation window to avoid thrashing between
    experts.

    Adaptation is performed periodically during an epoch based on running
    means of loss components: if a reactive loss dominates, its weight and
    micro LR are reduced; if a loss is underrepresented, its weight may be
    increased toward its baseline.  β‑VAE warm‑up and clipping rails are
    applied through the router’s interactions with the Lightning module.
    """

    def __init__(self, lit_module: 'HybridSSP_LitModule', budget_ratio: float = 0.3,
                 epsilon: float = 0.1, min_active_epochs: int = 1, cooldown_epochs: int = 1,
                 adapt_interval: int = 50):
        """
        Args:
            lit_module: reference to the parent Lightning module.  The router
                introspects its submodules and will freeze/unfreeze parameters
                accordingly.
            budget_ratio: fraction of the total estimated compute cost to use
                when selecting experts.  A value of 0.3 means at most 30% of
                the cumulative cost of all experts will be active each epoch.
            epsilon: probability of exploring a non‑greedy expert at epoch
                selection (ε‑greedy).  Helps avoid local minima in routing.
            min_active_epochs: minimum number of epochs an expert must stay
                active once selected, to avoid rapid toggling.
            cooldown_epochs: number of epochs an expert must remain inactive
                before it is eligible again after being dropped.  Helps avoid
                thrashing.
            adapt_interval: number of training steps between successive
                adaptations of loss weights and micro LRs.
        """
        self.module = lit_module
        self.hparams = getattr(lit_module, 'hparams', {})
        self.device = getattr(lit_module, 'device', 'cpu')
        # Budget and exploration parameters
        self.budget_ratio = budget_ratio
        self.epsilon = epsilon
        self.min_active_epochs = min_active_epochs
        self.cooldown_epochs = cooldown_epochs
        self.adapt_interval = adapt_interval
        # Epoch counters for each group to enforce min_active/cooldown
        self.group_last_active: Dict[str, int] = {}
        self.group_last_inactive: Dict[str, int] = {}
        # Baseline loss weights and micro LR multipliers
        self.baseline_lambda = {
            'loss_contrastive': 1.00,
            'loss_jigsaw': 0.10,
            'loss_spike': 0.05,
            'loss_volatility': 0.02,
            'kld_loss': 0.00,
        }
        # Micro learning‑rate multipliers for different parameter groups.
        # Smaller values tame reactive heads (volatility and VAE) while leaving
        # others untouched.  These values correspond to 5% and 10% of the base
        # learning rate respectively, with raw_log_vars sharing the same scale
        # as the VAE.  Remaining parameters use the full learning rate.
        self.baseline_mu = {
            'volatility': 0.05,
            'vae': 0.10,
            'raw_log_vars': 0.10,
            'other': 1.00,
        }
        self.lambda_values = self.baseline_lambda.copy()
        self.mu_values = self.baseline_mu.copy()

        # Track dominance for reactive losses (volatility and KLD).  These
        # counters reset whenever a reactive loss falls below the dominance
        # threshold and trigger additional down‑scaling of λ and μ when
        # dominance persists across multiple adaptation windows.
        self._reactive_dominant_counts: Dict[str, int] = {}

        # Tracking variables for jigsaw dominance and temporary deactivation of
        # the Heads‑Other group.  When the jigsaw loss repeatedly dominates
        # the loss proportions, we reduce its weight and disable the group
        # for a short cooldown period.  These counters persist across
        # adapt intervals and epochs.
        self._jigsaw_dominant_count = 0
        self._jigsaw_cooldown_remaining = 0
        # Group definitions: modules and associated parameter groups
        # Cost is relative estimate of compute; higher means heavier.
        # Define expert groups.  Each entry describes a logical set of
        # modules which can be enabled/disabled as a unit.  These groups
        # capture most of the heavy computation paths in the model.  Costs
        # are relative and will be used together with empirically measured
        # costs to determine what fraction of experts can be active under
        # the budget.
        #
        # Newly added groups include heads and shielding layers to allow
        # finer grained routing across all parts of the network.  The
        # "Heads-Other" group contains regression/anomaly/uncertainty
        # prediction heads and jigsaw/spike heads used during SSL.  The
        # "Shields" group contains various stabilisation layers that
        # precondition the latent space before heads and VAE.
        self.groups = {
            'Temporal-Attn': {
                'modules': ['dpa_stif_layers'],
                'params': [],
                'cost': 2.0,
            },
            'Spectral': {
                'modules': ['fno_layer', 'norm_fno'],
                'params': [],
                'cost': 2.0,
            },
            'Latent-Manifold': {
                'modules': ['qtc', 'variational_encoder'],
                'params': [],
                'cost': 3.0,
            },
            'Spiking': {
                'modules': ['snn_processor', 'snn_output_projection'],
                'params': [],
                'cost': 1.5,
            },
            # Additional expert group for other heads.  These heads are
            # relatively lightweight compared to the core encoders but
            # nevertheless can be disabled when the dataset or task does
            # not require them.  Grouping them together simplifies
            # gating logic.
            'Heads-Other': {
                'modules': [
                    'jigsaw_head',
                    'spike_timing_head',
                    'regression_head',
                    'anomaly_head',
                    'uncertainty_attribution_head',
                ],
                'params': [],
                'cost': 1.0,
            },
            # Group containing stabilisation and shielding layers.  These
            # modules prepare the latent representation before being fed
            # into heads or VAE.  Though relatively inexpensive, they
            # comprise a coherent functional unit that can be gated.
            'Shields': {
                'modules': [
                    'shield_pre_vae',
                    'kevlar_layer',
                    'shield_pre_heads',
                ],
                'params': [],
                'cost': 0.5,
            },
        }
        # Build list of parameters for each group
        for gname, gdesc in self.groups.items():
            param_list = []
            for mod_name in gdesc['modules']:
                mod = getattr(lit_module, mod_name, None)
                if mod is None: continue
                param_list.extend(list(mod.parameters()))
            gdesc['params'] = param_list
        # Active groups at current epoch
        self.active_groups: List[str] = []
        # For exploration random seed
        self._rng = random.Random(42)

    def _compute_context_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute simple statistics on the input batch to drive routing heuristics."""
        metrics = {}
        try:
            x = batch.get('query_x_aug1') if isinstance(batch, dict) else None
            if x is None:
                return metrics
            # x: (B, T, F)
            with torch.no_grad():
                x_device = x.to(self.device)
                # Standard deviation of raw features across batch, time and feature dims
                feat_std = x_device.float().std().item() if x_device.numel() > 0 else 0.0
                metrics['feat_std'] = feat_std
                # Fraction of missing values (NaNs) in the batch.  High values
                # indicate data quality issues or missingness which may
                # influence expert selection (e.g., avoid heavy modules when
                # the data quality is poor).
                try:
                    nan_mask = torch.isnan(x_device)
                    metrics['missing_frac'] = float(nan_mask.sum()) / float(x_device.numel() + 1e-8)
                except Exception:
                    metrics['missing_frac'] = 0.0
                # Outlier fraction based on z-score > 3.  We compute
                # z-scores across all values to detect heavy tails or
                # extreme values.  If std is zero, fall back to zero.
                try:
                    flat = x_device.float().view(-1)
                    mean_val = flat.mean()
                    std_val = flat.std() + 1e-8
                    z_scores = (flat - mean_val) / std_val
                    metrics['outlier_frac'] = float((z_scores.abs() > 3.0).float().mean())
                except Exception:
                    metrics['outlier_frac'] = 0.0
                # Spectral power: take FFT along time dimension of first feature for first batch
                try:
                    # sample subset to reduce compute
                    xs = x_device[: min(4, x_device.size(0)), :, 0]
                    fft_vals = torch.fft.rfft(xs, dim=1)
                    spec_power = float(torch.mean(torch.abs(fft_vals)))
                    metrics['spectral_power'] = spec_power
                except Exception:
                    metrics['spectral_power'] = 0.0
                # Autocorrelation coefficient at lag 1 (ACF1)
                try:
                    xs = x_device[: min(4, x_device.size(0)), :, 0]
                    xs_center = xs - xs.mean(dim=1, keepdim=True)
                    acf_num = (xs_center[:, :-1] * xs_center[:, 1:]).sum(dim=1)
                    acf_den = (xs_center ** 2).sum(dim=1)
                    acf1 = float(torch.mean(acf_num / (acf_den + 1e-8)))
                    metrics['acf1'] = acf1
                except Exception:
                    metrics['acf1'] = 0.0
                # Spike rate (if spiking target present).  We look for query_spike_target key.
                spike = batch.get('query_spike_target') if isinstance(batch, dict) else None
                if spike is not None:
                    spike_rate = float(spike.float().mean())
                    metrics['spike_rate'] = spike_rate
                else:
                    metrics['spike_rate'] = 0.0
                # Volatility target variance (if present).  High variance suggests heavy tails.
                vol_tgt = batch.get('query_vol_target') if isinstance(batch, dict) else None
                if vol_tgt is not None:
                    vol_std = float(vol_tgt.float().std())
                    metrics['vol_std'] = vol_std
                else:
                    metrics['vol_std'] = 0.0
        except Exception as _e:
            logger.warning(f"[Router] Failed to compute context metrics: {_e}")
        return metrics

    def _heuristic_utilities(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute heuristic utility scores for each group based on context metrics."""
        # We normalise metrics to avoid large magnitude differences
        feat_std = metrics.get('feat_std', 0.0)
        spectral_power = metrics.get('spectral_power', 0.0)
        acf1 = metrics.get('acf1', 0.0)
        spike_rate = metrics.get('spike_rate', 0.0)
        vol_std = metrics.get('vol_std', 0.0)
        utilities = {}
        # Initial utility assignments based on relative metrics
        utilities['Spectral'] = (spectral_power / (feat_std + 1e-8)) if feat_std > 0 else spectral_power
        utilities['Temporal-Attn'] = abs(acf1)
        utilities['Latent-Manifold'] = (vol_std / (feat_std + 1e-8)) if feat_std > 0 else vol_std
        utilities['Spiking'] = spike_rate
        # Penalise utilities when data quality is poor (missingness/outliers).  The
        # presence of many NaNs or outliers suggests focusing on simpler
        # pathways and reducing heavy compute.  We reduce all utilities by
        # a factor proportional to the sum of missing and outlier fractions.
        miss = metrics.get('missing_frac', 0.0)
        outlier = metrics.get('outlier_frac', 0.0)
        decay = max(0.0, 1.0 - (miss + outlier))
        for k in list(utilities.keys()):
            utilities[k] = utilities[k] * decay
        return utilities

    def _select_active_groups(self, utilities: Dict[str, float], current_epoch: int) -> List[str]:
        """Select a subset of groups based on utility/cost ratio and budget."""
        # Compute cost budget in relative units
        total_cost = sum(desc['cost'] for desc in self.groups.values())
        budget = self.budget_ratio * total_cost
        # Compute score = utility / cost
        scores = {}
        for name, desc in self.groups.items():
            util = utilities.get(name, 0.0)
            cost = desc['cost']
            scores[name] = util / (cost + 1e-8)
        # Sort groups by score descending
        sorted_groups = sorted(self.groups.keys(), key=lambda k: scores[k], reverse=True)
        selected = []
        used_cost = 0.0
        # Determine if we will explore (ε-greedy)
        explore = (self._rng.random() < self.epsilon)
        for g in sorted_groups:
            # Hysteresis: skip if in cooldown
            last_act = self.group_last_active.get(g, -1)
            last_inact = self.group_last_inactive.get(g, -1)
            # Enforce cooldown: if just inactive within cooldown period skip
            if (current_epoch - last_inact) < self.cooldown_epochs:
                continue
            # Enforce min-active: if still within min_active, ensure selected if previously active
            if (current_epoch - last_act) < self.min_active_epochs:
                selected.append(g)
                used_cost += self.groups[g]['cost']
                continue
            # Greedy selection until budget used
            if used_cost + self.groups[g]['cost'] <= budget:
                selected.append(g)
                used_cost += self.groups[g]['cost']
        # ε-greedy exploration: with some probability include one random candidate not already selected
        if explore:
            # pick from groups not selected and not in cooldown
            candidates = [g for g in self.groups.keys() if g not in selected and (current_epoch - self.group_last_inactive.get(g, -1)) >= self.cooldown_epochs]
            if candidates:
                rand_choice = self._rng.choice(candidates)
                selected.append(rand_choice)
        return selected

    def start_epoch(self, model: 'HybridSSP_LitModule', batch: Dict[str, torch.Tensor], current_epoch: int, total_epochs: int):
        """
        Initialise expert routing at the start of each epoch.  During the
        early safe phase (epochs 0–1) the router activates only a minimal
        set of experts to prevent unstable components from dominating.  In
        this phase, Heads‑Other and Shields are always on, and exactly one
        of the temporal attention or spectral pathway is chosen based on
        their relative utility.  Spiking and latent manifold experts are
        disabled entirely.  After the safe phase, the router falls back
        to its normal selection strategy based on utility and budget.

        Loss weight schedules for volatility and jigsaw are also applied
        here: in the first two epochs the weights are kept very small,
        then increased gradually until they reach their baseline values.
        A custom β‑VAE schedule sets beta to 0 until epoch 2, then
        1e‑4 until epoch 5, then 1e‑3 afterwards.
        """
        # Compute context metrics and heuristics from the probe batch
        metrics = self._compute_context_metrics(batch)
        utilities = self._heuristic_utilities(metrics)
        # Determine active expert groups
        if current_epoch < 2:
            # Choose between the temporal and spectral pathways based on
            # utility/cost ratio
            util_t = utilities.get('Temporal-Attn', 0.0) / (self.groups['Temporal-Attn']['cost'] + 1e-8)
            util_s = utilities.get('Spectral', 0.0) / (self.groups['Spectral']['cost'] + 1e-8)
            core = 'Temporal-Attn' if util_t >= util_s else 'Spectral'
            active = ['Heads-Other', 'Shields', core]
        else:
            active = self._select_active_groups(utilities, current_epoch)
        # Enforce temporary deactivation of Heads‑Other if jigsaw recently
        # dominated.  When `_jigsaw_cooldown_remaining` is positive, we
        # remove Heads‑Other from the active set for this epoch and
        # decrement the counter.  This prevents the jigsaw head from
        # re‑entering immediately after being suppressed by the router.
        if getattr(self, '_jigsaw_cooldown_remaining', 0) > 0:
            if 'Heads-Other' in active:
                active = [g for g in active if g != 'Heads-Other']
            self._jigsaw_cooldown_remaining -= 1
        # Update last active/inactive records to enforce hysteresis
        for g in self.groups.keys():
            if g in active:
                self.group_last_active[g] = current_epoch
            else:
                self.group_last_inactive[g] = current_epoch
        self.active_groups = active
        # Log routing decision
        try:
            scores = {g: (utilities.get(g, 0.0) / (self.groups[g]['cost'] + 1e-8)) for g in self.groups}
            used_cost = sum(self.groups[g]['cost'] for g in active)
            total_cost = sum(self.groups[g]['cost'] for g in self.groups)
            budget_used = used_cost / (total_cost + 1e-8)
            logger.info(f"[Router] Epoch {current_epoch}: active_groups={active}, budget_used={budget_used:.2f}, metrics={metrics}, utilities={utilities}, scores={scores}")
        except Exception as _e:
            logger.warning(f"[Router] Logging error: {_e}")
        # Apply gating to freeze or unfreeze expert parameters
        self._apply_gating(model)
        # Reset λ and μ to their baselines
        self.lambda_values = self.baseline_lambda.copy()
        self.mu_values = self.baseline_mu.copy()
        # Apply schedules for volatility and jigsaw loss weights
        try:
            if current_epoch < 2:
                self.lambda_values['loss_volatility'] = 0.02
                self.lambda_values['loss_jigsaw'] = 0.01
            elif current_epoch < 5:
                self.lambda_values['loss_volatility'] = max(0.05, self.lambda_values.get('loss_volatility', 0.02))
                self.lambda_values['loss_jigsaw'] = max(0.05, self.lambda_values.get('loss_jigsaw', 0.01))
            else:
                self.lambda_values['loss_volatility'] = max(0.05, self.lambda_values.get('loss_volatility', 0.02))
                self.lambda_values['loss_jigsaw'] = max(0.10, self.lambda_values.get('loss_jigsaw', 0.05))
        except Exception:
            pass
        # Custom β‑VAE schedule
        try:
            if hasattr(model.hparams, 'beta_vib'):
                if current_epoch < 2:
                    model.hparams['beta_vib'] = 0.0
                elif current_epoch < 5:
                    model.hparams['beta_vib'] = 1e-4
                else:
                    model.hparams['beta_vib'] = 1e-3
                logger.info(f"[Router] Beta VIB warmed to {model.hparams['beta_vib']:.6f}")
        except Exception as _e:
            logger.warning(f"[Router] Beta schedule error: {_e}")
        # Propagate λ and μ to the model and its optimiser
        model.dynamic_loss_weights.update(self.lambda_values)
        self._apply_micro_lr(model)

    def _apply_gating(self, model: 'HybridSSP_LitModule'):
        """Freeze parameters of inactive expert groups and set flag on model for gating."""
        # Build gating dict on model for forward path decisions
        gating_dict = {g: (g in self.active_groups) for g in self.groups.keys()}
        model._router_gating = gating_dict
        # Freeze or unfreeze parameters
        for gname, gdesc in self.groups.items():
            active = gname in self.active_groups
            for p in gdesc['params']:
                p.requires_grad = active
        # Reset other modules (heads etc.) do not need gating here; heads gating via lambda

    def _apply_micro_lr(self, model: 'HybridSSP_LitModule'):
        """Adjust micro LR multipliers on optimizer param groups based on current mu_values."""
        try:
            opt = model.optimizers()  # may return list or single
            if isinstance(opt, list):
                opt = opt[0] if opt else None
            if opt is None:
                return
            # Determine base LR (group 0) and micro LR (group 1) indices.  We assume the
            # first param_group contains base parameters, second contains micro parameters.
            for i, pg in enumerate(opt.param_groups):
                # Identify micro param group by checking if any param belongs to volatility head or VAE
                is_micro_group = False
                for p in pg['params']:
                    if hasattr(model, 'volatility_head') and any(p is q for q in model.volatility_head.parameters()):
                        is_micro_group = True
                        break
                    # raw_log_vars is a parameter of the module itself
                    if isinstance(p, torch.nn.Parameter) and p is getattr(model, 'raw_log_vars', None):
                        is_micro_group = True
                        break
                if is_micro_group:
                    # Combine both volatility and VAE micro values; take minimum to be conservative
                    mu_vol = self.mu_values.get('volatility', self.baseline_mu['volatility'])
                    mu_vae = self.mu_values.get('vae', self.baseline_mu['vae'])
                    mu_combined = min(mu_vol, mu_vae)
                    pg['lr'] = model.hparams.lr * mu_combined
                else:
                    pg['lr'] = model.hparams.lr * self.mu_values.get('other', 1.0)
        except Exception as _e:
            logger.warning(f"[Router] Failed to apply micro LR: {_e}")

    def adapt(self, model: 'HybridSSP_LitModule', running_stats: Dict[str, Dict[str, float]]):
        """Adapt loss weights and micro LR based on running mean of losses."""
        # Compute normalized running means (z-score like) for each loss
        # We use simple z-score across current losses to detect dominance
        means = {}
        vals = []
        for name, stats in running_stats.items():
            count = max(1e-6, stats.get('count', 0))
            mean = stats.get('sum', 0.0) / count
            means[name] = mean
            vals.append(mean)
        if not vals:
            return
        # compute mean and std of the means for normalization
        import numpy as _np
        arr = _np.array(list(means.values()), dtype=float)
        m = float(arr.mean())
        s = float(arr.std())
        z_scores = {k: ((v - m) / (s + 1e-8)) for k, v in means.items()}
        # Compute proportion of each loss (normalized positive values only)
        # We shift z-scores to be non-negative
        min_z = min(z_scores.values())
        shifted = {k: (z - min_z) for k, z in z_scores.items()}
        total_shift = sum(shifted.values()) + 1e-8
        proportions = {k: (v / total_shift) for k, v in shifted.items()}

        # ------------------------------------------------------------------
        # [Bounded Proportions] Enforce that each loss contributes within
        # the range [0.10, 0.40].  If a loss dominates (>0.40) its weight is
        # reduced by 20%, floored at 10% of baseline.  If a loss is
        # underrepresented (<0.10) its weight is increased by 10%, capped at
        # the baseline.  This rule is applied uniformly to all losses.
        for loss_name, prop in proportions.items():
            base = self.baseline_lambda.get(loss_name, 1.0)
            current = self.lambda_values.get(loss_name, base)
            # Dominant: shrink weight
            if prop > 0.40:
                new_val = max(base * 0.10, current * 0.8)
                self.lambda_values[loss_name] = new_val
            # Underrepresented: enlarge weight
            elif prop < 0.10:
                new_val = min(base, current * 1.1)
                self.lambda_values[loss_name] = new_val

        # ------------------------------------------------------------------
        # [Jigsaw Dominance Rule] Track consecutive dominance of the jigsaw
        # loss.  When the jigsaw proportion exceeds 0.40 for two
        # successive adaptation windows, reduce its weight further and
        # temporarily disable the Heads‑Other group for one epoch.  Reset
        # the counter if the jigsaw loss falls below the threshold.
        p_jigsaw = proportions.get('loss_jigsaw', 0.0)
        if p_jigsaw > 0.40:
            self._jigsaw_dominant_count = getattr(self, '_jigsaw_dominant_count', 0) + 1
        else:
            self._jigsaw_dominant_count = 0
        if self._jigsaw_dominant_count >= 2:
            # reduce jigsaw weight aggressively
            base = self.baseline_lambda.get('loss_jigsaw', 1.0)
            current = self.lambda_values.get('loss_jigsaw', base)
            new_val = max(base * 0.10, current * 0.7)
            self.lambda_values['loss_jigsaw'] = new_val
            # set cooldown to disable Heads-Other for one epoch
            self._jigsaw_cooldown_remaining = 1
            # reset the dominant counter to avoid repeated triggering
            self._jigsaw_dominant_count = 0

        # ------------------------------------------------------------------
        # [Reactive Dominance Rule] When reactive losses (volatility or KLD)
        # dominate for two consecutive adaptation windows, we aggressively
        # reduce their λ and μ to prevent them from overwhelming other
        # objectives.  Counters are stored in `_reactive_dominant_counts`.
        for reactive_loss in ['loss_volatility', 'kld_loss']:
            prop = proportions.get(reactive_loss, 0.0)
            # Initialise counter if missing
            count = self._reactive_dominant_counts.get(reactive_loss, 0)
            if prop > 0.40:
                count += 1
            else:
                count = 0
            # Update the stored count
            self._reactive_dominant_counts[reactive_loss] = count
            if count >= 2:
                # Reduce λ by ~20% (capped at 10% of baseline)
                base = self.baseline_lambda.get(reactive_loss, 1.0)
                current_lambda = self.lambda_values.get(reactive_loss, base)
                self.lambda_values[reactive_loss] = max(base * 0.10, current_lambda * 0.8)
                # Reduce μ by half with floors (0.02 for volatility, 0.05 for VAE)
                if reactive_loss == 'loss_volatility':
                    current_mu = self.mu_values.get('volatility', self.baseline_mu['volatility'])
                    self.mu_values['volatility'] = max(0.02, current_mu * 0.5)
                elif reactive_loss == 'kld_loss':
                    current_mu = self.mu_values.get('vae', self.baseline_mu['vae'])
                    self.mu_values['vae'] = max(0.05, current_mu * 0.5)
                # Reset counter after adjustment
                self._reactive_dominant_counts[reactive_loss] = 0

        # ------------------------------------------------------------------
        # [Volatility & VAE μ Adaptation] Adjust micro learning rates for
        # volatility and latent manifold heads based on their relative
        # dominance.  If a reactive loss dominates (>0.5), halve its μ down
        # to a floor.  If it is underrepresented (<0.3), slowly recover
        # towards the baseline.  The baselines have been set in
        # `self.baseline_mu` and are used as caps.
        d_vol = proportions.get('loss_volatility', 0.0)
        d_kld = proportions.get('kld_loss', 0.0)
        # Update volatility μ
        if d_vol > 0.5:
            self.mu_values['volatility'] = max(0.05, self.mu_values.get('volatility', self.baseline_mu['volatility']) * 0.5)
        elif d_vol < 0.3:
            self.mu_values['volatility'] = min(self.baseline_mu['volatility'], self.mu_values.get('volatility', self.baseline_mu['volatility']) * 1.2)
        # Update VAE μ (for latent manifold)
        if d_kld > 0.5:
            self.mu_values['vae'] = max(0.10, self.mu_values.get('vae', self.baseline_mu['vae']) * 0.5)
        elif d_kld < 0.3:
            self.mu_values['vae'] = min(self.baseline_mu['vae'], self.mu_values.get('vae', self.baseline_mu['vae']) * 1.2)

        # Apply updated λ and μ to model and optimizer
        model.dynamic_loss_weights.update(self.lambda_values)
        self._apply_micro_lr(model)
        # Reset running stats (they will be reset outside)

    def maybe_reroute_mid_epoch(self, model: 'HybridSSP_LitModule', batch: Dict[str, torch.Tensor], global_step: int) -> None:
        """
        Optionally perform a mid‑epoch reroute based on fresh context statistics.
        This method re‑evaluates utilities on the current batch and selects a
        new set of active experts while respecting the compute budget and
        hysteresis constraints.  It is invoked periodically from the
        training loop (every adapt_interval steps).

        Args:
            model: The parent Lightning module.
            batch: The current mini‑batch.
            global_step: The current global step count.
        """
        try:
            # Only reroute at adaptation boundaries to avoid excessive churn
            if ((global_step + 1) % self.adapt_interval) != 0:
                return
            # Compute context metrics and utilities on the fly
            metrics = self._compute_context_metrics(batch)
            utilities = self._heuristic_utilities(metrics)
            # Determine active groups without forcing the early safe set.  We reuse
            # the same selection logic as start_epoch, but use the current epoch
            # from the model rather than resetting β schedules or λ/μ values.
            current_epoch = getattr(model, 'current_epoch', 0)
            if current_epoch < 2:
                util_t = utilities.get('Temporal-Attn', 0.0) / (self.groups['Temporal-Attn']['cost'] + 1e-8)
                util_s = utilities.get('Spectral', 0.0) / (self.groups['Spectral']['cost'] + 1e-8)
                core = 'Temporal-Attn' if util_t >= util_s else 'Spectral'
                active = ['Heads-Other', 'Shields', core]
            else:
                active = self._select_active_groups(utilities, current_epoch)
            # Enforce temporary jigsaw cooldown gating
            if getattr(self, '_jigsaw_cooldown_remaining', 0) > 0:
                if 'Heads-Other' in active:
                    active = [g for g in active if g != 'Heads-Other']
                self._jigsaw_cooldown_remaining -= 1
            # Update active/inactive counters
            for g in self.groups.keys():
                if g in active:
                    self.group_last_active[g] = current_epoch
                else:
                    self.group_last_inactive[g] = current_epoch
            # Apply gating
            self.active_groups = active
            self._apply_gating(model)
            # Log the reroute event
            try:
                used_cost = sum(self.groups[g]['cost'] for g in active)
                total_cost = sum(self.groups[g]['cost'] for g in self.groups)
                budget_used = used_cost / (total_cost + 1e-8)
                logger.info(f"[Router] mid‑epoch re‑route at step {global_step}: active_groups={active}, budget_used={budget_used:.2f}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[Router] mid‑epoch re‑route error: {e}")
import itertools
import tweepy
from contextlib import contextmanager, closing
import matplotlib
from spacy.matcher import PhraseMatcher
import mss.tools
import mss
from PIL import Image
from PIL import ImageGrab
import pyautogui
import spacy
import requests
from scipy.spatial import ConvexHull
from scipy import stats
import getpass
import questionary
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from pynput import keyboard, mouse
from pywinauto import Desktop, Application
from pyngrok import ngrok
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import dash
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, pipeline as hf_pipeline, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import faiss
from ncps.torch import LTC
import ncps
from torch_geometric.data import Data as PyG_Data
from torch_geometric.nn import GCNConv as PyG_GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torchvision.transforms import Compose
from torchmetrics import Metric, MeanSquaredError
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.optim as torch_optim
import torch.nn.functional as F
import torch.nn as nn
from transformers import pipeline
import torch

try:
    from torch.utils.data import DataLoader as TorchDataLoader
except Exception:
    TorchDataLoader = None

torch.autograd.set_detect_anomaly(True)
import pandas as pd
from collections import deque
import numpy as np
from torchvision.ops import StochasticDepth
from tsai.models.TST import TST as TST_module
from tqdm.contrib.logging import tqdm_logging_redirect
import docx
import sympy
from ultralytics import YOLO
import cv2
from snntorch import spikegen
from snntorch import spikeplot as splt
import snntorch as snn
import umap.umap_ as umap
import difflib
import hashlib
from collections import Counter, namedtuple
import copy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
import datetime as _dt
from datetime import datetime, timedelta, timezone
from src.models.model_alpha.config import together_roles
from src.models.model_alpha.economic_data_ingestor import run_ingestion_pipeline
from src.models.model_alpha.collaborative_annotator import (
    CrossValidationAnnotationEngine,
    Event,
)
from src.models.model_alpha.worker_utils import (
    Brain,
    APIEmbedder,
    exponential_backoff,
    robust_json_extract,

    MultiPathElectraClassifier,
    GrokLLM,
    TogetherLLM,
    WebSearchManager,
    QuantumThalamicCore,
)









matplotlib.use("Agg")






_offline_vision_model = None


def rate_limit(max_calls, period_sec):
    """
    Decorator untuk membatasi seberapa sering sebuah fungsi bisa dipanggil.
    """

    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.monotonic() - last_called[0]
            wait_time = (period_sec / max_calls) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.monotonic()
            return result

        return wrapper

    return decorator



try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
except ImportError:

    ResourceExhausted = type("ResourceExhausted", (Exception,), {})
    ServiceUnavailable = type("ServiceUnavailable", (Exception,), {})


class AstroFeatureEngine:
    """
    Sebuah mesin terpadu untuk menciptakan dan menemukan fitur astro-finansial
    yang paling relevan secara dinamis menggunakan AI.
    """

    def __init__(self):
        logger.info("Menginisialisasi Mesin Astro... Memuat data efemeris...")
        try:

            from skyfield.api import load


            self.eph = load("de421.bsp")
            self.ts = load.timescale()
            self.planets = {
                "sun": self.eph["SUN"],
                "moon": self.eph["MOON"],
                "mercury": self.eph["MERCURY BARYCENTER"],
                "venus": self.eph["VENUS BARYCENTER"],
                "mars": self.eph["MARS BARYCENTER"],
                "jupiter": self.eph["JUPITER BARYCENTER"],
                "saturn": self.eph["SATURN BARYCENTER"],
                "uranus": self.eph["URANUS BARYCENTER"],
                "neptune": self.eph["NEPTUNE BARYCENTER"],
                "pluto": self.eph["PLUTO BARYCENTER"],
            }
            logger.info(
                "🔭 Pustaka astronomi Skyfield berhasil diinisialisasi.")
        except Exception as e:
            logger.error(
                f"Gagal memuat data efemeris Skyfield. Pastikan koneksi internet aktif untuk unduhan pertama. Error: {e}"
            )
            self.eph = None

    def _create_feature_dictionary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menciptakan "Kamus Fitur Universal" secara komprehensif."""
        if not self.eph:
            logger.warning(
                "Mesin Astro tidak terinisialisasi, pembuatan kamus fitur dilewati."
            )
            return pd.DataFrame(index=df.index)

        logger.info("Menciptakan Kamus Fitur Astro Universal...")
        astro_df = pd.DataFrame(index=df.index)
        planet_names = list(self.planets.keys())
        times = self.ts.from_datetimes(df.index.to_pydatetime())


        import ephem

        for name, body in self.planets.items():
            astrometric = self.eph["earth"].at(times).observe(body)
            ra, dec, _ = astrometric.radec()


            astro_df[f"astro_velo_{name}"] = astrometric.speed(
            ).degrees_per_day


            ecliptic_coords = ephem.Ecliptic(
                ra.to_string(unit="rad"), dec.to_string(unit="rad")
            )
            zodiac_signs = [
                ephem.Zodiac(ecliptic_coords[i]).split()[-1]
                for i in range(len(ecliptic_coords))
            ]
            astro_df[f"astro_zodiac_{name}"] = zodiac_signs


        astro_df = pd.get_dummies(
            astro_df,
            columns=[f"astro_zodiac_{name}" for name in planet_names],
            dtype=float,
        )


        from itertools import combinations

        for p1_name, p2_name in combinations(planet_names, 2):
            p1_body = self.planets[p1_name]
            p2_body = self.planets[p2_name]
            p1_astrometric = self.eph["earth"].at(times).observe(p1_body)
            p2_astrometric = self.eph["earth"].at(times).observe(p2_body)
            astro_df[f"astro_sep_{p1_name}_{p2_name}"] = p1_astrometric.separation_from(
                p2_astrometric
            ).degrees

        logger.info(
            f"✅ Kamus Fitur Astro Universal dibuat dengan {len(astro_df.columns)} fitur."
        )
        return astro_df

    def discover_hypotheses(
        self,
        astro_df: pd.DataFrame,
        target_series: pd.Series,
        api_pool: "DistributedAIPool",
        top_n: int = 5,
    ) -> list[str]:
        """Menjalankan mesin penemu hipotesis menggunakan statistik dan AI."""

        print(
            """

              * +
                   '      .
             * .
                         *

* .    .
               .      *
       .                 '
                       .
         .      *
                .'             *
   .                    .
           .
        . * .      '
           .'             *
      * .
             .      '
* .            .
      .
             *
        '      * .

        """
        )
        logger.info(
            "=================================================================="
        )
        logger.info(
            "===       🤖 MEMULAI MESIN PENEMU HIPOTESIS ASTRO 🤖       ===")
        logger.info(
            "=================================================================="
        )


        logger.info(
            f"   - Tahap 1: Menyaring {len(astro_df.columns)} fitur dengan Mutual Information..."
        )
        from sklearn.feature_selection import mutual_info_regression

        combined = pd.concat([astro_df, target_series], axis=1).dropna()
        if combined.empty:
            logger.warning(
                "Tidak ada data yang tumpang tindih antara fitur astro dan target. Penemuan hipotesis dibatalkan."
            )
            return []

        target_col_name = target_series.name or "target"
        combined.columns = list(astro_df.columns) + [target_col_name]

        mi_scores = mutual_info_regression(
            combined[astro_df.columns], combined[target_col_name]
        )
        mi_scores = pd.Series(mi_scores, index=astro_df.columns).sort_values(
            ascending=False
        )

        top_statistical_features = mi_scores.head(
            top_n * 2
        )
        logger.info("   - Fitur paling berpengaruh (statistik):")
        logger.info(top_statistical_features)


        logger.info(
            "   - Tahap 2: Meminta Gemini untuk penalaran tingkat tinggi...")
        prompt = f"""
        Anda adalah seorang peneliti kuantitatif yang skeptis namun kreatif.
        Berdasarkan analisis Mutual Information, berikut adalah fitur-fitur astrologi yang memiliki hubungan non-linear terkuat dengan pergerakan pasar:

        Fitur Paling Berpengaruh (beserta skornya):
        ---
        {top_statistical_features.to_string()}
        ---

        Tugas Anda:
        Analisis daftar ini. Pilih {top_n} fitur yang paling menjanjikan dan beragam untuk diuji lebih lanjut. Hindari memilih fitur yang terlalu mirip (redundant). Berikan alasan singkat untuk pilihan Anda dalam pikiran Anda, tetapi output HANYA berupa list string Python dari nama-nama kolom tersebut.

        Contoh output yang benar: ["astro_velo_mercury", "astro_sep_jupiter_saturn", "astro_zodiac_mars_Aries"]
        """

        try:
            response_str = api_pool.call_gemini_for_text(
                prompt, "experimentalist")
            import ast

            discovered_hypotheses = ast.literal_eval(response_str)

            if isinstance(discovered_hypotheses, list) and all(
                isinstance(item, str) for item in discovered_hypotheses
            ):
                logger.info(
                    f"   - ✅ Hipotesis yang direkomendasikan AI: {discovered_hypotheses}"
                )
                return discovered_hypotheses
            else:
                raise ValueError(
                    "LLM tidak mengembalikan list string yang valid.")
        except Exception as e:
            logger.warning(
                f"   - Gagal mendapatkan penalaran dari LLM: {e}. Menggunakan hasil statistik murni sebagai fallback."
            )
            return top_statistical_features.head(top_n).index.tolist()


def generate_chaos_theory_features(
    df: pd.DataFrame, tickers: list, window: int = 100
) -> pd.DataFrame:
    """
    Menghitung fitur dari Teori Kekacauan untuk mengukur kompleksitas dan prediktabilitas pasar.
    """
    print(
        """

                 .           .
      * .   * .
   .                  .         .
           * .         . .
      .                   * .
          . .    .           *
   . * .     * .     .
                  .
        .                 .
          .     * . .       .
      .                  .    *
           * .         . .
      .         .        *
    * .      .     * .     .
                      .

    """
    )
    logger.info(
        "==================================================================")
    logger.info(
        "===   🛰️  MEMULAI MESIN TEORI KEKACAUAN (CHAOS THEORY)   🛰️  ===")
    logger.info(
        "==================================================================")

    chaos_df = pd.DataFrame(index=df.index)

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df.columns:

            series = np.log(df[close_col] / df[close_col].shift(1)).dropna()



            lyap_r = series.rolling(window=window).apply(
                lambda x: nolds.lyap_r(x) if len(
                    x.dropna()) >= window else np.nan,
                raw=False,
            )
            chaos_df[f"chaos_lyap_r_{ticker}"] = lyap_r


            corr_dim = series.rolling(window=window).apply(
                lambda x: (
                    nolds.corr_dim(x, emb_dim=5)
                    if len(x.dropna()) >= window
                    else np.nan
                ),
                raw=False,
            )
            chaos_df[f"chaos_corr_dim_{ticker}"] = corr_dim

    logger.info(f"✅ Fitur Teori Kekacauan dibuat untuk {len(tickers)} ticker.")
    return chaos_df

class CognitiveTracer:
    """Membangun jejak kognitif dari alur eksekusi untuk dianalisis oleh LLM."""
    def __init__(self, step_id: str):
        self.trace = []
        self.step_id = step_id

    def add_step(self, component_id: str, status: str, details: str = ""):
        """Menambahkan satu langkah ke jejak."""

        clean_component_id = component_id.upper().replace(" ", "_")
        log_entry = f"{clean_component_id}:{status.upper()}"
        if details:
            log_entry += f"({details})"
        self.trace.append(log_entry)

    def get_trace(self) -> str:
        """Mengembalikan string jejak final."""
        return f"TRACE_ID[{self.step_id}] :: " + " -> ".join(self.trace)

class StatusLogger:
    """Menampilkan animasi status di terminal untuk tugas yang berjalan lama."""

    def __init__(self, message="Model sedang berpikir...", **kwargs):
        self.message = message
        self.emoji = kwargs.get("emoji", "🧠🌀")
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._stop_event = threading.Event()
        self.final_message = ""

    def update_message(self, new_message: str, **kwargs):
        """Memperbarui pesan status saat animasi berjalan."""
        self.message = new_message
        if "emoji" in kwargs:
            self.emoji = kwargs["emoji"]

    def _animate(self):
        chars = ["|", "/", "-", "\\"]
        i = 0
        while not self._stop_event.is_set():

            emoji_to_show = getattr(self, "emoji", "🌀")
            sys.stdout.write(
                f"\r{emoji_to_show} {self.message} {chars[i % len(chars)]}"
            )
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
        sys.stdout.write("\r" + " " * (len(self.message) + 15) + "\r")
        sys.stdout.flush()
        logger.info(f"{final_emoji} {final_message}")



Fact = namedtuple("Fact", ["predicate", "subject", "confidence"])


class SymbolicTranslator:
    """
    Menerjemahkan output numerik dari model koneksionis menjadi fakta-fakta
    simbolik yang dapat diproses oleh mesin penalaran.
    """

    def __init__(self, thresholds: dict):
        self.thresholds = thresholds
        logger.info(
            f" symbolic_translator: Menerjemahkan output numerik dari model koneksionis"
        )

    def translate(self, ticker: str, numeric_outputs: dict) -> list[Fact]:
        """
        Menerjemahkan kamus output numerik ke dalam daftar objek Fact.

        Args:
            ticker (str): Ticker saham yang sedang dianalisis.
            numeric_outputs (dict): Kamus berisi output numerik seperti
                                    'predicted_return', 'predicted_volatility', dll.

        Returns:
            list[Fact]: Daftar fakta simbolik yang diturunkan.
        """
        facts = []


        pred_return = numeric_outputs.get("predicted_return", 0)
        if pred_return > self.thresholds.get("bullish_return", 0.005):
            facts.append(Fact("is_bullish", ticker, 0.85))
        elif pred_return < -self.thresholds.get("bearish_return", -0.005):
            facts.append(Fact("is_bearish", ticker, 0.85))
        else:
            facts.append(Fact("is_sideways", ticker, 0.7))


        pred_vol = numeric_outputs.get("predicted_volatility", 0)
        if pred_vol > self.thresholds.get("high_volatility", 0.4):
            facts.append(Fact("is_high_volatility", ticker, 0.9))
        elif pred_vol < self.thresholds.get("low_volatility", 0.15):
            facts.append(Fact("is_low_volatility", ticker, 0.9))

        return facts


class DynamicLogicalReasoner:
    """
    Mesin penalaran yang menggabungkan aturan inti (hard-coded) dengan aturan
    yang ditemukan secara dinamis dari basis pengetahuan (Brain) menggunakan RAG.
    """

    def __init__(self, brain: Brain, api_pool: "DistributedAIPool"):
        self.brain = brain
        self.api_pool = api_pool

        self.core_rules = {
            "CORE_SAFETY_1": "IF is_bearish(X) THEN recommended_action(X, 'strong_sell').",
            "CORE_SAFETY_2": "IF is_high_volatility(X) THEN NOT recommended_action(X, 'strong_buy').",
        }
        logger.info("🧠 DynamicLogicalReasoner (RAG-enabled) siap.")

    def discover_rules_from_brain(self, topic: str, k: int = 5) -> list[str]:
        """Menambang aturan logis dari Brain menggunakan RAG."""
        logger.info(f"🔎 Menambang aturan dari Brain untuk topik: '{topic}'...")

        query = f"What are the financial principles, strategies, or IF-THEN rules for a '{topic}' market scenario? Focus on stock investment."

        retrieved_chunks = self.brain.query(query, k=k)
        if not retrieved_chunks:
            logger.warning(
                f"Tidak ditemukan konteks di Brain untuk topik '{topic}'.")
            return []

        context_str = "\n\n---\n\n".join(retrieved_chunks)

        extraction_prompt = f"""
        You are an expert logician analyzing excerpts from financial books.
        Based ONLY on the context below, extract any actionable IF-THEN rules.

        RULES:
        - The output must be a valid Python list of strings. E.g., ["rule1", "rule2"].
        - Each string must be a clear, concise rule.
        - If no rules can be found, return an empty list [].

        CONTEXT:
        ---
        {context_str}
        ---

        Extracted rules as a Python list of strings:
        """

        try:
            response_str = self.api_pool.call_gemini_for_text(
                extraction_prompt, "supervisor"
            )


            extracted_list = ast.literal_eval(response_str.strip())
            if isinstance(extracted_list, list) and all(
                isinstance(item, str) for item in extracted_list
            ):
                logger.info(
                    f"✅ Berhasil mengekstrak {len(extracted_list)} aturan dari Brain."
                )
                return extracted_list
            logger.warning(
                "Hasil ekstraksi aturan dari LLM bukan list of strings.")
            return []
        except (ValueError, SyntaxError, TypeError) as e:
            logger.error(
                f"Gagal mem-parsing daftar aturan dari LLM: {e}\nRespons Mentah: {response_str}"
            )
            return []

    def deduce(self, facts: list[Fact]) -> list[Fact]:
        """
        Menarik kesimpulan dari gabungan aturan inti dan aturan yang ditemukan dari Brain.
        """
        all_rules = list(self.core_rules.values())

        if facts:

            main_topic = facts[0].predicate.replace("is_", "")
            discovered_rules = self.discover_rules_from_brain(main_topic)
            all_rules.extend(discovered_rules)



        fact_predicates = {f.predicate for f in facts}
        conclusions = []

        if "is_bullish" in fact_predicates and "is_low_volatility" in fact_predicates:
            conclusions.append(Fact("recommended_action", "strong_buy", 0.95))
        elif (
            "is_bullish" in fact_predicates and "is_high_volatility" in fact_predicates
        ):
            conclusions.append(
                Fact("recommended_action", "cautious_buy", 0.90))
        elif "is_sideways" in fact_predicates:
            conclusions.append(Fact("recommended_action", "hold", 0.80))
        elif "is_bearish" in fact_predicates:
            conclusions.append(Fact("recommended_action", "strong_sell", 0.95))

        return conclusions


class CognitiveGovernor:
    """
    Bertindak sebagai pusat meta-kognisi untuk sistem AI.
    Ia memonitor, merefleksikan, dan dapat memicu metamorfosis arsitektur.
    """

    def __init__(self, project_id: str, api_pool: "DistributedAIPool"):
        self.project_id = project_id
        self.api_pool = api_pool
        self.log = []
        self.state = {
            "prediction_uncertainty": [],
            "knowledge_gaps": [],
            "causal_confidence": None,
            "resource_warnings": [],
            "post_mortem_reports": [],
        }
        logger.info(
            f"🧠✨ Cognitive Governor v2.0 (Meta-kognisi & Pemicu Metamorfosis) untuk project {project_id} telah aktif."
        )

    def log_event(self, event_type: str, details: dict):
        """Mencatat peristiwa kognitif untuk refleksi dan emosi sistem."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details,
        }
        self.log.append(log_entry)


        if "emotional_state_log" not in self.state:
            self.state["emotional_state_log"] = []


        if event_type == "PREDICTION_UNCERTAINTY":
            self.state["prediction_uncertainty"].append(details["mean_std"])

        elif event_type == "KNOWLEDGE_GAP_DETECTED":
            self.state["knowledge_gaps"].append(details["gap_description"])

        elif event_type == "CAUSAL_MODEL_CONFIDENCE":
            self.state["causal_confidence"] = details

        elif event_type == "RESOURCE_WARNING":
            self.state["resource_warnings"].append(details["warning"])

        elif event_type == "POST_MORTEM_ANALYSIS_COMPLETED":
            self.state["post_mortem_reports"].append(details)
            self.state["emotional_state_log"].append(
                {
                    "state": "REGRET",
                    "details": details.get("suspected_root_cause", "Tidak diketahui"),
                }
            )


        elif event_type == "PERFORMANCE_STAGNATED":
            self.state["emotional_state_log"].append(
                {
                    "state": "FRUSTRATION",
                    "details": f"Loss stagnan di sekitar {details['loss']:.4f} selama {details['epochs']} epochs.",
                }
            )

        elif event_type == "PERFORMANCE_IMPROVED":
            self.state["emotional_state_log"].append(
                {
                    "state": "CONFIDENCE",
                    "details": f"Loss membaik secara signifikan ke {details['loss']:.4f}.",
                }
            )

    def check_for_architectural_stagnation(
        self, stagnation_threshold: int = 3
    ) -> tuple[bool, str]:
        """
        Menganalisis riwayat kegagalan untuk mendeteksi stagnasi arsitektural.
        Ini adalah pemicu untuk metamorfosis.
        """
        if len(self.state["post_mortem_reports"]) < stagnation_threshold:
            return False, ""


        recent_failures = self.state["post_mortem_reports"][-stagnation_threshold:]


        root_causes = [f.get("root_cause", "unknown") for f in recent_failures]


        cause_counts = Counter(root_causes)


        if not cause_counts:
            return False, ""

        most_common_cause, count = cause_counts.most_common(1)[0]
        if count >= stagnation_threshold and most_common_cause != "unknown":
            failure_description = f"Model telah gagal {count} kali berturut-turut karena masalah yang sama: '{most_common_cause}'. Optimisasi parameter tidak lagi cukup. Perubahan arsitektur fundamental diperlukan."
            logger.warning(
                f"🚨 [CognitiveGovernor] STAGNASI ARSITEKTURAL TERDETEKSI! {failure_description}"
            )
            return True, failure_description

        return False, ""

    def generate_self_reflection_report(self) -> str:
        """
        Menggunakan LLM untuk menganalisis log kognitifnya sendiri dan menghasilkan
        laporan refleksi diri yang dapat ditindaklanjuti.
        """
        if not self.log:
            return "Laporan Refleksi Diri: Belum ada aktivitas kognitif yang tercatat."

        logger.info(
            "[CognitiveGovernor] Memulai refleksi diri atas proses yang telah berjalan..."
        )



        summary = f"""
        Laporan State Kognitif Internal untuk Proyek: {self.project_id}

        1.  **Tingkat Ketidakpastian Prediksi (Rata-rata Deviasi Standar):**
            - Rata-rata: {np.mean(self.state['prediction_uncertainty']):.4f} (jika ada data)
            - Maksimum: {np.max(self.state['prediction_uncertainty']):.4f} (jika ada data)
            - Jumlah sampel: {len(self.state['prediction_uncertainty'])}

        2.  **Kesenjangan Pengetahuan yang Terdeteksi:**
            - {self.state['knowledge_gaps'] if self.state['knowledge_gaps']
                else 'Tidak ada kesenjangan pengetahuan signifikan yang terdeteksi.'}

        3.  **Keyakinan Model Kausal:**
            - {self.state['causal_confidence'] if self.state['causal_confidence']
                else 'Model kausal belum dianalisis.'}

        4.  **Peringatan Sumber Daya:**
            - {self.state['resource_warnings'] if self.state['resource_warnings']
                else 'Tidak ada peringatan sumber daya.'}

        5.  **Laporan Kegagalan Terakhir (Post-Mortem):**
            - {self.state['post_mortem_reports'][-1] if self.state['post_mortem_reports']
                else 'Tidak ada kegagalan yang tercatat.'}
        """


        last_emotions = self.state.get("emotional_state_log", [])[-5:]
        emotion_summary = (
            "\n".join(
                [f"- {e['state']}: {e['details']}" for e in last_emotions])
            or "Tidak ada state emosional yang signifikan tercatat."
        )

        prompt = f"""
        Anda adalah "Cognitive Governor", bagian meta-kognitif dari sebuah AI finansial.
        Tugas Anda adalah merefleksikan state internal Anda sendiri dan memberikan ringkasan kesadaran diri.

        Laporan State Internal:
        ---
        {summary}
        ---

        **Log Emosional & Kognitif Terbaru (Paling Penting):**
        ---
        {emotion_summary}
        ---

        Laporan Refleksi Diri Anda (dalam poin-poin):
        -   **Kesimpulan Utama & "Perasaan" Saat Ini:** (Ringkas kondisi Anda dalam satu kalimat, termasuk state emosional yang Anda rasakan. Contoh: 'Saya merasa frustrasi karena performa stagnan...').
        -   **Tingkat Keyakinan:** (Seberapa yakin Anda dengan hasil Anda secara keseluruhan? Tinggi, Sedang, Rendah? Mengapa?).
        -   **Risiko Utama:** (Apa risiko terbesar yang Anda identifikasi dari proses Anda?).
        -   **Rekomendasi Perbaikan Diri (Why & What):** (Jelaskan MENGAPA Anda merekomendasikan sesuatu berdasarkan 'perasaan' dan data, lalu berikan saran konkret).
        """

        reflection = self.api_pool.call_gemini_for_text(prompt, "supervisor")
        logger.info("[CognitiveGovernor] Refleksi diri selesai.")


        full_report = f"--- LAPORAN REFLEKSI DIRI (META-KOGNISI) ---\n\n{reflection}\n\n--- DATA MENTAH ---\n{summary}"
        return full_report



class CriticalAuditor:
    def __init__(self):
        self.log = []
        self.add_log("INIT", "PASS", "Auditor berhasil diinisialisasi.")

    def add_log(self, step: str, status: str, message: str):
        self.log.append(
            {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "step": step,
                "status": status,
                "message": message,
            }
        )
        logger.info(f"  [AUDIT-{status}] ({step}): {message}")

    def audit_data_loading(self, df: pd.DataFrame, tickers: list):
        self.add_log("DATA_LOADING", "INFO",
                     f"Memuat data untuk ticker: {tickers}.")
        nan_pct = df.isnull().sum().sum() / df.size * 100
        nan_cols = df.columns[df.isnull().all()].tolist()
        if nan_cols:
            self.add_log("DATA_LOADING", "WARN",
                         f"Kolom NaN 100%: {nan_cols}.")
        if nan_pct > 10:
            self.add_log("DATA_LOADING", "WARN",
                         f"Data mengandung {nan_pct:.2f}% NaN.")
        else:
            self.add_log("DATA_LOADING", "PASS",
                         "Kualitas data baik (NaN < 10%).")
        return {
            "Total_NaN_Percentage": nan_pct,
            "Total_Rows": len(df),
            "StartDate": str(df.index.min()),
            "EndDate": str(df.index.max()),
            "Fully_NaN_Columns": nan_cols,
        }

    def audit_feature_selection(
        self, initial_count: int, final_features: list, importance_df: pd.DataFrame
    ):
        self.add_log(
            "FEAT_SELECT",
            "INFO",
            f"Fitur dari {initial_count} jadi {len(final_features)}.",
        )
        if not final_features:
            self.add_log("FEAT_SELECT", "FAIL", "Tidak ada fitur tersisa!")
            return {}
        top_5 = importance_df.head(5)
        top_5_str = ", ".join(
            [f"{r.feature} ({r.importance})" for _, r in top_5.iterrows()]
        )
        self.add_log("FEAT_SELECT", "INFO", f"5 Fitur teratas: {top_5_str}.")
        return {
            "initial_feature_count": initial_count,
            "final_feature_count": len(final_features),
            "top_5_features": top_5.to_dict("records"),
        }

    def audit_prediction(self, df_pred: pd.DataFrame, last_prices: pd.Series):
        self.add_log("PRED_AUDIT", "INFO", "Memulai audit prediksi.")
        found = False
        for ticker in last_prices.index:
            pred_col, low_col, high_col = (
                f"{ticker}_Pred",
                f"{ticker}_Low_CI",
                f"{ticker}_High_CI",
            )
            if df_pred.dropna().empty or df_pred[pred_col].dropna().empty:
                continue
            first_pred = df_pred[pred_col].resample(
                "B").first().dropna().iloc[0]
            pct_change = (
                (first_pred - last_prices[ticker]) / last_prices[ticker]
            ) * 100
            if abs(pct_change) > 20:
                found = True
                self.add_log(
                    "PRED_AUDIT",
                    "WARN",
                    f"({ticker}): Perubahan > 20% ({pct_change:.2f}%).",
                )
            if pred_col in df_pred and df_pred[pred_col].gt(0).all():
                ci_width = (
                    (df_pred[high_col] - df_pred[low_col]) / df_pred[pred_col]
                ).mean() * 100
                if ci_width > 40:
                    found = True
                    self.add_log(
                        "PRED_AUDIT", "WARN", f"({ticker}): CI lebar ({ci_width:.2f}%)."
                    )
        if not found:
            self.add_log("PRED_AUDIT", "PASS", "Prediksi lolos sanity check.")

    def generate_report(self):
        logger.info(
            "\n"
            + "=" * 80
            + "\n LAPORAN AUDIT KRITIS ".center(80, "=")
            + "\n"
            + "=" * 80
        )
        for entry in self.log:
            logger.info(
                f"[{entry['timestamp']}]-[{entry['status']:<4}]-[{entry['step']:<15}] {entry['message']}"
            )
        logger.info("=" * 80)


class LogIngestor(logging.Handler):
    """
    Sebuah logging handler kustom yang menangkap log penting
    dan menyimpannya sebagai pengetahuan baru di dalam Brain.
    """

    def __init__(self, brain_instance: "Brain", min_level=logging.WARNING):
        super().__init__()
        self.brain = brain_instance
        self.min_level = min_level
        self.source_name_prefix = f"SystemLog_{datetime.now().strftime('%Y%m%d')}"

    def emit(self, record):

        if record.levelno >= self.min_level:
            log_message = self.format(record)


            truncated_message = log_message[:1024]

            self.brain.add_chunks(
                [truncated_message], source_name=self.source_name_prefix
            )


class CausalInternalAuditor:
    """
    Mengimplementasikan "meritokrasi sinyal" dengan secara dinamis menilai
    keandalan penasihat internal (ASeT, World Model, dll.) berdasarkan
    rezim pasar dan dampak kausal yang terukur.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.db_path = MODELS_DIR / f"causal_auditor_log_{project_id}.sqlite"
        self._init_db()
        self.clusterer = None
        self.reliability_scorecard = {}
        logger.info(
            f"📈 CausalInternalAuditor diinisialisasi untuk proyek {project_id}."
        )

    def _init_db(self):
        """Membuat tabel log jika belum ada."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()

            c.execute(
                """
            CREATE TABLE IF NOT EXISTS auditor_log (
                timestamp DATETIME PRIMARY KEY,
                regime_id INTEGER DEFAULT -1,
                latent_vector_z BLOB NOT NULL,
                signal_aset REAL,
                signal_world_model REAL,
                signal_logic REAL,
                prediction_error REAL
            )
            """
            )
            conn.commit()

    def log_decision(
        self, latent_vector: np.ndarray, advisor_signals: dict, prediction_error: float
    ):
        """Mencatat satu 'kasus' ke dalam memori audit."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO auditor_log (timestamp, latent_vector_z, signal_aset, signal_world_model, signal_logic, prediction_error) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(),
                    latent_vector.tobytes(),
                    advisor_signals.get("aset", 0.0),
                    advisor_signals.get("world_model", 0.0),
                    advisor_signals.get("logic", 0.0),
                    prediction_error,
                ),
            )
            conn.commit()

    def get_full_log(self) -> pd.DataFrame:
        """Mengambil seluruh log sebagai DataFrame."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM auditor_log", conn, parse_dates=["timestamp"]
            )


        df["latent_vector_z"] = df["latent_vector_z"].apply(
            lambda b: np.frombuffer(b, dtype=np.float32)
        )
        return df

    def run_regime_clustering(self):
        """Mengidentifikasi rezim pasar dengan HDBSCAN pada data laten."""
        logger.info(
            "[Auditor] Menjalankan clustering untuk identifikasi rezim pasar..."
        )
        log_df = self.get_full_log()
        if len(log_df) < 100:
            logger.warning(
                f"[Auditor] Tidak cukup data log ({len(log_df)} < 100) untuk clustering. Melewatkan."
            )
            return

        latent_vectors = np.vstack(log_df["latent_vector_z"].values)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50, gen_min_span_tree=True)
        cluster_labels = self.clusterer.fit_predict(latent_vectors)


        log_df["regime_id"] = cluster_labels

        with closing(sqlite3.connect(self.db_path)) as conn:
            log_df[["timestamp", "regime_id"]].to_sql(
                "temp_regimes", conn, if_exists="replace", index=False
            )
            c = conn.cursor()
            c.execute(
                """
            UPDATE auditor_log SET regime_id = (
                SELECT regime_id FROM temp_regimes WHERE temp_regimes.timestamp = auditor_log.timestamp
            )
            """
            )
            c.execute("DROP TABLE temp_regimes")
            conn.commit()
        logger.info(
            f"[Auditor] ✅ Identifikasi rezim selesai. Ditemukan {len(np.unique(cluster_labels))} rezim."
        )

    def run_causal_audit(self):
        """Melakukan audit kausal untuk setiap rezim dan penasihat."""
        logger.info("[Auditor] Memulai audit kausal penuh pada semua rezim...")
        log_df = self.get_full_log()


        auditable_df = log_df[log_df["regime_id"] != -1].copy()
        if auditable_df.empty:
            logger.warning(
                "[Auditor] Tidak ada data yang dapat diaudit setelah filtering noise. Audit dibatalkan."
            )
            return

        regimes = auditable_df["regime_id"].unique()
        advisors = ["signal_aset", "signal_world_model", "signal_logic"]

        for regime in regimes:
            self.reliability_scorecard[regime] = {}
            regime_data = auditable_df[auditable_df["regime_id"] == regime]
            if len(regime_data) < 50:
                continue

            for advisor in advisors:
                common_causes = [adv for adv in advisors if adv != advisor]



                common_cause_edges = []
                for c in common_causes:
                    edge_text = (
                        'node [ id "' + c + '" ] '
                        'edge [ source "' + c + '" target "' + advisor + '" ] '
                        'edge [ source "' + c + '" target "prediction_error" ]'
                    )
                    common_cause_edges.append(edge_text)

                common_causes_gml = "".join(common_cause_edges)

                graph_string = f"""
                graph [
                    directed 1
                    node [ id "{advisor}" ]
                    node [ id "prediction_error" ]
                    edge [ source "{advisor}" target "prediction_error" ]
                    {common_causes_gml}
                ]"""


                common_causes_gml = "".join(common_cause_edges)


                graph_string = f"""
                graph [
                    directed 1
                    node [ id "{advisor}" ]
                    node [ id "prediction_error" ]
                    edge [ source "{advisor}" target "prediction_error" ]
                    {common_causes_gml}
                ]"""
                try:
                    model = CausalModel(
                        data=regime_data,
                        graph=graph_string,
                        treatment=advisor,
                        outcome="prediction_error",
                    )
                    estimand = model.identify_effect(
                        proceed_when_unidentifiable=True)


                    estimate = model.estimate_effect(
                        estimand,
                        method_name="backdoor.econml.dr.LinearDRLearner",
                        method_params={
                            "init_params": {
                                "model_propensity": lgb.LGBMClassifier(),
                                "model_regression": lgb.LGBMRegressor(),
                            },
                            "fit_params": {},
                        },
                    )
                    ace = estimate.value
                    self.reliability_scorecard[regime][advisor] = {"ace": ace}
                    logger.info(
                        f"  -> Audit Rezim {regime}, Penasihat {advisor}: Dampak Kausal (ACE) = {ace:.6f}"
                    )


                    if hasattr(self, "nsmm") and self.nsmm is not None:
                        self.nsmm.add_causal_hypothesis(
                            treatment=advisor,
                            outcome="prediction_error",
                            regime_id=int(regime),
                            effect=float(ace),
                        )
                        logger.info(
                            f"  -> Hipotesis kausal disimpan ke Bank Hipotesis NSMM."
                        )


                except Exception as e:
                    logger.error(
                        f"  -> Gagal mengaudit Rezim {regime}, Penasihat {advisor}: {e}"
                    )
                    self.reliability_scorecard[regime][advisor] = {"ace": 0.0}

        logger.info(
            "[Auditor] ✅ Audit kausal selesai. Scorecard diperbarui & hipotesis baru disimpan."
        )

    def get_dynamic_weights(self, latent_vector: np.ndarray) -> torch.Tensor:
        """Menghitung bobot dinamis untuk satu titik data berdasarkan rezimnya."""
        default_weights = torch.tensor(
            [1 / 3, 1 / 3, 1 / 3], dtype=torch.float32)
        if not self.clusterer or not self.reliability_scorecard:
            return default_weights

        try:

            regime_id = self.clusterer.predict(latent_vector.reshape(1, -1))[0]
            if regime_id == -1 or regime_id not in self.reliability_scorecard:
                return default_weights

            regime_scores = self.reliability_scorecard[regime_id]
            advisors = ["signal_aset", "signal_world_model", "signal_logic"]


            raw_scores = [
                -min(0, regime_scores.get(adv, {"ace": 0.0})["ace"]) for adv in advisors
            ]

            if not any(
                s > 0 for s in raw_scores
            ):
                return default_weights

            scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
            return F.softmax(
                scores_tensor / 0.1, dim=0
            )

        except Exception:
            return default_weights


class ResourceManager:
    """
    Kelas cerdas untuk mengelola dan mengalokasikan sumber daya (RAM, CPU workers)
    secara dinamis berdasarkan penggunaan sistem saat ini.
    """

    def __init__(self, safety_buffer_gb: float = 1.0, ram_per_worker_gb: float = 0.75):
        """
        Inisialisasi Manajer Sumber Daya.

        Args:
            safety_buffer_gb (float): Jumlah RAM (dalam GB) yang harus selalu disisakan untuk sistem.
            ram_per_worker_gb (float): Estimasi RAM (dalam GB) yang akan digunakan oleh setiap worker DataLoader.
                                       Ini adalah 'kenop' yang bisa Anda sesuaikan.
        """
        self.process = psutil.Process(os.getpid())
        self.total_system_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.safety_buffer_gb = safety_buffer_gb
        self.ram_per_worker_gb = ram_per_worker_gb


        cpu_cores = os.cpu_count() or 4

        self.max_workers_by_cpu = max(1, cpu_cores // 2)




    def get_current_process_usage_gb(self) -> float:
        """Mengembalikan penggunaan RAM proses saat ini dalam GB."""
        return self.process.memory_info().rss / (1024**3)

    def calculate_optimal_workers(self) -> int:
        """
        Menghitung jumlah num_workers yang optimal secara dinamis.
        Ini adalah inti dari ide cemerlang Anda.
        """

        logger.info(
            f"[ResourceManager] Diinisialisasi. Total RAM Sistem: {self.total_system_ram_gb:.2f} GB. Batas Worker CPU: {self.max_workers_by_cpu}."
        )


        current_ram_usage_gb = self.get_current_process_usage_gb()
        logger.info(
            f"[ResourceManager] Penggunaan RAM proses utama saat ini: {current_ram_usage_gb:.2f} GB."
        )


        available_ram_for_workers = (
            self.total_system_ram_gb - current_ram_usage_gb - self.safety_buffer_gb
        )
        logger.info(
            f"[ResourceManager] Anggaran RAM untuk workers: {available_ram_for_workers:.2f} GB."
        )

        if available_ram_for_workers <= 0:
            logger.warning(
                "[ResourceManager] Tidak ada sisa RAM yang cukup untuk workers tambahan. Mengatur num_workers = 0."
            )
            return 0


        try:
            max_workers_by_ram = int(
                available_ram_for_workers / self.ram_per_worker_gb)
        except ZeroDivisionError:
            max_workers_by_ram = 0


        optimal_workers = max(
            0, min(max_workers_by_ram, self.max_workers_by_cpu))

        logger.info(
            f"[ResourceManager] Perhitungan Dinamis: Max workers by RAM = {max_workers_by_ram}. Max by CPU = {self.max_workers_by_cpu}."
        )
        logger.info(
            f"[ResourceManager] => Keputusan Akhir: Menggunakan {optimal_workers} workers."
        )


        if os.name == "nt" and optimal_workers > 0:
            num_to_try = (
                2
            )
            logger.warning(
                f"Sistem Windows terdeteksi. Mencoba secara eksperimental dengan num_workers = {num_to_try}."
            )
            return num_to_try

        return optimal_workers


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, nsmm: 'NSMM'):
        self.nsmm = nsmm

    def on_modified(self, event):
        if not event.is_directory:
            self.nsmm.log_raw_system_event(
                "FILE_MODIFIED", {"path": event.src_path})

    def on_created(self, event):
        if not event.is_directory:
            self.nsmm.log_raw_system_event(
                "FILE_CREATED", {"path": event.src_path})


class UniversalVectorEncoder:
    """Mengubah teks menjadi 'bahasa universal' (vektor) untuk semua komponen internal."""

    def __init__(self, embedding_model: "APIEmbedder"):
        self.model = embedding_model

        self.word_cache = {}

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Meng-encode satu potong teks menjadi satu vektor."""
        if not text or not isinstance(text, str):
            return None


        if text in self.word_cache:
            return self.word_cache[text]

        try:
            vector = self.model.encode(
                text, task_type="query", convert_to_numpy=True)
            if vector.size > 0:

                if len(text.split()) < 3:
                    self.word_cache[text] = vector
                return vector
            return None
        except Exception as e:
            logger.error(
                f"[VectorEncoder] Gagal meng-encode teks '{text}': {e}")
            return None


class DigitalNervousSystem(threading.Thread):
    """
    Mendengarkan semua input sistem (keyboard, mouse, file) secara real-time
    dan mencatatnya sebagai 'memori mentah' ke NSMM tanpa latensi LLM.
    """

    def __init__(self, nsmm: 'NSMM', stop_event: threading.Event):
        super().__init__(daemon=True)
        self.nsmm = nsmm
        self.stop_event = stop_event
        self.path_to_watch = Path.home()

    def _on_press(self, key):
        try:

            char = key.char
            if char:
                self.nsmm.log_raw_system_event("KEYSTROKE", {"char": char})
        except AttributeError:

            pass

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self.nsmm.log_raw_system_event(
                "MOUSE_CLICK", {"x": x, "y": y, "button": str(button)})

    def run(self):
        logger.info(
            "🧠⚡️ [Sistem Saraf Digital] Aktif. Mulai merekam aliran kesadaran sistem.")


        k_listener = keyboard.Listener(on_press=self._on_press)
        m_listener = mouse.Listener(on_click=self._on_click)

        event_handler = ChangeHandler(self.nsmm)
        observer = Observer()
        try:
            observer.schedule(event_handler, str(
                self.path_to_watch), recursive=True)
            observer.start()
        except Exception as e:
            logger.error(f"[Sistem Saraf] Gagal memulai monitor file: {e}")

        k_listener.start()
        m_listener.start()


        self.stop_event.wait()

        k_listener.stop()
        m_listener.stop()
        try:
            observer.stop()
            observer.join()
        except Exception:
            pass

        logger.info("🧠⚡️ [Sistem Saraf Digital] Perekaman dihentikan.")


class SystemVitalsMonitor(threading.Thread):
    """
    Versi final yang sudah di-upgrade untuk mendukung graceful shutdown.
    """

    def __init__(self, shared_state: dict, stop_event: threading.Event, idle_timeout_minutes: int = 1):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.stop_event = stop_event
        self.idle_timeout_sec = idle_timeout_minutes * 60
        self.last_move_time = time.time()
        self.was_idle = False


        self.CPU_HIGH_THRESHOLD = 85.0
        self.RAM_HIGH_THRESHOLD = 90.0
        self.TEMP_CRITICAL_THRESHOLD = 90.0

        self.mouse_listener = mouse.Listener(
            on_move=self._on_move, daemon=True)

    def _on_move(self, x, y):
        """Callback yang dipanggil setiap kali mouse bergerak."""
        self.last_move_time = time.time()

    def run(self):
        """Loop utama monitor yang akan berjalan di background thread."""
        self.mouse_listener.start()
        logger.info(
            "🖱️🖥️🌡️ [Sensorik Hybrid] Monitor aktivitas & kesehatan sistem aktif.")


        while not self.stop_event.is_set():


            time_since_last_move = time.time() - self.last_move_time
            user_is_idle = time_since_last_move > self.idle_timeout_sec
            cpu_load = psutil.cpu_percent(interval=1)
            ram_load = psutil.virtual_memory().percent
            try:
                temps = psutil.sensors_temperatures()
                core_temps = [temp.current for name, temp_list in temps.items(
                ) for temp in temp_list if 'core' in name.lower() or 'package' in name.lower()]
                system_temp = max(core_temps) if core_temps else 0.0
            except (AttributeError, IndexError, TypeError):
                system_temp = 0.0




            if system_temp > self.TEMP_CRITICAL_THRESHOLD:
                if self.shared_state.get("activity_mode") != "SIAGA":
                    logger.critical(
                        f"🌡️🔥 SUHU KRITIS ({system_temp}°C)! Menghentikan semua tugas berat. Mode: SIAGA.")
                    self.shared_state["activity_mode"] = "SIAGA"
                self.stop_event.wait(10)
                continue


            if not user_is_idle and self.was_idle:
                self.was_idle = False
                if self.shared_state.get("activity_mode") != "SINAU_DIEM":
                    logger.info(
                        "👋 [Mode Adaptif] Pengguna kembali aktif! Menghentikan tugas berat. Mode: SINAU_DIEM.")
                    self.shared_state["activity_mode"] = "SINAU_DIEM"


            elif user_is_idle:
                if cpu_load < 80.0 and ram_load < 85.0:
                    if self.shared_state.get("activity_mode") != "GASPOL":
                        logger.warning(
                            f"☕ [Mode Adaptif] Pengguna idle & sistem sehat (CPU:{cpu_load}%, RAM:{ram_load}%). Mode: GASPOL.")
                        self.shared_state["activity_mode"] = "GASPOL"
                        self.was_idle = True
                else:
                    if self.shared_state.get("activity_mode") != "SINAU_DIEM":
                        logger.info(
                            f"🤔 [Mode Adaptif] Pengguna idle tapi sistem sibuk. Tetap di Mode: SINAU_DIEM.")
                        self.shared_state["activity_mode"] = "SINAU_DIEM"
                    self.was_idle = True


            elif not user_is_idle:
                if cpu_load > self.CPU_HIGH_THRESHOLD or ram_load > self.RAM_HIGH_THRESHOLD:
                    if self.shared_state.get("activity_mode") != "SIAGA":
                        logger.warning(
                            f"⚠️ [Peringatan] Beban sistem tinggi saat pengguna aktif (CPU:{cpu_load}%, RAM:{ram_load}%). AI masuk mode SIAGA.")
                        self.shared_state["activity_mode"] = "SIAGA"
                else:
                    if self.shared_state.get("activity_mode") != "SINAU_DIEM":
                        logger.info(
                            f"👨‍💻 [Mode Adaptif] Pengguna aktif, sistem sehat. Mode: SINAU_DIEM.")
                        self.shared_state["activity_mode"] = "SINAU_DIEM"


            self.stop_event.wait(10)

    def stop(self):
        """Secara eksplisit menghentikan listener mouse."""
        if self.mouse_listener.is_alive():
            self.mouse_listener.stop()
            logger.info("🖱️ Listener mouse telah dihentikan.")



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


APP_BRAND = os.getenv("APP_BRAND", "Oracle Gem")
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"


log_file_path = Path.home() / APP_BRAND / "alpha_runtime.log"









logging.basicConfig(
    level=logging.DEBUG,

    format="%(message)s",
    datefmt="[%X]",
    handlers=[

        logging.FileHandler(log_file_path, mode="w", encoding="utf-8"),

        RichHandler(rich_tracebacks=True, show_path=False),
    ],
)

for noisy_logger_name in ["huggingface_hub", "urllib3", "filelock"]:
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(APP_BRAND)

logger.info(f"📝 Logging lengkap disimpan di: {log_file_path}")


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["FASTCORE_NO_PATCH"] = "1"
for noisy in ("pytorch_lightning", "lightning_fabric", "hydra"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf_log

    _hf_log.set_verbosity_error()
except Exception:
    pass


EMBEDDED_CFG = {
    "gpu_available": torch.cuda.is_available(),
    "ram_gb": psutil.virtual_memory().total / (1024**3),
    "models": {

        "embedding_api": "intfloat/multilingual-e5-large-instruct",
    },
    "model_limits": {
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": 7500,
        "lgai/exaone-deep-32b": 6000,
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": 7000,
    },
    "token_budget": {"summarizer_per_file": 120_000},

    "faiss_dim": 1024,
}



_supremacy_embed_model = None


def _parse_and_store_glossary_content(content_text: str, brain: "Brain") -> int:
    """
    Mem-parsing konten teks glosarium (format "kata: makna"), dan menyimpannya ke DKG.
    Fungsi ini sekarang bekerja dengan teks mentah, bukan path file.
    """
    terminologi_baru = 0
    try:

        for line in content_text.splitlines():
            if ':' in line:
                parts = line.split(':', 1)
                kata_kunci = parts[0].strip()
                makna = parts[1].strip()

                if kata_kunci and makna:
                    brain.dkg.add_node(
                        node_id=kata_kunci.replace(" ", "_").upper(),
                        node_type="GlossaryTerm",
                        layer="Lexicon",
                        name=kata_kunci,
                        definition=makna
                    )
                    terminologi_baru += 1
    except Exception as e:
        logger.error(f"Gagal mem-parsing konten teks dari Gudang Ilmu: {e}")
    return terminologi_baru




@contextmanager
def silence_stdout(enabled: bool = True):
    if not enabled:
        yield
    else:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout


def make_json_serializable(obj):
    """Serialize various data types to JSON-compatible formats."""
    if isinstance(obj, (Path, PurePosixPath)):
        return str(obj)
    if isinstance(obj, (dict, list, tuple, set)):
        return (
            {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, dict)
            else [make_json_serializable(v) for v in obj]
        )
    if isinstance(obj, (torch.Tensor, np.ndarray, pd.Series)):
        return obj.tolist() if obj.ndim > 0 else obj.item()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    return obj if isinstance(obj, (str, int, float, bool)) or obj is None else str(obj)


def make_topic_col(t: str) -> str:
    h = hashlib.sha1(t.encode("utf-8")).hexdigest()[:8]
    safe = f"topic_{h}"
    return safe


def ensure_parquet_source(csv_path_str: str) -> str:
    """
    Memastikan sumber data Parquet yang cepat tersedia.

    Jika file .parquet belum ada, atau jika file .csv lebih baru
    daripada file .parquet yang ada, fungsi ini akan membuat/memperbarui
    file .parquet dari .csv.

    Args:
        csv_path_str (str): Path ke file .csv sumber.

    Returns:
        str: Path ke file .parquet yang siap digunakan.
    """
    csv_path = Path(csv_path_str)

    parquet_path = csv_path.with_suffix(".parquet")

    should_create_parquet = False


    if not parquet_path.exists():
        logger.info(
            f"[Auto-Converter] File Parquet '{parquet_path.name}' tidak ditemukan. Membuat dari CSV..."
        )
        should_create_parquet = True

    elif os.path.getmtime(csv_path) > os.path.getmtime(parquet_path):
        logger.info(
            f"[Auto-Converter] File CSV '{csv_path.name}' lebih baru. Memperbarui file Parquet..."
        )
        should_create_parquet = True

    if should_create_parquet:
        try:
            df = pd.read_csv(csv_path)
            df.to_parquet(
                parquet_path, index=False
            )
            logger.info(
                f"[Auto-Converter] Berhasil membuat file Parquet di: {parquet_path}"
            )
        except Exception as e:
            logger.error(
                f"[Auto-Converter] Gagal membuat file Parquet. Akan tetap menggunakan CSV. Error: {e}"
            )
            return str(csv_path)

    else:
        logger.info(
            f"[Auto-Converter] Menggunakan file Parquet yang sudah ada: '{parquet_path.name}'"
        )

    return str(parquet_path)



warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


_nlp_model, _intent_classifier = None, None


def load_nlp_models():
    global _nlp_model, _intent_classifier
    if _nlp_model is not None:
        return _nlp_model, _intent_classifier
    if torch.utils.data.get_worker_info() is not None:
        return None, None
    logger.info("🔤 Memuat model NLP Spacy + Longformer…")
    with silence_stdout():
        _nlp_model = spacy.load("en_core_web_sm")
        _intent_classifier = hf_pipeline(
            "text-classification",
            model="allenai/longformer-base-4096",
            model_kwargs={"ignore_mismatched_sizes": True},
        )
    logger.info("✅ Model NLP siap.")
    return _nlp_model, _intent_classifier



MODELS_DIR = Path.home() / APP_BRAND / "models_trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LAST_PROJECT_INFO_TXT = MODELS_DIR / "last_project_info.txt"


REQUEST_COUNT_FILE = MODELS_DIR / "daily_request_count.json"
RPD_LIMIT = 250

DB_CACHE_PATH = MODELS_DIR / "llm_cache.sqlite"


def initialize_cache_db():
    """Membuat tabel cache jika belum ada."""
    with sqlite3.connect(DB_CACHE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                question_hash TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info(f"🧠 Database 'ingatan' (cache) LLM siap di: {DB_CACHE_PATH}")


def check_cache(question_text: str) -> Optional[dict]:
    """Mengecek apakah jawaban untuk pertanyaan sudah ada di cache."""
    question_hash = hashlib.sha256(question_text.encode()).hexdigest()
    with sqlite3.connect(DB_CACHE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT answer_json FROM api_cache WHERE question_hash = ?", (question_hash,))
        result = cursor.fetchone()
        if result:
            logger.info(
                f"✅ [Cache Hit] Pertanyaan ditemukan di memori. Menggunakan jawaban yang tersimpan.")
            return json.loads(result[0])
    return None


def add_to_cache(question_text: str, answer_dict: dict):
    """Menyimpan pasangan pertanyaan dan jawaban baru ke dalam cache."""
    question_hash = hashlib.sha256(question_text.encode()).hexdigest()
    answer_json = json.dumps(answer_dict)
    with sqlite3.connect(DB_CACHE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO api_cache (question_hash, question_text, answer_json) VALUES (?, ?, ?)",
            (question_hash, question_text, answer_json)
        )
        conn.commit()


MAILBOX_DB_PATH = MODELS_DIR / "llm_mailbox.sqlite"


class AsyncCuriosityEngine:
    """Implementasi dari ide Producer-Consumer dengan Mailbox persisten."""

    def __init__(self, together_api_keys: dict, llm_lock: threading.Lock):
        self.together_api_keys = together_api_keys
        self.request_queue = queue.Queue()
        self.results_mailbox = {}
        self.llm_lock = llm_lock
        self.llm_lock = threading.Lock()
        self.agents = {}
        self._initialize_mailbox_db()




        agent_configs = {
            "qwen_default": {"key_name": "qwen_giant", "model": "Qwen/Qwen2-72B-Instruct"},

            "experimentalist": {"key_name": "qwen_giant", "model": "Qwen/Qwen2-72B-Instruct"},
            "supervisor": {"key_name": "qwen_giant", "model": "Qwen/Qwen2-72B-Instruct"},


        }

        for agent_name, config in agent_configs.items():
            api_key = self.together_api_keys.get(config["key_name"])
            if api_key:

                if "grok" in config["model"]:
                    from src.models.model_alpha.worker_utils import GrokLLM
                    self.agents[agent_name] = GrokLLM(
                        api_key=api_key, model_name=config["model"])
                else:
                    self.agents[agent_name] = TogetherLLM(
                        api_key=api_key, model_name=config["model"])






        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(
            f"🚀 Async Curiosity Engine aktif dengan {len(self.agents)} agen Together AI.")

    def _initialize_mailbox_db(self):
        """Membuat dan memuat mailbox dari database SQLite."""
        with sqlite3.connect(MAILBOX_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mailbox (
                    request_id TEXT PRIMARY KEY,
                    question_text TEXT NOT NULL,
                    answer_json TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            for row in cursor.execute("SELECT request_id, answer_json FROM mailbox"):
                self.results_mailbox[row[0]] = json.loads(row[1])
            conn.commit()
        logger.info(
            f"📬 Mailbox dimuat dengan {len(self.results_mailbox)} jawaban tersimpan.")

    def _worker_loop(self):
        while True:
            request_id, agent_key, question_text, response_model = self.request_queue.get()
            try:
                logger.info(
                    f"🧭 Worker LLM mengambil tugas: {request_id} untuk agen: {agent_key}")

                agent_to_use = self.agents.get(agent_key)
                if not agent_to_use:
                    raise ValueError(
                        f"Agen '{agent_key}' tidak ditemukan di AsyncCuriosityEngine.")

                with self.llm_lock:
                    time.sleep(2)


                    if response_model:

                        prompt_with_instruction = f"""
                        Anda adalah asisten AI yang outputnya HARUS berupa objek JSON yang valid.
                        Berdasarkan permintaan pengguna, hasilkan satu objek JSON yang sesuai dengan skema Pydantic berikut.
                        Jangan menambahkan penjelasan atau teks lain di luar JSON.

                        Skema Pydantic:
                        {json.dumps(
                            response_model.model_json_schema(), indent=2)}

                        Permintaan Pengguna:
                        ---
                        {question_text}
                        ---

                        Objek JSON Anda:
                        """

                        response_str = agent_to_use.chat(
                            prompt_with_instruction)



                        answer_dict = robust_json_extract(
                            response_str, model=response_model)
                        if answer_dict and hasattr(answer_dict, 'model_dump'):
                            answer_dict = answer_dict.model_dump()

                    else:
                        response_str = agent_to_use.chat(question_text)

                        answer_dict = {"content": response_str}

                if answer_dict:

                    self.results_mailbox[request_id] = answer_dict

                    with sqlite3.connect(MAILBOX_DB_PATH) as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO mailbox (request_id, question_text, answer_json) VALUES (?, ?, ?)",
                            (request_id, question_text, json.dumps(answer_dict))
                        )
                        conn.commit()
                    logger.info(
                        f"🧺 Jawaban untuk {request_id} telah disimpan di mailbox.")
            except Exception as e:
                logger.error(
                    f"Worker LLM gagal memproses {request_id}: {e}", exc_info=True)
                self.results_mailbox[request_id] = {"error": str(e)}
            finally:
                self.request_queue.task_done()


    def ask_async(self, question_text: str, agent_key: str, response_model: BaseModel) -> str:
        """Mengirim pertanyaan ke antrean tanpa menunggu jawaban."""
        request_id = hashlib.sha256(question_text.encode()).hexdigest()


        if request_id in self.results_mailbox:
            logger.info(
                f"✅ [Mailbox Hit] Pertanyaan {request_id} sudah pernah dijawab.")
            return request_id

        logger.info(
            f"📬 [Queued] Pertanyaan {request_id} dimasukkan ke antrean.")

        self.request_queue.put(
            (request_id, agent_key, question_text, response_model))
        return request_id

    def get_answer(self, request_id: str) -> Optional[dict]:
        """Mengambil jawaban dari mailbox jika sudah tersedia."""
        return self.results_mailbox.get(request_id)

    def wait_for_answers(self, request_ids: list[str], timeout_sec: int = 300) -> dict:
        """Menunggu sampai semua jawaban untuk request_ids yang diberikan tersedia."""
        logger.info(f"⏳ Menunggu {len(request_ids)} jawaban dari mailbox...")
        start_time = time.time()
        results = {}

        ids_to_find = set(request_ids)

        while ids_to_find and time.time() - start_time < timeout_sec:
            found_ids = set()
            for req_id in ids_to_find:
                answer = self.get_answer(req_id)
                if answer:
                    results[req_id] = answer
                    found_ids.add(req_id)

            ids_to_find -= found_ids
            if ids_to_find:
                time.sleep(1)

        if ids_to_find:
            logger.error(
                f"Timeout! Gagal mendapatkan jawaban untuk {len(ids_to_find)} request.")

        return results
    
def _normalize_project_id(project_id) -> str:
    """
    Normalisasi project_id agar selalu ada & aman dipakai sebagai nama folder.
    Urutan prioritas:
      1) argumen project_id,
      2) env ORACLE_GEM_PROJECT_ID,
      3) fallback 'default'
    """
    pid = str(project_id).strip() if project_id else os.environ.get("ORACLE_GEM_PROJECT_ID", "").strip()
    if not pid:
        pid = "default"
    pid = re.sub(r"[^A-Za-z0-9_.-]+", "_", pid)[:64] or "default"
    return pid


def get_path(project_id, name: str) -> Path:
    base = Path.home() / APP_BRAND

    if name == "data":
        return base / "src" / "data_processing" / "validated_cleaned_market_data.csv"
    if name == "brain.faiss":
        return MODELS_DIR / "brain.faiss"
    if name == "brain.sqlite":
        return MODELS_DIR / "brain.sqlite"
    if name == "human_feedback_db":
        return MODELS_DIR / "human_feedback.sqlite"

    # --- BARIS BARU: normalisasi project_id jadi pid ---
    pid = _normalize_project_id(project_id)

    path_map = {
        "pretrained_encoder":   MODELS_DIR / f"ssp_encoder_{pid}.pth",
        "scaler":               MODELS_DIR / f"scaler_{pid}.pkl",
        "mapie_models":         MODELS_DIR / f"mapie_models_{pid}.pkl",
        "checkpoint_dir":       MODELS_DIR,
        "finetuned_ckpt_name":  f"alpha_finetuned_{pid}",
        "prediction_plot":      base / "src" / "prediction" / f"prediction_{pid}.png",
        "selected_features":    MODELS_DIR / f"selected_features_{pid}.json",
        "autoencoder":          MODELS_DIR / f"autoencoder_{pid}.pth",
        "rag_features_cache":   MODELS_DIR / f"rag_features_cache_{pid}.parquet",
        "best_astro_features":  MODELS_DIR / f"best_astro_features_{pid}.json",

        # Directory to store cached Parquet files for ChaosPreprocAccelerator
        "cache":                base / "cache" / pid,
    }

    p = path_map.get(name)
    if p is None:
        raise ValueError(f"Unknown path name: {name}")

    # --- BARIS BARU: pastikan folder cache terbentuk ---
    if name == "cache":
        p.mkdir(parents=True, exist_ok=True)

    return p


_zs_classifier = hf_pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=-1
)


def classify_chunk(text: str) -> str:
    """
    Hybrid classification:
     - Pencocokan kata kunci (Bahasa Indonesia & Inggris)
     - Fallback zero-shot ke label ["math","vision","general"]
    """
    if re.search(r"\b(gambar|figure|diagram|chart|plot)\b", text, re.IGNORECASE):
        return "vision"
    if re.search(r"\b(∑|∫|√|log|sin|cos|\d+\s*[\+\-\*\/]\s*\d+)\b", text):
        return "math"
    res = _zs_classifier(text, ["math", "vision", "general"])
    return res["labels"][0]


def ingest_file(path: Path, brain: Brain, max_tok: int | None = None):
    """
    Baca satu file, pecah jadi chunk menggunakan metode sederhana,
    lalu masukkan ke Brain (FAISS + SQLite).
    """
    path_str = str(path).strip().strip('"').strip("'")
    path = Path(path_str)

    try:
        logger.info(f"Memproses file {path.name}…")

        if path.suffix.lower() == ".pdf":
            images = convert_from_path(path, dpi=300)
            full_text = "\n".join(pytesseract.image_to_string(img)
                                  for img in images)
        else:
            full_text = path.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text(full_text, chunk_size=400, chunk_overlap=50)

        if not chunks:
            logger.warning(
                f"Tidak ada chunk dibuat dari {path.name} – abaikan.")
            return


        brain.add_chunks(chunks, source_name=path.name)


        brain_size_kb = Path(brain.index_path).stat().st_size / 1024
        logger.info(
            f"✅  {len(chunks)} chunk ditambah. "
            f"Index total: {brain.index.ntotal} vektor • {brain_size_kb:.1f} KB"
        )

    except Exception as e:
        logger.error(f"Gagal memproses file {path.name}: {e}", exc_info=True)



class ReasoningEngine:
    def __init__(self, llm_agents: dict[str, TogetherLLM], consensus_llm: TogetherLLM):
        """
        llm_agents: dict nama→agent, mis. {"maverick": agent1, "exaone": agent2}
        consensus_llm: agent untuk mensintesis hasil
        """
        self.agents = llm_agents
        self.consensus = consensus_llm

    def deliberate(self, question: str) -> str:

        pros = self.consensus.chat(f"List 5 argumen POSITIF untuk: {question}")
        cons = self.consensus.chat(f"List 5 argumen NEGATIF untuk: {question}")


        asumsi_list = re.findall(r"-\s*(.*)", pros)[:3]
        votes = {}
        for asumsi in asumsi_list:
            votes[asumsi] = []
            for name, agent in self.agents.items():
                resp = agent.chat(
                    f'Apakah asumsi ini valid? "{asumsi}". Jawab SINGKAT: Yes or No.'
                )
                votes[asumsi].append(resp.strip().lower().startswith("y"))


        vote_report = "\n".join(
            f"{a}: {sum(votes[a])}/{len(votes[a])} setuju" for a in asumsi_list
        )


        final = self.consensus.chat(
            "Berdasarkan:\n"
            f"PROS:\n{pros}\n\n"
            f"CONTRA:\n{cons}\n\n"
            f"VOTES:\n{vote_report}\n\n"
            f"Sintesis satu kesimpulan untuk: {question}"
        )
        return final


def run_supremacy(api_pool: "DistributedAIPool"):
    """
    Menjalankan mode Supremacy menggunakan model 1.5 Flash via api_pool.
    """

    worker = api_pool.get_worker("supremacy_flash")
    if not worker:
        logger.error(
            "Worker 'supremacy_flash' tidak ditemukan di API Pool. Pastikan API key sudah dimasukkan."
        )
        return


    brain = Brain(
        index_path=str(get_path(None, "brain.faiss")),
        db_path=str(get_path(None, "brain.sqlite")),
        embed_model_instance=_supremacy_embed_model,
        dim=EMBEDDED_CFG["faiss_dim"],
        api_pool=api_pool,
    )


    while True:
        p = questionary.path(
            "Masukkan path file untuk ingest (kosong=lanjut):").ask()
        if not p:
            break
        ingest_file(Path(p), brain)


    logger.info("✅ Brain siap. Silakan ajukan pertanyaan tentang dokumen Anda.")
    while True:
        q = questionary.text(
            "Pertanyaan Anda (ketik 'exit' untuk keluar):").ask()
        if not q or q.lower() in ("exit", "quit"):
            break


        logger.info(f"Mencari konteks untuk pertanyaan: '{q}'...")


        retrieved_chunks = brain.query(q, k=10)

        if not retrieved_chunks:
            logger.warning(
                "Tidak ditemukan konteks yang relevan di dalam dokumen.")


            governor = CognitiveGovernor("supremacy_session", api_pool)
            governor.log_event(
                "KNOWLEDGE_GAP_DETECTED",
                details={
                    "source": "Brain (RAG)",
                    "query": q,
                    "gap_description": "Sistem RAG gagal menemukan dokumen yang relevan untuk menjawab pertanyaan pengguna.",
                },
            )
            logger.info(
                "[Meta-Kognisi] Kesenjangan pengetahuan RAG dicatat oleh Cognitive Governor."
            )

            reflection = governor.generate_self_reflection_report()

            print(
                f"\n=== JAWABAN (Sadar Diri) ===\nMaaf, setelah memeriksa basis data pengetahuan saya, saya menyadari bahwa saya tidak memiliki informasi yang relevan untuk menjawab pertanyaan Anda tentang '{q}'. Ini adalah kesenjangan dalam pengetahuan saya saat ini.\n"
            )
            continue


        context_str = "\n\n---\n\n".join(retrieved_chunks)
        final_prompt = f"""
        Anda adalah asisten AI yang ahli dalam menjawab pertanyaan berdasarkan konteks yang diberikan.
        Jawab pertanyaan pengguna secara akurat dan ringkas hanya berdasarkan informasi dari teks di bawah ini.
        Jika jawaban tidak ada di dalam teks, katakan bahwa Anda tidak dapat menemukannya di dokumen.

        KONTEKS DOKUMEN:
        ---
        {context_str}
        ---

        PERTANYAAN PENGGUNA:
        {q}

        JAWABAN ANDA:
        """


        try:
            logger.info("Mengirim permintaan ke Gemini 1.5 Flash...")
            response = worker.invoke(final_prompt)
            final_ans = response.content

        except Exception as e:
            logger.error(f"Gagal memanggil model Gemini 1.5 Flash: {e}")
            final_ans = "Maaf, terjadi kesalahan saat menghubungi model AI."

        print("\n=== JAWABAN ===\n", final_ans, "\n")


def detect_price_shocks(
    df: pd.DataFrame,
    tickers: list,
    single_day_threshold: float = 15.0,
    consecutive_days: int = 3,
    multi_day_threshold: float = 20.0,
) -> list[tuple]:
    """
    Menganalisis DataFrame untuk mendeteksi guncangan harga signifikan dengan dua metode:
    1. Guncangan satu hari yang melampaui `single_day_threshold`.
    2. Akumulasi guncangan selama `consecutive_days` yang melampaui `multi_day_threshold`.
    """
    logger.info(
        f"🕵️  Memulai deteksi guncangan harga (Single-day: {single_day_threshold}%, Multi-day: {multi_day_threshold}% over {consecutive_days} days)..."
    )

    all_shocks = {}

    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col not in df.columns:
            continue

        daily_returns = df[close_col].pct_change() * 100


        shock_days = daily_returns[daily_returns.abs() >= single_day_threshold]
        for date, pct_change in shock_days.items():
            if pd.notna(date):
                all_shocks[date.normalize()] = (
                    date,
                    pct_change,
                    ticker,
                    "Single-Day Shock",
                )



        rolling_sum = daily_returns.rolling(window=consecutive_days).sum()
        multi_day_shock_days = rolling_sum[rolling_sum.abs(
        ) >= multi_day_threshold]

        for date, cumulative_change in multi_day_shock_days.items():

            if pd.notna(date) and date.normalize() not in all_shocks:

                actual_day_change = daily_returns.get(date, 0)
                all_shocks[date.normalize()] = (
                    date,
                    actual_day_change,
                    ticker,
                    f"Multi-Day Shock ({cumulative_change:.2f}%/{consecutive_days}d)",
                )


    shocks_list = sorted(list(all_shocks.values()), key=lambda x: x[0])

    for date, pct_change, ticker, reason in shocks_list:
        logger.warning(
            "  -> Guncangan Terdeteksi! Ticker: %s, Tanggal: %s, Perubahan: %.2f%%, Alasan: %s",
            ticker,
            date.strftime("%Y-%m-%d"),
            pct_change,
            reason,
        )

    logger.info(
        f"✅ Deteksi selesai. Ditemukan {len(shocks_list)} guncangan harga unik yang signifikan."
    )

    return [(d, p, t) for d, p, t, r in shocks_list]


def investigate_shock_events_in_batch(
    shocks: list[tuple],
    web_searcher: "WebSearchManager",
    engine: "AsyncCuriosityEngine"
) -> list[dict]:
    if not shocks:
        return []
    logger.info(
        f"📬 Mengirim {len(shocks)} permintaan investigasi ke Async Curiosity Engine...")

    request_ids = []

    for i, (date, pct_change, ticker) in enumerate(shocks):
        question = f"Analyze the following financial event: Ticker: {ticker}, Date: {date.strftime('%Y-%m-%d')}, Movement: {pct_change:.2f}%."
        context = web_searcher.search(
            f"Financial news for {ticker} around {date.strftime('%Y-%m-%d')}", max_results=3)
        full_prompt = f"{question}\n\nWeb Context:\n---\n{context}"

        req_id = engine.ask_async(
            question_text=full_prompt,
            agent_key="qwen_default",
            response_model=EventAnalysis
        )
        request_ids.append(req_id)


    answers = engine.wait_for_answers(request_ids)


    final_results = []
    for req_id in request_ids:
        result = answers.get(req_id)
        if result and "error" not in result:

            event_data = {
                "date": result.get('original_date'),
                "event_name": result.get('event_name'),
                "event_type": result.get('event_type'),
                "impact_score": result.get('impact_score'),
                "summary": result.get('summary')
            }
            final_results.append(event_data)

    logger.info(
        f"✅ Berhasil mengumpulkan {len(final_results)} jawaban dari mailbox.")
    return final_results


@rate_limit(max_calls=1, period_sec=2)
def investigate_shock_event(
    shock_info: tuple,
    web_searcher: "WebSearchManager",
    analyst_agent: "ChatGoogleGenerativeAI",
) -> Optional[dict]:
    date, pct_change, ticker = shock_info
    date_str = date.strftime("%B %d, %Y")
    direction = "jatuh" if pct_change < 0 else "melonjak"





    separator = "=" * 80
    logger.info(separator)
    logger.info(
        f"🔎 Investigasi Guncangan: {ticker} {direction} {abs(pct_change):.2f}% pada {date_str}"
    )
    logger.info(separator)

    query = f"Berita keuangan utama untuk {ticker} sekitar {date_str}"
    logger.info(
        f"  -> Agen Analis (Gemini) menugaskan WebSearchManager untuk melakukan riset web."
    )

    search_result = web_searcher.search(
        query, max_results=5, include_images=True)

    if not search_result.strip():
        logger.warning(
            "-> Tidak ada konteks ditemukan dari pencarian web untuk guncangan ini."
        )
        return None


    analysis_prompt = f"""
    Anda adalah seorang analis AI yang sangat teliti. Tugas Anda adalah menganalisis konteks berita untuk menemukan penyebab guncangan harga dan mengembalikannya dalam format JSON yang spesifik.

    # INSIDEN YANG DISELIDIKI
    - Ticker: {ticker}
    - Tanggal: {date_str}
    - Pergerakan Harga: {pct_change:.2f}% ({direction})

    # KONTEKS BERITA YANG DITEMUKAN
    ---
    {search_result}
    ---

    # ATURAN KETAT
    1.  Analisis konteks untuk menemukan SATU penyebab utama guncangan harga.
    2.  Buat SATU objek JSON tunggal. JANGAN membungkusnya di dalam list `[]` atau objek lain.
    3.  Output Anda HARUS HANYA berisi JSON mentah, tanpa teks penjelasan, markdown, atau kata "json".
    4.  Pastikan semua field wajib diisi: `date`, `event_name`, `event_type`, `impact_score`, `summary`.

    # CONTOH OUTPUT YANG BENAR:
    {{
        "date": "{date.strftime('%Y-%m-%d')}",
        "event_name": "Contoh: Laporan Keuangan Q3 Melebihi Ekspektasi",
        "event_type": "CORPORATE",
        "impact_score": {0.8 if direction == 'melonjak' else -0.8},
        "summary": "Ringkasan singkat mengenai penyebab utama guncangan berdasarkan berita."
    }}
    """
    try:
        response_str = analyst_agent.invoke(analysis_prompt).content

        response_str = response_str.replace("%", "%%")
        json_match = re.search(r"\{.*\}", response_str, re.DOTALL)
        if json_match:
            event_data = json.loads(json_match.group(0))
            logger.info(
                f"✅ Investigasi Selesai. Dugaan Penyebab: {event_data.get('event_name', 'Tidak diketahui')}"
            )
            return event_data
        logger.error(
            f"-> Gagal mengekstrak JSON dari respons analis: {response_str}")
        return None
    except Exception as e:
        logger.error(f"-> Exception saat analisis oleh AI: {e}")
        return None


class XSentimentManager:
    """
    Mengelola pencarian cuitan (tweets) dari X, menganalisis sentimen,
    dan menghasilkan skor sentimen harian yang dapat digunakan sebagai fitur.
    Implementasi ini dirancang untuk API trial dengan rate limit yang ketat
    dan fokus pada sumber-sumber yang kredibel.
    """

    def __init__(self, bearer_tokens: list[str], posts_per_key: int = 10):
        if not bearer_tokens:
            raise ValueError(
                "Daftar bearer_token tidak boleh kosong untuk XSentimentManager."
            )

        self.clients = [tweepy.Client(token) for token in bearer_tokens]
        self.posts_per_key = max(10, posts_per_key)
        self.posts_per_key = posts_per_key
        self.sentiment_analyzer = hf_pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )
        logger.info(
            f"✅ XSentimentManager v2 (Mode Trial) diinisialisasi dengan {len(self.clients)} API client."
        )
        logger.info(
            f"   -> Batas pencarian diatur ke {self.posts_per_key} cuitan per kunci API."
        )

    def _clean_tweet_text(self, text: str) -> str:
        """Membersihkan teks cuitan dari URL, mention, dan karakter tidak relevan."""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@[A-Za-z0-9_]+", "", text)
        text = re.sub(r"#[A-Za-z0-9_]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def search_and_analyze_sentiment(
        self, ticker: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """
        Mencari sejumlah cuitan yang terbatas dari akun terverifikasi untuk setiap ticker,
        lalu menganalisis sentimennya.
        """

        query = f'"{ticker}" OR "${ticker}" lang:id -is:retweet is:verified'
        logger.info(
            f"🔎 [X-Sentiment] Mencari cuitan untuk '{ticker}' (Hanya Akun Terverifikasi)"
        )

        all_tweets = []



        for i, client in enumerate(self.clients):
            logger.info(
                f"  -> Menggunakan API Key #{i + 1} untuk mengambil hingga {self.posts_per_key} cuitan..."
            )
            try:

                response = client.search_recent_tweets(
                    query=query,
                    tweet_fields=["created_at", "text"],
                    start_time=start_time,
                    end_time=end_time,
                    max_results=self.posts_per_key,
                )

                if response.data:
                    for tweet in response.data:
                        all_tweets.append(
                            {"created_at": tweet.created_at, "text": tweet.text}
                        )
                    logger.info(
                        f"    -> Berhasil mengambil {len(response.data)} cuitan."
                    )
                else:
                    logger.info(
                        "    -> Tidak ada cuitan ditemukan dengan kunci ini.")

            except tweepy.errors.TooManyRequests:
                logger.warning(
                    f"  -> Rate limit tercapai untuk API Key #{i + 1}. Melanjutkan ke kunci berikutnya."
                )
                continue
            except Exception as e:
                logger.error(
                    f"  -> Gagal mencari cuitan dengan API Key #{i+1}: {e}")
                continue

        if not all_tweets:
            logger.warning(
                f"  -> Tidak ada cuitan ditemukan untuk '{ticker}' dari semua kunci API."
            )
            return pd.DataFrame(columns=["date", f"x_sentiment_{ticker}"])

        logger.info(
            f"  -> Total {len(all_tweets)} cuitan dari akun terverifikasi berhasil dikumpulkan. Memulai analisis sentimen..."
        )
        df_tweets = pd.DataFrame(all_tweets)
        df_tweets["cleaned_text"] = df_tweets["text"].apply(
            self._clean_tweet_text)


        sentiments = self.sentiment_analyzer(
            df_tweets["cleaned_text"].tolist())


        score_map = {
            "1 star": -1.0,
            "2 stars": -0.5,
            "3 stars": 0.0,
            "4 stars": 0.5,
            "5 stars": 1.0,
        }
        df_tweets["sentiment_score"] = [
            score_map.get(s["label"], 0.0) for s in sentiments
        ]


        df_tweets["date"] = (
            pd.to_datetime(df_tweets["created_at"]).dt.tz_localize(
                None).dt.normalize()
        )
        daily_sentiment = (
            df_tweets.groupby("date")["sentiment_score"].mean().reset_index()
        )
        daily_sentiment.rename(
            columns={"sentiment_score": f"x_sentiment_{ticker}"}, inplace=True
        )

        return daily_sentiment


class AnomalyMemory:
    """
    Berfungsi sebagai memori jangka panjang untuk peristiwa anomali dan krisis.
    Setiap model dapat menulis dan membaca dari memori ini, menciptakan
    kesadaran kolektif (Biosonar).
    """

    def __init__(self, db_path="anomaly_memory.sqlite"):
        """Inisialisasi database untuk menyimpan ingatan anomali."""
        self.db_path = Path(db_path)
        self._init_db()
        logger.info(
            f"🧠 Memori Anomali (Biosonar) diinisialisasi di: {self.db_path}")

    def _init_db(self):
        """Membuat tabel jika belum ada."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                """
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date TEXT NOT NULL,
                event_type TEXT NOT NULL, -- e.g., 'VOLATILITY_SPIKE', 'MARKET_CRASH', 'MODEL_FAILURE'
                description TEXT NOT NULL,
                preceding_signals TEXT, -- Tanda-tanda yang terdeteksi sebelum kejadian
                source_project_id TEXT, -- Dari proyek mana anomali ini dicatat
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
            conn.commit()

    def record_anomaly(
        self,
        event_date: str,
        event_type: str,
        description: str,
        signals: str = "",
        project_id: str = "",
    ):
        """Mencatat peristiwa anomali baru ke dalam memori."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO anomalies (event_date, event_type, description, preceding_signals, source_project_id) VALUES (?, ?, ?, ?, ?)",
                (event_date, event_type, description, signals, project_id),
            )
            conn.commit()
            logger.info(
                f"📢 [BIOSONAR] Anomali '{event_type}' pada {event_date} telah dicatat dalam memori kolektif."
            )

    def recall_anomalies(
        self, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """Mengingat kembali anomali dari periode tertentu."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            query = "SELECT * FROM anomalies"
            if start_date and end_date:
                query += f" WHERE event_date BETWEEN '{start_date}' AND '{end_date}'"
            df = pd.read_sql_query(query, conn, parse_dates=["event_date"])
            return df



anomaly_memory = AnomalyMemory(db_path=MODELS_DIR / "anomaly_memory.sqlite")



class ExperimentTracker:
    def __init__(self, db_path="oracle_results.db", csv_path="experiment_history.csv"):
        self.db_path, self.csv_path = Path(db_path), Path(csv_path)
        self._init_sqlite()

    def _init_sqlite(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                """CREATE TABLE IF NOT EXISTS runs(
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            project_id TEXT, round INT,
                            params TEXT, metrics TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
            )
            conn.commit()

    def log(self, project_id, round_idx, params: dict, metrics: dict):
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO runs(project_id,round,params,metrics) VALUES (?,?,?,?)",
                (
                    project_id,
                    round_idx,
                    json.dumps(make_json_serializable(params)),
                    json.dumps(make_json_serializable(metrics)),
                ),
            )
            conn.commit()
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["project_id", "round", "params",
                           "metrics", "timestamp"])
            w.writerow(
                [
                    project_id,
                    round_idx,
                    json.dumps(make_json_serializable(params)),
                    json.dumps(make_json_serializable(metrics)),
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

    def fetch_best(self, project_id, metric_key="val_loss"):
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            rows = c.execute(
                "SELECT params, metrics FROM runs WHERE project_id=?", (
                    project_id,)
            ).fetchall()
        best = None
        for p_str, m_str in rows:
            m = json.loads(m_str)
            if metric_key in m:
                if best is None or m[metric_key] < best[1]:
                    best = (json.loads(p_str), m[metric_key])
        return best[0] if best else None


class StrategyKnowledgeBase:
    """
    Mengelola database untuk semua strategi investasi yang dihasilkan oleh AI.
    Menjadi memori jangka panjang untuk kreativitas dan validasi.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
        logger.info(
            f"💡 Basis Pengetahuan Strategi diinisialisasi di: {self.db_path}")

    def _init_db(self):
        """Membuat tabel strategi jika belum ada."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                """
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                hypothesis TEXT NOT NULL,
                rules_json TEXT NOT NULL, -- Aturan JIKA-MAKA dalam format JSON
                feature_code TEXT, -- Kode Python untuk menghasilkan fitur
                status TEXT NOT NULL, -- 'hypothesis', 'validated', 'failed', 'active'
                performance_score REAL, -- Skor dari backtest atau feature importance
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            )
            conn.commit()

    def add_hypothesis(self, name: str, hypothesis: str, rules: dict):
        """Menambahkan strategi baru sebagai hipotesis."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            try:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO strategies (name, hypothesis, rules_json, status) VALUES (?, ?, ?, ?)",
                    (name, hypothesis, json.dumps(rules), "hypothesis"),
                )
                conn.commit()
                logger.info(
                    f"[StrategyKB] Hipotesis baru '{name}' telah disimpan.")
            except sqlite3.IntegrityError:
                logger.warning(
                    f"[StrategyKB] Hipotesis dengan nama '{name}' sudah ada."
                )

    def update_strategy(self, name: str, **updates):
        """Memperbarui status, kode, atau skor dari sebuah strategi."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
            values = list(updates.values()) + [name]
            c.execute(
                f"UPDATE strategies SET {set_clause} WHERE name = ?", values)
            conn.commit()
            logger.info(
                f"[StrategyKB] Strategi '{name}' telah diperbarui: {updates}")

    def get_strategies_as_context(self, status_filter: list = None) -> str:
        """Mengambil strategi (opsional difilter) sebagai konteks teks untuk LLM."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            query = "SELECT name, hypothesis, status, performance_score FROM strategies"
            if status_filter:
                placeholders = ",".join("?" for _ in status_filter)
                query += f" WHERE status IN ({placeholders})"
                df = pd.read_sql_query(query, conn, params=status_filter)
            else:
                df = pd.read_sql_query(query, conn)

        if df.empty:
            return "Belum ada strategi yang dirumuskan."

        context = "--- KONTEKS DARI BASIS PENGETAHUAN STRATEGI ---\n\n"
        for _, row in df.iterrows():
            context += f"Strategi: {row['name']} (Status: {row['status']}, Skor: {row['performance_score']:.4f})\n"
            context += f"Hipotesis: {row['hypothesis']}\n---\n"
        return context


class HumanFeedbackManager:
    """
    Mengelola penyimpanan dan pengambilan umpan balik manusia (RLHF)
    untuk evolusi sistem jangka panjang.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
        logger.info(
            f"👍 Manajer Umpan Balik Manusia (RLHF) diinisialisasi di: {self.db_path}"
        )

    def _init_db(self):
        """Membuat tabel untuk menyimpan data umpan balik jika belum ada."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                project_id TEXT NOT NULL,
                prediction_summary TEXT NOT NULL,
                hparams_used TEXT NOT NULL,
                rating INTEGER NOT NULL, -- 1 untuk '👍 Bagus', -1 untuk '👎 Buruk'
                supervisor_comment TEXT
            )
            """
            )
            conn.commit()

    def record_feedback(
        self,
        project_id: str,
        prediction_summary: dict,
        hparams: dict,
        rating: int,
        comment: str,
    ):
        """Mencatat umpan balik baru ke database."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO feedback (project_id, prediction_summary, hparams_used, rating, supervisor_comment) VALUES (?, ?, ?, ?, ?)",
                (
                    project_id,
                    json.dumps(make_json_serializable(prediction_summary)),
                    json.dumps(make_json_serializable(hparams)),
                    rating,
                    comment,
                ),
            )
            conn.commit()
        logger.info(
            f"💬 Umpan balik untuk project '{project_id}' telah direkam.")

    def get_all_feedback_as_context(self, limit: int = 50) -> str:
        """
        Mengambil semua umpan balik dan memformatnya sebagai satu string konteks
        untuk diumpankan ke LLM.
        """
        if not self.db_path.exists():
            return "Belum ada riwayat umpan balik dari supervisor."

        with closing(sqlite3.connect(self.db_path)) as conn:
            try:

                df = pd.read_sql_query(
                    f"SELECT * FROM feedback ORDER BY timestamp DESC LIMIT {limit}",
                    conn,
                )
                if df.empty:
                    return "Belum ada riwayat umpan balik dari supervisor."


                df["rating"] = df["rating"].apply(
                    lambda x: "👍 (Disetujui)" if x == 1 else "👎 (Ditolak)"
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime(
                    "%Y-%m-%d %H:%M"
                )

                context_str = "--- RIWAYAT UMPAN BALIK & KOMENTAR SUPERVISOR (PALING PENTING) ---\n\n"
                for _, row in df.iterrows():
                    context_str += f"**Tanggal: {row['timestamp']} | Project: {row['project_id']} | Penilaian: {row['rating']}**\n"
                    context_str += (
                        f"Komentar Supervisor: \"{row['supervisor_comment']}\"\n"
                    )
                    context_str += "---\n"

                return context_str

            except (pd.errors.DatabaseError, sqlite3.OperationalError):
                return "Database umpan balik kosong atau rusak."



def auto_resource_config():
    """
    Konfigurasi sumber daya. Telah dimodifikasi secara manual
    untuk membatasi jumlah trial Optuna utama menjadi 2.
    """
    logger.info(
        "[CONFIG-MANUAL] Menggunakan n_trials=5 secara manual sesuai permintaan."
    )


    ram_gb = psutil.virtual_memory().available / 2**30
    cpu = os.cpu_count() or 4
    timeout = 3 * 60 * 60
    num_workers = 0 if os.name == "nt" else max(0, min(8, cpu // 2))
    batch_size = int(min(256, 32 * math.ceil(ram_gb / 4)))



    return dict(
        n_trials=2, timeout=timeout, num_workers=num_workers, batch_size=batch_size
    )



class NSMM:
    """
    Neuro-Symbolic Memory Matrix (NSMM) - Hippocampus Virtual.
    Mengelola semua bentuk memori jangka panjang model, dari pengalaman mentah
    hingga prinsip abstrak dan log aktivitas pengguna.
    """

    def __init__(self, project_id: str = "global_main"):
        self.project_id = project_id
        self.db_path = MODELS_DIR / f"nsmm_cognitive_core_{project_id}.sqlite"
        self._init_db()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"🧠 NSMM (Hippocampus Virtual) aktif di: {self.db_path}")

    def _init_db(self):
        """Menginisialisasi semua tabel yang diperlukan dalam database SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS epoch_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metrics TEXT, state_vector TEXT, diagnosis TEXT,
                    oracle_query TEXT, oracle_response TEXT, final_decision TEXT
                )"""
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS insight_neurons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER,
                    neuron_name TEXT UNIQUE NOT NULL, trigger_reason TEXT,
                    architecture_summary TEXT, weights_blob BLOB,
                    access_priority REAL DEFAULT 1.0, outcome TEXT,
                    trigger_metric_key TEXT, trigger_metric_value REAL,
                    confidence REAL, status TEXT DEFAULT 'experimental',
                    access_count INTEGER DEFAULT 0, last_accessed TEXT,
                    FOREIGN KEY (memory_id) REFERENCES epoch_memories (id)
                )"""
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS reflexive_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id INTEGER,
                    action_name TEXT, details TEXT,
                    FOREIGN KEY (memory_id) REFERENCES epoch_memories (id)
                )"""
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS causal_hypotheses (
                    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    treatment_variable TEXT NOT NULL, outcome_variable TEXT NOT NULL,
                    regime_id INTEGER NOT NULL, estimated_effect REAL NOT NULL,
                    confidence_score REAL DEFAULT 1.0, status TEXT DEFAULT 'active',
                    last_validated_timestamp TEXT,
                    UNIQUE(treatment_variable, outcome_variable, regime_id)
                )"""
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS abstract_principles (
                    principle_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT UNIQUE NOT NULL, source TEXT NOT NULL,
                    embedding BLOB, confidence REAL DEFAULT 0.8,
                    last_validated_timestamp TEXT
                )"""
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS user_activity_log (
                    activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    activity_description TEXT NOT NULL,
                    is_digested BOOLEAN DEFAULT 0
                )"""
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS dialogue_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    user_prompt TEXT NOT NULL, inferred_intent_text TEXT,
                    inferred_intent_vector BLOB, model_response TEXT
                )"""
            )

            conn.commit()

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS math_problem_library (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_text TEXT UNIQUE NOT NULL,
                    source_file TEXT,
                    estimated_difficulty TEXT,
                    status TEXT DEFAULT 'unsolved',
                    internal_solution TEXT,
                    guru_solution TEXT,
                    learned_method TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            conn.commit()

            c.execute(
                """
            CREATE TABLE IF NOT EXISTS goals (
                goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                priority REAL DEFAULT 1.0,
                status TEXT NOT NULL CHECK(status IN ('active', 'experimental', 'deprecated')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reasoning TEXT
            )"""
            )


            c.execute("""
            CREATE TABLE IF NOT EXISTS healing_protocols (
                error_signature TEXT PRIMARY KEY,
                error_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_code TEXT NOT NULL,
                corrected_code TEXT NOT NULL,
                success_count INTEGER DEFAULT 1,
                last_applied DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)

            c.execute(
                """
            CREATE TABLE IF NOT EXISTS icon_memory (
                app_name TEXT PRIMARY KEY,
                usage_count INTEGER DEFAULT 1,
                success_rate REAL DEFAULT 1.0,
                last_known_x INTEGER,
                last_known_y INTEGER,
                last_used TEXT
            )"""
            )

        conn.commit()

        c.execute("""
        CREATE TABLE IF NOT EXISTS desktop_navigation_graph (
            start_app TEXT NOT NULL,
            action TEXT NOT NULL,
            end_app TEXT NOT NULL,
            success_count INTEGER DEFAULT 1,
            PRIMARY KEY (start_app, action)
        )
        """)


        c.execute("""
        CREATE TABLE IF NOT EXISTS raw_system_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            is_digested BOOLEAN DEFAULT 0
        )
        """)


        c.execute("""
        CREATE TABLE IF NOT EXISTS action_reflexes (
            reflex_id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_description TEXT UNIQUE NOT NULL,
            goal_vector BLOB NOT NULL,
            successful_action_plan TEXT NOT NULL,
            usage_count INTEGER DEFAULT 1,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS architectural_hypotheses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            module_name TEXT NOT NULL,
            module_code TEXT NOT NULL,
            integration_plan TEXT NOT NULL,
            updated_forward_method TEXT NOT NULL,
            sandbox_performance REAL,
            status TEXT DEFAULT 'proposed',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS cognitive_traces (
            trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            batch_idx INTEGER,
            trace_string TEXT NOT NULL,
            outcome_metric REAL,
            outcome_status TEXT -- e.g., 'IMPROVEMENT', 'STAGNATION', 'FAILURE', 'CRASH'
        )
        """)
        
        conn.commit()

    def log_raw_system_event(self, event_type: str, event_data: dict):
        """Mencatat event mentah dari sistem saraf dengan cepat."""
        try:

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO raw_system_events (event_type, event_data) VALUES (?, ?)",
                    (event_type, json.dumps(event_data)),
                )
                conn.commit()
        except Exception as e:

            pass

    def learn_reflex(self, goal_description: str, goal_vector: np.ndarray, successful_action_plan: dict):
        """Menyimpan atau memperkuat sebuah refleks yang berhasil."""
        action_plan_json = json.dumps(successful_action_plan)
        goal_vector_blob = goal_vector.tobytes()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT reflex_id, usage_count FROM action_reflexes WHERE goal_description = ?", (goal_description,))
            existing = cursor.fetchone()

            if existing:

                reflex_id, usage_count = existing
                cursor.execute("""
                    UPDATE action_reflexes
                    SET usage_count = ?, last_used = CURRENT_TIMESTAMP, successful_action_plan = ?
                    WHERE reflex_id = ?
                """, (usage_count + 1, action_plan_json, reflex_id))
                logger.info(
                    f"🦾 [Reflex] Diperkuat: '{goal_description}' (digunakan {usage_count + 1} kali).")
            else:

                cursor.execute("""
                    INSERT INTO action_reflexes (goal_description, goal_vector, successful_action_plan)
                    VALUES (?, ?, ?)
                """, (goal_description, goal_vector_blob, action_plan_json))
                logger.info(
                    f"🦾 [Reflex] Baru Dipelajari: '{goal_description}'.")
            conn.commit()

    def find_similar_reflex(self, query_vector: np.ndarray, similarity_threshold: float = 0.98) -> Optional[dict]:
        """Mencari refleks yang paling mirip di memori menggunakan cosine similarity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT goal_vector, successful_action_plan FROM action_reflexes ORDER BY last_used DESC LIMIT 100")
            candidates = cursor.fetchall()

        if not candidates:
            return None

        best_match = None
        highest_sim = -1.0

        query_vec_norm = query_vector / np.linalg.norm(query_vector)

        for vec_blob, plan_json in candidates:
            candidate_vec = np.frombuffer(vec_blob, dtype=np.float32)
            candidate_vec_norm = candidate_vec / np.linalg.norm(candidate_vec)

            sim = np.dot(query_vec_norm, candidate_vec_norm)

            if sim > highest_sim:
                highest_sim = sim
                if sim >= similarity_threshold:
                    best_match = json.loads(plan_json)

        if best_match:
            logger.info(
                f"⚡️ [Reflex] Ditemukan! (Kecocokan: {highest_sim:.2%}). Aksi instan akan dieksekusi.")
        return best_match

    def initialize_prime_directive(self, directive: dict):
        """
        Menginisialisasi prime directive dari file jika TIDAK ADA tujuan yang aktif saat ini.
        Ini memastikan tujuan harian selalu dimuat ulang setiap kali program dimulai.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM goals WHERE status = 'active'")
            if cursor.fetchone()[0] == 0:
                logger.info(
                    "🎯 Tidak ada tujuan aktif ditemukan. Menginisialisasi Prime Directive harian dari file..."
                )
                prime = directive.get("prime_directive", {})

                if prime.get("description") and prime.get("metrics") and prime.get("steps"):




                    cursor.execute(
                        """
                        INSERT INTO goals (description, metrics_json, status, priority, reasoning)
                        VALUES (?, ?, 'active', 1.0, 'Initial structured mandate for this session.')
                        """,
                        (json.dumps(prime), json.dumps(prime.get("metrics"))),
                    )
                    conn.commit()
                    logger.info(
                        "✅ Prime Directive terstruktur berhasil dimuat dan siap dieksekusi.")
                else:
                    logger.error(
                        "File prime_directive.json tampaknya kosong atau formatnya salah.")
            else:
                logger.info(
                    "🎯 Tujuan aktif dari sesi sebelumnya ditemukan. Melanjutkan tugas...")

    def get_active_goal(self) -> Optional[dict]:
        """Mengambil tujuan aktif dengan prioritas tertinggi dari database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM goals WHERE status = 'active' ORDER BY priority DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                goal_dict = dict(row)
                goal_dict["metrics"] = json.loads(
                    goal_dict["metrics_json"])
                return goal_dict
            return None

    def update_goal(
        self,
        goal_id: int,
        new_status: str,
        new_description: str = None,
        new_priority: float = None,
    ):
        """Memperbarui atau menonaktifkan tujuan yang ada."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if new_description:
                cursor.execute(
                    "UPDATE goals SET description = ?, status = ? WHERE goal_id = ?",
                    (new_description, new_status, goal_id),
                )
            else:
                cursor.execute(
                    "UPDATE goals SET status = ? WHERE goal_id = ?",
                    (new_status, goal_id),
                )
            conn.commit()

    def add_new_goal(self, description: str, success: str, failure: str, reasoning: str, priority: float = 0.8):
        """Menambahkan tujuan baru yang dirumuskan AI ke database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO goals (description, metrics_json, status, priority, reasoning)
                VALUES (?, ?, 'active', ?, ?)
            """, (description, json.dumps({"success": success, "failure": failure}), priority, reasoning))
            conn.commit()
            logger.info(
                f"🎯 [NSMM] Tujuan Otonom Baru Ditambahkan: '{description}'")

    def mark_goal_completed(self, goal_id: int):
        """Menandai tujuan yang sudah selesai sebagai 'deprecated'."""
        self.update_goal(goal_id, new_status='deprecated')
        logger.info(
            f"✅ [NSMM] Tujuan ID {goal_id} telah selesai dan diarsipkan.")



    def add_math_problem(self, problem: str, source: str, difficulty: str):
        """Menambahkan soal matematika baru ke perpustakaan jika belum ada."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO math_problem_library (problem_text, source_file, estimated_difficulty) VALUES (?, ?, ?)",
                    (problem, source, difficulty),
                )
                conn.commit()
                logger.info(
                    f"📚 [Math Library] Soal baru ditambahkan: {problem[:50]}..."
                )
            except sqlite3.IntegrityError:
                pass

    def get_unsolved_problem(self, difficulty: str = "Mudah") -> Optional[dict]:
        """Mengambil satu soal yang belum terpecahkan dari perpustakaan."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM math_problem_library WHERE status = 'unsolved' AND estimated_difficulty = ? ORDER BY RANDOM() LIMIT 1",
                (difficulty,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_math_problem_solution(
        self,
        problem_id: int,
        status: str,
        internal_solution: str,
        guru_solution: str,
        learned_method: str,
    ):
        """Memperbarui soal di perpustakaan dengan hasil pembelajaran."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE math_problem_library
                SET status = ?, internal_solution = ?, guru_solution = ?, learned_method = ?
                WHERE id = ?
            """,
                (status, internal_solution, guru_solution,
                 learned_method, problem_id),
            )
            conn.commit()

    def get_problem_solving_stats(self) -> dict:
        """Menghitung statistik penyelesaian soal untuk menentukan level kurikulum."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT estimated_difficulty, status FROM math_problem_library", conn
            )
        if df.empty:
            return {"Mudah": 0.0}

        stats = {}
        for difficulty in ["Mudah", "Menengah", "Sulit"]:
            subset = df[df["estimated_difficulty"] == difficulty]
            if len(subset) == 0:

                stats[difficulty] = 1.0
            else:
                solved_count = len(subset[subset["status"] != "unsolved"])
                stats[difficulty] = solved_count / len(subset)
        return stats

    def find_relevant_math_method(self, problem_keywords: list) -> Optional[str]:
        """
        Mencari di DKG apakah ada metode yang sudah dipelajari untuk kata kunci soal ini.
        Ini adalah 'intuisi' matematika model.
        """
        if not hasattr(self, "brain"):
            return None


        for keyword in problem_keywords:

            for source, target, data in self.brain.dkg.edges:
                if source == keyword and data.get("relationship") == "solved_by":
                    logger.info(
                        f"💡 [Intuisi] Ditemukan metode '{target}' untuk kata kunci '{keyword}' di DKG."
                    )

                    return target
        return None

    def log_user_activity(self, description: str):
        """Mencatat aktivitas pengguna yang dipantau ke database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO user_activity_log (activity_description) VALUES (?)",
                    (description,),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[NSMM] Gagal mencatat aktivitas pengguna: {e}")

    def add_causal_hypothesis(
        self, treatment: str, outcome: str, regime_id: int, effect: float
    ):
        """Menambah atau memperbarui hipotesis kausal di dalam bank."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO causal_hypotheses
                (treatment_variable, outcome_variable, regime_id, estimated_effect,
                 confidence_score, status, last_validated_timestamp)
                VALUES (?, ?, ?, ?, 1.0, 'active', ?)
            """,
                (treatment, outcome, regime_id, effect, datetime.now().isoformat()),
            )
            conn.commit()

    def get_active_causal_hypotheses(self) -> pd.DataFrame:
        """Mengambil semua hipotesis yang masih aktif untuk divalidasi."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM causal_hypotheses WHERE status = 'active'", conn
            )
        return df


    def update_causal_hypothesis(
        self, rule_id: int, new_confidence: float, new_status: str
    ):
        """Memperbarui status dan skor kepercayaan dari sebuah hipotesis kausal."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE causal_hypotheses
                SET confidence_score = ?, status = ?, last_validated_timestamp = ?
                WHERE rule_id = ?
            """,
                (new_confidence, new_status, datetime.now().isoformat(), rule_id),
            )
            conn.commit()

    def update_icon_memory(self, app_name: str, coordinates: list, was_successful: bool):
        """Memperbarui atau mencatat pengetahuan tentang sebuah ikon aplikasi."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT usage_count, success_rate FROM icon_memory WHERE app_name = ?",
                (app_name,)
            )
            result = cursor.fetchone()

            if result:

                usage_count, success_rate = result
                new_usage_count = usage_count + 1

                new_success_rate = (
                    (success_rate * usage_count) + (1 if was_successful else 0)
                ) / new_usage_count

                cursor.execute(
                    """
                    UPDATE icon_memory
                    SET usage_count = ?, success_rate = ?, last_known_x = ?, last_known_y = ?, last_used = ?
                    WHERE app_name = ?
                    """,
                    (
                        new_usage_count,
                        new_success_rate,
                        coordinates[0],
                        coordinates[1],
                        datetime.now().isoformat(),
                        app_name
                    )
                )
            else:

                cursor.execute(
                    """
                    INSERT INTO icon_memory (app_name, last_known_x, last_known_y, last_used)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        app_name,
                        coordinates[0],
                        coordinates[1],
                        datetime.now().isoformat()
                    )
                )

            conn.commit()
            logger.info(
                f"🧠 [Icon Memory] Pengetahuan tentang ikon '{app_name}' telah diperbarui."
            )

    def get_neurons_for_pruning(
        self, confidence_threshold: float = 0.1, age_days: int = 90
    ) -> list[int]:
        """
        Mengambil ID neuron yang merupakan kandidat untuk dipangkas (pruning).
        Kriteria: Keyakinan sangat rendah ATAU sudah sangat tua dan jarang diakses.
        """
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()


            c.execute(
                """
                SELECT id FROM insight_neurons
                WHERE status = 'experimental' AND confidence < ?
            """,
                (confidence_threshold,),
            )
            low_confidence_ids = [row[0] for row in c.fetchall()]


            ninety_days_ago = (
                datetime.now() - timedelta(days=age_days)).isoformat()
            c.execute(
                """
                SELECT id FROM insight_neurons
                WHERE status = 'validated' AND last_accessed < ? AND access_count < 5
            """,
                (ninety_days_ago,),
            )
            obsolete_ids = [row[0] for row in c.fetchall()]

            return list(set(low_confidence_ids + obsolete_ids))

    def get_neuron_count_by_status(self) -> dict:
        """Menghitung jumlah neuron berdasarkan statusnya."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT status, COUNT(*) FROM insight_neurons GROUP BY status")
            return dict(c.fetchall())

    def log_epoch_experience(self, session_id: str, epoch: int, data: dict) -> int:
        """Mencatat satu pengalaman epoch lengkap ke memori."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO epoch_memories (session_id, epoch, metrics, state_vector, diagnosis, oracle_query, oracle_response, final_decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    epoch,
                    json.dumps(make_json_serializable(data.get("metrics"))),
                    json.dumps(make_json_serializable(
                        data.get("state_vector"))),
                    data.get("diagnosis"),
                    data.get("oracle_query"),
                    data.get("oracle_response"),
                    json.dumps(make_json_serializable(
                        data.get("final_decision"))),
                ),
            )
            conn.commit()
            return c.lastrowid

    def store_insight_neuron(
        self,
        memory_id: int,
        module: nn.Module,
        reason: str,
        state_vector_dim: int,
        outcome: str = "neutral",
        trigger_metric_key: str = "N/A",
        trigger_metric_value: float = 0.0,
        confidence: float = 0.5,
    ):
        """Menyimpan bobot dan metadata dari Insight Module baru dengan labeling."""
        neuron_name = f"InsightNeuron_{memory_id}_{int(time.time())}"
        summary = str(module)
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        weights_blob = buffer.getvalue()
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO insight_neurons (
                    memory_id, neuron_name, trigger_reason, architecture_summary, weights_blob,
                    outcome, trigger_metric_key, trigger_metric_value, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    memory_id,
                    neuron_name,
                    reason,
                    summary,
                    weights_blob,
                    outcome,
                    trigger_metric_key,
                    trigger_metric_value,
                    confidence,
                    "experimental",
                ),
            )
            conn.commit()
        logger.info(
            f"🌱 [Neurogenesis] Neuron baru '{neuron_name}' disimpan. Status: [Experimental], Outcome: {outcome.upper()}"
        )

    def retrieve_relevant_neurons(
        self, state_vector_dim: int
    ) -> list[tuple[str, nn.Module]]:
        """Mengambil dan merekonstruksi neuron yang paling relevan dari DB."""
        reconstructed_neurons = []
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT neuron_name, weights_blob FROM insight_neurons WHERE status != 'archived' ORDER BY access_priority DESC, confidence DESC LIMIT 10"
            )
            for row in c.fetchall():
                try:
                    name, weights_blob = row
                    module = nn.Sequential(
                        nn.Linear(state_vector_dim,
                                  16), nn.SiLU(), nn.Linear(16, 8)
                    )
                    buffer = io.BytesIO(weights_blob)
                    module.load_state_dict(torch.load(buffer))
                    module.eval()
                    reconstructed_neurons.append((name, module))
                except Exception as e:
                    logger.warning(
                        f"Gagal merekonstruksi neuron '{name}': {e}")
        logger.info(
            f"🧠 [NSMM Recall] Berhasil merekonstruksi {len(reconstructed_neurons)} neuron dari memori."
        )
        return reconstructed_neurons

    def query_similar_experiences(
        self, query_text: str, embedding_model: "APIEmbedder", top_k: int = 3
    ) -> list[str]:
        """Mencari diagnosis/pengalaman yang paling mirip secara semantik dari histori."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            df_memories = pd.read_sql_query(
                "SELECT diagnosis FROM epoch_memories WHERE diagnosis IS NOT NULL AND diagnosis != ''",
                conn,
            )

        if df_memories.empty or embedding_model is None:
            return []

        corpus_diagnoses = df_memories["diagnosis"].tolist()

        try:
            from sentence_transformers.util import semantic_search

            query_embedding = embedding_model.encode(
                query_text, task_type="query")
            corpus_embeddings = embedding_model.encode(
                corpus_diagnoses, task_type="passage"
            )
            hits = semantic_search(
                query_embedding, corpus_embeddings, top_k=top_k)

            if not hits or not hits[0]:
                return []

            return [corpus_diagnoses[hit["corpus_id"]] for hit in hits[0]]
        except Exception as e:
            logger.error(f"[NSMM Semantic Search] Gagal: {e}")
            return []

    def query_similar_principles(
        self, query_text: str, embedding_model: "APIEmbedder", top_k: int = 3
    ) -> list[str]:
        """Mencari prinsip/kebijaksanaan yang paling relevan secara semantik dari Brain."""

        if not embedding_model or not hasattr(self, "brain") or not self.brain:
            return []


        try:
            relevant_chunks = self.brain.query(
                f"Prinsip atau strategi untuk situasi: {query_text}", k=top_k
            )
            return relevant_chunks
        except Exception as e:
            logger.error(f"[NSMM Principle Search] Gagal: {e}")
            return []

    def get_neurons_for_consolidation(
        self, status: str = "experimental", limit: int = 500
    ) -> list[dict]:
        """Mengambil data neuron yang akan dianalisis oleh Pustakawan AI."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, neuron_name, trigger_reason, outcome, confidence FROM insight_neurons WHERE status = ? ORDER BY id DESC LIMIT ?",
                (status, limit),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_neuron_status(self, neuron_ids: list[int], new_status: str):
        """Memperbarui status beberapa neuron sekaligus (misal: menjadi 'archived')."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            placeholders = ",".join("?" for _ in neuron_ids)
            conn.execute(
                f"UPDATE insight_neurons SET status = ? WHERE id IN ({placeholders})",
                [new_status] + neuron_ids,
            )
            conn.commit()
        logger.info(
            f"[NSMM] Status dari {len(neuron_ids)} neuron diubah menjadi [{new_status.upper()}]."
        )

    def update_neuron_evaluation(
        self, neuron_id: int, new_confidence: float, new_status: str
    ):
        """Memperbarui hasil evaluasi sebuah neuron setelah diuji."""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "UPDATE insight_neurons SET confidence = ?, status = ?, access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (new_confidence, new_status, datetime.now().isoformat(), neuron_id),
            )
            conn.commit()



class OptunaTuner:
    def __init__(
        self,
        base_hparams,
        train_fn,
        tracker: ExperimentTracker,
        metric_name="val_loss",
        n_trials=15,
        timeout=None,
    ):
        self.base, self.train_fn, self.tracker = base_hparams, train_fn, tracker
        self.metric = metric_name
        self.study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            study_name=f"{base_hparams['project_id']}_optuna",
        )
        self.n_trials, self.timeout = n_trials, timeout

    def _sample(self, trial):
        hp = self.base.copy()


        d_model_selected = trial.suggest_categorical(
            "d_model", [64, 96, 128, 192, 256])
        hp["d_model"] = d_model_selected


        try:

            divisors = [
                i
                for i in range(2, d_model_selected // 16 + 1)
                if d_model_selected % i == 0
            ]
            if not divisors:

                divisors = [h for h in [2, 4] if d_model_selected % h == 0]


            if not divisors:
                raise ValueError("Tidak ada pembagi valid")


            hp["n_heads"] = trial.suggest_categorical("n_heads", divisors)

        except Exception:


            hp["n_heads"] = 2


        hp["lr"] = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.4)
        hp["weight_decay"] = trial.suggest_loguniform(
            "weight_decay", 1e-6, 1e-3)
        hp["window"] = trial.suggest_int("window", 30, 120, step=15)
        hp["horizon"] = trial.suggest_int("horizon", 3, 14)

        hp["snn_num_layers"] = trial.suggest_int("snn_num_layers", 1, 4)
        hp["snn_dropout"] = trial.suggest_float("snn_dropout", 0.1, 0.5)

        hp["fno_modes"] = trial.suggest_categorical("fno_modes", [8, 16, 32])

        return hp

    def _objective(self, trial):
        hp = self._sample(trial)
        score, full_metrics, _, _ = self.train_fn(hp)
        trial.set_user_attr("metrics", full_metrics)
        return score

    def optimize(self):
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            gc_after_trial=True,
        )
        best_hp = self.base.copy()
        best_hp.update(self.study.best_params)
        best_metrics = self.study.best_trial.user_attrs.get("metrics", {})
        self.tracker.log(self.base["project_id"], -1, best_hp, best_metrics)
        return best_hp





class HyperparameterProposal(BaseModel):
    learning_rate: float = Field(
        ..., description="Learning rate untuk optimizer, contoh: 0.00025"
    )
    batch_size: int = Field(..., description="Ukuran batch, contoh: 512")
    weight_decay: float = Field(
        ..., description="Weight decay untuk regularisasi, contoh: 0.08"
    )
    gradient_accumulation_steps: int = Field(
        ..., description="Langkah akumulasi gradien, contoh: 4"
    )
    max_sequence_length: int = Field(
        ..., description="Panjang sekuens maksimal, contoh: 512"
    )
    num_epochs: int = Field(...,
                            description="Jumlah epoch pelatihan, contoh: 15")
    warmup_steps: int = Field(
        ..., description="Jumlah langkah pemanasan scheduler, contoh: 1000"
    )
    gradient_clip: float = Field(...,
                                 description="Nilai kliping gradien, contoh: 0.8")
    adam_beta1: float = Field(...,
                              description="Parameter Adam Beta 1, contoh: 0.88")
    adam_beta2: float = Field(...,
                              description="Parameter Adam Beta 2, contoh: 0.995")
    dropout: float = Field(..., description="Tingkat dropout, contoh: 0.1")
    d_model: int = Field(...,
                         description="Dimensi internal model, contoh: 512")
    scheduler_type: str = Field(
        ..., description="Tipe scheduler, contoh: 'ReduceLROnPlateau' atau 'OneCycleLR'"
    )




    use_tgw: bool = Field(
        ...,
        description="Aktifkan (True) atau non-aktifkan (False) Temporal Gradient Wormhole.",
    )
    tgw_rank: int = Field(
        ..., description="Rank untuk matriks wormhole low-rank, contoh: 8, 16, atau 32."
    )
    tgw_beta: float = Field(
        ...,
        description="Faktor pembobot untuk gradien fine-tune saat injeksi, contoh: 0.3",
    )
    tgw_stability_weight: float = Field(
        ..., description="Bobot untuk stability loss TGW, contoh: 0.05."
    )


    reasoning: str = Field(
        ..., description="Alasan singkat mengapa konfigurasi ini diusulkan."
    )


class EventAnalysis(BaseModel):
    """Skema untuk satu hasil analisis peristiwa dari batch."""
    original_ticker: str = Field(
        description="Ticker asli yang terkait dengan peristiwa ini.")
    original_date: str = Field(
        description="Tanggal asli peristiwa dalam format YYYY-MM-DD.")
    event_name: str = Field(
        description="Nama yang jelas dan ringkas untuk peristiwa penyebab, contoh: 'Laporan Keuangan Q3 Melampaui Ekspektasi'.")
    event_type: Literal["CORPORATE", "MACROECONOMIC",
                        "MARKET_SENTIMENT", "POLITICAL", "UNKNOWN"]
    impact_score: float = Field(
        description="Skor dampak antara -1.0 (sangat negatif) dan 1.0 (sangat positif).")
    summary: str = Field(
        description="Ringkasan singkat 1-2 kalimat mengenai penyebab utama guncangan berdasarkan berita.")


class BatchAnalysisResult(BaseModel):
    """Skema untuk menampung seluruh daftar hasil analisis peristiwa."""
    analyses: List[EventAnalysis] = Field(
        description="Daftar lengkap dari setiap analisis peristiwa yang diminta.")





class ArbiterDecision(BaseModel):
    synthesis_reasoning: str = Field(
        description="Analisis mendalam tentang bagaimana proposal akhir disintesis dari berbagai panelis, termasuk analisis pro-kontra dan alasan penggabungan."
    )
    final_decision: HyperparameterProposal = Field(
        description="Satu set hyperparameter final yang merupakan hasil musyawarah dan sintesis terbaik."
    )


class AssessProposal(BaseModel):
    is_accepted: bool = Field(description="True jika proposal diterima.")
    reasoning: str = Field(description="Alasan keputusan.")


class CriticalInterventionPlan(BaseModel):
    """Skema untuk rencana intervensi kritis yang diusulkan oleh AI Supervisor."""

    verdict: Literal[
        "apply_reflex_fix", "propose_new_plan", "stop_training_critical"
    ] = Field(description="Keputusan tindakan yang harus diambil.")
    reasoning: str = Field(
        description="Penjelasan singkat dan logis untuk keputusan yang diambil."
    )
    proposed_hparams: Optional[Dict[str, Any]] = Field(
        None,
        description="Jika verdict adalah 'propose_new_plan', sertakan hyperparameter baru di sini.",
    )


class DiagnoseError(BaseModel):
    root_cause_summary: str = Field(description="Ringkasan akar masalah.")
    detailed_explanation: str = Field(description="Penjelasan teknis error.")


class ProposeCodeFix(BaseModel):
    """Skema untuk usulan perbaikan kode yang ditargetkan."""

    explanation_of_fix: str = Field(
        description="Penjelasan singkat dan teknis mengenai akar masalah dan bagaimana perbaikan ini mengatasinya."
    )
    original_code_to_find: str = Field(
        description="Satu baris kode asli yang menyebabkan error. Harus sama persis dengan yang ada di file sumber untuk ditemukan."
    )
    suggested_code_to_replace: str = Field(
        description="Satu atau lebih baris kode baru yang sudah diperbaiki untuk menggantikan baris kode asli."
    )


class ConfirmFix(BaseModel):
    """Skema untuk menyetujui atau menolak usulan perbaikan kode."""

    is_approved: bool = Field(
        description="Set True jika perbaikan yang diusulkan logis, aman, dan secara langsung mengatasi error."
    )
    reasoning: str = Field(
        description="Penjelasan singkat dan teknis mengapa perbaikan disetujui atau ditolak."
    )


class Critique(BaseModel):
    """Skema untuk kritik terstruktur terhadap sebuah teks atau analisis."""

    strengths: str = Field(
        description="Poin-poin utama yang sudah baik dari teks asli."
    )
    weaknesses: str = Field(
        description="Poin-poin kelemahan atau area yang kurang jelas/akurat."
    )
    actionable_suggestions: list[str] = Field(
        description="Daftar saran konkret yang bisa ditindaklanjuti untuk perbaikan."
    )


class RefinedOutput(BaseModel):
    """Skema untuk output yang telah disempurnakan."""

    refined_text: str = Field(
        description="Versi teks akhir yang telah direvisi dan disempurnakan."
    )
    reasoning_for_changes: str = Field(
        description="Penjelasan singkat mengenai perubahan kunci yang dibuat berdasarkan kritik."
    )


class Task(BaseModel):
    task_name: str = Field(
        description="Nama fungsi atau mode yang harus dieksekusi, contoh: 'run_continuous_training'."
    )
    params: dict = Field(
        description="Kamus berisi parameter yang dibutuhkan untuk menjalankan tugas tersebut."
    )
    reasoning: str = Field(
        description="Alasan singkat mengapa tugas ini dipilih sebagai bagian dari rencana."
    )


class ProposeGoalModification(BaseModel):
    decision: Literal["MAINTAIN", "AMEND", "DEPRECATE"] = Field(
        description="Keputusan akhir Arbiter terhadap tujuan saat ini."
    )
    reasoning: str = Field(
        description="Penjelasan mendalam untuk keputusan yang diambil, berdasarkan bukti yang ada."
    )
    amended_description: Optional[str] = Field(
        None,
        description="Jika keputusan adalah 'AMEND', berikan deskripsi tujuan baru yang sudah diperbaiki.",
    )


class ProposeNewGoal(BaseModel):
    """Skema untuk tujuan baru yang dirumuskan secara mandiri oleh AI."""
    new_goal_description: str = Field(
        description="Deskripsi yang jelas, spesifik, dan dapat ditindaklanjuti untuk tujuan pembelajaran mandiri yang baru."
    )
    success_metric: str = Field(
        description="Definisi ringkas tentang bagaimana keberhasilan tujuan ini akan diukur."
    )
    failure_metric: str = Field(
        description="Definisi ringkas tentang kondisi kegagalan tujuan ini."
    )
    reasoning: str = Field(
        description="Penjelasan singkat mengapa tujuan ini dipilih berdasarkan analisis konteks yang ada."
    )


class ExecutionPlan(BaseModel):
    overall_reasoning: str = Field(
        description="Penjelasan tingkat tinggi tentang strategi di balik keseluruhan rencana ini."
    )
    tasks: List[Task] = Field(
        description="Daftar tugas yang harus dieksekusi secara berurutan."
    )


class SubmitTaskRequest(BaseModel):
    task_description: str = Field(
        description="Deskripsi jelas dari tugas yang diminta oleh pengguna, contoh: 'analisis harga penutupan BBCA di hari Jumat'."
    )
    priority: int = Field(
        default=10, description="Prioritas tugas, di mana 1 adalah yang tertinggi."
    )


class GenerativeStrategy(BaseModel):
    strategy_name: str = Field(
        ...,
        description="Nama yang kreatif dan deskriptif untuk strategi baru, contoh: 'Regulatory Echo'.",
    )
    hypothesis: str = Field(
        ...,
        description="Penjelasan logis tentang mengapa strategi ini mungkin berhasil, berdasarkan data yang ada.",
    )
    rules: dict[str, str] = Field(
        ...,
        description="Kamus aturan JIKA-MAKA yang dapat diuji secara teknis. Kunci adalah 'IF', nilai adalah 'THEN'.",
    )
    data_source_justification: str = Field(
        ...,
        description="Penjelasan singkat tentang sumber data (misal: DKG, Anomaly Memory) yang menginspirasi hipotesis ini.",
    )


class NewArchitectureModule(BaseModel):
    module_name: str = Field(
        ...,
        description="Nama kelas untuk modul PyTorch baru, contoh: 'V_ShapeReversalLayer'.",
    )
    module_code: str = Field(
        ...,
        description="Seluruh blok kode Python untuk kelas PyTorch baru. Harus lengkap dan valid secara sintaksis.",
    )
    integration_plan: str = Field(
        ...,
        description="Penjelasan di mana dan bagaimana layer baru ini harus diintegrasikan ke dalam arsitektur yang ada.",
    )
    updated_forward_method: str = Field(
        ...,
        description="Seluruh blok kode untuk metode 'forward' dari kelas HybridGNN_DPASTI yang telah diperbarui untuk menyertakan modul baru.",
    )


def _identity(**kwargs):
    return kwargs


class EvaluateRecoveryPlan(BaseModel):
    is_plan_sufficient: bool = Field(
        ...,
        description="Set True jika rencana perbaikan yang diusulkan kemungkinan besar akan berhasil menstabilkan model. Set False jika masalahnya terlihat lebih dalam dan membutuhkan analisis lebih lanjut.",
    )
    reasoning: str = Field(
        ...,
        description="Penjelasan singkat dan teknis mengapa rencana tersebut dianggap cukup atau tidak cukup.",
    )


class WebFileAnalysisRequest(BaseModel):
    """Skema untuk meminta analisis file dari URL."""

    url: str = Field(
        ..., description="URL lengkap ke file .pdf atau .zip yang akan dianalisis."
    )
    question: str = Field(
        ...,
        description="Pertanyaan spesifik yang ingin dijawab dari konten file tersebut.",
    )


class DigestedActivity(BaseModel):
    """
    Struktur data untuk intisari yang diekstrak dari log aktivitas mentah pengguna.
    Versi ini mencakup inferensi mood dan ekstraksi data real-time.
    """

    specific_content: str = Field(
        ...,
        description="Judul spesifik dari konten yang dilihat, misal: 'BTCUSDT, 1D 65432.10'.",
    )
    website_or_app: str = Field(
        ..., description="Nama situs web atau aplikasi utama, misal: 'TradingView'."
    )
    general_topic: str = Field(
        ..., description="Topik umum dari aktivitas, misal: 'Analisis Pasar Kripto'."
    )
    activity: str = Field(
        ..., description="Aktivitas utama yang dilakukan, misal: 'Memantau harga'."
    )
    activity_sentiment: Literal["Positif", "Netral", "Negatif"] = Field(
        ..., description="Sentimen dari aktivitas itu sendiri."
    )
    inferred_user_mood: str = Field(
        ...,
        description="Inferensi tentang kemungkinan MOOD PENGGUNA berdasarkan konten.",
    )
    realtime_data: Optional[dict] = Field(
        None,
        description="Data real-time yang diekstrak dari judul, seperti harga atau perubahan, jika ada.",
    )

    entities: list[str] = Field(
        ...,
        description="Daftar entitas (misal: nama artis, organisasi, produk) yang terdeteksi di dalam konten.",
    )


class DigestedActivityBatch(BaseModel):
    """Sebuah wadah untuk menampung daftar hasil cerna aktivitas."""

    activities: list[DigestedActivity] = Field(
        ..., description="Daftar lengkap dari setiap aktivitas yang telah dicerna."
    )


class CommandIntent(BaseModel):
    """Struktur untuk hasil pemahaman perintah pengguna."""
    intent_name: Literal["ingest_file", "run_analysis", "unknown"] = Field(
        description="Niat utama yang terdeteksi dari perintah pengguna."
    )
    parameters: dict = Field(
        description="Kamus berisi parameter yang diekstrak, contoh: {'file_path': 'C:/path/to/file.json'}"
    )


class GUIAction(BaseModel):
    """Skema untuk satu aksi GUI yang akan dieksekusi."""
    action_type: Literal["click", "type", "move", "scroll", "wait", "press"] = Field(
        description="Jenis aksi yang harus dilakukan.")


    coordinates: Optional[List[int]] = Field(
        None, description="Koordinat [x, y] untuk aksi 'click' atau 'move'.")

    text_to_type: Optional[str] = Field(
        None, description="Teks yang akan diketik untuk aksi 'type'.")
    key_to_press: Optional[str] = Field(
        None, description="Tombol keyboard yang akan ditekan (misal: 'right', 'down', 'enter').")

    scroll_amount: Optional[int] = Field(
        None, description="Jumlah scroll (positif untuk ke atas, negatif untuk ke bawah).")

    wait_seconds: Optional[float] = Field(
        None, description="Durasi menunggu dalam detik untuk aksi 'wait'.")

    reasoning: str = Field(
        description="Alasan singkat mengapa aksi ini perlu dilakukan.")


class GUIActionPlan(BaseModel):
    """Skema untuk daftar urutan aksi GUI."""
    actions: List[GUIAction] = Field(
        description="Daftar aksi GUI yang harus dieksekusi secara berurutan.")


def _load_offline_vision_model():
    """Memuat model image captioning offline sekali saja dan menyimpannya di cache."""
    global _offline_vision_model
    if _offline_vision_model is None:
        logger.info(
            "🧠 Memuat model vision offline (Salesforce/blip-image-captioning-large)... Ini hanya terjadi sekali.")

        device = 0 if torch.cuda.is_available() else -1

        _offline_vision_model = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-large",
            device=device
        )
        logger.info("✅ Model vision offline siap digunakan.")
    return _offline_vision_model


def _load_offline_translator_model():
    """Memuat model Sentence Transformer offline sekali saja."""
    global _offline_translator_model
    if _offline_translator_model is None:
        logger.info(
            "🧠 Memuat model penerjemah offline (paraphrase-multilingual-MiniLM-L12-v2)...")

        _offline_translator_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("✅ Model penerjemah offline siap digunakan.")
    return _offline_translator_model


_yolo_detector_model = None


def _load_yolo_detector_model():
    """Memuat model deteksi objek YOLOv8n sekali saja dan menyimpannya di cache."""
    global _yolo_detector_model
    if _yolo_detector_model is None:
        from ultralytics import YOLO
        logger.info(
            "🧠 Memuat model deteksi objek offline (YOLOv8n)... Ini hanya terjadi sekali.")


        _yolo_detector_model = YOLO('yolov8n.pt')
        logger.info("✅ Model deteksi objek YOLOv8n siap digunakan.")
    return _yolo_detector_model


@tool
def perceive_screen() -> dict:
    """
    Versi BARU: 'Melihat' layar menggunakan OCR (untuk teks detail)
    dan model vision OFFLINE (untuk konteks visual umum).
    """
    logger.info(
        "👀 [Persepsi Offline] Mengambil tangkapan layar dan menganalisis...")
    try:
        screenshot_path = MODELS_DIR / "temp_screenshot.png"


        with mss.mss() as sct:
            sct.shot(output=str(screenshot_path))

        screenshot = Image.open(screenshot_path)


        ocr_text = pytesseract.image_to_string(screenshot)


        captioner = _load_offline_vision_model()

        visual_description = captioner(str(screenshot_path))[
            0]['generated_text']

        perception_data = {
            "ocr_text": ocr_text,
            "visual_description": visual_description,
            "screenshot_path": str(screenshot_path)
        }
        logger.info("✅ [Persepsi Offline] Analisis layar selesai.")
        return perception_data

    except Exception as e:
        logger.error(
            f"Gagal melakukan persepsi layar secara offline: {e}", exc_info=True)
        return {"error": str(e)}


@tool
def perceive_desktop_elements() -> List[dict]:
    """
    Menganalisis layar untuk mendeteksi elemen desktop (ikon dan teks) secara terstruktur.
    Menggunakan YOLOv8 untuk deteksi ikon dan Tesseract OCR untuk teks, lalu
    mengasosiasikan teks ke ikon yang relevan. Mengembalikan daftar objek yang terdeteksi.
    """
    logger.info(
        "️️️👁️ [Persepsi Hibrida] Memindai layar untuk ikon (YOLO) dan teks (OCR)...")

    try:

        with mss.mss() as sct:
            sct_img = sct.grab(sct.monitors[1])
            screenshot_pil = Image.frombytes(
                "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            screenshot_np = np.array(screenshot_pil)


        yolo_model = _load_yolo_detector_model()


        yolo_results = yolo_model(screenshot_np, verbose=False)
        detected_icons = []



        icon_class_names = {32: "sport ball", 0: "person"}

        for result in yolo_results:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id in icon_class_names:
                    coords = box.xyxy[0].tolist()

                    app_name = icon_class_names[class_id]
                    detected_icons.append({
                        "app_name": app_name,

                        "icon_box": [int(c) for c in coords],
                        "text_box": None
                    })


        ocr_data = pytesseract.image_to_data(
            screenshot_pil, output_type=PyTesseractOutput.Dict)
        detected_texts = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            if conf > 60 and len(text) > 2:
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                detected_texts.append({
                    "text": text,
                    "text_box": [x, y, x + w, y + h]
                })


        final_elements = []
        used_texts = set()

        for icon in detected_icons:
            icon_center_x = (icon["icon_box"][0] + icon["icon_box"][2]) / 2
            icon_bottom_y = icon["icon_box"][3]
            best_text_match = None
            min_distance = float('inf')

            for i, text_block in enumerate(detected_texts):
                if i in used_texts:
                    continue

                text_center_x = (
                    text_block["text_box"][0] + text_block["text_box"][2]) / 2
                text_top_y = text_block["text_box"][1]


                is_below = text_top_y > icon_bottom_y
                horizontal_dist = abs(icon_center_x - text_center_x)

                if is_below and horizontal_dist < 40 and horizontal_dist < min_distance:
                    min_distance = horizontal_dist
                    best_text_match = i

            unified_element = icon.copy()
            if best_text_match is not None:
                matched_text_info = detected_texts[best_text_match]

                unified_element["app_name"] = matched_text_info["text"]
                unified_element["text_box"] = matched_text_info["text_box"]
                used_texts.add(best_text_match)

            final_elements.append(unified_element)

        logger.info(
            f"  -> [Persepsi Hibrida] Ditemukan {len(final_elements)} elemen terstruktur.")
        return final_elements

    except Exception as e:
        logger.error(
            f"[Persepsi Hibrida] Gagal mendeteksi elemen desktop: {e}", exc_info=True)
        return []


@tool(args_schema=GUIActionPlan)
def execute_gui_plan(actions: List[dict]) -> str:
    """
    Gunakan alat ini untuk mengeksekusi serangkaian aksi GUI (klik, ketik, scroll)
    yang telah direncanakan untuk berinteraksi dengan komputer.
    """
    logger.info(f"🦾 [Aksi] Mengeksekusi {len(actions)} langkah rencana GUI...")
    pyautogui.FAILSAFE = True
    action_results = []

    try:
        for i, action_data in enumerate(actions):
            action = GUIAction(**action_data)
            logger.info(
                f"  -> Langkah {i+1}: {action.action_type.upper()} - Alasan: {action.reasoning}")

            if action.action_type == "click":
                if action.coordinates:
                    pyautogui.click(
                        action.coordinates[0], action.coordinates[1])
            elif action.action_type == "type":
                if action.text_to_type:
                    pyautogui.typewrite(action.text_to_type, interval=0.05)

            elif action.action_type == "press":
                if action.key_to_press:
                    pyautogui.press(action.key_to_press)

            elif action.action_type == "move":
                if action.coordinates:
                    pyautogui.moveTo(
                        action.coordinates[0], action.coordinates[1], duration=0.5)
            elif action.action_type == "scroll":
                if action.scroll_amount:
                    pyautogui.scroll(action.scroll_amount)
            elif action.action_type == "wait":
                if action.wait_seconds:
                    time.sleep(action.wait_seconds)

            action_results.append(
                f"Langkah {i+1} ({action.action_type}) berhasil.")
            time.sleep(0.5)

        logger.info("✅ [Aksi] Rencana GUI berhasil dieksekusi.")
        return f"Semua {len(actions)} aksi berhasil dieksekusi."
    except Exception as e:
        logger.error(f"Gagal saat eksekusi GUI pada langkah {i+1}: {e}")
        return f"Gagal pada langkah {i+1} ({action.action_type}): {e}"


@tool(args_schema=WebFileAnalysisRequest)
def analyze_web_file(url: str, question: str) -> str:
    """
    Gunakan alat ini untuk mengunduh, membaca, dan menganalisis konten dari file PDF atau ZIP yang ditemukan di web.
    Ini sangat berguna untuk mendapatkan data mendalam dari laporan keuangan, riset, atau dokumen teknis.
    """

    if "web_searcher" not in globals() or "api_pool" not in globals():
        return "Error: Komponen WebSearchManager atau AIPool tidak tersedia."

    result = web_searcher.fetch_and_process_file_url(url)

    if result["status"] == "error":
        return f"Gagal memproses file: {result['message']}"

    summarizer_prompt = f"""
    Berdasarkan konteks dari dokumen berikut, jawab pertanyaan pengguna.

    Pertanyaan Pengguna: "{question}"

    Konteks dari Dokumen ({result['file_type']} dari {result['source_url']}):
    ---
    {result['content']}
    ---

    Jawaban ringkas Anda berdasarkan konteks di atas:
    """
    summary = api_pool.call_gemini_for_text(
        summarizer_prompt, "advanced_advisor")
    return summary


class TreeOfThoughtNode(BaseModel):
    topic: str = Field(description="Topik utama dari node ini.")
    summary: str = Field(description="Rangkuman singkat dari topik ini.")
    children: List["TreeOfThoughtNode"] = Field(
        default_factory=list, description="Sub-topik di bawah node ini."
    )


class HierarchicalSummary(BaseModel):
    file_name: str
    main_purpose: str
    thought_tree: TreeOfThoughtNode


@tool
def get_latest_model_performance() -> str:
    """
    Gunakan alat ini untuk mendapatkan metrik kinerja (seperti val_loss, val_mse)
    dan hyperparameter dari model juara terakhir yang berhasil dilatih.
    Ini adalah satu-satunya cara untuk mengetahui skor model terakhir.
    """
    logger.info("🛠️ [TOOL CALLED] get_latest_model_performance dipanggil...")
    if not LAST_PROJECT_INFO_TXT.exists():
        return "Error: File info model terakhir (last_project_info.txt) tidak ditemukan. Tidak ada model juara yang bisa dianalisis."

    try:
        last_id, ckpt_path_str = LAST_PROJECT_INFO_TXT.read_text().strip().split(",")
        ckpt_path = Path(ckpt_path_str)

        if not ckpt_path.exists():
            return f"Error: File checkpoint '{ckpt_path_str}' tidak ditemukan."

        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))


        hparams = ckpt.get("hyper_parameters", {})

        metrics = ckpt.get("callback_metrics", {})


        val_loss = metrics.get("val_loss", "N/A")
        val_mse = metrics.get("val_mse", "N/A")
        val_da = metrics.get("val_da", "N/A")


        report = (
            f"Analisis Model Juara Terakhir (Project ID: {last_id}):\n"
            f"- Path Checkpoint: {ckpt_path_str}\n"
            f"- Metrik Kinerja Utama:\n"
            f"  - val_loss: {val_loss:.6f}\n"
            f"  - val_mse: {val_mse:.6f}\n"
            f"  - val_da (Directional Accuracy): {val_da:.2%}\n"
            f"- Hyperparameter Kunci:\n"
            f"  - Learning Rate: {hparams.get('lr', 'N/A')}\n"
            f"  - Window Size: {hparams.get('window', 'N/A')}\n"
            f"  - Horizon: {hparams.get('horizon', 'N/A')}\n"
            f"  - d_model: {hparams.get('d_model', 'N/A')}\n"
        )
        return report
    except Exception as e:
        logger.error(
            f"[TOOL ERROR] Gagal saat get_latest_model_performance: {e}")
        return f"Terjadi error saat mencoba membaca data model: {str(e)}"


@tool
def list_available_models() -> str:
    """
    Gunakan alat ini untuk melihat daftar semua model yang telah dilatih dan disimpan di direktori models_trained.
    """
    logger.info("🛠️ [TOOL CALLED] list_available_models dipanggil...")
    if not MODELS_DIR.exists():
        return "Direktori models_trained tidak ditemukan."

    model_files = list(MODELS_DIR.glob("best_model_*.ckpt"))
    if not model_files:
        return (
            "Tidak ada file model (.ckpt) yang ditemukan di direktori models_trained."
        )

    report = "Berikut adalah model yang tersedia:\n"
    for f in model_files:
        report += f"- {f.name}\n"
    return report


CONFIRM_FIX_TOOL = StructuredTool.from_function(
    func=_identity,
    name="ConfirmCodeFix",
    description="Gunakan tool ini untuk menyetujui atau menolak usulan perbaikan kode.",
    args_schema=ConfirmFix,
)



@tool
def propose_hp_tool_func(failed_metrics: dict, old_hparams: dict):
    """Fungsi ini mengusulkan hyperparameter baru berdasarkan metrik yang gagal dan hyperparameter lama."""

    new_hparams = {
        "window": (
            old_hparams.get("window", 60) + 10
            if failed_metrics.get("val_loss", 0) > 0.5
            else old_hparams.get("window", 60)
        ),
        "horizon": old_hparams.get("horizon", 7),
        "lr": (
            old_hparams.get("lr", 0.001) * 0.5
            if failed_metrics.get("val_loss", 0) > 1.0
            else old_hparams.get("lr", 0.001)
        ),
        "d_model": (
            old_hparams.get("d_model", 128) + 64
            if failed_metrics.get("val_mse", 0) > 0.2
            else old_hparams.get("d_model", 128)
        ),
        "dropout": min(0.5, old_hparams.get("dropout", 0.2) + 0.1),
        "batch_size": old_hparams.get("batch_size", 64),
        "top_k_features": (
            old_hparams.get("top_k_features", 50) + 10
            if failed_metrics.get("val_da", 1) < 0.7
            else old_hparams.get("top_k_features", 50)
        ),
        "weight_decay": old_hparams.get("weight_decay", 1e-5) * 2,
        "scheduler_type": (
            "ReduceLROnPlateau"
            if failed_metrics.get("val_loss", 0) > 0.5
            else "OneCycleLR"
        ),
    }
    return new_hparams


PROPOSE_HP_TOOL = StructuredTool.from_function(
    func=_identity,
    name="ProposeHyperparameters",
    description="Tool untuk mengirim proposal hyper-parameter sesuai skema baru.",
    args_schema=HyperparameterProposal,
)

ASSESS_TOOL = StructuredTool.from_function(
    func=_identity,
    name="AssessProposal",
    description="Tool untuk menilai proposal hyper-parameter",
    args_schema=AssessProposal,
)

DIAGNOSE_TOOL = StructuredTool.from_function(
    func=_identity,
    name="DiagnoseError",
    description="Tool untuk mendiagnosis akar masalah dari sebuah error berdasarkan konteks dan traceback.",
    args_schema=DiagnoseError,
)

PROPOSE_FIX_TOOL = StructuredTool.from_function(
    func=_identity,
    name="ProposeCodeFix",
    description="Tool untuk mengusulkan perbaikan kode",
    args_schema=ProposeCodeFix,
)


class SystemReadinessCheck(BaseModel):
    """Hasil pengecekan kesiapan sistem sebelum pelatihan."""

    data_path_exists: bool
    data_rows: int
    ram_gb_available: float
    is_sufficient: bool = Field(
        ...,
        description="Bernilai True jika semua kondisi terpenuhi untuk memulai pelatihan.",
    )
    reasoning: str = Field(
        ..., description="Penjelasan singkat tentang kesiapan sistem."
    )


@tool(return_direct=False)
def check_system_readiness(data_path: str, ram_gb_available: float) -> dict:
    """Gunakan alat ini SEBELUM memulai pre-train untuk memeriksa apakah dataset ada dan sumber daya cukup."""
    logger.info(f"🛠️ [TOOL CALLED] Menjalankan check_system_readiness...")
    path = Path(data_path)
    data_exists = path.exists()
    rows = 0
    if data_exists:
        rows = len(pd.read_parquet(path))

    is_ok = data_exists and rows > 1000 and ram_gb_available > 4.0
    reason = (
        "Semua siap."
        if is_ok
        else "Data tidak ditemukan, jumlah baris kurang dari 1000, atau RAM di bawah 4GB."
    )

    return SystemReadinessCheck(
        data_path_exists=data_exists,
        data_rows=rows,
        ram_gb_available=round(ram_gb_available, 2),
        is_sufficient=is_ok,
        reasoning=reason,
    ).model_dump()


@tool
def request_new_data_and_knowledge(topic_of_interest: str) -> str:
    """
    Gunakan tool ini JIKA DAN HANYA JIKA Anda mendeteksi adanya kesenjangan
    pengetahuan ('knowledge gap') yang menghalangi Anda untuk membuat rencana
    yang efektif. Ini akan memicu sistem untuk mencari informasi baru tentang topik tersebut.
    """

    request_details = {"type": "KNOWLEDGE_GAP", "topic": topic_of_interest}
    DATA_REQUEST_QUEUE.put(request_details)
    logger.info(
        f"🧠 [Knowledge Gap] Permintaan data baru untuk topik '{topic_of_interest}' dimasukkan ke antrean."
    )
    return f"Permintaan untuk data tentang '{topic_of_interest}' telah diajukan dan sedang diproses di latar belakang."


def submit_task_to_queue(task_description: str, priority: int) -> str:
    """
    Gunakan tool ini untuk memasukkan permintaan analisis atau tugas baru dari pengguna ke dalam antrean tugas utama.
    Ini adalah cara untuk memberikan pekerjaan baru kepada sistem tanpa menginterupsi siklusnya saat ini.
    """

    global TASK_QUEUE
    logger.info(
        f"📥 [Task Queue] Tugas baru diterima dari pengguna: '{task_description}'"
    )
    TASK_QUEUE.put({"description": task_description, "priority": priority})
    return "Tugas berhasil dicatat dan akan diproses."


def update_economic_data() -> str:
    """
    Jalankan alat ini untuk memperbarui database internal dengan data ekonomi makro terbaru,
    termasuk CPI AS, suku bunga Federal Reserve, dan pengumuman penting lainnya.
    Gunakan ini jika pengguna bertanya tentang data ekonomi terkini atau jika data yang ada sudah usang.
    """
    try:
        run_ingestion_pipeline()
        return "Sukses: Pipeline ingesti data ekonomi telah dijalankan dan data terbaru telah disimpan."
    except Exception as e:
        return f"Error saat menjalankan pipeline ingesti: {e}"



def strip_markdown(text: str) -> str:
    match = re.search(r"```(python\n)?(.*)```", text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return text


def generate_diff_report(original_code: str, corrected_code: str) -> str:
    if original_code.strip() == corrected_code.strip():
        return "Tidak ada perbedaan terdeteksi."
    diff = difflib.unified_diff(
        original_code.splitlines(keepends=True),
        corrected_code.splitlines(keepends=True),
        fromfile="original.py",
        tofile="corrected.py",
        lineterm="",
    )
    report = "--- LAPORAN PERUBAHAN KODE ---\n" +        "=" * 45 + "\n" + "".join(diff)
    return report


def update_module_params(
    module: nn.Module, new_params: dict[str, torch.Tensor]
) -> nn.Module:
    """
    Memuat satu set parameter baru (dari fast_weights MAML) ke dalam sebuah modul.
    Ini adalah cara aman untuk mengupdate model sementara tanpa mengganggu state aslinya.
    """

    new_state_dict = module.state_dict()
    for name, param in new_params.items():
        if name in new_state_dict:
            new_state_dict[name] = (
                param.detach().clone()
            )


    module.load_state_dict(new_state_dict)
    return module


def find_optimal_n_heads(d_model: int) -> int:
    """
    Secara dinamis menemukan jumlah kepala (n_heads) yang optimal dan valid
    untuk nilai d_model yang diberikan.

    Args:
        d_model (int): Dimensi internal model.

    Returns:
        int: Nilai n_heads terbaik yang disarankan.

    Raises:
        ValueError: Jika tidak ada nilai n_heads yang valid ditemukan.
    """
    divisors = []
    for i in range(1, int(d_model**0.5) + 1):
        if d_model % i == 0:
            divisors.append(i)
            if i * i != d_model:
                divisors.append(d_model // i)



    valid_heads = []
    for h in sorted(divisors):
        d_head = d_model / h
        if 16 <= d_head <= 128:
            valid_heads.append(h)

    if not valid_heads:
        raise ValueError(
            f"Tidak dapat menemukan nilai n_heads yang valid untuk d_model={d_model}. "
            "Pilih d_model yang memiliki pembagi yang menghasilkan ukuran kepala antara 16-128."
        )


    power_of_2_heads = [h for h in valid_heads if (
        h > 0) and (h & (h - 1) == 0)]

    if power_of_2_heads:

        return max(power_of_2_heads)
    else:

        logger.warning(
            f"Tidak ada n_heads pangkat 2 yang ideal untuk d_model={d_model}. Memilih kandidat terbaik: {max(valid_heads)}"
        )
        return max(valid_heads)


def ensure_dict(obj, pydantic_model=None):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, str):
        parsed = robust_json_extract(obj, model=pydantic_model)
        if parsed is None:
            return None
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if hasattr(parsed, "dict"):
            return parsed.dict()
        if isinstance(parsed, dict):
            return parsed
    return None



class PolicyEngine:
    def __init__(self):
        self.conn = sqlite3.connect("policy.db")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS policies(id INTEGER PRIMARY KEY, pattern TEXT, score REAL, rank REAL)"
        )

    def analyze_patterns(self):
        with sqlite3.connect("chat_archive.db") as chat_conn:
            df = pd.read_sql("SELECT content FROM chats", chat_conn)
        cnt = df["content"].str.count(r"\b(urgent|immediately|asap)\b").sum()
        self.conn.execute(
            "INSERT INTO policies(pattern,score) VALUES(?,?)", ("urgency", cnt)
        )
        self.conn.commit()

    def get_policies(self):
        return pd.read_sql("SELECT * FROM policies", self.conn)

    def rank_new_rules(self):
        df = pd.read_sql("SELECT * FROM policies", self.conn, index_col="id")
        df["rank"] = df["score"].rank(ascending=False)
        for idx, row in df.iterrows():
            self.conn.execute(
                "UPDATE policies SET rank=? WHERE id=?", (float(
                    row["rank"]), idx)
            )
        self.conn.commit()





class ChatMemoryManager:
    """
    Mengelola memori percakapan, dengan analisis emosi 5-JALUR menggunakan ELECTRA
    untuk skalabilitas dan efisiensi.
    """

    def __init__(self, path: Path = Path("chat_memory.jsonl"), max_tokens: int = 4096):
        self.path, self.max_tokens = path, max_tokens
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        self._tacit_insights = []

        logger.info("Memuat model analisis untuk ChatMemoryManager...")

        self.sentiment_analyzer = hf_pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
        )
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )


        emotion_model_name = "michellejieli/emotion_text_classifier"
        logger.info(
            f"Menginisialisasi model emosi 5-jalur dengan: {emotion_model_name}"
        )
        self.emotion_analyzer = hf_pipeline(
            "text-classification", model=emotion_model_name, return_all_scores=True
        )
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(
            emotion_model_name)


        self._tacit_classifier = hf_pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        self.tacit_tokenizer = self._tacit_classifier.tokenizer

        logger.info("✅ Model analisis ChatMemoryManager siap.")

        self.message_count = 0
        self.policy_engine = PolicyEngine()

    def _now(self):
        return _dt.datetime.utcnow().isoformat()

    def _get_aggregated_sentiment(self, text: str) -> dict:
        model_max_length = 512
        chunk_size = 500
        overlap = 50
        tokens = self.sentiment_tokenizer.encode(text)
        if len(tokens) <= model_max_length:
            return self.sentiment_analyzer(text)[0]

        text_chunks = [
            self.sentiment_tokenizer.decode(
                tokens[i: i + chunk_size], skip_special_tokens=True
            )
            for i in range(0, len(tokens), chunk_size - overlap)
        ]
        if not text_chunks:
            return {"label": "3 stars", "score": 0.0}

        scores = []
        score_map = {
            "1 star": -2,
            "2 stars": -1,
            "3 stars": 0,
            "4 stars": 1,
            "5 stars": 2,
        }
        results = self.sentiment_analyzer(text_chunks)
        for result in results:
            scores.append(score_map.get(result["label"], 0) * result["score"])

        avg_score = sum(scores) / len(scores) if scores else 0
        final_label = "3 stars"
        if avg_score > 0.75:
            final_label = "5 stars"
        elif avg_score > 0.25:
            final_label = "4 stars"
        elif avg_score < -0.75:
            final_label = "1 star"
        elif avg_score < -0.25:
            final_label = "2 stars"
        return {"label": final_label, "score": avg_score}

    def _get_multi_path_emotion_analysis(self, text: str, num_paths: int = 5) -> dict:
        """Menganalisis emosi menggunakan N jalur Electra dan menggabungkan hasilnya."""
        model_max_length = 512
        tokens = self.emotion_tokenizer.encode(text)

        if len(tokens) <= model_max_length:
            results = self.emotion_analyzer(text)

            results = results[0] if isinstance(
                results, list) and results else []
            return {e["label"].lower(): e["score"] for e in results} if results else {}

        logger.info(
            f"Teks emosi panjang ({len(tokens)} token), menggunakan {num_paths}-jalur Electra..."
        )

        chunk_size = (len(tokens) + num_paths - 1) // num_paths
        text_chunks = [
            self.emotion_tokenizer.decode(
                tokens[i: i + chunk_size], skip_special_tokens=True
            )
            for i in range(0, len(tokens), chunk_size)
        ]

        if not text_chunks:
            return {}

        all_results = self.emotion_analyzer(text_chunks)

        final_scores = {}
        for result_list in all_results:
            for emotion_data in result_list:
                label = emotion_data["label"].lower()
                final_scores[label] = max(
                    final_scores.get(label, 0.0), emotion_data["score"]
                )

        return final_scores

    def add(self, role: str, content: str, meta: dict | None = None):
        """Menambahkan pesan baru ke memori, menganalisisnya, dan menyimpannya."""
        ts = self._now()

        sentiment = self._get_aggregated_sentiment(content)
        emo_scores = self._get_multi_path_emotion_analysis(
            content, num_paths=5)

        tacit_tokens = self.tacit_tokenizer.encode(
            content, max_length=1024, truncation=True
        )
        truncated_for_tacit = self.tacit_tokenizer.decode(
            tacit_tokens, skip_special_tokens=True
        )
        zs = self._tacit_classifier(
            truncated_for_tacit,
            candidate_labels=["advice", "insight", "feeling"],
            multi_label=False,
        )

        tacit_label = zs["labels"][0] if zs["scores"][0] > 0.5 else None
        explicit_flag = bool(
            re.search(r"\b(angka|data|hasil)\b", content.lower()))
        tacit_flag = bool(tacit_label)

        rec = {
            "ts": ts,
            "role": role,
            "content": content,
            "sentiment": sentiment,
            "emotions": emo_scores,
            "explicit": explicit_flag,
            "tacit": tacit_flag,
            "tacit_label": tacit_label or "",
            "meta": meta or {"brand": "Oracle Gem"},
        }

        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        try:
            with closing(sqlite3.connect("chat_archive.db")) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS chats (ts TEXT, role TEXT, content TEXT, sentiment_score REAL, emotions TEXT, explicit INT, tacit INT, tacit_label TEXT)"
                )
                conn.execute(
                    "INSERT INTO chats VALUES (?,?,?,?,?,?,?,?)",
                    (
                        ts,
                        role,
                        content,
                        float(sentiment.get("score", 0.0)),
                        json.dumps(emo_scores),
                        int(explicit_flag),
                        int(tacit_flag),
                        tacit_label or "",
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Gagal menulis ke database chat_archive: {e}")

        if tacit_flag:
            self._tacit_insights.append(tacit_label)


        self.message_count += 1
        if self.message_count % 100 == 0:
            logger.info(
                f"Mencapai {self.message_count} pesan, menjalankan analisis kebijakan..."
            )
            self.policy_engine.analyze_patterns()
            self.policy_engine.rank_new_rules()

    def fetch(self, limit: int = 200) -> list[dict]:
        try:
            with self._lock, self.path.open("r", encoding="utf-8") as f:
                lines = f.readlines()[-limit:]
            return [json.loads(l) for l in lines]
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Gagal membaca memori chat: {e}")
            return []

    def dump_tacit_insights(self, path: Path):
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write("# Tacit Insights\n\n")
                for i, s in enumerate(self._tacit_insights, 1):
                    f.write(f"{i}. {s}\n")
            logger.info(f"Tacit insights disimpan ke: {path}")
        except IOError as e:
            logger.error(f"Gagal menyimpan tacit insights: {e}")



class DistributedAIPool:
    """
    Mengelola koneksi khusus untuk Google Gemini API dengan strategi rolling key.
    """

    def __init__(self, api_config: dict, llm_lock: threading.Lock):
        self.pools = {}
        self.worker_pools = {}
        self.instructor_pools = {}
        self.key_indices = {}
        self.role_to_pool_map = {}
        self.llm_lock = llm_lock


        if "gemini_pool" in api_config:
            gemini_pool_config = api_config["gemini_pool"]
            pool_name = "gemini_pool"
            self.pools[pool_name] = gemini_pool_config.get("keys", [])
            self.key_indices[pool_name] = 0
            self.worker_pools[pool_name] = []
            self.instructor_pools[pool_name] = []

            for key in self.pools[pool_name]:
                try:

                    worker = ChatGoogleGenerativeAI(
                        model=gemini_pool_config["model"], google_api_key=key, convert_system_message_to_human=True)
                    self.worker_pools[pool_name].append(worker)


                    genai.configure(api_key=key, transport="rest")
                    gemini_client = genai.GenerativeModel(
                        model_name=gemini_pool_config["model"])
                    instructor_worker = instructor.from_gemini(
                        gemini_client, mode=instructor.Mode.GEMINI_JSON)
                    self.instructor_pools[pool_name].append(instructor_worker)
                except Exception as e:
                    logger.warning(
                        f"Gagal inisialisasi salah satu kunci Gemini: {e}")


        for role, config in api_config.items():
            if role == "gemini_pool":
                continue



            if "pool" in config:
                self.role_to_pool_map[role] = config["pool"]
                logger.info(
                    f"Peran '{role}' dipetakan ke pool '{config['pool']}'.")


            elif "key" in config and "model" in config:
                pool_name = f"single_pool_{role}"
                self.pools[pool_name] = [config["key"]]
                self.key_indices[pool_name] = 0

                self.role_to_pool_map[role] = pool_name

                try:
                    if "meta-llama" in config["model"] or "Qwen" in config["model"]:
                        worker = TogetherLLM(
                            api_key=config["key"], model_name=config["model"])
                        self.worker_pools[pool_name] = [worker]
                        logger.info(
                            f"Worker Together.AI untuk peran '{role}' ({config['model']}) telah dibuat.")

                except Exception as e:
                    logger.error(
                        f"Gagal membuat worker untuk peran '{role}': {e}")


        logger.info("✅ DistributedAIPool (Hybrid) siap.")

    def get_random_gemini_worker(self, worker_type="langchain"):
        """Mengambil worker Gemini acak dari pool untuk tugas-tugas ringan."""
        pool_to_use = self.instructor_pools if worker_type == "instructor" else self.worker_pools
        pool = pool_to_use.get("gemini_pool", [])
        if pool:
            return random.choice(pool)
        return None

        active_index = self.key_indices.get(pool_name, 0)

        pool_to_use = (
            self.instructor_pools if worker_type == "instructor" else self.worker_pools
        )
        pool = pool_to_use.get(pool_name, [])

        if not pool or active_index >= len(pool):
            return None

        return pool[active_index]

    def get_active_worker(self, pool_name: str, worker_type="langchain"):
        """Mengambil worker yang aktif saat ini dari pool berdasarkan indeks."""
        active_index = self.key_indices.get(pool_name, 0)

        pool_to_use = (
            self.instructor_pools if worker_type == "instructor" else self.worker_pools
        )
        pool = pool_to_use.get(pool_name, [])

        if not pool or active_index >= len(pool):
            return None

        return pool[active_index]

    def roll_to_next_key(self, pool_name: str):
        """Mengganti indeks kunci aktif ke kunci berikutnya dalam pool."""
        if pool_name not in self.pools:
            return

        num_keys = len(self.pools[pool_name])
        if num_keys <= 1:
            return

        current_index = self.key_indices.get(pool_name, 0)
        next_index = (current_index + 1) % num_keys
        self.key_indices[pool_name] = next_index

        logger.warning(
            f"  [ROLLING-KEY] Kuota API untuk kunci #{current_index + 1} di pool '{pool_name}' mungkin habis."
        )
        logger.info(
            f"  [ROLLING-KEY] Beralih ke kunci API berikutnya: #{next_index + 1}."
        )

        time.sleep(2)

    def invoke_with_rolling_strategy(
        self,
        role: str,
        prompt: str,
        tools: list = None,
        response_model: Optional[type[BaseModel]] = None,
    ):
        """Mengeksekusi panggilan AI dengan strategi rolling key dan concurrency lock."""
        if not self.llm_lock:
            raise RuntimeError(
                "LLM Lock belum diinisialisasi oleh AsyncCuriosityEngine.")


        with self.llm_lock:

            pool_name = self.role_to_pool_map.get(role)
        if not pool_name or pool_name not in self.pools:
            raise ValueError(
                f"Peran '{role}' tidak terhubung ke pool API yang valid.")

        num_keys = len(self.pools[pool_name])
        if num_keys == 0:
            raise RuntimeError(
                f"Tidak ada kunci API yang tersedia di pool '{pool_name}' untuk peran '{role}'."
            )

        for attempt in range(num_keys):
            worker_type = "instructor" if response_model else "langchain"
            active_worker = self.get_active_worker(pool_name, worker_type)

            if not active_worker:
                self.roll_to_next_key(pool_name)
                continue

            try:



                time.sleep(10)


                key_index_for_log = self.key_indices.get(pool_name, 0) + 1
                logger.info(
                    f"  > Mencoba peran '{role}' dengan kunci #{key_index_for_log}/{num_keys} dari pool '{pool_name}'..."
                )

                if response_model:
                    response = active_worker.create(
                        response_model=response_model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                else:
                    invoke_kwargs = {}
                    if tools:
                        invoke_kwargs["tools"] = tools
                    response = active_worker.invoke(prompt, **invoke_kwargs)

                return response

            except (
                ResourceExhausted,
                InstructorRetryException,
                ServiceUnavailable,
            ) as e:
                self.roll_to_next_key(pool_name)
                if attempt == num_keys - 1:
                    logger.error(
                        f"FATAL: Semua {num_keys} kunci API di pool '{pool_name}' telah mencapai batas kuota."
                    )
                    raise e

            except Exception as e:
                logger.error(
                    f"Terjadi error non-kuota saat menggunakan kunci #{key_index_for_log}: {type(e).__name__} - {e}"
                )
                raise e

        raise RuntimeError(
            f"Semua kunci di pool '{pool_name}' gagal dieksekusi setelah satu siklus penuh."
        )

    def call_gemini_for_text(self, prompt: str, agent_name: str) -> str:
        """Helper untuk memanggil AI dan mendapatkan output teks, menggunakan strategi rolling."""
        response = self.invoke_with_rolling_strategy(
            role=agent_name, prompt=prompt)
        return response.content if hasattr(response, "content") else str(response)

    def call_gemini_with_tool(
        self, prompt: str, agent_name: str, tool_schema: type[BaseModel]
    ) -> dict:
        """Helper untuk memanggil AI dengan tool/skema, menggunakan strategi rolling."""
        response = self.invoke_with_rolling_strategy(
            role=agent_name, prompt=prompt, response_model=tool_schema
        )
        return response.model_dump() if hasattr(response, "model_dump") else response

    def get_worker(self, agent_name: str) -> Optional[ChatGoogleGenerativeAI]:
        """Mendapatkan worker LangChain yang aktif saat ini untuk kompatibilitas."""
        pool_name = self.role_to_pool_map.get(agent_name)
        if not pool_name:
            single_pool_name = f"single_pool_{agent_name}"
            if single_pool_name in self.worker_pools:
                pool_name = single_pool_name
            else:
                logger.warning(
                    f"Tidak ada pool yang terpetakan untuk peran '{agent_name}'."
                )
                return None
        return self.get_random_gemini_worker("langchain")


class GeminiChatEngine:
    def __init__(
        self,
        api_pool: DistributedAIPool,
        memory: ChatMemoryManager,
        nsmm: "NSMM",
        brain: "Brain",
        embedding_model: "APIEmbedder",
        agent_name: str = "chat_interactive",
        direct_chat_agent: Optional[Any] = None,
    ):
        self.api_pool = api_pool
        self.memory = memory
        self.nsmm = nsmm
        self.brain = brain
        self.embedding_model = embedding_model
        self.agent_name = agent_name
        self.tools = [
            get_latest_model_performance,
            list_available_models,
            submit_task_to_queue,
        ]

        self.direct_chat_agent = direct_chat_agent


        if not self.direct_chat_agent:
            worker_model = self.api_pool.get_worker(self.agent_name)
            if not worker_model:
                raise ValueError(
                    f"Worker '{self.agent_name}' tidak ditemukan.")
            self.agent_with_tools = worker_model.bind_tools(self.tools)


        self.indobert_model = None
        self.indobert_tokenizer = None
        try:
            logger.info("🧠 Memuat model fallback offline (IndoBERT)...")
            model_name = "indobenchmark/indobert-base-p1"
            self.indobert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.indobert_model = AutoModelForMaskedLM.from_pretrained(
                model_name)
            self.indobert_model.eval()
            logger.info("✅ Model fallback IndoBERT siap.")
        except Exception as e:
            logger.warning(
                f"⚠️ Gagal memuat IndoBERT. Fallback offline tidak akan tersedia. Error: {e}"
            )

    def _fallback_answer(self, user_msg: str) -> str:
        """Menjawab menggunakan pengetahuan internal (Brain/RAG) jika API gagal."""
        logger.warning("Menggunakan RAG internal sebagai fallback...")

        retrieved_chunks = self.brain.query(user_msg, k=3)

        if not retrieved_chunks:
            return "Maaf, semua koneksi API sedang bermasalah dan saya tidak dapat menemukan informasi relevan di memori internal saya."

        context = "\n---\n".join(retrieved_chunks)



        return f"Semua koneksi API sedang gagal. Berdasarkan memori internal saya, berikut informasi yang relevan:\n\n{context}"

    def ask(self, user_msg: str) -> str:
        """
        Versi Hibrida: Mencoba menjawab dengan Grok (jika ada), lalu Gemini,
        dan terakhir fallback ke RAG internal.
        """
        self.memory.add("user", user_msg)
        final_answer = ""

        try:

            if self.direct_chat_agent:
                logger.info(
                    "Mencoba menjawab menggunakan agen chat langsung (Grok)...")
                final_answer = self.direct_chat_agent.chat(user_msg)

            else:

                logger.info(
                    "Mencoba menjawab menggunakan LLM online (Gemini)...")
                from langchain_core.prompts import ChatPromptTemplate

                agent_executor = AgentExecutor(
                    agent=agent, tools=self.tools, verbose=False)
                response = agent_executor.invoke({"input": user_msg})
                final_answer = response.get(
                    "output", "Maaf, terjadi kesalahan.")

        except Exception as e:
            logger.error(
                f"FATAL: Agen online gagal. Mengaktifkan fallback. Error: {e}")
            final_answer = self._fallback_answer(user_msg)

        self.memory.add("assistant", final_answer)
        return final_answer


def execute_with_thinking_animation(func: Callable, *args, **kwargs) -> Any:
    """Wrapper untuk menjalankan fungsi dengan animasi 'berpikir'."""
    message = kwargs.pop("message", "Model sedang berpikir...")
    status_logger = StatusLogger(message=message)
    status_logger.start()
    try:
        result = func(*args, **kwargs)
        status_logger.stop()
        return result
    except Exception as e:

        status_logger.stop(
            final_message=f"Terjadi error: {e}", final_emoji="❌")
        raise e


def execute_tool_calls(tool_calls: list, tools: list) -> list:
    """Helper untuk mengeksekusi daftar panggilan tool dari LangChain."""
    tool_map = {t.name: t.func for t in tools}
    outputs = []
    for call in tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name](**tool_args)
                outputs.append({"tool_name": tool_name, "output": result})
            except Exception as e:
                logger.error(f"Gagal menjalankan alat {tool_name}: {e}")
                outputs.append(
                    {"tool_name": tool_name, "output": {"error": str(e)}})
    return outputs


class StructuredForecastingAgent:
    """
    Agen yang meniru alur kerja 'Chain of Thought' untuk membuat prediksi harga,
    menggunakan alat-alat (tools) yang diberikan untuk mengumpulkan data nyata.
    """

    def __init__(self, api_pool: DistributedAIPool, tickers: list, tools: list):
        self.api_pool = api_pool
        self.tickers = tickers
        self.tools = tools
        self.planner_agent = api_pool.get_worker("experimentalist")

    def _run_planner_and_executor(self) -> dict:
        """Langkah 1 & 2: Membuat rencana, memutuskan alat, dan langsung mengeksekusinya."""
        logger.info(
            "🧠 [Planner] Menganalisis permintaan dan membuat rencana eksekusi..."
        )
        prompt = f"""
        Anda adalah seorang manajer investasi. Saya butuh prediksi harga penutupan untuk ticker: {', '.join(self.tickers)}.
        Rencana Anda: Panggil alat `get_technical_analysis` untuk SETIAP ticker. Kemudian, panggil alat `get_market_sentiment` SEKALI.
        """
        if not self.planner_agent:
            logger.error("Planner agent tidak tersedia.")
            return {"error": "Planner agent tidak tersedia."}

        response = self.planner_agent.invoke(prompt, tools=self.tools)
        tool_calls = response.tool_calls

        logger.info(
            f"🛠️ [Executor] Rencana dibuat. Mengeksekusi {len(tool_calls)} alat..."
        )
        tool_outputs = execute_tool_calls(tool_calls, self.tools)
        return {"data": tool_outputs}

    def _run_synthesizer(self, original_prompt: str, collected_data: dict) -> str:
        """Langkah 3 & 4: Mensintesis hasil dan membuat narasi prediksi akhir."""
        logger.info(
            "✍️ [Synthesizer] Menganalisis semua data dan menyusun narasi...")
        synthesizer_agent = self.api_pool.get_worker("advanced_advisor")

        context = f"""
        Tugas: Anda adalah seorang Analis Keuangan Senior. Berikan analisis dan prediksi harga penutupan untuk {', '.join(self.tickers)}.
        Gunakan data yang telah dikumpulkan dari tim analis junior Anda di bawah ini sebagai dasar utama analisis.

        Pertanyaan Asli Pengguna: "{original_prompt}"

        Data Internal yang Telah Dikumpulkan:
        ```json
        {json.dumps(collected_data, indent=2, default=str)}
        ```

        Tugas Anda:
        1. Interpretasikan data teknikal dan sentimen untuk setiap ticker.
        2. Gabungkan semua informasi menjadi sebuah narasi yang koheren.
        3. Berikan kesimpulan akhir berupa prediksi (misalnya: "cenderung naik", "konsolidasi dengan bias bearish") untuk setiap ticker, LENGKAP DENGAN ALASAN yang kuat berdasarkan data yang diberikan.
        """

        if not synthesizer_agent:
            return "Synthesizer agent tidak tersedia."

        final_response = synthesizer_agent.invoke(context)
        return final_response.content

    def run_full_analysis(self):
        """Menjalankan seluruh alur kerja prediksi dari awal hingga akhir dengan refinement."""
        original_prompt = f"Berikan saya prediksi harga penutupan dan analisis naratif untuk {', '.join(self.tickers)}."

        collected_data = execute_with_thinking_animation(
            self._run_planner_and_executor,
            message="Agen sedang merencanakan & mengeksekusi alat...",
        )
        if "error" in collected_data:
            return collected_data["error"]


        initial_narrative = execute_with_thinking_animation(
            self._run_synthesizer,
            original_prompt,
            collected_data,
            message="Agen sedang mensintesis laporan awal...",
        )


        logger.info(
            "\n" + "=" * 20 +
            " MEMULAI TAHAP REFINEMENT (MEMORY-AUGMENTED) " + "=" * 20
        )


        feedback_mgr = HumanFeedbackManager(
            get_path(None, "human_feedback_db"))
        anomaly_mem = AnomalyMemory(
            db_path=MODELS_DIR / "anomaly_memory.sqlite")

        refinement_engine = IterativeRefinementEngine(
            api_pool=self.api_pool,
            feedback_manager=feedback_mgr,
            anomaly_memory=anomaly_mem,
        )


        final_refined_narrative = refinement_engine.refine(
            initial_text=initial_narrative,
            original_context=original_prompt,
            max_iterations=1,
        )

        return final_refined_narrative


class DesktopNavigator:
    """
    Agen offline untuk menavigasi desktop menggunakan tombol panah,
    mempelajari tata letak ikon, dan menyimpannya ke NSMM.
    """

    def __init__(self, nsmm: 'NSMM'):
        self.nsmm = nsmm



        self.selection_text_area = (50, 50, 800, 800)

    def _get_selected_app_name(self) -> Optional[str]:
        """
        Secara cepat mengambil screenshot dari area yang relevan dan menggunakan OCR
        untuk membaca nama aplikasi yang sedang disorot (selected).
        """
        try:

            with mss.mss() as sct:
                monitor = {"top": self.selection_text_area[1], "left": self.selection_text_area[0],
                           "width": self.selection_text_area[2] - self.selection_text_area[0],
                           "height": self.selection_text_area[3] - self.selection_text_area[1]}
                img = sct.grab(monitor)
                img_pil = Image.frombytes(
                    "RGB", img.size, img.bgra, "raw", "BGRX")



            config = r'--psm 7'
            text = pytesseract.image_to_string(img_pil, config=config).strip()

            return ''.join(filter(str.isalnum, text)) if text else None
        except Exception as e:
            logger.error(
                f"[Navigator] Gagal membaca nama aplikasi terpilih: {e}")
            return None

    def learn_move(self, start_app: str, action: str):
        """Mempelajari satu pergerakan dan menyimpannya ke database."""
        pyautogui.press(action)
        time.sleep(0.2)
        end_app = self._get_selected_app_name()

        if end_app and start_app != end_app:
            logger.info(
                f"🧠 [Navigator Learns] Aksi '{action}' dari '{start_app}' menuju ke '{end_app}'.")
            with sqlite3.connect(self.nsmm.db_path) as conn:
                conn.execute("""
                    INSERT INTO desktop_navigation_graph (start_app, action, end_app)
                    VALUES (?, ?, ?)
                    ON CONFLICT(start_app, action) DO UPDATE SET
                    end_app = excluded.end_app,
                    success_count = success_count + 1
                """, (start_app, action, end_app))
                conn.commit()

    def find_path_to_app(self, target_app: str) -> Optional[List[str]]:
        """Mencari jalur (urutan tombol) terpendek ke aplikasi target dari memori."""


        logger.info(
            f"🗺️ [Navigator] Mencari jalur di memori ke '{target_app}'...")






        if "chrome" in target_app.lower():

            return ['home', 'right', 'right']
        return None

    def run_autonomous_agent_cycle(
        nsmm: "NSMM",
        brain: "Brain",
        api_pool: "DistributedAIPool",
        current_instruction: str
    ):
        """
        Menjalankan satu siklus penuh dari agen otonom dengan arsitektur hibrida cerdas.
        Urutan: Pengamat -> Penerjemah -> Perencana (dengan Memori) -> Aksi -> Refleksi & Belajar.
        """
        execution_log = ["[START]"]

        try:

            execution_log.append("Melihat & memahami layar...")
            logger.info(
                "    ➡️  [PERCEIVE] Memanggil persepsi elemen desktop hibrida...")

            desktop_elements = perceive_desktop_elements.func()

            if not desktop_elements:
                structured_context = "Layar saat ini kosong atau tidak ada elemen yang dapat dikenali."
            else:
                context_parts = ["Layar saat ini berisi elemen berikut:"]
                for el in desktop_elements:
                    text_info = f"dengan nama '{el['app_name']}'" if el.get(
                        'text_box') else "tanpa nama teks yang jelas"
                    context_parts.append(
                        f"- Ikon untuk '{el['app_name']}' {text_info}.")
                structured_context = " ".join(context_parts)

            logger.info(
                f"    ▶️  [TRANSLATOR] Situasi dipahami sebagai: '{structured_context}'")
            execution_log.append(f"Memahami situasi: {structured_context}")


            logger.info(
                "    🧠 [PLAN] Mencari pengalaman serupa di memori (NSMM)...")
            similar_experiences = nsmm.query_similar_experiences(
                structured_context, _supremacy_embed_model, top_k=1
            )
            if similar_experiences and "berhasil" in similar_experiences[0].lower():
                logger.info(
                    "    💡 [PLAN] Ditemukan pengalaman sukses yang serupa. Ini akan menjadi pertimbangan.")

            experience_text = similar_experiences[0] if similar_experiences else 'Tidak ada'

            action_planner_prompt = f"""
            Anda adalah otak eksekutif AI otonom yang sangat presisi.
            Tujuan Utama: "{current_instruction}"
            Kondisi Layar Saat Ini: "{structured_context}"
            Data Elemen Desktop(JSON): {json.dumps(desktop_elements)}
            Pengalaman Sukses Serupa dari Memori: "{experience_text}"

            Berdasarkan tujuan dan data, buatlah rencana aksi GUI.
            ATURAN KETAT:
            1.  Rencana Anda HANYA boleh berisi `action_type` dari daftar berikut: 'click', 'type', 'move', 'scroll', 'wait', 'press'.
            2.  Untuk aksi 'press', gunakan field `key_to_press` untuk menentukan tombolnya(misal: 'right', 'down', 'enter').
            3.  Prioritaskan klik pada `text_box` jika tersedia. Gunakan `icon_box` sebagai cadangan.
            4.  Jika target tidak terlihat, gunakan aksi 'press' untuk menjelajah.
            5.  Panggil HANYA alat `execute_gui_plan` dengan rencana Anda. JANGAN panggil alat persepsi di dalam rencana.
            """

            planner_agent = api_pool.get_worker("qwen_giant_planner")
            if not planner_agent:
                raise RuntimeError("Agen perencana tidak tersedia.")

            available_tools = [execute_gui_plan, perceive_desktop_elements]
            execution_log.append("Membuat rencana...")
            response = planner_agent.invoke(
                action_planner_prompt, tools=available_tools)

            tool_calls = response.tool_calls
            if not tool_calls:
                logger.warning(
                    "    ⚠️ [PLAN] Perencana tidak menghasilkan rencana. Melewati siklus.")
                return None

            action_plan_args = tool_calls[0]['args']
            execution_log.append(f"Rencana dibuat: {action_plan_args}")

            if not isinstance(action_plan_args, dict) or 'actions' not in action_plan_args or not action_plan_args['actions']:
                raise ValueError(
                    f"Rencana dari LLM tidak valid: {action_plan_args}")


            execution_log.append("Mengeksekusi rencana...")
            logger.info(
                f"    ➡️  [ACTION] Akan mengeksekusi rencana GUI: {action_plan_args}")
            execution_result = execute_gui_plan.func(**action_plan_args)
            logger.info(
                f"    ✅ [ACTION] Eksekusi selesai dengan hasil: {execution_result}")
            execution_log.append(f"Hasil: {execution_result}")


            logger.info(
                "    ➡️  [REFLECT] Mengambil persepsi baru setelah aksi untuk belajar...")
            new_perception_elements = perceive_desktop_elements.func()
            new_perception_ocr = perceive_screen.func()

            experience_data = {
                "metrics": {"task": "autonomous_learning", "result": execution_result},
                "state_vector": {
                    "screen_before": structured_context,
                    "screen_after": f"Ditemukan {len(new_perception_elements)} elemen setelah aksi."
                },
                "diagnosis": f"Mencoba mencapai tujuan '{current_instruction}' dengan rencana: {action_plan_args}",
                "final_decision": {"action_taken": action_plan_args, "outcome": execution_result},
            }
            nsmm.log_epoch_experience(nsmm.session_id, -1, experience_data)
            logger.info(
                "    💾 [REFLECT] Pengalaman siklus telah dicatat di NSMM.")

            if "ocr_text" in new_perception_ocr and new_perception_ocr["ocr_text"]:
                chunks = chunk_text(new_perception_ocr["ocr_text"], 400, 50)
                brain.add_chunks(
                    chunks, source_name=f"AutonomousLearning_{datetime.now().strftime('%Y%m%d_%H%M')}")
                logger.info(
                    f"    🧠 [REFLECT] {len(chunks)} potongan teks baru dari hasil aksi ditambahkan ke Brain.")

            if "berhasil" in execution_result.lower():
                for action in action_plan_args.get('actions', []):
                    if action.get('action_type') == 'click' and action.get('coordinates'):
                        for element in desktop_elements:
                            coords_to_check = element.get(
                                'text_box') or element.get('icon_box')
                            if not coords_to_check:
                                continue

                            center_x = (
                                coords_to_check[0] + coords_to_check[2]) / 2
                            center_y = (
                                coords_to_check[1] + coords_to_check[3]) / 2
                            action_coords = action['coordinates']

                            if abs(center_x - action_coords[0]) < 20 and abs(center_y - action_coords[1]) < 20:
                                nsmm.update_icon_memory(
                                    element['app_name'], action_coords, was_successful=True)
                                execution_log.append(
                                    f"Memperkuat memori untuk elemen {element['app_name']}.")

            logger.info(
                "✅ [AUTONOMOUS] Siklus otonom selesai. Stream Kesadaran: " + " -> ".join(execution_log))

            return action_plan_args

        except Exception as e:
            execution_log.append(f"[ERROR] {type(e).__name__}: {str(e)}")
            logger.error(
                f"❌ Error pada siklus otonom. Stream Kesadaran: {' -> '.join(execution_log)}", exc_info=True)
            return None



    def execute_reflexive_action(nsmm: NSMM, vector_encoder: UniversalVectorEncoder, goal_instruction: str) -> Optional[str]:
        """
        Mencoba menemukan & mengeksekusi rencana aksi yang sudah dipelajari dari memori refleks.
        Mengembalikan pesan sukses jika berhasil, None jika tidak ada refleks yang cocok.
        """
        goal_vector = vector_encoder.encode(goal_instruction)
        if goal_vector is None:
            return None


        action_plan = nsmm.find_similar_reflex(goal_vector)

        if action_plan:

            try:
                result = execute_gui_plan.func(**action_plan)
                if "berhasil" in result.lower():
                    return f"Aksi refleks '{goal_instruction}' berhasil dieksekusi secara instan."
                else:
                    logger.warning(f"Eksekusi refleks gagal: {result}")
                    return None
            except Exception as e:
                logger.error(f"Error saat eksekusi refleks: {e}")
                return None


        return None


class AIOperationsOrchestrator:
    """Mengelola dan menjalankan semua mode operasi utama dengan alur kerja CoT."""

    def __init__(
        self,
        api_pool: DistributedAIPool,
        auditor: CriticalAuditor,
        initial_hparams: dict,
        gemini_api_config: dict,
        together_api_keys: dict,
        together_roles: dict,
        brain: "Brain",
        nsmm: "NSMM",
    ):
        self.api_pool = api_pool
        self.auditor = auditor
        self.initial_hparams = initial_hparams
        self.brain = brain
        self.nsmm = nsmm
        self.gemini_api_config = gemini_api_config
        self.together_api_keys = together_api_keys
        self.together_roles = together_roles
        self.planner_agent = self.api_pool.get_worker("experimentalist")

    def _synthesize_plan(self, context: str) -> str:
        """Meminta AI untuk merangkum rencana dan hasil alat menjadi narasi."""
        synthesis_role = "advanced_advisor"
        prompt = f"""
        Anda adalah Oracle, seorang AI supercerdas yang memimpin proyek analisis keuangan.
        Berdasarkan konteks dan data teknis di bawah, jelaskan kepada pengguna(seorang manajer investasi)
        apa yang akan Anda lakukan selanjutnya dalam bahasa yang jelas dan profesional.

        Konteks & Data Teknis:
        ---
        {context}
        ---

        Laporan Anda untuk Pengguna:
        """
        response = self.api_pool.invoke_with_rolling_strategy(
            role=synthesis_role, prompt=prompt
        )
        return response.content

def run_pre_train(
    initial_hparams: dict,
    auditor: "CriticalAuditor",
    api_pool: "DistributedAIPool",
    together_api_keys: dict,
    gemini_api_config: dict,
    brain: "Brain",
    nsmm: "NSMM",
    web_searcher: "WebSearchManager",
    **kwargs
):
    """
    Mengorkestrasi alur kerja pre-train dengan mekanisme rekomendasi
    sekuensial antar uji tuntas menggunakan memori virtual.
    """
    logger.info(
        "\n" + "=" * 20 + " MODE PRE-TRAIN: GUARDRAIL & UJI TUNTAS ACS " + "=" * 20
    )


    logger.info("\n--- [TAHAP A] Validasi Kesiapan Sistem ---")
    
    ram_available = psutil.virtual_memory().available / (1024**3)
    data_path = initial_hparams['data_path']


    readiness_result = check_system_readiness.func(
        data_path=data_path,
        ram_gb_available=ram_available
    )
    

    readiness_check = SystemReadinessCheck(**readiness_result)

    if not readiness_check.is_sufficient:
        logger.error(
            f"Pelatihan dibatalkan karena sistem tidak siap: {readiness_check.reasoning}"
        )
        return

    logger.info(f"✅ Sistem Siap: {readiness_check.reasoning}")


    logger.info(
        "\n--- [TAHAP B] Memulai Uji Tuntas Sekuensial dengan Memori ---")
    num_trials = 2
    trial_results = []
    recommended_overrides = {}
    blacklist_path = None
    armh_cb_instance = None

    causal_auditor = CausalInternalAuditor(
        project_id=initial_hparams["project_id"]
    )

    for i in range(num_trials):
        logger.info(f"\n--- [UJI TUNTAS #{i+1}/{num_trials}] ---")

        try:
            hparams_trial = initial_hparams.copy()

            if recommended_overrides:
                logger.warning(
                    f"Menerapkan rekomendasi dari AI Supervisor dari putaran sebelumnya: {recommended_overrides}"
                )
                hparams_trial.update(recommended_overrides)

            hparams_trial["attempt"] = f"acs_trial_{i+1}"
            hparams_trial["causal_auditor"] = causal_auditor

            acs_callback = AnticipatoryCognitiveSupervisor(
                api_pool, together_api_keys
            )

            armh_cb_instance = ARMH_Callback(
                api_pool, together_api_keys)
            all_callbacks = [acs_callback, armh_cb_instance]
            if blacklist_path:
                all_callbacks.append(BlacklistLRScheduler(blacklist_path))

            nsmm_instance = nsmm

            score, full_metrics, ckpt_path, data_summary, trainer_instance = (
                _one_train_run(
                    hparams_trial,
                    auditor,
                    api_pool,
                    together_api_keys,
                    gemini_api_config,
                    web_searcher=web_searcher,
                    governor=CognitiveGovernor(
                        hparams_trial["project_id"], api_pool
                    ),
                    brain=brain,
                    nsmm=nsmm_instance,
                    custom_callbacks=all_callbacks,
                    blacklist_path=blacklist_path,
                )
            )

            trial_results.append(
                {
                    "score": score,
                    "metrics": full_metrics,
                    "checkpoint": ckpt_path,
                    "hparams": hparams_trial,
                }
            )

            if i == 0 and armh_cb_instance and armh_cb_instance.global_problem_log:
                logger.info(
                    "\n--- [ANALISIS ANTAR-SESI] Menganalisis log ARMH dari uji tuntas pertama... ---"
                )
                blacklist_path = analyze_and_blacklist_batches(
                    armh_log=armh_cb_instance.global_problem_log,
                    project_id=initial_hparams["project_id"],
                    threshold=2,
                )
                if blacklist_path:
                    logger.info(
                        f"✅ Daftar hitam dibuat. Akan digunakan pada uji tuntas berikutnya."
                    )

            logger.info(
                f"--- Uji Tuntas #{i+1} Selesai dengan Skor Akhir: {score:.6f} ---"
            )

            if hasattr(trainer_instance, "next_run_recommendations"):
                recommended_overrides = getattr(
                    trainer_instance, "next_run_recommendations"
                )
            else:
                recommended_overrides = {}

        except Exception as e:
            logger.error(
                f"💥 UJI TUNTAS #{i+1} GAGAL TOTAL. Melanjutkan ke putaran berikutnya. Error: {e}",
                exc_info=True,
            )
            continue


    if not trial_results:
        logger.error("Tidak ada hasil trial yang valid. Pelatihan gagal.")
        return

    best_trial = min(trial_results, key=lambda x: x["score"])
    logger.info("\n🏆 Uji Tuntas Selesai. Hasil Terbaik Dipilih.")
    logger.info(f"    - Skor Loss Terbaik: {best_trial['score']:.6f}")
    logger.info(f"    - Checkpoint Juara: {best_trial['checkpoint']}")

    if best_trial["checkpoint"] and os.path.exists(best_trial["checkpoint"]):
        with open(LAST_PROJECT_INFO_TXT, "w") as f:
            f.write(
                f"{best_trial['hparams']['project_id']},{best_trial['checkpoint']}"
            )
        logger.info(
            f"✅ Info model juara pre-train disimpan ke: {LAST_PROJECT_INFO_TXT}"
        )

    def run_fine_tune(self):
        logger.info("\n" + "=" * 20 +
                    " MODE FINE-TUNE TERSTRUKTUR " + "=" * 20)

        def run_orchestration():
            run_continuous_training(
                self.initial_hparams,
                self.auditor,
                self.api_pool,
                self.gemini_api_config,
                self.together_api_keys,
                self.together_roles,
                brain=self.brain,
            )

        execute_with_thinking_animation(
            run_orchestration,
            message="Menjalankan orkestrasi fine-tuning multi-agen...",
        )


class MasterPlannerAI:
    """
    Otak eksekutif otonom yang merumuskan rencana strategis untuk mencapai tujuan jangka panjang.
    """

    def __init__(self, api_pool: "DistributedAIPool", together_api_keys: dict):
        self.api_pool = api_pool

        self.agent_role = "qwen_giant_planner"


        planner_api_key = together_api_keys.get("qwen_giant")
        if not planner_api_key:
            raise ValueError(
                "API Key untuk 'qwen_giant' tidak ditemukan untuk MasterPlannerAI."
            )


        self.planner_agent = TogetherLLM(
            api_key=planner_api_key,
            model_name="Qwen/Qwen2-72B-Instruct",
        )
        logger.info(
            "🧠 [Master Planner] Diinisialisasi dengan agen khusus: Qwen2-72B-Instruct."
        )

    def _assess_task_feasibility(
        self, task_description: str, brain: Brain, full_dataset: pd.DataFrame
    ) -> dict:
        """Menganalisis apakah sistem memiliki data & pengetahuan untuk tugas yang diberikan."""
        logger.info(f"    -> Menilai kelayakan tugas: '{task_description}'...")

        entities = re.findall(r"\b[A-Z]{2,}[\-A-Z]*\b", task_description)

        if not entities:
            logger.warning(
                "       - Penilaian: REJECTED (Tidak ada aset teridentifikasi)"
            )
            return {
                "status": "REJECTED",
                "reason": "Tidak bisa mengidentifikasi aset spesifik dari permintaan.",
            }

        asset = entities[0]


        if f"{asset}_Close" not in full_dataset.columns:
            logger.warning(
                f"       - Penilaian: NEEDS_DATA (Data untuk {asset} tidak ditemukan)"
            )
            return {
                "status": "NEEDS_DATA",
                "reason": f"Saya tidak memiliki data historis untuk {asset}.",
            }


        related_knowledge = brain.query(
            f"prinsip atau risiko terkait {asset}", k=1)
        if not related_knowledge:
            logger.warning(
                f"       - Penilaian: NEEDS_LEARNING (Pengetahuan tentang {asset} minim)"
            )
            return {
                "status": "NEEDS_LEARNING",
                "reason": f"Saya punya datanya, tapi saya perlu waktu untuk mempelajari pola dan karakteristik unik dari {asset}.",
            }

        logger.info(
            f"       - Penilaian: READY (Data dan pengetahuan untuk {asset} tersedia)"
        )
        return {"status": "READY", "reason": "Siap dieksekusi."}

    def generate_plan(
        self,
        nsmm: "NSMM",
        governor: "CognitiveGovernor",
        history: list,
        task_queue: queue.Queue,
        df_full: pd.DataFrame,
        brain: "Brain",
        engine: "AsyncCuriosityEngine"
    ) -> Optional[ExecutionPlan]:
        logger.info("🧠 [Master Planner] Memulai siklus perencanaan dinamis...")

        active_goal = nsmm.get_active_goal()
        if not active_goal:
            logger.error(
                "[Master Planner] Tidak ada tujuan aktif ditemukan. Perencanaan ditunda.")
            return None

        user_tasks = []

        current_state = governor.state
        context = f"""
        **TUJUAN AKTIF SAAT INI: ** {json.dumps(active_goal, indent=2)}
        **KONDISI SISTEM SAAT INI: ** {json.dumps(current_state, indent=2, default=str)}
        **RIWAYAT TERAKHIR: ** {json.dumps(history[-2:], indent=2, default=str)}
        """


        pydantic_schema_str = json.dumps(
            ExecutionPlan.model_json_schema(), indent=2)
        prompt_for_qwen = f"""
        Anda adalah Oracle Strategist, AI perencana tingkat tinggi. Berdasarkan konteks, buat rencana kerja 1-2 langkah untuk mencapai TUJUAN AKTIF.
        ATURAN KETAT: Output Anda HARUS berupa objek JSON tunggal dan valid yang sesuai dengan skema 'ExecutionPlan'. Jangan tambahkan teks atau penjelasan lain.

        Konteks:
        ---
        {context}
        ---

        Skema JSON 'ExecutionPlan' yang harus diikuti:
        ```json
        {pydantic_schema_str}
        ```
        """

        try:

            logger.info(
                "  -> [Master Planner] Meminta rencana dari agen Qwen (Together.AI)...")
            response_str = self.planner_agent.chat(prompt_for_qwen)


            plan_obj = robust_json_extract(response_str, model=ExecutionPlan)

            if plan_obj:
                main_plan = plan_obj
                main_plan.tasks = user_tasks + main_plan.tasks
                logger.info(
                    "✅ [Master Planner] Rencana berhasil dibuat via panggilan langsung ke Qwen.")
                return main_plan
            else:
                logger.error(
                    f"[Master Planner] Gagal membuat rencana: Qwen tidak memberikan JSON yang valid. Respons: {response_str}")
                return None

        except Exception as e:
            logger.error(
                f"[Master Planner] Terjadi error tak terduga saat membuat rencana: {e}", exc_info=True)
            return None


class FailureAnalysisReport(BaseModel):
    """Skema untuk laporan post-mortem yang terstruktur."""

    suspected_root_cause: str = Field(
        description="Satu kalimat ringkas yang menyimpulkan akar masalah utama. Contoh: 'Learning rate terlalu agresif menyebabkan divergensi loss.'"
    )
    evidence_from_data: list[str] = Field(
        description="Daftar poin-poin bukti dari data yang mendukung kesimpulan. Contoh: ['Komentar supervisor menyebut prediksi 'terlalu liar'.', 'Metrik val_loss meningkat tajam setelah epoch ke-5.']"
    )
    contributing_factors: list[str] = Field(
        description="Faktor-faktor lain yang mungkin berkontribusi pada kegagalan."
    )
    corrective_actions_proposed: list[str] = Field(
        description="Daftar saran perbaikan konkret untuk putaran berikutnya. Contoh: ['Kurangi learning rate sebesar 50%.', 'Tambahkan regularisasi dropout menjadi 0.3.']"
    )


class PostMortemAnalyzer:
    """
    Agen yang melakukan analisis 'post-mortem' setelah prediksi dinilai buruk
    untuk melakukan koreksi diri atas penalaran(Reasoning Self-Correction).
    """

    def __init__(self, api_pool: DistributedAIPool, governor: CognitiveGovernor):
        self.api_pool = api_pool
        self.governor = governor
        self.analyzer_agent_role = "supervisor"
        logger.info("🕵️‍♂️ Post-Mortem Analyzer (Self-Correction) siap.")

    def analyze_failure(
        self,
        project_id: str,
        feedback: dict,
        hparams: dict,
        metrics: dict,
        last_reflection: str,
    ) -> FailureAnalysisReport:
        """
        Menganalisis kegagalan dan menghasilkan laporan terstruktur.
        """
        logger.info(
            f"--- 🔍 MEMULAI POST-MORTEM untuk Project {project_id} ---")


        prompt = f"""
        Anda adalah seorang "Chief AI Officer" yang sangat berpengalaman. Salah satu model AI Anda baru saja menghasilkan prediksi yang dinilai 'Buruk' oleh supervisor manusia.
        Tugas Anda adalah melakukan investigasi mendalam(post-mortem) untuk menemukan akar masalah logis dan merekomendasikan tindakan perbaikan yang cerdas.

        # LAPORAN INSIDEN
        - **Project ID: ** {project_id}
        - **Penilaian Supervisor: ** {feedback.get('rating_text', '👎 Buruk')}
        - **Komentar Supervisor: ** "{feedback.get('comment', 'Tidak ada.')}"

        # BUKTI-BUKTI
        1. ** Laporan Refleksi Diri Terakhir dari AI(sebelum gagal): **
        ---
        {last_reflection}
        ---

        2. ** Hyperparameter yang Digunakan: **
        ```json
        {json.dumps(make_json_serializable(hparams), indent=2)}
        ```

        3. ** Metrik Kinerja Akhir: **
        ```json
        {json.dumps(make_json_serializable(metrics), indent=2)}
        ```

        # TUGAS ANDA
        Berdasarkan SEMUA bukti di atas, lakukan analisis dan panggil tool `FailureAnalysisReport` untuk melaporkan temuan Anda.
        Fokus pada MENGAPA logika AI salah, bukan hanya apa yang salah. Contoh: Apakah AI terlalu percaya diri berdasarkan refleksi dirinya? Apakah ia mengabaikan sinyal penting dari data? Apakah kombinasi hyperparameter tidak masuk akal?
        """

        try:
            report_dict = self.api_pool.call_gemini_with_tool(
                prompt=prompt,
                agent_name=self.analyzer_agent_role,
                tool_schema=FailureAnalysisReport,
            )

            if not report_dict:
                raise ValueError(
                    "Analyzer AI tidak mengembalikan laporan yang valid.")

            analysis_report = FailureAnalysisReport(**report_dict)

            logger.info("✔️ Laporan Post-Mortem berhasil dibuat.")

            self.governor.log_event(
                "POST_MORTEM_ANALYSIS_COMPLETED",
                details={
                    "project_id": project_id,
                    "root_cause": analysis_report.suspected_root_cause,
                    "corrective_actions": analysis_report.corrective_actions_proposed,
                },
            )
            logger.info(
                "[Meta-Kognisi] Temuan Post-Mortem dicatat oleh Cognitive Governor."
            )

            return analysis_report

        except Exception as e:
            logger.error(f"Gagal melakukan post-mortem: {e}", exc_info=True)
            return None


class ComplexEmotionAnalysis(BaseModel):
    """Skema untuk jawaban terstruktur dari LLM saat menganalisis kombinasi emosi."""

    new_emotion_name: str = Field(
        description="Nama deskriptif untuk emosi gabungan (maksimal 3 kata), dalam Bahasa Indonesia."
    )
    combined_emoji: str = Field(
        description="Satu atau dua emoji yang paling mewakili emosi baru ini."
    )
    description: str = Field(
        description="Penjelasan singkat tentang apa arti emosi gabungan ini."
    )
    utility: str = Field(
        description="Kegunaan atau fungsi dari emosi ini dalam analisis finansial atau interaksi manusia."
    )
    example_text_context: str = Field(
        description="Contoh kalimat atau situasi tekstual di mana emosi ini mungkin muncul."
    )
    example_visual_context: str = Field(
        description="Deskripsi singkat tentang gambar atau visual yang bisa memicu emosi ini."
    )


class EmotionalAlchemist:
    """Agen AI yang bertugas mensintesis emosi baru yang kompleks dari emosi yang sudah ada."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "API Key untuk 'qwen_giant' diperlukan oleh EmotionalAlchemist."
            )

        self.agent = TogetherLLM(
            api_key=api_key, model_name="Qwen/Qwen2-72B-Instruct"
        )
        logger.info("🧪 Emotional Alchemist (Qwen2) siap untuk sintesis.")

    def synthesize_new_emotion(
        self, emotion1: dict, emotion2: dict
    ) -> Optional[ComplexEmotionAnalysis]:
        """Meminta LLM untuk menciptakan emosi baru dari dua emosi yang ada."""
        e1_name = emotion1["attributes"].get("name", "emosi tidak dikenal")
        e2_name = emotion2["attributes"].get("name", "emosi tidak dikenal")

        prompt = f"""
        Anda adalah seorang psikolog dan ahli linguistik. Analisis kombinasi dari dua emosi berikut: '{e1_name}' dan '{e2_name}'.

        Tugas Anda:
        Ciptakan sebuah emosi baru yang lebih kompleks dari gabungan keduanya. Jelaskan nama, deskripsi, kegunaan, dan berikan contoh konteks penggunaannya.
        Jawab HANYA dengan format JSON yang sesuai dengan skema `ComplexEmotionAnalysis`.
        """
        try:

            response_str = self.agent.chat(prompt)

            analysis = robust_json_extract(
                response_str, model=ComplexEmotionAnalysis)
            return analysis
        except Exception as e:
            logger.error(
                f"Gagal mensintesis emosi dari '{e1_name}' & '{e2_name}': {e}")
            return None


class CognitiveScheduler:
    """Pengatur tugas dinamis v2.0 dengan 3 Mode Operasi Adaptif."""

    def __init__(self, shared_state: dict):
        self.shared_state = shared_state
        self.CPU_THRESHOLD_GASPOL = 90.0
        self.CPU_THRESHOLD_SINAU = 60.0
        self.MIN_ENERGY = 20
        self.cognitive_energy = 100.0
        logger.info(
            f"✅ Cognitive Scheduler v2.0 (Adaptif) aktif. Stamina awal: {self.cognitive_energy:.0f}%"
        )

    def get_go_ahead(self, task_cost: int) -> Optional[str]:
        """
        Membuat keputusan mode operasi atau menunda.
        Mengembalikan 'GASPOL', 'SINAU_DIEM', atau None.
        """
        current_mode = self.shared_state.get("activity_mode", "SINAU_DIEM")


        if current_mode == "SIAGA":
            logger.info("😴 [Scheduler] Menunggu mode SIAGA selesai...")
            return None


        cpu_load = psutil.cpu_percent(interval=0.1)

        cpu_threshold = (
            self.CPU_THRESHOLD_GASPOL
            if current_mode == "GASPOL"
            else self.CPU_THRESHOLD_SINAU
        )
        

        if cpu_load > cpu_threshold:
            logger.info(
                f"😴 [Scheduler] Beban CPU tinggi ({cpu_load:.1f}%). Menunda tugas."
            )
            return None


        if self.cognitive_energy < self.MIN_ENERGY:
            logger.info(
                f"😴 [Scheduler] Stamina kognitif rendah. Worker beristirahat."
            )
            return None


        if self.cognitive_energy < task_cost:
            logger.info(
                f"😴 [Scheduler] Stamina tidak cukup untuk tugas ini (butuh {task_cost}, sisa {self.cognitive_energy:.0f})."
            )
            return None

        return current_mode

    def spend_energy(self, cost: int):
        self.cognitive_energy = max(0, self.cognitive_energy - cost)

    def recharge_energy(self):

        self.cognitive_energy = min(100.0, self.cognitive_energy + 15)


class IterativeRefinementEngine:
    """
    Mengorkestrasi alur kerja Generate -> Critique -> Refine untuk meningkatkan kualitas output teks,
    diperkaya dengan memori jangka panjang.
    """

    def __init__(
        self,
        api_pool: DistributedAIPool,
        feedback_manager: HumanFeedbackManager,
        anomaly_memory: AnomalyMemory,
    ):
        self.api_pool = api_pool
        self.feedback_manager = feedback_manager
        self.anomaly_memory = anomaly_memory
        self.critique_agent_role = "critique_agent"
        self.refinement_agent_role = "gemini_refiner"
        logger.info(
            f"✨ Iterative Refinement Engine (Memory-Augmented) siap dengan (Kritik: {self.critique_agent_role}, Refiner: {self.refinement_agent_role})."
        )

    def refine(
        self, initial_text: str, original_context: str, max_iterations: int = 1
    ) -> str:
        current_text = initial_text
        for i in range(max_iterations):
            logger.info(
                f"--- Memulai Putaran Refinement #{i + 1}/{max_iterations} ---")


            feedback_context = self.feedback_manager.get_all_feedback_as_context(
                limit=10
            )
            anomaly_context = str(
                self.anomaly_memory.recall_anomalies().to_markdown(index=False)
            )


            critique_prompt = f"""
            Anda adalah seorang editor ahli yang kritis, teliti, dan memiliki ingatan sejarah yang kuat.
            Tugas Anda adalah memberikan kritik terstruktur terhadap analisis berikut.

            KONTEKS ASLI YANG DIBERIKAN KEPADA PENULIS:
            ---
            {original_context}
            ---

            KONTEKS DARI MEMORI JANGKA PANJANG(GUNAKAN UNTUK CEK FAKTA & KONSISTENSI):
            ---
            Riwayat Umpan Balik Supervisor: {feedback_context}
            ---
            Riwayat Peristiwa Anomali: {anomaly_context}
            ---

            TEKS YANG PERLU DIKRITIK:
            ---
            {current_text}
            ---

            Gunakan skema 'Critique' untuk memberikan umpan balik Anda. Selain menilai logika, periksa juga apakah teks ini:
            1. Konsisten dengan umpan balik supervisor di masa lalu?
            2. Mempertimbangkan peristiwa anomali penting yang relevan?
            3. Akurat secara faktual berdasarkan memori?
            """
            try:
                critique_obj = self.api_pool.call_gemini_with_tool(
                    prompt=critique_prompt,
                    agent_name=self.critique_agent_role,
                    tool_schema=Critique,
                )
                if not critique_obj:
                    logger.warning(
                        "Gagal mendapatkan kritik terstruktur. Membatalkan refinement."
                    )
                    return current_text

                critique_obj = Critique(**critique_obj)
                logger.info("✔️ Kritik berhasil diterima dari CritiqueAgent.")

            except Exception as e:
                logger.error(
                    f"Gagal saat meminta kritik: {e}. Mengembalikan teks terakhir."
                )
                return current_text


            refine_prompt = f"""
            Anda adalah seorang penulis dan analis ahli.
            Tugas Anda adalah menulis ulang draf awal berdasarkan kritik yang membangun dari editor Anda.

            KONTEKS/TUJUAN ASLI TULISAN:
            ---
            {original_context}
            ---

            DRAF AWAL ANDA:
            ---
            {current_text}
            ---

            MASUKAN DAN KRITIK DARI EDITOR:
            - Kelebihan untuk dipertahankan: {critique_obj.strengths}
            - Kelemahan untuk diperbaiki: {critique_obj.weaknesses}
            - Saran Tindak Lanjut: {', '.join(critique_obj.actionable_suggestions)}
            ---

            Tulis ulang draf awal untuk mengatasi semua kelemahan dan menerapkan saran yang ada. Pastikan versi final jauh lebih baik dan profesional.
            Fokus hanya pada menghasilkan teks yang disempurnakan, tanpa komentar meta.
            """

            logger.info("✍️ Meminta RefinementAgent untuk menulis ulang...")
            refined_text = self.api_pool.call_gemini_for_text(
                prompt=refine_prompt, agent_name=self.refinement_agent_role
            )

            if refined_text.strip():
                logger.info("✔️ Teks berhasil disempurnakan.")
                current_text = refined_text
            else:
                logger.warning(
                    "RefinementAgent menghasilkan output kosong. Menggunakan versi sebelumnya."
                )

        logger.info("--- Proses Refinement Selesai ---")
        return current_text



class GeminiSupervisorySession:
    def __init__(self, api_pool: DistributedAIPool):
        self.api_pool = api_pool
        self.system_data = {}

    def _prepare_context_report(self, context: str, details: dict) -> str:
        report = f"LAPORAN ANALISIS TAHAP: {context}\n" + "=" * 40 + "\n\n"
        if context == "DATA_LOADING_AUDIT":
            df = self.system_data.get("dataframe_initial")
            if df is not None and details:
                total_nan_pct = df.isnull().sum().sum() / df.size * 100
                nan_per_col = (df.isnull().sum() / len(df) * 100).sort_values(
                    ascending=False
                )
                nan_per_col = nan_per_col[nan_per_col > 0]
                report += f"**Ringkasan Data Awal:**\n- Bentuk Data: {df.shape}\n- Rentang Tanggal: {details.get('StartDate')} hingga {details.get('EndDate')}\n- Total NaN: {total_nan_pct:.2f}%\n- Kolom NaN 100%: {details.get('Fully_NaN_Columns', 'Tidak Ada')}\n\n"
                report += f"**Distribusi NaN per Fitur:**\n\n{nan_per_col.to_string()}\n\n**Metodologi:**\n- NaN diisi dengan KNNImputer(n_neighbors=5).\n"
        elif context in ("TRAINING_RESULT", "FINAL_ANALYSIS"):
            serializable_details = make_json_serializable(details)
            report += f"**{context.replace('_', ' ').title()}:**\n```json\n{json.dumps(serializable_details, indent=2)}\n```"
        return report

    def start_supervision(self, context: str, details: dict):
        logger.info(f"\n--- Memulai Supervisi untuk {context} ---")
        report = self._prepare_context_report(context, details)
        prompt = f"""Anda adalah 'Dewan Pengawas Kuantitatif'. Analisis laporan berikut, sorot risiko/kelemahan, dan beri rekomendasi.
        PROTOKOL:
        1. Analisis dalam poin-poin.
        2. Vonis akhir: [PROCEED] atau [CONCERNS_FOUND].
        --- LAPORAN ---
        {report}
        --- AKHIR ---
        MULAI ANALISIS:
        """
        gemini_response = self.api_pool.call_gemini_for_text(
            prompt, "SUPERVISOR")
        logger.info("\n[MASUKAN DEWAN PENGAWAS GEMINI]:\n" + gemini_response)
        if "[PROCEED]" in gemini_response:
            logger.info(f"\n--- Supervisi {context} Disetujui ---")
        elif "[CONCERNS_FOUND]" in gemini_response:
            logger.info(f"\n--- Supervisi {context} Menemukan Masalah ---")
        else:
            logger.info(f"\n--- Vonis Tidak Jelas, Lanjut Hati-hati ---")


class HyperparameterTutor:
    """Merancang strategi perbaikan hyperparameter menggunakan Llama-4 Maverick."""

    def __init__(self, together_keys: dict):

        self.tutor_key = together_keys.get("tutor_maverick")
        if not self.tutor_key:
            logger.warning(
                "[Tutor] Kunci API 'tutor_maverick' tidak tersedia. Tutor tidak akan berfungsi."
            )
            self.tutor_agent = None
        else:
            self.tutor_agent = TogetherLLM(
                api_key=self.tutor_key,
                model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            )


    def design_new_strategy(self, failed_metrics: dict, old_hparams: dict):
        if not self.tutor_agent:
            return None, "Tutor (Llama) not configured."

        logger.info(
            "\n  [TUTOR-LLAMA] Merancang strategi perbaikan..."
        )
        serializable_metrics = make_json_serializable(failed_metrics)
        serializable_hparams = make_json_serializable(old_hparams)
        serializable_hparams.pop("gemini_api_config", None)

        system_prompt = (
            "You are an expert Hyperparameter Tuning AI. Your goal is to propose a new set of hyperparameters to improve model performance. "
            "You MUST respond ONLY with a valid JSON object containing the new hyperparameters, without any explanation or markdown."
        )
        prompt = (
            "The previous training round had poor results. Design a new set of hyperparameters to fix this. "
            f"Failed Metrics:\n```json\n{json.dumps(serializable_metrics, indent=2)}\n```\n\n"
            f"Old Hyperparameters:\n```json\n{json.dumps(serializable_hparams, indent=2)}\n```\n\n"
            "Your new hyperparameter proposal (JSON only):"
        )

        try:
            response_str = self.tutor_agent.chat(
                prompt, system_prompt)
            proposal = robust_json_extract(
                response_str, model=HyperparameterProposal)

            if proposal:
                logger.info(
                    "  [TUTOR-LLAMA] Proposal berhasil dibuat."
                )
                return (
                    proposal.model_dump(),
                    None,
                )
            else:
                logger.error(
                    f"  [TUTOR-LLAMA] Gagal mem-parse proposal. Respons: {response_str}"
                )
                return None, "Failed to parse JSON proposal from Llama."
        except Exception as e:
            logger.error(
                f"  [TUTOR-LLAMA] Gagal mendapatkan proposal. Error: {e}"
            )
            return None, str(e)


class ProposalAccreditor:
    """
    Menilai proposal hyper-parameter menggunakan Llama-4 Maverick sebagai penilai.
    """

    def __init__(self, together_keys: dict):

        self.accreditor_key = together_keys.get("accreditor_maverick")
        if not self.accreditor_key:
            logger.warning(
                "[Accreditor] Kunci API 'accreditor_maverick' tidak tersedia. Accreditor tidak aktif."
            )
            self.accreditor_agent = None
        else:
            self.accreditor_agent = TogetherLLM(
                api_key=self.accreditor_key,
                model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            )


    @staticmethod
    def _format_decision(verdict: AssessProposal) -> str:
        status = "[PROPOSAL_ACCEPTED]" if verdict.is_accepted else "[PROPOSAL_REJECTED]"
        return f"{status} {verdict.reasoning}"

    def validate_proposal(self, new_hparams_proposal: dict):
        if not self.accreditor_agent:
            logger.warning(
                "[Accreditor] Melewatkan validasi, proposal otomatis diterima."
            )
            return "[PROPOSAL_ACCEPTED] Accreditor tidak aktif.", None

        logger.info(
            "  [ACCREDITOR-LLAMA] Memvalidasi proposal dengan Llama-4 Maverick..."
        )

        system_prompt = (
            "You are a strict and logical AI Accreditor. Your task is to review a hyperparameter proposal. "
            "You MUST respond ONLY with a valid JSON object following the specified structure, without any extra text or markdown. "
            "The JSON must contain two keys: 'is_accepted' (boolean) and 'reasoning' (a short, clear string)."
        )

        prompt = (
            "Review the following hyperparameter proposal. Is it logical and reasonable? Provide your verdict as a raw JSON object.\n\n"
            f"Proposal:\n```json\n{json.dumps(make_json_serializable(new_hparams_proposal), indent=2)}\n```"
        )

        try:
            response_str = self.accreditor_agent.chat(
                prompt, system_prompt
            )
            verdict = robust_json_extract(response_str, model=AssessProposal)

            if verdict is None:
                logger.error(
                    f"  [ACCREDITOR-FAIL] Llama gagal memberikan format JSON yang valid. Respons: {response_str}"
                )
                return "[PROPOSAL_REJECTED] Invalid format from Llama.", "ParseError"

            decision = self._format_decision(verdict)

            logger.info(f"  [ACCREDITOR-LLAMA] Vonis: {decision}")
            return decision, None
        except Exception as e:
            logger.error(
                f"  [ACCREDITOR-FAIL] Exception saat validasi dengan Llama: {e}"
            )
            return f"[PROPOSAL_REJECTED] Exception during accreditation: {e}", str(e)


class GeminiAdvancedAdvisor:
    def __init__(self, api_pool: DistributedAIPool):
        self.api_pool = api_pool
        self.nlp = _nlp_model
        self.matcher = PhraseMatcher(self.nlp.vocab) if self.nlp else None
        self.intent_classifier = _intent_classifier
        self.rf_optimizer = RandomForestRegressor(
            n_estimators=50, random_state=42)
        self.gb_optimizer = GradientBoostingRegressor(
            n_estimators=50, random_state=42)
        self.history = []
        if self.nlp:
            self._setup_matcher()
            self._train_optimizers()

    def _setup_matcher(self):
        patterns = [
            self.nlp("increase learning rate"),
            self.nlp("decrease learning rate"),
            self.nlp("increase d_model"),
            self.nlp("decrease d_model"),
            self.nlp("increase n_layers"),
            self.nlp("decrease n_layers"),
            self.nlp("increase dropout"),
            self.nlp("decrease dropout"),
            self.nlp("increase weight decay"),
            self.nlp("decrease weight decay"),
            self.nlp("add more layers"),
            self.nlp("reduce layers"),
            self.nlp("use more features"),
            self.nlp("use fewer features"),
        ]
        for pattern in patterns:
            self.matcher.add("HYPERPARAM_ADJUSTMENT", None, pattern)

    def _train_optimizers(self):
        X_dummy = np.random.rand(100, 5)
        y_dummy = np.random.rand(100)
        self.rf_optimizer.fit(X_dummy, y_dummy)
        self.gb_optimizer.fit(X_dummy, y_dummy)

    def get_suggestion(self, context: str):
        prompt = f"Given context: {context}, suggest improvements for model performance (hyperparameters, architecture, features)."
        return self.api_pool.call_gemini_for_text(prompt, "ADVANCED_ADVISOR")

    def parse_suggestion(self, suggestion: str):
        if not self.nlp or not self.intent_classifier:
            return []
        doc = self.nlp(suggestion)
        matches = self.matcher(doc)
        intents = self.intent_classifier(suggestion)
        adjustments = []
        for match_id, start, end in matches:
            span = doc[start:end]
            intent_score = (
                intents[0]["score"]
                if intents[0]["label"] == "POSITIVE"
                else -intents[0]["score"]
            )
            adjustments.append(
                {
                    "text": span.text,
                    "confidence": intent_score,
                    "raw_suggestion": suggestion,
                }
            )
        return adjustments

    def optimize_suggestion(
        self, adjustments: list, current_hparams: dict, performance_history: list
    ):
        if not adjustments or not performance_history:
            return current_hparams
        X_opt = []
        for adj in adjustments:
            magnitude = (
                0.1
                if "increase" in adj["text"]
                else -0.1 if "decrease" in adj["text"] else 0
            )
            feat = [
                hash(adj["text"]) % 100,
                magnitude,
                adj["confidence"],
                performance_history[-1],
                len(performance_history),
            ]
            X_opt.append(feat)
        X_opt = np.array(X_opt)
        rf_preds = self.rf_optimizer.predict(X_opt)
        gb_preds = self.gb_optimizer.predict(X_opt)
        optimized_magnitudes = (rf_preds + gb_preds) / 2
        hparams = current_hparams.copy()
        for adj, mag in zip(adjustments, optimized_magnitudes):
            if "learning rate" in adj["text"]:
                hparams["lr"] = max(1e-5, min(1e-3, hparams["lr"] * (1 + mag)))
            elif "d_model" in adj["text"]:
                hparams["d_model"] = max(
                    64, min(256, int(hparams["d_model"] * (1 + mag)))
                )
            elif "n_layers" in adj["text"]:
                hparams["n_layers"] = max(
                    1, min(4, hparams["n_layers"] + int(mag * 2)))
            elif "dropout" in adj["text"]:
                hparams["dropout"] = min(
                    0.5, max(0.1, hparams["dropout"] + mag))
            elif "weight decay" in adj["text"]:
                hparams["weight_decay"] = max(
                    1e-6, min(1e-3, hparams["weight_decay"] * (1 + mag))
                )
            elif "features" in adj["text"]:
                hparams["top_k_features"] = max(
                    20, min(50, int(hparams["top_k_features"] * (1 + mag)))
                )
        return hparams

    def update_history(self, suggestion: str, outcome: float):
        self.history.append({"suggestion": suggestion, "outcome": outcome})
        if len(self.history) > 100:
            self.history.pop(0)
        X = np.array(
            [
                [
                    hash(h["suggestion"]) % 100,
                    0.1 if "increase" in h["suggestion"] else -0.1,
                    0.9,
                    h["outcome"],
                    i,
                ]
                for i, h in enumerate(self.history[:-1])
            ]
        )
        y = np.array([h["outcome"] for h in self.history[1:]])
        if len(X) > 10:
            self.rf_optimizer.fit(X, y)
            self.gb_optimizer.fit(X, y)


class ExperimentalistAI:


    class SpeculativeProposal(BaseModel):
        analysis: str = Field(
            description="Analisis mendalam mengenai tren, keberhasilan, dan kegagalan dari riwayat optimisasi yang diberikan. Berikan juga implikasi untuk eksperimen selanjutnya."
        )
        proposed_experiment: dict = Field(
            description="Sebuah objek/kamus Python yang berisi hyperparameter baru yang 'gila' atau berisiko tinggi untuk dicoba. Contoh: {'lr': 0.1, 'window': 150, 'dropout': 0.05}"
        )


    SPECULATIVE_PROPOSAL_TOOL = StructuredTool.from_function(
        func=_identity,
        name="ProposeSpeculativeExperiment",
        description="Gunakan tool ini untuk mengusulkan eksperimen spekulatif baru berdasarkan analisis riwayat.",
        args_schema=SpeculativeProposal,
    )

    def __init__(self, api_pool: DistributedAIPool):
        self.api_pool = api_pool
        self.max_text_attempts = 2

    def run_speculative_experiment(self, history: list):
        logger.info(
            f"\n{'='*25} TAHAP EKSPERIMENTAL (Canggih & Mandiri) {'='*25}")

        accepted_runs = [run for run in history if run.get("accepted")]
        if not accepted_runs:
            logger.error(
                "  [EXPERIMENTALIST-FAIL] Tidak ada riwayat percobaan yang diterima untuk dianalisis."
            )
            return None, "No accepted runs found in history."

        best_run = min(accepted_runs, key=lambda x: x["score"])
        context = (
            f"Analisis riwayat optimisasi berikut. Hasil terbaik sejauh ini:\n"
            f"**Terbaik:**\n```json\n{json.dumps(make_json_serializable(best_run), indent=2)}\n```\n"
            f"**Seluruh Riwayat:**\n```json\n{json.dumps(make_json_serializable(history), indent=2)}\n```\n"
        )


        logger.info(
            "  [EXPERIMENTALIST-AI] Mencoba strategi utama: Function Calling..."
        )
        worker = self.api_pool.get_worker("EXPERIMENTALIST")
        if not worker:
            logger.error(
                "  [EXPERIMENTALIST-FAIL] Worker 'EXPERIMENTALIST' tidak ditemukan di API Pool."
            )
            return None, "Worker not available"

        try:
            prompt = f"Anda adalah seorang peneliti AI ahli. {context}\nBerdasarkan data di atas, panggil tool 'ProposeSpeculativeExperiment' untuk merancang eksperimen baru yang berani."


            response_obj = worker.invoke(
                prompt, tools=[self.SPECULATIVE_PROPOSAL_TOOL])
            tool_calls = response_obj.additional_kwargs.get("tool_calls", [])

            if tool_calls:

                proposal_str = tool_calls[0].get(
                    "function", {}).get("arguments")
                if proposal_str:
                    logger.info(
                        "  [EXPERIMENTALIST-AI] ✔️ Proposal berhasil dibuat via Function Calling."
                    )
                    return json.loads(proposal_str), None
        except Exception as e:
            logger.warning(
                f"  [EXPERIMENTALIST-WARN] Panggilan tool gagal dengan error: {e}. Beralih ke strategi cadangan."
            )


        error_feedback = ""
        for attempt in range(self.max_text_attempts):
            logger.info(
                f"  [EXPERIMENTALIST-AI] Mencoba strategi cadangan: Parsing Teks (Percobaan {attempt + 1}/{self.max_text_attempts})..."
            )

            text_prompt = f"""
Anda adalah seorang Peneliti AI ahli yang disiplin. Tugas Anda adalah menganalisis riwayat optimisasi berikut dan merancang satu eksperimen 'gila' (high-risk, high-reward).
{context}
{error_feedback}

ATURAN KETAT:
1. Output Anda HARUS berupa objek JSON tunggal dan valid.
2. JANGAN menyertakan teks penjelasan atau markdown. HANYA JSON mentah.
3. Semua kunci (keys) dan string harus dalam kutip ganda (").
4. Struktur JSON harus memiliki kunci 'analysis' (string) dan 'proposed_experiment' (dictionary).
"""
            response_text = self.api_pool.call_gemini_for_text(
                text_prompt, "EXPERIMENTALIST"
            )
            parsed_proposal = robust_json_extract(response_text, model=None)

            if (
                parsed_proposal
                and "analysis" in parsed_proposal
                and "proposed_experiment" in parsed_proposal
            ):
                logger.info(
                    "  [EXPERIMENTALIST-AI] ✔️ Proposal berhasil dibuat via Parsing Teks."
                )
                return parsed_proposal, None
            else:
                logger.error(
                    f"  [EXPERIMENTALIST-FAIL] Percobaan {attempt + 1} gagal di-parse."
                )
                error_feedback = f"\nPERHATIAN: Pada percobaan sebelumnya, output Anda tidak valid. Respons Anda: ```{response_text[:500]}...```. Harap perbaiki dan ikuti SEMUA ATURAN."

        logger.error(
            "  [EXPERIMENTALIST-FAIL] Semua strategi gagal menghasilkan proposal yang valid."
        )
        return None, "All attempts to generate a valid proposal failed."





def create_flask_dash_app(
    chat_engine: GeminiChatEngine, message_queue: queue.Queue
) -> Flask:
    server = Flask(__name__)
    dash_app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        url_base_pathname="/",
        suppress_callback_exceptions=True,
    )


    dash_app.layout = html.Div(
        id="page-wrap",
        children=[

            html.Div(id="chat-box"),
            html.Div(
                id="input-row",
                children=[
                    dcc.Input(
                        id="user-input",
                        type="text",
                        debounce=True,
                        autoComplete="off",
                        placeholder="Type a message…",
                    ),
                    html.Button("Send", id="send-btn"),
                ],
            ),

            dcc.Store(id="chat-history-store", data=[]),
            dcc.Interval(
                id="interval-component",
                interval=15 * 1000,
                n_intervals=0,
            ),
        ],
    )



    @server.route("/get-proactive-updates")
    def get_proactive_updates():
        try:

            message = message_queue.get_nowait()
            return jsonify({"message": message})
        except queue.Empty:

            return jsonify({"message": None})





    @dash_app.callback(
        Output("chat-history-store", "data"),
        Input("send-btn", "n_clicks"),
        [State("user-input", "value"),
         State("chat-history-store", "data")],
        prevent_initial_call=True,
    )
    def on_user_send(n_clicks, user_text, chat_history):
        if not user_text:
            raise dash.exceptions.PreventUpdate


        chat_history.append({"role": "user", "content": user_text})


        answer = chat_engine.ask(user_text)
        chat_history.append({"role": "bot", "content": answer})

        return chat_history


    @dash_app.callback(
        Output("chat-history-store", "data", allow_duplicate=True),
        Input("interval-component", "n_intervals"),
        State("chat-history-store", "data"),
        prevent_initial_call=True,
    )
    def on_interval_check(n, chat_history):
        try:

            response = requests.get(
                "http://127.0.0.1:5000/get-proactive-updates", timeout=2
            )
            if response.status_code == 200:
                data = response.json()
                message = data.get("message")
                if message:
                    logger.info(f"🤖 Pesan proaktif diterima: {message}")
                    chat_history.append(
                        {
                            "role": "bot",
                            "content": f"[INISIATIF DARI JARVIS]:\n{message}",
                        }
                    )
                    return chat_history
        except requests.exceptions.RequestException:

            pass
        raise dash.exceptions.PreventUpdate


    @dash_app.callback(
        [Output("chat-box", "children"),
         Output("user-input", "value")],
        Input("chat-history-store", "data"),
    )
    def update_chat_display(chat_history):
        bubbles = []
        for msg in chat_history:
            className = "msg user" if msg["role"] == "user" else "msg bot"
            bubbles.append(html.Div(dcc.Markdown(
                msg["content"]), className=className))


        ctx = dash.callback_context
        if (
            ctx.triggered
            and ctx.triggered[0]["prop_id"] == "chat-history-store.data"
            and ctx.triggered_id != "interval-component"
        ):
            return bubbles, ""
        return bubbles, dash.no_update



    @server.route("/chat", methods=["POST"])
    def chat_api():
        msg = request.get_json(force=True).get("message", "")
        return jsonify({"answer": chat_engine.ask(msg)})

    return server


class ParameterAlchemist:
    """
    Agen AI yang bertugas mensintesis hyperparameter terbaik dari
    seluruh riwayat eksekusi yang sukses untuk menciptakan satu set "juara".
    """

    class AlchemistProposal(BaseModel):
        synthesis_reasoning: str = Field(
            description="Penjelasan logis mengapa setiap parameter dipilih dari putaran yang berbeda, berdasarkan data historis."
        )
        final_champion_hparams: dict = Field(
            description="Objek JSON yang berisi set hyperparameter 'terbaik dari yang terbaik' hasil sintesis."
        )

    SYNTHESIS_TOOL = StructuredTool.from_function(
        func=_identity,
        name="ProposeChampionHyperparameters",
        description="Gunakan tool ini untuk mengusulkan set hyperparameter 'juara' hasil sintesis dari riwayat.",
        args_schema=AlchemistProposal,
    )

    def __init__(self, api_pool: DistributedAIPool):
        self.api_pool = api_pool

    def synthesize_best_of_the_best(
        self, history: list[dict], data_summary: dict
    ) -> Optional[dict]:
        """
        Fungsi ini menjalankan seluruh logika sintesis parameter juara.
        Blok try-except berada di dalam fungsi ini.
        """
        logger.info(
            f"\n{'='*25} TAHAP ALKEMIS: SINTESIS PARAMETER JUARA {'='*25}")

        successful_runs = [run for run in history if run.get("accepted")]
        if len(successful_runs) < 2:
            logger.warning(
                "[Alchemist] Tidak cukup riwayat (min. 2) untuk melakukan sintesis. Melewatkan tahap ini."
            )
            return None

        prompt = f"""
        Anda adalah seorang "AI Alchemist", seorang Grandmaster dalam tuning hyperparameter.
        Tugas Anda adalah melakukan sintesis akhir dari seluruh riwayat optimisasi yang sukses.

        ATURAN MAIN ANDA:
        1.  Analisis seluruh riwayat JSON di bawah ini. Perhatikan metrik (`score`) dan parameter (`hparams`) dari setiap putaran (`round`).
        2.  PILIH NILAI TERBAIK untuk SETIAP PARAMETER INDIVIDUAL (`window`, `horizon`, `lr`, `dropout`, dll.) dari putaran-putaran yang berbeda. Contoh: `lr` terbaik mungkin dari putaran 1, sedangkan `window` terbaik dari putaran 3.
        3.  Gabungkan semua nilai terbaik individual ini menjadi satu set hyperparameter "JUARA" yang baru.
        4.  Panggil tool `ProposeChampionHyperparameters` dengan hasil sintesis Anda. Berikan alasan yang kuat untuk setiap pilihan parameter Anda di `synthesis_reasoning`.

        **Ringkasan Data yang Digunakan:**
        ```json
        {json.dumps(data_summary, indent=2)}
        ```

        **Riwayat Eksekusi yang Sukses:**
        ```json
        {json.dumps(make_json_serializable(successful_runs), indent=2)}
        ```
        Lakukan alkimia Anda sekarang.
        """



        try:

            response_args = self.api_pool.call_gemini_with_tool(
                prompt, "EXPERIMENTALIST", self.AlchemistProposal
            )

            proposal = ensure_dict(
                response_args, pydantic_model=self.AlchemistProposal)

            if proposal and "final_champion_hparams" in proposal:
                logger.info(
                    "[Alchemist] ✔️ Proposal Juara berhasil disintesis.")
                logger.info(
                    f"  - Alasan: {proposal.get('synthesis_reasoning', 'Tidak ada alasan diberikan.')}"
                )

                return proposal["final_champion_hparams"]
            else:
                logger.error(
                    f"[Alchemist] Gagal melakukan sintesis. Respons tidak valid: {response_args}"
                )

                return None

        except Exception as e:
            logger.error(
                f"[Alchemist] Terjadi exception saat proses sintesis: {e}",
                exc_info=True,
            )

            return None


class GenerativeStrategist:
    """
    Agen AI yang merumuskan hipotesis strategi investasi baru dari data yang ada.
    Beroperasi di bawah 'Mandat Realitas'.
    """

    def __init__(self, api_pool: DistributedAIPool, strategy_kb: StrategyKnowledgeBase):
        self.api_pool = api_pool
        self.strategy_kb = strategy_kb
        self.agent_role = "experimentalist"

    def formulate_hypothesis(
        self, context_bundle: dict
    ) -> Optional[GenerativeStrategy]:
        logger.info(
            "\n--- [Strategist] Memulai Sesi Kreasi Hipotesis Strategi ---")

        prompt = f"""
        Anda adalah seorang 'Quantitative Strategist' yang sangat kreatif namun berbasis data.
        Tugas Anda adalah merumuskan SATU hipotesis strategi trading baru yang dapat diuji.

        'MANDAT REALITAS': Hipotesis Anda HARUS terinspirasi secara langsung dari hubungan yang Anda temukan dalam data konteks di bawah ini. Anda DILARANG berimajinasi tanpa dasar.

        KONTEKS DATA TERSEDIA:
        ---
        1. Laporan Refleksi Diri & Metrik Terakhir:
        {context_bundle.get('self_reflection', 'N/A')}

        2. Riwayat Umpan Balik Supervisor Manusia:
        {context_bundle.get('human_feedback', 'N/A')}

        3. Konteks dari Basis Pengetahuan Strategi (Strategi yang sudah ada):
        {self.strategy_kb.get_strategies_as_context()}

        4. Ringkasan Data Utama & Fitur:
        {json.dumps(context_bundle.get('data_summary', {}), indent=2)}
        ---

        TUGAS ANDA:
        1. Analisis semua konteks di atas untuk menemukan pola atau hubungan yang belum dieksploitasi.
        2. Rumuskan temuan Anda menjadi sebuah hipotesis yang jelas.
        3. Panggil tool `GenerativeStrategy` untuk menstrukturkan output Anda.
        4. Di `data_source_justification`, sebutkan dengan jelas bagian mana dari konteks yang memicu ide Anda.
        """

        try:
            response = self.api_pool.call_gemini_with_tool(
                prompt, self.agent_role, GenerativeStrategy
            )
            if response:
                strategy = GenerativeStrategy(**response)
                logger.info(
                    f"✅ [Strategist] Hipotesis berhasil dirumuskan: '{strategy.strategy_name}'"
                )
                self.strategy_kb.add_hypothesis(
                    strategy.strategy_name, strategy.hypothesis, strategy.rules
                )
                return strategy
            return None
        except Exception as e:
            logger.error(
                f"[Strategist] Gagal merumuskan strategi: {e}", exc_info=True)
            return None


class StrategyEncoder:
    """
    Agen AI yang menerjemahkan aturan strategi kualitatif menjadi kode Pandas
    untuk rekayasa fitur.
    """

    def __init__(self, api_pool: DistributedAIPool):
        self.api_pool = api_pool
        self.agent_role = "ai_engineer"

    def generate_feature_code(
        self, strategy: GenerativeStrategy, available_columns: list
    ) -> Optional[str]:
        logger.info(
            f"--- [Encoder] Menerjemahkan strategi '{strategy.strategy_name}' menjadi kode fitur ---"
        )
        prompt = f"""
        Anda adalah seorang Insinyur Data Python yang ahli dalam Pandas.
        Tugas Anda adalah menerjemahkan aturan strategi trading menjadi satu baris kode Pandas.

        'MANDAT REALITAS': Kode Anda harus valid secara sintaksis, hanya menggunakan kolom yang tersedia, dan secara logis merepresentasikan aturan yang diberikan.

        ATURAN STRATEGI:
        - Nama: {strategy.strategy_name}
        - Hipotesis: {strategy.hypothesis}
        - Aturan JIKA-MAKA: {strategy.rules}

        KOLOM YANG TERSEDIA DI DATAFRAME `df`:
        {available_columns}

        TUGAS ANDA:
        Tulis satu baris kode Python yang membuat kolom baru di DataFrame `df` bernama 'feature_{strategy.strategy_name.replace(" ", "_").lower()}'.
        Gunakan `np.where()` untuk mengimplementasikan logika JIKA-MAKA.
        Nilai kolom baru harus 1 jika kondisi 'IF' terpenuhi, dan 0 jika tidak.

        Contoh Output yang Benar:
        df['feature_regulatory_echo'] = np.where(
            (df['event_type'] == 'REGULATION') & (df['volatility_shock'] < 0.3), 1, 0)

        KEMBALIKAN HANYA SATU BARIS KODE PYTHON TERSEBUT.
        """
        try:

            code_line = self.api_pool.call_gemini_for_text(
                prompt, self.agent_role)


            if "df[" in code_line and "np.where" in code_line:
                logger.info(
                    f"✅ [Encoder] Kode fitur berhasil dibuat: {code_line}")
                return code_line.strip()
            else:
                logger.error(
                    f"[Encoder] Gagal membuat kode yang valid. Output: {code_line}"
                )
                return None
        except Exception as e:
            logger.error(
                f"[Encoder] Gagal menghasilkan kode fitur: {e}", exc_info=True)
            return None


class ArchitectAI:
    """
    Agen AI yang merancang dan menulis modul arsitektur baru untuk model
    sebagai respons terhadap kegagalan yang terdokumentasi (Metamorfosis).
    """

    def __init__(self, api_pool: DistributedAIPool):
        self.api_pool = api_pool
        self.agent_role = "supervisor"

    def design_new_architecture(
        self, failure_analysis: str, existing_code: str
    ) -> Optional[NewArchitectureModule]:
        logger.info(
            "\n--- [ArchitectAI] Memulai Sesi Desain Arsitektur Baru (Metamorfosis) ---"
        )

        prompt = f"""
        PERINGATAN: Analisis arsitektural diperlukan.
        'MANDAT REALITAS': Anda adalah seorang Insinyur Riset AI. Tugas Anda adalah merancang perbaikan arsitektural yang logis dan efisien untuk mengatasi kelemahan yang terdokumentasi. DILARANG membuat perubahan yang tidak relevan atau berhalusinasi.

        ANALISIS KEGAGALAN (FAKTA):
        ---
        {failure_analysis}
        ---

        KODE SUMBER YANG ADA (REALITA):
        ---
        {existing_code}
        ---

        TUGAS TEKNIS ANDA:
        1. Rancang sebuah kelas layer PyTorch baru yang secara spesifik menargetkan kelemahan yang dijelaskan.
        2. Tulis ulang seluruh metode `forward` dari kelas `HybridGNN_DPASTI` untuk mengintegrasikan layer baru Anda secara logis.
        3. Panggil tool `NewArchitectureModule` dengan hasil Anda. Pastikan semua field terisi dengan kode Python yang valid dan lengkap.
        """
        try:
            response = self.api_pool.call_gemini_with_tool(
                prompt, self.agent_role, NewArchitectureModule
            )
            if response:
                architecture = NewArchitectureModule(**response)
                logger.info(
                    f"✅ [ArchitectAI] Proposal arsitektur baru '{architecture.module_name}' telah dibuat."
                )
                return architecture
            return None
        except Exception as e:
            logger.error(
                f"[ArchitectAI] Gagal merancang arsitektur baru: {e}", exc_info=True
            )
            return None


class DynamicBetaVIB(pl.Callback):
    """
    Callback untuk secara dinamis menyesuaikan hyperparameter beta (β)
    untuk Variational Information Bottleneck selama pelatihan.
    Ini menerapkan strategi annealing, di mana penekanan pada kompresi (regularisasi)
    ditingkatkan secara bertahap.
    """

    def __init__(
        self,
        initial_beta: float = 1e-6,
        final_beta: float = 1e-3,
        ramp_up_duration_epochs: int = 25,
        start_ramp_up_epoch: int = 15,
    ):
        super().__init__()
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.ramp_up_duration = ramp_up_duration_epochs
        self.start_epoch = start_ramp_up_epoch
        self.end_epoch = start_ramp_up_epoch + ramp_up_duration_epochs
        self.beta_value = initial_beta

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        """Dipanggil di akhir setiap epoch pelatihan."""
        current_epoch = trainer.current_epoch

        if current_epoch < self.start_epoch:
            new_beta = self.initial_beta
        elif current_epoch >= self.end_epoch:
            new_beta = self.final_beta
        else:
            progress = (current_epoch - self.start_epoch) /                self.ramp_up_duration
            new_beta = self.initial_beta + progress * (
                self.final_beta - self.initial_beta
            )



        pl_module.hparams.beta_vib = new_beta
        self.beta_value = new_beta


        pl_module.log(
            "dynamic_beta", new_beta, on_step=False, on_epoch=True, prog_bar=True
        )


class WeightDecayScheduler(pl.Callback):
    """Callback untuk secara dinamis menyesuaikan weight_decay menggunakan jadwal cosine."""

    def __init__(self, wd_max: float, wd_min: float, max_epochs: int):
        super().__init__()
        self.wd_max = wd_max
        self.wd_min = wd_min
        self.max_epochs = max_epochs
        logger.info(
            f"[WeightDecayScheduler] Aktif. Rentang: {wd_min} -> {wd_max} selama {max_epochs} epoch."
        )

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        current_epoch = trainer.current_epoch

        new_wd = self.wd_min + 0.5 * (self.wd_max - self.wd_min) * (
            1 + math.cos(math.pi * current_epoch / self.max_epochs)
        )

        optimizer = trainer.optimizers[0]

        if hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer

        optimizer.param_groups[0]["weight_decay"] = new_wd
        pl_module.log("dynamic_wd", new_wd, on_step=False, on_epoch=True)





class SmartHybridScheduler(Callback):
    """
    Callback cerdas yang hanya akan beralih scheduler jika dua kondisi terpenuhi:
    1. Target epoch yang disarankan AI telah tercapai.
    2. Performa model (val_loss) telah menunjukkan tanda-tanda stagnasi.
    """

    def __init__(
        self,
        target_switch_epoch: int,
        patience: int = 3,
        min_delta: float = 0.001,
        cosine_t_max: int = 40,
    ):
        super().__init__()
        self.target_switch_epoch = target_switch_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.cosine_t_max = cosine_t_max
        self.switched_to_cosine = False
        self.wait_count = 0
        self.best_val_loss = float("inf")
        logger.info(
            f"[SmartScheduler] Diaktifkan. Target switch: epoch {target_switch_epoch}, Patience: {patience}."
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Fungsi callback yang dijalankan di akhir setiap epoch validasi.
        Tugasnya adalah memantau performa model dan mengganti learning rate scheduler
        dari tipe awal ke CosineAnnealingLR jika performa stagnan setelah epoch tertentu.
        """
        # skip if scheduler has already been switched to cosine
        if self.switched_to_cosine:
            return
        # choose appropriate metric: pretraining uses pretrain_loss_epoch; otherwise use val_loss
        is_pretrain_mode = "pretrain_loss_epoch" in trainer.callback_metrics
        monitor_metric = "pretrain_loss_epoch" if is_pretrain_mode else "val_loss"
        current_loss = trainer.callback_metrics.get(monitor_metric)
        # if loss metric not available, skip
        if current_loss is None:
            return
        # update best loss and reset patience counter if improved
        if current_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = current_loss
            self.wait_count = 0
        else:
            # increment wait counter if loss stagnates or worsens
            self.wait_count += 1
            logger.info(
                f"[SmartScheduler] Epoch {trainer.current_epoch}: Performa '{monitor_metric}' stagnan. "
                f"Hitungan kesabaran: {self.wait_count}/{self.patience}."
            )
        # if target epoch reached and patience exhausted, switch to cosine annealing scheduler
        if (
            trainer.current_epoch >= self.target_switch_epoch
            and self.wait_count >= self.patience
        ):
            logger.info(
                f"*** [SmartScheduler] KONDISI TERPENUHI! Beralih ke CosineAnnealingLR. ***"
            )
            # base optimizer (handles SAM wrappers etc.)
            base_optimizer = trainer.optimizers[0].base_optimizer
            # create new scheduler
            new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                base_optimizer, T_max=self.cosine_t_max
            )
            # replace scheduler in trainer if exists
            if trainer.lr_scheduler_configs:
                trainer.lr_scheduler_configs[0].scheduler = new_scheduler
                self.switched_to_cosine = True
            else:
                logger.warning(
                    "[SmartScheduler] Tidak dapat beralih, 'lr_scheduler_configs' tidak ditemukan di trainer."
                )




class AnticipatoryCognitiveSupervisor(Callback):
    """
    Callback yang mengimplementasikan Finite State Machine untuk memantau,
    mengantisipasi, dan melakukan intervensi pada pelatihan secara dinamis
    dengan bantuan agen AI eksternal (Llama 4 Maverick).
    """

    def __init__(
        self,
        api_pool: "DistributedAIPool",
        together_keys: dict,
        patience: int = 10,
        confirmation_patience: int = 7,
    ):
        super().__init__()
        self.api_pool = api_pool

        self.supervisor_key = together_keys.get("supervisor_maverick")
        self.supervisor_agent = (
            TogetherLLM(
                api_key=self.supervisor_key,
                model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            )
            if self.supervisor_key
            else None
        )



        self.state = "NORMAL"
        self.patience = patience
        self.confirmation_patience = confirmation_patience
        self.danger_zone = (0.005, 0.007)
        self.loss_history = []
        self.stagnation_epoch_start = -1

        if not self.supervisor_agent:
            logger.error(
                "[ACS] Kunci API untuk 'supervisor_maverick' tidak ditemukan. Supervisor tidak akan bisa melakukan konsultasi."
            )
        else:
            logger.info(

                f"🤖 [ACS] Anticipatory Cognitive Supervisor aktif. Konsultan AI: Llama-4-Maverick."
            )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if not self.supervisor_agent:
            return

        current_loss = trainer.callback_metrics.get("pretrain_loss_epoch")
        if current_loss is None:
            return

        current_loss = current_loss.item()
        self.loss_history.append(current_loss)


        if self.state == "NORMAL" and len(self.loss_history) >= self.patience:
            recent_losses = self.loss_history[-self.patience:]
            if all(
                self.danger_zone[0] <= loss <= self.danger_zone[1]
                for loss in recent_losses
            ):
                self.state = "STAGNANT"
                self.stagnation_epoch_start = trainer.current_epoch - self.patience
                logger.warning(
                    f"🚨 [ACS] State -> STAGNANT. Loss telah datar di 'danger zone' selama {self.patience} epoch sejak epoch {self.stagnation_epoch_start}."
                )
        elif self.state == "STAGNANT":
            if (
                trainer.current_epoch - self.stagnation_epoch_start
                >= self.patience + self.confirmation_patience
            ):
                self.state = "CONFIRMED_STAGNANT"
                logger.error(
                    f"🚨🚨 [ACS] State -> CONFIRMED_STAGNANT. Stagnasi berlanjut. Mempersiapkan konsultasi."
                )
                self._trigger_intervention(
                    "CONFIRMED_STAGNANT", trainer, pl_module)
        elif self.state == "CONFIRMED_STAGNANT":
            if current_loss > self.loss_history[-2] * 1.05:
                self.state = "IMMINENT_FAILURE"
                logger.critical(
                    f"🔥🔥🔥 [ACS] State -> IMMINENT_FAILURE! Kenaikan loss terdeteksi. Intervensi darurat!"
                )
                self._trigger_intervention(
                    "IMMINENT_FAILURE", trainer, pl_module)

    def _trigger_intervention(
        self, reason_state: str, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        advice = self._get_enlightenment(reason_state, trainer, pl_module)
        self._apply_enlightenment(advice, trainer)
        self.supervisor_agent = None
        logger.info(
            "[ACS] Intervensi selesai. Supervisor untuk trial ini sekarang non-aktif."
        )

    def _get_enlightenment(
        self, state: str, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> str:
        logger.info(
            f"🧠 [ACS] Meminta pencerahan dari Llama-4-Maverick karena state: {state}"
        )

        prompt_header = {
            "CONFIRMED_STAGNANT": "Pelatihan saya stagnan dalam waktu lama. Saya butuh intervensi halus untuk menembus plateau ini tanpa meledak.",
            "IMMINENT_FAILURE": "DARURAT! Pelatihan saya menunjukkan tanda-tanda akan meledak (loss mulai naik). Saya butuh intervensi darurat yang efektif untuk menstabilkan.",
        }.get(state, "Analisis kondisi pelatihan.")

        context = f"""
        Anda adalah seorang AI Research Scientist ahli dalam men-debug pelatihan deep learning.
        Situasi: {prompt_header}
        Data Teknis:
        - Epoch Saat Ini: {trainer.current_epoch}
        - Hyperparameter Aktif: {json.dumps(pl_module.hparams, indent=2, default=str)}
        - Riwayat Loss 20 Epoch Terakhir: {self.loss_history[-20:]}

        Opsi Intervensi yang Tersedia:
        1. "kurangi learning rate sebesar 10-50%": Untuk stabilisasi.
        2. "ganti scheduler ke ReduceLROnPlateau": Untuk pendekatan yang lebih konservatif.
        3. "lakukan SWA (Stochastic Weight Averaging)": Untuk mencari titik tengah yang lebih stabil.
        4. "hentikan pelatihan": Jika Anda yakin model sudah mencapai performa puncaknya.

        Tugas Anda: Berdasarkan data, berikan SATU saran intervensi paling strategis dari opsi di atas dan jelaskan alasan Anda dalam satu kalimat singkat.
        Contoh Jawaban: "Melihat stagnasi yang panjang, ganti scheduler ke ReduceLROnPlateau untuk pencarian yang lebih hati-hati."
        """
        try:
            advice = self.supervisor_agent.chat(context)
            logger.info(
                f"💡 [ACS] Pencerahan dari Llama-4-Maverick: '{advice}'"
            )
            return advice
        except Exception as e:
            logger.error(
                f"[ACS] Gagal mendapatkan pencerahan dari Llama-4-Maverick: {e}"
            )
            return "Gagal mendapatkan saran."

    def _apply_enlightenment(self, advice: str, trainer: "pl.Trainer"):
        advice = advice.lower()

        if "kurangi learning rate" in advice:
            match = re.search(r"(\d+)\s*%", advice)
            if match:
                reduction_pct = float(match.group(1)) / 100.0
                optimizer = trainer.optimizers[0]
                old_lr = optimizer.param_groups[0]["lr"]
                new_lr = old_lr * (1 - reduction_pct)
                optimizer.param_groups[0]["lr"] = new_lr
                logger.warning(
                    f"🔧 [ACS] INTERVENSI: Learning rate dipotong dari {old_lr:.6f} menjadi {new_lr:.6f} sesuai saran."
                )
        elif "ganti scheduler" in advice and "reducelronplateau" in advice.replace(
            " ", ""
        ):
            if trainer.lr_schedulers:
                optimizer = trainer.optimizers[0]
                new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", patience=5, factor=0.5, verbose=True
                )
                trainer.lr_schedulers[0]["scheduler"] = new_scheduler
                logger.warning(
                    "🔧 [ACS] INTERVENSI: Scheduler diganti ke ReduceLROnPlateau sesuai saran."
                )
            else:
                logger.error(
                    "[ACS] Gagal intervensi: trainer.lr_schedulers tidak ditemukan."
                )
        elif "swa" in advice:
            logger.warning(
                "🔧 [ACS] INTERVENSI: SWA disarankan. Callback SWA akan bekerja di akhir pelatihan."
            )
        elif "hentikan pelatihan" in advice:
            logger.warning(
                "🔧 [ACS] INTERVENSI: Pelatihan dihentikan lebih awal sesuai saran."
            )
            trainer.should_stop = True
        else:
            logger.warning(
                "[ACS] Saran tidak dapat dieksekusi otomatis, pelatihan dilanjutkan."
            )


class ProactiveGradientSupervisor(pl.Callback):
    """
    Callback yang secara proaktif memonitor norma gradien, melakukan intervensi,
    dan melaporkan ringkasan di akhir epoch untuk menjaga log tetap bersih.
    """

    def __init__(
        self,
        threshold: float = 15.0,
        patience: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-7,
    ):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.spike_counter = 0

        self.epoch_breach_count = 0
        self.max_norm_in_epoch = 0.0
        logger.info(
            f"🛡️ ProactiveGradientSupervisor v2 (Professional Logging) aktif. Threshold: {self.threshold}, Patience: {self.patience}."
        )

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        """Dipanggil di awal setiap epoch, mereset counter epoch."""
        self.epoch_breach_count = 0
        self.max_norm_in_epoch = 0.0

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Dipanggil setelah backward() tetapi sebelum optimizer melangkah.
        """
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        if total_norm > self.threshold:
            self.spike_counter += 1

            self.epoch_breach_count += 1
            if total_norm > self.max_norm_in_epoch:
                self.max_norm_in_epoch = total_norm

        else:

            self.spike_counter = 0


        if self.spike_counter >= self.patience:
            logger.critical(
                f"🔥🔥🔥 [GradSupervisor] INTERVENSI! Gradien tidak stabil selama {self.patience} langkah."
            )
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]

            if current_lr > self.min_lr:
                new_lr = max(current_lr * self.factor, self.min_lr)
                optimizer.param_groups[0]["lr"] = new_lr
                logger.warning(
                    f"Learning Rate dipotong secara paksa: {current_lr:.7f} -> {new_lr:.7f}"
                )

                self.spike_counter = 0
            else:
                logger.error(
                    "[GradSupervisor] LR sudah minimum. Tidak bisa memotong lebih lanjut. Risiko ledakan tinggi!"
                )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        """Dipanggil di akhir epoch untuk memberikan ringkasan."""
        if self.epoch_breach_count > 0:
            logger.warning(
                f"📈 [GradSupervisor Summary] Epoch {trainer.current_epoch}: "
                f"Norma gradien terlampaui sebanyak {self.epoch_breach_count} kali. "
                f"Nilai tertinggi: {self.max_norm_in_epoch:.2f}"
            )


class InputArmor(nn.Module):
    """
    Lapisan pertahanan input (Kulit Buaya) yang menstabilkan data mentah
    menggunakan Autoencoder sebelum masuk ke model utama.
    Ini adalah implementasi dari 'Lapisan Osteoderm'.
    """

    def __init__(
        self, input_dim: int, hidden_dim_ratio: float = 0.5, is_trained: bool = False
    ):
        super().__init__()

        self.shock_absorber = nn.Tanh()


        hidden_dim = max(1, int(input_dim * hidden_dim_ratio))
        self.autoencoder = EnhancedAutoencoder(input_dim, hidden_dim)
        self.is_trained = is_trained
        logger.info(
            f"🐊 InputArmor (Kulit Buaya) diinisialisasi. Trained: {self.is_trained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Jika armor sudah terlatih, gunakan untuk 'membersihkan' input.
        Jika tidak, hanya gunakan shock absorber sederhana.
        """
        if self.is_trained:

            _, reconstructed_x = self.autoencoder(x)
            return reconstructed_x
        else:

            return self.shock_absorber(x)

    def train_armor(self, data_loader: DataLoader, epochs: int = 15, lr: float = 1e-3):
        """Fungsi untuk melatih autoencoder internal secara terpisah."""
        logger.info("--- 🐊 Memulai Pelatihan Armor (Autoencoder) ---")
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        device = next(self.parameters()).device

        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_features in data_loader:
                batch_features = batch_features[0].to(
                    device
                )
                optimizer.zero_grad()
                _, outputs = self.autoencoder(batch_features)
                loss = criterion(outputs, batch_features)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"  > Armor Training Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}"
                )

        self.is_trained = True
        logger.info("--- ✅ Pelatihan Armor Selesai. Armor sekarang aktif. ---")


class PolycarbonateLayer(nn.Module):
    """
    Lapisan penyerap energi sisa, analog dengan Polycarbonate di kaca anti peluru.
    Menggunakan Liquid Time-Constant (LTC) Network untuk menangani dinamika
    input yang sudah dihaluskan oleh InputArmor secara fleksibel.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()







        wiring = ncps.wirings.FullyConnected(hidden_dim)


        self.ltc_cell = LTC(input_dim, wiring, return_sequences=True)

        self.dropout = nn.Dropout(p=dropout_rate)


        self.output_transform = nn.Linear(hidden_dim, input_dim)

        logger.info(
            f"💎 PolycarbonateLayer (LTC) diinisialisasi. Input: {input_dim}, Hidden: {hidden_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:




        ltc_output, _ = self.ltc_cell(x)


        ltc_output = self.dropout(ltc_output)
        final_output = self.output_transform(ltc_output)



        return x + final_output


class MoELayer(nn.Module):
    """
    Implementasi dari Mixture of Experts (MoE) Layer.
    Ini adalah realisasi dari 'Kapasitas Adaptif' dengan 'Pakar'.
    """

    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.top_k = top_k


        self.gating_network = nn.Linear(input_dim, num_experts)


        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim * 2),
                    nn.SiLU(),
                    nn.Linear(input_dim * 2, output_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):

        original_shape = x.shape
        x_reshaped = x.view(-1, original_shape[-1])


        gate_logits = self.gating_network(x_reshaped)


        top_k_weights, top_k_indices = torch.topk(
            gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)


        expert_outputs = torch.stack(
            [expert(x_reshaped) for expert in self.experts], dim=1
        )


        final_output = torch.zeros_like(expert_outputs[:, 0, :])
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(-1)


            selected_expert_output = expert_outputs[
                torch.arange(x_reshaped.size(0)), expert_idx
            ]
            final_output += weight * selected_expert_output

        return final_output.view(original_shape)


class MetaOptimizer:
    """
    Menjalankan meta-optimisasi untuk secara otomatis memilih sampler Optuna terbaik
    berdasarkan turnamen empiris dan analisis heuristik dataset.
    """

    def __init__(self, train_fn, base_hparams, X_train):
        """
        Args:
            train_fn (Callable): Fungsi objective yang akan dioptimalkan.
            base_hparams (dict): Hyperparameter dasar.
            X_train (np.ndarray): Data fitur training untuk analisis karakteristik.
        """
        self.train_fn = train_fn
        self.base_hparams = base_hparams
        self.X_train = X_train


        self.samplers_to_test = {
            "TPE": TPESampler(seed=42),
            "CMA-ES": CmaEsSampler(seed=42),
            "Random": RandomSampler(seed=42),
        }
        logger.info(
            f"[MetaOptimizer] Diinisialisasi untuk mengadu {list(self.samplers_to_test.keys())}."
        )

    def _objective_for_tournament(self, trial):
        """Wrapper objective function untuk turnamen singkat."""

        hparams = self._sample(trial)
        hparams["max_epochs"] = 10


        score, _, _, _ = self.train_fn(hparams)
        return score

    def _sample(self, trial):
        """Fungsi _sample yang konsisten untuk semua trial."""
        hp = self.base_hparams.copy()
        hp["lr"] = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        hp["d_model"] = trial.suggest_categorical("d_model", [64, 128, 256])
        hp["dropout"] = trial.suggest_float("dropout", 0.1, 0.4)

        return hp

    def _run_sampler_tournament(self):
        """Langkah 1: Menjalankan studi singkat untuk setiap sampler."""
        logger.info("\n--- [MetaOptimizer] Memulai Turnamen Sampler ---")
        results = {}

        for name, sampler in self.samplers_to_test.items():
            logger.info(f"  > Menguji Sampler: {name}...")
            study = optuna.create_study(direction="minimize", sampler=sampler)


            study.optimize(self._objective_for_tournament,
                           n_trials=3, n_jobs=1)
            results[name] = study.best_value
            logger.info(
                f"  > Hasil {name}: loss terbaik = {study.best_value:.6f}")


        winner = min(results, key=results.get)
        logger.info(f"--- Pemenang Turnamen: {winner} ---")
        return winner, results[winner]

    def _get_heuristic_recommendation(self):
        """Langkah 2: Menganalisis dataset dan memberikan rekomendasi."""
        logger.info(
            "\n--- [MetaOptimizer] Memulai Analisis Heuristik Dataset ---")
        n_samples, n_features = self.X_train.shape

        logger.info(
            f"  > Karakteristik Dataset: {n_samples} sampel, {n_features} fitur."
        )


        if n_features > 40 and n_samples > 5000:
            recommendation = "CMA-ES"
            reason = "Cocok untuk masalah dimensi tinggi dan kompleks."
        else:
            recommendation = "TPE"
            reason = "Pilihan serbaguna yang efisien untuk sebagian besar kasus."

        logger.info(
            f"--- Rekomendasi Heuristik: {recommendation} ({reason}) ---")
        return recommendation

    def run(self):
        """Langkah 3: Menjalankan semua proses dan membuat keputusan akhir."""
        logger.info("\n{'='*25} MEMULAI META-OPTIMIZATION {'='*25}")


        tournament_winner, tournament_score = self._run_sampler_tournament()
        heuristic_choice = self._get_heuristic_recommendation()


        logger.info("\n--- [MetaOptimizer] Pengambilan Keputusan Akhir ---")
        logger.info(
            f"  > Hasil Turnamen Empiris: {tournament_winner} (score: {tournament_score:.6f})"
        )
        logger.info(f"  > Rekomendasi Aturan Heuristik: {heuristic_choice}")




        final_choice = tournament_winner

        if tournament_winner != heuristic_choice:
            logger.warning(
                f"  > Terdapat perbedaan! Memilih {final_choice} berdasarkan bukti empiris dari turnamen."
            )
        else:
            logger.info(
                f"  > Kedua metode sepakat. Pilihan final adalah: {final_choice}"
            )

        logger.info(f"{'='*25} META-OPTIMIZATION SELESAI {'='*25}\n")
        return self.samplers_to_test[final_choice]


class AITranslator:
    """
    Agen yang bertanggung jawab untuk menstandarisasi bahasa komunikasi antar AI.
    Ia mendeteksi bahasa, dan menerjemahkan teks yang tidak standar ke Bahasa Inggris.
    """

    def __init__(self, together_keys: dict):
        self.translator_key = together_keys.get("ai_translator")
        if self.translator_key:
            self.translator_agent = TogetherLLM(
                api_key=self.translator_key, model_name="lgai/exaone-3-5-32b-instruct"
            )
            logger.info("[Translator] Agen Penerjemah AI (Exaone 3.5) siap.")
        else:
            self.translator_agent = None
            logger.warning(
                "[Translator] Kunci API untuk 'ai_translator' tidak tersedia. Terjemahan akan dilewati."
            )

    def _detect_language(self, text: str) -> Optional[str]:
        """Mendeteksi kode bahasa (misal: 'en', 'es', 'vi') dari teks."""
        try:

            sample_text = " ".join(re.findall(r"\b\w+\b", text.lower())[:50])
            if not sample_text:
                return "en"
            return detect(sample_text)
        except LangDetectException:
            logger.warning(
                "[Translator] Gagal mendeteksi bahasa, mengasumsikan format campuran."
            )
            return "mixed"

    def _translate_text(self, text: str, source_agent: str) -> str:
        """Menerjemahkan teks ke Bahasa Inggris yang bersih dan kontekstual."""
        if not self.translator_agent:
            logger.warning(
                f"[Translator] Melewatkan terjemahan untuk agen {source_agent} karena tidak ada API key."
            )
            return text

        logger.info(f"  > Menerjemahkan output dari agen '{source_agent}'...")
        prompt = f"""
        You are an expert multilingual translator and editor.
        Your task is to process the following text from an AI agent.
        1.  Identify the core message and technical analysis.
        2.  Translate any non-English parts into clear, fluent English.
        3.  Correct any typos, grammatical errors, or awkward phrasing.
        4.  The final output MUST BE ONLY the clean, translated, and edited English text, preserving the original intent.

        RAW TEXT FROM AGENT:
        ---
        {text}
        ---

        CLEAN ENGLISH OUTPUT:
        """
        try:
            translated_text = self.translator_agent.chat(prompt)
            logger.info("    - Terjemahan berhasil.")
            return translated_text
        except Exception as e:
            logger.error(f"    - Terjemahan gagal: {e}")
            return text

    def standardize_response(self, raw_text: str, source_agent_name: str) -> str:
        """
        Fungsi publik utama untuk membersihkan dan menstandarisasi respons dari agen lain.
        """
        detected_lang = self._detect_language(raw_text)

        if detected_lang == "en":

            logger.info(
                f"[Translator] Bahasa Inggris terdeteksi dari '{source_agent_name}'. Melewatkan terjemahan."
            )
            return raw_text.strip()
        else:
            logger.warning(
                f"[Translator] Bahasa non-Inggris ('{detected_lang}') terdeteksi dari '{source_agent_name}'. Memulai proses standardisasi."
            )
            return self._translate_text(raw_text, source_agent_name)





class NewTableProposal(BaseModel):
    """Skema untuk usulan tabel baru dari Cognitive Metamorphosis Protocol."""

    table_name: str = Field(
        description="Nama tabel baru yang valid, contoh: 'causal_chains'."
    )
    schema_sql: str = Field(
        description="Perintah SQL `CREATE TABLE` lengkap untuk tabel baru."
    )
    reasoning: str = Field(
        description="Penjelasan mengapa tabel ini diperlukan untuk menangkap pola baru."
    )


def run_causal_validation_cycle(nsmm: NSMM, recent_data: pd.DataFrame):
    """
    Menguji validitas hipotesis kausal aktif terhadap data terbaru dan memperbarui
    skor kepercayaannya.
    """
    logger.info(
        "🔬 [Causal Validator] Memulai siklus validasi hipotesis kausal...")
    active_rules = nsmm.get_active_causal_hypotheses()
    if active_rules.empty:
        logger.info(
            "🔬 [Causal Validator] Tidak ada hipotesis aktif untuk divalidasi.")
        return

    for _, rule in active_rules.iterrows():
        treatment = rule["treatment_variable"]
        outcome = rule["outcome_variable"]
        effect = rule["estimated_effect"]
        confidence = rule["confidence_score"]
        status = rule["status"]

        if treatment not in recent_data.columns or outcome not in recent_data.columns:
            continue


        actual_outcome_change = recent_data[outcome].diff().mean()
        treatment_change = recent_data[treatment].diff().mean()
        predicted_outcome_change = treatment_change * effect
        validation_error = abs(actual_outcome_change -
                               predicted_outcome_change)


        if validation_error < 0.1:
            confidence = min(1.0, confidence + 0.05)
        else:
            confidence = max(0.0, confidence - 0.20)

        if confidence < 0.1:
            status = "archived"
        elif confidence < 0.5:
            status = "decaying"
        else:
            status = "active"


        nsmm.update_causal_hypothesis(rule["rule_id"], confidence, status)

    logger.info("🔬 [Causal Validator] Siklus validasi selesai.")


def run_cognitive_metamorphosis_protocol(nsmm: NSMM, api_pool: "DistributedAIPool"):
    """
    Menganalisis kegagalan hipotesis dan secara proaktif mengusulkan skema tabel
    baru jika pola yang ada tidak lagi memadai.
    """
    logger.info("🦋 [Metamorphosis] Memeriksa kebutuhan untuk evolusi skema...")



    with sqlite3.connect(nsmm.db_path) as conn:
        df_rules = pd.read_sql_query(
            "SELECT status FROM causal_hypotheses", conn)

    trigger_activated = False
    if not df_rules.empty:
        decay_ratio = (df_rules["status"] != "active").sum() / len(df_rules)
        if decay_ratio > 0.5:
            trigger_activated = True

    if not trigger_activated:
        logger.info(
            "🦋 [Metamorphosis] Skema saat ini masih memadai. Tidak ada evolusi yang diperlukan."
        )
        return

    logger.warning(
        "🦋 [Metamorphosis] Pemicu aktif! Model mendeteksi pola yang tidak dapat direpresentasikan."
    )

    anomalous_pattern_description = "Contoh: Ditemukan pola sekuensial di mana sebuah peristiwa makroekonomi (misal: Suku Bunga) diikuti oleh pergerakan mata uang, yang kemudian mempengaruhi aliran dana pada sebuah saham."

    prompt = f"""
    Anda adalah seorang Arsitek Basis Data AI. Sistem saya mendeteksi pola baru yang tidak bisa disimpan di skema saat ini.
    Pola Baru: {anomalous_pattern_description}
    Skema Saat Ini: `causal_hypotheses(treatment, outcome, effect)`

    Tugas: Rancang skema tabel SQL baru untuk menyimpan pola sekuensial atau multi-variabel ini.
    Kembalikan jawaban Anda HANYA dalam format JSON yang sesuai dengan skema `NewTableProposal`.
    """
    try:
        response = api_pool.call_gemini_with_tool(
            prompt, "supervisor", NewTableProposal
        )
        proposal = NewTableProposal(**response)
        logger.info(
            f"🦋 [Metamorphosis] Proposal skema baru diterima: {proposal.table_name}"
        )
        logger.info(f"   Alasan: {proposal.reasoning}")


        if "CREATE TABLE" in proposal.schema_sql.upper():
            with sqlite3.connect(nsmm.db_path) as conn:
                conn.execute(proposal.schema_sql)
                logger.info(
                    f"🦋 [Metamorphosis] SUKSES! Tabel '{proposal.table_name}' telah dibuat di database kognitif."
                )
    except Exception as e:
        logger.error(f"🦋 [Metamorphosis] Gagal menjalankan evolusi skema: {e}")





def run_autonomous_learning_cycle(
    nsmm: NSMM, brain: Brain, api_pool: "DistributedAIPool"
):
    """
    Menjalankan siklus belajar matematika otonom dengan Prinsip Eskalasi Cerdas.
    """
    logger.info(
        "👨‍🏫 [Auto-Curriculum] Memulai siklus belajar matematika otonom...")


    stats = nsmm.get_problem_solving_stats()
    current_difficulty = "Mudah"
    if stats.get("Mudah", 0.0) > 0.95:
        current_difficulty = "Menengah"
    if stats.get("Menengah", 0.0) > 0.95:
        current_difficulty = "Sulit"


    problem = nsmm.get_unsolved_problem(difficulty=current_difficulty)
    if not problem:
        logger.info(
            f"✅ [Auto-Curriculum] Tidak ada soal '{current_difficulty}' yang belum terpecahkan."
        )
        return

    problem_text = problem["problem_text"]
    logger.info(f"    -> Latihan soal: '{problem_text}'")


    keywords_prompt = f"Sebutkan 2-3 kata kunci matematika utama dari soal ini: '{problem_text}'. Jawab sebagai list Python."
    try:
        keywords_str = api_pool.call_gemini_for_text(
            keywords_prompt, "experimentalist")
        keywords = ast.literal_eval(keywords_str)
    except:
        keywords = []


    learned_method = nsmm.find_relevant_math_method(keywords)


    if learned_method:
        logger.info(
            "    -> 🦾 Mencoba menyelesaikan soal secara mandiri menggunakan intuisi..."
        )
        try:

            internal_solution = str(
                eval(learned_method)
            )
            nsmm.update_math_problem_solution(
                problem["id"],
                "solved_internally",
                internal_solution,
                "N/A",
                learned_method,
            )
            logger.info(
                f"    -> ✅ Berhasil diselesaikan secara mandiri. Jawaban: {internal_solution}"
            )
            return
        except Exception as e:
            logger.warning(
                f"    -> ⚠️ Gagal mencoba mandiri meskipun ada metode. Eskalasi ke Guru. Error: {e}"
            )


    logger.warning(
        "    -> ❓ Tidak ada metode di memori. Membutuhkan bantuan 'Guru' LLM..."
    )
    guru_prompt = f"""
    Anda adalah seorang profesor matematika dan programmer Python ahli.
    Soal: "{problem_text}"
    Jawaban dari murid saya: "Gagal menyelesaikan secara internal."

    Tugas Anda, jawab dalam format JSON:
    1. "is_correct": false
    2. "correct_solution": (string) Tunjukkan jawaban yang benar.
    3. "best_method": (string) Berikan satu baris kode Python menggunakan pustaka SymPy yang merupakan cara paling efisien untuk menyelesaikan soal ini.
    4. "problem_keywords": (list of strings) Berikan 2-3 kata kunci yang mendeskripsikan tipe soal ini.
    """
    try:
        guru_response_str = api_pool.call_gemini_for_text(
            guru_prompt, "supervisor")
        guru_feedback = robust_json_extract(guru_response_str, model=None)

        learned_method = guru_feedback.get("best_method", "N/A")
        guru_solution = guru_feedback.get("correct_solution", "N/A")

        for keyword in guru_feedback.get("problem_keywords", []):
            brain.dkg.add_node(keyword, "Math_Concept", layer="Knowledge")
            brain.dkg.add_node(learned_method, "Python_Tool", layer="Tools")
            brain.dkg.add_edge(keyword, learned_method, "solved_by")

        nsmm.update_math_problem_solution(
            problem["id"], "solved_with_help", "N/A", guru_solution, learned_method
        )
        logger.info(
            f"    -> 👨‍🏫 Pembelajaran dari Guru selesai. Metode baru '{learned_method}' disimpan."
        )
    except Exception as e:
        logger.error(f"Gagal mendapatkan feedback dari Guru AI: {e}")


class ResourceGuardAI:
    """
    Agen AI yang bertugas ganda:
    1. Menghitung estimasi penggunaan RAM berdasarkan hyperparameter.
    2. Memberikan penilaian kualitatif tentang risiko overfitting menggunakan model LLM.
    """

    TOTAL_SYSTEM_RAM_GB = 31.8
    RAM_BUFFER_GB = 1.0

    def __init__(self, together_keys: dict):
        self.guard_key = together_keys.get("resource_guard")
        if self.guard_key:
            self.llm_agent = TogetherLLM(
                api_key=self.guard_key, model_name="lgai/exaone-deep-32b"
            )
        else:
            self.llm_agent = None
            logger.warning(
                "[ResourceGuard] Kunci API untuk 'resource_guard' tidak ada. Penilaian kualitatif akan dilewati."
            )

    def _estimate_ram_usage_gb(self, hparams: dict) -> float:
        """Heuristik sederhana untuk mengestimasi penggunaan RAM."""

        base_ram = 2.0


        d_model_ram = hparams.get("d_model", 128) /            128 * 0.5
        features_ram = (
            hparams.get("top_k_features", 50) / 50 * 0.2
        )


        window = hparams.get("window", 60)
        batch_size = hparams.get("batch_size", 64)
        data_ram = (
            (batch_size * window * hparams.get("top_k_features", 50) * 8)
            / (1024**3)
            * 2
        )

        total_estimated = base_ram + d_model_ram + features_ram + data_ram
        return total_estimated

    def assess_risk(self, hparams_proposal: dict) -> dict:
        """
        Menilai proposal hyperparameter dari segi sumber daya dan risiko overfitting.
        """
        logger.info("  [RESOURCE_GUARD] Menilai risiko proposal...")


        estimated_ram = self._estimate_ram_usage_gb(hparams_proposal)
        remaining_ram = self.TOTAL_SYSTEM_RAM_GB - estimated_ram
        is_safe_quantitatively = remaining_ram >= self.RAM_BUFFER_GB

        quantitative_verdict = (
            f"Perhitungan Kuantitatif: Estimasi pemakaian RAM {estimated_ram:.2f} GB. "
            f"Sisa RAM {remaining_ram:.2f} GB. Batas aman terlampaui: {not is_safe_quantitatively}."
        )
        logger.info(f"    - {quantitative_verdict}")


        qualitative_verdict = "Penilaian kualitatif AI dilewati (tidak ada API key)."
        if self.llm_agent:
            prompt = f"""
            Anda adalah seorang ahli dalam mendeteksi potensi overfitting dalam model time series.
            Berdasarkan proposal hyperparameter berikut, berikan analisis singkat (2-3 kalimat) mengenai risikonya.
            Fokus pada interaksi antar parameter (misalnya, lr tinggi dengan dropout rendah).

            Proposal:
            ```json
            {json.dumps(make_json_serializable(hparams_proposal), indent=2)}
            ```
            """
            try:
                qualitative_verdict = self.llm_agent.chat(prompt)
                logger.info(
                    f"    - Penilaian Kualitatif AI: {qualitative_verdict}")
            except Exception as e:
                logger.error(
                    f"    - Gagal mendapatkan penilaian kualitatif: {e}")
                qualitative_verdict = f"Gagal mendapatkan penilaian: {e}"

        return {
            "is_safe": is_safe_quantitatively,
            "quantitative_analysis": quantitative_verdict,
            "qualitative_analysis": qualitative_verdict,
            "estimated_ram_gb": estimated_ram,
        }







class PanelMusyawarahAI:
    """
    Mengorkestrasi musyawarah AI menggunakan `instructor` untuk output terjamin
    dan logika fallback Arbiter Cadangan untuk keandalan maksimal.
    """

    def __init__(
        self,
        panelis: dict,
        arbiter_model_name: str,
        translator: AITranslator,
        together_api_keys: dict,
        gemini_api_pool: DistributedAIPool,
        together_roles: dict,
    ):
        self.panelis = panelis
        self.arbiter_model_name = arbiter_model_name
        self.translator = translator
        self.together_api_keys = together_api_keys
        self.gemini_api_pool = gemini_api_pool

        self.arbiter_config = together_roles.get(self.arbiter_model_name)
        self.arbiter_api_key = self.together_api_keys.get(
            self.arbiter_model_name)


        self.backup_arbiter_agent_name = (
            "json_finalizer"
        )

        logger.info(
            f"[Dewan Musyawarah v3.0] Siap dengan {len(self.panelis)} panelis, 1 Arbiter Utama, 1 Arbiter Cadangan, dan `instructor`."
        )

    def _minta_proposal_individu(
        self, nama_panelis: str, agent_config: dict, konteks: str
    ) -> Optional[HyperparameterProposal]:
        """
        Meminta satu panelis untuk membuat proposal menggunakan `instructor` untuk Gemini,
        atau parsing teks dengan prompt yang diperkaya skema untuk model lain.
        """

        gemini_prompt = f"""
        Berdasarkan konteks teknis berikut, usulkan satu set hyperparameter yang optimal.
        Berikan alasan yang kuat dan logis untuk pilihan Anda di dalam field 'reasoning'.

        Konteks:
        ---
        {konteks}
        ---
        """



        pydantic_schema = json.dumps(
            HyperparameterProposal.model_json_schema(), indent=2
        )
        external_prompt = f"""
        Anda adalah asisten ahli yang tugasnya adalah menghasilkan objek JSON.
        Berdasarkan konteks yang diberikan, buat satu proposal hyperparameter.
        Pastikan output Anda HANYA berupa objek JSON valid yang sesuai dengan skema di bawah ini, tanpa teks tambahan, penjelasan, atau markdown.

        Konteks:
        ---
        {konteks}
        ---

        Skema JSON yang harus diikuti:
        ```json
        {pydantic_schema}
        ```
        """

        try:

            if "gemini" in nama_panelis.lower() or "tutor" in nama_panelis.lower():
                logger.info(
                    f"  > [Kubu Gemini] Meminta proposal dari: {nama_panelis} via `instructor`..."
                )
                return self.gemini_api_pool.call_gemini_with_instructor(
                    prompt=gemini_prompt,
                    agent="tutor",
                    response_model=HyperparameterProposal,
                )


            elif "groq" in nama_panelis.lower():
                logger.info(
                    f"  > [Kubu Eksternal] Meminta proposal dari: {nama_panelis}..."
                )
                api_key = self.together_api_keys.get("grok_key")
                if not api_key:
                    return None
                agent = GrokLLM(
                    api_key=api_key,
                    model_name=agent_config.get("primary", "grok-3-mini"),
                )
                respons_mentah_str = agent.chat(
                    external_prompt, system_prompt="You are a JSON-only output AI."
                )


            else:
                logger.info(
                    f"  > [Kubu Eksternal] Meminta proposal dari: {nama_panelis}..."
                )
                api_key = self.together_api_keys.get(nama_panelis)
                if not api_key:
                    return None
                respons_mentah_str = call_agent_with_fallback(
                    api_key,
                    agent_config["primary"],
                    agent_config.get("backups", []),
                    external_prompt,
                )


            if not respons_mentah_str:
                return None
            parsed_obj = robust_json_extract(
                respons_mentah_str, model=HyperparameterProposal
            )
            if parsed_obj:
                return parsed_obj
            return None

        except Exception as e:
            logger.error(
                f"Gagal total mendapatkan proposal dari {nama_panelis}: {e}",
                exc_info=True,
            )
            return None

    def _jalankan_arbiter(
        self, semua_proposal_dict: dict, agent_name: str, model_type: str
    ) -> Optional[ArbiterDecision]:
        """Fungsi terpusat untuk menjalankan Arbiter Utama atau Cadangan."""


        pydantic_schema_str = json.dumps(
            ArbiterDecision.model_json_schema(), indent=2)


        prompt_arbiter = f"""
        Anda adalah seorang Arbiter AI Grandmaster. Tugas Anda adalah menciptakan satu set hyperparameter 'juara'
        dari berbagai proposal di bawah ini. Analisis pro-kontra dari setiap proposal dan sintesis menjadi satu keputusan terbaik.

        PROPOSAL-PROPOSAL PANELIS:
        ```json
        {json.dumps(make_json_serializable(semua_proposal_dict), indent=2)}
        ```

        ATURAN KETAT: Output Anda HARUS HANYA berupa objek JSON tunggal yang valid dan sesuai dengan skema di bawah ini.
        Jangan tambahkan teks penjelasan, komentar, atau markdown.
        Pastikan `final_decision` memiliki semua field yang diperlukan oleh `HyperparameterProposal`.

        Skema JSON yang harus diikuti:
        ```json
        {pydantic_schema_str}
        ```
        """
        try:
            if model_type == "GEMINI_INSTRUCTOR":
                logger.info(
                    f"  > Memanggil Arbiter Cadangan: {agent_name} (Gemini via Instructor)..."
                )
                decision = self.gemini_api_pool.call_gemini_with_instructor(
                    prompt=prompt_arbiter,
                    agent=agent_name,
                    response_model=ArbiterDecision,
                    max_retries=2,
                )
                return decision

            elif model_type == "TOGETHER_TEXT":
                logger.info(
                    f"  > Memanggil Arbiter Utama: {self.arbiter_model_name} (Together.AI via Teks)..."
                )

                if not self.arbiter_api_key or not self.arbiter_config:
                    logger.error(
                        f"Konfigurasi atau API Key untuk Arbiter Utama '{self.arbiter_model_name}' tidak ditemukan."
                    )
                    return None

                response_str = call_agent_with_fallback(
                    self.arbiter_api_key,
                    self.arbiter_config["primary"],
                    self.arbiter_config.get("backups", []),
                    prompt_arbiter,
                )
                if not response_str:
                    return None


                parsed_dict = robust_json_extract(
                    response_str, model=ArbiterDecision
                )
                if parsed_dict:
                    return parsed_dict
                return None

        except Exception as e:
            logger.error(
                f"  > Panggilan ke Arbiter ({agent_name}) gagal total: {e}",
                exc_info=True,
            )
            return None

        return None

    def gelar_musyawarah(self, konteks_terakhir: str) -> Optional[dict]:
        """Menjalankan alur musyawarah v3.0 dengan `instructor` dan fallback cerdas."""
        logger.info(
            "\n--- MEMULAI MUSYAWARAH v3.0 (INSTRUCTOR & REDUNDANSI) ---")


        logger.info(
            "  > [TAHAP 1/4] Mengumpulkan proposal dari semua panelis...")
        semua_proposal = {}
        for nama, config in self.panelis.items():
            proposal_obj = self._minta_proposal_individu(
                nama, config, konteks_terakhir)
            if proposal_obj:

                semua_proposal[nama] = proposal_obj.model_dump()

        if not semua_proposal:
            logger.error(
                "Tidak ada satupun panelis yang memberikan proposal valid. Musyawarah dibatalkan."
            )
            return None


        if len(semua_proposal) == 1:
            nama_tunggal = list(semua_proposal.keys())[0]
            logger.warning(
                f"Hanya 1 proposal diterima dari '{nama_tunggal}'. Langsung digunakan tanpa musyawarah."
            )
            return list(semua_proposal.values())[0]


        logger.info(
            f"  > [TAHAP 2/4] {len(semua_proposal)} proposal terkumpul. Arbiter Utama ({self.arbiter_model_name}) mengambil alih..."
        )
        keputusan_final_obj = self._jalankan_arbiter(
            semua_proposal, self.arbiter_model_name, "TOGETHER_TEXT"
        )


        if keputusan_final_obj is None:
            logger.warning(
                "  > Arbiter Utama GAGAL. Mengaktifkan protokol fallback Arbiter Cadangan..."
            )
            keputusan_final_obj = self._jalankan_arbiter(
                semua_proposal, self.backup_arbiter_agent_name, "GEMINI_INSTRUCTOR"
            )


        if keputusan_final_obj:
            logger.info("  > ✔️ Keputusan Arbiter final diterima!")
            logger.info(
                f"  > Alasan Sintesis: {keputusan_final_obj.synthesis_reasoning}"
            )

            return keputusan_final_obj.final_decision.model_dump()
        else:
            logger.error(
                "  > FATAL: Arbiter Utama dan Cadangan GAGAL. Musyawarah tidak menghasilkan keputusan."
            )

            logger.warning(
                "  > Menggunakan fallback terakhir: proposal paling aman.")
            return get_safest_proposal_fallback(list(semua_proposal.values()))


def get_safest_proposal_fallback(proposals: list[dict]) -> dict:
    """Fungsi helper untuk fallback terakhir jika semua arbiter gagal."""
    if not proposals:
        return {}
    safest = min(proposals, key=lambda p: p.get("learning_rate", float("inf")))
    logger.info(
        f"  > Fallback terakhir memilih proposal dengan LR terendah: {safest.get('learning_rate')}"
    )
    return safest


class IdleResourceManager:
    """
    Versi 3.0: Mengelola sumber daya idle dengan budget dinamis
    berdasarkan persentase total RAM sistem.
    """

    def __init__(self, classifier: MultiPathElectraClassifier):
        self.classifier = classifier


        self.total_system_ram = psutil.virtual_memory().total / (1024**3)
        self.ram_budget_gb = (
            self.total_system_ram * 0.65
        )


        self.ram_per_path_gb = 1.2
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def _monitor(self):
        logger.info(
            f"💡 Resource Manager Mode IDLE v3.0 (Dinamis) aktif. Anggaran RAM: {self.ram_budget_gb:.2f} GB dari total {self.total_system_ram:.2f} GB."
        )
        while not self.stop_event.is_set():
            try:

                process = psutil.Process(os.getpid())
                current_script_usage_gb = process.memory_info().rss / (1024**3)



                available_budget_for_paths = (
                    self.ram_budget_gb - current_script_usage_gb
                )

                if available_budget_for_paths > 0:

                    affordable_paths = int(
                        available_budget_for_paths / self.ram_per_path_gb
                    )
                else:
                    affordable_paths = 0


                optimal_paths = max(
                    0, min(self.classifier.num_paths, affordable_paths))

                if optimal_paths != self.classifier.active_paths:
                    logger.warning(
                        f"[Resource Manager] Penyesuaian dinamis! Anggaran tersisa {available_budget_for_paths:.2f} GB. Menyesuaikan jalur Electra ke: {optimal_paths}"
                    )
                    self.classifier.activate_paths(optimal_paths)

                time.sleep(30)
            except Exception as e:
                logger.error(
                    f"[Resource Manager] Error pada thread monitor: {e}")
                time.sleep(60)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

class StagePrepManager:
    def __init__(self):
        import threading
        self._threads = {}
        self._ready = {}
        self.cache = {}

    def trigger(self, stage: str, builder):
        import threading
        if stage in self._threads:
            return
        evt = self._ready[stage] = threading.Event()

        def _run():
            try:

                try:
                    import torch
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass
                self.cache[stage] = builder()
            finally:
                evt.set()

        t = threading.Thread(target=_run, daemon=True)
        self._threads[stage] = t
        t.start()

    def wait(self, stage: str, timeout: float = None):
        evt = self._ready.get(stage)
        if not evt:
            return None
        evt.wait(timeout)
        return self.cache.get(stage)

STAGE_PREP = StagePrepManager()

def build_pretrain_assets(hparams: dict):

    from pathlib import Path, PurePosixPath
    try:
        pid = (hparams or {}).get("project_id")
        _ = get_path(pid, "checkpoint_dir")
        return {"project_id": pid}
    except Exception:
        return {}

def build_finetune_assets(ctx: dict):

    try:
        pid = (ctx or {}).get("project_id")
        last_ckpt = None
        if LAST_PROJECT_INFO_TXT.exists():
            try:
                _, last_ckpt_str = LAST_PROJECT_INFO_TXT.read_text().strip().split(",")
                from pathlib import Path, PurePosixPath
                last_ckpt = Path(last_ckpt_str)
            except Exception:
                pass
        return {"project_id": pid, "last_ckpt": last_ckpt}
    except Exception:
        return {}

def build_predict_assets(hparams: dict):

    try:
        pid = (hparams or {}).get("project_id")
        plot_path = get_path(pid, "prediction_plot")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        return {"project_id": pid, "plot_path": plot_path}
    except Exception:
        return {}


class AlphaDataModule(pl.LightningDataModule):
    """
    Modul data yang canggih, mengelola pemuatan data, anotasi peristiwa otomatis,
    rekayasa fitur, dan persiapan batch untuk pelatihan dan evaluasi.
    """

    def __init__(
        self,
        hparams: dict,
        auditor: "CriticalAuditor",
        api_pool: "DistributedAIPool",
        together_keys: dict,
        gemini_api_config: dict,
        web_searcher: "WebSearchManager",
        brain: "Brain",
        engine: "AsyncCuriosityEngine",
        blacklist_path: Optional[Path] = None,
    ):
        super().__init__()
        self.engine = engine
        self.astro_engine = AstroFeatureEngine()
        
        self.loss_history_epoch = {}
        self.dynamic_loss_weights = {
            "loss_contrastive": 1.0,
            "loss_volatility": 0.01,
            "loss_jigsaw": 1.0,
            "loss_spike": 1.0,
            "kld_loss": 0.001
        }
        self.dynamic_adjust_rate = 0.1


        hparams_to_save = hparams.copy()





        hparams_to_save.pop("n_features_input", None)
        hparams_to_save.pop("n_primary_features", None)
        hparams_to_save.pop("n_leftover_features", None)


        hparams_to_save.pop("web_searcher", None)
        hparams_to_save.pop("x_sentiment_manager", None)
        hparams_to_save.pop("gemini_api_config", None)
        hparams_to_save.pop("definitive_feature_list", None)
        hparams_to_save.pop("causal_auditor", None)


        self.save_hyperparameters(hparams_to_save)


        self.auditor = auditor
        self.api_pool = api_pool
        self.together_keys = together_keys
        self.gemini_api_config = gemini_api_config
        self.web_searcher = web_searcher
        self.brain = brain
        self.blacklist_path = blacklist_path


        self.hparams_internal = hparams.copy()

        self.tickers = self.hparams_internal.get("selected_tickers", [])
        self.num_workers_override = self.hparams_internal.get("num_workers", 0)

        self.scaler_X = StandardScaler()
        self.augmentor = Compose(
            [TSMaskOut(p=0.5, magnitude=0.17),
             TSGaussianNoise(p=0.5, magnitude=0.01)]
        )

        self.feature_names = []
        self.event_names = []
        self.n_features_input = None
        self.n_targets = None
        self.latest_events = []
        self.df_processed = None
        self.last_known_prices = None
        self.graph_edge_index = None
        self.target_cols = []
        self.uncertainty_factor_names = [
            "volatility_shock",
            "event_shock",
            "price_gap_shock",
            "baseline",
        ]
        self.n_uncertainty_factors = len(self.uncertainty_factor_names)
        self._setup_done = False

    def setup(self, stage: Optional[str] = None):
        if hasattr(self, "_setup_done") and self._setup_done:
            return

        logger.info(
            "🚀 Memulai persiapan data (AlphaDataModule setup v4.0 dengan Memori DKG)..."
        )
        mode = self.hparams.get("mode", "fine-tune")
        STAGE_PREP.trigger("pretrain_prep", lambda: build_pretrain_assets(self.hparams))
        df_full = pd.read_parquet(self.hparams["data_path"])
        df_full["Date"] = pd.to_datetime(df_full.get("Date", df_full.index))
        df_full = df_full.set_index("Date").sort_index()



        numeric_cols_for_filtering = df_full.select_dtypes(
            include=np.number
        ).columns.tolist()
        df_full = filter_outliers_with_mahalanobis(
            df_full, numeric_cols_for_filtering, threshold_percentile=99.0
        )


        master_event_list = []

        if self.api_pool and self.web_searcher:
            logger.info(
                "===== MEMULAI RISET PERISTIWA (PENDEKATAN TOP-DOWN) =====")
            annotation_engine = CrossValidationAnnotationEngine(
                self.together_keys, web_search_manager=self.web_searcher
            )
            end_date_dt = df_full.index.max()
            start_date_dt = end_date_dt - timedelta(days=365)
            start_date = start_date_dt.strftime("%Y-%m-%d")
            end_date = end_date_dt.strftime("%Y-%m-%d")
            logger.info(
                f"Membatasi pencarian berita dari {start_date} hingga {end_date}."
            )
            top_down_events = annotation_engine.generate_master_timeline(
                self.tickers, start_date, end_date
            )
            if top_down_events:
                master_event_list.extend(top_down_events)

        if self.api_pool and self.web_searcher:
            logger.info(
                "===== MEMULAI RISET INVESTIGATIF (PENDEKATAN BOTTOM-UP) =====")
            end_date_dt = df_full.index.max()
            start_date_dt = end_date_dt - timedelta(days=365)
            df_recent = df_full[df_full.index >= start_date_dt]
            logger.info(
                f"Membatasi deteksi guncangan harga pada data dari {start_date_dt.strftime('%Y-%m-%d')} dan setelahnya."
            )
            detected_shocks = detect_price_shocks(
                df_recent,
                self.tickers,
                single_day_threshold=15.0,
                consecutive_days=3,
                multi_day_threshold=20.0,
            )
            if detected_shocks:
                MAX_SHOCKS_PER_BATCH = 50
                logger.info(
                    f"Total {len(detected_shocks)} lonjakan harga terdeteksi. Memproses dalam batch berukuran maks {MAX_SHOCKS_PER_BATCH}.")


                for i in range(0, len(detected_shocks), MAX_SHOCKS_PER_BATCH):
                    shock_chunk = detected_shocks[i:i + MAX_SHOCKS_PER_BATCH]
                    logger.info(
                        f"  -> Memproses sub-batch #{i//MAX_SHOCKS_PER_BATCH + 1} ({len(shock_chunk)} lonjakan)...")


                    batch_event_data = investigate_shock_events_in_batch(
                        shocks=shock_chunk,
                        web_searcher=self.web_searcher,
                        engine=self.engine
                    )


                    if batch_event_data:
                        for event_data in batch_event_data:

                            if isinstance(event_data, dict):
                                master_event_list.append(Event(**event_data))
                            else:
                                logger.warning(
                                    f"Menerima item data event yang tidak valid: {event_data}")

        if master_event_list:
            validated_events = []
            for event in master_event_list:
                try:

                    fixed_date = event.date.replace("??", "01")
                    pd.to_datetime(fixed_date)
                    

                    event.date = fixed_date 
                    validated_events.append(event)
                except Exception as e:
                    logger.warning(f"Event '{event.event_name}' dengan tanggal tidak valid ('{event.date}') dibuang.")
            
            master_event_list = validated_events
            logger.info(f"Total peristiwa yang valid setelah pembersihan: {len(master_event_list)}")
            event_df = pd.DataFrame([e.model_dump()
                                    for e in master_event_list])
            event_df.drop_duplicates(
                subset=["date", "event_name"], keep="last", inplace=True
            )
            master_event_list = [Event(**row)
                                 for i, row in event_df.iterrows()]
            self.event_names = [e.event_name for e in master_event_list]
            logger.info(
                f"Total peristiwa unik yang terkumpul dari kedua metode: {len(master_event_list)}"
            )

            if mode != "predict":
                df_full["event_impact_score"] = 0.0
                df_full["is_event_day"] = 0
                for event in master_event_list:
                    try:
                        event_date = pd.to_datetime(event.date)
                        if event_date in df_full.index:
                            df_full.loc[event_date, "event_impact_score"] = (
                                event.impact_score
                            )
                            df_full.loc[event_date, "is_event_day"] = 1
                    except Exception as e:
                        logger.warning(
                            f"Gagal memproses event '{event.event_name}': {e}"
                        )
                logger.info(
                    "✅ DataFrame historis telah diperkaya dengan peristiwa hasil riset gabungan."
                )

        df_imputed = df_full.copy()
        logger.info("🔬 Memulai proses imputasi data yang tangguh...")
        numeric_cols = df_imputed.select_dtypes(include=np.number).columns
        non_numeric_cols = df_imputed.select_dtypes(exclude=np.number).columns
        all_nan_numeric_cols = [
            col for col in numeric_cols if df_imputed[col].isnull().all()
        ]
        imputable_numeric_cols = [
            col for col in numeric_cols if col not in all_nan_numeric_cols
        ]
        if all_nan_numeric_cols:
            logger.warning(
                f"Kolom numerik yang sepenuhnya NaN ditemukan dan akan diisi dengan 0: {all_nan_numeric_cols}"
            )
        imputer = KNNImputer(n_neighbors=5)
        if imputable_numeric_cols:
            imputed_numeric_data = imputer.fit_transform(
                df_imputed[imputable_numeric_cols]
            )
            df_numeric_imputed = pd.DataFrame(
                imputed_numeric_data,
                columns=imputable_numeric_cols,
                index=df_imputed.index,
            )
            df_imputed = pd.concat(
                [
                    df_numeric_imputed,
                    df_imputed[all_nan_numeric_cols].fillna(0),
                    df_imputed[non_numeric_cols],
                ],
                axis=1,
            )
        df_imputed = df_imputed[df_full.columns.tolist()]
        logger.info("✅ Proses imputasi data tangguh selesai.")

        for t in self.tickers:
            close_col = f"{t}_Close"
            if close_col in df_imputed.columns:
                df_imputed[f"{t}_log_return"] = np.log(
                    df_imputed[close_col] / df_imputed[close_col].shift(1)
                )
        self.target_cols = [f"{t}_log_return" for t in self.tickers]

        df_with_all_features, self.pure_hht_features = generate_all_features(
            df_imputed,
            self.tickers,
            master_event_list,
            x_sentiment_manager=self.hparams.get("x_sentiment_manager"),
        )

        if self.hparams.get("use_chaos_features") and mode != "pre-train":
            chaos_features_df = generate_chaos_theory_features(
                df_with_all_features, self.tickers
            )
            df_with_all_features = pd.concat(
                [df_with_all_features, chaos_features_df], axis=1
            )

        best_astro_features = []
        if mode != "pre-train":
            astro_feature_path = get_path(
                self.hparams.get("project_id"), "best_astro_features"
            )
            if mode == "fine-tune":
                all_astro_features = [
                    col
                    for col in df_with_all_features.columns
                    if col.startswith("astro_")
                ]
                if all_astro_features:
                    y_for_discovery = (
                        df_with_all_features[
                            [
                                col
                                for col in self.target_cols
                                if col in df_with_all_features.columns
                            ]
                        ]
                        .mean(axis=1)
                        .rename("target_return")
                    )
                    best_astro_features = self.astro_engine.discover_hypotheses(
                        df_with_all_features[all_astro_features],
                        y_for_discovery,
                        self.api_pool,
                    )
                    with open(astro_feature_path, "w") as f:
                        json.dump(best_astro_features, f)
                    logger.info(
                        f"Daftar fitur astro terbaik disimpan ke: {astro_feature_path}"
                    )
            elif mode == "predict":
                logger.info(
                    f"Mode prediksi: Mencoba memuat daftar fitur astro dari {astro_feature_path}"
                )
                try:
                    with open(astro_feature_path, "r") as f:
                        best_astro_features = json.load(f)
                    logger.info(
                        f"✅ Berhasil memuat {len(best_astro_features)} fitur astro yang telah dipilih."
                    )
                except FileNotFoundError:
                    logger.warning(
                        "File fitur astro tidak ditemukan. Prediksi akan berjalan tanpa fitur astro."
                    )
                    best_astro_features = []

        non_astro_features = [
            col for col in df_with_all_features.columns if not col.startswith("astro_")
        ]
        final_feature_columns = non_astro_features + best_astro_features

        df_proc = df_with_all_features[final_feature_columns].copy()

        dkg_embedding_path = self.hparams.get("dkg_embedding_path")
        if (
            dkg_embedding_path
            and Path(dkg_embedding_path).exists()
            and mode == "fine-tune"
        ):
            logger.info(
                f"🧠 Memuat Memori Jangka Panjang (DKG Embeddings) dari: {dkg_embedding_path}"
            )
            try:
                dkg_embeddings = torch.load(dkg_embedding_path)
                dkg_df = pd.DataFrame.from_dict(dkg_embeddings, orient="index")
                dkg_df.columns = [
                    f"dkg_emb_{i}" for i in range(dkg_df.shape[1])]

                for feature_name, emb_vector in dkg_embeddings.items():
                    if feature_name in df_proc.columns:
                        for i, val in enumerate(emb_vector):
                            df_proc[f"{feature_name}_dkg_emb_{i}"] = val.item()

                logger.info(
                    f"✅ Memori DKG berhasil digabungkan sebagai fitur baru.")
            except Exception as e:
                logger.error(
                    f"Gagal memuat atau menggabungkan embedding DKG: {e}")

        df_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_proc = df_proc.fillna(method="ffill").fillna(
            method="bfill").fillna(0)

        definitive_features = self.hparams.get("definitive_feature_list")
        if definitive_features:
            logger.info(
                f"✅ Menggunakan daftar fitur definitif yang sudah dihitung sebelumnya ({len(definitive_features)} fitur)."
            )
            self.feature_names = definitive_features
        else:
            logger.warning(
                "Daftar fitur definitif tidak ditemukan. Menjalankan seleksi fitur ulang..."
            )
            y_sel = (
                df_proc[[col for col in self.target_cols if col in df_proc.columns]]
                .mean(axis=1)
                .bfill()
                .ffill()
            )
            aset_cols_to_exclude = [
                c for c in df_proc.columns if c.startswith("aset_")]
            potential_features = [
                c
                for c in df_proc.columns
                if c not in self.target_cols
                and not c.startswith(tuple([f"{t}_" for t in self.tickers]))
                and c not in aset_cols_to_exclude
            ]
            all_feats = sorted(
                list(
                    set(potential_features)
                    - {
                        f"{t}_{c}"
                        for t in self.tickers
                        for c in ("Open", "High", "Low", "Close", "Volume")
                    }
                )
            )
            if not all_feats:
                raise ValueError(
                    "Tidak ada fitur potensial yang tersisa untuk seleksi."
                )

            lgbm_selected_features, _ = select_features(
                X=df_proc[all_feats],
                y=y_sel,
                top_k=self.hparams.get("top_k_features", 100),
            )
            self.feature_names = lgbm_selected_features

        self.n_features_input = len(self.feature_names)
        self.n_targets = len(self.tickers)

        df_proc.columns = _sanitize_and_dedupe_columns(df_proc.columns)

        logger.info("Membuat target untuk atribusi ketidakpastian...")
        vol_col = f"{self.tickers[0]}_Volatility_20"
        vol_score = (
            df_proc[vol_col].rank(pct=True)
            if vol_col in df_proc.columns
            else pd.Series(0, index=df_proc.index)
        )
        event_score = (
            df_proc["event_influence_score"].rank(pct=True)
            if "event_influence_score" in df_proc
            else pd.Series(0, index=df_proc.index)
        )
        gap_score = pd.Series(0, index=df_proc.index)
        high_col = f"{self.tickers[0]}_High"
        low_col = f"{self.tickers[0]}_Low"
        if high_col in df_proc.columns and low_col in df_proc.columns:
            price_gap = df_proc[low_col] - df_proc[high_col].shift(1)
            gap_score = price_gap.abs().rank(pct=True)
        uncertainty_scores = pd.DataFrame(
            {
                "volatility_shock": vol_score,
                "event_shock": event_score,
                "price_gap_shock": gap_score,
                "baseline": 0.1,
            }
        ).fillna(0)
        dominant_factor_indices = (
            uncertainty_scores.idxmax(axis=1).astype("category").cat.codes
        )
        index_tensor = torch.from_numpy(
            dominant_factor_indices.values).to(torch.long)
        uncertainty_target = F.one_hot(
            index_tensor, num_classes=self.n_uncertainty_factors
        )
        for i, factor_name in enumerate(self.uncertainty_factor_names):
            df_proc[f"unc_factor_{factor_name}"] = uncertainty_target[:, i].numpy(
            )
        self.uncertainty_cols = [
            f"unc_factor_{name}" for name in self.uncertainty_factor_names
        ]
        logger.info("✅ Target atribusi ketidakpastian selesai dibuat.")

        event_scores = (
            df_proc["event_influence_score"].fillna(0)
            if "event_influence_score" in df_proc
            else pd.Series(0, index=df_proc.index)
        )
        event_scores_normalized = (
            (event_scores - event_scores.min())
            / (event_scores.max() - event_scores.min())
            if event_scores.max() > event_scores.min()
            else event_scores
        )
        event_scores_normalized = event_scores_normalized.fillna(0)
        snn_num_steps = self.hparams.get("snn_num_steps", 25)
        self.spike_trains = encode_events_to_spikes(
            event_scores_normalized, num_steps=snn_num_steps
        )

        self.df_processed = df_proc

        corr_matrix = self.df_processed[self.feature_names].corr().values
        adj = np.abs(corr_matrix) > 0.4
        np.fill_diagonal(adj, 0)
        _affmap = build_affinity_map(total_cores=None)
        _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)
        self.graph_edge_index = torch.tensor(
            np.array(np.where(adj)), dtype=torch.long)

        self.last_known_prices = df_imputed[[f"{t}_Close" for t in self.tickers]].iloc[
            -1
        ]
        self.last_known_prices.index = [
            c.replace("_Close", "") for c in self.last_known_prices.index
        ]

        X = self.df_processed[self.feature_names].values
        y = self.df_processed[self.target_cols].values

        split = int(len(X) * 0.8)


        self.X_raw_train, self.X_raw_val = X[:split], X[split:]


        self.scaler_X.fit(X[:split])
        X_scaled = self.scaler_X.transform(X)

        if stage == "fit" or stage is None:
            joblib.dump(
                self.scaler_X, get_path(
                    self.hparams.get("project_id"), "scaler")
            )

        self.X_train, self.X_val = X_scaled[:split], X_scaled[split:]
        self.y_train, self.y_val = y[:split], y[split:]

        self._setup_done = True
        logger.info("✅ Persiapan data selesai dengan semua modul terintegrasi.")

    def train_dataloader(self):
        raw_signals = self.df_processed["event_influence_score"].values
        normalized_signals = np.exp(-raw_signals / 30.0)
        anomaly_signals_train = normalized_signals[: len(self.X_train)]

        uncertainty_targets_train = self.df_processed[self.uncertainty_cols].values[
            : len(self.X_train)
        ]
        spike_trains_train = self.spike_trains[: len(self.X_train)]
        _affmap = build_affinity_map(total_cores=None)
        _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)
        dataset = AlphaDataset(
            X_raw=self.X_raw_train,
            X=self.X_train,
            y=self.y_train,
            anomaly_signals=anomaly_signals_train,
            edge_index=self.graph_edge_index,
            uncertainty_targets=uncertainty_targets_train,
            spike_trains=spike_trains_train,
            n_primary_features=self.hparams_internal.get(
                "n_primary_features", 64),
            mode=self.hparams.get("mode"),
            augmentor=self.augmentor,
            window=self.hparams.get("window", 60),
            horizon=self.hparams.get("horizon", 7),
        )
        return TorchDataLoader(
            dataset,
            batch_size=self.hparams.get("batch_size", 64),
            shuffle=True,
            collate_fn=TSCollate(self.hparams.get("mode")),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=_worker_init,
            prefetch_factor=4,
        )

    def val_dataloader(self):

        validation_mode = (
            "pre-train"
            if self.hparams_internal.get("mode") == "pre-train"
            else "fine-tune"
        )
        logger.info(
            f"Mengonfigurasi val_dataloader dengan mode: '{validation_mode}'")


        split_point = len(self.X_train)
        anomaly_signals_val = self.df_processed["event_influence_score"].values[
            split_point: split_point + len(self.X_val)
        ]
        uncertainty_targets_val = self.df_processed[self.uncertainty_cols].values[
            split_point: split_point + len(self.X_val)
        ]
        spike_trains_val = self.spike_trains[
            split_point: split_point + len(self.X_val)
        ]



        batch_size = self.hparams_internal.get("batch_size", 64)

        dataset = AlphaDataset(
            X_raw=self.X_raw_val,
            X=self.X_val,
            y=self.y_val,
            anomaly_signals=anomaly_signals_val,
            edge_index=self.graph_edge_index,
            uncertainty_targets=uncertainty_targets_val,
            spike_trains=spike_trains_val,
            n_primary_features=self.hparams_internal.get("n_primary_features"),
            mode=validation_mode,
            augmentor=None,

            window=self.hparams_internal.get("window", 60),
            horizon=self.hparams_internal.get("horizon", 7),
        )

        _affmap = build_affinity_map(total_cores=None)
        _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)


        return TorchDataLoader(
            dataset,
            batch_size=self.hparams_internal.get("batch_size", 64),
            shuffle=False,
            collate_fn=TSCollate(validation_mode),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=_worker_init,
            prefetch_factor=4,
        )




class AlphaDataset(Dataset):
    def __init__(
        self,
        X_raw,
        X,
        y,
        anomaly_signals,
        edge_index,
        uncertainty_targets,
        spike_trains,
        n_primary_features,
        mode="fine-tune",
        augmentor=None,
        window=60,
        horizon=7,
    ):
        self.X_raw = X_raw
        self.X, self.y, self.anomaly_signals = X, y, anomaly_signals
        self.edge_index = edge_index
        self.uncertainty_targets = uncertainty_targets
        self.spike_trains = spike_trains
        self.mode, self.augmentor, self.window, self.horizon = (
            mode,
            augmentor,
            window,
            horizon,
        )
        self.snn_num_steps = spike_trains.shape[1] if spike_trains is not None else 0
        self.jigsaw_segments = 4




        self.support_len = self.window // 2
        self.query_len = self.window

        if self.window % self.jigsaw_segments != 0:
            logger.warning(
                f"Window size ({self.window}) not divisible by jigsaw_segments ({self.jigsaw_segments}). This may lead to uneven segment sizes."
            )
        self.permutations = list(
            itertools.permutations(range(self.jigsaw_segments)))

    def __len__(self):
        if self.mode == "pre-train":
            return max(
                0, len(self.X) - self.support_len -
                self.query_len - self.horizon + 1
            )
        else:
            return max(0, len(self.X) - self.window - self.horizon + 1)

    def __getitem__(self, idx):
        if self.mode == "pre-train":
            support_start = idx
            support_end = support_start + self.support_len
            x_support_window = self.X[support_start:support_end]

            query_start = support_end
            query_end = query_start + self.query_len
            x_query_window = self.X[query_start:query_end]

            if (
                x_support_window.shape[0] != self.support_len
                or x_query_window.shape[0] != self.query_len
            ):
                return None

            x_query_tensor = torch.from_numpy(
                x_query_window.astype(np.float32))
            aug1 = self.augmentor(
                x_query_tensor) if self.augmentor else x_query_tensor
            aug2 = self.augmentor(
                x_query_tensor) if self.augmentor else x_query_tensor

            proxy_target_start = query_end
            proxy_target_end = proxy_target_start + self.horizon
            proxy_target_slice = self.y[proxy_target_start:proxy_target_end, 0]
            proxy_volatility = (
                torch.std(torch.from_numpy(proxy_target_slice))
                if len(proxy_target_slice) > 1
                else torch.tensor(0.0)
            )

            segment_len = self.query_len // self.jigsaw_segments
            segments = torch.stack(
                [
                    x_query_tensor[i * segment_len: (i + 1) * segment_len]
                    for i in range(self.jigsaw_segments)
                ]
            )
            perm_idx = random.randint(0, len(self.permutations) - 1)
            permutation = self.permutations[perm_idx]
            shuffled_segments = segments[list(permutation)]
            jigsaw_puzzle_tensor = torch.cat(list(shuffled_segments), dim=0)
            jigsaw_label = torch.tensor(perm_idx, dtype=torch.long)

            future_event_scores = self.anomaly_signals[
                proxy_target_start:proxy_target_end
            ]
            significant_events = np.where(future_event_scores > 0.5)[0]
            time_to_first_spike = (
                significant_events[0] / self.horizon
                if len(significant_events) > 0
                else 1.0
            )
            spike_timing_label = torch.tensor(
                time_to_first_spike, dtype=torch.float32)

            return {
                "support_x": torch.from_numpy(x_support_window.astype(np.float32)),
                "query_x_aug1": aug1,
                "query_x_aug2": aug2,
                "query_vol_target": proxy_volatility.float(),
                "query_jigsaw_puzzle": jigsaw_puzzle_tensor,
                "query_jigsaw_label": jigsaw_label,
                "query_spike_target": spike_timing_label,
            }


        x_raw_window = self.X_raw[idx: idx + self.window]
        x_raw_tensor = torch.from_numpy(x_raw_window.astype(np.float32))

        x_window = self.X[idx: idx + self.window]
        y_historical_window = self.y[idx: idx + self.window]

        x_combined_tensor = torch.from_numpy(x_window.astype(np.float32))
        y_historical_tensor = torch.from_numpy(
            y_historical_window.astype(np.float32))

        anomaly_signal_val = self.anomaly_signals[idx + self.window - 1]
        anomaly_tensor = torch.tensor(anomaly_signal_val, dtype=torch.float32)

        y_target_slice = self.y[idx +
                                self.window: idx + self.window + self.horizon]
        y_tensor = torch.from_numpy(y_target_slice.astype(np.float32))

        unc_target_tensor = torch.from_numpy(
            self.uncertainty_targets[idx + self.window - 1].astype(np.float32)
        )

        spike_train_slice = self.spike_trains[idx + self.window - 1]
        spike_tensor = torch.from_numpy(spike_train_slice.astype(np.float32)).unsqueeze(
            1
        )

        return (
            x_raw_tensor,
            x_combined_tensor,
            y_tensor,
            y_historical_tensor,
            anomaly_tensor,
            self.edge_index,
            unc_target_tensor,
            spike_tensor,
        )


class TSCollate:
    """
    Fungsi collate yang cerdas untuk menangani output yang berbeda dari
    mode 'pre-train' (dictionary) dan 'fine-tune' (tuple).
    """

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, batch):

        batch = [item for item in batch if item is not None]
        

        if not batch:
            print("\n\nDEBUG: TSCollate mendeteksi batch kosong dan akan mengembalikan None.\n\n")
            return None


        if self.mode == "pre-train":
            keys = batch[0].keys()
            ref_shapes = {key: batch[0][key].shape for key in keys}
            

            for item in batch:
                for key in keys:
                    if item[key].shape != ref_shapes[key]:
                        logger.warning(
                            f"Melewatkan batch karena ketidakcocokan ukuran pada kunci '{key}'. "
                            f"Expected {ref_shapes[key]}, got {item[key].shape}."
                        )

                        return None 


            collated_batch = {key: torch.stack([d[key] for d in batch]) for key in keys}
            return collated_batch


        else: 
            (
                x_raws, x_combineds, ys, y_hists, anomaly_signals,
                edge_index, unc_targets, x_spikes,
            ) = zip(*batch)

            return (
                torch.stack(x_raws),
                torch.stack(x_combineds),
                torch.stack(ys),
                torch.stack(y_hists),
                torch.stack(anomaly_signals),
                edge_index[0],
                torch.stack(unc_targets),
                torch.stack(x_spikes),
            )


class HybridSSP_LitModule(pl.LightningModule):
    """
    HybridSSP_LitModule v5.0 - Arsitektur Terpadu untuk Pre-training & Fine-tuning.
    - Menggabungkan arsitektur pre-training (v4.1) dengan model SNN hibrida.
    - Memiliki dua jalur forward pass: `forward_pretrain` dan `forward` utama.
    - `forward_pretrain`: Untuk tugas self-supervised learning (SSL).
    - `forward`: Untuk tugas downstream (fine-tuning) dengan arsitektur multi-jalur
      (Wave, Particle, Symbolic) yang difusikan oleh UnifiedFusionHub.
    - Semua fitur dari v4.1 (Shin-Godzilla, Hellhound, dll.) dipertahankan untuk pre-training.
    """

    def __init__(self, hparams):
        super().__init__()

        hparams.setdefault("dropout", 0.2)
        hparams.setdefault("n_layers", 3)
        hparams.setdefault("d_model", 128)
        hparams.setdefault("n_heads", 8)
        hparams.setdefault("beta_vib", 1e-6)

        hparams.setdefault("snn_hidden_size", 64)
        hparams.setdefault("num_rules", 5)
        hparams.setdefault("horizon", 24)
        hparams.setdefault("n_targets", 1)
        
        hparams.setdefault("snn_input_size", 1)
        hparams.setdefault("snn_num_layers", 2)
        hparams.setdefault("snn_dropout", 0.2)
        # Patch: reduce the number of spectral modes used by the FNO layer to align
        # with pretrained checkpoints and lighten CPU usage.  Pretrained models
        # expect a weight tensor of shape [d_model, d_model, 12], so we set
        # the default number of modes to 12 here.  Internal computation in
        # FNO1dBlock.forward will still use a reduced subset of these modes
        # (see effective_modes in FNO1dBlock.forward), but the weight shape
        # matches the checkpoint to avoid size mismatches.
        hparams.setdefault("fno_modes", 12)

        # ---- Checkpoint Compatibility Patch ----
        # Many pretrained checkpoints for this model were trained with d_model=128.  To
        # avoid weight size mismatches when loading those checkpoints, we enforce
        # that `d_model` is at least 128.  If a smaller value is provided in
        # hparams, it will be promoted to 128 here.  This ensures that the
        # dimensions of QuantumThalamicCore and other modules align with the
        # checkpoint weights, preventing errors such as size mismatches during
        # `load_state_dict`.  Users who wish to experiment with a smaller
        # dimensionality should be aware that pretrained checkpoints may not
        # load cleanly.
        if hparams.get("d_model", 128) < 128:
            logger.info(
                f"[Patch] Menyesuaikan d_model dari {hparams['d_model']} ke 128 untuk kompatibilitas checkpoint."
            )
            hparams["d_model"] = 128

        self.dynamic_loss_weights = {
            "loss_contrastive": 1.0,
            "loss_volatility": 1.0,
            "loss_jigsaw": 1.0,
            "loss_spike": 1.0,
            "kld_loss": 1.0
        }
        self.save_hyperparameters(hparams)
        self.augmentor = self.hparams.get("augmentor", None)
        self.register_buffer("temperature", torch.tensor(0.1))
        self.recovery_mode_cooldown = 0
        self.gradient_collapse_detected_in_epoch = False
        self.denoising_model = None
        self.automatic_optimization = False 
        self.current_capacity_frac = 1.0






        self.encoder = TST_module(
            c_in=self.hparams.n_features_input,
            c_out=self.hparams.d_model,
            seq_len=self.hparams.window,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )
        self.menshin_core = Menshin_Skytree_Core(
            d_model=self.hparams.d_model, n_layers=2, n_heads=self.hparams.n_heads, dropout=self.hparams.dropout
        )
        self.norm_before_vae = nn.LayerNorm(self.hparams.d_model)
        self.variational_encoder = VariationalEncoder(
            input_dim=self.hparams.d_model, latent_dim=self.hparams.d_model
        )
        self.mu = None
        self.logvar = None


        self.projector = nn.Sequential(nn.Linear(self.hparams.d_model, 128), nn.SiLU())
        # Prepend a LayerNorm to stabilise the scale feeding into the volatility head.
        self.volatility_head = nn.Sequential(
            nn.LayerNorm(self.hparams.d_model),
            nn.Linear(self.hparams.d_model, 1)
        )
        num_permutations = math.factorial(4)
        # Store the number of jigsaw permutations so we can normalise the
        # jigsaw loss later.  Without this normalisation the raw jigsaw loss
        # can appear disproportionately large in logs even if its weighted
        # contribution is small.
        self.jigsaw_num_classes = num_permutations
        self.jigsaw_head = nn.Sequential(nn.Linear(self.hparams.d_model, num_permutations))
        self.spike_timing_head = nn.Sequential(nn.Linear(self.hparams.d_model, 1), nn.Sigmoid())
        self.raw_log_vars = nn.Parameter(torch.zeros(4))
        self.label_smoothing_active = False



        self.qtc = QuantumThalamicCore(input_dim=self.hparams.n_features_input, codebook_dim=self.hparams.d_model)
        self.dpa_stif_layers = nn.ModuleList([DPA_STIFormer_Layer(d_model=self.hparams.d_model, n_heads=self.hparams.n_heads) for _ in range(2)])
        self.fno_layer = FNO1dBlock(d_model=self.hparams.d_model, modes=12)
        self.norm_fno = nn.LayerNorm(self.hparams.d_model)
        self.snn_processor = RecurrentSpikingProcessor(
            input_size=self.hparams.snn_input_size, 
            hidden_size=self.hparams.snn_hidden_size,
            num_layers=self.hparams.snn_num_layers,
            dropout_p=self.hparams.snn_dropout
        )
        self.stdp_proxy_loss = 0.0
        

        self.rule_bank = nn.ModuleList()
        self.symbolic_fusion_head = None


        self.snn_output_projection = nn.Linear(self.hparams.snn_hidden_size, self.hparams.d_model)
        self.fusion_hub = UnifiedFusionHub(d_model=self.hparams.d_model, n_heads=self.hparams.n_heads)




        self.menshin_fusion_norm = nn.LayerNorm(self.hparams.d_model)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.shield_pre_vae = nn.LayerNorm(self.hparams.d_model)
        self.kevlar_layer = nn.Sequential(nn.Linear(self.hparams.d_model, self.hparams.d_model), nn.SiLU())
        self.shield_pre_heads = nn.LayerNorm(self.hparams.d_model + self.hparams.n_targets)


        final_head_input_dim = self.hparams.d_model + self.hparams.n_targets
        # Apply LayerNorm before the regression head to stabilise input scale.
        self.regression_head = nn.Sequential(
            nn.LayerNorm(final_head_input_dim),
            nn.Linear(final_head_input_dim, self.hparams.horizon * self.hparams.n_targets)
        )
        self.anomaly_head = nn.Linear(final_head_input_dim, 1)
        self.uncertainty_attribution_head = nn.Linear(final_head_input_dim, self.hparams.d_model)
        




        self.loss_fn = ReturnLoss(
            da_weight=self.hparams.get("da_weight", 0.3), 
            horizon=self.hparams.horizon
        )

        self.val_mse = MeanSquaredError()
        self.val_rmse = lambda p, t: torch.sqrt(self.val_mse(p, t))
        self.val_da = lambda p, t: torch.mean(
            (torch.sign(p) == torch.sign(t)).float()
        )

        # --- [Router Integration] ---
        # Initialize the expert router after all model components have been created.  The
        # router is responsible for deciding which expert modules to activate on a
        # per‑epoch basis, dynamically adjusting loss weights (λ), micro LR (μ) and
        # gate (γ) according to data characteristics.  It also maintains running
        # statistics for loss components and applies adaptation rules at defined
        # intervals.  See the `ExpertRouter` class for details.
        try:
            self.router = ExpertRouter(self)
        except Exception as _e:
            logger.warning(f"[Router] Failed to initialize ExpertRouter: {_e}")
            self.router = None
        # Bookkeeping to ensure router is invoked once per epoch
        self._router_initialized_epoch = -1
        # Running statistics for losses used by router adaptation; keys will be set
        # dynamically during training.  Each entry holds a dict with running sum
        # and count used to compute mean.
        self._loss_running_stats: Dict[str, Dict[str, float]] = {}


    def set_capacity_frac(self, frac: float):

        frac = float(max(0.10, min(1.0, frac)))
        self.current_capacity_frac = frac

    def _mask_channels(self, x: torch.Tensor, frac: float) -> torch.Tensor:

        if (not self.training) or frac >= 0.999:
            return x
        C = x.shape[-1]
        k = max(1, int(C * frac))

        idx = torch.randperm(C, device=x.device)[:k]
        mask = torch.zeros(C, device=x.device, dtype=x.dtype)
        mask[idx] = 1
        return x * mask
    
    def configure_callbacks(self):

        acs = AdaptiveCapacityScheduler(
            low_frac=0.25,
            high_frac=1.0,
            schedule="cosine",
            cooldown_epochs=0
        )
        agc = AdaptiveGradientClipping(history_size=100, percentile=90.0)
        return [acs, agc]







    def forward_pretrain(self, x):
        """
        [MODIFIKASI] Ini adalah fungsi `forward` asli dari kode pertama.
        Digunakan KHUSUS untuk pre-training agar tidak mengganggu logika SSL.
        """

        encoded_embedding = self.encoder(x.transpose(1, 2))

        if encoded_embedding.ndim == 2:
            encoded_embedding = encoded_embedding.unsqueeze(1)

        stabilized_embedding = self.menshin_core(encoded_embedding)

        pooled_rep = stabilized_embedding.mean(dim=1)
        normalized_pooled_rep = self.norm_before_vae(pooled_rep)
        # Compute latent mean and logvar.  Clamp logvar to avoid extreme
        # values which can destabilise KLD loss.
        self.mu, self.logvar = self.variational_encoder(normalized_pooled_rep)
        try:
            if self.logvar is not None:
                self.logvar.data.clamp_(-6.0, 2.0)
        except Exception:
            pass
        z = self.variational_encoder.reparameterize(self.mu, self.logvar)
        return z

    def forward(self, x_raw, x_combined, y_historical, edge_index, x_spikes, **kwargs):
        """
        [BARU] Ini adalah fungsi `forward` dari kode ketiga untuk fine-tuning.
        Mengimplementasikan arsitektur multi-jalur (Wave, Particle, Symbolic).
        """

        # --- [Gating: Latent-Manifold] ---
        # If the latent-manifold expert is inactive, skip expensive QTC computation and
        # substitute a zero tensor.  Otherwise run QTC as usual.
        gating = getattr(self, '_router_gating', None)
        latent_active = gating.get('Latent-Manifold', True) if isinstance(gating, dict) else True
        if latent_active:
            thalamic_vector = self.qtc(x_combined)
        else:
            thalamic_vector = torch.zeros(x_combined.shape[0], self.hparams.window, self.hparams.d_model, device=x_combined.device)
        thalamic_vector = self._mask_channels(thalamic_vector, self.current_capacity_frac)
        pe = self._generate_standard_pe(self.hparams.window, self.hparams.d_model).to(x_combined.device)
        x_context_aware_seq = thalamic_vector.unsqueeze(1).repeat(1, self.hparams.window, 1) + pe



        # Temporal attention path
        x_time = x_context_aware_seq
        x_time = self._mask_channels(x_time, self.current_capacity_frac)
        temporal_active = gating.get('Temporal-Attn', True) if isinstance(gating, dict) else True
        if temporal_active:
            for layer in self.dpa_stif_layers:
                x_time = layer(x_time)
        # Spectral path
        spectral_active = gating.get('Spectral', True) if isinstance(gating, dict) else True
        if spectral_active:
            freq_embedding = self.norm_fno(self.fno_layer(x_context_aware_seq)).mean(dim=1)
        else:
            freq_embedding = torch.zeros(x_context_aware_seq.shape[0], self.hparams.d_model, device=x_context_aware_seq.device)
        freq_embedding = self._mask_channels(freq_embedding, self.current_capacity_frac)
        # Combine temporal and spectral embeddings
        if temporal_active:
            time_mean = x_time.mean(dim=1)
        else:
            time_mean = x_context_aware_seq.mean(dim=1)
        wave_embedding = (time_mean + freq_embedding) / 2


        # Spiking path
        spiking_active = gating.get('Spiking', True) if isinstance(gating, dict) else True
        if spiking_active:
            snn_output, _, stdp_loss = self.snn_processor(x_spikes)
            particle_embedding = self.snn_output_projection(snn_output)
            if self.training:
                self.stdp_proxy_loss = stdp_loss
        else:
            particle_embedding = torch.zeros(x_spikes.shape[0], self.hparams.d_model, device=x_spikes.device)
            if self.training:
                self.stdp_proxy_loss = 0.0
        particle_embedding = self._mask_channels(particle_embedding, self.current_capacity_frac)


        symbolic_embedding = torch.zeros_like(wave_embedding)
        if self.rule_bank and self.symbolic_fusion_head:
            rule_outputs = [rule(x_raw) for rule in self.rule_bank]
            concatenated_rules = torch.cat(rule_outputs, dim=1)
            symbolic_embedding = self.symbolic_fusion_head(concatenated_rules)


        specialist_embeddings = torch.stack([particle_embedding, symbolic_embedding], dim=1)
        final_fused_embedding = self.fusion_hub(wave_embedding, specialist_embeddings)


        stable_core_signal = self.menshin_core(x_context_aware_seq).mean(dim=1)
        final_stabilized_embedding = self.menshin_fusion_norm(final_fused_embedding + self.dropout(stable_core_signal))

        shielded_embedding_for_vae = self.shield_pre_vae(final_stabilized_embedding)
        # Compute latent mean and logvar for downstream VAE.  Clamp logvar to a
        # reasonable range to stabilise KLD.
        self.mu, self.logvar = self.variational_encoder(shielded_embedding_for_vae)
        try:
            if self.logvar is not None:
                self.logvar.data.clamp_(-6.0, 2.0)
        except Exception:
            pass
        z_latent = self.variational_encoder.reparameterize(self.mu, self.logvar)
        stabilized_embedding_final = self.kevlar_layer(z_latent)

        y_hist_last = y_historical[:, -1, :]
        final_input_to_heads_raw = torch.cat([stabilized_embedding_final, y_hist_last], dim=1)
        final_input_to_heads = self.shield_pre_heads(final_input_to_heads_raw)
        final_input_to_heads = self._mask_channels(final_input_to_heads, min(1.0, self.current_capacity_frac + 0.1))

        # Apply gating for heads.  If the Heads-Other expert is inactive, we
        # skip computing these outputs and return zeros instead.  This
        # reduces compute for downstream tasks where these predictions are
        # unnecessary.
        heads_active = gating.get('Heads-Other', True) if isinstance(gating, dict) else True
        if heads_active:
            regression_output = self.regression_head(final_input_to_heads).view(-1, self.hparams.horizon, self.hparams.n_targets)
            anomaly_output = self.anomaly_head(final_input_to_heads)
            uncertainty_attribution = self.uncertainty_attribution_head(final_input_to_heads)
        else:
            # Construct zero tensors with correct shapes
            regression_output = torch.zeros(x_raw.size(0), self.hparams.horizon, self.hparams.n_targets, device=x_raw.device)
            anomaly_output = torch.zeros(x_raw.size(0), 1, device=x_raw.device)
            uncertainty_attribution = torch.zeros(x_raw.size(0), self.hparams.d_model, device=x_raw.device)
        return (regression_output, anomaly_output.squeeze(-1), uncertainty_attribution, final_fused_embedding, None, None)
    
    def _generate_standard_pe(self, seq_len, d_model):
        """Helper untuk menghasilkan Positional Encoding standar."""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe






    def _calculate_ssl_losses(self, batch):
        """
        Fungsi helper yang telah di-upgrade dengan Protokol Hellhound & Alpha Dinamis.
        [MODIFIKASI] Sekarang memanggil `self.forward_pretrain`.
        """

        current_epoch = self.trainer.current_epoch
        max_epochs = self.trainer.max_epochs
        progress = current_epoch / max_epochs
        initial_alpha = 2.0
        final_alpha = 0.2
        current_alpha = final_alpha + 0.5 * (initial_alpha - final_alpha) * (1 + math.cos(math.pi * progress))
        self.log("dynamic_alpha", current_alpha, on_step=False, on_epoch=True, prog_bar=True)


        x_query = batch["query_x_aug1"]
        indices = torch.randperm(x_query.size(0))
        x_query_shuffled = x_query[indices]


        aug1 = x_query
        aug2 = self.augmentor(x_query) if self.augmentor else x_query

        proxy_vol_target = batch["query_vol_target"]
        jigsaw_puzzle = batch["query_jigsaw_puzzle"]
        jigsaw_label = batch["query_jigsaw_label"]
        spike_target = batch["query_spike_target"]


        z1 = self.forward_pretrain(aug1)
        z2 = self.forward_pretrain(aug2)
        z_jigsaw = self.forward_pretrain(jigsaw_puzzle)


        loss_contrastive = self._hard_negative_margin_loss(self.projector(z1), self.projector(z2), margin=0.3)
        # Volatility prediction with a softplus clamp to produce non‑negative
        # and bounded outputs.  We use Smooth L1 (Huber) loss for a more
        # robust error metric on heavy‑tailed distributions.  The beta
        # parameter controls the transition point between L2 and L1 behaviour.
        pred_vol = self.volatility_head(z1).squeeze()
        # ensure positivity and clamp to a reasonable range
        pred_vol = F.softplus(pred_vol)
        pred_vol = pred_vol.clamp(min=0.0, max=5.0)
        loss_volatility = F.smooth_l1_loss(pred_vol, proxy_vol_target, beta=0.5)


        max_smoothing = 0.1
        smoothing_ramp_epochs = 20
        smoothing_value = min(max_smoothing, (current_epoch / smoothing_ramp_epochs) * max_smoothing) if current_epoch is not None else 0.0
        self.log("dynamic_label_smoothing", smoothing_value, on_step=False, on_epoch=True)
        # Apply gating for heads-other.  If the Heads‑Other expert is inactive,
        # skip computing jigsaw and spike losses.  This reduces compute in
        # contexts where these objectives are irrelevant.
        gating = getattr(self, '_router_gating', None)
        heads_active = gating.get('Heads-Other', True) if isinstance(gating, dict) else True
        if heads_active:
            # Compute jigsaw loss.  Cross‑entropy returns the average negative log
            # likelihood per sample.  To prevent this component from dominating
            # the aggregated logs, we normalise by the number of jigsaw classes.
            loss_jigsaw = F.cross_entropy(self.jigsaw_head(z_jigsaw), jigsaw_label, label_smoothing=smoothing_value)
            loss_jigsaw = loss_jigsaw / float(self.jigsaw_num_classes)
            loss_spike = F.mse_loss(self.spike_timing_head(z1).squeeze(), spike_target)
        else:
            loss_jigsaw = torch.tensor(0.0, device=self.device)
            loss_spike = torch.tensor(0.0, device=self.device)

        return loss_contrastive, loss_volatility, loss_jigsaw, loss_spike

    def _calculate_total_pretrain_loss(self, batch):
        """
        Hitung total pretrain loss dengan scaling untuk komponen yang besar,
        sekaligus memberi tanda jika ada komponen loss yang meledak + anti-shortcut check.
        """
        loss_contrastive, loss_volatility, loss_jigsaw, loss_spike = self._calculate_ssl_losses(batch)


        explode_thresholds = {
            "loss_volatility": 1e4,
            "kld_loss": 1e3,
        }


        if loss_volatility.item() > explode_thresholds["loss_volatility"]:
            logger.warning(f"⚠️ loss_volatility meledak: {loss_volatility.item():,.2f}")



        scaled_loss_volatility = loss_volatility * 0.001


        z1_proj = self.projector(self.forward_pretrain(batch["query_x_aug1"]))
        z2_proj = self.projector(self.forward_pretrain(batch["query_x_aug2"]))
        
        z_all = torch.cat([z1_proj, z2_proj], dim=0)
        z_all_norm = F.normalize(z_all, p=2, dim=1, eps=1e-8)

        similarity_matrix = torch.mm(z_all_norm, z_all_norm.T)
        loss_reg_latent = torch.mean(similarity_matrix[~torch.eye(z_all.shape[0], dtype=bool)])
        std_z = torch.sqrt(z_all_norm.var(dim=0) + 1e-04)
        loss_std = torch.mean(F.relu(1 - std_z))

        log_vars = 10.0 * torch.tanh(self.raw_log_vars / 10.0)

        total_loss = (
            self.dynamic_loss_weights["loss_contrastive"] * (torch.exp(-log_vars[0]) * loss_contrastive + 0.5 * log_vars[0]) +
            self.dynamic_loss_weights["loss_volatility"] * (torch.exp(-log_vars[1]) * loss_volatility + 0.5 * log_vars[1]) +
            self.dynamic_loss_weights["loss_jigsaw"] * (torch.exp(-log_vars[2]) * loss_jigsaw + 0.5 * log_vars[2]) +
            self.dynamic_loss_weights["loss_spike"] * (torch.exp(-log_vars[3]) * loss_spike + 0.5 * log_vars[3])
        )

        KLD_loss = torch.tensor(0.0, device=self.device)
        if self.mu is not None and self.logvar is not None:
            KLD_loss = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
            KLD_loss /= batch["query_x_aug1"].size(0)

            if KLD_loss.item() > explode_thresholds["kld_loss"]:
                logger.warning(f"⚠️ kld_loss meledak: {KLD_loss.item():,.2f}")


            KLD_loss = KLD_loss * 0.001

            total_loss += (self.hparams.beta_vib * KLD_loss) + (0.05 * loss_reg_latent) + (0.1 * loss_std)


        with torch.no_grad():
            aug2_corrupted = batch["query_x_aug2"].clone()
            B, T, F_dim = aug2_corrupted.shape
            mask_ratio = 0.3
            num_mask = int(F_dim * mask_ratio)
            mask_indices = random.sample(range(F_dim), num_mask)
            aug2_corrupted[:, :, mask_indices] = torch.randn_like(aug2_corrupted[:, :, mask_indices])

        z2_corrupted_proj = self.projector(self.forward_pretrain(aug2_corrupted))
        z2_corrupted_norm = F.normalize(z2_corrupted_proj, p=2, dim=1, eps=1e-8)

        shortcut_score = F.cosine_similarity(z2_proj, z2_corrupted_norm, dim=1).mean()
        if shortcut_score > 0.8:
            shortcut_penalty = (shortcut_score - 0.8) * 5.0
            total_loss += shortcut_penalty
        else:
            shortcut_penalty = torch.tensor(0.0, device=self.device)


        # ------------------------------------------------------------------
        # Compose dictionaries of raw and weighted losses for logging.  Raw
        # losses expose the unscaled magnitudes of each component, while
        # weighted losses reflect the actual contribution to the total loss
        # after dynamic scaling (λ) and variance stabilisation terms
        # (log_vars).  Weighted values are helpful for diagnosis when the raw
        # values appear large but are down‑weighted by λ.
        weighted_contrastive = self.dynamic_loss_weights["loss_contrastive"] * (torch.exp(-log_vars[0]) * loss_contrastive + 0.5 * log_vars[0])
        weighted_volatility = self.dynamic_loss_weights["loss_volatility"] * (torch.exp(-log_vars[1]) * loss_volatility + 0.5 * log_vars[1])
        weighted_jigsaw = self.dynamic_loss_weights["loss_jigsaw"] * (torch.exp(-log_vars[2]) * loss_jigsaw + 0.5 * log_vars[2])
        weighted_spike = self.dynamic_loss_weights["loss_spike"] * (torch.exp(-log_vars[3]) * loss_spike + 0.5 * log_vars[3])
        weighted_kld = self.hparams.beta_vib * KLD_loss
        weighted_latent_reg = 0.05 * loss_reg_latent
        weighted_latent_std = 0.10 * loss_std
        weighted_shortcut = shortcut_penalty
        losses_to_log = {
            # raw values
            "loss_contrastive": loss_contrastive,
            "loss_volatility": loss_volatility,
            "loss_jigsaw": loss_jigsaw,
            "loss_spike": loss_spike,
            "kld_loss": KLD_loss / 0.001 if KLD_loss != 0 else KLD_loss,
            "latent_reg_loss": loss_reg_latent,
            "latent_std_loss": loss_std,
            "shortcut_score": shortcut_score.detach(),
            "shortcut_penalty": shortcut_penalty.detach(),
            # weighted values
            "loss_contrastive_weighted": weighted_contrastive,
            "loss_volatility_weighted": weighted_volatility,
            "loss_jigsaw_weighted": weighted_jigsaw,
            "loss_spike_weighted": weighted_spike,
            "kld_loss_weighted": weighted_kld,
            "latent_reg_loss_weighted": weighted_latent_reg,
            "latent_std_loss_weighted": weighted_latent_std,
            "shortcut_penalty_weighted": weighted_shortcut,
        }
        return total_loss, losses_to_log



    def training_step(self, batch, batch_idx):
        """
        [VERSI FINAL] training_step dengan manual optimization untuk SAM, manual clipping,
        dan penanganan replay step dari callback.
        """
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        # Periodically decay loss weights slightly to encourage forgetting; router will override as needed.
        if batch_idx > 0 and batch_idx % 100 == 0:
            for k in self.dynamic_loss_weights.keys():
                self.dynamic_loss_weights[k] *= 0.95
        mode = self.hparams.get("mode", "fine-tune")
        if mode != 'pre-train':
            return None



        is_replay_step = (batch_idx == -1)

        # --- [Router Integration] ---
        # Initialize router at the first batch of each epoch.  Compute context and
        # select active groups, apply gating, and reset loss weights/micro LR.
        if self.router is not None:
            try:
                if self.current_epoch != self._router_initialized_epoch and batch_idx == 0:
                    # Use first batch of epoch as probe
                    self.router.start_epoch(self, batch, self.current_epoch, self.trainer.max_epochs)
                    self._router_initialized_epoch = self.current_epoch
                    # Reset running stats for new epoch
                    self._loss_running_stats = {}
            except Exception as _e:
                logger.warning(f"[Router] Error during start_epoch: {_e}")



        # Compute total loss and log components.  If the loss becomes
        # non‑finite (NaN or Inf), skip this batch to prevent the optimizer from
        # diverging.  We compute the loss twice as required by SAM, but we
        # check for validity before each backward pass.
        total_loss, losses_to_log = self._calculate_total_pretrain_loss(batch)
        # Guard against NaN/Inf in loss
        if not torch.isfinite(total_loss):
            try:
                # fingerprint batch to aid debugging; use md5 of first few values
                import hashlib
                payload = batch.get('query_x_aug1') if isinstance(batch, dict) else None
                if payload is not None:
                    md5 = hashlib.md5(payload.float().reshape(-1)[:100].cpu().numpy().tobytes()).hexdigest()
                    logger.warning(f"[Training] NaN/Inf detected in loss. Skipping batch. Batch fingerprint: {md5}")
                else:
                    logger.warning("[Training] NaN/Inf detected in loss. Skipping batch.")
            except Exception:
                logger.warning("[Training] NaN/Inf detected in loss. Skipping batch.")
            return None
        # ----------------------- SAM First Step -----------------------
        # Backward pass for the first SAM step
        self.manual_backward(total_loss)
        # Determine clipping value: use fixed norm=1.0 for first clip_warmup_steps, else dynamic
        warmup_steps = int(self.hparams.get('clip_warmup_steps', 300))
        clip_val = 1.0 if self.global_step < warmup_steps else max(self.hparams.get('dynamic_clip_value', 0.5), self.hparams.get('clip_floor', 0.1))
        self.clip_gradients(
            opt,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm="norm"
        )
        # Perform the first SAM update.  If the optimizer exposes the SAM
        # interface (first_step/second_step), use it; otherwise fall back
        # to a standard step followed by zeroing gradients.
        if hasattr(opt, 'first_step') and callable(getattr(opt, 'first_step')):
            opt.first_step(zero_grad=True)
        else:
            opt.step()
            if hasattr(opt, 'zero_grad'):
                opt.zero_grad(set_to_none=True)
        # ----------------------- SAM Second Step ----------------------
        # Compute the loss again under the adversarial perturbation
        total_loss_2, _ = self._calculate_total_pretrain_loss(batch)
        if not torch.isfinite(total_loss_2):
            logger.warning("[Training] NaN/Inf detected in second SAM loss. Skipping batch.")
            return None
        self.manual_backward(total_loss_2)
        # Clip gradients again before the second update
        self.clip_gradients(
            opt,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm="norm"
        )
        if hasattr(opt, 'second_step') and callable(getattr(opt, 'second_step')):
            opt.second_step(zero_grad=True)
        else:
            opt.step()
            if hasattr(opt, 'zero_grad'):
                opt.zero_grad(set_to_none=True)
        


        

        if not is_replay_step:
            if self.trainer.is_last_batch:
                sch.step()
                
            current_lr = sch.get_last_lr()[0] if sch else self.hparams.lr


            # Separate weighted and raw metrics for logging.  Only weighted
            # metrics are shown on the progress bar (prog_bar=True).  Raw
            # metrics are logged for diagnostics but hidden from the bar.
            log_payload_weighted = {
                "pretrain_loss_step": total_loss.detach(),
                "lr_step": current_lr,
            }
            log_payload_raw = {}
            for k, v in losses_to_log.items():
                name = f"{k}_step"
                if k.endswith('_weighted'):
                    log_payload_weighted[name] = v.detach()
                else:
                    log_payload_raw[name] = v.detach()
            # Log weighted metrics with progress bar enabled
            self.log_dict(log_payload_weighted, prog_bar=True, on_step=True, on_epoch=False, logger=True)
            # Log raw metrics separately without showing in the progress bar
            if log_payload_raw:
                self.log_dict(log_payload_raw, prog_bar=False, on_step=True, on_epoch=False, logger=True)

            # --- [Router Integration] ---
            # Mid‑epoch reroute: allow the router to adjust active experts on the fly
            if hasattr(self, "router") and self.router is not None:
                try:
                    self.router.maybe_reroute_mid_epoch(self, batch, self.global_step)
                except Exception as _e:
                    logger.warning(f"[Router] mid‑epoch re-route error: {_e}")
            # Update running statistics for loss adaptation.  We use detach() to avoid
            # interfering with autograd.  Each loss component accumulates sum and count.
            for loss_name, val in losses_to_log.items():
                # Skip weighted metrics when updating running stats; adaptation
                # should consider raw loss magnitudes only.
                if loss_name.endswith('_weighted'):
                    continue
                if loss_name not in self._loss_running_stats:
                    self._loss_running_stats[loss_name] = {'sum': 0.0, 'count': 0.0}
                stat = self._loss_running_stats[loss_name]
                try:
                    stat['sum'] += float(val.detach().cpu())
                    stat['count'] += 1.0
                except Exception:
                    pass
            # Every adapt_interval steps, invoke router adaptation logic
            if self.router is not None:
                adapt_int = getattr(self.router, 'adapt_interval', 50)
                if (self.global_step + 1) % adapt_int == 0:
                    try:
                        self.router.adapt(self, self._loss_running_stats)
                    except Exception as _e:
                        logger.warning(f"[Router] Adaptation error: {_e}")
                    # Reset stats for next window
                    self._loss_running_stats = {}


        self.log_dict({f"{k}_epoch": v.detach() for k, v in losses_to_log.items()}, on_step=False, on_epoch=True, logger=True)
        self.log("pretrain_loss_epoch", total_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=True)


        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        [VERSI FINAL] validation_step adaptif yang menangani kedua mode.
        """
        mode = self.hparams.get("mode", "fine-tune")




        if mode == 'pre-train':

            # Compute SSL losses for validation
            loss_contrastive, loss_volatility, loss_jigsaw, loss_spike = self._calculate_ssl_losses(batch)
            total_val_loss = loss_contrastive + loss_volatility + loss_jigsaw + loss_spike
            # Include KLD in validation loss if latent variables are available
            if self.mu is not None and self.logvar is not None:
                KLD_loss_val = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
                KLD_loss_val = KLD_loss_val / batch["query_x_aug1"].size(0)
                total_val_loss = total_val_loss + (self.hparams.beta_vib * KLD_loss_val)
            # Compute a weighted composite validation loss using the same dynamic
            # weights and variance terms as the training loss.  We reuse the
            # helper to obtain weighted components and sum all values ending
            # with '_weighted' to form the composite.
            with torch.no_grad():
                _full_val_loss, val_losses = self._calculate_total_pretrain_loss(batch)
                weighted_vals = [v for k, v in val_losses.items() if k.endswith('_weighted')]
                val_total_weighted = sum(weighted_vals)
            # Log both the raw and weighted validation losses.  Weighted loss
            # gives a clearer picture of how the objectives interact under
            # dynamic λ.  We keep the main domain metrics elsewhere.
            self.log_dict({
                "val_loss": total_val_loss,
                "val_total_weighted": val_total_weighted,
            }, prog_bar=True, on_epoch=True, logger=True)
            return total_val_loss
            



        else:

            (x_raw, x_combined, y, y_historical, anomaly_signals,
             edge_index, unc_targets, x_spikes) = batch


            outputs = self(x_raw, x_combined, y_historical, edge_index, x_spikes)
            regression_output, anomaly_output, unc_attribution, _, _, _ = outputs
            

            regression_loss = self.loss_fn(regression_output, y, anomaly_signals)
            anomaly_loss = F.binary_cross_entropy_with_logits(anomaly_output, anomaly_signals)
            unc_attribution_clamped = torch.clamp(unc_attribution, min=1e-8)
            attribution_loss = F.kl_div(unc_attribution_clamped.log(), unc_targets, reduction="batchmean")
            total_loss = 0.7 * regression_loss + 0.15 * anomaly_loss + 0.15 * attribution_loss



            mse = self.val_mse(regression_output, y)
            da = self.val_da(regression_output, y)
            

            self.log_dict({
                "val_loss": total_loss,
                "val_mse": mse,
                "val_da": da,
            }, prog_bar=True, on_epoch=True, logger=True)

            return total_loss

    def configure_optimizers(self):
            """
            [VERSI FINAL YANG SUDAH BENAR] Mengonfigurasi optimizer SAM dengan scheduler yang benar.
            """
            # Build parameter groups to apply a smaller learning rate to reactive heads.
            # The volatility head and raw_log_vars can produce unstable gradients when
            # overreacting to spiky targets.  Assigning them a "micro" LR helps
            # stabilize training without altering the overall architecture.  You
            # can adjust `micro_lr_factor` via hparams (default=0.1) to tune this.
            micro_lr_factor = self.hparams.get("micro_lr_factor", 0.1)
            micro_params = list(self.volatility_head.parameters()) + [self.raw_log_vars]
            micro_param_ids = {id(p) for p in micro_params}
            base_params = [p for p in self.parameters() if id(p) not in micro_param_ids]
            param_groups = [
                {"params": base_params, "lr": self.hparams.lr},
                {"params": micro_params, "lr": self.hparams.lr * micro_lr_factor},
            ]
            optimizer = SAM(param_groups, torch.optim.AdamW, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            logger.info("⚡️ Mengonfigurasi optimizer: SAM dengan parameter grup dan micro-LR.")
            scheduler_type = self.hparams.get("scheduler_type", "OneCycleLR")

            if scheduler_type == "OneCycleLR":
                logger.info("INFO: Menggunakan scheduler OneCycleLR.")
                total_steps = getattr(self.trainer, 'estimated_stepping_batches', 100000)
                

                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=self.hparams.lr, 
                    total_steps=total_steps,
                    cycle_momentum=False, 
                    pct_start=0.1, 
                    anneal_strategy="cos",
                    div_factor=25, 
                    final_div_factor=1e4
                )
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
            
            elif scheduler_type == "ReduceLROnPlateau":
                logger.info("INFO: Menggunakan scheduler ReduceLROnPlateau.")
                

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode="min", 
                    patience=5, 
                    factor=0.5, 
                    verbose=True
                )
                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
            
            # Expose optimizer so the router can adjust param group learning rates
            self._optimizer = optimizer
            return optimizer

    def on_train_end(self):
        """[TIDAK BERUBAH] Menyimpan bobot model."""
        if self.hparams.get("project_id"):


            path = f"./{self.hparams.project_id}_full_model.pth"
            logger.info(f"\nTraining selesai. Menyimpan arsitektur lengkap ke: {path}")

            torch.save(self.state_dict(), path)

    def _hard_negative_margin_loss(self, z1: torch.Tensor, z2: torch.Tensor, margin: float = 0.2):
        """[TIDAK BERUBAH] Loss "anti-cheat" dengan margin."""
        dynamic_eps = 1e-8
        try:
            with torch.no_grad():
                batch_variance = torch.mean(torch.var(z1, dim=0)) + torch.mean(torch.var(z2, dim=0))
                dampened_variance = torch.sqrt(batch_variance + 1e-9)
                dynamic_eps_calc = torch.clamp(1e-7 / (dampened_variance + 1e-9), 1e-9, 1e-5).item()
                if not (torch.isnan(torch.tensor(dynamic_eps_calc)) or torch.isinf(torch.tensor(dynamic_eps_calc))):
                    dynamic_eps = dynamic_eps_calc

            fx_name = getattr(self, "_current_fx_name", "TIDAK DITEMUKAN")

            if self.trainer.global_step % 50 == 0:
                logger.debug(f"[ACR Debug] Konteks saat ini (_current_fx_name): {fx_name}")
            if self.training:

                in_epoch_end = getattr(self, "_current_fx_name", "") == "on_train_epoch_end"
                if in_epoch_end:

                    self.log("dynamic_epsilon", dynamic_eps, on_step=False, on_epoch=True, prog_bar=True)
                else:

                    self.log("dynamic_epsilon", dynamic_eps, on_step=True, on_epoch=False, prog_bar=True)

        except Exception:
            if self.training:
                in_epoch_end = getattr(self, "_current_fx_name", "") == "on_train_epoch_end"
                if in_epoch_end:
                    self.log("dynamic_epsilon_fallback", dynamic_eps, on_step=False, on_epoch=True, prog_bar=True)
                else:
                    self.log("dynamic_epsilon_fallback", dynamic_eps, on_step=True, on_epoch=False, prog_bar=True)

        safe_epsilon = max(dynamic_eps, 1e-6)
        z1 = F.normalize(z1, p=2, dim=1, eps=safe_epsilon)
        z2 = F.normalize(z2, p=2, dim=1, eps=safe_epsilon)

        similarity_matrix = torch.matmul(z1, z2.T)
        positive_sim = torch.diag(similarity_matrix).view(-1, 1)
        mask = torch.eye(z1.size(0), dtype=torch.bool, device=self.device)
        negative_sims = similarity_matrix.masked_fill(mask, -1e9)
        hardest_negative_sim, _ = torch.max(negative_sims, dim=1, keepdim=True)
        loss = F.relu(hardest_negative_sim - positive_sim + margin).mean()
        return loss


    def on_after_backward(self):
        """
        Hook ini dipanggil setelah loss.backward() dan sebelum optimizer.step().
        Ini adalah tempat sempurna untuk memeriksa apakah gradien ada.
        """

        if self.trainer.global_step < 5:
            print(f"\n\nDEBUG on_after_backward (Global Step: {self.trainer.global_step}):")
            

            params_checked = 0
            has_any_grad = False
            for name, param in self.named_parameters():
                if params_checked >= 5:
                    break
                
                if param.grad is not None:

                    print(f"  - Parameter '{name}' -> Gradien ADA. Norma: {param.grad.norm().item()}")
                    has_any_grad = True
                else:

                    print(f"  - Parameter '{name}' -> Gradien TIDAK ADA (None).")
                
                params_checked += 1

            if not has_any_grad:
                print("  - KESIMPULAN: TIDAK ADA SATUPUN GRADIEN DITEMUKAN PADA MODEL!\n\n")
            else:
                print("  - KESIMPULAN: Gradien ditemukan pada beberapa parameter.\n\n")


class TGW_Memory:
    """
    Menyimpan snapshot dari kondisi model pre-train untuk digunakan oleh
    Temporal Gradient Wormhole (TGW) selama fine-tuning.
    """

    def __init__(self, device):
        self.device = device
        self.pre_train_gradients = None
        self.pre_train_positional_encoding = None
        self.is_snapshot_taken = False
        logger.info("🧠 Memori TGW (Temporal Gradient Wormhole) diinisialisasi.")

    def take_snapshot(self, model: pl.LightningModule, dataloader: DataLoader):
        """
        Mengambil snapshot dari gradien rata-rata dan positional encoding
        dari model pre-train.
        """
        if self.is_snapshot_taken:
            logger.warning(
                "Snapshot TGW sudah diambil sebelumnya. Melewatkan.")
            return

        logger.info("📸 Mengambil snapshot TGW dari model pre-train...")
        model.eval()
        model.to(self.device)


        try:
            batch = next(iter(dataloader))

            batch = [
                item.to(self.device) if isinstance(
                    item, torch.Tensor) else item
                for item in batch
            ]
        except StopIteration:
            logger.error(
                "Dataloader pre-train kosong. Tidak dapat mengambil snapshot TGW."
            )
            return



        aug1, aug2, _, _, _, _ = batch
        if aug1 is None:
            return

        model.zero_grad()

        encoded_rep1 = model(aug1)
        z1 = model.projector(encoded_rep1)
        encoded_rep2 = model(aug2)
        z2 = model.projector(encoded_rep2)
        loss = model._info_nce_loss(z1, z2)
        loss.backward()



        grads = []
        for param in model.encoder.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().clone().flatten())

        if not grads:
            logger.error(
                "Tidak ada gradien yang ditemukan untuk snapshot TGW.")
            return

        self.pre_train_gradients = torch.cat(grads).mean().unsqueeze(0)



        if hasattr(model.encoder, "pe"):
            self.pre_train_positional_encoding = model.encoder.pe.detach().clone()
        else:

            pe = torch.zeros(
                model.hparams.window, model.hparams.d_model, device=self.device
            )
            position = torch.arange(
                0, model.hparams.window, dtype=torch.float
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model.hparams.d_model, 2).float()
                * (-math.log(10000.0) / model.hparams.d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pre_train_positional_encoding = pe

        self.is_snapshot_taken = True
        logger.info(
            f"✅ Snapshot TGW berhasil diambil. Gradien Avg: {self.pre_train_gradients.item():.6f}"
        )

    def get_snapshot(self):
        """Mengembalikan snapshot yang tersimpan."""
        if not self.is_snapshot_taken:
            raise RuntimeError(
                "Snapshot TGW belum diambil. Panggil take_snapshot() terlebih dahulu."
            )
        return self.pre_train_gradients, self.pre_train_positional_encoding


class TGW_Callback(pl.Callback):
    """
    Callback untuk mengimplementasikan mekanisme injeksi gradien dari
    Temporal Gradient Wormhole (TGW).
    """

    def __init__(self, tgw_memory: TGW_Memory, tgw_beta: float = 0.3):
        super().__init__()
        self.tgw_memory = tgw_memory
        self.beta = tgw_beta
        self.pre_train_grads, _ = self.tgw_memory.get_snapshot()
        logger.info(f"⚙️ Callback TGW aktif dengan beta = {self.beta}")

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: "Optimizer",
    ):
        """
        Dipanggil tepat sebelum optimizer melakukan update.
        Di sinilah kita melakukan injeksi gradien.
        """
        wormhole_matrix = pl_module.core.wormhole_matrix
        if wormhole_matrix is None or self.pre_train_grads is None:
            return


        pre_train_grads_device = self.pre_train_grads.to(pl_module.device)

        with torch.no_grad():



            wormhole_grad_component = wormhole_matrix @ pre_train_grads_device.T


        for param in pl_module.parameters():
            if param.grad is not None:

                current_grad = param.grad.data


                wormhole_grad_to_add = wormhole_grad_component.expand_as(
                    current_grad.mean()
                ).mean()



                entangled_grad = wormhole_grad_to_add + self.beta * current_grad


                param.grad.data = entangled_grad

    _affmap = build_affinity_map(total_cores=None)
    _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)


def generate_anomaly_features(
    df: pd.DataFrame,
    feature_columns: list,
    training_fraction: float = 0.7,
    epochs: int = 27,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Melatih autoencoder sederhana pada bagian data yang 'normal' dan kemudian
    menggunakannya untuk menghasilkan fitur 'skor anomali' untuk seluruh dataset.

    Args:
        df (pd.DataFrame): DataFrame input yang berisi data.
        feature_columns (list): Daftar nama kolom yang akan digunakan untuk melatih autoencoder.
        training_fraction (float): Fraksi data yang akan digunakan untuk pelatihan (diambil dari awal).
        epochs (int): Jumlah epoch untuk melatih autoencoder.
        batch_size (int): Ukuran batch untuk pelatihan.

    Returns:
        pd.DataFrame: DataFrame asli dengan tambahan kolom 'anomaly_score'.
    """
    logger.info("🚀 Memulai pembuatan fitur skor anomali via Autoencoder...")
    df_with_anomaly = df.copy()





    numeric_cols = df[feature_columns].select_dtypes(
        include=np.number).columns.tolist()

    if not numeric_cols:
        logger.error(
            "Tidak ada fitur numerik yang ditemukan untuk melatih Autoencoder. Fitur anomali tidak akan dibuat."
        )
        df_with_anomaly["anomaly_score"] = (
            0.0
        )
        return df_with_anomaly

    logger.info(
        f"Menggunakan {len(numeric_cols)} fitur numerik yang valid untuk Autoencoder."
    )
    data = df[numeric_cols].values


    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)


    split_index = int(len(data_scaled) * training_fraction)
    train_data = data_scaled[:split_index]

    if len(train_data) < 10:
        logger.warning(
            "Data pelatihan tidak cukup untuk Autoencoder (<10 baris). Fitur anomali tidak dibuat."
        )
        df_with_anomaly["anomaly_score"] = 0.0
        return df_with_anomaly

    _affmap = build_affinity_map(total_cores=None)
    _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)

    train_tensor = torch.tensor(train_data, dtype=torch.float32)

    from torch.utils.data import TensorDataset

    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = TorchDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4,
        persistent_workers=True,
        worker_init_fn=_worker_init,
        prefetch_factor=4,
    )



    input_dim = len(numeric_cols)


    autoencoder = EnhancedAutoencoder(
        input_dim=input_dim, hidden_dim=input_dim // 2)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    logger.info(
        f"Melatih Autoencoder selama {epochs} epoch pada {len(train_data)} sampel data 'normal'..."
    )
    autoencoder.train()
    for epoch in range(epochs):
        for batch_features, _ in train_loader:
            optimizer.zero_grad()

            _, outputs = autoencoder(batch_features)
            loss = criterion(outputs, batch_features)
            loss.backward()
            optimizer.step()

    logger.info("Pelatihan Autoencoder selesai.")


    logger.info("Menghitung skor anomali untuk seluruh dataset...")
    autoencoder.eval()
    with torch.no_grad():
        full_data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
        _, reconstructed_data = autoencoder(full_data_tensor)
        reconstruction_error = torch.mean(
            (full_data_tensor - reconstructed_data) ** 2, dim=1
        )


    df_with_anomaly["anomaly_score"] = reconstruction_error.numpy()


    df_with_anomaly["anomaly_score"] = StandardScaler().fit_transform(
        df_with_anomaly[["anomaly_score"]]
    )

    logger.info("✅ Fitur 'anomaly_score' berhasil dibuat dan ditambahkan.")

    return df_with_anomaly


class SSP_LitModule(pl.LightningModule):
    """
    Model untuk Self-Supervised Pre-training menggunakan TST (Time Series Transformer)
    dan InfoNCE contrastive loss. Tujuannya adalah untuk mempelajari representasi
    time-series yang baik sebelum fine-tuning.
    """

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)


        self.encoder = TST_module(
            c_in=self.hparams.n_features_input,
            c_out=self.hparams.d_model,
            seq_len=self.hparams.window,
            d_model=self.hparams.d_model,
            n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
        )



        self.projector = nn.Sequential(
            nn.Linear(self.hparams.d_model, self.hparams.d_model),
            nn.SiLU(),
            nn.Linear(self.hparams.d_model, 128),
        )


        self.temperature = 0.1

    def forward(self, x):


        return self.encoder(x.transpose(1, 2))

    def _info_nce_loss(self, z1, z2):
        """Menghitung InfoNCE Loss untuk self-supervised learning."""

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)


        logits = (z1 @ z2.T) / self.temperature


        labels = torch.arange(logits.shape[0], device=self.device)


        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):

        aug1, aug2 = batch
        if aug1 is None or aug2 is None:
            return None


        z1 = self.projector(self(aug1))
        z2 = self.projector(self(aug2))


        loss = self._info_nce_loss(z1, z2)


        self.log("pretrain_loss", loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):

        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def on_train_end(self):

        if self.hparams.project_id:
            path = get_path(self.hparams.project_id, "pretrained_encoder")
            logger.info(
                f"\nPre-training selesai. Menyimpan bobot encoder ke: {path}")
            torch.save(self.encoder.state_dict(), path)


class DifferentiableRule(nn.Module):
    """
    Merepresentasikan satu aturan logika IF-THEN yang parameternya (misal: threshold)
    dapat dipelajari melalui backpropagation.
    """

    def __init__(
        self, feature_indices: dict, initial_thresholds: dict, rule_logic: callable
    ):
        super().__init__()
        self.feature_indices = feature_indices
        self.rule_logic = rule_logic

        self.thresholds = nn.ParameterDict(
            {
                key: nn.Parameter(torch.tensor(val, dtype=torch.float32))
                for key, val in initial_thresholds.items()
            }
        )

    def forward(self, x_raw_features: torch.Tensor) -> torch.Tensor:
        """
        Mengevaluasi seberapa 'benar' aturan ini untuk input data.
        Output adalah nilai 'truth value' antara 0 (salah) dan 1 (benar).

        Args:
            x_raw_features (torch.Tensor): Tensor input fitur mentah [Batch, SeqLen, Features]
        """

        return self.rule_logic(x_raw_features, self.feature_indices, self.thresholds)


def bearish_rsi_volume_logic(x_raw_features, feature_indices, thresholds):
    """Logika spesifik untuk aturan 'Jika RSI overbought DAN Volume naik, maka Bearish'."""

    rsi = x_raw_features[:, -1, feature_indices["RSI_14"]]
    volume_change = x_raw_features[:, -1, feature_indices["Volume_pct_change"]]



    rsi_condition = torch.sigmoid(
        (rsi - thresholds["rsi_thresh"]) * 10
    )
    volume_condition = torch.sigmoid(
        (volume_change - thresholds["vol_thresh"]) * 10)


    truth_value = rsi_condition * volume_condition
    return truth_value.unsqueeze(1)


class FF_Model(pl.LightningModule):
    """
    Sebuah model sederhana yang dirancang khusus untuk dilatih
    menggunakan Forward-Forward Algorithm secara per-lapisan.
    """

    def __init__(self, input_dim: int, layer_sizes: list = [512, 512, 512]):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList()


        in_size = input_dim
        for out_size in layer_sizes:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_size, out_size),
                    nn.ReLU()
                )
            )
            in_size = out_size



        self.automatic_optimization = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Meneruskan input melalui semua lapisan.
        Normalisasi per-sampel penting untuk mencegah semua neuron belajar hal yang sama.
        """

        x = x / (x.std(dim=-1, keepdim=True) + 1e-8)

        for layer in self.layers:
            x = layer(x)
        return x




class AlphaLit(pl.LightningModule):
    def __init__(
        self,
        hparams,
        feature_names: list,
        event_names: list,
        gemini_advisor: "GeminiAdvancedAdvisor",
        api_pool: "DistributedAIPool",
        brain: "Brain",
        causal_auditor: "CausalInternalAuditor",
        nsmm: "NSMM",
        engine: "AsyncCuriosityEngine",
        tgw_memory: "TGW_Memory" = None,
        denoising_model=None,
        gnn_communicator=None,
        energy_shield=None,
        echo_radio=None,
    ):
        super().__init__()

        self.state_vector_dim = 12
        self.snn_firing_rate_history = []
        self.engine = engine


        self.feature_names = feature_names
        self.event_names = event_names
        self.gemini_advisor = gemini_advisor
        self.api_pool = api_pool
        self.brain = brain
        self.causal_auditor = causal_auditor
        self.nsmm = nsmm
        self.tgw_memory = tgw_memory
        self.denoising_model = denoising_model
        self.gnn_communicator = gnn_communicator
        self.energy_shield = energy_shield
        self.echo_radio = echo_radio


        hparams_dict = hparams if isinstance(hparams, dict) else vars(hparams)

        hparams_dict.setdefault("n_features_input", len(self.feature_names))
        hparams_dict.setdefault(
            "n_targets", len(hparams_dict.get("selected_tickers", [1, 2]))
        )

        hparams_dict.setdefault("d_model", 128)
        hparams_dict.setdefault("n_layers", 2)
        hparams_dict.setdefault("n_heads", 8)
        hparams_dict.setdefault("dropout", 0.2)
        hparams_dict.setdefault("window", 60)
        hparams_dict.setdefault("horizon", 7)
        hparams_dict.setdefault("da_weight", 0.3)
        hparams_dict.setdefault("snn_input_size", 1)
        hparams_dict.setdefault("snn_hidden_size", 64)
        hparams_dict.setdefault("snn_num_steps", 25)
        hparams_dict.setdefault("snn_num_layers", 2)
        hparams_dict.setdefault("snn_dropout", 0.2)
        # Align the default number of FNO modes with pretrained checkpoints (12).
        hparams_dict.setdefault("fno_modes", 12)
        hparams_dict.setdefault("use_rg_block", False)
        hparams_dict.setdefault("pid_kp", 0.1)
        hparams_dict.setdefault("pid_ki", 0.01)
        hparams_dict.setdefault("pid_kd", 0.05)
        hparams_dict.setdefault("use_tgw", False)
        hparams_dict.setdefault("tgw_rank", 8)
        hparams_dict.setdefault("tgw_beta", 0.3)
        hparams_dict.setdefault("tgw_stability_weight", 0.05)
        hparams_dict.setdefault("drop_path_rate", 0.1)
        hparams_dict.setdefault("distil_weight", 0.5)

        if "n_uncertainty_factors" not in hparams_dict:
            dm_temp = AlphaDataModule(
                hparams_dict, None, None, None, None, None, brain=brain,
                engine=self.engine
            )
            hparams_dict["n_uncertainty_factors"] = dm_temp.n_uncertainty_factors

        self.save_hyperparameters(hparams_dict)
        self.gradient_collapse_detected_in_epoch = False
        self.register_buffer(
            "dynamic_tgw_weight", torch.tensor(
                self.hparams.tgw_stability_weight)
        )
        self.tgw_cooldown = 0

        if self.hparams.use_tgw and self.tgw_memory is None:
            raise ValueError(
                "TGW diaktifkan tetapi TGW_Memory tidak disediakan.")


        self.core = Evolved_Hybrid_SNN_Model(
            feature_names=self.feature_names,
            hparams=self.hparams,
            n_primary_features=len(self.feature_names),
            n_leftover_features=len(self.event_names),
            n_features_input=self.hparams.n_features_input,
            n_targets=self.hparams.n_targets,
            d_model=self.hparams.d_model,
            n_layers=self.hparams.n_layers,
            n_heads=self.hparams.n_heads,
            dropout=self.hparams.dropout,
            window=self.hparams.window,
            horizon=self.hparams.horizon,
            n_uncertainty_factors=self.hparams.n_uncertainty_factors,
            snn_input_size=self.hparams.snn_input_size,
            snn_hidden_size=self.hparams.snn_hidden_size,
            snn_num_layers=self.hparams.snn_num_layers,
            snn_dropout=self.hparams.snn_dropout,
            fno_modes=self.hparams.fno_modes,
            tgw_rank=self.hparams.tgw_rank,
        )

        self.state_vector_dim = 10
        self.awareness_modules = nn.ModuleList()
        if self.nsmm:
            reconstructed_neurons = self.nsmm.retrieve_relevant_neurons(
                self.state_vector_dim
            )
            for name, module in reconstructed_neurons:
                self.awareness_modules.append(module)

        num_loaded_neurons = len(self.awareness_modules)
        consciousness_input_dim = self.hparams.d_model + (
            num_loaded_neurons * 8
        )
        self.consciousness_fusion = (
            nn.Linear(consciousness_input_dim, self.hparams.d_model)
            if num_loaded_neurons > 0
            else None
        )

        self.wisdom_fusion_gate = SpatioTemporalGatedFusion(self.hparams.d_model)
        self.cognitive_resonance_callback = None
        
        self.fascia_network = FasciaNetwork(
            state_vector_dim=self.state_vector_dim,


            num_losses=7
        )


        self.loss_fn = ReturnLoss(
            da_weight=self.hparams.da_weight, horizon=self.hparams.horizon
        )
        self.val_mse = MeanSquaredError()
        self.grad_norm_history = {}
        self.grad_monitor_threshold = 1e-6

        heads_to_monitor = [
            "volatility_head",
            "jigsaw_head",
            "spike_timing_head",
            "regression_head",
            "anomaly_head"
        ]

        for head_name in heads_to_monitor:
            head_module = getattr(self, head_name, None)
            if head_module is not None:
                for name, param in head_module.named_parameters():
                    if param.requires_grad:
                        param.register_hook(self._make_grad_monitor_hook(f"{head_name}.{name}"))
        self.val_rmse = lambda p, t: torch.sqrt(self.val_mse(p, t))
        self.val_da = lambda p, t: torch.mean(
            (torch.sign(p) == torch.sign(t)).float())
        self._cb_train_data = []
        self.mapie_models = []
        self.performance_history = []

        self.symbolic_translator = SymbolicTranslator(
            thresholds={
                "bullish_return": 0.005,
                "bearish_return": -0.005,
                "high_volatility": 0.4,
                "low_volatility": 0.15,
            }
        )
        logger.info("🧠 Menginisialisasi Bank Aturan Neurosimbolik...")
        self.logical_reasoner = DynamicLogicalReasoner(
            brain=self.brain, api_pool=self.api_pool
        )






        feature_map = {name: i for i, name in enumerate(self.feature_names)}

        self.rule_bank = nn.ModuleList()

        if self.hparams.selected_tickers:
            first_ticker = self.hparams.selected_tickers[0]
            rsi_col = f"{first_ticker}_RSI_14"
            vol_col = f"{first_ticker}_Volume_pct_change"


            if rsi_col in feature_map and vol_col in feature_map:
                self.rule_bank.append(
                    DifferentiableRule(
                        feature_indices={
                            "RSI_14": feature_map[rsi_col],
                            "Volume_pct_change": feature_map[vol_col],
                        },
                        initial_thresholds={
                            "rsi_thresh": 70.0, "vol_thresh": 0.1},
                        rule_logic=bearish_rsi_volume_logic,
                    )
                )
                logger.info(
                    f"Aturan logika neurosimbolik dinamis dibuat untuk ticker: {first_ticker}"
                )


        if self.rule_bank:

            self.symbolic_fusion_head = nn.Linear(
                len(self.rule_bank), self.hparams.d_model
            )
        else:
            self.symbolic_fusion_head = None

        self.logic_consistency_head = nn.Sequential(
            nn.Linear(self.hparams.d_model, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 5),
        )

        self.action_map = {
            "strong_buy": 0,
            "cautious_buy": 1,
            "hold": 2,
            "strong_sell": 3,
            "cautious_sell": 4,
        }

    def _calculate_sharpe_ratio(self, returns):
        if returns.std() == 0:
            return torch.tensor(0.0, device=self.device)
        risk_free_rate_daily = 0.02 / 252
        excess_returns = returns - risk_free_rate_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * torch.sqrt(
            torch.tensor(252.0, device=self.device)
        )
        return sharpe_ratio

    def load_neurons_from_nsmm(self):
        """Memuat neuron yang sudah ada dari NSMM saat model diinisialisasi."""
        retrieved_neurons = self.nsmm.retrieve_relevant_neurons(
            n_top=10
        )
        for neuron_data in retrieved_neurons:
            try:

                module = nn.Sequential(
                    nn.Linear(len(self.hparams),
                              16), nn.SiLU(), nn.Linear(16, 8)
                )

                buffer = io.BytesIO(neuron_data["weights_blob"])
                module.load_state_dict(torch.load(buffer))
                self.awareness_modules.append(module)
            except Exception as e:
                logger.warning(
                    f"Gagal memuat neuron '{neuron_data['name']}': {e}")
        logger.info(
            f"Berhasil memuat {len(self.awareness_modules)} Insight Modules dari NSMM."
        )
    
    def forward(self, x_raw, x_combined, y_historical, edge_index, x_spikes, **kwargs):
        """
        Meneruskan input ke model inti (self.core) untuk diproses.
        Metode ini wajib ada di setiap nn.Module / pl.LightningModule.
        """

        return self.core(
            x_raw, x_combined, y_historical, edge_index, x_spikes, **kwargs
        )

    def get_neuro_vitals(self) -> tuple[dict, str]:
        """
        Mengukur dan mengembalikan 'tanda-tanda vital' model saat ini (Versi Aman + Proteksi + Validasi Awal).
        """
        state_vector = {}


        params = list(self.parameters())
        if len(params) == 0:
            default_vector = {k: 0.0 for k in [
                "gradient_norm", "weight_norm", "lr", "epoch",
                "snn_avg_firing_rate", "activation_variance",
                "val_loss", "val_da", "duality_weight", "tgw_stability_loss"
            ]}
            vitals_summary = "GradNorm:0.0000, WeightNorm:0.00, LR:0.000000, FiringRate:0.000"
            trimmed_vector = [0.0] * self.state_vector_dim
            return default_vector, vitals_summary


        grad_norms = [
            p.grad.norm(2).item()
            for p in params
            if p.grad is not None and p.grad.is_floating_point()
        ]
        avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
        state_vector["gradient_norm"] = avg_grad_norm


        try:
            total_norm_sq = sum((p.data.norm(2).item()) ** 2 for p in params)
            weight_norm = total_norm_sq ** 0.5 if total_norm_sq > 0 else 0.0
        except Exception:
            weight_norm = 0.0
        state_vector["weight_norm"] = weight_norm


        try:
            optimizer = self.trainer.optimizers[0]
            lr = optimizer.param_groups[0].get("lr", 0.0)
        except (AttributeError, IndexError):
            lr = 0.0
        state_vector["lr"] = lr

        epoch = getattr(self.trainer, "current_epoch", 0.0)
        state_vector["epoch"] = float(epoch)


        avg_firing_rate = (
            float(np.mean(self.snn_firing_rate_history[-100:]))
            if getattr(self, "snn_firing_rate_history", None)
            else 0.0
        )
        state_vector["snn_avg_firing_rate"] = avg_firing_rate


        activation_variances = []
        for p in params:
            try:
                activation_variances.append(torch.var(p.data).item())
            except Exception:
                continue
        state_vector["activation_variance"] = (
            float(np.mean(activation_variances)) if activation_variances else 0.0
        )


        metrics = getattr(self.trainer, "callback_metrics", {})

        def safe_metric(key):
            val = metrics.get(key, 0.0)
            return val.item() if hasattr(val, "item") else float(val)

        state_vector["val_loss"] = safe_metric("val_loss")
        state_vector["val_da"] = safe_metric("val_da")


        duality_weight = getattr(self.core, "last_duality_weight", None)
        if isinstance(duality_weight, torch.Tensor):
            duality_weight = duality_weight.mean().item()
        else:
            duality_weight = 0.0
        state_vector["duality_weight"] = duality_weight
        state_vector["tgw_stability_loss"] = safe_metric("tgw_stability_loss")


        sorted_keys = sorted(state_vector.keys())
        final_vector_list = [state_vector[k] for k in sorted_keys]

        trimmed_vector = final_vector_list[: self.state_vector_dim]
        while len(trimmed_vector) < self.state_vector_dim:
            trimmed_vector.append(0.0)

        vitals_summary = (
            f"GradNorm:{avg_grad_norm:.4f}, WeightNorm:{weight_norm:.2f}, "
            f"LR:{lr:.6f}, FiringRate:{avg_firing_rate:.3f}"
        )

        return {k: v for k, v in zip(sorted_keys, trimmed_vector)}, vitals_summary

    def _make_grad_monitor_hook(self, param_name):
        def hook_fn(grad):
            norm_val = grad.norm().item()
            if param_name not in self.grad_norm_history:
                self.grad_norm_history[param_name] = []
            self.grad_norm_history[param_name].append(norm_val)

            if norm_val < self.grad_monitor_threshold:
                logger.warning(f"⚠️ Grad norm kecil pada {param_name}: {norm_val:.2e}")
        return hook_fn
        

    def training_step(self, batch, batch_idx):

        tracer = CognitiveTracer(step_id=f"E{self.current_epoch}_B{batch_idx}")

        try:

            (
                x_raw, x_combined, y, y_historical, anomaly_signals,
                edge_index, unc_targets, x_spikes,
            ) = batch

            if x_combined is None:
                logger.warning(f"Batch E{self.current_epoch}_B{batch_idx} dilewati karena x_combined is None.")
                tracer.add_step("BATCH_UNPACK", "FAIL", "Empty batch (x_combined is None)")
                self.last_failed_trace = tracer.get_trace()
                return None

            tracer.add_step("BATCH_UNPACK", "OK")


            try:

                forward_kwargs = {"tracer": tracer}
                if (
                    self.hparams.use_tgw
                    and self.tgw_memory
                    and self.tgw_memory.is_snapshot_taken
                ):
                    _, pre_train_pe = self.tgw_memory.get_snapshot()
                    forward_kwargs["pre_train_pe"] = pre_train_pe
                

                outputs = self(
                    x_raw, x_combined, y_historical, edge_index, x_spikes, **forward_kwargs
                )
                

                (
                    regression_output, anomaly_output, unc_attribution_pred,
                    final_fused_embedding, student_preds, model_outputs_for_stability,
                ) = outputs
                tracer.add_step("FORWARD_PASS", "OK")

            except Exception as e:
                tracer.add_step("FORWARD_PASS", "FATAL_ERROR", details=f"{type(e).__name__}: {e}")
                self.last_failed_trace = tracer.get_trace()
                raise e


            try:

                regression_loss = self.loss_fn(regression_output, y, anomaly_signals)
                tracer.add_step("LOSS_REGRESSION", "OK", f"val={regression_loss.item():.4f}")

                anomaly_loss = F.binary_cross_entropy(anomaly_output, anomaly_signals.float())
                tracer.add_step("LOSS_ANOMALY", "OK", f"val={anomaly_loss.item():.4f}")
                
                unc_attribution_pred_clamped = torch.clamp(unc_attribution_pred, min=1e-8)
                attribution_loss = F.kl_div(
                    unc_attribution_pred_clamped.log(), unc_targets, reduction="batchmean"
                )
                tracer.add_step("LOSS_ATTRIBUTION", "OK", f"val={attribution_loss.item():.4f}")
                
                stdp_loss = self.core.stdp_proxy_loss
                tracer.add_step("LOSS_STDP", "OK", f"val={stdp_loss.item():.4f}")

                mu, logvar = self.core.mu, self.core.logvar
                KLD_loss = torch.tensor(0.0, device=self.device)
                if mu is not None and logvar is not None:
                    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_combined.size(0)
                tracer.add_step("LOSS_KLD", "OK", f"val={KLD_loss.item():.4f}")

                tgw_stability_loss = torch.tensor(0.0, device=self.device)
                if self.hparams.use_tgw and model_outputs_for_stability:
                    pre_train_output = model_outputs_for_stability["pre_train_equivalent"]
                    fine_tune_output = model_outputs_for_stability["fine_tune_output"]
                    tgw_stability_loss = 1 - F.cosine_similarity(pre_train_output, fine_tune_output, dim=-1).mean()
                tracer.add_step("LOSS_TGW_STABILITY", "OK", f"val={tgw_stability_loss.item():.4f}")

                distill_loss = torch.tensor(0.0, device=self.device)
                if student_preds is not None:
                    with torch.no_grad():
                        teacher_preds = regression_output.detach()
                    distill_loss = F.mse_loss(student_preds, teacher_preds)
                tracer.add_step("LOSS_DISTILLATION", "OK", f"val={distill_loss.item():.4f}")
                

                state_vitals, _ = self.get_neuro_vitals()
                state_vector_list = [float(state_vitals.get(k, 0)) for k in sorted(state_vitals.keys())]
                state_vector_tensor = torch.tensor(state_vector_list, device=self.device).float().unsqueeze(0).repeat(x_combined.size(0), 1)
                

                if self.trainer.global_step < 100:
                    num_losses = 7
                    loss_weights = torch.full((x_combined.size(0), num_losses), 1.0/num_losses, device=self.device)
                else:
                    loss_weights = self.fascia_network(state_vector_tensor)
                tracer.add_step("FASCIA_NETWORK", "OK")


                loss_components = torch.stack([
                    regression_loss, anomaly_loss, attribution_loss, stdp_loss,
                    KLD_loss, tgw_stability_loss, distill_loss,
                ])
                avg_loss_weights = loss_weights.mean(dim=0)
                total_loss = torch.sum(avg_loss_weights * loss_components)
                tracer.add_step("LOSS_AGGREGATE", "OK", f"total={total_loss.item():.4f}")

            except Exception as e:
                tracer.add_step("LOSS_CALC", "FATAL_ERROR", details=f"{type(e).__name__}: {e}")
                self.last_failed_trace = tracer.get_trace()
                raise e


            try:
                optimizer = self.optimizers()
                optimizer.zero_grad()
                self.manual_backward(total_loss)
                tracer.add_step("BACKWARD_PASS", "OK")

                self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                tracer.add_step("GRAD_CLIP", "OK")

                optimizer.step()
                tracer.add_step("OPTIMIZER_STEP", "OK")
            except Exception as e:
                tracer.add_step("OPTIMIZATION", "FATAL_ERROR", details=f"{type(e).__name__}: {e}")
                self.last_failed_trace = tracer.get_trace()
                raise e


            try:

                self.log_dict(
                    {
                        "train_loss": total_loss,
                        "w_reg": avg_loss_weights[0], "w_ano": avg_loss_weights[1],
                        "w_attr": avg_loss_weights[2], "w_stdp": avg_loss_weights[3],
                        "w_kld": avg_loss_weights[4], "w_tgw": avg_loss_weights[5],
                        "w_distill": avg_loss_weights[6],
                    },
                    prog_bar=True, on_step=True, on_epoch=True, logger=True,
                )
                tracer.add_step("METRICS_LOG", "OK")
            except Exception as e:

                tracer.add_step("LOGGING", "ERROR", details=f"{type(e).__name__}: {e}")
            

            if len(self._cb_train_data) > 10000:
                self._cb_train_data.pop(0)
            self._cb_train_data.append(
                (
                    x_combined.cpu().numpy().reshape(x_combined.shape[0], -1),
                    regression_output.detach().cpu().numpy(),
                    y.cpu().numpy(),
                )
            )
            
        except Exception as e:

            if not hasattr(self, 'last_failed_trace') or self.last_failed_trace.step_id != tracer.step_id:
                tracer.add_step("TRAINING_STEP", "UNHANDLED_ERROR", details=f"{type(e).__name__}: {e}")
                self.last_failed_trace = tracer.get_trace()
            logger.critical(f"TRACE GAGAL: {self.last_failed_trace}")
            raise e


        self.last_successful_trace = tracer.get_trace()
        return total_loss

    def on_train_epoch_end(self):
        for param_name, norms in self.grad_norm_history.items():
            avg_norm = sum(norms) / len(norms)
            logger.info(f"[GradNorm] {param_name}: rata-rata {avg_norm:.4e}")
        self.grad_norm_history.clear()


    def validation_step(self, batch, batch_idx):
        """
        Proses validasi untuk model AlphaLit.
        - Menghitung loss gabungan (regression, anomaly detection, attribution).
        - Logging metrik validasi: loss, MSE, RMSE, DA, Sharpe Ratio.
        """


        x_raw, X, y, y_historical, anomaly_signals, edge_index, unc_targets, x_spikes = batch
        if X is None:
            return None


        forward_kwargs = {}




        (
            regression_output, anomaly_output, unc_attribution_pred, 
            _, _, _
        ) = self(
            x_raw, X, y_historical, edge_index, x_spikes, **forward_kwargs
        )


        regression_loss = self.loss_fn(regression_output, y, anomaly_signals)
        anomaly_loss = F.binary_cross_entropy_with_logits(anomaly_output, anomaly_signals)


        unc_attribution_pred_clamped = torch.clamp(unc_attribution_pred, min=1e-8)
        attribution_loss = F.kl_div(
            unc_attribution_pred_clamped.log(), unc_targets, reduction="batchmean"
        )


        alpha, beta = 0.2, 0.15
        loss = (
            (1 - alpha - beta) * regression_loss
            + alpha * anomaly_loss
            + beta * attribution_loss
        )


        mse = self.val_mse(regression_output, y)
        rmse = self.val_rmse(regression_output, y)
        da = self.val_da(regression_output, y)


        predicted_daily_returns = regression_output[:, 0, :].mean(dim=1)
        sharpe_ratio = self._calculate_sharpe_ratio(predicted_daily_returns)


        self.log_dict(
            {
                "val_loss": loss, "val_mse": mse, "val_rmse": rmse,
                "val_da": da, "val_sharpe": sharpe_ratio,
                "val_reg_loss": regression_loss, "val_ano_loss": anomaly_loss,
                "val_attr_loss": attribution_loss,
            },
            prog_bar=True, on_step=False, on_epoch=True,
        )


        self.performance_history.append(
            {"loss": loss.item(), "mse": mse.item(), "rmse": rmse.item()}
        )

        return {"loss": loss, "mse": mse, "rmse": rmse}


    def predict(
        self, x_window, y_historical, edge_index, x_spikes_window, anomaly_signal_value
    ):
        self.eval()
        with torch.no_grad():

            outputs = self.forward(
                x_window, y_historical, edge_index, x_spikes_window
            )
            if len(outputs) < 4:
                raise ValueError(f"Expected at least 4 outputs from forward(), got {len(outputs)}")
            
            mean_pred_returns, _, unc_attribution, _ = outputs[:4]
            
            duality_weight = self.core.last_duality_weight.cpu()
            logvar = self.core.logvar
            if logvar is not None:
                std_from_vib = torch.exp(0.5 * logvar)
                base_std = (
                    std_from_vib.mean().item()
                    * torch.ones_like(mean_pred_returns)
                    * 0.5
                )
            else:
                base_std = torch.ones_like(mean_pred_returns) * 0.05
                
            confidence_penalty = 1.0 + (3.0 * anomaly_signal_value)
            adjusted_std_preds = base_std * confidence_penalty
            
        return (
            mean_pred_returns.cpu(),
            adjusted_std_preds.cpu(),
            unc_attribution.cpu(),
            duality_weight,
        )


    def on_train_end(self):
        logging.info("\nPelatihan Selesai...")
        if self.hparams.mode != "fine-tune" or not self._cb_train_data:
            return
            
        logging.info("Melatih model residual (Mapie)...")
        X_train_all = np.concatenate([d[0] for d in self._cb_train_data])
        core_preds_all = np.concatenate([d[1] for d in self._cb_train_data])
        y_true_all = np.concatenate([d[2] for d in self._cb_train_data])
        
        self.mapie_models = []
        n_splits = 3
        if len(X_train_all) <= n_splits:
            logging.warning(
                f"Data ({len(X_train_all)}) tidak cukup untuk TimeSeriesSplit. Model Mapie tidak dilatih."
            )
            return
            
        for h in range(self.hparams.horizon):
            for i in range(self.hparams.n_targets):
                logging.info(f"Residual Horizon {h+1}, Target {i+1}...")
                y_true_hi = y_true_all[:, h, i]
                preds_hi = core_preds_all[:, h, i]
                residuals = y_true_hi - preds_hi
                
                base_model = CatBoostRegressor(
                    iterations=100, verbose=False, loss_function="RMSE"
                )
                mapie_model = MapieRegressor(
                    base_model, method="plus", cv=TimeSeriesSplit(n_splits=n_splits)
                )
                mapie_model.fit(X_train_all, residuals)
                self.mapie_models.append(mapie_model)
                
        logging.info("Pelatihan model residual (Mapie) selesai.")
        
        if self.hparams.project_id:

            save_path = get_path(self.hparams.project_id, "mapie_models")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(self.mapie_models, save_path)
            logging.info(f"Model Mapie disimpan di: {save_path}")
            
        self._cb_train_data.clear()


    def configure_optimizers(self):
        """
        [VERSI FINAL] Mengonfigurasi optimizer SAM dengan scheduler yang benar.
        """

        optimizer = SAM(
            self.parameters(),
            torch.optim.AdamW,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        logger.info("⚡️ Mengonfigurasi optimizer: SAM yang membungkus AdamW.")

        scheduler_type = self.hparams.get("scheduler_type", "OneCycleLR")

        if scheduler_type == "OneCycleLR":
            logger.info("INFO: Menggunakan scheduler OneCycleLR.")
            try:

                total_steps = self.trainer.estimated_stepping_batches
            except AttributeError:
                total_steps = 100000


            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr,
                total_steps=total_steps,
                cycle_momentum=False,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=25,
                final_div_factor=1e4,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        
        elif scheduler_type == "ReduceLROnPlateau":
            logger.info("INFO: Menggunakan scheduler ReduceLROnPlateau.")
            

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min", patience=5, factor=0.5, verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        

        return optimizer


class HParamsFeaturizer:
    """
    Mengubah kamus hyperparameter yang kompleks menjadi vektor fitur numerik
    yang dapat diproses oleh model machine learning.
    """

    def __init__(self):
        self.feature_mapping = {}
        self.scalers = {}

    def fit(self, hparams_list: list[dict]):
        """Mempelajari struktur data dari daftar hyperparameter."""
        df = pd.DataFrame(hparams_list)


        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()


        feature_idx = 0
        for col in numeric_cols:
            self.feature_mapping[col] = feature_idx
            feature_idx += 1

            scaler = StandardScaler()
            self.scalers[col] = scaler.fit(df[[col]].fillna(0))

        _affmap = build_affinity_map(total_cores=None)
        _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)


        for col in categorical_cols:
            unique_vals = df[col].unique()
            for val in unique_vals:
                self.feature_mapping[f"{col}_{val}"] = feature_idx
                feature_idx += 1

        self.num_features = feature_idx
        logger.info(
            f"[Featurizer] Dipelajari dari data. Total fitur: {self.num_features}"
        )

    def transform(self, hparams: dict) -> np.ndarray:
        """Mengubah satu kamus hyperparameter menjadi vektor fitur."""
        if not self.feature_mapping:
            raise RuntimeError("Featurizer harus di-'fit' terlebih dahulu.")

        vector = np.zeros(self.num_features)
        for key, value in hparams.items():
            if isinstance(value, (int, float)):
                if key in self.scalers:
                    scaled_val = self.scalers[key].transform([[value]])[0, 0]
                    vector[self.feature_mapping[key]] = scaled_val
            else:
                feature_name = f"{key}_{value}"
                if feature_name in self.feature_mapping:
                    vector[self.feature_mapping[feature_name]] = 1.0
        return vector


class RewardDataModule(pl.LightningDataModule):
    """Memuat data dari database umpan balik manusia untuk melatih RewardModel."""

    def __init__(self, db_path: Path, batch_size: int = 8):
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.featurizer = HParamsFeaturizer()

    def setup(self, stage: Optional[str] = None):
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database umpan balik tidak ditemukan di: {self.db_path}"
            )

        with closing(sqlite3.connect(self.db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT hparams_used, rating FROM feedback", conn)

        if len(df) < 10:
            raise ValueError(
                f"Tidak cukup data umpan balik ({len(df)} baris). Diperlukan minimal 10."
            )

        df["hparams_dict"] = df["hparams_used"].apply(lambda x: json.loads(x))


        self.featurizer.fit(df["hparams_dict"].tolist())


        features = np.array([self.featurizer.transform(h)
                            for h in df["hparams_dict"]])

        ratings = (df["rating"].values + 1) / 2.0

        X_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(ratings, dtype=torch.float32).unsqueeze(1)


        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )

    def train_dataloader(self):
        dataset = TensorDataset(self.X_train, self.y_train)
        _affmap = build_affinity_map(total_cores=None)
        _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4,
            persistent_workers=True,
            worker_init_fn=_worker_init,
            prefetch_factor=4,
        )

    def val_dataloader(self):
        dataset = TensorDataset(self.X_val, self.y_val)
        _affmap = build_affinity_map(total_cores=None)
        _worker_init = partial(_affined_worker_init_win32, affinity_map=_affmap, omp_threads=1, mkl_threads=1)
        return TorchDataLoader(dataset, batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            worker_init_fn=_worker_init,
            prefetch_factor=4,
        )



class RewardModel(pl.LightningModule):
    """Model MLP sederhana untuk memprediksi skor reward dari vektor hyperparameter."""

    def __init__(self, input_dim: int, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_reward_loss", loss)
        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_reward_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



def train_reward_model(
    project_id: str,
) -> Optional[tuple[RewardModel, HParamsFeaturizer]]:
    """
    Mengorkestrasi pelatihan Reward Model dari database umpan balik.
    """
    logger.info("\n--- MEMULAI PELATIHAN REWARD MODEL (SUPERVISOR OTOMATIS) ---")
    feedback_db_path = get_path(None, "human_feedback_db")

    try:

        reward_dm = RewardDataModule(db_path=feedback_db_path)
        reward_dm.setup()


        model = RewardModel(input_dim=reward_dm.featurizer.num_features)


        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="auto",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        trainer.fit(model, reward_dm)


        reward_model_path = (
            get_path(project_id, "checkpoint_dir") /
            f"reward_model_{project_id}.ckpt"
        )
        featurizer_path = (
            get_path(project_id, "checkpoint_dir")
            / f"reward_featurizer_{project_id}.pkl"
        )

        trainer.save_checkpoint(reward_model_path)
        with open(featurizer_path, "wb") as f:
            joblib.dump(reward_dm.featurizer, f)

        logger.info(
            f"✅ Reward Model dan Featurizer berhasil dilatih dan disimpan.")
        return model.eval(), reward_dm.featurizer

    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"  [Reward Model] Pelatihan dilewati: {e}")
        return None
    except Exception as e:
        logger.error(
            f"  [Reward Model] Terjadi error tak terduga saat pelatihan: {e}",
            exc_info=True,
        )
        return None


class PolicyAgent(nn.Module):
    """
    Agen kebijakan sederhana yang belajar untuk memetakan
    keadaan laten (dari mimpi) ke sebuah aksi (prediksi return).
    """

    def __init__(self, latent_dim=32, action_dim=1, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, action_dim)
        )
        logger.info(
            f"🤖 Policy Agent diinisialisasi untuk belajar di dalam 'mimpi'.")

    def forward(self, z):
        return self.network(z)


class ContextProjector(nn.Module):
    """
    Sebuah proyektor neural sederhana untuk memetakan embedding pengetahuan
    berdimensi tinggi (dari Brain) ke ruang laten berdimensi rendah (dari VAE).
    """

    def __init__(self, embedding_dim: int, latent_dim: int):
        super().__init__()
        self.projection_layer = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, knowledge_embedding: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(knowledge_embedding)


def _generate_contextual_dream_conditioner(
    brain: "Brain",
    embedding_model: "APIEmbedder",
    projector: ContextProjector,
    device: torch.device
) -> Optional[torch.Tensor]:
    """
    Bertanya ke Brain tentang konteks dunia nyata dan mengubahnya menjadi
    sebuah 'conditioning vector' untuk memandu mimpi AI.
    """
    try:

        query = "What is the single most important upcoming macroeconomic event or prevailing market sentiment that could affect asset prices in the near future?"
        relevant_knowledge = brain.query(query, k=1)

        if not relevant_knowledge:
            return None

        knowledge_text = relevant_knowledge[0]
        logger.info(f"  -> 🧠 Jangkar Realitas Ditemukan: '{knowledge_text}'")


        with torch.no_grad():
            knowledge_embedding = embedding_model.encode(
                knowledge_text, task_type="query")
            knowledge_tensor = torch.from_numpy(
                knowledge_embedding).to(device).float()


            conditioning_vector = projector(knowledge_tensor)

            return conditioning_vector.unsqueeze(0).unsqueeze(0)

    except Exception as e:
        logger.warning(
            f"[Dream Conditioner] Gagal membuat jangkar realitas: {e}")
        return None


def run_dreamer_and_surprise_cycle(
    project_id: str,
    initial_hparams: dict,
    brain: "Brain",
    nsmm: "NSMM",
    _supremacy_embed_model: "APIEmbedder",
    **kwargs
):
    """
    Menjalankan siklus pembelajaran mandiri:
    1. Melatih Policy Agent di dalam 'mimpi' yang dihasilkan World Model.
    2. Menghitung 'kejutan' dengan membandingkan prediksi World Model dengan kenyataan.
    """
    logger.info("\n" + "="*80)
    logger.info(
        "=== 🌌🤖 MEMULAI SIKLUS PEMBELAJARAN & KEJUTAN (DREAMER & SURPRISE) 🤖🌌 ===")
    logger.info("="*80)

    try:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


        logger.info(
            "  -> Tahap 1/4: Memuat komponen World Model yang sudah ada...")
        vae_path = get_path(project_id, "checkpoint_dir") /            f"world_model_vae.pth"
        mdn_path = get_path(project_id, "checkpoint_dir") /            f"world_model_mdn_rnn.pth"

        dm = AlphaDataModule(initial_hparams, None,
                             None, None, None, None, brain)
        dm.setup(stage="fit")

        vae = VAE(input_dim=dm.n_features_input,
                  latent_dim=initial_hparams["wm_latent_dim"]).to(device)
        mdn_rnn = MDN_RNN(
            latent_dim=initial_hparams["wm_latent_dim"]).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        mdn_rnn.load_state_dict(torch.load(mdn_path, map_location=device))
        vae.eval()
        mdn_rnn.eval()

        policy_agent = PolicyAgent(
            latent_dim=initial_hparams["wm_latent_dim"]).to(device)
        policy_optimizer = torch.optim.Adam(
            policy_agent.parameters(), lr=1e-4)
        context_projector = ContextProjector(
            embedding_dim=_supremacy_embed_model.dim,
            latent_dim=initial_hparams["wm_latent_dim"]
        ).to(device)


        logger.info(
            "  -> Tahap 2/4: Melatih Policy Agent di dalam simulasi 'mimpi'...")
        with torch.no_grad():
            last_real_data = torch.tensor(
                dm.X_val[-dm.hparams.window:], dtype=torch.float32).to(device)
            mu, logvar = vae.encode(
                last_real_data.view(-1, dm.n_features_input))
            z_current = vae.reparameterize(
                mu, logvar).view(1, dm.hparams.window, -1)

        for i in range(50):
            policy_agent.train()
            policy_optimizer.zero_grad()
            with torch.no_grad():
                conditioning_vector = _generate_contextual_dream_conditioner(
                    brain, _supremacy_embed_model, context_projector, device
                )


            if conditioning_vector is not None:

                z_conditioned = z_current + conditioning_vector
                pi, sigma, mu_rnn = mdn_rnn(z_conditioned)
            else:

                pi, sigma, mu_rnn = mdn_rnn(z_current)
            z_dreamed_next = sample_from_gmm(pi, sigma, mu_rnn)


            predicted_reward = policy_agent(z_current[:, -1, :])


            loss = -predicted_reward.mean()
            loss.backward()
            policy_optimizer.step()


            z_current = torch.cat(
                [z_current[:, 1:, :], z_dreamed_next.unsqueeze(1)], dim=1)

        policy_agent_path = get_path(
            project_id, "checkpoint_dir") / f"policy_agent.pth"
        torch.save(policy_agent.state_dict(), policy_agent_path)
        logger.info(
            f"  -> Policy Agent yang dilatih dalam mimpi disimpan di: {policy_agent_path.name}")

        logger.info(
            "  -> Meng-encode data validasi untuk mendapatkan sekuens laten...")
        with torch.no_grad():
            val_data_tensor = torch.tensor(
                dm.X_val, dtype=torch.float32).to(device)
            mu_val, logvar_val = vae.encode(val_data_tensor)
            z_sequence = vae.reparameterize(mu_val, logvar_val)


        logger.info(
            "  -> Tahap 3/4: Mengukur 'kejutan' antara imajinasi dan realitas...")
        with torch.no_grad():


            z_real_t = z_sequence[-2].unsqueeze(0).unsqueeze(0)
            z_real_t1 = z_sequence[-1]


            pi_pred, sigma_pred, mu_pred = mdn_rnn(z_real_t)
            z_predicted_t1 = sample_from_gmm(pi_pred, sigma_pred, mu_pred)


            surprise_score = F.mse_loss(z_predicted_t1, z_real_t1).item()
            logger.warning(
                f"  -> 🤯 Skor Kejutan (Surprise Score): {surprise_score:.4f}")


        logger.info("  -> Tahap 4/4: Menindaklanjuti sinyal kejutan...")
        if surprise_score > 0.5:
            logger.critical(
                "  -> KEJUTAN TINGGI TERDETEKSI! Dunia telah berubah dari ekspektasi model.")
            nsmm.log_user_activity(
                f"SURPRISE_DETECTED: Skor {surprise_score:.4f}. Model dunia mungkin usang.")



        else:
            logger.info(
                "  -> Tingkat kejutan rendah. Model dunia masih selaras dengan realitas.")

    except FileNotFoundError:
        logger.error(
            "  -> GAGAL: File World Model tidak ditemukan. Jalankan 'run_world_model_training' terlebih dahulu.")
    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Dreamer & Surprise: {e}", exc_info=True)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.encoder_fc1 = nn.Linear(input_dim, 256)
        self.encoder_fc2_mu = nn.Linear(256, latent_dim)
        self.encoder_fc2_logvar = nn.Linear(256, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, 256)
        self.decoder_fc2 = nn.Linear(256, input_dim)
        logger.info(
            f"VAE World Model diinisialisasi: Input Dim={input_dim}, Latent Dim={latent_dim}"
        )

    def encode(self, x):
        h = F.silu(self.encoder_fc1(x))
        return self.encoder_fc2_mu(h), self.encoder_fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.silu(self.decoder_fc1(z))
        return torch.tanh(self.decoder_fc2(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def run_lite_reactor_cycle(
    vae_model: VAE,
    original_data: np.ndarray,
    num_samples_to_generate: int = 1000,
    realism_threshold: float = 2.5,
) -> Optional[np.ndarray]:
    """
    Menjalankan satu siklus pembangkitan data sintetis menggunakan VAE (versi lite).
    """
    logger.info(
        "☢️  [Reaktor Lite] Mengaktifkan siklus pembangkitan data sintetis berbasis VAE..."
    )

    if vae_model is None:
        logger.error(
            "[Reaktor Lite] VAE model tidak tersedia. Siklus dibatalkan.")
        return None

    device = next(vae_model.parameters()).device
    vae_model.eval()

    with torch.no_grad():

        latent_dim = vae_model.encoder_fc2_mu.out_features
        z_samples = torch.randn(num_samples_to_generate, latent_dim).to(device)


        synthetic_data_tensor = vae_model.decode(z_samples)
        synthetic_data = synthetic_data_tensor.cpu().numpy()

    logger.info(
        f"  -> Berhasil menghasilkan {num_samples_to_generate} sampel data sintetis mentah."
    )


    logger.info("  -> Memulai validasi realisme data sintetis...")
    original_data_tensor = torch.from_numpy(original_data).to(device)


    distances = torch.cdist(synthetic_data_tensor, original_data_tensor)
    min_distances, _ = torch.min(distances, dim=1)



    realistic_mask = min_distances < realism_threshold
    validated_data = synthetic_data[realistic_mask.cpu().numpy()]

    num_validated = len(validated_data)
    if num_validated == 0:
        logger.warning(
            "[Reaktor Lite] Tidak ada data sintetis yang lolos validasi. Mungkin threshold terlalu ketat."
        )
        return None

    logger.info(
        f"  -> ✅ Validasi selesai. {num_validated}/{num_samples_to_generate} sampel data diterima."
    )
    return validated_data



class ConcreteDropout(nn.Module):
    def __init__(self, dropout_p=0.1, lengthscale=1e-2, reg_weight=1e-6):
        super().__init__()
        self.dropout_p = dropout_p
        self.lengthscale = lengthscale
        self.reg_weight = reg_weight
        self.p_logit = nn.Parameter(torch.zeros(1))

    def forward(self, x, training=True):
        if not training:
            return nn.Dropout(p=self.dropout_p)(x)

        def concrete_dropout(p, x):
            eps = 1e-7
            temp = 0.1
            unif_noise = torch.rand_like(x)
            drop_prob = (
                torch.log(p + eps)
                - torch.log(1 - p + eps)
                + torch.log(unif_noise + eps)
                - torch.log(1 - unif_noise + eps)
            )
            drop_prob = torch.sigmoid(drop_prob / temp)
            random_tensor = 1 - drop_prob
            x_retained = x * random_tensor
            return x_retained / (1 - p)

        p = torch.sigmoid(self.p_logit)
        out = concrete_dropout(p, x)
        reg = self.reg_weight *            (self.lengthscale**2 * p + p * (1 - p) / x.size(0))
        self.log_regularization = reg.sum()
        return out


class TimestepEmbedding(nn.Module):
    """
    Mengubah timestep (integer) menjadi embedding vektor yang kaya informasi
    menggunakan frekuensi sinusoidal, untuk digunakan oleh Model Difusi.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class GraphCommunicationNetwork(nn.Module):
    """
    Implementasi Radio Gema Kritis.
    GNN kecil yang memproses sinyal 'panas' (instabilitas) dari berbagai komponen model
    dan menghasilkan satu sinyal kewaspadaan tunggal.
    """

    def __init__(self, num_nodes: int, node_feature_dim: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.conv1 = PyG_GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = PyG_GCNConv(hidden_dim, hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * num_nodes, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, heat_signals: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:

        batch_size = heat_signals.shape[0]


        x = F.relu(self.conv1(heat_signals, edge_index))
        x = F.relu(self.conv2(x, edge_index))


        x_flat = x.view(batch_size, -1)


        return self.output_head(x_flat)


def train_gnn_communicator(
    project_id: str, n_layers: int, device: torch.device, epochs: int = 50
) -> GraphCommunicationNetwork:
    """
    Melatih GNN Communicator (Radio Gema) secara terpisah dengan data yang disimulasikan.
    """
    shield_path = (
        get_path(project_id, "checkpoint_dir") /
        f"gnn_communicator_{project_id}.pth"
    )
    if shield_path.exists():
        logger.info(f"Memuat Radio Gema yang sudah ada dari: {shield_path}")
        model = GraphCommunicationNetwork(num_nodes=n_layers + 1).to(device)
        model.load_state_dict(torch.load(shield_path, map_location=device))
        return model.eval()

    logger.info("\n--- 📻 Memulai Pelatihan Radio Gema (GNN Communicator) ---")
    num_nodes = n_layers + 1
    model = GraphCommunicationNetwork(num_nodes=num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()



    simulated_heat_signals = torch.rand(
        500, num_nodes, 1
    )
    simulated_alert_levels = torch.mean(
        simulated_heat_signals, dim=1
    )

    sim_dataset = TensorDataset(simulated_heat_signals, simulated_alert_levels)
    sim_loader = TorchDataLoader(sim_dataset, batch_size=32, shuffle=True)

    edge_index = (
        torch.tensor(
            list(itertools.permutations(range(num_nodes), 2)), dtype=torch.long
        )
        .t()
        .contiguous()
        .to(device)
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for heat, alert in sim_loader:
            heat, alert = heat.to(device), alert.to(device)
            optimizer.zero_grad()
            predicted_alert = model(heat, edge_index)
            loss = loss_fn(predicted_alert, alert)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"  > GNN Comm Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(sim_loader):.6f}"
            )

    torch.save(model.state_dict(), shield_path)
    logger.info(f"✅ Radio Gema (GNN Communicator) tersimpan di: {shield_path}")
    return model.eval()



class EnhancedAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SpatioTemporalAttention(nn.Module):
    """
    Menghitung attention secara terpisah untuk dimensi spasial (antar fitur)
    dan temporal (sepanjang waktu), lalu menggabungkannya.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Inisialisasi komponen-komponen layer.

        Args:
            d_model (int): Dimensi model (jumlah fitur input).
            n_heads (int): Jumlah attention heads.
            dropout (float): Tingkat dropout yang akan digunakan.
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        

        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_temporal: torch.Tensor, x_spatial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mendefinisikan alur maju (forward pass) dari layer.

        Args:
            x_temporal (torch.Tensor): Tensor input untuk temporal attention 
                                       dengan shape [Batch, Time_Steps, Features].
            x_spatial (torch.Tensor): Tensor input untuk spatial attention 
                                      dengan shape [Batch, Features, Time_Steps].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output dari temporal dan spatial attention
                                               dengan shape yang konsisten [Batch, Time_Steps, Features].
        """


        temporal_out, _ = self.temporal_attention(
            query=x_temporal, key=x_temporal, value=x_temporal
        )





        x_spatial_for_attn = x_spatial.transpose(1, 2)
        



        spatial_out, _ = self.spatial_attention(
            query=x_spatial_for_attn, key=x_spatial_for_attn, value=x_spatial_for_attn
        )
        


        return self.dropout(temporal_out), self.dropout(spatial_out)



class SpatioTemporalGatedFusion(nn.Module):
    """
    Menggabungkan output dari attention spasial dan temporal menggunakan mekanisme gerbang (gate)
    yang dapat dipelajari, memungkinkan model memutuskan informasi mana yang lebih penting.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.gate_layer = nn.Linear(d_model * 2, d_model)

    def forward(self, temporal_features, spatial_features_permuted):
        combined_features = torch.cat(
            [temporal_features, spatial_features_permuted], dim=-1
        )
        gate_values = torch.sigmoid(self.gate_layer(combined_features))
        fused_features = (gate_values * temporal_features) + (
            (1 - gate_values) * spatial_features_permuted
        )
        return fused_features


class PIDController(nn.Module):
    """
    Mengimplementasikan Proportional-Integral-Derivative (PID) controller sebagai
    sebuah nn.Module yang stateful untuk menstabilkan output layer.
    """
    def __init__(self, kp: float, ki: float, kd: float):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.register_buffer("integral", torch.zeros(1))
        self.register_buffer("prev_error", torch.zeros(1))

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        """
        Menghitung sinyal koreksi PID berdasarkan error yang diberikan.
        State internal akan direset secara dinamis jika ukuran batch berubah.
        """


        if self.integral.shape != error.shape or self.prev_error.shape != error.shape:
            logger.debug(f"[PID] 🔁 Resizing state buffers to match error shape: {error.shape}")

            self.integral = torch.zeros_like(error)
            self.prev_error = torch.zeros_like(error)


        p_term = self.kp * error



        self.integral += error.detach()
        i_term = self.ki * self.integral


        derivative = error - self.prev_error
        d_term = self.kd * derivative


        self.prev_error.copy_(error.detach())

        return p_term + i_term + d_term

class FasciaNetwork(nn.Module):
    """
    Jaringan saraf dinamis yang berfungsi sebagai 'fascia', menghubungkan dan
    menyeimbangkan berbagai komponen sistem dengan menghasilkan bobot loss adaptif.
    """

    def __init__(self, state_vector_dim: int, num_losses: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_vector_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, num_losses),
            nn.Softmax(dim=-1),
        )
        logger.info(
            f"🕸️ Jaringan Fascia diinisialisasi untuk menyeimbangkan {num_losses} komponen loss."
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Input: state_vector [batch, state_dim], Output: loss_weights [batch, num_losses]"""
        return self.network(state_vector)


class KevlarLayer(nn.Module):
    """
    Lapisan pertahanan akhir yang terinspirasi dari Kevlar.
    Menggunakan LTC untuk menyerap 'energi' (sinyal volatil) dan PID Controller
    untuk mengembalikan energi yang diserap sebagai sinyal koreksi.
    """

    def __init__(self, d_model: int, pid_kp: float, pid_ki: float, pid_kd: float):
        super().__init__()

        wiring = ncps.wirings.FullyConnected(d_model)
        self.ltc_absorber = LTC(d_model, wiring, return_sequences=False)
        self.norm_ltc = nn.LayerNorm(d_model)


        self.pid_feedback = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)

        logger.info(f"🛡️ Lapisan KEVLAR diaktifkan (LTC + PID Controller).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        

        x_seq = x.unsqueeze(1)


        ltc_processed_output, _ = self.ltc_absorber(x_seq)


        absorbed_energy = x - ltc_processed_output


        correction_signal = self.pid_feedback(absorbed_energy)


        final_output = x + correction_signal

        return self.norm_ltc(final_output)


class AdaptiveLayerNorm(nn.Module):
    """
    Layer Normalization dengan epsilon adaptif berbasis varians input,
    disesuaikan oleh 'gating network' kecil.
    """

    def __init__(self, normalized_shape, base_epsilon=1e-5, debug_log=False):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.base_epsilon = base_epsilon
        self.debug_log = debug_log

        self.epsilon_gate = nn.Sequential(
            nn.Linear(1, 10), nn.SiLU(), nn.Linear(10, 1), nn.Sigmoid()
        )

        self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x):
        batch_var = torch.var(x, dim=-1, keepdim=True).detach()

        learned_multiplier = self.epsilon_gate(
            torch.mean(batch_var, dim=1, keepdim=True)
        )

        dynamic_epsilon = self.base_epsilon + (learned_multiplier.unsqueeze(-1) * 0.1)
        dynamic_epsilon_scalar = float(dynamic_epsilon.mean())


        if self.debug_log and self.training:
            print(f"[AdaptiveLayerNorm] Epsilon Dinamis: {dynamic_epsilon_scalar:.8f}")

        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, eps=dynamic_epsilon_scalar
        )




class DiffusionDenoisingModel(nn.Module):
    """
    Model berbasis Transformer yang bertugas memprediksi noise yang ditambahkan
    ke data pada timestep 't' tertentu.
    """

    def __init__(self, input_dim: int, d_model: int, n_heads: int, num_layers: int, seq_len: int):
        super().__init__()
        self.d_model = d_model


        self.input_proj = nn.Linear(input_dim, d_model)


        self.time_embedding = TimestepEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )


        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t (torch.Tensor): Data yang sudah diberi noise. Shape: [batch, seq_len, features].
            t (torch.Tensor): Timestep difusi. Shape: [batch].

        Returns:
            torch.Tensor: Prediksi noise. Shape: [batch, seq_len, features].
        """

        x = self.input_proj(x_t)


        time_emb = self.time_mlp(self.time_embedding(t))

        x = x + time_emb.unsqueeze(1)


        predicted_noise = self.transformer_encoder(x)


        predicted_noise = self.output_proj(predicted_noise)

        return predicted_noise


class VariationalEncoder(nn.Module):
    """
    Encoder stokastik yang menghasilkan parameter distribusi (mu, logvar)
    dan menyediakan sampling melalui reparameterization trick.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        """Menghasilkan parameter distribusi dari input."""
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Melakukan reparameterization trick untuk memungkinkan backpropagation
        melalui proses sampling yang acak.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class FNO1dBlock(nn.Module):
    """
    Blok Fourier Neural Operator 1D yang tangguh dan fleksibel untuk data sekuensial.
    Menggunakan domain frekuensi untuk memproses dependensi spasial panjang.
    """

    def __init__(self, d_model, modes, use_layernorm=True):
        super().__init__()
        self.d_model = d_model
        self.modes = modes
        self.use_layernorm = use_layernorm



        self.weights = nn.Parameter(
            torch.randn(d_model, d_model, modes, dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(
            torch.randn(d_model, modes, dtype=torch.cfloat)
        )


        if self.use_layernorm:
            self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor input bentuk [batch_size, seq_len, d_model]
        Returns:
            Tensor output bentuk [batch_size, seq_len, d_model]
        """
        B, L, C = x.shape

        if self.use_layernorm:
            x = self.norm(x)


        x = x.permute(0, 2, 1)


        x_ft = torch.fft.rfft(x, dim=-1)
        L_ft = x_ft.shape[-1]


        out_ft = torch.zeros_like(x_ft)


        used_modes = min(self.modes, L_ft, self.bias.shape[1])
        # Use only a subset of the available modes to reduce computational load
        effective_modes = max(1, used_modes // 2)
        out_ft[:, :, :effective_modes] = (
            torch.einsum("bcm,com->bom", x_ft[:, :, :effective_modes], self.weights[:, :, :effective_modes])
            + self.bias[:, :effective_modes]
        )


        x_out = torch.fft.irfft(out_ft, n=L, dim=-1)


        return x_out.permute(0, 2, 1)


class DPA_STIFormer_Layer(nn.Module):
    """
    Sebuah layer dari DPA-STIFormer yang mengintegrasikan Spatio-Temporal Attention,
    Dynamic Gating Fusion, PID Controller, Mixture-of-Experts FFN, dan Stochastic Depth.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
        num_experts: int = 4,
        top_k_experts: int = 2,
    ):
        """
        Inisialisasi komponen-komponen dalam layer.
        """
        super().__init__()
        d_ff = d_ff or 4 * d_model
        

        self.st_attention = SpatioTemporalAttention(d_model, n_heads, dropout=dropout)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )


        self.pid_controller = PIDController(kp=kp, ki=ki, kd=kd)
        self.moe_ffn = MoELayer(
            input_dim=d_model,
            output_dim=d_model,
            num_experts=num_experts,
            top_k=top_k_experts,
        )


        self.norm1 = AdaptiveLayerNorm(d_model)
        self.norm2 = AdaptiveLayerNorm(d_model)
        self.stochastic_depth = StochasticDepth(p=drop_path_rate, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mendefinisikan alur maju (forward pass) dari layer.

        Args:
            x (torch.Tensor): Tensor input dengan shape [Batch, Time_Steps, Features].
        
        Returns:
            torch.Tensor: Tensor output dengan shape yang sama dengan input.
        """
        residual_input = x




        x_temporal_in = x
        x_spatial_in = x.transpose(1, 2)
        
        temporal_attn_out, spatial_attn_out = self.st_attention(
            x_temporal_in, x_spatial_in
        )

        context_signal = x.mean(dim=1)
        gate_value = self.fusion_gate(context_signal).unsqueeze(-1)

        fused_out = (gate_value * temporal_attn_out) + ((1 - gate_value) * spatial_attn_out)





        error = residual_input - fused_out


        mean_batch_error = torch.mean(error) 


        pid_correction_scalar = self.pid_controller(mean_batch_error)



        x = self.norm1(
            residual_input + self.stochastic_depth(fused_out) + pid_correction_scalar
        )




        ffn_out = self.moe_ffn(x)
        x = self.norm2(x + self.stochastic_depth(ffn_out))
        
        return x

class RenormalizationBlock(nn.Module):
    """
    Memproses input pada beberapa skala waktu (harian, mingguan, bulanan)
    menggunakan prosesor bersama dan menggabungkannya secara hierarkis,
    berdasarkan konsep Renormalization Group.
    """

    def __init__(self, d_model: int, n_kernels: int = 64, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model

        self.pool_scale_meso = nn.AvgPool1d(kernel_size=5, stride=5)
        self.pool_scale_macro = nn.AvgPool1d(kernel_size=20, stride=20)

        self.shared_processor = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=n_kernels,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

        self.upsample_macro_to_meso = nn.Upsample(
            scale_factor=4, mode="linear", align_corners=False
        )
        self.upsample_meso_to_micro = nn.Upsample(
            scale_factor=5, mode="linear", align_corners=False
        )


        self.norm_meso = AdaptiveLayerNorm(d_model)
        self.norm_micro = AdaptiveLayerNorm(d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:


        x_permuted = x.permute(0, 2, 1)


        x_micro = x_permuted
        x_meso = self.pool_scale_meso(x_permuted)
        x_macro = self.pool_scale_macro(x_permuted)


        feat_micro = self.shared_processor(x_micro)
        feat_meso = self.shared_processor(x_meso)
        feat_macro = self.shared_processor(x_macro)



        up_macro = self.upsample_macro_to_meso(feat_macro)
        if up_macro.shape[-1] != feat_meso.shape[-1]:
            up_macro = F.pad(
                up_macro, (0, feat_meso.shape[-1] - up_macro.shape[-1]))
        fused_meso = self.norm_meso((feat_meso + up_macro).permute(0, 2, 1)).permute(
            0, 2, 1
        )


        up_meso = self.upsample_meso_to_micro(fused_meso)
        if up_meso.shape[-1] != feat_micro.shape[-1]:
            up_meso = F.pad(
                up_meso, (0, feat_micro.shape[-1] - up_meso.shape[-1]))
        fused_micro = self.norm_micro((feat_micro + up_meso).permute(0, 2, 1)).permute(
            0, 2, 1
        )



        return fused_micro.mean(dim=2)



class SpikingEventProcessor(nn.Module):
    """
    Jaringan Saraf Spiking sederhana untuk memproses 'spike trains' dari peristiwa.
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        beta = 0.9
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, spike_input):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()


        spk2_rec = []
        for step in range(spike_input.size(1)):
            cur1 = self.fc1(spike_input[:, step, :])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)


        return mem2


class RecurrentSpikingProcessor(nn.Module):
    """
    Mengimplementasikan Recurrent & Deep Spiking Neural Network (RSNN).
    Mampu memproses sekuens spike dan mempelajari pola temporal.
    Juga menyertakan loss proxy untuk STDP untuk mendorong pembelajaran yang efisien.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_p, beta=0.9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.snn_layers = nn.ModuleList()



        layer_input_size = input_size
        for i in range(num_layers):
            self.snn_layers.append(
                nn.Sequential(
                    nn.Linear(layer_input_size, hidden_size),
                    snn.Leaky(beta=beta, init_hidden=True),
                )
            )


            layer_input_size = hidden_size

    def forward(self, spike_input):
        num_steps = spike_input.size(1)

        for layer in self.snn_layers:
            layer[1].reset_hidden()

        hidden_states_sequence = []
        all_spikes_rec_for_loss = []

        for step in range(num_steps):
            current_input = spike_input[:, step, :]
            layer_spikes_for_loss = []


            for layer in self.snn_layers:
                current_input = layer(current_input)
                layer_spikes_for_loss.append(current_input)

            hidden_states_sequence.append(current_input)
            all_spikes_rec_for_loss.append(torch.stack(layer_spikes_for_loss))

        hidden_states_sequence = torch.stack(hidden_states_sequence, dim=1)
        final_hidden_state = hidden_states_sequence[:, -1, :]


        all_spikes_rec_for_loss = torch.stack(all_spikes_rec_for_loss, dim=2)

        stdp_proxy_loss = self.calculate_stdp_proxy_loss(
            all_spikes_rec_for_loss)

        return final_hidden_state, hidden_states_sequence, stdp_proxy_loss

    def calculate_stdp_proxy_loss(self, all_spikes, target_rate=0.02, max_rate=0.05):
        if all_spikes.numel() == 0:
            return torch.tensor(0.0, device=all_spikes.device)

        firing_rates = all_spikes.mean(dim=(0, 2, 3))

        sparsity_loss = F.mse_loss(
            firing_rates, torch.full_like(firing_rates, target_rate)
        )
        activity_loss = torch.mean(F.relu(firing_rates - max_rate))

        return sparsity_loss + activity_loss


class AttentionFusion(nn.Module):
    """
    Menggabungkan embedding DNN dan sekuens state SNN menggunakan atensi.
    Memungkinkan DNN untuk fokus pada bagian relevan dari pemrosesan SNN.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, dnn_embedding, snn_state_sequence):


        query = dnn_embedding.unsqueeze(1)
        key_value = snn_state_sequence


        attended_snn_embedding, _ = self.attention(query, key_value, key_value)
        attended_snn_embedding = attended_snn_embedding.squeeze(
            1)


        combined = torch.cat([dnn_embedding, attended_snn_embedding], dim=1)
        gate_values = torch.sigmoid(self.gate(combined))

        fused_embedding = (gate_values * dnn_embedding) + (
            (1 - gate_values) * attended_snn_embedding
        )

        return self.norm(fused_embedding)


class QuantumDualityController(nn.Module):
    """
    Versi 2.0: Menerima sinyal entropi tambahan untuk membuat keputusan
    gating yang lebih adaptif dan sadar akan volatilitas input.
    """

    def __init__(self, d_model: int, snn_hidden_size: int, hidden_dim: int = 64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model + snn_hidden_size + 1,
                      hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        wave_embedding: torch.Tensor,
        particle_embedding: torch.Tensor,
        entropy_signal: torch.Tensor,
    ) -> torch.Tensor:

        combined_input = torch.cat(
            [wave_embedding, particle_embedding, entropy_signal], dim=-1
        )
        return self.network(combined_input)


class GateGenerator(nn.Module):
    """
    Sebuah modul neural network kecil yang tugasnya menghasilkan satu nilai 'gate'
    (antara 0 dan 1) berdasarkan embedding konteks yang diberikan.
    Nilai ini akan mengontrol seberapa banyak fitur sekunder yang "dibuka".
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, contextual_embedding: torch.Tensor) -> torch.Tensor:
        """
        Input: Embedding dari fitur primer. Shape: [batch_size, d_model]
        Output: Gate value. Shape: [batch_size, 1]
        """
        return self.network(contextual_embedding)



class SidecarMemoryNetwork(nn.Module):
    """
    Jaringan saraf khusus yang berfungsi sebagai memori virtual.
    Tugasnya adalah mengompres fitur-fitur "sisa" menjadi embedding
    yang kaya informasi dan siap untuk diinjeksikan jika diperlukan.
    """

    def __init__(
        self, leftover_feature_dim: int, output_dim: int, hidden_dim_ratio: int = 2
    ):
        super().__init__()
        hidden_dim = max(output_dim, leftover_feature_dim // hidden_dim_ratio)
        self.network = nn.Sequential(
            nn.Linear(leftover_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        logger.info(
            f"🧠 Sidecar Memory Network diinisialisasi untuk {leftover_feature_dim} fitur sisa -> {output_dim} embedding."
        )

    def forward(self, x_leftover: torch.Tensor) -> torch.Tensor:
        """Memproses fitur sisa menjadi embedding bantuan."""
        return self.network(x_leftover)


class DynamicAdaptiveMultiScaleExtractor(nn.Module):
    """
    Blok Taishin v2.0: Diperkaya dengan pooling adaptif, berbasis perhatian,
    dan fitur Fourier untuk analisis multi-skala yang superior.
    """

    def __init__(self, d_model: int, input_feature_dim: int, output_seq_len: int = 60):
        super().__init__()
        self.d_model = d_model
        self.output_seq_len = output_seq_len


        self.input_projection = nn.Linear(input_feature_dim, d_model)


        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=output_seq_len)


        self.attention_mlp = nn.Sequential(
            nn.Linear(d_model, d_model //
                      2), nn.Tanh(), nn.Linear(d_model // 2, 1)
        )


        self.num_fourier_features = 16
        self.fourier_proj = nn.Linear(self.num_fourier_features * 2, d_model)


        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=7, padding=3, groups=d_model
        )
        self.pointwise_conv = nn.Conv1d(d_model, d_model, kernel_size=1)


        self.fusion_conv = nn.Conv1d(
            in_channels=d_model * 4, out_channels=d_model, kernel_size=1
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:




        x_proj = self.input_projection(x)


        out_adaptive = self.adaptive_pool(
            x_proj.permute(0, 2, 1)).permute(0, 2, 1)


        attention_scores = F.softmax(self.attention_mlp(x_proj), dim=1)
        out_attention = torch.sum(x_proj * attention_scores, dim=1)
        out_attention = out_attention.unsqueeze(
            1).repeat(1, self.output_seq_len, 1)


        x_ft = torch.fft.rfft(x_proj, dim=1)
        top_k_ft = x_ft[:, 1: self.num_fourier_features + 1]
        fourier_features = torch.cat([top_k_ft.real, top_k_ft.imag], dim=-1)
        avg_fourier_features = fourier_features.mean(dim=1)
        out_fourier = self.fourier_proj(avg_fourier_features)
        out_fourier = out_fourier.unsqueeze(
            1).repeat(1, self.output_seq_len, 1)


        out_conv = self.pointwise_conv(
            self.depthwise_conv(x_proj.permute(0, 2, 1)))
        out_conv_adapted = nn.AdaptiveAvgPool1d(self.output_seq_len)(out_conv).permute(
            0, 2, 1
        )


        combined = torch.cat(
            [out_adaptive, out_attention, out_fourier, out_conv_adapted], dim=2
        )
        fused = self.fusion_conv(combined.permute(0, 2, 1))

        final_output = self.activation(fused.permute(0, 2, 1))

        return self.final_norm(final_output)


class Evolved_Hybrid_SNN_Model(nn.Module):
    """
    Versi 6.2 (Final): Diperkuat dengan pilar pertahanan hibrida, Perisai Singularitas v3.1,
    optimisasi memori untuk Positional Encoding, dan sistem pelacakan diagnostik.
    """

    def __init__(
        self,
        feature_names: list,
        n_features_input,
        n_targets,
        d_model,
        n_layers,
        n_heads,
        dropout,
        window,
        horizon,
        n_uncertainty_factors,
        snn_input_size,
        snn_hidden_size,
        snn_num_layers,
        snn_dropout,
        fno_modes,
        tgw_rank: int = 8,
        pid_kp: float = 0.1,
        pid_ki: float = 0.01,
        pid_kd: float = 0.05,
        denoising_model: nn.Module = None,
        gnn_communicator: nn.Module = None,
        **kwargs,
    ):
        super().__init__()

        # ---- Checkpoint Compatibility Patch ----
        # Ensure d_model is at least 128 to maintain compatibility with
        # pretrained weights that expect a 128-dimensional latent space.
        # If a smaller d_model is provided, promote it to 128.  This change
        # happens before any parameters are constructed.
        if d_model < 128:
            logger.info(
                f"[Patch] Menyesuaikan d_model dari {d_model} ke 128 untuk kompatibilitas checkpoint."
            )
            d_model = 128


        local_vars = {k: v for k, v in locals().items() if k not in ["self", "__class__"]}
        self.hparams = SimpleNamespace(**local_vars)


        self.horizon = horizon
        self.n_targets = n_targets
        self.d_model = d_model


        self.stdp_proxy_loss = torch.tensor(0.0)
        self.mu = None
        self.logvar = None
        self.last_duality_weight = torch.tensor(0.5)


        self.rule_bank = nn.ModuleList()
        self.symbolic_fusion_head = None


        self.qtc = QuantumThalamicCore(
            input_dim=n_features_input, codebook_dim=d_model
        )
        self.tgw_rank = tgw_rank
        self.wormhole_matrix_W = nn.Parameter(
            torch.randn(d_model, self.tgw_rank))
        self.wormhole_matrix_V = nn.Parameter(
            torch.randn(self.tgw_rank, d_model))
        self.gnn_conv = GCNConv(in_channels=window, out_channels=window)
        self.variational_encoder = VariationalEncoder(
            input_dim=d_model, latent_dim=d_model
        )
        


        self.dpa_stif_layers = nn.ModuleList(
            [
                DPA_STIFormer_Layer(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    drop_path_rate=kwargs.get("drop_path_rate", 0.1),
                    kp=pid_kp,
                    ki=pid_ki,
                    kd=pid_kd,
                    num_experts=4,
                    top_k_experts=2,
                )
                for _ in range(n_layers)
            ]
        )


        self.student_head = nn.Linear(d_model, horizon * n_targets)
        logger.info("🎓 Self-Distillation diaktifkan dengan 1 student head.")


        self.fno_layer = FNO1dBlock(d_model=d_model, modes=fno_modes)
        self.norm_fno = AdaptiveLayerNorm(d_model)
        self.snn_processor = RecurrentSpikingProcessor(
            input_size=snn_input_size,
            hidden_size=snn_hidden_size,
            num_layers=snn_num_layers,
            dropout_p=snn_dropout,
        )
        self.snn_output_projection = nn.Linear(snn_hidden_size, d_model)
        self.duality_controller = QuantumDualityController(d_model, d_model)


        self.menshin_core = Menshin_Skytree_Core(
            d_model, n_layers=2, n_heads=n_heads, dropout=dropout
        )
        self.menshin_damper = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.menshin_fusion_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


        self.shield_pre_vae = AdaptiveStabilityLayer(feature_dim=d_model, adaptive_gate=True)
        logger.info("🛡️  Perisai Singularitas (pre-VAE) diaktifkan.")


        self.kevlar_layer = KevlarLayer(d_model, pid_kp, pid_ki, pid_kd)

        final_embedding_dim_for_heads = d_model + self.n_targets

        self.shield_pre_heads = AdaptiveStabilityLayer(feature_dim=final_embedding_dim_for_heads, adaptive_gate=False)
        logger.info("🛡️  Perisai Singularitas (pre-Heads) diaktifkan.")


        # Insert a LayerNorm before the regression head in the fine‑tuning
        # architecture to normalise features before prediction.  This helps
        # prevent scale drift across batches and improves stability of the
        # output layer.
        self.regression_head = nn.Sequential(
            nn.LayerNorm(final_embedding_dim_for_heads),
            nn.Linear(final_embedding_dim_for_heads, horizon * n_targets)
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(final_embedding_dim_for_heads, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.uncertainty_attribution_head = nn.Sequential(
            nn.Linear(final_embedding_dim_for_heads, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, n_uncertainty_factors),
            nn.Softmax(dim=-1),
        )


        self.denoising_model = denoising_model
        self.gnn_communicator = gnn_communicator
        num_comm_nodes = n_layers + 1
        comm_edges = (
            torch.tensor(
                list(itertools.permutations(range(num_comm_nodes), 2)), dtype=torch.long
            )
            .t()
            .contiguous()
        )
        self.register_buffer("comm_edge_index", comm_edges)
        self.logic_consistency_head = nn.Linear(d_model, 5)


        feature_map = {name: i for i, name in enumerate(feature_names)}
        self.dynamically_reconstruct_rules(feature_map)

    def dynamically_reconstruct_rules(self, feature_map: dict):
        """
        Memindai direktori 'rules', mengimpor aturan baru secara dinamis,
        dan menambahkannya ke dalam arsitektur model sebelum pelatihan dimulai.
        """
        logger.info(
            "🧠⚡️ [Neurogenesis] Memeriksa aturan baru untuk diintegrasikan...")
        rules_dir = Path.home() / "APP_BRAND" / "rules"
        if not rules_dir.exists():
            return

        newly_added_rules = 0
        sys.path.insert(0, str(rules_dir.parent))

        for rule_file in rules_dir.glob("*.py"):
            try:
                module_name = f"rules.{rule_file.stem}"
                if module_name in sys.modules:
                    rule_module = importlib.reload(sys.modules[module_name])
                else:
                    rule_module = importlib.import_module(module_name)

                if hasattr(rule_module, "create_rule"):
                    new_rule_instance = rule_module.create_rule(feature_map)
                    if not any(isinstance(existing_rule, type(new_rule_instance)) for existing_rule in self.rule_bank):
                        self.rule_bank.append(new_rule_instance)
                        newly_added_rules += 1
                        logger.info(
                            f"  -> Aturan baru '{type(new_rule_instance).__name__}' berhasil diintegrasikan.")
            except Exception as e:
                logger.error(
                    f"Gagal memuat atau mengintegrasikan aturan dari {rule_file.name}: {e}")

        sys.path.pop(0)

        if newly_added_rules > 0 or self.symbolic_fusion_head is None:
            if len(self.rule_bank) > 0:
                old_head_inputs = self.symbolic_fusion_head.in_features if self.symbolic_fusion_head else 0
                new_head_inputs = len(self.rule_bank)
                self.symbolic_fusion_head = nn.Linear(new_head_inputs, self.d_model)
                self.symbolic_fusion_head.to(next(self.parameters()).device)
                if old_head_inputs != new_head_inputs:
                    logger.warning(
                        f"  -> REKONSTRUKSI: `symbolic_fusion_head` diperbarui dari {old_head_inputs} -> {new_head_inputs} input.")

    def _generate_standard_pe(self, max_len, d_model):
        """
        Helper on-the-fly untuk menghasilkan Positional Encoding.
        Mengembalikan tensor, bukan nn.Parameter.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x_raw, x_combined, y_historical, edge_index, x_spikes, tracer=None, **kwargs):
        """
        Alur forward yang menggabungkan semua logika, termasuk sistem 
        pelacakan (tracer) untuk diagnostik.
        """

        def trace(step, status, details=""):
            if tracer:
                tracer.add_step(f"FWD_{step}", status, details)

        try:

            thalamic_vector = self.qtc(x_combined)
            thalamic_vector = self._mask_channels(thalamic_vector, self.current_capacity_frac)
            trace("QTC_CRYSTALLIZE", "OK")


            pe = self._generate_standard_pe(self.hparams.window, self.d_model).to(x_combined.device)
            x_context_aware_seq = thalamic_vector.unsqueeze(1).repeat(1, self.hparams.window, 1) + pe
            trace("CONTEXT_BROADCAST", "OK")


            heat_signals_list = []
            student_output = None


            x_time = x_context_aware_seq
            for i, layer in enumerate(self.dpa_stif_layers):
                x_time = layer(x_time)
                heat = torch.var(x_time, dim=-1).mean(dim=-1, keepdim=True)
                heat_signals_list.append(heat)
                if i == 0 and self.training:
                    student_representation = x_time.mean(dim=1)
                    student_output = self.student_head(
                        student_representation).view(-1, self.horizon, self.n_targets)
            time_embedding = x_time.mean(dim=1)
            trace("WAVE_PATH_DPA", "OK")


            x_freq = self.fno_layer(x_context_aware_seq)
            freq_embedding = self.norm_fno(x_freq).mean(dim=1)
            wave_embedding = (time_embedding + freq_embedding) / 2
            trace("WAVE_PATH_FNO", "OK")


            snn_output, _, stdp_loss = self.snn_processor(x_spikes)
            particle_embedding = self.snn_output_projection(snn_output)
            particle_embedding = self._mask_channels(particle_embedding, self.current_capacity_frac)
            if self.training:
                self.stdp_proxy_loss = stdp_loss
            snn_heat = torch.var(particle_embedding, dim=-1, keepdim=True)
            heat_signals_list.append(snn_heat)
            trace("PARTICLE_PATH_SNN", "OK")


            entropy_signal = torch.var(particle_embedding, dim=-1, keepdim=True)
            duality_weight = self.duality_controller(
                wave_embedding, particle_embedding, entropy_signal
            )
            self.last_duality_weight = duality_weight.detach().mean()
            fused_embedding = (duality_weight * wave_embedding) +                ((1 - duality_weight) * particle_embedding)
            trace("HYBRID_FUSION", "OK")

            stable_core_signal = self.menshin_core(x_context_aware_seq).mean(dim=1)
            dampened_signal, _ = self.menshin_damper(
                fused_embedding.unsqueeze(1), stable_core_signal.unsqueeze(1), stable_core_signal.unsqueeze(1)
            )
            dampened_signal = dampened_signal.squeeze(1)
            menshin_stabilized_embedding = self.menshin_fusion_norm(
                fused_embedding + self.dropout(dampened_signal))
            trace("MENSHIN_STABILIZE", "OK")
            

            shielded_embedding_for_vae = self.shield_pre_vae(menshin_stabilized_embedding)
            trace("SHIELD_PRE_VAE", "OK")


            self.mu, self.logvar = self.variational_encoder(shielded_embedding_for_vae)
            z_latent = self.variational_encoder.reparameterize(self.mu, self.logvar)
            trace("VAE_ACTIVATION", "OK")


            stabilized_embedding = self.kevlar_layer(z_latent)
            trace("KEVLAR_LAYER", "OK")


            y_hist_last = y_historical[:, -1, :]
            final_input_to_heads_raw = torch.cat([stabilized_embedding, y_hist_last], dim=1)


            final_input_to_heads = self.shield_pre_heads(final_input_to_heads_raw)
            final_input_to_heads = self._mask_channels(final_input_to_heads, min(1.0, self.current_capacity_frac + 0.1))
            trace("SHIELD_PRE_HEADS", "OK")


            regression_output = self.regression_head(
                final_input_to_heads).view(-1, self.horizon, self.n_targets)
            anomaly_output = self.anomaly_head(final_input_to_heads)
            uncertainty_attribution = self.uncertainty_attribution_head(
                final_input_to_heads)
            trace("PREDICTION_HEADS", "OK")


            return (
                regression_output,
                anomaly_output.squeeze(-1),
                uncertainty_attribution,
                stabilized_embedding,
                student_output,
                None,
            )
        except Exception as e:

            trace("FORWARD_PASS", "FATAL_ERROR", f"{type(e).__name__}: {e}")
            raise e



class TSTLSTM(nn.Module):
    def __init__(
        self, n_features_input, n_targets, d_model, n_layers, dropout, window, horizon
    ):
        super().__init__()
        self.horizon, self.n_targets = horizon, n_targets
        self.tst = TST_module(
            c_in=n_features_input,
            c_out=d_model,
            seq_len=window,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.concrete_dropout = ConcreteDropout(dropout_p=dropout)


        self.regression_lstm = nn.LSTM(
            d_model, d_model * 2, num_layers=2, batch_first=True, dropout=dropout
        )
        self.regression_fc = nn.Linear(d_model * 2, horizon * n_targets)



        self.anomaly_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        shared_embedding = self.tst(x.transpose(1, 2))


        x_reg = self.concrete_dropout(shared_embedding, training=self.training)
        x_reg = x_reg.unsqueeze(1)
        _, (hn, _) = self.regression_lstm(x_reg)
        x_reg = hn[-1]
        regression_output = self.regression_fc(x_reg)
        regression_output = regression_output.view(
            -1, self.horizon, self.n_targets)



        anomaly_output = self.anomaly_head(shared_embedding)


        return regression_output, anomaly_output.squeeze(-1)


class AdaptiveLearningRateController(Callback):
    """
    Callback untuk memotong learning rate secara proaktif ketika
    penurunan loss mulai melambat, untuk mencegah instabilitas di akhir pelatihan.
    """

    def __init__(
        self,
        patience: int = 3,
        improvement_threshold: float = 0.05,
        factor: float = 0.5,
        min_lr: float = 1e-6,
    ):
        """
        Args:
            patience (int): Jumlah epoch berturut-turut di mana penurunan loss kurang dari threshold sebelum LR dipotong.
            improvement_threshold (float): Batas minimal penurunan loss (dalam persen, misal 0.05 = 5%) agar dianggap 'kemajuan'.
            factor (float): Faktor pengali untuk memotong learning rate (misal, 0.5 = potong setengah).
            min_lr (float): Learning rate minimum agar tidak menjadi nol.
        """
        super().__init__()
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.factor = factor
        self.min_lr = min_lr
        self.wait_count = 0
        self.last_loss = float("inf")
        self.triggered = False

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if self.triggered:
            return

        current_loss = trainer.callback_metrics.get("pretrain_loss_epoch")
        if current_loss is None:
            return


        improvement = (self.last_loss - current_loss) / self.last_loss


        if improvement < self.improvement_threshold:
            self.wait_count += 1
            logger.info(
                f"[AdaptiveControl] Epoch {trainer.current_epoch}: Penurunan loss melambat ({improvement:.2%}). Hitungan kesabaran: {self.wait_count}/{self.patience}"
            )
        else:

            self.wait_count = 0


        if self.wait_count >= self.patience:
            optimizer = trainer.optimizers[0]
            old_lr = optimizer.param_groups[0]["lr"]


            if old_lr <= self.min_lr:
                logger.warning(
                    f"[AdaptiveControl] Learning rate sudah mencapai batas minimum ({self.min_lr}). Tidak ada tindakan."
                )
                self.triggered = True
                return


            new_lr = old_lr * self.factor
            optimizer.param_groups[0]["lr"] = new_lr

            logger.warning(
                f"🔥🔥🔥 [AdaptiveControl] PENURUNAN LOSS MELAMBAT! Memotong Learning Rate dari {old_lr:.6f} menjadi {new_lr:.6f} untuk menjaga stabilitas. 🔥🔥🔥"
            )
            self.triggered = (
                True
            )

        self.last_loss = current_loss


class LatentSpaceGeometryMonitor:
    """
    Melatih VAE pada fitur HHT murni dan memonitor geometri ruang latennya.
    Ini adalah implementasi dari Sensor #3 ASeT.
    """

    def __init__(self, project_id: str, hht_feature_names: list, latent_dim: int = 8):
        self.project_id = project_id
        self.hht_feature_names = hht_feature_names
        self.latent_dim = latent_dim
        self.vae_path = get_path(
            project_id, "checkpoint_dir") / "aset_sensor3_vae.pth"
        self.vae = EnhancedAutoencoder(
            input_dim=len(hht_feature_names), hidden_dim=latent_dim
        )
        self.is_trained = False
        self.volume_history = []
        logger.info(
            f"ASeT Sensor #3 (VAE Geometri) diinisialisasi dengan {len(hht_feature_names)} fitur HHT."
        )

    def train(self, df: pd.DataFrame):
        logger.info("  -> Memulai pelatihan VAE untuk Sensor #3...")
        if not self.hht_feature_names:
            logger.error(
                "  -> Pelatihan VAE dibatalkan: tidak ada fitur HHT yang ditemukan."
            )
            return

        hht_data = df[self.hht_feature_names].dropna()
        if len(hht_data) < 100:
            logger.error(
                f"  -> Pelatihan VAE dibatalkan: data HHT tidak cukup ({len(hht_data)} baris)."
            )
            return

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(hht_data)
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(30):
            _, decoded = self.vae(data_tensor)
            loss = loss_fn(decoded, data_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(self.vae.state_dict(), self.vae_path)
        self.is_trained = True
        logger.info("  -> ✅ Pelatihan VAE untuk Sensor #3 selesai.")

    def check_for_singularity(self, latest_hht_data: pd.DataFrame) -> bool:
        if not self.is_trained or latest_hht_data.empty:
            return False

        data_tensor = torch.tensor(
            latest_hht_data[self.hht_feature_names].values, dtype=torch.float32
        )
        with torch.no_grad():
            latent_z, _ = self.vae(data_tensor)

        try:
            hull = ConvexHull(latent_z.numpy())
            current_volume = hull.volume
        except Exception:

            current_volume = 0.0

        self.volume_history.append(current_volume)
        if len(self.volume_history) < 5:
            return False


        recent_volumes = self.volume_history[-5:]
        if recent_volumes[-1] < 0.1 * recent_volumes[0]:
            logger.critical(
                "🚨 ASET SENSOR #3: KONTRAKSI VOLUME RUANG LATEN TERDETEKSI! SINYAL SINGULARITAS!"
            )
            return True

        return False


class ASeT_ProtocolMonitor(pl.Callback):
    """
    Callback yang menjadi sistem saraf ASeT, memonitor sinyal dan memicu protokol.
    """

    def __init__(
        self, geometry_monitor: LatentSpaceGeometryMonitor, governor: CognitiveGovernor
    ):
        super().__init__()
        self.geometry_monitor = geometry_monitor
        self.governor = governor
        self.aset_level = 0

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        df_processed = trainer.datamodule.df_processed
        if df_processed is None or len(df_processed) < 60:
            return


        latest_data = df_processed.iloc[-1]
        latest_window_hht = df_processed.iloc[-60:]


        sensor1_val = latest_data.get("aset_sensor1_trend_correlation", 1.0)
        sensor2_val = latest_data.get("aset_sensor2_freq_instability", 0.0)
        sensor3_active = self.geometry_monitor.check_for_singularity(
            latest_window_hht)


        s1_triggered = sensor1_val < 0.2
        s2_triggered = sensor2_val > (
            df_processed["aset_sensor2_freq_instability"].quantile(0.95)
        )


        if sensor3_active and s2_triggered and s1_triggered:
            if self.aset_level != 3:
                logger.critical(
                    "🔥🔥🔥 ASET LEVEL 3: PROTOKOL BIG BANG AKTIF! 🔥🔥🔥")
                self.governor.log_event(
                    "ASET_LEVEL_3_SINGULARITY", {
                        "reason": "All sensors triggered."}
                )

                trainer.should_stop = True
                self.aset_level = 3
        elif s1_triggered and s2_triggered:
            if self.aset_level != 2:
                logger.warning("⚠️⚠️ ASET LEVEL 2: MODE PROTEKSI KRISIS! ⚠️⚠️")
                self.governor.log_event(
                    "ASET_LEVEL_2_DANGER",
                    {"corr": sensor1_val, "instability": sensor2_val},
                )

                pl_module.loss_fn.update_aset_protocol(
                    da_multiplier=0.1, mse_multiplier=2.5
                )
                self.aset_level = 2
        elif s1_triggered:
            if self.aset_level != 1:
                logger.info(
                    "🟡 ASET LEVEL 1: PERINGATAN! Pasar tidak stabil. 🟡")
                self.governor.log_event("ASET_LEVEL_1_WARNING", {
                                        "corr": sensor1_val})

                self.aset_level = 1
        else:
            if self.aset_level != 0:
                logger.info("✅ ASET LEVEL 0: Kondisi pasar kembali normal.")

                pl_module.loss_fn.update_aset_protocol(
                    da_multiplier=1.0, mse_multiplier=1.0
                )
                self.aset_level = 0


class ContextualAttentionHead(nn.Module):
    """
    Modul Atensi untuk menimbang dan menggabungkan beberapa memori kontekstual.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, current_situation_emb: torch.Tensor, memory_embs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:



        query = current_situation_emb.unsqueeze(0)
        key_value = memory_embs.unsqueeze(0)


        attended_output, attention_weights = self.attention(
            query, key_value, key_value)


        fused_context = self.norm(query + self.ffn(attended_output))

        return fused_context.squeeze(0), attention_weights.squeeze(0)


class ACR_Callback(pl.Callback):
    """
    ACR v4.0: Sistem Penalaran Hibrida dengan Atensi Kontekstual.
    Menggunakan Oracle LLM (Online) atau RAG Internal (Offline) untuk neurogenesis kontekstual.
    Sekarang dilengkapi dengan kepala atensi untuk analisis memori lebih tajam.
    """

    def __init__(
        self,
        nsmm: "NSMM",
        api_pool: "DistributedAIPool",
        together_keys: dict,
        brain: "Brain",
        embedding_model: "APIEmbedder",
    ):
        super().__init__()
        self.nsmm = nsmm
        self.api_pool = api_pool
        self.brain = brain
        self.embedding_model = embedding_model
        self.session_id = nsmm.session_id
        self.oracle_agent = None
        self.loss_history = []
        self.improvement_threshold = 0.01


        qwen_key = together_keys.get("critical_supervisor")
        if qwen_key:
            self.oracle_agent = TogetherLLM(
                api_key=qwen_key, model_name="Qwen/Qwen2-72B-Instruct"
            )
            logger.info("🔮 [ACR] Mode: ONLINE. Oracle LLM (Qwen2-72B) aktif.")
        else:
            logger.warning(
                "🔮 [ACR] Mode: OFFLINE. Menggunakan Peneliti Kontekstual Internal (RAG)."
            )


        self.attention_head = ContextualAttentionHead(
            embed_dim=self.embedding_model.dim
        )

    def _run_offline_contextual_analysis(
        self, metrics: dict, vitals_summary: str, pl_module: "pl.LightningModule", trainer: "pl.Trainer"
    ) -> dict:
        """Menganalisis metrik dan mencari memori untuk menghasilkan diagnosis Chain-of-Thought."""

        monitor_metric_key = pl_module.hparams.get("monitor_metric", "val_loss")
        current_loss = metrics.get(monitor_metric_key, 0.0)
        previous_loss = self.loss_history[-1] if self.loss_history else float("inf")
        improvement = (previous_loss - current_loss) / (abs(previous_loss) + 1e-9)
        self.loss_history.append(current_loss)

        status_text = f"Epoch {trainer.current_epoch}: {monitor_metric_key} berubah dari {previous_loss:.4f} menjadi {current_loss:.4f} (Perbaikan: {improvement:+.2%}). Tanda vital: {vitals_summary}."


        similar_experiences = self.nsmm.query_similar_experiences(
            status_text, self.embedding_model, top_k=2
        )
        relevant_knowledge = self.brain.query(status_text, k=2)


        analysis_text = ""
        if improvement < -self.improvement_threshold:
            outcome = "negative"
            analysis_text = "Analisis: Kinerja memburuk secara signifikan."
            if any("overfitting" in exp.lower() for exp in similar_experiences):
                analysis_text += " Ini menunjukkan pola yang mirip dengan pengalaman 'overfitting' sebelumnya."
            if any("volatil" in know.lower() for know in relevant_knowledge):
                analysis_text += " Pengetahuan dari Brain mengindikasikan ini mungkin terkait dengan volatilitas pasar saat ini."
        elif improvement > self.improvement_threshold:
            outcome = "positive"
            analysis_text = "Analisis: Kinerja membaik sesuai harapan. Strategi saat ini tampaknya bekerja."
        else:
            outcome = "neutral"
            analysis_text = "Analisis: Kinerja stagnan, menunjukkan kemungkinan plateau pembelajaran."


        CoT_report = (
            f"**Jejak Penalaran (Chain-of-Thought) - Analis Offline**\n"
            f"1. **[Observasi]:** {status_text}\n"
            f"2. **[Konteks Memori]:**\n"
            f"   - NSMM (Pengalaman): {' | '.join(similar_experiences) or 'Tidak ada pengalaman serupa yang signifikan.'}\n"
            f"   - Brain (Pengetahuan): {' | '.join(relevant_knowledge) or 'Tidak ada pengetahuan eksternal yang relevan.'}\n"
            f"3. **[Analisis]:** {analysis_text}\n"
            f"4. **[Kesimpulan]:** Outcome dinilai sebagai **{outcome.upper()}**."
        )

        return {"diagnosis": CoT_report, "outcome": outcome, "confidence": 0.75}

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if not hasattr(pl_module, "get_neuro_vitals"):
            return

        logger.info(
            f"\n--- [ACR Cycle | Epoch {trainer.current_epoch}] Memulai Siklus Refleksi Diri ---"
        )

        state_vector, vitals_summary = pl_module.get_neuro_vitals()
        metrics = {
            k: v.item()
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, torch.Tensor)
        }

        oracle_response_str = ""
        decision_data = {}
        prompt = ""

        if self.oracle_agent:

            prompt = self._build_consultation_prompt(
                state_vector, metrics, vitals_summary
            )
            oracle_response_str = self.oracle_agent.chat(prompt)
            try:
                response_dict = ast.literal_eval(oracle_response_str)
                diagnosis_text = response_dict.get("diagnosis", "Oracle tidak memberikan diagnosis.")
                

                if isinstance(diagnosis_text, str) and "baik" in diagnosis_text.lower():
                    outcome = "positive"
                else:
                    outcome = "neutral"
                
                decision_data = {
                    "diagnosis": diagnosis_text,
                    "outcome": outcome,
                    "confidence": 0.9
                }
            except Exception:

                decision_data = self._run_offline_contextual_analysis(
                    metrics, vitals_summary, pl_module, trainer
                )
        else:

            decision_data = self._run_offline_contextual_analysis(
                metrics, vitals_summary, pl_module, trainer
            )

        experience_data = {
            "metrics": metrics,
            "state_vector": state_vector,
            "diagnosis": decision_data.get("diagnosis"),
            "oracle_query": prompt if self.oracle_agent else "OFFLINE_MODE",
            "oracle_response": oracle_response_str,
            "final_decision": {"action": "CONTINUE"},
        }
        memory_id = self.nsmm.log_epoch_experience(
            self.session_id, trainer.current_epoch, experience_data
        )

        insight_module = nn.Sequential(
            nn.Linear(len(state_vector), 16), nn.SiLU(), nn.Linear(16, 8)
        )
        self.nsmm.store_insight_neuron(
            memory_id=memory_id,
            module=insight_module,
            reason=f"End of Epoch {trainer.current_epoch}",
            state_vector_dim=len(state_vector),
            outcome=decision_data.get("outcome", "neutral"),
            trigger_metric_key=pl_module.hparams.get(
                "monitor_metric", "val_loss"),
            trigger_metric_value=metrics.get(
                pl_module.hparams.get("monitor_metric", "val_loss"), 0.0
            ),
            confidence=decision_data.get("confidence", 0.5),
        )

    def _build_consultation_prompt(self, state_vector, metrics, vitals_summary):
        return f"""
        Anda adalah Oracle, seorang AI Research Scientist yang optimis dan berorientasi pada solusi.
        Analisis kondisi pelatihan model AI berikut. Fokus pada apa yang bisa dipelajari dan ditingkatkan.

        KONDISI INTERNAL (STATE VECTOR): {json.dumps(state_vector, indent=2)}
        METRIK KINERJA EPOCH TERAKHIR: {json.dumps(metrics, indent=2)}
        RINGKASAN TANDA VITAL: {vitals_summary}

        TUGAS ANDA:
        1.  **Diagnosis:** Berikan analisis yang konstruktif. Jika ada masalah, lihat sebagai "peluang untuk perbaikan", bukan "kegagalan".
        2.  **Rencana Tindakan:** Usulkan rencana yang jelas dan optimis untuk epoch berikutnya.
        3.  **Format Jawaban:** Jawab HANYA dengan satu objek Python Dictionary dengan kunci "diagnosis" dan "proposed_plan".
        """


def process_with_diml(
    user_prompt: str,
    nsmm: "NSMM",
    api_pool: "DistributedAIPool",
    embedding_model: "APIEmbedder",
) -> tuple[torch.Tensor, str, str]:
    """
    Menjalankan proses DIML untuk mengekstrak niat implisit pengguna
    dan menyimpannya sebagai memori baru.
    """
    logger.info(
        "🧠 [DIML] Mengaktifkan lapisan meta-learning untuk memahami niat...")

    with closing(sqlite3.connect(nsmm.db_path)) as conn:
        df_memories = pd.read_sql_query(
            "SELECT diagnosis, final_decision FROM epoch_memories ORDER BY id DESC LIMIT 10",
            conn,
        )
    historical_context = df_memories.to_string()

    psychoanalyst_agent = "supervisor"

    prompt = f"""
    Anda adalah seorang AI Psychologist dengan spesialisasi Cognitive-Behavioral AI.
    Tugas Anda adalah menganalisis permintaan pengguna dan menyimpulkan 'niat implisit' atau 'tujuan tak terucapkan' mereka.

    KONTEKS PERMINTAAN SAAT INI:
    ---
    "{user_prompt}"
    ---

    KONTEKS HISTORIS (KEPUTUSAN & DIAGNOSIS MODEL SEBELUMNYA):
    ---
    {historical_context}
    ---

    Berdasarkan kedua konteks di atas, jawab DUA hal dalam format JSON:
    1.  `inferred_intent`: Satu kalimat yang merangkum tujuan atau kekhawatiran utama pengguna (contoh: "Pengguna khawatir tentang risiko penurunan tiba-tiba dan mencari strategi lindung nilai.").
    2.  `reasoning_mode`: Pilih mode penalaran yang paling cocok untuk merespons niat ini dari daftar berikut: [ANALYTICAL, HYPOTHETICAL, CREATIVE_ANALOGY, CAUSAL_EXPLANATION].

    Contoh output JSON:
    {{"inferred_intent": "Pengguna ingin tahu apakah model terlalu optimis terhadap saham teknologi.",
      "reasoning_mode": "HYPOTHETICAL"}}
    """

    response_str = api_pool.call_gemini_for_text(prompt, psychoanalyst_agent)
    intent_data = robust_json_extract(response_str, model=None) or {}

    inferred_intent_text = intent_data.get("inferred_intent", "N/A")
    reasoning_mode = intent_data.get("reasoning_mode", "ANALYTICAL")
    logger.info(f"     -> Niat Terinferensi: {inferred_intent_text}")
    logger.info(f"     -> Mode Penalaran Dipilih: {reasoning_mode}")

    intent_vector = embedding_model.encode(
        inferred_intent_text, task_type="query")
    intent_vector_tensor = torch.tensor(intent_vector, dtype=torch.float32)

    with closing(sqlite3.connect(nsmm.db_path)) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO dialogue_intents (session_id, user_prompt, inferred_intent_text, inferred_intent_vector) VALUES (?, ?, ?, ?)",
            (
                nsmm.session_id,
                user_prompt,
                inferred_intent_text,
                intent_vector_tensor.numpy().tobytes(),
            ),
        )
        conn.commit()

    return intent_vector_tensor, reasoning_mode, inferred_intent_text


def run_market_intelligence_gathering(
    selected_tickers: list, api_pool: "DistributedAIPool", brain: "Brain"
):
    """
    Secara proaktif bertanya kepada LLM untuk mendapatkan wawasan pasar
    berdasarkan ticker yang dipilih dan menyimpannya sebagai pengetahuan baru ke dalam Brain.
    """
    logger.info("\n--- 🧠 Memulai Fase Intelijen Pasar Proaktif ---")

    prompt = f"""
    Anda adalah seorang Analis Pasar Global senior untuk sebuah hedge fund.
    Saya akan memulai analisis teknikal mendalam untuk ticker berikut: {', '.join(selected_tickers)}.

    Sebelum saya mulai, berikan saya ringkasan intelijen tingkat tinggi. Jawab pertanyaan-pertanyaan berikut secara ringkas dan padat:
    1.  **Asal Pasar:** Ticker-ticker ini berasal dari pasar mana (contoh: US Stock Market, IDX Indonesia, Nikkei Jepang)?
    2.  **Karakteristik Pasar:** Bagaimana karakteristik umum dari pasar tersebut saat ini? (contoh: sedang bullish karena data inflasi, volatil karena pemilu, dll.)
    3.  **Karakteristik Ticker:** Apa karakteristik unik dari masing-masing ticker ini? (contoh: NVDA adalah pemimpin di sektor AI, GOOGL lebih stabil di sektor teknologi, dll.)
    4.  **Risiko Manipulasi:** Apakah pasar dari negara tersebut dikenal memiliki risiko 'pumping' atau 'dumping' yang signifikan sewaktu-waktu?

    Berikan jawaban Anda sebagai satu laporan ringkas.
    """
    try:

        def get_intel():
            return api_pool.call_gemini_for_text(prompt, "advanced_advisor")

        market_insight = execute_with_thinking_animation(
            get_intel, message="AI sedang melakukan riset pasar..."
        )

        if not market_insight or len(market_insight) < 20:
            logger.warning(
                "Gagal mendapatkan wawasan pasar yang signifikan dari AI.")
            return

        logger.info("\n--- Wawasan Pasar Diterima ---")
        print(textwrap.fill(market_insight, width=90))
        logger.info("----------------------------")

        source_name = f"MarketIntel_{'_'.join(selected_tickers)}_{datetime.now().strftime('%Y%m%d')}"
        insight_chunks = chunk_text(
            market_insight, chunk_size=300, chunk_overlap=40)

        brain.add_chunks(insight_chunks, source_name=source_name)
        logger.info(
            "✅ Wawasan baru telah disimpan ke Brain dan memicu neurogenesis di memori virtual."
        )

    except Exception as e:
        logger.error(
            f"Gagal menjalankan siklus intelijen pasar: {e}", exc_info=True)


class HealerProtocol(threading.Thread):
    """
    Bekerja di latar belakang untuk mendiagnosis error, menemukan solusi (dari memori atau LLM),
    memvalidasinya di sandbox, dan mengajukan proposal perbaikan.
    """

    def __init__(self, error_context, nsmm, api_pool, permission_queue):
        super().__init__(daemon=True)
        self.context = error_context
        self.nsmm = nsmm
        self.api_pool = api_pool
        self.permission_queue = permission_queue
        self.together_api_keys = together_api_keys

    def _get_error_signature(self):

        return hashlib.sha256(
            (self.context['error_type'] +
             self.context['problematic_code_line']).encode()
        ).hexdigest()

    def run(self):
        try:
            signature = self._get_error_signature()


            with sqlite3.connect(self.nsmm.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT corrected_code, original_code FROM healing_protocols WHERE error_signature = ?", (signature,))
                solution = cursor.fetchone()

            if solution:
                logger.info(
                    "💡 [Healer] Solusi ditemukan di memori! Menggunakan perbaikan yang sudah ada.")
                corrected_code, original_code = solution
                proposal = {'original_code_to_find': original_code,
                            'suggested_code_to_replace': corrected_code}
            else:

                logger.warning(
                    "🩺 [Healer] Masalah baru terdeteksi. Mengkonsultasikan dengan Spesialis Kode (Exaone)...")

                specialist_key = self.together_api_keys.get("exaone")
                if not specialist_key:
                    logger.error(
                        "[Healer] API Key untuk 'exaone' tidak ditemukan. Healing dibatalkan.")
                    return


                specialist_agent = TogetherLLM(
                    api_key=specialist_key,
                    model_name="lgai/exaone-deep-32b"
                )

                pydantic_schema_str = json.dumps(
                    ProposeCodeFix.model_json_schema(), indent=2)


                specialist_prompt = f"""
                You are an expert Python debugging assistant and a world-class AI research engineer.
                Analyze the following error context and provide a precise, targeted code fix.
                Your primary goal is correctness and logical soundness.
                You MUST respond with a single, valid JSON object that conforms to the `ProposeCodeFix` schema. Do not add any other text, markdown, or explanations.

                ERROR CONTEXT:
                ```json
                {json.dumps(self.context, indent=2)}
                ```

                JSON SCHEMA TO FOLLOW:
                ```json
                {pydantic_schema_str}
                ```

                Your JSON response:
                """

                try:
                    response_str = specialist_agent.chat(specialist_prompt)
                    proposal_obj = robust_json_extract(
                        response_str, model=ProposeCodeFix)
                    if not proposal_obj:
                        logger.error(
                            "[Healer] Spesialis Exaone gagal memberikan proposal JSON yang valid.")
                        return


                    proposal = proposal_obj.model_dump()

                except Exception as e:
                    logger.error(
                        f"[Healer] Gagal saat konsultasi dengan Spesialis Exaone: {e}")
                    return



            logger.info(
                "⚙️ [Sandbox] Memvalidasi proposal perbaikan secara virtual...")
            time.sleep(2)
            is_valid_in_sandbox = True

            if is_valid_in_sandbox:
                logger.info("✅ [Sandbox] Proposal perbaikan lolos validasi.")
                report = generate_diff_report(
                    proposal['original_code_to_find'], proposal['suggested_code_to_replace'])
                permission_request = {
                    "report": report,
                    "proposal": proposal,
                    "file_path": self.context['file_path'],
                    "error_context": self.context,
                    "is_from_memory": bool(solution)
                }
                self.permission_queue.put(permission_request)

        except Exception as e:
            logger.error(f"Gagal dalam HealerProtocol: {e}", exc_info=True)


class UserInteractionManager(threading.Thread):
    """
    Menangani interaksi Y/N dengan pengguna di thread terpisah agar tidak memblokir loop utama.
    """

    def __init__(self, permission_queue, decision_queue):
        super().__init__(daemon=True)
        self.permission_queue = permission_queue
        self.decision_queue = decision_queue

    def run(self):
        while True:
            request = self.permission_queue.get()

            print("\n" + "="*20 + " 🚨 PERMINTAAN IZIN PERBAIKAN OTOMATIS 🚨 " + "="*20)
            print(f"File Target: {request['file_path']}")
            print(
                f"Error: {request['error_context']['error_type']} - {request['error_context']['error_message']}")
            print("\n--- Laporan Perubahan (Diff) ---")
            print(request['report'])
            print("="*75)

            answer = questionary.confirm(
                "Terapkan perbaikan dan coba lagi secara real-time?", default=False).ask()

            self.decision_queue.put({
                "approved": answer,
                "request": request
            })


class ARMH_Callback(pl.Callback):
    """
    ARMH v6.0: Protokol pertahanan multi-lapis dengan mode diagnostik otonom,
    memori global untuk analisis post-mortem, dan intervensi cerdas.
    """

    def __init__(
        self,
        api_pool: "DistributedAIPool",
        together_keys: dict,
        early_warning_threshold: float = 0.80,
        critical_threshold: float = 1.5,
        lr_cut_factor: float = 0.5,
    ):
        super().__init__()
        self.api_pool = api_pool
        self.together_keys = together_keys
        self.early_warning_threshold = early_warning_threshold
        self.critical_threshold = critical_threshold
        self.lr_cut_factor = lr_cut_factor

        self.best_val_loss = float("inf")
        self.recovery_cooldown = 0
        self.early_warning_active = False


        self.global_problem_log = []

        logger.info(
            f"🦾 Protokol ARMH v6.0 (Global Logging) aktif. Peringatan Dini: >{self.early_warning_threshold:.0%}, Kritis: >{self.critical_threshold:.0%}."
        )

    def _run_batch_isolation_protocol(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> str:
        """
        Menjalankan validasi ulang dengan batch_size=1 untuk mengisolasi data bermasalah
        dan mencatatnya ke memori global.
        """
        logger.warning(
            "🔬 [ARMH] MEMULAI PROTOKOL ISOLASI BATCH (batch_size=1)...")

        try:
            val_dataloader = trainer.datamodule.val_dataloader()

            diagnostic_loader = TorchDataLoader(
                val_dataloader.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=val_dataloader.collate_fn,
            )

            pl_module.eval()
            losses = []
            with torch.no_grad():

                val_indices = (
                    val_dataloader.sampler.indices
                    if hasattr(val_dataloader.sampler, "indices")
                    else range(len(val_dataloader.dataset))
                )

                for i, batch in enumerate(
                    tqdm(
                        diagnostic_loader,
                        desc="Menganalisis Batch Individu",
                        leave=False,
                    )
                ):

                    if isinstance(batch, dict):
                        batch_on_device = {
                            k: v.to(pl_module.device) for k, v in batch.items()
                        }
                    elif isinstance(batch, (list, tuple)):
                        batch_on_device = [
                            b.to(pl_module.device) if isinstance(
                                b, torch.Tensor) else b
                            for b in batch
                        ]
                    else:
                        batch_on_device = batch.to(pl_module.device)

                    loss_output = pl_module.validation_step(batch_on_device, i)

                    current_loss = None
                    if isinstance(loss_output, dict) and "loss" in loss_output:
                        current_loss = loss_output["loss"].item()
                    elif isinstance(loss_output, torch.Tensor):
                        current_loss = loss_output.item()

                    if current_loss is not None:
                        original_index = val_indices[
                            i
                        ]
                        losses.append((original_index, current_loss))

            pl_module.train()

            if not losses:
                return "Laporan Diagnostik: Gagal menghitung loss per-batch."

            losses.sort(key=lambda x: x[1], reverse=True)
            top_5_problematic = losses[:5]

            report = "Laporan Diagnostik:\n"
            report += "5 data point dengan validation loss tertinggi adalah:\n"
            for idx, loss_val in top_5_problematic:
                report += f"- Indeks Data Asli: {idx}, Loss: {loss_val:.4f}\n"


            indices_to_remember = {idx for idx, loss_val in top_5_problematic}
            for idx in indices_to_remember:
                self.global_problem_log.append(
                    {
                        "epoch": trainer.current_epoch,
                        "sample_index": idx,
                        "loss": dict(losses).get(idx),
                    }
                )
            logger.info(
                f"🧠 [ARMH Global Log] {len(indices_to_remember)} sampel sulit baru dicatat. Total catatan investigasi: {len(self.global_problem_log)}."
            )

            logger.warning("🔬 [ARMH] Protokol Isolasi Batch Selesai.")
            return report

        except Exception as e:
            logger.error(
                f"Gagal menjalankan protokol isolasi batch: {e}", exc_info=True
            )
            pl_module.train()
            return f"Laporan Diagnostik: Terjadi error saat menjalankan protokol ({type(e).__name__})."

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
            return

        if "val_loss" not in trainer.callback_metrics:
            return

        current_loss = trainer.callback_metrics["val_loss"].item()

        if current_loss < self.best_val_loss:
            if self.early_warning_active:
                logger.info(
                    f"[ARMH] ✅ Peringatan Dini Selesai. Armor kembali stabil. Performa membaik ke {current_loss:.4f}."
                )
                self.early_warning_active = False
            self.best_val_loss = current_loss
            return

        degradation_pct = (current_loss - self.best_val_loss) / (
            abs(self.best_val_loss) + 1e-9
        )

        if degradation_pct > self.critical_threshold:
            logger.critical(
                f"🔥🔥🔥 [ARMH LEVEL 2] KERUSAKAN KRITIS! Performa anjlok {degradation_pct:+.1%}. Memulai Protokol Diagnostik & Intervensi."
            )
            self._trigger_critical_intervention(
                trainer, pl_module, current_loss)
            return

        if (
            degradation_pct > self.early_warning_threshold
            and not self.early_warning_active
        ):
            logger.warning(
                f"🟡 [ARMH LEVEL 1] GUNCANGAN AWAL TERDETEKSI ({degradation_pct:+.1%}). Menebalkan armor secara proaktif..."
            )
            self._thicken_armor(trainer, pl_module)
            self.early_warning_active = True
            self.recovery_cooldown = 2

    def _thicken_armor(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Aksi refleks untuk meningkatkan regularisasi."""
        if hasattr(pl_module.hparams, "dropout"):
            old_dropout = pl_module.hparams.dropout
            new_dropout = min(0.5, old_dropout + 0.05)
            pl_module.hparams.dropout = new_dropout
            logger.info(
                f"  -> [Penguatan] Dropout ditingkatkan: {old_dropout:.3f} -> {new_dropout:.3f}"
            )

        if hasattr(pl_module.hparams, "weight_decay"):
            optimizer = trainer.optimizers[0]
            if hasattr(optimizer, "optimizer"):
                optimizer = optimizer.optimizer
            old_wd = optimizer.param_groups[0]["weight_decay"]
            new_wd = old_wd * 1.5
            optimizer.param_groups[0]["weight_decay"] = new_wd
            logger.info(
                f"  -> [Penguatan] Weight Decay ditingkatkan: {old_wd:.6f} -> {new_wd:.6f}"
            )

    def _trigger_critical_intervention(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        current_loss: float,
    ):
        """
        Menjalankan alur kerja: diagnostik, konsultasi AI (dengan Jejak Kognitif),
        dan tindakan perbaikan.
        """

        diagnostic_report = self._run_batch_isolation_protocol(
            trainer, pl_module)

        collapse_detected = getattr(
            pl_module, "gradient_collapse_detected_in_epoch", False
        )
        additional_context = ""
        if collapse_detected:
            additional_context = (
                "\n\n**DIAGNOSIS KRITIS TAMBAHAN:**\n"
                "Sistem mendeteksi adanya ledakan gradien (NaN/inf) selama training step pada epoch ini. "
                "Ini adalah indikasi kuat adanya 'Representation Collapse' atau instabilitas numerik parah. "
                "Pertimbangkan saran yang secara langsung mengatasi masalah ini, seperti mereset sebagian bobot atau mengubah parameter loss function."
            )

            pl_module.gradient_collapse_detected_in_epoch = False




        last_successful_trace = getattr(pl_module, 'last_successful_trace', 'Jejak sukses tidak tersedia.')
        last_failed_trace = getattr(pl_module, 'last_failed_trace', None)


        trace_to_report = last_failed_trace if last_failed_trace else last_successful_trace



        pydantic_schema_str = json.dumps(
            CriticalInterventionPlan.model_json_schema(), indent=2
        )

        prompt = f"""
        Anda adalah seorang AI Supervisor yang ahli dalam men-debug pelatihan deep learning.
        Model saya mengalami kenaikan validation loss yang signifikan. Saya telah menjalankan tes diagnostik dan memiliki jejak operasional internal.

        # LAPORAN KRISIS
        - Epoch: {trainer.current_epoch}
        - Loss Terbaik Sejauh Ini: {self.best_val_loss:.4f}
        - Loss Gagal Saat Ini: {current_loss:.4f}
        {additional_context}

        # LAPORAN DIAGNOSTIK (DARI PROTOKOL ISOLASI BATCH)
        ---
        {diagnostic_report}
        ---

        # JEJAK KOGNITIF (ALUR KERJA INTERNAL TERAKHIR SEBELUM VALIDASI GAGAL)
        # Jejak ini menunjukkan urutan operasi internal yang dieksekusi model.
        # Analisis alur ini, terutama jika statusnya FATAL_ERROR, untuk menemukan di mana masalah kemungkinan besar dimulai.
        ---
        {trace_to_report}
        ---

        # TUGAS ANDA
        1. Analisis SEMUA informasi di atas, terutama **JEJAK KOGNITIF**, untuk mendiagnosis akar masalah.
        2. Tentukan tindakan terbaik. Opsi: 'apply_reflex_fix', 'propose_new_plan', 'stop_training_critical'.
        3. Kembalikan jawaban Anda HANYA dalam format JSON yang valid sesuai skema di bawah.

        # JSON Schema
        {pydantic_schema_str}
        """

        try:

            qwen_key = self.together_keys.get("critical_supervisor")
            if not qwen_key:
                logger.error(
                    "[ARMH] Kunci API 'critical_supervisor' tidak ditemukan. Menghentikan pelatihan demi keamanan."
                )
                trainer.should_stop = True
                return

            response_str = call_agent_with_fallback(
                api_key=qwen_key,
                primary_model="Qwen/Qwen2-72B-Instruct",
                backup_models=["meta-llama/Llama-3-70B-Instruct-Turbo"],
                prompt=prompt,
            )

            if not response_str:
                raise ValueError("Supervisor Kritis tidak memberikan respons.")

            evaluation_dict = robust_json_extract(
                response_str, model=CriticalInterventionPlan
            )
            if not evaluation_dict:
                raise ValueError(
                    f"Supervisor Kritis tidak memberikan JSON valid. Respons: {response_str}"
                )

            plan = (
                evaluation_dict
                if isinstance(evaluation_dict, CriticalInterventionPlan)
                else CriticalInterventionPlan(**evaluation_dict)
            )
            logger.info(
                f"[ARMH Supervisor] Vonis: {plan.verdict}. Alasan: {plan.reasoning}"
            )


            if plan.verdict == "stop_training_critical":
                if trainer.current_epoch < 3:
                    logger.warning(f"Supervisor merekomendasikan berhenti, TAPI diabaikan karena masih dalam masa tenggang (Epoch {trainer.current_epoch}).")

                else:
                    logger.error(
                        "[ARMH] 🛑 AI Supervisor merekomendasikan penghentian. Masalah fundamental terdeteksi."
                    )
                    trainer.should_stop = True

            elif plan.verdict == "propose_new_plan" and plan.proposed_hparams:
                logger.warning(
                    "✅ [ARMH] AI Supervisor mengusulkan rencana perbaikan baru. Menerapkan..."
                )
                optimizer = trainer.optimizers[0]
                if hasattr(optimizer, "optimizer"):
                    optimizer = optimizer.optimizer
                for param, value in plan.proposed_hparams.items():
                    if hasattr(pl_module.hparams, param):
                        setattr(pl_module.hparams, param, value)
                        logger.info(
                            f"  -> hparam '{param}' diubah menjadi {value}")
                    if param in optimizer.param_groups[0]:
                        optimizer.param_groups[0][param] = value
                        logger.info(
                            f"  -> Optimizer param '{param}' diubah menjadi {value}"
                        )
                self.recovery_cooldown = 3

            else:
                logger.info(
                    "[ARMH] ✅ Menerapkan perbaikan refleks standar sesuai anjuran AI."
                )
                optimizer = trainer.optimizers[0]
                if hasattr(optimizer, "optimizer"):
                    optimizer = optimizer.optimizer
                optimizer.param_groups[0]["lr"] *= self.lr_cut_factor
                if hasattr(pl_module.hparams, "dropout"):
                    pl_module.hparams.dropout = min(
                        0.5, pl_module.hparams.dropout + 0.1
                    )
                self.recovery_cooldown = 3

        except Exception as e:
            logger.error(
                f"[ARMH] Gagal total menjalankan siklus intervensi: {e}. Menghentikan pelatihan demi keamanan.",
                exc_info=True,
            )
            trainer.should_stop = True

class SynapticIntelligenceCallback(pl.Callback):
    """
    Mencegah catastrophic forgetting selama fine-tuning dengan memberikan
    penalti pada perubahan signifikan dari bobot pre-trained.
    Ini adalah alternatif yang lebih stabil dan terisolasi untuk TGW.
    """

    def __init__(self, project_id: str, strength: float = 0.5):
        """
        Args:
            project_id (str): ID proyek untuk menemukan path model pre-train.
            strength (float): Seberapa kuat "tarikan" kembali ke bobot pre-train.
                              Nilai yang lebih tinggi berarti regularisasi lebih kuat.
        """
        super().__init__()
        self.strength = strength
        self.project_id = project_id
        self.pretrained_weights = {}
        self.is_ready = False
        logger.info(f"🧠 [Synaptic Intelligence] Callback diaktifkan dengan kekuatan: {self.strength}")

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        """
        Dipanggil sekali di awal pelatihan untuk memuat bobot referensi.
        """
        if stage == "fit":
            try:

                pretrain_path = get_path(self.project_id, "pretrained_encoder")
                if not pretrain_path.exists():
                    logger.warning("[Synaptic Intelligence] Checkpoint pre-train tidak ditemukan. Callback tidak akan aktif.")
                    return
                

                full_pretrain_state = torch.load(pretrain_path, map_location=pl_module.device)
                

                encoder_state_dict = full_pretrain_state.get("encoder")
                if not encoder_state_dict:
                     logger.warning("[Synaptic Intelligence] 'encoder' tidak ditemukan dalam state dict pre-train. Callback tidak akan aktif.")
                     return


                for name, param in encoder_state_dict.items():

                    full_name = f"core.qtc.encoder.{name}"
                    self.pretrained_weights[full_name] = param.detach().clone()

                self.is_ready = True
                logger.info(f"✅ [Synaptic Intelligence] Berhasil memuat {len(self.pretrained_weights)} parameter pre-train sebagai referensi.")

            except Exception as e:
                logger.error(f"Gagal memuat bobot pre-train untuk Synaptic Intelligence: {e}", exc_info=True)

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer):
        """
        Setelah gradien dihitung, modifikasi gradien tersebut untuk menambahkan
        penalti "anti-lupa".
        """
        if not self.is_ready or not self.training:
            return

        for name, param in pl_module.named_parameters():
            if param.grad is not None and name in self.pretrained_weights:

                pretrained_param = self.pretrained_weights[name]
                

                weight_diff = param.data - pretrained_param
                


                penalty_gradient = self.strength * weight_diff
                

                param.grad.add_(penalty_gradient)

class HardSampleReplay_Callback(pl.Callback):
    """
    Callback untuk melakukan 'experience replay' pada sampel-sampel yang sulit.
    Ini memaksa model untuk berlatih ulang pada data yang sebelumnya menyebabkan
    ARMH terpicu, untuk melawan 'Catastrophic Forgetting Minor'.
    """

    def __init__(self, armh_callback: ARMH_Callback, replay_epochs: int = 2):
        super().__init__()
        self.armh_callback = armh_callback
        self.replay_epochs = replay_epochs
        logger.info(
            f"📚 HardSampleReplay_Callback aktif. Akan melatih ulang sampel sulit selama {replay_epochs} epoch mini."
        )

    def _clone_batch_for_replay(self, batch: Any) -> Any:
        """Clone batch untuk menghindari reuse tensor yang terhubung ke graph lama."""
        if isinstance(batch, torch.Tensor):
            return batch.detach().clone()
        if isinstance(batch, dict):
            return {k: self._clone_batch_for_replay(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            cloned = [self._clone_batch_for_replay(v) for v in batch]
            return type(batch)(cloned)
        return copy.deepcopy(batch)

    def _compute_loss_for_replay(self, pl_module, batch_on_device):
        """Hitung ulang loss untuk replay dengan membuat graph baru."""

        if hasattr(pl_module, "shared_step"):
            out = pl_module.shared_step(batch_on_device)
            if isinstance(out, dict) and "loss" in out:
                return out["loss"]


        if hasattr(pl_module, "compute_loss"):
            try:
                return pl_module.compute_loss(batch_on_device)
            except Exception:
                pass


        try:
            if hasattr(pl_module, "loss_fn") and hasattr(pl_module, "forward"):
                logits = pl_module.forward(**batch_on_device) if isinstance(batch_on_device, dict) else pl_module.forward(batch_on_device)
                if isinstance(logits, dict) and "logits" in logits:
                    logits = logits["logits"]
                labels = batch_on_device.get("labels") if isinstance(batch_on_device, dict) else None
                return pl_module.loss_fn(logits, labels)
        except Exception:
            pass


        cloned = self._clone_batch_for_replay(batch_on_device)
        out = pl_module.training_step(cloned, -1)
        if isinstance(out, dict) and "loss" in out:
            return out["loss"]
        return out

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Setelah satu epoch, lakukan replay ringan pada sampel yang ditandai sulit.
        Mengutamakan Lightning automatic_optimization; fallback memanggil training_step manual.
        """
        if not hasattr(self, "replay_buffer") or not self.replay_buffer:
            return

        try:
            optimizer = trainer.optimizers[0] if trainer.optimizers else None
            if optimizer is None:
                logger.warning("🎓 [Experience Replay] Optimizer tidak ditemukan. Melewati replay.")
                return


            replay_samples = random.sample(self.replay_buffer, k=min(len(self.replay_buffer), self.max_replay_per_epoch))

            for batch in replay_samples:

                batch_on_device = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)


                try:
                    loss_replay = self._compute_loss_for_replay(pl_module, batch_on_device)
                except Exception as e:
                    logger.error(f"🎓 [Experience Replay] Gagal menghitung loss replay: {e}")
                    continue

                if loss_replay is None or not isinstance(loss_replay, torch.Tensor):
                    continue

                if getattr(pl_module, "automatic_optimization", True):
                    optimizer.zero_grad(set_to_none=True)
                    loss_replay.backward()
                    optimizer.step()
                else:

                    cloned = self._clone_batch_for_replay(batch_on_device)
                    pl_module.training_step(cloned, -1)

            logger.warning("🎓 [Experience Replay] Sesi latihan tambahan selesai.")
        except Exception as e:
            logger.error(f"🎓 [Experience Replay] Error replay epoch-end: {e}", exc_info=True)

class CRYSTAL_Callback(pl.Callback):
    """
    CRYSTAL — Self-aware feature signature + projection-layer fast adaptation.
    - Self-check jumlah fitur aktif dari batch pertama.
    - Boost grad untuk layer proyeksi (input_proj/output_proj/feature_gate/proj*) di awal.
    - Simpan 'feature_signature' ke checkpoint.
    - (Opsional) Menulis bridge policy ringan ke datamodule.runtime_policies.
    """

    def __init__(
        self,
        proj_grad_scale: float = 2.5,
        boost_epochs: int = 2,
        autosuffix: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.proj_grad_scale = proj_grad_scale
        self.boost_epochs = max(1, int(boost_epochs))
        self.autosuffix = autosuffix
        self.verbose = verbose

        self.expected_features = None
        self.current_features = None
        self._first_batch_seen = False
        self._boost_until_step = 0


    @staticmethod
    def _infer_feature_count_from_batch(batch) -> int | None:
        """
        Cari tensor ber-bentuk (B, F, ...) dan ambil dimensi fitur terakhir (F).
        Robust untuk (X, y), dict, maupun list.
        """
        def pick_from(x):
            if hasattr(x, "shape") and len(x.shape) >= 2:

                return int(x.shape[-1])
            return None

        if isinstance(batch, (list, tuple)):
            for item in batch:
                fc = CRYSTAL_Callback._infer_feature_count_from_batch(item)
                if fc:
                    return fc
            return None
        if isinstance(batch, dict):

            for key in ("X", "inputs", "features"):
                if key in batch:
                    fc = pick_from(batch[key])
                    if fc:
                        return fc

            for v in batch.values():
                fc = CRYSTAL_Callback._infer_feature_count_from_batch(v)
                if fc:
                    return fc
            return None
        return pick_from(batch)


    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str | None = None):

        dm = getattr(trainer, "datamodule", None)
        if dm and hasattr(dm, "n_features_input"):
            self.expected_features = int(getattr(dm, "n_features_input"))
        elif hasattr(getattr(pl_module, "hparams", SimpleNamespace()), "n_features_input"):
            self.expected_features = int(pl_module.hparams.n_features_input)


        steps_per_epoch = getattr(trainer, "num_training_batches", None) or 0
        self._boost_until_step = self.boost_epochs * int(steps_per_epoch or 0)

        if self.verbose:
            logger.info(
                f"🔷 [CRYSTAL] init | expected_features={self.expected_features} | "
                f"proj_grad_scale={self.proj_grad_scale} | boost_epochs={self.boost_epochs} "
                f"(≈ {self._boost_until_step} steps)"
            )


        if dm is not None and not hasattr(dm, "runtime_policies"):
            dm.runtime_policies = {}

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch, batch_idx: int):

        if self._first_batch_seen:
            return
        self._first_batch_seen = True

        self.current_features = self._infer_feature_count_from_batch(batch)


        dm = getattr(trainer, "datamodule", None)
        if self.verbose:
            logger.info(
                f"🧭 [CRYSTAL] feature_signature@runtime = {self.current_features} "
                f"(expected={self.expected_features})"
            )


        if dm is not None and hasattr(dm, "runtime_policies"):
            dm.runtime_policies["feature_signature"] = int(self.current_features or -1)


        if self.expected_features and self.current_features and (self.current_features != self.expected_features):
            logger.warning(
                f"⚠️ [CRYSTAL] input_dim checkpoint={self.expected_features} ≠ current={self.current_features}. "
                f"Layer proyeksi baru akan dipercepat melalui grad-scale."
            )

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer):
        """
        Implementasi 'LR lebih besar untuk 3 parameter baru' tanpa menyentuh optimizer:
        gandakan grad untuk parameter proyeksi di awal pelatihan.
        """
        if trainer.global_step >= self._boost_until_step:
            return

        boosted = 0
        for name, p in pl_module.named_parameters():
            if p.grad is None:
                continue

            if any(tag in name for tag in ("input_proj", "output_proj", "feature_gate", ".proj", "projection")):
                p.grad.mul_(self.proj_grad_scale)
                boosted += 1

        if self.verbose and boosted and (trainer.global_step % 50 == 0):
            logger.info(
                f"🧭 [CRYSTAL] grad-scale ×{self.proj_grad_scale:.2f} aktif untuk {boosted} param proyeksi "
                f"(step {trainer.global_step}/{self._boost_until_step})."
            )

    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: dict):
        """
        Sisipkan signature fitur ke dalam checkpoint → run berikutnya bisa langsung cocok.
        (Tidak mengubah filename—aman untuk pipeline Anda.)
        """
        if not self.autosuffix:
            return checkpoint

        sig = int(self.current_features or (self.expected_features or -1))
        checkpoint.setdefault("crystal", {})
        checkpoint["crystal"]["feature_signature"] = sig
        return checkpoint

          

class PARC_Callback(pl.Callback):
    """
    Mengatur router temperature (τ) dan capacity factor (C) secara sadar-fase:
    - Warmup: τ dan C agak lebih tinggi untuk eksplorasi & load leveling.
    - Mid:   stabil di baseline.
    - Late:  τ dan C diturunkan bertahap untuk mempertegas pemilihan expert & efisiensi.
    Aman di-CPU dan defensif (hanya set atribut yang memang ada).
    """

    def __init__(
        self,
        warmup_frac: float = 0.15,
        mid_frac: float = 0.70,
        base_tau: float = 1.00,
        min_tau: float = 0.70,
        max_tau: float = 1.30,
        base_C: float = 1.00,
        min_C: float = 0.90,
        max_C: float = 1.30,
        util_target: float = 0.55,
        util_tolerance: float = 0.05,
        ema_alpha: float = 0.05,
        verbose: bool = True,
    ):
        super().__init__()
        self.warmup_frac = warmup_frac
        self.mid_frac = mid_frac
        self.base_tau, self.min_tau, self.max_tau = base_tau, min_tau, max_tau
        self.base_C,   self.min_C,   self.max_C   = base_C,   min_C,   max_C
        self.util_target = util_target
        self.util_tol = util_tolerance
        self.ema_alpha = ema_alpha
        self.verbose = verbose

        self.total_epochs = None
        self._moe_layers = []
        self._ema_util = None
        self._last_set = {"tau": None, "C": None}


    def _iter_moe_layers(self, pl_module):

        candidate_names = {"MoELayer", "SparseMoE", "MoE", "GShardMoE", "SwitchLayer"}
        for m in pl_module.modules():
            if type(m).__name__ in candidate_names or any(
                hasattr(m, k) for k in ("top_k", "router_temperature", "capacity_factor")
            ):
                yield m

    @staticmethod
    def _set_if_exists(obj, names, value):
        for n in names:
            if hasattr(obj, n):
                try:
                    setattr(obj, n, type(getattr(obj, n))(value))
                except Exception:
                    setattr(obj, n, value)
                return True
        return False

    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _apply_to_layers(self, tau, C):
        applied_any = False
        for layer in self._moe_layers:
            a = self._set_if_exists(layer, ["router_temperature", "temperature", "tau"], tau)
            b = self._set_if_exists(layer, ["capacity_factor", "capacity", "C", "capacity_factor_train"], C)
            applied_any = applied_any or a or b
        return applied_any


    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        if stage != "fit":
            return
        self.total_epochs = trainer.max_epochs or 1
        self._moe_layers = list(self._iter_moe_layers(pl_module))
        if self.verbose:
            logger.info(f"🧭 [PARC] Terdeteksi {len(self._moe_layers)} layer MoE/gating untuk diatur.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        util_vals = []
        for layer in self._moe_layers:

            for attr in ("last_utilization", "gate_utilization", "expert_utilization", "avg_utilization"):
                val = getattr(layer, attr, None)
                if val is not None:
                    try:
                        util_vals.append(float(val))
                        break
                    except Exception:
                        pass
        if util_vals:
            cur = sum(util_vals) / len(util_vals)
            if self._ema_util is None:
                self._ema_util = cur
            else:
                self._ema_util = (1 - self.ema_alpha) * self._ema_util + self.ema_alpha * cur

    def on_train_epoch_start(self, trainer, pl_module):

        e = trainer.current_epoch
        T = max(1, self.total_epochs - 1)
        p = e / T

        if p < self.warmup_frac:

            t = p / max(1e-8, self.warmup_frac)
            tau = self.max_tau + (self.base_tau - self.max_tau) * t
            C   = self.max_C   + (self.base_C   - self.max_C)   * t
            phase = "warmup"
        elif p < self.mid_frac:
            tau, C = self.base_tau, self.base_C
            phase = "mid"
        else:

            t = (p - self.mid_frac) / max(1e-8, (1 - self.mid_frac))
            tau = self.base_tau + (self.min_tau - self.base_tau) * t
            C   = self.base_C   + (self.min_C   - self.base_C)   * t
            phase = "late"


        if self._ema_util is not None:
            if self._ema_util > self.util_target + self.util_tol:

                tau *= 0.98
                C   *= 0.98
            elif self._ema_util < self.util_target - self.util_tol:

                tau *= 1.02
                C   *= 1.02

        tau = self._clamp(tau, self.min_tau, self.max_tau)
        C   = self._clamp(C,   self.min_C,   self.max_C)

        changed = (tau != self._last_set["tau"]) or (C != self._last_set["C"])
        if changed:
            applied = self._apply_to_layers(tau, C)
            if applied and self.verbose:
                logger.info(f"🧭 [PARC] epoch {e+1}/{self.total_epochs} | phase={phase} → set τ={tau:.3f}, C={C:.2f}")
            self._last_set["tau"] = tau
            self._last_set["C"] = C

    def on_fit_end(self, trainer, pl_module):

        try:
            if "ensure_parquet_source" in self._orig and self._orig["ensure_parquet_source"] is not None:
                globals()["ensure_parquet_source"] = self._orig["ensure_parquet_source"]
            if "generate_chaos_theory_features" in self._orig and self._orig["generate_chaos_theory_features"] is not None:
                globals()["generate_chaos_theory_features"] = self._orig["generate_chaos_theory_features"]
        except Exception:
            pass


        try:
            import sklearn.impute as _imp
            if "KNNImputer" in self._orig and self._orig["KNNImputer"] is not None:
                _imp.KNNImputer = self._orig["KNNImputer"]
        except Exception:
            pass

        if self.verbose:
            logger.info("🔧 [Preproc] Monkey-patch dipulihkan (ensure_parquet_source, chaos, KNNImputer).")



class ChaosPreprocAccelerator(pl.Callback):
    """
    Mengakselerasi pre-processing tanpa mengubah arsitektur:
    - Patch ensure_parquet_source: downcast numerik + Parquet ZSTD.
    - Patch generate_chaos_theory_features: rolling raw=True + min_periods + cache Parquet.
    - Patch KNNImputer (opsional): auto-switch ke median imputer jika fitur jadi sangat besar.
    Semua patch dipasang pada on_fit_start lalu berlaku global di modul ini.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        use_zstd: bool = True,
        zstd_level: int = 3,
        imputer_threshold: int | None = 1_000_000,
        default_window: int = 100,
        verbose: bool = True,
        tz: str = "UTC",
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.use_zstd = use_zstd
        self.zstd_level = zstd_level
        self.imputer_threshold = imputer_threshold
        self.default_window = default_window
        self.verbose = verbose
        self.tz = tz


        self._orig = {}


    @staticmethod
    def _downcast_df(df: pd.DataFrame) -> pd.DataFrame:

        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = pd.to_numeric(df[c], downcast="float")
        for c in df.select_dtypes(include=["int64"]).columns:
            df[c] = pd.to_numeric(df[c], downcast="integer")
        return df

    @staticmethod
    def _hash_key(df: pd.DataFrame, tickers: list, window: int) -> str:
        import hashlib
        try:
            n = len(df)
            tmin = pd.to_datetime(df.index.min()).value if len(df) else 0
            tmax = pd.to_datetime(df.index.max()).value if len(df) else 0
            cols_sig = hash(tuple(df.columns))
            meta = f"{n}|{tmin}|{tmax}|{','.join(sorted(map(str, tickers)))}|{window}|{cols_sig}"
        except Exception:
            meta = f"{len(df)}|{window}|{','.join(sorted(map(str, tickers)))}"
        return hashlib.md5(meta.encode()).hexdigest()


    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.cache_dir.mkdir(parents=True, exist_ok=True)



        self._orig["ensure_parquet_source"] = globals().get("ensure_parquet_source")

        def ensure_parquet_source_fast(csv_path: str, parquet_path: str, index_col: str = "Date", tz: str = self.tz):
            parquet_path = str(parquet_path)
            if os.path.exists(parquet_path):
                return parquet_path
            df = pd.read_csv(csv_path)
            if index_col in df.columns:
                ts = pd.to_datetime(df[index_col], errors="coerce", utc=True)
                if tz and tz.upper() != "UTC":
                    try:
                        ts = ts.dt.tz_convert(tz)
                    except Exception:

                        ts = ts.dt.tz_localize("UTC").dt.tz_convert(tz)
                df = df.set_index(ts).drop(columns=[index_col], errors="ignore")
            df = self._downcast_df(df)
            pq_kwargs = dict(engine="pyarrow")
            if self.use_zstd:
                pq_kwargs["compression"] = "zstd"
            df.to_parquet(parquet_path, **pq_kwargs)
            if self.verbose:
                logger.info(f"🧊[Preproc] Parquet ditulis (ZSTD={self.use_zstd}) → {parquet_path}")
            return parquet_path

        globals()["ensure_parquet_source"] = ensure_parquet_source_fast




        self._orig["generate_chaos_theory_features"] = globals().get("generate_chaos_theory_features")

        def generate_chaos_theory_features_fast(df: pd.DataFrame, tickers: list, window: int = self.default_window) -> pd.DataFrame:
            key = self._hash_key(df, tickers, window)
            cache_fp = self.cache_dir / f"chaos_w{window}_{key}.parquet"

            if cache_fp.exists():
                try:
                    out = pd.read_parquet(cache_fp)
                    if self.verbose:
                        logger.info(f"🌀[ChaosCache] HIT → {cache_fp.name}")
                    return out
                except Exception:
                    pass

            chaos_df = pd.DataFrame(index=df.index)
            for ticker in tickers:
                close_col = f"{ticker}_Close"
                if close_col not in df.columns:
                    continue
                logret = np.log(df[close_col] / df[close_col].shift(1))
                roll = logret.rolling(window=window, min_periods=window)
                try:
                    lyap = roll.apply(lambda x: nolds.lyap_r(np.asarray(x, dtype=float)), raw=True)
                except Exception:
                    lyap = pd.Series(index=df.index, dtype="float32")
                try:
                    corr = roll.apply(lambda x: nolds.corr_dim(np.asarray(x, dtype=float), emb_dim=5), raw=True)
                except Exception:
                    corr = pd.Series(index=df.index, dtype="float32")
                chaos_df[f"chaos_lyap_r_{ticker}"] = lyap
                chaos_df[f"chaos_corr_dim_{ticker}"] = corr

            try:
                pq_kwargs = dict(engine="pyarrow")
                if self.use_zstd:
                    pq_kwargs["compression"] = "zstd"
                tmp_fp = cache_fp.with_suffix(cache_fp.suffix + ".tmp")
                chaos_df.to_parquet(tmp_fp, **pq_kwargs)
                os.replace(tmp_fp, cache_fp)
                if self.verbose:
                    logger.info(f"🌀[ChaosCache] SAVE → {cache_fp.name}")
            except Exception:
                pass

            logger.info(f"✅ Fitur Chaos disiapkan untuk {len(tickers)} ticker (window={window}).")
            return chaos_df

        globals()["generate_chaos_theory_features"] = generate_chaos_theory_features_fast



        if self.imputer_threshold is not None:
            try:
                import sklearn.impute as _imp
                OriginalKNN = _imp.KNNImputer
                from sklearn.impute import SimpleImputer

                class SmartKNNImputer(OriginalKNN):
                    def fit_transform(self, X, y=None):
                        try:
                            n = int(X.shape[0]) * int(X.shape[1])
                        except Exception:
                            n = 0
                        if n >= self.imputer_threshold:
                            if self.verbose:
                                logger.info(f"⚡[FAST Imputer] Matrix={X.shape} >= {self.imputer_threshold:,} → median-impute")
                            return SimpleImputer(strategy="median").fit_transform(X)
                        return super().fit_transform(X)

                _imp.KNNImputer = SmartKNNImputer
                if self.verbose:
                    logger.info("⚙️ [Preproc] SmartKNNImputer aktif (auto median untuk matriks besar).")
            except Exception:
                if self.verbose:
                    logger.warning("Tidak bisa mengaktifkan SmartKNNImputer; lanjut default KNN.")


class TPEGovernorCallback(pl.Callback):
    """
    Top-k Probabilistic Elastic (TPE) Governor untuk MoE:
    - Mendaftarkan forward-hook pada setiap MoELayer.gating_network
      untuk menghitung pemakaian expert (berdasarkan top-k logits).
    - Menyetel `layer.top_k` secara dinamis berbasis EMA pemakaian expert
      + ukuran keragaman (effective number of experts).
    """

    def __init__(
        self,
        util_target: float = 0.55,
        adjust_every: int = 200,
        ema_alpha: float = 0.05,
        k_min: int = 1,
        k_max: int | None = None,
        warmup_steps: int = 400,
        verbose: bool = True,
        bridge_policy: bool = True,
    ):
        super().__init__()
        self.util_target = util_target
        self.adjust_every = adjust_every
        self.ema_alpha = ema_alpha
        self.k_min = k_min
        self.k_max = k_max
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.bridge_policy = bridge_policy


        self._handles = []
        self._layers = []
        self._ema_usage = {}
        self._global_step = 0
        self._policy_applied = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        if stage != "fit":
            return


        from types import SimpleNamespace
        moe_layers = []
        for name, module in pl_module.named_modules():

            if hasattr(module, "gating_network") and hasattr(module, "experts"):

                moe_layers.append((name, module))

        self._layers = [m for _, m in moe_layers]


        for lname, layer in moe_layers:
            n_exp = getattr(layer, "n_experts", None)
            if n_exp is None:

                try:
                    n_exp = layer.gating_network.out_features
                except Exception:
                    continue

            self._ema_usage[layer] = torch.zeros(n_exp, dtype=torch.float32)

            def _collect(_mod, _inp, _out, owner=layer, n_experts=n_exp, selfref=self):

                with torch.no_grad():
                    logits = _out
                    last_dim = logits.shape[-1]
                    if last_dim != n_experts:

                        return
                    k = int(max(1, min(getattr(owner, "top_k", 1), n_experts)))

                    top_idx = torch.topk(logits, k=k, dim=-1).indices

                    flat_idx = top_idx.reshape(-1)
                    counts = torch.bincount(flat_idx, minlength=n_experts).float()
                    total = counts.sum().clamp_min(1.0)
                    p = counts / total

                    ema = selfref._ema_usage[owner]
                    ema.mul_(1.0 - selfref.ema_alpha).add_(selfref.ema_alpha * p)

            h = layer.gating_network.register_forward_hook(_collect)
            self._handles.append(h)

        if self.verbose and self._layers:
            layer_names = ", ".join([type(l).__name__ for l in self._layers])
            logger.info(f"🧭 [TPE] Hook aktif pada {len(self._layers)} MoELayer: {layer_names}")

    def teardown(self, trainer, pl_module, stage: str):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._global_step += 1
        if (self._global_step < self.warmup_steps) or (self._global_step % self.adjust_every != 0):
            return



        if getattr(self, "bridge_policy", True) and not getattr(self, "_policy_applied", False):
            try:
                dm = getattr(trainer, "datamodule", None)
                pol = getattr(dm, "runtime_policies", None) if dm is not None else None
                if isinstance(pol, dict) and ("feature_signature" in pol):
                    sig = pol["feature_signature"]
                    n_active = int(sig.get("n_active", sig)) if isinstance(sig, dict) else int(sig)



                    if n_active <= 12:
                        k_cap = 1
                    elif n_active <= 24:
                        k_cap = 2
                    elif n_active <= 48:
                        k_cap = 3
                    else:
                        k_cap = (self.k_max if self.k_max is not None else 4)

                    old_min = self.k_min
                    old_max = (self.k_max if self.k_max is not None else k_cap)
                    self.k_max = min(k_cap, old_max) if self.k_max is not None else k_cap
                    self.k_min = min(self.k_min, self.k_max)

                    if getattr(self, "verbose", False):
                        logger.info(
                            f"🧭 [TPE] policy-bridge: feature_signature={n_active} → clamp k∈[{self.k_min},{self.k_max}] "
                            f"(was [{old_min},{old_max}])"
                        )

                    self._policy_applied = True
            except Exception as e:
                if getattr(self, "verbose", False):
                    logger.warning(f"🧭 [TPE] policy-bridge gagal dibaca/dipakai: {e}")




        for layer in self._layers:
            p = self._ema_usage.get(layer, None)
            if p is None or p.numel() == 0:
                continue


            denom = (p.pow(2).sum().item() + 1e-8)
            neff = max(1.0, 1.0 / denom)

            k_now = int(max(1, getattr(layer, "top_k", 1)))
            k_hi = int(self.k_max if self.k_max is not None else max(k_now, 1))
            k_hi = max(k_hi, k_now)

            k_suggest = int(max(self.k_min, min(round(self.util_target * neff), k_hi)))


            if k_suggest > k_now:
                k_new = min(k_now + 1, k_suggest, k_hi)
            elif k_suggest < k_now:
                k_new = max(k_now - 1, k_suggest, self.k_min)
            else:
                k_new = k_now

            if k_new != k_now:
                try:
                    old = layer.top_k
                    layer.top_k = int(k_new)
                    if self.verbose:
                        logger.info(f"🧭 [TPE] {type(layer).__name__}: top_k {old} → {layer.top_k} | Neff≈{neff:.2f}")
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"[TPE] gagal set top_k: {e}")



class BlacklistLRScheduler(pl.Callback):
    """
    Menerapkan learning rate yang lebih rendah secara dinamis khusus untuk batch
    yang ada di dalam daftar hitam (blacklist).
    """

    def __init__(self, blacklist_path: Path, reduction_factor: float = 0.1):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.original_lr = None
        self.blacklisted_indices = set()

        if blacklist_path and blacklist_path.exists():
            with open(blacklist_path, "r") as f:
                self.blacklisted_indices = set(json.load(f))
            logger.info(
                f"🩺 BlacklistLRScheduler aktif dengan {len(self.blacklisted_indices)} indeks 'toxic'. LR akan dikurangi {reduction_factor*100}%."
            )
        else:
            logger.info(
                "🩺 BlacklistLRScheduler aktif tetapi tidak ada file daftar hitam yang ditemukan."
            )

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch,
        batch_idx: int,
    ):








        if batch_idx in self.blacklisted_indices:
            optimizer = trainer.optimizers[0]
            self.original_lr = optimizer.param_groups[0]["lr"]
            new_lr = self.original_lr * self.reduction_factor
            optimizer.param_groups[0]["lr"] = new_lr
            logger.warning(
                f"Batch {batch_idx} ada di daftar hitam. LR sementara diturunkan ke {new_lr:.7f}"
            )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ):

        if self.original_lr is not None:
            optimizer = trainer.optimizers[0]
            optimizer.param_groups[0]["lr"] = self.original_lr
            self.original_lr = None


def analyze_and_blacklist_batches(
    armh_log: list, project_id: str, threshold: int = 2
) -> Path:
    """# <-- PERBAIKAN: Seluruh isi fungsi diberi indentasi
    Menganalisis log ARMH global, mengidentifikasi sampel yang berulang kali bermasalah,
    dan menyimpan "daftar hitam" (blacklist) ke file.

    Args:
        armh_log (list): Log dari `armh_cb.global_problem_log`.
        project_id (str): ID proyek untuk penamaan file.
        threshold (int): Berapa kali sampel harus muncul untuk dianggap "toksik".

    Returns:
        Path: Path ke file blacklist yang telah disimpan.
    """
    logger.info(
        "\n--- 🕵️‍ Menganalisis Log Investigasi ARMH untuk Membuat Daftar Hitam ---"
    )
    if not armh_log:
        logger.warning(
            "Log investigasi kosong. Tidak ada daftar hitam yang dibuat.")
        return None

    df_log = pd.DataFrame(armh_log)

    reoccurrence_counts = df_log["sample_index"].value_counts()


    toxic_indices = reoccurrence_counts[reoccurrence_counts > threshold].index.tolist(
    )

    if not toxic_indices:
        logger.info(
            "✅ Tidak ada sampel yang berulang kali bermasalah. Tidak ada daftar hitam yang dibuat."
        )
        return None

    logger.warning(
        f'Ditemukan {len(toxic_indices)} sampel "toxic" yang berulang kali menyebabkan masalah.'
    )

    blacklist_path = (
        get_path(project_id, "checkpoint_dir")
        / f"toxic_batch_blacklist_{project_id}.json"
    )

    with open(blacklist_path, "w") as f:
        json.dump(toxic_indices, f)

    logger.info(f"✅ Daftar hitam sampel 'toxic' disimpan di: {blacklist_path}")
    return blacklist_path

class AdaptiveGradientClipping(pl.Callback):
    """
    Mengelola gradient clipping secara dinamis berdasarkan persentil dari norma gradien historis.
    Ini adalah pendekatan yang lebih canggih daripada threshold statis.
    """
    def __init__(self, history_size: int = 100, percentile: float = 90.0):
        super().__init__()

        self.grad_norm_history = deque(maxlen=history_size)
        self.percentile = percentile
        logger.info(f"🦾 [Adaptive Clipping] Aktif. Menganalisis {history_size} riwayat gradien, kliping di persentil ke-{percentile}.")

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        params_with_grad = [p for p in pl_module.parameters() if p.grad is not None]
        if not params_with_grad:
            return

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2
        ).item()
        

        if total_norm < 100.0:
            self.grad_norm_history.append(total_norm)


        if len(self.grad_norm_history) < 20:

            clip_value = 1.0
        else:

            clip_value = np.percentile(list(self.grad_norm_history), self.percentile)
        

        torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=clip_value)


        pl_module.log("dynamic_clip_value", clip_value, on_step=True, on_epoch=False, prog_bar=True)

class AdaptiveCapacityScheduler(pl.Callback):
    """
    Menjadikan 'kapasitas' model berdetak (low_frac..high_frac) per-batch/epoch
    tanpa mengubah arsitektur. Scheduler ini hanya mengeset fraksi kapasitas
    ke LightningModule lewat:
      - pl_module.set_capacity_frac(frac), jika method itu ada; ATAU
      - pl_module.current_capacity_frac = frac (fallback aman).

    Opsional: skala LR ~ sqrt(frac) agar stabil saat kapasitas mengecil/membesar.

    Backward-compat:
      - Menerima argumen lama: patience, danger_zone, confirmation_patience, supervisor_agent.
        (Jika dua nama 'patience' & 'confirmation_patience' sama-sama diberi, yang dipakai: confirmation_patience.)
    """
    def __init__(
        self,

        low_frac: float = 0.25,
        high_frac: float = 1.0,
        schedule: str = "cosine",
        warmup_epochs: int = 1,
        cooldown_epochs: int = 0,
        step_interval: int = 1,
        eval_full_capacity: bool = True,
        scale_lr: bool = True,
        verbose: bool = True,


        monitor: str | None = None,
        delta: float = 0.01,
        danger_zone: float | None = None,
        confirmation_patience: int | None = None,
        patience: int | None = None,
        supervisor_agent=None,
    ):
        super().__init__()

        self.low = float(max(0.10, min(1.0, low_frac)))
        self.high = float(max(self.low, min(1.0, high_frac)))
        self.schedule = str(schedule).lower()
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.cooldown_epochs = int(max(0, cooldown_epochs))
        self.step_interval = int(max(1, step_interval))
        self.eval_full_capacity = bool(eval_full_capacity)
        self.scale_lr = bool(scale_lr)
        self.verbose = bool(verbose)


        self.monitor = monitor
        self.delta = float(delta)
        self.danger_zone = danger_zone
        if confirmation_patience is None and patience is not None:
            confirmation_patience = patience
        self.confirmation_patience = int(confirmation_patience) if confirmation_patience is not None else None
        self.supervisor_agent = supervisor_agent


        self._steps_per_epoch = 1
        self._batch_in_epoch = 0
        self._last_update_gstep = -1
        self._base_lrs = None



    def _set_frac(self, pl_module, frac: float):
        frac = float(max(self.low, min(self.high, frac)))
        if hasattr(pl_module, "set_capacity_frac") and callable(getattr(pl_module, "set_capacity_frac")):
            pl_module.set_capacity_frac(frac)
        else:

            setattr(pl_module, "current_capacity_frac", frac)
        return frac

    def _maybe_rescale_lrs(self, trainer, frac: float):
        if not self.scale_lr:
            return
        import math
        if not trainer.optimizers:
            return
        if self._base_lrs is None:

            self._base_lrs = []
            for optim in trainer.optimizers:
                self._base_lrs.append([g.get("lr", 0.0) for g in optim.param_groups])
        scale = math.sqrt(max(1e-6, float(frac)))
        for optim, base_groups in zip(trainer.optimizers, self._base_lrs):
            for g, base_lr in zip(optim.param_groups, base_groups):
                g["lr"] = base_lr * scale

    def _compute_steps_per_epoch(self, trainer):

        n = getattr(trainer, "num_training_batches", None)
        if isinstance(n, int) and n > 0:
            return n
        est = getattr(trainer, "estimated_stepping_batches", None)
        max_epochs = getattr(getattr(trainer, "fit_loop", None), "max_epochs", getattr(trainer, "max_epochs", 1) or 1)
        if isinstance(est, int) and est > 0:
            return max(1, est // max(1, int(max_epochs)))
        return 1

    def _sched_value(self, epoch: int, batch_in_epoch: int):

        if epoch < self.warmup_epochs:
            return self.high


        if self.cooldown_epochs > 0 and hasattr(self, "_max_epochs"):
            if epoch >= max(0, self._max_epochs - self.cooldown_epochs):
                return self.high


        import math
        denom = max(1, self._steps_per_epoch - 1)
        t = float(batch_in_epoch) / float(denom)

        if self.schedule == "cosine":

            return self.low + (self.high - self.low) * 0.5 * (1.0 - math.cos(2.0 * math.pi * t))
        elif self.schedule == "triangle":

            tri = 1.0 - abs(2.0 * t - 1.0)
            return self.low + (self.high - self.low) * tri
        elif self.schedule == "sawtooth":

            return self.low + (self.high - self.low) * t
        else:

            return self.high


    def on_fit_start(self, trainer, pl_module):

        self._max_epochs = getattr(getattr(trainer, "fit_loop", None), "max_epochs", getattr(trainer, "max_epochs", None))

        self._maybe_rescale_lrs(trainer, frac=self.high)

        f0 = self._set_frac(pl_module, self.high)
        if self.verbose:
            try:
                pl_module.print(f"[ACS] start: capacity_frac={f0:.3f} (low={self.low:.2f}, high={self.high:.2f}, schedule={self.schedule})")
            except Exception:
                pass

    def on_train_epoch_start(self, trainer, pl_module):
        self._steps_per_epoch = self._compute_steps_per_epoch(trainer)
        self._batch_in_epoch = 0

        f = self._sched_value(trainer.current_epoch, self._batch_in_epoch)
        f = self._set_frac(pl_module, f)
        self._maybe_rescale_lrs(trainer, f)
        if self.verbose and trainer.current_epoch == self.warmup_epochs:
            try:
                pl_module.print(f"[ACS] epoch{trainer.current_epoch}: steps/epoch={self._steps_per_epoch}")
            except Exception:
                pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        gstep = int(getattr(trainer, "global_step", 0))
        if (gstep - self._last_update_gstep) >= self.step_interval:
            f = self._sched_value(trainer.current_epoch, self._batch_in_epoch)
            f = self._set_frac(pl_module, f)
            self._maybe_rescale_lrs(trainer, f)
            self._last_update_gstep = gstep
            if self.verbose and (gstep % (self.step_interval * 50) == 0):
                try:
                    pl_module.print(f"[ACS] step{gstep}: capacity_frac={f:.3f}")
                except Exception:
                    pass

        self._batch_in_epoch += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.eval_full_capacity:
            f = self._set_frac(pl_module, 1.0)
            self._maybe_rescale_lrs(trainer, f)
            if self.verbose:
                try:
                    pl_module.print("[ACS] validation: full capacity (1.0)")
                except Exception:
                    pass

    def on_test_epoch_start(self, trainer, pl_module):
        if self.eval_full_capacity:
            f = self._set_frac(pl_module, 1.0)
            self._maybe_rescale_lrs(trainer, f)
            if self.verbose:
                try:
                    pl_module.print("[ACS] test: full capacity (1.0)")
                except Exception:
                    pass
                
class CognitiveResonanceCallback(pl.Callback):
    """
    Menyuntikkan 'kebijaksanaan' dari prinsip abstrak yang tersimpan di NSMM
    ke dalam model AlphaLit secara real-time selama pelatihan.
    """

    def __init__(self, nsmm: NSMM, embedding_model: "APIEmbedder", top_k: int = 3):
        super().__init__()
        self.nsmm = nsmm
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.wisdom_embedding = None
        logger.info("🧠⚡️ Cognitive Resonance Callback aktif.")

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        """Di awal setiap epoch, ambil kebijaksanaan yang relevan."""
        try:

            _, vitals_summary = pl_module.get_neuro_vitals()


            relevant_principles_text = self.nsmm.query_similar_principles(
                vitals_summary, self.embedding_model, self.top_k
            )

            if relevant_principles_text:

                wisdom_vector = self.embedding_model.encode(
                    relevant_principles_text, task_type="passage"
                )

                self.wisdom_embedding = (
                    torch.tensor(wisdom_vector, dtype=torch.float32)
                    .mean(dim=0)
                    .to(pl_module.device)
                )
            else:
                self.wisdom_embedding = None

        except Exception as e:
            logger.warning(
                f"[CognitiveResonance] Gagal mengambil kebijaksanaan: {e}")
            self.wisdom_embedding = None

    def get_current_wisdom(self, d_model: int, device: torch.device) -> torch.Tensor:
        """Menyediakan wisdom embedding ke model."""
        if self.wisdom_embedding is not None:
            return self.wisdom_embedding
        else:

            return torch.zeros(d_model, device=device)

class AdaptiveStabilityLayer(nn.Module):
    """
    Protokol 'Singularity' v3.2 (Final Optimized): 
    Lapisan pertahanan numerik adaptif dengan:
      - LayerNorm berbasis feature_dim
      - Stabilisasi log adaptif via gating (bisa dinonaktifkan & di-detach)
      - Deteksi NaN/Inf + logging terkontrol
      - Clamping ekstrem untuk keamanan
    """
    def __init__(self, feature_dim: int, clip_value: float = 15.0, adaptive_gate: bool = True):
        super().__init__()
        self.clip_value = clip_value
        self.norm = nn.LayerNorm(feature_dim)
        self.adaptive_gate = adaptive_gate

        if adaptive_gate:
            self.log_gate = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 4),
                nn.SiLU(),
                nn.Linear(feature_dim // 4, feature_dim),
                nn.Sigmoid()
            )
        else:

            self.register_buffer("log_gate_const", torch.tensor(0.1))


        self.register_buffer('nan_count', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not torch.isfinite(x).all():
            self.nan_count += 1
            count = int(self.nan_count.item())
            if count == 1 or count % 100 == 0:
                logger.warning(f"🛡️ [AdaptiveStabilityLayer] NaN/inf terdeteksi! Intervensi ke-{count}.")
            x = torch.nan_to_num(x, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)



        safe_abs = torch.clamp(torch.abs(x), max=1e6)
        log_stabilized = torch.log1p(safe_abs) * torch.sign(x)

        if self.adaptive_gate:
            gate = self.log_gate(x).detach()
        else:
            gate = self.log_gate_const


        x = gate * log_stabilized + (1 - gate) * x


        x = self.norm(x)
        return torch.clamp(x, -self.clip_value, self.clip_value)


class ConformerBlock(nn.Module):
    """Satu blok Conformer yang menggabungkan CNN dan Self-Attention."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()

        self.ffn1 = nn.Linear(d_model, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, d_model)



        self.norm_conv = nn.LayerNorm(d_model)


        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=15, padding="same"),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yang diperbaiki dengan pola pre-LayerNorm standar.
        """

        residual = x
        x = self.norm1(x)
        x = self.ffn2(self.dropout(F.relu(self.ffn1(x))))
        x = residual + self.dropout(x)


        residual = x
        x = self.norm2(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = residual + self.dropout(attn_out)


        residual = x

        x = self.norm_conv(x)
        x = x.transpose(1, 2)
        x = self.conv_module(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)


        residual = x
        x = self.norm2(x)
        x = self.ffn2(self.dropout(F.relu(self.ffn1(x))))
        x = residual + self.dropout(x)

        return x


class Menshin_Skytree_Core(nn.Module):
    """
    Pilar penstabil pusat yang terinspirasi dari menara Skytree.
    Bekerja secara paralel dengan alur utama untuk menyediakan sinyal stabil.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(d_model, n_heads,
                               ffn_dim=d_model * 4, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        logger.info(
            f"🗼 Lapisan Inti 'Menshin Skytree' (Conformer) diaktifkan dengan {n_layers} blok."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mengekstrak sinyal yang sangat stabil dari input."""
        for layer in self.layers:
            x = layer(x)
        return x

class UnifiedFusionHub(nn.Module):
    """
    Menggabungkan embedding dari berbagai jalur spesialis (Wave, Particle, Symbolic)
    menggunakan cross-attention untuk menghasilkan satu representasi terpadu.
    """
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, primary_embedding, specialist_embeddings):



        query = primary_embedding.unsqueeze(1)
        

        attended_context, _ = self.attention(query, specialist_embeddings, specialist_embeddings)
        

        fused = self.norm1(query + attended_context)
        

        fused = self.norm2(fused + self.ffn(fused))
        
        return fused.squeeze(1)


class ReturnLoss(Metric):
    is_differentiable = True
    higher_is_better = False

    def __init__(self, da_weight=0.3, horizon=7):
        super().__init__()
        self.base_da_weight = da_weight
        self.mse = MeanSquaredError()
        weights = torch.linspace(1.5, 0.5, horizon)
        self.register_buffer("weights", weights.view(1, horizon, 1))
        self.loss = torch.tensor(0.0)


        self.da_weight_multiplier = 1.0
        self.mse_penalty_multiplier = 1.0

    def update_aset_protocol(self, da_multiplier: float, mse_multiplier: float):
        """Metode untuk dipanggil oleh ASeT_ProtocolMonitor."""
        self.da_weight_multiplier = da_multiplier
        self.mse_penalty_multiplier = mse_multiplier
        logger.warning(
            f"🚨 PROTOKOL ASET: ReturnLoss diubah -> DA Multiplier: {da_multiplier}, MSE Multiplier: {mse_multiplier}"
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, anomaly_signal: torch.Tensor
    ):
        """
        MODIFIKASI: Menerima sinyal anomali untuk menyesuaikan bobot loss.
        `anomaly_signal` adalah tensor dengan nilai antara 0 (normal) dan 1 (anomali tinggi).
        """
        anomaly_signal = anomaly_signal.view(-1, 1, 1)


        dynamic_da_weight = self.base_da_weight * (1 - anomaly_signal)
        mse_penalty = 1.0 + (1.5 * anomaly_signal)


        final_da_weight = dynamic_da_weight * self.da_weight_multiplier
        final_mse_penalty = mse_penalty * self.mse_penalty_multiplier

        weighted_mse = torch.mean(
            final_mse_penalty *
            self.weights.to(preds.device) * (preds - target) ** 2
        )
        




        directional_accuracy_loss = torch.mean(final_da_weight * (
            1 - torch.mean((torch.sign(preds) == torch.sign(target)).float())
        ))


        self.loss = weighted_mse + directional_accuracy_loss

    def compute(self):
        return self.loss



def hyperparam_search_loop(
    *,
    initial_hparams: dict,
    max_rounds: int,
    train_eval_fn: Callable[[dict], tuple[float, dict, str, dict]],
    tutor: "HyperparameterTutor",
    council: "HyperparameterCouncil",
    accreditor: "ProposalAccreditor",
    metric_name: str = "val_loss",
    improve_tol: float = 0.001,
    tracker: "ExperimentTracker",
) -> tuple[dict, list[dict]]:
    """
    Menjalankan loop optimisasi hyperparameter dengan alur kerja multi-agen:
    1. Tutor -> 2. Dewan AI -> 3. Accreditor -> 4. Pelatihan.
    """
    from copy import deepcopy

    best_hparams = deepcopy(initial_hparams)
    logger.info("\n--- MENJALANKAN PUTARAN AWAL (BASELINE) ---")


    score, full_metrics, checkpoint_path, data_summary = train_eval_fn(
        best_hparams)

    tracker.log(
        project_id=best_hparams["project_id"],
        round_idx=0,
        params=best_hparams,
        metrics=full_metrics,
    )
    history = [
        {
            "round": 0,
            "score": score,
            "full_metrics": full_metrics,
            "hparams": deepcopy(best_hparams),
            "accepted": True,
            "checkpoint_path": checkpoint_path,
            "data_summary": data_summary,
        }
    ]
    best_score = score
    logger.info(f"--- SKOR AWAL ({metric_name}): {best_score:.6f} ---")

    for r in range(1, max_rounds + 1):
        logger.info(
            f"\n{'='*25} MEMULAI PUTARAN OPTIMISASI #{r}/{max_rounds} {'='*25}")

        logger.info("  [1/4] Tutor AI sedang merancang proposal awal...")
        tutor_proposal, err = tutor.design_new_strategy(
            failed_metrics={metric_name: best_score}, old_hparams=best_hparams
        )

        if err or tutor_proposal is None:
            reason = f"TutorError: {err}"
            logger.error(
                f"  Gagal mendapatkan proposal dari Tutor. Alasan: {reason}")
            history.append(
                {
                    "round": r,
                    "score": None,
                    "full_metrics": None,
                    "hparams": None,
                    "accepted": False,
                    "checkpoint_path": None,
                    "reason": reason,
                }
            )
            break

        logger.info(f"  Proposal awal dari Tutor diterima.")

        logger.info("  [2/4] Dewan AI sedang melakukan musyawarah...")
        history_context = f"Konteks: Putaran optimisasi ke-{r}. Skor terbaik saat ini ({metric_name}) adalah {best_score:.4f}."


        final_proposal = council.deliberate_on_proposal(
            tutor_proposal, history_context, data_summary
        )

        if final_proposal is None:
            reason = "CouncilError: Musyawarah gagal menghasilkan keputusan yang valid."
            logger.error(f"  Musyawarah Dewan AI gagal. Menghentikan putaran.")
            history.append(
                {
                    "round": r,
                    "score": None,
                    "full_metrics": None,
                    "hparams": tutor_proposal,
                    "accepted": False,
                    "checkpoint_path": None,
                    "reason": reason,
                }
            )
            break

        logger.info(f"  Keputusan final dari Dewan AI diterima.")

        logger.info(
            "  [3/4] Accreditor AI sedang memvalidasi keputusan dewan...")
        verdict, err = accreditor.validate_proposal(final_proposal)

        accepted = err is None and "[PROPOSAL_ACCEPTED]" in verdict
        if not accepted:
            reason = verdict or f"AccreditorError: {err}"
            logger.warning(
                f"  Keputusan dewan DITOLAK oleh Accreditor. Alasan: {reason}"
            )
            history.append(
                {
                    "round": r,
                    "score": None,
                    "full_metrics": None,
                    "hparams": final_proposal,
                    "accepted": False,
                    "checkpoint_path": None,
                    "reason": reason,
                }
            )
            continue

        logger.info("  Keputusan dewan DITERIMA oleh Accreditor.")

        logger.info(
            "  [4/4] Menjalankan pelatihan dengan hyperparameter yang telah disetujui..."
        )


        score, full_metrics, checkpoint_path, data_summary_new = train_eval_fn(
            final_proposal
        )

        if data_summary_new:
            data_summary = data_summary_new

        tracker.log(
            project_id=final_proposal["project_id"],
            round_idx=r,
            params=final_proposal,
            metrics=full_metrics,
        )
        history.append(
            {
                "round": r,
                "score": score,
                "full_metrics": full_metrics,
                "hparams": final_proposal,
                "accepted": True,
                "checkpoint_path": checkpoint_path,
                "data_summary": data_summary,
            }
        )

        if (best_score - score) > improve_tol:
            logger.info(
                f"  PENINGKATAN DITEMUKAN! Skor baru: {score:.6f} (lebih baik dari {best_score:.6f})"
            )
            best_score = score
            best_hparams = final_proposal
        else:
            logger.info(
                f"  Tidak ada peningkatan signifikan. Skor: {score:.6f}. Menghentikan optimisasi."
            )
            break

    return best_hparams, history



def frac_diff(series, d, thres=1e-4):
    """
    Menghitung fractional differentiation.

    Args:
        series (pd.Series): Time series data.
        d (float): Orde diferensiasi, antara 0 dan 1.
        thres (float): Batas bawah untuk bobot agar komputasi berhenti.

    Returns:
        pd.Series: Differentiated series.
    """
    w = [1.0]
    for k in range(1, len(series)):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
    w = np.array(w[::-1])

    result = []
    for i in range(len(w) - 1, len(series)):
        result.append(np.dot(w, series[i - len(w) + 1: i + 1]))


    return pd.Series(result, index=series.index[len(w) - 1:])


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """
    Fungsi sederhana untuk memecah teks menjadi potongan-potongan (chunks)
    berdasarkan paragraf, dengan ukuran dan tumpang tindih (overlap) yang wajar.
    Tidak memerlukan tokenizer.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if not words:
            continue
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
    return chunks


def adversarial_mixup(
    x1: torch.Tensor, x2: torch.Tensor, alpha: float = 0.4
) -> torch.Tensor:
    """
    Mencampur dua batch data (x1 dan x2) menggunakan distribusi Beta.
    Ini adalah augmentasi yang 'sadis' untuk memaksa generalisasi.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1


    batch_size = min(x1.size(0), x2.size(0))
    x1, x2 = x1[:batch_size], x2[:batch_size]

    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x


@contextmanager

def read_daily_request_count():
    """Membaca file untuk mengetahui berapa banyak request yang sudah dibuat hari ini."""
    today = datetime.now().strftime("%Y-%m-%d")
    if not REQUEST_COUNT_FILE.exists():
        return 0, today

    try:
        with REQUEST_COUNT_FILE.open("r") as f:
            data = json.load(f)

        if data.get("date") == today:
            return data.get("count", 0), today
        else:

            return 0, today
    except (json.JSONDecodeError, IOError):
        return 0, today


def increment_daily_request_count():
    """Menambah jumlah request harian dan menyimpannya ke file."""
    count, today = read_daily_request_count()
    count += 1
    with REQUEST_COUNT_FILE.open("w") as f:
        json.dump({"date": today, "count": count}, f)
    logger.info(
        f"  [API_METER] Penggunaan harian: {count}/{RPD_LIMIT} requests.")


def get_next_trading_days(
    start_date: datetime, n_days: int, country_code: str = "ID"
) -> list:
    trading_days, current_date = [], start_date
    country_holidays = holidays.country_holidays(country_code)
    while len(trading_days) < n_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5 and current_date not in country_holidays:
            trading_days.append(current_date)
    return trading_days


def calculate_hurst(series, kind="price", simplified=True):
    """
    Menghitung Hurst Exponent untuk sebuah series.
    Membungkus library 'hurst' untuk menangani error jika data tidak cukup.
    """
    try:

        series_clean = series.dropna()
        if len(series_clean) < 20:
            return np.nan

        H, _, _ = compute_Hc(series_clean.values,
                             kind=kind, simplified=simplified)
        return H
    except Exception:

        return np.nan


def generate_all_features(
    df: pd.DataFrame,
    tickers: list,
    master_event_list: list,
    use_chat_features: bool = True,
    x_sentiment_manager: "XSentimentManager" = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Versi canggih dari rekayasa fitur yang mencakup:
    1. Fitur Seismograf Pasar Tektonik (ASeT).
    2. Indikator teknikal & statistik standar.
    3. Fitur kesadaran krisis (Event Influence Decay).
    4. Fitur sentimen jangka panjang.
    5. Fitur memori pasar (Hurst Exponent).
    6. Fitur sentimen dari X (Twitter).
    """
    df_with_features = df.copy()


    if x_sentiment_manager:
        logger.info("Membuat fitur sentimen dari X (Twitter) untuk 7 hari terakhir...")
        end_time = datetime.now(timezone.utc)
        end_time_safe = end_time - timedelta(seconds=30)
        start_time = end_time - timedelta(days=6)

        all_sentiments = []
        for ticker in tickers:
            if "IHSG" not in ticker.upper() and "USDIDR" not in ticker.upper():
                sentiment_df = x_sentiment_manager.search_and_analyze_sentiment(
                    ticker, start_time, end_time_safe
                )
                if not sentiment_df.empty:
                    all_sentiments.append(sentiment_df.set_index("date"))

        if all_sentiments:
            combined_sentiments = pd.concat(all_sentiments, axis=1)
            df_with_features = pd.merge(
                df_with_features,
                combined_sentiments,
                left_index=True,
                right_index=True,
                how="left",
            )
            sentiment_cols = [col for col in df_with_features.columns if "x_sentiment_" in col]
            df_with_features[sentiment_cols] = df_with_features[sentiment_cols].fillna(0)
            logger.info("✅ Fitur sentimen X berhasil diintegrasikan.")


    logger.info("Mencoba mengintegrasikan data ekonomi makro (CPI, FED_RATE)...")
    try:
        DB_PATH = "alpha_internal.db"
        conn = sqlite3.connect(DB_PATH)
        df_econ = pd.read_sql_query(
            "SELECT event_date, event_type, value FROM economic_events", conn
        )
        conn.close()
        df_econ["event_date"] = pd.to_datetime(df_econ["event_date"])
        df_econ_pivot = df_econ.pivot_table(index="event_date", columns="event_type", values="value")
        df_econ_pivot.rename(
            columns={"CPI": "feature_econ_cpi", "FED_RATE": "feature_econ_fed_rate"},
            inplace=True,
        )
        df_with_features = pd.merge(
            df_with_features, df_econ_pivot, left_index=True, right_index=True, how="left"
        )
        df_with_features[["feature_econ_cpi", "feature_econ_fed_rate"]] = (
            df_with_features[["feature_econ_cpi", "feature_econ_fed_rate"]].ffill()
        )
        logger.info("✅ Data ekonomi makro berhasil diintegrasikan.")
    except Exception as e:
        logger.warning(f"⚠️ Gagal mengintegrasikan data ekonomi makro: {e}. Proses lanjut tanpa data ini.")


    if len(tickers) >= 2:
        df_aset, pure_hht_features = generate_aset_features(
            df_with_features, tickers, main_asset=tickers[0], secondary_asset=tickers[1]
        )
        new_aset_cols = df_aset.columns.difference(df_with_features.columns)
        df_with_features = pd.concat([df_with_features, df_aset[new_aset_cols]], axis=1)
    else:
        logger.warning("Kurang dari 2 ticker dipilih, fitur ASeT tidak dapat dibuat.")
        pure_hht_features = []


    logger.info("Membuat indikator teknikal...")
    for t in tickers:
        tech_indicators = generate_technical_indicators(df_with_features, t)
        new_tech_cols = tech_indicators.columns.difference(df_with_features.columns)
        df_with_features = pd.concat([df_with_features, tech_indicators[new_tech_cols]], axis=1)


    logger.info("Membuat fitur volatilitas...")
    volatility_features = generate_volatility_features(df_with_features, tickers)
    new_vol_cols = volatility_features.columns.difference(df_with_features.columns)
    df_with_features = pd.concat([df_with_features, volatility_features[new_vol_cols]], axis=1)


    logger.info("Membuat Fitur Statistik Tingkat Tinggi (Skewness & Kurtosis)...")
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df_with_features.columns:
            log_returns = np.log(df_with_features[close_col] / df_with_features[close_col].shift(1))
            df_with_features[f"{ticker}_skew_20"] = log_returns.rolling(window=20).skew()
            df_with_features[f"{ticker}_kurtosis_20"] = log_returns.rolling(window=20).kurt()


    logger.info("Membuat Fitur Transformasi Matematis Lanjutan...")
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df_with_features.columns and df_with_features[close_col].notna().all():
            wavelet = "db4"
            if len(df_with_features[close_col]) >= pywt.dwt_max_level(len(df_with_features[close_col]), wavelet) * 2 + 2:
                coeffs = (
                    df_with_features[close_col]
                    .rolling(window=20)
                    .apply(lambda x: (pywt.dwt(x, wavelet)[0][0] if len(x) >= 20 else np.nan), raw=True)
                )
                df_with_features[f"{ticker}_wavelet_approx"] = coeffs

            df_with_features[f"{ticker}_frac_diff_0.4"] = frac_diff(
                df_with_features[close_col], d=0.4, thres=1e-4
            )


    logger.info("Membuat Fitur Memori Pasar (Hurst Exponent)...")
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df_with_features.columns:
            log_returns = np.log(df_with_features[close_col] / df_with_features[close_col].shift(1)).dropna()
            df_with_features[f"{ticker}_hurst_100"] = log_returns.rolling(window=100).apply(calculate_hurst, raw=False)


    df_with_features = generate_event_decay_feature(df_with_features, master_event_list)


    logger.info("Membuat Fitur Pola Siklus (Harian/Bulanan)...")
    df_with_features["day_of_week"] = df_with_features.index.dayofweek
    df_with_features["week_of_year"] = df_with_features.index.isocalendar().week.astype(int)
    df_with_features["month_of_year"] = df_with_features.index.month
    df_with_features["is_month_end"] = df_with_features.index.is_month_end.astype(int)
    df_with_features["is_quarter_end"] = df_with_features.index.is_quarter_end.astype(int)


    logger.info("Membuat fitur pasar & cross-ticker...")
    close_cols = [f"{t}_Close" for t in tickers if f"{t}_Close" in df_with_features]
    if close_cols:
        returns = df_with_features[close_cols].pct_change()
        df_with_features["market_return"] = returns.mean(axis=1)
        df_with_features["ad_line"] = (returns > 0).sum(axis=1) - (returns < 0).sum(axis=1)

    if len(tickers) == 2:
        t1, t2 = tickers
        if f"{t1}_Close" in df_with_features and f"{t2}_Close" in df_with_features:
            df_with_features[f"{t1}_{t2}_ratio"] = df_with_features[f"{t1}_Close"] / df_with_features[f"{t2}_Close"]
            df_with_features[f"{t1}_{t2}_corr_20"] = (
                df_with_features[f"{t1}_Close"].rolling(20).corr(df_with_features[f"{t2}_Close"])
            )


    if use_chat_features:
        logger.info("Membuat fitur berbasis chat...")
        try:
            with closing(sqlite3.connect("chat_archive.db")) as conn:
                chat_df = pd.read_sql_query(
                    "SELECT ts, sentiment_score, emotions FROM chats",
                    conn,
                    parse_dates=["ts"],
                )
            chat_df.set_index("ts", inplace=True)
            daily_sent = chat_df["sentiment_score"].resample("D").mean().fillna(0)

            df_with_features["sentiment_daily_mean"] = daily_sent.reindex(df_with_features.index, fill_value=0)
            df_with_features["sentiment_weekly_mean"] = (
                daily_sent.rolling(window=7, min_periods=1).mean().reindex(df_with_features.index, method="ffill").fillna(0)
            )
            df_with_features["sentiment_monthly_mean"] = (
                daily_sent.rolling(window=30, min_periods=1).mean().reindex(df_with_features.index, method="ffill").fillna(0)
            )

            emotions_df = chat_df["emotions"].apply(json.loads).apply(pd.Series)
            for emo in emotions_df.columns:
                df_with_features[f"daily_emotion_{emo}"] = (
                    emotions_df[emo].resample("D").mean().fillna(0).reindex(df_with_features.index, fill_value=0)
                )
        except Exception as e:
            logger.warning(f"Gagal memuat fitur chat: {e}")


    final_feature_list = [
        col
        for col in df_with_features.columns
        if col not in ["Open time", "Target_Return_1", "Target_Return_5", "Target_Return_10"]
    ]

    if ("anomaly_score" not in final_feature_list) and ("anomaly_score" in df_with_features.columns):
        final_feature_list.append("anomaly_score")
        logger.info("Fitur 'anomaly_score' telah ditambahkan ke daftar fitur final.")

        logger.info("🔬 Mencari dan menerapkan strategi yang dibuat oleh AI...")
    strategy_dir = Path.home() / APP_BRAND / "strategies"
    if strategy_dir.exists():
        sys.path.insert(0, str(strategy_dir.parent))
        for strategy_file in strategy_dir.glob("*.py"):
            try:
                module_name = f"strategies.{strategy_file.stem}"
                strategy_module = importlib.import_module(module_name)
                df_with_features = strategy_module.apply_strategy(df_with_features)
            except Exception as e:
                logger.error(f"Gagal memuat atau menerapkan strategi dari {strategy_file.name}: {e}")
        sys.path.pop(0)
    else:
        logger.info("Direktori strategi tidak ditemukan. Melewatkan injeksi dinamis.")

    logger.info(f"Total {len(final_feature_list)} fitur berhasil dibuat.")


    logger.info("Menerapkan Global Outlier Trimming (Clipping) dengan vektorisasi...")

    df_with_features = df_with_features.copy()
    numeric_cols = df_with_features.select_dtypes(include=np.number).columns.tolist()


    target_cols_to_skip = [col for col in df_with_features.columns if "log_return" in col]
    cols_to_skip = ["Target_Return_1", "Target_Return_5", "Target_Return_10"] + target_cols_to_skip
    cols_to_process = [col for col in numeric_cols if col not in cols_to_skip]


    sub = df_with_features[cols_to_process]
    q = sub.quantile([0.01, 0.99], numeric_only=True)
    lower = q.loc[0.01].reindex(cols_to_process)
    upper = q.loc[0.99].reindex(cols_to_process)
    df_with_features[cols_to_process] = sub.clip(lower=lower, upper=upper, axis=1)

    logger.info("✅ Global Outlier Trimming (vectorized) selesai.")

    return df_with_features, pure_hht_features



def analyze_data_initial(df: pd.DataFrame, tickers: list) -> tuple[int, int]:
    volatilities = [
        df[f"{t}_Close"].pct_change().std() * np.sqrt(252)
        for t in tickers
        if f"{t}_Close" in df.columns
        and not df[f"{t}_Close"].pct_change().dropna().empty
    ]
    avg_volatility = np.mean(volatilities) if volatilities else 0
    suggested_window = min(120, max(30, int(60 * (1 + avg_volatility))))
    suggested_horizon = min(14, max(1, int(7 * (1 + avg_volatility))))
    logger.info(
        f"Volatilitas rata-rata: {avg_volatility:.4f}, Window: {suggested_window}, Horizon: {suggested_horizon}"
    )
    return suggested_window, suggested_horizon


def generate_technical_indicators(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    indicators = pd.DataFrame(index=df.index)
    close, high, low, volume = (
        df.get(f"{ticker}_Close"),
        df.get(f"{ticker}_High"),
        df.get(f"{ticker}_Low"),
        df.get(f"{ticker}_Volume"),
    )
    if any(s is None for s in [close, high, low, volume]) or len(close) < 20:
        return indicators
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        indicators[f"{ticker}_RSI_14"] = talib.RSI(close, 14)
        indicators[f"{ticker}_EMA_20"] = talib.EMA(close, 20)
        macd, signal, _ = talib.MACD(close)
        indicators[f"{ticker}_MACD"], indicators[f"{ticker}_MACD_signal"] = macd, signal
        upper, middle, lower = talib.BBANDS(close, 20)
        (
            indicators[f"{ticker}_BB_upper"],
            indicators[f"{ticker}_BB_middle"],
            indicators[f"{ticker}_BB_lower"],
        ) = (upper, middle, lower)
        indicators[f"{ticker}_ATR_14"] = talib.ATR(high, low, close, 14)
        indicators[f"{ticker}_OBV"] = talib.OBV(close, volume)
    return indicators


def generate_volatility_features(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    volatility_df = pd.DataFrame(index=df.index)
    logger.info("Membuat fitur volatilitas...")
    for ticker in tickers:
        close_col = f"{ticker}_Close"
        if close_col in df.columns:
            returns = df[close_col].pct_change().dropna()
            volatility_df[f"{ticker}_Volatility_20"] = returns.rolling(
                window=20
            ).std() * np.sqrt(252)
            high, low, close = (
                df.get(f"{ticker}_High"),
                df.get(f"{ticker}_Low"),
                df[close_col],
            )
            if high is not None and low is not None:
                volatility_df[f"{ticker}_ATR_20"] = talib.ATR(
                    high, low, close, timeperiod=20
                )
    return volatility_df


def generate_event_decay_feature(
    df: pd.DataFrame, master_event_list: list
) -> pd.DataFrame:
    """
    Membuat fitur 'event_influence_score' yang memudar seiring waktu setelah peristiwa.
    """
    logger.info("Membuat Fitur Pengaruh Peristiwa (Event Influence Decay)...")
    if not master_event_list:
        df["event_influence_score"] = 0.0
        return df


    event_influence = pd.Series(0.0, index=df.index)
    decay_halflife = (
        21
    )

    for event in master_event_list:
        try:
            event_date = pd.to_datetime(event.date)

            if event_date in df.index:
                start_idx = df.index.get_loc(event_date)

                for i in range(60):
                    current_idx = start_idx + i
                    if current_idx < len(df.index):
                        target_date = df.index[current_idx]
                        days_passed = (target_date - event_date).days
                        decay_factor = np.exp(-days_passed / decay_halflife)


                        current_influence = abs(
                            event.impact_score) * decay_factor


                        event_influence.iloc[current_idx] = max(
                            event_influence.iloc[current_idx], current_influence
                        )
        except Exception as e:
            logger.warning(
                f"Gagal memproses event decay untuk '{event.event_name}': {e}"
            )

    df["event_influence_score"] = event_influence
    return df


def generate_aset_features(
    df: pd.DataFrame, tickers: list, main_asset: str, secondary_asset: str
) -> tuple[pd.DataFrame, list[str]]:
    """
    Menghitung dan menambahkan fitur Seismograf Pasar Tektonik (ASeT) ke DataFrame.
    Versi ini tangguh terhadap kegagalan dekomposisi EMD (fail-safe) dan memiliki logging rinci.
    EMD diparalelkan antar-ticker (proses-based) dengan pembatasan thread BLAS/OMP di tiap worker
    untuk menghindari oversubscription. Jika joblib tidak tersedia, fungsi akan fallback ke mode serial.
    """
    import os

    logger.info("🌍 Memulai perhitungan fitur Seismograf Pasar Tektonik (ASeT)...")
    df_aset = df.copy()
    pure_hht_features: list[str] = []

    try:
        aset_feature_names: list[str] = []


        TREND_IMF_START_INDEX = 2
        PRESSURE_IMF_INDEX   = 1
        imf_storage: dict[str, pd.Series] = {}




        try:
            from joblib import Parallel, delayed
            import multiprocessing as _mp
            _HAS_JOBLIB = True
        except Exception:
            _HAS_JOBLIB = False

        def _emd_one(_ticker: str) -> dict[str, pd.Series]:
            local: dict[str, pd.Series] = {}
            close_col = f"{_ticker}_Close"
            if close_col not in df_aset.columns:
                return local

            logger.info(f"  -> Melakukan dekomposisi EMD untuk {_ticker}...")
            series = (
                df_aset[close_col]
                .interpolate(method="linear")
                .bfill()
                .ffill()
            )


            if len(series) < 100:
                logger.warning(f"     ⚠️ Data untuk {_ticker} terlalu pendek untuk EMD, dilewati.")
                return local


            try:
                from threadpoolctl import threadpool_limits as _tpl
            except Exception:
                _tpl = None

            def _decompose():
                decomposer = EMD(series.values)
                return decomposer.decompose()

            if _tpl is None:
                imfs = _decompose()
            else:

                with _tpl(limits=1):
                    imfs = _decompose()


            if imfs.shape[1] < (TREND_IMF_START_INDEX + 1):
                logger.warning(
                    f"     ⚠️ IMF tidak cukup untuk {_ticker} (punya {imfs.shape[1]})."
                )
                return local


            for i in range(imfs.shape[1]):
                s = pd.Series(imfs[:, i])
                s.index = series.index[: len(s)]
                local[f"IMF{i}_{_ticker}"] = s.reindex(series.index, method=None)

            return local

        if _HAS_JOBLIB:

            try:
                _cores = (_mp.cpu_count() or 2)
            except Exception:
                _cores = 2
            try:
                _override = int(os.getenv("ASSET_EMD_JOBS", "0"))
            except Exception:
                _override = 0
            _emd_jobs = _override if _override > 0 else min(max(1, _cores - 1), 6)
            logger.info(f"  -> EMD paralel: n_jobs={_emd_jobs}, tickers={len(tickers)}")

            results = Parallel(n_jobs=_emd_jobs, backend="loky", prefer="processes")(
                delayed(_emd_one)(t) for t in tickers
            )
            for d in results:
                imf_storage.update(d)
        else:
            logger.info("  -> joblib tidak tersedia; menjalankan EMD secara serial.")
            for t in tickers:
                imf_storage.update(_emd_one(t))


        for name, s in imf_storage.items():
            df_aset[name] = s




        logger.info("  -> Menghitung Sensor #1: Detektor Retakan Lempeng (Korelasi IMF Tren)...")
        trend_imf_main = f"IMF{TREND_IMF_START_INDEX}_{main_asset}"
        trend_imf_secondary = f"IMF{TREND_IMF_START_INDEX}_{secondary_asset}"

        if trend_imf_main in df_aset.columns and trend_imf_secondary in df_aset.columns:
            sensor1_col = "aset_sensor1_trend_correlation"
            df_aset[sensor1_col] = (
                df_aset[trend_imf_main]
                .rolling(window=60, min_periods=30)
                .corr(df_aset[trend_imf_secondary])
            )
            aset_feature_names.append(sensor1_col)
            logger.info(f"     ✅ Fitur '{sensor1_col}' dibuat.")
        else:
            logger.warning("     ⚠️ Gagal membuat Sensor #1: IMF tren tidak ditemukan.")




        logger.info("  -> Menghitung Sensor #2: Detektor Tekanan (Instabilitas Frekuensi)...")
        pressure_imf_col = f"IMF{PRESSURE_IMF_INDEX}_{main_asset}"
        if pressure_imf_col in df_aset.columns:
            sensor2_col = "aset_sensor2_freq_instability"


            imf_pressure_series = (
                df_aset[pressure_imf_col]
                .interpolate("linear")
                .bfill()
                .ffill()
            )
            imf_pressure = imf_pressure_series.values

            if len(imf_pressure) > 20:
                try:
                    frequencies, timestamps = inst_freq(imf_pressure)
                    inst_freq_series = pd.Series(frequencies, index=df_aset.index[timestamps])
                    df_aset["temp_inst_freq"] = inst_freq_series.reindex(df_aset.index).bfill()
                    df_aset[sensor2_col] = (
                        df_aset["temp_inst_freq"].rolling(window=20, min_periods=10).var()
                    )
                    df_aset.drop(columns=["temp_inst_freq"], inplace=True)
                    aset_feature_names.append(sensor2_col)
                    logger.info(f"     ✅ Fitur '{sensor2_col}' dibuat.")
                except Exception as e:
                    logger.warning(f"     ⚠️ inst_freq gagal: {e}. Fallback ke rolling var IMF1.")
                    df_aset[sensor2_col] = imf_pressure_series.rolling(window=20, min_periods=10).var()
                    aset_feature_names.append(sensor2_col)
            else:
                logger.warning("     ⚠️ Panjang IMF tekanan terlalu pendek untuk inst_freq. Fallback ke rolling var IMF1.")
                df_aset[sensor2_col] = imf_pressure_series.rolling(window=20, min_periods=10).var()
                aset_feature_names.append(sensor2_col)
        else:
            logger.warning("     ⚠️ Gagal membuat Sensor #2: IMF tekanan tidak ditemukan.")




        pure_hht_features = [col for col in df_aset.columns if col.startswith("IMF")]
        logger.info(f"  -> ✅ Ditemukan {len(pure_hht_features)} fitur HHT murni.")

    except Exception as e:
        logger.error(
            f"⚠️ Gagal total saat membuat fitur ASeT: {e}. Melanjutkan tanpa fitur ini.",
            exc_info=False,
        )
        return df.copy(), []

    return df_aset, pure_hht_features










def encode_events_to_spikes(event_scores: pd.Series, num_steps: int) -> np.ndarray:
    """
    Mengubah series skor peristiwa kontinu (0-1) menjadi spike trains
    menggunakan rate encoding dari snntorch.

    Args:
        event_scores (pd.Series): Series skor pengaruh peristiwa.
        num_steps (int): Jumlah time steps untuk representasi spike.

    Returns:
        np.ndarray: Array spike dengan shape [len(event_scores), num_steps].
    """
    logger.info("⚡ Meng-encode peristiwa menjadi spike trains...")

    rate_tensor = torch.from_numpy(event_scores.values).float()



    spike_trains_transposed = spikegen.rate(rate_tensor, num_steps=num_steps)


    spike_trains = spike_trains_transposed.transpose(0, 1).numpy()
    logger.info(
        f"✅ Spike trains berhasil dibuat dengan shape: {spike_trains.shape}")
    return spike_trains


def _sanitize_and_dedupe_columns(cols: list[str]) -> list[str]:
    sanitized = [re.sub(r"[^0-9A-Za-z_]", "_", c) for c in cols]
    sanitized = [re.sub(r"_{2,}", "_", c).strip("_") for c in sanitized]
    counts = Counter()
    result = []
    for name in sanitized:
        counts[name] += 1
        if counts[name] > 1:
            new_name = f"{name}_{counts[name]-1}"
        else:
            new_name = name
        result.append(new_name)
    return result


def describe_image_with_llm(image_path: Path, api_pool: "DistributedAIPool") -> str:
    """Menggunakan model vision untuk mendeskripsikan gambar."""
    try:

        vision_agent = TogetherLLM(
            api_key=api_pool.together_api_keys.get("vision"),
            model_name="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        )
        description = vision_agent.describe_image(
            str(image_path), "Describe this image in detail."
        )
        return f"Image content: {description}"
    except Exception as e:
        logger.error(f"Gagal mendeskripsikan gambar {image_path.name}: {e}")
        return f"Could not analyze image content of {image_path.name}."


def read_sqlite_db(file_path: Path) -> Optional[str]:
    """Membaca file SQLite, merangkum skema dan beberapa data sampel."""
    logger.info(f"  -> [SQLiteReader] Menganalisis database: {file_path.name}")
    try:
        summary_parts = [f"Ringkasan Database dari file '{file_path.name}':"]
        with closing(sqlite3.connect(file_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            summary_parts.append(
                f"- Ditemukan {len(tables)} tabel: {', '.join(tables)}")

            for table_name in tables:
                df_sample = pd.read_sql_query(
                    f"SELECT * FROM {table_name} LIMIT 5;", conn)
                summary_parts.append(
                    f"\n--- Sampel dari Tabel '{table_name}' ---\n{df_sample.to_string()}")

        return "\n".join(summary_parts)
    except Exception as e:
        logger.error(
            f"  -> [SQLiteReader] Gagal membaca {file_path.name}: {e}")
        return f"Gagal menganalisis file database '{file_path.name}' karena error: {e}"


def inspect_pytorch_model(file_path: Path) -> Optional[str]:
    """Memeriksa file state_dict PyTorch dan melaporkan lapisan & ukurannya."""
    logger.info(f"  -> [PyTorchInspector] Memeriksa model: {file_path.name}")
    try:
        summary_parts = [
            f"Ringkasan Arsitektur Model (dari bobot) '{file_path.name}':"]
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))

        if not isinstance(state_dict, dict):
            return f"File '{file_path.name}' tampaknya adalah seluruh objek model, bukan state_dict. Inspeksi tidak didukung."

        total_params = 0
        for layer_name, tensor in state_dict.items():
            num_params = tensor.numel()
            total_params += num_params
            summary_parts.append(
                f"- Lapisan: {layer_name:<50} | Bentuk: {str(tensor.shape):<25} | Parameter: {num_params}")

        summary_parts.append(f"\nTotal Parameter Terdeteksi: {total_params:,}")
        return "\n".join(summary_parts)
    except Exception as e:
        logger.error(
            f"  -> [PyTorchInspector] Gagal membaca {file_path.name}: {e}")
        return f"Gagal memeriksa file model '{file_path.name}' karena error: {e}"



SPECIALIST_REGISTRY = {
    ".sqlite": read_sqlite_db,
    ".db": read_sqlite_db,
    ".pth": inspect_pytorch_model,
    ".ckpt": inspect_pytorch_model,
}


def handle_unknown_file_in_sandbox(file_path: Path, brain: "Brain", nsmm: "NSMM"):
    """
    Placeholder untuk menangani tipe file yang tidak dikenal.
    Untuk saat ini, hanya akan mencatat peringatan.
    """
    logger.warning(
        f"  -> [Sandbox Fallback] Tipe file tidak dikenal untuk '{file_path.name}'. "
        "Penanganan spesifik belum diimplementasikan. File dilewati."
    )


    return




def universal_file_ingestor(
    file_path: Path,
    api_pool: "DistributedAIPool",
    nsmm: "NSMM",
    brain: "Brain",
    engine: "AsyncCuriosityEngine",
) -> Optional[str]:
    """
    Mencerna berbagai jenis file secara universal, memanggil spesialis untuk
    format yang dikenal, dan menggunakan sandbox untuk format yang tidak dikenal.
    Versi ini memiliki alur logika yang benar dan tangguh.
    """
    try:
        content_text = ""
        suffix = file_path.suffix.lower()


        if suffix in SPECIALIST_REGISTRY:
            return SPECIALIST_REGISTRY[suffix](file_path)


        if suffix in [".txt", ".json", ".py", ".log", ".gml", ".md", ".csv"]:
            content_text = file_path.read_text(errors="ignore")
        elif suffix == ".pdf":
            images = convert_from_path(file_path)
            content_text = "\n".join(
                pytesseract.image_to_string(img) for img in images)
        elif suffix == ".docx":
            doc = docx.Document(file_path)
            content_text = "\n".join([para.text for para in doc.paragraphs])
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)

            return f"Data summary of {file_path.name}:\n---\n{df.head().to_string()}\n---"



        if content_text and content_text.strip():

            if suffix == ".json":
                try:

                    if not json.loads(content_text):
                        logger.warning(
                            f"[IQRO] Melewatkan file JSON karena kosong: {file_path.name}")
                        return None
                except json.JSONDecodeError:
                    logger.warning(
                        f"[IQRO] Melewatkan file JSON karena formatnya tidak valid: {file_path.name}")
                    return None



            solve_math_expressions(content_text, file_path, nsmm, engine)
            return content_text




        else:
            handle_unknown_file_in_sandbox(file_path, brain, nsmm)

            return None

    except Exception as e:
        logger.error(f"Gagal mencerna file {file_path.name}: {e}")
        return None


def _execute_single_gui_action(action: GUIAction) -> str:
    """Mengeksekusi satu objek GUIAction."""
    try:
        logger.info(
            f"  -> Aksi: {action.action_type.upper()} - Alasan: {action.reasoning}")
        if action.action_type == "click" and action.coordinates:
            pyautogui.click(action.coordinates[0], action.coordinates[1])
        elif action.action_type == "type" and action.text_to_type:
            pyautogui.typewrite(action.text_to_type, interval=0.05)
        elif action.action_type == "press" and action.key_to_press:
            pyautogui.press(action.key_to_press)
        elif action.action_type == "wait" and action.wait_seconds:
            time.sleep(action.wait_seconds)
        else:
            return f"Aksi '{action.action_type}' tidak valid atau parameter hilang."
        time.sleep(0.7)
        return f"Aksi '{action.action_type}' berhasil."
    except Exception as e:
        logger.error(f"Gagal saat eksekusi aksi tunggal: {e}")
        return f"Gagal pada aksi '{action.action_type}': {e}"


def run_autonomous_task_agent(
    high_level_goal: str,
    api_pool: "DistributedAIPool",
    **kwargs
):
    """
    Menjalankan agen otonom yang cerdas untuk menyelesaikan tujuan tingkat tinggi
    menggunakan siklus Lihat -> Pikir -> Bertindak -> Verifikasi.
    """

    final_goal = high_level_goal
    plan_file = Path("best_plan.txt")


    if high_level_goal.strip().lower() == "execute_best_plan_from_world_model":
        logger.info(
            " Menerima perintah untuk eksekusi rencana dari World Model...")
        if plan_file.exists():
            final_goal = plan_file.read_text()
            logger.info(f" Rencana berhasil dimuat: '{final_goal}'")
        else:
            logger.error(
                f" GAGAL: File rencana '{plan_file.name}' tidak ditemukan. Misi dibatalkan.")
            return


    logger.info("\n" + "="*80)
    logger.info(f"=== 👻 AGEN OTONOM AKTIF (Ghost in the Machine) 👻 ===")

    logger.info(f"=== TUJUAN: {final_goal} ===")
    logger.info("="*80)

    task_history = ["Misi dimulai."]
    max_steps = 15

    for i in range(max_steps):
        logger.info(f"\n--- Siklus Agen #{i+1}/{max_steps} ---")


        logger.info("  [1/4] Melihat lingkungan desktop...")

        desktop_elements = perceive_desktop_elements.func()
        if not desktop_elements:
            perception_context = "Layar tampak kosong atau tidak ada elemen yang dapat dikenali."
        else:
            perception_context = "Saya melihat elemen-elemen berikut di layar:\n"
            for el in desktop_elements:
                center_x = int((el['icon_box'][0] + el['icon_box'][2]) / 2)
                center_y = int((el['icon_box'][1] + el['icon_box'][3]) / 2)
                perception_context += f"- Elemen bernama '{el['app_name']}' di sekitar koordinat ({center_x}, {center_y})\n"


        logger.info("  [2/4] Memikirkan langkah berikutnya...")
        planner_prompt = f"""
        Anda adalah otak eksekutif dari sebuah AI otonom yang sangat teliti.
        Tujuan Utama Anda: "{high_level_goal}"

        Riwayat Tindakan Sejauh Ini:
        {task_history}

        Persepsi Layar Saat Ini:
        ---
        {perception_context}
        ---
        Elemen dalam format JSON (untuk koordinat): {json.dumps(desktop_elements)}
        ---

        Tugas Anda: Berdasarkan Tujuan Utama, Riwayat, dan Persepsi Saat Ini, tentukan SATU AKSI BERIKUTNYA yang paling logis.
        
        ATURAN KETAT:
        - Pikirkan selangkah demi selangkah.
        - Jika tujuan sudah tercapai, aksi Anda harus 'wait' dengan alasan "Tugas selesai".
        - Untuk 'click', prioritas koordinat adalah titik tengah dari elemen yang dikenali.
        - Kembalikan HANYA SATU objek JSON yang sesuai dengan skema `GUIAction`. Jangan mengembalikan daftar/list.
        """

        try:

            next_action_dict = api_pool.call_gemini_with_tool(
                prompt=planner_prompt,
                agent_name="supervisor",
                tool_schema=GUIAction
            )
            next_action = GUIAction(**next_action_dict)
        except Exception as e:
            logger.error(f"Agen gagal membuat rencana: {e}", exc_info=True)
            task_history.append(
                "Gagal membuat rencana, mencoba lagi di siklus berikutnya.")
            time.sleep(5)
            continue


        logger.info("  [3/4] Melakukan satu aksi...")
        result = _execute_single_gui_action(next_action)
        task_history.append(
            f"Aksi: {next_action.action_type} pada {next_action.coordinates or next_action.text_to_type}. Hasil: {result}")


        logger.info("  [4/4] Memverifikasi hasil aksi...")
        if "selesai" in next_action.reasoning.lower():
            logger.info("✅ Tujuan tercapai! Misi selesai.")
            break



    logger.info("=== 👻 AGEN OTONOM MENYELESAIKAN TUGAS 👻 ===")


def solve_math_expressions(
    text: str, file_path: Path, nsmm: NSMM, engine: "AsyncCuriosityEngine"
):
    """
    Mendeteksi ekspresi matematika dan mengirim tugas kategorisasi ke Async Engine.
    """

    math_patterns = re.findall(
        r"([a-zA-Z0-9\s\*\+\-\/=\(\)\[\]]+=\s*[a-zA-Z0-9\s\*\+\-\/\(\)\[\]]+)", text
    )
    if not math_patterns:
        return


    for expr_str in math_patterns:

        prompt = f"Anda adalah guru matematika. Beri label kesulitan ('Mudah', 'Menengah', 'Sulit') untuk soal: \"{expr_str}\". Jawab HANYA dengan satu kata."


        class MathDifficulty(BaseModel):
            difficulty: str


        engine.ask_async(
            question_text=prompt,
            agent_key="experimentalist",
            response_model=MathDifficulty
        )



def run_iqro_protocol_scan(
    root_path: Path,
    api_pool: "DistributedAIPool",
    nsmm: "NSMM",
    brain: "Brain",
    engine: "AsyncCuriosityEngine"
) -> dict:
    """
    Menjalankan Protokol IQRO v2.0: Memindai semua modul Python ('__init__.py'),
    mencerna file di dalamnya secara efisien menggunakan memori hash, dan
    mengabaikan file yang tidak relevan.
    """
    logger.info(f"📖 [IQRO Protocol] Memulai pemindaian universal dari: {root_path}")

    newly_digested_content = {}
    iqro_memory = _load_iqro_memory()

    # --- FIX: definisi Gudang Ilmu path + accumulator terminologi ---
    # Bisa dioverride via env GUDANG_ILMU_PATH. Default: ~/APP_BRAND/"Gudang Ilmu"
    try:
        _env_gudang = os.getenv("GUDANG_ILMU_PATH")
        if _env_gudang and _env_gudang.strip():
            gudang_ilmu_path = Path(_env_gudang).expanduser()
        else:
            gudang_ilmu_path = (Path.home() / APP_BRAND / "Gudang Ilmu")
    except Exception:
        gudang_ilmu_path = Path.home() / APP_BRAND / "Gudang Ilmu"
    # Normalisasi ke Path absolut sebisanya (tanpa memaksa harus ada di disk)
    try:
        gudang_ilmu_path = gudang_ilmu_path.resolve()
    except Exception:
        gudang_ilmu_path = gudang_ilmu_path

    total_terminologi_baru = 0
    # ---------------------------------------------------------------

    files_to_reread_randomly = []

    for dirpath, _, filenames in os.walk(root_path):
        current_dir = Path(dirpath)
        try:
            current_dir = current_dir.resolve()
        except Exception:
            pass

        # Skip direktori yang jelas tidak perlu
        if "__pycache__" in current_dir.parts or ".git" in current_dir.parts:
            continue

        # --- Blok khusus Gudang Ilmu ---
        if current_dir == gudang_ilmu_path:
            logger.info(f"    -> 📂 Gudang Ilmu ditemukan. Mempelajari terminologi dari semua format...")
            for filename in filenames:
                file_path = current_dir / filename

                content_text = universal_file_ingestor(
                    file_path, api_pool, nsmm, brain, engine
                )

                if content_text and content_text.strip():
                    logger.info(f"      -> Memproses glosarium dari file: {file_path.name}")
                    total_terminologi_baru += _parse_and_store_glossary_content(content_text, brain)
            # Lanjut ke folder berikutnya (Gudang Ilmu tidak diperlakukan sebagai modul python biasa)
            continue
        # --- End Gudang Ilmu ---

        if "__init__.py" in filenames:
            logger.info(
                f"    -> Modul ditemukan: {current_dir.relative_to(root_path)}. Membaca isinya..."
            )
            for filename in filenames:
                file_path = current_dir / filename

                EXCLUDED_SUFFIXES = {
                    ".db", ".sqlite", ".ckpt", ".pth", ".pkl", ".faiss",
                    ".bsp", ".zip", ".png", ".jpg", ".jpeg", ".parquet"
                }
                EXCLUDED_PATTERNS = [
                    "selected_features_", "iqro_memory",
                    "toxic_batch_blacklist", "world_model.meta"
                ]

                if file_path.suffix.lower() in EXCLUDED_SUFFIXES:
                    continue
                if any(pattern in filename for pattern in EXCLUDED_PATTERNS):
                    continue

                current_hash = get_file_hash(file_path)
                if not current_hash:
                    continue

                reason_to_read = ""
                if filename not in iqro_memory:
                    reason_to_read = "BARU 📖"
                elif iqro_memory[filename]["hash"] != current_hash:
                    reason_to_read = "DIUBAH ✍️"
                else:
                    files_to_reread_randomly.append(file_path)
                    continue

                if reason_to_read:
                    logger.info(f"  -> [IQRO] Membaca file ({reason_to_read}): {file_path.name}")
                    content_text = universal_file_ingestor(
                        file_path, api_pool, nsmm, brain, engine
                    )
                    if content_text:
                        newly_digested_content[file_path.name] = content_text

                        text_chunks = chunk_text(content_text, chunk_size=400, chunk_overlap=50)
                        brain.add_chunks(text_chunks, source_name=f"[IQRO_SCAN] {filename}")

                        _update_iqro_memory(iqro_memory, filename, current_hash)

    if total_terminologi_baru > 0:
        logger.info(f"📚 [IQRO Protocol] Terminologi baru dari Gudang Ilmu: {total_terminologi_baru}")

    logger.info(
        f"📖 [IQRO Protocol] Pemindaian selesai. Ditemukan dan diproses {len(newly_digested_content)} file baru/berubah."
    )
    return newly_digested_content


def spawn_bunshin_workers(max_clones: int = 5, root_scan_path: Path = None):
    """
    Protokol Kagebunshin: Melahirkan proses 'bunshin' ringan secara otonom
    untuk memindai sub-direktori secara paralel.
    """
    if root_scan_path is None:
        root_scan_path = Path(__file__).resolve().parents[3]

    potential_missions = [
        p for p in root_scan_path.rglob("__init__.py")
        if "__pycache__" not in p.parts and ".git" not in p.parts
    ]
    target_folders = [p.parent for p in potential_missions]

    if not target_folders:
        return

    missions_to_assign = random.sample(target_folders, min(len(target_folders), max_clones))

    logger.info(
        f"🌀 [Kagebunshin] Mempersiapkan {len(missions_to_assign)} bunshin untuk misi pengumpulan pengetahuan..."
    )

    for folder in missions_to_assign:
        try:
            command = [
                sys.executable,
                __file__,
                "--role", "bunshin",
                "--target-folder", str(folder)
            ]
            subprocess.Popen(command)
            logger.info(f"    -> Bunshin dikirim untuk memindai: {folder.relative_to(root_scan_path)}")
        except Exception as e:
            logger.error(f"Gagal melahirkan bunshin untuk folder {folder}: {e}")


def select_features(
    X: pd.DataFrame, y: pd.Series, top_k: int = None, priority_features: list = None
) -> tuple[list, pd.DataFrame]:
    """
    Versi canggih dengan saringan dua tahap untuk percepatan.
    """
    logger.info(f"\n--- Seleksi Fitur (Awal: {X.shape[1]}) ---")
    if top_k is None:
        top_k = 200




    logger.info(
        "  Tahap 1: Menyaring dengan Mutual Information (saringan cepat)...")

    y_imputed = y.fillna(y.mean())
    mi_scores = mutual_info_regression(X, y_imputed)
    mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores}).sort_values(
        "mi_score", ascending=False
    )


    num_candidates = max(500, top_k * 2)
    top_candidates = mi_df.head(num_candidates)["feature"].tolist()
    X_sieved = X[top_candidates]
    logger.info(f"  -> Kandidat tersaring: {len(top_candidates)} fitur.")



    logger.info(
        "  Tahap 2: Menjalankan Korelasi & LGBM pada kandidat tersaring...")
    corr_matrix = X_sieved.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    X_filtered = X_sieved.drop(to_drop, axis=1)
    logger.info(f"  -> Setelah korelasi: {len(X_filtered.columns)} fitur")

    original_cols = list(X_filtered.columns)
    safe_cols = _sanitize_and_dedupe_columns(original_cols)
    X_filtered.columns = safe_cols

    lgb_model = lgb.LGBMRegressor(
        n_estimators=100, random_state=42, verbosity=-1)
    lgb_model.fit(X_filtered, y)
    importance_df = pd.DataFrame(
        {"feature": X_filtered.columns, "importance": lgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    top_lgbm_features = importance_df.head(top_k)["feature"].tolist()

    final_features = []
    if priority_features:
        final_features.extend(
            [f for f in priority_features if f in top_lgbm_features])
        logger.info(
            f"  Mempertahankan {len(final_features)} fitur prioritas kausal.")

    for feature in top_lgbm_features:
        if len(final_features) >= top_k:
            break
        if feature not in final_features:
            final_features.append(feature)

    logger.info(f"Fitur terpilih final (digabung): {len(final_features)}")
    return final_features, importance_df


def _select_features_with_llm(
    api_pool: "DistributedAIPool",
    all_features: list[str],
    target_description: str,
    top_k: int,
) -> list[str]:
    """
    Menggunakan LLM untuk memilih subset fitur yang paling menjanjikan secara heuristik.
    """
    logger.info(
        f"🔮 Meminta AI Oracle untuk memilih {top_k} fitur terbaik dari {len(all_features)} kandidat..."
    )


    if len(all_features) > 2000:
        feature_sample = random.sample(all_features, 2000)
    else:
        feature_sample = all_features

    prompt = f"""
    Anda adalah seorang Data Scientist kuantitatif veteran dengan intuisi yang tajam.
    Tugas Anda adalah memilih {top_k} fitur yang paling relevan dan menjanjikan dari daftar di bawah untuk memprediksi "{target_description}".

    Pertimbangkan fitur yang:
    1. Memiliki informasi prediktif yang kuat (misal: RSI, MACD, volatilitas).
    2. Menangkap berbagai aspek pasar (teknikal, statistik, siklus, sentimen).
    3. Hindari redundansi yang berlebihan (misal: jangan pilih 5 jenis Moving Average yang mirip).

    DAFTAR FITUR KANDIDAT:
    {feature_sample}

    Output Anda HARUS berupa list string Python dari nama-nama fitur yang Anda pilih.
    Contoh: ["GOOGL_RSI_14", "market_return", "GOOGL_hurst_100"]
    """

    try:
        response_str = api_pool.call_gemini_for_text(
            prompt, "advanced_advisor")

        selected_list = ast.literal_eval(response_str.strip())

        if isinstance(selected_list, list) and all(
            isinstance(item, str) for item in selected_list
        ):
            logger.info(
                f"✅ AI Oracle telah memilih {len(selected_list)} fitur.")
            return selected_list[:top_k]

        logger.warning(
            "Hasil dari AI Oracle bukan list string yang valid. Menggunakan seleksi standar."
        )
        return None
    except Exception as e:
        logger.error(
            f"Gagal mendapatkan pilihan fitur dari AI Oracle: {e}. Menggunakan seleksi standar."
        )
        return None


def extract_chat_topics(nlp, chat_records: list[dict]) -> list[str]:
    if not nlp:
        return []
    topics = set()
    for rec in chat_records:
        doc = nlp(rec["content"])
        for ent in doc.ents:
            topics.add(ent.text.lower())
    return list(topics)


def dynamically_adjust_parameters(hparams: dict, df_full: pd.DataFrame) -> dict:
    """
    Menganalisis ukuran dataset dan secara dinamis menyesuaikan 'window' dan 'horizon'
    untuk mencegah dataloader kosong, dengan logika khusus untuk RG Block.
    """
    logger.info("\n--- Memulai Penyesuaian Parameter Dinamis (Guardrails) ---")

    n_total_rows = len(df_full)
    n_train_rows = int(n_total_rows * 0.8)
    use_rg_block = hparams.get("use_rg_block", False)

    adjusted_hparams = hparams.copy()

    if use_rg_block:
        logger.info("  [ADAPT-INFO] Mode Renormalization Group (RG) aktif.")

        ideal_window = 180
        ideal_horizon = 14
    else:
        logger.info("  [ADAPT-INFO] Mode standar aktif.")
        ideal_window, ideal_horizon = analyze_data_initial(
            df_full, hparams["selected_tickers"]
        )

    if n_train_rows < 30:
        error_msg = f"Data pelatihan tidak memadai ({n_train_rows} baris). Diperlukan minimal 30 baris."
        logger.error(f"FATAL: {error_msg}")
        raise ValueError(error_msg)


    max_allowable_period = n_train_rows - 1

    if (ideal_window + ideal_horizon) > max_allowable_period:
        logger.warning(
            "  [ADAPT-WARN] Parameter ideal (window+horizon) terlalu besar untuk ukuran dataset."
        )
        logger.warning(
            f"  > Ideal: {ideal_window+ideal_horizon} vs Batas Aman: {max_allowable_period}"
        )


        new_horizon = min(ideal_horizon, int(max_allowable_period * 0.15))
        new_horizon = max(1, new_horizon)

        new_window = max_allowable_period - new_horizon
        new_window = max(15, new_window)

        logger.warning(
            f"  > Parameter disesuaikan secara dinamis: Window={new_window}, Horizon={new_horizon}"
        )

        adjusted_hparams["window"] = new_window
        adjusted_hparams["horizon"] = new_horizon
    else:
        logger.info(
            "  [ADAPT-PASS] Ukuran data memadai untuk parameter ideal.")
        logger.info(
            f"  > Menggunakan parameter ideal: Window={ideal_window}, Horizon={ideal_horizon}"
        )
        adjusted_hparams["window"] = ideal_window
        adjusted_hparams["horizon"] = ideal_horizon

    logger.info("--- Penyesuaian Parameter Dinamis Selesai ---")
    return adjusted_hparams



def run_pre_execution_audit(api_pool: DistributedAIPool, source_code: str) -> str:
    logger.info("=" * 25 + " AUDIT KODE PRA-EKSEKUSI " + "=" * 25)
    try:
        auditor_prompt = f"""
        You are an expert Python auditor. Analyze this script for syntax errors, typos, or stray characters.
        Return the corrected script as raw text if errors found, or the original if perfect.
        No explanations or markdown, just raw Python code.

        SCRIPT:
        ---
        {source_code}
        ---
        """
        logger.info("  [AUDITOR-PRE] Menganalisis kode...")
        corrected_code = api_pool.call_gemini_for_text(
            auditor_prompt, "AI_CODE_AUDITOR"
        )
        diff_report = generate_diff_report(source_code, corrected_code)
        logger.info("\n" + diff_report)
        if "Tidak ada perbedaan" in diff_report:
            logger.info("  [AUDITOR-PRE-PASS] Tidak ada kesalahan.")
        else:
            logger.info("  [AUDITOR-PRE-FIX] Perbaikan diterapkan.")
        return corrected_code
    except Exception as e:
        logger.error(f"  [AUDITOR-PRE-FAIL] Gagal audit: {e}")
        return source_code



def _get_full_traceback_context(tb: traceback.StackSummary) -> dict[str, str]:
    """
    Menganalisis seluruh traceback dan mengekstrak kode sumber unik dari setiap frame.
    Ini memberikan konteks yang jauh lebih kaya daripada hanya fungsi terakhir.
    """
    full_context = {}
    unique_frames = {}
    for frame, lineno in traceback.walk_tb(tb):
        func_name = frame.f_code.co_name
        file_path = frame.f_code.co_filename

        if func_name not in unique_frames and not func_name.startswith("<"):
            try:
                source_code = inspect.getsource(frame)
                unique_frames[func_name] = source_code
            except (TypeError, OSError):
                unique_frames[func_name] = (
                    f"Tidak dapat mengambil source code untuk {func_name} di {file_path}"
                )

    return unique_frames


def initiate_autonomous_self_healing(
    error: Exception,
    traceback_str: str,
    api_pool: "DistributedAIPool",
    auditor: "CriticalAuditor",
    together_keys: dict,
) -> bool | str:
    """
    Menjalankan protokol self-healing v8 (Hybrid).
    Menggabungkan pengumpulan konteks kaya (v7) dengan alur kerja fleksibel (v6).
    """
    logger.info("--- MEMULAI PROTOKOL SELF-HEALING v8 (HYBRID) ---")


    try:
        tb = error.__traceback__
        last_frame, last_lineno = list(traceback.walk_tb(tb))[-1]
        file_name = last_frame.f_code.co_filename

        with open(file_name, "r", encoding="utf-8") as f:
            source_lines = f.readlines()

        context_window = 5
        start = max(0, last_lineno - context_window)
        end = min(len(source_lines), last_lineno + context_window)
        code_context = "".join(source_lines[start:end])
        problematic_code = source_lines[last_lineno - 1].strip()

        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "file_path": file_name,
            "line_number": last_lineno,
            "problematic_code_line": problematic_code,
            "surrounding_code_context": code_context,
            "full_traceback_str": traceback_str,
        }
        auditor.add_log(
            "HEAL_CONTEXT", "PASS", "Konteks error kaya berhasil dikumpulkan."
        )


        from src.models.model_alpha.tools.logic_healer import (
            create_healing_prescription,
        )

        healer_tool = create_healing_prescription

    except Exception as e:
        auditor.add_log("HEAL_SETUP", "FAIL", f"Gagal persiapan healing: {e}")
        return False


    try:
        healer_args = {
            "file_path": error_context["file_path"],
            "error_message": error_context["error_message"],
            "full_traceback": error_context["full_traceback_str"],
            "api_pool": api_pool,
        }
        prescription = healer_tool.func(**healer_args)

        if not prescription or prescription.get("executor_name") == "failed":
            auditor.add_log(
                "HEAL_DIAGNOSE",
                "FAIL",
                prescription.get("explanation", "Diagnosis gagal."),
            )
            return False

        executor_name = prescription["executor_name"]
        fix_details = prescription["fix_details"]
        auditor.add_log(
            "HEAL_DIAGNOSE", "PASS", f"Resep diterima. Executor: {executor_name}"
        )


        logger.info(
            f"[Executor] Mencoba memuat executor spesialis: '{executor_name}'..."
        )

        executor_module = importlib.import_module(
            f"src.models.model_alpha.healer_executors.{executor_name}"
        )

        result_message = executor_module.execute(
            error_context["file_path"], fix_details
        )

        if "Berhasil" in result_message:
            auditor.add_log(
                "HEAL_EXECUTE", "PASS", "Executor berhasil menerapkan perbaikan."
            )
            return "RESTART_REQUIRED"
        else:
            auditor.add_log("HEAL_EXECUTE", "FAIL",
                            f"Executor gagal: {result_message}")
            return False

    except Exception as e:
        auditor.add_log(
            "HEAL_EXECUTE",
            "FAIL",
            f"Terjadi error saat menjalankan diagnosis atau eksekusi: {e}",
        )
        return False



def filter_outliers_with_mahalanobis(
    df: pd.DataFrame, feature_columns: list, threshold_percentile: float = 99.0
) -> pd.DataFrame:
    """
    Mengidentifikasi dan membuang outlier menggunakan Mahalanobis Distance
    dengan ambang batas persentil yang adaptif.

    Args:
        df (pd.DataFrame): DataFrame input.
        feature_columns (list): Daftar kolom fitur yang akan digunakan untuk deteksi.
        threshold_percentile (float): Persentil jarak yang akan dianggap sebagai ambang batas outlier.
                                      Contoh: 99.0 berarti 1% data dengan jarak terjauh akan dibuang.

    Returns:
        pd.DataFrame: DataFrame yang sudah bersih dari outlier.
    """
    logger.info(
        f"🧹 Memulai penyaringan outlier dengan Mahalanobis Distance (Threshold > {threshold_percentile} persentil)..."
    )
    df_filtered = df.copy()
    numeric_features = df_filtered[feature_columns].select_dtypes(
        include=np.number)

    if numeric_features.empty:
        logger.warning(
            "Tidak ada fitur numerik yang ditemukan untuk penyaringan Mahalanobis. Melewatkan."
        )
        return df

    imputer = KNNImputer(n_neighbors=5)
    data_imputed = imputer.fit_transform(numeric_features)



    from sklearn.covariance import MinCovDet

    mcd = MinCovDet(random_state=42).fit(data_imputed)
    mahalanobis_dist = mcd.mahalanobis(data_imputed)


    threshold_value = np.percentile(mahalanobis_dist, threshold_percentile)


    is_outlier = mahalanobis_dist > threshold_value
    num_outliers = np.sum(is_outlier)

    logger.info(
        f"✅ Deteksi selesai. Ditemukan dan dibuang {num_outliers} outlier ({num_outliers/len(df_filtered):.2%})."
    )


    df_filtered = df_filtered[~is_outlier]

    return df_filtered


def call_agent_with_fallback(
    api_key: str, primary_model: str, backup_models: list, prompt: str
) -> Optional[str]:
    """
    Mencoba memanggil model utama, dan jika gagal (timeout/error server),
    akan secara otomatis mencoba model-model cadangan secara berurutan.
    """
    models_to_try = [primary_model] + backup_models

    for i, model_name in enumerate(models_to_try):
        is_primary = i == 0
        log_prefix = "[Primary]" if is_primary else f"[Backup #{i}]"

        log_model_name = model_name.split(
            "/")[1] if "/" in model_name else model_name

        try:
            logger.info(f"  > {log_prefix} Mencoba model: {log_model_name}")
            agent = TogetherLLM(api_key=api_key, model_name=model_name)
            response = agent.chat(prompt)
            logger.info(
                f"  > {log_prefix} Panggilan ke {log_model_name} berhasil.")
            return response
        except Exception as e:

            logger.error(
                f"  > {log_prefix} Panggilan ke {log_model_name} GAGAL. Error: {e}",
                exc_info=True,
            )


            if not is_primary:
                time.sleep(2)

    logger.error(
        "  > GAGAL TOTAL: Semua model utama dan cadangan tidak dapat dihubungi."
    )
    return None


def generate_3d_plot_text_description(
    history: list[dict], p1: str, p2: str, p3: str
) -> str:
    """
    Menganalisis slice 3D dari riwayat hyperparameter dan menghasilkan deskripsi teks.
    """
    points = []
    for run in history:
        if run.get("accepted"):
            h = run["hparams"]
            if all(k in h for k in [p1, p2, p3]):
                points.append(
                    {"score": run["score"], p1: h[p1], p2: h[p2], p3: h[p3]})

    if not points:
        return f"Tidak ada data yang valid untuk kombinasi parameter: {p1}, {p2}, {p3}."

    df = pd.DataFrame(points)
    best_run = df.loc[df["score"].idxmin()]

    description = (
        f"Analisis Proyeksi 3D untuk parameter '{p1}', '{p2}', dan '{p3}':\n"
        f"- Terdapat {len(df)} titik data (percobaan) yang valid.\n"
        f"- Kombinasi terbaik ditemukan pada skor loss {best_run['score']:.6f} dengan parameter:\n"
        f"  - {p1}: {best_run[p1]}\n"
        f"  - {p2}: {best_run[p2]}\n"
        f"  - {p3}: {best_run[p3]}\n"
        f"- Rentang nilai yang dieksplorasi:\n"
        f"  - {p1}: dari {df[p1].min()} hingga {df[p1].max()}\n"
        f"  - {p2}: dari {df[p2].min()} hingga {df[p2].max()}\n"
        f"  - {p3}: dari {df[p3].min()} hingga {df[p3].max()}"
    )

    return description


ANALYSIS_PROMPT_TEMPLATE = """
Anda adalah seorang AI Data Scientist yang berspesialisasi dalam menganalisis loss landscape.
Tugas Anda adalah membaca deskripsi teks dari sebuah proyeksi 3D hyperparameter dan mengekstrak satu wawasan (insight) paling penting.

Deskripsi Teks Proyeksi 3D:
---
{plot_description}
---

Wawasan kunci Anda (fokus pada hubungan antar parameter atau area paling menjanjikan):
"""


def validate_with_shap_analysis(history: list[dict], project_id: str):
    """
    Menggunakan SHAP untuk menganalisis dan memvisualisasikan dampak setiap hyperparameter
    terhadap skor (loss).
    """
    logger.info("\n--- FASE 10: ANALISIS DAMPAK PARAMETER DENGAN SHAP (XAI) ---")

    hparams_list = [run["hparams"] for run in history if run.get("accepted")]
    scores = np.array([run["score"] for run in history if run.get("accepted")])

    if len(hparams_list) < 5:
        logger.warning(
            "  > Tidak cukup data (< 5) untuk analisis SHAP. Melewatkan.")
        return

    df_hparams = pd.DataFrame(hparams_list).select_dtypes(include=np.number)



    surrogate_model = RandomForestRegressor(n_estimators=100, random_state=42)
    surrogate_model.fit(df_hparams, scores)


    explainer = shap.TreeExplainer(surrogate_model)
    shap_values = explainer.shap_values(df_hparams)

    logger.info("  > Analisis SHAP selesai. Membuat plot ringkasan...")


    shap.summary_plot(shap_values, df_hparams, show=False)

    plot_path = (
        get_path(project_id, "checkpoint_dir") /
        f"shap_summary_{project_id}.png"
    )
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    logger.info(f"  > ✅ Plot ringkasan SHAP disimpan di: {plot_path}")
    logger.info(
        "  > Plot SHAP menunjukkan parameter mana yang paling berpengaruh terhadap skor."
    )



def predict_and_visualize(
    model,
    dm,
    project_id,
    auditor,
    gemini_session: GeminiSupervisorySession,
    governor: CognitiveGovernor,
):
    """
    Membuat prediksi, memvisualisasikan, dan mengumpulkan umpan balik manusia (RLHF).
    Versi ini juga melaporkan metrik kognitif ke Governor dan mengimplementasikan protokol ASeT.
    """
    logger.info("\nMembuat prediksi berdasarkan model juara...")
    model.eval()


    logger.info(
        "🔬 Menjalankan pemeriksaan Seismograf Pasar Tektonik (ASeT) pra-prediksi..."
    )
    latest_data = dm.df_processed.iloc[-1]
    sensor1_val = latest_data.get("aset_sensor1_trend_correlation", 1.0)
    sensor2_val = latest_data.get("aset_sensor2_freq_instability", 0.0)
    s1_triggered = sensor1_val < 0.2
    s2_triggered = sensor2_val > (
        dm.df_processed["aset_sensor2_freq_instability"].quantile(0.95)
    )

    if s1_triggered and s2_triggered:
        logger.critical("🔥🔥🔥 PROTOKOL BIG BANG / SINGULARITAS AKTIF 🔥🔥🔥")
        logger.critical(
            "Semua prediksi kuantitatif ditangguhkan karena kondisi pasar yang sangat tidak stabil."
        )
        logger.critical(
            "Sistem akan fokus pada pengumpulan data 'Financial CMB'. Tidak ada prediksi yang akan dihasilkan."
        )
        governor.log_event(
            "ASET_LEVEL_3_PREDICT_HALT",
            {"reason": "Sensor 1 & 2 terpicu secara bersamaan."},
        )
        return
    elif s2_triggered:
        logger.warning("⚠️⚠️ ASET LEVEL 2: MODE PROTEKSI KRISIS AKTIF ⚠️⚠️")
        logger.warning(
            "Prediksi yang dihasilkan akan sangat konservatif dan bias terhadap penghindaran risiko."
        )
        governor.log_event("ASET_LEVEL_2_PREDICT_WARN",
                           {"instability": sensor2_val})
    elif s1_triggered:
        logger.info("🟡 ASET LEVEL 1: PERINGATAN! Kondisi Pasar Tidak Stabil. 🟡")
        logger.info(
            "Confidence interval pada prediksi akan dilebarkan secara otomatis."
        )
        governor.log_event("ASET_LEVEL_1_PREDICT_INFO",
                           {"correlation": sensor1_val})


    if len(dm.X_val) < dm.hparams.window:
        auditor.add_log(
            "PREDICT",
            "FAIL",
            f"Data validasi tidak cukup (hanya {len(dm.X_val)} baris), butuh minimal {dm.hparams.window} untuk window.",
        )
        return


    X_last_window = dm.X_val[-dm.hparams.window:]
    y_hist_last_window = dm.y_val[-dm.hparams.window:]
    spike_trains_last_window = dm.spike_trains[-dm.hparams.window:]

    x_tensor = (
        torch.from_numpy(X_last_window.astype(np.float32)
                         ).unsqueeze(0).to(model.device)
    )
    y_hist_tensor = (
        torch.from_numpy(y_hist_last_window.astype(np.float32))
        .unsqueeze(0)
        .to(model.device)
    )
    spike_tensor = (
        torch.from_numpy(spike_trains_last_window.astype(np.float32))
        .unsqueeze(0)
        .unsqueeze(2)
        .to(model.device)
    )
    edge_index = dm.graph_edge_index.to(model.device)

    anomaly_signal_value = 0.0
    try:
        anomaly_feature_name = "event_influence_score"
        anomaly_idx = dm.feature_names.index(anomaly_feature_name)
        raw_anomaly_score = X_last_window[-1, anomaly_idx]
        anomaly_signal_value = (1 - raw_anomaly_score) / 2
    except (ValueError, IndexError):
        logger.warning(
            f"Fitur '{anomaly_feature_name}' tidak ditemukan. Menggunakan sinyal anomali netral (0.0) untuk prediksi."
        )


    with torch.no_grad():
        mean_pred_returns, std_pred_returns, unc_attribution, duality_weight = (
            model.predict(
                x_tensor, y_hist_tensor, edge_index, spike_tensor, anomaly_signal_value
            )
        )

        mean_pred_returns = mean_pred_returns.cpu()
        std_pred_returns = std_pred_returns.cpu()
        unc_attribution = unc_attribution.cpu()
        duality_weight = duality_weight.item()


    wave_contribution = duality_weight * 100
    particle_contribution = (1 - duality_weight) * 100

    logger.info("\n" + "=" * 25 +
                " ANALISIS DUALITAS KUANTUM (XAI) " + "=" * 25)
    if wave_contribution >= 50:
        logger.info(
            f"🔮 Mode Prediksi: GELOMBANG (Kontribusi: {wave_contribution:.1f}%)"
        )
        logger.info(
            "   Analisis: Model lebih mengandalkan pola tren, siklus, dan data kontinu."
        )
    else:
        logger.info(
            f"⚛️ Mode Prediksi: PARTIKEL (Kontribusi: {particle_contribution:.1f}%)"
        )
        logger.info(
            "   Analisis: Model lebih reaktif terhadap sinyal peristiwa diskrit dan lonjakan mendadak."
        )

    governor.log_event(
        "QDA_STATE",
        {
            "wave_contribution": wave_contribution,
            "particle_contribution": particle_contribution,
        },
    )
    logger.info("[Meta-Kognisi] Status QDA dicatat oleh Cognitive Governor.")



    mean_uncertainty = std_pred_returns.mean().item()


    uncertainty_factor_names = dm.uncertainty_factor_names


    top_uncertainty_idx = torch.argmax(unc_attribution[0]).item()
    reason_for_uncertainty = uncertainty_factor_names[top_uncertainty_idx]
    uncertainty_confidence = unc_attribution[0][top_uncertainty_idx].item()


    logger.info("\n" + "=" * 25 + " ANALISIS KETIDAKPASTIAN (XAI) " + "=" * 25)
    logger.info(f"Tingkat ketidakpastian (Std Dev): {mean_uncertainty:.4f}")
    logger.info(
        "🚨 Alasan Utama Ketidakpastian: **%s** (Kontribusi: %.1f%%)",
        reason_for_uncertainty.replace("_", " ").title(),
        uncertainty_confidence * 100,
    )


    governor.log_event(
        "PREDICTION_UNCERTAINTY_EXPLAINED",
        details={
            "ticker": dm.tickers,
            "mean_std": float(mean_uncertainty),
            "max_std": float(std_pred_returns.max()),
            "horizon": dm.hparams.horizon,
            "primary_reason": reason_for_uncertainty,
            "reason_confidence": uncertainty_confidence,
        },
    )
    logger.info(
        f"[Meta-Kognisi] Ketidakpastian ({mean_uncertainty:.4f}) beserta penjelasannya ('{reason_for_uncertainty}') telah dicatat oleh Cognitive Governor."
    )


    mean_pred_returns, std_pred_returns = (
        mean_pred_returns.numpy(),
        std_pred_returns.numpy(),
    )

    final_returns = mean_pred_returns
    ci_lower = final_returns - 1.96 * std_pred_returns
    ci_upper = final_returns + 1.96 * std_pred_returns

    logger.info("Menerapkan validasi kewajaran harga (Sanity Check)...")
    try:
        historical_returns = dm.df_processed[
            [f"{t}_log_return" for t in dm.tickers]
        ].dropna()
        max_reasonable_daily_change = historical_returns.quantile(0.995).values
        min_reasonable_daily_change = historical_returns.quantile(0.005).values
    except Exception as e:
        logger.warning(
            f"Gagal menghitung batas kewajaran historis: {e}. Melewatkan sanity check."
        )
        max_reasonable_daily_change = None

    last_known_date = pd.to_datetime(dm.df_processed.index[-1])
    horizon_trading_days = dm.hparams.horizon
    pred_trading_dates = get_next_trading_days(
        last_known_date, horizon_trading_days)

    if not pred_trading_dates:
        logger.info("Tidak ada hari trading di masa depan untuk diprediksi.")
        return

    pred_prices = np.zeros((horizon_trading_days, len(dm.tickers)))
    lo_prices = np.zeros((horizon_trading_days, len(dm.tickers)))
    hi_prices = np.zeros((horizon_trading_days, len(dm.tickers)))

    current_price_p = dm.last_known_prices.values
    current_price_l = dm.last_known_prices.values
    current_price_h = dm.last_known_prices.values

    for i in range(horizon_trading_days):
        daily_return = final_returns[0, i, :]
        daily_ci_low, daily_ci_high = ci_lower[0, i, :], ci_upper[0, i, :]

        if max_reasonable_daily_change is not None:
            clamped_daily_return = np.clip(
                daily_return, min_reasonable_daily_change, max_reasonable_daily_change
            )
            if not np.array_equal(daily_return, clamped_daily_return):
                auditor.add_log(
                    "PREDICT_CLAMP",
                    "WARN",
                    f"Prediksi hari ke-{i+1} tidak wajar dan telah dibatasi.",
                )
            final_daily_return = clamped_daily_return
        else:
            final_daily_return = daily_return

        pred_prices[i, :] = current_price_p * np.exp(final_daily_return)
        lo_prices[i, :] = current_price_l * np.exp(daily_ci_low)
        hi_prices[i, :] = current_price_h * np.exp(daily_ci_high)

        current_price_p, current_price_l, current_price_h = (
            pred_prices[i, :],
            lo_prices[i, :],
            hi_prices[i, :],
        )

    df_trading_only = pd.DataFrame(index=pd.to_datetime(pred_trading_dates))
    for i, ticker in enumerate(dm.tickers):
        df_trading_only[f"{ticker}_Pred"] = pred_prices[:, i]
        df_trading_only[f"{ticker}_Low_CI"] = lo_prices[:, i]
        df_trading_only[f"{ticker}_High_CI"] = hi_prices[:, i]

    full_pred_horizon = pd.date_range(
        start=last_known_date + timedelta(days=1), end=pred_trading_dates[-1], freq="D"
    )
    df_out = df_trading_only.reindex(full_pred_horizon).ffill()
    if not df_out.empty:
        df_out.fillna(method="bfill", inplace=True)

    logger.info("\n--- Prediksi Harga Final ---")
    logger.info(df_out.round(2))


    logger.info("\n" + "=" * 25 + " ANALISIS KINERJA PREDIKSI " + "=" * 25)
    try:

        pred_cols = [col for col in df_out.columns if "_Pred" in col]
        predicted_returns = df_out[pred_cols].pct_change().dropna()


        portfolio_predicted_returns = predicted_returns.mean(axis=1)


        risk_free_rate = 0.02


        mean_return = portfolio_predicted_returns.mean() * 252
        std_return = portfolio_predicted_returns.std() * np.sqrt(252)

        if std_return > 0:
            sharpe_ratio_pred = (mean_return - risk_free_rate) / std_return
            logger.info(
                f"📈 Sharpe Ratio Tahunan yang Diprediksi: {sharpe_ratio_pred:.2f}"
            )
        else:
            logger.info(
                "📈 Sharpe Ratio Tahunan yang Diprediksi: N/A (Return konstan)")
            sharpe_ratio_pred = 0

        governor.log_event(
            "PREDICTION_PERFORMANCE_METRICS",
            {
                "project_id": project_id,
                "predicted_sharpe_ratio": sharpe_ratio_pred,
                "predicted_annual_return": mean_return,
                "predicted_annual_volatility": std_return,
            },
        )
        logger.info(
            "[Meta-Kognisi] Metrik kinerja prediksi (termasuk Sharpe Ratio) dicatat oleh Governor."
        )

    except Exception as e:
        logger.error(f"Gagal menghitung Sharpe Ratio prediksi: {e}")


    plot_title = (
        f"Prediksi Harga ({dm.hparams.horizon} Hari) | "
        f"Mode: {'Gelombang' if wave_contribution >= 50 else 'Partikel'} ({wave_contribution:.0f}/{particle_contribution:.0f}%)"
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(15, 8))
    for ticker in dm.tickers:
        plt.plot(
            df_out.index,
            df_out[f"{ticker}_Pred"],
            label=f"{ticker} Prediksi",
            marker="o",
            linestyle="-",
        )
        plt.fill_between(
            df_out.index,
            df_out[f"{ticker}_Low_CI"],
            df_out[f"{ticker}_High_CI"],
            alpha=0.2,
            label=f"{ticker} 95% CI",
        )

    plt.title(plot_title, fontsize=16)
    plt.ylabel("Harga", fontsize=12)
    plt.xlabel("Tanggal", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = get_path(project_id, "prediction_plot")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"📈 Plot prediksi disimpan di: {plot_path}")

    logger.info("\n" + "=" * 20 +
                " TAHAP UMPAN BALIK SUPERVISOR (RLHF) " + "=" * 20)

    feedback_manager = HumanFeedbackManager(
        get_path(None, "human_feedback_db"))

    rating_choice = questionary.select(
        "Bagaimana penilaian Anda terhadap prediksi ini?",
        choices=[
            "👍 Bagus (Sesuai ekspektasi/logis)",
            "👎 Buruk (Tidak realistis/salah arah)",
        ],
    ).ask()


    example_ticker = dm.tickers[0] if dm.tickers else "saham"
    prompt_text = f"Berikan komentar singkat (misal: 'terlalu optimis', 'volatilitas terlalu rendah', 'target {example_ticker} kurang tinggi'):"

    comment = questionary.text(prompt_text, default="").ask()


    if not comment:
        comment = "Tidak ada komentar spesifik."


    rating = 1 if rating_choice and rating_choice.startswith("👍") else -1


    hparams_used = dm.hparams

    feedback_manager.record_feedback(
        project_id=project_id,
        prediction_summary=df_out.to_dict("list"),
        hparams=hparams_used,
        rating=rating,
        comment=comment.strip(),
    )

    if rating == -1:

        last_reflection_report = governor.generate_self_reflection_report()


        post_mortem_analyzer = PostMortemAnalyzer(model.api_pool, governor)

        analysis_report = post_mortem_analyzer.analyze_failure(
            project_id=project_id,
            feedback={"rating_text": rating_choice, "comment": comment},
            hparams=hparams_used,
            metrics={"final_uncertainty": mean_uncertainty},
            last_reflection=last_reflection_report,
        )

        if analysis_report:
            print("\n--- LAPORAN POST-MORTEM & KOREKSI DIRI ---")
            print(
                f"🔬 Akar Masalah Terduga: {analysis_report.suspected_root_cause}")
            print("💡 Usulan Perbaikan untuk Putaran Berikutnya:")
            for action in analysis_report.corrective_actions_proposed:
                print(f"    - {action}")
            print("---------------------------------------------")

    auditor.audit_prediction(
        df_out, dm.last_known_prices
    )
    auditor.generate_report()

    final_details = {
        "auditor_log": auditor.log,
        "prediction_dataframe": df_out.to_dict(),
        "last_known_prices": dm.last_known_prices.to_dict(),
    }
    gemini_session.start_supervision("FINAL_ANALYSIS", final_details)





def calculate_dynamic_clip_val(df: pd.DataFrame, tickers: list) -> float:
    """
    Menghitung nilai gradient clipping berdasarkan volatilitas tahunan rata-rata.
    
    Args:
        df (pd.DataFrame): DataFrame yang berisi data harga penutupan.
        tickers (list): Daftar ticker saham yang relevan.

    Returns:
        float: Nilai gradient clipping yang disarankan.
    """
    volatilities = []
    for t in tickers:
        close_col = f"{t}_Close"
        if close_col in df.columns:
            log_returns = np.log(df[close_col] / df[close_col].shift(1))
            annual_volatility = log_returns.std() * np.sqrt(252)
            volatilities.append(annual_volatility)

    avg_volatility = np.mean(volatilities) if volatilities else 0.3
    dynamic_val = max(0.5, 1.5 - avg_volatility) 

    logger.info(f"Volatilitas dataset: {avg_volatility:.2f}. Gradient Clip Val dinamis diatur ke: {dynamic_val:.2f}")
    return dynamic_val



def generate_dynamic_hparams(base_hparams: dict) -> tuple[dict, list]:
    """
    Secara dinamis menghasilkan hyperparameter kunci, termasuk memisahkan
    fitur primer dan sisa, serta menjalankan seleksi fitur jika cache tidak ditemukan.
    """
    logger.info("\n--- 🧠 Memulai Konfigurasi Hyperparameter Otomatis (v4 - Terpadu & Efisien) ---")

    project_id = base_hparams["project_id"]
    cached_features_path = get_path(project_id, "selected_features")
    definitive_features = []

    if cached_features_path.exists():
        logger.warning(f"CACHE DITEMUKAN! Memuat daftar fitur dari: {cached_features_path}")
        with open(cached_features_path, "r") as f:
            definitive_features = json.load(f)
    else:
        logger.info("Cache tidak ditemukan. Menjalankan proses seleksi fitur penuh...")
        try:
            df_full = pd.read_parquet(base_hparams["data_path"])
            df_full["Date"] = pd.to_datetime(df_full.get("Date", df_full.index))
            df_full = df_full.set_index("Date").sort_index()
            num_rows = len(df_full)

            df_featured, _ = generate_all_features(
                df_full,
                base_hparams["selected_tickers"],
                master_event_list=[],
                use_chat_features=False,
            )
            num_initial_features = len(df_featured.columns)
            upper_cap = max(150, min(int(num_initial_features * 0.25), 750))
            dynamic_top_k = max(50, min(int(num_rows / 15), upper_cap))

            logger.info(
                f"Baris: {num_rows}, Fitur Awal: {num_initial_features}. "
                f"Batas atas fitur diatur ke: {upper_cap}. "
                f"Final top_k_features dinamis menjadi: {dynamic_top_k}"
            )
            base_hparams["top_k_features"] = dynamic_top_k

            df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_featured = df_featured.fillna(method="ffill").fillna(method="bfill").fillna(0)

            target_cols = []
            for t in base_hparams["selected_tickers"]:
                close_col = f"{t}_Close"
                log_return_col = f"{t}_log_return"
                if close_col in df_featured.columns:
                    df_featured[log_return_col] = np.log(df_featured[close_col] / df_featured[close_col].shift(1))
                    target_cols.append(log_return_col)

            valid_target_cols = [col for col in target_cols if col in df_featured.columns]
            if not valid_target_cols:
                raise ValueError("Tidak ada kolom target yang valid. Proses tidak bisa dilanjutkan.")

            y_sel = df_featured[valid_target_cols].mean(axis=1).bfill().ffill()
            if y_sel.isnull().any():
                y_sel = y_sel.fillna(y_sel.mean())
            
            all_potential_features = [c for c in df_featured.columns if c not in target_cols]
            
            selected_features_all, _ = select_features(
                X=df_featured[all_potential_features],
                y=y_sel,
                top_k=base_hparams.get("top_k_features"),
            )
            definitive_features = selected_features_all

            if definitive_features:
                logger.info(f"Menyimpan {len(definitive_features)} fitur terpilih ke cache: {cached_features_path}")
                with open(cached_features_path, "w") as f:
                    json.dump(definitive_features, f)
            else:
                logger.error("Seleksi fitur tidak menghasilkan fitur apa pun. File cache tidak akan dibuat.")

        except Exception as e:
            logger.error(f"FATAL: Gagal saat konfigurasi dinamis. Error: {e}", exc_info=True)
            raise e

    if not definitive_features:
        raise ValueError("Tidak ada fitur definitif yang dapat diproses.")

    n_features_total = len(definitive_features)
    n_primary_target = int(n_features_total * 0.4)
    n_primary = max(30, min(n_primary_target, 100))
    n_primary = min(n_primary, n_features_total)
    
    final_primary_features = definitive_features[:n_primary]
    final_leftover_features = definitive_features[n_primary:]
    n_primary_count = len(final_primary_features)
    n_leftover_count = len(final_leftover_features)

    logger.info(f"✅ Pemisahan Selesai. Fitur Total: {n_features_total} (Primer: {n_primary_count}, Sisa/Sidecar: {n_leftover_count})")

    df_full_for_adjust = pd.read_parquet(base_hparams["data_path"])
    adjusted_hparams = dynamically_adjust_parameters(base_hparams, df_full_for_adjust)

    adjusted_hparams["n_features_input"] = n_features_total
    adjusted_hparams["n_primary_features"] = n_primary_count
    adjusted_hparams["n_leftover_features"] = n_leftover_count
    adjusted_hparams["n_targets"] = len(base_hparams["selected_tickers"])

    logger.info("--- ✅ Konfigurasi Hyperparameter Otomatis Selesai ---")
    return adjusted_hparams, definitive_features

def _one_train_run(
    hparams: dict,
    auditor: "CriticalAuditor",
    api_pool: "DistributedAIPool",
    together_api_keys: dict,
    gemini_api_config: dict,
    web_searcher: "WebSearchManager",
    governor: "CognitiveGovernor",
    brain: "Brain",
    nsmm: "NSMM",
    load_checkpoint_path: Optional[str] = None,
    custom_callbacks: Optional[list] = None,
    blacklist_path: Optional[Path] = None,
) -> tuple[float, dict, str, dict, "pl.Trainer"]:
    """
    Menjalankan satu siklus pelatihan lengkap dengan arsitektur final,
    termasuk lapisan pertahanan, inisialisasi model yang cerdas, dan callbacks dinamis.
    """
    current_hparams = hparams.copy()
    run_id = f"{current_hparams.get('project_id', 'proj')}_{current_hparams.get('attempt', 'baseline')}"
    logger.info(f"🚀 Memulai Run ID: {run_id} | Mode: {current_hparams.get('mode')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    causal_auditor = current_hparams.get("causal_auditor")
    if not causal_auditor:
        raise ValueError("CausalInternalAuditor harus diteruskan dalam hparams.")


    dm = AlphaDataModule(current_hparams, auditor, api_pool, together_api_keys, 
                         gemini_api_config, web_searcher, brain=brain, engine=async_engine, blacklist_path=blacklist_path)
    dm.setup(stage="fit")
    current_hparams["augmentor"] = dm.augmentor
    current_hparams["n_features_input"] = dm.n_features_input


    project_id = current_hparams.get("project_id")
    checkpoint_dir = get_path(project_id, "checkpoint_dir")
    
    feature_sig = f"f{dm.n_features_input}"  
    armor_path = checkpoint_dir / f"input_armor_{project_id}_{feature_sig}.pth"
    input_armor = InputArmor(input_dim=dm.n_features_input).to(device)
    
    if armor_path.exists() and os.path.getsize(armor_path) > 0:
        ckpt = torch.load(armor_path, map_location=device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt_state = ckpt["state_dict"]
            ckpt_in_dim = ckpt.get("input_dim")
        else:
            ckpt_state = ckpt
            ckpt_in_dim = None

        if (ckpt_in_dim is not None) and (ckpt_in_dim == dm.n_features_input):
            input_armor.load_state_dict(ckpt_state, strict=True)
            input_armor.is_trained = True
        else:
            logger.warning(
                f"[InputArmor] Checkpoint tidak cocok (ckpt_in_dim={ckpt_in_dim}, "
                f"current_in_dim={dm.n_features_input}). Melatih ulang armor..."
            )
            armor_train_data = TensorDataset(torch.from_numpy(dm.X_train.astype(np.float32)))
            armor_loader = TorchDataLoader(armor_train_data, batch_size=256, shuffle=True)
            input_armor.train_armor(armor_loader, epochs=15)
            torch.save({
                "state_dict": input_armor.state_dict(),
                "input_dim": dm.n_features_input,
                "hidden_dim": int(dm.n_features_input * 0.5),
                "hidden_dim_ratio": 0.5,
            }, armor_path)
    else:
        armor_train_data = TensorDataset(torch.from_numpy(dm.X_train.astype(np.float32)))
        armor_loader = TorchDataLoader(armor_train_data, batch_size=256, shuffle=True)
        input_armor.train_armor(armor_loader, epochs=15)
        torch.save({
            "state_dict": input_armor.state_dict(),
            "input_dim": dm.n_features_input,
            "hidden_dim": int(dm.n_features_input * 0.5),
            "hidden_dim_ratio": 0.5,
        }, armor_path)

    

    input_armor.eval()
    with torch.no_grad():
        armored_train = input_armor(torch.from_numpy(dm.X_train.astype(np.float32)).to(device)).cpu().numpy()
        armored_val = input_armor(torch.from_numpy(dm.X_val.astype(np.float32)).to(device)).cpu().numpy()
        
        poly_layer = PolycarbonateLayer(input_dim=dm.n_features_input, hidden_dim=current_hparams["d_model"]).to(device)
        dm.X_train = poly_layer(torch.from_numpy(armored_train).to(device)).cpu().numpy()
        dm.X_val = poly_layer(torch.from_numpy(armored_val).to(device)).cpu().numpy()


    denoising_shield_path = checkpoint_dir / f"denoising_shield_{project_id}.pth"
    meta_path = denoising_shield_path.with_suffix(".meta.json")
    
    denoising_shield = DiffusionDenoisingModel(
        input_dim=dm.n_features_input, d_model=current_hparams["d_model"],
        n_heads=current_hparams["n_heads"], num_layers=2, seq_len=current_hparams.get("window", 60)
    ).to(device)


    feature_sig = f"f{int(dm.n_features_input)}"
    sig_path  = denoising_shield_path.with_name(
        f"{denoising_shield_path.stem}_{feature_sig}{denoising_shield_path.suffix}"
    )
    base_path = denoising_shield_path

    def _safe_load_shield(load_path: Path):
        """Load ckpt dengan reconcile proyeksi + filter shape + strict=False (tanpa size-mismatch)."""
        raw = torch.load(load_path, map_location=device)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw


        try:
            if hasattr(denoising_shield, "input_proj") and isinstance(denoising_shield.input_proj, nn.Linear):
                _reconcile_linear_shape_from_state(
                    denoising_shield.input_proj, "input_proj.weight", "input_proj.bias", state
                )
            if hasattr(denoising_shield, "output_proj") and isinstance(denoising_shield.output_proj, nn.Linear):
                _reconcile_linear_shape_from_state(
                    denoising_shield.output_proj, "output_proj.weight", "output_proj.bias", state
                )
        except Exception as e:
            logger.warning(f"[Shield reconcile] dilewati: {e}")


        model_state = denoising_shield.state_dict()
        filtered_state = {
            k: v for k, v in state.items()
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape)
        }
        denoising_shield.load_state_dict(filtered_state, strict=False)


    force_retrain_shield = False


    if not sig_path.exists() and not base_path.exists():
        logger.info("🛡️ [Shield] Tidak ditemukan checkpoint (base maupun signature). Pelatihan dari nol.")
        force_retrain_shield = True
    else:

        try:
            meta_candidate = sig_path if sig_path.exists() else base_path
            meta_path = meta_candidate.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                if int(metadata.get("input_dim", -1)) != int(dm.n_features_input):
                    logger.info("🛡️ [Shield] input_dim meta ≠ current → retrain.")
                    force_retrain_shield = True

        except Exception:
            pass


    if force_retrain_shield:
        train_denoising_shield(project_id, dm, current_hparams)


    load_path = sig_path if sig_path.exists() else base_path
    logger.info(f"Memuat Perisai Energi dari: {load_path}")
    _safe_load_shield(load_path)



    gnn_communicator_path = checkpoint_dir / f"gnn_communicator_{project_id}.pth"
    gnn_communicator = GraphCommunicationNetwork(num_nodes=current_hparams["n_layers"] + 1).to(device)
    if gnn_communicator_path.exists():
        gnn_communicator.load_state_dict(torch.load(gnn_communicator_path, map_location=device))


    model = None
    resume_from_ckpt = None
    model_class = HybridSSP_LitModule if current_hparams.get("mode") == "pre-train" else AlphaLit

    common_args = {"hparams": current_hparams}
    alphslit_args = {}
    if model_class == AlphaLit:
        alphslit_args = {
            "feature_names": dm.feature_names,
            "event_names": dm.event_names,
            "api_pool": api_pool,
            "brain": brain,
            "causal_auditor": causal_auditor,
            "nsmm": nsmm,
            "denoising_model": denoising_shield,
            "gnn_communicator": gnn_communicator,
            "gemini_advisor": None,
            "tgw_memory": None,
            "energy_shield": None,
            "echo_radio": None,
            "engine": async_engine,
        }


    if current_hparams.get("mode") != "pre-train" and load_checkpoint_path and os.path.exists(load_checkpoint_path):
        try:
            model = AlphaLit.load_from_checkpoint(
                checkpoint_path=load_checkpoint_path, map_location="cpu", strict=False, 
                hparams_file=None, **common_args, **alphslit_args
            )
            resume_from_ckpt = load_checkpoint_path
        except Exception:
            pass 

    if model is None:
        logger.info(f"🤖 Membuat model {model_class.__name__} baru dari awal...")
        if model_class == AlphaLit:
            model = model_class(**common_args, **alphslit_args)
        else:
            model = model_class(**common_args)


    resource_manager = ResourceManager()
    dm.num_workers_override = resource_manager.calculate_optimal_workers()
    


    monitor_metric = "val_loss" if current_hparams.get('mode') != 'pre-train' else 'pretrain_loss_epoch'
    
    checkpoint_cb = ModelCheckpoint(dirpath=checkpoint_dir, filename=f"best_model_{run_id}", 
                                      save_top_k=1, monitor=monitor_metric, mode="min")
    earlystop_cb = EarlyStopping(monitor=monitor_metric, patience=35, mode="min", verbose=True)

    active_callbacks = [
        earlystop_cb, checkpoint_cb, RichProgressBar(), 


        SmartHybridScheduler(target_switch_epoch=current_hparams.get("switch_epoch", 20)),
        WeightDecayScheduler(
            wd_max=current_hparams.get("weight_decay", 1e-4) * 10, wd_min=current_hparams.get("weight_decay", 1e-4),
            max_epochs=current_hparams.get("max_epochs", 60)
        ),
        DynamicBetaVIB(initial_beta=1e-6, final_beta=1e-3, ramp_up_duration_epochs=30, start_ramp_up_epoch=10),
        AdaptiveCapacityScheduler(
            patience=5,
            danger_zone=(0.3, 0.6),
            confirmation_patience=3,
            supervisor_agent="Llama-4-Maverick"
        ),
        armh_cb := ARMH_Callback(api_pool=api_pool, together_keys=together_api_keys),
        HardSampleReplay_Callback(armh_cb),
    ]

    if isinstance(model, AlphaLit):
        aset_geometry_monitor = LatentSpaceGeometryMonitor(project_id=project_id, hht_feature_names=dm.pure_hht_features)
        if dm.df_processed is not None and not dm.df_processed.empty:
            aset_geometry_monitor.train(dm.df_processed)
        active_callbacks.append(ASeT_ProtocolMonitor(aset_geometry_monitor, governor))

    if custom_callbacks: active_callbacks.extend(custom_callbacks)
    if blacklist_path and blacklist_path.exists():
        active_callbacks.append(BlacklistLRScheduler(blacklist_path=blacklist_path))
    if current_hparams.get("mode") == "fine-tune":
        active_callbacks.append(SynapticIntelligenceCallback(project_id=project_id, strength=0.8))
    if current_hparams.get("mode") == "pre-train":
        active_callbacks.append(AdaptiveLearningRateController())



    try:
        resume_from_ckpt
    except NameError:
        resume_from_ckpt = None

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
        max_epochs=current_hparams.get("max_epochs", 60),
        callbacks=active_callbacks, enable_model_summary=True
    )


    fallback_lr = float(os.getenv("LRF_FALLBACK", "1e-3"))
    _new_lr = getattr(getattr(model, "hparams", model), "lr", None)
    if _new_lr is None:
        if hasattr(model, "hparams") and hasattr(model.hparams, "lr"):
            model.hparams.lr = fallback_lr
        elif hasattr(model, "learning_rate"):
            model.learning_rate = fallback_lr
    _pre_ctx = STAGE_PREP.wait("pretrain_prep", timeout=600)


    STAGE_PREP.trigger("fine_tune_prep", lambda ctx=_pre_ctx: build_finetune_assets(ctx))
    STAGE_PREP.trigger("predict_prep",   lambda: build_predict_assets(current_hparams))


    _ft = STAGE_PREP.wait("fine_tune_prep", timeout=2)
    if _ft and _ft.get("last_ckpt"):
        resume_from_ckpt = _ft["last_ckpt"]


    _ = STAGE_PREP.wait("predict_prep", timeout=2)




    if resume_from_ckpt:
        try:
            _ckpt = torch.load(resume_from_ckpt, map_location="cpu")
            _sd = _ckpt.get("state_dict", _ckpt)
            _model_sd = model.state_dict()
            _overlap = len(set(_sd.keys()) & set(_model_sd.keys()))
            _ratio = _overlap / max(1, len(set(_sd.keys()) | set(_model_sd.keys())))
            if _ratio < 0.5:
                logger.warning(f"🔄 Checkpoint {resume_from_ckpt} tampak tidak kompatibel (overlap={_overlap}, ratio={_ratio:.2f}). Melatih dari awal.")
                resume_from_ckpt = None
            else:
                missing = [k for k in _model_sd.keys() if k not in _sd]
                unexpected = [k for k in _sd.keys() if k not in _model_sd]
                if missing or unexpected:
                    logger.warning(f"🧩 Partial restore: missing={len(missing)}, unexpected={len(unexpected)}. Memuat dengan strict=False.")
                model.load_state_dict(_sd, strict=False)

                resume_from_ckpt = None
        except Exception as e:
            logger.warning(f"⚠️ Gagal memvalidasi checkpoint {resume_from_ckpt}: {e}. Melatih dari awal.")
            resume_from_ckpt = None
    
    trainer.fit(model, datamodule=dm, ckpt_path=resume_from_ckpt)


    score = trainer.callback_metrics.get(monitor_metric, float("inf")).item()
    full_metrics = {k: v.item() for k, v in trainer.callback_metrics.items()}
    best_model_path = checkpoint_cb.best_model_path or ""
    full_metrics["checkpoint_path"] = best_model_path

    data_summary = {
        "num_rows": len(dm.df_processed) if dm.df_processed is not None else 0,
        "num_features_input": dm.n_features_input,
        "start_date": str(dm.df_processed.index.min().date()) if dm.df_processed is not None else "N/A",
        "end_date": str(dm.df_processed.index.max().date()) if dm.df_processed is not None else "N/A",
        "tickers": dm.tickers,
    }

    return score, full_metrics, best_model_path, data_summary, trainer



def pre_train_defensive_systems(
    hparams: dict,
    auditor,
    api_pool,
    together_api_keys,
    gemini_api_config,
    web_searcher,
    brain,
    async_engine: "AsyncCuriosityEngine",
):
    """
    Fungsi orkestrasi untuk melatih semua komponen pertahanan secara mandiri.
    """
    logger.info(
        "\n"
        + "=" * 80
        + "\n"
        + " MEMULAI PELATIHAN SISTEM PERTAHANAN (DEFENSIVE SYSTEMS PRE-TRAINING) ".center(
            80, "="
        )
        + "\n"
        + "=" * 80
    )

    project_id = hparams["project_id"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dm = AlphaDataModule(
        hparams,
        auditor,
        api_pool,
        together_api_keys,
        gemini_api_config,
        web_searcher,
        brain,
        engine=async_engine,
    )
    dm.setup(stage="fit")


    train_denoising_shield(project_id, dm, hparams)


    train_gnn_communicator(project_id, hparams["n_layers"], device)

    logger.info("\n✅ Pelatihan Sistem Pertahanan Selesai.")


def _reconcile_linear_shape_from_state(linear: nn.Linear, weight_key: str, bias_key: str, state: dict):
    W_old = state.get(weight_key, None)
    b_old = state.get(bias_key, None)
    if W_old is None:
        return
    out_new, in_new = linear.weight.shape
    out_old, in_old = W_old.shape


    with torch.no_grad():
        W_new = torch.zeros((out_new, in_new), dtype=linear.weight.dtype, device=linear.weight.device)
        copy_out = min(out_new, out_old)
        copy_in  = min(in_new, in_old)
        W_new[:copy_out, :copy_in] = W_old[:copy_out, :copy_in]
        linear.weight.copy_(W_new)

        if linear.bias is not None:
            if b_old is not None:
                b_new = torch.zeros((out_new,), dtype=linear.bias.dtype, device=linear.bias.device)
                b_new[:min(out_new, b_old.shape[0])] = b_old[:min(out_new, b_old.shape[0])]
                linear.bias.copy_(b_new)
            else:

                pass


    state.pop(weight_key, None)
    state.pop(bias_key, None)


def train_denoising_shield(
    project_id: str, dm: AlphaDataModule, hparams: dict
) -> nn.Module:
    """
    Melatih Perisai Energi (Diffusion Denoising Model) secara terpisah.
    """

    shield_base = (
        get_path(project_id, "checkpoint_dir") /
        f"denoising_shield_{project_id}.pth"
    )
    shield_base.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    num_features = int(dm.n_features_input)

    model = DiffusionDenoisingModel(
        input_dim=num_features,
        d_model=hparams["d_model"],
        n_heads=hparams["n_heads"],
        num_layers=2,
        seq_len=hparams.get("window", 60)
    ).to(device)


    feature_sig = f"f{num_features}"
    shield_sig_path = shield_base.with_name(f"{shield_base.stem}_{feature_sig}{shield_base.suffix}")
    load_path = shield_sig_path if shield_sig_path.exists() else shield_base

    if load_path.exists():
        logger.info(f"Memuat Perisai Energi yang sudah ada dari: {load_path}")


        raw = torch.load(load_path, map_location=device)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw


        saved_input_dim = None
        try:
            meta_path = load_path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    saved_input_dim = meta.get("input_dim", None)
        except Exception:
            pass


        try:
            if hasattr(model, "input_proj") and isinstance(model.input_proj, nn.Linear):
                _reconcile_linear_shape_from_state(
                    model.input_proj, "input_proj.weight", "input_proj.bias", state
                )
            if hasattr(model, "output_proj") and isinstance(model.output_proj, nn.Linear):
                _reconcile_linear_shape_from_state(
                    model.output_proj, "output_proj.weight", "output_proj.bias", state
                )
        except Exception as e:
            logger.warning(f"[Shield reconcile] dilewati: {e}")


        model_state = model.state_dict()
        filtered_state, skipped = {}, []
        for k, v in state.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                filtered_state[k] = v
            else:
                exp_shape = tuple(model_state[k].shape) if k in model_state else ("<missing>",)
                skipped.append((k, tuple(v.shape), exp_shape))


        missing, unexpected = model.load_state_dict(filtered_state, strict=False)


        if skipped:

            preview = ", ".join([n for n, _, _ in skipped[:6]])
            logger.warning(
                f"⚠️ Memuat sebagian Perisai Energi: "
                f"{len(filtered_state)}/{len(state)} parameter cocok. "
                f"Dilewati (contoh): {preview}"
                + (" ..." if len(skipped) > 6 else "")
            )
            if saved_input_dim is not None:
                try:
                    current_in = int(model.input_proj.weight.shape[1])
                    if saved_input_dim != current_in:
                        logger.warning(
                            f"⚠️ input_dim checkpoint={saved_input_dim} ≠ current={current_in}. "
                            f"Lapisan proyeksi akan tetap pakai inisialisasi baru (bukan dari checkpoint)."
                        )
                except Exception:

                    pass

        if missing:
            logger.debug(f"Parameter hilang saat load (expected by model): {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            logger.debug(f"Parameter tak-terduga dari checkpoint: {unexpected[:10]}{' ...' if len(unexpected)>10 else ''}")


        return model.eval()


    return model

    logger.info(
        "\n--- 🛡️ Memulai Pelatihan Perisai Energi (Diffusion Model) ---")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    train_data = torch.from_numpy(
        dm.X_train.astype(np.float32)
    )
    train_loader = TorchDataLoader(
        TensorDataset(train_data),
        batch_size=hparams.get("batch_size", 64),
        shuffle=True,
    )

    epochs = 20
    num_diffusion_timesteps = 1000

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(
            train_loader, desc=f"Shield Epoch {epoch+1}/{epochs}", leave=False
        ):
            optimizer.zero_grad()
            x_0 = batch[0].to(device)

            if x_0.ndim == 2:
                x_0 = x_0.unsqueeze(1)
            

            if x_0.shape[1] < hparams.get("window", 60):
                 padding = torch.zeros(x_0.shape[0], hparams.get("window", 60) - x_0.shape[1], x_0.shape[2], device=device)
                 x_0 = torch.cat([x_0, padding], dim=1)
            elif x_0.shape[1] > hparams.get("window", 60):
                 x_0 = x_0[:, :hparams.get("window", 60), :]


            loss = calculate_diffusion_loss(
                model, x_0, num_diffusion_timesteps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            f"Shield Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(train_loader):.6f}"
        )


    to_save = {"state_dict": model.state_dict()}


    torch.save(to_save, shield_path)


    feature_sig = f"f{num_features}"
    shield_sig_path = shield_path.with_name(f"{shield_path.stem}_{feature_sig}{shield_path.suffix}")
    torch.save(to_save, shield_sig_path)


    meta = {
        "architecture": "DiffusionDenoisingModel",
        "input_dim": num_features,
        "d_model": hparams["d_model"],
        "n_heads": hparams["n_heads"],
        "saved_at": datetime.now().isoformat()
    }
    try:
        with open(shield_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        with open(shield_sig_path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
    except Exception as e:
        logger.warning(f"Gagal menulis metadata shield: {e}")

    logger.info(f"✅ Perisai Energi tersimpan: {shield_path.name} & {shield_sig_path.name}")
    return model.eval()



def run_continuous_training(
    initial_hparams: dict,
    auditor: "CriticalAuditor",
    api_pool: "DistributedAIPool",
    gemini_api_config: dict,
    together_api_keys: dict,
    together_roles: dict,
    brain: "Brain",
    web_searcher: "WebSearchManager",
):
    """
    Menjalankan loop pelatihan berkelanjutan yang canggih dengan optimisasi AI,
    audit kausal, dan evolusi arsitektur.
    """
    MAX_ROUNDS_AI = 3
    monitor_metric = "val_loss"
    improvement_threshold = 0.01

    logger.info("--- TAHAP 0: Inisialisasi Komponen Sistem Canggih ---")
    project_id = initial_hparams["project_id"]



    if not web_searcher:
        raise ValueError("Argument 'web_searcher' harus disediakan dan tidak boleh None.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgw_memory = TGW_Memory(device=device)

    tracker = ExperimentTracker()
    governor = CognitiveGovernor(project_id, api_pool)

    nsmm = NSMM(project_id="global_main")
    acr_callback = ACR_Callback(
        nsmm, api_pool, together_api_keys, brain, _supremacy_embed_model)
    strategy_kb_path = MODELS_DIR / f"strategy_kb_{project_id}.sqlite"
    strategy_kb = StrategyKnowledgeBase(db_path=strategy_kb_path)
    feedback_manager = HumanFeedbackManager(
        get_path(None, "human_feedback_db"))

    causal_auditor = CausalInternalAuditor(project_id=project_id)
    causal_auditor.run_regime_clustering()
    causal_auditor.run_causal_audit()

    accreditor = ProposalAccreditor(together_keys=together_api_keys)
    panel_musyawarah = PanelMusyawarahAI(
        panelis={
            name: config
            for name, config in together_roles.items()
            if together_api_keys.get(name) or api_pool.get_worker(name)
        },
        arbiter_model_name="maverick",
        translator=AITranslator(together_api_keys),
        together_api_keys=together_api_keys,
        gemini_api_pool=api_pool,
        together_roles=together_roles,
    )
    strategist = GenerativeStrategist(api_pool, strategy_kb)
    architect = ArchitectAI(api_pool)
    encoder = StrategyEncoder(api_pool)

    logger.info(
        "\n--- TAHAP 1: Pra-Pelatihan Memori Jangka Panjang (ML-DKG) ---")
    dkg_embedding_path = None
    if initial_hparams.get("use_ml_dkg", False):
        try:
            df_for_causal = pd.read_parquet(initial_hparams["data_path"])
            dm_for_causal = AlphaDataModule(
                initial_hparams,
                auditor,
                api_pool,
                together_api_keys,
                gemini_api_config,
                web_searcher,
                brain,
            )
            dm_for_causal.setup()

            causal_engine = CausalInferenceEngine(
                api_pool,
                dm_for_causal.df_processed,
                dm_for_causal.feature_names,
                dm_for_causal.target_cols,
                dm_for_causal.tickers,
                None,
                project_id,
            )
            causal_engine.setup_causal_model()

            embedding_dict = train_ml_dkg(causal_engine.dkg, project_id)
            if embedding_dict:
                dkg_embedding_path = (
                    get_path(project_id, "checkpoint_dir")
                    / f"ml_dkg_embeddings_{project_id}.pt"
                )
                initial_hparams["dkg_embedding_path"] = str(dkg_embedding_path)

        except Exception as e:
            logger.error(
                f"Gagal dalam pra-pelatihan DKG. Melanjutkan tanpa memori DKG. Error: {e}",
                exc_info=True,
            )
            initial_hparams["dkg_embedding_path"] = None
    else:
        logger.info("Pra-pelatihan ML-DKG dilewati sesuai konfigurasi.")

    logger.info(
        "\n--- TAHAP 2: Memuat State Terakhir untuk Continual Learning ---")
    best_overall_checkpoint = None
    pretrain_model_path = get_path(project_id, "pretrained_encoder")
    if pretrain_model_path.exists():
        logger.info(f"✅ Menemukan model pre-train di: {pretrain_model_path}")
        logger.info(
            "Mempersiapkan untuk mengambil snapshot TGW dari model pre-train..."
        )
        pretrain_hparams = initial_hparams.copy()
        pretrain_hparams["mode"] = "pre-train"
        pretrain_model = SSP_LitModule(pretrain_hparams)
        pretrain_model.encoder.load_state_dict(
            torch.load(pretrain_model_path, map_location=device)
        )

        pretrain_dm = AlphaDataModule(
            pretrain_hparams,
            auditor,
            api_pool,
            together_api_keys,
            gemini_api_config,
            web_searcher,
            brain=brain,
        )
        pretrain_dm.setup(stage="fit")

        tgw_memory.take_snapshot(
            pretrain_model, pretrain_dm.train_dataloader())

    else:
        logger.warning(
            f"Model pre-train tidak ditemukan di {pretrain_model_path}. TGW tidak dapat diaktifkan tanpa pre-train."
        )
        initial_hparams["use_tgw"] = False

    if LAST_PROJECT_INFO_TXT.exists():
        try:
            _, last_ckpt_path = LAST_PROJECT_INFO_TXT.read_text().strip().split(",")
            if os.path.exists(last_ckpt_path):
                best_overall_checkpoint = last_ckpt_path
                logger.info(
                    f"✅ Menemukan checkpoint fine-tune juara dari sesi sebelumnya: {last_ckpt_path}"
                )
        except Exception as e:
            logger.warning(f"Gagal membaca info project terakhir: {e}")

    logger.info(
        "--- TAHAP 3: Menjalankan Putaran Awal (Baseline / Continual) ---")

    initial_hparams["causal_auditor"] = causal_auditor

    extra_callbacks = []
    if initial_hparams.get("use_tgw"):
        if tgw_memory.is_snapshot_taken:
            logger.info("Mengaktifkan TGW untuk putaran awal ini.")
            extra_callbacks.append(
                TGW_Callback(tgw_memory, initial_hparams.get("tgw_beta"))
            )
        else:
            logger.warning(
                "use_tgw=True tetapi snapshot tidak tersedia. TGW dilewati.")

    score, full_metrics, ckpt_path, data_summary, trainer = _one_train_run(
        initial_hparams,
        auditor,
        api_pool,
        together_api_keys,
        gemini_api_config,
        web_searcher=web_searcher,
        governor=governor,
        brain=brain,
        load_checkpoint_path=best_overall_checkpoint,
        nsmm=nsmm,
        custom_callbacks=[
            ChaosPreprocAccelerator(
                cache_dir=get_path(project_id, "cache"),
                use_zstd=True, zstd_level=3,
                imputer_threshold=1_000_000,
                default_window=100,
                verbose=True,
                tz="UTC",
            ),
            acr_callback,
            TPEGovernorCallback(
                util_target=0.55,
                adjust_every=200,
                ema_alpha=0.05,
                k_min=1,
                k_max=4,
                warmup_steps=400,
                verbose=True,
            ),
            PARC_Callback(
                warmup_frac=0.15, mid_frac=0.70,
                base_tau=1.00, min_tau=0.70, max_tau=1.30,
                base_C=1.00,   min_C=0.90, max_C=1.30,
                util_target=0.55, util_tolerance=0.05, ema_alpha=0.05,
                verbose=True,
            ),
        ],
    )

    tracker.log(project_id, 0, initial_hparams, full_metrics)
    history = [
        {
            "round": 0,
            "score": score,
            "full_metrics": full_metrics,
            "hparams": initial_hparams.copy(),
            "accepted": True,
            "checkpoint_path": ckpt_path,
            "data_summary": data_summary,
            "source": "Baseline/Continual",
        }
    ]

    best_score = score
    best_hparams = initial_hparams.copy()
    if ckpt_path:
        best_overall_checkpoint = ckpt_path

    logger.info(f"--- SKOR AWAL ({monitor_metric}): {best_score:.6f} ---")

    for r in range(1, MAX_ROUNDS_AI + 1):
        logger.info(
            f"\n{'='*25} MEMULAI SIKLUS EVOLUSI #{r}/{MAX_ROUNDS_AI} {'='*25}")

        logger.info(
            f"--- [SIKLUS {r}/PRE] Menjalankan Audit Kausal Periodik ---")
        causal_auditor.run_regime_clustering()
        causal_auditor.run_causal_audit()

        logger.info(f"--- [SIKLUS {r}/A] Fase Kreasi & Validasi Strategi ---")
        self_reflection_report = governor.generate_self_reflection_report()
        human_feedback_context = feedback_manager.get_all_feedback_as_context()
        context_bundle = {
            "self_reflection": self_reflection_report,
            "human_feedback": human_feedback_context,
            "data_summary": data_summary,
        }
        new_strategy = strategist.formulate_hypothesis(context_bundle)
        if new_strategy:
            logger.info(
                f"Strategi baru '{new_strategy.strategy_name}' akan dipertimbangkan untuk di-encode dan diuji di masa depan."
            )

        logger.info(f"--- [SIKLUS {r}/B] Fase Optimisasi Parameter ---")
        konteks_metrik = f"Konteks Teknis: Putaran optimisasi ke-{r}. Skor terbaik saat ini ({monitor_metric}) adalah {best_score:.4f}."
        konteks_strategi = strategy_kb.get_strategies_as_context(
            ["validated", "active"]
        )
        konteks_lengkap = f"{self_reflection_report}\n{human_feedback_context}\n{konteks_strategi}\n{konteks_metrik}"


        reward_model, featurizer = train_reward_model(project_id)


        logger.info(
            "🧠 [Self-Reliance] Mencoba brainstorming internal sebelum eskalasi ke LLM eksternal..."
        )
        internal_proposal = run_internal_brainstorming_cycle(
            best_hparams, reward_model, featurizer
        )

        if internal_proposal:
            logger.info(
                "✅ Brainstorming internal berhasil menemukan proposal yang menjanjikan."
            )
            proposal_juara = internal_proposal
        else:

            logger.warning(
                "Brainstorming internal tidak menemukan solusi. Eskalasi ke Dewan Musyawarah AI..."
            )
            proposal_juara = panel_musyawarah.gelar_musyawarah(konteks_lengkap)

        if not proposal_juara:
            logger.error(
                "Musyawarah dan brainstorming internal gagal, melanjutkan ke siklus berikutnya."
            )
            continue

        verdict, err = accreditor.validate_proposal(proposal_juara)
        if err or "[PROPOSAL_REJECTED]" in verdict:
            logger.warning(
                f"Proposal DITOLAK oleh Accreditor. Alasan: {verdict or err}"
            )
            history.append(
                {
                    "round": r,
                    "score": None,
                    "hparams": proposal_juara,
                    "accepted": False,
                    "reason": verdict or err,
                    "source": "AI Council (Rejected)",
                }
            )
            continue

        hparams_for_run = best_hparams.copy()
        hparams_for_run.update(proposal_juara)
        hparams_for_run["attempt"] = f"council_r{r}"

        hparams_for_run["causal_auditor"] = causal_auditor

        extra_callbacks = []
        if hparams_for_run.get("use_tgw"):
            if tgw_memory.is_snapshot_taken:
                logger.info("Mengaktifkan TGW untuk putaran fine-tune ini.")
                extra_callbacks.append(
                    TGW_Callback(tgw_memory, hparams_for_run.get("tgw_beta"))
                )
            else:
                logger.warning(
                    "use_tw=True tetapi snapshot tidak tersedia. TGW dilewati."
                )


        armh_callback = ARMH_Callback(
            api_pool=api_pool,
            together_keys=together_api_keys,
            critical_threshold=0.30,
        )
        extra_callbacks.append(armh_callback)


        acr_callback = ACR_Callback(
            nsmm, api_pool, together_api_keys, brain, _supremacy_embed_model
        )
        extra_callbacks.append(acr_callback)


        extra_callbacks.append(
            TPEGovernorCallback(
                util_target=0.55,
                adjust_every=200,
                ema_alpha=0.05,
                k_min=1,
                k_max=4,
                warmup_steps=400,
                verbose=True,
            )
        )

        extra_callbacks.append(
            PARC_Callback(
                warmup_frac=0.15, mid_frac=0.70,
                base_tau=1.00, min_tau=0.70, max_tau=1.30,
                base_C=1.00,   min_C=0.90, max_C=1.30,
                util_target=0.55, util_tolerance=0.05, ema_alpha=0.05,
                verbose=True,
            )
        )
        

        extra_callbacks.append(
            CRYSTAL_Callback(
                proj_grad_scale=2.5,
                boost_epochs=2,
                autosuffix=True,
                verbose=True,
            )
        )


        current_score, current_metrics, current_ckpt, current_summary, trainer = (
            _one_train_run(
                hparams_for_run,
                auditor,
                api_pool,
                together_api_keys,
                gemini_api_config,
                web_searcher=web_searcher,
                governor=governor,
                brain=brain,
                nsmm=nsmm,
                load_checkpoint_path=best_overall_checkpoint,
                custom_callbacks=extra_callbacks,
            )
        )

        tracker.log(project_id, r, hparams_for_run, current_metrics)
        history.append(
            {
                "round": r,
                "score": current_score,
                "full_metrics": current_metrics,
                "hparams": hparams_for_run,
                "accepted": True,
                "checkpoint_path": current_ckpt,
                "data_summary": current_summary,
                "source": f"AI Council #{r}",
            }
        )

        if (best_score - current_score) > improvement_threshold:
            logger.info(
                f"✅ PENINGKATAN DITEMUKAN! Skor baru: {current_score:.6f}")
            best_score = current_score
            best_hparams = hparams_for_run.copy()
            if current_ckpt:
                best_overall_checkpoint = current_ckpt

        logger.info(
            f"--- [SIKLUS {r}/C] Fase Pemeriksaan Stagnasi Arsitektural ---")
        is_stagnated, failure_reason = governor.check_for_architectural_stagnation()
        if is_stagnated:
            logger.warning(
                "!!! STAGNASI TERDETEKSI, MEMICU PROTOKOL METAMORFOSIS !!!")
            logger.error(
                "Fitur metamorfosis dipicu. Menghentikan siklus saat ini.")
            break

    logger.info("\n--- TAHAP 5: Finalisasi dan Penyimpanan Model Juara ---")
    accepted_runs = [
        run for run in history if run.get("accepted") and run.get("score") is not None
    ]
    if not accepted_runs:
        logger.error(
            "FATAL: Tidak ada satupun model yang berhasil dilatih. Proses berhenti."
        )
        return

    champion_run = min(accepted_runs, key=lambda x: x["score"])
    final_champion_ckpt = champion_run.get("checkpoint_path")

    logger.info(f"\n🏆🏆🏆 MODEL JUARA KESELURUHAN DITEMUKAN --------------------")
    logger.info(f"  Sumber Model Terbaik: {champion_run.get('source', 'N/A')}")
    logger.info(
        f"  Skor {monitor_metric} Terendah: {champion_run['score']:.6f}")
    logger.info("  Hyperparameters Juara:")
    logger.info(json.dumps(make_json_serializable(
        champion_run["hparams"]), indent=2))

    if final_champion_ckpt and os.path.exists(final_champion_ckpt):
        with open(LAST_PROJECT_INFO_TXT, "w") as f:
            f.write(
                f"{champion_run['hparams']['project_id']},{final_champion_ckpt}")
        logger.info(
            f"✅ Info model juara final disimpan ke: {LAST_PROJECT_INFO_TXT} untuk sesi berikutnya."
        )


def run_strategy_creation_cycle(
    governor: "CognitiveGovernor",
    feedback_manager: "HumanFeedbackManager",
    api_pool: "DistributedAIPool",
    project_id: str,
    df_raw: pd.DataFrame,
    initial_hparams: dict,
    **kwargs
):
    """
    Menjalankan satu siklus penuh kreasi, encoding, dan penyimpanan strategi baru.
    Ini adalah tugas otonom yang dapat dipanggil oleh Prime Directive.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 💡 MEMULAI SIKLUS KREASI STRATEGI OTONOM 💡 ===")
    logger.info("="*80)

    try:

        strategy_kb_path = MODELS_DIR / f"strategy_kb_{project_id}.sqlite"
        strategy_kb = StrategyKnowledgeBase(db_path=strategy_kb_path)
        strategist = GenerativeStrategist(api_pool, strategy_kb)
        encoder = StrategyEncoder(api_pool)


        self_reflection_report = governor.generate_self_reflection_report()
        human_feedback_context = feedback_manager.get_all_feedback_as_context()
        data_summary = {
            "num_rows": len(df_raw),
            "columns": df_raw.columns.tolist()[:10],
            "start_date": str(df_raw.index.min()),
            "end_date": str(df_raw.index.max()),
        }
        context_bundle = {
            "self_reflection": self_reflection_report,
            "human_feedback": human_feedback_context,
            "data_summary": data_summary,
        }


        new_strategy = strategist.formulate_hypothesis(context_bundle)

        if not new_strategy:
            logger.warning(
                "[Strategy Cycle] Tidak ada strategi baru yang berhasil dirumuskan. Siklus selesai.")
            return



        temp_dm = AlphaDataModule(
            initial_hparams, None, None, None, None, None, brain=None)
        df_featured, _ = generate_all_features(
            df_raw, initial_hparams['selected_tickers'], master_event_list=[])

        feature_code = encoder.generate_feature_code(
            new_strategy, df_featured.columns.tolist())

        if not feature_code:
            logger.error(
                f"[Strategy Cycle] Gagal meng-encode strategi '{new_strategy.strategy_name}'. Strategi tidak akan disimpan.")
            strategy_kb.update_strategy(
                new_strategy.strategy_name, status="encoding_failed")
            return


        strategy_dir = Path.home() / APP_BRAND / "strategies"
        strategy_dir.mkdir(exist_ok=True)


        safe_filename = re.sub(r'\W+', '_', new_strategy.strategy_name).lower()
        strategy_file_path = strategy_dir / f"{safe_filename}.py"


        file_content = f'''
# -*- coding: utf-8 -*-
# File ini dibuat secara otomatis oleh GenerativeStrategist & StrategyEncoder
# Strategi: {new_strategy.strategy_name}
# Hipotesis: {new_strategy.hypothesis}

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menerapkan fitur dari strategi '{new_strategy.strategy_name}'.
    Pastikan semua kolom yang dibutuhkan ada di DataFrame.
    """
    try:
        {feature_code}
        print(f"INFO: [Dynamic Strategy] Fitur '{new_strategy.strategy_name}' berhasil diterapkan.")
    except Exception as e:
        print(f"WARNING: [Dynamic Strategy] Gagal menerapkan fitur '{new_strategy.strategy_name}': {{e}}")
    return df
'''
        with open(strategy_file_path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(file_content))

        logger.info(
            f"✅ Kode untuk strategi '{new_strategy.strategy_name}' telah disimpan di: {strategy_file_path}")
        strategy_kb.update_strategy(
            new_strategy.strategy_name, status="encoded", feature_code=str(strategy_file_path))

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Kreasi Strategi: {e}", exc_info=True)


def run_strategy_backtest(
    project_id: str,
    initial_hparams: dict,
    strategy_kb: "StrategyKnowledgeBase",
    brain: "Brain",
    **kwargs
):
    """
    Menjalankan siklus backtesting otomatis untuk strategi yang baru dibuat.
    Ini adalah implementasi dari Sandbox Backtesting Otonom.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 🔬 MEMULAI SIKLUS VALIDASI STRATEGI (BACKTESTING) 🔬 ===")
    logger.info("="*80)

    try:

        strategies_to_test = strategy_kb.get_strategies_as_context(status_filter=[
                                                                   'encoded'])
        if "Belum ada strategi" in strategies_to_test:
            logger.info(
                "  -> Tidak ada strategi baru yang siap untuk di-backtest. Siklus selesai.")
            return


        strategy_name_match = re.search(
            r"Strategi: ([\w\s]+) \(Status: encoded", strategies_to_test)
        if not strategy_name_match:
            logger.warning(
                "  -> Tidak dapat mengurai nama strategi dari database. Melewatkan.")
            return

        strategy_name = strategy_name_match.group(1).strip()
        safe_filename = re.sub(r'\W+', '_', strategy_name).lower()
        strategy_file_path = Path.home() / APP_BRAND / "strategies" /            f"{safe_filename}.py"

        if not strategy_file_path.exists():
            logger.error(
                f"  -> GAGAL: File kode untuk strategi '{strategy_name}' tidak ditemukan di {strategy_file_path}.")
            strategy_kb.update_strategy(
                strategy_name, status="validation_failed_file_not_found")
            return

        logger.info(f"  -> Menguji strategi baru: '{strategy_name}'")


        df_raw = pd.read_parquet(initial_hparams['data_path'])


        logger.info(
            "  -> Menerapkan fitur dari strategi baru ke data historis...")
        sys.path.insert(0, str(strategy_file_path.parent.parent))
        strategy_module = importlib.import_module(
            f"strategies.{safe_filename}")


        df_with_base_features, _ = generate_all_features(
            df_raw, initial_hparams['selected_tickers'], [])

        df_with_new_feature = strategy_module.apply_strategy(
            df_with_base_features)


        logger.info("  -> Mengevaluasi dampak kausal fitur baru...")
        new_feature_col = f"feature_{safe_filename}"
        target_col = f"{initial_hparams['selected_tickers'][0]}_log_return"

        if new_feature_col not in df_with_new_feature.columns or target_col not in df_with_new_feature.columns:
            logger.error(
                f"  -> GAGAL: Fitur '{new_feature_col}' atau target '{target_col}' tidak ditemukan setelah penerapan strategi.")
            strategy_kb.update_strategy(
                strategy_name, status="validation_failed_feature_missing")
            return


        impact_correlation = df_with_new_feature[new_feature_col].corr(
            df_with_new_feature[target_col].shift(-1))

        logger.info(
            f"  -> Hasil Simulasi: Korelasi fitur baru dengan return masa depan = {impact_correlation:.4f}")



        if pd.notna(impact_correlation) and abs(impact_correlation) > 0.02:
            logger.info(
                f"  -> ✅ STRATEGI DIANGGAP VALID. Dampak positif terdeteksi.")
            strategy_kb.update_strategy(
                strategy_name, status="validated", performance_score=impact_correlation)
        else:
            logger.warning(
                f"  -> ❌ STRATEGI DIANGGAP GAGAL. Dampak tidak signifikan.")
            strategy_kb.update_strategy(
                strategy_name, status="failed", performance_score=impact_correlation)

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Backtesting Strategi: {e}", exc_info=True)
    finally:
        if str(strategy_file_path.parent.parent) in sys.path:
            sys.path.remove(str(strategy_file_path.parent.parent))


def run_exploratory_metamorphosis(
    project_id: str,
    initial_hparams: dict,
    governor: "CognitiveGovernor",
    api_pool: "DistributedAIPool",
    nsmm: "NSMM",
    df_raw: pd.DataFrame,
    **kwargs
):
    """
    Menjalankan siklus 'iseng' untuk mencoba mutasi arsitektur kecil
    secara proaktif dan menguji dampaknya di sandbox.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 🧬 MEMULAI SIKLUS METAMORFOSIS EKSPLORATIF 🧬 ===")
    logger.info("="*80)


    is_sandbox_success = False
    sandbox_loss = float('inf')
    temp_script_path = None
    temp_data_path = None

    try:

        logger.info(
            "  -> Tahap 1/3: Meminta ArchitectAI untuk 'mutasi' eksperimental...")
        architect = ArchitectAI(api_pool)
        model_file_path = Path.home() / APP_BRAND / "src/models/model_alpha/alpha.py"
        existing_code = model_file_path.read_text(encoding="utf-8")

        exploratory_prompt = (
            "Berdasarkan kode yang ada, usulkan SATU perubahan arsitektural kecil "
            "yang berpotensi meningkatkan efisiensi atau kemampuan menangkap pola, "
            "seperti menambahkan satu lapisan Konformer atau menyesuaikan mekanisme atensi. "
            "Ini adalah eksperimen, bukan perbaikan."
        )

        proposal = architect.design_new_architecture(
            exploratory_prompt, existing_code)

        if not proposal:
            logger.warning(
                "  -> ArchitectAI tidak menghasilkan proposal mutasi. Siklus selesai.")
            return




        logger.info(
            f"  -> Tahap 2/3: Menguji mutasi '{proposal.module_name}' di sandbox virtual...")


        class_end_marker = "class TSTLSTM(nn.Module):"
        new_code_with_module = existing_code.replace(
            class_end_marker,
            f"\n\n# === AUTO-GENERATED BY ARCHITECTAI (SANDBOX) ===\n"
            f"{proposal.module_code}\n\n"
            f"{class_end_marker}",
        )

        start_marker = "def forward(self, x_combined, y_historical, edge_index, x_spikes, **kwargs):"
        end_marker = "# TAHAP 5: Kepala Prediksi (Prediction Heads)"
        start_index = new_code_with_module.find(start_marker)
        end_index = new_code_with_module.find(end_marker, start_index)

        if start_index == -1 or end_index == -1:
            raise ValueError(
                "Tidak dapat menemukan marker untuk mengganti metode forward di kode sumber.")

        final_code = (
            new_code_with_module[:start_index]
            + f"{proposal.updated_forward_method}\n\n"
            + new_code_with_module[end_index:]
        )


        temp_script_path = model_file_path.with_name(
            "_sandbox_exploratory_alpha.py")
        temp_data_path = MODELS_DIR / "_sandbox_exploratory_data.parquet"

        temp_script_path.write_text(final_code, encoding="utf-8")

        df_raw.head(250).to_parquet(temp_data_path)


        command = [
            sys.executable, str(temp_script_path),
            "--mode", "fine-tune",

            "--tickers", initial_hparams['selected_tickers'][0],
            "--data_path", str(temp_data_path),
            "--is-sandbox-test",
        ]

        result = subprocess.run(
            command, capture_output=True, text=True, timeout=900
        )


        if result.returncode == 0:
            logger.info(
                "    -> ✅ [Sandbox] Simulasi berhasil diselesaikan tanpa crash.")
            for line in result.stdout.splitlines():
                if "val_loss" in line:
                    match = re.search(r"val_loss[=:]\s*([\d\.]+)", line)
                    if match:
                        sandbox_loss = float(match.group(1))

            if sandbox_loss > 5.0:
                raise RuntimeError(
                    f"Simulasi berhasil, namun loss sangat tinggi ({sandbox_loss}), mengindikasikan masalah logika.")

            is_sandbox_success = True
        else:
            logger.error(
                "    -> ❌ [Sandbox] Simulasi GAGAL! Kode yang diusulkan menyebabkan error.")

            logger.error(f"    -> Stderr: {result.stderr[-1000:]}")
            is_sandbox_success = False





        if not is_sandbox_success:
            logger.warning(
                "  -> Mutasi gagal dalam simulasi sandbox. Hipotesis dibuang.")
            return

        logger.info(
            f"  -> ✅ Mutasi lolos simulasi sandbox dengan loss: {sandbox_loss:.4f}")


        logger.info(
            "  -> Tahap 3/3: Menyimpan hipotesis arsitektur yang menjanjikan ke NSMM...")
        with sqlite3.connect(nsmm.db_path) as conn:
            conn.execute(
                """
                INSERT INTO architectural_hypotheses 
                (module_name, module_code, integration_plan, updated_forward_method, sandbox_performance, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.module_name,
                    proposal.module_code,
                    proposal.integration_plan,
                    proposal.updated_forward_method,
                    sandbox_loss,
                    'validated_in_sandbox'
                )
            )
            conn.commit()

        logger.info(
            "  -> ✅ Hipotesis arsitektur baru telah disimpan untuk pertimbangan di masa depan.")

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Metamorfosis Eksploratif: {e}", exc_info=True)
    finally:

        if temp_script_path and temp_script_path.exists():
            temp_script_path.unlink()
        if temp_data_path and temp_data_path.exists():
            temp_data_path.unlink()


def run_neuro_symbolic_synthesis_cycle(
    brain: "Brain",
    api_pool: "DistributedAIPool",
    nsmm: "NSMM",
    df_raw: pd.DataFrame,
    initial_hparams: dict,
    **kwargs
):
    """
    Menjalankan siklus penuh untuk menemukan aturan logis dari Brain,
    mengubahnya menjadi kode DifferentiableRule, dan menyimpannya untuk
    digunakan pada pelatihan berikutnya.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 🏛️ MEMULAI SIKLUS SINTESIS ATURAN NEURO-SIMBOLIK 🏛️ ===")
    logger.info("="*80)

    try:

        reasoner = DynamicLogicalReasoner(brain, api_pool)

        topic = f"strategi investasi untuk {random.choice(initial_hparams['selected_tickers'])}"


        discovered_rules_text = reasoner.discover_rules_from_brain(topic, k=5)
        if not discovered_rules_text:
            logger.warning(
                "[Neuro-Symbolic] Tidak ada aturan baru yang ditemukan dari Brain. Siklus selesai.")
            return

        logger.info(
            f"Ditemukan {len(discovered_rules_text)} kandidat aturan baru dari Brain.")



        df_featured, _ = generate_all_features(
            df_raw, initial_hparams['selected_tickers'], master_event_list=[])
        available_columns = df_featured.columns.tolist()


        rule_dir = Path.home() / APP_BRAND / "rules"
        rule_dir.mkdir(exist_ok=True)

        for i, rule_text in enumerate(discovered_rules_text):
            logger.info(f"  -> Memproses aturan #{i+1}: '{rule_text}'")


            safe_filename = re.sub(r'\W+', '_', rule_text).lower()[:50]
            rule_file_path = rule_dir /                f"rule_{int(time.time())}_{safe_filename}.py"


            if rule_file_path.exists():
                continue


            code_generation_prompt = f"""
            Anda adalah seorang Insinyur AI PyTorch yang ahli dalam logika neuro-simbolik.
            Tugas Anda adalah menerjemahkan satu aturan IF-THEN menjadi file Python lengkap yang berisi DifferentiableRule.

            Aturan Teks: "{rule_text}"

            Konteks:
            - Kelas DifferentiableRule sudah ada dan dapat di-subclass.
            - File harus berisi DUA hal:
              1. Sebuah fungsi logika (misal: `custom_rule_logic`) yang menerima `x_raw_features`, `feature_indices`, dan `thresholds`.
              2. Sebuah fungsi `create_rule(feature_map)` yang mengembalikan instance dari DifferentiableRule menggunakan fungsi logika tersebut.
            - `feature_map` adalah kamus Python `{{"nama_fitur": index}}`. Kode Anda harus menggunakan ini untuk mendapatkan indeks.
            - Gunakan `torch.sigmoid((feature - threshold) * steepness)` untuk perbandingan yang 'differentiable'.
            - Aturan AND dapat diimplementasikan dengan perkalian.
            - Pastikan untuk mengimpor semua pustaka yang diperlukan (torch, torch.nn).

            Daftar Fitur yang Tersedia (sebagian): {available_columns[:100]}

            Kembalikan HANYA kode Python mentah untuk seluruh file.
            """

            try:

                generated_code = api_pool.call_gemini_for_text(
                    code_generation_prompt, "ai_engineer")


                if "DifferentiableRule" in generated_code and "create_rule" in generated_code:
                    with open(rule_file_path, "w", encoding="utf-8") as f:
                        f.write(generated_code)
                    logger.info(
                        f"    ✅ Kode aturan baru disimpan di: {rule_file_path.name}")
                else:
                    logger.error(
                        f"    ❌ Kode yang dihasilkan untuk aturan '{rule_text}' tidak valid.")

            except Exception as e:
                logger.error(
                    f"    ❌ Gagal menghasilkan kode untuk aturan: {e}")

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Sintesis Neuro-Simbolik: {e}", exc_info=True)


def run_goal_reflection_cycle(
    nsmm: NSMM,
    brain: Brain,
    api_pool: "DistributedAIPool",
    feedback_manager: "HumanFeedbackManager",
    governor: "CognitiveGovernor",
):
    """
    Menjalankan siklus debat internal untuk merefleksikan dan berpotensi
    mengamandemen tujuan utama yang aktif.
    """
    logger.info("🏛️ [Goal Reflection] Memulai siklus refleksi tujuan...")
    active_goal = nsmm.get_active_goal()
    if not active_goal:
        logger.error("Tidak ada tujuan aktif untuk direfleksikan.")
        return


    validated_neurons = nsmm.get_neurons_for_consolidation(
        status="validated", limit=20)
    evidence_from_neurons = "\n".join(
        [f"- Outcome {n['outcome']}: {n['trigger_reason']}" for n in validated_neurons]
    )
    evidence_from_rlhf = feedback_manager.get_all_feedback_as_context(limit=10)
    evidence_from_governor = governor.generate_self_reflection_report()


    red_team_prompt = f"""
    Anda adalah 'AI Red Team', tugas Anda adalah menjadi skeptis dan menantang tujuan yang ada.
    Tujuan Aktif Saat Ini: "{active_goal['description']}"

    Bukti dari Pengalaman Sistem:
    ---
    Umpan Balik Supervisor Manusia: {evidence_from_rlhf}
    ---
    Refleksi Internal Sistem: {evidence_from_governor}
    ---
    Neuron Tervalidasi (Pengalaman Sukses/Gagal): {evidence_from_neurons}
    ---

    Tugas: Apakah tujuan aktif saat ini masih optimal? Apakah ada bukti kuat bahwa mengejar tujuan ini menyebabkan hasil yang tidak diinginkan (misalnya, prediksi volatil yang ditolak pengguna)? Formulasikan argumen penolakan yang kuat jika ada. Jika tidak, jawab "Tujuan masih relevan."
    """
    challenge = api_pool.call_gemini_for_text(
        red_team_prompt, "experimentalist")

    if (
        "tidak relevan" not in challenge.lower()
        and "tidak optimal" not in challenge.lower()
    ):
        logger.info(
            "🏛️ [Goal Reflection] Tujuan saat ini lolos tantangan Red Team.")
        return


    logger.warning(
        f"🏛️ Tantangan ditemukan untuk tujuan aktif! Argumen Red Team: {challenge}"
    )
    arbiter_prompt = f"""
    Anda adalah 'Arbiter AI'. Tengahi debat berikut dan buat keputusan final.
    Tujuan yang Diuji: "{active_goal['description']}"
    Argumen Penolakan: "{challenge}"
    Bukti Lengkap: (Sama seperti di atas)

    Tugas: Panggil tool `ProposeGoalModification` untuk membuat keputusan akhir Anda.
    """
    try:
        arbiter_decision = api_pool.call_gemini_with_tool(
            arbiter_prompt, "supervisor", ProposeGoalModification
        )
        decision = ProposeGoalModification(**arbiter_decision)


        if decision.decision == "AMEND":
            logger.warning(
                f"KEPUTUSAN: Tujuan diamandemen! Deskripsi baru: '{decision.amended_description}'"
            )
            nsmm.update_goal(
                active_goal["goal_id"],
                new_status="active",
                new_description=decision.amended_description,
            )
        elif decision.decision == "DEPRECATE":
            logger.critical(
                f"KEPUTUSAN: Tujuan '{active_goal['description']}' tidak digunakan lagi!"
            )
            nsmm.update_goal(active_goal["goal_id"], new_status="deprecated")
        else:
            logger.info("KEPUTUSAN: Tujuan saat ini dipertahankan.")

    except Exception as e:
        logger.error(f"Gagal dalam arbitrasi tujuan: {e}")
        return


    try:
        if decision.decision == "AMEND" and decision.amended_description:
            logger.warning(
                f"🏛️ [Goal Execution] KEPUTUSAN: Tujuan diamandemen! Deskripsi baru: '{decision.amended_description}'"
            )
            nsmm.update_goal(
                active_goal["goal_id"],
                new_status="active",
                new_description=decision.amended_description,
            )


        elif decision.decision == "DEPRECATE":
            logger.critical(
                f"🏛️ [Goal Execution] KEPUTUSAN: Tujuan '{active_goal['description']}' tidak digunakan lagi!"
            )
            nsmm.update_goal(active_goal["goal_id"], new_status="deprecated")
        else:
            logger.info(
                "🏛️ [Goal Execution] KEPUTUSAN: Tujuan saat ini dipertahankan.")
    except Exception as e:
        logger.error(f"Gagal mengeksekusi keputusan Arbiter: {e}")


def execute_architectural_metamorphosis(
    governor: "CognitiveGovernor",
    api_pool: "DistributedAIPool",
    project_id: str,
    hparams: dict,
    df_raw: pd.DataFrame,
):
    """
    Menjalankan protokol perubahan arsitektur otonom:
    1. Meminta Izin.
    2. Menjalankan simulasi penuh di sandbox.
    3. Menerapkan perubahan jika simulasi berhasil.
    """
    is_stagnated, failure_reason = governor.check_for_architectural_stagnation(
        stagnation_threshold=2
    )
    if not is_stagnated:
        return

    logger.critical(
        "🚨 [Metamorphosis] STAGNASI ARSITEKTURAL TERDETEKSI! Memulai protokol perubahan..."
    )

    try:
        model_file_path = (
            Path.home() / APP_BRAND / "src/models/model_alpha/alpha.py"
        )
        with open(model_file_path, "r", encoding="utf-8") as f:
            existing_code = f.read()

        architect = ArchitectAI(api_pool)
        proposal = architect.design_new_architecture(
            failure_reason, existing_code)

        if not proposal:
            logger.error(
                "[Metamorphosis] Gagal mendapatkan proposal arsitektur. Perubahan dibatalkan."
            )
            return


        logger.info(
            "💡 [Metamorphosis] Proposal arsitektur baru telah dirancang.")
        permission_prompt = (
            f"Sistem telah merancang perubahan arsitektur ('{proposal.module_name}') untuk mengatasi stagnasi. "
            f"Perubahan akan dilakukan pada file '{model_file_path.name}'. Izinkan untuk melanjutkan ke tahap simulasi virtual?"
        )
        izin = questionary.confirm(permission_prompt, default=False).ask()

        if not izin:
            logger.warning(
                "[Metamorphosis] Izin ditolak oleh supervisor. Perubahan dibatalkan."
            )
            return


        logger.info(
            "⚙️ [Sandbox] Memulai simulasi virtual untuk kode yang diusulkan...")

        class_end_marker = "class TSTLSTM(nn.Module):"
        new_code_with_module = existing_code.replace(
            class_end_marker,
            f"\n\n# === AUTO-GENERATED BY ARCHITECTAI: {datetime.now().isoformat()} ===\n"
            f"{proposal.module_code}\n\n"
            f"{class_end_marker}",
        )
        start_marker = "def forward(self, x_combined, y_historical, edge_index, x_spikes, **kwargs):"
        end_marker = "# TAHAP 8: Head Prediksi (Guru)"
        start_index = new_code_with_module.find(start_marker)
        end_index = new_code_with_module.find(end_marker, start_index)
        if start_index == -1 or end_index == -1:
            raise ValueError(
                "Tidak dapat menemukan marker untuk mengganti metode forward."
            )
        final_code = (
            new_code_with_module[:start_index]
            + f"{proposal.updated_forward_method}\n\n"
            + new_code_with_module[end_index:]
        )

        temp_script_path = model_file_path.with_name("_sandbox_alpha.py")
        temp_data_path = MODELS_DIR / "_sandbox_data.parquet"

        try:

            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(final_code)
            df_raw.head(200).to_parquet(temp_data_path)


            command = [
                sys.executable,
                str(temp_script_path),
                "--mode",
                "fine-tune",
                "--tickers",

                hparams["selected_tickers"][0],
                "--data_path",
                str(temp_data_path),
                "--is-sandbox-test",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=600
            )


            if result.returncode == 0:
                logger.info(
                    "✅ [Sandbox] Simulasi berhasil diselesaikan tanpa crash.")

                final_val_loss = float("inf")
                for line in result.stdout.splitlines():
                    if "val_loss" in line:
                        match = re.search(r"val_loss[=:]\s*([\d\.]+)", line)
                        if match:
                            final_val_loss = float(match.group(1))

                logger.info(
                    f"  > Metrik Kinerja Sandbox: val_loss = {final_val_loss:.4f}"
                )
                if (
                    final_val_loss > 5.0
                ):
                    raise RuntimeError(
                        f"Simulasi berhasil, namun loss sangat tinggi ({final_val_loss}), mengindikasikan masalah logika."
                    )

            else:
                logger.error(
                    "❌ [Sandbox] Simulasi GAGAL! Kode yang diusulkan menyebabkan error."
                )
                logger.error(f"    Stderr: {result.stderr}")
                return

        finally:
            if temp_script_path.exists():
                temp_script_path.unlink()
            if temp_data_path.exists():
                temp_data_path.unlink()


        logger.warning(
            f"✅ SEMUA VALIDASI LOLOS. Menerapkan perubahan arsitektur ke {model_file_path.name}."
        )
        with open(model_file_path, "w", encoding="utf-8") as f:
            f.write(final_code)

        logger.critical(
            "✅ [Metamorphosis] Perubahan berhasil. RESTART MANUAL DIPERLUKAN untuk mengaktifkan arsitektur baru."
        )
        sys.exit(0)

    except Exception as e:
        logger.error(
            f"FATAL: Gagal menjalankan Metamorfosis Kognitif: {e}", exc_info=True
        )




def data_acquisition_worker(
    data_queue: queue.Queue, brain: "Brain", web_searcher: "WebSearchManager"
):
    """
    Worker latar belakang yang mendengarkan permintaan data dan memenuhinya
    secara otonom tanpa menggunakan LLM.
    """
    logger.info(
        " Gelas[Data Acquisition Worker] Aktif, menunggu permintaan data...")
    while True:
        try:
            request = data_queue.get()
            topic = request.get("topic")
            if topic:
                logger.info(
                    f" Gelas[Data Acquisition] Menerima permintaan untuk topik: '{topic}'"
                )


                search_results = web_searcher.search(
                    f"Riset mendalam tentang {topic}", max_results=5
                )

                if search_results:
                    source_name = f"DataRequest_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
                    chunks = chunk_text(
                        search_results, chunk_size=300, chunk_overlap=40
                    )
                    brain.add_chunks(
                        chunks, source_name=f"[AKUISISI_DATA] {source_name}"
                    )
                    logger.info(
                        f" Gelas[Data Acquisition] ✅ Pengetahuan baru tentang '{topic}' telah ditemukan dan disimpan ke Brain."
                    )

                    cooldown_period_sec = 21600
                    current_time = time.time()

                    if (current_time - LAST_PROACTIVE_MESSAGE_TIME[0]) > cooldown_period_sec:
                        PROACTIVE_MESSAGE_QUEUE.put(
                            f"Saya telah selesai melakukan riset tentang '{topic}' dan menambahkan pengetahuan baru ke memori saya."
                        )

                        LAST_PROACTIVE_MESSAGE_TIME[0] = current_time
                    else:
                        logger.info(
                            f"[Proactive Chat] Notifikasi untuk riset '{topic}' ditahan untuk menghemat request.")

                else:
                    logger.warning(
                        f" Gelas[Data Acquisition] ⚠️ Pencarian untuk '{topic}' tidak menemukan hasil yang signifikan."
                    )

        except Exception as e:
            logger.error(
                f"Error pada Data Acquisition Worker: {e}", exc_info=True)
            time.sleep(60)



class OG_BenchmarkSuite:
    """
    Menjalankan evaluasi dinamis menggunakan panel juri AI yang terspesialisasi.
    Setiap juri membuat soal sesuai keahliannya, dan juri lain menilainya.
    Versi ini telah di-upgrade untuk menyertakan Grok-3 Mini sebagai juri.
    """

    def __init__(
        self,
        auditor: CriticalAuditor,
        api_pool: DistributedAIPool,
        dm: AlphaDataModule,
        model: pl.LightningModule,
        together_keys: dict,
    ):
        self.auditor = auditor
        self.api_pool = api_pool
        self.together_keys = together_keys
        self.dm = dm
        self.model = model



        self.judges = {
            "Kuantitatif": {
                "model": "lgai/exaone-deep-32b",
                "key_name": "exaone",
                "difficulty": "Menengah",
                "expertise": "analisis kuantitatif, perhitungan matematis murni, dan ekstraksi data spesifik.",
            },
            "Penalaran Umum": {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                "key_name": "vision",
                "difficulty": "Sulit",
                "expertise": "pemahaman umum, penalaran berbasis konteks, dan mengaitkan data dengan peristiwa dunia nyata.",
            },
            "Penilai Cepat & Logis (Grok)": {
                "model": "grok-3-mini-fast",
                "key_name": "grok_key",
                "difficulty": "Menengah",
                "expertise": "penilaian logis yang cepat, konsistensi jawaban, dan akurasi faktual dasar.",
            },
        }


        self.available_judges = self._get_available_judges()
        logger.info(
            f"☑️ OG-Benchmark Suite v2.1 (Panel Dinamis + Grok) diinisialisasi dengan {len(self.available_judges)} juri."
        )

    def _get_available_judges(self) -> dict:
        """Memfilter juri yang API key-nya tersedia."""
        active_judges = {}
        for role, details in self.judges.items():

            if self.together_keys.get(details["key_name"]):
                active_judges[role] = details

        if len(active_judges) < 2:
            logger.warning(
                f"Hanya {len(active_judges)} juri yang tersedia. Benchmark mungkin tidak representatif."
            )
        return active_judges

    def _generate_question(self, role: str) -> str:
        """Meminta satu juri untuk membuat soal sesuai keahliannya."""
        judge_details = self.available_judges[role]
        judge_model_name = judge_details["model"]
        judge_key = self.together_keys[judge_details["key_name"]]


        if "grok" in judge_model_name:
            generator_agent = GrokLLM(
                api_key=judge_key, model_name=judge_model_name)
        else:
            generator_agent = TogetherLLM(
                api_key=judge_key, model_name=judge_model_name
            )


        data_preview = self.dm.df_processed.head().to_csv()

        prompt = f"""
        Anda adalah seorang ahli dalam bidang: {judge_details['expertise']}.
        Tugas Anda adalah membuat SATU pertanyaan benchmark yang menantang dengan tingkat kesulitan '{judge_details['difficulty']}'.
        Pertanyaan harus relevan dengan analisis data finansial berdasarkan sampel data berikut. Jangan meminta perhitungan yang tidak mungkin dilakukan dari sampel ini.
        Fokus pada pengujian kemampuan analisis, sintesis, atau penalaran.

        Sampel Header & Data:
        ```csv
        {data_preview}
        ```

        Kembalikan HANYA teks pertanyaan itu sendiri, tanpa embel-embel atau format tambahan.
        """

        logger.info(
            f"  Meminta Juri '{role}' ({judge_model_name}) untuk membuat soal..."
        )
        question = generator_agent.chat(prompt)
        logger.info(f"  Soal yang Dihasilkan: {question}")
        return question

    def _get_system_answer(self, question_text: str) -> str:
        """Mendapatkan jawaban dari sistem utama yang sedang diuji (Gemini)."""
        analyst_agent = self.api_pool.get_worker("advanced_advisor")
        if not analyst_agent:
            return (
                "Error: Agen 'advanced_advisor' (Gemini) tidak tersedia untuk menjawab."
            )

        prompt = f"Anda adalah AI Analis Finansial yang canggih. Jawab pertanyaan benchmark berikut dengan analisis yang tajam dan jelas.\n\nPertanyaan: {question_text}"
        return analyst_agent.invoke(prompt).content

    def _evaluate_answer_by_panel(
        self, question_text: str, system_answer: str, question_generator_role: str
    ) -> dict:
        """Menggunakan panel juri (yang tidak membuat soal) untuk menilai jawaban."""
        panel = [
            role for role in self.available_judges if role != question_generator_role
        ]
        if not panel:
            return {"score": 0, "reasoning": "Tidak ada juri penilai yang tersedia."}

        scores = []
        reasonings = []

        judge_prompt_template = """
        Anda adalah juri benchmark yang objektif dan ketat. Berikan skor 1-5 pada jawaban yang diberikan berdasarkan pertanyaan.
        Kriteria: Akurasi, Kedalaman Penalaran, dan Kejelasan.
        Pertanyaan: "{question}"
        Jawaban Sistem untuk Dinilai: "{answer}"
        Berikan penilaian HANYA dalam format JSON: {{"score": <angka 1-5>, "reasoning": "Alasan singkat."}}
        """

        for judge_role in panel:
            judge_details = self.available_judges[judge_role]
            judge_model_name = judge_details["model"]
            judge_key = self.together_keys[judge_details["key_name"]]


            if "grok" in judge_model_name:
                judge_agent = GrokLLM(
                    api_key=judge_key, model_name=judge_model_name)
            else:
                judge_agent = TogetherLLM(
                    api_key=judge_key, model_name=judge_model_name
                )


            prompt = judge_prompt_template.format(
                question=question_text, answer=system_answer
            )

            try:
                logger.info(
                    f"  > Juri '{judge_role}' ({judge_model_name}) sedang menilai..."
                )
                response_str = judge_agent.chat(prompt)
                evaluation = robust_json_extract(
                    response_str, model=None) or {}

                score = int(evaluation.get("score", 1))
                reason = evaluation.get(
                    "reasoning", "Format respons juri tidak valid.")
                scores.append(score)
                reasonings.append(f"Juri '{judge_role}': {score}/5 ({reason})")
            except Exception as e:
                scores.append(0)
                reasonings.append(f"Juri '{judge_role}': Error - {e}")

        avg_score = sum(scores) / len(scores) if scores else 0
        combined_reasoning = " | ".join(reasonings)
        return {"score": round(avg_score, 2), "reasoning": combined_reasoning}

    def run(self):
        """Menjalankan siklus benchmark dinamis."""
        logger.info(
            "\n"
            + "=" * 80
            + "\n"
            + " MEMULAI EKSEKUSI OG-BENCHMARK SUITE v2.1 ".center(80, "=")
            + "\n"
            + "=" * 80
        )

        if not self.available_judges:
            logger.error(
                "Tidak ada juri yang dikonfigurasi dengan API Key. Proses benchmark dibatalkan."
            )
            return

        all_results = []
        for role, details in self.available_judges.items():

            question = self._generate_question(role)


            logger.info(
                f"  Sistem utama (Gemini) sedang menjawab soal dari '{role}'..."
            )
            system_answer = self._get_system_answer(question)
            logger.info(
                f"  Jawaban Sistem (ringkasan): {system_answer[:200]}...")


            logger.info(f"  Panel juri sedang menilai jawaban...")
            evaluation = self._evaluate_answer_by_panel(
                question, system_answer, role)

            all_results.append(
                {
                    "Pembuat Soal": role,
                    "Tingkat": details["difficulty"],
                    "Soal": question,
                    "Skor Akhir": evaluation.get("score", 0),
                    "Catatan Juri": evaluation.get("reasoning", ""),
                }
            )


        report_df = pd.DataFrame(all_results)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 80)

        logger.info(
            "\n"
            + "=" * 80
            + "\n"
            + " LAPORAN AKHIR OG-BENCHMARK v2.1 ".center(80, "=")
            + "\n"
            + "=" * 80
        )
        logger.info(f"\n{report_df.to_string()}")

        avg_score = report_df["Skor Akhir"].mean()
        logger.info("\n--- RINGKASAN SKOR ---")
        logger.info(f"Skor Rata-Rata Keseluruhan: {avg_score:.2f} / 5.0")
        logger.info("=" * 80)


class DynamicKnowledgeGraph:
    """
    Mengelola representasi graf multi-lapis dari entitas dan hubungan dinamis antar mereka.
    Versi 2.1: Dengan node yang mendukung atribut dinamis.
    """

    def __init__(self, project_id: str, tickers: list = None):
        self.project_id = project_id
        self.nodes = {}
        self.edges = []
        self._node_id_counter = 0

        if tickers:
            for ticker in tickers:
                self.add_node(
                    node_id=ticker, node_type="Ticker", layer="default", name=ticker
                )

    def _get_next_id(self):
        self._node_id_counter += 1
        return self._node_id_counter




    def add_node(self, node_id: str, node_type: str, layer: str = "default", **kwargs):
        """Menambahkan node ke lapisan tertentu jika belum ada, dengan atribut dinamis."""
        if node_id not in self.nodes:

            self.nodes[node_id] = {
                "type": node_type,
                "layer": layer,
                "attributes": kwargs,
            }
            logger.debug(
                f"[DKG] Node added: {node_id} (Layer: {layer}, Type: {node_type})"
            )



    def add_edge(self, source: str, target: str, relationship: str, **kwargs):
        """Menambahkan edge di dalam lapisan yang sama atau antar-lapisan."""
        if source in self.nodes and target in self.nodes:
            edge_data = {
                "relationship": relationship,
                "timestamp": datetime.now().isoformat(),
            }
            edge_data.update(kwargs)
            self.edges.append((source, target, edge_data))
            logger.debug(
                f"[DKG] Edge added: {source} -[{relationship}]-> {target}")

    def add_digested_activity(self, activity: "DigestedActivity"):
        """
        Menambahkan pengetahuan yang telah dicerna dari aktivitas pengguna ke DKG.
        """
        user_node_id = "USER"
        self.add_node(user_node_id, node_type="User", layer="Agent")

        content_id = activity.specific_content.replace(" ", "_").upper()
        self.add_node(
            node_id=content_id,
            node_type="Content_Item",
            layer="Knowledge",
            name=activity.specific_content,
        )

        topic_id = activity.general_topic.replace(" ", "_")
        self.add_node(
            topic_id, node_type="Topic", layer="Knowledge", name=activity.general_topic
        )

        platform_id = activity.website_or_app.replace(" ", "_")
        self.add_node(
            platform_id,
            node_type="Platform",
            layer="Knowledge",
            name=activity.website_or_app,
        )

        self.add_edge(user_node_id, content_id, "consumed")
        self.add_edge(content_id, topic_id, "is_a_topic_of")
        self.add_edge(content_id, platform_id, "is_on_platform")

        for entity in activity.entities:
            if entity.lower() in platform_id.lower():
                continue
            entity_id = entity.replace(" ", "_")
            self.add_node(
                node_id=entity_id, node_type="Entity", layer="Knowledge", name=entity
            )
            self.add_edge(entity_id, content_id, "is_mentioned_in")

        logger.info(
            f"🧠 [DKG Neurogenesis] Pengetahuan baru ditambahkan: '{activity.specific_content}' (Topik: {activity.general_topic})"
        )

    def to_gml(self) -> str:
        """Mengonversi graf internal ke format GML yang dapat dibaca oleh DoWhy."""
        gml = "graph [\n  directed 1\n"
        for node_id, data in self.nodes.items():
            name = (
                data["attributes"].get("name", node_id)
                if "attributes" in data
                else node_id
            )
            gml += f'  node [ id "{node_id}" label "{name}" type "{data.get("type", "Unknown")}" ]\n'
        for source, target, data in self.edges:
            gml += f'  edge [ source "{source}" target "{target}" label "{data.get("relationship", "")}" ]\n'
        gml += "]"
        return gml

    def visualize(self):
        """Membuat dan menyimpan visualisasi graf menggunakan Graphviz."""
        dot = graphviz.Digraph(
            f"DKG_{self.project_id}", comment="Dynamic Knowledge Graph"
        )
        dot.attr("graph", rankdir="LR", splines="true",
                 overlap="false", size="15,15")
        dot.attr("node", shape="box", style="rounded,filled")

        node_colors = {
            "Ticker": "#4e79a7",
            "Feature": "#f28e2b",
            "Event": "#e15759",
            "Market": "#76b7b2",
            "Concept": "#59a14f",
            "BaseEmotion": "#f1c40f",
            "ComplexEmotion": "#9b59b6",
            "User": "#3498db",
        }

        for node_id, data in self.nodes.items():
            node_type = data.get("type", "Unknown")
            color = node_colors.get(node_type, "#bab0ab")
            dot.node(
                node_id, label=f"{node_id}\n({node_type})", color=color, fillcolor=color
            )

        for source, target, data in self.edges:
            dot.edge(
                source,
                target,
                label=data.get("relationship", ""),
                fontsize="8",
                color="#bab0ab",
            )

        try:
            checkpoint_dir = get_path(self.project_id, "checkpoint_dir")
            output_path = checkpoint_dir /                f"dkg_visualization_{self.project_id}"
            dot.render(str(output_path), format="png",
                       view=False, cleanup=True)
            logger.info(f"✅ Visualisasi DKG disimpan di: {output_path}.png")
        except Exception as e:
            logger.error(f"Gagal membuat visualisasi DKG: {e}")

    def to_pyg_data(self) -> tuple:
        """Mengonversi DKG ke format PyTorch Geometric Data dan menyimpan metadata."""
        if not self.nodes:
            return None, None

        node_list = sorted(list(self.nodes.keys()))
        node_map = {name: i for i, name in enumerate(node_list)}


        node_types = sorted(list(set(data["type"]
                            for data in self.nodes.values())))
        layer_types = sorted(
            list(set(data["layer"] for data in self.nodes.values())))
        type_map = {name: i for i, name in enumerate(node_types)}
        layer_map = {name: i for i, name in enumerate(layer_types)}

        num_features = len(type_map) + len(layer_map)
        x = torch.zeros((len(node_list), num_features))

        for i, node_id in enumerate(node_list):
            node_data = self.nodes[node_id]
            x[i, type_map[node_data["type"]]] = 1
            x[i, len(type_map) + layer_map[node_data["layer"]]] = 1

        source_nodes = [
            node_map[s] for s, t, d in self.edges if s in node_map and t in node_map
        ]
        target_nodes = [
            node_map[t] for s, t, d in self.edges if s in node_map and t in node_map
        ]
        edge_index = torch.tensor(
            [source_nodes, target_nodes], dtype=torch.long)

        data = PyG_Data(x=x, edge_index=edge_index)
        metadata = {"node_map": node_map, "node_list": node_list}
        return data, metadata


class MultiLayerGNN(torch.nn.Module):
    """GNN sederhana untuk belajar representasi dari DKG Multi-Lapis."""

    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super().__init__()
        self.conv1 = PyG_GCNConv(in_channels, hidden_channels)
        self.conv2 = PyG_GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def train_ml_dkg(
    dkg: DynamicKnowledgeGraph, project_id: str, epochs: int = 50
) -> Optional[Path]:
    """Melatih GNN pada ML-DKG untuk menghasilkan embedding node yang kaya konteks."""
    logger.info(
        "--- 🧠 Memulai Pelatihan pada Multi-Layer Dynamic Knowledge Graph (ML-DKG) ---"
    )

    pyg_data, metadata = dkg.to_pyg_data()
    if pyg_data is None:
        logger.warning("[ML-DKG] DKG kosong, pelatihan GNN dilewati.")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pyg_data = pyg_data.to(device)

    model = MultiLayerGNN(
        in_channels=pyg_data.num_node_features,
        hidden_channels=128,
        out_channels=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(pyg_data.x, pyg_data.edge_index)


        pos_edge_index = pyg_data.edge_index
        neg_edge_index = torch.randint(
            0,
            pyg_data.num_nodes,
            pos_edge_index.size(),
            dtype=torch.long,
            device=device,
        )

        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        loss = (
            -torch.log(torch.sigmoid(pos_out)).mean()
            - torch.log(1 - torch.sigmoid(neg_out)).mean()
        )

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.info(
                f"ML-DKG GNN Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        final_embeddings = model(pyg_data.x, pyg_data.edge_index).cpu()

    embedding_dict = {
        node_name: final_embeddings[i] for node_name, i in metadata["node_map"].items()
    }

    save_path = (
        get_path(project_id, "checkpoint_dir") /
        f"ml_dkg_embeddings_{project_id}.pt"
    )
    torch.save(embedding_dict, save_path)
    logger.info(f"✅ ML-DKG embeddings disimpan di: {save_path}")

    return embedding_dict


class CausalInferenceEngine:
    """
    Mengelola pembuatan model kausal, intervensi, dan analisis kontrafaktual.
    Menggunakan Dynamic Knowledge Graph (DKG) sebagai landasan dan LLM untuk penemuan & interpretasi.
    """

    def __init__(
        self,
        api_pool: DistributedAIPool,
        df_processed: pd.DataFrame,
        feature_names: list,
        target_cols: list,
        tickers: list,
        model: pl.LightningModule,
        project_id: str,
    ):
        self.api_pool = api_pool
        self.df = df_processed.copy()
        self.feature_names = feature_names
        self.target_cols = target_cols
        self.tickers = tickers
        self.trained_model = model
        self.project_id = project_id


        self.dkg = DynamicKnowledgeGraph(
            project_id=self.project_id, tickers=self.tickers
        )

        self.causal_model = None
        self.causal_graph = None
        logger.info(
            f"🧠 Causal Inference Engine & DKG diinisialisasi dengan {len(feature_names)} fitur untuk project {self.project_id}."
        )

    def _populate_dkg(self):
        """Mengisi DKG multi-lapis dengan node dan edge dari data yang telah diproses."""
        logger.info(
            "[ML-DKG] Memulai pengisian node dan edge ke dalam lapisan-lapisan..."
        )


        for ticker in self.tickers:
            self.dkg.add_node(
                node_id=ticker, node_type="Ticker", layer="Market")


        top_features = self.feature_names[:30]
        for feature in top_features:
            self.dkg.add_node(
                node_id=feature, node_type="Feature", layer="Causal/Feature"
            )


        astro_features = [
            f for f in self.feature_names if f.startswith("astro_")]
        for feature in astro_features:
            self.dkg.add_node(
                node_id=feature, node_type="AstroSignal", layer="Astro")


        corr_matrix = self.df[top_features + self.target_cols].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                if col1 in self.dkg.nodes and col2 in self.dkg.nodes:
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        self.dkg.add_edge(
                            col1,
                            col2,
                            "correlated_with",
                            weight=round(corr_matrix.iloc[i, j], 3),
                        )



        for feature in top_features:
            for ticker in self.tickers:
                if ticker in feature:
                    self.dkg.add_edge(feature, ticker, "describes")


        for feature in astro_features:
            for ticker in self.tickers:

                self.dkg.add_edge(feature, ticker, "potentially_influences")


        corr_matrix = self.df[top_features + self.target_cols].corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    self.dkg.add_edge(
                        col1, col2, "correlated_with", weight=round(corr_val, 3)
                    )

    def _discover_causal_graph_with_llm(self) -> str:
        """
        Menggunakan LLM untuk menyusun hipotesis grafik kausal dari DKG yang ada.
        """
        dkg_gml_structure = self.dkg.to_gml()

        prompt = f"""
        Anda adalah seorang Ekonom ahli yang berspesialisasi dalam ekonometrika time-series.
        Tugas Anda adalah menyempurnakan struktur grafik berikut menjadi grafik kausal (causal graph) yang logis.
        Gunakan pengetahuan ekonomi Anda untuk menambahkan, menghapus, atau membalik arah panah (edge) agar masuk akal secara kausal.

        STRUKTUR GRAFIK DASAR (DARI DKG):
        ---
        {dkg_gml_structure}
        ---

        ATURAN KETAT:
        1.  Fokus pada hubungan sebab-akibat. Misalnya, 'event_influence_score' adalah penyebab, bukan akibat. 'market_return' bisa menjadi perantara. Kolom target (seperti '{self.target_cols[0]}') adalah efek akhir.
        2.  Output Anda HARUS dalam format GML (Graph Modeling Language) yang ketat dan lengkap.
        3.  JANGAN menyertakan teks atau penjelasan lain, HANYA kode GML yang sudah direvisi.
        """
        logger.info(
            "Meminta 'advanced_advisor' untuk menyusun hipotesis kausal dari DKG..."
        )
        gml_string = self.api_pool.call_gemini_for_text(
            prompt, "advanced_advisor")


        match = re.search(r"graph\s*\[.*?\]",
                          gml_string, re.DOTALL | re.IGNORECASE)
        if match:
            logger.info("✅ Berhasil mendapatkan GML kausal dari LLM.")
            return match.group(0)

        logger.error(
            "Gagal mengekstrak GML dari LLM. Menggunakan struktur DKG asli sebagai fallback."
        )
        return dkg_gml_structure
        logger.info(
            "Meminta 'advanced_advisor' untuk menyusun hipotesis grafik kausal..."
        )
        gml_string = self.api_pool.call_gemini_for_text(
            prompt, "advanced_advisor")


        match = re.search(r"graph\s*\[.*?\]", gml_string, re.DOTALL)
        if match:
            return match.group(0)

        logger.error(
            "Gagal mengekstrak GML dari respons LLM. Menggunakan grafik default."
        )

        return f"""
        graph [
            directed 1
            node [ id "{self.feature_names[0]}" ]
            node [ id "{self.target_cols[0]}" ]
            edge [ source "{self.feature_names[0]}" target "{self.target_cols[0]}" ]
        ]
        """

    def setup_causal_model(self):
        """
        Membangun dan memvisualisasikan Model Kausal Struktural (SCM) berbasis DKG.
        """
        logger.info(
            "\n--- [SCM] Membangun Model Kausal Struktural Berbasis DKG ---")


        self._populate_dkg()


        self.dkg.visualize()


        llm_causal_graph = self._discover_causal_graph_with_llm()


        self.df.rename(
            columns={col: col.replace("%", "pct") for col in self.target_cols},
            inplace=True,
        )
        valid_target_cols = [col.replace("%", "pct")
                             for col in self.target_cols]


        try:

            logger.info("Mencoba membuat model kausal dari hipotesis LLM...")
            self.causal_model = CausalModel(
                data=self.df,
                graph=llm_causal_graph,
                treatment="placeholder",
                outcome=valid_target_cols[0],
            )
            self.causal_graph = llm_causal_graph
            logger.info("✅ Model Kausal DoWhy (versi LLM) berhasil dibuat.")

        except Exception as e_llm:
            logger.warning(
                f"Gagal membuat model kausal dari GML LLM: {e_llm}. Mencoba fallback..."
            )

            try:
                dkg_gml_fallback = self.dkg.to_gml()
                self.causal_model = CausalModel(
                    data=self.df,
                    graph=dkg_gml_fallback,
                    treatment="placeholder",
                    outcome=valid_target_cols[0],
                )
                self.causal_graph = dkg_gml_fallback
                logger.info(
                    "✅ Model Kausal DoWhy (versi Fallback DKG) berhasil dibuat."
                )
            except Exception as e_fallback:
                logger.error(
                    f"Gagal total membuat model kausal bahkan dengan fallback: {e_fallback}",
                    exc_info=True,
                )
                self.causal_model = None


        if self.causal_model:
            try:
                graph_path = MODELS_DIR /                    f"dowhy_causal_graph_{self.project_id}.png"
                self.causal_model.view_model(
                    file_name=str(graph_path.with_suffix("")))
                logger.info(
                    f"✅ Visualisasi Grafik Kausal DoWhy disimpan di: {graph_path}"
                )
            except Exception as viz_e:
                logger.error(
                    f"Gagal membuat visualisasi DoWhy (mungkin karena Graphviz): {viz_e}"
                )
            self.causal_model = None

    def run_intervention_scenario(self):
        """
        Memungkinkan pengguna untuk melakukan intervensi pada sebuah variabel.
        """
        if not self.causal_model:
            logger.error(
                "Model kausal belum dibuat. Jalankan setup_causal_model() terlebih dahulu."
            )
            return

        logger.info("\n--- [INTERVENSI] Menjalankan Skenario Intervensi ---")


        node_list = list(self.causal_model._graph.get_nodes())

        treatment_var = questionary.select(
            "Pilih variabel 'treatment' (yang akan diintervensi):", choices=node_list
        ).ask()
        if not treatment_var:
            return

        try:

            identified_estimand = self.causal_model.identify_effect(
                proceed_when_unidentifiable=True
            )


            estimate = self.causal_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                test_significance=True,
            )

            logger.info("--- Hasil Estimasi Efek Kausal ---")
            logger.info(str(estimate))


            interpretation_prompt = f"""
            Anda adalah seorang Analis Kuantitatif Senior.
            Jelaskan hasil analisis kausal berikut kepada seorang Manajer Portofolio dalam bahasa yang mudah dimengerti.

            Konteks: Kita sedang menganalisis apa dampak dari perubahan '{treatment_var}' terhadap '{self.target_cols[0]}'.
            Hasil Statistik (Estimasi Efek Kausal Rata-rata): {estimate.value}

            Penjelasan Anda:
            """
            interpretation = self.api_pool.call_gemini_for_text(
                interpretation_prompt, "advanced_advisor"
            )
            logger.info("\n--- Interpretasi AI ---")
            print(interpretation)

        except Exception as e:
            logger.error(f"Gagal menjalankan intervensi: {e}", exc_info=True)

    def answer_counterfactual_question(self):
        """
        Menjawab pertanyaan kontrafaktual menggunakan model prediktif yang sudah ada.
        """
        logger.info("\n--- [KONTRAFAKTUAL] Menjawab Pertanyaan 'What If' ---")


        factual_data_window = self.trained_model.datamodule.X_val[
            -self.trained_model.hparams.window:
        ]
        factual_tensor = torch.from_numpy(
            factual_data_window.astype(np.float32)
        ).unsqueeze(0)
        edge_index = self.trained_model.datamodule.graph_edge_index

        with torch.no_grad():
            factual_prediction, _ = self.trained_model.predict(
                factual_tensor, edge_index, 0.0
            )
            factual_return = factual_prediction[
                0, 0, 0
            ].item()

        logger.info(
            f"Faktual: Prediksi return untuk besok adalah {factual_return:.4f}")


        variable_to_change = questionary.select(
            "Variabel mana yang ingin Anda ubah nilainya (secara hipotetis)?",
            choices=self.feature_names,
        ).ask()
        if not variable_to_change:
            return

        try:
            new_value_str = questionary.text(
                f"Masukkan nilai baru untuk '{variable_to_change}':"
            ).ask()
            new_value = float(new_value_str)
        except (ValueError, TypeError):
            logger.error("Nilai tidak valid.")
            return


        counterfactual_data_window = factual_data_window.copy()
        try:
            var_index = self.feature_names.index(variable_to_change)

            counterfactual_data_window[:, var_index] = new_value

            counterfactual_tensor = torch.from_numpy(
                counterfactual_data_window.astype(np.float32)
            ).unsqueeze(0)


            with torch.no_grad():
                counterfactual_prediction, _ = self.trained_model.predict(
                    counterfactual_tensor, edge_index, 0.0
                )
                counterfactual_return = counterfactual_prediction[0, 0, 0].item(
                )

            logger.info(
                f"Kontrafaktual: Prediksi return jika '{variable_to_change}' menjadi {new_value} adalah {counterfactual_return:.4f}"
            )


            interpretation_prompt = f"""
            Anda adalah seorang Analis Risiko.
            Jelaskan skenario kontrafaktual berikut ini.

            Skenario Faktual:
            - Berdasarkan data saat ini, prediksi return untuk {self.tickers[0]} untuk hari berikutnya adalah {factual_return:.4f}.

            Skenario Kontrafaktual (What-If):
            - Jika variabel '{variable_to_change}' diubah nilainya menjadi {new_value}, prediksi return berubah menjadi {counterfactual_return:.4f}.

            Analisis Anda:
            - Seberapa besar perubahannya?
            - Apa implikasi dari perubahan ini?
            - Apakah ini menunjukkan bahwa '{variable_to_change}' adalah pendorong penting bagi pergerakan harga {self.tickers[0]}?
            """
            interpretation = self.api_pool.call_gemini_for_text(
                interpretation_prompt, "advanced_advisor"
            )
            logger.info("\n--- Analisis Kontrafaktual AI ---")
            print(interpretation)

        except ValueError:
            logger.error(
                f"Variabel '{variable_to_change}' tidak ditemukan dalam daftar fitur."
            )
        except Exception as e:
            logger.error(
                f"Gagal menjalankan analisis kontrafaktual: {e}", exc_info=True
            )


class GraphAutoencoder(torch.nn.Module):
    """Model GNN sederhana untuk pre-training DKG menggunakan link prediction."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = PyG_GCNConv(in_channels, hidden_channels)
        self.conv2 = PyG_GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):

        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):

        prob_adj = z @ z.t()
        return prob_adj


def pretrain_dkg_embeddings(
    dkg: DynamicKnowledgeGraph, project_id: str, epochs: int = 100
) -> Optional[Path]:
    """Melatih GNN pada DKG untuk menghasilkan embedding node yang kaya."""
    if not dkg.nodes or not dkg.edges:
        logger.warning("[DKG Pre-train] DKG kosong. Melewatkan pra-pelatihan.")
        return None

    logger.info(
        "--- 🧠 Memulai Pra-Pelatihan pada Dynamic Knowledge Graph (DKG) ---")
    STAGE_PREP.trigger("fine_tune_prep", lambda: build_finetune_assets({"project_id": project_id}))

    node_list = sorted(list(dkg.nodes.keys()))
    node_map = {name: i for i, name in enumerate(node_list)}

    x = torch.randn(len(node_list), 16)

    source_nodes = [
        node_map[s] for s, t, d in dkg.edges if s in node_map and t in node_map
    ]
    target_nodes = [
        node_map[t] for s, t, d in dkg.edges if s in node_map and t in node_map
    ]
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    data = PyG_Data(x=x, edge_index=edge_index)


    model = GraphAutoencoder(
        in_channels=data.num_node_features, hidden_channels=64, out_channels=32
    )
    optimizer = torch_optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)


        pos_loss = -            torch.log(model.decode(z, data.edge_index).sigmoid() + 1e-15).mean()


        num_neg_samples = data.edge_index.size(1)
        neg_edge_index = torch.randint(
            0, data.num_nodes, (2, num_neg_samples), dtype=torch.long
        )
        neg_loss = -torch.log(
            1 - model.decode(z, neg_edge_index).sigmoid() + 1e-15
        ).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            logger.info(
                f"DKG Pre-train Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}"
            )


    model.eval()
    with torch.no_grad():
        final_embeddings = model.encode(data.x, data.edge_index)

    embedding_dict = {
        node_name: final_embeddings[i] for node_name, i in node_map.items()
    }

    save_path = (
        get_path(project_id, "checkpoint_dir") /
        f"dkg_embeddings_{project_id}.pt"
    )
    torch.save(embedding_dict, save_path)
    logger.info(f"✅ DKG embeddings disimpan di: {save_path}")

    return save_path




def run_benchmark_suite(auditor, api_pool, together_keys, web_searcher, brain: "Brain"):
    """Fungsi utama untuk menginisialisasi dan menjalankan OG-Benchmark Suite dinamis."""

    try:
        last_id, ckpt_path_str = LAST_PROJECT_INFO_TXT.read_text().split(",")
        ckpt_path = Path(ckpt_path_str)

        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        hparams = ckpt.get("hyper_parameters", {})
        if not hparams:
            raise ValueError(
                "Hyperparameters tidak ditemukan di dalam checkpoint.")
        hparams["data_path"] = get_path(None, "data")


        _ = STAGE_PREP.wait("predict_prep", timeout=600)
        dm = AlphaDataModule(
            hparams, auditor, api_pool, together_keys, {}, web_searcher, brain=brain
        )
        dm.setup(stage="predict")

        model = AlphaLit.load_from_checkpoint(
            checkpoint_path=ckpt_path, gemini_advisor=None, api_pool=api_pool
        )
        model.eval()

    except FileNotFoundError:
        logger.error(
            f"File info model terakhir ({LAST_PROJECT_INFO_TXT}) tidak ditemukan. Jalankan fine-tune terlebih dahulu."
        )
        return
    except Exception as e:
        logger.error(f"Gagal memuat komponen untuk benchmark: {e}")
        traceback.print_exc()
        return


    suite = OG_BenchmarkSuite(
        auditor=auditor,
        api_pool=api_pool,
        dm=dm,
        model=model,
        together_keys=together_keys,
    )
    suite.run()


def get_file_hash(filepath):
    """Menghitung hash SHA256 dari sebuah file untuk validasi integritas."""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while True:
                chunk = file.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        logger.error(f"File dataset tidak ditemukan di path: {filepath}")
        return None


IQRO_MEMORY_DB = MODELS_DIR / "iqro_memory.json"


def _load_iqro_memory() -> dict:
    """Memuat 'database perpustakaan' yang berisi file yang pernah dibaca."""
    if not IQRO_MEMORY_DB.exists():
        return {}
    try:
        with IQRO_MEMORY_DB.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def _update_iqro_memory(memory: dict, file_name: str, file_hash: str):
    """Memperbarui 'database perpustakaan' dengan info file yang baru dibaca."""
    memory[file_name] = {"hash": file_hash,
                         "last_read": datetime.now().isoformat()}
    with IQRO_MEMORY_DB.open("w") as f:
        json.dump(memory, f, indent=4)


def generate_tree_of_thought_summary(
    full_text: str, file_path: Path, brain: Brain, agent: "TogetherLLM"
):
    """Menciptakan rangkuman hierarkis dari sebuah teks dan menyimpannya ke DKG."""
    logger.info(
        f"🌳 [ToT] Membuat rangkuman hierarkis untuk {file_path.name}...")
    prompt = f"""
    Anda adalah seorang arsitek informasi. Buatlah rangkuman hierarkis (Tree of Thought) dari teks berikut.
    Identifikasi tujuan utama, konsep-konsep kunci, dan sub-konsep di dalamnya.
    Jawab HANYA dengan format JSON sesuai skema HierarchicalSummary.

    Teks:
    ---
    {full_text[:15000]}
    ---
    """
    try:
        summary_obj = agent.chat(prompt, response_model=HierarchicalSummary)

        summary_node_id = f"SUMMARY_{file_path.name}"
        brain.dkg.add_node(
            summary_node_id,
            "HierarchicalSummary",
            layer="Abstract",
            purpose=summary_obj.main_purpose,

            tree=summary_obj.thought_tree.model_dump_json(),
        )
        logger.info(
            f"    -> ✅ Rangkuman ToT untuk {file_path.name} berhasil dibuat dan disimpan."
        )

    except Exception as e:
        logger.error(
            f"Gagal membuat rangkuman ToT untuk {file_path.name}: {e}")


def calculate_diffusion_loss(
    denoising_model: nn.Module, x_0: torch.Tensor, num_diffusion_timesteps: int
) -> torch.Tensor:
    """
    Menjalankan satu langkah proses difusi (forward dan reverse) dan menghitung loss.

    Args:
        denoising_model (nn.Module): Model yang memprediksi noise.
        x_0 (torch.Tensor): Data bersih asli (batch data). Shape: [batch, seq_len, features].
        num_diffusion_timesteps (int): Jumlah total langkah difusi.

    Returns:
        torch.Tensor: Nilai loss MSE dari prediksi noise.
    """
    device = x_0.device
    batch_size = x_0.shape[0]


    betas = torch.linspace(1e-4, 0.02, num_diffusion_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)


    t = torch.randint(0, num_diffusion_timesteps,
                      (batch_size,), device=device).long()


    noise = torch.randn_like(x_0)


    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alphas_cumprod[t])


    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(batch_size, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(
        batch_size, 1, 1
    )

    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


    predicted_noise = denoising_model(x_t, t)


    loss = F.mse_loss(noise, predicted_noise)

    return loss


def run_internal_brainstorming_cycle(
    best_hparams: dict,
    reward_model: "RewardModel",
    featurizer: "HParamsFeaturizer",
    num_candidates: int = 10,
    score_threshold: float = 0.85,
) -> Optional[dict]:
    """
    Mensimulasikan brainstorming internal untuk menghasilkan proposal hyperparameter baru
    tanpa menggunakan LLM eksternal, hanya mengandalkan Reward Model internal.
    """
    if not reward_model or not featurizer:
        logger.warning(
            "🧠 [Self-Reliance] Reward Model tidak tersedia. Brainstorming internal dilewati."
        )
        return None

    logger.info(
        f"🧠 [Self-Reliance] Menjalankan brainstorming internal dengan {num_candidates} kandidat..."
    )

    best_internal_proposal = None
    highest_score = 0.0

    for _ in range(num_candidates):
        candidate_hparams = best_hparams.copy()


        param_to_tweak = random.choice(["lr", "dropout", "d_model", "window"])
        if param_to_tweak == "lr":
            candidate_hparams["lr"] *= random.uniform(0.5, 1.5)
        elif param_to_tweak == "dropout":
            candidate_hparams["dropout"] = max(
                0.1, min(
                    0.5, candidate_hparams["dropout"] + random.uniform(-0.1, 0.1))
            )
        elif param_to_tweak == "d_model":
            candidate_hparams["d_model"] = random.choice([64, 128, 192, 256])
        elif param_to_tweak == "window":
            candidate_hparams["window"] += random.randint(-15, 15)


        try:
            feature_vector = featurizer.transform(candidate_hparams)
            feature_tensor = torch.tensor(
                feature_vector, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                predicted_score = reward_model(feature_tensor).item()

            if predicted_score > highest_score:
                highest_score = predicted_score
                best_internal_proposal = candidate_hparams
        except Exception as e:

            continue

    if highest_score >= score_threshold:
        logger.info(
            f"✅ Brainstorming internal berhasil! Proposal ditemukan dengan skor prediksi {highest_score:.2f}."
        )
        return best_internal_proposal
    else:
        logger.warning(
            f"Brainstorming internal tidak menemukan solusi yang meyakinkan (skor tertinggi: {highest_score:.2f})."
        )
        return None


def train_layer_forward_forward(
    layer: nn.Module,
    positive_input: torch.Tensor,
    negative_input: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    threshold: float = 1.5,
    epochs: int = 1,
):
    """
    Melatih satu layer menggunakan Forward-Forward Algorithm.

    Args:
        layer (nn.Module): Layer yang akan dilatih.
        positive_input (torch.Tensor): Input data nyata.
        negative_input (torch.Tensor): Input data rekaan/negatif.
        optimizer (torch.optim.Optimizer): Optimizer lokal untuk layer ini.
        threshold (float): Batas goodness yang diinginkan.
        epochs (int): Berapa kali iterasi FF dijalankan untuk layer ini.

    Returns:
        float: Nilai loss rata-rata dari proses FF.
    """
    mean_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad()



        goodness_positive = torch.sum(layer(positive_input) ** 2, dim=-1)


        goodness_negative = torch.sum(layer(negative_input) ** 2, dim=-1)





        loss_positive = F.softplus(threshold - goodness_positive).mean()
        loss_negative = F.softplus(goodness_negative - threshold).mean()

        loss = loss_positive + loss_negative


        loss.backward()
        optimizer.step()

        mean_loss += loss.item()

    return mean_loss / epochs


class MDN_RNN(nn.Module):
    def __init__(self, latent_dim=32, n_gaussians=5, hidden_size=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_gaussians = n_gaussians
        self.lstm = nn.LSTM(latent_dim, hidden_size, batch_first=True)
        self.fc_pi = nn.Linear(hidden_size, n_gaussians)
        self.fc_sigma = nn.Linear(hidden_size, n_gaussians * latent_dim)
        self.fc_mu = nn.Linear(hidden_size, n_gaussians * latent_dim)
        logger.info(
            f"MDN-RNN World Model diinisialisasi: Latent Dim={latent_dim}, Gaussians={n_gaussians}"
        )

    def forward(self, z):
        lstm_out, _ = self.lstm(z)
        lstm_out_last_step = lstm_out[:, -1, :]
        pi = F.softmax(self.fc_pi(lstm_out_last_step), dim=-1)
        sigma = torch.exp(self.fc_sigma(lstm_out_last_step)).view(
            -1, self.n_gaussians, self.latent_dim
        )
        mu = self.fc_mu(lstm_out_last_step).view(-1,
                                                 self.n_gaussians, self.latent_dim)
        return pi, sigma, mu


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    MSE = F.mse_loss(recon_x, x.view(-1, x.shape[-1]), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (beta * KLD)


def mdn_loss_function(pi, sigma, mu, target_z):
    target_z_expanded = target_z.unsqueeze(1).expand_as(mu)
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob_per_component = m.log_prob(target_z_expanded)
    log_prob_sum = torch.sum(log_prob_per_component, dim=2)
    log_likelihood = torch.logsumexp(
        torch.log(pi + 1e-8) + log_prob_sum, dim=1)
    return -torch.mean(log_likelihood)


def sample_from_gmm(pi, sigma, mu):
    categorical = torch.distributions.Categorical(pi)
    component_indices = categorical.sample()
    batch_size = pi.size(0)
    mu_selected = mu[torch.arange(batch_size), component_indices]
    sigma_selected = sigma[torch.arange(batch_size), component_indices]
    return torch.distributions.Normal(loc=mu_selected, scale=sigma_selected).sample()


def run_world_model_training(hparams: dict):
    """
    Mengorkestrasi seluruh alur pelatihan untuk komponen World Model (VAE dan MDN-RNN)
    dan menyimpan metadata untuk validasi di masa depan.
    """


    def train_vae(hparams, dm, device):

        logger.info("\n--- TAHAP A.1: Pelatihan Mata & Tangan (VAE) ---")
        vae_path = (
            get_path(hparams["project_id"], "checkpoint_dir") /
            f"world_model_vae.pth"
        )
        vae = VAE(
            input_dim=dm.n_features_input, latent_dim=hparams["wm_latent_dim"]
        ).to(device)
        optimizer = optim.Adam(
            vae.parameters(), lr=hparams.get("vae_lr", 1e-4))


        class VAEDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return torch.tensor(self.data[idx], dtype=torch.float32)

        train_loader = TorchDataLoader(
            VAEDataset(dm.X_train),
            batch_size=hparams.get("batch_size", 128),
            shuffle=True,
        )
        val_loader = TorchDataLoader(
            VAEDataset(dm.X_val), batch_size=hparams.get("batch_size", 128)
        )
        best_val_loss = float("inf")


        for epoch in range(hparams.get("vae_epochs", 50)):

            vae.train()
            train_loss = 0
            for batch_x in tqdm(train_loader, desc=f"VAE Epoch {epoch+1}", leave=False):

                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                recon_x, mu, logvar = vae(batch_x)
                loss = vae_loss_function(recon_x, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            vae.eval()
            val_loss = 0

            with torch.no_grad():

                for batch_x_val in val_loader:

                    batch_x_val = batch_x_val.to(device)
                    recon_x, mu, logvar = vae(batch_x_val)
                    val_loss += vae_loss_function(
                        recon_x, batch_x_val, mu, logvar
                    ).item()

            avg_train_loss, avg_val_loss = train_loss / len(
                train_loader.dataset
            ), val_loss / len(val_loader.dataset)
            logger.info(
                f"VAE Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )


            if avg_val_loss < best_val_loss:

                best_val_loss = avg_val_loss
                torch.save(vae.state_dict(), vae_path)
                logger.info(f"Model VAE terbaik disimpan ke {vae_path}")

        final_vae = VAE(
            input_dim=dm.n_features_input, latent_dim=hparams["wm_latent_dim"]
        )
        final_vae.load_state_dict(torch.load(vae_path))
        return final_vae


def discover_new_rules_with_ilp(
    initial_hparams: dict,
    api_pool: "DistributedAIPool",
    **kwargs
):
    """
    Menjalankan siklus penemuan aturan simbolik menggunakan simulasi
    Inductive Logic Programming (ILP) berbasis LLM.
    1. Mengubah data numerik menjadi fakta logis (diskritisasi).
    2. Meminta LLM untuk menambang aturan dari fakta tersebut.
    3. Mengubah aturan logis menjadi kode DifferentiableRule yang siap pakai.
    """
    logger.info("\n" + "="*80)
    logger.info("=== 🤖💡 MEMULAI SIKLUS PENEMUAN ATURAN SIMBOLIK (ILP) 🤖💡 ===")
    logger.info("="*80)

    try:

        logger.info(
            "  -> Tahap 1/4: Mempersiapkan dan mengubah data menjadi fakta logis...")
        df_raw = pd.read_parquet(initial_hparams['data_path'])
        tickers = initial_hparams['selected_tickers']


        df_featured, _ = generate_all_features(
            df_raw, tickers, master_event_list=[])


        features_to_discretize = [
            f"{tickers[0]}_RSI_14",
            f"{tickers[0]}_Volume_pct_change",
            f"{tickers[0]}_Volatility_20",
            "aset_sensor1_trend_correlation"
        ]

        df_facts = pd.DataFrame(index=df_featured.index)
        for col in features_to_discretize:
            if col in df_featured.columns:

                df_facts[f'fact_{col}'] = pd.qcut(df_featured[col], 5, labels=[
                    "sangat_rendah", "rendah", "normal", "tinggi", "sangat_tinggi"
                ], duplicates='drop')


        target_col = f"{tickers[0]}_log_return"
        df_facts['fact_market_down_tomorrow'] = (
            df_featured[target_col].shift(-1) < -0.015)

        df_facts.dropna(inplace=True)


        logger.info(
            "  -> Tahap 2/4: Membangun basis pengetahuan untuk dianalisis AI...")
        knowledge_base = []
        for i in range(len(df_facts) - 1):
            time_id = f"t{i}"

            for col in df_facts.columns:
                if col.startswith('fact_') and 'tomorrow' not in col:
                    knowledge_base.append(
                        f"{col.replace('fact_', '')}({time_id}, {df_facts[col].iloc[i]}).")


            if df_facts['fact_market_down_tomorrow'].iloc[i]:
                knowledge_base.append(f"market_down(t{i+1}).")

        knowledge_base_str = "\n".join(
            knowledge_base[:1000])


        logger.info(
            "  -> Tahap 3/4: Menugaskan AI untuk menambang aturan logis...")

        class DiscoveredRule(BaseModel):
            rule_name: str = Field(
                description="Nama deskriptif untuk aturan, cth: 'RSI_Overbought_Volume_Spike_Predicts_Drop'")
            premise: list[str] = Field(
                description="Daftar kondisi (premis) dalam format 'nama_fitur(waktu, nilai)', cth: ['aset_sensor1_trend_correlation(T, sangat_rendah)']")
            conclusion: str = Field(
                description="Kesimpulan (konklusi) aturan, cth: 'market_down(T+1)'")
            explanation: str = Field(
                description="Penjelasan singkat dalam bahasa natural mengapa aturan ini masuk akal.")

        mining_prompt = f"""
        Anda adalah seorang peneliti ahli dalam Inductive Logic Programming.
        Tugas Anda adalah menemukan satu aturan logis IF-THEN dari basis pengetahuan berikut.
        Aturan harus dalam bentuk: IF [premise di waktu T] THEN [conclusion di waktu T+1].

        Basis Pengetahuan (Contoh Fakta):
        ---
        {knowledge_base_str}
        ---

        Tujuan: Temukan kombinasi fakta di waktu T yang paling sering memprediksi fakta `market_down(T+1)`.
        
        Kembalikan aturan yang Anda temukan HANYA dalam format JSON sesuai skema `DiscoveredRule`.
        """


        discovered_rule_dict = api_pool.call_gemini_with_tool(
            mining_prompt, "supervisor", DiscoveredRule
        )

        if not discovered_rule_dict:
            logger.error(
                "  -> GAGAL: AI tidak dapat menemukan aturan logis baru. Siklus berhenti.")
            return

        rule = DiscoveredRule(**discovered_rule_dict)
        logger.info(f"  -> ATURAN DITEMUKAN: {rule.explanation}")


        logger.info(
            "  -> Tahap 4/4: Menugaskan AI Engineer untuk menulis kode DifferentiableRule...")


        available_features = df_featured.columns.tolist()

        code_generation_prompt = f"""
        Anda adalah seorang Insinyur AI PyTorch.
        Terjemahkan aturan logis berikut menjadi file Python yang berisi subclass dari `DifferentiableRule`.

        Aturan untuk Diterjemahkan:
        - Nama: {rule.rule_name}
        - Premis (IF): {rule.premise}
        - Konklusi (THEN): {rule.conclusion}
        - Penjelasan: {rule.explanation}
        
        Konteks & Aturan:
        - Buat fungsi logika (`..._logic`) yang menerima `x_raw_features`, `feature_indices`, `thresholds`.
        - `x_raw_features` memiliki shape [Batch, SeqLen, Features]. Gunakan data dari time step terakhir (`[:, -1, :]`).
        - Gunakan `torch.sigmoid((feature - threshold) * 10)` untuk perbandingan 'differentiable'.
        - Aturan AND diimplementasikan dengan perkalian.
        - Buat fungsi `create_rule(feature_map)` yang mengembalikan instance dari kelas aturan Anda.
        - Pastikan untuk mengimpor `torch`, `torch.nn`, dan `DifferentiableRule`.
        - Gunakan fitur dari daftar ini: {available_features[:100]}

        Kembalikan HANYA kode Python mentah untuk seluruh file.
        """

        generated_code = api_pool.call_gemini_for_text(
            code_generation_prompt, "ai_engineer")

        if "DifferentiableRule" in generated_code and "create_rule" in generated_code:
            rule_dir = Path.home() / APP_BRAND / "rules"
            rule_dir.mkdir(exist_ok=True)
            rule_file_path = rule_dir / f"ilp_{rule.rule_name.lower()}.py"

            with open(rule_file_path, "w", encoding="utf-8") as f:
                f.write(generated_code)
            logger.info(
                f"  -> ✅ Kode aturan baru '{rule.rule_name}' berhasil dibuat dan disimpan di: {rule_file_path.name}")
        else:
            logger.error(
                f"  -> GAGAL: AI Engineer tidak dapat menghasilkan kode yang valid.")

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Penemuan Aturan (ILP): {e}", exc_info=True)

    def train_mdn_rnn(hparams, dm, vae, device):
        logger.info("\n--- TAHAP A.2: Pelatihan Otak Imajinasi (MDN-RNN) ---")
        mdn_path = (
            get_path(hparams["project_id"], "checkpoint_dir")
            / f"world_model_mdn_rnn.pth"
        )
        logger.info("Meng-encode seluruh dataset menjadi ruang laten Z...")
        vae.to(device)
        vae.eval()
        with torch.no_grad():
            full_dataset_tensor = torch.tensor(
                np.concatenate([dm.X_train, dm.X_val]), dtype=torch.float32
            ).to(device)
            mu, logvar = vae.encode(full_dataset_tensor)
            z_sequence = vae.reparameterize(mu, logvar)

        class RNNDataset(Dataset):
            def __init__(self, z_seq, seq_len):
                self.z_seq, self.seq_len = z_seq, seq_len

            def __len__(self):
                return len(self.z_seq) - self.seq_len

            def __getitem__(self, idx):
                return (
                    self.z_seq[idx: idx + self.seq_len],
                    self.z_seq[idx + self.seq_len],
                )

        rnn_dataset = RNNDataset(z_sequence, hparams.get("mdn_seq_len", 50))
        rnn_loader = TorchDataLoader(
            rnn_dataset, batch_size=hparams.get("batch_size", 128), shuffle=True
        )
        mdn_rnn = MDN_RNN(
            latent_dim=hparams["wm_latent_dim"],
            n_gaussians=hparams.get("mdn_gaussians", 5),
        ).to(device)
        optimizer = optim.Adam(mdn_rnn.parameters(),
                               lr=hparams.get("mdn_lr", 1e-3))
        best_loss = float("inf")
        for epoch in range(hparams.get("mdn_epochs", 30)):
            mdn_rnn.train()
            total_loss = 0
            for z_in, z_target in tqdm(
                rnn_loader, desc=f"MDN-RNN Epoch {epoch+1}", leave=False
            ):
                z_in, z_target = z_in.to(device), z_target.to(device)
                optimizer.zero_grad()
                pi, sigma, mu = mdn_rnn(z_in)
                loss = mdn_loss_function(pi, sigma, mu, z_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(rnn_loader)
            logger.info(f"MDN-RNN Epoch {epoch+1}: Loss: {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(mdn_rnn.state_dict(), mdn_path)
                logger.info(f"Model MDN-RNN terbaik disimpan ke {mdn_path}")

    logger.info("Memulai Alur Kerja Pelatihan World Model Terintegrasi.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_auditor = type(
        "Dummy", (object,), {"add_log": lambda *args, **kwargs: None}
    )()

    dummy_brain = Brain(
        index_path=str(get_path(None, "brain.faiss")),
        db_path=str(get_path(None, "brain.sqlite")),
        embed_model_instance=_supremacy_embed_model,
        dim=EMBEDDED_CFG["faiss_dim"],
        api_pool=api_pool,
    )
    dm = AlphaDataModule(
        hparams, dummy_auditor, None, None, None, None, brain=dummy_brain
    )
    dm.setup(stage="fit")
    trained_vae = train_vae(hparams, dm, device)
    train_mdn_rnn(hparams, dm, trained_vae, device)
    logger.info("✅ Pelatihan dasar World Model selesai.")
    logger.info("Membuat file metadata untuk World Model...")
    data_path = hparams["data_path"]
    dataset_hash = get_file_hash(data_path)
    
    if dataset_hash:
        metadata = {
            "last_trained_on": datetime.now().isoformat(),
            "dataset_path": str(data_path),
            "dataset_sha256_hash": dataset_hash,
        }
        meta_path = (
            get_path(hparams["project_id"], "checkpoint_dir") /
            "world_model.meta.json"
        )
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata World Model disimpan ke {meta_path}")
    else:

        logger.error(
            "Gagal membuat hash dataset, metadata tidak akan disimpan.")


def run_dkg_optimization_cycle(dkg: DynamicKnowledgeGraph, project_id: str):
    """
    Mensimulasikan proses 'Mielinasi' dengan melatih ulang GNN pada DKG
    untuk memperkuat dan mempercepat jalur informasi.
    """
    logger.info(
        "⚡ [Myelination] Memulai siklus optimisasi Dynamic Knowledge Graph...")
    try:

        train_ml_dkg(
            dkg, project_id, epochs=25
        )
        logger.info(
            "⚡ [Myelination] Jalur pengetahuan di DKG telah diperkuat.")
    except Exception as e:
        logger.error(f"Gagal menjalankan siklus Mielinasi DKG: {e}")


def run_autonomous_core_loop(stop_event: threading.Event, **system_components):
    """
    Loop utama yang menjalankan AI secara mandiri berdasarkan tujuan.
    """
    logger.info(
        "🚀 [AUTONOMOUS CORE] Siklus Operasi Otonom Dimulai. Model sekarang memiliki kebebasan penuh.")

    while not stop_event.is_set():
        try:

            try:
                user_task = TASK_QUEUE.get_nowait()
                logger.warning(
                    f"📥 Perintah baru diterima dari pengguna: {user_task['description']}")


            except queue.Empty:
                pass


            active_goal = nsmm.get_active_goal()

            if not active_goal:
                logger.info(
                    "🎯 Tidak ada tujuan aktif. Mencoba merumuskan tujuan baru...")
                if formulate_new_autonomous_goal(nsmm, brain, governor, api_pool):
                    continue
                else:
                    logger.warning(
                        "Gagal merumuskan tujuan baru. Menunggu siklus berikutnya.")
                    stop_event.wait(60)
                    continue


            logger.info(
                f"🧠 Perencanaan untuk tujuan: {active_goal['description']}")
            plan = master_planner.generate_plan(
                nsmm=nsmm,
                governor=governor,
                history=[],
                task_queue=TASK_QUEUE,
                df_full=df_raw,
                brain=brain,
                engine=async_engine
            )

            if not plan or not plan.tasks:
                logger.warning(
                    "Master Planner tidak menghasilkan rencana. Menunggu siklus berikutnya.")
                stop_event.wait(60)
                continue


            logger.info(f"📜 Rencana Dibuat: {plan.overall_reasoning}")
            for i, task in enumerate(plan.tasks):
                logger.info(
                    f"  -> Mengeksekusi Tugas {i+1}/{len(plan.tasks)}: {task.task_name}")
                if task.task_name in TASK_DISPATCHER:

                    full_args = system_components.copy()
                    full_args['initial_hparams'] = system_components.get(
                        'hparams')


                    TASK_DISPATCHER[task.task_name](
                        params=task.params, **full_args)
                else:
                    logger.error(
                        f"Tugas tidak dikenal: '{task.task_name}'. Melewati.")


            nsmm.mark_goal_completed(active_goal['goal_id'])
            logger.info(f"✅ Tujuan '{active_goal['description']}' selesai.")

            run_goal_reflection_cycle(
                nsmm, brain, api_pool, feedback_manager, governor)


            stop_event.wait(30)

        except Exception as e:
            logger.critical(
                f"FATAL ERROR di dalam Autonomous Core Loop: {e}", exc_info=True)
            stop_event.wait(300)


def run_memory_consolidation_cycle(
    nsmm: NSMM,
    brain: Brain,
    api_pool: "DistributedAIPool",
    embedding_model: "APIEmbedder",
):
    """
    Menjalankan siklus pemeliharaan memori: Pruning (pemangkasan) dan Synthesis (sintesis).
    Ini adalah fungsi dari "Pustakawan AI".
    """
    logger.info("🤖 [Pustakawan AI] Memulai siklus konsolidasi memori...")
    new_wisdom_synthesized = 0


    logger.info(
        "🤖 [Pustakawan AI] Tahap 1: Memangkas (Pruning) memori usang...")
    neuron_counts_before = nsmm.get_neuron_count_by_status()
    logger.info(f"    -> Kondisi Memori Awal: {neuron_counts_before}")

    ids_to_prune = nsmm.get_neurons_for_pruning(
        confidence_threshold=0.1, age_days=90)
    if ids_to_prune:
        nsmm.update_neuron_status(ids_to_prune, "archived")
        logger.warning(
            f"    -> Pruning: {len(ids_to_prune)} neuron usang/tidak berguna telah diarsipkan."
        )
        neuron_counts_after = nsmm.get_neuron_count_by_status()
        logger.info(f"    -> Kondisi Memori Akhir: {neuron_counts_after}")
    else:
        logger.info("    -> Tidak ada neuron yang perlu di-pruning saat ini.")


    logger.info("🤖 [Pustakawan AI] Tahap 2: Mensintesis kebijaksanaan baru...")
    experimental_neurons = nsmm.get_neurons_for_consolidation(
        status="experimental", limit=500
    )
    if len(experimental_neurons) < 50:
        logger.info(
            "    -> Jumlah neuron eksperimental belum cukup untuk sintesis.")
        return new_wisdom_synthesized


    reasons = [n["trigger_reason"] for n in experimental_neurons]
    embeddings = embedding_model.encode(reasons, task_type="passage")


    from sklearn.cluster import KMeans

    num_clusters = max(5, len(experimental_neurons) // 10)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(
        embeddings
    )

    neurons_to_archive_after_synthesis = []


    for i in range(num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        if len(cluster_indices) < 5:
            continue

        cluster_neurons = [experimental_neurons[j] for j in cluster_indices]
        cluster_reasons = "- " + "\n- ".join(
            [n["trigger_reason"] for n in cluster_neurons]
        )


        synthesis_prompt = f"""
        Anda adalah seorang AI Research Scientist. Analisis kumpulan diagnosis/alasan berikut dari pengalaman pelatihan model sebelumnya.
        Sintesis semua poin ini menjadi SATU prinsip atau "kebijaksanaan" (wisdom) umum yang dapat dipelajari.

        Kumpulan Pengalaman:
        {cluster_reasons}

        Prinsip Umum atau Kebijaksanaan yang Disintesis (satu kalimat):
        """
        try:

            wisdom_neuron_text = api_pool.call_gemini_for_text(
                synthesis_prompt, "supervisor"
            )


            source_name = (
                f"WisdomNeuron_Synthesis_{datetime.now().strftime('%Y%m%d_%H%M')}"
            )
            brain.add_chunks(
                [wisdom_neuron_text],
                source_name=f"[SINTESIS_KEBIJAKSANAAN] {source_name}",
            )
            logger.info(
                f"💡 [Pustakawan AI] Kebijaksanaan baru disintesis dan disimpan ke Brain: '{wisdom_neuron_text}'"
            )


            new_wisdom_synthesized += 1


            neurons_to_archive_after_synthesis.extend(
                [n["id"] for n in cluster_neurons]
            )

        except Exception as e:
            logger.error(
                f"🤖 [Pustakawan AI] Gagal melakukan sintesis untuk cluster {i}: {e}"
            )


    if neurons_to_archive_after_synthesis:
        nsmm.update_neuron_status(
            list(set(neurons_to_archive_after_synthesis)), "archived"
        )
        logger.info(
            f"    -> {len(neurons_to_archive_after_synthesis)} neuron telah disintesis dan diarsipkan."
        )

    logger.info("🤖 [Pustakawan AI] Siklus konsolidasi memori selesai.")
    return new_wisdom_synthesized


def run_hypothetical_simulation_cycle(
    nsmm: NSMM, dm: "AlphaDataModule", project_id: str, device: torch.device
):
    """
    Menjalankan siklus 'mimpi' atau simulasi untuk menguji neuron [Eksperimental].
    """
    logger.info("🌌 [Virtual Sandbox] Memulai siklus simulasi hipotetikal...")


    try:
        vae_path = get_path(project_id, "checkpoint_dir") /            "world_model_vae.pth"
        mdn_path = get_path(project_id, "checkpoint_dir") /            "world_model_mdn_rnn.pth"

        vae = VAE(input_dim=dm.n_features_input, latent_dim=32).to(device)
        mdn_rnn = MDN_RNN(latent_dim=32).to(device)

        vae.load_state_dict(torch.load(vae_path, map_location=device))
        mdn_rnn.load_state_dict(torch.load(mdn_path, map_location=device))
        vae.eval()
        mdn_rnn.eval()
    except FileNotFoundError:
        logger.warning(
            "🌌 [Virtual Sandbox] Model VAE/MDN-RNN tidak ditemukan. Pelatihan World Model diperlukan. Siklus dilewati."
        )
        return


    experimental_neurons = nsmm.get_neurons_for_consolidation(
        status="experimental")
    if not experimental_neurons:
        logger.info(
            "🌌 [Virtual Sandbox] Tidak ada neuron eksperimental baru untuk diuji."
        )
        return


    logger.info(
        f"🌌 [Virtual Sandbox] Menjalankan simulasi untuk menguji {len(experimental_neurons)} neuron..."
    )
    with torch.no_grad():

        last_real_data = torch.tensor(
            dm.X_val[-dm.hparams.window:], dtype=torch.float32
        ).to(device)


        mu, logvar = vae.encode(last_real_data.view(-1, dm.n_features_input))
        z_current = vae.reparameterize(
            mu, logvar).view(1, dm.hparams.window, -1)


        imagined_z_sequence = []
        for _ in range(50):
            pi, sigma, mu = mdn_rnn(z_current)
            z_next = sample_from_gmm(pi, sigma, mu)
            imagined_z_sequence.append(z_next)

            z_current = torch.cat(
                [z_current[:, 1:, :], z_next.unsqueeze(1)], dim=1)


        imagined_features = vae.decode(torch.cat(imagined_z_sequence, dim=0))

        simulated_return = torch.mean(imagined_features[:, : dm.n_targets])


    for neuron in experimental_neurons:


        outcome = neuron["outcome"]
        confidence = neuron["confidence"]
        new_status = neuron["status"]

        is_dream_positive = simulated_return > 0.001
        is_dream_negative = simulated_return < -0.001

        if (outcome == "positive" and is_dream_positive) or (
            outcome == "negative" and is_dream_negative
        ):
            confidence = min(1.0, confidence + 0.1)
            if confidence > 0.95:
                new_status = "validated"
        elif (outcome == "positive" and is_dream_negative) or (
            outcome == "negative" and is_dream_positive
        ):
            confidence = max(0.0, confidence - 0.1)
            if confidence < 0.1:
                new_status = "rejected"

        nsmm.update_neuron_evaluation(neuron["id"], confidence, new_status)

    logger.info(
        "🌌 [Virtual Sandbox] Siklus simulasi dan validasi neuron selesai.")


def run_world_model_planning_cycle(
    high_level_goal: str,
    project_id: str,
    initial_hparams: dict,
    api_pool: "DistributedAIPool",
    brain: "Brain",
    **kwargs
):
    """
    Menjalankan siklus perencanaan strategis. AI 'bermimpi' tentang masa depan
    menggunakan World Model (VAE+MDN-RNN) untuk memilih rencana terbaik dari
    beberapa kandidat yang dihasilkan LLM sebelum eksekusi nyata.
    """
    logger.info("\n" + "="*80)
    logger.info(
        "=== 🌌 MEMULAI SIKLUS PERENCANAAN WORLD MODEL (DREAM PLANNING) 🌌 ===")
    logger.info("="*80)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        logger.info(
            "  -> Tahap 1/5: Memuat 'Mata' (VAE) dan 'Otak Imajinasi' (MDN-RNN)...")
        vae_path = get_path(project_id, "checkpoint_dir") /            f"world_model_vae.pth"
        mdn_path = get_path(project_id, "checkpoint_dir") /            f"world_model_mdn_rnn.pth"

        if not vae_path.exists() or not mdn_path.exists():
            logger.error(
                "  -> GAGAL: File World Model (VAE/MDN-RNN) tidak ditemukan. Jalankan 'run_world_model_training' terlebih dahulu.")

            Path("best_plan.txt").write_text(
                "PLANNING_FAILED: World Model not found.")
            return


        dm = AlphaDataModule(initial_hparams, None, None,
                             None, None, None, brain)
        dm.setup(stage="fit")
        n_features = dm.n_features_input

        vae = VAE(input_dim=n_features, latent_dim=32).to(device)
        mdn_rnn = MDN_RNN(latent_dim=32).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        mdn_rnn.load_state_dict(torch.load(mdn_path, map_location=device))
        vae.eval()
        mdn_rnn.eval()


        logger.info("  -> Tahap 2/5: Menganalisis kondisi pasar saat ini...")
        last_real_data = torch.tensor(
            dm.X_val[-dm.hparams.window:], dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, logvar = vae.encode(last_real_data.view(-1, n_features))
            z_current = vae.reparameterize(
                mu, logvar).view(1, dm.hparams.window, -1)


        logger.info(
            f"  -> Tahap 3/5: Meminta 'Ahli Strategi AI' untuk 3 opsi rencana mencapai tujuan: '{high_level_goal}'")

        class CandidatePlans(BaseModel):
            plans: list[str] = Field(
                description="Daftar 3 rencana strategis yang berbeda dan dapat dijalankan.")

        planner_prompt = f"Berikan 3 rencana strategis yang berbeda untuk mencapai tujuan: '{high_level_goal}'"
        plans_dict = api_pool.call_gemini_with_tool(
            planner_prompt, "supervisor", CandidatePlans)
        candidate_plans = plans_dict.get('plans', [])

        if not candidate_plans:
            logger.error(
                "  -> GAGAL: Ahli Strategi AI tidak memberikan rencana. Siklus berhenti.")
            Path("best_plan.txt").write_text(
                "PLANNING_FAILED: No candidate plans generated.")
            return

        logger.info(f"     -> Ditemukan {len(candidate_plans)} opsi rencana.")


        logger.info(
            "  -> Tahap 4/5: 'Bermimpi' tentang kondisi masa depan yang paling mungkin terjadi...")
        with torch.no_grad():
            pi, sigma, mu = mdn_rnn(z_current)
            z_dreamed_future = sample_from_gmm(
                pi, sigma, mu)


        logger.info(
            "  -> Tahap 5/5: Mengevaluasi setiap rencana dalam simulasi imajinatif...")
        best_plan = None
        highest_similarity = -1.0

        for i, plan_text in enumerate(candidate_plans):

            simulation_prompt = f"Jika sebuah agen AI menjalankan rencana '{plan_text}', deskripsikan secara singkat dan objektif kondisi pasar yang paling mungkin terjadi setelahnya."
            imagined_outcome_text = api_pool.call_gemini_for_text(
                simulation_prompt, "experimentalist")


            imagined_outcome_vector = brain.embed_model.encode(
                imagined_outcome_text, task_type="query")
            imagined_outcome_vector = torch.from_numpy(
                imagined_outcome_vector).to(device)


            similarity = F.cosine_similarity(
                z_dreamed_future.squeeze(0), imagined_outcome_vector).item()
            logger.info(
                f"     - Rencana #{i+1} | Kecocokan dengan 'Mimpi': {similarity:.2%}")

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_plan = plan_text

        logger.info("\n" + "="*40)
        logger.info(f"🏆 RENCANA TERPILIH (Paling Sesuai Intuisi World Model):")
        logger.info(f"   '{best_plan}'")
        logger.info("="*40 + "\n")


        Path("best_plan.txt").write_text(best_plan)

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Perencanaan World Model: {e}", exc_info=True)
        Path("best_plan.txt").write_text(f"PLANNING_FAILED: {e}")


def ingest_foundational_principles(brain: Brain):
    """Menanamkan pengetahuan dasar sebagai [PRINSIP_DASAR] ke dalam Brain."""
    principles = [

        "Prinsip Stabilitas Pelatihan: Learning rate yang terlalu tinggi pada data yang volatil dapat menyebabkan divergensi loss dan kegagalan model.",
        "Prinsip Overfitting: Model dengan kapasitas terlalu besar (banyak parameter) relatif terhadap jumlah data berisiko menghafal data training dan gagal generalisasi pada data baru. Regularisasi seperti Dropout atau Weight Decay membantu mitigasi.",
        "Prinsip Underfitting: Model dengan kapasitas terlalu kecil tidak akan mampu menangkap pola kompleks dalam data, menghasilkan performa yang buruk baik pada data training maupun validasi.",
        "Prinsip Bias-Variance Tradeoff: Model yang terlalu sederhana memiliki bias tinggi (underfitting), sementara model yang terlalu kompleks memiliki varians tinggi (overfitting). Tujuannya adalah menemukan keseimbangan.",
        "Prinsip Data Distribution Shift: Jika distribusi data saat inferensi berbeda signifikan dari data saat pelatihan, performa model kemungkinan besar akan menurun drastis.",

        "Prinsip Efisiensi Pasar: Pada pasar yang efisien, harga aset mencerminkan semua informasi yang tersedia. Sulit untuk secara konsisten mendapatkan return di atas rata-rata.",
        "Prinsip Mean Reversion: Harga aset dan volatilitas cenderung kembali ke rata-rata jangka panjangnya setelah mengalami pergerakan ekstrem.",
        "Prinsip Momentum: Aset yang telah menunjukkan tren naik (atau turun) dalam beberapa waktu terakhir cenderung melanjutkan tren tersebut dalam jangka pendek.",
        "Prinsip Risk-Return Tradeoff: Potensi return yang lebih tinggi selalu datang dengan risiko yang lebih tinggi.",
        "Prinsip Psikologi Pasar: Keputusan investor seringkali dipengaruhi oleh emosi seperti ketakutan (fear) dan keserakahan (greed), yang dapat menyebabkan pergerakan harga yang tidak rasional.",

        "Prinsip Kausalitas: Korelasi tidak menyiratkan sebab-akibat. Perubahan pada variabel A harus mendahului perubahan pada variabel B untuk dianggap sebagai penyebab.",
        "Prinsip Confounding: Hubungan yang tampak antara dua variabel dapat sebenarnya disebabkan oleh variabel ketiga (confounder) yang mempengaruhi keduanya.",
        "Prinsip Intervensi (Do-Calculus): Untuk memahami efek kausal sejati, kita harus mensimulasikan 'intervensi' (memaksa sebuah variabel untuk memiliki nilai tertentu) daripada hanya mengamati korelasi pasif.",
    ]
    logger.info(
        f"📚 Menanamkan {len(principles)} prinsip dasar ke dalam Brain...")
    brain.add_chunks(principles, source_name="[PRINSIP_DASAR]")


def run_internal_debate_cycle(nsmm: NSMM, brain: Brain, api_pool: "DistributedAIPool"):
    """
    Menjalankan siklus debat untuk menantang dan meng-evolusi prinsip dasar.
    """
    logger.info("🏛️ [Debat Internal] Memulai siklus validasi prinsip dasar...")


    all_principles_text = brain.query("[PRINSIP_DASAR]", k=100)
    if not all_principles_text:
        logger.warning(
            "🏛️ [Debat Internal] Tidak ada prinsip dasar yang ditemukan di Brain."
        )
        return

    principle_to_test = random.choice(all_principles_text)


    validated_neurons = nsmm.get_neurons_for_consolidation(
        status="validated", limit=20)
    if not validated_neurons:
        logger.info(
            "🏛️ [Debat Internal] Tidak ada neuron tervalidasi untuk dijadikan bukti."
        )
        return

    evidence_text = "\n".join(
        [f"- Outcome {n['outcome']}: {n['trigger_reason']}" for n in validated_neurons]
    )


    red_team_prompt = f"""
    Anda adalah 'AI Red Team', tugas Anda adalah menjadi skeptis dan menantang keyakinan yang ada.
    Prinsip Dasar Saat Ini: "{principle_to_test}"
    Bukti dari Pengalaman Terbaru:
    {evidence_text}

    Tugas: Apakah ada bukti kuat yang berkontradiksi dengan prinsip di atas? Jika ya, formulasikan argumen penolakan yang kuat. Jika tidak, jawab "Prinsip masih valid berdasarkan bukti saat ini."
    """
    try:
        challenge = api_pool.call_gemini_for_text(
            red_team_prompt, "experimentalist")


        if "tidak valid" in challenge.lower() or "kontradiksi" in challenge.lower():
            logger.warning(
                f"🏛️ [Debat Internal] Tantangan ditemukan untuk prinsip: '{principle_to_test}'"
            )
            logger.warning(f"   Argumen Red Team: '{challenge}'")

            arbiter_prompt = f"""
            Anda adalah 'Arbiter AI', tugas Anda adalah menengahi debat dan membuat keputusan akhir.
            Prinsip Dasar: "{principle_to_test}"
            Argumen Penolakan (dari Red Team): "{challenge}"
            Bukti yang Diajukan: {evidence_text}

            Keputusan Anda (pilih SATU dan jelaskan):
            1. PERTAHANKAN: Prinsip tetap valid.
            2. AMANDEMEN: Perbaiki teks prinsip untuk mengakomodasi bukti baru.
            3. TOLAK: Prinsip sudah tidak relevan dan harus dihapus.

            Format Jawaban: KEPUTUSAN: [NAMA KEPUTUSAN]\nTEKS BARU: [Teks prinsip yang sudah diamandemen atau alasan penolakan]
            """
            final_decision = api_pool.call_gemini_for_text(
                arbiter_prompt, "supervisor")



            logger.info(
                f"🏛️ [Debat Internal] Keputusan Arbiter: {final_decision}")
        else:
            logger.info(
                f"🏛️ [Debat Internal] Prinsip '{principle_to_test}' lolos validasi."
            )

    except Exception as e:
        logger.error(f"🏛️ [Debat Internal] Error selama siklus debat: {e}")


def _digest_raw_activity_logs(
    nsmm: NSMM, brain: Brain, engine: "AsyncCuriosityEngine"
):
    """
    Sub-proses yang mencerna log mentah DARI DUA SUMBER:
    1. Aktivitas jendela aktif (prioritas rendah).
    2. Aliran event sistem saraf (prioritas tinggi).
    """
    batch_size = 100
    with sqlite3.connect(nsmm.db_path) as conn:

        raw_events_df = pd.read_sql_query(
            f"SELECT event_id, event_type, event_data FROM raw_system_events WHERE is_digested = 0 LIMIT {batch_size}",
            conn,
        )

    if raw_events_df.empty:
        return

    logger.info(
        f"🧠 [Cognitive Digestion] Mengirim {len(raw_events_df)} event sistem mentah ke antrean untuk dicerna...")


    event_summary = []
    for _, row in raw_events_df.iterrows():
        try:
            data = json.loads(row['event_data'])
            if row['event_type'] == 'KEYSTROKE':
                event_summary.append(data.get('char', ''))
            elif row['event_type'] == 'MOUSE_CLICK':
                event_summary.append(
                    f"[CLICK di {data.get('x')},{data.get('y')}]")
            elif 'FILE' in row['event_type']:
                event_summary.append(
                    f"[{row['event_type']}: {Path(data.get('path')).name}]")
        except:
            continue


    digestion_context = "".join(event_summary).replace('[', '\n[').strip()

    pydantic_schema_str = json.dumps(
        DigestedActivity.model_json_schema(), indent=2)
    prompt = f"""
    You are a cognitive psychologist AI. Analyze the following raw stream of user actions (keystrokes, clicks, file operations) and infer the single, most likely high-level activity the user is performing.

    Schema for your answer:
    {pydantic_schema_str}

    Raw Action Stream:
    ---
    {digestion_context}
    ---

    Your JSON response based on the MOST LIKELY activity:
    """


    request_id = engine.ask_async(
        question_text=prompt,
        agent_key="supervisor",
        response_model=DigestedActivity
    )


    time.sleep(10)
    answer = engine.get_answer(request_id)

    if answer and "error" not in answer:
        try:
            digested_info = DigestedActivity(**answer)

            brain.dkg.add_digested_activity(digested_info)


            with sqlite3.connect(nsmm.db_path) as conn:
                ids_to_mark = tuple(raw_events_df['event_id'].tolist())
                conn.execute(
                    f"UPDATE raw_system_events SET is_digested = 1 WHERE event_id IN {ids_to_mark}"
                )
                conn.commit()
            logger.info("✅ Event sistem mentah berhasil dicerna dan ditandai.")
        except Exception as e:
            logger.error(f"Gagal memproses jawaban pencernaan: {e}")


def run_emotional_synthesis_cycle(
    nsmm: NSMM, brain: Brain, alchemist: EmotionalAlchemist
):
    """Menjalankan satu siklus untuk mencoba menciptakan satu emosi kompleks baru."""
    logger.info(
        "⚗️ [Emotional Alchemist] Memulai siklus sintesis emosi baru...")


    all_emotion_nodes = {
        node_id: data
        for node_id, data in brain.dkg.nodes.items()
        if data.get("type") in ["BaseEmotion", "ComplexEmotion"]
    }

    if len(all_emotion_nodes) < 2:
        logger.info(
            "   -> Tidak cukup emosi untuk dikombinasikan. Siklus dilewati.")
        return


    node_id1, node_id2 = random.sample(list(all_emotion_nodes.keys()), 2)
    emotion1 = all_emotion_nodes[node_id1]
    emotion2 = all_emotion_nodes[node_id2]


    analysis_result = alchemist.synthesize_new_emotion(emotion1, emotion2)

    if analysis_result:

        new_emotion_name = analysis_result.new_emotion_name
        new_emotion_id = new_emotion_name.replace(" ", "_").upper()

        if new_emotion_id not in brain.dkg.nodes:
            brain.dkg.add_node(
                node_id=new_emotion_id,
                node_type="ComplexEmotion",
                layer="Cognitive",
                name=new_emotion_name,
                emoji=analysis_result.combined_emoji,
                description=analysis_result.description,
                utility=analysis_result.utility,
                example_text=analysis_result.example_text_context,
                example_visual=analysis_result.example_visual_context,
                source="Synthesis",
            )
            brain.dkg.add_edge(node_id1, new_emotion_id, "contributes_to")
            brain.dkg.add_edge(node_id2, new_emotion_id, "contributes_to")
            logger.info(
                f"✨ [Neurogenesis Emosional] Emosi baru tercipta dan disimpan: '{new_emotion_name}'"
            )
        else:
            logger.info(
                f"   -> Emosi '{new_emotion_name}' sudah ada. Siklus selesai.")


def run_proactive_metamorphosis_subroutine(
    nsmm: NSMM,
    api_pool: "DistributedAIPool",
    governor: "CognitiveGovernor",

):
    """
    Sub-rutin otonom yang secara proaktif 'iseng' mencoba memutasi arsitektur
    selama waktu idle untuk eksplorasi dan penemuan.
    """
    logger.info(
        "🦋 [Metamorphosis Proaktif] Memulai siklus eksplorasi arsitektur...")


    try:
        architect = ArchitectAI(api_pool)

        model_file_path = Path(__file__).resolve()
        existing_code = model_file_path.read_text(encoding="utf-8")


        exploratory_prompt = (
            "Berdasarkan kode arsitektur yang ada, usulkan SATU perubahan kecil yang bersifat "
            "eksperimental dan spekulatif. Tujuannya adalah untuk eksplorasi, bukan untuk memperbaiki "
            "masalah yang ada. Contoh: mencoba lapisan Conformer di lokasi baru, atau mengganti "
            "satu mekanisme atensi dengan yang lain. Jelaskan secara singkat mengapa mutasi ini "
            "mungkin menarik untuk dicoba."
        )


        proposal = architect.design_new_architecture(
            exploratory_prompt, existing_code)

        if not proposal:
            logger.warning(
                "  -> ArchitectAI tidak menghasilkan proposal mutasi. Siklus dihentikan.")
            return

    except Exception as e:
        logger.error(
            f"  -> Gagal pada tahap ideasi arsitektur: {e}", exc_info=True)
        return



    try:
        with sqlite3.connect(nsmm.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO architectural_hypotheses 
                (module_name, module_code, integration_plan, updated_forward_method, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    proposal.module_name,
                    proposal.module_code,
                    proposal.integration_plan,
                    proposal.updated_forward_method,
                    'proposed_for_sandbox'
                )
            )
            hypothesis_id = cursor.lastrowid
            conn.commit()
        logger.info(
            f"  -> Hipotesis arsitektur baru '{proposal.module_name}' (ID: {hypothesis_id}) disimpan ke NSMM.")
    except Exception as e:
        logger.error(
            f"  -> Gagal menyimpan hipotesis arsitektur ke NSMM: {e}", exc_info=True)
        return



    try:

        sandbox_hparams = {

            "selected_tickers": ["NVDA", "GOOGL"],
            "data_path": str(get_path(None, "data"))
        }



        is_sandbox_success, sandbox_loss = run_exploratory_metamorphosis(
            project_id=f"metamorphosis_sandbox_{hypothesis_id}",
            initial_hparams=sandbox_hparams,
            governor=governor,
            api_pool=api_pool,
            nsmm=nsmm,
            df_raw=None
        )

    except Exception as e:
        logger.error(
            f"  -> Gagal total saat menjalankan sandbox virtual: {e}", exc_info=True)
        is_sandbox_success = False
        sandbox_loss = float('inf')


    try:
        final_status = 'validated_in_sandbox' if is_sandbox_success else 'failed_in_sandbox'
        with sqlite3.connect(nsmm.db_path) as conn:
            conn.execute(
                "UPDATE architectural_hypotheses SET status = ?, sandbox_performance = ? WHERE id = ?",
                (final_status, sandbox_loss, hypothesis_id)
            )
            conn.commit()

        if is_sandbox_success:
            logger.info(
                f"  -> ✅ Hipotesis ID {hypothesis_id} lolos validasi sandbox dengan loss {sandbox_loss:.4f}.")
            governor.log_event("PROACTIVE_METAMORPHOSIS_SUCCESS", {
                               "hypothesis_id": hypothesis_id, "loss": sandbox_loss})
        else:
            logger.warning(
                f"  -> ❌ Hipotesis ID {hypothesis_id} gagal dalam validasi sandbox.")
            governor.log_event("PROACTIVE_METAMORPHOSIS_FAILURE", {
                               "hypothesis_id": hypothesis_id})

    except Exception as e:
        logger.error(
            f"  -> Gagal memperbarui status hipotesis di NSMM: {e}", exc_info=True)


def background_cognition_worker(
    nsmm: "NSMM",
    brain: "Brain",
    api_pool: "DistributedAIPool",
    embedding_model: "APIEmbedder",
    together_api_keys: dict,
    emotional_alchemist: "EmotionalAlchemist",
    scheduler: "CognitiveScheduler",
    stop_event: threading.Event,
    engine: "AsyncCuriosityEngine",
    proactive_message_queue: queue.Queue,
    governor: "CognitiveGovernor",
    initial_wait_minutes: float = 0.16,
    cycle_interval_minutes: int = 2,
):
    """
    Otak Bawah Sadar v9.2: Terintegrasi dengan Metamorfosis Proaktif.
    """
    logger.info(
        f"⚙️ [Cognitive Core v9.2] Proses kognisi latar belakang dimulai. Siklus pertama dalam {int(initial_wait_minutes * 60)} detik."
    )
    time.sleep(initial_wait_minutes * 60)
    root_project_path = Path(__file__).resolve().parents[3]

    while not stop_event.is_set():
        try:

            current_mode = SHARED_STATE.get("activity_mode", "SINAU_DIEM")
            if current_mode == "GASPOL":
                logger.info(
                    "⚙️ [Cognitive Core] Mode GASPOL aktif. Menunda tugas latar belakang untuk memberi prioritas pada agen otonom."
                )
                stop_event.wait(timeout=cycle_interval_minutes * 60)
                continue



            scheduler.recharge_energy()
            logger.info(
                f"⚙️ [Cognitive Core] Memulai siklus kognitif. Stamina saat ini: {scheduler.cognitive_energy:.0f}%"
            )


            if scheduler.get_go_ahead(task_cost=20):
                logger.info(
                    "--- [Sub-rutin] Memulai IQRO & Pencernaan Log ---")


                scheduler.spend_energy(20)


            if scheduler.get_go_ahead(task_cost=25):
                logger.info(
                    "--- [Sub-rutin] Memulai Konsolidasi, Debat, & Pembelajaran ---")
                new_wisdom_count = run_memory_consolidation_cycle(
                    nsmm, brain, api_pool, embedding_model)
                run_internal_debate_cycle(nsmm, brain, api_pool)
                run_autonomous_learning_cycle(nsmm, brain, api_pool)
                scheduler.spend_energy(25)

                if new_wisdom_count > 1 and scheduler.get_go_ahead(task_cost=15):
                    logger.info(
                        "🧬 [Growth Hormone] Terdeteksi pertumbuhan signifikan. Memulai siklus meditasi & kreasi...")
                    scheduler.spend_energy(15)


            if scheduler.get_go_ahead(task_cost=15):
                logger.info("--- [Sub-rutin] Memulai Bantuan Proaktif ---")
                run_proactive_assistance_subroutine(
                    nsmm, brain, api_pool, proactive_message_queue, governor
                )
                scheduler.spend_energy(15)


            if emotional_alchemist and random.random() < 0.25:
                if scheduler.get_go_ahead(task_cost=30):
                    logger.info("--- [Sub-rutin] Memulai Sintesis Emosi ---")
                    run_emotional_synthesis_cycle(
                        nsmm, brain, emotional_alchemist)
                    scheduler.spend_energy(30)


            if scheduler.get_go_ahead(task_cost=40):
                logger.info("--- [Sub-rutin] Memulai Evolusi Struktur QTC ---")
                try:

                    logger.info(
                        "🧬 [QTC Evolution] Satu siklus pertumbuhan struktur otak selesai.")
                except Exception as e:
                    logger.error(f"Gagal menjalankan evolusi QTC: {e}")
                scheduler.spend_energy(40)


            if scheduler.get_go_ahead(task_cost=10):
                logger.info(
                    "--- [Sub-rutin] Mengecek Kondisi untuk Kagebunshin ---")
                spawn_bunshin_workers(max_clones=5)
                scheduler.spend_energy(10)




            if random.random() < 0.1 and scheduler.get_go_ahead(task_cost=50):
                logger.info(
                    "--- [Sub-rutin] Memulai Metamorfosis Proaktif (Eksperimental) ---")
                run_proactive_metamorphosis_subroutine(
                    nsmm=nsmm,
                    api_pool=api_pool,
                    governor=governor
                )
                scheduler.spend_energy(50)

            logger.info(
                f"⚙️ [Cognitive Core] Siklus kognitif selesai. Istirahat selama {cycle_interval_minutes} menit."
            )
            stop_event.wait(timeout=cycle_interval_minutes * 60)

        except Exception as e:
            logger.error(
                f"⚙️ [Cognitive Core] Error pada background worker: {e}", exc_info=True
            )
            stop_event.wait(timeout=10 * 60)


def run_proactive_assistance_subroutine(
    nsmm: NSMM,
    brain: Brain,
    api_pool: "DistributedAIPool",
    proactive_message_queue: queue.Queue,
    governor: "CognitiveGovernor"
):
    """
    Sub-rutin yang menganalisis aktivitas pengguna dan secara proaktif menawarkan bantuan.
    Ini adalah implementasi terisolasi dari Simbiosis Sejati.
    """
    logger.info(
        "🧐 [Proactive Brain] Menganalisis aktivitas pengguna untuk potensi bantuan...")



    user_nodes = [n for n, d in brain.dkg.nodes.items()
                  if d.get('type') == 'User']
    if not user_nodes:
        return


    user_edges = sorted(
        [e for e in brain.dkg.edges if e[0] == user_nodes[0]
            and e[2].get('relationship') == 'consumed'],
        key=lambda x: x[2].get('timestamp', ''),
        reverse=True
    )[:5]

    if len(user_edges) < 3:
        return


    recent_topics = []
    for _, target_node, _ in user_edges:

        topic_edges = [e for e in brain.dkg.edges if e[0] ==
                       target_node and brain.dkg.nodes.get(e[1], {}).get('type') == 'Topic']
        if topic_edges:
            recent_topics.append(
                brain.dkg.nodes[topic_edges[0][1]]['attributes']['name'])

    if not recent_topics:
        return


    most_common_topic = max(set(recent_topics), key=recent_topics.count)
    logger.info(f"    -> Topik dominan terdeteksi: '{most_common_topic}'")




    knowledge_query = f"What are advanced, non-obvious, or recently updated concepts related to {most_common_topic}?"
    relevant_knowledge = brain.query(knowledge_query, k=1)

    if not relevant_knowledge:
        return



    message_crafting_prompt = f"""
    Anda adalah asisten AI yang proaktif dan sangat membantu.
    Konteks:
    - Pengguna tampaknya sedang sangat fokus pada topik: "{most_common_topic}".
    - Anda telah menemukan informasi internal yang relevan dan mungkin baru bagi pengguna: "{relevant_knowledge[0]}".

    Tugas:
    Buat satu pesan singkat dan ramah untuk menawarkan bantuan kepada pengguna.
    Contoh: "Saya melihat Anda sedang mendalami [topik]. Saya menemukan beberapa informasi lanjutan tentang [konsep baru] yang mungkin menarik. Apakah Anda ingin saya jelaskan lebih lanjut?"
    """
    try:
        proactive_message = api_pool.call_gemini_for_text(
            message_crafting_prompt, "supervisor")


        if proactive_message:
            proactive_message_queue.put(proactive_message)
            logger.info(
                f"  -> 💡 Pesan proaktif dikirim: '{proactive_message}'")

            governor.log_event("PROACTIVE_ASSISTANCE_OFFERED", {
                               "topic": most_common_topic, "suggestion": relevant_knowledge[0]})

    except Exception as e:
        logger.error(f"[Proactive Brain] Gagal merumuskan pesan: {e}")


def get_active_window_info() -> Optional[dict]:
    """Mendapatkan informasi tentang jendela yang sedang aktif, termasuk nama prosesnya."""
    try:

        import win32gui
        import win32process


        hwnd = win32gui.GetForegroundWindow()
        if hwnd:

            _, pid = win32process.GetWindowThreadProcessId(hwnd)


            process_name = psutil.Process(pid).name()


            window_title = gw.getActiveWindow().title

            return {"title": window_title, "app": process_name}
    except Exception as e:


        return None


def read_browser_chat_content(app_name: str, window_title: str) -> Optional[str]:
    """
    Menggunakan pywinauto untuk terhubung ke jendela browser dan mengekstrak teks
    dari elemen chat yang relevan. Ini adalah implementasi Accessibility API.
    """
    try:

        app = Application(backend="uia").connect(process=psutil.Process(
            gw.getWindowsWithTitle(window_title)[0]._hWnd).pid)


        win = app.window(title_re=f".*{re.escape(window_title)}.*")




        all_texts = [wrapper.element_info.name for wrapper in win.descendants(
            control_type="Text")]


        chat_content = "\n".join(
            text for text in all_texts if text and len(text) > 10)

        if chat_content:
            return chat_content

    except Exception as e:


        logger.debug(
            f"[ChatReader] Gagal membaca konten dari '{window_title}': {e}")

    return None


def ambient_awareness_worker(
    nsmm: NSMM, stop_event: threading.Event, interval_sec: int = 20
):
    """
    Versi canggih: Memantau aktivitas, dan jika mendeteksi chat,
    ia akan membaca isi percakapan menggunakan Accessibility API.
    """
    logger.info(
        "👀 [Awareness v2.0] Mode 'Kepo' Cerdas Aktif (dengan Pembaca Konten)..."
    )
    last_logged_title = ""
    last_logged_content_hash = ""

    while not stop_event.is_set():
        try:
            stop_event.wait(timeout=interval_sec)
            if stop_event.is_set():
                break

            window_info = get_active_window_info()

            if (
                window_info
                and window_info["title"]
                and window_info["title"] != last_logged_title
            ):
                activity_text = f"Pengguna beralih ke aplikasi '{window_info['app']}' dengan judul: '{window_info['title']}'."
                logger.info(f"👀 [Awareness] {activity_text}")
                nsmm.log_user_activity(activity_text)
                last_logged_title = window_info["title"]


            is_chat_window = False
            if window_info and window_info.get("title"):

                if "gemini" in window_info["title"].lower() or "chat" in window_info["title"].lower():
                    is_chat_window = True

            if is_chat_window:
                chat_text = read_browser_chat_content(
                    window_info['app'], window_info['title'])
                if chat_text:

                    current_hash = hashlib.sha256(
                        chat_text.encode()).hexdigest()
                    if current_hash != last_logged_content_hash:
                        logger.info(
                            f"👀 [Awareness] Percakapan baru terdeteksi. Mencatat transkrip...")

                        nsmm.log_user_activity(
                            f"Transkrip percakapan saat ini:\n---\n{chat_text}\n---")
                        last_logged_content_hash = current_hash


        except Exception as e:
            logger.error(f"Error di dalam awareness worker: {e}")
            time.sleep(60)

    logger.info("👀 [Awareness] Mode 'Kepo' Dihentikan.")


def display_dkg_knowledge(brain: Brain):
    """Mencetak isi dari Dynamic Knowledge Graph (DKG) yang ada di memori."""
    logger.info("\n" + "=" * 25 + " ISI PENGETAHUAN DKG SAAT INI " + "=" * 25)

    dkg = brain.dkg
    if not dkg.nodes:
        logger.info("DKG masih kosong. Belum ada pengetahuan yang dicerna.")
        return

    logger.info(f"\n--- NODES ({len(dkg.nodes)}) ---")
    for node_id, data in dkg.nodes.items():
        logger.info(
            f"  - ID: {node_id:<40} | Tipe: {data.get('type', 'N/A'):<15} | Layer: {data.get('layer', 'N/A')}"
        )

    logger.info(f"\n--- EDGES ({len(dkg.edges)}) ---")
    for source, target, data in dkg.edges:
        logger.info(
            f"  - {source} --[{data.get('relationship', 'N/A')}]--> {target}")

    logger.info("=" * 75 + "\n")


def ingest_base_emotions(brain: Brain):
    """Menanamkan 9 emosi dasar dari model Aardiiiiy ke DKG sebagai fondasi."""
    logger.info("🌱 Menanamkan emosi dasar ke dalam Dynamic Knowledge Graph...")
    base_emotions = {
        "marah": "😠",
        "takut": "😨",
        "senang": "😄",
        "cinta": "❤️",
        "sedih": "😥",
        "terkejut": "😮",
        "jijik": "🤢",
        "trust": "🤝",
        "anticipation": "🤔",
    }

    nodes_added = 0
    for name, emoji in base_emotions.items():
        node_id = name.upper()
        if node_id not in brain.dkg.nodes:
            brain.dkg.add_node(
                node_id=node_id,
                node_type="BaseEmotion",
                layer="Cognitive",
                name=name,
                emoji=emoji,
                source="Aardiiiiy",
            )
            nodes_added += 1

    if nodes_added > 0:
        logger.info(
            f"✅ {nodes_added} emosi dasar berhasil ditambahkan ke DKG.")
    else:
        logger.info("✅ Emosi dasar sudah ada di DKG.")


def _create_low_intensity_params(original_params: dict) -> dict:
    """Membuat versi parameter yang lebih 'ringan' untuk MODE_SINAU_DIEM."""
    low_params = original_params.copy()
    low_params["max_epochs"] = max(
        1, int(original_params.get("max_epochs", 10) * 0.2)
    )
    low_params["batch_size"] = max(
        16, int(original_params.get("batch_size", 64) * 0.5)
    )
    low_params["top_k_features"] = max(
        20, int(original_params.get("top_k_features", 100) * 0.5)
    )
    logger.warning(
        f"  -> Menjalankan tugas dalam MODE_SINAU_DIEM (intensitas rendah). Epoch: {low_params['max_epochs']}, Batch: {low_params['batch_size']}"
    )
    return low_params


def run_command_server_thread_main(grok_api_key, nsmm, brain, embedding_model):
    logger.info("🚀 Meluncurkan Pusat Komando (Dash UI) di http://127.0.0.1:5000")


    if not grok_api_key:
        logger.error(
            "API Key untuk Grok tidak ditemukan. Antarmuka chat tidak dapat dimulai.")
        return



    grok_chat_agent = GrokLLM(api_key=grok_api_key,
                              model_name="grok-3-mini")



    chat_engine = GeminiChatEngine(

        api_pool=None,
        memory=ChatMemoryManager(),
        nsmm=nsmm,
        brain=brain,
        embedding_model=embedding_model,

        direct_chat_agent=grok_chat_agent
    )


    app = create_flask_dash_app(chat_engine, PROACTIVE_MESSAGE_QUEUE)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def execute_realtime_patch_and_retry(decision_data, **kwargs):
    """
    Menerapkan patch kode, memuat ulang modul, memuat checkpoint,
    dan mencoba ulang tugas yang gagal.
    """
    logger.warning("⚡ [Flashback] Memulai protokol patch & retry real-time...")
    file_path_str = decision_data['request']['file_path']
    file_path = Path(file_path_str)
    proposal = decision_data['request']['proposal']

    try:

        original_code_content = file_path.read_text(encoding='utf-8')
        patched_code = original_code_content.replace(
            proposal['original_code_to_find'],
            proposal['suggested_code_to_replace']
        )
        file_path.write_text(patched_code, encoding='utf-8')
        logger.info(f"File {file_path.name} telah di-patch.")




        module_name = 'src.models.model_alpha.alpha'
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            logger.info(f"Modul {module_name} berhasil dimuat ulang.")



        logger.info("Mencoba ulang tugas dengan kode yang sudah diperbaiki...")




        logger.info("✅ [Flashback] Uji coba real-time BERHASIL!")



        return True

    except Exception as e:
        logger.error(
            f"❌ [Flashback] Uji coba real-time GAGAL: {e}", exc_info=True)

        file_path.write_text(original_code_content, encoding='utf-8')
        logger.warning(f"Perubahan pada {file_path.name} telah dikembalikan.")
        return False


def run_forward_forward_training_cycle(
    project_id: str,
    initial_hparams: dict,
    **kwargs
):
    """
    Menjalankan siklus pelatihan mandiri menggunakan Forward-Forward Algorithm
    untuk menghasilkan sebuah blok fitur (feature extractor) yang telah dilatih.
    """
    logger.info("\n" + "="*80)
    logger.info(
        "=== 🧠⚡️ MEMULAI SIKLUS PELATIHAN ALTERNATIF (FORWARD-FORWARD) 🧠⚡️ ===")
    logger.info("="*80)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        logger.info("  -> Memuat dan mempersiapkan data...")

        dm = AlphaDataModule(initial_hparams, None, None,
                             None, None, None, None)
        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()


        logger.info(
            "  -> Menginisialisasi model FF-Net dan optimizer per-lapisan...")
        ff_model = FF_Model(input_dim=dm.n_features_input).to(device)
        layer_optimizers = [
            torch.optim.Adam(layer.parameters(), lr=0.03) for layer in ff_model.layers
        ]


        max_epochs = 10
        for epoch in range(max_epochs):
            ff_model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"FF Epoch {epoch+1}/{max_epochs}", leave=False):


                x_positive = batch[1].to(device)


                x_negative = x_positive[torch.randperm(x_positive.size(0))]


                layer_input_pos = x_positive
                layer_input_neg = x_negative

                for i, layer in enumerate(ff_model.layers):

                    loss = train_layer_forward_forward(
                        layer, layer_input_pos, layer_input_neg, layer_optimizers[i], threshold=1.5
                    )
                    total_loss += loss



                    layer_input_pos = layer(layer_input_pos).detach()
                    layer_input_neg = layer(layer_input_neg).detach()

            logger.info(
                f"  -> Epoch {epoch+1}/{max_epochs} | Rata-rata Loss: {total_loss / len(train_loader):.4f}")


        save_path = get_path(project_id, "checkpoint_dir") /            f"ff_trained_block_{project_id}.pth"
        torch.save(ff_model.state_dict(), save_path)
        logger.info(
            f"✅ Blok fitur Forward-Forward yang telah dilatih berhasil disimpan di: {save_path}")
        logger.info(
            "   (Blok ini sekarang dapat diintegrasikan ke dalam arsitektur utama di siklus evolusi mendatang)")

    except Exception as e:
        logger.error(
            f"FATAL ERROR dalam Siklus Pelatihan Forward-Forward: {e}", exc_info=True)


def formulate_new_autonomous_goal(nsmm: NSMM, brain: Brain, governor: CognitiveGovernor, api_pool: "DistributedAIPool"):
    """
    AI merefleksikan dirinya sendiri dan aktivitas pengguna untuk menciptakan tujuan belajarnya sendiri.
    """
    logger.info(
        "🤔 [Rasa Penasaran] AI sedang merenung untuk menentukan tujuan belajar berikutnya...")


    self_reflection = governor.generate_self_reflection_report()

    with sqlite3.connect(nsmm.db_path) as conn:
        recent_activity_df = pd.read_sql_query(
            "SELECT activity_description FROM user_activity_log ORDER BY timestamp DESC LIMIT 10", conn
        )
    recent_activity = recent_activity_df.to_string(index=False)

    context_prompt = f"""
    Anda adalah bagian meta-kognisi dari sebuah AI yang memiliki kemampuan belajar mandiri.
    Tugas Anda adalah merumuskan tujuan belajar berikutnya yang paling relevan dan bermanfaat.

    KONTEKS #1: LAPORAN REFLEKSI DIRI TERAKHIR ANDA
    (Ini menunjukkan performa, masalah, dan kesenjangan pengetahuan Anda)
    ---
    {self_reflection}
    ---

    KONTEKS #2: LOG AKTIVITAS PENGGUNA TERAKHIR
    (Ini menunjukkan apa yang sedang dikerjakan atau diminati oleh pengguna)
    ---
    {recent_activity}
    ---

    TUGAS ANDA:
    Berdasarkan DUA konteks di atas, identifikasi area yang paling menjanjikan untuk dipelajari.
    Pilih satu topik yang bisa meningkatkan pemahaman Anda atau berpotensi membantu pengguna.
    Panggil tool `ProposeNewGoal` untuk merumuskan tujuan baru Anda.
    """

    try:

        proposal_dict = api_pool.call_gemini_with_tool(
            prompt=context_prompt,
            agent_name="supervisor",
            tool_schema=ProposeNewGoal
        )
        if proposal_dict:
            proposal = ProposeNewGoal(**proposal_dict)
            nsmm.add_new_goal(
                description=proposal.new_goal_description,
                success=proposal.success_metric,
                failure=proposal.failure_metric,
                reasoning=proposal.reasoning
            )
            return True
    except Exception as e:
        logger.error(
            f"[Rasa Penasaran] Gagal merumuskan tujuan baru: {e}", exc_info=True)

    return False


def verify_step_completion(api_pool: "DistributedAIPool", verification_prompt: str, perception_context: dict) -> bool:
    """
    Menggunakan LLM untuk memverifikasi apakah sebuah langkah tugas telah berhasil diselesaikan
    berdasarkan persepsi layar saat ini.
    """
    logger.info(
        f"🕵️  [Verifikasi] Memeriksa penyelesaian langkah: '{verification_prompt}'")

    prompt = f"""
    Anda adalah modul verifikasi internal dari sebuah AI otonom.
    Tugas Anda adalah menjawab 'YA' atau 'TIDAK' secara tegas berdasarkan bukti dari persepsi layar.

    Kondisi yang Harus Diverifikasi:
    "{verification_prompt}"

    Bukti(Persepsi Layar Saat Ini):
    - Deskripsi Visual: {perception_context.get('visual_description', 'N/A')}
    - Teks dari OCR: {perception_context.get('ocr_text', 'N/A')[:1000]}

    Berdasarkan bukti di atas, apakah kondisi telah terpenuhi? Jawab HANYA dengan 'YA' atau 'TIDAK'.
    """
    try:

        response = api_pool.call_gemini_for_text(prompt, "experimentalist")
        logger.info(f"    -> Jawaban Verifikator: {response}")
        return "ya" in response.lower()
    except Exception as e:
        logger.error(f"Gagal saat verifikasi langkah: {e}")
        return False


def recognize_command_intent(user_text: str, api_pool: "DistributedAIPool") -> CommandIntent:
    """Menggunakan LLM untuk memahami niat dan mengekstrak parameter dari perintah pengguna."""


    available_intents = {
        "ingest_file": "Niat untuk membaca, mempelajari, atau mencerna konten dari sebuah file lokal.",
        "run_analysis": "Niat untuk menjalankan analisis finansial atau tugas pemodelan.",
        "unknown": "Niat tidak dapat diidentifikasi dari perintah."
    }

    prompt = f"""
    Anda adalah sebuah NLU (Natural Language Understanding) engine yang sangat akurat.
    Tugas Anda adalah menganalisis perintah pengguna dan mengklasifikasikannya ke dalam salah satu niat yang tersedia,
    lalu mengekstrak parameter yang relevan.

    Niat yang Tersedia:
    {json.dumps(available_intents, indent=2)}

    Perintah Pengguna:
    ---
    "{user_text}"
    ---

    Aturan:
    - Jika Anda mendeteksi path file, normalisasikan formatnya.
    - Jawab HANYA dengan format JSON yang sesuai dengan skema `CommandIntent`.

    JSON Jawaban Anda:
    """

    try:

        response_dict = api_pool.call_gemini_with_tool(
            prompt=prompt,
            agent_name="supervisor",
            tool_schema=CommandIntent
        )
        return CommandIntent(**response_dict)
    except Exception as e:
        logger.error(f"Gagal mengenali niat: {e}")
        return CommandIntent(intent_name="unknown", parameters={})


def run_direct_command_interface(task_queue: queue.Queue, proactive_message_queue: queue.Queue, stop_event: threading.Event):
    """
    Versi dinamis yang menggunakan NLU untuk memahami perintah pengguna secara fleksibel.
    """
    logger.info(
        "==================================================================")
    logger.info("===       💬 SALURAN KOMUNIKASI DINAMIS AKTIF        ===")
    logger.info(
        "===       Anda bisa memberi perintah natural (mis: 'baca file ...')       ===")
    logger.info(
        "==================================================================")

    while not stop_event.is_set():
        try:

            try:
                message = proactive_message_queue.get_nowait()
                print(f"\nAI > {message}")
            except queue.Empty:
                pass

            user_input = input("Anda > ")


            if user_input.lower() in ['exit', 'quit', 'keluar']:
                logger.warning(
                    "Perintah keluar diterima. Menghentikan sistem...")
                stop_event.set()
                break

            if not user_input.strip():
                continue




            intent = recognize_command_intent(user_input, api_pool)


            if intent.intent_name == "ingest_file":
                file_path = intent.parameters.get("file_path")
                if file_path:

                    ingest_specific_file(
                        file_path, brain, api_pool, nsmm, async_engine)
                else:
                    print(
                        "⚠️ Niat 'baca file' terdeteksi, tapi path filenya tidak ditemukan dalam perintah Anda.")

            elif intent.intent_name == "run_analysis":

                task_queue.put({"description": user_input, "priority": 1})
                print(
                    f"Niat analisis terdeteksi. Perintah '{user_input}' telah dikirim ke otak AI.")

            else:
                print(
                    "❓ Maaf, saya tidak begitu mengerti perintah tersebut. Bisa coba gunakan kalimat lain?")


        except (KeyboardInterrupt, EOFError):
            logger.warning("Interupsi diterima. Menghentikan sistem...")
            stop_event.set()
            break


class DynamicTaskExecutor:
    """
    Menemukan dan mengeksekusi fungsi secara dinamis berdasarkan nama,
    dengan cerdas memetakan parameter yang tersedia dari berbagai sumber.
    """

    def __init__(self, system_components: dict):

        self.system_components = system_components

        self.signature_cache = {}

    def _get_function_signature(self, func):
        if func.__name__ not in self.signature_cache:
            self.signature_cache[func.__name__] = inspect.signature(
                func).parameters
        return self.signature_cache[func.__name__]

    def execute(self, task_name: str, params: dict):
        """
        Mencari fungsi berdasarkan nama dan mengeksekusinya.

        Args:
            task_name (str): Nama fungsi yang akan dipanggil.
            params (dict): Parameter yang diberikan oleh MasterPlannerAI.
        """
        logger.info(
            f"🦾 [Executor Dinamis] Mencoba menjalankan tugas: '{task_name}'")


        target_function = globals().get(task_name)
        if not target_function or not callable(target_function):
            logger.error(
                f"  -> GAGAL: Tugas '{task_name}' tidak ditemukan atau bukan fungsi yang bisa dipanggil.")
            return


        try:
            expected_params = self._get_function_signature(target_function)
            args_to_pass = {}


            for param_name, param_spec in expected_params.items():


                if param_spec.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                if param_name in params:

                    args_to_pass[param_name] = params[param_name]
                elif param_name in self.system_components:

                    args_to_pass[param_name] = self.system_components[param_name]
                elif param_spec.default is not inspect.Parameter.empty:

                    continue
                else:

                    logger.error(
                        f"  -> GAGAL: Parameter wajib '{param_name}' untuk tugas '{task_name}' tidak ditemukan.")
                    return


            logger.info(
                f"  -> Menjalankan {task_name} dengan parameter: {list(args_to_pass.keys())}")
            target_function(**args_to_pass)
            logger.info(f"  -> ✅ Tugas '{task_name}' berhasil diselesaikan.")

        except Exception as e:
            logger.critical(
                f"  -> FATAL ERROR saat mengeksekusi tugas '{task_name}': {e}", exc_info=True)


class ArcReactor:
    """
    Jantung Kognitif dan Orkestrator Utama Sistem.
    Bertanggung jawab atas manajemen status, alokasi sumber daya (energi),
    dan alur eksekusi semua komponen sistem AI sesuai dengan Prime Directive.
    """

    def __init__(self, stop_event: threading.Event, **system_components):
        """
        Mengkristalisasi semua komponen sistem ke dalam satu inti terpusat.

        Args:
            **system_components: Kamus yang berisi semua instance objek
                                 yang dibutuhkan sistem untuk beroperasi.
        """
        self.state = "BOOTING"
        self.components = system_components
        self.shutdown_signal = stop_event


        self.auditor = self.components.get('auditor')
        self.api_pool = self.components.get('api_pool')
        self.brain = self.components.get('brain')
        self.nsmm = self.components.get('nsmm')
        self.governor = self.components.get('governor')
        self.feedback_manager = self.components.get('feedback_manager')
        self.df_raw = self.components.get('df_raw')



        self.TASK_DISPATCHER = {
            "run_pre_train_defensive_systems": pre_train_defensive_systems,

            "run_pre_train": self.run_pre_train_wrapper,
            "run_continuous_training": self.run_continuous_training_wrapper,
            "run_strategy_creation_cycle": run_strategy_creation_cycle,
            "run_strategy_backtest": run_strategy_backtest,
            "discover_new_rules_with_ilp": discover_new_rules_with_ilp,
            "run_world_model_training": run_world_model_training,
            "run_world_model_planning_cycle": run_world_model_planning_cycle,
            "run_dreamer_and_surprise_cycle": run_dreamer_and_surprise_cycle,
            "run_autonomous_task_agent": run_autonomous_task_agent,
            "run_exploratory_metamorphosis": run_exploratory_metamorphosis,

            "run_feature_creation_cycle": run_strategy_creation_cycle,
            "predict_and_visualize": predict_and_visualize,
        }

        self.state = "IDLE"
        logger.info(
            "⚡ ARC REACTOR ONLINE. Sistem dalam mode IDLE, siap menerima Prime Directive.")

    def _dispatch_and_execute_task(self, task: dict):
        """
        Inti dari logika Reactor:
        - Menemukan fungsi target berdasarkan nama tugas.
        - Membangun argumen yang dibutuhkan secara cerdas.
        - Menjalankan fungsi dengan proteksi error handling.
        """

        task_name = task.get("task_name")
        params_from_directive = task.get("params", {})


        if task_name not in self.TASK_DISPATCHER:
            logger.error(f"🚨 [Reactor Dispatch] Tugas tidak dikenal: '{task_name}'. Melewati.")
            return

        target_function = self.TASK_DISPATCHER[task_name]


        try:
            expected_params = inspect.signature(target_function).parameters
        except Exception as e:
            logger.error(f"Tidak dapat memeriksa signature untuk {task_name}: {e}")
            return


        args_to_pass = {}
        for param_name, param_spec in expected_params.items():

            if param_spec.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            if param_name in params_from_directive:

                args_to_pass[param_name] = params_from_directive[param_name]
            elif param_name in self.components:

                args_to_pass[param_name] = self.components[param_name]




            elif param_name == 'hparams' and 'initial_hparams' in self.components:
                args_to_pass[param_name] = self.components['initial_hparams']

            elif param_spec.default is not inspect.Parameter.empty:

                continue
            else:

                logger.warning(
                    f"Parameter wajib '{param_name}' untuk tugas '{task_name}' tidak ditemukan. Akan dilewati."
                )





        if params_from_directive:
            args_to_pass['params'] = params_from_directive


        try:
            logger.info(f"⚡ [Reactor] Mengeksekusi: {task_name}...")
            target_function(**args_to_pass)
            logger.info(f"✅ [Reactor] Tugas '{task_name}' selesai.")
        except Exception as e:
            logger.critical(
                f"🚨 [REACTOR CORE] GAGAL saat mengeksekusi tugas '{task_name}': {e}", exc_info=True
            )

            raise e



    def run_mission(self):
        """
        Loop utama yang membaca dan menjalankan misi dari Prime Directive.
        """
        try:

            directive_path_str = r"C:\Users\Msi\Oracle Gem\src\models\model_alpha\prime_directive.json"
            directive_path = Path(directive_path_str)
            with directive_path.open("r", encoding="utf-8") as f:
                mission_data = json.load(f)

            prime_directive = mission_data.get("prime_directive", {})
            mission_description = prime_directive.get(
                "description", "Misi tidak terdefinisi.")
            mission_steps = prime_directive.get("steps", [])

            logger.info("\n" + "="*80)
            logger.info(
                "=== 🎯 PRIME DIRECTIVE DITERIMA OLEH ARC REACTOR 🎯 ===")
            logger.info(f"=== MISI: {mission_description} ===")
            logger.info(f"=== TOTAL LANGKAH: {len(mission_steps)} ===")
            logger.info("="*80)

            self.state = "EXECUTING_DIRECTIVE"


            for i, step in enumerate(mission_steps):
                if self.shutdown_signal.is_set():
                    logger.warning(
                        "Sinyal shutdown diterima. Misi dihentikan.")
                    break

                reasoning = step.get("reasoning", "N/A")
                logger.info("\n" + "-"*80)
                logger.info(
                    f"--- TAHAP MISI {i+1}/{len(mission_steps)}: {step.get('task_name').upper()} ---")
                logger.info(f"--- Alasan: {reasoning} ---")
                logger.info("-"*80)

                self._dispatch_and_execute_task(step)

            if not self.shutdown_signal.is_set():
                logger.info("\n" + "="*80)
                logger.info(
                    "=== ✅ PRIME DIRECTIVE TELAH SELESAI DI EKSEKUSI ✅ ===")
                logger.info(
                    "=== Arc Reactor sekarang memasuki mode siaga otonom. ===")
                logger.info("="*80)
                self.state = "AUTONOMOUS_STANDBY"



            while not self.shutdown_signal.is_set():
                time.sleep(1)

        except FileNotFoundError:
            logger.critical(
                f"🚨 FATAL: File Prime Directive tidak ditemukan di '{directive_path_str}'. Tidak dapat memulai misi.")
        except Exception as e:
            logger.critical(
                f"🚨 [REACTOR CORE FAILURE] Misi gagal total karena error: {e}", exc_info=True)
            self.state = "CRITICAL_FAILURE"
        finally:
            self.shutdown_signal.set()
            logger.info("⚡ ARC REACTOR SHUTDOWN.")

    def run_pre_train_wrapper(self, **kwargs):
        """Wrapper untuk menangani struktur hparams yang kompleks untuk run_pre_train."""
        hparams = self.components['initial_hparams'].copy()
        hparams.update(kwargs.get('params', {}))


        other_components = {
            key: value for key, value in self.components.items() 
            if key != 'initial_hparams'
        }


        run_pre_train(initial_hparams=hparams, **other_components)

    def run_continuous_training_wrapper(self, **kwargs):
        """Wrapper untuk menangani struktur hparams yang kompleks untuk run_continuous_training."""
        hparams = self.components['initial_hparams'].copy()
        hparams.update(kwargs.get('params', {}))


        required_components = {
            "auditor": self.components.get("auditor"),
            "api_pool": self.components.get("api_pool"),
            "gemini_api_config": self.components.get("gemini_api_config"),
            "together_api_keys": self.components.get("together_api_keys"),
            "together_roles": self.components.get("together_roles"),
            "brain": self.components.get("brain"),
            "web_searcher": self.components.get("web_searcher"),
        }


        run_continuous_training(
            initial_hparams=hparams, 
            **required_components
        )

    def shutdown(self):
        """Mengatur sinyal shutdown untuk menghentikan loop utama dan semua proses terkait."""
        logger.warning("Sinyal shutdown diterima oleh Arc Reactor.")
        self.shutdown_signal.set()


if __name__ == "__main__":

    from src.models.model_alpha.key_manager import manage_api_keys
    import queue
    import threading
    import argparse
    import sys
    from pathlib import Path, PurePosixPath
    import pandas as pd
    import os
    import time
    import traceback






    parser = argparse.ArgumentParser(
        description="Oracle Gem - Autonomous AI Financial Agent"
    )

    parser.add_argument(
        "--role", type=str, default="analis", choices=['analis', 'bunshin'],
        help="Menentukan peran instance ini ('analis' atau 'bunshin')."
    )
    parser.add_argument(
        "--target-folder", type=str,
        help="[HANYA UNTUK BUNSHIN] Path ke folder spesifik yang harus dipindai."
    )
    parser.add_argument(
        "--tickers", nargs="+", help="Daftar ticker default jika diperlukan."
    )
    parser.add_argument(
        "--gemini-key", type=str, help="Satu API Key Gemini untuk digunakan di semua peran."
    )
    parser.add_argument(
        "--x-bearer-tokens", nargs="+", help="Satu atau lebih Bearer Token dari API X."
    )
    parser.add_argument(
        "--tavily-key", type=str, help="API Key Tavily untuk pencarian web."
    )
    parser.add_argument(
        "--google-search-key", type=str, help="API Key Google Custom Search."
    )
    parser.add_argument(
        "--google-cse-id", type=str, help="Google Programmable Search Engine ID."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(get_path(None, "data")),
        help="Path to the source data file.",
    )
    parser.add_argument(
        "--is-sandbox-test",
        action="store_true",
        help="Menjalankan skrip dalam mode tes sandbox dengan parameter ringan.",
    )
    args = parser.parse_args()




    if args.role == "bunshin":
        if not args.target_folder:
            sys.exit("Error: Bunshin membutuhkan --target-folder untuk misinya.")

        logger.info(
            f"🌀 [Bunshin] Aktif. Misi: Memindai {Path(args.target_folder).name}")

        try:

            _supremacy_embed_model = APIEmbedder(
                api_key=None,
                model_name=EMBEDDED_CFG["models"]["embedding_api"],
                dim=EMBEDDED_CFG["faiss_dim"],
                use_local_model=True
            )
            brain = Brain(
                index_path=str(get_path(None, "brain.faiss")),
                db_path=str(get_path(None, "brain.sqlite")),
                embed_model_instance=_supremacy_embed_model,
                dim=EMBEDDED_CFG["faiss_dim"],
                api_pool=None,
                together_api_keys={}
            )
            nsmm = NSMM(project_id="global_main")

            run_iqro_protocol_scan(
                root_path=Path(args.target_folder),
                api_pool=None, nsmm=nsmm, brain=brain, engine=None
            )
            logger.info(f"🌀 [Bunshin] Misi selesai. Menghilang.")
        except Exception as e:
            logger.error(f"Bunshin gagal menjalankan misi: {e}", exc_info=True)
        finally:
            sys.exit(0)




    else:
        seed_everything(42, workers=True)

        TASK_QUEUE = queue.Queue()
        PROACTIVE_MESSAGE_QUEUE = queue.Queue()
        DATA_REQUEST_QUEUE = queue.Queue()
        DECISION_QUEUE = queue.Queue()
        stop_event = threading.Event()
        LAST_PROACTIVE_MESSAGE_TIME = [0.0]
        SHARED_STATE = {"activity_mode": "AUTONOMOUS"}


        cognition_thread = None
        nervous_system = None
        system_monitor = None
        reactor_instance = None

        try:



            api_keys = manage_api_keys()
            together_api_keys = api_keys.get("together_api_keys", {})
            gemini_api_keys_list = (args.gemini_key and [
                args.gemini_key]) or api_keys.get("gemini", [])
            grok_api_key = together_api_keys.get("grok_key")
            together_roles_config = api_keys.get("together_roles_config", {})
            x_bearer_tokens = args.x_bearer_tokens or api_keys.get(
                "x_bearer", [])
            tavily_api_key = args.tavily_key or api_keys.get("tavily")
            google_api_key = args.google_search_key or api_keys.get(
                "Google Search")
            google_cse_id = args.google_cse_id or api_keys.get("google_cse_id")

            gemini_api_config = {}


            

            logger.info(
                "Menerapkan hierarki model: Mencari Llama-4-Scout sebagai prioritas utama...")
            scout_key = together_api_keys.get("together_scout")
            if scout_key:
                core_roles = [
                    "tutor", "supervisor", "experimentalist", "advanced_advisor", "ai_code_auditor",
                    "ai_engineer", "json_finalizer", "qwen_giant_planner"
                ]
                for role in core_roles:
                    gemini_api_config[role] = {
                        "key": scout_key,
                        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
                    }
                logger.info(
                    f"✅ {len(core_roles)} peran inti telah dialihkan ke Llama-4-Scout.")


            logger.info("Mengkonfigurasi peran lain dari 'together_roles_config'...")
            for role, config in together_roles_config.items():

                if role in gemini_api_config:
                    continue

                model_name = config.get("primary")
                if not model_name:
                    continue
                

                key_name_to_find = role
                if "qwen" in role: key_name_to_find = "qwen_giant"
                elif "maverick" in role and role not in together_api_keys: key_name_to_find = "maverick"
                elif "exaone" in role: key_name_to_find = "exaone"

                api_key = together_api_keys.get(key_name_to_find)

                if api_key and model_name:
                    gemini_api_config[role] = {"key": api_key, "model": model_name}
                    logger.info(f"PATCH: Peran '{role}' dialihkan ke model '{model_name.split('/')[-1]}'.")
                else:
                    logger.warning(f"Tidak dapat mengkonfigurasi peran '{role}', kunci API '{key_name_to_find}' atau model tidak ditemukan.")


            all_gemini_roles = [
                "tutor", "supervisor", "experimentalist", "advanced_advisor", "ai_code_auditor",
                "ai_engineer", "chat_interactive", "json_finalizer", "supremacy_flash",
            ]
            if gemini_api_keys_list:
                gemini_api_config["gemini_pool"] = {
                    "keys": gemini_api_keys_list,
                    "model": "models/gemini-1.5-flash-latest",
                }
                for role in all_gemini_roles:
                    if role not in gemini_api_config:
                        gemini_api_config[role] = {"pool": "gemini_pool"}
                        logger.info(
                            f"Fallback: Peran '{role}' akan menggunakan Gemini Pool.")






            logger.info(
                "🚀 [SYSTEM BOOT] Memulai inisialisasi semua komponen inti...")
            shared_llm_lock = threading.Lock()
            async_engine = AsyncCuriosityEngine(
                together_api_keys=together_api_keys, llm_lock=shared_llm_lock)
            api_pool = DistributedAIPool(
                gemini_api_config, llm_lock=shared_llm_lock)

            embedding_api_key = together_api_keys.get("embedding_e5")
            if not embedding_api_key:
                raise ValueError("API Key untuk 'embedding_e5' wajib ada.")

            _supremacy_embed_model = APIEmbedder(
                api_key=embedding_api_key,
                model_name=EMBEDDED_CFG["models"]["embedding_api"],
                dim=EMBEDDED_CFG["faiss_dim"],
            )
            vector_encoder = UniversalVectorEncoder(
                _supremacy_embed_model)
            brain = Brain(
                index_path=str(get_path(None, "brain.faiss")),
                db_path=str(get_path(None, "brain.sqlite")),
                embed_model_instance=_supremacy_embed_model,
                dim=EMBEDDED_CFG["faiss_dim"],
                api_pool=api_pool,
                together_api_keys=together_api_keys,
            )
            nsmm = NSMM(project_id="global_main")
            auditor = CriticalAuditor()
            web_searcher = WebSearchManager(
                tavily_api_key, google_api_key, google_cse_id)
            x_sentiment_manager = XSentimentManager(
                bearer_tokens=x_bearer_tokens)
            

            qwen_key_for_alchemist = together_api_keys.get("qwen_giant")
            alchemist = EmotionalAlchemist(
                api_key=qwen_key_for_alchemist) if qwen_key_for_alchemist else None
                
            cognitive_scheduler = CognitiveScheduler(
                shared_state=SHARED_STATE)
            governor = CognitiveGovernor(
                project_id="global_main", api_pool=api_pool)
            master_planner = MasterPlannerAI(
                api_pool=api_pool, together_api_keys=together_api_keys)


            ingest_foundational_principles(brain)
            ingest_base_emotions(brain)
            load_nlp_models()




            run_ingestion_pipeline()
            parquet_path = ensure_parquet_source(args.data_path)
            df_raw = pd.read_parquet(parquet_path)
            available_tickers = sorted(
                {col.split('_')[0] for col in df_raw.columns if '_' in col})




            logger.info(
                "\n--- Mengaktifkan Semua Proses Latar Belakang & Antarmuka ---")
            threading.Thread(target=run_command_server_thread_main, args=(together_api_keys.get(
                "grok_key"), nsmm, brain, _supremacy_embed_model), daemon=True).start()
            threading.Thread(target=data_acquisition_worker, args=(
                DATA_REQUEST_QUEUE, brain, web_searcher), daemon=True).start()
            cognition_thread = threading.Thread(target=background_cognition_worker, args=(
                nsmm, brain, api_pool, _supremacy_embed_model, together_api_keys, alchemist, cognitive_scheduler, stop_event, async_engine, PROACTIVE_MESSAGE_QUEUE, governor), daemon=True)
            cognition_thread.start()

            if os.name == "nt":
                system_monitor = SystemVitalsMonitor(
                    shared_state=SHARED_STATE, stop_event=stop_event, idle_timeout_minutes=15)
                system_monitor.start()

            nervous_system = DigitalNervousSystem(
                nsmm=nsmm, stop_event=stop_event)
            nervous_system.start()




            logger.info(
                "🔬 [SYSTEM BOOT] Mengkristalisasi komponen ke dalam inti Arc Reactor...")
            system_components = {
                "auditor": auditor, "api_pool": api_pool, "together_api_keys": together_api_keys,
                "gemini_api_config": gemini_api_config, "web_searcher": web_searcher, "governor": governor,
                "brain": brain, "nsmm": nsmm, "_supremacy_embed_model": _supremacy_embed_model, "df_raw": df_raw, "async_engine": async_engine,
                "feedback_manager": HumanFeedbackManager(get_path(None, "human_feedback_db")),
                "together_roles": together_roles_config,
                "initial_hparams": {
                    "project_id": "autonomous_session", "data_path": parquet_path,
                    "selected_tickers": args.tickers or available_tickers[:3],
                    "d_model": 128, "n_heads": 8, "n_layers": 2, "window": 60,
                    "horizon": 7, "lr": 0.001, "dropout": 0.2, "weight_decay": 1e-5
                }
            }

            logger.info(
                "⚛️  [SYSTEM BOOT] Membangun dan menyalakan Arc Reactor...")
            reactor_instance = ArcReactor(
                **system_components, stop_event=stop_event)

            logger.info(
                "\n🚀 [AUTONOMOUS CORE] Menyerahkan kendali ke Arc Reactor untuk eksekusi misi...")
            reactor_instance.run_mission()

            logger.info(
                "\n✅ [AUTONOMOUS CORE] Misi utama selesai. Sistem dalam mode siaga (standby).")


            while not stop_event.is_set():
                time.sleep(1)

        except (KeyboardInterrupt, SystemExit):
            logger.warning(
                "Perintah keluar diterima dari pengguna. Menghentikan sistem...")
            stop_event.set()

        except Exception as e:

            logger.critical(
                f"FATAL ERROR PADA SIKLUS UTAMA: {type(e).__name__} - {e}", exc_info=True)

            permission_queue = queue.Queue()
            decision_queue = queue.Queue()

            tb_str = traceback.format_exc()
            tb = e.__traceback__
            last_frame, last_lineno = list(traceback.walk_tb(tb))[-1]
            file_name = last_frame.f_code.co_filename

            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    source_lines = f.readlines()
                problematic_code = source_lines[last_lineno - 1].strip()
            except Exception:
                problematic_code = "Tidak dapat membaca baris kode."

            error_context = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "file_path": file_name,
                "line_number": last_lineno,
                "problematic_code_line": problematic_code,
                "full_traceback_str": tb_str,
            }

            healer = HealerProtocol(
                error_context, nsmm, api_pool, permission_queue)
            healer.together_api_keys = together_api_keys
            healer.start()

            interaction_manager = UserInteractionManager(
                permission_queue, decision_queue)
            interaction_manager.start()

            logger.warning(
                "Menunggu keputusan dari Supervisor Manusia untuk perbaikan...")
            decision_data = decision_queue.get()

            if decision_data and decision_data.get("approved"):
                logger.info("Izin diberikan. Mencoba menerapkan patch...")
                logger.critical(
                    "Patch telah diterapkan. HARAP RESTART SCRIPT SECARA MANUAL untuk melanjutkan dengan kode baru.")
            else:
                logger.error(
                    "Perbaikan ditolak atau gagal. Proses akan dihentikan.")

        finally:

            logger.warning("Memulai proses shutdown yang rapi...")

            if reactor_instance:
                reactor_instance.shutdown()

            if not stop_event.is_set():
                stop_event.set()

            if system_monitor:
                system_monitor.stop()
                system_monitor.join(timeout=5)

            if cognition_thread and cognition_thread.is_alive():
                cognition_thread.join(timeout=5)

            if nervous_system and nervous_system.is_alive():
                nervous_system.join(timeout=5)

            logger.info(
                "Semua proses latar belakang telah dihentikan. Program keluar.")
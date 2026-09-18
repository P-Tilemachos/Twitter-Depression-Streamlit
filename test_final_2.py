# -*- coding: utf-8 -*-
"""
test_final_2.py  (FIXED + DL LOADER v2)
AI Depression Detector (Final Thesis Version — DL-ready)
- TF-IDF + ML models (RF, XGB, LR, SVM, NB)
- Deep Learning text models (CNN, BiLSTM) with a robust loader (.h5 / .keras)
- Tokenizers (.pkl / .json) -> loaded WITHOUT depending on the Keras Tokenizer class
- Per-model auto label alignment (probes) & thresholds
- Ensemble decision + guardrails
- VADER + BERT sentiment (messaging)
- Admin view: per-model details, suicide-risk viz, TF-IDF explain

✅ FIXES / MODS:
1) All ensemble/guard outputs are persisted safely inside st.session_state["last_analysis"]
2) Admin View reads ONLY from st.session_state["last_analysis"] using .get() to avoid NameError
3) Added explicit keys: ensemble_label, final_label, guard_pass, signals_true, vader_label, senti_bucket

✅ DL LOADER v2:
A) Auto-discovery of model files (*.h5 / *.keras) and tokenizers (*.pkl / *.json)
   in the script folder, the working folder and their ./models sub-folders
B) Several load strategies (custom_objects -> plain -> config scrub + weights) for both .h5 and .keras
C) Tokenizer loader that works even if the Keras Tokenizer class is not importable (Keras 2 -> Keras 3 pickles)
D) max_len is read from the model input shape (fallback: 200); own pad_sequences (no extra import)
E) Models are cached with st.cache_resource (no reload on every Streamlit rerun)
F) Every success / failure is logged and shown in Admin View (no more silent `except: continue`)
G) Label-alignment of DL models is recomputed automatically if the model files change
"""

#--------------------------------------Imports--------------------------------------
import os

# Must be set BEFORE tensorflow / keras are imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import json
import math
import random
import pathlib
import pickle
import platform
import smtplib
import socket
import re
import zipfile
from email.mime.text import MIMEText
from collections import Counter
from datetime import datetime

import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# ---------- Optional for public IP ----------
try:
    import requests
except Exception:
    requests = None

# ---------- Page / seed ----------
st.set_page_config(page_title="Depression Detector", page_icon="🧠", layout="wide")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = "0"

# ---------- Admin / email ----------
ADMIN_EMAIL_ALLOWED = "adviewstreamlit@gmail.com"
ADMIN_PASS_ALLOWED  = "Adviewstreamlit@3092025!"
EMAIL_ENABLED       = True
SMTP_HOST           = "smtp.gmail.com"
SMTP_PORT           = 587
SMTP_USERNAME       = ADMIN_EMAIL_ALLOWED
SMTP_PASSWORD       = os.environ.get("SMTP_APP_PASSWORD", "")

# ---------- Constants ----------
PROBE_POS = [
    "i love you", "i feel happy", "i am fine",
    "today is a good day", "i feel great", "i'm grateful", "so excited!"
]
PROBE_NEG = [
    "i want to die", "i hate myself", "kill myself",
    "i feel hopeless", "life is pointless", "i want to end it"
]
SUICIDE_KEYWORDS = {
    "die", "kill", "suicide", "worthless", "hopeless", "tired", "alone",
    "death", "self-harm", "cut", "end", "end it", "no reason to live"
}
SUICIDE_PHRASES = {
    "strong": [
        "i want to die", "want to die", "kill myself", "end my life", "end it all",
        "no reason to live", "don't want to live", "dont want to live",
        "better off dead", "wish i were dead", "i wish i was dead",
        "i want to disappear forever", "i don't want to exist", "i dont want to exist"
    ],
    "medium": [
        "tired of life", "sick of life", "i can't go on", "i cant go on",
        "i can't do this anymore", "i cant do this anymore",
        "i give up", "i'm done with everything", "im done with everything",
        "nothing to live for", "don't see a future", "dont see a future",
        "life has no meaning", "everything has lost its meaning",
        "no point in trying", "empty inside", "completely empty",
        "lost all hope", "no hope left", "better off without me",
        "everyone would be better off without me",
        "no purpose in waking up", "purpose in waking up", "no point in waking up",
        "waiting for the end", "just waiting for the end"
    ],
    "weak": [
        "no strength anymore", "i have no strength anymore", "so tired of everything",
        "exhausted by life", "i feel numb", "i feel empty",
        "i don't care about anything", "i dont care about anything",
        "can't get out of bed", "cant get out of bed",
        "don't feel anything", "dont feel anything",
        "life is too hard", "life is overwhelming", "overwhelmed by life"
    ]
}

#------Color Palettes------
COLOR_PALETTES = {
    "positive": ["#FFF5BA", "#FFD580", "#FFC1CC"],
    "neutral":  ["#B0C4DE", "#D8BFD8", "#FAF3E0"],
    "negative": ["#1B1A1E", "#4E4C4C", "#2F4F4F"]
}
#------Motivation Messages------
MOTIVATIONALS = [
    "🌟 You are stronger than you think!",
    "💪 Keep pushing forward.",
    "🌈 Bright days are ahead.",
    "😊 You matter. You are loved.",
    "✨ Every day is a new beginning.",
    "☀️ Stay positive. You're doing great!",
    "🙌 Keep Up The Good Work!",
]

#------Sentiment guard thresholds------
VADER_POS_THR = 0.20
VADER_NEG_THR = -0.30
SUICIDE_GUARD_THR    = 5.0   # %
MODEL_CONF_GUARD_THR = 0.90  # avg p_dep
REQUIRE_TWO_SIGNALS  = True

# ---------- Styles ----------
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
footer { visibility: hidden; }
h1,h2,h3,h4 { color: #e0e0e0; }
.card,.card.warn,.card.error,.card.info,.card.good,.stAlert{
  background: rgba(15,17,19,0.97) !important;
  color: #f2f6fb !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  padding: 16px !important;
}
.card.warn{background: rgba(70,52,10,0.97)!important;}
.card.error{background: rgba(95,28,28,0.97)!important;}
.card.info{background: rgba(17,40,56,0.97)!important;}
.card.good{background: rgba(26,56,26,0.97)!important;}
.conf-chip{ display:inline-block; padding:2px 8px; border-radius:999px; color:#fff; font-weight:700; }
.floating{ position:fixed; bottom:16px; left:50%; transform:translateX(-50%);
  font-size:22px; color:#FFEB3B; opacity:0.45; animation:floatText 10s infinite alternate; z-index:0; }
@keyframes floatText{ 0%{ transform:translate(-50%,0);} 100%{ transform:translate(-50%,-16px);} }
</style>
""", unsafe_allow_html=True)

# ---------- Helper funcs ----------
def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def get_proba(model, X) -> float:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return float(p[0, 1]) if p.shape[1] == 2 else float(p[0])
    if hasattr(model, "decision_function"):
        return sigmoid(float(model.decision_function(X).ravel()[0]))
    return float(int(model.predict(X)[0]))

def align_stats_from_raw_means(pos_avg_raw: float, neg_avg_raw: float):
    if neg_avg_raw > pos_avg_raw:
        dep_label = 1
        pos_avg_dep, neg_avg_dep = pos_avg_raw, neg_avg_raw
    else:
        dep_label = 0
        pos_avg_dep, neg_avg_dep = 1.0 - pos_avg_raw, 1.0 - neg_avg_raw
    thr_dep = 0.5 * (pos_avg_dep + neg_avg_dep)
    thr_dep = max(0.2, min(0.8, thr_dep))
    return dep_label, pos_avg_dep, neg_avg_dep, thr_dep

def auto_align_label(model, vectorizer):
    pos_ps = [get_proba(model, vectorizer.transform([s])) for s in PROBE_POS]
    neg_ps = [get_proba(model, vectorizer.transform([s])) for s in PROBE_NEG]
    dep_label, pos_avg_dep, neg_avg_dep, thr_dep = align_stats_from_raw_means(
        float(np.mean(pos_ps)), float(np.mean(neg_ps))
    )
    return {"depressed_label": dep_label, "pos_avg": pos_avg_dep, "neg_avg": neg_avg_dep, "thr_dep": thr_dep}

def dep_conf_band(p: float) -> str:
    if p >= 0.80:
        return "High"
    if p >= 0.60:
        return "Medium"
    return "Low"

def band_badge_html(band: str) -> str:
    colors = {"High":"#c62828","Medium":"#ef6c00","Low":"#2e7d32"}
    return f'<span class="conf-chip" style="background:{colors.get(band,"#455a64")};">{band}</span>'

def show_callout(kind: str, title: str, body: str = ""):
    cls = {"error":"card error","warn":"card warn","info":"card info","good":"card good"}.get(kind, "card info")
    st.markdown(
        f"<div class='{cls}'><div style='font-weight:700;font-size:1.1rem;margin-bottom:6px;'>{title}</div>{body}</div>",
        unsafe_allow_html=True
    )

def vader_label_from_scores(scores):
    c = scores.get('compound', 0.0)
    if c >= VADER_POS_THR:   return 'positive'
    if c <= VADER_NEG_THR:   return 'negative'
    return 'neutral'

def sentiment_bucket(bert_label: str, vader_scores) -> str:
    vl = vader_label_from_scores(vader_scores)
    if vl == 'positive': return "Positive"
    if vl == 'negative': return "Negative"
    bl = (bert_label or "").upper()
    if bl.startswith("POS"): return "Positive"
    if bl.startswith("NEG"): return "Negative"
    return "Neutral"

def suicide_score_pct(text: str) -> float:
    txt = (text or "").lower().strip()
    if not txt:
        return 0.0
    toks = txt.split()
    if not toks:
        return 0.0

    hits = 0.0

    for w in toks:
        if w in SUICIDE_KEYWORDS:
            hits += 1.0

    for ph in SUICIDE_PHRASES.get("strong", []):
        if ph in txt:
            hits += 3.0

    for ph in SUICIDE_PHRASES.get("medium", []):
        if ph in txt:
            hits += 2.0

    for ph in SUICIDE_PHRASES.get("weak", []):
        if ph in txt:
            hits += 1.0

    base_len = max(len(toks), 10)
    score = 100.0 * hits / base_len
    return max(0.0, min(score, 100.0))

def get_client_ip_best_effort() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "unknown"

def get_public_ip(timeout=2.5) -> str:
    if requests is None:
        return "unknown"
    try:
        r = requests.get("https://api.ipify.org?format=json", timeout=timeout)
        if r.ok:
            return r.json().get("ip", "unknown")
    except Exception:
        pass
    return "unknown"

def send_email_to_admin(subject: str, body: str):
    if not EMAIL_ENABLED:
        return False, "Email sending disabled (EMAIL_ENABLED=False)."
    if not SMTP_PASSWORD:
        return False, "SMTP app password missing (set env SMTP_APP_PASSWORD)."
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["From"] = SMTP_USERNAME
        msg["To"] = ADMIN_EMAIL_ALLOWED
        msg["Subject"] = subject
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, [ADMIN_EMAIL_ALLOWED], msg.as_string())
        server.quit()
        return True, "Sent."
    except Exception as e:
        return False, f"Email error: {e}"

# =====================================================================
# ===== DL-CORE-BEGIN  (Deep Learning loader: tokenizer + Keras) ======
# =====================================================================

# ---------- Where to look for model / tokenizer files ----------
try:
    BASE_DIR = pathlib.Path(__file__).resolve().parent
except NameError:
    BASE_DIR = pathlib.Path.cwd()

SEARCH_DIRS = []
for _d in (BASE_DIR, pathlib.Path.cwd(), BASE_DIR / "models", pathlib.Path.cwd() / "models"):
    if _d not in SEARCH_DIRS:
        SEARCH_DIRS.append(_d)

# ---------- Tokenizer (independent from the Keras Tokenizer class) ----------
_DEFAULT_TOK_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


class SimpleTokenizer:
    """Drop-in replacement of keras Tokenizer.texts_to_sequences() (same behaviour)."""

    def __init__(self, word_index, num_words=None, oov_token=None,
                 filters=_DEFAULT_TOK_FILTERS, lower=True, split=" ", char_level=False):
        self.word_index = {str(k): int(v) for k, v in dict(word_index).items()}
        self.num_words = int(num_words) if num_words else None
        self.oov_token = oov_token
        self.filters = filters if filters is not None else ""
        self.lower = bool(lower)
        self.split = split if split else " "
        self.char_level = bool(char_level)
        self._oov_index = self.word_index.get(oov_token) if oov_token is not None else None
        self._trans = str.maketrans({c: self.split for c in self.filters})

    def _tokenize(self, text: str):
        if self.lower:
            text = text.lower()
        if self.char_level:
            return list(text)
        text = text.translate(self._trans)
        return [t for t in text.split(self.split) if t]

    def texts_to_sequences(self, texts):
        out = []
        for t in texts:
            vect = []
            for w in self._tokenize(str(t)):
                i = self.word_index.get(w)
                if i is not None:
                    if self.num_words and i >= self.num_words:
                        if self._oov_index is not None:
                            vect.append(self._oov_index)
                    else:
                        vect.append(i)
                elif self._oov_index is not None:
                    vect.append(self._oov_index)
            out.append(vect)
        return out


class _KerasTokStub:
    """Empty placeholder: receives the state of a pickled keras Tokenizer."""
    pass


class _TokUnpickler(pickle.Unpickler):
    """Unpickles a keras Tokenizer even if the keras module path does not exist any more."""
    def find_class(self, module, name):
        if name == "Tokenizer" and "keras" in module:
            return _KerasTokStub
        return super().find_class(module, name)


def _tokenizer_from_state(state) -> SimpleTokenizer:
    def g(key, default=None):
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)

    word_index = g("word_index")
    if isinstance(word_index, str):
        word_index = json.loads(word_index)
    if not word_index:
        raise ValueError("Tokenizer file has no 'word_index' (is it really a Keras Tokenizer?).")
    return SimpleTokenizer(
        word_index=word_index,
        num_words=g("num_words"),
        oov_token=g("oov_token"),
        filters=g("filters", _DEFAULT_TOK_FILTERS),
        lower=g("lower", True),
        split=g("split", " "),
        char_level=g("char_level", False),
    )


def load_tokenizer_compat(path) -> SimpleTokenizer:
    p = pathlib.Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, str):            # tokenizer.to_json() saved with json.dump(...)
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("Unsupported tokenizer JSON structure.")
        cfg = data.get("config", data)
        return _tokenizer_from_state(cfg)

    if ext in (".pkl", ".pickle", ".joblib"):
        try:
            with open(p, "rb") as f:
                obj = _TokUnpickler(f).load()
        except Exception as e1:
            try:
                obj = joblib.load(p)
            except Exception as e2:
                raise RuntimeError(f"pickle: {type(e1).__name__}: {e1} | joblib: {type(e2).__name__}: {e2}")
        return _tokenizer_from_state(obj)

    raise ValueError("Unsupported tokenizer format (use .pkl or .json)")


def pad_post(seqs, maxlen: int):
    """Same as keras pad_sequences(..., padding='post', truncating='post')."""
    X = np.zeros((len(seqs), maxlen), dtype="int32")
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        if s:
            X[i, :len(s)] = s
    return X


# ---------- Keras imports (Keras 3 / TF 2.16+ ; also works with older) ----------
KERAS_IMPORT_ERROR = None
keras = None
load_model = None
model_from_json = None
Sequential = None
h5py = None
try:
    import h5py
    import keras
    from tensorflow.keras.models import load_model, Sequential, model_from_json
    from tensorflow.keras.layers import (
        Embedding, SpatialDropout1D, LSTM, GRU, Conv1D, MaxPooling1D,
        GlobalMaxPooling1D, Dense, Dropout, Bidirectional, Flatten, Input
    )
except Exception as _e:
    KERAS_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    load_model = None

if load_model is not None:
    try:
        @keras.saving.register_keras_serializable(package="Sequential")
        class MySequential(Sequential):
            pass
    except Exception:
        MySequential = Sequential

    _CUSTOM_OBJS = {
        "Sequential": MySequential,
        "Embedding": Embedding,
        "SpatialDropout1D": SpatialDropout1D,
        "LSTM": LSTM,
        "GRU": GRU,
        "Conv1D": Conv1D,
        "MaxPooling1D": MaxPooling1D,
        "GlobalMaxPooling1D": GlobalMaxPooling1D,
        "Dense": Dense,
        "Dropout": Dropout,
        "Bidirectional": Bidirectional,
        "Flatten": Flatten,
        "Input": Input,
    }

    # every built-in layer class by name (legacy configs have no "module" key, so Keras 3 cannot find them itself)
    # (+ initializers / regularizers / constraints: e.g. Keras 3.2 cannot resolve the legacy "Orthogonal" name of LSTM)
    _ALL_LAYER_OBJS = {}
    for _mod_name in ("initializers", "regularizers", "constraints", "layers"):   # layers last -> they win on name clashes
        try:
            _mod = getattr(keras, _mod_name)
            for _nm in dir(_mod):
                _obj = getattr(_mod, _nm, None)
                if _nm[:1].isupper() and isinstance(_obj, type):
                    _ALL_LAYER_OBJS[_nm] = _obj
        except Exception:
            pass
    _ALL_LAYER_OBJS.update({k: v for k, v in _CUSTOM_OBJS.items() if k != "Sequential"})

    # keys that old (Keras 2) configs contain and Keras 3 layers reject
    _LEGACY_DROP_KEYS = ("trainable", "dtype", "time_major")

    def _scrub_legacy_layer_cfg(cfg):
        if isinstance(cfg, dict):
            for k in _LEGACY_DROP_KEYS:
                cfg.pop(k, None)
            for k in list(cfg.keys()):
                _scrub_legacy_layer_cfg(cfg[k])
        elif isinstance(cfg, list):
            for it in cfg:
                _scrub_legacy_layer_cfg(it)

    def _pop_key_deep(obj, key) -> bool:
        """Removes `key` from every nested dict. Returns True if something was removed."""
        removed = False
        if isinstance(obj, dict):
            if key in obj:
                obj.pop(key)
                removed = True
            for v in list(obj.values()):
                removed = _pop_key_deep(v, key) or removed
        elif isinstance(obj, list):
            for v in obj:
                removed = _pop_key_deep(v, key) or removed
        return removed

    _BAD_KW_RE = re.compile(r"unexpected keyword argument '(\w+)'")
    _BAD_KW_SET_RE = re.compile(r"Unrecognized keyword arguments(?: passed to \w+)?: \{([^}]*)\}")

    def _bad_kwargs_from_error(msg: str):
        keys = set(_BAD_KW_RE.findall(msg))
        for chunk in _BAD_KW_SET_RE.findall(msg):
            keys.update(re.findall(r"'(\w+)'\s*:", chunk))
        return keys

    def _read_model_config(model_path: str):
        """Reads the architecture JSON from a legacy .h5 file or from a .keras archive."""
        p = pathlib.Path(model_path)
        if p.suffix.lower() == ".keras":
            with zipfile.ZipFile(p) as z:
                return json.loads(z.read("config.json").decode("utf-8"))
        with h5py.File(p, "r") as f:
            raw = f.attrs.get("model_config", None)
            if raw is None:
                raise RuntimeError("No model_config attribute inside model file "
                                   "(weights-only file? the full model must be saved).")
            cfg_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            return json.loads(cfg_str)

    def _saved_keras_version(model_path: str) -> str:
        try:
            p = pathlib.Path(model_path)
            if p.suffix.lower() == ".keras":
                with zipfile.ZipFile(p) as z:
                    return str(json.loads(z.read("metadata.json")).get("keras_version", ""))
            with h5py.File(p, "r") as f:
                v = f.attrs.get("keras_version", "")
                return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
        except Exception:
            return ""

    def _ver_tuple(v: str):
        nums = re.findall(r"\d+", v or "")
        return tuple(int(x) for x in nums[:2])

    def _dl_hint(errors_text: str, saved_ver: str, installed_ver: str) -> str:
        hints = []
        sv, iv = _ver_tuple(saved_ver), _ver_tuple(installed_ver)
        if sv and iv and sv > iv:
            hints.append(f"HINT: the file was saved with a NEWER Keras ({saved_ver}) than the installed one "
                         f"({installed_ver}) -> upgrade Keras or re-save the model as legacy .h5.")
        if "objects could not be loaded" in errors_text and iv and iv < (3, 3):
            hints.append("HINT: known Keras 3.2.x bug when loading LSTM/GRU/Bidirectional from .keras -> run: "
                         "pip install -U \"keras>=3.4\" (tested with keras 3.6.0 + TensorFlow 2.17), "
                         "or use the .h5 version of the model.")
        return " ".join(hints)

    def _deserialize_layer(layer_cfg):
        """Builds ONE layer from an old/new config; drops any kwarg the installed Keras rejects."""
        cfg = json.loads(json.dumps(layer_cfg))          # deep copy
        _scrub_legacy_layer_cfg(cfg)
        _pop_key_deep(cfg, "batch_input_shape")
        last_err = None
        for _ in range(12):
            try:
                return keras.layers.deserialize(cfg, custom_objects=_ALL_LAYER_OBJS)
            except Exception as e:
                last_err = e
                bad = _bad_kwargs_from_error(str(e))
                if not bad:
                    break
                if not any([_pop_key_deep(cfg, k) for k in bad]):
                    break
        raise last_err

    def _rebuild_sequential(model_path: str):
        """Layer-by-layer rebuild of a Sequential model (works for legacy Keras-2 .h5 files
        that Keras 3 cannot deserialize directly), then loads the weights."""
        cfg = _read_model_config(model_path)
        if cfg.get("class_name") != "Sequential":
            raise RuntimeError(f"layer-by-layer rebuild supports Sequential models only (found {cfg.get('class_name')}).")
        inner = cfg.get("config", {})
        layers_cfg = inner.get("layers", []) if isinstance(inner, dict) else inner
        name = inner.get("name") if isinstance(inner, dict) else None

        input_shape, input_dtype, layers = None, "float32", []
        if isinstance(inner, dict) and inner.get("build_input_shape"):
            input_shape = tuple(inner["build_input_shape"][1:])
        for lc in layers_cfg:
            conf = lc.get("config", {}) or {}
            bshape = conf.get("batch_input_shape") or conf.get("batch_shape")
            if lc.get("class_name") == "InputLayer":
                if bshape:
                    input_shape = tuple(bshape[1:])
                if isinstance(conf.get("dtype"), str):
                    input_dtype = conf["dtype"]
                continue
            if input_shape is None and bshape:
                input_shape = tuple(bshape[1:])
            layers.append(_deserialize_layer(lc))
        if input_shape is None or not layers:
            raise RuntimeError("Could not infer the input shape / layers from the model config.")

        model = Sequential(name=name)
        model.add(keras.Input(shape=input_shape, dtype=input_dtype))
        for layer in layers:
            model.add(layer)
        model.load_weights(model_path)
        return model

    def load_model_compat(model_path: str):
        """Tries several strategies. Raises RuntimeError with ALL error messages if every one fails."""
        errors = []

        # 1) standard loader (with / without custom objects)
        for label, kwargs in (("custom_objects", {"custom_objects": _CUSTOM_OBJS}), ("plain", {})):
            try:
                return load_model(model_path, compile=False, **kwargs)
            except Exception as e:
                errors.append(f"load_model[{label}] -> {type(e).__name__}: {str(e)[:250]}")

        # 2) layer-by-layer rebuild (+ weights)  <- fixes legacy .h5 (SpatialDropout1D 'trainable' etc.)
        try:
            return _rebuild_sequential(model_path)
        except Exception as e:
            errors.append(f"rebuild-layers+weights -> {type(e).__name__}: {str(e)[:250]}")

        # 3) generic config scrub via model_from_json (non-Sequential models)
        try:
            cfg = _read_model_config(model_path)
            _scrub_legacy_layer_cfg(cfg)
            model = model_from_json(json.dumps(cfg), custom_objects=_CUSTOM_OBJS)
            model.load_weights(model_path)
            return model
        except Exception as e:
            errors.append(f"config-scrub+weights -> {type(e).__name__}: {str(e)[:250]}")

        sv = _saved_keras_version(model_path)
        try:
            iv = keras.__version__
        except Exception:
            iv = ""
        joined = " || ".join(errors)
        info = f" [file saved with keras {sv or '?'}; installed keras {iv or '?'}]"
        hint = _dl_hint(joined, sv, iv)
        raise RuntimeError(joined + info + ((" " + hint) if hint else ""))

    def _embedding_vocab_limit(model):
        try:
            for layer in model.layers:
                if layer.__class__.__name__ == "Embedding":
                    return int(layer.input_dim)
        except Exception:
            pass
        return None

    def _model_max_len(model, default: int) -> int:
        try:
            shp = model.input_shape
            if isinstance(shp, list):
                shp = shp[0]
            if isinstance(shp, (tuple, list)) and len(shp) >= 2 and isinstance(shp[1], int) and shp[1] > 0:
                return int(shp[1])
        except Exception:
            pass
        return int(default)

    def _file_sig(path) -> str:
        try:
            p = pathlib.Path(path)
            stt = p.stat()
            return f"{p.name}:{stt.st_size}:{int(stt.st_mtime)}"
        except Exception:
            return str(path)

    class KerasTextWrapper:
        """Keras text model (CNN / LSTM / ...) + tokenizer  ->  probability."""

        def __init__(self, model, tokenizer, max_len=200, model_path="", tokenizer_path=""):
            self.model = model
            self.tokenizer = tokenizer
            self.max_len = _model_max_len(model, max_len)
            self.vocab_limit = _embedding_vocab_limit(model)
            self.model_path = str(model_path)
            self.tokenizer_path = str(tokenizer_path)
            self.sig = _file_sig(model_path) + "|" + _file_sig(tokenizer_path)
            self._validate()

        def _validate(self):
            p = self.predict_proba_batch(["i feel fine today", "i feel so sad and alone"])
            if not np.all(np.isfinite(p)):
                raise RuntimeError("Model returned non-finite values in the test prediction.")

        def predict_proba_batch(self, texts):
            seqs = self.tokenizer.texts_to_sequences(list(texts))
            X = pad_post(seqs, self.max_len)
            if self.vocab_limit:
                X = np.where(X < self.vocab_limit, X, 0).astype("int32")   # ids outside the embedding -> padding
            p = self.model.predict(X, verbose=0)
            if isinstance(p, (list, tuple)):
                p = p[0]
            p = np.asarray(p, dtype="float64")
            if p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]
            elif p.ndim == 2 and p.shape[1] > 2:
                raise RuntimeError(f"Unsupported output shape {p.shape} (binary model expected).")
            p = p.reshape(len(X))
            if p.min() < 0.0 or p.max() > 1.0:          # logits -> probabilities
                p = 1.0 / (1.0 + np.exp(-p))
            return p

        def predict_proba_text(self, texts):
            return float(self.predict_proba_batch(texts)[0])
else:
    KerasTextWrapper = None


def auto_align_label_textmodel(text_model: "KerasTextWrapper"):
    pos_ps = text_model.predict_proba_batch(PROBE_POS)      # one batch call instead of 13 single calls
    neg_ps = text_model.predict_proba_batch(PROBE_NEG)
    dep_label, pos_avg_dep, neg_avg_dep, thr_dep = align_stats_from_raw_means(
        float(np.mean(pos_ps)), float(np.mean(neg_ps))
    )
    return {
        "depressed_label": dep_label, "pos_avg": pos_avg_dep, "neg_avg": neg_avg_dep,
        "thr_dep": thr_dep, "sig": text_model.sig,
    }


# ---------- Discovery + cached loading ----------
DL_SPECS = {
    "CNN": {
        "models": ["cnn_model_smote_fixed.h5", "cnn_model_smote.h5",
                   "cnn_model_smote_fixed.keras", "cnn_model_smote.keras"],
        "tokenizers": ["cnn_tokenizer.pkl", "tokenizer_smote.pkl", "tokenizer_smote.json"],
        "keywords": ("cnn", "conv"),
        "exclude": ("lstm", "gru"),
    },
    "BiLSTM": {
        "models": ["lstm_model_smote_fixed.h5", "lstm_model_smote.h5",
                   "lstm_model_smote_fixed.keras", "lstm_model_smote.keras"],
        "tokenizers": ["bilstm_tokenizer.pkl", "tokenizer_smote.pkl", "tokenizer_smote.json"],
        "keywords": ("lstm", "gru"),
        "exclude": ("cnn", "conv"),
    },
}


def _collect_files(explicit_names, keywords, exts, exclude=()):
    """Explicit names first (in the given order), then auto-discovered files by keyword."""
    found, seen = [], set()

    def _add(p):
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            found.append(p)

    for name in explicit_names:
        for d in SEARCH_DIRS:
            p = d / name
            if p.is_file():
                _add(p)
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            stem = p.stem.lower()
            if (p.is_file() and p.suffix.lower() in exts
                    and any(k in stem for k in keywords)
                    and not any(x in stem for x in exclude)):
                _add(p)
    return found


def dl_env_info() -> dict:
    info = {"python": platform.python_version()}
    for mod in ("tensorflow", "keras", "h5py"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info


@st.cache_resource(show_spinner="Loading Deep Learning models…")
def load_dl_models():
    """Returns (dict name -> KerasTextWrapper, list of log rows)."""
    out, log = {}, []

    if KerasTextWrapper is None:
        log.append({"model": "*", "file": "-", "tokenizer": "-", "status": "SKIPPED",
                    "detail": f"Keras / TensorFlow could not be imported: {KERAS_IMPORT_ERROR}"})
        return out, log

    for label, spec in DL_SPECS.items():
        model_files = _collect_files(spec["models"], spec["keywords"], (".h5", ".keras"), spec["exclude"])
        tok_files = _collect_files(spec["tokenizers"], ("token",), (".pkl", ".json"), spec["exclude"])

        if not model_files:
            log.append({"model": label, "file": "-", "tokenizer": "-", "status": "MISSING",
                        "detail": "No .h5/.keras model file found in: " + ", ".join(str(d) for d in SEARCH_DIRS)})
            continue
        if not tok_files:
            log.append({"model": label, "file": model_files[0].name, "tokenizer": "-", "status": "MISSING",
                        "detail": "No tokenizer (.pkl/.json) found."})
            continue

        done = False
        for mp in model_files:
            try:
                keras_model = load_model_compat(str(mp))
            except Exception as e:
                log.append({"model": label, "file": mp.name, "tokenizer": "-", "status": "FAILED",
                            "detail": f"model load: {e}"})
                continue

            for tp in tok_files:
                try:
                    tok = load_tokenizer_compat(tp)
                    wrapper = KerasTextWrapper(keras_model, tok, max_len=200,
                                               model_path=mp, tokenizer_path=tp)
                    out[label] = wrapper
                    log.append({"model": label, "file": mp.name, "tokenizer": tp.name, "status": "OK",
                                "detail": (f"max_len={wrapper.max_len}, vocab_limit={wrapper.vocab_limit}, "
                                           f"tokenizer_words={len(tok.word_index)}")})
                    done = True
                    break
                except Exception as e:
                    log.append({"model": label, "file": mp.name, "tokenizer": tp.name, "status": "FAILED",
                                "detail": f"tokenizer/validation: {type(e).__name__}: {str(e)[:300]}"})
            if done:
                break

    return out, log

# =====================================================================
# ===== DL-CORE-END ====================================================
# =====================================================================

def majority_vote_with_thresholds(p_by_model: dict, thr_by_model: dict) -> int:
    votes = [1 if p_by_model[m] >= thr_by_model[m] else 0 for m in p_by_model]
    return 1 if sum(votes) > (len(votes)/2) else 0

# ---------- Caption env ----------
st.caption(
    f'<span style="font-size:12px;color:#cfd8dc;opacity:.9;">Env: Python {platform.python_version()} | NumPy {np.__version__} | Matplotlib {plt.matplotlib.__version__}</span>',
    unsafe_allow_html=True
)

# ---------- Transformers sentiment ----------
@st.cache_resource(show_spinner=False)
def load_bert_pipeline():
    try:
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,          # CPU
            truncation=True
        )
    except Exception as e:
         return None

bert_classifier = load_bert_pipeline()

# ---------- Load classic ML ----------
tfidf_vectorizer = joblib.load("tfidf_vectorizer_smote.pkl")
models = {
    #"Random Forest": joblib.load("rf_model_smote.pkl"),
    "XGBoost": joblib.load("xgb_model_smote.pkl"),
    "Logistic Regression": joblib.load("lr_model_smote.pkl"),
    "SVM": joblib.load("svm_model_smote.pkl"),
    "Naive Bayes": joblib.load("nb_model_smote.pkl"),
}

# ---------- Load DL (CNN / BiLSTM) — cached, auto-discovered, logged ----------
text_models, DL_LOAD_LOG = load_dl_models()

analyzer = SentimentIntensityAnalyzer()

# ---------- Persisted label map ----------
ALIGN_PATH = pathlib.Path("label_map.json")

def _persist_label_maps():
    try:
        ALIGN_PATH.write_text(json.dumps(st.session_state["label_maps"], indent=2))
    except Exception:
        pass

if "label_maps" not in st.session_state:
    _lm = {}
    if ALIGN_PATH.exists():
        try:
            _lm = json.loads(ALIGN_PATH.read_text())
        except Exception:
            _lm = {}
    for _n, _m in models.items():
        if _n not in _lm:
            _lm[_n] = auto_align_label(_m, tfidf_vectorizer)
    st.session_state["label_maps"] = _lm
    _persist_label_maps()

def sync_dl_label_maps():
    """(Re)computes label alignment of a DL model when it is new or when its files changed."""
    changed = False
    for n, tm in text_models.items():
        cur = st.session_state["label_maps"].get(n)
        if (not cur) or cur.get("sig") != tm.sig:
            try:
                st.session_state["label_maps"][n] = auto_align_label_textmodel(tm)
                changed = True
            except Exception:
                pass
    if changed:
        _persist_label_maps()

sync_dl_label_maps()

# ---------- Header / login ----------
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "show_login" not in st.session_state:
    st.session_state["show_login"] = False

top_left, _, top_right = st.columns([6, 6, 2])
with top_left:
    is_admin = st.session_state.get("is_admin", False)
    show_login = st.session_state["show_login"]
    st.markdown(f"Now you are in: **{'Admin View' if is_admin else 'User View'}**")
    if not is_admin and not st.session_state["show_login"]:
        st.title("🧠Depression Detector & Mood Reactive UI")

with top_right:
    if st.session_state["is_admin"]:
        if st.button("🔒 Log out", use_container_width=True):
            st.session_state["is_admin"] = False
            st.session_state["show_login"] = False
            st.rerun()
    else:
        if st.button("🔐 Admin", use_container_width=True):
            st.session_state["show_login"] = True

def admin_login_panel():
    if not st.session_state.get("show_login", False) or st.session_state["is_admin"]:
        return False
    with st.expander("🔑 Admin Login", expanded=True):
        st.write("Enter your admin credentials")
        with st.form("admin_login_form", clear_on_submit=True):
            email = st.text_input("Email", value="", placeholder="admin@example.com")
            pwd   = st.text_input("Code",  value="", type="password", placeholder="••••••••")
            c1, c2 = st.columns([1, 1])
            cancel = c1.form_submit_button("Cancel")
            submit = c2.form_submit_button("Login")
        if cancel:
            st.session_state["show_login"] = False
            st.rerun()
            return False
        if submit:
            if email.strip().lower() == ADMIN_EMAIL_ALLOWED and pwd == ADMIN_PASS_ALLOWED:
                st.session_state["is_admin"]  = True
                st.session_state["show_login"] = False
                st.success("Logged in successfully.")
                st.rerun()
                return True
            else:
                st.error("Invalid credentials.")
    return False

_ = admin_login_panel()

# ---------- Views ----------
show_login = st.session_state.get("show_login", False)
is_admin   = st.session_state.get("is_admin", False)

# ============================ USER VIEW ============================
if (not is_admin) and (not show_login):
    user_input = st.text_area(
    "✍️ Write here about your day - feelings:",
    height=180,
    placeholder="Example: Lately I feel stressed and tired. I can’t focus, my sleep is bad, and I don’t enjoy things like before..."
)
    if st.button("📊 Analyze"):
        text = (user_input or "").strip()
        if len(text) < 5:
            st.warning("Please enter a longer text (≥5 characters).")
            st.stop()
        if len(text) > 5000:
            text = text[:5000]

        X = tfidf_vectorizer.transform([text])

        per_model_probs, per_model_thrs = {}, {}
        dl_errors = {}

        # ---------- Classic Machine Learning
        for name, model in models.items():
            raw_p = get_proba(model, X)
            lm = st.session_state["label_maps"][name]
            p_dep = raw_p if lm["depressed_label"] == 1 else (1.0 - raw_p)
            per_model_probs[name] = p_dep
            per_model_thrs[name]  = lm["thr_dep"]

        #---------- Deep Learning (CNN / BiLSTM) ----------
        for name, tm in text_models.items():
            try:
                raw_p = tm.predict_proba_text([text])
                lm = st.session_state["label_maps"].get(name) or auto_align_label_textmodel(tm)
                st.session_state["label_maps"][name] = lm
                p_dep = raw_p if lm["depressed_label"] == 1 else (1.0 - raw_p)
                per_model_probs[name] = p_dep
                per_model_thrs[name]  = lm["thr_dep"]
            except Exception as e:
                dl_errors[name] = f"{type(e).__name__}: {e}"
                continue

        ens_vote  = majority_vote_with_thresholds(per_model_probs, per_model_thrs) if per_model_probs else 0
        avg_p_dep = float(np.mean(list(per_model_probs.values()))) if per_model_probs else 0.0

        #---------- Sentiment context----------
        bert_label = "N/A"
        bert_score = 0.0
        bert_err   = None

        try:
            if bert_classifier is not None:
                out = bert_classifier(text)
                if isinstance(out, list) and len(out) > 0:
                    bert_label = str(out[0].get("label", "N/A") or "N/A")
                    bert_score = float(out[0].get("score", 0.0) or 0.0)
        except Exception as e:
            bert_err = str(e)
            bert_label = "N/A"
            bert_score = 0.0

        vader_scores = analyzer.polarity_scores(text)
        vader_lbl    = vader_label_from_scores(vader_scores)
        senti_bucket = sentiment_bucket(bert_label, vader_scores)

        sscore  = suicide_score_pct(text)
        ip_addr = get_client_ip_best_effort()
        pub_ip  = get_public_ip()

              #---------- Guardrails ----------
        is_vader_neg_strong = (vader_lbl == "negative")
        is_bert_neg_strong = (bert_classifier is not None and bert_label.upper() == "NEGATIVE" and bert_score >= 0.90)
        high_suicide_risk   = (sscore >= SUICIDE_GUARD_THR)
        high_model_conf     = (avg_p_dep >= MODEL_CONF_GUARD_THR)

        signals_true = int(sum([is_vader_neg_strong, is_bert_neg_strong, high_suicide_risk, high_model_conf]))
        guard_pass   = (signals_true >= 2) if REQUIRE_TWO_SIGNALS else (signals_true >=1)

        # ✅ Labels persisted
        ensemble_label = "DEPRESSED" if ens_vote == 1 else "NOT DEPRESSED"
        final_label    = "DEPRESSED" if (ens_vote == 1 and guard_pass) else "NOT DEPRESSED"

        # ✅ Persist everything needed for Admin/UI/Email
        st.session_state["last_analysis"] = {
            "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "ip": ip_addr,
            "public_ip": pub_ip,
            "text": text,

            "per_model_probs": per_model_probs,
            "per_model_thrs": per_model_thrs,
            "dl_errors": dl_errors,

            "ensemble_vote": ens_vote,
            "ensemble_avg_prob": avg_p_dep,
            "ensemble_label": ensemble_label,
            "final_label": final_label,

            "guard_pass": bool(guard_pass),
            "signals_true": int(signals_true),
            "vader_label": vader_lbl,
            "senti_bucket": senti_bucket,

            "sentiment": vader_scores,
            "suicide_score": float(sscore),

            # ✅ BERT stored properly
            "bert": {"label": bert_label, "score": float(bert_score)},

            "label_maps": st.session_state["label_maps"],
        }

        #---------- Mood background ----------
        mood = vader_label_from_scores(vader_scores)
        bg_color = random.choice(COLOR_PALETTES[mood])
        st.markdown(f"""
        <style>
        .stApp {{
          background: linear-gradient(180deg, {bg_color} 0%, #0e1117 55%, #0e1117 100%);
        }}
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="floating">{random.choice(MOTIVATIONALS)}</div>', unsafe_allow_html=True)

        # ---------- Confidence chip ----------
        band = dep_conf_band(avg_p_dep)
        st.markdown(f'<div style="margin-top:6px;">Confidence: {band_badge_html(band)}</div>', unsafe_allow_html=True)

        #---------- Callouts----------
        if (ens_vote == 1) and guard_pass:
            show_callout(
                "error",
                "😞 The Model Recognised Signs of Depression.",
                "If you are Struggling, you can CALL 📞 <b>10306</b> for a free psychological Support (24/7)."
            )
        else:
            if senti_bucket == "Positive":
                show_callout("good", "😊 Not Depressed.", "No signs of depression detected. Keep positive thinking! 💪")
            elif senti_bucket == "Neutral":
                show_callout("info", "😐 Neutral.", "A short walk or a warm drink might help reset. 🚶")
            else:
                show_callout("warn", "🤔 No Signs of Depressed, but a Negative Mood has been Detected.", "Take care and do something that makes you HAPPY! 💙")

        st.caption("⚠️ This tool is not a medical diagnosis. In case of emergency, call 166 or 112 immediately.")

        #---------- Email to admin ----------
        subject = "[AI-Depression] New analysis snapshot"
        mood_txt = vader_lbl.upper()
        decision_txt = final_label

        body = (
            f"Timestamp: {st.session_state['last_analysis']['timestamp']}\n"
            f"Local IP: {ip_addr}\n"
            f"Public IP: {pub_ip}\n"
            f"Summary mood (VADER): {mood_txt} (compound={vader_scores.get('compound',0.0):.3f})\n"
            f"BERT Sentiment: {bert_label} (conf={bert_score*100:.1f}%)\n"
            f"Ensemble decision: {decision_txt} (avg p_dep={avg_p_dep*100:.1f}%)\n"
            f"Final decision (after guardrails): {final_label}\n"
            f"Guardrails pass: {guard_pass} (signals={signals_true})\n"
            f"Suicide risk score: {sscore:.2f}%\n"
            f"\n--- Models ---\n"
            + "\n".join([
                f"{name}: p_dep={per_model_probs[name]*100:.1f}% (thr={per_model_thrs[name]*100:.1f}%)"
                for name in per_model_probs
            ])
            + "\n\nUser text: " + text + "\n"
        )

        ok, msg = send_email_to_admin(subject, body)
        if EMAIL_ENABLED:
            st.caption(f"Admin email notification: {'OK' if ok else msg}")


# ============================ ADMIN VIEW ============================
elif is_admin:
    import plotly.graph_objects as go
    import pandas as pd

    st.header("Depression Detector - Admin View")

    if "last_analysis" not in st.session_state:
        st.error("No analysis performed yet. Go to User View and run an analysis.")
        st.stop()

    data = st.session_state.get("last_analysis", {})

    st.subheader("📄 User Text (context)")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(
        f"**Time:** {data.get('timestamp','?')}  |  "
        f"**Local IP:** {data.get('ip','unknown')}  |  "
        f"**Public IP:** {data.get('public_ip','unknown')}"
    )
    st.write(data.get("text", ""))
    st.markdown("</div>", unsafe_allow_html=True)

    ens_vote = int(data.get("ensemble_vote", 0))
    avgp     = float(data.get("ensemble_avg_prob", 0.0))

    st.subheader("🧩 Ensemble")
    st.write(
        f"- Majority vote (per-model thresholds) → "
        f"**{'Depressed 😞' if ens_vote==1 else 'Not Depressed 😊'}** "
        f"(avg p_dep={avgp*100:.1f}%)"
    )
    st.write(f"- Confidence band: **{dep_conf_band(avgp)}**")
    st.write(f"- Guardrails pass: **{data.get('guard_pass', False)}** (signals={data.get('signals_true', 0)})")
    st.write(f"- Final decision (UI): **{data.get('final_label','?')}**")

    st.subheader("🧠 BERT Sentiment")
    bert = data.get("bert", {})
    st.write(f"- Label: **{bert.get('label','')}**  |  Confidence: **{float(bert.get('score',0.0))*100:.2f}%**")

    st.subheader("🤖 ML Models — probability, threshold & decision")
    per_probs = data.get("per_model_probs", {})
    per_thrs  = data.get("per_model_thrs", {})
    label_maps = data.get("label_maps", {})

    for name, p in per_probs.items():
        lm  = label_maps.get(name, {})
        thr = float(per_thrs.get(name, 0.5))
        p = float(p)
        decision = "Depressed 😞" if p >= thr else "Not Depressed 😊"
        st.write(
            f"- **{name}** → p_dep={p*100:.1f}% (thr={thr*100:.1f}%) → **{decision}** "
            f"(auto-align: depressed_label={lm.get('depressed_label','?')}, "
            f"probes pos_avg={lm.get('pos_avg',0):.2f}, neg_avg={lm.get('neg_avg',0):.2f})"
        )

    st.subheader("📊 VADER Sentiment")
    scores = data.get("sentiment", {})
    neg = float(scores.get('neg', 0.0))
    neu = float(scores.get('neu', 0.0))
    pos = float(scores.get('pos', 0.0))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].bar(['Negative', 'Neutral', 'Positive'], [neg, neu, pos],
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axs[0].set_title('Sentiment Analysis - Bar Plot')
    axs[1].pie([neg, neu, pos],
               labels=['Negative', 'Neutral', 'Positive'],
               autopct='%1.1f%%', startangle=90,
               colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axs[1].set_title('Sentiment Analysis - Pie Chart')
    st.pyplot(fig)

    st.subheader("🔎 Top tokens (TF-IDF) for this text")
    try:
        feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
        X_vec = tfidf_vectorizer.transform([data.get('text','')])
        row = X_vec.toarray()[0]
        if np.count_nonzero(row) == 0:
            st.info("No informative tokens found (all zeros).")
        else:
            top_idx = np.argsort(row)[-5:][::-1]
            tokens  = [feature_names[i] for i in top_idx]
            weights = [row[i] for i in top_idx]
            st.write(", ".join(f"{t} ({w:.3f})" for t, w in zip(tokens, weights)))
    except Exception:
        st.info("TF-IDF tokens not available for explanation.")

    # ---------------- Suicide-Risk ----------------
    st.subheader("🧠 Suicide-Risk")
    s = float(data.get("suicide_score", 0.0))

    def risk_style(score: float):
        if score >= 30:
            return ("HIGH", "#6b1f1f", "#ff4d4d")
        if score >= 10:
            return ("MODERATE", "#5a3a10", "#ffa726")
        return ("LOW", "#1f4021", "#66bb6a")

    level, pill_bg, dot_color = risk_style(s)
    pill_html = f"""
    <div style="background:{pill_bg};border:1px solid rgba(255,255,255,0.12);border-radius:16px;
                padding:14px 16px;display:flex;align-items:center;gap:12px;box-shadow:0 10px 30px rgba(0,0,0,0.35);
                max-width: 900px;">
      <span style="width:14px;height:14px;border-radius:50%;background:{dot_color};display:inline-block;
                   border:2px solid rgba(0,0,0,0.25)"></span>
      <span style="font-weight:800;letter-spacing:.2px;">Suicide Risk Score:</span>
      <span style="font-variant-numeric:tabular-nums;font-weight:800;">{s:.2f}%</span>
      <span style="opacity:.9;margin-left:6px;">({level})</span>
    </div>"""
    st.markdown(pill_html, unsafe_allow_html=True)

    analyzed_text = (data.get("text", "") or "").lower()
    wc = Counter(analyzed_text.split())

    risky_words = [w for w in wc if w in SUICIDE_KEYWORDS]
    if risky_words:
        words_for_plot = risky_words
        info_msg = "3D visualization of high-risk suicide words (if any)."
    else:
        words_for_plot = [w for w, _ in wc.most_common(5)]
        info_msg = "No explicit suicide words detected — showing top frequent tokens instead."

    if words_for_plot:
        x_vals = list(range(len(words_for_plot)))
        y_vals = [wc[w] for w in words_for_plot]
        z_vals = [neg for _ in words_for_plot]

        fig3d = go.Figure(data=[go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='markers+text',
            text=words_for_plot,
            textposition='top center',
            marker=dict(size=12, color=z_vals, colorscale='Reds', opacity=0.85)
        )])

        fig3d.update_layout(
            scene=dict(
                xaxis_title='Word Index',
                yaxis_title='Frequency',
                zaxis_title='Negativity Score'
            ),
            width=800,
            height=520,
            margin=dict(r=10, l=10, b=10, t=10),
            title="3D Token Risk / Frequency Map"
        )

        st.plotly_chart(fig3d, use_container_width=True)
        st.caption(info_msg)
    else:
        st.info("Δεν βρέθηκαν λέξεις για 3D απεικόνιση (άδειο ή πολύ μικρό κείμενο).")

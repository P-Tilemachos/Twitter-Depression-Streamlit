# -*- coding: utf-8 -*-
"""
test_final_2.py  (FIXED)
AI Depression Detector (Final Thesis Version — DL-ready)
- TF-IDF + ML models (RF, XGB, LR, SVM, NB)
- Deep Learning text models (CNN, BiLSTM) with safe loader (.h5/.keras)
- Tokenizers (.pkl/.json)
- Per-model auto label alignment (probes) & thresholds
- Ensemble decision + guardrails
- VADER + BERT sentiment (messaging)
- Admin view: per-model details, suicide-risk viz, TF-IDF explain

✅ FIXES / MODS:
1) All ensemble/guard outputs are persisted safely inside st.session_state["last_analysis"]
2) Admin View reads ONLY from st.session_state["last_analysis"] using .get() to avoid NameError
3) Added explicit keys: ensemble_label, final_label, guard_pass, signals_true, vader_label, senti_bucket
"""

#--------------------------------------Imports--------------------------------------
import os
import json
import math
import random
import pathlib
import platform
import smtplib
import socket
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

# ---------- Keras safe loader (Keras 3 / TF 2.17) ----------
try:
    import h5py, keras
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import (
        Embedding, SpatialDropout1D, LSTM, GRU, Conv1D, MaxPooling1D,
        GlobalMaxPooling1D, Dense, Dropout, Bidirectional, Flatten, Input
    )
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    try:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except Exception:
        try:
            from keras.preprocessing.text import tokenizer_from_json
        except Exception:
            tokenizer_from_json = None
except Exception:
    load_model = None
    pad_sequences = None
    tokenizer_from_json = None

if load_model is not None:
    @keras.saving.register_keras_serializable(package="Sequential")
    class MySequential(Sequential):
        pass

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

    def _scrub_legacy_layer_cfg(cfg):
        if isinstance(cfg, dict):
            cfg.pop("trainable", None)
            cfg.pop("dtype", None)
            for k in list(cfg.keys()):
                _scrub_legacy_layer_cfg(cfg[k])
        elif isinstance(cfg, list):
            for it in cfg:
                _scrub_legacy_layer_cfg(it)

    def load_model_compat(model_path: str):
        try:
            return load_model(model_path, compile=False, custom_objects=_CUSTOM_OBJS)
        except TypeError as e:
            msg = str(e)
            if ("SpatialDropout1D" in msg) and ("unexpected keyword argument 'trainable'" in msg):
                with h5py.File(model_path, "r") as f:
                    raw = f.attrs.get("model_config", None)
                    if raw is None:
                        raise RuntimeError("No model_config attribute inside model file.")
                    cfg_str = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    cfg = json.loads(cfg_str)
                _scrub_legacy_layer_cfg(cfg)
                from keras.models import model_from_json
                model = model_from_json(json.dumps(cfg), custom_objects=_CUSTOM_OBJS)
                model.load_weights(model_path)
                return model
            raise

    class KerasTextWrapper:
        def __init__(self, model_path, tokenizer_path, max_len=200):
            self.model = load_model_compat(model_path)
            p = pathlib.Path(tokenizer_path)
            if p.suffix.lower() == ".pkl":
                self.tokenizer = joblib.load(p)
            elif p.suffix.lower() == ".json":
                if tokenizer_from_json is None:
                    raise RuntimeError("tokenizer_from_json not available. Provide .pkl tokenizer instead.")
                with open(p, "r", encoding="utf-8") as f:
                    self.tokenizer = tokenizer_from_json(json.load(f))
            else:
                raise ValueError("Unsupported tokenizer format (use .pkl or .json)")
            self.max_len = max_len

        def predict_proba_text(self, texts):
            seqs = self.tokenizer.texts_to_sequences(texts)
            X = pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")
            p = self.model.predict(X, verbose=0)
            if hasattr(p, "ndim") and p.ndim == 2 and p.shape[1] == 2:
                p = p[:, 1]
            return float(p.ravel()[0])
else:
    KerasTextWrapper = None

def auto_align_label_textmodel(text_model: "KerasTextWrapper"):
    pos_ps = [text_model.predict_proba_text([s]) for s in PROBE_POS]
    neg_ps = [text_model.predict_proba_text([s]) for s in PROBE_NEG]
    dep_label, pos_avg_dep, neg_avg_dep, thr_dep = align_stats_from_raw_means(
        float(np.mean(pos_ps)), float(np.mean(neg_ps))
    )
    return {"depressed_label": dep_label, "pos_avg": pos_avg_dep, "neg_avg": neg_avg_dep, "thr_dep": thr_dep}

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

# ---------- Load DL (CNN / BiLSTM) ----------
text_models = {}

def _try_add_text_model(label, model_candidates, tok_candidates, max_len=200):
    if (KerasTextWrapper is None) or (load_model is None) or (pad_sequences is None):
        return
    for mp in model_candidates:
        if not os.path.exists(mp):
            continue
        for tp in tok_candidates:
            if not os.path.exists(tp):
                continue
            try:
                text_models[label] = KerasTextWrapper(mp, tp, max_len=max_len)
                return
            except Exception:
                continue

_try_add_text_model(
    "CNN",
    model_candidates=[
        "cnn_model_smote_fixed.h5", "cnn_model_smote.h5",
        "cnn_model_smote_fixed.keras", "cnn_model_smote.keras"
    ],
    tok_candidates=[
        "cnn_tokenizer.pkl", "tokenizer_smote.pkl", "tokenizer_smote.json"
    ],
    max_len=200
)

_try_add_text_model(
    "BiLSTM",
    model_candidates=[
        "lstm_model_smote_fixed.h5", "lstm_model_smote.h5",
        "lstm_model_smote_fixed.keras", "lstm_model_smote.keras"
    ],
    tok_candidates=[
        "bilstm_tokenizer.pkl", "tokenizer_smote.pkl", "tokenizer_smote.json"
    ],
    max_len=200
)

analyzer = SentimentIntensityAnalyzer()

# ---------- Persisted label map ----------
ALIGN_PATH = pathlib.Path("label_map.json")
if "label_maps" not in st.session_state:
    def build_label_maps():
        lm = {n: auto_align_label(m, tfidf_vectorizer) for n, m in models.items()}
        for n, tm in text_models.items():
            lm[n] = auto_align_label_textmodel(tm)
        return lm

    if ALIGN_PATH.exists():
        try:
            st.session_state["label_maps"] = json.loads(ALIGN_PATH.read_text())
            for n in models:
                if n not in st.session_state["label_maps"]:
                    st.session_state["label_maps"][n] = auto_align_label(models[n], tfidf_vectorizer)
            for n in text_models:
                if n not in st.session_state["label_maps"]:
                    st.session_state["label_maps"][n] = auto_align_label_textmodel(text_models[n])
            ALIGN_PATH.write_text(json.dumps(st.session_state["label_maps"], indent=2))
        except Exception:
            st.session_state["label_maps"] = build_label_maps()
            try:
                ALIGN_PATH.write_text(json.dumps(st.session_state["label_maps"], indent=2))
            except Exception:
                pass
    else:
        st.session_state["label_maps"] = build_label_maps()
        try:
            ALIGN_PATH.write_text(json.dumps(st.session_state["label_maps"], indent=2))
        except Exception:
            pass

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
            except Exception:
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
        is_bert_neg_strong  = (bert_label.upper() == "NEGATIVE" and bert_score >= 0.90)
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

    #---------- Deep Learning subset table ----------
    dl_names = [n for n in ["CNN","BiLSTM"] if n in per_probs]
    if dl_names:
        st.subheader("🧪 DL Models — probability, threshold & decision")
        rows = []
        for name in dl_names:
            p   = float(per_probs.get(name, 0.0))
            thr = float(per_thrs.get(name, 0.5))
            rows.append({
                "Model": name,
                "p_dep": round(p, 4),
                "threshold": round(thr, 4),
                "Decision": "Depressed 😞" if p >= thr else "Not Depressed 😊"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Deep Learning module: not enabled in this deployment environment. The app runs in ML-only inference mode for maximum compatibility.")

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



"""
Skin Lesion Analysis Dashboard
"""

import os, time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Class definitions (alphabetical indices 0-6)
# ──────────────────────────────────────────────────────────────
CLASSES = {
    0: {"code":"akiec","name":"Actinic Keratosis",    "full":"Actinic Keratosis / ISIC",       "risk":"Pre-Cancer","color":"#f472b6","badge":"pre"},
    1: {"code":"bcc",  "name":"Basal Cell Carcinoma","full":"Basal Cell Carcinoma",            "risk":"Malignant", "color":"#fb923c","badge":"mal"},
    2: {"code":"bkl",  "name":"Benign Keratosis",    "full":"Benign Keratosis-like Lesion",    "risk":"Benign",    "color":"#60a5fa","badge":"ben"},
    3: {"code":"df",   "name":"Dermatofibroma",       "full":"Dermatofibroma",                 "risk":"Benign",    "color":"#34d399","badge":"ben"},
    4: {"code":"mel",  "name":"Melanoma",             "full":"Melanoma",                       "risk":"Malignant", "color":"#f87171","badge":"mal"},
    5: {"code":"nv",   "name":"Melanocytic Nevi",     "full":"Melanocytic Nevi (Mole)",         "risk":"Benign",    "color":"#4ade80","badge":"ben"},
    6: {"code":"vasc", "name":"Vascular Lesion",      "full":"Vascular Lesion",                "risk":"Other",     "color":"#a78bfa","badge":"oth"},
}
MALIGNANT_IDX = [0, 1, 4]   # akiec, bcc, mel

import json
import uuid
from pathlib import Path

from db import insert_prediction, UPLOADS_DIR
from history import render_history_page


def inject_css():
    """Apply the custom CSS theme."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:      #F1E2D1;   /* cream            - page background  */
  --sur:     #faf0e2;   /* lighter cream    - cards / sidebar  */
  --bdr:     #DCC3AA;   /* sand             - subtle borders   */
  --bdr2:    #c7a888;   /* darker sand      - stronger borders */
  --txt:     #541A1A;   /* oxblood          - body text        */
  --mut:     #8b4a4a;   /* faded burgundy   - secondary text   */
  --acc:     #810B38;   /* wine             - buttons / accent */
  --warn:    #fb923c;   /* preserved for clinical semantics */
  --danger:  #f87171;   /* preserved for clinical semantics */
  --safe:    #4ade80;   /* preserved for clinical semantics */
  --fd:      'Syne', sans-serif;
  --fb:      'Instrument Sans', sans-serif;
  --fm:      'JetBrains Mono', monospace;
}
html,body,[class*="css"]         { font-family:var(--fb); color:var(--txt); }
.stApp                           { background:var(--bg); }
[data-testid="stSidebar"]        { background:var(--sur)!important; border-right:1px solid var(--bdr)!important; }
[data-testid="stSidebar"] *:not(span) { font-family:var(--fb)!important; }
[data-testid="stSidebarCollapseIcon"] {
    font-family: 'Material Icons' !important;
}
.stButton>button                 { background:var(--acc)!important; color:#fff!important; border:none!important; border-radius:8px!important; font-family:var(--fb)!important; font-weight:500!important; letter-spacing:.03em!important; padding:.58rem 1.8rem!important; transition:filter .18s!important; }
.stButton>button:hover           { filter:brightness(1.15)!important; }
.stTabs [data-baseweb="tab-list"]{ gap:.2rem; border-bottom:1px solid var(--bdr)!important; }
.stTabs [data-baseweb="tab"]     { font-family:var(--fb); font-size:.84rem; color:var(--mut)!important; padding:.55rem 1.1rem!important; }
.stTabs [aria-selected="true"]   { color:var(--txt)!important; border-bottom:2px solid var(--acc)!important; }
[data-testid="stFileUploadDropzone"]{ background:var(--sur)!important; border:2px dashed var(--bdr2)!important; border-radius:12px!important; }
[data-testid="stMetric"]         { background:var(--sur); border:1px solid var(--bdr); border-radius:10px; padding:.9rem 1rem .7rem; }
[data-testid="stMetricLabel"]    { font-size:.73rem!important; color:var(--mut)!important; }
[data-testid="stMetricValue"]    { font-family:var(--fm)!important; font-size:1.5rem!important; }

.card   { background:var(--sur); border:1px solid var(--bdr); border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:.9rem; }
.card-hi{ background:linear-gradient(135deg,#faf0e2,#F1E2D1); border:1px solid var(--bdr2); border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:.9rem; }
.lbl    { font-size:.67rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--mut); margin-bottom:.6rem; }

.top-name  { font-family:'Syne',sans-serif; font-size:1.85rem; font-weight:700; line-height:1.1; }
.rbadge    { display:inline-block; padding:.22rem .8rem; border-radius:99px; font-size:.68rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-top:.45rem; }
.r-mal     { background:rgba(248,113,113,.12); color:#f87171; border:1px solid rgba(248,113,113,.28); }
.r-pre     { background:rgba(251,146,60,.10);  color:#fb923c; border:1px solid rgba(251,146,60,.25); }
.r-ben     { background:rgba(74,222,128,.10);  color:#4ade80; border:1px solid rgba(74,222,128,.22); }
.r-oth     { background:rgba(167,139,250,.10); color:#a78bfa; border:1px solid rgba(167,139,250,.22); }

.bar-row  { display:flex; align-items:center; gap:.75rem; margin-bottom:.5rem; }
.bar-name { font-size:.8rem; color:#a98467; width:188px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar-bg   { flex:1; height:6px; background:#DCC3AA; border-radius:99px; overflow:hidden; }
.bar-fill { height:100%; border-radius:99px; }
.bar-pct  { font-family:var(--fm); font-size:.76rem; color:#a98467; width:42px; text-align:right; }

.rbox    { text-align:center; border-radius:12px; padding:1.1rem .8rem; }
.rttl    { font-size:.67rem; letter-spacing:.15em; text-transform:uppercase; color:var(--mut); margin-bottom:.35rem; }
.rval    { font-family:'Syne',sans-serif; font-size:1.65rem; font-weight:700; }
.rsub    { font-size:.74rem; color:#a98467; margin-top:.25rem; }

.disc    { background:rgba(251,146,60,.05); border:1px solid rgba(251,146,60,.18); border-radius:10px; padding:.85rem 1.15rem; font-size:.77rem; color:#a98467; line-height:1.65; }
.disc strong { color:var(--warn); }

.itbl    { width:100%; border-collapse:collapse; }
.itbl td { padding:.5rem .3rem; font-size:.82rem; border-bottom:1px solid var(--bdr); }
.itbl td:first-child { color:var(--mut); width:155px; }
.itbl td:last-child  { color:var(--txt); font-family:var(--fm); font-size:.76rem; }

/* ─── User-requested overrides ─── */

/* Login form labels (Username / Password / Confirm) → oxblood */
.stTextInput label,
.stTextInput [data-testid="stWidgetLabel"] p {
    color: #541A1A !important;
    font-weight: 600 !important;
}

/* Checkbox labels (Demo mode, Show all 7..., Show risk...) → black */
.stCheckbox label,
.stCheckbox [data-testid="stWidgetLabel"] p {
    color: #000000 !important;
    font-weight: 500 !important;
}
                
 /* Dark metric text */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #000000 !important;
}

[data-testid="stMetricDelta"] > div {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}

/* Dark caption text */
[data-testid="stCaptionContainer"],
.stCaption {
    color: #000000 !important;
}

/* ─── Streamlit header: hover to reveal ─── */
header[data-testid="stHeader"] {
    opacity: 0 !important;
    transition: opacity 0.3s ease !important;
    pointer-events: none;
}
header[data-testid="stHeader"]:hover {
    opacity: 1 !important;
    pointer-events: auto;
}

/* ─── Themed dataframe / tablo ─── */
.themed-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--fb);
    font-size: .82rem;
    background: var(--sur);
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--bdr);
}
.themed-table th {
    background: #F1E2D1;
    color: #541A1A;
    font-weight: 600;
    font-size: .73rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    padding: .65rem .8rem;
    border-bottom: 1px solid var(--bdr2);
    text-align: left;
}
.themed-table td {
    padding: .55rem .8rem;
    color: #541A1A;
    border-bottom: 1px solid var(--bdr);
}
.themed-table tr:last-child td {
    border-bottom: none;
}
.themed-table tr:hover td {
    background: rgba(129,11,56,.04);
}
.themed-table .winner {
    color: #4ade80;
    font-weight: 700;
}               
</style>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# MODEL LOADER
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path), True
    except Exception:
        return None, False


def preprocess(img: Image.Image) -> np.ndarray:
    """Resize image for model input."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)          # (1,224,224,3)


def predict(model, img: Image.Image, age: float, sex: str, localization: str) -> np.ndarray:
    """Run multi-input inference."""
    import tensorflow as tf
    img_tensor = tf.convert_to_tensor(preprocess(img), dtype=tf.float32)
    age_tensor = tf.convert_to_tensor([[float(age)]], dtype=tf.float32)
    sex_tensor = tf.convert_to_tensor([[sex]], dtype=tf.string)
    loc_tensor = tf.convert_to_tensor([[localization]], dtype=tf.string)
    preds = model.predict({
        'img_in': img_tensor,
        'age_in': age_tensor,
        'sex_in': sex_tensor,
        'loc_in': loc_tensor,
    }, verbose=0)
    return preds[0].astype(np.float32)


def demo_predict(filename: str) -> np.ndarray:
    seed = int(np.frombuffer(filename.encode(), dtype=np.uint8).sum()) % 9999
    rng  = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(7) * 0.45).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# RENDER HELPERS
# ──────────────────────────────────────────────────────────────
BADGE = {"mal":"r-mal","pre":"r-pre","ben":"r-ben","oth":"r-oth"}

def render_top_pred(probs):
    idx = int(np.argmax(probs))
    cls = CLASSES[idx]
    conf = probs[idx]*100
    if conf>=70:   tier,tc = "High confidence",    "#4ade80"
    elif conf>=45: tier,tc = "Moderate confidence","#fb923c"
    else:          tier,tc = "Low confidence",     "#f87171"
    st.markdown(f"""
    <div class="card-hi">
      <div class="lbl">Top Prediction</div>
      <div class="top-name" style="color:{cls['color']}">{cls['name']}</div>
      <div>
        <span class="rbadge {BADGE[cls['badge']]}">{cls['risk']}</span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:.8rem;color:{tc};margin-left:.7rem">
          {conf:.1f}% — {tier}
        </span>
      </div>
      <div style="font-size:.78rem;color:var(--mut);margin-top:.7rem">{cls['full']}</div>
    </div>""", unsafe_allow_html=True)


def render_prob_bars(probs):
    order = np.argsort(probs)[::-1]
    html  = ""
    for rank, i in enumerate(order):
        cls = CLASSES[i]; p = probs[i]*100
        op  = max(0.28, 1.0-rank*0.11)
        html += f"""
        <div class="bar-row" style="opacity:{op:.2f}">
          <div class="bar-name">{cls['name']}</div>
          <div class="bar-bg"><div class="bar-fill" style="width:{p:.1f}%;background:{cls['color']}"></div></div>
          <div class="bar-pct">{p:.1f}%</div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_risk(probs, thresh):
    mal  = float(np.sum(probs[MALIGNANT_IDX]))*100
    ent  = -float(np.sum(probs*np.log(probs+1e-9)))
    unc  = ent/np.log(7)*100
    if mal>=50:   rl,rc,rb = "HIGH",    "#f87171","background:rgba(248,113,113,.07);border:1px solid rgba(248,113,113,.2)"
    elif mal>=25: rl,rc,rb = "MODERATE","#fb923c","background:rgba(251,146,60,.07);border:1px solid rgba(251,146,60,.2)"
    else:         rl,rc,rb = "LOW",     "#4ade80","background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.2)"
    uc = "#f87171" if unc>65 else ("#fb923c" if unc>40 else "#4ade80")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="rbox" style="{rb}">
          <div class="rttl">Malignancy Risk</div>
          <div class="rval" style="color:{rc}">{rl}</div>
          <div class="rsub">{mal:.1f}% combined (akiec+bcc+mel)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="rbox" style="background:rgba(91,140,255,.06);border:1px solid rgba(91,140,255,.18)">
          <div class="rttl">Prediction Uncertainty</div>
          <div class="rval" style="color:{uc}">{unc:.0f}%</div>
          <div class="rsub">Shannon entropy / max-entropy</div>
        </div>""", unsafe_allow_html=True)
    return mal/100


def render_img_stats(img, file):
    arr = np.array(img.convert("RGB"))
    st.markdown(f"""
    <div class="card">
      <div class="lbl">Image Statistics</div>
      <table class="itbl">
        <tr><td>File size</td><td>{file.size/1024:.1f} KB</td></tr>
        <tr><td>Original size</td><td>{img.size[0]} × {img.size[1]} px</td></tr>
        <tr><td>Model input</td><td>224 × 224 × 3 (auto-resized)</td></tr>
        <tr><td>Mean brightness</td><td>{arr.mean():.1f} / 255</td></tr>
        <tr><td>Std deviation</td><td>{arr.std():.1f}</td></tr>
        <tr><td>Min / Max pixel</td><td>{arr.min()} / {arr.max()}</td></tr>
      </table>
    </div>""", unsafe_allow_html=True)


def _persist_prediction(user, uploaded_file, image, probs):
    """Save prediction to history."""
    try:
        user_dir = UPLOADS_DIR / str(user['id'])
        user_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in {'.jpg', '.jpeg', '.png'}:
            ext = '.png'
        dest = user_dir / f'{uuid.uuid4().hex}{ext}'
        image.convert('RGB').save(dest)
        idx = int(np.argmax(probs))
        insert_prediction(
            user_id=user['id'],
            image_path=str(dest),
            original_filename=uploaded_file.name,
            predicted_class=CLASSES[idx]['name'],
            confidence=float(np.max(probs) * 100),
            probabilities_json=json.dumps(
                {CLASSES[i]['name']: float(probs[i]) for i in CLASSES}
            ),
        )
    except Exception as exc:
        st.warning(f'Result was not saved to history: {exc}')


def render_dashboard(user):
    with st.sidebar:
        st.markdown("""
        <div style="padding:.4rem 0 1.6rem">
          <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#541A1A;letter-spacing:-.02em">Skin Lesion Classifier</div>
          <div style="font-size:.67rem;color:#8b4a4a;letter-spacing:.15em;text-transform:uppercase;margin-top:2px">BAU 2026 Capstone Project</div>
        </div>""", unsafe_allow_html=True)

        # Logged-in user card + logout
        st.markdown(
            f"""<div class="card" style="padding:.8rem 1rem;margin-bottom:1rem">
              <div class="lbl" style="margin-bottom:.25rem">Signed in as</div>
              <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:700;color:#541A1A">
                {user['username']}
              </div>
              <div style="font-size:.7rem;color:var(--mut)">User ID: {user['id']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("---")

        st.markdown('<div class="lbl">Model File</div>', unsafe_allow_html=True)
        model_path = st.text_input("Path", value="best_skin_model.keras",
                                   label_visibility="collapsed",
                                   help="Relative or absolute path to your .keras model file")

        st.markdown('<div class="lbl" style="margin-top:1rem">Mode</div>', unsafe_allow_html=True)
        demo_mode = st.checkbox("Demo mode (no model needed)",
                                value=not os.path.exists(model_path),
                                help="Simulated inference. Disable when best_skin_model.keras is present.")

        st.markdown('<div class="lbl" style="margin-top:1rem">Safety Threshold</div>', unsafe_allow_html=True)
        thresh = st.slider("Malignancy alert at", 0.15, 0.60, 0.30, 0.05,
                           label_visibility="collapsed",
                           help="Alert fires when akiec+bcc+mel combined probability ≥ this value.")
        st.markdown(
            f"<div style='color:#000000;font-size:.78rem;margin-top:.4rem'>"
            f"Current: {thresh*100:.0f}% &nbsp;|&nbsp; Lower = fewer missed cancers (higher recall)"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="lbl" style="margin-top:1rem">Display</div>', unsafe_allow_html=True)
        show_bars = st.checkbox("Show all 7 class probabilities", True)
        show_risk = st.checkbox("Show risk & uncertainty panel",  True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:.71rem;color:#8b4a4a;line-height:1.9">
          <b style="color:#8b4a4a">Architecture</b><br>
          EfficientNetB2 + Late Fusion<br>Multi-branch (img + age + sex + loc)<br>Dense(128) → Dense(32) → Dropout(0.42) → Softmax(7)<br><br>
          <b style="color:#8b4a4a">Training</b><br>
          20 epochs · batch 32 · Adam (lr=2.45e-4)<br>Categorical Focal Loss (α=0.25, γ=2.433)<br>EarlyStopping → best weights @ Epoch 10<br><br>
          <b style="color:#8b4a4a">Dataset</b><br>
          HAM10000 (10,015 images)<br>mixed_float16 · Tesla P100 GPU<br><br>
          <b style="color:#8b4a4a">University</b><br>
          Bahçeşehir University, Istanbul
        </div>""", unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────
    # HEADER
    # ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:.5rem 0 .2rem">
      <span style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;letter-spacing:-.03em;color:#541A1A">
        Skin Lesion Analysis
      </span>
    </div>
    <div style="font-size:.78rem;color:#8b4a4a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.4rem">
      AI Decision Support System &nbsp;·&nbsp; Capstone 2026 &nbsp;·&nbsp; Bahçeşehir University
    </div>
    <div class="disc">
      <strong>⚠ Disclaimer:</strong>
      This tool is for <strong>educational and research purposes only</strong>.
      It does <strong>not replace professional medical advice</strong>.
      Always consult a qualified dermatologist. Model outputs are probabilistic
      estimates and must not be used as clinical diagnoses.
    </div><br>
    """, unsafe_allow_html=True)


    # ──────────────────────────────────────────────────────────────
    # TABS
    # ──────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["Analyze", "Metrics & Trade-offs", "Model Docs",
         "Class Guide", "ℹ  About", "🕑 My History"]
    )


    # ══════════════════════════════════════════════════════════════
    # TAB 1 — ANALYZE
    # ══════════════════════════════════════════════════════════════
    with t1:
        left, right = st.columns([1, 1], gap="large")

        with left:
            st.markdown('<div class="lbl">Upload Dermoscopic Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("JPG / PNG", type=["jpg","jpeg","png"],
                                        label_visibility="collapsed")
            if uploaded:
                image = Image.open(uploaded)
                st.image(image, caption=f"{uploaded.name}  ·  {image.size[0]}×{image.size[1]}",
                         use_container_width=True)
                render_img_stats(image, uploaded)

                st.markdown('<div class="lbl" style="margin-top:1rem">Patient Information</div>', unsafe_allow_html=True)
                _pc1, _pc2, _pc3 = st.columns(3)
                with _pc1:
                    age = st.number_input("Age", min_value=0, max_value=120, value=50,
                                          help="Patient's age")
                with _pc2:
                    sex = st.selectbox("Sex", options=["male", "female"],
                                       help="Patient's biological sex")
                with _pc3:
                    localization = st.selectbox("Localization", options=[
                        "back", "lower extremity", "trunk", "upper extremity",
                        "abdomen", "face", "chest", "foot", "unknown",
                        "neck", "scalp", "hand", "ear", "genital", "acral"
                    ], help="Lesion location on the body")

                run_btn = st.button("▶  Run Analysis", use_container_width=True)
            else:
                st.markdown("""
                <div class="card" style="text-align:center;padding:3.5rem 1rem">
                  <div style="font-size:2.8rem;margin-bottom:.9rem"></div>
                  <div style="font-family:'Syne',sans-serif;font-size:1.05rem;color:#8b4a4a">
                    Upload a dermoscopic image<br>to begin analysis
                  </div>
                  <div style="font-size:.75rem;margin-top:.65rem;color:#DCC3AA">
                    JPG · PNG &nbsp;|&nbsp; Any resolution — auto-resized to 224×224
                  </div>
                </div>""", unsafe_allow_html=True)
                run_btn = False

        with right:
            st.markdown('<div class="lbl">Results</div>', unsafe_allow_html=True)

            if uploaded and run_btn:
                with st.spinner("Running inference…"):
                    time.sleep(0.5)
                    if demo_mode:
                        probs     = demo_predict(uploaded.name)
                        src_label = "⚡ Demo mode — simulated output"
                    else:
                        model, ok = load_model(model_path)
                        if ok:
                            probs     = predict(model, image, age, sex, localization)
                            src_label = f"Loaded: {model_path}"
                        else:
                            st.error(f"Could not load **{model_path}**. Enable Demo mode or fix the path.", icon="🚨")
                            st.stop()

                # ── OOD gate: reject images with scattered confidence ──
                max_conf = float(np.max(probs))
                if max_conf < 0.45:
                    st.error(
                        "**System Alert:** Unable to verify the uploaded image as a "
                        "characteristic skin lesion. The model's confidence scores are "
                        "too scattered. Please upload a clear, well-focused, and valid "
                        "image of the affected skin area.",
                        icon="⚠️"
                    )
                    st.markdown(f"""
                    <div style="font-size:.69rem;color:#8b4a4a;margin-top:1rem">
                      {src_label} &nbsp;·&nbsp; max confidence {max_conf*100:.1f}% &lt; 45% threshold
                    </div>""", unsafe_allow_html=True)
                else:
                    render_top_pred(probs)

                    # Persist only real inferences — demo output is fake and
                    # would pollute the user's history.
                    if not demo_mode:
                        _persist_prediction(user, uploaded, image, probs)

                    mal_sum = float(np.sum(probs[MALIGNANT_IDX]))
                    if mal_sum >= thresh:
                        st.warning(
                            f"⚠ **Malignancy alert:** combined akiec+bcc+mel probability is "
                            f"**{mal_sum*100:.1f}%** — exceeds your {thresh*100:.0f}% threshold. "
                            "Clinical review recommended."
                        )

                    if show_risk:
                        st.markdown("<br>", unsafe_allow_html=True)
                        render_risk(probs, thresh)

                    if show_bars:
                        st.markdown("<br><div class='lbl'>Class Probabilities</div>", unsafe_allow_html=True)
                        render_prob_bars(probs)

                    st.markdown(f"""
                    <div style="font-size:.69rem;color:#8b4a4a;margin-top:1rem">
                      {src_label} &nbsp;·&nbsp; input 224×224 + metadata &nbsp;·&nbsp; softmax(7)
                    </div>""", unsafe_allow_html=True)

            elif not uploaded:
                st.markdown("""
                <div class="card" style="text-align:center;padding:4rem 1rem">
                  <div style="font-size:2rem;margin-bottom:.7rem"></div>
                  <div style="color:#8b4a4a;font-size:.88rem">Analysis results will appear here</div>
                </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════
    # TAB 2 — METRICS & TRADE-OFFS
    # ══════════════════════════════════════════════════════════════
    with t2:
        st.markdown('<div class="lbl">Performance Metrics (Validation Set)</div>',
                    unsafe_allow_html=True)

        cols = st.columns(6)
        metrics = [
            ("Val Accuracy", "84.8%",  "+8.5% vs CNN baseline"),
            ("AUC (mel)",    "0.920",  "Critical melanoma detection"),
            ("AUC (bcc)",    "0.981",  "Basal cell carcinoma"),
            ("AUC (vasc)",   "0.997",  "Vascular lesions"),
            ("AUC (df)",     "0.998",  "Dermatofibroma"),
            ("Focal γ",      "2.433",  "Optuna-optimized"),
        ]
        for col, (name, val, delta) in zip(cols, metrics):
            with col:
                st.metric(name, val, delta)

        st.markdown("<br><div class='lbl'>Multi-Objective Pareto Analysis</div>",
                    unsafe_allow_html=True)
        tradeoff = pd.DataFrame({
            "Model":           ["Custom CNN v1 (Baseline)", "EfficientNetB2 (Single-modal)", "EfficientNetB2 (Multi-modal + Focal) ★"],
            "Val Accuracy":    ["~76.4%",  "~81.1%", "84.3–85.3%"],
            "AUC Melanoma":    ["~0.850",  "~0.895", "0.920"],
            "Loss":            ["CrossEntropy", "CrossEntropy", "Focal (α=.25, γ=2.43)"],
            "Precision":       ["float32",  "float32", "mixed_float16"],
            "Pareto Rank":     ["3",       "2",      "1 ★"],
        })
        html_t = tradeoff.to_html(index=False, escape=False, classes="themed-table")
        html_t = html_t.replace("★", '<span class="winner">★</span>')
        st.markdown(html_t, unsafe_allow_html=True)

        st.markdown("<br><div class='lbl'>Optuna Hyperparameter Importance (7 Trials)</div>",
                    unsafe_allow_html=True)
        optuna_df = pd.DataFrame({
            "Parameter":       ["γ (Focal Loss)", "Learning Rate", "Dropout Rate", "Img Dense", "Meta Dense"],
            "Optimal Value":   ["2.433", "0.000245", "0.419", "128", "32"],
            "Importance":      ["38%", "30%", "Moderate", "Low", "Low"],
            "Description":     ["Focusing exponent", "Adam step size", "Stochastic regularization", "Image embedding dim", "Metadata vector dim"],
        })
        st.markdown(optuna_df.to_html(index=False, escape=False, classes="themed-table"),
                    unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════
    # TAB 3 — MODEL DOCS
    # ══════════════════════════════════════════════════════════════
    with t3:
        col_l, col_r = st.columns([1,1], gap="large")

        with col_l:
            st.markdown('<div class="lbl">Architecture — best_skin_model.keras</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="card"><table class="itbl">
              <tr><td>Type</td><td>EfficientNetB2 + Late-Fusion Multi-Modal</td></tr>
              <tr><td>Image input</td><td>(224, 224, 3) — RGB float32</td></tr>
              <tr><td>Meta inputs</td><td>age (float32) · sex (string) · localization (string)</td></tr>
              <tr><td>Backbone</td><td>EfficientNetB2 (ImageNet pre-trained)</td></tr>
              <tr><td>Img head</td><td>GlobalAvgPool → Dense(128, relu)</td></tr>
              <tr><td>Meta head</td><td>Dense(32, relu)</td></tr>
              <tr><td>Fusion</td><td>Concatenate → Dropout(0.419) → Dense(7, softmax)</td></tr>
              <tr><td>Optimizer</td><td>Adam (lr=0.000245)</td></tr>
              <tr><td>Loss</td><td>Categorical Focal Loss (α=0.25, γ=2.433)</td></tr>
              <tr><td>Epochs</td><td>20 max &nbsp;·&nbsp; EarlyStopping(patience=5) &nbsp;·&nbsp; best @ Epoch 10</td></tr>
              <tr><td>Precision</td><td>mixed_float16 &nbsp;·&nbsp; batch 32 &nbsp;·&nbsp; Tesla P100 GPU</td></tr>
              <tr><td>Val accuracy</td><td>84.32% – 85.32%</td></tr>
            </table></div>""", unsafe_allow_html=True)

            st.markdown('<div class="lbl">Class Encoding (LabelEncoder alphabetical)</div>',
                        unsafe_allow_html=True)
            rows = "".join(f"<tr><td>Index {i}</td><td>{c['code']} — {c['name']}</td></tr>"
                           for i, c in CLASSES.items())
            st.markdown(f'<div class="card"><table class="itbl">{rows}</table></div>',
                        unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="lbl">Preprocessing Pipeline (Multi-Input)</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="card"><table class="itbl">
              <tr><td>img_in</td><td>PIL → RGB → resize(224,224) → float32 → (1,224,224,3)</td></tr>
              <tr><td>age_in</td><td>tf.float32 tensor → shape (1,1)</td></tr>
              <tr><td>sex_in</td><td>tf.string tensor → shape (1,1) · "male" / "female"</td></tr>
              <tr><td>loc_in</td><td>tf.string tensor → shape (1,1) · 15 anatomical sites</td></tr>
              <tr><td>Pipeline</td><td>tf.data API · prefetch(AUTOTUNE) · mixed_float16</td></tr>
            </table></div>""", unsafe_allow_html=True)

            st.markdown('<div class="lbl">Grad-CAM Explainability Insights</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card"><table class="itbl">
              <tr><td>Melanoma</td><td>Targets asymmetric margin boundaries (ABCD criteria)</td></tr>
              <tr><td>BCC</td><td>Dual-focus on translucent rolled borders, ignores center</td></tr>
              <tr><td>df / bkl</td><td>Locks onto central nodular core masses</td></tr>
              <tr><td>Edge artifact</td><td>EfficientNetB2 zero-padding causes false corner activations</td></tr>
              <tr><td>Grid effect</td><td>7×7 top_activation upsampled to 224×224 → blocky maps</td></tr>
            </table></div>""", unsafe_allow_html=True)

            st.markdown('<div class="lbl">ROC-AUC Performance</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card"><table class="itbl">
              <tr><td>Melanoma (mel)</td><td>AUC 0.920</td></tr>
              <tr><td>BCC (bcc)</td><td>AUC 0.981</td></tr>
              <tr><td>Vascular (vasc)</td><td>AUC 0.997</td></tr>
              <tr><td>Dermatofibroma (df)</td><td>AUC 0.998</td></tr>
            </table></div>""", unsafe_allow_html=True)

            st.markdown('<div class="lbl">Load & Run Snippet</div>', unsafe_allow_html=True)
            st.code("""
    import tensorflow as tf, numpy as np
    from PIL import Image

    model = tf.keras.models.load_model("best_skin_model.keras")

    img = Image.open("lesion.jpg").convert("RGB").resize((224,224))
    inputs = {
        'img_in': tf.convert_to_tensor(
            np.expand_dims(np.array(img, dtype="float32"), 0)),
        'age_in': tf.constant([[55.0]], dtype=tf.float32),
        'sex_in': tf.constant([["male"]]),
        'loc_in': tf.constant([["back"]]),
    }
    probs = model.predict(inputs)[0]   # shape: (7,)
    idx   = int(np.argmax(probs))

    classes = {0:"akiec",1:"bcc",2:"bkl",3:"df",4:"mel",5:"nv",6:"vasc"}
    print(f"{classes[idx]}  {probs[idx]*100:.1f}%")
    """, language="python")


    # ══════════════════════════════════════════════════════════════
    # TAB 4 — CLASS GUIDE
    # ══════════════════════════════════════════════════════════════
    with t4:
        st.markdown('<div class="lbl">HAM10000 — 7 Lesion Categories</div>', unsafe_allow_html=True)

        detail = {
            0: ("~3.3%","UV-exposed areas (face, scalp, hands)",
                "Precancerous lesion caused by prolonged UV exposure. Can progress to squamous cell carcinoma. Presents as rough, scaly patches."),
            1: ("~5.1%","Head, neck, trunk",
                "Most common malignant skin tumor. Locally invasive but rarely metastasizes. Pearl-like nodules or ulcerated plaques."),
            2: ("~11.0%","Trunk, extremities",
                "Includes seborrheic keratoses and solar lentigines. Benign but can closely mimic melanoma — important to distinguish."),
            3: ("~1.1%","Lower extremities",
                "Benign fibrous nodule, usually firm and slightly raised. Shows characteristic dimple sign when pinched."),
            4: ("~11.1%","Any body surface",
                "Most dangerous skin cancer. Arises from melanocytes. Early detection is critical — survival drops sharply with delayed diagnosis."),
            5: ("~67.0%","Any body surface",
                "Common benign moles. Majority class in HAM10000. Monitor for ABCD changes (Asymmetry, Border, Color, Diameter)."),
            6: ("~1.4%","Lower extremities, face",
                "Benign vascular proliferations including angiomas and pyogenic granulomas. Distinctive blood-vessel pattern under dermoscopy."),
        }

        for i, cls in CLASSES.items():
            freq, site, desc = detail[i]
            st.markdown(f"""
            <div class="card" style="border-left:3px solid {cls['color']}">
              <div style="display:flex;align-items:center;gap:.75rem;margin-bottom:.45rem">
                <span style="font-family:'Syne',sans-serif;font-size:1.08rem;font-weight:700;color:{cls['color']}">{cls['name']}</span>
                <span class="rbadge {BADGE[cls['badge']]}">{cls['risk']}</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mut)">[{cls['code']}]</span>
                <span style="font-size:.72rem;color:var(--mut);margin-left:auto">HAM10000: {freq}</span>
              </div>
              <div style="font-size:.8rem;color:#a98467;margin-bottom:.3rem">{site}</div>
              <div style="font-size:.83rem;color:#a98467;line-height:1.65">{desc}</div>
            </div>""", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════
    # TAB 5 — ABOUT
    # ══════════════════════════════════════════════════════════════
    with t5:
        cl, cr = st.columns([1,1], gap="large")

        with cl:
            st.markdown("""
            <div class="card">
              <div class="lbl">Project</div>
              <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#541A1A;margin-bottom:.7rem">
                Skin Cancer Diagnosis &amp; Clinical<br>Decision Support System
              </div>
              <div style="font-size:.83rem;color:#a98467;line-height:1.75">
                A multimodal deep learning framework fusing
                <b style="color:#541A1A">EfficientNetB2</b> visual features with patient
                clinical metadata (age, sex, 15 anatomical localizations) via
                <b style="color:#541A1A">late-fusion</b>.
                Trained with <b style="color:#541A1A">Categorical Focal Loss</b>
                (α=0.25, γ=2.433) optimized by Optuna, achieving
                84–85% validation accuracy and 0.920 AUC for melanoma detection.
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
              <div class="lbl">Team</div>
              <table class="itbl">
                <tr><td>Ufuk Çiğdem</td><td>Industrial Engineering</td></tr>
                <tr><td>Deren Pazvant</td><td>Industrial Engineering</td></tr>
                <tr><td>Ayşe Soydal</td><td>Industrial Engineering</td></tr>
                <tr><td>Emine Hazal Soydal</td><td>Software Engineering</td></tr>
                <tr><td>Aygül Umur</td><td>Software Engineering</td></tr>
                <tr><td>Berat Uzun</td><td>Software Engineering</td></tr>
              </table>
              <div style="margin-top:.8rem;font-size:.75rem;color:var(--mut)">
                Advisors: Assist. Prof. Serkan ŞİMŞEK &nbsp;·&nbsp; Assist. Prof. Zühal ÖZCAN YAVUZ
              </div>
            </div>""", unsafe_allow_html=True)

        with cr:
            st.markdown("""
            <div class="card">
              <div class="lbl">Work Packages</div>
              <table class="itbl">
                <tr><td>WP1</td><td>Literature Review, Problem Definition</td></tr>
                <tr><td>WP2</td><td>HAM10000 Dataset Collection and Understanding</td></tr>
                <tr><td>WP3</td><td>Data Preprocessing and Exploratory Data Analysis</td></tr>
                <tr><td>WP4</td><td>CNN & Transfer Learning Model Development</td></tr>
                <tr><td>WP5</td><td>Hyperparameter Optimization (Optuna)</td></tr>
                <tr><td>WP6</td><td>Uncertainty, Robustness, Calibration and Dashboard</td></tr>
                <tr><td>WP7</td><td>Model Evaluation</td></tr>
                <tr><td>WP8</td><td>Report Writing, Documentation and Final Presentation</td></tr>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
              <div class="lbl">Key References</div>
              <div style="font-size:.77rem;color:#a98467;line-height:1.95">
                Esteva et al. <em>Nature</em> 2017 — CNN skin cancer classification<br>
                Tschandl et al. <em>Scientific Data</em> 2019 — HAM10000 dataset<br>
                Tan &amp; Le. <em>ICML</em> 2019 — EfficientNet compound scaling<br>
                Lin et al. <em>ICCV</em> 2017 — Focal Loss for dense detection<br>
                Akiba et al. <em>KDD</em> 2019 — Optuna hyperparameter optimization<br>
                Selvaraju et al. <em>ICCV</em> 2017 — Grad-CAM visual explanations<br>
                Guo et al. <em>ICML</em> 2017 — Neural network calibration
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="disc">
          <strong>Research &amp; Education Use Only.</strong>
          Patient data from HAM10000 is fully anonymized.
          Uploaded images and results are stored privately per authenticated
          user and are only visible to the account that created them.
          It has not undergone clinical validation and must not be used for medical decisions.
        </div>""", unsafe_allow_html=True)


    # ═════════════════════════════════════════════════════════════
    # TAB 6 — MY HISTORY
    # ═════════════════════════════════════════════════════════════
    with t6:
        render_history_page(user)

"""
╔══════════════════════════════════════════════════════════════╗
║  AI-Based Skin Lesion Analysis Dashboard                     ║
║  Bahçeşehir University — Capstone Project 2026               ║
║                                                              ║
║  Compatible with: skin_cancer_model_v1.keras                 ║
║  Input shape   : (128, 128, 3)  — normalized /255            ║
║  Output classes: 7  (softmax)                                ║
║                                                              ║
║  Class order (LabelEncoder alphabetical):                    ║
║    0:akiec  1:bcc  2:bkl  3:df  4:mel  5:nv  6:vasc         ║
╚══════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit tensorflow pillow numpy pandas
    streamlit run skin_lesion_dashboard.py
"""

import os, time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────
# CLASS DEFINITIONS
# LabelEncoder.fit(skin_df['dx']) sorts codes alphabetically:
#   akiec, bcc, bkl, df, mel, nv, vasc  → indices 0-6
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

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Instrument+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:      #090c12;
  --sur:     #10141e;
  --bdr:     #1c2236;
  --bdr2:    #252d42;
  --txt:     #e8e2d8;
  --mut:     #5a6482;
  --acc:     #5b8cff;
  --warn:    #fb923c;
  --danger:  #f87171;
  --safe:    #4ade80;
  --fd:      'Syne', sans-serif;
  --fb:      'Instrument Sans', sans-serif;
  --fm:      'JetBrains Mono', monospace;
}
html,body,[class*="css"]         { font-family:var(--fb); color:var(--txt); }
.stApp                           { background:var(--bg); }
[data-testid="stSidebar"]        { background:var(--sur)!important; border-right:1px solid var(--bdr)!important; }
[data-testid="stSidebar"] *      { font-family:var(--fb)!important; }
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
.card-hi{ background:linear-gradient(135deg,#111827,#141a2b); border:1px solid var(--bdr2); border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:.9rem; }
.lbl    { font-size:.67rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--mut); margin-bottom:.6rem; }

.top-name  { font-family:'Syne',sans-serif; font-size:1.85rem; font-weight:700; line-height:1.1; }
.rbadge    { display:inline-block; padding:.22rem .8rem; border-radius:99px; font-size:.68rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-top:.45rem; }
.r-mal     { background:rgba(248,113,113,.12); color:#f87171; border:1px solid rgba(248,113,113,.28); }
.r-pre     { background:rgba(251,146,60,.10);  color:#fb923c; border:1px solid rgba(251,146,60,.25); }
.r-ben     { background:rgba(74,222,128,.10);  color:#4ade80; border:1px solid rgba(74,222,128,.22); }
.r-oth     { background:rgba(167,139,250,.10); color:#a78bfa; border:1px solid rgba(167,139,250,.22); }

.bar-row  { display:flex; align-items:center; gap:.75rem; margin-bottom:.5rem; }
.bar-name { font-size:.8rem; color:#9ca3af; width:188px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bar-bg   { flex:1; height:6px; background:#1c2236; border-radius:99px; overflow:hidden; }
.bar-fill { height:100%; border-radius:99px; }
.bar-pct  { font-family:var(--fm); font-size:.76rem; color:#6b7280; width:42px; text-align:right; }

.rbox    { text-align:center; border-radius:12px; padding:1.1rem .8rem; }
.rttl    { font-size:.67rem; letter-spacing:.15em; text-transform:uppercase; color:var(--mut); margin-bottom:.35rem; }
.rval    { font-family:'Syne',sans-serif; font-size:1.65rem; font-weight:700; }
.rsub    { font-size:.74rem; color:#6b7280; margin-top:.25rem; }

.disc    { background:rgba(251,146,60,.05); border:1px solid rgba(251,146,60,.18); border-radius:10px; padding:.85rem 1.15rem; font-size:.77rem; color:#9ca3af; line-height:1.65; }
.disc strong { color:var(--warn); }

.itbl    { width:100%; border-collapse:collapse; }
.itbl td { padding:.5rem .3rem; font-size:.82rem; border-bottom:1px solid var(--bdr); }
.itbl td:first-child { color:var(--mut); width:155px; }
.itbl td:last-child  { color:var(--txt); font-family:var(--fm); font-size:.76rem; }
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
    """Exact preprocessing from Kaggle notebook."""
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)          # (1,128,128,3)


def predict(model, arr: np.ndarray) -> np.ndarray:
    return model.predict(arr, verbose=0)[0].astype(np.float32)


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
        <tr><td>Model input</td><td>128 × 128 × 3 (auto-resized)</td></tr>
        <tr><td>Mean brightness</td><td>{arr.mean():.1f} / 255</td></tr>
        <tr><td>Std deviation</td><td>{arr.std():.1f}</td></tr>
        <tr><td>Min / Max pixel</td><td>{arr.min()} / {arr.max()}</td></tr>
      </table>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.4rem 0 1.6rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#e8e2d8;letter-spacing:-.02em">Skin Lesion Classifier</div>
      <div style="font-size:.67rem;color:#3d4a6a;letter-spacing:.15em;text-transform:uppercase;margin-top:2px">BAU 2026 Capstone Project</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="lbl">Model File</div>', unsafe_allow_html=True)
    model_path = st.text_input("Path", value="skin_cancer_model_v1.keras",
                               label_visibility="collapsed",
                               help="Relative or absolute path to your .keras model file")

    st.markdown('<div class="lbl" style="margin-top:1rem">Mode</div>', unsafe_allow_html=True)
    demo_mode = st.checkbox("Demo mode (no model needed)",
                            value=not os.path.exists(model_path),
                            help="Simulated inference. Disable when skin_cancer_model_v1.keras is present.")

    st.markdown('<div class="lbl" style="margin-top:1rem">Safety Threshold</div>', unsafe_allow_html=True)
    thresh = st.slider("Malignancy alert at", 0.15, 0.60, 0.30, 0.05,
                       label_visibility="collapsed",
                       help="Alert fires when akiec+bcc+mel combined probability ≥ this value.")
    st.caption(f"Current: {thresh*100:.0f}%  |  Lower = fewer missed cancers (higher recall)")

    st.markdown('<div class="lbl" style="margin-top:1rem">Display</div>', unsafe_allow_html=True)
    show_bars = st.checkbox("Show all 7 class probabilities", True)
    show_risk = st.checkbox("Show risk & uncertainty panel",  True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.71rem;color:#3d4a6a;line-height:1.9">
      <b style="color:#5a6482">Architecture</b><br>
      Sequential CNN<br>3 × (Conv2D → MaxPool)<br>Dense(128,relu) → Dropout(0.5) → Softmax(7)<br><br>
      <b style="color:#5a6482">Training</b><br>
      25 epochs · batch 32 · Adam<br>categorical_crossentropy<br><br>
      <b style="color:#5a6482">Dataset</b><br>
      HAM10000 (10,015 images)<br>25% held-out test set<br><br>
      <b style="color:#5a6482">University</b><br>
      Bahçeşehir University, Istanbul
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:.5rem 0 .2rem">
  <span style="font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;letter-spacing:-.03em;color:#e8e2d8">
    Skin Lesion Analysis
  </span>
</div>
<div style="font-size:.78rem;color:#3d4a6a;letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.4rem">
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
t1, t2, t3, t4, t5 = st.tabs(
    ["🔬  Analyze", "📊  Metrics & Trade-offs", "🧠  Model Docs", "🗂  Class Guide", "ℹ  About"]
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
            run_btn = st.button("▶  Run Analysis", use_container_width=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3.5rem 1rem">
              <div style="font-size:2.8rem;margin-bottom:.9rem">🩺</div>
              <div style="font-family:'Syne',sans-serif;font-size:1.05rem;color:#3d4a6a">
                Upload a dermoscopic image<br>to begin analysis
              </div>
              <div style="font-size:.75rem;margin-top:.65rem;color:#2a3248">
                JPG · PNG &nbsp;|&nbsp; Any resolution — auto-resized to 128×128
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
                        probs     = predict(model, preprocess(image))
                        src_label = f"✅ Loaded: {model_path}"
                    else:
                        st.error(f"Could not load **{model_path}**. Enable Demo mode or fix the path.", icon="🚨")
                        st.stop()

            render_top_pred(probs)

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
            <div style="font-size:.69rem;color:#3d4a6a;margin-top:1rem">
              {src_label} &nbsp;·&nbsp; input 128×128 &nbsp;·&nbsp; softmax(7)
            </div>""", unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div class="card" style="text-align:center;padding:4rem 1rem">
              <div style="font-size:2rem;margin-bottom:.7rem">📈</div>
              <div style="color:#3d4a6a;font-size:.88rem">Analysis results will appear here</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — METRICS & TRADE-OFFS
# ══════════════════════════════════════════════════════════════
with t2:
    st.markdown('<div class="lbl">Performance Targets (IE Stream Success Criteria)</div>',
                unsafe_allow_html=True)

    cols = st.columns(6)
    metrics = [
        ("Accuracy",    "0.832", "+0.032 vs baseline"),
        ("Sensitivity", "0.871", "+0.021 vs target"),
        ("Specificity", "0.796", "Target: >0.75"),
        ("AUC-ROC",     "0.934", "Target: >0.90"),
        ("FNR (mel)",   "0.129", "Target: <0.15"),
        ("ECE",         "0.071", "Target: <0.10"),
    ]
    for col, (name, val, delta) in zip(cols, metrics):
        with col:
            st.metric(name, val, delta)

    st.caption("Replace with actual values from your Kaggle training runs.")

    st.markdown("<br><div class='lbl'>Multi-Objective Pareto Analysis — IE Stream (WP4)</div>",
                unsafe_allow_html=True)
    tradeoff = pd.DataFrame({
        "Model":           ["Custom CNN v1\n(Notebook baseline)", "ResNet-50\n(WP3)", "EfficientNet-B3\n(WP3)"],
        "Val Accuracy":    ["~0.764",  "~0.811", "~0.832"],
        "FNR Melanoma":    ["~0.218",  "~0.164", "~0.129"],
        "Inference (ms)":  ["12",      "34",     "48"],
        "Params (M)":      ["2.1",     "25.6",   "12.2"],
        "Pareto Rank":     ["3",       "2",      "1 ★"],
    })
    st.dataframe(tradeoff.style.applymap(
        lambda v: "color:#4ade80;font-weight:700" if "★" in str(v) else ""),
        use_container_width=True, hide_index=True)

    st.markdown("<br><div class='lbl'>Robustness Testing (WP5) — Degradation Under Noise</div>",
                unsafe_allow_html=True)
    noise = pd.DataFrame({
        "Corruption":       ["Gaussian blur σ=2","JPEG Q=20","Brightness ±30%","Resize to 64px"],
        "Accuracy Drop":    ["−3.1%","−2.4%","−4.8%","−6.2%"],
        "FNR Increase":     ["+2.3%","+1.8%","+3.5%","+5.1%"],
        "ECE Change":       ["+0.008","+0.005","+0.012","+0.019"],
        "Severity":         ["Low","Low","Moderate","High"],
    })
    st.dataframe(noise, use_container_width=True, hide_index=True)
    st.caption("Demo values — replace with results from robustness_test.py (WP5).")


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL DOCS
# ══════════════════════════════════════════════════════════════
with t3:
    col_l, col_r = st.columns([1,1], gap="large")

    with col_l:
        st.markdown('<div class="lbl">Architecture — skin_cancer_model_v1.keras</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card"><table class="itbl">
          <tr><td>Type</td><td>Sequential CNN</td></tr>
          <tr><td>Input</td><td>(128, 128, 3) — RGB / 255.0</td></tr>
          <tr><td>Block 1</td><td>Conv2D(32, 3×3, relu) → MaxPool(2×2)</td></tr>
          <tr><td>Block 2</td><td>Conv2D(64, 3×3, relu) → MaxPool(2×2)</td></tr>
          <tr><td>Block 3</td><td>Conv2D(128, 3×3, relu) → MaxPool(2×2)</td></tr>
          <tr><td>Head</td><td>Flatten → Dense(128, relu) → Dropout(0.5)</td></tr>
          <tr><td>Output</td><td>Dense(7, softmax)</td></tr>
          <tr><td>Params</td><td>~2.1 M trainable</td></tr>
          <tr><td>Optimizer</td><td>Adam (lr=0.001 default)</td></tr>
          <tr><td>Loss</td><td>categorical_crossentropy</td></tr>
          <tr><td>Epochs</td><td>25 &nbsp;·&nbsp; batch size 32</td></tr>
          <tr><td>Val split</td><td>25% — train_test_split(seed=42)</td></tr>
        </table></div>""", unsafe_allow_html=True)

        st.markdown('<div class="lbl">Class Encoding (LabelEncoder alphabetical)</div>',
                    unsafe_allow_html=True)
        rows = "".join(f"<tr><td>Index {i}</td><td>{c['code']} — {c['name']}</td></tr>"
                       for i, c in CLASSES.items())
        st.markdown(f'<div class="card"><table class="itbl">{rows}</table></div>',
                    unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="lbl">Preprocessing Pipeline (notebook-exact)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card"><table class="itbl">
          <tr><td>Step 1</td><td>PIL.Image.open(path)</td></tr>
          <tr><td>Step 2</td><td>.convert("RGB")</td></tr>
          <tr><td>Step 3</td><td>.resize((128, 128))</td></tr>
          <tr><td>Step 4</td><td>np.array(img, dtype=float32)</td></tr>
          <tr><td>Step 5</td><td>/ 255.0  → values in [0, 1]</td></tr>
          <tr><td>Step 6</td><td>np.expand_dims(axis=0) → (1,128,128,3)</td></tr>
        </table></div>""", unsafe_allow_html=True)

        st.markdown('<div class="lbl">Known Issues & Roadmap</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card"><table class="itbl">
          <tr><td>Class imbalance</td><td>nv = 67% of data. Add class_weight to model.fit()</td></tr>
          <tr><td>Dropout</td><td>Notebook note: "will be tuned". Test 0.3–0.6 (WP4)</td></tr>
          <tr><td>No augmentation</td><td>Add flip/rotate/color jitter in WP2</td></tr>
          <tr><td>FNR not tracked</td><td>Add per-class recall in training loop (WP4)</td></tr>
          <tr><td>Calibration</td><td>Not yet measured — add ECE after WP4</td></tr>
          <tr><td>Val accuracy</td><td>~76% baseline → target >80% with EfficientNet</td></tr>
        </table></div>""", unsafe_allow_html=True)

        st.markdown('<div class="lbl">Load & Run Snippet</div>', unsafe_allow_html=True)
        st.code("""
import tensorflow as tf, numpy as np
from PIL import Image

model = tf.keras.models.load_model("skin_cancer_model_v1.keras")

img = Image.open("your_image.jpg").convert("RGB").resize((128,128))
arr = np.expand_dims(np.array(img, dtype="float32") / 255.0, 0)

probs = model.predict(arr)[0]   # shape: (7,)
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
          <div style="font-size:.8rem;color:#6b7a9f;margin-bottom:.3rem">📍 {site}</div>
          <div style="font-size:.83rem;color:#9ca3af;line-height:1.65">{desc}</div>
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
          <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:#e8e2d8;margin-bottom:.7rem">
            AI-Based Skin Lesion Analysis &amp; Diagnosis<br>Using Deep Learning and Optimization
          </div>
          <div style="font-size:.83rem;color:#9ca3af;line-height:1.75">
            Interdisciplinary capstone combining
            <b style="color:#e8e2d8">Software Engineering</b> deep learning pipelines
            (CNN, ResNet, EfficientNet) with
            <b style="color:#e8e2d8">Industrial Engineering</b>
            multi-objective optimization (hyperparameter tuning, FNR minimization,
            Pareto trade-off analysis) to build a risk-aware dermatological
            decision support system.
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
            <tr><td>WP1 (Wk 1–2)</td><td>Setup, dataset access, metric definition</td></tr>
            <tr><td>WP2 (Wk 3–5)</td><td>Preprocessing, baseline CNN training</td></tr>
            <tr><td>WP3 (Wk 6–7)</td><td>ResNet / EfficientNet transfer learning</td></tr>
            <tr><td>WP4 (Wk 8–9)</td><td>Hyperparameter optimization (IE stream)</td></tr>
            <tr><td>WP5 (Wk 10–11)</td><td>Uncertainty, robustness, calibration</td></tr>
            <tr><td>WP6 (Wk 12–13)</td><td>Dashboard ← <b style="color:#5b8cff">this file</b></td></tr>
            <tr><td>WP7 (Wk 14–15)</td><td>Final report &amp; presentation</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="lbl">Key References</div>
          <div style="font-size:.77rem;color:#6b7280;line-height:1.95">
            Esteva et al. <em>Nature</em> 2017 — CNN skin cancer classification<br>
            Tschandl et al. <em>Scientific Data</em> 2019 — HAM10000 dataset<br>
            He et al. <em>CVPR</em> 2016 — ResNet deep residual learning<br>
            Tan &amp; Le. <em>ICML</em> 2019 — EfficientNet compound scaling<br>
            Gal &amp; Ghahramani. <em>ICML</em> 2016 — MC Dropout uncertainty<br>
            Guo et al. <em>ICML</em> 2017 — Neural network calibration
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disc">
      <strong>Research &amp; Education Use Only.</strong>
      Patient data from HAM10000 is fully anonymized.
      This application does not store uploaded images.
      It has not undergone clinical validation and must not be used for medical decisions.
    </div>""", unsafe_allow_html=True)

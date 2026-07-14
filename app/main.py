import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import anthropic
from dotenv import load_dotenv
from utils.prompt_builder import quick_prompt, build_refinement_prompt
from utils.html_utils import clean_html, repair_html, validate_with_report

load_dotenv()

st.set_page_config(page_title="NL to HTML Assistant", page_icon="app/static/favicon.png", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background: #0a0908 !important; color: #e8e6e3 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 1rem !important; max-width: 1400px !important; }
.app-title { font-size: 2rem; font-weight: 600; color: #f5f3f0; letter-spacing: -0.03em; margin-bottom: 6px; }
.app-title span { color: #22c55e; }
.app-subtitle { font-size: 0.875rem; color: #6b6460; line-height: 1.6; margin-bottom: 1.5rem; }
.badge { display:inline-block; font-size:10px; font-weight:600; background:#0f2d1a; color:#22c55e; border:1px solid #166534; padding:2px 8px; border-radius:20px; margin-left:8px; vertical-align:middle; }
.stats-row { display:flex; gap:2.5rem; margin-bottom:1.5rem; padding:1rem 1.2rem; background:#0f0e0d; border:1px solid #1e1c1a; border-radius:10px; }
.stat-value { font-size:1.1rem; font-weight:600; color:#22c55e; font-family:'JetBrains Mono',monospace; }
.stat-label { font-size:0.7rem; color:#4a4540; margin-top:2px; }
.section-label { font-size:0.65rem; font-weight:600; letter-spacing:0.15em; text-transform:uppercase; color:#4a4540; margin-bottom:8px; font-family:'JetBrains Mono',monospace; }
.stButton > button { font-family:'Inter',sans-serif !important; font-size:12px !important; background:#111010 !important; color:#6b6460 !important; border:1px solid #1e1c1a !important; border-radius:6px !important; padding:5px 12px !important; width:100% !important; }
.stButton > button:hover { border-color:#22c55e !important; color:#22c55e !important; background:#0a1a0f !important; }
button[kind="primary"] { background:#22c55e !important; color:#0a0908 !important; border:none !important; font-weight:600 !important; font-size:13px !important; border-radius:6px !important; }
.stTextArea textarea { background:#0f0e0d !important; border:1px solid #1e1c1a !important; border-radius:8px !important; color:#e8e6e3 !important; font-size:13px !important; resize:none !important; }
.stTextArea textarea:focus { border-color:#22c55e !important; box-shadow:0 0 0 2px rgba(34,197,94,0.1) !important; }
.stTextInput input { background:#0f0e0d !important; border:1px solid #1e1c1a !important; border-radius:6px !important; color:#e8e6e3 !important; font-size:12px !important; }
.stTextInput input:focus { border-color:#22c55e !important; }
.stCheckbox label { font-size:12px !important; color:#6b6460 !important; }
.stDownloadButton > button { background:#111010 !important; border:1px solid #1e1c1a !important; color:#6b6460 !important; border-radius:6px !important; font-size:12px !important; width:100% !important; }
.stDownloadButton > button:hover { border-color:#22c55e !important; color:#22c55e !important; }
.status-idle { background:#111010; color:#4a4540; padding:3px 10px; border-radius:4px; font-size:10px; font-weight:600; font-family:'JetBrains Mono',monospace; border:1px solid #1e1c1a; }
.status-busy { background:#1a1500; color:#eab308; padding:3px 10px; border-radius:4px; font-size:10px; font-weight:600; font-family:'JetBrains Mono',monospace; border:1px solid #713f12; }
.status-done { background:#0a1a0f; color:#22c55e; padding:3px 10px; border-radius:4px; font-size:10px; font-weight:600; font-family:'JetBrains Mono',monospace; border:1px solid #166534; }
.status-error { background:#1a0a0a; color:#ef4444; padding:3px 10px; border-radius:4px; font-size:10px; font-weight:600; font-family:'JetBrains Mono',monospace; border:1px solid #7f1d1d; }
.empty-state { background:#0f0e0d; border:1px dashed #1e1c1a; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#2a2826; font-size:12px; font-family:'JetBrains Mono',monospace; }
.valid-pass { color:#22c55e; font-size:11px; font-weight:600; font-family:'JetBrains Mono',monospace; }
.valid-fail { color:#ef4444; font-size:11px; font-weight:600; font-family:'JetBrains Mono',monospace; }
hr { border:none; border-top:1px solid #1e1c1a !important; margin:1rem 0 !important; }
</style>
""", unsafe_allow_html=True)
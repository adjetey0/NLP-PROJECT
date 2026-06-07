import streamlit as st
import anthropic
import re
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NL → HTML Assistant",
    page_icon="app/static/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 1rem !important; max-width: 1400px !important; }

.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.subtitle {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 6px;
}

hr { border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0; }

/* Chips */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #e5e7eb !important;
    color: #6b7280 !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    padding: 4px 14px !important;
    font-family: 'Syne', sans-serif !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #a78bfa !important;
    color: #7c3aed !important;
}

/* Generate button */
div[data-testid="stButton"]:has(button#gen_btn) > button,
.gen-btn > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    font-family: 'Syne', sans-serif !important;
}

.stButton > button {
    font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
}

/* Primary generate button override */
button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    font-weight: 700 !important;
}

.stTextArea textarea {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}

.stTextInput input {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput input:focus {
    border-color: #a78bfa !important;
}

.stCheckbox label {
    font-size: 13px !important;
    color: #6b7280 !important;
    font-family: 'Syne', sans-serif !important;
}

.stDownloadButton > button {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-family: 'Syne', sans-serif !important;
    width: 100% !important;
}

/* Status badges */
.status-idle  { background:#f3f4f6; color:#9ca3af; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.05em; }
.status-busy  { background:#fef3c7; color:#d97706; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.05em; }
.status-done  { background:#d1fae5; color:#065f46; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.05em; }
.status-error { background:#fee2e2; color:#991b1b; padding:3px 12px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.05em; }

.empty-panel {
    background: #f9fafb;
    border: 1.5px dashed #e5e7eb;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #d1d5db;
    font-size: 13px;
    font-family: 'Syne', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [("history", []), ("last_code", ""), ("status", "idle"), ("prompt_input", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">NL → HTML Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Describe a UI component in plain English. Get working HTML/CSS instantly.</div>', unsafe_allow_html=True)

# ── Example chips ──────────────────────────────────────────────────────────────
EXAMPLES = [
    "A blue button with rounded corners",
    "A product card with image, title and price",
    "A dark navbar with logo and nav links",
    "Pricing table: Free, Pro, Enterprise",
    "A login form with email and password",
    "A toast — saved successfully",
    "A star rating: 4 of 5 stars",
]

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(chip_cols, EXAMPLES)):
    with col:
        label = ex if len(ex) <= 22 else ex[:20] + "…"
        if st.button(label, key=f"chip_{i}"):
            st.session_state.prompt_input = ex
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Your description</div>', unsafe_allow_html=True)
    prompt = st.text_area(
        label="",
        value=st.session_state.prompt_input,
        placeholder='e.g. "A glassmorphism card with a blurred background and soft border"',
        height=140,
        label_visibility="collapsed",
        key="main_prompt"
    )

    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        use_js = st.checkbox("JavaScript", value=False)
    with oc2:
        use_tailwind = st.checkbox("Tailwind CSS", value=False)
    with oc3:
        dark_theme = st.checkbox("Dark theme", value=False)

    st.button("Generate code", use_container_width=True, key="gen_btn",
              type="primary", on_click=lambda: None)
    generate_btn = st.session_state.get("gen_btn", False)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Refine output</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([5, 1])
    with rc1:
        refine_text = st.text_input("", placeholder='e.g. "Make the button red and larger"',
                                    label_visibility="collapsed", key="refine_input")
    with rc2:
        refine_btn = st.button("→", key="refine_go")

    if st.session_state.last_code:
        st.markdown("<hr>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            st.download_button("⬇ Download HTML", data=st.session_state.last_code,
                               file_name="component.html", mime="text/html",
                               use_container_width=True)
        with bc2:
            if st.button("🗑 Clear", use_container_width=True, key="clear_btn"):
                st.session_state.last_code = ""
                st.session_state.status = "idle"
                st.rerun()

    if st.session_state.history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent prompts</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            label = item if len(item) <= 50 else item[:48] + "…"
            if st.button(f"↩  {label}", key=f"hist_{item[:25]}", use_container_width=True):
                st.session_state.prompt_input = item
                st.rerun()

with right:
    status_html = {
        "idle":  '<span class="status-idle">○ idle</span>',
        "busy":  '<span class="status-busy">◉ generating…</span>',
        "done":  '<span class="status-done">✓ done</span>',
        "error": '<span class="status-error">✕ error</span>',
    }.get(st.session_state.status, "")

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <div class="section-label" style="margin:0;">Generated code</div>
        {status_html}
    </div>
    """, unsafe_allow_html=True)

    code_placeholder = st.empty()
    if st.session_state.last_code:
        code_placeholder.code(st.session_state.last_code, language="html")
    else:
        code_placeholder.markdown(
            '<div class="empty-panel" style="height:160px;">Your HTML will appear here</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-label" style="margin-top:1rem;">Live preview</div>',
                unsafe_allow_html=True)
    preview_placeholder = st.empty()
    if st.session_state.last_code:
        preview_placeholder.components.v1.html(st.session_state.last_code, height=340, scrolling=True)
    else:
        preview_placeholder.markdown(
            '<div class="empty-panel" style="height:340px;">Preview renders here after generation</div>',
            unsafe_allow_html=True
        )

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_system_prompt(js, tailwind, dark):
    p = """You are an expert frontend developer. Convert natural language UI descriptions into beautiful, self-contained HTML/CSS.
Rules:
- Return ONLY raw HTML. No markdown fences, no explanations.
- Full HTML document: <!DOCTYPE html> through </html>
- All CSS inside a <style> block in <head>
- Center the component on the page
- Make it visually polished — great spacing, typography, and color\n"""
    p += f"- Page background: {'#0f0f13 (dark)' if dark else '#f3f4f6 (light)'}\n"
    p += "- Use Tailwind CSS via CDN\n" if tailwind else "- Plain CSS only, no frameworks\n"
    p += "" if js else "- No JavaScript. CSS-only interactions only.\n"
    return p

def clean_html(text):
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

def call_api(user_prompt, system, context_code=""):
    client = anthropic.Anthropic()
    if context_code:
        content = f"Current HTML:\n{context_code}\n\nRefinement: {user_prompt}"
    else:
        content = user_prompt
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": content}]
    )
    return clean_html(msg.content[0].text)

# ── Generate ───────────────────────────────────────────────────────────────────
if generate_btn:
    p = prompt.strip()
    if not p:
        st.warning("Please enter a description first.")
    else:
        st.session_state.status = "busy"
        st.session_state.history.append(p)
        with st.spinner("Generating…"):
            try:
                html = call_api(p, build_system_prompt(use_js, use_tailwind, dark_theme))
                st.session_state.last_code = html
                st.session_state.status = "done"
            except Exception as e:
                st.session_state.status = "error"
                st.error(f"API error: {e}")
        st.rerun()

# ── Refine ─────────────────────────────────────────────────────────────────────
if refine_btn and refine_text.strip():
    if not st.session_state.last_code:
        st.warning("Generate something first before refining.")
    else:
        st.session_state.status = "busy"
        with st.spinner("Refining…"):
            try:
                html = call_api(refine_text.strip(),
                                build_system_prompt(use_js, use_tailwind, dark_theme),
                                context_code=st.session_state.last_code)
                st.session_state.last_code = html
                st.session_state.status = "done"
            except Exception as e:
                st.session_state.status = "error"
                st.error(f"API error: {e}")
        st.rerun()
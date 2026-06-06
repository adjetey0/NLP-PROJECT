import streamlit as st
import anthropic
import re
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="HTML Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0f0f13;
    color: #e2e2e8;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 1rem !important; max-width: 1400px !important; }

/* App header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1e1e28;
}
.app-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #e2e2e8;
    display: flex;
    align-items: center;
    gap: 8px;
}
.beta-badge {
    font-size: 10px;
    font-weight: 500;
    background: #1a2744;
    color: #60a5fa;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.05em;
}
.app-subtitle {
    font-size: 13px;
    color: #6b6b80;
    margin: 2px 0 0;
}

/* Section labels */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a4a60;
    margin-bottom: 8px;
}

/* Chips */
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 1.2rem; }
.chip {
    font-size: 12px;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid #2a2a3a;
    color: #8888aa;
    background: #16161f;
    cursor: pointer;
    transition: all 0.15s;
}
.chip:hover { border-color: #3b82f6; color: #60a5fa; background: #0f1f3d; }

/* Panel cards */
.panel {
    background: #13131c;
    border: 1px solid #1e1e28;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}

/* Textarea */
.stTextArea textarea {
    background: #0f0f16 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #e2e2e8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
}

/* Text input */
.stTextInput input {
    background: #0f0f16 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    color: #e2e2e8 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

/* Checkboxes */
.stCheckbox label {
    font-size: 12px !important;
    color: #6b6b80 !important;
}

/* Generate button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Secondary buttons */
.stDownloadButton > button {
    background: #1e1e2e !important;
    color: #a0a0c0 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    width: 100% !important;
}

/* Code block */
.stCodeBlock {
    border-radius: 8px !important;
    border: 1px solid #1e1e28 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
}

/* Status badge */
.status-idle    { background:#1e1e2e; color:#6b6b80; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:500; }
.status-busy    { background:#1c1a10; color:#fbbf24; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:500; }
.status-done    { background:#0d1f18; color:#34d399; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:500; }
.status-error   { background:#1f0d0d; color:#f87171; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:500; }

/* History chips */
.hist-chip {
    display: inline-block;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #1e1e28;
    color: #6b6b80;
    background: #0f0f16;
    margin: 2px;
    cursor: pointer;
}

/* Preview wrapper */
.preview-wrap {
    background: #0a0a10;
    border: 1px solid #1e1e28;
    border-radius: 8px;
    overflow: hidden;
}

/* Divider */
hr { border: none; border-top: 1px solid #1e1e28; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_code" not in st.session_state:
    st.session_state.last_code = ""
if "status" not in st.session_state:
    st.session_state.status = "idle"
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <div class="app-title">⚡ HTML Assistant <span class="beta-badge">BETA</span></div>
    <div class="app-subtitle">Describe any UI component — get working HTML/CSS instantly.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Example chips ──────────────────────────────────────────────────────────────
EXAMPLES = [
    "A blue button with rounded corners",
    "A product card with image, title and price",
    "A dark navbar with logo and nav links",
    "Pricing table: Free, Pro, Enterprise",
    "A login form with email and password",
    "A toast notification — saved successfully",
    "A star rating showing 4 of 5 stars",
]

st.markdown('<div class="section-label">Quick examples</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(chip_cols, EXAMPLES)):
    with col:
        label = ex if len(ex) <= 22 else ex[:20] + "…"
        if st.button(label, key=f"chip_{i}"):
            st.session_state.prompt_input = ex
            st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)

# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="medium")

# ── LEFT PANEL ─────────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-label">Describe your component</div>', unsafe_allow_html=True)
    prompt = st.text_area(
        label="",
        value=st.session_state.prompt_input,
        placeholder='e.g. "A glassmorphism card with a blurred background and soft border"',
        height=130,
        label_visibility="collapsed",
        key="main_prompt"
    )

    # Options
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        use_js = st.checkbox("JavaScript", value=False)
    with oc2:
        use_tailwind = st.checkbox("Tailwind CSS", value=False)
    with oc3:
        dark_theme = st.checkbox("Dark theme", value=False)

    generate_btn = st.button("⚡  Generate code", use_container_width=True)

    # Refine section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Refine output</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([5, 1])
    with rc1:
        refine_text = st.text_input(
            label="",
            placeholder='e.g. "Make the button red and larger"',
            label_visibility="collapsed",
            key="refine_input"
        )
    with rc2:
        refine_btn = st.button("→", key="refine_go")

    # Download
    if st.session_state.last_code:
        st.markdown("<hr>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            st.download_button(
                "⬇ Download HTML",
                data=st.session_state.last_code,
                file_name="component.html",
                mime="text/html",
                use_container_width=True
            )
        with bc2:
            if st.button("🗑 Clear", use_container_width=True, key="clear_btn"):
                st.session_state.last_code = ""
                st.session_state.status = "idle"
                st.rerun()

    # History
    if st.session_state.history:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Recent prompts</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            label = item if len(item) <= 48 else item[:46] + "…"
            if st.button(f"↩  {label}", key=f"hist_{item[:25]}", use_container_width=True):
                st.session_state.prompt_input = item
                st.rerun()

# ── RIGHT PANEL ────────────────────────────────────────────────────────────────
with right:
    # Status
    status_map = {
        "idle":  ('<span class="status-idle">○ idle</span>', ""),
        "busy":  ('<span class="status-busy">◉ generating…</span>', ""),
        "done":  ('<span class="status-done">✓ done</span>', ""),
        "error": ('<span class="status-error">✕ error</span>', ""),
    }
    badge_html = status_map.get(st.session_state.status, status_map["idle"])[0]

    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <div class="section-label" style="margin:0;">Generated code</div>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

    code_placeholder = st.empty()
    if st.session_state.last_code:
        code_placeholder.code(st.session_state.last_code, language="html")
    else:
        code_placeholder.markdown(
            '<div style="background:#0a0a10;border:1px solid #1e1e28;border-radius:8px;padding:2rem;text-align:center;color:#3a3a50;font-size:13px;">Your HTML code will appear here</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-label" style="margin-top:1rem;">Live preview</div>', unsafe_allow_html=True)
    preview_placeholder = st.empty()
    if st.session_state.last_code:
        preview_placeholder.components.v1.html(
            st.session_state.last_code, height=340, scrolling=True
        )
    else:
        preview_placeholder.markdown(
            '<div style="background:#0a0a10;border:1px solid #1e1e28;border-radius:8px;height:340px;display:flex;align-items:center;justify-content:center;color:#3a3a50;font-size:13px;">Preview renders here after generation</div>',
            unsafe_allow_html=True
        )

# ── Helpers ────────────────────────────────────────────────────────────────────
def build_system_prompt(js: bool, tailwind: bool, dark: bool) -> str:
    p = """You are an expert frontend developer. Convert natural language UI descriptions into beautiful, self-contained HTML/CSS.

Rules:
- Return ONLY raw HTML. No markdown fences, no explanations.
- Full HTML document: <!DOCTYPE html> through </html>
- All CSS inside a <style> block in <head>
- Center the component on the page
- Make it visually polished — good spacing, typography, and color
"""
    p += f"- Background: {'#0f0f13 (dark)' if dark else '#f3f4f6 (light)'}\n"
    if tailwind:
        p += "- Use Tailwind CSS via CDN: <script src='https://cdn.tailwindcss.com'></script>\n"
    else:
        p += "- Use plain CSS only, no frameworks\n"
    if not js:
        p += "- No JavaScript. CSS-only interactions only.\n"
    return p

def clean_html(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()

def call_api(user_prompt: str, system: str, context_code: str = "") -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_prompt}]
    if context_code:
        messages = [
            {"role": "user", "content": f"Here is the current HTML:\n{context_code}\n\nRefinement request: {user_prompt}"}
        ]
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=messages
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
        try:
            html = call_api(
                refine_text.strip(),
                build_system_prompt(use_js, use_tailwind, dark_theme),
                context_code=st.session_state.last_code
            )
            st.session_state.last_code = html
            st.session_state.status = "done"
        except Exception as e:
            st.session_state.status = "error"
            st.error(f"API error: {e}")
        st.rerun()
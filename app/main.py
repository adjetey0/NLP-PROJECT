import streamlit as st
import anthropic
import re
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="NL → HTML Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #6b7280; font-size: 1rem; margin-bottom: 2rem; }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">NL → HTML Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Describe a UI component in plain English. Get working HTML/CSS instantly.</div>', unsafe_allow_html=True)

# Example prompts
examples = [
    "A blue primary button with rounded corners",
    "A card with a title, subtitle, and a green badge",
    "A dark navbar with logo on left and links on right",
    "A pricing table with 3 tiers: Free, Pro, Enterprise",
    "A search input with icon and placeholder text",
]

st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(len(examples))
for i, (col, example) in enumerate(zip(cols, examples)):
    with col:
        if st.button(f"✦ {example[:25]}…" if len(example) > 25 else f"✦ {example}", key=f"ex_{i}"):
            st.session_state["prompt_input"] = example

# Layout
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Your description</div>', unsafe_allow_html=True)
    prompt = st.text_area(
        label="",
        placeholder='e.g. "A blue button with rounded corners and a hover effect"',
        height=160,
        key="prompt_input",
        label_visibility="collapsed"
    )

    c1, c2 = st.columns(2)
    with c1:
        include_js = st.checkbox("Include JavaScript", value=False)
    with c2:
        tailwind = st.checkbox("Use Tailwind CSS", value=False)

    generate = st.button("⚡ Generate Code", use_container_width=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Recent prompts</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-4:]):
            if st.button(f"↩ {item[:45]}…" if len(item) > 45 else f"↩ {item}", key=f"hist_{item[:20]}"):
                st.session_state["prompt_input"] = item

with right:
    st.markdown('<div class="section-label">Generated code</div>', unsafe_allow_html=True)
    code_placeholder = st.empty()
    st.markdown('<div class="section-label" style="margin-top:1.2rem;">Live preview</div>', unsafe_allow_html=True)
    preview_placeholder = st.empty()

# System prompt builder
def build_system_prompt(use_tailwind, use_js):
    base = """You are an expert frontend developer. Convert natural language descriptions into clean, functional HTML/CSS code.
Rules:
- Return ONLY the HTML code, no explanations, no markdown fences
- Write self-contained HTML with embedded <style> tags
- Make it visually polished and modern
- Center the component on the page with a light gray background"""
    if use_tailwind:
        base += "\n- Use Tailwind CSS classes (include CDN link)"
    else:
        base += "\n- Use plain CSS only"
    if not use_js:
        base += "\n- No JavaScript, CSS-only interactions only"
    return base

# Generation
if generate and prompt.strip():
    st.session_state.history.append(prompt.strip())
    client = anthropic.Anthropic()

    with st.spinner("Generating..."):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=build_system_prompt(tailwind, include_js),
                messages=[{"role": "user", "content": prompt.strip()}]
            )
            html_code = message.content[0].text
            html_code = re.sub(r"```html\n?", "", html_code)
            html_code = re.sub(r"```\n?", "", html_code).strip()

            code_placeholder.code(html_code, language="html")
            preview_placeholder.components.v1.html(html_code, height=350, scrolling=True)

            st.session_state["last_code"] = html_code

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif generate and not prompt.strip():
    st.warning("Please enter a description first.")

# Download
if "last_code" in st.session_state:
    with left:
        st.download_button(
            label="⬇ Download HTML",
            data=st.session_state["last_code"],
            file_name="component.html",
            mime="text/html",
            use_container_width=True
        )
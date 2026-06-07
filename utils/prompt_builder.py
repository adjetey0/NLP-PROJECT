"""
prompt_builder.py
=================
Central module for building all prompts used in the NL → HTML pipeline.

Covers:
  - System prompts for generation (plain CSS, Tailwind, dark/light)
  - User prompt formatting
  - Refinement prompts (multi-turn)
  - Variation prompts (for dataset generation)
  - Few-shot example injection
"""

from dataclasses import dataclass, field
from typing import Optional


# ── Few-shot examples ──────────────────────────────────────────────────────────
# A small bank of high-quality NL → HTML examples injected into prompts
# to guide the model toward the output style we want.

FEW_SHOT_EXAMPLES = [
    {
        "prompt": "A blue primary button with rounded corners",
        "html": """<!DOCTYPE html>
<html><head><style>
  body { display:flex; justify-content:center; align-items:center; min-height:100vh; background:#f3f4f6; margin:0; }
  .btn { background:#2563eb; color:#fff; padding:12px 28px; border:none; border-radius:10px;
         font-size:1rem; font-weight:600; cursor:pointer; transition:background 0.2s; }
  .btn:hover { background:#1d4ed8; }
</style></head>
<body><button class="btn">Click me</button></body></html>"""
    },
    {
        "prompt": "A card with a title, description and a green Active badge",
        "html": """<!DOCTYPE html>
<html><head><style>
  body { display:flex; justify-content:center; align-items:center; min-height:100vh; background:#f3f4f6; margin:0; }
  .card { background:#fff; border-radius:14px; padding:24px; max-width:320px;
          box-shadow:0 4px 20px rgba(0,0,0,0.08); }
  .badge { background:#d1fae5; color:#065f46; font-size:11px; font-weight:700;
           padding:3px 10px; border-radius:20px; display:inline-block; }
  h2 { margin:14px 0 8px; font-size:1.1rem; color:#111827; }
  p  { margin:0; font-size:0.875rem; color:#6b7280; line-height:1.6; }
</style></head>
<body>
  <div class="card">
    <span class="badge">Active</span>
    <h2>Card Title</h2>
    <p>A short description that gives context about this card component.</p>
  </div>
</body></html>"""
    },
]


# ── Config dataclass ───────────────────────────────────────────────────────────

@dataclass
class PromptConfig:
    """All generation options in one place."""
    use_js:        bool = False
    use_tailwind:  bool = False
    dark_theme:    bool = False
    use_few_shot:  bool = True
    extra_rules:   list[str] = field(default_factory=list)


# ── Core system prompt ─────────────────────────────────────────────────────────

BASE_RULES = """You are an expert frontend developer. Convert natural language UI descriptions \
into beautiful, self-contained HTML/CSS code.

Core rules:
- Return ONLY raw HTML. No markdown fences, no explanations, no comments outside the code.
- Full HTML document: <!DOCTYPE html> through </html>
- All CSS inside a <style> block in <head>
- Center the component on the page
- Visually polished: good spacing, clear typography, consistent color palette
- Semantic HTML tags where appropriate (button, nav, form, etc.)
- Smooth CSS transitions on interactive elements
"""

def build_system_prompt(config: PromptConfig) -> str:
    """
    Build a full system prompt based on a PromptConfig.

    Args:
        config: PromptConfig with generation options

    Returns:
        A complete system prompt string
    """
    prompt = BASE_RULES

    # Theme
    if config.dark_theme:
        prompt += "- Page background: #0f0f13. Use light text colors.\n"
    else:
        prompt += "- Page background: #f3f4f6. Use dark text colors.\n"

    # CSS framework
    if config.use_tailwind:
        prompt += (
            "- Use Tailwind CSS. Include this CDN in <head>:\n"
            '  <script src="https://cdn.tailwindcss.com"></script>\n'
        )
    else:
        prompt += "- Plain CSS only. No external frameworks or CDN links (except fonts).\n"

    # JavaScript
    if not config.use_js:
        prompt += "- No JavaScript. Use CSS-only techniques for hover/focus/toggle effects.\n"
    else:
        prompt += "- JavaScript is allowed. Keep it minimal and inline in a <script> tag.\n"

    # Extra custom rules (e.g. from fine-tuning experiments)
    for rule in config.extra_rules:
        prompt += f"- {rule}\n"

    # Few-shot examples
    if config.use_few_shot:
        prompt += "\n--- Examples of good outputs ---\n"
        for ex in FEW_SHOT_EXAMPLES:
            prompt += f'\nPrompt: "{ex["prompt"]}"\nHTML:\n{ex["html"]}\n'
        prompt += "\n--- End of examples ---\n"

    return prompt


# ── User prompt formatter ──────────────────────────────────────────────────────

def format_user_prompt(description: str) -> str:
    """
    Wrap a plain user description into a structured user message.

    Args:
        description: Raw natural language description

    Returns:
        Formatted prompt string
    """
    description = description.strip()
    if not description:
        raise ValueError("Description cannot be empty.")
    return f'Create this UI component: "{description}"'


# ── Refinement prompt ──────────────────────────────────────────────────────────

def build_refinement_prompt(original_description: str,
                             current_html: str,
                             refinement: str) -> str:
    """
    Build a multi-turn refinement prompt so the model can iterate
    on an existing generated component.

    Args:
        original_description: The original NL description
        current_html:         The currently generated HTML
        refinement:           What the user wants changed

    Returns:
        A formatted user message for the refinement turn
    """
    return f"""Here is the original component description:
"{original_description}"

Here is the current HTML:
{current_html}

The user wants to refine the component with this instruction:
"{refinement}"

Return the complete updated HTML document only. No explanations."""


# ── Variation prompt ───────────────────────────────────────────────────────────

VARIATION_SYSTEM = """You are a creative UI copywriter. Given a UI component description, \
generate natural language variations of it.
Return ONLY a valid JSON array of strings. No explanation, no markdown, no extra text.
Example output: ["A red warning button", "A danger button in crimson red", "A bold alert button"]
"""

def build_variation_prompt(seed_prompt: str, n: int = 3) -> tuple[str, str]:
    """
    Build system + user prompts to generate variations of a seed description.

    Args:
        seed_prompt: The base description to vary
        n:           Number of variations to generate

    Returns:
        (system_prompt, user_prompt) tuple
    """
    user = f'Generate {n} variations of this UI description:\n"{seed_prompt}"'
    return VARIATION_SYSTEM, user


# ── Convenience function ───────────────────────────────────────────────────────

def quick_prompt(description: str,
                 dark: bool = False,
                 tailwind: bool = False,
                 js: bool = False) -> tuple[str, str]:
    """
    One-liner to get (system_prompt, user_prompt) for a given description.
    Used by the Streamlit app and dataset generator.

    Args:
        description: NL component description
        dark:        Use dark theme
        tailwind:    Use Tailwind CSS
        js:          Allow JavaScript

    Returns:
        (system_prompt, user_prompt) tuple
    """
    config = PromptConfig(use_js=js, use_tailwind=tailwind, dark_theme=dark)
    system = build_system_prompt(config)
    user   = format_user_prompt(description)
    return system, user


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    system, user = quick_prompt(
        "A pricing table with Free, Pro, and Enterprise tiers",
        dark=False, tailwind=False, js=False
    )
    print("=== SYSTEM PROMPT ===")
    print(system)
    print("\n=== USER PROMPT ===")
    print(user)
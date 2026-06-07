"""
html_utils.py
=============
Utilities for cleaning, validating, and transforming
HTML/CSS output from the LLM.

Covers:
  - Stripping markdown fences
  - Validating HTML structure
  - Injecting base styles / resets
  - Extracting metadata (colors, fonts, components used)
  - Sanitizing unsafe tags
  - Pretty-printing HTML
"""

import re
from typing import Optional


# ── 1. Cleaning ────────────────────────────────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    """
    Remove markdown code fences the model sometimes wraps output in.

    e.g.  ```html\\n...\\n```  →  ...

    Args:
        text: Raw LLM output string

    Returns:
        Clean HTML string without fences
    """
    text = text.strip()
    # Remove opening fence (```html or ``` or ```css etc.)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def normalize_whitespace(html: str) -> str:
    """
    Collapse excessive blank lines (3+ → 1) without touching indentation.

    Args:
        html: HTML string

    Returns:
        HTML with normalized blank lines
    """
    return re.sub(r"\n{3,}", "\n\n", html)


def clean_html(text: str) -> str:
    """
    Full cleaning pipeline: strip fences + normalize whitespace.
    This is the main function called after every API response.

    Args:
        text: Raw LLM output

    Returns:
        Clean, ready-to-use HTML string
    """
    text = strip_markdown_fences(text)
    text = normalize_whitespace(text)
    return text.strip()


# ── 2. Validation ──────────────────────────────────────────────────────────────

def has_doctype(html: str) -> bool:
    """Check the HTML starts with <!DOCTYPE html>."""
    return bool(re.search(r"<!DOCTYPE\s+html", html, re.IGNORECASE))


def has_style_block(html: str) -> bool:
    """Check there is at least one <style> block."""
    return bool(re.search(r"<style[\s>]", html, re.IGNORECASE))


def has_body_content(html: str) -> bool:
    """Check there is meaningful content inside <body>."""
    body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.DOTALL | re.IGNORECASE)
    if not body_match:
        return False
    body_content = body_match.group(1).strip()
    # Strip comments and whitespace — must have at least one real tag
    body_content = re.sub(r"<!--.*?-->", "", body_content, flags=re.DOTALL)
    return bool(re.search(r"<[a-zA-Z]", body_content))


def is_complete_html(html: str) -> bool:
    """
    Check that the HTML is a full document (not just a snippet).

    Returns:
        True if it looks like a complete HTML document
    """
    return (
        has_doctype(html)
        and "</html>" in html.lower()
        and has_body_content(html)
    )


def is_valid_output(html: str) -> bool:
    """
    Full validation: complete document + has style + has body content.
    Used to decide whether to accept or retry an LLM response.

    Args:
        html: Cleaned HTML string

    Returns:
        True if output is usable
    """
    return is_complete_html(html) and has_style_block(html)


def validate_with_report(html: str) -> dict:
    """
    Return a detailed validation report — useful for debugging
    and for the evaluation pipeline.

    Args:
        html: Cleaned HTML string

    Returns:
        Dict with individual check results and overall pass/fail
    """
    report = {
        "has_doctype":     has_doctype(html),
        "has_style_block": has_style_block(html),
        "has_body":        has_body_content(html),
        "is_complete":     is_complete_html(html),
    }
    report["valid"] = all(report.values())
    return report


# ── 3. Injection & Repair ──────────────────────────────────────────────────────

CSS_RESET = """
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; }
"""

def inject_css_reset(html: str) -> str:
    """
    Inject a minimal CSS reset into the first <style> block found.
    Ensures consistent rendering across browsers.

    Args:
        html: HTML string

    Returns:
        HTML with reset injected
    """
    if "<style>" in html:
        return html.replace("<style>", f"<style>{CSS_RESET}", 1)
    elif "<style " in html:
        return re.sub(r"(<style[^>]*>)", rf"\1{CSS_RESET}", html, count=1)
    return html


def ensure_viewport_meta(html: str) -> str:
    """
    Inject a viewport meta tag if not already present.
    Ensures mobile-friendly rendering in preview.

    Args:
        html: HTML string

    Returns:
        HTML with viewport meta tag
    """
    viewport = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    if "viewport" not in html:
        html = html.replace("<head>", f"<head>\n  {viewport}", 1)
    return html


def wrap_snippet(snippet: str, dark: bool = False) -> str:
    """
    Wrap an HTML snippet (not a full document) into a complete HTML page.
    Useful when the model returns just a component without a full doc structure.

    Args:
        snippet: HTML fragment (e.g. just a <div>...</div>)
        dark:    Use dark background

    Returns:
        Full HTML document string
    """
    bg = "#0f0f13" if dark else "#f3f4f6"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      background: {bg};
      font-family: system-ui, -apple-system, sans-serif;
      padding: 2rem;
    }}
  </style>
</head>
<body>
  {snippet}
</body>
</html>"""


def repair_html(html: str, dark: bool = False) -> str:
    """
    Attempt to repair a broken or incomplete HTML response.
    - If it's a full doc: inject reset + viewport
    - If it's a snippet: wrap it in a full page

    Args:
        html:  Cleaned HTML string
        dark:  Dark theme flag for snippet wrapping

    Returns:
        Best-effort repaired HTML string
    """
    if is_complete_html(html):
        html = inject_css_reset(html)
        html = ensure_viewport_meta(html)
        return html
    else:
        # Treat as a snippet and wrap it
        return wrap_snippet(html, dark=dark)


# ── 4. Metadata extraction ─────────────────────────────────────────────────────

def extract_colors(html: str) -> list[str]:
    """
    Extract hex color values used in the HTML/CSS.

    Args:
        html: HTML string

    Returns:
        Sorted list of unique hex colors found
    """
    colors = re.findall(r"#([0-9a-fA-F]{3,6})\b", html)
    return sorted(set(f"#{c.upper()}" for c in colors))


def extract_fonts(html: str) -> list[str]:
    """
    Extract font-family names referenced in the HTML.

    Args:
        html: HTML string

    Returns:
        List of font names found
    """
    fonts = re.findall(r"font-family\s*:\s*([^;\"'}{]+)", html)
    cleaned = []
    for f in fonts:
        name = f.split(",")[0].strip().strip("'\"")
        if name:
            cleaned.append(name)
    return list(dict.fromkeys(cleaned))  # deduplicate, preserve order


def extract_components(html: str) -> list[str]:
    """
    Detect which UI components are present in the HTML
    based on tag and class name patterns.

    Args:
        html: HTML string

    Returns:
        List of detected component names
    """
    checks = {
        "button":     r"<button",
        "input":      r"<input",
        "form":       r"<form",
        "navbar":     r"<nav",
        "card":       r'class=["\'][^"\']*card',
        "modal":      r'class=["\'][^"\']*modal',
        "table":      r"<table",
        "badge":      r'class=["\'][^"\']*badge',
        "alert":      r'class=["\'][^"\']*alert',
        "dropdown":   r'class=["\'][^"\']*dropdown',
        "checkbox":   r'type=["\']checkbox',
        "radio":      r'type=["\']radio',
        "image":      r"<img",
        "link":       r"<a\s",
        "list":       r"<ul|<ol",
        "grid":       r"display\s*:\s*grid",
        "flexbox":    r"display\s*:\s*flex",
    }
    found = []
    for name, pattern in checks.items():
        if re.search(pattern, html, re.IGNORECASE):
            found.append(name)
    return found


def summarize(html: str) -> dict:
    """
    Return a full metadata summary of a generated HTML output.
    Useful for dataset labeling and evaluation.

    Args:
        html: HTML string

    Returns:
        Dict with validation, colors, fonts, components, line count
    """
    return {
        "validation":  validate_with_report(html),
        "colors":      extract_colors(html),
        "fonts":       extract_fonts(html),
        "components":  extract_components(html),
        "line_count":  html.count("\n") + 1,
        "char_count":  len(html),
    }


# ── 5. Sanitization ────────────────────────────────────────────────────────────

UNSAFE_TAGS = ["script", "iframe", "object", "embed", "form"]

def sanitize_html(html: str, allow_js: bool = False) -> str:
    """
    Remove potentially unsafe tags from generated HTML.
    Called before rendering in the live preview.

    Args:
        html:     HTML string
        allow_js: If True, keep <script> tags

    Returns:
        Sanitized HTML string
    """
    tags = UNSAFE_TAGS if allow_js else UNSAFE_TAGS
    if allow_js and "script" in tags:
        tags = [t for t in tags if t != "script"]

    for tag in tags:
        html = re.sub(
            rf"<{tag}[\s>].*?</{tag}>", "", html,
            flags=re.DOTALL | re.IGNORECASE
        )
    return html


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """```html
<!DOCTYPE html>
<html>
<head>
<style>
  body { display:flex; justify-content:center; align-items:center;
         min-height:100vh; background:#f3f4f6; font-family:Inter,sans-serif; }
  .btn { background:#2563eb; color:#fff; padding:12px 28px;
         border:none; border-radius:10px; font-size:1rem; cursor:pointer; }
  .btn:hover { background:#1d4ed8; }
</style>
</head>
<body>
  <button class="btn">Click me</button>
</body>
</html>
```"""

    html = clean_html(sample)
    print("=== CLEANED ===")
    print(html[:200])

    print("\n=== VALIDATION ===")
    print(validate_with_report(html))

    print("\n=== SUMMARY ===")
    s = summarize(html)
    for k, v in s.items():
        print(f"  {k}: {v}")
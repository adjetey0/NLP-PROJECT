"""
generate_dataset.py
===================
Generates NL → HTML/CSS pairs using OpenRouter API (free models).
Uses prompt_builder.py and html_utils.py for clean, consistent output.
Saves to data/dataset.json with auto-resume support.

Usage:
  python data/generate_dataset.py                   # full run with variations
  python data/generate_dataset.py --no-variations   # seed prompts only (faster)
  python data/generate_dataset.py --output data/my_dataset.json
  python data/generate_dataset.py --delay 1.0       # slower, safer rate limit
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from utils.prompt_builder import build_system_prompt, build_variation_prompt, PromptConfig
from utils.html_utils import clean_html, is_valid_output, summarize

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# Free models on OpenRouter — tries each one in order if previous fails
FREE_MODELS = [
    "qwen/qwen3-coder:free",
    "deepseek/deepseek-r1:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


# Seed prompts

SEED_PROMPTS = [
    # Buttons
    "A blue primary button with rounded corners and a hover effect",
    "A red delete button with a trash icon",
    "A ghost button with a border and transparent background",
    "A large green submit button that takes full width",
    "A disabled gray button that looks unclickable",

    # Cards
    "A product card with an image, title, price, and Add to Cart button",
    "A user profile card with avatar, name, role, and social links",
    "A blog post card with a cover image, title, date, and read more link",
    "A stats card showing a number, label, and percentage change",
    "A dark card with a glowing border effect",

    # Navigation
    "A dark navbar with a logo on the left and nav links on the right",
    "A sidebar navigation with icons and labels",
    "A breadcrumb navigation showing Home > Products > Detail",
    "A tab bar with 4 tabs where the first one is active",
    "A sticky top navbar with a shadow",

    # Forms
    "A login form with email, password fields and a submit button",
    "A search bar with a magnifying glass icon inside",
    "A newsletter signup with an email input and subscribe button",
    "A contact form with name, email, message fields and a send button",
    "A dropdown select menu for choosing a country",

    # Badges & Tags
    "A green success badge saying Completed",
    "A row of colorful skill tags like Python, React, Machine Learning",
    "A notification badge with a red dot on a bell icon",
    "A status indicator showing Online with a green dot",
    "A set of filter chips that can be selected or deselected",

    # Layout
    "A pricing table with Free, Pro, and Enterprise tiers",
    "A hero section with a headline, subtitle, and two CTA buttons",
    "A footer with 3 columns of links and a copyright notice",
    "A modal dialog with a title, message, and confirm/cancel buttons",
    "A toast notification saying Saved successfully with a close button",

    # Misc
    "A progress bar at 70% completion",
    "A dark/light mode toggle switch",
    "A star rating component showing 4 out of 5 stars",
    "An avatar group showing 4 overlapping profile pictures",
    "A timeline showing 3 events with dates and descriptions",
]


# ── Generation config ──────────────────────────────────────────────────────────

GENERATION_CONFIG = PromptConfig(
    use_js=False,
    use_tailwind=False,
    dark_theme=False,
    use_few_shot=True,
)


# ── OpenRouter API call ────────────────────────────────────────────────────────

def call_openrouter(system: str, user: str, model: str) -> str | None:
    """
    Call the OpenRouter API with a given model.

    Args:
        system: System prompt
        user:   User message
        model:  OpenRouter model string

    Returns:
        Response text or None on failure
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/nlp-project",
        "X-Title":       "NL-to-HTML Dataset Generator",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": 2000,
        "temperature": 0.7,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── Variation generator ────────────────────────────────────────────────────────

def generate_variations(seed: str, n: int = 3) -> list[str]:
    """Generate n natural language variations of a seed prompt."""
    system, user = build_variation_prompt(seed, n=n)
    for model in FREE_MODELS:
        try:
            text = call_openrouter(system, user, model)
            if text:
                # Extract JSON array from response
                start = text.find("[")
                end   = text.rfind("]") + 1
                if start != -1 and end > start:
                    variations = json.loads(text[start:end])
                    if isinstance(variations, list):
                        return [v for v in variations if isinstance(v, str)]
        except Exception as e:
            print(f"  ⚠ Variation error ({model}): {e}")
            continue
    return []


# ── HTML generator ─────────────────────────────────────────────────────────────

def generate_html(prompt: str,
                  config: PromptConfig,
                  retries: int = 2) -> dict | None:
    """Generate HTML for a prompt, validate it, and return a dataset entry."""
    system = build_system_prompt(config)
    user   = f'Create this UI component: "{prompt}"'

    for attempt in range(1, retries + 2):
        for model in FREE_MODELS:
            try:
                raw  = call_openrouter(system, user, model)
                html = clean_html(raw)

                if not is_valid_output(html):
                    print(f"  ↺ Invalid output ({model}), attempt {attempt}")
                    continue

                meta = summarize(html)
                return {
                    "prompt":     prompt,
                    "html":       html,
                    "model":      model,
                    "components": meta["components"],
                    "colors":     meta["colors"],
                    "fonts":      meta["fonts"],
                    "char_count": meta["char_count"],
                    "valid":      meta["validation"]["valid"],
                }
            except Exception as e:
                print(f"  ⚠ API error ({model}) attempt {attempt}: {e}")
                time.sleep(2)
                continue

        if attempt <= retries:
            print(f"  ↺ All models failed, retrying ({attempt}/{retries})...")
            time.sleep(3)

    print(f"  ✗ Skipping '{prompt[:45]}' after all attempts")
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    use_variations:      bool  = True,
    output_path:         str   = "data/dataset.json",
    delay:               float = 1.0,
    variation_seeds:     int   = 10,
    variations_per_seed: int   = 3,
):
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        return

    print(f"\n{'═' * 50}")
    print(f"  NL → HTML Dataset Generator (OpenRouter)")
    print(f"{'═' * 50}")
    print(f"  Models : {', '.join(FREE_MODELS)}")
    print(f"  Output : {output_path}")
    print(f"{'═' * 50}\n")

    # Resume support
    dataset = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            dataset = json.load(f)
        existing_prompts = {d["prompt"] for d in dataset}
        print(f"📂 Resuming — {len(dataset)} pairs already saved")
    else:
        existing_prompts = set()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build prompt list
    all_prompts = list(SEED_PROMPTS)

    if use_variations:
        print(f"\n🔀 Generating variations for {variation_seeds} seed prompts...")
        for seed in tqdm(SEED_PROMPTS[:variation_seeds], desc="Variations"):
            variations = generate_variations(seed, n=variations_per_seed)
            all_prompts.extend(variations)
            time.sleep(delay)
        print(f"   → {len(all_prompts)} total prompts after variations")

    new_prompts = [p for p in all_prompts if p not in existing_prompts]
    print(f"\n⚡ Generating HTML for {len(new_prompts)} new prompts...\n")

    if not new_prompts:
        print("✅ Nothing new to generate.")
        return

    # Generation loop
    skipped = 0
    for i, prompt in enumerate(tqdm(new_prompts, desc="Generating"), 1):
        entry = generate_html(prompt, GENERATION_CONFIG)
        if entry:
            dataset.append(entry)
        else:
            skipped += 1

        if i % 5 == 0:
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)

        time.sleep(delay)

    # Final save
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n{'─' * 50}")
    print(f"✅ Done!")
    print(f"   Total pairs saved : {len(dataset)}")
    print(f"   Skipped (invalid) : {skipped}")
    print(f"   Output file       : {output_path}")
    print(f"{'─' * 50}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NL → HTML/CSS dataset via OpenRouter")
    parser.add_argument("--no-variations",       action="store_true",
                        help="Skip variation generation (faster)")
    parser.add_argument("--output",              default="data/dataset.json",
                        help="Output JSON file path")
    parser.add_argument("--delay",               type=float, default=1.0,
                        help="Delay between API calls in seconds")
    parser.add_argument("--variation-seeds",     type=int, default=10,
                        help="Number of seed prompts to generate variations for")
    parser.add_argument("--variations-per-seed", type=int, default=3,
                        help="Number of variations per seed prompt")
    args = parser.parse_args()

    main(
        use_variations=not args.no_variations,
        output_path=args.output,
        delay=args.delay,
        variation_seeds=args.variation_seeds,
        variations_per_seed=args.variations_per_seed,
    )
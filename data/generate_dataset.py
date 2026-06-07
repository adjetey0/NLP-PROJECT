"""
generate_dataset.py
===================
Generates NL → HTML/CSS pairs using the Anthropic API.
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

import anthropic
import json
import time
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

from utils.prompt_builder import build_system_prompt, build_variation_prompt, PromptConfig
from utils.html_utils import clean_html, is_valid_output, summarize

load_dotenv()


# ── Seed prompts ───────────────────────────────────────────────────────────────

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


# ── Variation generator ────────────────────────────────────────────────────────

def generate_variations(client: anthropic.Anthropic, seed: str, n: int = 3) -> list[str]:
    """Generate n natural language variations of a seed prompt."""
    system, user = build_variation_prompt(seed, n=n)
    try:
        msg = client.messages.create(
            model="claude-haiku-20240307",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        text = msg.content[0].text.strip()
        variations = json.loads(text)
        if isinstance(variations, list):
            return [v for v in variations if isinstance(v, str)]
    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠ Variation error for '{seed[:40]}': {e}")
    return []


# ── HTML generator ─────────────────────────────────────────────────────────────

def generate_html(client: anthropic.Anthropic,
                  prompt: str,
                  config: PromptConfig,
                  retries: int = 2) -> dict | None:
    """Generate HTML for a prompt, validate it, and return a dataset entry."""
    system = build_system_prompt(config)

    for attempt in range(1, retries + 2):
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": f'Create this UI component: "{prompt}"'}]
            )
            html = clean_html(msg.content[0].text)

            if not is_valid_output(html):
                if attempt <= retries:
                    print(f"  ↺ Invalid output on attempt {attempt}, retrying...")
                    time.sleep(1)
                    continue
                else:
                    print(f"  ✗ Skipping '{prompt[:45]}' after {retries + 1} attempts")
                    return None

            meta = summarize(html)
            return {
                "prompt":     prompt,
                "html":       html,
                "components": meta["components"],
                "colors":     meta["colors"],
                "fonts":      meta["fonts"],
                "char_count": meta["char_count"],
                "valid":      meta["validation"]["valid"],
            }

        except Exception as e:
            print(f"  ⚠ API error on attempt {attempt}: {e}")
            if attempt <= retries:
                time.sleep(2)

    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    use_variations: bool = True,
    output_path: str = "data/dataset.json",
    delay: float = 0.5,
    variation_seeds: int = 10,
    variations_per_seed: int = 3,
):
    client = anthropic.Anthropic()

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
            variations = generate_variations(client, seed, n=variations_per_seed)
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
        entry = generate_html(client, prompt, GENERATION_CONFIG)
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
    parser = argparse.ArgumentParser(description="Generate NL → HTML/CSS dataset")
    parser.add_argument("--no-variations",       action="store_true",
                        help="Skip variation generation (faster)")
    parser.add_argument("--output",              default="data/dataset.json",
                        help="Output JSON file path")
    parser.add_argument("--delay",               type=float, default=0.5,
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
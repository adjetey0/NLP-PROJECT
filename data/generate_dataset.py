"""
Dataset Generation Script
Generates NL → HTML/CSS pairs using the Anthropic API.
Saves output to data/dataset.json
"""

import anthropic
import json
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ─── Seed prompts ────────────────────────────────────────────────────────────
# These are the base descriptions. The script will also generate variations.

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

    # Layout components
    "A pricing table with Free, Pro, and Enterprise tiers",
    "A hero section with a headline, subtitle, and two CTA buttons",
    "A footer with 3 columns of links and a copyright notice",
    "A modal dialog with a title, message, and confirm/cancel buttons",
    "A toast notification saying 'Saved successfully' with a close button",

    # Misc
    "A progress bar at 70% completion",
    "A dark/light mode toggle switch",
    "A star rating component showing 4 out of 5 stars",
    "An avatar group showing 4 overlapping profile pictures",
    "A timeline showing 3 events with dates and descriptions",
]


# ─── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert frontend developer. Convert natural language UI descriptions into clean, self-contained HTML/CSS code.

Rules:
- Return ONLY the HTML code. No explanations, no markdown fences, no comments.
- Include a <style> block inside <head> for all CSS
- Make the component visually polished and modern
- Center the component on the page with a light gray (#f3f4f6) background
- The full output must be a complete HTML document starting with <!DOCTYPE html>
- Use plain CSS only, no frameworks
- No JavaScript unless it's essential for the component to make sense
"""


# ─── Variation generator ─────────────────────────────────────────────────────

VARIATION_SYSTEM = """You are a creative UI copywriter. Given a UI component description, generate 3 natural language variations of it.
Return ONLY a JSON array of 3 strings. No explanation, no markdown. Example:
["A red warning button", "A danger button in crimson", "A bold alert button with red background"]
"""

def generate_variations(client, prompt: str) -> list[str]:
    """Generate variations of a seed prompt."""
    try:
        msg = client.messages.create(
            model="claude-haiku-20240307",
            max_tokens=300,
            system=VARIATION_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        return json.loads(text)
    except Exception:
        return []


# ─── HTML generator ───────────────────────────────────────────────────────────

def generate_html(client, prompt: str) -> str | None:
    """Generate HTML/CSS for a given prompt."""
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        html = msg.content[0].text.strip()
        # Strip markdown fences if present
        if html.startswith("```"):
            html = html.split("\n", 1)[1]
            html = html.rsplit("```", 1)[0]
        return html.strip()
    except Exception as e:
        print(f"  ⚠ Error generating HTML: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(
    use_variations: bool = True,
    output_path: str = "data/dataset.json",
    delay: float = 0.5,
):
    client = anthropic.Anthropic()
    dataset = []

    # Load existing dataset to allow resuming
    if os.path.exists(output_path):
        with open(output_path) as f:
            dataset = json.load(f)
        existing_prompts = {d["prompt"] for d in dataset}
        print(f"📂 Resuming — {len(dataset)} pairs already exist")
    else:
        existing_prompts = set()

    # Build full prompt list
    all_prompts = list(SEED_PROMPTS)

    if use_variations:
        print("🔀 Generating prompt variations...")
        for seed in tqdm(SEED_PROMPTS[:10], desc="Variations"):  # limit to first 10 seeds
            variations = generate_variations(client, seed)
            all_prompts.extend(variations)
            time.sleep(delay)

    # Filter already-generated prompts
    all_prompts = [p for p in all_prompts if p not in existing_prompts]
    print(f"\n⚡ Generating HTML for {len(all_prompts)} prompts...\n")

    for prompt in tqdm(all_prompts, desc="Generating"):
        html = generate_html(client, prompt)
        if html:
            dataset.append({"prompt": prompt, "html": html})

        # Save after every 5 generations
        if len(dataset) % 5 == 0:
            with open(output_path, "w") as f:
                json.dump(dataset, f, indent=2)

        time.sleep(delay)

    # Final save
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Done! {len(dataset)} pairs saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate NL → HTML dataset")
    parser.add_argument("--no-variations", action="store_true", help="Skip variation generation")
    parser.add_argument("--output", default="data/dataset.json", help="Output file path")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    main(
        use_variations=not args.no_variations,
        output_path=args.output,
        delay=args.delay,
    )
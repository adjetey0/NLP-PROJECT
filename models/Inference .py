"""
inference.py
============
Run inference with either the fine-tuned CodeT5+ model
or the baseline Anthropic API — for side-by-side comparison.

Usage:
  # Single prompt with fine-tuned model
  python models/inference.py --prompt "A blue button with rounded corners"

  # Compare fine-tuned vs baseline API
  python models/inference.py --prompt "A blue button" --compare

  # Batch inference on a prompts file
  python models/inference.py --batch data/test_prompts.json \
                              --output data/predictions.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

import anthropic
from dotenv import load_dotenv
from utils.prompt_builder import quick_prompt
from utils.html_utils import clean_html, repair_html, is_valid_output

load_dotenv()

MODEL_NAME  = "Salesforce/codet5p-220m"
MAX_INPUT   = 128
MAX_OUTPUT  = 1024


# ── Load fine-tuned model ──────────────────────────────────────────────────────

def load_finetuned_model(model_dir: str):
    """
    Load the fine-tuned CodeT5+ model and tokenizer.

    Args:
        model_dir: Path to saved model directory

    Returns:
        (model, tokenizer) tuple
    """
    print(f"📦 Loading fine-tuned model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base      = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model     = PeftModel.from_pretrained(base, model_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print("   ✓ Model loaded")
    return model, tokenizer


# ── Fine-tuned inference ───────────────────────────────────────────────────────

def generate_finetuned(prompt: str,
                       model,
                       tokenizer,
                       temperature: float = 0.7,
                       num_beams:   int   = 4) -> str:
    """
    Generate HTML using the fine-tuned CodeT5+ model.

    Args:
        prompt:      NL description
        model:       Loaded PEFT model
        tokenizer:   Loaded tokenizer
        temperature: Sampling temperature (higher = more creative)
        num_beams:   Beam search width (higher = better quality, slower)

    Returns:
        Generated HTML string
    """
    input_text = f"Generate HTML for: {prompt}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_OUTPUT,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    html = tokenizer.decode(outputs[0], skip_special_tokens=True)
    html = clean_html(html)
    html = repair_html(html)
    return html


# ── Baseline API inference ─────────────────────────────────────────────────────

def generate_baseline(prompt: str,
                      dark:     bool = False,
                      tailwind: bool = False,
                      js:       bool = False) -> str:
    """
    Generate HTML using the Anthropic API (baseline).

    Args:
        prompt:   NL description
        dark:     Dark theme flag
        tailwind: Tailwind CSS flag
        js:       JavaScript flag

    Returns:
        Generated HTML string
    """
    client = anthropic.Anthropic()
    system, user = quick_prompt(prompt, dark=dark, tailwind=tailwind, js=js)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": user}]
    )
    html = clean_html(msg.content[0].text)
    html = repair_html(html)
    return html


# ── Comparison ─────────────────────────────────────────────────────────────────

def compare(prompt: str, model_dir: str) -> dict:
    """
    Run the same prompt through both models and return both outputs.

    Args:
        prompt:    NL description
        model_dir: Path to fine-tuned model

    Returns:
        Dict with finetuned and baseline HTML + validity flags
    """
    model, tokenizer = load_finetuned_model(model_dir)

    print("\n🔧 Fine-tuned model generating...")
    ft_html  = generate_finetuned(prompt, model, tokenizer)
    ft_valid = is_valid_output(ft_html)

    print("🌐 Baseline API generating...")
    bl_html  = generate_baseline(prompt)
    bl_valid = is_valid_output(bl_html)

    return {
        "prompt": prompt,
        "finetuned": {
            "html":  ft_html,
            "valid": ft_valid,
            "chars": len(ft_html),
        },
        "baseline": {
            "html":  bl_html,
            "valid": bl_valid,
            "chars": len(bl_html),
        },
    }


# ── Batch inference ────────────────────────────────────────────────────────────

def batch_inference(prompts_path: str,
                    output_path:  str,
                    model_dir:    str,
                    use_baseline: bool = False) -> None:
    """
    Run inference on a list of prompts and save predictions.

    Args:
        prompts_path: JSON file with list of prompt strings or dicts
        output_path:  Where to save predictions JSON
        model_dir:    Path to fine-tuned model
        use_baseline: If True, use API instead of fine-tuned model
    """
    with open(prompts_path) as f:
        data = json.load(f)

    # Accept either a list of strings or list of dicts with 'prompt' key
    prompts = [d if isinstance(d, str) else d["prompt"] for d in data]
    print(f"📂 {len(prompts)} prompts loaded from {prompts_path}")

    predictions = []

    if not use_baseline:
        model, tokenizer = load_finetuned_model(model_dir)

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{len(prompts)}] {prompt[:55]}...")
        try:
            if use_baseline:
                html = generate_baseline(prompt)
            else:
                html = generate_finetuned(prompt, model, tokenizer)
            predictions.append({"prompt": prompt, "html": html})
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            predictions.append({"prompt": prompt, "html": ""})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    valid = sum(1 for p in predictions if is_valid_output(p["html"]))
    print(f"\n✅ Saved {len(predictions)} predictions to {output_path}")
    print(f"   Valid: {valid}/{len(predictions)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned CodeT5+")
    parser.add_argument("--prompt",    type=str, default=None,
                        help="Single NL prompt to generate HTML for")
    parser.add_argument("--model-dir", type=str, default="models/codet5-html",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--compare",   action="store_true",
                        help="Compare fine-tuned vs baseline API output")
    parser.add_argument("--batch",     type=str, default=None,
                        help="Path to JSON file with list of prompts")
    parser.add_argument("--output",    type=str, default="data/predictions.json",
                        help="Output path for batch predictions")
    parser.add_argument("--baseline",  action="store_true",
                        help="Use baseline API instead of fine-tuned model")
    args = parser.parse_args()

    # Single prompt
    if args.prompt and not args.compare:
        if args.baseline:
            print("🌐 Generating with baseline API...")
            html = generate_baseline(args.prompt)
        else:
            model, tokenizer = load_finetuned_model(args.model_dir)
            print("🔧 Generating with fine-tuned model...")
            html = generate_finetuned(args.prompt, model, tokenizer)

        print("\n" + "─" * 50)
        print(html[:500] + ("..." if len(html) > 500 else ""))
        print("─" * 50)
        print(f"Valid: {is_valid_output(html)} | Chars: {len(html)}")

    # Comparison mode
    elif args.prompt and args.compare:
        result = compare(args.prompt, args.model_dir)
        print(f"\n{'═'*50}")
        print(f"  FINE-TUNED  | valid={result['finetuned']['valid']} | {result['finetuned']['chars']} chars")
        print(f"  BASELINE    | valid={result['baseline']['valid']}  | {result['baseline']['chars']} chars")
        print(f"{'═'*50}")

    # Batch mode
    elif args.batch:
        batch_inference(args.batch, args.output, args.model_dir, args.baseline)

    else:
        parser.print_help()
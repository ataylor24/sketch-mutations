import argparse
import os
import json
from typing import List, Dict

from tqdm.auto import tqdm
import yaml

from utils import format_prompt
from api_inference import _inference, load_inference_client  # type: ignore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_dataset(path: str, mode_cfg: dict, split: str | None = None) -> List[Dict]:
    """Load the ProofNet jsonl file and optionally filter by split (valid / test)."""
    examples: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dp = json.loads(line)
            if split is None or split == "all" or dp.get("split") == split:
                examples.append(dp)
    if mode_cfg["development"]:
        examples = examples[:mode_cfg["n_problems"]]
    return examples


def build_statement(dp: Dict) -> str:
    """Construct a Lean 4 statement from a ProofNet entry.

    The dataset already contains separate fields:
      * header          – import and open lines
      * informal_prefix – doc-string description of theorem
      * formal_statement – Lean theorem header ending with ':='
    We append a minimal proof skeleton ("by sorry") so the snippet
    compiles syntactically while keeping the proof body empty.
    """
    header = dp.get("header", "").rstrip()
    informal_prefix = dp.get("informal_prefix", "").rstrip()
    theorem_stmt = dp.get("formal_statement", "").rstrip()

    # Ensure the statement ends with a proof placeholder
    if theorem_stmt.endswith(":="):
        theorem_stmt = f"{theorem_stmt} by\n  sorry"
    elif theorem_stmt.endswith("by"):
        theorem_stmt = f"{theorem_stmt}\n  sorry"

    parts = [segment for segment in [header, informal_prefix, theorem_stmt] if segment]
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    # ------------------------------------------------------------------
    # Load YAML configuration
    # ------------------------------------------------------------------
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve parameters: CLI overrides YAML
    dataset_path = cfg["input_jsonl"]

    # Output directory: prefer CLI, else stage2_out, else default "output"
    prompt_dir = cfg["stage1_out_json"]
    lean_statement_dir = cfg["stage1_out_lean"]

    # Sketcher settings
    sketcher_cfg = cfg["mutator_sketcher"]
    model = sketcher_cfg["model_name"]
    reasoning = sketcher_cfg["reasoning"]

    # Prompt type decides informal2formal flag
    cfg_prompt_type = sketcher_cfg["prompt_type"]

    # Num examples – CLI overrides YAML mode.n_problems
    mode_cfg = cfg["mode"]
    
    # ------------------------------------------------------------------
    # Prepare inference client using shared helper
    # ------------------------------------------------------------------
    # Apply CLI overrides to sketcher_cfg and reload client
    sketcher_cfg["model_name"] = model
    sketcher_cfg["reasoning"] = reasoning

    inference_client = load_inference_client(sketcher_cfg)

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(lean_statement_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Iterate over examples
    # ------------------------------------------------------------------
    data = load_dataset(dataset_path, mode_cfg, split="test")
    for idx, dp in enumerate(tqdm(data, desc="Generating proofs")):
        file_stub = dp.get("name", f"sample_{idx}")
        json_path = os.path.join(prompt_dir, f"{file_stub}.json")
        lean_path = os.path.join(lean_statement_dir, f"{file_stub}.lean")
                
        if os.path.exists(json_path):
            print(f"Skipping {file_stub} because it already exists")
            continue
        
        statement = build_statement(dp)
        prompt = format_prompt(statement, format_type=cfg_prompt_type)
        # API call via shared _inference helper
        # We temporarily construct a minimal config dict expected by _inference.
        try:
            response_text = _inference(inference_client, prompt, sketcher_cfg)
        except Exception as e:
            print(f"Error during API call: {e}")
            response_text = f"ERROR: {e}"
        # ------------------------------------------------------------------
        # Persist results per-example
        # ------------------------------------------------------------------
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"statement": statement, "response": response_text}, f, indent=4, ensure_ascii=False)

        # Try to extract the last Lean 4 block (if any)
        lean_resp = ""
        if "```lean4" in response_text:
            lean_resp = response_text.split("```lean4")[-1].split("```", 1)[0].strip()
        else:
            lean_resp = response_text.split("```lean")[-1].split("```", 1)[0].strip()

        lean_path = os.path.join(lean_statement_dir, f"{file_stub}.lean")
        with open(lean_path, "w", encoding="utf-8") as f:
            f.write(lean_resp)

        # # Convenience: print a short preview to stdout
        # print(f"\nExample {idx + 1}: {file_stub}\n{'-' * 80}\n{lean_resp}\n{'-' * 80}\n")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate informal proofs & Lean sketches on ProofNet with o3.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

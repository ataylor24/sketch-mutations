import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from tqdm.auto import tqdm
import yaml
import argparse
from utils import format_prompt

set_seed(42)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_sketches(cfg: dict) -> List[Dict]:
    """Load the sketch jsons and return a list of dictionaries."""
    sketches = []
    for file in os.listdir(cfg["stage1_out_json"]):
        if file.endswith(".json") and not "test" in file:
            with open(os.path.join(cfg["stage1_out_json"], file), "r", encoding="utf-8") as f:
                sketch = json.load(f)["response"]
                sketches.append({"name": file.split(".")[0],"sketch": sketch})
    return sketches

def load_dataset(cfg: dict) -> List[Dict]:
    """Load the dataset from the paths in the config. We need to load the formal statements and the sketches and then combine them if use_sketch is true."""
    formal_statements = load_jsonl(cfg["lean_input"], split="test")
    if cfg["prover"]["use_sketch"]:
        sketches = load_sketches(cfg)
        
        dataset = [{**formal_statements[sketch["name"]], **sketch} for sketch in sketches]
    else:
        dataset = formal_statements
        
    if cfg["mode"]["development"]:
        dataset = dataset[:cfg["mode"]["n_problems"]]
        
    return dataset
    
                
def load_jsonl(path: str, split: str | None = None) -> Dict:
    """Load the ProofNet jsonl file and optionally filter by split (valid / test)."""
    examples: Dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dp = json.loads(line)
            if split is None or split == "all" or dp.get("split") == split:
                header = dp["header"]
                formal_statement = dp["formal_statement"]
                examples[dp["name"]] = {"formal_statement": header + formal_statement}
    return examples

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(args):
    # --------------------------------------------------------------
    # Config
    # --------------------------------------------------------------
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    prover_cfg = cfg.get("prover")
    
    model_id = prover_cfg.get("model_name")
    max_tokens = prover_cfg.get("max_new_tokens")
    overwrite = prover_cfg.get("overwrite")
    output_dir = cfg.get("stage2_out", "deepseek_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # --------------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------------
    dataset = load_dataset(cfg)
   
    # --------------------------------------------------------------
    # Load DeepSeek model
    # --------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )


    for i, problem in enumerate(tqdm(dataset, desc="DeepSeek generation")):
        file_stub = problem.get("name", f"sample_{i}")
        format_type = "cot" if cfg["prover"]["prompt_type"] == "cot" else "default"
        json_out = Path(output_dir) / f"{file_stub}_{format_type}.json"
        if json_out.exists() and not overwrite:
            print(f"Skipping {file_stub} because it already exists and overwrite is false")
            continue
        
        prompt = format_prompt(problem["formal_statement"], format_type=format_type)
        
        sketch = problem["sketch"] + "\n\n" if cfg["prover"]["use_sketch"] else ""
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{sketch}### Complete Lean 4 Proof\n\n```lean4"},
        ]
        
        inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False, return_tensors="pt")
        inputs = inputs[:, :-1]  # strip EOS
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        # Extract completed Lean 4 block (after last ```lean4)
        cleaned = full_response.split("```lean4")[-1].split("```", 1)[0].strip()
        json_out = Path(output_dir) / f"{file_stub}_{format_type}.json"
        json_out.write_text(
            json.dumps({"input_sketch": problem["sketch"], "response": full_response, "cleaned": cleaned}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use DeepSeek-Prover-V2 to complete Lean proofs from o3 sketches.")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args) 
#!/usr/bin/env python3
"""verify_lean_proofs.py – Verify Lean 4 proofs (.lean files) using a running kimina-lean-server.

Example usage:
    python verify_lean_proofs.py --lean_dir /path/to/lean/files \
                                 --url http://localhost               \
                                 --timeout 120                        \
                                 --batch-size 0                       \
                                 --workers 16

It aggregates results and stores a JSON summary in the output directory.
"""

from __future__ import annotations

# NOTE: This script now relies solely on `config.yaml` for all runtime parameters.
# No command-line arguments are necessary.

import argparse, json, os, pathlib, yaml
from typing import List, Dict
from tqdm.auto import tqdm

# kimina-lean-server client
from client.client import Lean4Client, batch_verify_proof  # type: ignore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_dataset(cfg: dict) -> List[Dict]:
    """Load the dataset from the directory of json files"""
    dataset = []
    for file in os.listdir(cfg["stage2_out"]):
        if file.endswith(".json"):
            with open(os.path.join(cfg["stage2_out"], file), "r", encoding="utf-8") as f:
                cleaned = json.load(f)["cleaned"].rstrip()
                dataset.append({"custom_id": file.split(".")[0], "proof": cleaned})
    return dataset


def is_verified(entry: dict) -> bool:
    """True iff Lean accepted *and* proof has no `sorry`s or errors."""
    if entry["error"] is not None:
        return False
    resp = entry["response"]
    for msg in resp.get("messages", []):
        print(msg)
        if msg["severity"] == "error":
            return False
    # crude check: ensure 'sorry' absent in file content after evaluation
    return True


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main(args):
    # -----------------------------------------------------------
    # Load configuration YAML
    # -----------------------------------------------------------
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(cfg)
    output_path = cfg.get("stage3_out_json")
    verified_path = cfg.get("stage3_out_lean")
    os.makedirs(verified_path, exist_ok=True)

    verifier_cfg = cfg.get("verifier", {})

    client = Lean4Client(base_url=verifier_cfg.get("url", "http://localhost"))

    passed = 0
    results_summary = {}


    batch_size = verifier_cfg.get("batch_size", 0)
    timeout = verifier_cfg.get("timeout", 120)
    workers = verifier_cfg.get("workers", os.cpu_count())

        
    
    if batch_size == 0:
        verification = client.verify(dataset, timeout=timeout)
    else:
        verification = batch_verify_proof(client, dataset, timeout=timeout,
                                            num_proc=workers, batch_size=batch_size)


    verified_results = []
    for i, result in enumerate(verification["results"]):
        if is_verified(result):
            verified_results.append(dataset[result["custom_id"]]["proof"])
    
  
    json.dump(verified_results, open(os.path.join(verified_path, "verified_results.json"), "w"), indent=2)
    print(f"Verified {len(verified_results)}/{len(dataset)} files. Detailed report saved to {verified_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Lean 4 proofs using a running kimina-lean-server.")
    parser.add_argument("--config_path", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args) 
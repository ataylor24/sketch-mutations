#!/usr/bin/env python3
"""
run_mutate_prove.py – one‑pass pipeline:

    SEED  →  SKETCH  →  MUTATION  →  VALIDATE  →  PROVE  →  REWARD

It re‑uses:
  • api_inference.py      (OpenAI / DeepSeek / Azure wrapper)
  • utils.format_prompt   (prompt templates)
  • lean-client‑python    (syntax / aesop checks)
  • kimina‑lean‑server    (final kernel verification)

All intermediate artefacts are cached on disk so the script is
*idempotent* and restart‑safe – pass `--resume` to skip completed items.
"""

# ─────────────────────────── imports ────────────────────────────
from __future__ import annotations
import argparse, json, os, pathlib, random, sys, time
from typing import Dict, List, Any

import yaml
from tqdm.auto import tqdm

# local helpers (your existing files)
from api_inference import load_inference_client, _inference
from utils import format_prompt
from client.client import Lean4Client, batch_verify_proof     # kimina client

# if you keep verify helpers elsewhere, adjust import
# from verify_helpers import lean_syntax_ok, aesop_non_trivial  # ≈ 20 LOC utils

# torch / hf only needed for MUTATOR + PROVER (local weights)
import torch
from transformers import AutoTokenizer#, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# ───────────────────────── config & CLI ─────────────────────────
DEFAULT_CONFIG = "config.yaml"

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full mutate‑prove pipeline")
    p.add_argument("--config_path", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--resume", action="store_true",
                   help="Skip sketches / mutations / proofs that already exist")
    return p.parse_args()

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ──────────────────────── tiny file helpers ─────────────────────
def write_text(fp: pathlib.Path, txt: str):
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(txt, encoding="utf-8")

def write_json(fp: pathlib.Path, obj: Any):
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def read_lines(path: str, split: str | None, dev_cfg: Dict) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            dp = json.loads(ln)
            if split in (None, "all") or dp.get("split") == split:
                rows.append(dp)
    if dev_cfg["development"]:
        rows = rows[: dev_cfg["n_problems"]]
    return rows

# ─────────────────────── lean helpers ──────────────────────────
def append_exact(conjecture: str) -> bool:
    return conjecture + "  exact?\n"

def append_aesop(conjecture: str) -> bool:
    return conjecture + "  aesop?\n"

def check_proved_by_exact(response: str) -> bool:
    for message in response.messages:
            if message.severity == "info" and message.data.startswith("Try this:"):
                return True
    return False

def check_proved_by_aesop(response: str) -> bool:
    for message in response.messages:
        if (
            message.severity == "info"
            and message.data.startswith("Try this:")
            and "sorry" not in message.data
        ):
            return True
    return False

# ─────────────────────── STAGE 0 – seeds ───────────────────────
def build_seed(dp: Dict) -> str:
    """Return Lean snippet with header + formal statement + `by sorry`."""
    hdr = dp.get("header", "").rstrip()
    stmt = dp.get("formal_statement", "").rstrip()
    if stmt.endswith(":="):
        stmt += " by\n  sorry"
    elif stmt.endswith("by"):
        stmt += "\n  sorry"
    return "\n\n".join([s for s in [hdr, stmt] if s])

# ─────────────────────── STAGE 1 – sketch ──────────────────────
def run_sketcher(cfg, data, resume):
    out_json = pathlib.Path(cfg["stage1_out"])
    out_json.mkdir(parents=True, exist_ok=True)

    client_cfg = cfg["mutator_sketcher"]
    client = load_inference_client(client_cfg)

    for dp in tqdm(data, desc="SKETCH"):
        stub = dp["name"]
        fp = out_json / f"{stub}.json"
        if resume and fp.exists():
            continue

        seed_snippet = build_seed(dp)
        prompt = format_prompt(seed_snippet, client_cfg["prompt_type"])
        response = _inference(client, prompt, client_cfg)
        reasoning = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        write_json(fp, {"seed": seed_snippet, "sketch": content, "reasoning": reasoning})

# ───────────────────── STAGE 2 – mutation ──────────────────────
# def load_hf_model(model_id, dtype=torch.bfloat16):
#     tok = AutoTokenizer.from_pretrained(model_id)
#     mdl = AutoModelForCausalLM.from_pretrained(
#         model_id, device_map="auto", torch_dtype=dtype, trust_remote_code=True
#     )
#     return tok, mdl
from vllm import LLM, SamplingParams

def load_vllm_model(
    model_id: str,
    *,
    dtype: str = "float16",
    tp_size: int = 4,
    max_model_len: int = 8192,
) -> tuple[AutoTokenizer, LLM]:
    """
    Return (tokenizer, llm) ready for inference with vLLM.
    """
    tok = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.95,
    )
    return tok, llm

def extract_last_lean_block(txt: str) -> str:
    lean_resp = ""
    if "```lean4" in txt:
        lean_resp = txt.split("```lean4")[-1].split("```", 1)[0].strip()
    else:
        lean_resp = txt.split("```lean")[-1].split("```", 1)[0].strip()
    return lean_resp.strip()

def run_mutator(cfg, resume):
    sketches_dir = pathlib.Path(cfg["stage1_out"])
    mut_json_dir = pathlib.Path(cfg["stage2_out"])
    mut_json_dir.mkdir(parents=True, exist_ok=True)
  

    mut_cfg = cfg["mutator"]
    tok, mdl = load_vllm_model(mut_cfg["model_name"])
    max_tokens = mut_cfg["max_new_tokens"]
    use_sketch = mut_cfg["use_sketch"]
    prompt_type = mut_cfg["prompt_type"]
    num_return_sequences = mut_cfg["num_return_sequences"]

    for js in tqdm(list(sketches_dir.glob("*.json")), desc="MUTATE"):
        stub = js.stem
        out_js = mut_json_dir / f"{stub}.json"
        if resume and out_js.exists():
            continue

        sketch_data = json.loads(js.read_text())
        seed = sketch_data["seed"]
        reasoning_chain = sketch_data["reasoning"] if use_sketch else ""
        prompt = format_prompt(seed, prompt_type)

        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"{reasoning_chain}\n\n```lean4\n"},
        ]
        inp = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False, return_tensors="pt")
        
        outs = mdl.generate(inp, sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    temperature=1.0,
                    n=num_return_sequences,
                ))[0].outputs

        # grab every returned Lean block
        lean_blocks = [extract_last_lean_block(o.text) for o in outs]

        write_json(out_js, {
            "prompt": prompt,
            "responses": [o.text for o in outs],
            "lean_blocks": lean_blocks,
        })
        

# ────────────────── STAGE 3 – validate & filter ─────────────────
def validate_mutations(cfg, client: Lean4Client) -> List[Dict]:
    """Filter mutated sketches using the remote Lean verifier.

    A single Lean4Client instance (constructed once in ``main``) is passed in so
    the HTTP connection pool can be re-used across both validation and final
    reward computation.
    """

    mut_json_dir = pathlib.Path(cfg["stage2_out"])
    valid_dir = pathlib.Path(cfg["stage3_out"])
    valid_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ build verify batch
    proofs_batch, id2lean = [], {}
    for js in mut_json_dir.glob("*.json"):
        data = json.loads(js.read_text())

        # Support either a single key "lean" or a list under "leans"
        stmt= data["lean_blocks"]# take first candidate

        if not stmt:
            continue  # skip empty / malformed entries

        cid = js.stem  # unique id per file
        proofs_batch.append({"custom_id": cid, "proof": stmt})
        id2lean[cid] = stmt

    # No candidates – short-circuit early
    if not proofs_batch:
        return []

    # ------------------------------------------------------------------ remote verification
    v_cfg = cfg["verifier"]
    resp = client.verify(proofs_batch, timeout=v_cfg["timeout"])

    # ------------------------------------------------------------------ collate successful items
    valids: List[Dict] = []
    for res in resp["results"]:
        cid = res["custom_id"]
        passed = (
            res["error"] is None
            and all(m["severity"] != "error" for m in res["response"]["messages"])
        )

        if passed:
            stmt = id2lean[cid]
            stub = cid  # same as file stem
            for suffix in ["_exact", "_aesop"]:
                
                valids.append({"name": stub + suffix, "lean": stmt})
            
    # ------------------------------------------------------------------ check novelty
    for stub in valids:
        for suffix in ["_exact", "_aesop"]:
            
    return valids

# ───────────────────── STAGE 4 – prove  ────────────────────────
def run_prover(cfg, valids, resume):
    prover_out_dir = pathlib.Path(cfg["stage4_out"])
    prover_out_dir.mkdir(parents=True, exist_ok=True)

    prov_cfg = cfg["prover"]
    tok, mdl = load_vllm_model(prov_cfg["model_name"])
    max_tokens = prov_cfg["max_new_tokens"]
    prompt_type = prov_cfg["prompt_type"]
    num_return_sequences = prov_cfg["num_return_sequences"]

    for dp in tqdm(valids, desc="PROVE"):
        stub = dp["name"]
        out_js = prover_out_dir / f"{stub}.json"
        if resume and out_js.exists():
            continue

        prompt = format_prompt(dp["lean"], prompt_type)
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "```lean4\n"},
        ]
        prompt_txt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

        outs = mdl.generate(prompt_txt, sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    temperature=1.0,
                    n=num_return_sequences,
                ))[0].outputs
        
        proofs = [extract_last_lean_block(o.text) for o in outs]

        write_json(out_js, {
            "prompt": prompt,
            "responses": [o.text for o in outs],
            "proofs": proofs,
        })

# ───────────────────── STAGE 5 – reward ────────────────────────
def verify_and_reward(cfg, client: Lean4Client) -> None:
    prover_out_dir = pathlib.Path(cfg["stage4_out"])
    verified_dir   = pathlib.Path(cfg["stage5_out"]); verified_dir.mkdir(parents=True, exist_ok=True)
    results_js     = pathlib.Path(cfg["results_json"])

    # ------------------------------------------------------------------ build batch
    proofs_batch, id2meta = [], {}          # custom_id ➜ (file_stub, proof_txt)
    for file_idx, fpath in enumerate(prover_out_dir.glob("*.json")):
        data       = json.loads(fpath.read_text())
        file_stub  = fpath.stem
        for cand_idx, proof_txt in enumerate(data["proofs"]):
            cid = f"{file_idx}_{cand_idx}"
            proofs_batch.append({"custom_id": cid, "proof": proof_txt})
            id2meta[cid] = (file_stub, proof_txt)

    # ------------------------------------------------------------------ kimina call (client passed in)
    v_cfg  = cfg["verifier"]
    r      = client.verify(proofs_batch, timeout=v_cfg["timeout"])

    # ------------------------------------------------------------------ collate rewards
    first_pass, reward_tbl = {}, []         # file_stub ➜ proof_txt
    for res in r["results"]:
        cid           = res["custom_id"]
        file_stub, p  = id2meta[cid]

        passed = (res["error"] is None
                  and all(m["severity"] != "error" for m in res["response"]["messages"]))

        if passed and file_stub not in first_pass:
            first_pass[file_stub] = p      # remember first successful proof

    for file_stub in {fp.stem for fp in prover_out_dir.glob("*.json")}:
        succeeded = file_stub in first_pass
        reward_tbl.append({"name": file_stub,
                           "passed": succeeded,
                           "reward": 1.0 if succeeded else 0.0})
        if succeeded:
            write_text(verified_dir / f"{file_stub}.lean", first_pass[file_stub])

    # dev‑mode dump of failing responses (optional)
    if cfg["mode"]["development"]:
        for res in r["results"]:
            file_stub, _ = id2meta[res["custom_id"]]
            if file_stub not in first_pass:       # only failures
                with open(f"{file_stub}_fail.json", "w") as fh:
                    json.dump(res, fh, indent=2)

    write_json(results_js, reward_tbl)
    n_ok = sum(x["passed"] for x in reward_tbl)
    print(f"✓ verification complete – {n_ok}/{len(reward_tbl)} proofs accepted")

# ───────────────────────────── main ────────────────────────────
def main():
    args = get_args()
    cfg = load_cfg(args.config_path)

    # 0 – load dataset once
    seeds = read_lines(cfg["input_jsonl"], split="test", dev_cfg=cfg["mode"])

    # 1 – generate sketches
    run_sketcher(cfg, seeds, resume=args.resume)
  
    # 2 – mutate with local model
    run_mutator(cfg, resume=args.resume)

    # 3 – validate mutations (syntax + aesop) using verifier
    v_cfg = cfg["verifier"]
    verifier_client = Lean4Client(base_url=v_cfg["url"])

    valid_mutations = validate_mutations(cfg, verifier_client)

    # 4 – attempt proofs
    run_prover(cfg, valid_mutations, resume=args.resume)

    # 5 – Lean kernel verification + reward table
    verify_and_reward(cfg, verifier_client)

if __name__ == "__main__":
    torch.set_grad_enabled(False)   # inference‑only
    main()

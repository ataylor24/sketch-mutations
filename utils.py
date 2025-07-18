def format_prompt(
    statement: str,
    format_type: str = "default",
    *,
    topic: str | None = None,
) -> str:
    """
    Build a prompt string for the chosen `format_type`.

    Parameters
    ----------
    statement : str
        Lean 4 snippet that contains header + theorem skeleton (typically ends with `by sorry`).
    format_type : str, default "default"
        One of:
            - "mutation"          : ask for ONE new theorem that is harder but related
            - "mutation_cot"      : mutation + natural‑language reasoning before the Lean code
            - "cot"               : chain‑of‑thought plan before full proof
            - "informal2formal"   : informal explanation then Lean proof
            - "informal_ds"       : *DeepSeek‑friendly* informal first, then proof
            - "informal_o3"       : same but for o3 / GPT‑4o
            - "default"           : just ask to complete the Lean proof
    topic : str | None
        Optional topic keyword (e.g. "Cauchy sequences").  Used only by mutation prompts.
    """
    if format_type == "mutation":
        prompt = """
        You are an expert mathematician working in Lean 4.

        **Task**: Produce exactly ONE new Lean 4 *theorem statement* (no proof) that is
        • still about {topic},  
        • strictly stronger or otherwise non‑trivial compared to the original,
        • re‑uses symbol names where sensible,
        • compiles when its body is replaced by `by sorry`.

        Return only the new Lean declaration inside ```lean4 fences – no informal text.  
        End the code block with `-- END` so downstream parsing can stop there.

        Original theorem:
        ```lean4
        {stmt}
        """.strip()
        return prompt.format(stmt=statement, topic=topic or "the same topic")
    elif format_type == "mutation_cot":
        prompt = """
        You are an expert mathematician working in Lean 4.

        Step 1 – Briefly explain (in natural language) how you will strengthen or
        modify the following theorem to make it more challenging yet still plausible.

        Step 2 – Output exactly ONE new Lean 4 theorem statement (no proof) that
        meets those criteria. Put it in a ```lean4 block and end with -- END.

        Original theorem:
        ```lean4
        {stmt}
        """.strip()
        return prompt.format(stmt=statement, topic=topic or "the same topic")
    # --------------- existing prompt types, tightened ---------------
    elif format_type == "cot":
        prompt = """
        Complete the following Lean 4 code:

        {stmt}
        Before you write the formal Lean proof, output a detailed proof plan
        explaining the key lemmas and tactics you will use. Then give the Lean proof
        inside a single ```lean4 block (no extra commentary after the code).
        """.strip()
        return prompt.format(stmt=statement)

    elif format_type == "informal2formal":
        prompt = """
        For the theorem below, first give an informal, English‑language proof,
        then formalise it in Lean 4:

        {stmt}
        """.strip()
        return prompt.format(stmt=statement)

    elif format_type == "informal_ds":     # DeepSeek tends to ignore "no Lean" instructions
        prompt = """
        Complete the following Lean 4 code:

        {stmt}
        Before producing the Lean proof, provide a brief informal proof summary.
        """.strip()
        return prompt.format(stmt=statement)

    elif format_type == "informal_o3":
        prompt = """
        Provide a natural‑language, informal proof for the following theorem:

        {stmt}
        """.strip()
        return prompt.format(stmt=statement)

    elif format_type == "default":
        prompt = """
        Complete the following Lean 4 code:

        {stmt}
        """.strip()
        return prompt.format(stmt=statement)

    else:
        raise ValueError(f"Unknown format_type: {format_type}")




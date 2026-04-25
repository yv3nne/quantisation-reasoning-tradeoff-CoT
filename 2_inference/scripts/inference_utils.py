import logging, re, subprocess, threading
from typing import Optional

import torch


# PORTED LOGITS PROCESSOR FOR/FROM VLLM

class ThinkingBudgetLogitsProcessor:
    """
    Ported from vLLM 0.12.0 custom logits processors to vLLM 0.7.3 (Volta/sm_70, CUDA 12).

    vLLM >= 0.12.0 introduced native CoT reasoning support via --enable-reasoning
    and guided-decoding-based think-token injection. That version requires
    torch 2.9 + CUDA 12.9 which drops Volta (sm_70) entirely.

    This class reimplements the same behaviour using SamplingParams.logits_processors,
    which is available from vLLM 0.7.3.
    It forces the end-of-thinking close tag (</think> for Qwen3, <|/think|> for
    other models) at exactly position `budget` if the token has not already been
    emitted, pushing CoT models to transition from reasoning to answer
    generation within the specified token budget.

    Stateless design: relies only on len(token_ids), making it safe for batched
    vLLM offline generation where the same SamplingParams may be reused.
    """

    def __init__(self, budget: int, think_end_token_id: int):
        self.budget = budget
        self.think_end_token_id = think_end_token_id

    def __call__(self, token_ids: list, logits: torch.Tensor) -> torch.Tensor:
        # at position `budget`, if </think> hasn't appeared yet, force it.
        if len(token_ids) == self.budget and self.think_end_token_id not in token_ids:
            logits = torch.full_like(logits, float("-inf")) # sets all other probabilities to -inf
            logits[self.think_end_token_id] = 0.0 # leaves only think token id as prediction option
        return logits


def get_think_end_token_id(tokenizer) -> int:
    """
    Resolve the single-token end-of-thinking marker from the tokenizer vocabulary.
    Tries common tags in order; falls back to full vocab scan.
    """
    for candidate in ["</think>", "<|/think|>", "<|endofthink|>"]:
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            logging.info(f"[THINK] End-think token: '{candidate}' → id {ids[0]}")
            return ids[0]
    for token, token_id in tokenizer.get_vocab().items():
        if token in ("</think>", "<|/think|>"):
            logging.info(f"[THINK] End-think token (vocab scan): '{token}' → id {token_id}")
            return token_id
    raise ValueError(
        "Cannot find end-think token in tokenizer vocabulary. "
        "Ensure a Qwen3 or compatible CoT model tokenizer is loaded."
    )


# DATASET UTILITIES FOR CHAT FORMAT AND RESPONSE POSTPROCESSING

def _extract_boxed(text: str) -> Optional[str]:
    """Return the content of the outermost \\boxed{} in text, handling nested braces."""
    # model outputs on math latex heavy, spawning need for bracket content extraction
    idx = text.find(r"\boxed{")
    if idx == -1: return None
    start = idx + len(r"\boxed{")
    depth, i = 1, start
    while i < len(text) and depth:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def extract_answer(text: str) -> str:
    """
    Strip thinking blocks and extract the final answer from model output.
    Priority order: <result>…</result> -> \\boxed{} -> explicit phrase -> last line.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|think\|>.*?<\|/think\|>", "", text, flags=re.DOTALL)
    text = text.strip()

    for m in (re.search(r"<result>(.*?)</result>", text, re.DOTALL), re.search(r"\\result\{([^}]*)\}", text)):
        if m: return m.group(1).strip()

    boxed = _extract_boxed(text)
    if boxed is not None: return boxed

    m = re.search(
        r"(?:answer\s*(?:is\s*)?[:=]?\s*|the answer is\s*|=\s*)([^\n\.]+)",
        text, re.IGNORECASE,
    )
    if m: return m.group(1).strip().lower()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return lines[-1].lower() if lines else text.lower()


def build_prompt(tok, raw_prompt: str, thinking_budget: Optional[int] = None) -> str:
    """Apply chat template, passing thinking budget where supported."""
    messages = [{"role": "user", "content": raw_prompt}]
    kwargs: dict = dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    try: return tok.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("thinking_budget", None)
        try: return tok.apply_chat_template(messages, **kwargs)
        except Exception: return raw_prompt
    except Exception: return raw_prompt


# GPU UTILITIES

class GPUMonitor:
    """Background thread that samples mean nvidia-smi power draw and memory BW."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.power: list = []
        self.mem_bw: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    text=True,
                ).strip().split(",")
                self.power.append(float(out[0]))
                self.mem_bw.append(float(out[1].strip()))
            except Exception:
                pass
            self._stop.wait(self.interval)

    @property
    def mean_power(self) -> float:
        return sum(self.power) / len(self.power) if self.power else float("nan")

    @property
    def mean_mem_bw(self) -> float:
        return sum(self.mem_bw) / len(self.mem_bw) if self.mem_bw else float("nan")

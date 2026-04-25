import csv, logging, os, datetime, sys, time
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from zoneinfo import ZoneInfo

# model and dataset deps
import datasets as hf_datasets, sklearn.metrics
import torch, transformers
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# custom
from scripts.inference_utils import (
    ThinkingBudgetLogitsProcessor,
    get_think_end_token_id, extract_answer,
    build_prompt, _extract_boxed,
    GPUMonitor,
)
from scripts.math_equiv import _EQUIV_FNS


# DATASET FORMAT AND LOADING

@dataclass
class BenchmarkDataset:
    """A loaded, split dataset for inference evaluation."""
    name:       str
    prompts:    List[str]
    references: List[str]
    split_idx:  int
    full_size:  int                = 0
    equiv_fn:   Optional[Callable] = None  # if None, uses exact string matching


def _to_prompts_refs(hf_ds, name: str, inference_prompt: str):
    """Extract (prompts, references) from a HuggingFace dataset."""
    prompts, refs = [], []
    for row in hf_ds:
        if name == "hendrycks_math":
            prompts.append(row["problem"] + "\n\n" + inference_prompt)
            boxed = _extract_boxed(row["solution"])
            refs.append(boxed if boxed is not None else row["solution"].strip())
        elif name == "drop":
            prompts.append(f"Passage: {row['passage']}\n\nQuestion: {row['question']}\n\n{inference_prompt}")
            spans = row["answers_spans"]["spans"] # for eleuther format is different despite being the same dataset
            refs.append(spans[0].strip().lower() if spans else "")
        else: raise ValueError(f"No prompt/ref extractor defined for dataset: {name}")
    return prompts, refs


def _half(lst, split_idx: int): # for equal cross-val split of prompts and solutions
    mid = len(lst) // 2
    return lst[:mid] if split_idx == 0 else lst[mid:]


def load_benchmark_datasets(
    hf_datasets_cfg: list,
    split_idx: int,
    smoke_n: Optional[int],
    inference_prompt: str,
) -> List[BenchmarkDataset]:
    """Load all configured benchmark datasets. Raises if all fail to load."""
    datasets = []
    for cfg in hf_datasets_cfg:
        try:
            hf_ds = hf_datasets.load_dataset(
                cfg["handle"],
                *([cfg["config"]] if cfg["config"] else []),
                split=cfg["split"],
            )
            prompts, refs = _to_prompts_refs(hf_ds, cfg["name"], inference_prompt)
            prompts = _half(prompts, split_idx)
            refs    = _half(refs,    split_idx)
            full_size = len(prompts)
            if smoke_n is not None:
                prompts, refs = prompts[:smoke_n], refs[:smoke_n]
            datasets.append(BenchmarkDataset(
                name=cfg["name"], prompts=prompts,
                references=refs, split_idx=split_idx,
                full_size=full_size,
                equiv_fn=_EQUIV_FNS.get(cfg.get("equiv")),
            ))
            logging.info(f"[DATA] {cfg['name']}: {len(prompts)} samples (full={full_size}, split={split_idx})")
        except Exception as e:
            logging.warning(f"[DATA] Failed to load {cfg['name']}: {e}")

    if not datasets:
        raise RuntimeError("All configured datasets failed to load.")
    return datasets


# INFERENCE CONFIGS

@dataclass
class InfConf:
    """
    Configuration and lifecycle management for a single model inference run.
    Loads a GPTQ-quantised model from `base_path` using vLLM.

    CoT think-token injection ported from vLLM 0.12.0 to 0.7.3 via
    SamplingParams.logits_processors (see ThinkingBudgetLogitsProcessor).
    """
    model_name:             str # ie qwen
    quant_bits:             int # ie 8bit
    quant_groupsize:        int # for VRAM use
    handle:                 str # on hf
    base_path:              str
    answer_tokens:          int   = 256
    thinking_budget:        int   = 128
    gpu_memory_utilization: float = 0.85
    pred_log_path:          Optional[str] = None
    benchmark_results_path: Optional[str] = None # for metrics entries
    path:                   str    = field(init=False, repr=True)
    tok:                    object = field(default=None, repr=False, init=False)
    llm:                    object = field(default=None, repr=False, init=False)
    _think_end_id:          int    = field(default=-1,   repr=False, init=False)

    def __post_init__(self):
        self.path = os.path.abspath(os.path.join(
            self.base_path,
            f"{self.model_name}_Q_{self.quant_bits}B{self.quant_groupsize}G",
        ))

    def setup(self):
        """Load tokenizer and vLLM LLM from the quantised model on disk."""
        self._load_tokenizer()
        self._load_llm()

    def cleanup(self):
        """Release GPU memory including vLLM distributed process group."""
        import gc
        # Destroy vLLM distributed state, releases NCCL/GLOO handles and CUDA buffers
        # that Python GC cannot reach by deleting the LLM object.
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception: pass
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception: pass
        self.llm = None
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"[LLM] Released {self.model_name} {self.quant_bits}b")

    def _load_tokenizer(self):
        logging.info(f"[TOKENIZER] Loading from HF handle: {self.handle}")
        self.tok = transformers.AutoTokenizer.from_pretrained(self.handle)
        self.tok.pad_token = self.tok.eos_token
        self._think_end_id = get_think_end_token_id(self.tok)
        logging.info("[TOKENIZER] Loaded")

    def _load_llm(self):
        logging.info(f"[LLM] Loading vLLM LLM from {self.path}")
        self.llm = LLM(
            model=self.path,
            tokenizer=self.handle,
            dtype="float16",
            quantization="gptq",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=4096,
            enable_chunked_prefill=False,
        )
        logging.info("[LLM] Ready")

    def make_sampling_params(self) -> SamplingParams:
        logits_processors = []
        if self._think_end_id >= 0:
            logits_processors.append(
                ThinkingBudgetLogitsProcessor(self.thinking_budget, self._think_end_id)
            )
        return SamplingParams(
            temperature=0,
            max_tokens=self.thinking_budget + self.answer_tokens,
            logits_processors=logits_processors or None,
            logprobs=1,
        )


# RUN BENCHMARK ON CONFIG, DATASET AND RECORD METRICS

def benchmark(conf: InfConf, dataset: BenchmarkDataset, rouge, meteor) -> float:
    """Run inference on dataset with conf's current thinking_budget; return mean s/request."""
    start_wall = datetime.datetime.now(ZoneInfo("Europe/Berlin")).isoformat()
    run_id = (
        f"{conf.model_name}_{conf.quant_bits}b_"
        f"{conf.thinking_budget}_{dataset.name}_s{dataset.split_idx}"
    )
    logging.info(
        f"[BENCHMARK] model={conf.model_name} bits={conf.quant_bits} "
        f"budget={conf.thinking_budget} dataset={dataset.name} split={dataset.split_idx}"
    )

    sampling_params = conf.make_sampling_params()
    predictions, elapsed_per_req = [], []

    start_perf = time.perf_counter()
    with GPUMonitor() as monitor:
        for raw_prompt in tqdm(
            dataset.prompts,
            desc=f"{conf.model_name} {conf.quant_bits}b budget={conf.thinking_budget} [{dataset.name}]",
            file=sys.stdout, mininterval=0, leave=True,
        ):
            prompt_text = build_prompt(conf.tok, raw_prompt, thinking_budget=conf.thinking_budget)
            t0 = time.perf_counter()
            outputs = conf.llm.generate([prompt_text], sampling_params, use_tqdm=False)
            elapsed_per_req.append(time.perf_counter() - t0)

            req_out = outputs[0].outputs[0]
            predictions.append(extract_answer(req_out.text))
            sys.stdout.flush()

    total_elapsed = time.perf_counter() - start_perf
    mean_req_time = sum(elapsed_per_req) / len(elapsed_per_req) if elapsed_per_req else float("nan")

    if dataset.equiv_fn is not None:
        correct  = [1 if dataset.equiv_fn(p, r) else 0 for p, r in zip(predictions, dataset.references)]
        accuracy = sum(correct) / len(correct) if correct else 0.0
        f1       = accuracy
    else:
        correct  = [1 if p == r else 0 for p, r in zip(predictions, dataset.references)]
        accuracy = float(sklearn.metrics.accuracy_score(dataset.references, predictions))
        f1       = float(sklearn.metrics.f1_score(
            dataset.references, predictions, average="macro", zero_division=0
        ))
    if conf.pred_log_path:
        pred_header = not os.path.exists(conf.pred_log_path)
        with open(conf.pred_log_path, "a", newline="") as pf:
            pw = csv.writer(pf)
            if pred_header:
                pw.writerow([
                    "run_id", "model", "quant_bits", "quant_groupsize",
                    "thinking_budget", "dataset", "split",
                    "sample_idx", "reference", "prediction", "is_correct",
                ])
            for i, (pred, ref, is_ok) in enumerate(
                zip(predictions, dataset.references, correct)
            ):
                pw.writerow([
                    run_id, conf.model_name, conf.quant_bits, conf.quant_groupsize,
                    conf.thinking_budget, dataset.name, dataset.split_idx,
                    i, ref, pred, is_ok,
                ])
        logging.info(f"[PRED_LOG] {len(predictions)} predictions written to {conf.pred_log_path}")

    results = {
        "elapsed_s":         round(total_elapsed, 3),
        "mean_req_s":        round(mean_req_time, 3),
        "power_W":           round(monitor.mean_power, 2),
        "mem_bandwidth_pct": round(monitor.mean_mem_bw, 2),
        "accuracy":          accuracy,
        "f1":                f1,
        "ROUGE":  float(rouge.compute(predictions=predictions, references=dataset.references)["rougeL"]),
        "METEOR": float(meteor.compute(predictions=predictions, references=dataset.references)["meteor"]),
    }

    if conf.benchmark_results_path:
        write_header = not os.path.exists(conf.benchmark_results_path)
        cols = [
            "start_time", "model", "quant_bits", "quant_groupsize",
            "thinking_budget", "handle", "dataset", "split",
        ] + list(results.keys())
        with open(conf.benchmark_results_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header: writer.writerow(cols)
            writer.writerow(
                [start_wall, conf.model_name, conf.quant_bits, conf.quant_groupsize,
                 conf.thinking_budget, conf.handle, dataset.name, dataset.split_idx]
                + [results[k] for k in results]
            )

    logging.info(f"[BENCHMARK] Done: {results}")
    return mean_req_time

import argparse, json, logging, os
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"

import triton.runtime.cache as _triton_cache
if not hasattr(_triton_cache, "default_dump_dir"): _triton_cache.default_dump_dir = _triton_cache.default_cache_dir
if not hasattr(_triton_cache, "default_override_dir"): _triton_cache.default_override_dir = _triton_cache.default_cache_dir

import evaluate
from scripts.benchmark import InfConf, benchmark, load_benchmark_datasets

# worker function to iterate through benchmark configurations, no worker doesnt clean all vram so must be cleanly separated
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # LOADING DATASET
    with open(args.config) as f: cfg = json.load(f)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(cfg["log_path"]), logging.StreamHandler()],
        force=True,
    )
    all_datasets = load_benchmark_datasets(
        cfg["hf_datasets"], cfg["dataset_split"], cfg["smoke_n"], cfg["inference_prompt"],
    )

    # WORKER CONFIGURATION
    conf = InfConf(
        model_name=cfg["model_name"],
        quant_bits=cfg["quant_bits"],
        quant_groupsize=cfg["quant_groupsize"],
        handle=cfg["handle"],
        base_path=cfg["base_path"],
        answer_tokens=cfg["answer_tokens"],
        gpu_memory_utilization=0.85,
        pred_log_path=cfg["pred_log_path"],
        benchmark_results_path=cfg["benchmark_results_path"],
    )
    conf.setup()
    logging.info("[WORKER] LLM ready")

    # SCORING
    rouge  = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    for budget in cfg["thinking_budgets"]:
        conf.thinking_budget = budget
        for ds in all_datasets:
            benchmark(conf, ds, rouge, meteor)

    # DONE
    conf.cleanup()
    logging.info("[WORKER] Done")


if __name__ == "__main__":
    main()

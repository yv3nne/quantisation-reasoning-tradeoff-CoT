import argparse, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))

STAGES = [ # for notebooks
    {
        "name": "quantisation",
        "dir":  os.path.join(ROOT, "1_quantisation"),
        "nb":   "project.ipynb",
    },
    {
        "name": "inference",
        "dir":  os.path.join(ROOT, "2_inference"),
        "nb":   "project.ipynb",
    },
]

def run_notebook(stage: dict, timeout: int = 7200, extra_env: dict = None) -> None:
    """Execute a Jupyter notebook via nbconvert inside its uv environment."""
    # set paths
    nb_path  = os.path.join(stage["dir"], stage["nb"])
    log_path = os.path.join(stage["dir"], f"{stage['name']}_nbconvert.log")
    cmd = [
        "uv", "run","jupyter", "nbconvert","--to", "notebook","--execute","--inplace",
        "--ExecutePreprocessor.timeout", str(timeout),
        nb_path,
    ]
    env = os.environ.copy()
    if extra_env: env.update(extra_env)

    # print settings
    split_tag = f" [split={extra_env.get('DATASET_SPLIT', '—')}]" if extra_env else ""
    run_tag   = f" [run={extra_env.get('RUN_INDEX', '—')}]" if extra_env and "RUN_INDEX" in extra_env else ""
    print(f"\n{'='*60}")
    print(f"[{stage['name'].upper()}]{split_tag}{run_tag} Running: {nb_path}")
    print(f"  env  → SMOKE_TEST={env.get('SMOKE_TEST','1')}  "
          f"DATASET_SPLIT={env.get('DATASET_SPLIT','0')}  "
          f"CROSS_VAL={env.get('CROSS_VAL','0')}")
    print(f"  log  → {log_path}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    with open(log_path, "w") as log_f: # run
        proc = subprocess.run(
            cmd,
            cwd=stage["dir"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elapsed = time.perf_counter() - t0 # log time

    if proc.returncode != 0: # success or not
        print(f"\n[ERROR] {stage['name']} notebook failed (exit {proc.returncode}).\n  See log: {log_path}")
        sys.exit(proc.returncode)
    print(f"[OK] {stage['name']} completed in {elapsed:.1f}s")


def main(): # args
    parser = argparse.ArgumentParser(description="Run quantisation → inference pipeline")
    parser.add_argument(
        "--quant-only", action="store_true",
        help="Run quantisation notebook only",
    )
    parser.add_argument(
        "--inf-only", action="store_true",
        help="Run inference notebook only (assumes quantised models exist in _temp/)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help=(
            "Number of inference runs (--inf-only only). Each run appends to "
            "benchmark_results.csv and prediction_log.csv for CI computation. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--no-smoke", action="store_true",
        help="Disable 5-sample smoke test; run the full dataset (default: smoke enabled)",
    )
    parser.add_argument(
        "--subset", type=int, default=None, metavar="N",
        help=(
            "Run only the first N samples per dataset (implies --no-smoke). "
            "Useful for a faster-than-full run to check results before committing to the full set."
        ),
    )
    parser.add_argument(
        "--cross-val", action="store_true",
        help=(
            "Run the pipeline twice, swapping calibration/benchmark splits. "
            "Results land in _temp/split_0/ and _temp/split_1/."
        ),
    )
    parser.add_argument(
        "--timeout", type=int, default=7200,
        help="Per-notebook timeout in seconds (default: 7200)",
    )
    args = parser.parse_args()

    # incompatible setting handling
    if args.quant_only and args.inf_only:
        print("[ERROR] Cannot combine --quant-only and --inf-only")
        sys.exit(1)

    if args.runs != 1 and not args.inf_only:
        print("[ERROR] --runs requires --inf-only")
        sys.exit(1)

    if args.runs < 1:
        print("[ERROR] --runs must be >= 1")
        sys.exit(1)

    stages = STAGES # for quantisation
    if args.quant_only: stages = [s for s in STAGES if s["name"] == "quantisation"]
    if args.inf_only: stages = [s for s in STAGES if s["name"] == "inference"]
    if args.subset is not None and args.subset < 1:
        print("[ERROR] --subset must be >= 1")
        sys.exit(1)

    # if --subset, --no-smoke also false
    smoke_val  = "0" if (args.no_smoke or args.subset is not None) else "1"
    subset_val = str(args.subset) if args.subset is not None else ""
    cross_val  = "1" if args.cross_val else "0"
    splits = [0, 1] if args.cross_val else [0]

    t_start = time.perf_counter()

    for split_idx in splits: # cross val or not
        for run_idx in range(args.runs): # per run
            extra_env = {
                "SMOKE_TEST":    smoke_val,
                "SUBSET_N":      subset_val,
                "CROSS_VAL":     cross_val,
                "DATASET_SPLIT": str(split_idx),
                "RUN_INDEX":     str(run_idx),
            }

            # build labels for stdout
            parts = []
            if args.cross_val: parts.append(f"split {split_idx}/1")
            if args.runs > 1: parts.append(f"run {run_idx + 1}/{args.runs}")
            label = ", ".join(parts) if parts else "single run"
            print(f"\nPipeline ({label}): {' → '.join(s['name'] for s in stages)}")

            # run
            for stage in stages: run_notebook(stage, timeout=args.timeout, extra_env=extra_env)

    # finished
    total = time.perf_counter() - t_start
    print(f"\n{'='*60}\nPipeline complete in {total:.1f}s")
    inf_dir = os.path.join(ROOT, "2_inference")
    temp    = os.path.join(ROOT, "1_quantisation", "_temp")
    if args.cross_val:
        for s in [0, 1]:
            print(f"  Results (split {s}): {os.path.join(temp, f'split_{s}', 'benchmark_results.csv')}")
    else: print(f"  Results:     {os.path.join(inf_dir, 'benchmark_results.csv')}")

    print(f"  Predictions: {os.path.join(inf_dir, 'prediction_log.csv')}")
    if args.runs > 1: print(f"  ({args.runs} runs x {len(splits)} split(s) appended)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
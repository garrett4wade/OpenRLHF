import json
import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import argparse
from scipy.stats import t
from pathlib import Path
import sys

# sys.path.append("/home/fw/openrlhf-bundle/OpenRLHF/ts_mlsys25")

from settings import MODEL_SIZE_TO_N_NODES_BAISC, N_NODES_TO_BATCH_SIZE, CTX, PROMPT_LEN

pd.set_option("display.precision", 2)  # Set precision to 4 decimal places
np.set_printoptions(precision=2)  # Set precision to 4 decimal places


def t_score_ci(data):
    # Calculate mean and standard deviation
    mean = np.mean(data)
    std_dev = np.std(data)

    # Define confidence level (e.g., 95%)
    confidence_level = 0.95

    # Degrees of freedom (n-1 for a sample)
    degrees_of_freedom = len(data) - 1

    # Calculate the critical value based on the confidence level and degrees of freedom
    t_score = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = t_score * (std_dev / np.sqrt(len(data)))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound


benchmark_db = defaultdict(list)


def _parselog(
    actor_size: int,
    critic_size: int,
    bs: int,
    ctx: int, prompt_len: int,
    rollout_n_mbs:int,
    train_n_mbs: int,
):
    exp_name = "mlsys"
    trial_name = f"a{actor_size}c{critic_size}b{bs}ct{ctx}p{prompt_len}nr{rollout_n_mbs}nt{train_n_mbs}"

    logpath = str(Path("/mnt/bs_fs/openrlhf-logs") / exp_name / trial_name / "output.log")

    oom = False
    time_records = []
    tflops_records = []
    try:
        with open(logpath, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if "ray.exceptions.RayTaskError(OutOfMemoryError)" in line or "CUDA out of memory" in line:
                    oom = True
                elif (
                    "torch.distributed.DistBackendError: NCCL error"
                    in line
                ):
                    oom = True
                    break
                if "End-to-End" in line:
                    step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                    time_records.append(step_time)
                    tflops = float(line.split("TFLOPs: ")[1].split(",")[0])
                    tflops_records.append(tflops)
                if "Benchmarking finishes" in line:
                    oom = False
    except FileNotFoundError:
        # print(f"File not found: {logpath}")
        return False, oom

    time_records = time_records[1:]
    if not oom:
        if len(time_records) == 0 or len(tflops_records) == 0:
            return False, oom
        avg_time = np.mean(time_records)
    else:
        avg_time = float("inf")
    w = MODEL_SIZE_TO_N_NODES_BAISC[actor_size] * 8
    d = dict(
        a=actor_size,
        c=critic_size,
        ctx=ctx,
        plen=prompt_len,
        n_gpus=w,
        avg_t=avg_time,
        log_path=logpath,
    )
    if not oom:
        for k, v in d.items():
            benchmark_db[k].append(v)
    return True, oom


def main():
    for size in [7, 13, 34, 70]:
        for scale_both in [True, False]:
            factors = [2**x for x in range(7)]
            for nr, nt in itertools.product(factors, factors):
                parse_success, oom = _parselog(
                    actor_size=size,
                    critic_size=7 if not scale_both else 13,
                    bs=N_NODES_TO_BATCH_SIZE[MODEL_SIZE_TO_N_NODES_BAISC[size]],
                    ctx=CTX,
                    prompt_len=PROMPT_LEN,
                    rollout_n_mbs=nr,
                    train_n_mbs=nt,
                )
                if parse_success and not oom:
                    break
    df = pd.DataFrame(benchmark_db)
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()

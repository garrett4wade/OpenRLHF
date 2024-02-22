import json
import os
import subprocess
import numpy as np
import itertools
import pandas as pd
from collections import defaultdict
import argparse

benchmark_db = defaultdict(list)


def _parselog(
    model_size: int,
    zero_stage: int,
    seqlen: int,
    bs: int,
    offload: bool,
):
    exp_name = f"or-a{model_size}s{seqlen // 100}g{bs}"
    if offload:
        exp_name += "-of"
    logpath = f"/lustre/aigc/llm/logs/fw/{exp_name}/1/rlhf.log"
    oom = False
    time_records = []
    tflops_records = []
    thpt_records = []
    max_mem = 0.0
    try:
        with open(logpath, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if "CUDA out of memory" in line:
                    oom = True
                    break
                if "End-to-End" in line:
                    step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                    time_records.append(step_time)
                    tflops = float(line.split("TFLOPs: ")[1].split(",")[0])
                    tflops_records.append(tflops)
                    thpt = float(line.split(", Samples/sec: ")[1].split(",")[0])
                    thpt_records.append(thpt)
                if "Compute Utilization - " in line:
                    mem = float(line.split("Used Memory - ")[1].split("MB,")[0])
                    max_mem = max(max_mem, mem)
    except FileNotFoundError:
        # print(f"File not found: {logpath}")
        return False
    if not oom:
        if len(time_records) == 0 or len(tflops_records) == 0 or len(thpt_records) == 0 or max_mem == 0.0:
            return False
        avg_time = np.mean(time_records)
        avg_tflops = np.mean(tflops_records)
        thpt = np.mean(thpt_records)
    else:
        avg_time = float("inf")
        avg_tflops = -float("inf")
        thpt = -float("inf")
        max_mem = 0.0
    d = dict(
        model_size=model_size,
        zero_stage=zero_stage,
        seqlen=seqlen,
        bs=bs,
        offload=offload,
        OOM=oom,
        Throughput=thpt,
        MaxGPUMemory=max_mem,
    )
    for k, v in d.items():
        benchmark_db[k].append(v)
    return True


def parselog(model_size: int):
    if model_size <= 13:
        zero_stage = 2
    else:
        zero_stage = 3
    bszs = range(1, 100)
    seqlens = [256, 512, 1024]
    offloads = [True, False]
    for max_answer_len, bs, offload in itertools.product(seqlens, bszs, offloads):
        _parselog(model_size, zero_stage, max_answer_len, bs, offload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, default=7, choices=[7, 13, 34, 70], nargs="+")
    parser.add_argument("--max", action="store_true")
    args = parser.parse_args()
    for model_size in args.model_size:
        parselog(model_size)
    df = pd.DataFrame(benchmark_db)
    if not args.max:
        print(df.to_string(index=False))
    else:
        max_throughput_df = df.loc[df.groupby(["model_size", "seqlen"])["Throughput"].idxmax()]
        print(max_throughput_df.to_string(index=False))
